# Tensor parallel for dense Qwen3.5 / 3.8 (design, work in progress)

Goal: dense Qwen3.8-27B decode near 250 tok/s on two 16 GB GPUs without peer access (GeForce on Windows). The layer split
(`TQLLM_SPLIT`) makes the cards take turns inside every verify block, so a step costs the sum of both halves; tensor
parallelism makes both cards read their half of every layer's weights at once.

## Why, in numbers (2x RTX 5060 Ti, measured)

- Layer split, TQ_L: a verify step is 42 ms (each card ~21 ms, in turn); ~5.8 tokens accepted per step on code ->
  137 tok/s. Ceiling of the layer split at full bandwidth: ~175 tok/s (33 ms, 5.8 tokens).
- All-reduce of an 8 x 5120 bf16 partial between the cards (`tools/tp_bench/tp_bench.cu`, `test-tp-bench.bat`):
  copies through pinned host memory + events 143-154 us; **zero-copy (producer kernel stores into mapped pinned host
  memory, consumer kernel reads it) + cross-device events 42-47 us**; spinning on host-memory flags ~1.8 ms (WDDM batches
  submissions; unusable). 128 all-reduces per step (two per layer) -> ~5.6 ms.
- Projection (per-card ~350 GB/s effective, +1 ms head and draft): TQ_L ~210 tok/s code (~225 with tighter sync),
  **TQ_S ~235-255**, TQ_XS ~245-270. TQ_S on the layer split today: 153-160 code, 86 prose, 18/20 on the coding check.

## Shapes (Qwen3.8-27B) - every split is even

| | Full | Per GPU |
|---|--:|--:|
| hidden | 5120 | 5120 (replicated) |
| attention heads / KV heads / head dim | 24 / 4 / 256 | 12 / 2 / 256 |
| attention o-proj input | 6144 | 3072 (= 24 x 128) |
| GDN k-heads / v-heads / dk / dv | 16 / 48 / 128 / 128 | 8 / 24 / 128 / 128 |
| GDN out-proj input (inner) | 6144 | 3072 |
| FFN intermediate | 17408 | 8704 (= 68 x 128 = 34 x 256) |

## Plan

Megatron-style: column-parallel in (q/k/v/gate, GDN qkv/z/a/b, FFN gate/up: split by output rows, per head group),
row-parallel out (attention o, GDN out, FFN down: split by input columns, partial sums all-reduced). The residual
stream, norms and embeddings are replicated; each card holds half the heads, half the KV cache and half the GDN state.

- **Weights.** GGUF block formats are row-major runs of fixed-size blocks (TQ4: 66-byte blocks of 128 weights; K-quants:
  256-weight superblocks), so a row range is a byte range and a column range at a block boundary is a per-row block range.
  Slice the raw tensor bytes before `WeightFormat::load()` - format-agnostic. Fused matrices (`g_cat`: q|k|v(|gate),
  GDN mix) are rebuilt per card from each component's head slice. TQ rotations act per 128-block of the input, so column
  slices at multiples of 128 keep the basis.
- **Compute.** Per-layer code runs with halved head / intermediate counts on each card. Two hooks per layer after the
  row-parallel projections: zero-copy all-reduce of the partial (store to mapped host buffer, event, peer add).
- **Head.** Vocabulary split by rows; argmax / top-16 merged across the cards (small).
- **Execution.** M1: eager, interleaved host enqueue per segment (`use_dev` per card). M2: one captured graph per card
  per step with external event record / wait nodes; DFlash2 drafter on one card (it reads the replicated features).

## Engine integration (qwen35 engine, found while reading the code)

- **Device context.** Reuse the split's machinery: `cuda_of[2]`, `dstash[2]`, `use_dev(d)` (stream, side stream, events,
  cuBLAS, the ~85 `dev_bufs()` scratch buffers, per-GPU `Int8Gemm`). TP adds to `DevStash`, swapped by `use_dev` in O(1):
  `std::vector<Layer> layer` (each GPU's weight shards), `std::vector<KvRef> key_cache, value_cache` and the KV
  `kv_mem / kv_pool / kv_seq` (half the KV heads each), `std::vector<Buf> conv_state, rec_state` (half the GDN heads),
  `v_stash2[2]` (verify stash, sized by mix rows).
- **Shapes.** After load, `cfg.heads`, `cfg.kvheads`, `cfg.ffn`, `cfg.lin_kheads`, `cfg.lin_vheads` hold the per-GPU
  values (both GPUs are symmetric); `hidden`, `head_dim`, `lin_dk`, `lin_dv`, `vocab` stay full.
- **Loading.** `g_mat` / `g_cat` read whole tensors with `f.read(t)`; TP wraps the read with a row slice (column-parallel:
  q / k / v / gate per head, GDN mix rows per head group in the `[q k | v | z | b | a]` order, FFN gate / up) or a
  column slice at a block boundary (row-parallel: `wo`, GDN `w_out`, `w_down`), then calls the format's `load()`.
  Per-head vectors (q / k norms are per head-dim and replicated; GDN `conv_w`, `neg_a`, `dt_bias`, `gnorm_w` follow the
  head slice).
- **Slices for rank r (Qwen3.8-27B; checked against the kernels).**
  - Attention: `attn_q` holds `[q(256) | gate(256)]` per head (`o[h*dim+d] *= sigmoid(qg[h*2*dim+dim+d])`): rows
    `[r*12*512, (r+1)*12*512)`; `attn_k` / `attn_v`: rows `[r*2*256, (r+1)*2*256)` (GQA: q-heads 0-11 use KV heads 0-1);
    `attn_output`: columns `[r*12*256, (r+1)*12*256)`; `q_norm` / `k_norm` (per head-dim) replicated.
  - Gated DeltaNet: GGUF keeps the value heads **tiled** (`cfg.tiled`, `kh = h % kheads` in `gdn_multi_k`), so rank r
    owns k-heads `[8r, 8r+8)` and v-heads `{16g + 8r .. 16g + 8r + 7 : g = 0, 1, 2}` - three runs of 8 heads, which keep
    `h % kheads` right locally (local h' = 8g + i <-> global 16g + 8r + i). `attn_qkv` rows: q `[8r*128, +1024)`, k
    `2048 + [8r*128, +1024)`, v `4096 + run(g)*128`; `attn_gate` (z) rows, `ssm_out` columns: the v runs x 128;
    `ssm_beta` / `ssm_alpha` rows, `ssm_a`, `ssm_dt.bias`: the v runs; `ssm_conv1d` (per channel, conv_k taps): the
    q / k / v channels as `attn_qkv`; `ssm_norm` (per dv) replicated.
  - FFN: `ffn_gate` / `ffn_up` rows `[r*8704, (r+1)*8704)`; `ffn_down` columns the same.
- **Hooks** (all-reduce of a bf16 partial, `n x hidden`):
  1. mixer output `M` (attention `wo` in `attn_mixer`, GDN `mm_h(L.w_out, M, n)`), before `ffn_block`'s first
     `add_sumsq(M, resid)`;
  2. FFN output `X` after `mm_h(L.w_down, X, n)`, before the closing `add_sumsq(X, M)` (two places in `ffn_block`).
- **Head.** M1: the full LM head on the home GPU only (~2 ms a step); vocabulary split later.
- **Order of work per layer (eager, M1):** rank 0 mixer, rank 1 mixer, exchange, rank 0 FFN, rank 1 FFN, exchange. The
  layer loop in `verify_body` becomes per-segment calls so both ranks' segments interleave on the host.

## Results so far (2x RTX 5060 Ti, Windows 11, branch `tp`)

| Qwen3.8-27B, 8K context | Code tok/s | Prose tok/s | Prefill 9.7K | Coding check |
|---|--:|--:|--:|--:|
| TQ_XS, layer split + DFlash2 | 196-200 | 80-85 | - | 17/20 |
| TQ_XS, **TP** + DFlash2 (segment graphs) | **206-216** | **122** | - | - |
| TQ_S, layer split + DFlash2 | 153-160 | 86 | 1,643 | 18/20 |
| TQ_S, **TP** + DFlash2 (segment graphs, commit graphs) | **202-212** | **113** | 1,212 (eager) | 18/20 |
| TQ_L, layer split + DFlash2 (131K) | 128-133 | 82-83 | 1,650 | - |
| TQ_L, **TP** + DFlash2 (262K, `--prompt-cache on`, home on the second card) | **190-197** | **111-113** | 969 (eager) | 20/20 |
| TQ_XS without a drafter: one GPU / layer split / TP | 33.6 / 31.6 / 37.7 | | | |

- Correctness: TP answers are coherent; 2 of 4 golden answers identical to one GPU, the others diverge late (two
  partial sums per row-parallel projection). Graph vs eager TP: identical.
- Where a TQ_S verify step (27.8 ms + 1.3 ms commit) goes: ~15.7 ms weights (6.3 GB per GPU), ~5 ms the 128
  cross-GPU exchanges (~40 us each: WDDM event latency), ~2 ms drafter and tree, ~1.6 ms LM head (home only), the
  rest launch overhead. GPU0 86 % / GPU1 76 % busy.
- Tried: eager TP (host-bound: 167-171 / 98 on TQ_XS); one graph per GPU per step synchronised by flags in mapped
  host memory: ~1.6 ms an exchange under WDDM (`tools/tp_bench` mode D) - unusable.
- 250 tok/s needs the exchange latency well under 40 us (Linux without WDDM, or peer access / NVLink) or fewer
  exchanges. Next ideas: vocabulary-split head (~0.7 ms), prefill pipelined like the layer split, Linux numbers.

## Milestones

| | Scope | Done when |
|---|---|---|
| M1 | sharded load, single-token decode without a drafter, eager | TQ_XS answers identical to one GPU? (numerics change: partial sums are added in a different order, so "identical" may become "same within tolerance"; measure) + ms per token |
| M2 | DFlash2 verify blocks, graphs per card | tok/s on TQ_S |
| M3 | prefill under TP | long prompts |
| M4 | sync tuning, release | >= 235 tok/s code on TQ_S |

Open question for M1: bit-exactness. The row-parallel projections sum two partials (one per card) instead of one full dot
product, so results can differ in the last bits from one GPU; greedy text may diverge late in long answers. Check with
the golden prompts and report how often, rather than assume.
