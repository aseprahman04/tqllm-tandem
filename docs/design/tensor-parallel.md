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
