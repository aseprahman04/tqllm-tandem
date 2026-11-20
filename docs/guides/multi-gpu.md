# Two GPUs

[Documentation](../README.md) / [Project home](../../README.md)

Two ways to use two NVIDIA GPUs with one model: a single process that splits the model across both cards (the
multi-GPU fork addition), or two processes, one per card, talking over NCCL (upstream tqLLM). Pick by what you have:
one PC with two cards in one process, or two boxes.

## Single process: layer split (`TQLLM_SPLIT`)

*Multi-GPU fork addition: not part of upstream tqLLM.*

One Qwen3.8-27B model can run across two NVIDIA cards in one PC. The layers are split in two contiguous ranges: the
first `K` layers run on CUDA device 1, the rest, the output head and the DFlash2 drafter on CUDA device 0. Each layer's
weights, KV cache and recurrent state live on its own card, so the two cards hold together what neither holds alone:
a bigger quantization, or a far longer context.

This is pipeline (layer) parallelism, not tensor parallelism: a verify block's rows cross from one card to the other
once (about 10 KB a token), not twice a layer. No NVLink or peer-to-peer access is needed; the copy goes through host
memory when the cards have none (GeForce cards on Windows). The design follows the layer split of
[PolyStrata](https://github.com/simplexai-labs/PolyStrata)'s strata-poly engine; the code is new.

### Using it

Set `TQLLM_SPLIT` to the first layer of the second range. Everything else is a normal `tqserve` command line:

```bat
set CUDA_VISIBLE_DEVICES=0,1
set TQLLM_SPLIT=44
tqserve.exe --model Qwen3.8-27B-TQ_L.gguf ^
  -md drafters\Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab mtp-vocab\atx_65536.txt ^
  --context 524288 --kv tq3 --kv-host 0 --prompt-cache on ^
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144
```

- **`TQLLM_SPLIT=K`**: layers `0 .. K-1` on CUDA device 1, `K .. 63` plus the head and the drafter on device 0.
  Device 0 is the "home" card: every request thread starts on it. Put the display on device 0 and give device 1 more
  layers (a larger `K`), since device 0 also holds the head, the drafter and the desktop.
- **`--kv-host 0`** is required for now: all KV stays on the two cards.
- Greedy output is the one-card output token for token (checked on Qwen3.8-27B TQ_XS, which fits one card).

### What runs

| | |
|---|---|
| Models | dense Qwen3.5 / 3.6 / 3.8 GGUFs (`qwen35`) and their MoE models (`qwen35moe`, every expert on the GPUs), TQ formats and llama.cpp K-quants |
| Drafter | DFlash2 (`-md`), or none |
| Slots | one (`--np 1`) |
| KV | on the GPUs (`--kv-host 0`); any KV format |
| Prompt reuse | `--prompt-cache on` |
| Not yet | `--kv-host` (a host KV tail), `--kv-compact`, `--np > 1`, experts in host memory (`--moe-host-mb`), images |

A verify block runs as two captured CUDA graphs, one per card, with the hand-off between them. Prefill chunks run eagerly
and overlap across the cards: device 1 starts the next chunk while device 0 finishes this one (`split_prefill`). Commits
and tree gathers run eagerly.

### Measured

2x RTX 5060 Ti 16 GB, Windows 11, CUDA 13.2, greedy, thinking off, 512 output tokens. Code and prose prompts are short;
the long prompt is a 26.9K-token source file with a question.

| Model, context | Split | Code tok/s | Prose tok/s | Long prompt |
|---|---|---:|---:|---|
| TQ_XS, 8K, one card | none | 205 | 81 | — |
| TQ_XS, 8K | `K=32` | 196-200 | 80-85 | — |
| TQ_L, 512K (YaRN x2) | `K=44` | 127-140 | 81 | prefill 1,231 tok/s (26.9K tokens), decode 69-72 tok/s |
| TQ_L, 131K | `K=40` | 127-140 | 81 | prefill 1,650 tok/s (36.7K tokens) |
| UD-Q6_K, 131K | `K=36` | 86-96 | 57 | prefill 1,576 tok/s (36.7K tokens) |

TQ_L (14.7 GB) does not load on one 16 GB card at any context. On two it serves the trained 262,144 tokens with room
to spare and 524,288 with YaRN x2 (VRAM 12.7 GB on device 0 with the display, 14.9 GB on device 1).

Long-context retrieval (a passphrase planted in tqLLM's own source text, greedy answer):

| Context | Prompt tokens | Depth | Prefill | Result |
|---|---:|---:|---|---|
| 262,144 | 159,491 | 25 % | 574 tok/s, 278 s | found |
| 262,144 | 248,390 | 50 % | 478 tok/s, 722 s | found |
| 524,288 (YaRN x2) | 459,444 | 35 % | 348 tok/s, 1,319 s | found |

A 20-task coding check (Python functions with unit tests) passes 19 of 20 with TQ_L split, with or without YaRN.

### Memory

With `--kv tq3` a position takes 224 bytes per KV head in each attention layer: 14.3 KB a token for Qwen3.8-27B's 16
attention layers, 3.8 GB at 262,144 tokens and 7.5 GB at 524,288. Each card holds the KV of its own attention layers
(every fourth layer is one), so `K` also sets how the KV divides. A million tokens (15 GB of KV) does not fit two 16 GB
cards with TQ_L.

### Notes

- During a long prefill both cards work at once on consecutive chunks; the first card has more to do (11 of the 16
  attention layers at `K=44`), so it sets the pace.
- Kernels that need more than 48 KB of shared memory have their limit set on each card (`PerDeviceLimit`,
  `src/core/per_device_once.cuh`); the limit is a per-device attribute.
- With a split the token embeddings live in portable host memory, so both cards read them (one row per token).
- Code: `src/models/qwen/qwen35/qwen35_split.cpp` and the hooks marked "layer split" in the qwen35 engine.

## Single process: tensor parallel (`TQLLM_TP=1`)

The other way to use two cards in one process: every layer is halved, and both cards work on the same layer at once.
Each card holds half of every layer's attention heads, Gated DeltaNet heads and FFN rows (the slice map is in
[`docs/design/tensor-parallel.md`](../design/tensor-parallel.md)); after the mixer and after the FFN the two partial
sums are exchanged through mapped host memory, so both cards keep the same residual stream. The LM head, sampling and
the DFlash2 drafter run on CUDA device 0, which needs ~3 GB more: put the card without the display first.

```bat
set CUDA_VISIBLE_DEVICES=1,0
set TQLLM_TP=1
build\tqserve.exe --model models\Qwen3.8-27B-TQ_L.gguf ^
  -md models\drafters\Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab models\mtp-vocab\atx_65536.txt ^
  --context 262144 --kv tq3 --kv-host 0 --prompt-cache on --port 8086
```

(`set TANDEM_TP=1` before `scripts\windows\tandem-serve.bat` does the same.) Measured on 2x RTX 5060 Ti, Windows 11:

| TQ_L + DFlash2 | Code tok/s | Prose tok/s | Prefill 9.7K / 36.7K | VRAM (device 0 / 1) |
|---|--:|--:|--:|--:|
| Layer split, `K=40`, 131,072 | 128-133 | 82-83 | 1,662 / 1,650 | |
| **Tensor parallel, 262,144** | **200-210** | **118-124** | **1,996 / 1,923** | 13.2 / ~11.7 GB (+0.9 GB with `--mmproj`) |

Decode is ~50 % faster and prefill ~10 % faster than the layer split. Prefill runs two chunks per pass (mixer of
chunk 0, mixer of chunk 1, FFN of chunk 0, FFN of chunk 1), so each chunk's exchanges cross the bus through the copy
engine while the cards compute the other chunk; without that pairing (`TQLLM_DEBUG=tp_nopair`, same outputs) the cards
wait on each other 128 times per chunk. The decode head's 4-bit pass is split by vocabulary rows across both cards.
With `--prompt-cache on` a follow-up turn reads only its new tokens. A needle at 229,716 tokens was found. The answers
are not bit-identical to one card (two partial sums per row-parallel projection), so `tandem_golden.py` does not apply;
the 20-task coding check passes 20 / 20.

Images work under TP: add `--mmproj` (Qwen3.8-27B: Unsloth's `mmproj-F16.gguf` from `unsloth/Qwen3.8-27B-GGUF`). The
vision encoder runs on CUDA device 0; the image rows and their M-RoPE positions are copied to the other card, and both
scatter them into their embeddings. TQ_XS under TP answers image questions exactly as on one card.

Limits so far: one slot (`--np 1`), dense models (no MoE), no `--mtp`, all KV on the GPUs (no `--kv-host`,
`--kv-compact`), not together with `TQLLM_SPLIT`, 262,144 tokens tested. A decode exchange costs ~25 us plus the
bytes (16 verify columns: ~70 us here, where the second card sits on PCIe 4.0 x4); WDDM gives each card one copy
engine. Linux is not measured.

## Two processes: `--split layer` (NCCL)

*Upstream tqLLM.* `--split layer` runs one dense Qwen3.5 / 3.6 / 3.8 model (GGUF) across two NVIDIA GPUs: each GPU holds
half the layers, its share of the KV cache and recurrent state, so each card needs less memory than one card holding the
whole model. The output is the same as on one GPU (greedy text identical, token for token). It does not make one request
faster: the two GPUs run one after the other.

One `tqserve` process runs per GPU. Rank 0 serves the API; rank 1 follows it and has no HTTP port.

### One PC with two GPUs

```
CUDA_VISIBLE_DEVICES=1 tqserve --model Qwen3.8-27B-TQ_L.gguf --context 65536 --np 4 \
    --split layer --tp-rank 1 --tp-peer 127.0.0.1 &
CUDA_VISIBLE_DEVICES=0 tqserve --model Qwen3.8-27B-TQ_L.gguf --context 65536 --np 4 \
    --split layer --tp-rank 0
```

Both processes take the same model file and the same options (`--context`, `--np`, `-md`, `-ub`). Rank 1 waits up to
10 minutes for rank 0. Stopping rank 0 stops rank 1; if rank 1 stops, rank 0 answers the running request with an error.

### Two machines

Each machine runs one rank against its own copy of the model file. `--tp-peer` is rank 0's address on the link between
them (e.g. the 200 Gb ConnectX ports of two DGX Spark / GB10 boxes), and both machines must use the same NCCL version:

```
# machine A (rank 0)
NCCL_SOCKET_IFNAME=enp1s0f0 tqserve --model MODEL.gguf --context 32768 --split layer --tp-rank 0
# machine B (rank 1)
NCCL_SOCKET_IFNAME=enp1s0f0 tqserve --model MODEL.gguf --context 32768 --split layer --tp-rank 1 --tp-peer 10.10.0.55
```

## Flash-Next: experts over two boxes (`--split experts`)

*Upstream tqLLM.* With every expert resident on both boxes (`slots=512` in the manifest), `--split experts` lets the
second box compute half of each layer's experts (e % 2 == 1) for the first box's decode steps whenever the second box has
no request of its own; when it is busy, the first box computes all of them. The output is the same as one box either way.
Run both behind one address with `--peer`, so a lone request lands on the first box and is helped, and several users
spread over both. On two GB10s (NVFP4 experts, 32 unseen questions, one user at a time): through the router below,
64.4 -> 68.4 tok/s (median, faster on all 32; several users unchanged); a dedicated pair of `--np 1` servers,
66.8 -> 70.5.

```
# both boxes: the two halves of the ConnectX port (each about 100 Gb/s), four channels per peer
export NCCL_SOCKET_IFNAME=enp1s0f0 NCCL_IB_GID_INDEX=3 NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_NCHANNELS_PER_NET_PEER=4
# machine B (rank 1): serves its own users, helps A while idle
tqserve --model flashnext.tqfn --np 8 --context 262144 --host 10.10.0.44 --port 8080 --split experts --tp-rank 1 --tp-peer 10.10.0.55
# machine A (rank 0, the front door)
tqserve --model flashnext.tqfn --np 8 --context 262144 --port 8080 --peer http://10.10.0.44:8080 --split experts --tp-rank 0
```

If rank 1 stops, the request in flight fails and rank 0 continues on its own (every expert is resident there, same output).

## Notes (upstream `--split`)

- The memory fit does not run: give `--context` and `--np` that fit each GPU.
- `--split-at N` moves the boundary (layers N and up on rank 1) when the cards differ in size.
- Not yet with `--split layer`: `--kv-host`, `--kv-compact`, image input, MoE models.
- Built from source, multi-GPU needs NCCL: a system `libnccl-dev` is found by itself, or pass `-DTQLLM_NCCL_DIR=...`.
  The Docker image includes it.
