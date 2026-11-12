# Two GPUs (layer split)

[Documentation](../README.md) / [Project home](../../README.md)

*Multi-GPU fork addition: not part of upstream tqLLM.*

One Qwen3.8-27B model can run across two NVIDIA cards in one PC. The layers are split in two contiguous ranges: the
first `K` layers run on CUDA device 1, the rest, the output head and the DFlash2 drafter on CUDA device 0. Each layer's
weights, KV cache and recurrent state live on its own card, so the two cards hold together what neither holds alone:
a bigger quantization, or a far longer context.

This is pipeline (layer) parallelism, not tensor parallelism: a verify block's rows cross from one card to the other
once (about 10 KB a token), not twice a layer. No NVLink or peer-to-peer access is needed; the copy goes through host
memory when the cards have none (GeForce cards on Windows). The design follows the layer split of
[PolyStrata](https://github.com/simplexai-labs/PolyStrata)'s strata-poly engine; the code is new.

## Using it

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

## What runs

| | |
|---|---|
| Models | dense Qwen3.5 / 3.6 / 3.8 GGUFs (`qwen35`), TQ formats and llama.cpp K-quants |
| Drafter | DFlash2 (`-md`), or none |
| Slots | one (`--np 1`) |
| KV | on the GPUs (`--kv-host 0`); any KV format |
| Prompt reuse | `--prompt-cache on` |
| Not yet | `--mtp`, `--kv-host` (a host KV tail), `--kv-compact`, `--np > 1`, MoE models, images |

A verify block runs as two captured CUDA graphs, one per card, with the hand-off between them; prefill chunks, commits
and tree gathers run eagerly.

## Measured

2x RTX 5060 Ti 16 GB, Windows 11, CUDA 13.2, greedy, thinking off, 512 output tokens. Code and prose prompts are short;
the long prompt is a 26.9K-token source file with a question.

| Model, context | Split | Code tok/s | Prose tok/s | Long prompt |
|---|---|---:|---:|---|
| TQ_XS, 8K, one card | none | 205 | 81 | — |
| TQ_XS, 8K | `K=32` | 196-200 | 80-85 | — |
| TQ_L, 512K (YaRN x2) | `K=44` | 127-140 | 81 | prefill 826 tok/s, decode 63 tok/s |

TQ_L (14.7 GB) does not load on one 16 GB card at any context. On two it serves the trained 262,144 tokens with room
to spare and 524,288 with YaRN x2 (VRAM 12.7 GB on device 0 with the display, 14.9 GB on device 1).

Long-context retrieval (a passphrase planted in tqLLM's own source text, greedy answer):

| Context | Prompt tokens | Depth | Prefill | Result |
|---|---:|---:|---|---|
| 262,144 | 159,491 | 25 % | 574 tok/s, 278 s | found |
| 262,144 | 248,390 | 50 % | 478 tok/s, 722 s | found |
| 524,288 (YaRN x2) | 459,444 | 35 % | 348 tok/s, 1,319 s | found |

A 20-task coding check (Python functions with unit tests) passes 19 of 20 with TQ_L split, with or without YaRN.

## Memory

With `--kv tq3` a position takes 224 bytes per KV head in each attention layer: 14.3 KB a token for Qwen3.8-27B's 16
attention layers, 3.8 GB at 262,144 tokens and 7.5 GB at 524,288. Each card holds the KV of its own attention layers
(every fourth layer is one), so `K` also sets how the KV divides. A million tokens (15 GB of KV) does not fit two 16 GB
cards with TQ_L.

## Notes

- During a long prefill the first card does most of the work: it holds 11 of the 16 attention layers at `K=44`.
- Kernels that need more than 48 KB of shared memory have their limit set on each card (`PerDeviceLimit`,
  `src/core/per_device_once.cuh`); the limit is a per-device attribute.
- With a split the token embeddings live in portable host memory, so both cards read them (one row per token).
- Code: `src/models/qwen/qwen35/qwen35_split.cpp` and the hooks marked "layer split" in the qwen35 engine.
