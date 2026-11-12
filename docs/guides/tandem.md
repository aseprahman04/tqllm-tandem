# Tandem guide: install, run, use, fix

[Documentation](../README.md) / [Project home](../../README.md) · how the split works: [multi-gpu.md](multi-gpu.md)

*Multi-GPU fork addition: not part of upstream tqLLM.*

## 1. Before you start

| You need | Notes |
|---|---|
| Two NVIDIA GPUs | RTX 20 series or newer. Measured on 2x RTX 5060 Ti 16 GB. Two cards of the same model are the tested case. |
| NVIDIA driver for CUDA 13 | R580 or newer (`nvidia-smi` shows "CUDA Version: 13.x"). |
| System RAM | 32 GB or more. The model files are memory-mapped while loading. |
| Disk | about 16 GB for TQ_L, the drafter and its vocabulary. |
| Windows 10 / 11 | Linux builds the same code but has no measured numbers yet. |

No NVLink, no SLI bridge and no peer-to-peer support are needed: the two cards exchange about 10 KB per token through
host memory.

## 2. Install

**From a release** (no compiler needed): download both zips from the
[releases page](https://github.com/aseprahman04/tqllm-tandem/releases), and unzip them into one folder:

- `tandem-v*-win-x64-cuda13.zip`: `tqserve.exe` (RTX 30 / 40 / 50), `tandem-serve.bat`, LICENSE, NOTICE.
- `cudart-tandem-cuda13-win-x64.zip`: the CUDA 13 libraries `tqserve.exe` loads. Skip it if the CUDA 13 toolkit's
  `bin\x64` folder is on your `PATH`.

Check the SHA-256 of each zip against the release notes before you run anything.

**From source**: see the [quick start](../../README.md#quick-start-windows) (Visual Studio 2022 Build Tools, CUDA 12.8+,
CMake, Ninja).

## 3. Download the model

From [YTan2000/Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ), into a `models` folder next to
`tqserve.exe`:

```bat
pip install -U huggingface_hub
hf download YTan2000/Qwen3.8-27B-TQ Qwen3.8-27B-TQ_L.gguf drafters/Qwen3.8-27B-DFlash2-TQ3.gguf ^
  mtp-vocab/atx_65536.txt --local-dir models
```

| File | Size | Why |
|---|--:|---|
| `Qwen3.8-27B-TQ_L.gguf` | 14.7 GB | the model (4-bit class); needs two 16 GB cards |
| `Qwen3.8-27B-TQ_XS.gguf` (optional) | 11.9 GB | 3-bit, fits one card: faster, weaker; useful for the correctness check |
| `drafters/Qwen3.8-27B-DFlash2-TQ3.gguf` | 0.8 GB | the DFlash2 drafter (speculative decoding) |
| `mtp-vocab/atx_65536.txt` | 0.4 MB | the drafter's vocabulary |

## 4. First run

```bat
tandem-serve.bat
```

A healthy start prints, among others:

```
tqllm: layer split: layers 0-43 on GPU 1, 44-63, the head and the drafter on GPU 0
tqserve: token embeddings in host memory (0.72 GB of VRAM freed)
tqserve: listening on 127.0.0.1:8086 (model=tqllm, context=524288)
```

Loading takes about a minute. Then check it answers:

```bat
curl http://127.0.0.1:8086/v1/chat/completions -H "content-type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":20}"
```

The read-only monitor (live prefill and decode rates, GPU use) is at `http://127.0.0.1:8086/monitor`.

## 5. Choosing the split and the context

Set these before `tandem-serve.bat` (or pass the same flags to `tqserve.exe` yourself):

| Variable | Default | Meaning |
|---|---|---|
| `TQLLM_SPLIT` | `44` | layers `0 .. K-1` on CUDA device 1, the rest + head + drafter on device 0 |
| `TANDEM_CONTEXT` | `524288` | context tokens; above 262,144 the script adds YaRN x2 (x4 above 524,288) |
| `TANDEM_MODEL` | `models\Qwen3.8-27B-TQ_L.gguf` | or pass the model as the first argument |
| `TANDEM_PORT` | `8086` | the API port |
| `CUDA_VISIBLE_DEVICES` | `0,1` | which two cards, and their order |

Rules of thumb:

- **Device 0 is the home card**: it holds the head, the drafter and usually your desktop. Give device 1 more layers
  (a larger `K`). If device 0 runs out of memory, raise `K`; if device 1 does, lower it.
- **Every fourth layer holds KV**, so `K` also decides how the context's memory divides between the cards.
- KV per token with `--kv tq3` is 14.3 KB for Qwen3.8-27B: 3.8 GB at 262,144 tokens, 7.5 GB at 524,288.

Measured on 2x RTX 5060 Ti 16 GB with TQ_L:

| Context | `TQLLM_SPLIT` | VRAM device 0 / device 1 |
|--:|--:|---|
| 131,072 | 40 | 13.7 / 11.0 GB |
| 262,144 | 40 | 14.1 / 11.3 GB |
| 524,288 (YaRN x2) | 44 | 12.7 / 14.9 GB |

1,048,576 tokens does not fit 2x 16 GB with TQ_L: its KV alone is 15 GB.

## 6. Use it with apps and coding agents

**One command for a terminal coding agent**: `tandem-agent.bat` starts the split server through tqLLM's launcher
(`tools/launch/tqllm.py`) and opens the agent already pointed at it; when the agent exits, the server stops.

```bat
tandem-agent.bat claude     & rem Claude Code (Anthropic Messages API)
tandem-agent.bat codex      & rem Codex (Responses API)
tandem-agent.bat opencode   & rem or pi
tandem-agent.bat serve      & rem only the server, until Ctrl+C
```

It needs Python 3 and the agent on `PATH`, and reads the same variables as `tandem-serve.bat`. The agent's own
configuration files are never written: it gets the local server through its environment and command line. (Checked:
the launcher brings up the split server and both the Messages and Responses APIs answer; the interactive agents
themselves were not exercised in testing.)

`tqserve` speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, streaming, tools) and the
Anthropic Messages API (`/v1/messages`), so other clients only need a base URL:

| Client | Setting |
|---|---|
| OpenAI SDKs, Open WebUI, Continue, Cline, Aider, Codex | base URL `http://127.0.0.1:8086/v1`, any API key text unless you set `--api-key` |
| Claude Code and other Anthropic clients | `ANTHROPIC_BASE_URL=http://127.0.0.1:8086` |
| Agents that size their context from llama.cpp's `GET /props` | works: `/props` reports `n_ctx` |

More detail for agents (thinking control, tool calls, long sessions) is in upstream tqLLM's
[coding agents guide](coding-agents.md). Two Tandem-specific tips:

- Keep `--prompt-cache on` (the script does): each turn re-reads only what is new, not the whole conversation.
- A very long first prompt is slow to read (a 459,444-token prompt took 22 minutes); conversations that grow turn by
  turn do not pay that cost again.

## 7. Security

- `tqserve` listens on `127.0.0.1` only. Leave it that way unless you need another machine to reach it.
- To serve a LAN, add `--host 0.0.0.0 --api-key <secret>` (or `--api-key-file`). **Never expose it without an API key**:
  anyone who reaches the port can run prompts on your GPUs.
- Only use `tqserve.exe` and the CUDA libraries from this repository's releases, and check their SHA-256 first. The
  bundled CUDA DLLs are NVIDIA's signed redistributables (right-click > Properties > Digital Signatures shows
  "NVIDIA Corporation").

## 8. Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `TQLLM_SPLIT: needs two CUDA devices; this process sees 1` | `CUDA_VISIBLE_DEVICES` lists one card, or the driver sees one. Set `CUDA_VISIBLE_DEVICES=0,1`. |
| `TQLLM_SPLIT: all KV on the GPUs so far: pass --kv-host 0` | a host KV tail is not supported under the split yet. Add `--kv-host 0`. |
| `TQLLM_SPLIT: one slot so far: pass --np 1` | the split serves one request at a time for now. |
| `TQLLM_SPLIT: no --kv-compact yet` / `no --mtp yet` / `dense models only` | not supported under the split yet; drop the option or run without `TQLLM_SPLIT`. |
| `model does not fit in device memory` | lower `TANDEM_CONTEXT`, or move layers between the cards with `TQLLM_SPLIT`; close GPU-heavy apps on device 0. |
| `cublas64_13.dll was not found` (or cuBLASLt / cuSOLVER / cuSPARSE) | unzip `cudart-tandem-cuda13-win-x64.zip` next to `tqserve.exe`, or put the CUDA 13 toolkit's `bin\x64` on `PATH`. |
| The program exits at once with a CUDA driver error | update the NVIDIA driver to R580 or newer. |
| Answers differ from a one-card run | that should never happen: run `tools/tandem_golden.py` on a model that fits one card and open an issue with the result. |

## 9. FAQ

**Why a layer split and not tensor parallelism?** Tensor parallelism synchronizes the cards twice per layer; over plain
PCIe without peer access that costs more than it gains on consumer cards. A layer split sends one small set of rows per
verify block, and with captured graphs the split costs almost nothing (196–200 tok/s split against 205 on one card).

**Can I mix two different cards?** Not tested. It should work if both are supported by the build (RTX 30 / 40 / 50); put
the card with more free memory as device 1 or adjust `K`.

**More than two cards?** Not yet: it is on the [roadmap](../../README.md#roadmap).

**Does the split change the model's answers?** No. Greedy output is identical to one card, token for token; that is the
project's correctness test.

**Is TQ_L as good as Q6?** Close: 19/20 against 20/20 on a 20-task coding check, with 8x the context of the llama.cpp
Q6 setup it was compared with. Inside an agent loop that runs tests and fixes its own code, the gap matters less.
