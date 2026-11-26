# Inkling packs on Blackwell

The optional CUDA build links the Basalt Inkling model library into
`tqserve`. It accepts the Inkling NVFP4/MXFP8 `.basalt` pack, a matching Inkling GGUF
tokenizer, one slot, and one or two visible SM120 devices. On two cards the backend
splits layers according to free VRAM. Use `--split-at N` to move the boundary, or
`--split experts` to keep the attention layers on the first card while the second
card handles experts. The GGUF backend remains available in the same build.

The Windows SM120 binary requires an NVIDIA driver compatible with CUDA 13.2, the
CUDA 13.2 cuBLAS and cuSOLVER libraries on PATH, and the Microsoft Visual C++ x64 runtime.
The pack contains the weights. The matching GGUF supplies the tokenizer and chat template.

```sh
git clone https://github.com/jesdga95/basalt
git -C basalt checkout --detach 0810b58d54eda7517d96fe1e0a8dd1556c8512ec
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DTQLLM_BASALT_SOURCE=/absolute/path/to/basalt
cmake --build build --config Release --target tqserve
tqserve --model Inkling-2-Mini-NVFP4-MX8.basalt --tokenizer Inkling-2-Mini-UD-Q4_K_XL.gguf --kv fp8 --exact off --context 264192 --prompt-cache on --host 127.0.0.1 --port 19558
```

Generation defaults to plain decode. For greedy requests, `--mtp --draft-max 3`
enables the DSpark drafter shipped in the same pack. The target model verifies
each proposed token. Set `TQLLM_INKLING2_SAMPLED_DRAFT=1` to use the drafter
at nonzero temperature. Each verifier row draws from the target distribution using
the seed and its output position. A mismatch commits the sampled target token and
discards the remaining proposals. By default, sampled requests use single-token
decoding. Repetition, frequency and presence penalties also select that path. The API returns
the proposal and acceptance counts in `timings.draft_n` and `timings.draft_n_accepted`.

Draft limits from one to seven use fixed DSpark windows. A larger limit enables
adaptive verification, using the drafter's confidence and the elapsed time of
previous verification rounds within the current request. The embedded drafter
predicts at most seven tokens. With `--lookup 15`, a repeated suffix can propose
up to 15 tokens from the current prompt or from tokens already generated in this
request. The minimum matching suffix is eight tokens, adjustable with
`--lookup-first`. Each proposal still passes through the target model.

For long-context greedy coding, the optional local-draft pack limits DSpark's
middle proposal layer to the same 1,024-token window as its other two layers.
The target model verifies every proposed token against the full context. The
conversion copies the original tensor sections and checks their digests.
It needs disk space for one additional pack and a volume supporting hard links.

```sh
python tools/inkling_local_draft.py Inkling-2-Mini-NVFP4-MX8.basalt Inkling-MX8-local-draft.basalt --basalt-source /absolute/path/to/basalt
```

Start the new pack with `--mtp --draft-max 4`. The shorter proposal window can
change the acceptance rate. Compare decode time and task results on your workload.

Prefill uses the pack's batched CUDA kernels. With
`--prompt-cache on`, Tandem saves sliding attention rings and convolution state for
both devices in host memory. Full attention KV remains on its owning device.
Tandem checkpoints roughly every quarter of the context, at the end of the longest
user turn of at least 4,096 tokens, and at the prompt tail. Only checkpoints preceding the first differing
token remain eligible. Generated text is never returned from cache.

Read `prefill_cached_tokens` per request from `/api/monitor/v1/snapshot`. A compactor
that moves a summary in front of a retained document changes the causal prefix.
The document then requires fresh prefill even if its text is identical. Redistribute
the optional build with Basalt's LICENSE and THIRD_PARTY_LICENSES.

To exercise compaction, start an idle server with at least 16,384 context and the
cache enabled. Set `TQSERVE_INKLING_PACK_TEST_URL` to its local URL and run
`python tests/test_inkling_pack_cache_live.py`. The test changes the history after
a long user turn, checks the monitor's reuse counter, and compares the answer to a
fresh prefill of the same compacted request.

### Profiling a warm decode

On Windows SM120 with CUDA 13.2 we captured node-level traces with Nsight
Systems 2026.6.1. Version 2025.6.3 caused an illegal CUDA memory access during
the cold 256K request on the same binary and machine. Use a named Nsight
session, launch the server with `nsys launch --session-new=inkling -t cuda,nvtx
--cuda-graph-trace=node tqserve.exe <server arguments>`, and warm the desired
prefix. Start capture with `nsys start --session=inkling --sample=none
--cpuctxsw=none -o warm`. Send one post-compaction request, wait for its reply,
then run `nsys stop --session=inkling`. Export with
`nsys export --type=sqlite -o warm.sqlite warm.nsys-rep`, then run
`python tools/inkling_trace.py warm.sqlite`.

The trace marks prefill and first token, individual decode steps, DSpark
proposals, and target verification. Phase times include their nested ranges.
GPU kernel durations can overlap. The summary also reports the time with at
least one active kernel. Measure decode throughput without the profiler.
