# Inkling packs on Blackwell

The optional Windows/Linux CUDA build links the Basalt 0.5.7 Inkling model library into
`tqserve`. It accepts the Inkling NVFP4/MXFP8 `.basalt` pack, a matching Inkling GGUF
tokenizer, one slot, and one or two visible SM120 devices. On two cards the backend
splits layers according to free VRAM. The GGUF backend remains available in the same build.

The Windows SM120 binary requires an NVIDIA driver compatible with CUDA 13.2, the
CUDA 13.2 cuBLAS and cuSOLVER libraries on PATH, and the Microsoft Visual C++ x64 runtime.
The pack contains the weights. The matching GGUF supplies the tokenizer and chat template.

```sh
git clone --branch v0.5.7 --depth 1 https://github.com/jesdga95/basalt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DTQLLM_BASALT_SOURCE=/absolute/path/to/basalt
cmake --build build --config Release --target tqserve
tqserve --model Inkling-2-Mini-NVFP4-MX8.basalt --tokenizer Inkling-2-Mini-UD-Q4_K_XL.gguf --kv fp8 --exact off --context 264192 --prompt-cache on --host 127.0.0.1 --port 19558
```

Generation uses plain decode. Prefill uses the pack's batched CUDA kernels. With
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
