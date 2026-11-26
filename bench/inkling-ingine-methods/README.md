# Testing Ingine CUDA techniques inside Tandem

The Inkling pack backend in `tqserve` links Basalt's model library. Tandem owns the request, the reusable prompt prefix and the speculative decoding loop. Each experiment below replaces one part of the linked library and keeps the pack, tokenizer, drafter, KV format and client request fixed. An external Ingine server does not participate in these measurements.

## Variants

| Patch | Change inside the linked model |
|---|---|
| `fused.patch` | Pair the gate and up projections for each expert feature tile. Apply softcapped SwiGLU and write the quantized activation in the same kernel. Keep Basalt's NVFP4 matrices and dot-product order. |
| `bf16-native.patch` | Group up to eight routed tokens per tile. Expand activations to BF16, convert packed weights with the Blackwell E2M1 instruction, and run BF16 tensor-core MMA for gate and up. |
| `bf16-lut-after-native.patch` | Replace the native weight conversion in the BF16 experiment with a lookup implementation. Apply this patch on top of `bf16-native.patch`. |
| `stream.patch` | Stream the two aligned NVFP4 expert weight words through 32-bit loads with L1 no-allocation and an L2 prefetch hint. Ingine reserves its newer L2 eviction hint for toolkits from 13.4. |
| `attention-16.patch` | Use up to sixteen parallel time slices per KV head for long FP8 global attention. Preserve Basalt's cache layout, partial buffers and merge kernel. This tests the scheduling choice independently of Ingine's cluster-wide fusion. |

The implementation adapts the methods to Basalt's buffers. The upstream implementations and the pinned integration are available in the [Ingine MoE kernel](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/blob/556acc7cf42749536684854a6df820e41258f929/native/src/kernels/moe/moe.cu), [Ingine attention kernel](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/blob/556acc7cf42749536684854a6df820e41258f929/native/src/kernels/attn/attn.cu), [Ingine cache policy](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/blob/556acc7cf42749536684854a6df820e41258f929/native/src/kernels/common/l2.cuh) and [Basalt source](https://github.com/jesdga95/basalt/tree/0810b58d54eda7517d96fe1e0a8dd1556c8512ec).

## Build one experimental executable

Use a separate Basalt checkout for each variant at commit `0810b58d54eda7517d96fe1e0a8dd1556c8512ec`. The measured Tandem base was `f8af10e306aa921872768debc91451d40c744486`. From the Tandem root, substitute absolute checkout and patch paths as appropriate.

```sh
git -C /path/to/basalt apply /path/to/tandem/bench/inkling-ingine-methods/fused.patch
cmake -S . -B build-fused -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DTQLLM_BASALT_SOURCE=/path/to/basalt
cmake --build build-fused --target tqserve
```

For native BF16, apply `bf16-native.patch` instead and configure a fresh `build-native` directory. The following two commands compile only Basalt's expert translation unit for SM120a. The lookup variant uses the ordinary SM120 build after applying both BF16 patches.

```sh
python bench/inkling-ingine-methods/native_ninja.py build-native
ninja -C build-native -f inkling-native.ninja tqserve
```

## Measurement

Two RTX 5060 Ti cards with 16 GB each served a single greedy SQLite coding task on Windows, CUDA 13.2, with the long-context Inkling pack, FP8 KV, four DSpark proposals and thinking disabled. The runtime split the 24 layers evenly across the cards. Each independent server start received one cold post-compaction request followed by three alternating pairs of original-tail and compacted-tail requests. The unchanged prefix contained 262,144 tokens, the compacted request contained 262,216 tokens, and the client timed each request from submission until receipt of the full response.

Decode rates come from the server's `timings.predicted_per_second` field. Each warm median uses the three compacted-tail requests with profiling disabled. A separate Nsight Systems 2026.6.1 run checks actual GPU execution. The 12 checks execute the generated SQL against SQLite and compare it with an independent oracle. They exercise one coding task. The hashes and timings allow comparisons across binaries without publishing the private request.

## Results on the 256K task

| Method | Median warm decode (tok/s) | Median warm wall (s) | Cold wall (s) | Lowest SQLite score across seven requests |
|---|---:|---:|---:|---:|
| Basalt linked control | 86.30 | 2.016 | 136.12 | 12/12 |
| Streaming expert weights | 86.48 | 1.890 | 138.51 | 12/12 |
| Fused gate/up and SwiGLU | 71.14 | 2.273 | 137.35 | 12/12 |
| BF16 MMA, lookup | 69.57 | 2.263 | 137.17 | 12/12 |
| BF16 MMA, native FP4 | 79.74 | 2.006 | 138.27 | 12/12 |
| 16 global attention splits | 83.84 | 1.933 | 139.78 | 12/12 |
| Control restart | 87.68 | 1.877 | 138.14 | 12/12 |
| Streaming restart | 85.88 | 1.886 | 138.94 | 12/12 |

Each variant produced the same SQL bytes as the control for the corresponding request and passed all 12 SQLite cases. Across six warm trials from two independent starts per binary, the median was 86.66 tok/s for the control and 85.90 tok/s for streaming (-0.88%). Their decode ranges overlapped. Pooled median end-to-end latency was 1.970 s for the control and 1.888 s for streaming. Streaming had lower latency on the first start and higher latency on the second. The serving kernel remains the profiling preview build.

Within 40 warm decode steps, Nsight captured 1,920 calls to the new BF16 expert kernel, 960 per GPU. Their summed duration was 474.2 ms. Global attention used 64 splits and 320 launches over 40 control steps. The 16-split variant used 328 launches over 41 steps. DSpark accepted 84/160 proposals with the control and 83/164 with 16 splits, producing identical final SQL. Global attention kernels averaged 1.295 ms and 1.319 ms per launch, respectively. Kernel durations may overlap other work. [Per-trial measurements](measurements.json).
