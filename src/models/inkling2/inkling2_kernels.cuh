#pragma once
// Inkling-2-Mini kernels (see inkling2.cu). FP16 activations, FP32 accumulate. The reference is ref/inkling2.py
// (float64); the engine (inkling2.cpp) orchestrates the multi-stage ops (projections via cuBLAS, then these kernels).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {
// y = x * rsqrt(mean(x^2) + eps) * w   (w null: weightless)
void ik2_rms_norm(const __half * x, const float * w, float eps, int C, __half * y, cudaStream_t s);
// out[nq][C] = softmax(q . s / sqrt(C)) over the normed states [blocks..., x], weighted sum of the normed states
// (states RMS-normed weightless, eps 1e-6)
void ik2_attn_res(const __half * blocks, int n_blocks, const __half * x, const float * q, int nq, int C, __half * out, cudaStream_t s);
// y[t][c] = sum_j w[j][c] x[t - (taps-1-j)]  (w [taps][C] folded, j = taps-1 the current token; x[k<0] = 0)
void ik2_shortconv(const float * w, int taps, const __half * x, __half * y, int C, int T, cudaStream_t s);
// out = (1 - tanh(forget g)) x + tanh(gain g) msg,  g = down @ [rms(x); rms(msg)]  (down [64][2C], forget/gain [C][64])
void ik2_control(const __half * x, const __half * msg, const float * down, const float * forget, const float * gain,
                 int C, __half * out, cudaStream_t s);
// q [H][D] -> L2-clip x clip[head][d] (cap), in place; one row of H heads
void ik2_l2clip(const __half * x, const float * clip, float cap, int H, int D, __half * y, cudaStream_t s);
// v [KV][D] -> weightless RMS norm (eps 1e-6), in place
void ik2_rms_weightless(const __half * x, int KV, int D, __half * y, cudaStream_t s);
// The relative-bias attention core for one query token at position pos. q [H][D] (clipped), k/v [KV][D] (clipped /
// normed), r [H][d_rel] (clipped), gate/sink [H], rel_bias [d_rel][extent], KV cache [context][KV][D] FP16.
// Writes out [H*D] = (softmax over the causal(+window) keys, with the relative bias and a per-head sink logit) v,
// times sigmoid(gate). The KV cache row for pos is written by the caller.
struct Ik2AttnCore {
    const __half * q, * k, * v, * r, * gate, * sink;
    const float * rel_bias;
    int H, KV, D, d_rel, extent, window;
    bool sliding;
    uint32_t pos;
    const __half * kc;  // [context][KV][D] FP16 (the keys before pos)
    const __half * vc;  // [context][KV][D] FP16
};
void ik2_attention_core(const Ik2AttnCore & a, __half * out, cudaStream_t s);
// logits[i] = cap * tanh(logits[i] / cap) for i < unpadded
void ik2_softcap(float * logits, int unpadded, float cap, cudaStream_t s);
}  // namespace tqllm_cuda
