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
// y[t][c] = sum_j w[j][c] x[t - (taps-1-j)]  (w [C][taps] folded, j = taps-1 the current token; x[k<0] = 0)
// hist [taps-1][C] holds the previous taps-1 tokens (hist[0] = oldest); nullptr for the first token.
void ik2_shortconv(const float * w, int taps, const __half * x, __half * y, int C, int T, const __half * hist, cudaStream_t s);
// Full-layer dynamic conv: y[c] = sum_j w[c][j] x[c-(taps-1-j)] + sum_lag dyn[kvh(c)][lag] x[c-lag]
// w [C][taps] folded, dyn [KV][taps] (per kv-head, per lag), hist [taps-1][C]
void ik2_shortconv_dyn(const float * w, int taps, const __half * x, __half * y, int C, int D, const float * dyn, const __half * hist, cudaStream_t s);
// out = (1 - tanh(forget g)) x + tanh(gain g) msg,  g = down @ [rms(x); rms(msg)]  (down [64][2C], forget/gain [C][64])
void ik2_control(const __half * x, const __half * msg, const float * down, const float * forget, const float * gain,
                 int C, __half * out, cudaStream_t s);
// q [H][D] -> L2-clip x clip[head][d] (cap), in place; one row of H heads
void ik2_l2clip(const __half * x, const float * clip, float cap, int H, int D, __half * y, cudaStream_t s);
// v [KV][D] -> weightless RMS norm (eps 1e-6), in place
void ik2_rms_weightless(const __half * x, int KV, int D, __half * y, cudaStream_t s);
// The relative-bias attention core for one query token at position pos. q [H][D] (clipped), k/v [KV][D] (clipped /
// normed), r [H][d_rel] (clipped), gate [H][D], sink [H], rel_bias [extent][d_rel], KV cache [context][KV][D] FP16.
// Writes out [H*D] = (softmax over the causal(+window) keys, with the relative bias and a per-head sink logit) v,
// times sigmoid(gate). The KV cache row for pos is written by the caller.
struct Ik2AttnCore {
    const __half * q, * r, * gate, * sink;
    const float * rel_bias;
    int H, KV, D, d_rel, extent, window;
    bool sliding;
    uint32_t pos;
    const __half * kc;  // [context][KV][D] FP16 (sliding visits use a window-sized ring)
    const __half * vc;  // [context][KV][D] FP16
};
void ik2_attention_core(const Ik2AttnCore & a, __half * out, cudaStream_t s);
// Inkling's sigmoid router picks by biased probability and weights by the original probability.
void ik2_route(const float * logits, const float * bias, int experts, int count, float scale,
               int * ids, float * weights, cudaStream_t stream);
void ik2_expert_activation(const __nv_bfloat16 * gate_up, const int * ids, const float * scales, int count,
                           int width, float gate_cap, float up_cap, __half * activated, cudaStream_t stream);
void ik2_expert_sum(const __nv_bfloat16 * rows, const int * ids, const float * scales, const float * weights,
                    int count, int width, __half * output, cudaStream_t stream);
// logits[i] = cap * tanh(logits[i] / cap) for i < unpadded
void ik2_softcap(float * logits, int unpadded, float cap, cudaStream_t s);
// hist [taps-1][C] <- shift + append cur [C]
void ik2_conv_hist_update(__half * hist, const __half * cur, int taps, int C, cudaStream_t s);
// y[i] = a[i] + b[i]  (device-resident residual add)
void ik2_hadd(const __half * a, const __half * b, __half * y, int n, cudaStream_t s);
// y[i] = a[i]  (device copy)
void ik2_hcopy(const __half * a, __half * y, int n, cudaStream_t s);
// y[i] *= s  (in place, device-resident scalar scale)
void ik2_hscale(__half * y, float scale, int n, cudaStream_t s);
}  // namespace tqllm_cuda
