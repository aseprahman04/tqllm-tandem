// Inkling-2-Mini kernels (see inkling2_kernels.cuh). FP16 activations, FP32 accumulate. The reference is
// ref/inkling2.py (float64). The engine (inkling2.cpp) orchestrates the multi-stage ops; these are the self-contained
// pieces it calls.
#include "inkling2_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <stdexcept>
#include <string>

namespace tqllm_cuda {
namespace {

// A launch the runtime refuses (a stream of another device, a bad launch configuration) leaves its error in the
// thread's CUDA error state and nowhere else. Every launcher below reads it right after its launch, so the message
// names this file and the launcher's line. (An earlier kernel that faulted while running can surface here as well:
// TQLLM_INKLING2_SYNC=1 in inkling2.cpp pins those to the group that launched them.)
void launched_at(int line) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("inkling2: ") + cudaGetErrorString(e) + " (inkling2.cu:" + std::to_string(line) + ")");
}
#define launched() launched_at(__LINE__)

constexpr float kResEps = 1e-6f;  // the attention residual's norm (transformers' Inkling2UnweightedRMSNorm default)

// block-wide sum of a per-thread partial (one block)
__device__ float block_sum(float v) {
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = 0.f;
        for (int i = 0; i < (int(blockDim.x) + 31) / 32; ++i) r += red[i];
        red[0] = r;
    }
    __syncthreads();
    return red[0];
}

// ---- RMS norm ----
__global__ void rms_norm_k(const __half * __restrict__ x, const float * __restrict__ w, float eps, int C, __half * __restrict__ y) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) { float v = __half2float(x[i]); ss += v * v; }
    const float scale = rsqrtf(block_sum(ss) / C + eps);
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = __half2float(x[i]) * scale;
        y[i] = __float2half(w ? v * w[i] : v);
    }
}

// ---- attention residual ----
__global__ void attn_res_k(const __half * __restrict__ blocks, int n_blocks, const __half * __restrict__ x,
                           const float * __restrict__ q, int nq, int C, __half * __restrict__ out) {
    const int qi = blockIdx.x;
    const int n = n_blocks + 1;
    __shared__ float scores[128];  // n <= 128
    __shared__ float norms[128];
    for (int si = 0; si < n; ++si) {
        const __half * st = (si < n_blocks) ? blocks + size_t(si) * C : x;
        float ss = 0.f, dot = 0.f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float value = __half2float(st[c]);
            ss += value * value;
            dot += q[qi * C + c] * value;
        }
        const float sum = block_sum(ss);
        __syncthreads();
        const float projection = block_sum(dot);
        if (threadIdx.x == 0) {
            norms[si] = rsqrtf(sum / C + kResEps);
            scores[si] = projection * norms[si] * rsqrtf(float(C));
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = scores[0], z = 0.f;
        for (int si = 1; si < n; ++si) m = fmaxf(m, scores[si]);
        for (int si = 0; si < n; ++si) {
            scores[si] = expf(scores[si] - m);
            z += scores[si];
        }
        for (int si = 0; si < n; ++si) scores[si] *= norms[si] / z;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float value = 0.f;
        for (int si = 0; si < n; ++si) {
            const __half * st = si < n_blocks ? blocks + size_t(si) * C : x;
            value += scores[si] * __half2float(st[c]);
        }
        out[qi * C + c] = __float2half(value);
    }
}

// ---- short conv with history ----
// w [C][taps] folded; hist [taps-1][C] (hist[0] = oldest, hist[taps-2] = most recent); nullptr = first token.
__global__ void shortconv_k(const float * __restrict__ w, int taps, const __half * __restrict__ x, __half * __restrict__ y,
                            int C, int T, const __half * __restrict__ hist) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * C;
    if (idx >= total) return;
    const int t = idx / C, c = idx % C;
    float acc = 0.f;
    for (int j = 0; j < taps; ++j) {
        const int lag = taps - 1 - j;  // 0 = current token, 1 = previous, ...
        float xv;
        if (lag == 0) {
            xv = __half2float(x[size_t(t) * C + c]);
        } else {
            if (!hist) continue;
            xv = __half2float(hist[size_t(lag - 1) * C + c]);
        }
        acc += w[c * taps + j] * xv;
    }
    y[idx] = __float2half(acc);
}

// ---- full-layer dynamic conv ----
// w [C][taps] folded (static), dyn [KV][taps] (dynamic taps: per kv-head, per lag; lag 0 = current token).
// Each channel c belongs to kv-head c/D; its dynamic taps are dyn[kvh*taps + lag].
__global__ void shortconv_dyn_k(const float * __restrict__ w, int taps, const __half * __restrict__ x, __half * __restrict__ y,
                                int C, int D, const float * __restrict__ dyn, const __half * __restrict__ hist) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int kvh = c / D;
    float acc = 0.f;
    for (int j = 0; j < taps; ++j) {
        const int lag = taps - 1 - j;
        float xv;
        if (lag == 0) xv = __half2float(x[c]);
        else { if (!hist) continue; xv = __half2float(hist[size_t(lag - 1) * C + c]); }
        acc += w[c * taps + j] * xv;
    }
    for (int lag = 0; lag < taps; ++lag) {
        float xv;
        if (lag == 0) xv = __half2float(x[c]);
        else { if (!hist) continue; xv = __half2float(hist[size_t(lag - 1) * C + c]); }
        acc += rsqrtf(float(2 * taps + 1)) * dyn[kvh * taps + lag] * xv;
    }
    y[c] = __float2half(acc);
}

// ---- control: g = down @ [rms(x); rms(msg)]; out = (1 - tanh(forget g)) x + tanh(gain g) msg ----
__global__ void control_k(const __half * __restrict__ x, const __half * __restrict__ msg, const float * __restrict__ down,
                          const float * __restrict__ forget, const float * __restrict__ gain, int C, __half * __restrict__ out) {
    // phase 1: 64 threads compute g[64]
    __shared__ float g[64];
    if (threadIdx.x < 64) {
        const int i = threadIdx.x;
        float gx = 0.f, gm = 0.f;
        for (int j = 0; j < C; ++j) { float a = __half2float(x[j]); gx += a * a; float b = __half2float(msg[j]); gm += b * b; }
        const float sx = rsqrtf(gx / C + kResEps), sm = rsqrtf(gm / C + kResEps);
        float s = 0.f;
        for (int j = 0; j < C; ++j) s += down[i * 2 * C + j] * (__half2float(x[j]) * sx) + down[i * 2 * C + C + j] * (__half2float(msg[j]) * sm);
        g[i] = s;
    }
    __syncthreads();
    // phase 2: C threads combine
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float f = 0.f, gn = 0.f;
        for (int i = 0; i < 64; ++i) {
            f += forget[c * 64 + i] * g[i];
            gn += gain[c * 64 + i] * g[i];
        }
        out[c] = __float2half((1.f - tanhf(f)) * __half2float(x[c]) + tanhf(gn) * __half2float(msg[c]));
    }
}

// ---- L2 clip: x[head][d] * (cap / (norm + 1e-6)).clamp(max=1) * clip[head][d] ----
__global__ void l2clip_k(const __half * __restrict__ x, const float * __restrict__ clip, float cap, int H, int D, __half * __restrict__ y) {
    const int h = blockIdx.x;
    if (h >= H) return;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = __half2float(x[h * D + i]) * (clip ? clip[h * D + i] : 1.f);
        ss += v * v;
    }
    const float scale = fminf(1.f, cap / (sqrtf(block_sum(ss)) + 1e-6f));
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        y[h * D + i] = __float2half(__half2float(x[h * D + i]) * (clip ? clip[h * D + i] : 1.f) * scale);
}

// ---- weightless RMS norm (v) ----
__global__ void rms_wl_k(const __half * __restrict__ x, int KV, int D, __half * __restrict__ y) {
    const int h = blockIdx.x;
    if (h >= KV) return;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float v = __half2float(x[h * D + i]); ss += v * v; }
    const float sc = rsqrtf(block_sum(ss) / D + kResEps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) y[h * D + i] = __float2half(__half2float(x[h * D + i]) * sc);
}

// ---- the attention core: one query token at position pos ----
// q [H][D] clipped, k/v [KV][D] (the current token's, conv'd + clipped/normed), r [H][d_rel] clipped, gate/sink [H],
// rel_bias [d_rel][extent], KV cache [context][KV][D] FP16. out [H*D] = (softmax over causal(+window) keys with the
// relative bias and a per-head sink logit) v, times sigmoid(gate).
__global__ void attn_core_k(Ik2AttnCore a, __half * __restrict__ out) {
    const int h = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kv = h / (a.H / a.KV), end = int(a.pos) + 1;
    const int begin = a.sliding ? max(0, end - a.window) : 0;
    constexpr int warps = 8;
    __shared__ float scores[warps], weights[warps], old_weight, denominator;
    float value = 0.f, maximum = __half2float(a.sink[h]);
    if (threadIdx.x == 0) denominator = 1.f;  // the sink has zero value
    for (int base = begin; base < end; base += warps) {
        const int t = base + warp;
        float dot = 0.f, bias = 0.f;
        if (t < end) {
            const int row = a.sliding ? t % a.window : t;
            const size_t k = (size_t(row) * a.KV + kv) * a.D;
            for (int d = lane; d < a.D; d += 32)
                dot += __half2float(a.q[h * a.D + d]) * __half2float(a.kc[k + d]);
            const int dist = int(a.pos) - t;
            if (dist < a.extent)
                for (int r = lane; r < a.d_rel; r += 32)
                    bias += __half2float(a.r[h * a.d_rel + r]) * a.rel_bias[size_t(dist) * a.d_rel + r];
        }
        float score = dot * rsqrtf(float(a.D)) + bias;
        for (int o = 16; o; o >>= 1) score += __shfl_down_sync(0xffffffffu, score, o);
        if (lane == 0) scores[warp] = t < end ? score : -CUDART_INF_F;
        __syncthreads();
        if (threadIdx.x == 0) {
            float next = maximum;
            for (int w = 0; w < warps; ++w) next = fmaxf(next, scores[w]);
            old_weight = expf(maximum - next);
            maximum = next;
            denominator *= old_weight;
            for (int w = 0; w < warps; ++w) {
                weights[w] = expf(scores[w] - next);
                denominator += weights[w];
            }
        }
        __syncthreads();
        const int d = threadIdx.x;
        if (d < a.D) {
            value *= old_weight;
            for (int w = 0; w < warps && base + w < end; ++w) {
                const int row = a.sliding ? (base + w) % a.window : base + w;
                value += weights[w] * __half2float(a.vc[(size_t(row) * a.KV + kv) * a.D + d]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < a.D) {
        const int i = h * a.D + threadIdx.x;
        out[i] = __float2half(value / denominator / (1.f + expf(-__half2float(a.gate[i]))));
    }
}

// ---- conv history update: shift and append ----
// hist [taps-1][C] half; cur [C] half (current token's input to the conv)
// After: hist[0] = cur, hist[i] = old hist[i-1] (i >= 1)
__global__ void conv_hist_update_k(__half * __restrict__ hist, const __half * __restrict__ cur, int taps, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    for (int i = taps - 2; i > 0; --i)
        hist[size_t(i) * C + c] = hist[size_t(i - 1) * C + c];
    hist[c] = cur[c];
}

// ---- softcap ----
__global__ void softcap_k(float * __restrict__ logits, int unpadded, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= unpadded) return;
    logits[i] = cap * tanhf(logits[i] / cap);
}

// ---- device-resident residual ops (the forward must not touch activations on the host) ----
// y[i] = a[i] + b[i]
__global__ void hadd_k(const __half * __restrict__ a, const __half * __restrict__ b, __half * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __hadd(a[i], b[i]);
}
// y[i] = a[i] (device copy)
__global__ void hcopy_k(const __half * __restrict__ a, __half * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i];
}
// y[i] = a[i] * s (in place)
__global__ void hscale_k(__half * __restrict__ y, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(__half2float(y[i]) * s);
}

__global__ void route_k(const float * logits, const float * bias, int experts, int count, float scale,
                        int * ids, float * weights) {
    extern __shared__ float candidates[];
    float * probability = candidates + experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        probability[e] = 1.f / (1.f + expf(-logits[e]));
        candidates[e] = probability[e] + (bias ? bias[e] : 0.f);
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    float sum = 0.f;
    for (int i = 0; i < count; ++i) {
        int best = 0;
        for (int e = 1; e < experts; ++e) if (candidates[e] > candidates[best]) best = e;
        ids[i] = best;
        sum += probability[best];
        candidates[best] = -CUDART_INF_F;
    }
    for (int i = 0; i < count; ++i) weights[i] = scale * probability[ids[i]] / (sum + 1e-20f);
}

__global__ void expert_activation_k(const __nv_bfloat16 * gu, const int * ids, const float * scales,
                                    int width, float gate_cap, float up_cap, __half * out) {
    const int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;
    const float s = scales ? scales[ids[i]] : 1.f;
    const float g = __bfloat162float(gu[(2*i)*width + j]) * s;
    const float u = __bfloat162float(gu[(2*i+1)*width + j]) * s;
    out[i*width+j] = __float2half(tanhf(g/gate_cap)*gate_cap / (1.f+expf(-g)) * tanhf(u/up_cap)*up_cap);
}

__global__ void expert_sum_k(const __nv_bfloat16 * rows, const int * ids, const float * scales, const float * weights,
                             int count, int width, __half * out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;
    float sum = 0.f;
    for (int i = 0; i < count; ++i)
        sum += weights[i] * (scales ? scales[ids[i]] : 1.f) * __bfloat162float(rows[i*width+j]);
    out[j] = __float2half(sum);
}

}  // namespace

void ik2_rms_norm(const __half * x, const float * w, float eps, int C, __half * y, cudaStream_t s) {
    int threads = std::min(1024, C < 32 ? 32 : 256);
    rms_norm_k<<<1, threads, 0, s>>>(x, w, eps, C, y);
    launched();
}

void ik2_attn_res(const __half * blocks, int n_blocks, const __half * x, const float * q, int nq, int C, __half * out, cudaStream_t s) {
    int threads = std::min(1024, C < 32 ? 32 : 256);
    attn_res_k<<<nq, threads, 0, s>>>(blocks, n_blocks, x, q, nq, C, out);
    launched();
}

void ik2_shortconv(const float * w, int taps, const __half * x, __half * y, int C, int T, const __half * hist, cudaStream_t s) {
    const int total = T * C;
    shortconv_k<<<(total + 255) / 256, 256, 0, s>>>(w, taps, x, y, C, T, hist);
    launched();
}

void ik2_shortconv_dyn(const float * w, int taps, const __half * x, __half * y, int C, int D, const float * dyn, const __half * hist, cudaStream_t s) {
    shortconv_dyn_k<<<(C + 255) / 256, 256, 0, s>>>(w, taps, x, y, C, D, dyn, hist);
    launched();
}

void ik2_control(const __half * x, const __half * msg, const float * down, const float * forget, const float * gain,
                 int C, __half * out, cudaStream_t s) {
    control_k<<<1, std::max(64, C < 32 ? 32 : 256), 0, s>>>(x, msg, down, forget, gain, C, out);
    launched();
}

void ik2_l2clip(const __half * x, const float * clip, float cap, int H, int D, __half * y, cudaStream_t s) {
    int threads = std::max(32, std::min(256, ((D + 31) / 32) * 32));
    l2clip_k<<<H, threads, 0, s>>>(x, clip, cap, H, D, y);
    launched();
}

void ik2_rms_weightless(const __half * x, int KV, int D, __half * y, cudaStream_t s) {
    int threads = std::max(32, std::min(256, ((D + 31) / 32) * 32));
    rms_wl_k<<<KV, threads, 0, s>>>(x, KV, D, y);
    launched();
}

void ik2_attention_core(const Ik2AttnCore & a, __half * out, cudaStream_t s) {
    attn_core_k<<<a.H, 256, 0, s>>>(a, out);
    launched();
}

void ik2_route(const float * logits, const float * bias, int experts, int count, float scale,
               int * ids, float * weights, cudaStream_t s) {
    route_k<<<1, 128, 2 * experts * sizeof(float), s>>>(logits, bias, experts, count, scale, ids, weights);
    launched();
}
void ik2_expert_activation(const __nv_bfloat16 * gu, const int * ids, const float * scales, int count, int width,
                           float gate_cap, float up_cap, __half * out, cudaStream_t s) {
    expert_activation_k<<<dim3((width+255)/256, count),256,0,s>>>(gu,ids,scales,width,gate_cap,up_cap,out);
    launched();
}
void ik2_expert_sum(const __nv_bfloat16 * rows, const int * ids, const float * scales, const float * weights, int count,
                    int width, __half * out, cudaStream_t s) {
    expert_sum_k<<<(width+255)/256,256,0,s>>>(rows,ids,scales,weights,count,width,out);
    launched();
}

void ik2_softcap(float * logits, int unpadded, float cap, cudaStream_t s) {
    softcap_k<<<(unpadded + 255) / 256, 256, 0, s>>>(logits, unpadded, cap);
    launched();
}

void ik2_conv_hist_update(__half * hist, const __half * cur, int taps, int C, cudaStream_t s) {
    conv_hist_update_k<<<(C + 255) / 256, 256, 0, s>>>(hist, cur, taps, C);
    launched();
}

void ik2_hadd(const __half * a, const __half * b, __half * y, int n, cudaStream_t s) {
    hadd_k<<<(n + 255) / 256, 256, 0, s>>>(a, b, y, n);
    launched();
}

void ik2_hcopy(const __half * a, __half * y, int n, cudaStream_t s) {
    hcopy_k<<<(n + 255) / 256, 256, 0, s>>>(a, y, n);
    launched();
}

void ik2_hscale(__half * y, float s, int n, cudaStream_t st) {
    hscale_k<<<(n + 255) / 256, 256, 0, st>>>(y, s, n);
    launched();
}

}  // namespace tqllm_cuda
