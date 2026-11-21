// Inkling-2-Mini kernels (see inkling2_kernels.cuh). FP16 activations, FP32 accumulate. The reference is
// ref/inkling2.py (float64). The engine (inkling2.cpp) orchestrates the multi-stage ops; these are the self-contained
// pieces it calls.
#include "inkling2_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {
namespace {

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
    for (int si = 0; si < n; ++si) {
        const __half * st = (si < n_blocks) ? blocks + size_t(si) * C : x;
        float ss = 0.f;
        for (int i = threadIdx.x; i < C; i += blockDim.x) { float v = __half2float(st[i]); ss += v * v; }
        const float sc = rsqrtf(block_sum(ss) / C + kResEps);
        float d = 0.f;
        for (int i = threadIdx.x; i < C; i += blockDim.x) d += q[qi * C + i] * __half2float(st[i]) * sc;
        // every thread takes part in the sum: block_sum shuffles and waits on barriers, and a warp shuffle or a barrier
        // that only thread 0 reaches is undefined (the block may never finish)
        const float dsum = block_sum(d);
        if (threadIdx.x == 0) scores[si] = dsum / sqrtf(C);
        __syncthreads();
    }
    __shared__ float m, Z;
    if (threadIdx.x == 0) {
        m = -1e30f;
        for (int si = 0; si < n; ++si) m = fmaxf(m, scores[si]);
        Z = 0.f;
        for (int si = 0; si < n; ++si) Z += expf(scores[si] - m);
    }
    __syncthreads();
    const int c = threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
    for (int si = 0; si < n; ++si) {
        const __half * st = (si < n_blocks) ? blocks + size_t(si) * C : x;
        float ss = 0.f;
        for (int i = threadIdx.x; i < C; i += blockDim.x) { float v = __half2float(st[i]); ss += v * v; }
        const float sc = rsqrtf(block_sum(ss) / C + kResEps);
        acc += (expf(scores[si] - m) / Z) * __half2float(st[c]) * sc;
        __syncthreads();
    }
    out[qi * C + c] = __float2half(acc);
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
        acc += dyn[kvh * taps + lag] * xv;
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
    const int c = threadIdx.x;
    if (c >= C) return;
    float f = 0.f, gn = 0.f;
    for (int i = 0; i < 64; ++i) { f += forget[c * 64 + i] * g[i]; gn += gain[c * 64 + i] * g[i]; }
    const float xf = (1.f - tanhf(f)) * __half2float(x[c]);
    const float mg = tanhf(gn) * __half2float(msg[c]);
    out[c] = __float2half(xf + mg);
}

// ---- L2 clip: x[head][d] * (cap / (norm + 1e-6)).clamp(max=1) * clip[head][d] ----
__global__ void l2clip_k(const __half * __restrict__ x, const float * __restrict__ clip, float cap, int H, int D, __half * __restrict__ y) {
    const int h = blockIdx.x;
    if (h >= H) return;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float v = __half2float(x[h * D + i]); ss += v * v; }
    const float scale = cap / (sqrtf(block_sum(ss)) + 1e-6f);
    const float cl = fminf(scale, 1.f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) y[h * D + i] = __float2half(__half2float(x[h * D + i]) * cl * clip[h * D + i]);
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
    const int h = blockIdx.x;  // query head
    if (h >= a.H) return;
    const int D = a.D, KV = a.KV, H = a.H;
    const int g = H / KV;  // GQA group size
    const int kvh = h / g;
    // the keys: the KV cache rows [lo, pos) plus the current token's k (at index pos)
    // For decode the current token's k/v are appended by the caller before this kernel; here we read the cache rows
    // [lo, pos] inclusive (the caller wrote the current token's k/v at row pos).
    const int lo = a.sliding ? (int(a.pos) > a.window ? int(a.pos) - a.window : 0) : 0;
    const int hi = int(a.pos) + 1;  // inclusive
    const int n = hi - lo;
    // scores over the n keys
    __shared__ float sc[1024];  // n <= window <= 1024
    if (threadIdx.x == 0) {
        const float inv = 1.f / sqrtf(D);
        for (int s = 0; s < n; ++s) {
            const int t = lo + s;
            float dot = 0.f;
            for (int d = 0; d < D; ++d) dot += __half2float(a.q[h * D + d]) * __half2float(a.kc[size_t(t) * KV * D + kvh * D + d]);
            // relative bias: r[h] . rel_bias[:, t - pos]  (0 <= pos - t < extent else 0)
            const int dist = int(a.pos) - t;
            float rel = 0.f;
            if (dist >= 0 && dist < a.extent) {
                for (int e = 0; e < a.d_rel; ++e) rel += __half2float(a.r[h * a.d_rel + e]) * a.rel_bias[e * a.extent + dist];
            }
            sc[s] = dot * inv + rel;
        }
        // sink logit: a per-head scalar in the softmax (its value zero)
        sc[n] = __half2float(a.sink[h]);
    }
    __syncthreads();
    // softmax over n + 1 (the sink)
    __shared__ float m, Z;
    if (threadIdx.x == 0) {
        m = -1e30f;
        for (int s = 0; s <= n; ++s) m = fmaxf(m, sc[s]);
        Z = 0.f;
        for (int s = 0; s <= n; ++s) Z += expf(sc[s] - m);
    }
    __syncthreads();
    // out[h][d] = sum_s p[s] v[s][d]  (the sink contributes zero value)
    const int d = threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int s = 0; s < n; ++s) {
        const int t = lo + s;
        acc += (expf(sc[s] - m) / Z) * __half2float(a.vc[size_t(t) * KV * D + kvh * D + d]);
    }
    out[h * D + d] = __float2half(acc * __expf(__half2float(a.gate[h])) / (1.f + __expf(__half2float(a.gate[h]))));  // sigmoid(gate)
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

}  // namespace

void ik2_rms_norm(const __half * x, const float * w, float eps, int C, __half * y, cudaStream_t s) {
    int threads = std::min(1024, C < 32 ? 32 : 256);
    rms_norm_k<<<1, threads, 0, s>>>(x, w, eps, C, y);
}

void ik2_attn_res(const __half * blocks, int n_blocks, const __half * x, const float * q, int nq, int C, __half * out, cudaStream_t s) {
    int threads = std::min(1024, C < 32 ? 32 : 256);
    attn_res_k<<<nq, threads, 0, s>>>(blocks, n_blocks, x, q, nq, C, out);
}

void ik2_shortconv(const float * w, int taps, const __half * x, __half * y, int C, int T, const __half * hist, cudaStream_t s) {
    const int total = T * C;
    shortconv_k<<<(total + 255) / 256, 256, 0, s>>>(w, taps, x, y, C, T, hist);
}

void ik2_shortconv_dyn(const float * w, int taps, const __half * x, __half * y, int C, int D, const float * dyn, const __half * hist, cudaStream_t s) {
    shortconv_dyn_k<<<(C + 255) / 256, 256, 0, s>>>(w, taps, x, y, C, D, dyn, hist);
}

void ik2_control(const __half * x, const __half * msg, const float * down, const float * forget, const float * gain,
                 int C, __half * out, cudaStream_t s) {
    control_k<<<1, std::max(64, C < 32 ? 32 : 256), 0, s>>>(x, msg, down, forget, gain, C, out);
}

void ik2_l2clip(const __half * x, const float * clip, float cap, int H, int D, __half * y, cudaStream_t s) {
    int threads = std::min(256, D);
    l2clip_k<<<H, threads, 0, s>>>(x, clip, cap, H, D, y);
}

void ik2_rms_weightless(const __half * x, int KV, int D, __half * y, cudaStream_t s) {
    int threads = std::min(256, D);
    rms_wl_k<<<KV, threads, 0, s>>>(x, KV, D, y);
}

void ik2_attention_core(const Ik2AttnCore & a, __half * out, cudaStream_t s) {
    attn_core_k<<<a.H, std::min(256, a.D), 0, s>>>(a, out);
}

void ik2_softcap(float * logits, int unpadded, float cap, cudaStream_t s) {
    softcap_k<<<(unpadded + 255) / 256, 256, 0, s>>>(logits, unpadded, cap);
}

void ik2_conv_hist_update(__half * hist, const __half * cur, int taps, int C, cudaStream_t s) {
    conv_hist_update_k<<<(C + 255) / 256, 256, 0, s>>>(hist, cur, taps, C);
}

void ik2_hadd(const __half * a, const __half * b, __half * y, int n, cudaStream_t s) {
    hadd_k<<<(n + 255) / 256, 256, 0, s>>>(a, b, y, n);
}

void ik2_hcopy(const __half * a, __half * y, int n, cudaStream_t s) {
    hcopy_k<<<(n + 255) / 256, 256, 0, s>>>(a, y, n);
}

void ik2_hscale(__half * y, float s, int n, cudaStream_t st) {
    hscale_k<<<(n + 255) / 256, 256, 0, st>>>(y, s, n);
}

}  // namespace tqllm_cuda
