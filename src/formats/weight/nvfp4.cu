// NVFP4 (modelopt) weight kernels, the Qwen3.8-Flash-Next expert format: codes [rows][k / 2] (E2M1 pairs, low nibble =
// even column), FP8 E4M3 scale per 16 [rows][k / 16], FP32 global scale; w = e2m1 * fp8(scale) * global.
// First version: correctness reference for the Flash-Next engine (one warp per row, FP32 accumulate), not tuned.
#include "common.cuh"
#include "quant_kernels.cuh"

#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace tqllm_cuda {
namespace {
__device__ __constant__ float E2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                          -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// grid (rows / 8), 256 threads: warp = row, lanes stride over the 16-weight groups; up to 8 columns
template <int NCOL>
__global__ void __launch_bounds__(256) nvfp4_matvec_k(const uint8_t * w, const uint8_t * sc, float gscale,
                                                      const __nv_bfloat16 * x, float * y, int rows, int k) {
    const int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const int ng = k / 16;
    const uint8_t * wr = w + int64_t(row) * (k / 2);
    const uint8_t * sr = sc + int64_t(row) * ng;
    float acc[NCOL] = {};
    for (int g = lane; g < ng; g += 32) {
        __nv_fp8_e4m3 s8;
        s8.__x = sr[g];
        const float s = float(s8);
        const uint64_t codes = *reinterpret_cast<const uint64_t *>(wr + g * 8);  // 16 nibbles
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            const float wv = E2M1[(codes >> (4 * e)) & 15u] * s;
#pragma unroll
            for (int c = 0; c < NCOL; ++c) acc[c] += wv * __bfloat162float(x[int64_t(c) * k + g * 16 + e]);
        }
    }
#pragma unroll
    for (int c = 0; c < NCOL; ++c) {
        float v = acc[c];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[int64_t(c) * rows + row] = v * gscale;
    }
}
}  // namespace

void nvfp4_matvec(const uint8_t * w, const uint8_t * sc, float gscale, const __nv_bfloat16 * x, float * y, int rows,
                  int k, int ncols, cudaStream_t s) {
    const dim3 grid((rows + 7) / 8);
    switch (ncols) {
        case 1: nvfp4_matvec_k<1><<<grid, 256, 0, s>>>(w, sc, gscale, x, y, rows, k); break;
        case 2: nvfp4_matvec_k<2><<<grid, 256, 0, s>>>(w, sc, gscale, x, y, rows, k); break;
        case 4: nvfp4_matvec_k<4><<<grid, 256, 0, s>>>(w, sc, gscale, x, y, rows, k); break;
        case 8: nvfp4_matvec_k<8><<<grid, 256, 0, s>>>(w, sc, gscale, x, y, rows, k); break;
        default:
            for (int c0 = 0; c0 < ncols; ++c0)
                nvfp4_matvec_k<1><<<grid, 256, 0, s>>>(w, sc, gscale, x + int64_t(c0) * k, y + int64_t(c0) * rows, rows, k);
    }
}

// ggml NVFP4 (block_nvfp4: d[4] UE4M3 + qs[32] E2M1 nibbles per 64 weights) expert matvec, decode in-register.
// 2-D grid: x = rows / 8 (8 rows per block), y = n * topk (assignments). 256 threads = 8 warps, one warp per row.
// w: [experts * rows][k / 64] block_nvfp4 (each block 36 bytes: d[4] then qs[32]); sc: [experts] F32 (null: 1.0);
// x: [n][k] BF16; ids: [n * topk] expert ids; y: [n * topk][rows] BF16.
namespace {
__device__ __forceinline__ float ue4m3_f(uint8_t b) {  // ggml's UE4M3 (unsigned E4M3, finite)
    if (b == 0 || (b == 0x7f && b != 0xff)) return 0.f;
    const int e = (b >> 3) & 15, m = b & 7;
    return e == 0 ? ldexpf(m, -9) : ldexpf(1.f + m / 8.f, e - 7);
}
__global__ void nvfp4_gguf_expert_matvec_k(const uint8_t * w, const float * sc, const __half * x,
                                           __nv_bfloat16 * y, int rows, int k, const int * ids, int topk, bool per_expert_input) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;  // output row within the expert
    if (row >= rows) return;
    const int a = blockIdx.y;  // assignment [0, n * topk)
    const int tok = a / topk, e = ids[a];
    const float scale = sc ? sc[e] : 1.f;
    const int nb = k / 64;  // block_nvfp4 per row
    const uint8_t * wr = w + (int64_t(e) * rows + row) * int64_t(nb) * 36;  // this expert's row
    const __half * xr = x + int64_t(per_expert_input ? a : tok) * k;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t * blk = wr + b * 36;
        const uint8_t * d = blk, * qs = blk + 4;
        // 4 sub-blocks of 16 weights; sub-block s uses scale d[s]
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float scs = ue4m3_f(d[s]);
            // 8 bytes of nibbles per sub-block (qs is 32 bytes for the block's 64 weights), as dequant.cpp's
            // GGML_TYPE_NVFP4 case reads them; a 16-byte step read the wrong codes from sub-block 1 on and ran 24
            // bytes past the block (past the buffer at the last block of the last expert)
            const uint8_t * q = qs + s * 8;
            // sub-block s covers weights [s*16, s*16+16): 16 weights = 8 bytes of nibbles
            const int base = b * 64 + s * 16;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t qbyte = q[i % 8];
                const int nib = (i >= 8) ? (qbyte >> 4) : (qbyte & 15);
                const float wv = E2M1[nib] * scs;
                acc += wv * __half2float(xr[base + i]);
            }
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[int64_t(a) * rows + row] = __float2bfloat16(acc * scale);
}
}  // namespace
void nvfp4_gguf_expert_matvec(const uint8_t * w, const float * sc, const __half * x, __nv_bfloat16 * y, int rows,
                              int k, const int * ids, int n, int topk, bool per_expert_input, cudaStream_t s) {
    const dim3 grid((rows + 7) / 8, n * topk);
    nvfp4_gguf_expert_matvec_k<<<grid, 256, 0, s>>>(w, sc, x, y, rows, k, ids, topk, per_expert_input);
}
}  // namespace tqllm_cuda
