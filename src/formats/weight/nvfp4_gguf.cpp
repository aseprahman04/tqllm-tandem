// GGUF type 40 (ggml NVFP4, block_nvfp4: 4 UE4M3 + 32 B E2M1 per 64 weights) expert format for Inkling-2-Mini.
// The expert tensors are stored as [rows][cols][experts] type 40 with a separate [experts] F32 .scale tensor
// (one scale per expert). The format loads the block_nvfp4 bytes in place and supplies an expert matvec that
// decodes in-register and applies the per-expert scale. Dense dequant is not provided (experts only).
#include "quant_kernels.cuh"
#include "weight_format.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {
namespace formats {

struct Nvfp4Gguf final : WeightFormat {
    const char * name() const override { return "NVFP4Gguf"; }
    Basis basis() const override { return Basis::Plain; }
    bool has_expert_matvec() const override { return true; }

    // raw: GGUF bytes of `rows` rows, each `cols` wide, type 40 (block_nvfp4, 36 bytes per 64 weights).
    // aux (via `up`): the per-expert F32 scales are loaded separately by the engine and passed at matvec time;
    // here aux is null (the engine passes the scale pointer through the Mat's aux at expert_matvec time).
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        if (cols % 64) throw std::runtime_error("NVFP4Gguf needs a multiple of 64 columns");
        const int64_t nb = int64_t(rows) * (cols / 64);
        if (int64_t(raw.size()) != nb * 36)
            throw std::runtime_error("NVFP4Gguf: tensor size does not match its shape (block_nvfp4 is 36 bytes)");
        const void * p = up(raw.data(), raw.size());
        return {p, 40, rows, cols, nullptr, this};
    }

    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("NVFP4Gguf: dense matvec unsupported (expert format)");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("NVFP4Gguf: dense dequant unsupported (expert format)");
    }

    // Grouped experts: m.p points at [experts * rows][cols / 2] block_nvfp4 bytes (the stacked expert weights);
    // m.aux points at [experts] F32 scales. ids: [n * topk] expert ids. y: [n * topk][rows] BF16.
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows, const int * ids, int n, int topk,
                       bool per_expert_input, cudaStream_t s) const override {
        (void) per_expert_input;
        tqllm_cuda::nvfp4_gguf_expert_matvec(static_cast<const uint8_t *>(m.p), static_cast<const float *>(m.aux),
                                             static_cast<const __nv_bfloat16 *>(x), y, rows, m.cols, ids, n, topk, s);
    }
};

void register_nvfp4_gguf() { register_weight_format(40 /* GGML_TYPE_NVFP4 */, std::make_unique<Nvfp4Gguf>()); }

}  // namespace formats
}  // namespace tqllm
