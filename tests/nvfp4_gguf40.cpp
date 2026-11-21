// Host-dequant parity for ggml NVFP4 (GGUF type 40, block_nvfp4: 4 UE4M3 + 32 B E2M1 per 64 weights).
// Builds random block_nvfp4 bytes, decodes them with dequantize_host(GGML_TYPE_NVFP4), and compares against an
// independent fp64 reconstruction of the same bytes. CPU only (no CUDA).
#include "dequant.hpp"
#include "gguf.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tqllm;

static double ue4m3_f64(uint8_t b) {  // independent fp64 reference
    if (b == 0 || (b == 0x7f && b != 0xff)) return 0.0;
    const int e = (b >> 3) & 15, m = b & 7;
    return e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + (double) m / 8.0, e - 7);
}
static const double kE2M1[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

static int run(int rows, int cols) {
    std::mt19937 rng(uint32_t(rows * 131 + cols));
    const int cb_per_row = cols / 64;
    const int64_t nb = int64_t(rows) * cb_per_row;
    std::vector<uint8_t> raw(size_t(nb) * 36, 0);
    std::vector<double> ref(size_t(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int cb = 0; cb < cb_per_row; ++cb) {
            uint8_t * blk = raw.data() + (int64_t(r) * cb_per_row + cb) * 36;
            for (int s = 0; s < 4; ++s) blk[s] = uint8_t(rng() & 0x7f);  // UE4M3 (finite, no 0x7f/0xff)
            // 32 nibble bytes: byte i holds weights (2i, 2i+1); low nibble = even weight
            for (int i = 0; i < 32; ++i) blk[4 + i] = uint8_t(rng());
            for (int s = 0; s < 4; ++s) {
                const double sc = ue4m3_f64(blk[s]);
                for (int i = 0; i < 16; ++i) {  // sub-block s: weights s*16 .. s*16+15, nibble bytes s*8 .. s*8+7
                    const uint8_t q = blk[4 + s * 8 + (i >> 1)];
                    const int nib = (i & 1) ? ((q >> 4) & 15) : (q & 15);
                    ref[int64_t(r) * cols + cb * 64 + s * 16 + i] = kE2M1[nib] * sc;
                }
            }
        }
    }
    std::vector<float> got(size_t(rows) * cols);
    dequantize_host(GGML_TYPE_NVFP4, raw.data(), got.data(), int64_t(rows) * cols);
    double maxrel = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::abs(double(got[i]) - ref[i]);
        const double m = std::max({1.0, std::abs(ref[i])});
        maxrel = std::max(maxrel, d / m);
    }
    if (maxrel > 1e-6) {
        std::printf("FAIL rows=%d cols=%d maxrel=%.3e\n", rows, cols, maxrel);
        return 1;
    }
    std::printf("ok rows=%d cols=%d maxrel=%.3e\n", rows, cols, maxrel);
    return 0;
}

int main() {
    int rc = 0;
    rc |= run(1, 64);
    rc |= run(3, 128);
    rc |= run(7, 2304);   // Inkling expert width
    rc |= run(2, 1152);   // Inkling down width
    rc |= run(16, 2304);
    return rc;
}
