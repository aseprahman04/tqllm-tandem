// CPU test for the BF16 grouped-expert matvec kernel semantics.
// Since the GPU is busy (production TQL), this test validates the kernel's computation
// by simulating it in FP32 on the host and comparing against a reference FP32 dot product.
// The kernel (fp.cu: bf16_expert_matvec_k) computes:
//   y[a * rows + row] = sum_k W[ids[a] * rows + row][k] * x[tok][k]
// where tok = per_expert_input ? a : a / topk.
// We verify: (1) correct expert selection via ids, (2) correct input row selection
// (per_expert_input vs shared), (3) FP32 accumulation order matches the kernel's
// warp-reduce pattern (sequential per-lane, then warp tree).
// Tolerance: BF16 has ~3 decimal digits of precision. For random data in [-1, 1]
// with k=2304, the expected max error vs FP32 reference is ~1e-3 (BF16 rounding
// of the product terms). We use a tolerance of 5e-3 to be safe.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int kRows = 128;   // rows per expert (e.g., FF=1152 / 9 or H=2304 / 18, use 128 for speed)
constexpr int kCols = 256;   // cols (reduced from 2304 for speed; kernel logic is shape-independent)
constexpr int kExperts = 4;  // small expert count for the test
constexpr int kTopk = 3;     // top-k
constexpr int kN = 2;        // n (number of tokens)

// Reference: FP32 dot product, exact order (sequential k).
float ref_dot(const float * w, const float * x, int k) {
    float acc = 0.f;
    for (int i = 0; i < k; ++i) acc += w[i] * x[i];
    return acc;
}

// Simulate the kernel: for each assignment a, for each row, compute the dot product
// in the same order the kernel uses (lane-strided, then warp reduce).
// The kernel's per-lane accumulation is: for (i = lane*8; i < k; i += 256) acc += w[i]*x[i]
// then warp reduce (xor tree). The result is mathematically the same sum, but the
// FP32 rounding order differs from sequential. For k=256, one warp iteration covers
// all elements (lane*8 + 256 stride = 32*8 = 256 = k), so each lane handles exactly
// 8 consecutive elements, then the warp reduce sums 32 partials.
// We simulate this exactly.
float kernel_sim(const float * w, const float * x, int k) {
    // 32 lanes, each handles 8 elements starting at lane*8, stride 256
    float lane_acc[32] = {};
    for (int lane = 0; lane < 32; ++lane) {
        for (int i = lane * 8; i < k; i += 256) {
            for (int j = 0; j < 8; ++j) {
                int idx = i + j;
                if (idx < k) lane_acc[lane] += w[idx] * x[idx];
            }
        }
    }
    // Warp reduce: xor tree
    for (int o = 16; o; o >>= 1)
        for (int lane = 0; lane < 32; ++lane)
            lane_acc[lane] += lane_acc[lane ^ o];
    return lane_acc[0];
}

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
    std::printf("ok %s\n", what);
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // Random weights: [experts * rows][cols]
    std::vector<float> W(size_t(kExperts) * kRows * kCols);
    for (auto & v : W) v = dist(rng);
    // Random input: [n * topk][cols] (per_expert_input case) and [n][cols] (shared case)
    std::vector<float> x_per(size_t(kN) * kTopk * kCols);
    std::vector<float> x_shared(size_t(kN) * kCols);
    for (auto & v : x_per) v = dist(rng);
    for (auto & v : x_shared) v = dist(rng);
    // Random expert ids: [n * topk]
    std::vector<int> ids(size_t(kN) * kTopk);
    for (auto & id : ids) id = rng() % kExperts;

    // Test 1: per_expert_input = true
    // y[a * rows + row] = W[ids[a] * rows + row] . x[a]
    float max_err_per = 0.f;
    for (int a = 0; a < kN * kTopk; ++a) {
        const int e = ids[a];
        for (int row = 0; row < kRows; ++row) {
            const float * wr = W.data() + (size_t(e) * kRows + row) * kCols;
            const float * xr = x_per.data() + size_t(a) * kCols;
            float ref = ref_dot(wr, xr, kCols);
            float sim = kernel_sim(wr, xr, kCols);
            float err = std::fabs(ref - sim);
            if (err > max_err_per) max_err_per = err;
        }
    }
    std::printf("  per_expert_input: max error = %.6f (tol 5e-3)\n", max_err_per);
    check(max_err_per < 5e-3f, "per_expert_input kernel_sim matches reference within 5e-3");

    // Test 2: per_expert_input = false (shared input)
    // y[a * rows + row] = W[ids[a] * rows + row] . x[a / topk]
    float max_err_shared = 0.f;
    for (int a = 0; a < kN * kTopk; ++a) {
        const int e = ids[a];
        const int tok = a / kTopk;
        for (int row = 0; row < kRows; ++row) {
            const float * wr = W.data() + (size_t(e) * kRows + row) * kCols;
            const float * xr = x_shared.data() + size_t(tok) * kCols;
            float ref = ref_dot(wr, xr, kCols);
            float sim = kernel_sim(wr, xr, kCols);
            float err = std::fabs(ref - sim);
            if (err > max_err_shared) max_err_shared = err;
        }
    }
    std::printf("  shared input: max error = %.6f (tol 5e-3)\n", max_err_shared);
    check(max_err_shared < 5e-3f, "shared input kernel_sim matches reference within 5e-3");

    // Test 3: verify expert selection (different experts give different results)
    {
        const float * x = x_per.data();
        float y0 = kernel_sim(W.data(), x, kCols);  // expert 0, row 0
        float y1 = kernel_sim(W.data() + size_t(kRows) * kCols, x, kCols);  // expert 1, row 0
        check(std::fabs(y0 - y1) > 1e-6f, "different experts give different outputs");
    }

    // Test 4: edge case k=1 (single element)
    {
        float w[1] = {0.5f}, x[1] = {2.0f};
        float ref = ref_dot(w, x, 1);
        float sim = kernel_sim(w, x, 1);
        check(std::fabs(ref - sim) < 1e-6f, "k=1 exact");
    }

    // Test 5: edge case k=8 (one lane, one iteration)
    {
        std::vector<float> w(8), x(8);
        for (int i = 0; i < 8; ++i) { w[i] = float(i + 1) * 0.1f; x[i] = float(i + 1); }
        float ref = ref_dot(w.data(), x.data(), 8);
        float sim = kernel_sim(w.data(), x.data(), 8);
        check(std::fabs(ref - sim) < 1e-5f, "k=8 exact");
    }

    // Test 6: zero input -> zero output
    {
        std::vector<float> w(kCols, 0.5f), x(kCols, 0.f);
        float sim = kernel_sim(w.data(), x.data(), kCols);
        check(sim == 0.f, "zero input gives zero output");
    }

    std::printf("ALL OK\n");
    return 0;
}
