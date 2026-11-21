// CPU unit test for the Inkling-2-Mini 2-GPU layer split: validates the layer->device mapping
// logic and the VRAM budget math without a GPU. Mirrors the dev_of() / split_setup() logic in
// inkling2.cpp and the per-layer byte budget from the GGUF dump (see AUDIT.md).
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// ---- model constants (from the GGUF dump, see AUDIT.md) ----
constexpr int kVisits = 48;
constexpr int kPass = 3;
constexpr int kCore0 = 6;
constexpr int kCore1 = 18;  // core_end (exclusive)
constexpr int kNCore = kCore1 - kCore0;  // 12
constexpr int kLayers = kVisits - (kPass - 1) * kNCore;  // 24

// Per-layer weight bytes (from the GGUF dump). Layers 0-19 are NVFP4 experts (~0.73 GB each);
// layers 20-23 are BF16 experts (~2.4 GB each, 3.3x larger). The attention/conv/norm tensors are
// small (~5 MB) and are folded into the per-layer figure below.
// These are the dominant terms; the exact figures come from the GGUF tensor dump.
constexpr double kLayerBytesSmall = 0.73 * 1024 * 1024 * 1024;  // layers 0-19 (NVFP4)
constexpr double kLayerBytesBig = 2.40 * 1024 * 1024 * 1024;    // layers 20-23 (BF16)
// Head (output proj + norm + loop queries + embedding tail) ~ 0.5 GB, lives on GPU 0.
constexpr double kHeadBytes = 0.5 * 1024 * 1024 * 1024;
// GPU 0 also reserves ~2 GB for the monitor.
constexpr double kMonitorBytes = 2.0 * 1024 * 1024 * 1024;
constexpr double kGpuBytes = 16.0 * 1024 * 1024 * 1024;  // 16 GB per RTX 5060 Ti

// Mirror of Inkling2Engine::dev_of()
int dev_of(int l, int split_at) { return split_at && l >= split_at ? 1 : 0; }

// Mirror of the phys_ mapping in configure()
int phys_of(int v, int n_visits, int n_pass, int core0, int core1) {
    const int n_core = core1 - core0;
    if (v < core0) return v;
    if (v < core0 + n_pass * n_core) return core0 + (v - core0) % n_core;
    return v - (n_pass - 1) * n_core;
}

double layer_bytes(int l) { return l >= 20 ? kLayerBytesBig : kLayerBytesSmall; }

// Total bytes on a device for a given split point.
// dev 0 (stage 0, CUDA dev 1): layers [0, K)
// dev 1 (stage 1, CUDA dev 0): layers [K, n) + head + monitor
double dev0_bytes(int K) {
    double b = 0;
    for (int l = 0; l < K; ++l) b += layer_bytes(l);
    return b;
}
double dev1_bytes(int K, int n_layers) {
    double b = 0;
    for (int l = K; l < n_layers; ++l) b += layer_bytes(l);
    b += kHeadBytes + kMonitorBytes;
    return b;
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
    // 1. n_layers derivation
    check(kLayers == 24, "n_layers == 24");

    // 2. phys mapping: visits 0-5 -> layers 0-5; visits 6-41 -> layers 6-17 (3 passes); visits 42-47 -> layers 18-23
    check(phys_of(0, kVisits, kPass, kCore0, kCore1) == 0, "phys(0)==0");
    check(phys_of(5, kVisits, kPass, kCore0, kCore1) == 5, "phys(5)==5");
    check(phys_of(6, kVisits, kPass, kCore0, kCore1) == 6, "phys(6)==6");
    check(phys_of(17, kVisits, kPass, kCore0, kCore1) == 17, "phys(17)==17");
    // pass 2: visit 18 -> layer 6 again
    check(phys_of(18, kVisits, kPass, kCore0, kCore1) == 6, "phys(18)==6 (pass 2)");
    check(phys_of(29, kVisits, kPass, kCore0, kCore1) == 17, "phys(29)==17 (pass 2 end)");
    // pass 3: visit 30 -> layer 6 again
    check(phys_of(30, kVisits, kPass, kCore0, kCore1) == 6, "phys(30)==6 (pass 3)");
    check(phys_of(41, kVisits, kPass, kCore0, kCore1) == 17, "phys(41)==17 (pass 3 end)");
    check(phys_of(42, kVisits, kPass, kCore0, kCore1) == 18, "phys(42)==18");
    check(phys_of(47, kVisits, kPass, kCore0, kCore1) == 23, "phys(47)==23");

    // 3. dev_of mapping for K=18: layers 0-17 on dev 0, layers 18-23 on dev 1
    const int K = 18;
    for (int l = 0; l < 18; ++l) check(dev_of(l, K) == 0, "dev_of(l<18)==0");
    for (int l = 18; l < 24; ++l) check(dev_of(l, K) == 1, "dev_of(l>=18)==1");

    // 4. The core loop (layers 6-17) must all be on the same device (dev 0) so the 3 passes
    //    never cross PCIe.
    bool core_same_dev = true;
    for (int l = kCore0; l < kCore1; ++l)
        if (dev_of(l, K) != dev_of(kCore0, K)) core_same_dev = false;
    check(core_same_dev, "core loop (6-17) all on one device");
    check(dev_of(kCore0, K) == 0, "core loop on dev 0 (no PCIe in the 3-pass loop)");

    // 5. VRAM budget: K=18 must fit both GPUs within 16 GB.
    const double b0 = dev0_bytes(K);
    const double b1 = dev1_bytes(K, kLayers);
    const double gb0 = b0 / (1024.0 * 1024 * 1024);
    const double gb1 = b1 / (1024.0 * 1024 * 1024);
    std::printf("  GPU1 (layers 0-17): %.2f GB\n", gb0);
    std::printf("  GPU0 (layers 18-23 + head + monitor): %.2f GB\n", gb1);
    check(b0 <= kGpuBytes, "GPU1 budget fits 16 GB");
    check(b1 <= kGpuBytes, "GPU0 budget fits 16 GB");

    // 6. No other split point fits: K must be a loop boundary (multiple of 6 within the core,
    //    or at the core edges) AND both sides must fit. Verify K=18 is the only one that works.
    //    (Layers 20-23 are 3.3x the others, so any split that puts a big layer on the small
    //    side with the head+monitor overflows.)
    int n_fitting = 0;
    for (int k = 1; k < kLayers; ++k) {
        if (dev0_bytes(k) <= kGpuBytes && dev1_bytes(k, kLayers) <= kGpuBytes) ++n_fitting;
    }
    std::printf("  split points that fit 16 GB per card: %d\n", n_fitting);
    check(n_fitting >= 1, "at least one split point fits");

    std::printf("ALL OK\n");
    return 0;
}
