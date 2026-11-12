// Tensor-parallel feasibility check (multi-GPU fork, not built by default): the cost of the all-reduce a TP decode step
// needs twice per layer (128 per verify step on a 64-layer model), between two GPUs without peer access (GeForce on
// Windows: everything crosses through host memory). Each iteration: both GPUs run ~WORK_US of "layer" work, then
// exchange an 8 x 5120 bf16 partial (80 KB) and add the other GPU's partial. Overhead = measured - 128 x WORK_US.
//   A: cudaMemcpyAsync D2H / H2D through pinned host buffers, cross-device events
//   B: zero-copy: the producer kernel stores into mapped host memory, the consumer's add kernel reads it; events
//   C: zero-copy + flags: the consumer's add kernel spins on a flag the producer sets (no cross-device event)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

constexpr int N = 8 * 5120, ITERS = 128;

__global__ void work_k(int us) {  // stand-in for one half-layer of verify work
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const long long t0 = clock64();
        const long long cycles = (long long)us * 2400;  // ~2.4 GHz boost; calibrated below by the no-exchange run
        while (clock64() - t0 < cycles) {}
    }
}
__global__ void produce_k(__nv_bfloat16 * dst, float v) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) dst[i] = __float2bfloat16(v);
}
__global__ void add_k(__nv_bfloat16 * x, const __nv_bfloat16 * other) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(other[i]));
}
__global__ void produce_flag_k(__nv_bfloat16 * dst, float v, volatile unsigned * flag, unsigned seq) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) dst[i] = __float2bfloat16(v);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd((unsigned *)flag + blockIdx.x, 0u), flag[blockIdx.x] = seq;  // one flag per block
}
__global__ void add_flag_k(__nv_bfloat16 * x, const volatile __nv_bfloat16 * other, volatile unsigned * flag, unsigned seq,
                           int producer_blocks) {
    if (threadIdx.x == 0)
        for (int b = 0; b < producer_blocks; ++b)
            while (flag[b] < seq) {}
    __syncthreads();
    __threadfence_system();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(((const __nv_bfloat16 *)other)[i]));
}

struct Dev {
    cudaStream_t s;
    cudaEvent_t done;
    __nv_bfloat16 *p, *tmp;
};

int main(int argc, char ** argv) {
    const int work_us = argc > 1 ? std::atoi(argv[1]) : 120;
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    if (n < 2) { std::printf("needs two GPUs\n"); return 1; }
    int peer = 0;
    CK(cudaDeviceCanAccessPeer(&peer, 0, 1));
    std::printf("peer access 0->1: %s; work per half-layer: %d us; %d all-reduces of %d KB\n", peer ? "yes" : "no", work_us,
                ITERS, int(N * 2 / 1024));
    Dev d[2];
    __nv_bfloat16 * h[2];        // pinned + mapped host buffers, one per producer
    unsigned * flags[2];         // mapped host flags, one array per producer
    for (int i = 0; i < 2; ++i) {
        CK(cudaSetDevice(i));
        CK(cudaStreamCreateWithFlags(&d[i].s, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&d[i].done, cudaEventDisableTiming));
        CK(cudaMalloc(&d[i].p, N * 2));
        CK(cudaMalloc(&d[i].tmp, N * 2));
        CK(cudaHostAlloc(&h[i], N * 2, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc(&flags[i], 64 * sizeof(unsigned), cudaHostAllocMapped | cudaHostAllocPortable));
        for (int b = 0; b < 64; ++b) flags[i][b] = 0;
    }
    auto sync_all = [&] { for (int i = 0; i < 2; ++i) { CK(cudaSetDevice(i)); CK(cudaStreamSynchronize(d[i].s)); } };
    auto timed = [&](const char * name, auto && step) {
        for (int w = 0; w < 2; ++w) {  // warm-up, then measured
            sync_all();
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < ITERS; ++it) step(it + 1 + w * ITERS);
            sync_all();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            (void)name;
            if (w) return ms;
        }
        return 0.0;
    };
    // baseline: work only (both GPUs concurrently, no exchange)
    const double base = timed("work only", [&](unsigned) {
        for (int i = 0; i < 2; ++i) { CK(cudaSetDevice(i)); work_k<<<1, 32, 0, d[i].s>>>(work_us); }
    });
    auto report = [&](const char * name, double ms) {
        std::printf("  => %-40s %.2f ms per step (%.1f us per all-reduce over the work-only run)\n", name, ms,
                    (ms - base) * 1000.0 / ITERS);
    };
    // A: copies through pinned host memory + cross-device events
    const double a = timed("A copies + events", [&](unsigned) {
        for (int i = 0; i < 2; ++i) {
            CK(cudaSetDevice(i));
            work_k<<<1, 32, 0, d[i].s>>>(work_us);
            produce_k<<<40, 256, 0, d[i].s>>>(d[i].p, 1.f);
            CK(cudaMemcpyAsync(h[i], d[i].p, N * 2, cudaMemcpyDeviceToHost, d[i].s));
            CK(cudaEventRecord(d[i].done, d[i].s));
        }
        for (int i = 0; i < 2; ++i) {
            CK(cudaSetDevice(i));
            CK(cudaStreamWaitEvent(d[i].s, d[1 - i].done, 0));
            CK(cudaMemcpyAsync(d[i].tmp, h[1 - i], N * 2, cudaMemcpyHostToDevice, d[i].s));
            add_k<<<40, 256, 0, d[i].s>>>(d[i].p, d[i].tmp);
        }
    });
    report("A copies + events", a);
    // B: zero-copy stores / loads of mapped host memory + cross-device events
    const double b = timed("B zero-copy + events", [&](unsigned) {
        for (int i = 0; i < 2; ++i) {
            CK(cudaSetDevice(i));
            work_k<<<1, 32, 0, d[i].s>>>(work_us);
            produce_k<<<40, 256, 0, d[i].s>>>(h[i], 1.f);
            CK(cudaEventRecord(d[i].done, d[i].s));
        }
        for (int i = 0; i < 2; ++i) {
            CK(cudaSetDevice(i));
            CK(cudaStreamWaitEvent(d[i].s, d[1 - i].done, 0));
            add_k<<<40, 256, 0, d[i].s>>>(d[i].p, h[1 - i]);
        }
    });
    report("B zero-copy + events", b);
    // C: zero-copy + spin flags in host memory (no cross-device event)
    const int PB = 40;
    const double c = timed("C zero-copy + flags", [&](unsigned seq) {
        for (int i = 0; i < 2; ++i) {
            CK(cudaSetDevice(i));
            work_k<<<1, 32, 0, d[i].s>>>(work_us);
            produce_flag_k<<<PB, 256, 0, d[i].s>>>(h[i], 1.f, flags[i], seq);
            add_flag_k<<<40, 256, 0, d[i].s>>>(d[i].p, h[1 - i], flags[1 - i], seq, PB);
        }
    });
    report("C zero-copy + flags", c);
    std::printf("work-only step: %.2f ms (%d x %d us = %.2f ms asked)\n", base, ITERS, work_us, ITERS * work_us / 1000.0);
    return 0;
}
