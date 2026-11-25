// Compare one decode position with a host softmax over logical cache positions.
#include "inkling2_kernels.cuh"
#include "quant_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T> struct Buffer {
    std::vector<T> host;
    T *device = nullptr;
    explicit Buffer(size_t size) : host(size) { check(cudaMalloc(&device, size * sizeof(T))); }
    ~Buffer() { cudaFree(device); }
    void upload() { check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice)); }
};

static void compare(int pos, bool sliding) {
    constexpr int H = 2, D = 128, R = 16, E = 64, window = 1024;
    const int capacity = sliding ? window : pos + 1;
    Buffer<__half> keys(capacity * D), values(capacity * D), query(H * D), rel(H * R),
                   sink(H), gate(H * D), output(H * D);
    Buffer<float> bias(E * R);
    // The discarded position just outside the window differs strongly from any retained row.
    // Create physical cache contents by writing logical positions in chronological order.
    for (int t = 0; t <= pos; ++t) {
        const int row = (t % capacity) * D;
        for (int d = 0; d < D; ++d) {
            keys.host[row+d] = __float2half(float((t * 3 + d) % 11 - 5) / 7);
            values.host[row+d] = __float2half(t == 0 ? 128.f : float((t + 2 * d) % 37 - 18) / 9);
        }
    }
    for (int i = 0; i < H * D; ++i) {
        query.host[i] = __float2half(float(i % 7 - 3) / 5);
        gate.host[i] = __float2half(float(i % 9 - 4));
    }
    for (int i = 0; i < H * R; ++i) rel.host[i] = __float2half(float(i % 5 - 2) / 2);
    for (int i = 0; i < E * R; ++i) bias.host[i] = float((i * 3) % 17 - 8) / 9;
    sink.host[0] = __float2half(0); sink.host[1] = __float2half(6);
    keys.upload(); values.upload(); query.upload(); rel.upload(); gate.upload(); sink.upload(); bias.upload();
    tqllm_cuda::Ik2AttnCore a{};
    a.q=query.device; a.r=rel.device; a.sink=sink.device; a.gate=gate.device; a.rel_bias=bias.device;
    a.H=H; a.KV=1; a.D=D; a.d_rel=R; a.extent=E; a.window=window;
    a.kc=keys.device; a.vc=values.device; a.sliding=sliding; a.pos=pos;
    tqllm_cuda::ik2_attention_core(a, output.device, nullptr);
    check(cudaMemcpy(output.host.data(), output.device, H * D * sizeof(__half), cudaMemcpyDeviceToHost));
    for (int h = 0; h < H; ++h) {
        double z = std::exp(__half2float(sink.host[h]));
        std::vector<double> sum(D, 0.);
        for (int t = 0; t <= pos; ++t) {
            const int distance = pos - t;
            if (sliding && distance >= window) continue;
            double dot = 0., offset = 0.;
            for (int d = 0; d < D; ++d)
                dot += __half2float(query.host[h*D+d]) * __half2float(keys.host[(t%capacity)*D+d]);
            if (distance < E)
                for (int r = 0; r < R; ++r) offset += __half2float(rel.host[h*R+r]) * bias.host[distance*R+r];
            const double weight = std::exp(dot / std::sqrt(double(D)) + offset);
            z += weight;
            for (int d = 0; d < D; ++d) sum[d] += weight * __half2float(values.host[(t%capacity)*D+d]);
        }
        for (int d = 0; d < D; ++d) {
            const double expected = sum[d] / z / (1. + std::exp(-__half2float(gate.host[h*D+d])));
            const double actual = __half2float(output.host[h*D+d]);
            if (!(std::abs(expected - actual) <= 0.001 * (1. + std::abs(expected)))) {
                std::fprintf(stderr, "pos=%d sliding=%d head=%d dim=%d expected=%.8f actual=%.8f\n",
                             pos, sliding, h, d, expected, actual);
                throw std::runtime_error("attention differs from CPU");
            }
        }
    }
}

static void check_channels() {
    constexpr int C = 2304;
    Buffer<__half> x(C), bank(C), out(C);
    Buffer<float> q(C);
    for (int c = 0; c < C; ++c) {
        x.host[c] = __float2half(c % 2 ? -2.f : 2.f);
        bank.host[c] = __float2half(1.f);
    }
    x.upload(); bank.upload(); q.upload();
    tqllm_cuda::ik2_attn_res(bank.device, 1, x.device, q.device, 1, C, out.device, nullptr);
    check(cudaMemcpy(out.host.data(), out.device, C*sizeof(__half), cudaMemcpyDeviceToHost));
    for (int c = 0; c < C; ++c)
        if (std::abs(__half2float(out.host[c]) - (c%2 ? 0.f : 1.f)) > 0.001f)
            throw std::runtime_error("residual mixture differs from CPU");
}

static void check_expert_rows() {
    // GGUF NVFP4 packs columns j and j+8 in the low and high nibbles of a byte.
    constexpr int width=64, rows=8, count=2;
    Buffer<uint8_t> matrix(count * rows * 36);
    Buffer<__half> input(count * width);
    Buffer<__nv_bfloat16> output(count * rows);
    Buffer<int> ids(count);
    ids.host={1,0};
    for (int e=0; e<count; ++e)
        for (int r=0; r<rows; ++r) {
            uint8_t * block=matrix.host.data()+(e*rows+r)*36;
            std::fill(block,block+4,0x38);  // UE4M3 1.0
            std::fill(block+4,block+36,0x42); // low=1, high=2
            block[4+r]=e ? 0x43 : 0x21;
        }
    for (int a=0; a<count; ++a)
        for (int j=0; j<width; ++j) input.host[a*width+j]=__float2half(float((a+1)*(j%13-6)));
    matrix.upload(); ids.upload(); input.upload();
    for (bool per_expert : {false,true}) {
        tqllm_cuda::nvfp4_gguf_expert_matvec(matrix.device,nullptr,input.device,output.device,rows,width,ids.device,
                                             1,count,per_expert,nullptr);
        check(cudaMemcpy(output.host.data(),output.device,count*rows*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
        for (int a=0; a<count; ++a)
            for (int r=0; r<rows; ++r) {
                float sum=0;
                for (int j=0; j<width; ++j) {
                    float weight=(j%16<8) ? 1.f : 2.f;
                    if (j==r) weight=ids.host[a] ? 1.5f : .5f;
                    if (j==r+8) weight=ids.host[a] ? 2.f : 1.f;
                    sum+=weight*__half2float(input.host[(per_expert ? a : 0)*width+j]);
                }
                if (__bfloat162float(output.host[a*rows+r])!=__bfloat162float(__float2bfloat16(sum)))
                    throw std::runtime_error("routed NVFP4 matvec differs from CPU");
            }
    }
}

int main() {
    // Cross-warp tie, unequal top two, highly confident final head, and untouched padding.
    for (float peak : {2.f, 4.f, 28.f}) {
        constexpr int vocab = 515;
        Buffer<float> logits(vocab + 1), ref(vocab + 1), scratch(1);
        for (int i = 0; i < vocab; ++i) {
            logits.host[i] = float(i % 9 - 6) * 0.125f;
            ref.host[i] = float(i % 7 - 3);
        }
        logits.host[13] = peak; logits.host[510] = 2.f; logits.host[vocab] = 123.f;
        const auto before = logits.host;
        logits.upload(); ref.upload();
        tqllm_cuda::ik2_loopcd(logits.device, ref.device, vocab, scratch.device, nullptr);
        check(cudaMemcpy(logits.host.data(), logits.device, (vocab + 1) * sizeof(float), cudaMemcpyDeviceToHost));
        double sum = 0;
        for (int i = 0; i < vocab; ++i) sum += std::exp(double(before[i] - peak));
        const double w = 0.5 * (1.0 - (1.0 - std::exp(2.0 - peak)) / sum);
        for (int i = 0; i < vocab; ++i)
            if (std::abs(logits.host[i] - (before[i] + w * (before[i] - ref.host[i]))) > 1e-5)
                throw std::runtime_error("adaptive LoopCD differs from the full-vocabulary CPU oracle");
        if (logits.host[vocab] != 123.f) throw std::runtime_error("LoopCD touched padded vocabulary");
    }
    for (int p : {0, 1022, 1023, 1024, 1258, 4096}) {
        compare(p, false);
        compare(p, true);
    }
    check_channels();
    check_expert_rows();
    std::puts("12 full/sliding attention cases, 2304 residual channels and shared/routed NVFP4 rows match CPU");
}
