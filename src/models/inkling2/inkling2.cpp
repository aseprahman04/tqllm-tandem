// Inkling-2-Mini (GGUF arch `inkling2`, the looped 26B MoE) engine on the GPU.
//
// 24 physical layers, 48 visits (layers 6..17 run 3 times: a 3-pass loop). Per visit (reference ref/inkling2.py):
//   h_in = attn_res(blocks, h)            (layer 0: (h,h,h); a full layer's 3 queries give its q, k, v inputs)
//   [core, pass>0: ins_j = control(h_in_j, msg_j), m_in = control(m_in, msg_mlp)]
//   norm the inputs (sliding: one attn_norm; full: attn_{q,k,v}_inp_norm)
//   attention: q,k,v,r,rel,gate,sink projections; k,v through short convs (full layers add dynamic taps);
//              q L2-clip x attn_q_clip (cap qk_clip), k L2-clip (qk_clip), v weightless RMS, r L2-clip x attn_rel_clip
//              (rel_clip); full layers scale q,r by tau = 1 + alpha*ln(max(1,(pos+1)/n_floor));
//              logits = q.k/sqrt(d) + r[t].rel_bias[:,t-s] (0<=t-s<extent else 0), causal (+window if sliding),
//              a per-head sink logit in the softmax; output x sigmoid(gate) -> attn_output
//   a = shortconv_attn(attn_out)
//   block boundary (L % 6 == 0): blocks += [h], r = a; else r = h + a
//   m_in = attn_res(blocks, r) with ffn_res_query; [core pass>0: control(m_in, msg_mlp)]
//   m = moe(rms(m_in, ffn_norm)): sigmoid router, top-8 by score+exp_probs_b, weights normalised x route_scale;
//       expert g = W_g x s_g[e], u = W_u x s_u[e], h = tanh(g/4)*4*sigmoid(g)*tanh(u/25)*25, out = (h W_d) s_d[e]
//   m = shortconv_mlp(m); h = r + m
// end: h = attn_res(blocks, h) with output_res_query -> output_norm -> output -> 30*tanh(l/30), first unpadded logits
//
// The reference is ref/inkling2.py (float64). This engine is parity-tested against it seam by seam. The GGUF is the
// llama.cpp master layout (attn_rel, attn_rel_bias, attn_res_query, attn_k_inp_norm, fused ffn_gate_up_exps, short
// convs [taps][C] with the residual and scale folded into the last tap).
//
// NOTE (2026-11-21): the forward is complete for the no-loop prefix (layers 0..5) and the full 48-visit loop. GPU
// parity has not been run yet (the production GPUs are busy) - see GPU-TEST-READY.md.
#include "debug.hpp"
#include "dequant.hpp"
#include "dev_alloc.cuh"
#include "engine.hpp"
#include "kv_store.hpp"
#include "prefill_observer.hpp"
#include "token_sampling.hpp"
#include "common_kernels.cuh"
#include "moe_common.cuh"
#include "gguf.hpp"
#include "gguf_mapped.hpp"
#include "inkling2_kernels.cuh"
#include "quant_kernels.cuh"
#include "weight_format.hpp"

#include <chrono>
#include <random>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {
namespace {

using bf16 = __nv_bfloat16;

void ck_at(cudaError_t e, int line) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2: ") + cudaGetErrorString(e) + " (inkling2.cpp:" + std::to_string(line) + ")");
}
#define ck(e) ck_at((e), __LINE__)
// A blocking copy on stream s. This build has no legacy default stream (CMakeLists.txt defines
// CUDA_API_PER_THREAD_DEFAULT_STREAM), so a plain cudaMemcpy runs on the calling thread's own stream and does not wait
// for the kernels still queued on a stage's stream: a host read through it can see the buffer before they wrote it.
#define copy_sync(dst, src, n, kind, s) do { ck(cudaMemcpyAsync((dst), (src), (n), (kind), (s))); ck(cudaStreamSynchronize(s)); } while (0)
// A cuBLAS failure with its status and the line that asked for the operation.
// gemm_h(...) reports a failure at the line that called it
#define gemm_h(...) gemm_h_at(__VA_ARGS__, __LINE__)
// A checkpoint of the TQLLM_INKLING2_SYNC diagnostic mode (Inkling2Engine::sync_check_at); nothing when it is off.
#define sync_check(what) do { if (sync_) sync_check_at((what), __LINE__); } while (0)

struct Buf {
    void * p = nullptr;
    size_t bytes = 0;
    Buf() = default;
    explicit Buf(size_t n) : bytes(n) { if (bytes) ck(tqllm_cuda::dev_malloc(&p, bytes)); }
    ~Buf() { if (p) tqllm_cuda::dev_free(p); }
    Buf(Buf && o) noexcept : p(std::exchange(o.p, nullptr)), bytes(std::exchange(o.bytes, 0)) {}
    Buf & operator=(Buf && o) noexcept {
        if (this != &o) { if (p) tqllm_cuda::dev_free(p); p = std::exchange(o.p, nullptr); bytes = std::exchange(o.bytes, 0); }
        return *this;
    }
    Buf(const Buf &) = delete;
    Buf & operator=(const Buf &) = delete;
    template <class T> T * as() const { return static_cast<T *>(p); }
};

// A projection owned by its registered weight format.
struct M {
    Mat m;
    int rows = 0, cols = 0;
};

struct VisitState {
    __half * keys = nullptr, * values = nullptr;
    Buf k_hist, v_hist, post_hist, mlp_hist;
};

struct Layer {
    const float * attn_norm = nullptr;   // sliding layer: one norm for the three inputs
    const float * q_inp_norm = nullptr, * k_inp_norm = nullptr, * v_inp_norm = nullptr;  // full layer
    const float * ffn_norm = nullptr;
    M wq, wk, wv, wr, wgate, wsink, wo;
    const float * k_conv = nullptr, * v_conv = nullptr, * post_conv = nullptr, * mlp_conv = nullptr;  // [taps][C] folded
    M k_conv_proj, v_conv_proj;  // [C][taps] BF16 (full layers)
    const float * q_clip = nullptr;   // [hd][heads]
    const float * r_clip = nullptr;   // [d_rel][heads]
    const float * rel_bias = nullptr; // [d_rel][extent]
    const float * attn_res_q = nullptr;
    int attn_res_nq = 1;
    const float * ffn_res_q = nullptr;
    const float * loop_msg_query = nullptr;  // mainline: Q, K, V, FFN rows in one physical layer
    const float * loop_attn_down = nullptr, * loop_attn_forget = nullptr, * loop_attn_gain = nullptr;
    const float * loop_k_down = nullptr, * loop_k_forget = nullptr, * loop_k_gain = nullptr;
    const float * loop_v_down = nullptr, * loop_v_forget = nullptr, * loop_v_gain = nullptr;
    const float * loop_mlp_down = nullptr, * loop_mlp_forget = nullptr, * loop_mlp_gain = nullptr;
    M router;            // [experts][C]
    const float * probs_bias = nullptr, * e_gu_scale = nullptr, * e_down_scale = nullptr;
    Mat e_gu, e_down;
    bool sliding = true;
};

class Inkling2Engine final : public Engine {
public:
    Inkling2Engine(const std::string & path, uint32_t context)
        : context_(context ? context : 32768) {
        const char * sync = std::getenv("TQLLM_INKLING2_SYNC");
        sync_ = sync && *sync ? std::atoi(sync) : 0;
        if (sync_) std::fprintf(stderr, "inkling2: TQLLM_INKLING2_SYNC=%d: device sync + error check after every group of operations (diagnostic, slow)\n", sync_);
        meta_ = gguf_open(path);
        configure();
        split_setup();
        map_files(meta_);
        load();
        alloc_state();
    }
    ~Inkling2Engine() override {
        L_.clear();
        owned_.clear();
        for (int d = 0; d < 2; ++d) {
            if (stream_[d]) {
                cudaSetDevice(cuda_of_[d]);
                st_[d].kv_pool[0].reset();
                st_[d].kv_pool[1].reset();
                cudaStreamDestroy(stream_[d]);
            }
        }
        for (int d = 0; d < 2; ++d)
            if (hop_ev_[d]) cudaEventDestroy(hop_ev_[d]);
    }

    void synchronize() override { ck(cudaSetDevice(cuda_of_[cur_dev_])); ck(cudaStreamSynchronize(stream())); }
    uint32_t layers() const override { return uint32_t(n_layers_); }
    uint32_t hidden() const override { return uint32_t(H_); }
    uint32_t vocab() const override { return uint32_t(vocab_); }
    uint32_t eos() const override { return eos_; }
    uint32_t context() const override { return context_; }

    uint32_t step(uint32_t tok, uint32_t pos, bool sample, float *) override {
        // tqserve runs every request on a thread of its own, and a new thread starts on CUDA device 0: make the device
        // of the stage this engine is on current before any of that stage's streams is used.
        ck(cudaSetDevice(cuda_of_[cur_dev_]));
        run_token(tok, pos);
        if (sample) {
            compute_logits();
            return next_token();
        }
        return 0;
    }

    bool incremental() const override { return true; }
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence) override {
        sampling_.temperature = temperature;
        sampling_.top_p = top_p;
        sampling_.top_k = top_k;
        sampling_.presence = presence;
        rng_.seed(seed);
    }
    void set_penalties(float repeat, float frequency) override {
        sampling_.repeat = repeat;
        sampling_.frequency = frequency;
    }
    void set_ignore_eos(bool value) override { ignore_eos_ = value; }
    void spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int) override {
        if (prompt.empty() || uint64_t(prompt.size()) + count > context_) throw std::invalid_argument("inkling2: request exceeds context");
        generated_.clear();
        remaining_ = count;
        next_pos_ = uint32_t(prompt.size());
        const auto start = Clock::now();
        if (count) {
            for (uint32_t pos = 0; pos < next_pos_; ++pos) {
                const bool last = pos + 1 == next_pos_;
                const uint32_t next = step(prompt[pos], pos, last, nullptr);
                if (last) generated_.push_back(next);
                if (prefill_observer && (last || (pos + 1) % 32 == 0)) prefill_observer(pos + 1, next_pos_, 0);
            }
        }
        decode_start_ = Clock::now();
        prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(decode_start_ - start).count();
    }
    bool spec_advance() override {
        if (generated_.size() >= remaining_ || (!ignore_eos_ && generated_.back() == eos_)) return false;
        generated_.push_back(step(generated_.back(), next_pos_++, true, nullptr));
        return true;
    }
    const std::vector<uint32_t> & spec_tokens(int) const override { return generated_; }
    std::vector<uint32_t> spec_finish(uint64_t * stats) override {
        if (stats) {
            stats[0] = generated_.size(); stats[1] = stats[2] = 0;
            stats[3] = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - decode_start_).count();
            stats[4] = prefill_us_;
        }
        return std::move(generated_);
    }
    std::vector<uint32_t> generate_spec(const std::vector<uint32_t> & prompt, uint32_t count, int draft,
                                        uint64_t * stats) override {
        spec_begin(prompt, count, draft);
        while (spec_advance()) {}
        return spec_finish(stats);
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point decode_start_;
    uint64_t prefill_us_ = 0;
    SamplingParams sampling_;
    std::mt19937_64 rng_;
    std::vector<uint32_t> generated_;
    uint32_t next_pos_ = 0, remaining_ = 0;
    bool ignore_eos_ = false;
    // ---- config ----
    uint32_t context_ = 32768, eos_ = 0;
    int H_ = 0, n_visits_ = 0, n_pass_ = 1, n_layers_ = 0;
    int core0_ = 0, core1_ = 0;
    int n_head_ = 0, n_kv_ = 0, hd_ = 0;
    int d_rel_ = 0, extent_ = 0, extent_swa_ = 0, window_ = 0;
    int taps_ = 4, block_ = 6;
    int n_exp_ = 0, k_exp_ = 0;
    float route_scale_ = 1.f, eps_ = 0.f;
    float qk_clip_ = 0.f, rel_clip_ = 0.f, n_floor_ = 1.f, log_alpha_ = 0.f;
    float gate_cap_ = 4.f, up_cap_ = 25.f, logit_cap_ = 30.f;
    int vocab_ = 0, vocab_unpadded_ = 0;
    std::vector<int> phys_;
    std::vector<bool> sliding_;

    std::vector<Layer> L_;
    std::vector<VisitState> visits_;
    M out_;
    const float * embd_norm_ = nullptr, * out_norm_ = nullptr, * out_res_q_ = nullptr;
    const float * loop_res_q_ = nullptr;    // [2 * n_core][C]
    const float * loop_kv_res_q_ = nullptr; // [2 * n_core][C]

    // TQLLM_INKLING2_SYNC=1: after every group of operations wait for the stage's device and read the CUDA error state,
    // so a failing kernel is reported with the group that launched it instead of at some later, unrelated call.
    // =2 also prints every checkpoint before waiting on it: the last line names a group that never returns.
    int sync_ = 0;
    std::string where_;  // the token / visit the checkpoints belong to (kept in the diagnostic mode only)
    void sync_check_at(const char * what, int line) {
        int dev = -1;
        ck_at(cudaGetDevice(&dev), line);
        if (sync_ >= 2) {
            std::fprintf(stderr, "inkling2: sync %s%s, stage %d = CUDA device %d (inkling2.cpp:%d)\n", what, where_.c_str(), cur_dev_, dev, line);
            std::fflush(stderr);
        }
        cudaError_t e = cudaDeviceSynchronize();      // every stream of the thread's current device
        const cudaError_t last = cudaGetLastError();  // a launch the runtime refused leaves its error only here
        if (e == cudaSuccess) e = last;
        if (e == cudaSuccess && dev == cuda_of_[cur_dev_]) return;
        throw std::runtime_error(std::string("inkling2: ") + (e != cudaSuccess ? cudaGetErrorString(e) : "the thread's CUDA device is not the stage's") +
                                 " after " + what + where_ + ", stage " + std::to_string(cur_dev_) + " = CUDA device " + std::to_string(cuda_of_[cur_dev_]) +
                                 ", current device " + std::to_string(dev) + " (inkling2.cpp:" + std::to_string(line) + ")");
    }

    int n_blocks_ = 0;
    int prev_nb_ = 0;  // number of block states retained from the previous pass
    // 2-GPU layer split
    int split_at_ = 0;  // 0 = single GPU; layers [0, split_at_) on dev 1, [split_at_, n_layers_) on dev 0
    int cur_dev_ = 0;   // 0 or 1
    int load_dev_ = 0;  // device the next put()/load targets (set per layer during load())
    cudaStream_t stream_[2] = {nullptr, nullptr};
    cudaEvent_t hop_ev_[2] = {nullptr, nullptr};
    int cuda_of_[2] = {0, 1};  // stage 0 -> cuda dev 1, stage 1 -> cuda dev 0
    // Per-stage scratch: each GPU owns its own activations so a stage's buffers never alias the other's.
    struct Stage {
        std::unique_ptr<KvMemory> kv_memory;
        std::unique_ptr<KvPool> kv_pool[2]; // full and sliding visits, each with its own cache row layout
        Buf h_;          // [C] current stream
        Buf blocks_;     // [n_blocks][C] block states
        Buf prev_bank_;
        Buf prev_h_;     // [C] previous pass's final stream (for loop messages)
        Buf logits_;     // [vocab] F32 on the output stage
        Buf sample_scratch_, candidates_;
        Buf s1_, s2_, s3_, s4_;
        Buf projected_half_;
        Buf inputs_, normalized_, residual_, messages_, pre_core_, pre_blocks_;
        Buf a_, b_, c_, d_, e_, f_, g_;  // attention scratch (F32 or half, per use)
        Buf moe_ids_;    // [K] expert IDs
        Buf moe_gu_;     // [K * 2 * FF] gate+up output (BF16)
        Buf moe_act_;    // [K * FF] activation (half)
        Buf moe_dn_;     // [K * C] down output (BF16)
        Buf moe_wts_;    // [K] weights (F32)
    };
    Stage st_[2];
    Stage & st() { return st_[cur_dev_]; }
    std::vector<std::unique_ptr<MappedGguf>> maps_;
    std::map<std::string, const MappedGguf *> map_of_;
    std::vector<Buf> owned_;
    GgufFile meta_;

    std::string prefix_;
    double num(const std::string & k) const { return meta_.num(prefix_ + k); }
    double num(const std::string & k, double fb) const { return meta_.num(prefix_ + k, fb); }
    // First element of an array-valued KV (the root layout stores per-visit scalars as arrays);
    // falls back to the scalar value when the key holds a number.
    double arr0(const std::string & full_key) const {
        if (!meta_.has(full_key)) return 0.0;
        const auto & v = meta_.kv.at(full_key);
        if (!v.arr.empty()) return v.arr[0];
        return v.num;
    }

    // 2-GPU layer split: TQLLM_SPLIT=K puts layers [0,K) on GPU 1, [K,n) + head on GPU 0.
    // The loop (core layers) must stay on one GPU so the 3 passes never cross PCIe: with the
    // Inkling-2-Mini byte budget (layers 20-23 are 3.3x the others) only K at a loop boundary fits
    // 16 GB per card; see the budget table in PROGRESS.md.
    void split_setup() {
        const char * e = std::getenv("TQLLM_SPLIT");
        if (!e || !*e) return;
        const int k = std::atoi(e);
        int n = 0;
        ck(cudaGetDeviceCount(&n));
        if (n < 2) throw std::runtime_error("TQLLM_SPLIT: needs two CUDA devices");
        if (k <= 0 || k >= n_layers_) throw std::runtime_error("TQLLM_SPLIT: K must be in 1.." + std::to_string(n_layers_ - 1));
        if (k > core0_ && k < core1_) throw std::runtime_error("inkling2: keep the repeated core on one GPU");
        split_at_ = k;
        cuda_of_[0] = 1;  // stage 0 (layers [0,K)) on CUDA device 1
        cuda_of_[1] = 0;  // stage 1 (layers [K,n) + head) on CUDA device 0
        cur_dev_ = 0;
        load_dev_ = 0;
        std::fprintf(stderr, "inkling2: layer split: layers 0-%d on GPU 1, %d-%d + head on GPU 0\n", k - 1, k, n_layers_ - 1);
    }
    int head_stage() const { return split_at_ ? 1 : 0; }
    int dev_of(int l) const { return split_at_ && l >= split_at_ ? 1 : 0; }
    // Create the stage stream on its owning device.
    void ensure_dev(int d) {
        if (!stream_[d]) {
            ck(cudaSetDevice(cuda_of_[d]));
            ck(cudaStreamCreate(&stream_[d]));
        }
    }
    // Switch the active stage. Records an event on the old stream so the new stage's first work
    // (which reads the hopped activation) is ordered after the old stage finished producing it.
    void use_dev(int d) {
        if (!split_at_ || d == cur_dev_) return;
        // CUDA: an event and the stream it is recorded on must live on the same device, so one event per device.
        ck(cudaSetDevice(cuda_of_[cur_dev_]));
        if (!hop_ev_[cur_dev_]) ck(cudaEventCreateWithFlags(&hop_ev_[cur_dev_], cudaEventDisableTiming));
        ck(cudaEventRecord(hop_ev_[cur_dev_], stream_[cur_dev_]));
        ensure_dev(d);
        ck(cudaSetDevice(cuda_of_[d]));
        ck(cudaStreamWaitEvent(stream_[d], hop_ev_[cur_dev_], 0));
        cur_dev_ = d;
    }
    cudaStream_t stream() const { return stream_[cur_dev_]; }
    // The stream a load targets: load_dev_ during load(), cur_dev_ otherwise.
    cudaStream_t load_stream() const { return stream_[load_dev_]; }

    void configure() {
        const std::string arch = meta_.str("general.architecture");
        const bool physical_layout = arch == "inkling2_loop";
        if (!physical_layout && arch != "inkling2") throw std::runtime_error("inkling2: wrong architecture");
        prefix_ = arch + ".";
        if (physical_layout && (num("loop.varying", 0) != 0 || num("loop.gate_rank") != 64))
            throw std::runtime_error("inkling2: expected fixed loop and gate rank 64");
        H_ = int(num("embedding_length"));
        const int blocks = int(num("block_count"));
        n_pass_ = int(num(physical_layout ? "loop.count" : "num_loops", 1));
        core0_ = int(num(physical_layout ? "loop.start" : "loop.core_start", 0));
        core1_ = int(num(physical_layout ? "loop.end" : "loop.core_end", 0));
        if (core0_ < 0 || core1_ <= core0_ || core1_ > blocks || n_pass_ < 1)
            throw std::runtime_error("inkling2: invalid loop bounds");
        // head_count / head_count_kv are stored as arrays (one value per visit) in the root layout;
        // GgufFile::num returns 0 for an array, so read the first element explicitly.
        n_head_ = int(arr0(prefix_ + "attention.head_count"));
        n_kv_ = int(arr0(prefix_ + "attention.head_count_kv"));
        hd_ = int(num("attention.key_length"));
        d_rel_ = int(num("attention.relative_length"));
        extent_ = int(num("attention.relative_extent"));
        extent_swa_ = int(num("rel_extent_swa", extent_));
        window_ = int(num("attention.sliding_window"));
        taps_ = int(num("shortconv.l_cache", 4));
        block_ = int(num("attn_res.block_size", 6));
        n_exp_ = int(num("expert_count"));
        k_exp_ = int(num("expert_used_count"));
        route_scale_ = float(num("expert_weights_scale", 1.0));
        eps_ = float(num("attention.layer_norm_rms_epsilon", 1e-5));
        qk_clip_ = float(num("attention.qk_clip", 0.0));
        rel_clip_ = float(num("attention.relative_clip", 0.0));
        n_floor_ = float(num("attention.log_scale_floor", 1.0));
        log_alpha_ = float(num("attention.log_scale_alpha", 0.0));
        gate_cap_ = float(num("activation.situ_beta", 4.0));
        up_cap_ = float(num("activation.situ_linear_beta", 25.0));
        logit_cap_ = float(num("final_logit_softcapping", 30.0));
        vocab_ = int(num("vocab_size"));
        vocab_unpadded_ = int(num("vocab_size_unpadded", vocab_));
        if (meta_.has("tokenizer.ggml.eos_token_id")) eos_ = uint32_t(meta_.num("tokenizer.ggml.eos_token_id"));
        const int n_core = core1_ - core0_;
        const int64_t extra = int64_t(n_pass_ - 1) * n_core;
        if (physical_layout && extra > INT32_MAX - blocks) throw std::runtime_error("inkling2: loop count overflow");
        if (!physical_layout && extra >= blocks) throw std::runtime_error("inkling2: no physical layers");
        n_layers_ = physical_layout ? blocks : blocks - int(extra);
        n_visits_ = physical_layout ? blocks + int(extra) : blocks;
        if (n_layers_ <= 0 || core0_ < 0 || core1_ <= core0_ || core1_ > n_layers_ || n_pass_ < 1 ||
            vocab_unpadded_ < 16 || vocab_unpadded_ > vocab_ || window_ <= 0 || hd_ <= 0 || hd_ > 256 || n_head_ <= 0 || n_kv_ <= 0 || n_head_ % n_kv_ ||
            n_exp_ <= 0 || k_exp_ <= 0 || k_exp_ > n_exp_ || taps_ < 2 || block_ < 1 || n_layers_ / block_ + 2 > 64)
            throw std::runtime_error("inkling2: unsupported model dimensions");
        phys_.resize(n_visits_);
        for (int v = 0; v < n_visits_; ++v) {
            if (v < core0_) phys_[v] = v;
            else if (v < core0_ + n_pass_ * n_core) phys_[v] = core0_ + (v - core0_) % n_core;
            else phys_[v] = v - (n_pass_ - 1) * n_core;
        }
        // sliding_window_pattern: one bool per visit (48). A physical layer is "full" (not sliding)
        // if any of its visits has pattern=false. The root layout stores this as the only source of
        // the sliding/full distinction; full layers carry attn_{q,k,v}_inp_norm instead of attn_norm.
        sliding_.resize(n_layers_, true);
        const auto pattern = prefix_ + "attention.sliding_window_pattern";
        if (meta_.has(pattern)) {
            const auto & v = meta_.kv.at(pattern);
            if (v.arr.size() != size_t(blocks)) throw std::runtime_error("inkling2: wrong sliding window pattern length");
            for (size_t vi = 0; vi < v.arr.size(); ++vi) {
                const int visit = int(vi);
                const int p = physical_layout ? visit : phys_[visit];
                if (p >= 0 && p < n_layers_ && v.arr[vi] == 0.0) sliding_[p] = false;
            }
        }
    }

    // ---- tensor access ----
    const GgufTensor & tensor(const std::string & n) const {
        auto it = meta_.tensors.find(n);
        if (it == meta_.tensors.end()) throw std::runtime_error("inkling2: missing tensor " + n);
        return it->second;
    }
    const MappedGguf & map_for(const GgufTensor & t) const { return *map_of_.at(t.file.empty() ? meta_.path : t.file); }
    GgufBytes bytes(const std::string & n) const {
        const auto & t = tensor(n);
        if (!t.built()) return map_for(t).view(t);
        throw std::runtime_error("inkling2: built tensor " + n);
    }
    void map_files(const GgufFile & g) {
        for (const auto & [name, t] : g.tensors) {
            if (t.built()) continue;
            const std::string f = t.file.empty() ? g.path : t.file;
            if (map_of_.count(f)) continue;
            maps_.push_back(std::make_unique<MappedGguf>(f));
            map_of_[f] = maps_.back().get();
        }
    }
    const void * put(const void * host, size_t n) {
        // During load() the target device is load_dev_ (set per layer); during forward it is cur_dev_.
        const int d = load_dev_;
        ensure_dev(d);
        ck(cudaSetDevice(cuda_of_[d]));
        owned_.emplace_back(n);
        ck(cudaMemcpyAsync(owned_.back().p, host, n, cudaMemcpyHostToDevice, stream_[d]));
        ck(cudaStreamSynchronize(stream_[d]));
        return owned_.back().p;
    }
    // Point the next put()/load at device d (its stream must exist).
    void set_load_dev(int d) {
        ensure_dev(d);
        load_dev_ = d;
        ck(cudaSetDevice(cuda_of_[d]));
    }
    std::vector<float> host_f32(const std::string & n) const {
        const auto & t = tensor(n);
        uint64_t e = 1;
        for (auto d : t.ne) e *= d;
        std::vector<float> v(e);
        dequantize_host_rows(t.type, bytes(n).data, v.data(), int64_t(e / t.ne[0]), int64_t(t.ne[0]));
        return v;
    }
    const float * vec(const std::string & n) {
        const auto v = host_f32(n);
        return static_cast<const float *>(put(v.data(), v.size() * 4));
    }
    const float * expert_vec(const std::string & name) {
        const auto values = host_f32(name);
        if (int(values.size()) != n_exp_) throw std::runtime_error("inkling2: " + name + " is not one value per expert");
        return static_cast<const float *>(put(values.data(), values.size() * sizeof(float)));
    }
    // Fold a short-conv weight [C][taps] (master layout: unfolded): w *= sc; w[:, -1] += sc.
    // sc = 1/3 for full layers (dynamic taps add extra weight), 1/sqrt(taps+1) for sliding.
    // Returns a device pointer to the folded weight.
    const float * fold_conv_vec(const std::string & n, int C, int taps, bool full) {
        auto v = host_f32(n);
        const float sc = full ? 1.f / 3.f : 1.f / std::sqrt(float(taps + 1));
        for (int c = 0; c < C; ++c)
            for (int j = 0; j < taps; ++j) v[size_t(c) * taps + j] *= sc;
        for (int c = 0; c < C; ++c) v[size_t(c) * taps + (taps - 1)] += sc;
        return static_cast<const float *>(put(v.data(), v.size() * 4));
    }
    M load_mat(const std::string & n) {
        const auto & t = tensor(n);
        M r;
        r.cols = int(t.ne[0]);
        r.rows = int(t.ne[1]);
        const WeightFormat * wf = weight_format(t.type);
        if (!wf) throw std::runtime_error("inkling2: unsupported projection format for " + n);
        GgufBytes b = bytes(n);
        std::vector<char> raw(b.data, b.data + b.size);
        r.m = wf->load(raw, r.rows, r.cols, [this](const void * h, size_t bytes) { return put(h, bytes); }, load_stream());
        if (!r.m.fmt || r.m.fmt->basis() != Basis::Plain) throw std::runtime_error("inkling2: " + n + " format");
        return r;
    }
    Mat load_experts(const std::string & n) {
        const auto & t = tensor(n);
        if (t.ne.size() != 3 || int(t.ne[2]) != n_exp_) throw std::runtime_error("inkling2: " + n + " is not an expert stack");
        const WeightFormat * wf = weight_format(t.type);
        if (!wf) throw std::runtime_error("inkling2: no weight format for expert type " + std::to_string(t.type));
        GgufBytes b = bytes(n);
        std::vector<char> raw(b.data, b.data + b.size);
        Mat m = wf->load(raw, int(t.ne[1]) * n_exp_, int(t.ne[0]), [this](const void * h, size_t s) { return put(h, s); }, load_stream());
        if (!m.fmt || !m.fmt->has_expert_matvec()) throw std::runtime_error("inkling2: " + n + " has no grouped-expert kernel");
        return m;
    }

    void load() {
        // Normalize mapped embedding rows at the first layer; place the head at the last.
        set_load_dev(dev_of(0));
        embd_norm_ = vec("token_embd_norm.weight");
        set_load_dev(head_stage());
        out_ = load_mat("output.weight");
        out_norm_ = vec("output_norm.weight");
        out_res_q_ = vec("output_res_query.weight");
        // The global loop queries are read by the core layers' kernels (loop_messages), so they live on the core's
        // GPU: a kernel cannot read another device's memory.
        set_load_dev(dev_of(core0_));
        const bool physical_layout = prefix_ == "inkling2_loop.";
        if (!physical_layout) {
            loop_res_q_ = vec("loop_res_query.weight");
            loop_kv_res_q_ = vec("loop_kv_res_query.weight");
        }
        e_ff_ = int(num("expert_feed_forward_length", 1152));
        L_.resize(n_layers_);
        for (int l = 0; l < n_layers_; ++l) {
            set_load_dev(dev_of(l));  // this layer's weights land on its owner GPU
            auto & Y = L_[l];
            Y.sliding = sliding_[l];
            const std::string p = "blk." + std::to_string(l) + ".";
            Y.ffn_norm = vec(p + "ffn_norm.weight");
            if (sliding_[l]) Y.attn_norm = vec(p + "attn_norm.weight");
            else { Y.q_inp_norm = vec(p + "attn_q_inp_norm.weight"); Y.k_inp_norm = vec(p + "attn_k_inp_norm.weight"); Y.v_inp_norm = vec(p + "attn_v_inp_norm.weight"); }
            Y.wq = load_mat(p + "attn_q.weight");
            Y.wk = load_mat(p + "attn_k.weight");
            Y.wv = load_mat(p + "attn_v.weight");
            Y.wr = load_mat(p + "attn_rel.weight");
            Y.wgate = load_mat(p + "attn_gate.weight");
            Y.wsink = load_mat(p + "attn_sink_proj.weight");
            Y.wo = load_mat(p + "attn_output.weight");
            Y.k_conv = fold_conv_vec(p + "attn_k_conv.weight", n_kv_ * hd_, taps_, !sliding_[l]);
            Y.v_conv = fold_conv_vec(p + "attn_v_conv.weight", n_kv_ * hd_, taps_, !sliding_[l]);
            Y.post_conv = fold_conv_vec(p + "attn_post_conv.weight", H_, taps_, !sliding_[l]);
            Y.mlp_conv = fold_conv_vec(p + "ffn_post_conv.weight", H_, taps_, !sliding_[l]);
            Y.q_clip = vec(p + "attn_q_clip.weight");
            Y.r_clip = vec(p + "attn_rel_clip.weight");
            Y.rel_bias = vec(p + "attn_rel_bias.weight");
            if (meta_.tensors.count(p + "attn_res_query.weight")) {
                const auto & t = tensor(p + "attn_res_query.weight");
                Y.attn_res_q = vec(p + "attn_res_query.weight");
                Y.attn_res_nq = t.ne.size() > 1 ? int(t.ne[1]) : 1;
                if (Y.attn_res_nq != 1 && Y.attn_res_nq != 3)
                    throw std::runtime_error("inkling2: unexpected attention residual query count in " + p);
            } else if (l != 0) {
                throw std::runtime_error("inkling2: missing attention residual query in " + p);
            }
            Y.ffn_res_q = vec(p + "ffn_res_query.weight");
            if (!sliding_[l]) { Y.k_conv_proj = load_mat(p + "attn_k_conv_proj.weight"); Y.v_conv_proj = load_mat(p + "attn_v_conv_proj.weight"); }
            if (l >= core0_ && l < core1_) {
                if (physical_layout) {
                    const auto name = p + "loop_msg_query.weight";
                    const auto & dims = tensor(name).ne;
                    if (dims.size() != 2 || dims[0] != uint64_t(H_) || dims[1] != 4)
                        throw std::runtime_error("inkling2: expected four message queries in " + name);
                    Y.loop_msg_query = vec(name);
                }
                auto load_control = [&](const std::string & name, const float *& down, const float *& forget, const float *& gain) {
                    const auto base = p + "loop_" + name + (physical_layout ? "_ctrl_" : "_");
                    down = vec(base + "down.weight");
                    forget = vec(base + "forget.weight");
                    gain = vec(base + "gain.weight");
                };
                load_control(physical_layout ? "q" : "attn", Y.loop_attn_down, Y.loop_attn_forget, Y.loop_attn_gain);
                load_control("k", Y.loop_k_down, Y.loop_k_forget, Y.loop_k_gain);
                load_control("v", Y.loop_v_down, Y.loop_v_forget, Y.loop_v_gain);
                load_control(physical_layout ? "ffn" : "mlp", Y.loop_mlp_down, Y.loop_mlp_forget, Y.loop_mlp_gain);
            }
            Y.router = load_mat(p + "ffn_gate_inp.weight");
            if (meta_.tensors.count(p + "exp_probs_b.bias")) Y.probs_bias = expert_vec(p + "exp_probs_b.bias");
            Y.e_gu = load_experts(p + "ffn_gate_up_exps.weight");
            Y.e_down = load_experts(p + "ffn_down_exps.weight");
            if (meta_.tensors.count(p + "ffn_gate_up_exps.scale")) Y.e_gu_scale = expert_vec(p + "ffn_gate_up_exps.scale");
            if (meta_.tensors.count(p + "ffn_down_exps.scale")) Y.e_down_scale = expert_vec(p + "ffn_down_exps.scale");
        }

    }

    void alloc_state() {
        // Streams + cuBLAS for every device that will be used.
        ensure_dev(0);
        if (split_at_) ensure_dev(1);
        cur_dev_ = 0;
        // Per-stage scratch, allocated on that stage's own device.
        const size_t qd = size_t(n_head_) * hd_;
        const size_t w = std::max(qd, size_t(H_));
        for (int d = 0; d < 2; ++d) {
            if (!split_at_ && d != 0) continue;
            Stage & S = st_[d];
            set_dev_for_alloc(d);
            S.h_ = Buf(size_t(H_) * 2);
            S.inputs_ = Buf(3 * S.h_.bytes); S.normalized_ = Buf(3 * S.h_.bytes);
            S.residual_ = Buf(S.h_.bytes); S.messages_ = Buf(4 * S.h_.bytes);
            S.pre_core_ = Buf(S.h_.bytes); S.pre_blocks_ = Buf(64 * S.h_.bytes);
            S.blocks_ = Buf(size_t(64) * H_ * 2);
            S.prev_bank_ = Buf(size_t(64) * H_ * 2);
            S.prev_h_ = Buf(size_t(H_) * 2);
            S.s1_ = Buf(w * 2); S.s2_ = Buf(w * 2); S.s3_ = Buf(w * 2); S.s4_ = Buf(w * 2);
            S.a_ = Buf(w * 4); S.b_ = Buf(w * 4); S.c_ = Buf(w * 4); S.d_ = Buf(w * 4);
            // e_ takes the gate projection, and attn_gate has one row per head x dim (qd floats), not one per head
            S.e_ = Buf(w * 4); S.f_ = Buf(w * 4); S.g_ = Buf(w * 2);
            S.moe_ids_ = Buf(size_t(k_exp_) * 4);
            S.moe_gu_ = Buf(size_t(k_exp_) * 2 * e_ff_ * 2);  // BF16
            S.moe_act_ = Buf(size_t(k_exp_) * e_ff_ * 2);     // half
            S.moe_dn_ = Buf(size_t(k_exp_) * H_ * 2);         // BF16
            S.moe_wts_ = Buf(size_t(k_exp_) * 4);
            S.projected_half_ = Buf(2 * (2 * qd + 2 * n_kv_ * hd_ + n_head_ * d_rel_ + n_head_));
            if (d == head_stage()) {
                S.logits_ = Buf(size_t(vocab_) * sizeof(float));
                S.sample_scratch_ = Buf(64 * 16 * (sizeof(float) + sizeof(uint32_t)));
                S.candidates_ = Buf(sizeof(Candidates));
            }
        }
        visits_.resize(phys_.size());
        const size_t kd = size_t(n_kv_) * hd_;
        // Repeated visits share weights, but each has an independent cache and convolution history.
        for (int stage = 0; stage <= head_stage(); ++stage) {
            set_dev_for_alloc(stage);
            Stage & s = st_[stage];
            s.kv_memory = make_cuda_kv_memory(stream_[stage]);
            for (int ring = 0; ring < 2; ++ring) {
                int count = 0;
                for (int layer : phys_) if (dev_of(layer) == stage && sliding_[layer] == bool(ring)) ++count;
                if (!count) continue;
                const uint32_t capacity = ring ? std::min(context_, uint32_t(window_)) : context_;
                KvGeometry g{count, kd * sizeof(__half), kd * sizeof(__half), capacity};
                s.kv_pool[ring] = std::make_unique<KvPool>(*s.kv_memory, g, 1, 0);
                KvPool & pool = *s.kv_pool[ring];
                const int page = pool.take(KvTier::device);
                int cache = 0;
                for (size_t v = 0; v < phys_.size(); ++v) {
                    const int layer = phys_[v];
                    if (dev_of(layer) != stage || sliding_[layer] != bool(ring)) continue;
                    visits_[v].keys = static_cast<__half *>(pool.page_ptr(KvTier::device, cache, true, page));
                    visits_[v].values = static_cast<__half *>(pool.page_ptr(KvTier::device, cache, false, page));
                    ++cache;
                }
            }
        }
        for (size_t v = 0; v < visits_.size(); ++v) {
            const int layer = phys_[v];
            set_dev_for_alloc(dev_of(layer));
            VisitState & state = visits_[v];
            state.k_hist = Buf(size_t(taps_ - 1) * kd * sizeof(__half));
            state.v_hist = Buf(state.k_hist.bytes);
            state.post_hist = Buf(size_t(taps_ - 1) * H_ * sizeof(__half));
            state.mlp_hist = Buf(state.post_hist.bytes);
        }
    }
    // Set the CUDA device for the next Buf allocation (which uses the current device's cudaMalloc).
    void set_dev_for_alloc(int d) {
        ensure_dev(d);
        ck(cudaSetDevice(cuda_of_[d]));
    }

    // Dispatch each projection through its own weight format (BF16, F32 or quantized).
    // (called through the gemm_h macro, which passes the caller's line for the error message)
    void gemm_h_at(const M & w, const __half * x, float * y, int n, int line) {
        w.m.fmt->matvec_f32(w.m, x, y, n, stream());
        ck_at(cudaGetLastError(), line);
    }

    void run_token(uint32_t tok, uint32_t pos) {
        if (pos >= context_ || tok >= uint32_t(vocab_)) throw std::out_of_range("inkling2: token or position out of range");
        if (sync_) where_ = " [pos " + std::to_string(pos) + "]";
        use_dev(dev_of(0));
        const auto & tensor_info = tensor("token_embd.weight");
        const size_t stride = tensor_info.bytes / tensor_info.ne[1];
        const char * row = bytes("token_embd.weight").data + size_t(tok) * stride;
        std::vector<float> embedding(H_);
        dequantize_host_rows(tensor_info.type, row, embedding.data(), 1, H_);
        std::vector<__half> e(H_);
        for (int c = 0; c < H_; ++c) e[c] = __float2half(embedding[c]);
        Stage & first = st();
        ck(cudaMemcpyAsync(first.h_.p, e.data(), H_ * sizeof(__half), cudaMemcpyHostToDevice, stream()));
        tqllm_cuda::ik2_rms_norm(first.h_.as<__half>(), embd_norm_, eps_, H_, first.h_.as<__half>(), stream());
        n_blocks_ = 0;
        // pre-core layers
        for (int l = 0; l < core0_; ++l) {
            if (dev_of(l) != cur_dev_) { hop_to(dev_of(l)); use_dev(dev_of(l)); }
            visit(l, l, 0, pos, nullptr, 0);
        }
        {
            // The core loop runs entirely on one stage (its layers' owner), so the 3 passes never
            // cross PCIe. Capture the pre-core state on that stage.
            const int core_dev = dev_of(core0_);
            if (core_dev != cur_dev_) { hop_to(core_dev); use_dev(core_dev); }
            Stage & S = st();
            Buf & pre_h = S.pre_core_, & pre_blk = S.pre_blocks_;
            int pre_nb = n_blocks_;
            // (these device copies go on the stage's stream, after the visits that produce what they read)
            ck(cudaMemcpyAsync(pre_h.p, S.h_.p, H_ * 2, cudaMemcpyDeviceToDevice, stream()));
            ck(cudaMemcpyAsync(pre_blk.p, S.blocks_.p, size_t(pre_nb) * H_ * 2, cudaMemcpyDeviceToDevice, stream()));
            sync_check("loop: snapshot of the pre-core state");
            for (int p = 0; p < n_pass_; ++p) {
                if (p > 0) {
                    ck(cudaMemcpyAsync(S.h_.p, pre_h.p, H_ * 2, cudaMemcpyDeviceToDevice, stream()));
                    ck(cudaMemcpyAsync(S.blocks_.p, pre_blk.p, size_t(pre_nb) * H_ * 2, cudaMemcpyDeviceToDevice, stream()));
                    n_blocks_ = pre_nb;
                    sync_check("loop: restore of the pre-core state");
                }
                for (int l = core0_; l < core1_; ++l) {
                    int v = core0_ + p * (core1_ - core0_) + (l - core0_);
                    visit(l, v, p, pos, nullptr, 0);
                }
                if (p < n_pass_ - 1) {
                    ck(cudaMemcpyAsync(S.prev_h_.p, S.h_.p, H_ * 2, cudaMemcpyDeviceToDevice, stream()));
                    prev_nb_ = n_blocks_ - pre_nb - 1;
                    if (prev_nb_ > 0) ck(cudaMemcpyAsync(S.prev_bank_.p, S.blocks_.as<__half>() + size_t(pre_nb + 1) * H_,
                                                         size_t(prev_nb_) * H_ * sizeof(__half), cudaMemcpyDeviceToDevice, stream()));
                    sync_check("loop: copy of the pass's final stream");
                }
            }
        }
        // post-core layers
        for (int l = core1_; l < n_layers_; ++l) {
            if (split_at_ && dev_of(l) != cur_dev_) {
                hop_to(dev_of(l));  // move h_ + blocks_ to the next stage
                use_dev(dev_of(l));
                sync_check("hop of the stream and the blocks to the next stage");
            }
            visit(l, l + (n_pass_ - 1) * (core1_ - core0_), 0, pos, nullptr, 0);
        }
    }
    // Move the residual vector (and block states) from the current stage to stage d across PCIe.
    // Only the small activation crosses (hidden 2304 half + up to 64 blocks), never the weights.
    // Host-staged: consumer GPUs (RTX 5060 Ti) have no P2P, so cudaMemcpyPeerAsync would fail.
    // The data is tiny (2304 half + up to 64 blocks), so the host round-trip is cheap.
    void hop_to(int d) {
        if (!split_at_ || d == cur_dev_) return;
        Stage & from = st_[cur_dev_], & to = st_[d];
        const int from_c = cuda_of_[cur_dev_], to_c = cuda_of_[d];
        // Each side copies on its own stage's stream: the read waits for the kernels that produce the vector, and the
        // write lands before the next stage's kernels (queued on that same stream) read it.
        const cudaStream_t from_s = stream_[cur_dev_], to_s = stream_[d];
        // h_ : device(from) -> host -> device(to)
        std::vector<__half> hbuf(H_);
        ck(cudaSetDevice(from_c));
        copy_sync(hbuf.data(), from.h_.p, H_ * 2, cudaMemcpyDeviceToHost, from_s);
        ck(cudaSetDevice(to_c));
        copy_sync(to.h_.p, hbuf.data(), H_ * 2, cudaMemcpyHostToDevice, to_s);
        // blocks_ : same pattern
        if (n_blocks_) {
            std::vector<__half> bbuf(size_t(n_blocks_) * H_);
            ck(cudaSetDevice(from_c));
            copy_sync(bbuf.data(), from.blocks_.p, bbuf.size() * 2, cudaMemcpyDeviceToHost, from_s);
            ck(cudaSetDevice(to_c));
            copy_sync(to.blocks_.p, bbuf.data(), bbuf.size() * 2, cudaMemcpyHostToDevice, to_s);
        }
    }

    void visit(int l, int v, int pass, uint32_t pos, const __half * prev_last, int prev_nblocks) {
        Stage & S = st();
        if (prev_last == nullptr && pass > 0) { prev_last = S.prev_h_.as<__half>(); prev_nblocks = prev_nb_; }
        auto & Y = L_[l];
        auto & state = visits_[v];
        if (pos == 0) {
            for (Buf * buf : {&state.k_hist, &state.v_hist, &state.post_hist, &state.mlp_hist})
                ck(cudaMemsetAsync(buf->p, 0, buf->bytes, stream()));
        }
        const int C = H_;
        __half * h = S.h_.as<__half>();
        if (sync_) where_ = " [layer " + std::to_string(l) + ", visit " + std::to_string(v) + ", pass " + std::to_string(pass) + ", pos " + std::to_string(pos) + "]";

        // h_in = attn_res(blocks, h) (layer 0: (h,h,h))
        Buf & insbuf = S.inputs_;
        __half * ins[3];
        if (l == 0) {
            for (int j = 0; j < 3; ++j) {
                tqllm_cuda::ik2_hcopy(h, static_cast<__half *>(insbuf.p) + j * C, C, stream());
                ins[j] = static_cast<__half *>(insbuf.p) + j * C;
            }
        } else {
            tqllm_cuda::ik2_attn_res(S.blocks_.as<__half>(), n_blocks_, h, Y.attn_res_q, Y.attn_res_nq, C, static_cast<__half *>(insbuf.p), stream());
            auto * o = static_cast<__half *>(insbuf.p);
            if (Y.attn_res_nq == 1) {
                tqllm_cuda::ik2_hcopy(o, o + C, C, stream());
                tqllm_cuda::ik2_hcopy(o, o + 2 * C, C, stream());
            }
            for (int j = 0; j < 3; ++j) ins[j] = o + j * C;
        }
        sync_check("visit: residual mix of the attention inputs");
        if (pass > 0) {
            Buf & msgs = S.messages_;
            loop_messages(l, prev_last, prev_nblocks, static_cast<__half *>(msgs.p));
            tqllm_cuda::ik2_control(ins[0], static_cast<__half *>(msgs.p), Y.loop_attn_down, Y.loop_attn_forget, Y.loop_attn_gain, C, ins[0], stream());
            tqllm_cuda::ik2_control(ins[1], static_cast<__half *>(msgs.p) + C, Y.loop_k_down, Y.loop_k_forget, Y.loop_k_gain, C, ins[1], stream());
            tqllm_cuda::ik2_control(ins[2], static_cast<__half *>(msgs.p) + 2 * C, Y.loop_v_down, Y.loop_v_forget, Y.loop_v_gain, C, ins[2], stream());
            sync_check("visit: loop control of the attention inputs");
        }
        // norm the inputs
        Buf & xsbuf = S.normalized_;
        __half * xs[3];
        for (int j = 0; j < 3; ++j) {
            xs[j] = static_cast<__half *>(xsbuf.p) + j * C;
            if (sliding_[l]) tqllm_cuda::ik2_rms_norm(ins[j], Y.attn_norm, eps_, C, xs[j], stream());
            else tqllm_cuda::ik2_rms_norm(ins[j], (j == 0 ? Y.q_inp_norm : j == 1 ? Y.k_inp_norm : Y.v_inp_norm), eps_, C, xs[j], stream());
        }
        sync_check("visit: norm of the attention inputs");
        // attention
        __half * attn_out = S.s1_.as<__half>();
        __half * pre_kh = S.s2_.as<__half>();  // [kd]
        __half * pre_vh = S.s3_.as<__half>();  // [kd]
        attention(Y, state, pos, xs[0], xs[1], xs[2], attn_out, pre_kh, pre_vh);
        // a = shortconv_attn(attn_out)
        __half * a = S.g_.as<__half>();  // g_ is free after attention
        const __half * phist = pos > 0 ? state.post_hist.as<__half>() : nullptr;
        tqllm_cuda::ik2_shortconv(Y.post_conv, taps_, attn_out, a, C, 1, phist, stream());
        sync_check("visit: short conv after the attention");
        // update conv histories: shift and append current token's pre-conv inputs
        const int kd = n_kv_ * hd_;
        tqllm_cuda::ik2_conv_hist_update(state.k_hist.as<__half>(), pre_kh, taps_, kd, stream());
        tqllm_cuda::ik2_conv_hist_update(state.v_hist.as<__half>(), pre_vh, taps_, kd, stream());
        tqllm_cuda::ik2_conv_hist_update(state.post_hist.as<__half>(), attn_out, taps_, C, stream());
        // block boundary (device-resident): at a boundary, push h onto the block bank and r = a;
        // otherwise r = h + a.
        __half * r;
        Buf & rbuf = S.residual_;
        if (l % block_ == 0) {
            if (n_blocks_ < 64)
                tqllm_cuda::ik2_hcopy(h, S.blocks_.as<__half>() + size_t(n_blocks_) * C, C, stream());
            ++n_blocks_;
            tqllm_cuda::ik2_hcopy(a, rbuf.as<__half>(), C, stream());
            r = rbuf.as<__half>();
        } else {
            tqllm_cuda::ik2_hadd(h, a, rbuf.as<__half>(), C, stream());
            r = rbuf.as<__half>();
        }
        sync_check("visit: block bank push / residual add after the attention");
        // m_in = attn_res(blocks, r) with ffn_res_q
        __half * m_in = S.s3_.as<__half>();
        tqllm_cuda::ik2_attn_res(S.blocks_.as<__half>(), n_blocks_, r, Y.ffn_res_q, 1, C, m_in, stream());
        sync_check("visit: residual mix of the MoE input");
        if (pass > 0) {
            Buf & msgs = S.messages_;
            tqllm_cuda::ik2_control(m_in, static_cast<__half *>(msgs.p) + 3 * C, Y.loop_mlp_down, Y.loop_mlp_forget, Y.loop_mlp_gain, C, m_in, stream());
            sync_check("visit: loop control of the MoE input");
        }
        // m = moe(rms(m_in, ffn_norm))
        __half * mnorm = S.s4_.as<__half>();
        tqllm_cuda::ik2_rms_norm(m_in, Y.ffn_norm, eps_, C, mnorm, stream());
        sync_check("visit: norm of the MoE input");
        __half * m = S.g_.as<__half>();  // g_ is free after post_conv
        moe(Y, mnorm, m);
        // m = shortconv_mlp(m)
        const __half * mhist = pos > 0 ? state.mlp_hist.as<__half>() : nullptr;
        __half * filtered_m = S.s2_.as<__half>();
        tqllm_cuda::ik2_shortconv(Y.mlp_conv, taps_, m, filtered_m, C, 1, mhist, stream());
        tqllm_cuda::ik2_conv_hist_update(state.mlp_hist.as<__half>(), m, taps_, C, stream());
        sync_check("visit: short conv after the MoE");
        // h = r + m (device-resident)
        tqllm_cuda::ik2_hadd(r, filtered_m, h, C, stream());
        sync_check("visit: residual add after the MoE");
        sync_check("visit: conv history update");
        if (sync_) where_ = " [pos " + std::to_string(pos) + "]";
    }

    // The four loop messages (m_q, m_k, m_v, m_mlp) of core layer l: attn_res(bank, last) with the loop_res_query /
    // loop_kv_res_query rows for this core index. bank = the previous pass's blocks after its first (prev_nblocks ..
    // n_blocks_); last = the previous pass's final stream.
    void loop_messages(int l, const __half * prev_last, int prev_nblocks, __half * out) {
        const int i = l - core0_;
        const int C = H_;
        const __half * bank = st().prev_bank_.as<__half>();
        // Emit [attention, key, value, MLP] directly; both query matrices contain two rows per core layer.
        if (const float * rows = L_[l].loop_msg_query) {
            tqllm_cuda::ik2_attn_res(bank, prev_nblocks, prev_last, rows, 4, C, out, stream());
        } else {
            const float * q = loop_res_q_ + 2 * i * C;
            tqllm_cuda::ik2_attn_res(bank, prev_nblocks, prev_last, q, 1, C, out, stream());
            tqllm_cuda::ik2_attn_res(bank, prev_nblocks, prev_last, loop_kv_res_q_ + 2 * i * C, 2, C, out + C, stream());
            tqllm_cuda::ik2_attn_res(bank, prev_nblocks, prev_last, q + C, 1, C, out + 3 * C, stream());
        }
        sync_check("loop messages: reorder");
        ck(cudaStreamSynchronize(stream()));  // tmp is freed here: nothing queued may still use it
    }

    // The attention of one layer at position pos: project q,k,v,r,gate,sink; conv k,v; clip; relative bias; softmax;
    // gate; output projection. Writes out [C]. Uses dedicated scratch buffers (a..g) to avoid aliasing.
    // pre_kh/pre_vh: scratch for the pre-conv k/v (for conv history).
    void attention(Layer & Y, VisitState & state, uint32_t pos, const __half * xq, const __half * xk, const __half * xv, __half * out,
                   __half * pre_kh, __half * pre_vh) {
        const int C = H_, D = hd_, H = n_head_, KV = n_kv_;
        const int qd = H * D, kd = KV * D;
        Stage & S = st();
        // projections (F32 scratch)
        float * qf = S.a_.as<float>();  // [qd]
        gemm_h(Y.wq, xq, qf, 1);
        float * kf = S.b_.as<float>();  // [kd]
        gemm_h(Y.wk, xk, kf, 1);
        float * vf = S.c_.as<float>();  // [kd]
        gemm_h(Y.wv, xv, vf, 1);
        float * rf = S.d_.as<float>();  // [H*d_rel]
        gemm_h(Y.wr, xq, rf, 1);
        float * gate = S.e_.as<float>();  // [H]
        gemm_h(Y.wgate, xq, gate, 1);
        float * sink = S.f_.as<float>();  // [H]
        gemm_h(Y.wsink, xq, sink, 1);
        sync_check("attention: q / k / v / r / gate / sink projections");
        // to half (device-resident conversion)
        __half * qh = S.projected_half_.as<__half>();
        __half * kh = qh + qd, * vh = kh + kd, * rh = vh + kd;
        __half * gateh = rh + H * d_rel_, * sinkh = gateh + qd;
        tqllm_cuda::common::f32_to_half(qf, qh, qd, stream());
        tqllm_cuda::common::f32_to_half(kf, kh, kd, stream());
        tqllm_cuda::common::f32_to_half(vf, vh, kd, stream());
        tqllm_cuda::common::f32_to_half(rf, rh, H * d_rel_, stream());
        tqllm_cuda::common::f32_to_half(gate, gateh, qd, stream());
        tqllm_cuda::common::f32_to_half(sink, sinkh, H, stream());
        // save pre-conv k/v for conv history (device copy)
        tqllm_cuda::ik2_hcopy(kh, pre_kh, kd, stream());
        tqllm_cuda::ik2_hcopy(vh, pre_vh, kd, stream());
        sync_check("attention: conversion to half + pre-conv copies");
        // k/v short conv (with history)
        const __half * khist = pos > 0 ? state.k_hist.as<__half>() : nullptr;
        const __half * vhist = pos > 0 ? state.v_hist.as<__half>() : nullptr;
        if (Y.sliding) {
            tqllm_cuda::ik2_shortconv(Y.k_conv, taps_, kh, kh, kd, 1, khist, stream());
            tqllm_cuda::ik2_shortconv(Y.v_conv, taps_, vh, vh, kd, 1, vhist, stream());
        } else {
            // full layer: dynamic conv (static + dynamic taps from input)
            // dyn = x @ conv_proj^T -> [kd] (per-channel, per-lag)
            float * dynk = S.e_.as<float>();  // [kd] = [KV*D] — overwrites gateh, but gateh is already converted
            gemm_h(Y.k_conv_proj, xk, dynk, 1);
            float * dynv = S.f_.as<float>();  // [kd]
            gemm_h(Y.v_conv_proj, xv, dynv, 1);
            tqllm_cuda::ik2_shortconv_dyn(Y.k_conv, taps_, kh, kh, kd, hd_, dynk, khist, stream());
            tqllm_cuda::ik2_shortconv_dyn(Y.v_conv, taps_, vh, vh, kd, hd_, dynv, vhist, stream());
        }
        sync_check("attention: k / v short conv");
        // clip q, k; norm v; clip r
        tqllm_cuda::ik2_l2clip(qh, Y.q_clip, qk_clip_, H, D, qh, stream());
        tqllm_cuda::ik2_l2clip(kh, nullptr, qk_clip_, KV, D, kh, stream());  // k uses qk_clip (no per-head clip weight)
        tqllm_cuda::ik2_rms_weightless(vh, KV, D, vh, stream());
        tqllm_cuda::ik2_l2clip(rh, Y.r_clip, rel_clip_, H, d_rel_, rh, stream());
        // full layers: q, r *= tau = 1 + alpha * ln(max(1, (pos+1)/n_floor))
        if (!Y.sliding && log_alpha_ != 0.f) {
            const float tau = 1.f + log_alpha_ * std::log(std::max(1.f, float(pos + 1) / n_floor_));
            tqllm_cuda::ik2_hscale(qh, tau, qd, stream());
            tqllm_cuda::ik2_hscale(rh, tau, H * d_rel_, stream());
        }
        sync_check("attention: q / k / r clip, v norm, tau");
        // write the current token's k/v into the cache at row pos (device-to-device)
        {
            __half * krow = state.keys + size_t(Y.sliding ? pos % window_ : pos) * KV * D;
            __half * vrow = state.values + size_t(Y.sliding ? pos % window_ : pos) * KV * D;
            ck(cudaMemcpyAsync(krow, kh, kd * 2, cudaMemcpyDeviceToDevice, stream()));
            ck(cudaMemcpyAsync(vrow, vh, kd * 2, cudaMemcpyDeviceToDevice, stream()));
        }
        sync_check("attention: KV cache write");
        // attention core
        tqllm_cuda::Ik2AttnCore ac{};
        ac.q = qh; ac.r = rh; ac.gate = gateh; ac.sink = sinkh;
        ac.rel_bias = Y.rel_bias;
        ac.H = H; ac.KV = KV; ac.D = D; ac.d_rel = d_rel_; ac.extent = Y.sliding ? extent_swa_ : extent_; ac.window = window_;
        ac.sliding = Y.sliding; ac.pos = pos;
        ac.kc = state.keys; ac.vc = state.values;
        __half * att = S.g_.as<__half>();  // [qd]
        tqllm_cuda::ik2_attention_core(ac, att, stream());
        sync_check("attention: core (scores, softmax, gate)");
        // out = att @ wo^T [C] (F32 scratch -> half, device-resident)
        float * of = S.a_.as<float>();  // [C]
        gemm_h(Y.wo, att, of, 1);
        tqllm_cuda::common::f32_to_half(of, out, C, stream());
        sync_check("attention: output projection");
    }

    // The MoE: sigmoid router, top-k, routed experts. x [C], out [C].
    void moe(Layer & layer, const __half * x, __half * out) {
        Stage & s = st();
        float * logits = s.s1_.as<float>();
        int * ids = s.moe_ids_.as<int>();
        float * weights = s.moe_wts_.as<float>();
        gemm_h(layer.router, x, logits, 1);
        tqllm_cuda::ik2_route(logits, layer.probs_bias, n_exp_, k_exp_, route_scale_, ids, weights, stream());
        layer.e_gu.fmt->expert_matvec(layer.e_gu, x, s.moe_gu_.as<bf16>(), 2 * e_ff_, ids, 1, k_exp_, false, stream());
        ck(cudaGetLastError());
        tqllm_cuda::ik2_expert_activation(s.moe_gu_.as<bf16>(), ids, layer.e_gu_scale, k_exp_, e_ff_, gate_cap_, up_cap_,
                                          s.moe_act_.as<__half>(), stream());
        layer.e_down.fmt->expert_matvec(layer.e_down, s.moe_act_.as<__half>(), s.moe_dn_.as<bf16>(), H_, ids, 1,
                                        k_exp_, true, stream());
        ck(cudaGetLastError());
        tqllm_cuda::ik2_expert_sum(s.moe_dn_.as<bf16>(), ids, layer.e_down_scale, weights, k_exp_, H_, out, stream());
        sync_check("moe: expert projections and weighted sum");
    }

    int e_ff_ = 0;  // expert feed-forward width (set in load)

    void compute_logits() {
        // The output projection and its normalization share the final stage.
        use_dev(head_stage());
        Stage & S = st();
        __half * h = S.h_.as<__half>();
        __half * hres = S.s1_.as<__half>();
        tqllm_cuda::ik2_attn_res(S.blocks_.as<__half>(), n_blocks_, h, out_res_q_, 1, H_, hres, stream());
        __half * hnorm = S.s2_.as<__half>();
        tqllm_cuda::ik2_rms_norm(hres, out_norm_, eps_, H_, hnorm, stream());
        sync_check("head: residual mix + output norm");
        gemm_h(out_, hnorm, S.logits_.as<float>(), 1);
        tqllm_cuda::ik2_softcap(S.logits_.as<float>(), vocab_unpadded_, logit_cap_, stream());
        sync_check("head: output projection + softcap");
    }

    struct Candidates { float logits[16]; uint32_t ids[16]; };
    uint32_t next_token() {
        Stage & s = st();
        float * logits = s.logits_.as<float>();
        auto * candidates = s.candidates_.as<Candidates>();
        if (sampling_.temperature <= 0.f) {
            tqllm_cuda::argmax_big(logits, vocab_unpadded_, candidates->ids, s.sample_scratch_.p, stream());
            uint32_t id;
            copy_sync(&id, candidates->ids, sizeof(id), cudaMemcpyDeviceToHost, stream());
            return id;
        }
        tqllm_cuda::topk16_big(logits, vocab_unpadded_, 1, candidates->logits, candidates->ids, s.sample_scratch_.p, stream());
        Candidates host;
        copy_sync(&host, candidates, sizeof(host), cudaMemcpyDeviceToHost, stream());
        return draw(top_dist(host.logits, host.ids, sampling_, generated_), rng_);
    }

};

std::unique_ptr<Engine> make_inkling2_engine_impl(const std::string & path, uint32_t context) {
    return std::make_unique<Inkling2Engine>(path, context);
}
}  // namespace

bool is_inkling2_gguf(const std::string & path) {
    try {
        const auto arch = gguf_architecture(path);
        return arch == "inkling2" || arch == "inkling2_loop";
    } catch (...) { return false; }
}

std::unique_ptr<Engine> make_inkling2_engine(const std::string & path, uint32_t context) {
    return make_inkling2_engine_impl(path, context);
}

}  // namespace tqllm
