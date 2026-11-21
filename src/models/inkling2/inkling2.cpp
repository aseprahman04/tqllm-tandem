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
#include "common_kernels.cuh"
#include "moe_common.cuh"
#include "gguf.hpp"
#include "gguf_mapped.hpp"
#include "inkling2_kernels.cuh"
#include "quant_kernels.cuh"
#include "weight_format.hpp"

#include <cublas_v2.h>
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

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2: ") + cudaGetErrorString(e));
}

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

// A projection: either a device matrix in a weight format (m) or a dense F32 buffer (f32).
struct M {
    Mat m;
    const float * f32 = nullptr;
    int rows = 0, cols = 0;
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
    const float * loop_attn_down = nullptr, * loop_attn_forget = nullptr, * loop_attn_gain = nullptr;
    const float * loop_k_down = nullptr, * loop_k_forget = nullptr, * loop_k_gain = nullptr;
    const float * loop_v_down = nullptr, * loop_v_forget = nullptr, * loop_v_gain = nullptr;
    const float * loop_mlp_down = nullptr, * loop_mlp_forget = nullptr, * loop_mlp_gain = nullptr;
    M router;            // [experts][C]
    const float * probs_bias = nullptr;  // [experts]
    Mat e_gu, e_down;    // stacked NVFP4 expert weights (gate|up rows stacked)
    const float * e_gu_scale = nullptr;  // [experts]
    const float * e_down_scale = nullptr;
    bool experts_nvfp4 = false;
    Buf kc, vc;  // [context][kv heads][hd] FP16
    bool sliding = true;
    // conv history: [taps-1][C] half, per conv (k, v, post, mlp)
    Buf k_hist, v_hist, post_hist, mlp_hist;
    int conv_pos = 0;  // number of tokens processed through this layer's convs
};

class Inkling2Engine final : public Engine {
public:
    Inkling2Engine(const std::string & path, uint32_t context)
        : context_(context ? context : 32768) {
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
            if (cublas_[d]) cublasDestroy(cublas_[d]);
            if (stream_[d]) cudaStreamDestroy(stream_[d]);
        }
        for (int d = 0; d < 2; ++d)
            if (hop_ev_[d]) cudaEventDestroy(hop_ev_[d]);
    }

    void synchronize() override { ck(cudaStreamSynchronize(stream())); }
    uint32_t layers() const override { return uint32_t(n_layers_); }
    uint32_t hidden() const override { return uint32_t(H_); }
    uint32_t vocab() const override { return uint32_t(vocab_); }
    uint32_t eos() const override { return eos_; }
    uint32_t context() const override { return context_; }

    uint32_t step(uint32_t tok, uint32_t pos, bool sample, float *) override {
        run_token(tok, pos);
        if (sample) {
            compute_logits();
            ck(cudaStreamSynchronize(stream()));
            return greedy_token();
        }
        return 0;
    }

private:
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
    M embd_, out_;
    const float * embd_norm_ = nullptr, * out_norm_ = nullptr, * out_res_q_ = nullptr;
    const float * loop_res_q_ = nullptr;    // [2 * n_core][C]
    const float * loop_kv_res_q_ = nullptr; // [2 * n_core][C]

    int n_blocks_ = 0;
    int prev_nb_ = 0; // previous pass's n_blocks (bank start)
    // 2-GPU layer split
    int split_at_ = 0;  // 0 = single GPU; layers [0, split_at_) on dev 1, [split_at_, n_layers_) on dev 0
    int cur_dev_ = 0;   // 0 or 1
    int load_dev_ = 0;  // device the next put()/load targets (set per layer during load())
    cudaStream_t stream_[2] = {nullptr, nullptr};
    cublasHandle_t cublas_[2] = {nullptr, nullptr};
    cudaEvent_t hop_ev_[2] = {nullptr, nullptr};
    int cuda_of_[2] = {0, 1};  // stage 0 -> cuda dev 1, stage 1 -> cuda dev 0
    // Per-stage scratch: each GPU owns its own activations so a stage's buffers never alias the other's.
    struct Stage {
        Buf h_;          // [C] current stream
        Buf blocks_;     // [n_blocks][C] block states
        Buf prev_h_;     // [C] previous pass's final stream (for loop messages)
        Buf logits_;     // [vocab] F32 (stage 1 only)
        Buf s1_, s2_, s3_, s4_;
        Buf a_, b_, c_, d_, e_, f_, g_;  // attention scratch (F32 or half, per use)
        Buf moe_ids_;    // [K] expert IDs
        Buf moe_gu_;     // [K * 2 * FF] gate+up output (BF16)
        Buf moe_act_;    // [K * FF] activation (half)
        Buf moe_dn_;     // [K * C] down output (BF16)
        Buf moe_wts_;    // [K] weights (F32)
        Buf moe_scales_; // [K] down scales (F32)
    };
    Stage st_[2];
    Stage & st() { return st_[cur_dev_]; }
    std::vector<std::unique_ptr<MappedGguf>> maps_;
    std::map<std::string, const MappedGguf *> map_of_;
    std::vector<Buf> owned_;
    GgufFile meta_;

    double num(const std::string & k) const { return meta_.num("inkling2." + k); }
    double num(const std::string & k, double fb) const { return meta_.num("inkling2." + k, fb); }
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
        split_at_ = k;
        cuda_of_[0] = 1;  // stage 0 (layers [0,K)) on CUDA device 1
        cuda_of_[1] = 0;  // stage 1 (layers [K,n) + head) on CUDA device 0
        cur_dev_ = 0;
        load_dev_ = 0;
        std::fprintf(stderr, "inkling2: layer split: layers 0-%d on GPU 1, %d-%d + head on GPU 0\n", k - 1, k, n_layers_ - 1);
    }
    int dev_of(int l) const { return split_at_ && l >= split_at_ ? 1 : 0; }
    // Ensure device d has a stream + cuBLAS handle; set it current. Does not touch cur_dev_.
    void ensure_dev(int d) {
        if (!stream_[d]) {
            ck(cudaSetDevice(cuda_of_[d]));
            ck(cudaStreamCreate(&stream_[d]));
            if (!cublas_[d]) {
                if (cublasCreate(&cublas_[d]) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("inkling2: cuBLAS init failed");
                cublasSetStream(cublas_[d], stream_[d]);
            }
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
    cublasHandle_t cublas() const { return cublas_[cur_dev_]; }
    // The stream a load targets: load_dev_ during load(), cur_dev_ otherwise.
    cudaStream_t load_stream() const { return stream_[load_dev_]; }

    void configure() {
        if (meta_.str("general.architecture") != "inkling2") throw std::runtime_error("inkling2: wrong architecture");
        H_ = int(num("embedding_length"));
        n_visits_ = int(num("block_count"));
        n_pass_ = int(num("num_loops", 1));
        core0_ = int(num("loop.core_start", 0));
        core1_ = int(num("loop.core_end", 0));
        // head_count / head_count_kv are stored as arrays (one value per visit) in the root layout;
        // GgufFile::num returns 0 for an array, so read the first element explicitly.
        n_head_ = int(arr0("inkling2.attention.head_count"));
        n_kv_ = int(arr0("inkling2.attention.head_count_kv"));
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
        n_layers_ = n_visits_ - (n_pass_ - 1) * n_core;
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
        if (meta_.has("inkling2.attention.sliding_window_pattern")) {
            const auto & v = meta_.kv.at("inkling2.attention.sliding_window_pattern");
            for (size_t vi = 0; vi < v.arr.size(); ++vi) {
                const int visit = int(vi);
                const int p = phys_[visit];
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
    // Fold a short-conv weight [C][taps] (master layout: unfolded): w *= sc; w[:, -1] += sc, sc = 1/sqrt(taps+1).
    // Returns a device pointer to the folded weight.
    const float * fold_conv_vec(const std::string & n, int C, int taps) {
        auto v = host_f32(n);
        const float sc = 1.f / std::sqrt(float(taps + 1));
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
        if (wf) {
            GgufBytes b = bytes(n);
            std::vector<char> raw(b.data, b.data + b.size);
            r.m = wf->load(raw, r.rows, r.cols, [this](const void * h, size_t s) { return put(h, s); }, load_stream());
            if (!r.m.fmt || r.m.fmt->basis() != Basis::Plain) throw std::runtime_error("inkling2: " + n + " format");
        } else {
            auto v = host_f32(n);
            r.f32 = static_cast<const float *>(put(v.data(), v.size() * 4));
        }
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
        // Head, embedding and the global loop queries live on stage 1 (GPU 0, where the head runs).
        set_load_dev(1);
        embd_ = load_mat("token_embd.weight");
        embd_norm_ = vec("token_embd_norm.weight");
        out_ = load_mat("output.weight");
        out_norm_ = vec("output_norm.weight");
        out_res_q_ = vec("output_res_query.weight");
        loop_res_q_ = vec("loop_res_query.weight");
        loop_kv_res_q_ = vec("loop_kv_res_query.weight");
        e_ff_ = int(num("expert_feed_forward_length", 1152));
        L_.resize(n_layers_);
        for (int l = 0; l < n_layers_; ++l) {
            set_load_dev(dev_of(l));  // this layer's weights land on its owner GPU
            auto & Y = L_[l];
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
            Y.k_conv = fold_conv_vec(p + "attn_k_conv.weight", n_kv_ * hd_, taps_);
            Y.v_conv = fold_conv_vec(p + "attn_v_conv.weight", n_kv_ * hd_, taps_);
            Y.post_conv = fold_conv_vec(p + "attn_post_conv.weight", H_, taps_);
            Y.mlp_conv = fold_conv_vec(p + "ffn_post_conv.weight", H_, taps_);
            Y.q_clip = vec(p + "attn_q_clip.weight");
            Y.r_clip = vec(p + "attn_rel_clip.weight");
            Y.rel_bias = vec(p + "attn_rel_bias.weight");
            if (meta_.tensors.count(p + "attn_res_query.weight")) {
                const auto & t = tensor(p + "attn_res_query.weight");
                Y.attn_res_q = vec(p + "attn_res_query.weight");
                Y.attn_res_nq = t.ne.size() > 1 ? int(t.ne[1]) : 1;
            }
            Y.ffn_res_q = vec(p + "ffn_res_query.weight");
            if (!sliding_[l]) { Y.k_conv_proj = load_mat(p + "attn_k_conv_proj.weight"); Y.v_conv_proj = load_mat(p + "attn_v_conv_proj.weight"); }
            if (l >= core0_ && l < core1_) {
                Y.loop_attn_down = vec(p + "loop_attn_down.weight"); Y.loop_attn_forget = vec(p + "loop_attn_forget.weight"); Y.loop_attn_gain = vec(p + "loop_attn_gain.weight");
                Y.loop_k_down = vec(p + "loop_k_down.weight"); Y.loop_k_forget = vec(p + "loop_k_forget.weight"); Y.loop_k_gain = vec(p + "loop_k_gain.weight");
                Y.loop_v_down = vec(p + "loop_v_down.weight"); Y.loop_v_forget = vec(p + "loop_v_forget.weight"); Y.loop_v_gain = vec(p + "loop_v_gain.weight");
                Y.loop_mlp_down = vec(p + "loop_mlp_down.weight"); Y.loop_mlp_forget = vec(p + "loop_mlp_forget.weight"); Y.loop_mlp_gain = vec(p + "loop_mlp_gain.weight");
            }
            Y.router = load_mat(p + "ffn_gate_inp.weight");
            if (meta_.tensors.count(p + "exp_probs_b.bias")) Y.probs_bias = vec(p + "exp_probs_b.bias");
            Y.e_gu = load_experts(p + "ffn_gate_up_exps.weight");
            Y.e_down = load_experts(p + "ffn_down_exps.weight");
            Y.experts_nvfp4 = (tensor(p + "ffn_gate_up_exps.weight").type == 40);
            if (meta_.tensors.count(p + "ffn_gate_up_exps.scale")) Y.e_gu_scale = vec(p + "ffn_gate_up_exps.scale");
            if (meta_.tensors.count(p + "ffn_down_exps.scale")) Y.e_down_scale = vec(p + "ffn_down_exps.scale");
        }
        for (int d = 0; d < 2; ++d) {
            if (!cublas_[d]) {
                if (cublasCreate(&cublas_[d]) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("inkling2: cuBLAS init failed");
                cublasSetStream(cublas_[d], stream_[d]);
            }
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
            if (split_at_ && d != 0 && d != 1) continue;
            if (!split_at_ && d != 0) continue;
            Stage & S = st_[d];
            set_dev_for_alloc(d);
            S.h_ = Buf(size_t(H_) * 2);
            S.blocks_ = Buf(size_t(64) * H_ * 2);
            S.prev_h_ = Buf(size_t(H_) * 2);
            S.s1_ = Buf(w * 2); S.s2_ = Buf(w * 2); S.s3_ = Buf(w * 2); S.s4_ = Buf(w * 2);
            S.a_ = Buf(w * 4); S.b_ = Buf(w * 4); S.c_ = Buf(w * 4); S.d_ = Buf(w * 4);
            S.e_ = Buf(size_t(n_head_) * 4); S.f_ = Buf(size_t(n_head_) * 4); S.g_ = Buf(w * 2);
            S.moe_ids_ = Buf(size_t(k_exp_) * 4);
            S.moe_gu_ = Buf(size_t(k_exp_) * 2 * e_ff_ * 2);  // BF16
            S.moe_act_ = Buf(size_t(k_exp_) * e_ff_ * 2);     // half
            S.moe_dn_ = Buf(size_t(k_exp_) * H_ * 2);         // BF16
            S.moe_wts_ = Buf(size_t(k_exp_) * 4);
            S.moe_scales_ = Buf(size_t(k_exp_) * 4);
            if (d == 1) S.logits_ = Buf(size_t(vocab_) * 4);  // the head runs on stage 1
        }
        // Per-layer KV + conv history, on the layer's owner device.
        const size_t chist = size_t(taps_ - 1) * H_ * 2;  // [taps-1][C] half
        for (int l = 0; l < n_layers_; ++l) {
            set_dev_for_alloc(dev_of(l));
            auto & Y = L_[l];
            Y.kc = Buf(size_t(context_) * n_kv_ * hd_ * 2);
            Y.vc = Buf(size_t(context_) * n_kv_ * hd_ * 2);
            Y.k_hist = Buf(chist);
            Y.v_hist = Buf(chist);
            Y.post_hist = Buf(chist);
            Y.mlp_hist = Buf(chist);
        }
    }
    // Set the CUDA device for the next Buf allocation (which uses the current device's cudaMalloc).
    void set_dev_for_alloc(int d) {
        ensure_dev(d);
        ck(cudaSetDevice(cuda_of_[d]));
    }

    // y[n][rows] = W x[n][cols] via cuBLAS (W BF16 or F32)
    void gemm_h(const M & w, const __half * x, float * y, int n) {
        const float one = 1.f, zero = 0.f;
        if (w.f32) {
            if (cublasGemmEx(cublas(), CUBLAS_OP_T, CUBLAS_OP_N, w.rows, n, w.cols, &one, w.f32, CUDA_R_32F, w.cols, x, CUDA_R_16F, w.cols,
                             &zero, y, CUDA_R_32F, w.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("inkling2: cuBLAS GEMM failed");
        } else {
            if (cublasGemmEx(cublas(), CUBLAS_OP_T, CUBLAS_OP_N, w.rows, n, w.cols, &one, w.m.p, CUDA_R_16F, w.cols, x, CUDA_R_16F, w.cols,
                             &zero, y, CUDA_R_32F, w.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("inkling2: cuBLAS GEMM failed");
        }
    }
    // y[n][rows] = W x[n][cols] -> BF16 (for a format matvec input)
    void mv(const M & w, const __half * x, bf16 * y, int n) {
        if (w.f32) {
            float * tmp = st().s1_.as<float>();  // reuse scratch as F32
            gemm_h(w, x, tmp, n);
            for (int i = 0; i < w.rows * n; ++i) y[i] = __float2bfloat16(tmp[i]);
        } else {
            w.m.fmt->matvec(w.m, x, y, n, stream());
        }
    }

    // ---- forward (single token at position pos) ----
    void run_token(uint32_t tok, uint32_t pos) {
        // The embedding lives on stage 1 (GPU 0); the first layers run on stage 0 (GPU 1) when split.
        // Compute the embedding on stage 1, then hop the vector to stage 0 if the first layer is there.
        use_dev(1);
        {
            const auto & t = tensor("token_embd.weight");
            const int64_t row_bytes = int64_t(H_) * (t.type == 30 ? 2 : 4);
            GgufBytes b = bytes("token_embd.weight");
            const uint8_t * row = reinterpret_cast<const uint8_t *>(b.data) + int64_t(tok) * row_bytes;
            std::vector<__half> e(H_);
            for (int i = 0; i < H_; ++i) {
                float v;
                if (t.type == 30) { __nv_bfloat16 bh; std::memcpy(&bh, row + i * 2, 2); v = __bfloat162float(bh); }
                else { std::memcpy(&v, row + i * 4, 4); }
                e[i] = __float2half(v);
            }
            // e is host; upload to stage 1's h_, then rms-norm on stage 1.
            Stage & S1 = st_[1];
            ck(cudaMemcpyAsync(S1.h_.p, e.data(), H_ * 2, cudaMemcpyHostToDevice, stream()));
            tqllm_cuda::ik2_rms_norm(S1.h_.as<__half>(), embd_norm_, eps_, H_, S1.h_.as<__half>(), stream());
        }
        n_blocks_ = 0;
        // The first layer's device: stage 0 if split and layer 0 < split_at_, else stage 1.
        const int first_dev = dev_of(0);
        if (first_dev != cur_dev_) {
            hop_to(first_dev);  // move the embedding vector across
            use_dev(first_dev);
        }
        // pre-core layers
        for (int l = 0; l < core0_; ++l) {
            if (split_at_ && dev_of(l) != cur_dev_) use_dev(dev_of(l));
            visit(l, l, 0, pos, nullptr, 0);
        }
        if (n_pass_ > 1) {
            // The core loop runs entirely on one stage (its layers' owner), so the 3 passes never
            // cross PCIe. Capture the pre-core state on that stage.
            const int core_dev = dev_of(core0_);
            if (core_dev != cur_dev_) use_dev(core_dev);
            Stage & S = st();
            Buf pre_h(H_ * 2), pre_blk(size_t(64) * H_ * 2);
            int pre_nb = n_blocks_;
            ck(cudaMemcpy(pre_h.p, S.h_.p, H_ * 2, cudaMemcpyDeviceToDevice));
            ck(cudaMemcpy(pre_blk.p, S.blocks_.p, size_t(pre_nb) * H_ * 2, cudaMemcpyDeviceToDevice));
            for (int p = 0; p < n_pass_; ++p) {
                if (p > 0) {
                    ck(cudaMemcpy(S.h_.p, pre_h.p, H_ * 2, cudaMemcpyDeviceToDevice));
                    ck(cudaMemcpy(S.blocks_.p, pre_blk.p, size_t(pre_nb) * H_ * 2, cudaMemcpyDeviceToDevice));
                    n_blocks_ = pre_nb;
                }
                for (int l = core0_; l < core1_; ++l) {
                    int v = core0_ + p * (core1_ - core0_) + (l - core0_);
                    visit(l, v, p, pos, nullptr, 0);
                }
                if (p < n_pass_ - 1) {
                    ck(cudaMemcpy(S.prev_h_.p, S.h_.p, H_ * 2, cudaMemcpyDeviceToDevice));
                    prev_nb_ = n_blocks_;
                }
            }
        }
        // post-core layers
        for (int l = core1_; l < n_layers_; ++l) {
            if (split_at_ && dev_of(l) != cur_dev_) use_dev(dev_of(l));
            visit(l, l, 0, pos, nullptr, 0);
        }
    }
    // Move the residual vector (and block states) from the current stage to stage d across PCIe.
    // Only the small activation crosses (hidden 2304 half + up to 64 blocks), never the weights.
    void hop_to(int d) {
        if (!split_at_ || d == cur_dev_) return;
        Stage & from = st_[cur_dev_], & to = st_[d];
        const int from_c = cuda_of_[cur_dev_], to_c = cuda_of_[d];
        ck(cudaMemcpyPeerAsync(to.h_.p, to_c, from.h_.p, from_c, H_ * 2, stream_[cur_dev_]));
        if (n_blocks_)
            ck(cudaMemcpyPeerAsync(to.blocks_.p, to_c, from.blocks_.p, from_c, size_t(n_blocks_) * H_ * 2, stream_[cur_dev_]));
    }

    void visit(int l, int v, int pass, uint32_t pos, const __half * prev_last, int prev_nblocks) {
        Stage & S = st();
        if (prev_last == nullptr && pass > 0) { prev_last = S.prev_h_.as<__half>(); prev_nblocks = prev_nb_; }
        auto & Y = L_[l];
        const int C = H_;
        __half * h = S.h_.as<__half>();

        // h_in = attn_res(blocks, h) (layer 0: (h,h,h))
        Buf insbuf(3 * C * 2);
        __half * ins[3];
        if (l == 0) {
            for (int j = 0; j < 3; ++j) {
                tqllm_cuda::ik2_hcopy(h, static_cast<__half *>(insbuf.p) + j * C, C, stream());
                ins[j] = static_cast<__half *>(insbuf.p) + j * C;
            }
        } else {
            tqllm_cuda::ik2_attn_res(S.blocks_.as<__half>(), n_blocks_, h, Y.attn_res_q, Y.attn_res_nq, C, static_cast<__half *>(insbuf.p), stream());
            auto * o = static_cast<__half *>(insbuf.p);
            if (Y.attn_res_nq == 1) { for (int j = 0; j < 3; ++j) ins[j] = o; }
            else { ins[0] = o; ins[1] = o + C; ins[2] = o + 2 * C; }
        }
        if (pass > 0) {
            Buf msgs(4 * C * 2);
            loop_messages(l, prev_last, prev_nblocks, static_cast<__half *>(msgs.p));
            tqllm_cuda::ik2_control(ins[0], static_cast<__half *>(msgs.p), Y.loop_attn_down, Y.loop_attn_forget, Y.loop_attn_gain, C, ins[0], stream());
            tqllm_cuda::ik2_control(ins[1], static_cast<__half *>(msgs.p) + C, Y.loop_k_down, Y.loop_k_forget, Y.loop_k_gain, C, ins[1], stream());
            tqllm_cuda::ik2_control(ins[2], static_cast<__half *>(msgs.p) + 2 * C, Y.loop_v_down, Y.loop_v_forget, Y.loop_v_gain, C, ins[2], stream());
        }
        // norm the inputs
        Buf xsbuf(3 * C * 2);
        __half * xs[3];
        for (int j = 0; j < 3; ++j) {
            xs[j] = static_cast<__half *>(xsbuf.p) + j * C;
            if (sliding_[l]) tqllm_cuda::ik2_rms_norm(ins[j], Y.attn_norm, eps_, C, xs[j], stream());
            else tqllm_cuda::ik2_rms_norm(ins[j], (j == 0 ? Y.q_inp_norm : j == 1 ? Y.k_inp_norm : Y.v_inp_norm), eps_, C, xs[j], stream());
        }
        // attention
        __half * attn_out = S.s1_.as<__half>();
        __half * pre_kh = S.s2_.as<__half>();  // [kd]
        __half * pre_vh = S.s3_.as<__half>();  // [kd]
        attention(Y, pos, xs[0], xs[1], xs[2], attn_out, pre_kh, pre_vh);
        // a = shortconv_attn(attn_out)
        __half * a = S.g_.as<__half>();  // g_ is free after attention
        const __half * phist = Y.conv_pos > 0 ? Y.post_hist.as<__half>() : nullptr;
        tqllm_cuda::ik2_shortconv(Y.post_conv, taps_, attn_out, a, C, 1, phist, stream());
        // block boundary (device-resident): at a boundary, push h onto the block bank and r = a;
        // otherwise r = h + a.
        __half * r;
        Buf rbuf(C * 2);
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
        // m_in = attn_res(blocks, r) with ffn_res_q
        __half * m_in = S.s3_.as<__half>();
        tqllm_cuda::ik2_attn_res(S.blocks_.as<__half>(), n_blocks_, r, Y.ffn_res_q, 1, C, m_in, stream());
        if (pass > 0) {
            Buf msgs(4 * C * 2);
            loop_messages(l, prev_last, prev_nblocks, static_cast<__half *>(msgs.p));
            tqllm_cuda::ik2_control(m_in, static_cast<__half *>(msgs.p) + 3 * C, Y.loop_mlp_down, Y.loop_mlp_forget, Y.loop_mlp_gain, C, m_in, stream());
        }
        // m = moe(rms(m_in, ffn_norm))
        __half * mnorm = S.s4_.as<__half>();
        tqllm_cuda::ik2_rms_norm(m_in, Y.ffn_norm, eps_, C, mnorm, stream());
        __half * m = S.g_.as<__half>();  // g_ is free after post_conv
        moe(Y, mnorm, m);
        // m = shortconv_mlp(m)
        const __half * mhist = Y.conv_pos > 0 ? Y.mlp_hist.as<__half>() : nullptr;
        tqllm_cuda::ik2_shortconv(Y.mlp_conv, taps_, m, m, C, 1, mhist, stream());
        // h = r + m (device-resident)
        tqllm_cuda::ik2_hadd(r, m, h, C, stream());
        // update conv histories: shift and append current token's pre-conv inputs
        const int kd = n_kv_ * hd_;
        tqllm_cuda::ik2_conv_hist_update(Y.k_hist.as<__half>(), pre_kh, taps_, kd, stream());
        tqllm_cuda::ik2_conv_hist_update(Y.v_hist.as<__half>(), pre_vh, taps_, kd, stream());
        tqllm_cuda::ik2_conv_hist_update(Y.post_hist.as<__half>(), attn_out, taps_, C, stream());
        tqllm_cuda::ik2_conv_hist_update(Y.mlp_hist.as<__half>(), m, taps_, C, stream());
        ++Y.conv_pos;
    }

    // The four loop messages (m_q, m_k, m_v, m_mlp) of core layer l: attn_res(bank, last) with the loop_res_query /
    // loop_kv_res_query rows for this core index. bank = the previous pass's blocks after its first (prev_nblocks ..
    // n_blocks_); last = the previous pass's final stream.
    void loop_messages(int l, const __half * prev_last, int prev_nblocks, __half * out) {
        const int i = l - core0_;
        const int C = H_;
        const __half * bank = st().blocks_.as<__half>() + size_t(prev_nblocks) * C;
        const int nbank = n_blocks_ - prev_nblocks;
        // m_q, m_mlp from loop_res_query rows [2i, 2i+1]; m_k, m_v from loop_kv_res_query rows [2i, 2i+1]
        tqllm_cuda::ik2_attn_res(bank, nbank, prev_last, loop_res_q_ + 2 * i * C, 2, C, out, stream());          // m_q, m_mlp
        tqllm_cuda::ik2_attn_res(bank, nbank, prev_last, loop_kv_res_q_ + 2 * i * C, 2, C, out + 2 * C, stream());  // m_k, m_v
        // reorder: out = [m_q, m_k, m_v, m_mlp] (currently [m_q, m_mlp, m_k, m_v]); device copies.
        Buf tmp(C * 2);
        tqllm_cuda::ik2_hcopy(out + C, tmp.as<__half>(), C, stream());      // tmp = m_mlp
        tqllm_cuda::ik2_hcopy(out + 2 * C, out + C, C, stream());            // out[1] = m_k
        tqllm_cuda::ik2_hcopy(tmp.as<__half>(), out + 2 * C, C, stream());   // out[2] = m_v
        tqllm_cuda::ik2_hcopy(tmp.as<__half>(), out + 3 * C, C, stream());   // out[3] = m_mlp
    }

    // The attention of one layer at position pos: project q,k,v,r,gate,sink; conv k,v; clip; relative bias; softmax;
    // gate; output projection. Writes out [C]. Uses dedicated scratch buffers (a..g) to avoid aliasing.
    // pre_kh/pre_vh: scratch for the pre-conv k/v (for conv history).
    void attention(Layer & Y, uint32_t pos, const __half * xq, const __half * xk, const __half * xv, __half * out,
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
        // to half (device-resident conversion)
        __half * qh = S.a_.as<__half>();  // [qd]
        tqllm_cuda::common::f32_to_half(qf, qh, qd, stream());
        __half * kh = S.b_.as<__half>();  // [kd]
        tqllm_cuda::common::f32_to_half(kf, kh, kd, stream());
        __half * vh = S.c_.as<__half>();  // [kd]
        tqllm_cuda::common::f32_to_half(vf, vh, kd, stream());
        __half * rh = S.d_.as<__half>();  // [H*d_rel]
        tqllm_cuda::common::f32_to_half(rf, rh, H * d_rel_, stream());
        // save pre-conv k/v for conv history (device copy)
        tqllm_cuda::ik2_hcopy(kh, pre_kh, kd, stream());
        tqllm_cuda::ik2_hcopy(vh, pre_vh, kd, stream());
        // k/v short conv (with history)
        const __half * khist = Y.conv_pos > 0 ? Y.k_hist.as<__half>() : nullptr;
        const __half * vhist = Y.conv_pos > 0 ? Y.v_hist.as<__half>() : nullptr;
        if (Y.sliding) {
            tqllm_cuda::ik2_shortconv(Y.k_conv, taps_, kh, kh, kd, 1, khist, stream());
            tqllm_cuda::ik2_shortconv(Y.v_conv, taps_, vh, vh, kd, 1, vhist, stream());
        } else {
            // full layer: dynamic conv (static + dynamic taps from input)
            // dyn = x @ conv_proj^T -> [kd] (per-channel, per-lag)
            float * dynk = S.e_.as<float>();  // [kd] = [KV*D]
            gemm_h(Y.k_conv_proj, xk, dynk, 1);
            float * dynv = S.f_.as<float>();  // [kd]
            gemm_h(Y.v_conv_proj, xv, dynv, 1);
            tqllm_cuda::ik2_shortconv_dyn(Y.k_conv, taps_, kh, kh, kd, dynk, khist, stream());
            tqllm_cuda::ik2_shortconv_dyn(Y.v_conv, taps_, vh, vh, kd, dynv, vhist, stream());
        }
        // clip q, k; norm v; clip r
        tqllm_cuda::ik2_l2clip(qh, Y.q_clip, qk_clip_, H, D, qh, stream());
        tqllm_cuda::ik2_l2clip(kh, Y.q_clip, qk_clip_, KV, D, kh, stream());  // k uses qk_clip (no per-head clip weight)
        tqllm_cuda::ik2_rms_weightless(vh, KV, D, vh, stream());
        tqllm_cuda::ik2_l2clip(rh, Y.r_clip, rel_clip_, H, d_rel_, rh, stream());
        // full layers: q, r *= tau = 1 + alpha * ln(max(1, (pos+1)/n_floor))
        if (!Y.sliding && log_alpha_ != 0.f) {
            const float tau = 1.f + log_alpha_ * std::log(std::max(1.f, float(pos + 1) / n_floor_));
            tqllm_cuda::ik2_hscale(qh, tau, qd, stream());
            tqllm_cuda::ik2_hscale(rh, tau, H * d_rel_, stream());
        }
        // write the current token's k/v into the cache at row pos (device-to-device)
        {
            __half * krow = Y.kc.as<__half>() + size_t(pos) * KV * D;
            __half * vrow = Y.vc.as<__half>() + size_t(pos) * KV * D;
            ck(cudaMemcpyAsync(krow, kh, kd * 2, cudaMemcpyDeviceToDevice, stream()));
            ck(cudaMemcpyAsync(vrow, vh, kd * 2, cudaMemcpyDeviceToDevice, stream()));
        }
        // gate/sink to half (device-resident conversion)
        __half * gateh = S.e_.as<__half>();
        __half * sinkh = S.f_.as<__half>();
        tqllm_cuda::common::f32_to_half(gate, gateh, H, stream());
        tqllm_cuda::common::f32_to_half(sink, sinkh, H, stream());
        // attention core
        tqllm_cuda::Ik2AttnCore ac{};
        ac.q = qh; ac.k = kh; ac.v = vh; ac.r = rh; ac.gate = gateh; ac.sink = sinkh;
        ac.rel_bias = Y.rel_bias;
        ac.H = H; ac.KV = KV; ac.D = D; ac.d_rel = d_rel_; ac.extent = Y.sliding ? extent_swa_ : extent_; ac.window = window_;
        ac.sliding = Y.sliding; ac.pos = pos;
        ac.kc = Y.kc.as<const __half>(); ac.vc = Y.vc.as<const __half>();
        __half * att = S.g_.as<__half>();  // [qd]
        tqllm_cuda::ik2_attention_core(ac, att, stream());
        // out = att @ wo^T [C] (F32 scratch -> half, device-resident)
        float * of = S.a_.as<float>();  // [C]
        gemm_h(Y.wo, att, of, 1);
        tqllm_cuda::common::f32_to_half(of, out, C, stream());
    }

    // The MoE: sigmoid router, top-k, routed experts. x [C], out [C].
    // NOTE: the router read and the activation / down weighted-sum still round-trip through the
    // host (E=128 and K=8 are small). This is correct on the layer's own device but serializes;
    // a device-resident router + SwiGLU-softcap + weighted-sum is a follow-up (see PROGRESS.md).
    void moe(Layer & Y, const __half * x, __half * out) {
        Stage & S = st();
        const int C = H_, E = n_exp_, K = k_exp_, FF = e_ff_;
        // router logits = x @ router^T [E]
        float * logits = S.s1_.as<float>();
        gemm_h(Y.router, x, logits, 1);
        // sigmoid + top-k + weights (host: E is small)
        std::vector<float> sc(E);
        ck(cudaMemcpy(sc.data(), logits, E * 4, cudaMemcpyDeviceToHost));
        for (int e = 0; e < E; ++e) sc[e] = 1.f / (1.f + std::exp(-sc[e]));
        std::vector<int> order(E);
        for (int e = 0; e < E; ++e) order[e] = e;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            float va = sc[a] + (Y.probs_bias ? Y.probs_bias[a] : 0.f);
            float vb = sc[b] + (Y.probs_bias ? Y.probs_bias[b] : 0.f);
            return va > vb;
        });
        std::vector<int> ids(K);
        std::vector<float> wts(K), sds(K);
        float wsum = 0.f;
        for (int i = 0; i < K; ++i) { ids[i] = order[i]; wsum += sc[ids[i]]; }
        for (int i = 0; i < K; ++i) {
            wts[i] = sc[ids[i]] / (wsum + 1e-20f) * route_scale_;
            sds[i] = Y.e_down_scale ? Y.e_down_scale[ids[i]] : 1.f;
        }
        // upload ids, weights, scales to device
        ck(cudaMemcpyAsync(S.moe_ids_.p, ids.data(), K * 4, cudaMemcpyHostToDevice, stream()));
        ck(cudaMemcpyAsync(S.moe_wts_.p, wts.data(), K * 4, cudaMemcpyHostToDevice, stream()));
        ck(cudaMemcpyAsync(S.moe_scales_.p, sds.data(), K * 4, cudaMemcpyHostToDevice, stream()));
        // gate+up: e_gu is [E * 2 * FF][C]; expert_matvec with rows=2*FF gives [K][2*FF] BF16
        Y.e_gu.fmt->expert_matvec(Y.e_gu, x, S.moe_gu_.as<bf16>(), 2 * FF, S.moe_ids_.as<int>(), 1, K, false, stream());
        // activation: h = tanh(g/gate_cap)*gate_cap*sigmoid(g)*tanh(u/up_cap)*up_cap
        {
            std::vector<bf16> gu(size_t(K) * 2 * FF);
            ck(cudaMemcpy(gu.data(), S.moe_gu_.p, gu.size() * 2, cudaMemcpyDeviceToHost));
            std::vector<__half> act(size_t(K) * FF);
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < FF; ++j) {
                    float gv = __bfloat162float(gu[size_t(i) * 2 * FF + j]);
                    float uv = __bfloat162float(gu[size_t(i) * 2 * FF + FF + j]);
                    if (Y.e_gu_scale) { gv *= Y.e_gu_scale[ids[i]]; uv *= Y.e_gu_scale[ids[i]]; }
                    float hv = std::tanh(gv / gate_cap_) * gate_cap_ * (1.f / (1.f + std::exp(-gv))) * std::tanh(uv / up_cap_) * up_cap_;
                    act[size_t(i) * FF + j] = __float2half(hv);
                }
            ck(cudaMemcpyAsync(S.moe_act_.p, act.data(), act.size() * 2, cudaMemcpyHostToDevice, stream()));
        }
        // down: e_down is [E * C][FF]; expert_matvec with rows=C gives [K][C] BF16
        Y.e_down.fmt->expert_matvec(Y.e_down, S.moe_act_.as<__half>(), S.moe_dn_.as<bf16>(), C, S.moe_ids_.as<int>(), 1, K, true, stream());
        // out = sum_i wts[i] * sds[i] * moe_dn_[i]  (host: K is small; out is a device pointer,
        // so accumulate on the host and upload)
        std::vector<bf16> dn(size_t(K) * C);
        ck(cudaMemcpy(dn.data(), S.moe_dn_.p, dn.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<__half> o(C);
        for (int j = 0; j < C; ++j) {
            float acc = 0.f;
            for (int i = 0; i < K; ++i) acc += wts[i] * sds[i] * __bfloat162float(dn[size_t(i) * C + j]);
            o[j] = __float2half(acc);
        }
        ck(cudaMemcpyAsync(out, o.data(), C * 2, cudaMemcpyHostToDevice, stream()));
    }

    int e_ff_ = 0;  // expert feed-forward width (set in load)

    void compute_logits() {
        // The head (output.weight, out_norm, out_res_query) lives on stage 1; make sure we're there.
        use_dev(1);
        Stage & S = st();
        __half * h = S.h_.as<__half>();
        __half * hres = S.s1_.as<__half>();
        tqllm_cuda::ik2_attn_res(S.blocks_.as<__half>(), n_blocks_, h, out_res_q_, 1, H_, hres, stream());
        __half * hnorm = S.s2_.as<__half>();
        tqllm_cuda::ik2_rms_norm(hres, out_norm_, eps_, H_, hnorm, stream());
        gemm_h(out_, hnorm, S.logits_.as<float>(), 1);
        tqllm_cuda::ik2_softcap(S.logits_.as<float>(), vocab_unpadded_, logit_cap_, stream());
    }

    uint32_t greedy_token() {
        std::vector<float> l(vocab_unpadded_);
        ck(cudaMemcpy(l.data(), st_[1].logits_.as<float>(), vocab_unpadded_ * 4, cudaMemcpyDeviceToHost));
        int best = 0;
        for (int i = 1; i < vocab_unpadded_; ++i) if (l[i] > l[best]) best = i;
        return uint32_t(best);
    }
};

std::unique_ptr<Engine> make_inkling2_engine_impl(const std::string & path, uint32_t context) {
    return std::make_unique<Inkling2Engine>(path, context);
}
}  // namespace

bool is_inkling2_gguf(const std::string & path) {
    try { return gguf_architecture(path) == "inkling2"; } catch (...) { return false; }
}

std::unique_ptr<Engine> make_inkling2_engine(const std::string & path, uint32_t context) {
    return make_inkling2_engine_impl(path, context);
}

}  // namespace tqllm
