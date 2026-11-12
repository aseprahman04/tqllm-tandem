// qwen35 engine: a layer split across two GPUs (multi-GPU fork addition, not part of upstream tqLLM).
//
// TQLLM_SPLIT=K puts layers [0, K) on CUDA device 0 and [K, layers) on device 1, which also holds the head, the
// sampling buffers and the DFlash2 drafter. The engine's members always hold the current device's stream, cuBLAS
// handle and scratch buffers; use_dev() swaps them. The design follows PolyStrata's strata-poly layer split (one stage
// per card, rows handed on through host memory, no peer access needed); the code is new.
//
// What the split runs so far: one slot, dense and MoE models (every expert on the GPUs), all KV on the GPUs (--kv-host 0), --prompt-cache, DFlash2, --mtp or no
// drafter. A verify block runs as two captured graphs, one per GPU, with the hand-off between them (split_hop); prefill
// chunks, commits and tree gathers run eagerly. Greedy output equals the one-device run token for token (checked on
// Qwen3.8-27B TQ_XS); TQ_L, which no 16 GB card holds alone, serves 524,288 tokens (YaRN x2) on two.
#include "qwen35_engine.hpp"

#include <cuda.h>

namespace tqllm {
namespace qwen35_detail {

void Qwen35Engine::split_setup(int slots) {
    const char * e = std::getenv("TQLLM_SPLIT");
    if (!e || !*e) return;
    const int k = std::atoi(e);
    int n = 0;
    ck(cudaGetDeviceCount(&n));
    auto need = [](bool ok, const std::string & what) {
        if (!ok) throw std::runtime_error("TQLLM_SPLIT: " + what);
    };
    need(n >= 2, "needs two CUDA devices; this process sees " + std::to_string(n));
    need(slots == 1, "one slot so far: pass --np 1");
    need(k > 0 && k < int(cfg.layers), "K must be a layer in 1.." + std::to_string(cfg.layers - 1));
    need(!cfg.moe || (opt.moe_host_mb <= 0 && opt.moe_cache_mb <= 0 && opt.n_cpu_moe <= 0),
         "MoE models with every expert on the GPUs only so far (no --moe-host-mb / --moe-cache-mb / -ncmoe)");
    need(!cfg.tied, "models with an untied LM head only (the head lives on the second GPU)");
    need(opt.kv_host <= 0, "all KV on the GPUs so far: pass --kv-host 0");
    need(opt.kv_compact <= 0, "no --kv-compact yet");
    split_at = k;
    cuda_of[0] = 1;  // the first layers on CUDA device 1
    cuda_of[1] = 0;  // the rest, the head and the drafter on CUDA device 0, where the constructor already is
    cur_dev = 1;  // the stream and cuBLAS handle made so far are device 0's: the home stage's
    std::fprintf(stderr, "tqllm: layer split: layers 0-%d on GPU 1, %d-%u, the head and the drafter on GPU 0\n", k - 1,
                 k, cfg.layers - 1);
}

// Every scratch buffer the layers, the head or the drafter's inputs use: one copy per device.
std::vector<Buf *> Qwen35Engine::dev_bufs() {
    return {&x,      &norm,   &resid,  &mix,    &gate,      &up,     &act,      &logits_buf, &token,   &part,
            &xq,     &xs,     &xf,     &part_x, &part_m,    &amax,   &t_in,     &t_qg,       &t_q,     &t_k,
            &t_v,    &t_o,    &t_qkv,  &t_qkvc, &t_z,       &t_a,    &t_b,      &t_core,     &t_gn,    &v_x,
            &v_m,    &v_r,    &v_np,   &v_in,   &v_o,       &v_core, &v_act,    &v_xh,       &v_xf,    &v_px,
            &v_pm,   &v_logits, &v_tok, &v_amax, &v_q,      &v_qkvc, &v_part,   &s_id,       &s_lg,    &s_scr,
            &v_xb,   &v_mixa, &v_mixb, &v_xu,   &v_hid,     &v_cat,  &mtp_h,    &v_tokens,   &v_pos,   &v_rpos,
            &d_tpar, &d_tdep, &d_tcols, &d_chain, &d_cdep,  &d_gdesc, &d_pend,  &d_rep_qkvc, &gemm_w,  &gemm_c,
            &blas_ws, &fp8_sw, &v_feat,
            // MoE routing and expert scratch (every expert of a layer lives on that layer's GPU)
            &moe_logits, &moe_ids, &moe_counts, &moe_groups, &moe_xids, &moe_weights, &moe_shared_weights, &moe_gate,
            &moe_up, &moe_act, &moe_out, &moe_shared};
}

void Qwen35Engine::use_dev(int d, bool wait) {
    if (!split_at || d == cur_dev) return;
    // everything queued on the old device runs before what the new one queues from here on
    if (!hop_ev) ck(cudaEventCreateWithFlags(&hop_ev, cudaEventDisableTiming));
    if (wait) ck(cudaEventRecord(hop_ev, stream));
    const cudaEvent_t after = hop_ev;
    // the verify path keeps pointers into the tree / chain tables: carried across the swap by name
    const int par = !vpar ? 0 : vpar == d_tpar.as<int>() ? 1 : vpar == d_chain.as<int>() ? 2 : 3;
    const int dep = !vdep ? 0 : vdep == d_tdep.as<int>() ? 1 : vdep == d_cdep.as<int>() ? 2 : 3;
    if (par == 3 || dep == 3) throw std::logic_error("use_dev: a verify table pointer of no known buffer");
    DevStash & from = dstash[cur_dev];
    DevStash & to = dstash[d];
    from.stream = stream, from.side = side, from.ev_fork = ev_fork, from.ev_join = ev_join, from.hop = hop_ev;
    from.blas = blas;
    const auto list = dev_bufs();
    from.bufs.resize(list.size());
    to.bufs.resize(list.size());
    for (size_t i = 0; i < list.size(); ++i) {
        from.bufs[i] = std::move(*list[i]);
        *list[i] = std::move(to.bufs[i]);
    }
    from.i8 = std::move(i8);
    i8 = std::move(to.i8);
    stream = to.stream, side = to.side, ev_fork = to.ev_fork, ev_join = to.ev_join, hop_ev = to.hop, blas = to.blas;
    to.stream = to.side = nullptr, to.ev_fork = to.ev_join = to.hop = nullptr, to.blas = nullptr;  // the members own them now
    ck(cudaSetDevice(cuda_of[d]));
    cur_dev = d;
    if (!stream) {  // first visit: the device's own stream and cuBLAS handle
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cb(cublasCreate(&blas));
        cb(cublasSetStream(blas, stream));
        cb(cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH));
    }
    if (wait) ck(cudaStreamWaitEvent(stream, after, 0));
    vpar = par == 1 ? d_tpar.as<int>() : par == 2 ? d_chain.as<int>() : nullptr;
    vdep = dep == 1 ? d_tdep.as<int>() : dep == 2 ? d_cdep.as<int>() : nullptr;
}

// Stage 0's scratch: the same buffers allocate() and the drafter made on stage 1, of the same sizes, and the constant
// tables copied.
void Qwen35Engine::split_alloc_dev0() {
    if (!split_at) return;
    std::vector<size_t> bytes;
    for (Buf * b : dev_bufs()) {
        size_t sz = 0;
        if (b->p) {
            CUdeviceptr base = 0;
            if (cuMemGetAddressRange(&base, &sz, CUdeviceptr(b->p)) != CUDA_SUCCESS)
                throw std::runtime_error("split: cannot read a scratch buffer's size");
        }
        bytes.push_back(sz);
    }
    use_dev(0);
    const auto list = dev_bufs();
    for (size_t i = 0; i < list.size(); ++i)
        if (bytes[i] && !list[i]->p) *list[i] = Buf(bytes[i]);
    if (blas_ws.p) cb(cublasSetWorkspace(blas, blas_ws.p, size_t(32) << 20));
    if (dstash[1].i8) {  // --prefill-int8 on: stage 0's own int8 GEMM (cuBLASLt handle, weight and token buffers)
        i8 = std::make_unique<tqllm_cuda::Int8Gemm>();
        i8->init(i8_init[0], i8_init[1], int(i8_init[2]), int(i8_init[3]));
    }
    // the verify tables start as allocate() wrote them; d_pend starts at 0
    split_mirror_control();
    sync_ck();
    use_dev(1);
}

Buf & Qwen35Engine::other_stage(Buf & b) {
    const auto list = dev_bufs();
    return dstash[1 - cur_dev].bufs[size_t(std::find(list.begin(), list.end(), &b) - list.begin())];
}

// Called on stage 1 right after use_dev(1): stage 0's residual rows, their norm partials and the drafter features of
// its layers, for the first n columns, into stage 1's buffers.
void Qwen35Engine::split_hop(int n) {
    const size_t H = cfg.hidden;
    const int to = cuda_of[1], from = cuda_of[0];
    ck(cudaMemcpyPeerAsync(v_x.p, to, other_stage(v_x).p, from, size_t(n) * H * 2, stream));
    ck(cudaMemcpyPeerAsync(v_px.p, to, other_stage(v_px).p, from, size_t(n) * ((H + 255) / 256) * 4, stream));
    if (dfl.present && v_feat.p && other_stage(v_feat).p)
        ck(cudaMemcpyPeerAsync(v_feat.p, to, other_stage(v_feat).p, from, size_t(n) * dfl.target_ids.size() * H * 2, stream));
}

// Prefill chunk (n > NVER, no images): stage 0 needs only the chunk's tokens and positions, which it copies from the
// caller's staged host memory itself, and its own state and KV, which its stream already orders. So it does not wait for
// the home stage, only until the previous chunk's rows, norm partials and drafter features have been copied off it.
// The home stage then waits for this chunk's stage 0 (use_dev(1)) as before. Every kernel and its inputs are those of
// verify_body(n, bucket): only the cross-device ordering is relaxed, so results are unchanged.
void Qwen35Engine::split_prefill(const uint32_t * tokens, const int * pos_host, int n, int bucket) {
    if (!pf_hop_done) ck(cudaEventCreateWithFlags(&pf_hop_done, cudaEventDisableTiming));  // on the home device
    use_dev(0, false);
    ck(cudaStreamWaitEvent(stream, pf_hop_done, 0));
    ck(cudaMemcpyAsync(v_tokens.p, tokens, size_t(n) * 4, cudaMemcpyHostToDevice, stream));
    ck(cudaMemcpyAsync(v_pos.p, pos_host, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
    ck(cudaMemcpyAsync(v_rpos.p, pos_host, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
    verify_body(n, bucket, 0);
    use_dev(1);
    split_hop(n);
    ck(cudaEventRecord(pf_hop_done, stream));
    verify_body(n, bucket, 1);
}

// Called on stage 0 right after use_dev(0): stage 1 holds the tables as the host last wrote them (tokens, positions,
// tree / chain parents and depths, the pending commit length).
void Qwen35Engine::split_mirror_control() {
    auto peer = [&](Buf & b, size_t n) {
        const Buf & src = other_stage(b);
        if (b.p && src.p && n) ck(cudaMemcpyPeerAsync(b.p, cuda_of[0], src.p, cuda_of[1], n, stream));
    };
    peer(v_tokens, size_t(NBUF) * 4);
    peer(v_pos, 2 * size_t(nslots) * sizeof(int));
    peer(v_rpos, 2 * size_t(nslots) * sizeof(int));
    for (Buf * t : {&d_tpar, &d_tdep, &d_tcols, &d_chain, &d_cdep}) peer(*t, size_t(NB) * 4);
    peer(d_pend, sizeof(int));
}

}  // namespace qwen35_detail
}  // namespace tqllm
