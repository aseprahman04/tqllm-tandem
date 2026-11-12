// qwen35 engine: a layer split across two GPUs (multi-GPU fork addition, not part of upstream tqLLM).
//
// TQLLM_SPLIT=K puts layers [0, K) on CUDA device 0 and [K, layers) on device 1, which also holds the head, the
// sampling buffers and the DFlash2 drafter. The engine's members always hold the current device's stream, cuBLAS
// handle and scratch buffers; use_dev() swaps them. The design follows PolyStrata's strata-poly layer split (one stage
// per card, rows handed on through host memory, no peer access needed); the code is new.
//
// What the split runs so far: one slot, dense models, all KV on the GPUs (--kv-host 0), --prompt-cache, DFlash2 or no
// drafter. A verify block runs as two captured graphs, one per GPU, with the hand-off between them (split_hop); prefill
// chunks, commits and tree gathers run eagerly. Greedy output equals the one-device run token for token (checked on
// Qwen3.8-27B TQ_XS); TQ_L, which no 16 GB card holds alone, serves 524,288 tokens (YaRN x2) on two.
#include "qwen35_engine.hpp"

#include <cuda.h>

namespace tqllm {
namespace qwen35_detail {

void Qwen35Engine::split_setup() {
    const char * e = std::getenv("TQLLM_SPLIT");
    if (!e || !*e) return;
    const int k = std::atoi(e);
    int n = 0;
    ck(cudaGetDeviceCount(&n));
    auto need = [](bool ok, const std::string & what) {
        if (!ok) throw std::runtime_error("TQLLM_SPLIT: " + what);
    };
    need(n >= 2, "needs two CUDA devices; this process sees " + std::to_string(n));
    need(k > 0 && k < int(cfg.layers), "K must be a layer in 1.." + std::to_string(cfg.layers - 1));
    need(!cfg.moe, "dense models only so far (not qwen35moe)");
    need(!cfg.tied, "models with an untied LM head only (the head lives on the second GPU)");
    need(opt.kv_host <= 0, "all KV on the GPUs so far: pass --kv-host 0");
    need(opt.kv_compact <= 0, "no --kv-compact yet");
    need(!opt.native_mtp, "no --mtp yet (use -md with a DFlash2 drafter)");
    split_at = k;
    home_dev = 1;
    cuda_of[0] = 1;  // the first layers on CUDA device 1
    cuda_of[1] = 0;  // the rest, the head and the drafter on CUDA device 0, where the constructor already is
    cur_dev = home_dev;  // the stream and cuBLAS handle made so far are device 0's: the home stage's
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
            &blas_ws, &fp8_sw, &v_feat};
}

void Qwen35Engine::use_dev(int d) {
    if (!split_at || d == cur_dev) return;
    // everything queued on the old device runs before what the new one queues from here on
    if (!hop_ev) ck(cudaEventCreateWithFlags(&hop_ev, cudaEventDisableTiming));
    ck(cudaEventRecord(hop_ev, stream));
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
    ck(cudaStreamWaitEvent(stream, after, 0));
    vpar = par == 1 ? d_tpar.as<int>() : par == 2 ? d_chain.as<int>() : nullptr;
    vdep = dep == 1 ? d_tdep.as<int>() : dep == 2 ? d_cdep.as<int>() : nullptr;
}

// Device 0's scratch: the same buffers allocate() made on device 1, of the same sizes, and the constant tables copied.
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
    // the verify tables start as allocate() wrote them; d_pend starts at 0
    split_mirror_control();
    sync_ck();
    use_dev(home_dev);
}

// Called on device 1 right after use_dev(1): device 0's residual rows, their norm partials and the drafter features of
// its layers, for the first n columns, into device 1's buffers.
void Qwen35Engine::split_hop(int n) {
    const auto list = dev_bufs();
    auto dev0 = [&](Buf & b) -> const void * {
        return dstash[0].bufs[size_t(std::find(list.begin(), list.end(), &b) - list.begin())].p;
    };
    const size_t H = cfg.hidden;
    const int to = cuda_of[1], from = cuda_of[0];
    ck(cudaMemcpyPeerAsync(v_x.p, to, dev0(v_x), from, size_t(n) * H * 2, stream));
    ck(cudaMemcpyPeerAsync(v_px.p, to, dev0(v_px), from, size_t(n) * ((H + 255) / 256) * 4, stream));
    if (dfl.present && v_feat.p && dev0(v_feat))
        ck(cudaMemcpyPeerAsync(v_feat.p, to, dev0(v_feat), from, size_t(n) * dfl.target_ids.size() * H * 2, stream));
}

// Called on device 0 right after use_dev(0): device 1 holds the tables as the host last wrote them (tokens, positions,
// tree / chain parents and depths, the pending commit length).
void Qwen35Engine::split_mirror_control() {
    const auto list = dev_bufs();
    auto peer = [&](Buf & b, size_t n) {
        const size_t i = size_t(std::find(list.begin(), list.end(), &b) - list.begin());
        const Buf & src = dstash[1].bufs[i];
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
