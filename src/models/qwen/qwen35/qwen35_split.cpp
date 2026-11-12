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

void Qwen35Engine::tp_setup(int slots) {
    const char * e = std::getenv("TQLLM_TP");
    if (!e || std::atoi(e) != 1) return;
    int n = 0;
    ck(cudaGetDeviceCount(&n));
    auto need = [](bool ok, const std::string & what) {
        if (!ok) throw std::runtime_error("TQLLM_TP: " + what);
    };
    need(!split_at, "TQLLM_SPLIT and TQLLM_TP are two ways to use two GPUs; set one");
    need(n >= 2, "needs two CUDA devices; this process sees " + std::to_string(n));
    need(slots == 1, "one slot so far: pass --np 1");
    need(!cfg.moe, "dense models only so far");
    need(!cfg.tied, "models with an untied LM head only");
    need(cfg.heads % 2 == 0 && cfg.kvheads % 2 == 0 && cfg.lin_kheads % 2 == 0 && cfg.lin_vheads % 2 == 0 && cfg.ffn % 2 == 0,
         "head and FFN counts must split evenly");
    need(opt.draft_model.empty() && !opt.native_mtp, "no drafter yet (milestone 1: plain decoding)");
    need(opt.kv_host <= 0 && opt.kv_compact <= 0, "all KV on the GPUs, no --kv-compact");
    need(!opt.prompt_cache, "no --prompt-cache yet");
    tp = 2;
    cuda_of[0] = 1;  // rank 0 on CUDA device 1
    cuda_of[1] = 0;  // rank 1 (home: head, sampling) on CUDA device 0, where the constructor already is
    cur_dev = 1;
    std::fprintf(stderr, "tqllm: tensor parallel over two GPUs (each: %u of %u attention heads, %u of %u FFN rows)\n",
                 cfg.heads / 2, cfg.heads, cfg.ffn / 2, cfg.ffn);
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
    if ((!split_at && !tp) || d == cur_dev) return;
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
    if (tp) {  // tensor parallel: the weights, KV and recurrent state are each GPU's own shard
        std::swap(layer, from.layer), std::swap(layer, to.layer);
        std::swap(kv_mem, from.kv_mem), std::swap(kv_mem, to.kv_mem);
        std::swap(kv_pool, from.kv_pool), std::swap(kv_pool, to.kv_pool);
        std::swap(kv_seq, from.kv_seq), std::swap(kv_seq, to.kv_seq);
        std::swap(key_cache, from.key_cache), std::swap(key_cache, to.key_cache);
        std::swap(value_cache, from.value_cache), std::swap(value_cache, to.value_cache);
        std::swap(conv_state, from.conv_state), std::swap(conv_state, to.conv_state);
        std::swap(rec_state, from.rec_state), std::swap(rec_state, to.rec_state);
        for (int s = 0; s < 2; ++s) std::swap(v_stash2[s], from.stash[s]), std::swap(v_stash2[s], to.stash[s]);
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
    if (wait) ck(cudaStreamWaitEvent(stream, after, 0));
    vpar = par == 1 ? d_tpar.as<int>() : par == 2 ? d_chain.as<int>() : nullptr;
    vdep = dep == 1 ? d_tdep.as<int>() : dep == 2 ? d_cdep.as<int>() : nullptr;
}

// Stage 0's scratch: the same buffers allocate() and the drafter made on stage 1, of the same sizes, and the constant
// tables copied.
void Qwen35Engine::split_alloc_dev0() {
    if (!split_at && !tp) return;
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
    // the RoPE scaling table (YaRN, linear) is __constant__ memory, per device: the constructor loaded it on device 0
    // only, so the first GPU's attention layers ran unscaled RoPE under --rope-scaling until this line
    set_rope(false);
    const auto list = dev_bufs();
    for (size_t i = 0; i < list.size(); ++i)
        if (bytes[i] && !list[i]->p) *list[i] = Buf(bytes[i]);
    if (blas_ws.p) cb(cublasSetWorkspace(blas, blas_ws.p, size_t(32) << 20));
    if (dstash[1].i8) {  // --prefill-int8 on: stage 0's own int8 GEMM (cuBLASLt handle, weight and token buffers)
        i8 = std::make_unique<tqllm_cuda::Int8Gemm>();
        i8->init(i8_init[0], i8_init[1], int(i8_init[2]), int(i8_init[3]));
    }
    if (tp) {  // rank 0's own verify stash (home's sizes), KV pool and views, recurrent state, all-reduce buffers
        for (int s = 0; s < 2; ++s)
            for (const Buf & b : dstash[1].stash[s]) {
                CUdeviceptr base = 0;
                size_t sz = 0;
                if (b.p && cuMemGetAddressRange(&base, &sz, CUdeviceptr(b.p)) != CUDA_SUCCESS)
                    throw std::runtime_error("tp: cannot read a stash buffer's size");
                v_stash2[s].emplace_back(sz);
            }
        alloc_seq_state();  // ends on device 1
        use_dev(0);
        const size_t xb = size_t(NBUF) * cfg.hidden * sizeof(bf16);
        for (int r = 0; r < 2; ++r) {
            use_dev(r);
            for (int s = 0; s < 2; ++s) {
                ck(cudaHostAlloc(reinterpret_cast<void **>(&tp_host[r][s]), xb, cudaHostAllocMapped | cudaHostAllocPortable));
                ck(cudaEventCreateWithFlags(&tp_made[r][s], cudaEventDisableTiming));
                ck(cudaEventCreateWithFlags(&tp_used[r][s], cudaEventDisableTiming));
            }
        }
        use_dev(0);
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

// ---- tensor parallel: shards cut from the raw GGUF bytes (docs/design/tensor-parallel.md) ----

namespace {
int total(const std::vector<std::pair<int, int>> & r) {
    int n = 0;
    for (const auto & [a, b] : r) n += b - a;
    return n;
}
}  // namespace

// GGUF block formats store a matrix row by row, each row a run of fixed-size blocks: a row range is a byte range and a
// column range on a block boundary is a per-row block range, whatever the format.
std::vector<char> Qwen35Engine::tp_slice(const std::vector<char> & raw, uint32_t type, int rows, int cols, const Ranges & rr,
                                         const Ranges & cr) {
    uint64_t be = 0, bb = 0;
    if (!gguf_block_layout(type, be, bb)) throw std::runtime_error("tp: no block layout for GGUF type " + std::to_string(type));
    if (uint64_t(cols) % be) throw std::runtime_error("tp: a row is not a whole number of blocks");
    const size_t row_bytes = size_t(uint64_t(cols) / be * bb);
    if (raw.size() != row_bytes * size_t(rows)) throw std::runtime_error("tp: tensor size does not match its shape");
    size_t out_row = 0;
    for (const auto & [c0, c1] : cr) {
        if (c0 < 0 || c1 > cols || c0 >= c1 || uint64_t(c0) % be || uint64_t(c1) % be)
            throw std::runtime_error("tp: column range off a block boundary");
        out_row += size_t(uint64_t(c1 - c0) / be * bb);
    }
    std::vector<char> out(out_row * size_t(total(rr)));
    char * o = out.data();
    for (const auto & [r0, r1] : rr) {
        if (r0 < 0 || r1 > rows || r0 >= r1) throw std::runtime_error("tp: bad row range");
        for (int r = r0; r < r1; ++r)
            for (const auto & [c0, c1] : cr) {
                const size_t off = size_t(r) * row_bytes + size_t(uint64_t(c0) / be * bb), len = size_t(uint64_t(c1 - c0) / be * bb);
                std::memcpy(o, raw.data() + off, len);
                o += len;
            }
    }
    return out;
}

Mat Qwen35Engine::g_mat_cut(const GgufFile & f, const std::string & name, int rows, int cols, const Ranges & rr, const Ranges & cr) {
    const auto & t = gt(f, name);
    if (t.ne.size() != 2 || t.ne[0] != uint64_t(cols) || t.ne[1] != uint64_t(rows))
        throw std::runtime_error("unexpected shape for " + name);
    const int nr = total(rr), nc = total(cr);
    const auto host = tp_slice(f.read(t), t.type, rows, cols, rr, cr);
    if (t.type == GGML_F32) {
        const auto b = f32_to_bf16_host(host);
        return {put(b.data(), b.size()), GGML_BF16, nr, nc};
    }
    if (t.type == GGML_F16) {
        const auto b = f16_to_bf16_host(host);
        return {put(b.data(), b.size()), GGML_BF16, nr, nc};
    }
    if (t.type != GGML_BF16)
        if (const auto * wf = weight_format(t.type)) return adopt(wf->load(host, nr, nc, uploader(), stream));
    if (t.type != GGML_BF16) throw std::runtime_error("tp: unsupported matrix type " + std::to_string(t.type) + " for " + name);
    return {put(host.data(), host.size()), t.type, nr, nc};
}

Mat Qwen35Engine::g_cat_cut(const GgufFile & f, const std::vector<CutPart> & parts, int cols) {
    uint32_t type = 0;
    bool mixed = false;
    for (const auto & p : parts) {
        const auto & t = gt(f, p.name);
        if (t.ne.size() != 2 || t.ne[0] != uint64_t(cols) || t.ne[1] != uint64_t(p.rows))
            throw std::runtime_error("unexpected shape for " + p.name);
        mixed |= type && t.type != type;
        type = t.type;
    }
    const Ranges all_cols{{0, cols}};
    static const std::set<uint32_t> ok = {GGML_Q8_0, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_IQ3_XXS, GGML_IQ3_S,
                                          GGML_IQ4_XS, GGML_IQ4_NL, GGML_Q2_K, GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ1_S};
    if (mixed && !std::all_of(parts.begin(), parts.end(), [&](const CutPart & p) { return ok.count(gt(f, p.name).type) > 0; }))
        return {};  // not combinable: the caller loads the parts some other way (as g_cat)
    if (mixed) {  // per-part formats as one composite matrix (the same rule as g_cat)
        std::vector<Mat> ms;
        for (const auto & p : parts) ms.push_back(g_mat_cut(f, p.name, p.rows, cols, p.rr, all_cols));
        auto c = composite_format(std::move(ms));
        if (!c) return {};
        composites.push_back(std::move(c));
        return composite_mat(*composites.back());
    }
    int rows = 0;
    std::vector<char> host;
    for (const auto & p : parts) {
        const auto d = tp_slice(f.read(gt(f, p.name)), type, p.rows, cols, p.rr, all_cols);
        host.insert(host.end(), d.begin(), d.end());
        rows += total(p.rr);
    }
    if (type == GGML_BF16) return {put(host.data(), host.size()), type, rows, cols};
    if (type == GGML_F32) {
        const auto b = f32_to_bf16_host(host);
        return {put(b.data(), b.size()), GGML_BF16, rows, cols};
    }
    if (type == GGML_F16) {
        const auto b = f16_to_bf16_host(host);
        return {put(b.data(), b.size()), GGML_BF16, rows, cols};
    }
    if (const auto * wf = weight_format(type)) return adopt(wf->load(host, rows, cols, uploader(), stream));
    return {};
}

// All-reduce of v_m or v_x ([n][hidden] bf16) across the two GPUs through mapped host memory: each GPU stores its
// partial (zero-copy), then adds the other's. Two buffers per GPU alternate; a GPU overwrites one only after the other
// GPU has read what it held two all-reduces ago (tp_used). ~45 us for an 8-column block on 2x RTX 5060 Ti.
void Qwen35Engine::tp_allreduce(bool m_buf, int n) {
    const size_t cnt = size_t(n) * cfg.hidden;
    const int s = int(tp_k & 1u);
    for (int r = 0; r < 2; ++r) {
        use_dev(r, false);
        ck(cudaStreamWaitEvent(stream, tp_used[1 - r][s], 0));
        tqllm_cuda::tp_store((m_buf ? v_m : v_x).as<bf16>(), tp_host[r][s], cnt, stream);
        ck(cudaEventRecord(tp_made[r][s], stream));
    }
    for (int r = 0; r < 2; ++r) {
        use_dev(r, false);
        ck(cudaStreamWaitEvent(stream, tp_made[1 - r][s], 0));
        tqllm_cuda::tp_add((m_buf ? v_m : v_x).as<bf16>(), tp_host[1 - r][s], cnt, stream);
        ck(cudaEventRecord(tp_used[r][s], stream));
    }
    ++tp_k;
}

// verify_body under tensor parallelism (one slot, no drafter features, no images, no KV staging): both GPUs run every
// layer on their half of the heads / FFN rows; the host interleaves them segment by segment, the residual stream is
// the same on both after each all-reduce. The head runs on the home GPU.
void Qwen35Engine::verify_body_tp(int n, int max_seq) {
    const int H = int(cfg.hidden), lkh = int(cfg.lin_kheads), lvh = int(cfg.lin_vheads);
    const int dk = int(cfg.lin_dk), dv = int(cfg.lin_dv);
    const int qkv_dim = 2 * lkh * dk + lvh * dv, zdim = lvh * dv;
    use_dev(0);  // waits for the home GPU's token / position uploads
    split_mirror_control();
    for (int r = 0; r < 2; ++r) {
        use_dev(r, false);
        embed_multi(v_tokens.as<uint32_t>(), n, v_x.as<bf16>());
        tqllm_cuda::sumsq(v_x.as<bf16>(), v_px.as<float>(), n * H, stream);
    }
    uint32_t attn_i = 0, lin_i = 0;
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        for (int r = 0; r < 2; ++r) {
            use_dev(r, false);
            const auto & L = layer[l];
            auto *X = v_x.as<bf16>(), *M = v_m.as<bf16>(), *T = v_in.as<bf16>(), *O = v_o.as<bf16>(), *CORE = v_core.as<bf16>();
            auto * XH = v_xh.as<__half>();
            nrot_h(X, L.in_norm, v_px.as<float>(), XH, H, n);
            if (cfg.full[l]) {
                attn_mixer(L, key_cache[attn_i].as<bf16>(), value_cache[attn_i].as<bf16>(), n, max_seq);
            } else {
                const int rows = mix_rows(L);
                bf16 * TL = n <= NVER ? v_stash2[sp][lin_i].as<bf16>() : T;
                mix_mm(L, TL, n);
                bf16 * QKVC = v_qkvc.as<bf16>();
                tqllm_cuda::conv_multi(TL, rows, n, conv_slot(lin_i, cur), conv_slot(lin_i, cur + 1), L.conv_w, QKVC,
                                       qkv_dim, stream, vpar);
                const bf16 * B0 = TL + qkv_dim + zdim;  // row layout [qkv | z | b | a]
                tqllm_cuda::gdn_multi(QKVC, n, B0 + lvh, B0, rows, L.neg_a, L.dt_bias, rec_slot(lin_i, cur),
                                      rec_slot(lin_i, cur + 1), CORE, lkh, lvh, dk, dv, cfg.tiled, stream, vpar);
                if (basis_ == Basis::Tq3Rot32) {
                    tqllm_cuda::gated_rms_rot_h(CORE, TL + qkv_dim, rows, n, L.gnorm_w, XH, lvh, dv, cfg.eps, stream);
                } else {
                    tqllm_cuda::gated_rms_multi(CORE, TL + qkv_dim, rows, n, L.gnorm_w, O, lvh, dv, cfg.eps, stream);
                    rot_h(O, XH, zdim, n);
                }
                mm_h(L.w_out, M, n);
            }
        }
        cfg.full[l] ? ++attn_i : ++lin_i;
        tp_allreduce(true, n);  // the mixer's partial (o / GDN out projection)
        tp_hold = true;
        for (int r = 0; r < 2; ++r) {
            use_dev(r, false);
            ffn_block(layer[l], n, v_x.as<bf16>());  // residual + norm + gate / up + down (partial into v_x)
        }
        tp_hold = false;
        tp_allreduce(false, n);  // the FFN's partial (down projection)
        for (int r = 0; r < 2; ++r) {
            use_dev(r, false);
            tqllm_cuda::add_sumsq(v_x.as<bf16>(), v_m.as<bf16>(), v_px.as<float>(), n * H, stream);
        }
    }
    use_dev(1, false);
    auto * X = v_x.as<bf16>();
    ck(cudaMemcpyAsync(v_hid.p, X, size_t(n) * H * 2, cudaMemcpyDeviceToDevice, stream));
    if (!segs.empty()) {
        head_argmax(out_norm, n, 0);
    } else if (n > NVER) {
        if (prefill_final) {
            ck(cudaMemcpyAsync(X, X + size_t(n - 1) * H, size_t(H) * 2, cudaMemcpyDeviceToDevice, stream));
            head_argmax(out_norm, 1);
        }
    } else {
        head_argmax(out_norm, n);
    }
}

// Rank r's shard of one dense qwen35 layer (the slice map of docs/design/tensor-parallel.md), with cfg still whole.
void Qwen35Engine::load_layer_tp(const GgufFile & f, Layer & L, uint32_t l, int r) {
    const int H = int(cfg.hidden), F = int(cfg.ffn), hd = int(cfg.head_dim), nh = int(cfg.heads), nkv = int(cfg.kvheads);
    const int kh = int(cfg.lin_kheads), vh = int(cfg.lin_vheads), dk = int(cfg.lin_dk), dv = int(cfg.lin_dv);
    const int lk = kh * dk, lv = vh * dv, qkvd = 2 * lk + lv;
    const std::string p = "blk." + std::to_string(l) + ".";
    const Ranges all_h{{0, H}};
    auto half = [&](int count, int unit) { return Ranges{{r * (count / 2) * unit, (r + 1) * (count / 2) * unit}}; };
    auto scaled = [](const Ranges & rr, int unit, int base) {
        Ranges o;
        for (const auto & [a, b] : rr) o.push_back({base + a * unit, base + b * unit});
        return o;
    };
    // FFN: column-parallel gate / up (rows), row-parallel down (columns)
    const Ranges fr = half(F, 1);
    L.w_gu = g_cat_cut(f, {{p + "ffn_gate.weight", F, fr}, {p + "ffn_up.weight", F, fr}}, H);
    if (!L.w_gu.p) throw std::runtime_error("TQLLM_TP: " + p + "ffn_gate / ffn_up cannot be fused");
    L.w_down = g_mat_cut(f, p + "ffn_down.weight", H, F, all_h, fr);
    if (cfg.full[l]) {
        // attention: q (with its gate rows, [q | gate] per head), k, v by heads; o by its input columns
        const int qrows = (cfg.gated_attention ? 2 : 1) * hd;
        L.w_mix = g_cat_cut(f, {{p + "attn_q.weight", nh * qrows, half(nh, qrows)}, {p + "attn_k.weight", nkv * hd, half(nkv, hd)},
                                {p + "attn_v.weight", nkv * hd, half(nkv, hd)}}, H);
        if (!L.w_mix.p) throw std::runtime_error("TQLLM_TP: " + p + "attention q / k / v cannot be fused");
        L.wo = g_mat_cut(f, p + "attn_output.weight", H, nh * hd, all_h, half(nh, hd));
        L.q_norm = g_f32(f, p + "attn_q_norm.weight", hd);
        L.k_norm = g_f32(f, p + "attn_k_norm.weight", hd);
        return;
    }
    // Gated DeltaNet. GGUF keeps the value heads tiled (value head h reads key head h % kh), so rank r keeps key heads
    // [r * kh / 2, (r + 1) * kh / 2) and, of every group of kh value heads, the same kh / 2: local value head
    // g * kh / 2 + i is global g * kh + r * kh / 2 + i, and h % (kh / 2) still names its local key head.
    const int kh2 = kh / 2;
    Ranges vrun;  // value heads, in heads
    for (int g = 0; g < vh / kh; ++g) vrun.push_back({g * kh + r * kh2, g * kh + r * kh2 + kh2});
    Ranges qkv_rows{{r * kh2 * dk, (r + 1) * kh2 * dk}, {lk + r * kh2 * dk, lk + (r + 1) * kh2 * dk}};
    for (const auto & x : scaled(vrun, dv, 2 * lk)) qkv_rows.push_back(x);
    const Ranges z_rows = scaled(vrun, dv, 0);
    L.w_mix = g_cat_cut(f, {{p + "attn_qkv.weight", qkvd, qkv_rows}, {p + "attn_gate.weight", lv, z_rows},
                            {p + "ssm_beta.weight", vh, vrun}, {p + "ssm_alpha.weight", vh, vrun}}, H);
    if (!L.w_mix.p) {  // beta / alpha in a dense type beside quantised qkv / gate: two matrices (as the one-GPU load)
        L.w_mix = g_cat_cut(f, {{p + "attn_qkv.weight", qkvd, qkv_rows}, {p + "attn_gate.weight", lv, z_rows}}, H);
        L.w_ba = g_cat_cut(f, {{p + "ssm_beta.weight", vh, vrun}, {p + "ssm_alpha.weight", vh, vrun}}, H);
        if (!L.w_mix.p || L.w_ba.type != GGML_BF16) throw std::runtime_error("TQLLM_TP: " + p + "unexpected Gated DeltaNet types");
    }
    L.w_out = g_mat_cut(f, p + "ssm_out.weight", H, lv, all_h, z_rows);
    L.conv_w = g_f32_cut(f, p + "ssm_conv1d.weight", qkvd, int(cfg.conv_k), qkv_rows);
    L.neg_a = g_f32_cut(f, p + "ssm_a", vh, 1, vrun);
    L.dt_bias = g_f32_cut(f, p + "ssm_dt.bias", vh, 1, vrun);
    L.gnorm_w = g_f32(f, p + "ssm_norm.weight", cfg.lin_dv);
}

const float * Qwen35Engine::g_f32_cut(const GgufFile & f, const std::string & name, int rows, int row_len, const Ranges & rr) {
    const auto & t = gt(f, name);
    const uint64_t n = uint64_t(rows) * uint64_t(row_len);
    if (t.type != GGML_F32 || t.bytes != n * 4) throw std::runtime_error("tp: expected F32[" + std::to_string(n) + "] " + name);
    const auto host = f.read(t);
    std::vector<float> out;
    out.reserve(size_t(total(rr)) * size_t(row_len));
    const auto * v = reinterpret_cast<const float *>(host.data());
    for (const auto & [r0, r1] : rr) out.insert(out.end(), v + size_t(r0) * row_len, v + size_t(r1) * row_len);
    return static_cast<const float *>(put(out.data(), out.size() * 4));
}

}  // namespace qwen35_detail
}  // namespace tqllm
