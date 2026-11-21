// Dry-run (no GPU) tensor-name auditor for the Inkling-2-Mini GGUF (arch `inkling2`).
//
// Parses the GGUF header (pure file I/O, no CUDA), dumps every tensor (name + type + shape) to a
// text file, then enumerates the exact set of tensor names the loader (src/models/inkling2/inkling2.cpp)
// expects for 24 physical layers + head + embedding + loop queries, and reports every mismatch at once:
//   - expected but missing in the GGUF
//   - present in the GGUF but not consumed by the loader
//   - shape / type disagreement
//
// Usage: inkling2_names_check <model.gguf> [dump.txt]
// Exit code 0 = zero mismatches, 1 = mismatches found, 2 = usage / parse error.
#include "gguf.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using tqllm::GgufFile;
using tqllm::GgufTensor;

namespace {

std::string type_name(uint32_t t) {
    switch (t) {
        case 0: return "F32";
        case 1: return "F16";
        case 30: return "BF16";
        case 40: return "NVFP4";
        case 321: return "MXFP8";
        case 8: return "Q8_0";
        case 12: return "Q4_K";
        default: return "type" + std::to_string(t);
    }
}

std::string shape_str(const std::vector<uint64_t> & ne) {
    std::string s = "[";
    for (size_t i = 0; i < ne.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(ne[i]);
    }
    return s + "]";
}

// A tensor the loader expects: name, whether required, and the expected shape (empty = any).
struct Expect {
    std::string name;
    bool required = true;
    std::vector<uint64_t> shape;  // empty = don't check shape
    uint32_t type = 0;            // 0 = don't check type
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [dump.txt]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string dump_path = argc > 2 ? argv[2] : "gguf-tensors.txt";

    GgufFile f;
    try {
        f = tqllm::gguf_open_shard(path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: cannot open GGUF: %s\n", e.what());
        return 2;
    }

    // ---- 1. Dump every tensor (name, type, shape) to the text file ----
    {
        std::ofstream out(dump_path);
        if (!out) {
            std::fprintf(stderr, "FAIL: cannot write dump file %s\n", dump_path.c_str());
            return 2;
        }
        out << "# GGUF tensor dump: " << path << "\n";
        out << "# arch = " << (f.has("general.architecture") ? f.str("general.architecture") : std::string("?")) << "\n";
        out << "# n_tensors = " << f.tensors.size() << "\n\n";
        // Key metadata the loader reads (inkling2.* keys)
        out << "# ---- inkling2.* metadata ----\n";
        for (const auto & [k, v] : f.kv) {
            if (k.rfind("inkling2.", 0) == 0 || k == "general.architecture") {
                out << k << " = ";
                if (!v.str.empty()) out << v.str;
                else if (!v.arr.empty()) {
                    out << "[";
                    for (size_t i = 0; i < v.arr.size(); ++i) { if (i) out << ","; out << v.arr[i]; }
                    out << "]";
                } else out << v.num;
                out << "\n";
            }
        }
        out << "\n# ---- tensors (name, type, shape) ----\n";
        for (const auto & [name, t] : f.tensors)
            out << name << "\t" << type_name(t.type) << " (" << t.type << ")\t" << shape_str(t.ne) << "\n";
    }

    // ---- 2. Read the config the loader reads ----
    auto num = [&](const std::string & k, double fb) -> double {
        const std::string key = "inkling2." + k;
        return f.has(key) ? f.num(key) : fb;
    };
    const int H = int(num("embedding_length", 0));
    const int n_visits = int(num("block_count", 0));
    const int n_pass = int(num("num_loops", 1));
    const int core0 = int(num("loop.core_start", 0));
    const int core1 = int(num("loop.core_end", 0));
    const int n_core = core1 - core0;
    const int n_layers = n_visits - (n_pass - 1) * n_core;
    // head_count / head_count_kv are stored as arrays (one per visit); take the first element.
    auto arr0 = [&](const std::string & k) -> int {
        const std::string key = "inkling2." + k;
        if (!f.has(key)) return 0;
        const auto & v = f.kv.at(key);
        if (!v.arr.empty()) return int(v.arr[0]);
        return int(v.num);
    };
    const int n_head = arr0("attention.head_count");
    const int n_kv = arr0("attention.head_count_kv");
    const int hd = int(num("attention.key_length", 0));
    const int d_rel = int(num("attention.relative_length", 0));
    const int extent = int(num("attention.relative_extent", 0));
    const int window = int(num("attention.sliding_window", 0));
    const int taps = int(num("shortconv.l_cache", 4));
    const int block = int(num("attn_res.block_size", 6));
    const int n_exp = int(num("expert_count", 0));
    const int k_exp = int(num("expert_used_count", 0));
    const int e_ff = int(num("expert_feed_forward_length", 0));
    const int vocab = int(num("vocab_size", 0));
    const int vocab_unpadded = int(num("vocab_size_unpadded", vocab));

    std::fprintf(stderr, "config: H=%d n_visits=%d n_pass=%d core=[%d,%d) n_layers=%d n_head=%d n_kv=%d hd=%d "
                         "d_rel=%d extent=%d window=%d taps=%d block=%d n_exp=%d k_exp=%d e_ff=%d vocab=%d/%d\n",
                 H, n_visits, n_pass, core0, core1, n_layers, n_head, n_kv, hd, d_rel, extent, window, taps, block,
                 n_exp, k_exp, e_ff, vocab, vocab_unpadded);

    // sliding_window_pattern: 48 bools (one per visit). A physical layer is "full" if any of its
    // visits has pattern=false. This is the ONLY source of the sliding/full distinction in the GGUF.
    std::vector<bool> sliding(n_layers, true);
    if (f.has("inkling2.attention.sliding_window_pattern")) {
        const auto & v = f.kv.at("inkling2.attention.sliding_window_pattern");
        // The pattern is stored as a bool array (GGUF type 7 = bool, packed into v.arr as 0/1).
        std::vector<bool> pat;
        for (double x : v.arr) pat.push_back(x != 0.0);
        // Map each visit to its physical layer; if any visit of a layer is full, the layer is full.
        for (size_t vi = 0; vi < pat.size(); ++vi) {
            int v = int(vi);
            int p;
            if (v < core0) p = v;
            else if (v < core0 + n_pass * n_core) p = core0 + (v - core0) % n_core;
            else p = v - (n_pass - 1) * n_core;
            if (p >= 0 && p < n_layers && !pat[vi]) sliding[p] = false;
        }
    }
    // Count full layers for the report.
    int n_full = 0;
    for (int l = 0; l < n_layers; ++l) if (!sliding[l]) ++n_full;
    std::fprintf(stderr, "sliding: %d sliding, %d full (full layers:", n_layers - n_full, n_full);
    for (int l = 0; l < n_layers; ++l) if (!sliding[l]) std::fprintf(stderr, " %d", l);
    std::fprintf(stderr, ")\n");

    // ---- 3. Enumerate the expected tensors (mirror of inkling2.cpp load()) ----
    std::vector<Expect> exp;
    // Head + embedding (stage 1)
    exp.push_back({"token_embd.weight", true, {uint64_t(H), uint64_t(vocab)}, 0});
    exp.push_back({"token_embd_norm.weight", true, {uint64_t(H)}, 0});
    exp.push_back({"output.weight", true, {uint64_t(H), uint64_t(vocab)}, 0});
    exp.push_back({"output_norm.weight", true, {uint64_t(H)}, 0});
    exp.push_back({"output_res_query.weight", true, {uint64_t(H)}, 0});
    // loop_res_query / loop_kv_res_query: GGUF ne = [H, 2*n_core] (loader's load_mat swaps to rows=2*n_core, cols=H)
    exp.push_back({"loop_res_query.weight", true, {uint64_t(H), uint64_t(2 * n_core)}, 0});
    exp.push_back({"loop_kv_res_query.weight", true, {uint64_t(H), uint64_t(2 * n_core)}, 0});

    for (int l = 0; l < n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        exp.push_back({p + "ffn_norm.weight", true, {uint64_t(H)}, 0});
        if (sliding[l])
            exp.push_back({p + "attn_norm.weight", true, {uint64_t(H)}, 0});
        else {
            exp.push_back({p + "attn_q_inp_norm.weight", true, {uint64_t(H)}, 0});
            exp.push_back({p + "attn_k_inp_norm.weight", true, {uint64_t(H)}, 0});
            exp.push_back({p + "attn_v_inp_norm.weight", true, {uint64_t(H)}, 0});
        }
        // Attention projections. GGUF ne-order [ne0, ne1]; loader's load_mat reads cols=ne[0], rows=ne[1].
        //   attn_q [H, n_head*hd] = [2304, 3072]; attn_k/v [H, n_kv*hd] = [2304, 1024]
        //   attn_rel [H, d_rel*n_head] = [2304, 384]; attn_gate [H, n_head*hd] = [2304, 3072]
        //   attn_sink_proj [H, n_head] = [2304, 24]; attn_output [n_head*hd, H] = [3072, 2304]
        exp.push_back({p + "attn_q.weight", true, {uint64_t(H), uint64_t(n_head) * uint64_t(hd)}, 0});
        exp.push_back({p + "attn_k.weight", true, {uint64_t(H), uint64_t(n_kv) * uint64_t(hd)}, 0});
        exp.push_back({p + "attn_v.weight", true, {uint64_t(H), uint64_t(n_kv) * uint64_t(hd)}, 0});
        exp.push_back({p + "attn_rel.weight", true, {uint64_t(H), uint64_t(d_rel) * uint64_t(n_head)}, 0});
        exp.push_back({p + "attn_gate.weight", true, {uint64_t(H), uint64_t(n_head) * uint64_t(hd)}, 0});
        exp.push_back({p + "attn_sink_proj.weight", true, {uint64_t(H), uint64_t(n_head)}, 0});
        exp.push_back({p + "attn_output.weight", true, {uint64_t(n_head) * uint64_t(hd), uint64_t(H)}, 0});
        // Short convs [taps][C]
        exp.push_back({p + "attn_k_conv.weight", true, {uint64_t(taps), uint64_t(n_kv) * uint64_t(hd)}, 0});
        exp.push_back({p + "attn_v_conv.weight", true, {uint64_t(taps), uint64_t(n_kv) * uint64_t(hd)}, 0});
        exp.push_back({p + "attn_post_conv.weight", true, {uint64_t(taps), uint64_t(H)}, 0});
        exp.push_back({p + "ffn_post_conv.weight", true, {uint64_t(taps), uint64_t(H)}, 0});
        // Clips and relative bias
        exp.push_back({p + "attn_q_clip.weight", true, {uint64_t(hd), uint64_t(n_head)}, 0});
        exp.push_back({p + "attn_rel_clip.weight", true, {uint64_t(d_rel), uint64_t(n_head)}, 0});
        exp.push_back({p + "attn_rel_bias.weight", true, {uint64_t(d_rel), uint64_t(extent)}, 0});
        // attn_res_query: full layers have [H, 3] (3 queries), sliding layers have [H] (1 query).
        // The loader reads it optionally (meta_.tensors.count); layer 0 has none in the GGUF.
        // Mark optional so a missing one is not a hard error, but report the shape if present.
        if (sliding[l])
            exp.push_back({p + "attn_res_query.weight", false, {uint64_t(H)}, 0});
        else
            exp.push_back({p + "attn_res_query.weight", false, {uint64_t(H), 3}, 0});
        // ffn_res_query [H]
        exp.push_back({p + "ffn_res_query.weight", true, {uint64_t(H)}, 0});
        // Full layers: dynamic conv projections [H, taps*?] — from dump: [2304, 32] = [H, 8*taps]?
        // Actually [2304, 32]: rows=H, cols=32. The loader does load_mat (rows=ne[1]=32, cols=ne[0]=2304).
        if (!sliding[l]) {
            exp.push_back({p + "attn_k_conv_proj.weight", true, {uint64_t(H), 32}, 0});
            exp.push_back({p + "attn_v_conv_proj.weight", true, {uint64_t(H), 32}, 0});
        }
        // Core loop layers: loop_* tensors
        if (l >= core0 && l < core1) {
            exp.push_back({p + "loop_attn_down.weight", true, {uint64_t(2 * H), 64}, 0});
            exp.push_back({p + "loop_attn_forget.weight", true, {64, uint64_t(H)}, 0});
            exp.push_back({p + "loop_attn_gain.weight", true, {64, uint64_t(H)}, 0});
            exp.push_back({p + "loop_k_down.weight", true, {uint64_t(2 * H), 64}, 0});
            exp.push_back({p + "loop_k_forget.weight", true, {64, uint64_t(H)}, 0});
            exp.push_back({p + "loop_k_gain.weight", true, {64, uint64_t(H)}, 0});
            exp.push_back({p + "loop_v_down.weight", true, {uint64_t(2 * H), 64}, 0});
            exp.push_back({p + "loop_v_forget.weight", true, {64, uint64_t(H)}, 0});
            exp.push_back({p + "loop_v_gain.weight", true, {64, uint64_t(H)}, 0});
            exp.push_back({p + "loop_mlp_down.weight", true, {uint64_t(2 * H), 64}, 0});
            exp.push_back({p + "loop_mlp_forget.weight", true, {64, uint64_t(H)}, 0});
            exp.push_back({p + "loop_mlp_gain.weight", true, {64, uint64_t(H)}, 0});
        }
        // Router [n_exp, H]
        exp.push_back({p + "ffn_gate_inp.weight", true, {uint64_t(H), uint64_t(n_exp)}, 0});
        // exp_probs_b.bias [n_exp] (optional in loader)
        exp.push_back({p + "exp_probs_b.bias", false, {uint64_t(n_exp)}, 0});
        // Expert stacks. GGUF ne-order: gate_up [2*e_ff, H, n_exp] = [2304,2304,128]; down [e_ff, H, n_exp] = [1152,2304,128].
        // Loader's load_experts: rows = ne[1]*n_exp, cols = ne[0].
        exp.push_back({p + "ffn_gate_up_exps.weight", true, {uint64_t(e_ff) * 2, uint64_t(H), uint64_t(n_exp)}, 0});
        exp.push_back({p + "ffn_down_exps.weight", true, {uint64_t(e_ff), uint64_t(H), uint64_t(n_exp)}, 0});
        // Expert scales (optional)
        exp.push_back({p + "ffn_gate_up_exps.scale", false, {uint64_t(n_exp)}, 0});
        exp.push_back({p + "ffn_down_exps.scale", false, {uint64_t(n_exp)}, 0});
    }

    // ---- 4. Compare expected vs actual ----
    std::set<std::string> expected_names;
    for (const auto & e : exp) expected_names.insert(e.name);
    std::set<std::string> actual_names;
    for (const auto & [name, t] : f.tensors) actual_names.insert(name);

    int n_missing = 0, n_unused = 0, n_shape = 0, n_type = 0;
    std::vector<std::string> report;
    for (const auto & e : exp) {
        auto it = f.tensors.find(e.name);
        if (it == f.tensors.end()) {
            if (e.required) {
                ++n_missing;
                report.push_back("MISSING(required): " + e.name);
            }
            continue;
        }
        const GgufTensor & t = it->second;
        // Shape check
        if (!e.shape.empty() && t.ne != e.shape) {
            ++n_shape;
            report.push_back("SHAPE " + e.name + ": expected " + shape_str(e.shape) + " got " + shape_str(t.ne));
        }
        // Type check
        if (e.type && t.type != e.type) {
            ++n_type;
            report.push_back("TYPE " + e.name + ": expected " + type_name(e.type) + " got " + type_name(t.type));
        }
    }
    // Tensors in the GGUF but not expected
    for (const auto & [name, t] : f.tensors) {
        if (!expected_names.count(name)) {
            ++n_unused;
            report.push_back("UNUSED: " + name + " " + type_name(t.type) + " " + shape_str(t.ne));
        }
    }

    // ---- 5. Print report ----
    std::fprintf(stderr, "\n=== MISMATCH REPORT ===\n");
    std::fprintf(stderr, "missing(required): %d\n", n_missing);
    std::fprintf(stderr, "shape mismatch:    %d\n", n_shape);
    std::fprintf(stderr, "type mismatch:     %d\n", n_type);
    std::fprintf(stderr, "unused (in GGUF, not expected): %d\n", n_unused);
    for (const auto & line : report) std::fprintf(stderr, "  %s\n", line.c_str());
    const int total = n_missing + n_shape + n_type + n_unused;
    std::fprintf(stderr, "TOTAL mismatches: %d\n", total);
    std::fprintf(stderr, "dump written to: %s\n", dump_path.c_str());
    return total == 0 ? 0 : 1;
}
