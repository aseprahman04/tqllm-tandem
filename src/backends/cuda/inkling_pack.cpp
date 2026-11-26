// Single-slot Tandem backend for the Inkling NVFP4/MXFP8 pack. The CUDA model is linked
// into tqserve. Tandem owns tokenization, request scheduling and prefix selection.
#include "registry.hpp"
#include "prefill_observer.hpp"
#include "tqllm/tokenizer.hpp"
#include "basalt/core/checkpoint_transfer.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/spec/draft_policy.hpp"
#include "basalt/spec/suffix_drafter.hpp"

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>

namespace basalt::models::inkling2 {
std::unique_ptr<model::ModelAdapter> make_adapter();
}

namespace tqllm {
namespace {
using Clock = std::chrono::steady_clock;
void require(basalt::model::Result r) {
    if (!r.ok()) throw std::runtime_error(r.message);
}
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
class InklingPack final : public Engine {
public:
    InklingPack(const std::string &path, const std::string &tokenizer, uint32_t ctx,
                int slots, const EngineOptions &options) : capacity_(ctx), cache_(options.prompt_cache) {
        if (slots != 1 || options.tp_world != 1 || options.kv_compact || !options.draft_model.empty())
            throw std::invalid_argument("Inkling pack requires one slot and local GPUs; its optional DSpark drafter is in the pack");
        Tokenizer tokens;
        tokens.load(tokenizer);
        if (tokens.special_id("<|content_text|>") != 200004 || tokens.eos_id() != eos())
            throw std::invalid_argument("Inkling pack requires its matching Inkling tokenizer (--tokenizer)");
        user_token_ = tokens.special_id("<|message_user|>");
        end_message_ = tokens.special_id("<|end_message|>");
        basalt::model::LoadRequest load;
        load.paths = {path};
        load.max_context = ctx;
        load.draft = options.native_mtp;
        load.spec = load.draft ? kWindow : 1;
        if (load.draft && options.lookup > 0) {
            lookup_limit_ = std::min(options.lookup, kWindow - 1);
            lookup_match_ = options.lookup_first > 0 ? options.lookup_first : 8;
            lookup_.emplace(3, std::max(32, lookup_match_), ctx);
        }
        load.kv = options.kv.empty() ? "fp16" : options.kv;
        if (!options.cache_k.empty() || !options.cache_v.empty() ||
            (load.kv != "fp8" && load.kv != "fp16" && load.kv != "int8"))
            throw std::invalid_argument("Inkling pack supports --kv fp8, fp16 or int8");

        int count = 0;
        check_cuda(cudaGetDeviceCount(&count));
        if (count < 1 || count > 2) throw std::invalid_argument("Select one or two SM120 GPUs with CUDA_VISIBLE_DEVICES");
        for (int i = 0; i < count; ++i) {
            check_cuda(cudaSetDevice(i));
            cudaDeviceProp props{};
            check_cuda(cudaGetDeviceProperties(&props, i));
            if (props.major != 12 || props.minor != 0) throw std::invalid_argument("Inkling pack needs SM120 GPUs");
            size_t free_bytes, total_bytes;
            check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes));
            basalt::model::GpuCards::Card card;
            card.device = card.index = i;
            card.info.name = props.name;
            card.info.cc_major = props.major;
            card.info.cc_minor = props.minor;
            card.info.vram = total_bytes;
            card.free = free_bytes;
            card.label = "CUDA " + std::to_string(i) + " (" + props.name + ")";
            load.gpus.cards.push_back(std::move(card));
        }
        if (options.split_experts && (count != 2 || options.split_layers || options.split_at != -1))
            throw std::invalid_argument("Inkling expert split requires two local GPUs and no layer cut");
        if (options.split_at != -1 && (count != 2 || options.split_at < 1 || options.split_at >= 24))
            throw std::invalid_argument("Inkling --split-at requires two GPUs and a layer boundary from 1 to 23");
        if (count == 2) {
            load.gpus.mode = options.split_experts ? basalt::program::GpuMode::Peer : basalt::program::GpuMode::Split;
            if (options.split_at > 0) load.gpus.diag_cut = options.split_at;
        }
        check_cuda(cudaSetDevice(0));
        require(basalt::models::inkling2::make_adapter()->start(load, started_));
        require(session().prepare());
        dspark_ = load.draft;
        if (dspark_ && session().max_window() < 2) throw std::invalid_argument("Inkling pack has no usable DSpark drafter");
        note = "Inkling pack, Basalt CUDA in-process, KV " + load.kv;
        if (dspark_) note += ", DSpark";
        for (int i = 0; i < count; ++i) note += "; " + started_.model->gpu_report(i);
    }

    bool incremental() const override { return true; }
    bool has_drafter() const override { return dspark_; }
    bool native_mtp() const override { return dspark_; }
    uint32_t layers() const override { return uint32_t(started_.model->desc().n_layers); }
    uint32_t hidden() const override { return uint32_t(started_.model->desc().n_embd); }
    uint32_t vocab() const override { return uint32_t(started_.model->desc().n_vocab); }
    uint32_t eos() const override { return 200006; }
    uint32_t context() const override { return capacity_; }
    void synchronize() override { check_cuda(cudaDeviceSynchronize()); }
    void set_ignore_eos(bool ignore) override { ignore_eos_ = ignore; }
    void set_sampling(float temp, float p, int k, uint64_t seed, float presence) override {
        sampling_.temperature = temp;
        sampling_.greedy = temp <= 0;
        sampling_.top_p = p;
        sampling_.top_k = k;
        sampling_.seed = seed;
        sampling_.penalty_present = presence;
    }
    void set_penalties(float repeat, float frequency) override {
        sampling_.penalty_repeat = repeat;
        sampling_.penalty_freq = frequency;
    }
    uint32_t prefix_reuse(int, const std::vector<uint32_t> &ids) const override {
        if (checkpoints_.empty() || ids.empty()) return 0;
        // Leave the final input token to run the head with this request's sampling settings.
        const size_t limit = std::min(previous_.size(), ids.size() - 1);
        const auto mismatch = std::mismatch(ids.begin(), ids.begin() + limit, previous_.begin());
        const auto it = checkpoints_.upper_bound(uint32_t(mismatch.first - ids.begin()));
        return it == checkpoints_.begin() ? 0 : std::prev(it)->first;
    }
    void spec_abort(int) override { checkpoints_.clear(); previous_.clear(); }
    void spec_begin(const std::vector<uint32_t> &ids, uint32_t count, int draft_count) override {
        const nvtx3::scoped_range range{"Inkling prefill and first token"};
        if (ids.empty() || uint64_t(ids.size()) + count > capacity_)
            throw std::invalid_argument("Inkling pack request exceeds context");
        const auto start = Clock::now();
        check_cuda(cudaSetDevice(0));
        if (std::any_of(ids.begin(), ids.end(), [n=vocab()](uint32_t id){return id >= n;}))
            throw std::invalid_argument("Inkling token outside pack vocabulary");
        output_.clear();
        draft_limit_ = draft_count > 0 ? std::min(draft_count, kWindow - 1) : 4;
        policy_.reset();
        if (dspark_ && (draft_limit_ > kNeuralDrafts || lookup_))
            policy_.emplace(std::max(draft_limit_, lookup_limit_) + 1);
        next_drafts_ = 3;
        steps_ = drafted_ = accepted_ = 0;
        limit_ = count;
        const uint32_t tail = uint32_t(ids.size() - 1);
        const uint32_t reused = prefix_reuse(0, ids);
        uint32_t pos = reused;
        try {
            if (reused) transfer(reused, false);
            checkpoints_.erase(checkpoints_.upper_bound(reused), checkpoints_.end());
            // Save a long reference turn before the reply framing can diverge on compaction.
            uint32_t reference_end = 0, user_start = 0, longest = 0;
            for (uint32_t i = 0; cache_ && i < tail; ++i) {
                if (ids[i] == user_token_) user_start = i + 1;
                if (user_start && ids[i] == end_message_) {
                    const uint32_t length = i + 1 - user_start;
                    if (length >= 4096 && length > longest) {
                        longest = length;
                        reference_end = i + 1;
                    }
                    user_start = 0;
                }
            }
            // Save at roughly quarter-context intervals, at the reference boundary and at the prompt tail.
            const uint32_t stride = std::max(4096u, capacity_ / 4 / 4096 * 4096);
            std::erase_if(checkpoints_, [&](const auto &entry) {
                const auto pos = entry.first;
                return pos % stride && pos != reference_end && pos != tail;
            });
            std::vector<int32_t> input(ids.begin(), ids.end());
            if (lookup_) {
                lookup_->reset();
                lookup_->append(input.data(), input.size());
            }
            while (pos < tail) {
                uint32_t end = cache_ ? std::min(tail, (pos / stride + 1) * stride) : tail;
                if (pos < reference_end) end = std::min(end, reference_end);
                basalt::model::PrefillRequest r;
                r.tokens = input.data() + pos;
                r.pos = pos;
                r.n_tokens = int(end - pos);
                r.total_tokens = int(tail - pos);
                r.more = end < tail;
                // Each boundary must be independently resumable, including local attention and convolution rings.
                r.rows_after = 0;
                if (prefill_observer) r.progress = [&](int n){ prefill_observer(pos + n, uint32_t(ids.size()), reused); };
                require(session().prefill_chunk(r));
                pos = end;
                if (cache_) transfer(pos, true);
            }
            session().end_prompt();
            if (cache_) previous_ = ids;
            next_pos_ = tail;
            if (count) output_.push_back(pick(input.back()));
            if (lookup_ && count) lookup_->append(int32_t(output_.back()));
            if (prefill_observer) prefill_observer(uint32_t(ids.size()), uint32_t(ids.size()), reused);
        } catch (...) {
            spec_abort(0);
            session().end_prompt();
            throw;
        }
        decode_started_ = Clock::now();
        prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(decode_started_ - start).count();
    }
    bool spec_advance() override {
        if (output_.size() >= limit_ || (!ignore_eos_ && output_.back() == eos())) return false;
        const nvtx3::scoped_range range{"Inkling decode step"};
        const size_t before = output_.size();
        if (dspark_ && sampling_.greedy && !has_penalty() && limit_ - before > 1) verify_drafts();
        else output_.push_back(pick(int32_t(output_.back())));
        if (lookup_)
            for (size_t i = before; i < output_.size(); ++i) lookup_->append(int32_t(output_[i]));
        return true;
    }
    const std::vector<uint32_t>& spec_tokens(int) const override { return output_; }
    std::vector<uint32_t> spec_finish(uint64_t *stats) override {
        if (stats) {
            stats[0] = steps_;
            stats[1] = drafted_;
            stats[2] = accepted_;
            stats[3] = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - decode_started_).count();
            stats[4] = prefill_us_;
        }
        return std::move(output_);
    }

private:
    static constexpr int kWindow = 16;
    static constexpr int kNeuralDrafts = 7;
    basalt::model::ModelSession& session() { return *started_.sessions.front(); }
    bool has_penalty() const {
        return sampling_.penalty_present || sampling_.penalty_freq || sampling_.penalty_repeat != 1.f;
    }
    void commit_window(int keep) {
        auto *s = &session();
        require(s->decode_kept(&s, &keep, 1));
        ++steps_;
    }
    // Recent Basalt sessions can announce a placement change at a specific token.
    template<class Session>
    static int available_rows(Session &s, int64_t pos, int room) {
        if constexpr (requires { s.window_limit(pos); }) {
            const int limit = s.window_limit(pos);
            if (limit > 0) return std::min(room, limit);
        }
        return room;
    }
    void verify_drafts() {
        std::array<int32_t, kWindow> window{}, picks{};
        std::array<float, kWindow> confidence{};
        window[0] = int32_t(output_.back());
        const int room = std::min({std::max(draft_limit_, lookup_limit_) + 1, session().max_window(), int(limit_ - output_.size())});
        const int want = available_rows(session(), next_pos_, room) - 1;
        if (!want) {
            output_.push_back(pick(window[0]));
            return;
        }
        const auto start = Clock::now();
        int proposed = 0;
        int match_length = 0;
        if (lookup_) {
            proposed = lookup_->propose(std::min(lookup_limit_, want), window.data() + 1);
            match_length = lookup_->last_match();
            if (match_length < lookup_match_) proposed = 0;
        }
        const bool copied = proposed > 0;
        if (!copied) {
            const nvtx3::scoped_range range{"Inkling DSpark proposal"};
            const int ask = std::min({want, draft_limit_, policy_ ? next_drafts_ : kNeuralDrafts});
            require(session().draft_begin(window[0], next_pos_, ask));
            require(session().draft_end(window.data() + 1, confidence.data(), ask, proposed, {}));
            if (proposed < 0 || proposed > ask) throw std::runtime_error("Invalid Inkling draft count");
            if (policy_) proposed = policy_->window(confidence.data(), proposed) - 1;
        }
        basalt::model::DecodeRequest r;
        r.tokens = window.data();
        r.pos = next_pos_;
        r.n_tokens = proposed + 1;
        r.row_tokens = picks.data();
        sampling_.counter = output_.size();
        sampling_.penalty_last_n = 0;
        r.sampling = &sampling_;
        {
            const nvtx3::scoped_range range{"Inkling target verification"};
            require(session().decode_window(r));
        }
        int kept = 0;
        int matched = 0;
        for (int i = 0; i <= proposed; ++i) {
            if (picks[i] < 0 || uint32_t(picks[i]) >= vocab()) throw std::runtime_error("Invalid Inkling verifier token");
            output_.push_back(uint32_t(picks[i]));
            ++kept;
            const bool match = i < proposed && picks[i] == window[i + 1];
            if (match) ++matched;
            if (!match || (!ignore_eos_ && uint32_t(picks[i]) == eos())) break;
        }
        drafted_ += proposed;
        accepted_ += matched;
        next_pos_ += kept;
        commit_window(kept);
        if (policy_) {
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            if (copied) policy_->observe(true, proposed + 1, matched, match_length, ms);
            else {
                policy_->observe_window(confidence.data(), proposed + 1, matched, ms);
                next_drafts_ = std::clamp(proposed + 1, 2, kNeuralDrafts);
            }
        }
    }
    void transfer(uint32_t pos, bool save) {
        basalt::core::RunningState state;
        require(session().running_state(pos, save, state));
        auto &bytes = save ? checkpoints_[pos] : checkpoints_.at(pos);
        std::string error;
        if (save) {
            bytes.resize(state.bytes());
            std::copy(state.head.begin(), state.head.end(), bytes.begin());
            if (!basalt::core::running_copy_out(state, bytes.data(), error)) throw std::runtime_error(error);
        } else {
            if (bytes.size() != state.bytes()) throw std::runtime_error("Inkling checkpoint size changed");
            if (!basalt::core::running_copy_in(state, bytes.data(), error)) throw std::runtime_error(error);
            require(session().running_restored(pos, bytes.data()));
        }
    }
    uint32_t pick(int32_t token) {
        basalt::model::DecodeRequest request;
        int32_t next = -1;
        request.tokens = &token;
        request.n_tokens = 1;
        request.pos = next_pos_++;
        request.next_token = &next;
        sampling_.counter = output_.size();
        sampling_.penalty_last_n = has_penalty() ? int(std::min(size_t(64), output_.size())) : 0;
        std::vector<int32_t> history(output_.end() - sampling_.penalty_last_n, output_.end());
        request.history = history.data();
        request.history_len = int(history.size());
        request.sampling = &sampling_;
        require(session().decode_window(request));
        if (next < 0 || uint32_t(next) >= vocab()) throw std::runtime_error("Invalid Inkling output token");
        commit_window(1);
        return uint32_t(next);
    }
    basalt::model::Started started_;
    basalt::kernels::SamplerParams sampling_;
    std::vector<uint32_t> previous_, output_;
    std::map<uint32_t, std::vector<uint8_t>> checkpoints_;
    uint32_t user_token_, end_message_;
    uint32_t capacity_, limit_ = 0, next_pos_ = 0;
    bool cache_, ignore_eos_ = false;
    bool dspark_ = false;
    int draft_limit_ = 0;
    int lookup_limit_ = 0, lookup_match_ = 8, next_drafts_ = 3;
    std::optional<basalt::spec::DraftPolicy> policy_;
    std::optional<basalt::spec::SuffixDrafter> lookup_;
    uint64_t steps_ = 0, drafted_ = 0, accepted_ = 0;
    Clock::time_point decode_started_;
    uint64_t prefill_us_ = 0;
};

const bool registered = (register_engine({"inkling2-pack",
    [](const std::filesystem::path &p){ return p.extension() == ".basalt"; },
    [](const std::string &p, const std::string &t, uint32_t c, int slots, const EngineOptions &o) -> std::unique_ptr<Engine> {
        return std::make_unique<InklingPack>(p, t, c, slots, o);
    }, false}), true);
} // namespace
} // namespace tqllm
