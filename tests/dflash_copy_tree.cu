#include "dflash_kernels.cuh"
#include <array>
#include "verify_columns.hpp"
#include <cstdio>
#include <stdexcept>

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    constexpr int capacity = tqllm::kCopyVerifyColumns + 2;
    uint32_t* tokens;
    int* parents;
    int* depths;
    uint32_t *ids,*anchor;
    float *scores;
    ck(cudaMalloc(&ids,32*sizeof(*ids)));
    ck(cudaMalloc(&scores,272*sizeof(*scores)));
    ck(cudaMalloc(&anchor,sizeof(*anchor)));
    const uint32_t root=7;
    ck(cudaMemcpy(anchor,&root,sizeof(root),cudaMemcpyHostToDevice));
    ck(cudaMalloc(&tokens, capacity * sizeof(*tokens)));
    ck(cudaMalloc(&parents, capacity * sizeof(*parents)));
    ck(cudaMalloc(&depths, capacity * sizeof(*depths)));
    for (int len = 2; len < tqllm::kReplayColumns; ++len) {
        for (int duplicate = 0; duplicate <= 1; ++duplicate) {
            std::array<uint32_t,capacity> t{}, copy{};
            std::array<int,capacity> p{}, d{};
            t.fill(999); p.fill(-99); d.fill(-99);
            for (int i = 0; i < len; ++i) copy[i] = 40 + i;
            std::array<uint32_t,32> candidates{};
            std::array<float,272> lattice{};
            const int best=len%14;
            lattice[256+best]=lattice[257+best]=2.f;
            const uint32_t neural=duplicate ? copy[0] : 150;
            candidates[16+best]=neural; candidates[17+best]=17;
            ck(cudaMemcpy(ids,candidates.data(),sizeof(candidates),cudaMemcpyHostToDevice));
            ck(cudaMemcpy(scores,lattice.data(),sizeof(lattice),cudaMemcpyHostToDevice));
            auto et = t; auto ep = p; auto ed = d;
            et[0]=7; ep[0]=-1; ed[0]=0;
            for (int c=1; c <= len; ++c) {
                const bool sibling=c==len && !duplicate;
                et[c]=sibling ? neural : copy[c-1];
                ep[c]=sibling ? 0 : c-1;
                ed[c]=sibling ? 1 : c;
            }
            ck(cudaMemcpy(tokens,t.data(),sizeof(t),cudaMemcpyHostToDevice));
            ck(cudaMemcpy(parents,p.data(),sizeof(p),cudaMemcpyHostToDevice));
            ck(cudaMemcpy(depths,d.data(),sizeof(d),cudaMemcpyHostToDevice));
            tqllm_cuda::dflash_copy_tree(ids,scores,anchor,copy.data(),len,tokens,parents,depths,nullptr);
            ck(cudaMemcpy(t.data(),tokens,sizeof(t),cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(p.data(),parents,sizeof(p),cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(d.data(),depths,sizeof(d),cudaMemcpyDeviceToHost));
            if (t != et || p != ep || d != ed) return 1;
            tqllm_cuda::dflash_copy_chain(root,copy.data(),len,tokens,parents,depths,nullptr);
            ck(cudaMemcpy(t.data(),tokens,sizeof(t),cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(p.data(),parents,sizeof(p),cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(d.data(),depths,sizeof(d),cudaMemcpyDeviceToHost));
            for (int j = 0; j <= len; ++j) {
                et[j] = j ? copy[j-1] : root; ep[j] = j-1; ed[j] = j;
            }
            if (t != et || p != ep || d != ed) return 1;
        }
    }
    if constexpr (tqllm::kCopyVerifyColumns > tqllm::kReplayColumns) {
        std::array<uint32_t,capacity> copy{}, actual{};
        std::array<int,capacity> par{}, dep{};
        for (int c = 0; c < capacity; ++c) copy[c] = 321 + c;
        for (int len = tqllm::kReplayColumns; len < tqllm::kCopyVerifyColumns; ++len) {
            ck(cudaMemset(tokens, 0xff, sizeof(actual)));
            ck(cudaMemset(parents, 0xff, sizeof(par)));
            ck(cudaMemset(depths, 0xff, sizeof(dep)));
            tqllm_cuda::dflash_copy_chain(root, copy.data(), len, tokens, parents, depths, nullptr);
            ck(cudaMemcpy(actual.data(), tokens, sizeof(actual), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(par.data(), parents, sizeof(par), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(dep.data(), depths, sizeof(dep), cudaMemcpyDeviceToHost));
            for (int c = 0; c < capacity; ++c) {
                if (actual[c] != (c > len ? UINT32_MAX : c == 0 ? root : copy[c-1]) ||
                    par[c] != (c > len ? -1 : c-1) || dep[c] != (c > len ? -1 : c)) return 1;
            }
        }
    }
    ck(cudaFree(tokens)); ck(cudaFree(parents)); ck(cudaFree(depths));
    ck(cudaFree(ids)); ck(cudaFree(scores)); ck(cudaFree(anchor));
    std::puts("copy trees matched, including the root, sibling and untouched trailing columns");
}
