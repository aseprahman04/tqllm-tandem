#include "dflash_kernels.cuh"
#include <array>
#include <cstdio>
#include <stdexcept>

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
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
    ck(cudaMalloc(&tokens, 18 * sizeof(*tokens)));
    ck(cudaMalloc(&parents, 18 * sizeof(*parents)));
    ck(cudaMalloc(&depths, 18 * sizeof(*depths)));
    for (int len = 2; len <= 15; ++len) {
        for (int duplicate = 0; duplicate <= 1; ++duplicate) {
            std::array<uint32_t,18> t{}, copy{};
            std::array<int,18> p{}, d{};
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
        }
    }
    ck(cudaFree(tokens)); ck(cudaFree(parents)); ck(cudaFree(depths));
    ck(cudaFree(ids)); ck(cudaFree(scores)); ck(cudaFree(anchor));
    std::puts("28 copy trees matched, including the root, sibling and untouched trailing columns");
}
