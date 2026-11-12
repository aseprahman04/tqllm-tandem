#pragma once
// Multi-GPU fork addition (not part of upstream tqLLM).
// CUDA function attributes, such as a kernel's dynamic shared-memory limit, are set per device. A process-wide
// "set once" guard leaves every other device at the default (48 KB) and its launches fail with "invalid argument".
// PerDeviceOnce runs its body once on each device that reaches it.
#include <cuda_runtime.h>

namespace tqllm_cuda {

struct PerDeviceOnce {
    unsigned done = 0;  // one bit per device ordinal (0..31)
    template <class F>
    void operator()(F && f) {
        int d = 0;
        cudaGetDevice(&d);
        if (d < 0 || d >= 32) {
            f();
            return;
        }
        if (!(done >> d & 1u)) {
            f();
            done |= 1u << d;
        }
    }
};

// The "raise the limit when a launch needs more than before" guard, per device: raise(need) is true (and records need)
// when need is above what this device was last set to.
struct PerDeviceMax {
    long long most[32] = {};
    bool raise(long long need) {
        int d = 0;
        cudaGetDevice(&d);
        if (d < 0 || d >= 32) return true;
        if (need <= most[d]) return false;
        most[d] = need;
        return true;
    }
};

}  // namespace tqllm_cuda
