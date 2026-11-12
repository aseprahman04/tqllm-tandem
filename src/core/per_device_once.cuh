#pragma once
// Multi-GPU fork addition (not part of upstream tqLLM).
// CUDA function attributes, such as a kernel's dynamic shared-memory limit, are set per device. A process-wide
// "set once" guard leaves every other device at the default (48 KB) and its launches fail with "invalid argument".
// PerDeviceLimit::raise(need) is true when this device was not yet set to `need` or more, and records it; raise(1)
// is "once per device".
#include <cuda_runtime.h>

namespace tqllm_cuda {

struct PerDeviceLimit {
    long long most[32] = {};  // per device ordinal
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
