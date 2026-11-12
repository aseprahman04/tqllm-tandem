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

}  // namespace tqllm_cuda
