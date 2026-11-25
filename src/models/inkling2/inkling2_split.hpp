#pragma once

namespace tqllm {
// Stage zero owns the contiguous interval. An empty interval disables the split.
constexpr int inkling2_layer_stage(int layer, int begin, int end) {
    return end > begin && (layer < begin || layer >= end) ? 1 : 0;
}
}  // namespace tqllm
