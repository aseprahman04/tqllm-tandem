#pragma once
#ifndef TQLLM_VERIFY_COLUMNS
#define TQLLM_VERIFY_COLUMNS 16
#endif

namespace tqllm {
// Width of the largest CUDA tree that can be replayed by this build.
inline constexpr int kCopyVerifyColumns = TQLLM_VERIFY_COLUMNS < 16 ? 16 : TQLLM_VERIFY_COLUMNS;
inline constexpr int kReplayColumns = kCopyVerifyColumns < 32 ? kCopyVerifyColumns : 32;
static_assert(TQLLM_VERIFY_COLUMNS >= 8 && (TQLLM_VERIFY_COLUMNS <= 16 || TQLLM_VERIFY_COLUMNS == 24 || TQLLM_VERIFY_COLUMNS == 32 || TQLLM_VERIFY_COLUMNS == 48 || TQLLM_VERIFY_COLUMNS == 64 || TQLLM_VERIFY_COLUMNS == 96));
}
