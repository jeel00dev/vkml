#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vkml {

/// Element types vkml can store.
///
/// Deliberately small. ggml carries ~30 types because quantised inference needs
/// them; training does not, and every extra type multiplies the kernel matrix
/// (docs/ARCHITECTURE.md §9 lists quantisation as an explicit non-goal).
///
/// BF16 is absent because this project's target GPU does not support it
/// (measured: docs/ARCHITECTURE.md §1.1).
enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,
    I32 = 2,
    I64 = 3,
    Bool = 4,
};

inline constexpr int kNumDTypes = 5;

[[nodiscard]] constexpr size_t dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::I32: return 4;
        case DType::I64: return 8;
        case DType::Bool: return 1;
    }
    return 0;
}

[[nodiscard]] constexpr std::string_view dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::I32: return "i32";
        case DType::I64: return "i64";
        case DType::Bool: return "bool";
    }
    return "?";
}

[[nodiscard]] constexpr bool is_floating(DType dt) noexcept {
    return dt == DType::F32 || dt == DType::F16;
}

[[nodiscard]] constexpr bool is_integral(DType dt) noexcept {
    return dt == DType::I32 || dt == DType::I64;
}

/// Only floating tensors can carry gradients, matching PyTorch.
[[nodiscard]] constexpr bool is_differentiable(DType dt) noexcept { return is_floating(dt); }

// ---------------------------------------------------------------------------
// fp16 <-> fp32
//
// IEEE-754 binary16. Handles subnormals, infinities and NaN correctly; the
// naive bit-shuffle that most tutorials show silently flushes subnormals to
// zero, which would show up as a tolerance failure against PyTorch only for
// very small values and would be miserable to track down later.
//
// Reference for the branch-free approach: Fabian Giesen's float_to_half_fast3,
// the same lineage ggml's implementation comes from.
//
// WHAT COMPUTES:
//
//   F32   everything.
//   F16   the same operators as F32 on both backends, with one exception: prod
//         is CPU-only for everyone (see VulkanBackend::supports). Storage only,
//         never an accumulator -- values widen to float at the memory boundary
//         and narrow once on the store.
//   I32   storage and cast only. I64 additionally indexes -- index_select,
//   I64   scatter_add and the argmax/argmin results. Neither is an arithmetic
//         type: there are no integer kernels, and every operator that would
//         need one raises rather than reinterpreting the bytes.
//   Bool  masks: comparison results, and where()'s condition.
//
// This note replaced one saying f16 arithmetic would arrive in M9. It was a
// correct plan that the Phase 2 manifesto superseded by requiring mixed
// precision in P2, and it had drifted into being read as "f16 does not work".
// ---------------------------------------------------------------------------

[[nodiscard]] float fp16_to_fp32(uint16_t h) noexcept;

[[nodiscard]] uint16_t fp32_to_fp16(float f) noexcept;

/// Trivial storage wrapper. Not an arithmetic type on purpose: any accumulation
/// must be done in fp32 (docs/ARCHITECTURE.md §7.3), and an implicit-conversion
/// half type makes it far too easy to accumulate in 16 bits by accident.
struct Half {
    uint16_t bits = 0;

    Half() = default;

    explicit Half(float f) noexcept : bits(fp32_to_fp16(f)) {}

    [[nodiscard]] float to_float() const noexcept { return fp16_to_fp32(bits); }
};

static_assert(sizeof(Half) == 2);

}  // namespace vkml
