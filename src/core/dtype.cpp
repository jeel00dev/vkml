#include "vkml/core/dtype.h"

#include <bit>
#include <cmath>

namespace vkml {
namespace {

// C++20 std::bit_cast replaces the union punning that ggml's C implementation
// needs, and is well-defined rather than merely widely-supported.
[[nodiscard]] constexpr float bits_to_float(uint32_t w) noexcept {
    return std::bit_cast<float>(w);
}

[[nodiscard]] constexpr uint32_t float_to_bits(float f) noexcept {
    return std::bit_cast<uint32_t>(f);
}

}  // namespace

// Branch-free IEEE-754 binary16 conversion, after Fabian Giesen's float_to_half
// / half_to_float (public domain) -- the same lineage as ggml's implementation.
//
// The reason to use this rather than the obvious shift-and-mask version is
// subnormal handling. A naive conversion flushes fp16 subnormals (magnitudes
// below ~6.1e-5) to zero. That is invisible in most tests and then shows up as
// a mismatch against PyTorch only for very small activations -- exactly the
// class of bug this project's tolerance gates are meant to catch, and a
// miserable one to trace back to a conversion helper.
float fp16_to_fp32(uint16_t h) noexcept {
    const uint32_t w = static_cast<uint32_t>(h) << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112F;
    const float normalized = bits_to_float((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5F;
    const float denormalized = bits_to_float((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denorm_cutoff = UINT32_C(1) << 27;
    const uint32_t result =
        sign | (two_w < denorm_cutoff ? float_to_bits(denormalized) : float_to_bits(normalized));
    return bits_to_float(result);
}

uint16_t fp32_to_fp16(float f) noexcept {
    const float scale_to_inf = 0x1.0p+112F;
    const float scale_to_zero = 0x1.0p-110F;

    float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = float_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = bits_to_float((bias >> 1) + UINT32_C(0x07800000)) + base;

    const uint32_t bits = float_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;

    return static_cast<uint16_t>((sign >> 16) |
                                 (shl1_w > UINT32_C(0xFF000000) ? UINT32_C(0x7E00) : nonsign));
}

}  // namespace vkml
