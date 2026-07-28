#pragma once

#include <cstdint>

namespace vkml::cpu {

// Philox4x32-10, the counter-based generator docs/ARCHITECTURE.md §5.4 selects.
//
// Counter-based means the n-th value is computed *from* n rather than by
// advancing a state, which buys three things this project needs:
//
//   - reproducible: the same (seed, offset, index) always gives the same bits,
//     independently of how many values were drawn before or in what order;
//   - parallel: every invocation computes its own value with no shared state
//     and no synchronisation, which is what makes a GPU kernel possible at all;
//   - stateless: nothing to store, migrate between devices, or get out of step
//     between the two backends.
//
// The algorithm is Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
// (SC11). It is a reduced-strength block cipher: ten rounds of multiply-and-xor
// over a 128-bit counter keyed by 64 bits. Ten rounds is the standard choice and
// passes the full TestU01 BigCrush suite; fewer is faster and measurably worse.
//
// This will NOT match PyTorch's RNG bit-for-bit and must not try to
// (docs/ARCHITECTURE.md §7.2). RNG parity is tested distributionally instead.
//
// The GLSL twin lives in shaders/rand.comp and must stay identical, so the two
// backends agree exactly rather than within a tolerance. The constants are
// specified by the paper, so neither side is free to choose them.

/// Round keys, from the paper: the fractional parts of the golden ratio and of
/// sqrt(3), which is the usual "nothing up my sleeve" construction.
inline constexpr uint32_t kPhiloxWeyl0 = 0x9E3779B9U;
inline constexpr uint32_t kPhiloxWeyl1 = 0xBB67AE85U;

/// Multipliers, likewise specified by the paper.
inline constexpr uint32_t kPhiloxMul0 = 0xD2511F53U;
inline constexpr uint32_t kPhiloxMul1 = 0xCD9E8D57U;

inline constexpr int kPhiloxRounds = 10;

struct Philox4x32 {
    uint32_t v[4];
};

/// Ten rounds of Philox4x32 over a 128-bit counter and a 64-bit key.
[[nodiscard]] inline Philox4x32 philox4x32(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                           uint32_t k0, uint32_t k1) noexcept {
    for (int round = 0; round < kPhiloxRounds; ++round) {
        // The 64-bit product's halves are what the round mixes; doing it in
        // uint64_t is exactly how the GLSL side computes it too.
        const uint64_t p0 = static_cast<uint64_t>(kPhiloxMul0) * c0;
        const uint64_t p1 = static_cast<uint64_t>(kPhiloxMul1) * c2;

        const uint32_t hi0 = static_cast<uint32_t>(p0 >> 32U);
        const uint32_t lo0 = static_cast<uint32_t>(p0);
        const uint32_t hi1 = static_cast<uint32_t>(p1 >> 32U);
        const uint32_t lo1 = static_cast<uint32_t>(p1);

        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;

        // Bump the key rather than re-deriving one per round: this is the
        // "Weyl sequence" that keeps successive rounds from sharing structure.
        k0 += kPhiloxWeyl0;
        k1 += kPhiloxWeyl1;
    }
    return Philox4x32{{c0, c1, c2, c3}};
}

/// One uniform value in [0, 1) for element `index` of the draw identified by
/// `(seed, offset)`.
///
/// Takes the top 24 bits, which is the float significand's width: every result
/// is then exactly representable, evenly spaced, and strictly below 1. Scaling
/// the full 32 bits instead would round the largest value up to exactly 1.0,
/// which breaks any caller treating the range as half-open -- dropout with
/// p = 0 would then drop an element.
[[nodiscard]] inline float philox_uniform(uint64_t seed, uint64_t offset, uint64_t index) noexcept {
    const Philox4x32 r =
        philox4x32(static_cast<uint32_t>(index), static_cast<uint32_t>(index >> 32U),
                   static_cast<uint32_t>(offset), static_cast<uint32_t>(offset >> 32U),
                   static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32U));

    constexpr float kScale = 1.0F / 16777216.0F;  // 2^-24
    return static_cast<float>(r.v[0] >> 8U) * kScale;
}

}  // namespace vkml::cpu
