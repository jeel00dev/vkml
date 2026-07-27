#pragma once

#include "vkml/graph/node.h"

#include <cstdint>
#include <vector>

// Pairwise (tree) summation.
//
// WHY THIS IS NOT A NAIVE LOOP
// ----------------------------
// Sequential summation of n values in fp32 has a worst-case relative error
// bound of n·ε, with ε = 2^-23 ≈ 1.19e-7. The project's correctness gate is
// atol = rtol = 1e-5 against PyTorch (docs/ARCHITECTURE.md §7.3). So:
//
//     n =   784 (MNIST input features)   sequential ≈ 9.3e-5   FAILS by ~9x
//     n =  4096 (transformer hidden)     sequential ≈ 4.9e-4   FAILS by ~49x
//
// A naive accumulator does not merely lose a little precision here; it misses
// the acceptance criterion outright, and it would do so in a way that looks
// like a kernel bug. Pairwise summation splits the range recursively, giving a
// bound of roughly (B + log2(n/B))·ε for a sequential base case of size B:
//
//     B = 32, n =   784   ≈ 4.3e-6    2.3x margin
//     B = 32, n =  4096   ≈ 4.6e-6    2.2x margin
//     B = 32, n = 16384   ≈ 4.9e-6    2.0x margin
//
// B = 32 is chosen to keep at least 2x margin against the tolerance out to
// n = 16384, which covers every reduction length these models produce. Larger
// B (numpy uses 128) would still pass in practice, because real rounding errors
// random-walk rather than aligning, but designing to the worst-case bound costs
// nothing measurable and removes a whole class of "why is this test flaky at
// large K" investigation later.
//
// This is a deliberate correctness-over-speed choice, per the project's
// engineering standard: the recursion is slower than a flat loop and that is
// accepted.
//
// It also mirrors what the Vulkan backend will have to do. That GPU has no
// global float atomicAdd (measured, docs/ARCHITECTURE.md §1.1), so its
// reductions must be tree-shaped too -- subgroup reduction, then shared memory,
// then a deterministic second pass. Matching the CPU reference to that
// structure keeps the two comparable.

namespace vkml::cpu {

/// Sequential base case size. See the error analysis above.
inline constexpr int64_t kPairwiseBlock = 32;

/// Sums `load(i)` for i in [begin, end) using pairwise summation.
///
/// `load` may read from anywhere -- strided memory, a transformed value -- so
/// the same routine serves plain sums, sums of squares, and dot products
/// without materialising a temporary.
template <typename T, typename Load>
[[nodiscard]] T pairwise_sum(const Load& load, int64_t begin, int64_t end) {
    const int64_t n = end - begin;
    if (n <= 0) {
        return T{0};
    }
    if (n <= kPairwiseBlock) {
        T acc{0};
        for (int64_t i = begin; i < end; ++i) {
            acc += load(i);
        }
        return acc;
    }
    // Split at the midpoint rather than at a block boundary: an even split
    // keeps the tree balanced, which is what bounds the depth at log2(n/B).
    const int64_t mid = begin + n / 2;
    return pairwise_sum<T>(load, begin, mid) + pairwise_sum<T>(load, mid, end);
}

/// Decomposition of a reduction into the axes that survive and the axes that
/// collapse.
///
/// Both are expressed as Shapes carrying the *input's* strides, so iterating
/// `kept` walks output positions and iterating `reduced` walks the elements
/// folded into one output position. That makes every reduction kernel the same
/// two nested loops, with the numerics living entirely in `pairwise_sum`.
struct ReducePlan {
    Shape kept;     ///< extents and input strides of the surviving axes
    Shape reduced;  ///< extents and input strides of the collapsed axes
};

/// Builds a ReducePlan for `in` given a bitmask of axes to reduce.
[[nodiscard]] ReducePlan make_reduce_plan(const Shape& in, uint32_t axes_mask);

/// Output extents for a reduction, honouring keepdim.
[[nodiscard]] std::vector<int64_t> reduced_dims(const Shape& in, uint32_t axes_mask, bool keepdim);

/// Bitmask covering every axis, used when no axes are given (reduce-all).
[[nodiscard]] uint32_t all_axes_mask(int ndim) noexcept;

}  // namespace vkml::cpu
