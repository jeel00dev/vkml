#pragma once

#include "vkml/graph/node.h"
#include "vkml/util/assert.h"

#include <cstddef>
#include <cstring>

// Strided iteration helpers shared by every CPU kernel.
//
// DESIGN: mirror the GPU, then add a fast path
// --------------------------------------------
// The general form deliberately mirrors what a Vulkan compute kernel does:
// take a flat invocation index, decompose it into per-axis indices, multiply
// by that operand's strides to get a byte offset. Writing the reference
// implementation the same way means that when a Vulkan kernel later disagrees
// with the CPU oracle, the difference is a real bug rather than a difference
// in how the two traverse memory.
//
// That general form costs O(rank) integer ops per element, including a
// division per axis, which would make MNIST training unbearably slow. So every
// kernel first checks whether all operands are contiguous and takes a tight
// linear loop when they are. Real GPU kernel libraries do exactly the same
// thing -- llama.cpp compiles `_aligned` shader variants with the bounds and
// stride arithmetic removed, for the same reason.

namespace vkml::cpu {

/// Byte offset of the `linear`-th element in row-major logical order.
///
/// Handles broadcast views for free: a stride of 0 simply contributes nothing,
/// so every index along that axis reads the same element.
[[nodiscard]] inline int64_t linear_to_offset(int64_t linear, const Shape& s) noexcept {
    int64_t off = 0;
    for (int i = s.ndim() - 1; i >= 0; --i) {
        const int64_t extent = s.dim(i);
        const int64_t idx = linear % extent;
        linear /= extent;
        off += idx * s.stride(i);
    }
    return off;
}

/// Decomposes a flat index into per-axis indices, row-major.
inline void unravel(int64_t linear, const Shape& s, std::array<int64_t, kMaxDims>& idx) noexcept {
    for (int i = s.ndim() - 1; i >= 0; --i) {
        const int64_t extent = s.dim(i);
        idx[static_cast<size_t>(i)] = linear % extent;
        linear /= extent;
    }
}

template <typename T>
[[nodiscard]] inline T* base_ptr(Node& n) noexcept {
    return static_cast<T*>(n.data());
}

template <typename T>
[[nodiscard]] inline const T* base_ptr(const Node& n) noexcept {
    return static_cast<const T*>(n.data());
}

/// Widens a stored element to the type arithmetic is done in.
///
/// Every kernel computes in float whatever it stores, so that f16 is a storage
/// format and never an accumulator (`ARCHITECTURE.md` §7.3, and the note on
/// `Half` in dtype.h). These two overloads are the only place that conversion
/// lives, which is what keeps the f32 and f16 paths from drifting apart.
[[nodiscard]] inline float widen(float v) noexcept { return v; }

[[nodiscard]] inline float widen(Half v) noexcept { return v.to_float(); }

/// Reads element `linear` of a possibly-strided operand.
template <typename T>
[[nodiscard]] inline T load(const Node& n, int64_t linear) noexcept {
    const auto* bytes = static_cast<const std::byte*>(n.data());
    T value{};
    std::memcpy(&value, bytes + linear_to_offset(linear, n.shape), sizeof(T));
    return value;
}

/// Writes element `linear` of a possibly-strided operand.
template <typename T>
inline void store(Node& n, int64_t linear, T value) noexcept {
    auto* bytes = static_cast<std::byte*>(n.data());
    std::memcpy(bytes + linear_to_offset(linear, n.shape), &value, sizeof(T));
}

/// True when a whole-tensor walk can use a flat linear loop.
[[nodiscard]] inline bool all_contiguous(const Node& a) noexcept { return a.shape.is_contiguous(); }

template <typename... Rest>
[[nodiscard]] inline bool all_contiguous(const Node& a, const Rest&... rest) noexcept {
    return a.shape.is_contiguous() && all_contiguous(rest...);
}

/// Applies `f(a) -> out`, elementwise.
template <typename Out, typename In, typename F>
void map_unary(Node& out, const Node& a, F&& f) {
    const int64_t n = out.shape.numel();
    if (n == 0) {
        return;
    }
    VKML_DEBUG_ASSERT(a.shape.numel() == n, "unary operand count mismatch");

    if (all_contiguous(out, a)) {
        Out* o = base_ptr<Out>(out);
        const In* x = base_ptr<In>(a);
        for (int64_t i = 0; i < n; ++i) {
            o[i] = f(x[i]);
        }
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        store<Out>(out, i, f(load<In>(a, i)));
    }
}

/// Applies `f(a, b) -> out`, elementwise.
///
/// Broadcasting is *not* handled here. It is resolved during graph
/// construction by inserting Broadcast view nodes with stride-0 axes, so by the
/// time a kernel runs all three operands have identical extents and only their
/// strides differ. Keeping broadcast out of the kernels is what stops every
/// kernel from re-implementing the same rules -- and it is why the fast path
/// above is a plain `o[i] = f(x[i], y[i])`.
template <typename Out, typename In, typename F>
void map_binary(Node& out, const Node& a, const Node& b, F&& f) {
    const int64_t n = out.shape.numel();
    if (n == 0) {
        return;
    }
    VKML_DEBUG_ASSERT(a.shape.same_dims(out.shape) && b.shape.same_dims(out.shape),
                      "binary operands must already be broadcast to the output shape");

    if (all_contiguous(out, a, b)) {
        Out* o = base_ptr<Out>(out);
        const In* x = base_ptr<In>(a);
        const In* y = base_ptr<In>(b);
        for (int64_t i = 0; i < n; ++i) {
            o[i] = f(x[i], y[i]);
        }
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        store<Out>(out, i, f(load<In>(a, i), load<In>(b, i)));
    }
}

}  // namespace vkml::cpu
