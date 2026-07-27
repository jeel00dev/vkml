#pragma once

#include "vkml/api/tensor.h"

#include <span>

namespace vkml {

// Binary elementwise. Operands are broadcast against each other using NumPy
// rules; the broadcast is expressed as stride-0 views, never as a copy.
[[nodiscard]] Tensor add(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor sub(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor mul(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor div(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor pow(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor maximum(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor minimum(const Tensor& a, const Tensor& b);

// Scalar forms. The scalar adopts the tensor's dtype, so `x + 1` behaves as it
// does in PyTorch without a general type-promotion lattice (M0 scope).
[[nodiscard]] Tensor add(const Tensor& a, double s);
[[nodiscard]] Tensor sub(const Tensor& a, double s);
[[nodiscard]] Tensor mul(const Tensor& a, double s);
[[nodiscard]] Tensor div(const Tensor& a, double s);
[[nodiscard]] Tensor pow(const Tensor& a, double s);

// Comparison. Produces Bool.
[[nodiscard]] Tensor equal(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor less(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor greater(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor less_equal(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor greater_equal(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor not_equal(const Tensor& a, const Tensor& b);

// Unary elementwise.
[[nodiscard]] Tensor neg(const Tensor& a);
[[nodiscard]] Tensor abs(const Tensor& a);
[[nodiscard]] Tensor sign(const Tensor& a);
[[nodiscard]] Tensor square(const Tensor& a);
[[nodiscard]] Tensor sqrt(const Tensor& a);
[[nodiscard]] Tensor rsqrt(const Tensor& a);
[[nodiscard]] Tensor reciprocal(const Tensor& a);
[[nodiscard]] Tensor exp(const Tensor& a);
[[nodiscard]] Tensor log(const Tensor& a);
[[nodiscard]] Tensor erf(const Tensor& a);
[[nodiscard]] Tensor sin(const Tensor& a);
[[nodiscard]] Tensor cos(const Tensor& a);
[[nodiscard]] Tensor tanh(const Tensor& a);
[[nodiscard]] Tensor sigmoid(const Tensor& a);
[[nodiscard]] Tensor relu(const Tensor& a);
[[nodiscard]] Tensor gelu(const Tensor& a);
[[nodiscard]] Tensor silu(const Tensor& a);
[[nodiscard]] Tensor clamp(const Tensor& a, double lo, double hi);
[[nodiscard]] Tensor clamp_min(const Tensor& a, double lo);
[[nodiscard]] Tensor clamp_max(const Tensor& a, double hi);

// Shape/dtype/device-matching constants. Used heavily by backward rules, where
// a mask or a unit gradient has to match an existing tensor exactly.
[[nodiscard]] Tensor zeros_like(const Tensor& a);
[[nodiscard]] Tensor ones_like(const Tensor& a);
[[nodiscard]] Tensor full_like(const Tensor& a, double value);

/// Elementwise select. `cond` must be Bool; all three broadcast together.
[[nodiscard]] Tensor where(const Tensor& cond, const Tensor& a, const Tensor& b);

/// Joins tensors along `axis`. Every input must share the same rank, dtype,
/// device and extents on every axis except `axis`.
///
/// The graph node is binary, so a list of N is folded left into N-1 nodes.
/// That copies O(N^2) bytes for equal-sized inputs where O(N) is achievable
/// with an n-ary node. Deferred deliberately: N is 2 for every use in scope
/// (residual joins, UNet skips), and an n-ary node needs per-source extents in
/// push constants for a case that does not yet exist.
[[nodiscard]] Tensor cat(std::span<const Tensor> tensors, int axis = 0);

/// Two-input form, which is what the graph node actually is.
[[nodiscard]] Tensor cat(const Tensor& a, const Tensor& b, int axis = 0);

/// Replaces every element where `mask` is true with `value`, following
/// torch.masked_fill. `mask` must be Bool and broadcastable to `a`'s shape.
///
/// Composed from `where` rather than given its own kernel: the replacement is a
/// rank-0 tensor broadcast with stride 0, so the "fused" version would save one
/// graph node and four bytes. See the commit that added it.
[[nodiscard]] Tensor masked_fill(const Tensor& a, const Tensor& mask, double value);

/// Zeroes everything below the `diagonal`-th diagonal of the last two axes,
/// keeping the upper triangle. `diagonal = 0` keeps the main diagonal;
/// positive values move the boundary toward the upper-right. Rank must be at
/// least 2; leading axes are batched over.
[[nodiscard]] Tensor triu(const Tensor& a, int64_t diagonal = 0);

/// Mirror of `triu`, keeping the lower triangle.
[[nodiscard]] Tensor tril(const Tensor& a, int64_t diagonal = 0);

// Reductions. An empty `axes` reduces over every axis.
[[nodiscard]] Tensor sum(const Tensor& a, std::span<const int> axes = {}, bool keepdim = false);
[[nodiscard]] Tensor mean(const Tensor& a, std::span<const int> axes = {}, bool keepdim = false);
[[nodiscard]] Tensor prod(const Tensor& a, std::span<const int> axes = {}, bool keepdim = false);
[[nodiscard]] Tensor max(const Tensor& a, std::span<const int> axes = {}, bool keepdim = false);
[[nodiscard]] Tensor min(const Tensor& a, std::span<const int> axes = {}, bool keepdim = false);
[[nodiscard]] Tensor argmax(const Tensor& a, int axis, bool keepdim = false);
[[nodiscard]] Tensor argmin(const Tensor& a, int axis, bool keepdim = false);

// Braced-list overloads for axis lists, for the same reason as in tensor.h.
[[nodiscard]] inline Tensor sum(const Tensor& a, std::initializer_list<int> axes,
                                bool keepdim = false) {
    return sum(a, std::span<const int>{axes.begin(), axes.size()}, keepdim);
}

[[nodiscard]] inline Tensor mean(const Tensor& a, std::initializer_list<int> axes,
                                 bool keepdim = false) {
    return mean(a, std::span<const int>{axes.begin(), axes.size()}, keepdim);
}

[[nodiscard]] inline Tensor prod(const Tensor& a, std::initializer_list<int> axes,
                                 bool keepdim = false) {
    return prod(a, std::span<const int>{axes.begin(), axes.size()}, keepdim);
}

[[nodiscard]] inline Tensor max(const Tensor& a, std::initializer_list<int> axes,
                                bool keepdim = false) {
    return max(a, std::span<const int>{axes.begin(), axes.size()}, keepdim);
}

[[nodiscard]] inline Tensor min(const Tensor& a, std::initializer_list<int> axes,
                                bool keepdim = false) {
    return min(a, std::span<const int>{axes.begin(), axes.size()}, keepdim);
}

[[nodiscard]] Tensor softmax(const Tensor& a, int axis = -1);
[[nodiscard]] Tensor log_softmax(const Tensor& a, int axis = -1);

/// Matrix multiply, following torch.matmul for ranks 1-4: leading axes are
/// treated as batch and broadcast against each other.
[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b);

// Operator sugar.
[[nodiscard]] inline Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }

[[nodiscard]] inline Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }

[[nodiscard]] inline Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }

[[nodiscard]] inline Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }

[[nodiscard]] inline Tensor operator+(const Tensor& a, double s) { return add(a, s); }

[[nodiscard]] inline Tensor operator-(const Tensor& a, double s) { return sub(a, s); }

[[nodiscard]] inline Tensor operator*(const Tensor& a, double s) { return mul(a, s); }

[[nodiscard]] inline Tensor operator/(const Tensor& a, double s) { return div(a, s); }

[[nodiscard]] inline Tensor operator-(const Tensor& a) { return neg(a); }

}  // namespace vkml
