#pragma once

#include "vkml/api/tensor.h"

#include <array>
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
/// Complementary error function, 1 - erf(x), computed without forming that
/// difference. Exists because the subtraction is what destroys the result: erf
/// approaches +/-1, so `1 - erf(x)` and `1 + erf(x)` cancel and lose the
/// significand exactly where erfc is still perfectly representable. gelu's
/// value and its gradient are both built on it for that reason.
[[nodiscard]] Tensor erfc(const Tensor& a);
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

/// How a loss collapses its per-sample values, following torch's `reduction`.
enum class Reduction {
    Mean,  ///< average over every element; torch's default
    Sum,
    None,  ///< return the per-sample losses unreduced
};

/// Mean squared error between `input` and `target`, which must broadcast
/// together. Composed from sub/square and a reduction.
[[nodiscard]] Tensor mse_loss(const Tensor& input, const Tensor& target,
                              Reduction reduction = Reduction::Mean);

/// Softmax cross-entropy from raw logits and integer class labels, following
/// torch.nn.functional.cross_entropy. `logits` is (N, C) or (C,); `target` is
/// I64 holding a class index per sample.
///
/// TAKES LOGITS, NOT PROBABILITIES. Passing softmax output would apply the
/// normalisation twice and train against a wrong objective while still
/// producing finite, plausible numbers -- so the distinction is worth stating
/// twice.
///
/// Built on `log_softmax`, which already subtracts the row maximum. The naive
/// `log(softmax(x))` underflows to -inf as soon as the model becomes confident
/// -- softmax of a losing logit reaches 0 in fp32 around a 90-logit gap, and
/// log(0) then poisons every gradient in the batch. The stable form costs
/// nothing and is the only reason this trains at all.
///
/// The label is selected by multiplying against a one-hot mask rather than
/// gathering, so this composes from operators that already exist on both
/// backends. Exactly one term per row survives, which makes the row sum exact
/// rather than merely well-conditioned. The mask is built by comparing class
/// indices in F32, which is exact for any C below 2^24.
[[nodiscard]] Tensor cross_entropy(const Tensor& logits, const Tensor& target,
                                   Reduction reduction = Reduction::Mean);

/// Binary cross-entropy from raw logits, following
/// torch.nn.functional.binary_cross_entropy_with_logits. `target` holds
/// probabilities in [0, 1] -- usually 0 or 1, but soft labels are allowed.
///
/// TAKES LOGITS, NOT PROBABILITIES, for the same reason cross_entropy does.
/// There is deliberately no variant taking probabilities: computing
/// `log(p)` for a confident model underflows, and torch's version of that
/// function exists only with a clamp bolted on to hide it. Apply this to the
/// value you would have passed to sigmoid.
///
/// Evaluated as `max(x, 0) - x*y + log(1 + exp(-|x|))`, which is the standard
/// rearrangement that never evaluates exp on a positive argument: exp(-|x|)
/// lies in (0, 1], so the logarithm's argument stays in (1, 2] whatever the
/// logit magnitude. The naive `-[y log s(x) + (1-y) log(1-s(x))]` loses the
/// losing term to underflow well before x reaches 100.
[[nodiscard]] Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& target,
                                                      Reduction reduction = Reduction::Mean);

/// Kullback-Leibler divergence, following torch.nn.functional.kl_div.
///
/// `input` HOLDS LOG-PROBABILITIES and `target` holds probabilities, which is
/// torch's convention and trips people every time. Pass log_softmax output, not
/// softmax output. With `log_target` true, `target` is log-probabilities too.
///
/// Pointwise value is `target * (log(target) - input)`, defined as 0 wherever
/// `target` is 0 -- the limit of `t log t` as t goes to 0, which the arithmetic
/// would otherwise produce as 0 * -inf = NaN.
///
/// NOTE ON REDUCTION: Mean averages over every element, matching torch's
/// default, which is not the mathematical definition. The KL divergence of a
/// batch is the SUM over classes averaged over samples, so use Sum and divide
/// by the batch size. Reduction has no BatchMean member because it is shared
/// with every other loss here, and a value only one of them honours would be a
/// trap in the other three.
[[nodiscard]] Tensor kl_div(const Tensor& input, const Tensor& target,
                            Reduction reduction = Reduction::Mean, bool log_target = false);

/// Huber loss, following torch.nn.functional.huber_loss.
///
/// Quadratic within `delta` of the target and linear beyond it, so a single
/// outlier contributes a bounded gradient instead of dominating the batch the
/// way squared error lets it. The two pieces meet with matching value and slope
/// at |error| = delta, which is what makes it usable as a training objective.
///
/// Not torch's smooth_l1_loss: that is this divided by delta (`beta` there).
/// Both exist in torch and differ by exactly that factor, which is worth
/// knowing before comparing numbers against a reference implementation.
[[nodiscard]] Tensor huber_loss(const Tensor& input, const Tensor& target,
                                Reduction reduction = Reduction::Mean, double delta = 1.0);

/// Extracts sliding local blocks, following torch.nn.functional.unfold.
///
/// `(N, C, H, W)` becomes `(N, C * kernel_h * kernel_w, L)`, where `L` is the
/// number of window positions. Positions falling outside the padded image
/// contribute zero. This is the first half of convolution-as-GEMM: the second
/// half is an ordinary matmul against the flattened weights.
[[nodiscard]] Tensor im2col(const Tensor& input, std::array<int, 2> kernel,
                            std::array<int, 2> stride = {1, 1}, std::array<int, 2> padding = {0, 0},
                            std::array<int, 2> dilation = {1, 1});

/// Adjoint of `im2col`, following torch.nn.functional.fold: sums every window
/// contribution back into the image position it came from.
///
/// Overlapping windows mean one image position receives several contributions,
/// which is why this cannot be expressed as a gather and is one of the few
/// operations needing its own kernel. `image` gives the spatial extent to
/// reconstruct, which the column tensor does not determine.
[[nodiscard]] Tensor col2im(const Tensor& cols, std::array<int, 2> image, std::array<int, 2> kernel,
                            std::array<int, 2> stride = {1, 1}, std::array<int, 2> padding = {0, 0},
                            std::array<int, 2> dilation = {1, 1});

/// 2D convolution, following torch.nn.functional.conv2d.
///
/// `input` is `(N, C_in, H, W)`, `weight` is `(C_out, C_in, kernel_h,
/// kernel_w)`, and `bias` is `(C_out)` or undefined.
///
/// Implemented as convolution-as-GEMM: `im2col` to lay each window out as a
/// column, then one matmul against the flattened weights. That reuses the tuned
/// matmul instead of needing a direct convolution kernel, and its gradient
/// falls out of the gradients for im2col, matmul and reshape -- there is no
/// conv-specific backward rule anywhere.
///
/// GROUPS ARE NOT SUPPORTED. Grouped and depthwise convolution need either a
/// batched matmul over the group axis or a separate kernel, and nothing in the
/// current roadmap uses them. Passing a weight whose input-channel extent
/// disagrees with the input's is rejected rather than silently misinterpreted.
[[nodiscard]] Tensor conv2d(const Tensor& input, const Tensor& weight,
                            const Tensor& bias = Tensor{}, std::array<int, 2> stride = {1, 1},
                            std::array<int, 2> padding = {0, 0},
                            std::array<int, 2> dilation = {1, 1});

/// Uniform values in [0, 1), from a counter-based generator.
///
/// A PURE FUNCTION of `(seed, offset, element index)`: the same arguments always
/// give the same tensor, on either backend, however the work is divided. There
/// is no hidden global stream to advance, so two calls sharing a seed and an
/// offset produce identical values -- which is what makes a dropout mask
/// reproducible, and a bug if a training loop forgets to advance the offset.
///
/// Deliberately not bit-compatible with PyTorch's generator, and it does not
/// try to be (docs/ARCHITECTURE.md §7.2). Matching another framework's stream
/// would validate nothing about this library; parity is tested
/// distributionally instead.
[[nodiscard]] Tensor rand(std::span<const int64_t> dims, uint64_t seed, uint64_t offset = 0,
                          Device device = Device::cpu());

/// Zeroes each element independently with probability `p` and scales the rest
/// by `1 / (1 - p)`, following torch.nn.functional.dropout.
///
/// The scaling happens at training time -- "inverted dropout" -- so that
/// evaluation is the identity and needs no compensating factor. `training =
/// false` returns the input unchanged, which is why the flag is a parameter
/// rather than something the caller branches on.
///
/// The mask comes from `rand(seed, offset)`, so the caller owns reproducibility:
/// the same seed and offset give the same mask. A training loop must advance the
/// offset between steps, or every step drops the same elements.
[[nodiscard]] Tensor dropout(const Tensor& input, double p, uint64_t seed, uint64_t offset = 0,
                             bool training = true);

/// Applies a batch normalisation given the statistics to use:
/// `(input - mean) / sqrt(variance + eps) * weight + bias`.
///
/// `mean` and `variance` are rank-1 with one entry per channel, where the
/// channel is axis 1. `weight` and `bias` are optional and likewise per-channel.
///
/// TAKES THE STATISTICS; IT DOES NOT CHOOSE OR UPDATE THEM. Whether to use the
/// batch's own statistics or the running estimate is a property of training
/// mode, and updating the running estimate is a mutation across calls -- both
/// belong to the `nn` module, which owns that state, exactly as the optimisers
/// own theirs. Keeping this function pure is what makes it directly comparable
/// against `torch.nn.functional.batch_norm`.
///
/// A CALLER COMPUTING BATCH STATISTICS MUST USE THE BIASED VARIANCE (divide by
/// N, not N-1) here, while updating a running estimate with the UNBIASED one.
/// That is torch's behaviour, verified, and the asymmetry is deliberate on
/// their part: the biased estimate is the right normaliser for the batch in
/// hand, and the unbiased one the right estimator of the population. Using one
/// for both makes evaluation drift away from training as the running estimate
/// converges to the wrong value -- invisible in a single-step comparison.
[[nodiscard]] Tensor batch_norm(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                const Tensor& weight = Tensor{}, const Tensor& bias = Tensor{},
                                double eps = 1e-5);

/// 2D max pooling, following torch.nn.functional.max_pool2d.
///
/// PADS WITH -INFINITY, not zero. A padded window must not report 0 as its
/// maximum when every real element is negative, which is why this needs its own
/// kernel rather than composing as `max` over `im2col` -- im2col pads with zero
/// (verified against torch; `avg_pool2d` composes precisely because zero
/// padding is what *it* wants).
///
/// Its gradient goes to exactly one position per window -- the first maximum in
/// row-major order within the window, matching torch -- not split among ties.
[[nodiscard]] Tensor max_pool2d(const Tensor& input, std::array<int, 2> kernel,
                                std::array<int, 2> stride = {0, 0},
                                std::array<int, 2> padding = {0, 0},
                                std::array<int, 2> dilation = {1, 1});

/// 2D average pooling, following torch.nn.functional.avg_pool2d with
/// `count_include_pad=True`: padded positions contribute zero and are counted
/// in the divisor.
///
/// Composed from im2col and a mean, because zero padding is exactly the
/// semantics wanted here. Dilation is not accepted -- torch's avg_pool2d has no
/// such parameter, and offering one would invent a behaviour to match.
[[nodiscard]] Tensor avg_pool2d(const Tensor& input, std::array<int, 2> kernel,
                                std::array<int, 2> stride = {0, 0},
                                std::array<int, 2> padding = {0, 0});

/// Gathers along `axis` at the positions named by `index`, following
/// torch.index_select. `index` must be I64 and rank 1; the result takes its
/// extent on `axis` from the index length, and every other extent from `a`.
///
/// This is Embedding's forward pass.
[[nodiscard]] Tensor index_select(const Tensor& a, int axis, const Tensor& index);

/// Adjoint of `index_select`: accumulates each slice of `src` into the row of a
/// zero-filled result named by `index`, following torch.index_add.
/// `dim_size` is the extent of the result on `axis`.
///
/// One of the four operations docs/ARCHITECTURE.md records as genuinely needing
/// its own kernel rather than composing from forward ops -- repeated indices
/// mean several source slices land on one destination, which no elementwise or
/// reduction op expresses.
///
/// DETERMINISM. The target GPU has no global float atomicAdd
/// (`shaderBufferFloat32AtomicAdd = false`, measured), so the usual
/// atomic scatter is unavailable -- and unwanted, since its order varies run to
/// run. Both backends instead fold contributions in ascending index order,
/// which makes the result bit-reproducible and identical between them.
[[nodiscard]] Tensor scatter_add(const Tensor& src, int axis, const Tensor& index,
                                 int64_t dim_size);

/// Standardises over the last `normalized_axes` axes: subtract the mean,
/// divide by the standard deviation. No affine term -- `nn.LayerNorm` applies
/// its own weight and bias with `mul`/`add`, which the autograd handles without
/// a special case.
///
/// Composed from mean/sub/square/rsqrt rather than fused. That is the two-pass
/// algorithm, which is the numerically sound one: computing the variance as
/// E[x^2] - E[x]^2 in one pass cancels catastrophically when the mean is large
/// relative to the spread. A fused kernel would save bandwidth, not accuracy,
/// and is deferred until a profile says it matters.
///
/// `normalized_axes` counts trailing axes rather than naming a shape, which is
/// a deliberate divergence from torch's `normalized_shape`. It carries the same
/// information for every legal call and needs no shape validation.
[[nodiscard]] Tensor layer_norm(const Tensor& a, int normalized_axes = 1, double eps = 1e-5);

/// As `layer_norm` but without centring: divides by the root mean square.
/// Used by Llama-family models, where dropping the mean subtraction is most of
/// the speedup and costs nothing measurable in quality.
///
/// DEFAULT eps DIVERGES FROM TORCH, deliberately. `torch.nn.functional.rms_norm`
/// defaults eps to `finfo(dtype).eps` (1.19e-7 for f32), which is small enough
/// to be a no-op except for an exactly-zero input. vkml uses 1e-5, matching
/// `layer_norm` here and what Llama-family implementations actually ship. The
/// two are otherwise identical: pass eps explicitly and the results agree to
/// 7e-7, which is how the parity tests compare them.
[[nodiscard]] Tensor rms_norm(const Tensor& a, int normalized_axes = 1, double eps = 1e-5);

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
