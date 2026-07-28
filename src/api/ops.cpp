#include "vkml/api/ops.h"

#include "vkml/dispatch/executor.h"
#include "vkml/graph/grad_mode.h"
#include "vkml/graph/node.h"
#include "vkml/util/assert.h"

#include <algorithm>
#include <format>
#include <string_view>

namespace vkml {
namespace {

/// Overloads taking a name exist for operators that are *compositions* and so
/// have no OpKind of their own -- cross_entropy, mse_loss. Without them such an
/// operator would have to borrow an unrelated enumerator purely to phrase its
/// error, and the message would name the wrong thing.
void check_same_device(const Tensor& a, const Tensor& b, std::string_view op) {
    VKML_CHECK(a.device() == b.device(), DeviceError,
               "'{}' operands are on different devices: {} and {}", op, a.device().str(),
               b.device().str());
}

void check_same_device(const Tensor& a, const Tensor& b, OpKind op) {
    check_same_device(a, b, op_name(op));
}

/// M0 policy: no implicit type promotion between tensors.
///
/// PyTorch has a full promotion lattice (f32 + i64 -> f32). Reproducing it is a
/// meaningful amount of machinery and a rich source of silent surprises, and it
/// is not needed to train anything. Requiring a matching dtype is stricter than
/// PyTorch and therefore never *wrong* -- a program that works here works there.
/// Python scalars are still converted to the tensor's dtype, so `x + 1` behaves
/// as expected. Documented as a deliberate divergence.
void check_same_dtype(const Tensor& a, const Tensor& b, std::string_view op) {
    VKML_CHECK(a.dtype() == b.dtype(), DTypeError,
               "'{}' operands have different dtypes ({} and {}); vkml does not promote "
               "implicitly -- cast one explicitly with .to()",
               op, dtype_name(a.dtype()), dtype_name(b.dtype()));
}

void check_same_dtype(const Tensor& a, const Tensor& b, OpKind op) {
    check_same_dtype(a, b, op_name(op));
}

Tensor finish(NodePtr n) {
    Tensor t{std::move(n)};
    if (eager()) {
        t.realize();
    }
    return t;
}

/// Broadcasts both operands to their common shape and builds a binary node.
Tensor binary(OpKind op, const Tensor& a, const Tensor& b, DType out_dtype) {
    check_same_device(a, b, op);
    check_same_dtype(a, b, op);

    const std::vector<int64_t> dims = broadcast_dims(a.shape(), b.shape());

    // Broadcasting materialises nothing: each operand becomes a view with
    // stride 0 on the stretched axes. Kernels then see three operands of
    // identical extents and never implement broadcast rules themselves.
    const Tensor ba = a.shape() == dims ? a : a.broadcast_to(dims);
    const Tensor bb = b.shape() == dims ? b : b.broadcast_to(dims);

    auto n = make_node(op, Shape::contiguous(dims, dtype_size(out_dtype)), out_dtype, a.device());
    n->src[0] = ba.node();
    n->src[1] = bb.node();
    n->n_src = 2;
    n->requires_grad =
        grad_enabled() && (a.requires_grad() || b.requires_grad()) && is_differentiable(out_dtype);
    return finish(std::move(n));
}

Tensor unary(OpKind op, const Tensor& a) {
    auto n =
        make_node(op, Shape::contiguous(a.shape(), dtype_size(a.dtype())), a.dtype(), a.device());
    n->src[0] = a.node();
    n->n_src = 1;
    n->requires_grad = grad_enabled() && a.requires_grad();
    return finish(std::move(n));
}

/// Wraps a scalar as a rank-0 tensor of the peer's dtype and device.
Tensor scalar_like(const Tensor& peer, double value) {
    return Tensor::full({}, value, peer.dtype(), peer.device());
}

uint32_t axes_to_mask(std::span<const int> axes, int ndim) {
    if (axes.empty()) {
        return ndim <= 0 ? 0U : (1U << static_cast<uint32_t>(ndim)) - 1U;
    }
    uint32_t mask = 0;
    for (const int axis : axes) {
        const int a = normalize_dim(axis, ndim);
        const uint32_t bit = 1U << static_cast<uint32_t>(a);
        VKML_CHECK((mask & bit) == 0, ShapeError, "axis {} given more than once", a);
        mask |= bit;
    }
    return mask;
}

std::vector<int64_t> reduce_out_dims(const Tensor& a, uint32_t mask, bool keepdim) {
    std::vector<int64_t> out;
    for (int i = 0; i < a.ndim(); ++i) {
        const bool reduced = (mask & (1U << static_cast<uint32_t>(i))) != 0;
        if (!reduced) {
            out.push_back(a.size(i));
        } else if (keepdim) {
            out.push_back(1);
        }
    }
    return out;
}

Tensor reduction(OpKind op, const Tensor& a, std::span<const int> axes, bool keepdim,
                 DType out_dtype) {
    const uint32_t mask = axes_to_mask(axes, a.ndim());
    const std::vector<int64_t> dims = reduce_out_dims(a, mask, keepdim);

    auto n = make_node(op, Shape::contiguous(dims, dtype_size(out_dtype)), out_dtype, a.device());
    n->src[0] = a.node();
    n->n_src = 1;
    n->params.set(ReduceParams{.axes_mask = mask, .keepdim = keepdim});
    n->requires_grad = grad_enabled() && a.requires_grad() && is_differentiable(out_dtype);
    return finish(std::move(n));
}

}  // namespace

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b) { return binary(OpKind::Add, a, b, a.dtype()); }

Tensor sub(const Tensor& a, const Tensor& b) { return binary(OpKind::Sub, a, b, a.dtype()); }

Tensor mul(const Tensor& a, const Tensor& b) { return binary(OpKind::Mul, a, b, a.dtype()); }

Tensor div(const Tensor& a, const Tensor& b) { return binary(OpKind::Div, a, b, a.dtype()); }

Tensor pow(const Tensor& a, const Tensor& b) { return binary(OpKind::Pow, a, b, a.dtype()); }

Tensor maximum(const Tensor& a, const Tensor& b) {
    return binary(OpKind::Maximum, a, b, a.dtype());
}

Tensor minimum(const Tensor& a, const Tensor& b) {
    return binary(OpKind::Minimum, a, b, a.dtype());
}

Tensor add(const Tensor& a, double s) { return add(a, scalar_like(a, s)); }

Tensor sub(const Tensor& a, double s) { return sub(a, scalar_like(a, s)); }

Tensor mul(const Tensor& a, double s) { return mul(a, scalar_like(a, s)); }

Tensor div(const Tensor& a, double s) { return div(a, scalar_like(a, s)); }

Tensor pow(const Tensor& a, double s) { return pow(a, scalar_like(a, s)); }

Tensor equal(const Tensor& a, const Tensor& b) { return binary(OpKind::Equal, a, b, DType::Bool); }

Tensor less(const Tensor& a, const Tensor& b) { return binary(OpKind::Less, a, b, DType::Bool); }

Tensor greater(const Tensor& a, const Tensor& b) {
    return binary(OpKind::Greater, a, b, DType::Bool);
}

Tensor less_equal(const Tensor& a, const Tensor& b) {
    return binary(OpKind::LessEqual, a, b, DType::Bool);
}

Tensor greater_equal(const Tensor& a, const Tensor& b) {
    return binary(OpKind::GreaterEqual, a, b, DType::Bool);
}

Tensor not_equal(const Tensor& a, const Tensor& b) {
    return binary(OpKind::NotEqual, a, b, DType::Bool);
}

Tensor neg(const Tensor& a) { return unary(OpKind::Neg, a); }

Tensor abs(const Tensor& a) { return unary(OpKind::Abs, a); }

Tensor sign(const Tensor& a) { return unary(OpKind::Sign, a); }

Tensor square(const Tensor& a) { return unary(OpKind::Square, a); }

Tensor sqrt(const Tensor& a) { return unary(OpKind::Sqrt, a); }

Tensor rsqrt(const Tensor& a) { return unary(OpKind::Rsqrt, a); }

Tensor reciprocal(const Tensor& a) { return unary(OpKind::Reciprocal, a); }

Tensor exp(const Tensor& a) { return unary(OpKind::Exp, a); }

Tensor log(const Tensor& a) { return unary(OpKind::Log, a); }

Tensor erf(const Tensor& a) { return unary(OpKind::Erf, a); }

Tensor sin(const Tensor& a) { return unary(OpKind::Sin, a); }

Tensor cos(const Tensor& a) { return unary(OpKind::Cos, a); }

Tensor tanh(const Tensor& a) { return unary(OpKind::Tanh, a); }

Tensor sigmoid(const Tensor& a) { return unary(OpKind::Sigmoid, a); }

Tensor relu(const Tensor& a) { return unary(OpKind::Relu, a); }

Tensor gelu(const Tensor& a) { return unary(OpKind::Gelu, a); }

Tensor silu(const Tensor& a) { return unary(OpKind::Silu, a); }

namespace {

Tensor clamp_impl(const Tensor& a, ClampParams p) {
    auto n = make_node(OpKind::Clamp, Shape::contiguous(a.shape(), dtype_size(a.dtype())),
                       a.dtype(), a.device());
    n->src[0] = a.node();
    n->n_src = 1;
    n->params.set(p);
    n->requires_grad = grad_enabled() && a.requires_grad();
    return finish(std::move(n));
}

}  // namespace

Tensor clamp(const Tensor& a, double lo, double hi) {
    VKML_CHECK(lo <= hi, ShapeError, "clamp bounds are inverted: lo={} hi={}", lo, hi);
    return clamp_impl(a, ClampParams{.lo = static_cast<float>(lo),
                                     .hi = static_cast<float>(hi),
                                     .has_lo = true,
                                     .has_hi = true});
}

Tensor clamp_min(const Tensor& a, double lo) {
    return clamp_impl(a, ClampParams{.lo = static_cast<float>(lo), .has_lo = true});
}

Tensor clamp_max(const Tensor& a, double hi) {
    return clamp_impl(a, ClampParams{.hi = static_cast<float>(hi), .has_hi = true});
}

Tensor zeros_like(const Tensor& a) { return Tensor::zeros(a.shape(), a.dtype(), a.device()); }

Tensor ones_like(const Tensor& a) { return Tensor::ones(a.shape(), a.dtype(), a.device()); }

Tensor full_like(const Tensor& a, double value) {
    return Tensor::full(a.shape(), value, a.dtype(), a.device());
}

namespace {

Tensor tri_impl(const Tensor& a, int64_t diagonal, OpKind kind) {
    VKML_CHECK(a.shape().size() >= 2, ShapeError,
               "{}() needs a rank-2 or higher tensor, got rank {}", op_name(kind),
               a.shape().size());
    // The offset is stored as int32 to keep TriParams small; a diagonal beyond
    // that range could not index any tensor this backend can allocate, so
    // rejecting it here is a clearer failure than a silent narrowing.
    VKML_CHECK(diagonal >= std::numeric_limits<int32_t>::min() &&
                   diagonal <= std::numeric_limits<int32_t>::max(),
               IndexError, "{}() diagonal {} is out of range", op_name(kind), diagonal);

    auto n =
        make_node(kind, Shape::contiguous(a.shape(), dtype_size(a.dtype())), a.dtype(), a.device());
    n->src[0] = a.node();
    n->n_src = 1;
    n->params.set(TriParams{.diagonal = static_cast<int32_t>(diagonal)});
    n->requires_grad = grad_enabled() && a.requires_grad();
    return finish(std::move(n));
}

}  // namespace

namespace {

/// The trailing `count` axes, as the reduction API wants them.
std::vector<int> trailing_axes(const Tensor& a, int count, const char* what) {
    VKML_CHECK(count >= 1 && count <= a.ndim(), ShapeError,
               "{}() normalizes over {} trailing axes but the tensor has rank {}", what, count,
               a.ndim());
    std::vector<int> axes;
    axes.reserve(static_cast<size_t>(count));
    for (int i = a.ndim() - count; i < a.ndim(); ++i) {
        axes.push_back(i);
    }
    return axes;
}

}  // namespace

namespace {

Tensor apply_reduction(const Tensor& per_sample, Reduction r) {
    switch (r) {
        case Reduction::Mean: return mean(per_sample);
        case Reduction::Sum: return sum(per_sample);
        case Reduction::None: return per_sample;
    }
    VKML_UNREACHABLE("unhandled Reduction");
}

/// One-hot mask over `classes`, as F32, from I64 labels.
///
/// Comparison kernels are F32-only on both backends, so the labels are cast
/// rather than compared as integers. That is exact: every integer up to 2^24 is
/// representable in F32, and a class count anywhere near that is far beyond
/// what this library targets.
Tensor one_hot_f32(const Tensor& target, int64_t classes) {
    const Tensor indices =
        Tensor::arange(0.0, static_cast<double>(classes), 1.0, DType::F32, target.device());
    // (N, 1) against (C,) broadcasts to (N, C) with stride 0 on both stretched
    // axes, so the mask is never materialised at full size by the comparison
    // itself -- only its Bool result is.
    return equal(target.to(DType::F32).unsqueeze(-1), indices).to(DType::F32);
}

}  // namespace

Tensor mse_loss(const Tensor& input, const Tensor& target, Reduction reduction) {
    check_same_dtype(input, target, "mse_loss");
    check_same_device(input, target, "mse_loss");
    return apply_reduction(square(sub(input, target)), reduction);
}

Tensor cross_entropy(const Tensor& logits, const Tensor& target, Reduction reduction) {
    VKML_CHECK(target.dtype() == DType::I64, DTypeError,
               "cross_entropy() target must hold I64 class indices, got {}",
               dtype_name(target.dtype()));
    VKML_CHECK(logits.ndim() == 1 || logits.ndim() == 2, ShapeError,
               "cross_entropy() expects logits of rank 1 or 2, got rank {}", logits.ndim());
    check_same_device(logits, target, "cross_entropy");

    // Rank 1 is a single unbatched sample; lift it so one path handles both.
    const Tensor batched = logits.ndim() == 1 ? logits.unsqueeze(0) : logits;
    const Tensor labels = logits.ndim() == 1 ? target.unsqueeze(0) : target;

    VKML_CHECK(labels.ndim() == 1, ShapeError, "cross_entropy() target must be rank 1, got rank {}",
               labels.ndim());
    VKML_CHECK(labels.numel() == batched.size(0), ShapeError,
               "cross_entropy() has {} logit rows but {} labels", batched.size(0), labels.numel());

    const int64_t classes = batched.size(1);
    // Exactly one term per row is non-zero, so summing the masked row recovers
    // that term exactly -- adding zeros is exact in IEEE-754. The row sum is
    // therefore not a source of error, which is why the tolerance for this is
    // inherited from log_softmax rather than widened for a reduction over C.
    const Tensor picked = sum(mul(log_softmax(batched, -1), one_hot_f32(labels, classes)), {-1});
    return apply_reduction(neg(picked), reduction);
}

namespace {

/// Window positions along one axis, following the standard convolution formula.
/// Returns 0 when the padded image is smaller than the dilated kernel, which
/// the callers reject rather than silently producing an empty result.
[[nodiscard]] int64_t window_count(int64_t extent, int kernel, int stride, int pad, int dilation) {
    const int64_t effective = static_cast<int64_t>(dilation) * (kernel - 1) + 1;
    const int64_t span = extent + 2LL * pad - effective;
    return span < 0 ? 0 : span / stride + 1;
}

void check_window(std::array<int, 2> kernel, std::array<int, 2> stride, std::array<int, 2> padding,
                  std::array<int, 2> dilation, const char* what) {
    for (int i = 0; i < 2; ++i) {
        VKML_CHECK(kernel[i] >= 1, ShapeError, "{}: kernel must be positive, got {}", what,
                   kernel[i]);
        VKML_CHECK(stride[i] >= 1, ShapeError, "{}: stride must be positive, got {}", what,
                   stride[i]);
        VKML_CHECK(dilation[i] >= 1, ShapeError, "{}: dilation must be positive, got {}", what,
                   dilation[i]);
        VKML_CHECK(padding[i] >= 0, ShapeError, "{}: padding must not be negative, got {}", what,
                   padding[i]);
    }
}

UnfoldParams make_unfold_params(std::array<int, 2> kernel, std::array<int, 2> stride,
                                std::array<int, 2> padding, std::array<int, 2> dilation,
                                int64_t image_h, int64_t image_w) {
    return UnfoldParams{.kernel_h = kernel[0],
                        .kernel_w = kernel[1],
                        .stride_h = stride[0],
                        .stride_w = stride[1],
                        .pad_h = padding[0],
                        .pad_w = padding[1],
                        .dilation_h = dilation[0],
                        .dilation_w = dilation[1],
                        .image_h = static_cast<int32_t>(image_h),
                        .image_w = static_cast<int32_t>(image_w)};
}

}  // namespace

Tensor im2col(const Tensor& input, std::array<int, 2> kernel, std::array<int, 2> stride,
              std::array<int, 2> padding, std::array<int, 2> dilation) {
    VKML_CHECK(input.ndim() == 4, ShapeError, "im2col() expects (N, C, H, W), got rank {}",
               input.ndim());
    check_window(kernel, stride, padding, dilation, "im2col()");

    const int64_t out_h =
        window_count(input.size(2), kernel[0], stride[0], padding[0], dilation[0]);
    const int64_t out_w =
        window_count(input.size(3), kernel[1], stride[1], padding[1], dilation[1]);
    VKML_CHECK(out_h > 0 && out_w > 0, ShapeError,
               "im2col(): the dilated kernel is larger than the padded image "
               "({}x{} against {}x{})",
               dilation[0] * (kernel[0] - 1) + 1, dilation[1] * (kernel[1] - 1) + 1,
               input.size(2) + 2LL * padding[0], input.size(3) + 2LL * padding[1]);

    const std::vector<int64_t> dims{input.size(0), input.size(1) * kernel[0] * kernel[1],
                                    out_h * out_w};

    auto n = make_node(OpKind::Im2Col, Shape::contiguous(dims, dtype_size(input.dtype())),
                       input.dtype(), input.device());
    n->src[0] = input.node();
    n->n_src = 1;
    n->params.set(
        make_unfold_params(kernel, stride, padding, dilation, input.size(2), input.size(3)));
    n->requires_grad = grad_enabled() && input.requires_grad();
    return finish(std::move(n));
}

Tensor col2im(const Tensor& cols, std::array<int, 2> image, std::array<int, 2> kernel,
              std::array<int, 2> stride, std::array<int, 2> padding, std::array<int, 2> dilation) {
    VKML_CHECK(cols.ndim() == 3, ShapeError, "col2im() expects (N, C*kh*kw, L), got rank {}",
               cols.ndim());
    check_window(kernel, stride, padding, dilation, "col2im()");

    const int64_t patch = static_cast<int64_t>(kernel[0]) * kernel[1];
    VKML_CHECK(cols.size(1) % patch == 0, ShapeError,
               "col2im(): channel-patch extent {} is not a multiple of the kernel size {}",
               cols.size(1), patch);

    const int64_t out_h = window_count(image[0], kernel[0], stride[0], padding[0], dilation[0]);
    const int64_t out_w = window_count(image[1], kernel[1], stride[1], padding[1], dilation[1]);
    VKML_CHECK(out_h * out_w == cols.size(2), ShapeError,
               "col2im(): a {}x{} image with this geometry has {} window positions, but the "
               "column tensor has {}",
               image[0], image[1], out_h * out_w, cols.size(2));

    const std::vector<int64_t> dims{cols.size(0), cols.size(1) / patch, image[0], image[1]};

    auto n = make_node(OpKind::Col2Im, Shape::contiguous(dims, dtype_size(cols.dtype())),
                       cols.dtype(), cols.device());
    n->src[0] = cols.node();
    n->n_src = 1;
    n->params.set(make_unfold_params(kernel, stride, padding, dilation, image[0], image[1]));
    n->requires_grad = grad_enabled() && cols.requires_grad();
    return finish(std::move(n));
}

namespace {

/// torch's convention: a stride of 0 means "same as the kernel", which is what
/// makes `max_pool2d(x, 2)` halve the spatial extent as everyone expects.
std::array<int, 2> pool_stride(std::array<int, 2> stride, std::array<int, 2> kernel) {
    return {stride[0] == 0 ? kernel[0] : stride[0], stride[1] == 0 ? kernel[1] : stride[1]};
}

}  // namespace

Tensor rand(std::span<const int64_t> dims, uint64_t seed, uint64_t offset, Device device) {
    auto n = make_node(OpKind::Rand, Shape::contiguous(dims, dtype_size(DType::F32)), DType::F32,
                       device);
    n->n_src = 0;
    n->params.set(RandParams{.seed = seed, .offset = offset});
    return finish(std::move(n));
}

Tensor dropout(const Tensor& input, double p, uint64_t seed, uint64_t offset, bool training) {
    VKML_CHECK(p >= 0.0 && p < 1.0, ShapeError, "dropout() probability must be in [0, 1), got {}",
               p);
    if (!training || p == 0.0) {
        return input;
    }

    // Strict `>=` keeps an element, so p is exactly the probability of being
    // dropped: rand() is uniform on the half-open [0, 1), so P(u < p) == p. A
    // generator that could return 1.0 would make the two ends inconsistent,
    // which is why philox_uniform takes 24 bits rather than scaling 32.
    const Tensor keep = greater_equal(rand(input.shape(), seed, offset, input.device()),
                                      Tensor::full({}, p, DType::F32, input.device()));
    const Tensor scaled = mul(input, 1.0 / (1.0 - p));
    return where(keep, scaled, zeros_like(input));
}

Tensor batch_norm(const Tensor& input, const Tensor& mean, const Tensor& variance,
                  const Tensor& weight, const Tensor& bias, double eps) {
    VKML_CHECK(input.ndim() >= 2, ShapeError,
               "batch_norm() needs rank 2 or higher so axis 1 is the channel, got rank {}",
               input.ndim());
    const int64_t channels = input.size(1);

    // Every per-channel operand is reshaped to (1, C, 1, ...) so it broadcasts
    // across batch and space with stride 0 rather than being materialised.
    std::vector<int64_t> stat_shape(static_cast<size_t>(input.ndim()), 1);
    stat_shape[1] = channels;

    const auto as_channel_vector = [&](const Tensor& t, const char* what) {
        VKML_CHECK(t.ndim() == 1 && t.numel() == channels, ShapeError,
                   "batch_norm(): {} must be rank 1 with {} entries, got rank {} with {}", what,
                   channels, t.ndim(), t.numel());
        check_same_device(input, t, "batch_norm");
        check_same_dtype(input, t, "batch_norm");
        return t.reshape(stat_shape);
    };

    Tensor out = mul(sub(input, as_channel_vector(mean, "mean")),
                     rsqrt(add(as_channel_vector(variance, "variance"), eps)));

    if (weight.defined()) {
        out = mul(out, as_channel_vector(weight, "weight"));
    }
    if (bias.defined()) {
        out = add(out, as_channel_vector(bias, "bias"));
    }
    return out;
}

Tensor max_pool2d(const Tensor& input, std::array<int, 2> kernel, std::array<int, 2> stride,
                  std::array<int, 2> padding, std::array<int, 2> dilation) {
    VKML_CHECK(input.ndim() == 4, ShapeError, "max_pool2d() expects (N, C, H, W), got rank {}",
               input.ndim());
    const std::array<int, 2> step = pool_stride(stride, kernel);
    check_window(kernel, step, padding, dilation, "max_pool2d()");

    for (int i = 0; i < 2; ++i) {
        // torch requires this, and allowing it would place a window entirely
        // inside the padding, whose maximum is -infinity.
        VKML_CHECK(padding[i] * 2 <= kernel[i], ShapeError,
                   "max_pool2d(): padding {} must not exceed half the kernel extent {}", padding[i],
                   kernel[i]);
    }

    const int64_t out_h = window_count(input.size(2), kernel[0], step[0], padding[0], dilation[0]);
    const int64_t out_w = window_count(input.size(3), kernel[1], step[1], padding[1], dilation[1]);
    VKML_CHECK(out_h > 0 && out_w > 0, ShapeError,
               "max_pool2d(): the dilated kernel is larger than the padded image");

    const std::vector<int64_t> dims{input.size(0), input.size(1), out_h, out_w};

    auto n = make_node(OpKind::MaxPool2d, Shape::contiguous(dims, dtype_size(input.dtype())),
                       input.dtype(), input.device());
    n->src[0] = input.node();
    n->n_src = 1;
    n->params.set(
        make_unfold_params(kernel, step, padding, dilation, input.size(2), input.size(3)));
    n->requires_grad = grad_enabled() && input.requires_grad();
    return finish(std::move(n));
}

Tensor avg_pool2d(const Tensor& input, std::array<int, 2> kernel, std::array<int, 2> stride,
                  std::array<int, 2> padding) {
    VKML_CHECK(input.ndim() == 4, ShapeError, "avg_pool2d() expects (N, C, H, W), got rank {}",
               input.ndim());
    const std::array<int, 2> step = pool_stride(stride, kernel);

    const int64_t channels = input.size(1);
    const int64_t patch = static_cast<int64_t>(kernel[0]) * kernel[1];
    const int64_t out_h = window_count(input.size(2), kernel[0], step[0], padding[0], 1);
    const int64_t out_w = window_count(input.size(3), kernel[1], step[1], padding[1], 1);

    // im2col validates the geometry and pads with zero, which is exactly
    // count_include_pad=True: the padded entries contribute nothing to the sum
    // and the divisor stays kernel_h * kernel_w.
    const Tensor cols = im2col(input, kernel, step, padding, {1, 1});
    const Tensor grouped = cols.reshape({input.size(0), channels, patch, out_h * out_w});
    return mean(grouped, {2}).reshape({input.size(0), channels, out_h, out_w});
}

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              std::array<int, 2> stride, std::array<int, 2> padding, std::array<int, 2> dilation) {
    VKML_CHECK(input.ndim() == 4, ShapeError, "conv2d() expects (N, C, H, W) input, got rank {}",
               input.ndim());
    VKML_CHECK(weight.ndim() == 4, ShapeError,
               "conv2d() expects (C_out, C_in, kh, kw) weight, got rank {}", weight.ndim());
    check_same_dtype(input, weight, "conv2d");
    check_same_device(input, weight, "conv2d");
    VKML_CHECK(weight.size(1) == input.size(1), ShapeError,
               "conv2d(): weight has {} input channels but the input has {}; grouped and "
               "depthwise convolution are not supported",
               weight.size(1), input.size(1));

    const int64_t out_channels = weight.size(0);
    const std::array<int, 2> kernel{static_cast<int>(weight.size(2)),
                                    static_cast<int>(weight.size(3))};

    // (N, C*kh*kw, L). im2col validates the geometry, so no separate check here.
    const Tensor cols = im2col(input, kernel, stride, padding, dilation);
    const int64_t positions = cols.size(2);

    const int64_t out_h =
        window_count(input.size(2), kernel[0], stride[0], padding[0], dilation[0]);
    const int64_t out_w =
        window_count(input.size(3), kernel[1], stride[1], padding[1], dilation[1]);

    // (C_out, C*kh*kw) @ (N, C*kh*kw, L) -> (N, C_out, L). matmul broadcasts the
    // batch axes, so the weights are shared across the batch without a copy.
    const Tensor flat_weight = weight.reshape({out_channels, cols.size(1)});
    Tensor out = matmul(flat_weight, cols).reshape({input.size(0), out_channels, out_h, out_w});

    if (bias.defined()) {
        VKML_CHECK(bias.ndim() == 1 && bias.size(0) == out_channels, ShapeError,
                   "conv2d(): bias must be rank 1 with {} elements, got rank {} with {}",
                   out_channels, bias.ndim(), bias.numel());
        // (C_out, 1, 1) right-aligns against (N, C_out, H, W), so the bias
        // broadcasts across batch and space with stride 0 rather than a copy.
        out = add(out, bias.reshape({out_channels, 1, 1}));
    }

    VKML_ASSERT(positions == out_h * out_w, "im2col window count {} disagrees with {}x{}",
                positions, out_h, out_w);
    return out;
}

Tensor index_select(const Tensor& a, int axis, const Tensor& index) {
    VKML_CHECK(index.dtype() == DType::I64, DTypeError, "index_select() index must be I64, got {}",
               dtype_name(index.dtype()));
    VKML_CHECK(index.ndim() == 1, ShapeError, "index_select() index must be rank 1, got rank {}",
               index.ndim());
    check_same_device(a, index, OpKind::IndexSelect);

    const int ax = normalize_dim(axis, a.ndim());
    std::vector<int64_t> dims = a.shape();
    dims[static_cast<size_t>(ax)] = index.numel();

    auto n = make_node(OpKind::IndexSelect, Shape::contiguous(dims, dtype_size(a.dtype())),
                       a.dtype(), a.device());
    n->src[0] = a.node();
    n->src[1] = index.node();
    n->n_src = 2;
    n->params.set(AxisParams{.axis = ax});
    n->requires_grad = grad_enabled() && a.requires_grad();
    return finish(std::move(n));
}

Tensor scatter_add(const Tensor& src, int axis, const Tensor& index, int64_t dim_size) {
    VKML_CHECK(index.dtype() == DType::I64, DTypeError, "scatter_add() index must be I64, got {}",
               dtype_name(index.dtype()));
    VKML_CHECK(index.ndim() == 1, ShapeError, "scatter_add() index must be rank 1, got rank {}",
               index.ndim());
    check_same_device(src, index, OpKind::ScatterAdd);
    VKML_CHECK(dim_size >= 0, ShapeError, "scatter_add() dim_size must be non-negative, got {}",
               dim_size);

    const int ax = normalize_dim(axis, src.ndim());
    VKML_CHECK(src.size(ax) == index.numel(), ShapeError,
               "scatter_add() source extent on axis {} is {} but the index has {} entries", ax,
               src.size(ax), index.numel());

    std::vector<int64_t> dims = src.shape();
    dims[static_cast<size_t>(ax)] = dim_size;

    auto n = make_node(OpKind::ScatterAdd, Shape::contiguous(dims, dtype_size(src.dtype())),
                       src.dtype(), src.device());
    n->src[0] = src.node();
    n->src[1] = index.node();
    n->n_src = 2;
    n->params.set(AxisParams{.axis = ax});
    n->requires_grad = grad_enabled() && src.requires_grad();
    return finish(std::move(n));
}

Tensor layer_norm(const Tensor& a, int normalized_axes, double eps) {
    const std::vector<int> axes = trailing_axes(a, normalized_axes, "layer_norm");

    // keepdim so the mean broadcasts back over the reduced axes with stride 0
    // rather than needing an explicit reshape.
    const Tensor centered = sub(a, mean(a, axes, /*keepdim=*/true));
    const Tensor variance = mean(square(centered), axes, /*keepdim=*/true);
    return mul(centered, rsqrt(add(variance, eps)));
}

Tensor rms_norm(const Tensor& a, int normalized_axes, double eps) {
    const std::vector<int> axes = trailing_axes(a, normalized_axes, "rms_norm");
    const Tensor mean_square = mean(square(a), axes, /*keepdim=*/true);
    return mul(a, rsqrt(add(mean_square, eps)));
}

Tensor cat(const Tensor& a, const Tensor& b, int axis) {
    check_same_dtype(a, b, OpKind::Cat);
    check_same_device(a, b, OpKind::Cat);
    VKML_CHECK(a.ndim() == b.ndim(), ShapeError, "cat() operands have different ranks: {} and {}",
               a.ndim(), b.ndim());

    const int ax = normalize_dim(axis, a.ndim());
    for (int i = 0; i < a.ndim(); ++i) {
        VKML_CHECK(i == ax || a.size(i) == b.size(i), ShapeError,
                   "cat() operands differ on axis {} ({} vs {}); only the concatenated axis "
                   "may differ",
                   i, a.size(i), b.size(i));
    }

    // Bound to a named value: shape() returns by value, so calling it twice in
    // one expression would take begin() and end() from two different
    // temporaries, both destroyed at the semicolon.
    std::vector<int64_t> dims = a.shape();
    dims[static_cast<size_t>(ax)] = a.size(ax) + b.size(ax);

    auto n = make_node(OpKind::Cat, Shape::contiguous(dims, dtype_size(a.dtype())), a.dtype(),
                       a.device());
    n->src[0] = a.node();
    n->src[1] = b.node();
    n->n_src = 2;
    n->params.set(AxisParams{.axis = ax});
    n->requires_grad = grad_enabled() && (a.requires_grad() || b.requires_grad());
    return finish(std::move(n));
}

Tensor cat(std::span<const Tensor> tensors, int axis) {
    VKML_CHECK(!tensors.empty(), ShapeError, "cat() needs at least one tensor");
    if (tensors.size() == 1) {
        return tensors[0];
    }
    // Left fold. The axis is normalised against the first operand's rank, which
    // every operand shares, so re-normalising per step cannot change it.
    Tensor acc = cat(tensors[0], tensors[1], axis);
    for (size_t i = 2; i < tensors.size(); ++i) {
        acc = cat(acc, tensors[i], axis);
    }
    return acc;
}

Tensor masked_fill(const Tensor& a, const Tensor& mask, double value) {
    VKML_CHECK(mask.dtype() == DType::Bool, DTypeError, "masked_fill() mask must be Bool, got {}",
               dtype_name(mask.dtype()));
    check_same_device(a, mask, OpKind::Where);
    // where() broadcasts all three operands and already rejects incompatible
    // shapes, so no separate shape check is needed here.
    return where(mask, scalar_like(a, value), a);
}

Tensor triu(const Tensor& a, int64_t diagonal) { return tri_impl(a, diagonal, OpKind::Triu); }

Tensor tril(const Tensor& a, int64_t diagonal) { return tri_impl(a, diagonal, OpKind::Tril); }

Tensor where(const Tensor& cond, const Tensor& a, const Tensor& b) {
    VKML_CHECK(cond.dtype() == DType::Bool, DTypeError, "where() condition must be Bool, got {}",
               dtype_name(cond.dtype()));
    check_same_dtype(a, b, OpKind::Where);
    check_same_device(a, b, OpKind::Where);

    std::vector<int64_t> dims = broadcast_dims(a.shape(), b.shape());
    dims = broadcast_dims(dims, cond.shape());

    auto n = make_node(OpKind::Where, Shape::contiguous(dims, dtype_size(a.dtype())), a.dtype(),
                       a.device());
    n->src[0] = (cond.shape() == dims ? cond : cond.broadcast_to(dims)).node();
    n->src[1] = (a.shape() == dims ? a : a.broadcast_to(dims)).node();
    n->src[2] = (b.shape() == dims ? b : b.broadcast_to(dims)).node();
    n->n_src = 3;
    n->requires_grad = grad_enabled() && (a.requires_grad() || b.requires_grad());
    return finish(std::move(n));
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

Tensor sum(const Tensor& a, std::span<const int> axes, bool keepdim) {
    return reduction(OpKind::Sum, a, axes, keepdim, a.dtype());
}

Tensor mean(const Tensor& a, std::span<const int> axes, bool keepdim) {
    return reduction(OpKind::Mean, a, axes, keepdim, a.dtype());
}

Tensor prod(const Tensor& a, std::span<const int> axes, bool keepdim) {
    return reduction(OpKind::Prod, a, axes, keepdim, a.dtype());
}

Tensor max(const Tensor& a, std::span<const int> axes, bool keepdim) {
    return reduction(OpKind::Max, a, axes, keepdim, a.dtype());
}

Tensor min(const Tensor& a, std::span<const int> axes, bool keepdim) {
    return reduction(OpKind::Min, a, axes, keepdim, a.dtype());
}

Tensor argmax(const Tensor& a, int axis, bool keepdim) {
    const std::array<int, 1> axes{axis};
    return reduction(OpKind::ArgMax, a, axes, keepdim, DType::I64);
}

Tensor argmin(const Tensor& a, int axis, bool keepdim) {
    const std::array<int, 1> axes{axis};
    return reduction(OpKind::ArgMin, a, axes, keepdim, DType::I64);
}

namespace {

Tensor softmax_family(OpKind op, const Tensor& a, int axis) {
    const int norm = normalize_dim(axis, a.ndim());
    auto n =
        make_node(op, Shape::contiguous(a.shape(), dtype_size(a.dtype())), a.dtype(), a.device());
    n->src[0] = a.node();
    n->n_src = 1;
    n->params.set(AxisParams{.axis = norm});
    n->requires_grad = grad_enabled() && a.requires_grad();
    return finish(std::move(n));
}

}  // namespace

Tensor softmax(const Tensor& a, int axis) { return softmax_family(OpKind::Softmax, a, axis); }

Tensor log_softmax(const Tensor& a, int axis) {
    return softmax_family(OpKind::LogSoftmax, a, axis);
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

Tensor matmul(const Tensor& a, const Tensor& b) {
    check_same_device(a, b, OpKind::Matmul);
    check_same_dtype(a, b, OpKind::Matmul);
    VKML_CHECK(a.ndim() >= 1 && b.ndim() >= 1, ShapeError, "matmul needs rank >= 1 operands");

    // torch.matmul semantics: a rank-1 operand is temporarily promoted, and the
    // promoted axis is removed from the result.
    const bool a_vec = a.ndim() == 1;
    const bool b_vec = b.ndim() == 1;

    Tensor am = a_vec ? a.unsqueeze(0) : a;  // [K] -> [1, K]
    Tensor bm = b_vec ? b.unsqueeze(1) : b;  // [K] -> [K, 1]

    // Bind the shapes to locals. Tensor::shape() returns by value, so writing
    // `am.shape().begin()` and `am.shape().end()` would produce iterators into
    // two *different* temporaries -- undefined behaviour that manifests as
    // nonsense ranks.
    const std::vector<int64_t> am_shape = am.shape();
    const std::vector<int64_t> bm_shape = bm.shape();

    const int64_t m = am_shape[am_shape.size() - 2];
    const int64_t k = am_shape.back();
    const int64_t k2 = bm_shape[bm_shape.size() - 2];
    const int64_t n = bm_shape.back();

    VKML_CHECK(k == k2, ShapeError, "matmul inner dimensions disagree: {} vs {}", k, k2);

    // Batch axes broadcast against each other, then everything is normalised to
    // rank 4 so the kernel has exactly one shape to handle.
    const std::vector<int64_t> a_batch(am_shape.begin(), am_shape.end() - 2);
    const std::vector<int64_t> b_batch(bm_shape.begin(), bm_shape.end() - 2);
    std::vector<int64_t> batch = broadcast_dims(a_batch, b_batch);

    VKML_CHECK(batch.size() <= 2, ShapeError,
               "matmul supports at most 2 batch axes at rank {} (got {})", kMaxDims, batch.size());
    while (batch.size() < 2) {
        batch.insert(batch.begin(), 1);
    }

    std::vector<int64_t> a_target = batch;
    a_target.push_back(m);
    a_target.push_back(k);
    std::vector<int64_t> b_target = batch;
    b_target.push_back(k);
    b_target.push_back(n);

    const Tensor a4 = am.broadcast_to(a_target);
    const Tensor b4 = bm.broadcast_to(b_target);

    std::vector<int64_t> out_dims = batch;
    out_dims.push_back(m);
    out_dims.push_back(n);

    auto node = make_node(OpKind::Matmul, Shape::contiguous(out_dims, dtype_size(a.dtype())),
                          a.dtype(), a.device());
    node->src[0] = a4.node();
    node->src[1] = b4.node();
    node->n_src = 2;
    node->requires_grad = grad_enabled() && (a.requires_grad() || b.requires_grad());

    Tensor out = finish(std::move(node));

    // Undo the rank-1 promotions and drop batch axes that were only added to
    // reach rank 4. `a_batch`/`b_batch` are the pre-padding batch extents.
    const std::vector<int64_t> real_batch = broadcast_dims(a_batch, b_batch);

    std::vector<int64_t> final_dims;
    final_dims.insert(final_dims.end(), real_batch.begin(), real_batch.end());
    if (!a_vec) {
        final_dims.push_back(m);
    }
    if (!b_vec) {
        final_dims.push_back(n);
    }
    return out.reshape(final_dims);
}

}  // namespace vkml
