#include "vkml/api/ops.h"

#include "vkml/dispatch/executor.h"
#include "vkml/graph/grad_mode.h"
#include "vkml/graph/node.h"
#include "vkml/util/assert.h"

#include <algorithm>
#include <format>

namespace vkml {
namespace {

void check_same_device(const Tensor& a, const Tensor& b, OpKind op) {
    VKML_CHECK(a.device() == b.device(), DeviceError,
               "'{}' operands are on different devices: {} and {}", op_name(op), a.device().str(),
               b.device().str());
}

/// M0 policy: no implicit type promotion between tensors.
///
/// PyTorch has a full promotion lattice (f32 + i64 -> f32). Reproducing it is a
/// meaningful amount of machinery and a rich source of silent surprises, and it
/// is not needed to train anything. Requiring a matching dtype is stricter than
/// PyTorch and therefore never *wrong* -- a program that works here works there.
/// Python scalars are still converted to the tensor's dtype, so `x + 1` behaves
/// as expected. Documented as a deliberate divergence.
void check_same_dtype(const Tensor& a, const Tensor& b, OpKind op) {
    VKML_CHECK(a.dtype() == b.dtype(), DTypeError,
               "'{}' operands have different dtypes ({} and {}); vkml does not promote "
               "implicitly -- cast one explicitly with .to()",
               op_name(op), dtype_name(a.dtype()), dtype_name(b.dtype()));
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
