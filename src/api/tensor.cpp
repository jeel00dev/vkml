#include "vkml/api/tensor.h"

#include "vkml/backend/api/backend.h"
#include "vkml/dispatch/executor.h"
#include "vkml/graph/grad_mode.h"
#include "vkml/graph/node.h"
#include "vkml/util/assert.h"

#include <cmath>
#include <cstring>
#include <format>

namespace vkml {
namespace {

/// Creates a leaf holding freshly allocated storage, ready to be filled.
///
/// Bound AND computed: an Input leaf has no rule to evaluate, so once it has
/// memory there is nothing left for the scheduler to do with it. The caller
/// copies host bytes in immediately (`from_host`), which is the only reason
/// marking it here rather than after the copy is honest.
NodePtr make_leaf(std::span<const int64_t> dims, DType dtype, Device device) {
    auto n = make_node(OpKind::Input, Shape::contiguous(dims, dtype_size(dtype)), dtype, device);
    n->storage = backend_for(device).allocator().allocate(n->shape.nbytes());
    n->flags |= kFlagComputed;
    return n;
}

/// Creates an unrealised generator node (Full, Arange).
NodePtr make_generator(OpKind op, std::span<const int64_t> dims, DType dtype, Device device) {
    return make_node(op, Shape::contiguous(dims, dtype_size(dtype)), dtype, device);
}

void require_defined(const Tensor& t, const char* what) {
    VKML_CHECK(t.defined(), Error, "{} called on an undefined tensor", what);
}

/// Whether two realised nodes name overlapping bytes of the SAME storage.
///
/// Different storages never overlap: the allocator hands out disjoint
/// suballocations, and two distinct Storage objects cannot alias. So this is
/// only a question when both nodes point at one storage, and then it is an
/// ordinary interval test on `[offset, offset + nbytes)`.
[[nodiscard]] bool storages_overlap(const Node& a, const Node& b, size_t nbytes) {
    if (a.storage.get() != b.storage.get()) {
        return false;
    }
    const int64_t a_end = a.storage_offset + static_cast<int64_t>(nbytes);
    const int64_t b_end = b.storage_offset + static_cast<int64_t>(nbytes);
    return a.storage_offset < b_end && b.storage_offset < a_end;
}

}  // namespace

Tensor::Tensor() = default;
Tensor::~Tensor() = default;
Tensor::Tensor(const Tensor&) = default;
Tensor& Tensor::operator=(const Tensor&) = default;
Tensor::Tensor(Tensor&&) noexcept = default;
Tensor& Tensor::operator=(Tensor&&) noexcept = default;

Tensor::Tensor(std::shared_ptr<Node> node) : node_(std::move(node)) {}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Tensor Tensor::full(std::span<const int64_t> dims, double value, DType dtype, Device device) {
    auto n = make_generator(OpKind::Full, dims, dtype, device);
    n->params.set(FullParams{.value = value});
    Tensor t{std::move(n)};
    if (eager()) {
        t.realize();
    }
    return t;
}

Tensor Tensor::zeros(std::span<const int64_t> dims, DType dtype, Device device) {
    return full(dims, 0.0, dtype, device);
}

Tensor Tensor::ones(std::span<const int64_t> dims, DType dtype, Device device) {
    return full(dims, 1.0, dtype, device);
}

Tensor Tensor::arange(double start, double stop, double step, DType dtype, Device device) {
    VKML_CHECK(step != 0.0, ShapeError, "arange step must be non-zero");
    const double span = (stop - start) / step;
    const int64_t count = span <= 0.0 ? 0 : static_cast<int64_t>(std::ceil(span));

    const std::array<int64_t, 1> dims{count};
    auto n = make_generator(OpKind::Arange, dims, dtype, device);
    n->params.set(ArangeParams{.start = start, .step = step});
    Tensor t{std::move(n)};
    if (eager()) {
        t.realize();
    }
    return t;
}

Tensor Tensor::from_host(const void* data, std::span<const int64_t> dims, DType dtype,
                         Device device) {
    NodePtr n = make_leaf(dims, dtype, device);
    if (n->shape.nbytes() > 0) {
        VKML_CHECK(data != nullptr, Error,
                   "from_host received a null pointer for a non-empty "
                   "tensor");
        backend_for(device).copy_from_host(*n->storage, 0, data, n->shape.nbytes());
    }
    return Tensor{std::move(n)};
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

std::vector<int64_t> Tensor::shape() const {
    require_defined(*this, "shape()");
    const auto d = node_->shape.dims();
    return {d.begin(), d.end()};
}

int Tensor::ndim() const {
    require_defined(*this, "ndim()");
    return node_->shape.ndim();
}

int64_t Tensor::numel() const {
    require_defined(*this, "numel()");
    return node_->shape.numel();
}

int64_t Tensor::size(int axis) const {
    require_defined(*this, "size()");
    return node_->shape.dim(normalize_dim(axis, node_->shape.ndim()));
}

std::vector<int64_t> Tensor::strides() const {
    require_defined(*this, "strides()");
    const auto s = node_->shape.strides();
    return {s.begin(), s.end()};
}

DType Tensor::dtype() const {
    require_defined(*this, "dtype()");
    return node_->dtype;
}

Device Tensor::device() const {
    require_defined(*this, "device()");
    return node_->device;
}

bool Tensor::is_contiguous() const {
    require_defined(*this, "is_contiguous()");
    return node_->shape.is_contiguous();
}

std::string Tensor::str() const {
    if (!defined()) {
        return "Tensor(<undefined>)";
    }
    return std::format("Tensor(shape={}, dtype={}, device={}{})", node_->shape.str(),
                       dtype_name(node_->dtype), node_->device.str(),
                       node_->requires_grad ? ", requires_grad=True" : "");
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

const Tensor& Tensor::realize() const {
    require_defined(*this, "realize()");
    vkml::realize(node_);
    return *this;
}

void Tensor::to_host(void* dst) const {
    require_defined(*this, "to_host()");
    realize();

    const int64_t n = node_->shape.numel();
    if (n == 0) {
        return;
    }
    VKML_CHECK(dst != nullptr, Error, "to_host received a null destination");

    const size_t esz = dtype_size(node_->dtype);

    // A strided source cannot be memcpy'd out in one go, and the host buffer is
    // always dense row-major. Materialise a contiguous copy through the normal
    // graph machinery rather than duplicating gather logic here -- that keeps
    // one implementation of strided reads, in the backend where it belongs.
    if (!node_->shape.is_contiguous()) {
        contiguous().to_host(dst);
        return;
    }

    backend_for(node_->device)
        .copy_to_host(dst, *node_->storage, node_->storage_offset, static_cast<size_t>(n) * esz);
}

float Tensor::item() const {
    require_defined(*this, "item()");
    VKML_CHECK(numel() == 1, ShapeError, "item() requires exactly one element, got {}", numel());
    VKML_CHECK(node_->dtype == DType::F32, DTypeError, "item() currently supports F32 only, got {}",
               dtype_name(node_->dtype));
    float v = 0.0F;
    to_host(&v);
    return v;
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

Tensor Tensor::reshape(std::span<const int64_t> dims) const {
    require_defined(*this, "reshape()");

    if (const auto s = node_->shape.reshaped(dims); s.has_value()) {
        return Tensor{make_view(OpKind::Reshape, node_, *s, 0)};
    }
    // Not expressible as strides over this layout. Materialise, then retry --
    // and the copy is a visible Contiguous node rather than a hidden memcpy.
    return contiguous().reshape(dims);
}

Tensor Tensor::permute(std::span<const int> perm) const {
    require_defined(*this, "permute()");
    NodePtr n = make_view(OpKind::Permute, node_, node_->shape.permuted(perm), 0);

    // Record the permutation. Backward needs its inverse, and recovering that
    // from strides alone is ambiguous whenever two axes share a stride (which
    // happens whenever an extent is 1).
    PermuteParams p{};
    for (size_t i = 0; i < perm.size(); ++i) {
        p.perm[i] = normalize_dim(perm[i], node_->shape.ndim());
    }
    n->params.set(p);
    return Tensor{std::move(n)};
}

Tensor Tensor::transpose(int a, int b) const {
    require_defined(*this, "transpose()");
    const int nd = node_->shape.ndim();
    const int ia = normalize_dim(a, nd);
    const int ib = normalize_dim(b, nd);

    std::vector<int> perm(static_cast<size_t>(nd));
    for (int i = 0; i < nd; ++i) {
        perm[static_cast<size_t>(i)] = i;
    }
    std::swap(perm[static_cast<size_t>(ia)], perm[static_cast<size_t>(ib)]);
    return permute(perm);
}

Tensor Tensor::squeeze(int axis) const {
    require_defined(*this, "squeeze()");
    return Tensor{make_view(OpKind::Squeeze, node_, node_->shape.squeezed(axis), 0)};
}

Tensor Tensor::unsqueeze(int axis) const {
    require_defined(*this, "unsqueeze()");
    return Tensor{make_view(OpKind::Unsqueeze, node_, node_->shape.unsqueezed(axis), 0)};
}

Tensor Tensor::slice(int axis, int64_t start, int64_t stop, int64_t step) const {
    require_defined(*this, "slice()");
    const int norm = normalize_dim(axis, node_->shape.ndim());
    const SlicedShape r = node_->shape.sliced(norm, start, stop, step);
    NodePtr n = make_view(OpKind::Slice, node_, r.shape, r.offset_bytes);
    // Recorded for the backward rule: the adjoint has to know which positions
    // of the original extent this view selected.
    n->params.set(SliceParams{.axis = norm, .start = start, .stop = stop, .step = step});
    return Tensor{std::move(n)};
}

Tensor Tensor::broadcast_to(std::span<const int64_t> dims) const {
    require_defined(*this, "broadcast_to()");
    return Tensor{make_view(OpKind::Broadcast, node_, node_->shape.broadcast_to(dims), 0)};
}

// ---------------------------------------------------------------------------
// Copies
// ---------------------------------------------------------------------------

Tensor Tensor::contiguous() const {
    require_defined(*this, "contiguous()");
    if (node_->shape.is_contiguous()) {
        return *this;
    }

    auto n = make_node(OpKind::Contiguous,
                       Shape::contiguous(node_->shape.dims(), node_->shape.itemsize()),
                       node_->dtype, node_->device);
    n->src[0] = node_;
    n->n_src = 1;
    n->requires_grad = grad_enabled() && node_->requires_grad;

    Tensor t{std::move(n)};
    if (eager()) {
        t.realize();
    }
    return t;
}

Tensor Tensor::to(DType dtype) const {
    require_defined(*this, "to()");
    if (dtype == node_->dtype) {
        return *this;
    }

    auto n = make_node(OpKind::Cast, Shape::contiguous(node_->shape.dims(), dtype_size(dtype)),
                       dtype, node_->device);
    n->src[0] = node_;
    n->n_src = 1;
    n->params.set(CastParams{.target = dtype});
    // A cast to an integer type is not differentiable, so gradient tracking
    // stops here -- the same rule PyTorch applies.
    n->requires_grad = grad_enabled() && node_->requires_grad && is_differentiable(dtype);

    Tensor t{std::move(n)};
    if (eager()) {
        t.realize();
    }
    return t;
}

void Tensor::assign_(const Tensor& src) {
    require_defined(*this, "assign_()");
    VKML_CHECK(src.defined(), Error, "assign_() from an undefined tensor");
    VKML_CHECK(node_->shape.same_dims(src.node()->shape), ShapeError,
               "assign_() shape mismatch: {} vs {}", node_->shape.str(), src.node()->shape.str());
    VKML_CHECK(node_->dtype == src.dtype(), DTypeError, "assign_() dtype mismatch: {} vs {}",
               dtype_name(node_->dtype), dtype_name(src.dtype()));
    VKML_CHECK(node_->device == src.device(), DeviceError, "assign_() device mismatch");
    VKML_CHECK(node_->shape.is_contiguous(), ShapeError,
               "assign_() destination must be contiguous");

    realize();
    const Tensor flat = src.contiguous();
    flat.realize();

    const size_t nbytes = node_->shape.nbytes();
    if (nbytes == 0) {
        return;
    }

    Backend& backend = backend_for(node_->device);

    // Source and destination are on the same device -- checked above -- so the
    // bytes have no reason to visit the host. They used to: this was a full
    // device -> host -> device round trip for every assignment, which cost
    // three submissions against one for the same arithmetic and hit BatchNorm's
    // forward pass as well as every optimiser. See
    // docs/adr/0006-lazy-assign-and-submission-batching.md.
    //
    // The one case that still needs the host is an OVERLAPPING copy within a
    // single storage, which `t[0:5].assign_(t[2:7])` reaches: vkCmdCopyBuffer
    // requires disjoint regions when the source and destination buffers are the
    // same. Staging through host memory is what made that safe before, so that
    // is what it keeps doing.
    if (!storages_overlap(*node_, *flat.node(), nbytes)) {
        backend.copy_device_to_device(*node_->storage, node_->storage_offset, *flat.node()->storage,
                                      flat.node()->storage_offset, nbytes);
        return;
    }

    std::vector<std::byte> staging(nbytes);
    backend.copy_to_host(staging.data(), *flat.node()->storage, flat.node()->storage_offset,
                         nbytes);
    backend.copy_from_host(*node_->storage, node_->storage_offset, staging.data(), nbytes);
}

// ---------------------------------------------------------------------------
// Autograd
// ---------------------------------------------------------------------------

bool Tensor::requires_grad() const {
    require_defined(*this, "requires_grad()");
    return node_->requires_grad;
}

void Tensor::set_requires_grad(bool value) {
    require_defined(*this, "set_requires_grad()");
    if (value) {
        VKML_CHECK(is_differentiable(node_->dtype), DTypeError,
                   "only floating tensors can require grad, got {}", dtype_name(node_->dtype));
    }
    node_->requires_grad = value;
    if (value) {
        node_->flags |= kFlagParam;
    } else {
        node_->flags &= ~static_cast<uint32_t>(kFlagParam);
    }
}

Tensor Tensor::grad() const {
    require_defined(*this, "grad()");
    return node_->grad ? Tensor{node_->grad} : Tensor{};
}

void Tensor::set_grad(const Tensor& g) {
    require_defined(*this, "set_grad()");
    node_->grad = g.defined() ? g.node() : nullptr;
}

}  // namespace vkml
