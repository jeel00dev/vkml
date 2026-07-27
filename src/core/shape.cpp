#include "vkml/core/shape.h"

#include "vkml/util/assert.h"

#include <algorithm>
#include <format>

namespace vkml {

int normalize_dim(int dim, int ndim, int extra) {
    const int span = ndim + extra;
    VKML_CHECK(span > 0, IndexError, "a rank-0 shape has no axes to index (got axis {})", dim);
    VKML_CHECK(dim >= -span && dim < span, IndexError,
               "axis {} is out of range for a rank-{} shape (valid range [{}, {}])", dim, ndim,
               -span, span - 1);
    return dim < 0 ? dim + span : dim;
}

std::vector<int64_t> broadcast_dims(std::span<const int64_t> a, std::span<const int64_t> b) {
    const size_t nd = std::max(a.size(), b.size());
    VKML_CHECK(nd <= static_cast<size_t>(kMaxDims), ShapeError,
               "broadcast result would have rank {}, which exceeds kMaxDims={}", nd, kMaxDims);

    std::vector<int64_t> out(nd);
    // Right-aligned, per numpy broadcasting rules.
    for (size_t k = 0; k < nd; ++k) {
        const int64_t da = k < a.size() ? a[a.size() - 1 - k] : 1;
        const int64_t db = k < b.size() ? b[b.size() - 1 - k] : 1;

        int64_t r = 0;
        if (da == db) {
            r = da;
        } else if (da == 1) {
            r = db;
        } else if (db == 1) {
            r = da;
        } else {
            throw ShapeError(std::format(
                "shapes are not broadcastable: extent {} vs {} at axis -{}", da, db, k + 1));
        }
        out[nd - 1 - k] = r;
    }
    return out;
}

Shape Shape::contiguous(std::span<const int64_t> dims, size_t itemsize) {
    VKML_CHECK(dims.size() <= static_cast<size_t>(kMaxDims), ShapeError,
               "rank {} exceeds kMaxDims={}", dims.size(), kMaxDims);
    VKML_CHECK(itemsize > 0, ShapeError, "itemsize must be positive");

    Shape s;
    s.ndim_ = static_cast<int>(dims.size());
    s.itemsize_ = itemsize;

    for (size_t i = 0; i < dims.size(); ++i) {
        VKML_CHECK(dims[i] >= 0, ShapeError, "negative extent {} at axis {}", dims[i], i);
        s.dims_[i] = dims[i];
    }

    // Row-major: the innermost axis has the smallest stride.
    int64_t acc = static_cast<int64_t>(itemsize);
    for (int i = s.ndim_ - 1; i >= 0; --i) {
        s.strides_[static_cast<size_t>(i)] = acc;
        acc *= s.dims_[static_cast<size_t>(i)];
    }
    return s;
}

Shape Shape::strided(std::span<const int64_t> dims, std::span<const int64_t> strides_bytes,
                     size_t itemsize) {
    VKML_CHECK(dims.size() == strides_bytes.size(), ShapeError,
               "dims and strides differ in length ({} vs {})", dims.size(), strides_bytes.size());
    VKML_CHECK(dims.size() <= static_cast<size_t>(kMaxDims), ShapeError,
               "rank {} exceeds kMaxDims={}", dims.size(), kMaxDims);
    VKML_CHECK(itemsize > 0, ShapeError, "itemsize must be positive");

    Shape s;
    s.ndim_ = static_cast<int>(dims.size());
    s.itemsize_ = itemsize;
    for (size_t i = 0; i < dims.size(); ++i) {
        VKML_CHECK(dims[i] >= 0, ShapeError, "negative extent {} at axis {}", dims[i], i);
        s.dims_[i] = dims[i];
        s.strides_[i] = strides_bytes[i];
    }
    return s;
}

int64_t Shape::numel() const noexcept {
    int64_t n = 1;
    for (int i = 0; i < ndim_; ++i) {
        n *= dims_[static_cast<size_t>(i)];
    }
    return n;
}

size_t Shape::nbytes() const noexcept { return static_cast<size_t>(numel()) * itemsize_; }

bool Shape::is_contiguous() const noexcept {
    if (numel() == 0) {
        return true;  // nothing is ever addressed, so the strides cannot matter
    }
    int64_t expected = static_cast<int64_t>(itemsize_);
    for (int i = ndim_ - 1; i >= 0; --i) {
        const size_t u = static_cast<size_t>(i);
        if (dims_[u] == 1) {
            continue;  // an extent of 1 cannot constrain the layout
        }
        if (strides_[u] != expected) {
            return false;
        }
        expected *= dims_[u];
    }
    return true;
}

bool Shape::has_broadcast_stride() const noexcept {
    for (int i = 0; i < ndim_; ++i) {
        if (strides_[static_cast<size_t>(i)] == 0 && dims_[static_cast<size_t>(i)] > 1) {
            return true;
        }
    }
    return false;
}

int64_t Shape::offset_of(std::span<const int64_t> index) const {
    VKML_CHECK(index.size() == static_cast<size_t>(ndim_), IndexError,
               "index has {} components for a rank-{} shape", index.size(), ndim_);
    int64_t off = 0;
    for (size_t i = 0; i < index.size(); ++i) {
        VKML_CHECK(index[i] >= 0 && index[i] < dims_[i], IndexError,
                   "index {} is out of range for extent {} at axis {}", index[i], dims_[i], i);
        off += index[i] * strides_[i];
    }
    return off;
}

Shape Shape::permuted(std::span<const int> perm) const {
    VKML_CHECK(perm.size() == static_cast<size_t>(ndim_), ShapeError,
               "permutation has {} entries for a rank-{} shape", perm.size(), ndim_);

    std::array<bool, kMaxDims> seen{};
    Shape s;
    s.ndim_ = ndim_;
    s.itemsize_ = itemsize_;

    for (size_t i = 0; i < perm.size(); ++i) {
        const int axis = normalize_dim(perm[i], ndim_);
        VKML_CHECK(!seen[static_cast<size_t>(axis)], ShapeError,
                   "axis {} appears more than once in the permutation", axis);
        seen[static_cast<size_t>(axis)] = true;
        s.dims_[i] = dims_[static_cast<size_t>(axis)];
        s.strides_[i] = strides_[static_cast<size_t>(axis)];
    }
    return s;
}

Shape Shape::transposed(int a, int b) const {
    const int ia = normalize_dim(a, ndim_);
    const int ib = normalize_dim(b, ndim_);
    Shape s = *this;
    std::swap(s.dims_[static_cast<size_t>(ia)], s.dims_[static_cast<size_t>(ib)]);
    std::swap(s.strides_[static_cast<size_t>(ia)], s.strides_[static_cast<size_t>(ib)]);
    return s;
}

Shape Shape::squeezed(int dim) const {
    const int axis = normalize_dim(dim, ndim_);
    VKML_CHECK(dims_[static_cast<size_t>(axis)] == 1, ShapeError,
               "cannot squeeze axis {} with extent {}", axis, dims_[static_cast<size_t>(axis)]);

    Shape s;
    s.ndim_ = ndim_ - 1;
    s.itemsize_ = itemsize_;
    for (int i = 0, o = 0; i < ndim_; ++i) {
        if (i == axis) {
            continue;
        }
        s.dims_[static_cast<size_t>(o)] = dims_[static_cast<size_t>(i)];
        s.strides_[static_cast<size_t>(o)] = strides_[static_cast<size_t>(i)];
        ++o;
    }
    return s;
}

Shape Shape::unsqueezed(int dim) const {
    VKML_CHECK(ndim_ + 1 <= kMaxDims, ShapeError, "unsqueeze would produce rank {} > kMaxDims={}",
               ndim_ + 1, kMaxDims);
    const int axis = normalize_dim(dim, ndim_, /*extra=*/1);

    Shape s;
    s.ndim_ = ndim_ + 1;
    s.itemsize_ = itemsize_;

    // The new axis has extent 1, so its stride is unobservable; pick the value a
    // contiguous layout would have used so that debug output reads naturally.
    const int64_t new_stride = axis < ndim_
                                   ? strides_[static_cast<size_t>(axis)] *
                                         std::max<int64_t>(dims_[static_cast<size_t>(axis)], 1)
                                   : static_cast<int64_t>(itemsize_);

    for (int i = 0, o = 0; o < s.ndim_; ++o) {
        if (o == axis) {
            s.dims_[static_cast<size_t>(o)] = 1;
            s.strides_[static_cast<size_t>(o)] = new_stride;
            continue;
        }
        s.dims_[static_cast<size_t>(o)] = dims_[static_cast<size_t>(i)];
        s.strides_[static_cast<size_t>(o)] = strides_[static_cast<size_t>(i)];
        ++i;
    }
    return s;
}

Shape Shape::broadcast_to(std::span<const int64_t> target) const {
    VKML_CHECK(target.size() <= static_cast<size_t>(kMaxDims), ShapeError,
               "broadcast target rank {} exceeds kMaxDims={}", target.size(), kMaxDims);
    VKML_CHECK(target.size() >= static_cast<size_t>(ndim_), ShapeError,
               "cannot broadcast rank {} down to rank {}", ndim_, target.size());

    Shape s;
    s.ndim_ = static_cast<int>(target.size());
    s.itemsize_ = itemsize_;

    const size_t lead = target.size() - static_cast<size_t>(ndim_);
    for (size_t k = 0; k < target.size(); ++k) {
        if (k < lead) {
            // A brand new leading axis: every index into it aliases the same
            // element, which stride 0 expresses exactly and for free.
            s.dims_[k] = target[k];
            s.strides_[k] = 0;
            continue;
        }
        const size_t i = k - lead;
        if (dims_[i] == target[k]) {
            s.dims_[k] = target[k];
            s.strides_[k] = strides_[i];
        } else if (dims_[i] == 1) {
            s.dims_[k] = target[k];
            s.strides_[k] = 0;
        } else {
            throw ShapeError(
                std::format("cannot broadcast extent {} to {} at axis {}", dims_[i], target[k], k));
        }
    }
    return s;
}

std::optional<Shape> Shape::reshaped(std::span<const int64_t> dims) const {
    VKML_CHECK(dims.size() <= static_cast<size_t>(kMaxDims), ShapeError,
               "reshape target rank {} exceeds kMaxDims={}", dims.size(), kMaxDims);

    // Resolve a single inferred extent (-1).
    std::vector<int64_t> resolved(dims.begin(), dims.end());
    int infer_at = -1;
    int64_t known = 1;
    for (size_t i = 0; i < resolved.size(); ++i) {
        if (resolved[i] == -1) {
            VKML_CHECK(infer_at < 0, ShapeError, "at most one extent may be -1");
            infer_at = static_cast<int>(i);
        } else {
            VKML_CHECK(resolved[i] >= 0, ShapeError, "invalid extent {} at axis {}", resolved[i],
                       i);
            known *= resolved[i];
        }
    }

    const int64_t n = numel();
    if (infer_at >= 0) {
        VKML_CHECK(known > 0 && n % known == 0, ShapeError,
                   "cannot infer an extent: {} elements do not divide by {}", n, known);
        resolved[static_cast<size_t>(infer_at)] = n / known;
    } else {
        VKML_CHECK(known == n, ShapeError, "reshape changes the element count ({} -> {})", n,
                   known);
    }

    // A non-contiguous view generally cannot be re-expressed as strides over the
    // same memory. Report that rather than guessing, so the caller materialises
    // a copy explicitly and the copy shows up in the graph.
    if (!is_contiguous()) {
        return std::nullopt;
    }
    return Shape::contiguous(resolved, itemsize_);
}

SlicedShape Shape::sliced(int dim, int64_t start, int64_t stop, int64_t step) const {
    const int axis = normalize_dim(dim, ndim_);
    const int64_t extent = dims_[static_cast<size_t>(axis)];

    VKML_CHECK(step > 0, ShapeError, "slice step must be positive, got {}", step);
    VKML_CHECK(start >= 0 && start <= extent, IndexError,
               "slice start {} is out of range for extent {}", start, extent);
    VKML_CHECK(stop >= start && stop <= extent, IndexError,
               "slice stop {} is out of range for start {} and extent {}", stop, start, extent);

    Shape s = *this;
    s.dims_[static_cast<size_t>(axis)] = (stop - start + step - 1) / step;
    s.strides_[static_cast<size_t>(axis)] = strides_[static_cast<size_t>(axis)] * step;

    return SlicedShape{s, start * strides_[static_cast<size_t>(axis)]};
}

bool Shape::same_dims(const Shape& other) const noexcept {
    if (ndim_ != other.ndim_) {
        return false;
    }
    for (int i = 0; i < ndim_; ++i) {
        if (dims_[static_cast<size_t>(i)] != other.dims_[static_cast<size_t>(i)]) {
            return false;
        }
    }
    return true;
}

bool operator==(const Shape& a, const Shape& b) noexcept {
    if (a.ndim_ != b.ndim_ || a.itemsize_ != b.itemsize_) {
        return false;
    }
    for (int i = 0; i < a.ndim_; ++i) {
        const size_t u = static_cast<size_t>(i);
        if (a.dims_[u] != b.dims_[u] || a.strides_[u] != b.strides_[u]) {
            return false;
        }
    }
    return true;
}

std::string Shape::str() const {
    std::string out = "(";
    for (int i = 0; i < ndim_; ++i) {
        out += std::format("{}{}", i ? ", " : "", dims_[static_cast<size_t>(i)]);
    }
    if (ndim_ == 1) {
        out += ",";
    }
    out += ")";

    if (!is_contiguous()) {
        out += " strides=[";
        for (int i = 0; i < ndim_; ++i) {
            out += std::format("{}{}", i ? ", " : "", strides_[static_cast<size_t>(i)]);
        }
        out += "]B";
    }
    return out;
}

}  // namespace vkml
