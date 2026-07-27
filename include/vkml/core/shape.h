#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vkml {

/// Maximum tensor rank.
///
/// Fixed at 4 per docs/ARCHITECTURE.md §3 Fork 4. The reasoning is concrete
/// rather than aesthetic: every model in scope is rank <= 4 (CNN [N,C,H,W],
/// transformer [B,H,S,D], RNN [T,B,F]), and the target GPU's 256-byte push
/// constant limit means 3 tensors x (dims + strides) costs 96 bytes at rank 4
/// but 192 bytes at rank 8, which would force shape metadata into a uniform
/// buffer and add an indirection to every kernel.
///
/// Raising this later is a contained change (this constant, plus the push
/// constant layout when the Vulkan backend exists), but it taxes every kernel,
/// so it should be a deliberate decision rather than a drift.
inline constexpr int kMaxDims = 4;

/// Result of narrowing one axis. Declared before Shape so that Shape::sliced
/// can name it, and defined after, because it holds a Shape by value.
struct SlicedShape;

/// Describes a strided view over a flat byte buffer: rank, extents, byte
/// strides, and element size. Owns no memory.
///
/// DIMENSION ORDER -- note the deviation from ggml
/// ----------------------------------------------
/// vkml stores dimensions in ROW-MAJOR order: dims[0] is the outermost
/// (slowest-varying) axis and dims[ndim-1] is the innermost (fastest-varying).
/// This is the numpy / PyTorch / dlpack convention.
///
/// ggml uses the opposite order -- ne[0] is the *fastest*-varying axis, so a
/// PyTorch tensor of shape (M, K) is ne = {K, M} in ggml. The architecture
/// document's instruction to follow ggml applies to strides being measured in
/// BYTES (§4.2), which vkml does; it does not extend to axis order, and the
/// stated goal of a PyTorch-shaped API makes row-major the right call:
///
///   - `x.shape == (64, 128)` has to mean 64 rows of 128 for the API to feel
///     familiar at all.
///   - numpy, torch and dlpack all agree on row-major, so zero-copy interop
///     needs no axis reversal.
///   - ggml's reversed order is a documented, recurring source of confusion for
///     people arriving from PyTorch.
///
/// The cost is that anyone reading ggml kernels alongside vkml kernels must
/// mentally reverse the indices. That cost is paid once by us, rather than
/// continuously by every user of the Python API.
///
/// Strides are in BYTES, following ggml and numpy (`ndarray.strides` is also in
/// bytes). Bytes rather than elements makes broadcasting (stride 0) and future
/// mixed-dtype views expressible without special-casing.
class Shape {
public:
    /// Rank-0 scalar: numel() == 1.
    Shape() = default;

    /// Row-major contiguous layout over `dims`.
    [[nodiscard]] static Shape contiguous(std::span<const int64_t> dims, size_t itemsize);

    /// Arbitrary strided layout. Strides are in bytes and may be 0 (broadcast).
    [[nodiscard]] static Shape strided(std::span<const int64_t> dims,
                                       std::span<const int64_t> strides_bytes, size_t itemsize);

    [[nodiscard]] int ndim() const noexcept { return ndim_; }

    [[nodiscard]] int64_t dim(int i) const noexcept { return dims_[static_cast<size_t>(i)]; }

    [[nodiscard]] int64_t stride(int i) const noexcept { return strides_[static_cast<size_t>(i)]; }

    [[nodiscard]] std::span<const int64_t> dims() const noexcept {
        return {dims_.data(), static_cast<size_t>(ndim_)};
    }

    [[nodiscard]] std::span<const int64_t> strides() const noexcept {
        return {strides_.data(), static_cast<size_t>(ndim_)};
    }

    [[nodiscard]] size_t itemsize() const noexcept { return itemsize_; }

    /// Total element count. 1 for a rank-0 scalar, 0 if any extent is 0.
    [[nodiscard]] int64_t numel() const noexcept;

    /// Bytes a *contiguous* tensor of this rank and extents would occupy.
    [[nodiscard]] size_t nbytes() const noexcept;

    /// True if a row-major walk of the elements visits memory sequentially.
    /// Extents of 1 are ignored, because their stride cannot constrain the
    /// layout -- this matches numpy and torch, and matters because transposing
    /// a (1, N) tensor must not make it look non-contiguous.
    [[nodiscard]] bool is_contiguous() const noexcept;

    /// True if any stride is 0, i.e. some element is aliased by broadcasting.
    /// Such a view must never be written to.
    [[nodiscard]] bool has_broadcast_stride() const noexcept;

    /// Byte offset of an element from the start of the view.
    [[nodiscard]] int64_t offset_of(std::span<const int64_t> index) const;

    // -- transforms. All return a new Shape; none touch memory. --------------

    /// Reorders axes. `perm` must be a permutation of [0, ndim).
    [[nodiscard]] Shape permuted(std::span<const int> perm) const;

    /// Swaps two axes. Accepts negative indices.
    [[nodiscard]] Shape transposed(int a, int b) const;

    /// Removes an axis of extent 1. Accepts negative indices.
    [[nodiscard]] Shape squeezed(int dim) const;

    /// Inserts an axis of extent 1. Accepts negative indices in [-ndim-1, ndim].
    [[nodiscard]] Shape unsqueezed(int dim) const;

    /// Broadcasts to `target` using numpy rules, giving broadcast axes stride 0.
    /// Throws ShapeError if the shapes are incompatible.
    [[nodiscard]] Shape broadcast_to(std::span<const int64_t> target) const;

    /// Reinterprets the extents without moving data.
    ///
    /// Returns nullopt when this view is not contiguous, because the result
    /// could not then be expressed as strides. The caller is expected to
    /// materialise a contiguous copy and retry -- keeping that decision at the
    /// call site is what makes the copy visible in the graph instead of hidden
    /// inside a "reshape" (docs/ARCHITECTURE.md §3, "make this explicit").
    ///
    /// Exactly one extent may be -1 and is inferred.
    [[nodiscard]] std::optional<Shape> reshaped(std::span<const int64_t> dims) const;

    /// Half-open [start, stop) with a positive step, along one axis.
    /// Returns the narrowed shape together with the byte offset of its first
    /// element relative to this view.
    [[nodiscard]] SlicedShape sliced(int dim, int64_t start, int64_t stop, int64_t step = 1) const;

    [[nodiscard]] std::string str() const;

    friend bool operator==(const Shape& a, const Shape& b) noexcept;

    /// Same rank and extents, ignoring strides and element size.
    [[nodiscard]] bool same_dims(const Shape& other) const noexcept;

private:
    std::array<int64_t, kMaxDims> dims_{};
    std::array<int64_t, kMaxDims> strides_{};
    int ndim_ = 0;
    size_t itemsize_ = 4;
};

struct SlicedShape {
    Shape shape;
    int64_t offset_bytes = 0;
};

/// Converts a possibly-negative axis index into [0, ndim). Throws IndexError
/// when out of range. `extra` widens the accepted range, which unsqueeze needs
/// because inserting at position ndim is legal.
[[nodiscard]] int normalize_dim(int dim, int ndim, int extra = 0);

/// numpy broadcasting: right-align, extents must match or be 1.
/// Throws ShapeError when incompatible.
[[nodiscard]] std::vector<int64_t> broadcast_dims(std::span<const int64_t> a,
                                                  std::span<const int64_t> b);

}  // namespace vkml
