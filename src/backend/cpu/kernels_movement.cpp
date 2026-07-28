#include "iterate.h"
#include "kernels.h"
#include "philox.h"

#include "vkml/util/assert.h"

#include <cstring>
#include <format>

namespace vkml::cpu {
namespace {

/// Copies a possibly-strided view into a fresh contiguous buffer.
///
/// This is the op the graph builder inserts whenever a view cannot express what
/// is needed -- reshaping a transposed tensor, for instance. Keeping it an
/// explicit node rather than a hidden copy inside `reshape` is what makes the
/// cost visible in the graph and in profiles.
void k_contiguous(Node& out) {
    const Node& a = *out.src[0];
    VKML_ASSERT(out.dtype == a.dtype, "contiguous must not change dtype");

    const int64_t n = out.shape.numel();
    if (n == 0) {
        return;
    }

    const size_t esz = dtype_size(out.dtype);
    const auto* src = static_cast<const std::byte*>(a.data());
    auto* dst = static_cast<std::byte*>(out.data());

    if (a.shape.is_contiguous() && out.shape.is_contiguous()) {
        std::memcpy(dst, src, static_cast<size_t>(n) * esz);
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        std::memcpy(dst + linear_to_offset(i, out.shape), src + linear_to_offset(i, a.shape), esz);
    }
}

template <typename From, typename To>
void cast_typed(Node& out, const Node& a) {
    map_unary<To, From>(out, a, [](From v) { return static_cast<To>(v); });
}

/// Converts between element types.
///
/// F16 is handled explicitly rather than through static_cast, because Half is
/// deliberately not an arithmetic type -- accumulating in 16 bits by accident
/// is precisely what that design prevents (see core/dtype.h).
void k_cast(Node& out) {
    const Node& a = *out.src[0];
    const DType from = a.dtype;
    const DType to = out.dtype;

    if (from == to) {
        k_contiguous(out);
        return;
    }

    auto from_f32 = [&](auto tag) {
        using To = decltype(tag);
        cast_typed<float, To>(out, a);
    };

    switch (from) {
        case DType::F32:
            switch (to) {
                case DType::I32: from_f32(int32_t{}); return;
                case DType::I64: from_f32(int64_t{}); return;
                case DType::Bool:
                    map_unary<uint8_t, float>(out, a,
                                              [](float v) -> uint8_t { return v != 0.0F ? 1 : 0; });
                    return;
                case DType::F16:
                    map_unary<uint16_t, float>(out, a, [](float v) { return fp32_to_fp16(v); });
                    return;
                default: break;
            }
            break;
        case DType::F16:
            if (to == DType::F32) {
                map_unary<float, uint16_t>(out, a, [](uint16_t v) { return fp16_to_fp32(v); });
                return;
            }
            break;
        case DType::I32:
            if (to == DType::F32) {
                cast_typed<int32_t, float>(out, a);
                return;
            }
            if (to == DType::I64) {
                cast_typed<int32_t, int64_t>(out, a);
                return;
            }
            break;
        case DType::I64:
            if (to == DType::F32) {
                cast_typed<int64_t, float>(out, a);
                return;
            }
            if (to == DType::I32) {
                cast_typed<int64_t, int32_t>(out, a);
                return;
            }
            break;
        case DType::Bool:
            if (to == DType::F32) {
                map_unary<float, uint8_t>(out, a, [](uint8_t v) { return v != 0 ? 1.0F : 0.0F; });
                return;
            }
            if (to == DType::I64) {
                map_unary<int64_t, uint8_t>(out, a,
                                            [](uint8_t v) -> int64_t { return v != 0 ? 1 : 0; });
                return;
            }
            break;
    }

    throw DTypeError(
        std::format("cpu backend: unsupported cast {} -> {}", dtype_name(from), dtype_name(to)));
}

void k_full(Node& out) {
    const auto p = out.params.get<FullParams>();
    const int64_t n = out.shape.numel();

    switch (out.dtype) {
        case DType::F32: {
            const auto v = static_cast<float>(p.value);
            auto* d = static_cast<float*>(out.data());
            for (int64_t i = 0; i < n; ++i) {
                d[i] = v;
            }
            return;
        }
        case DType::I64: {
            const auto v = static_cast<int64_t>(p.value);
            auto* d = static_cast<int64_t*>(out.data());
            for (int64_t i = 0; i < n; ++i) {
                d[i] = v;
            }
            return;
        }
        case DType::I32: {
            const auto v = static_cast<int32_t>(p.value);
            auto* d = static_cast<int32_t*>(out.data());
            for (int64_t i = 0; i < n; ++i) {
                d[i] = v;
            }
            return;
        }
        case DType::Bool: {
            const uint8_t v = p.value != 0.0 ? 1 : 0;
            std::memset(out.data(), v, static_cast<size_t>(n));
            return;
        }
        default: unsupported_dtype(out);
    }
}

void k_arange(Node& out) {
    const auto p = out.params.get<ArangeParams>();
    const int64_t n = out.shape.numel();

    if (out.dtype == DType::F32) {
        auto* d = static_cast<float*>(out.data());
        for (int64_t i = 0; i < n; ++i) {
            // start + i*step rather than a running accumulator: the accumulator
            // compounds rounding error across the range, and torch.arange does
            // not.
            d[i] = static_cast<float>(p.start + static_cast<double>(i) * p.step);
        }
        return;
    }
    if (out.dtype == DType::I64) {
        auto* d = static_cast<int64_t*>(out.data());
        for (int64_t i = 0; i < n; ++i) {
            d[i] = static_cast<int64_t>(p.start + static_cast<double>(i) * p.step);
        }
        return;
    }
    unsupported_dtype(out);
}

/// Uniform values in [0, 1), one per element, from a counter-based generator.
///
/// The value depends only on (seed, offset, linear index), so it is identical
/// on both backends and independent of how the work is divided -- see philox.h.
void k_rand(Node& out) {
    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    const auto p = out.params.get<RandParams>();
    const int64_t n = out.shape.numel();

    for (int64_t i = 0; i < n; ++i) {
        store<float>(out, i, philox_uniform(p.seed, p.offset, static_cast<uint64_t>(i)));
    }
}

/// Adjoint of a strided narrowing.
///
/// The output has the ORIGINAL extent; the gradient has the narrowed extent.
/// Every position the slice did not select receives zero, and the selected
/// positions receive the incoming gradient. Written as an explicit kernel
/// because scattering into strided positions cannot be expressed with the
/// elementwise and reduction ops.
///
/// Uses `=` rather than `+=` because a slice selects each source position at
/// most once (the step is required to be positive), so no two gradient elements
/// can land on the same destination. ScatterAdd, which has no such guarantee,
/// will need accumulation and is where the missing global float atomics on the
/// target GPU first matters.
void k_slice_backward(Node& out) {
    const Node& g = *out.src[0];
    const auto p = out.params.get<SliceParams>();

    VKML_ASSERT(out.shape.is_contiguous(), "slice_backward output must be contiguous");
    const size_t esz = dtype_size(out.dtype);

    std::memset(out.data(), 0, out.shape.nbytes());

    const int64_t n = g.shape.numel();
    if (n == 0) {
        return;
    }

    const auto* src = static_cast<const std::byte*>(g.data());
    auto* dst = static_cast<std::byte*>(out.data());
    const auto axis = static_cast<size_t>(p.axis);

    std::array<int64_t, kMaxDims> idx{};
    for (int64_t i = 0; i < n; ++i) {
        unravel(i, g.shape, idx);

        // Map the position within the slice back to the original extent.
        int64_t off = 0;
        for (int d = 0; d < out.shape.ndim(); ++d) {
            const size_t u = static_cast<size_t>(d);
            const int64_t j = (u == axis) ? p.start + idx[u] * p.step : idx[u];
            off += j * out.shape.stride(d);
        }
        std::memcpy(dst + off, src + linear_to_offset(i, g.shape), esz);
    }
}

/// Joins two operands along one axis.
///
/// The linear index is split into (outer, along, inner) around the
/// concatenated axis: `inner` is the product of the extents after it, `outer`
/// everything before, so an element's position along the axis is
/// `(i / inner) % extent`. Positions below the first operand's extent come from
/// it, the rest from the second with the offset subtracted.
///
/// Each source index is rebuilt using the *source's* own axis extent, which is
/// what makes this correct when the two differ -- reusing the output's extent
/// would read past the end of the shorter operand.
///
/// Byte-wise rather than typed: concatenation moves elements without
/// interpreting them, so one implementation serves every dtype.
void k_cat(Node& out) {
    const Node& a = *out.src[0];
    const Node& b = *out.src[1];
    VKML_ASSERT(out.dtype == a.dtype && out.dtype == b.dtype, "cat must not change dtype");

    const int64_t n = out.shape.numel();
    if (n == 0) {
        return;
    }

    const int axis = out.params.get<AxisParams>().axis;
    const int nd = out.shape.ndim();

    int64_t inner = 1;
    for (int i = axis + 1; i < nd; ++i) {
        inner *= out.shape.dim(i);
    }
    const int64_t out_extent = out.shape.dim(axis);
    const int64_t a_extent = a.shape.dim(axis);
    const int64_t b_extent = b.shape.dim(axis);

    const size_t esz = dtype_size(out.dtype);
    auto* dst = static_cast<std::byte*>(out.data());
    const auto* a_src = static_cast<const std::byte*>(a.data());
    const auto* b_src = static_cast<const std::byte*>(b.data());

    for (int64_t i = 0; i < n; ++i) {
        const int64_t rest = i % inner;
        const int64_t along = (i / inner) % out_extent;
        const int64_t outer = i / (inner * out_extent);

        const bool from_a = along < a_extent;
        const int64_t extent = from_a ? a_extent : b_extent;
        const int64_t pos = from_a ? along : along - a_extent;
        const int64_t linear = (outer * extent + pos) * inner + rest;

        const Node& src = from_a ? a : b;
        const std::byte* base = from_a ? a_src : b_src;
        std::memcpy(dst + linear_to_offset(i, out.shape),
                    base + linear_to_offset(linear, src.shape), esz);
    }
}

/// Splits a linear index around `axis` into (outer, along, inner).
///
/// `inner` is the product of the extents after the axis and `outer` everything
/// before, so an element's position along the axis is `(i / inner) % extent`.
/// Shared by cat, index_select and scatter_add, which all remap exactly one
/// axis and leave the rest alone.
struct AxisSplit {
    int64_t inner = 1;
    int64_t extent = 1;
};

[[nodiscard]] AxisSplit axis_split(const Shape& s, int axis) {
    AxisSplit r;
    for (int i = axis + 1; i < s.ndim(); ++i) {
        r.inner *= s.dim(i);
    }
    r.extent = s.dim(axis);
    return r;
}

/// Gathers rows named by an index vector. Embedding's forward pass.
void k_index_select(Node& out) {
    const Node& a = *out.src[0];
    const Node& idx = *out.src[1];
    const int axis = out.params.get<AxisParams>().axis;

    const int64_t n = out.shape.numel();
    if (n == 0) {
        return;
    }

    const AxisSplit o = axis_split(out.shape, axis);
    const int64_t a_extent = a.shape.dim(axis);
    const size_t esz = dtype_size(out.dtype);

    auto* dst = static_cast<std::byte*>(out.data());
    const auto* src = static_cast<const std::byte*>(a.data());

    for (int64_t i = 0; i < n; ++i) {
        const int64_t rest = i % o.inner;
        const int64_t k = (i / o.inner) % o.extent;
        const int64_t outer = i / (o.inner * o.extent);

        const int64_t j = load<int64_t>(idx, k);
        VKML_CHECK(j >= 0 && j < a_extent, IndexError,
                   "index_select: index[{}] = {} is out of range for extent {}", k, j, a_extent);

        const int64_t from = (outer * a_extent + j) * o.inner + rest;
        std::memcpy(dst + linear_to_offset(i, out.shape), src + linear_to_offset(from, a.shape),
                    esz);
    }
}

/// Adjoint of index_select: accumulate each source slice into the row it came
/// from. Repeated indices mean several sources land on one destination, which
/// is what makes this irreducible to the elementwise and reduction ops.
///
/// Walks the SOURCE in ascending linear order, so for any destination the
/// contributions arrive in ascending index order. That fixed order is what
/// makes the result bit-reproducible, and it is the same order the Vulkan
/// kernel uses -- deliberately, so the two agree exactly rather than merely
/// within a tolerance. Fixing it costs nothing here and is the only way to
/// have it at all on the GPU, which has no global float atomicAdd.
void k_scatter_add(Node& out) {
    const Node& src = *out.src[0];
    const Node& idx = *out.src[1];
    const int axis = out.params.get<AxisParams>().axis;

    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    VKML_ASSERT(out.shape.is_contiguous(), "scatter_add output must be contiguous");
    std::memset(out.data(), 0, out.shape.nbytes());

    const int64_t n = src.shape.numel();
    if (n == 0) {
        return;
    }

    const AxisSplit s = axis_split(src.shape, axis);
    const int64_t out_extent = out.shape.dim(axis);
    auto* dst = static_cast<float*>(out.data());

    for (int64_t i = 0; i < n; ++i) {
        const int64_t rest = i % s.inner;
        const int64_t k = (i / s.inner) % s.extent;
        const int64_t outer = i / (s.inner * s.extent);

        const int64_t j = load<int64_t>(idx, k);
        VKML_CHECK(j >= 0 && j < out_extent, IndexError,
                   "scatter_add: index[{}] = {} is out of range for extent {}", k, j, out_extent);

        dst[(outer * out_extent + j) * s.inner + rest] += load<float>(src, i);
    }
}

/// Window geometry, recomputed identically by both kernels and by the shader.
struct Window {
    int64_t out_h = 0;  ///< window positions down the image
    int64_t out_w = 0;  ///< window positions across it
    int64_t channels = 0;
};

[[nodiscard]] Window window_of(const UnfoldParams& p, int64_t channels_times_patch) {
    const auto span = [](int64_t extent, int k, int s, int pad, int d) {
        return (extent + 2LL * pad - (static_cast<int64_t>(d) * (k - 1) + 1)) / s + 1;
    };
    Window w;
    w.out_h = span(p.image_h, p.kernel_h, p.stride_h, p.pad_h, p.dilation_h);
    w.out_w = span(p.image_w, p.kernel_w, p.stride_w, p.pad_w, p.dilation_w);
    w.channels = channels_times_patch / (static_cast<int64_t>(p.kernel_h) * p.kernel_w);
    return w;
}

/// Extracts sliding blocks. One output element is read from exactly one image
/// position, or is zero when the window falls outside the padded image -- so
/// this is a gather and needs no accumulation.
void k_im2col(Node& out) {
    const Node& in = *out.src[0];
    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    const auto p = out.params.get<UnfoldParams>();
    const Window w = window_of(p, out.shape.dim(1));

    const int64_t n_batch = out.shape.dim(0);
    const int64_t patch = static_cast<int64_t>(p.kernel_h) * p.kernel_w;
    const int64_t positions = w.out_h * w.out_w;

    for (int64_t n = 0; n < n_batch; ++n) {
        for (int64_t row = 0; row < out.shape.dim(1); ++row) {
            const int64_t c = row / patch;
            const int64_t ki = (row % patch) / p.kernel_w;
            const int64_t kj = (row % patch) % p.kernel_w;

            for (int64_t pos = 0; pos < positions; ++pos) {
                const int64_t oh = pos / w.out_w;
                const int64_t ow = pos % w.out_w;
                const int64_t h = oh * p.stride_h - p.pad_h + ki * p.dilation_h;
                const int64_t x = ow * p.stride_w - p.pad_w + kj * p.dilation_w;

                const bool inside = h >= 0 && h < p.image_h && x >= 0 && x < p.image_w;
                const int64_t src = ((n * w.channels + c) * p.image_h + h) * p.image_w + x;
                const int64_t dst = (n * out.shape.dim(1) + row) * positions + pos;
                store<float>(out, dst, inside ? load<float>(in, src) : 0.0F);
            }
        }
    }
}

/// Adjoint of im2col: every window contribution is summed back into the image
/// position it came from.
///
/// Walks the OUTPUT and pulls, rather than walking the input and scattering.
/// Overlapping windows mean an image position receives several contributions,
/// and pulling makes each output element the property of one iteration -- so
/// the fold order is fixed by construction and no accumulation races. The
/// Vulkan kernel uses the identical traversal for the identical reason, which
/// is why the two agree bit-for-bit rather than within a tolerance.
///
/// The scan is bounded by the kernel size, not by the number of windows: for a
/// given image position only kernel_h * kernel_w window offsets can reach it,
/// and each determines at most one window.
void k_col2im(Node& out) {
    const Node& cols = *out.src[0];
    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    const auto p = out.params.get<UnfoldParams>();
    const Window w = window_of(p, cols.shape.dim(1));

    const int64_t patch = static_cast<int64_t>(p.kernel_h) * p.kernel_w;
    const int64_t positions = w.out_h * w.out_w;
    const int64_t total = out.shape.numel();

    for (int64_t i = 0; i < total; ++i) {
        const int64_t x = i % p.image_w;
        const int64_t h = (i / p.image_w) % p.image_h;
        const int64_t c = (i / (p.image_w * p.image_h)) % w.channels;
        const int64_t n = i / (p.image_w * p.image_h * w.channels);

        float acc = 0.0F;
        for (int64_t ki = 0; ki < p.kernel_h; ++ki) {
            const int64_t top = h + p.pad_h - ki * p.dilation_h;
            if (top < 0 || top % p.stride_h != 0) {
                continue;
            }
            const int64_t oh = top / p.stride_h;
            if (oh >= w.out_h) {
                continue;
            }
            for (int64_t kj = 0; kj < p.kernel_w; ++kj) {
                const int64_t left = x + p.pad_w - kj * p.dilation_w;
                if (left < 0 || left % p.stride_w != 0) {
                    continue;
                }
                const int64_t ow = left / p.stride_w;
                if (ow >= w.out_w) {
                    continue;
                }
                const int64_t row = c * patch + ki * p.kernel_w + kj;
                acc += load<float>(cols,
                                   (n * cols.shape.dim(1) + row) * positions + oh * w.out_w + ow);
            }
        }
        store<float>(out, i, acc);
    }
}

/// Position of the first maximum in one pooling window, as a flat offset into
/// the window's (kernel_h x kernel_w) grid, or -1 if the window lies entirely
/// in the padding.
///
/// Scanned in row-major order with a strict `>`, so the FIRST maximum wins --
/// matching torch, and matching k_argmax's convention. Using `>=` would silently
/// select the last, which changes where the gradient lands whenever a window
/// contains a tie.
///
/// Padding is -infinity rather than zero, so a window of negative values reports
/// its true maximum. This is the reason max pooling needs a kernel at all
/// instead of composing as a max over im2col, which pads with zero.
template <typename Load>
[[nodiscard]] int64_t argmax_in_window(const UnfoldParams& p, int64_t oh, int64_t ow,
                                       const Load& load_at) {
    float best = -std::numeric_limits<float>::infinity();
    int64_t best_offset = -1;

    for (int64_t ki = 0; ki < p.kernel_h; ++ki) {
        const int64_t h = oh * p.stride_h - p.pad_h + ki * p.dilation_h;
        if (h < 0 || h >= p.image_h) {
            continue;
        }
        for (int64_t kj = 0; kj < p.kernel_w; ++kj) {
            const int64_t x = ow * p.stride_w - p.pad_w + kj * p.dilation_w;
            if (x < 0 || x >= p.image_w) {
                continue;
            }
            const float v = load_at(h, x);
            if (v > best) {
                best = v;
                best_offset = ki * p.kernel_w + kj;
            }
        }
    }
    return best_offset;
}

void k_max_pool2d(Node& out) {
    const Node& in = *out.src[0];
    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    const auto p = out.params.get<UnfoldParams>();

    const int64_t out_h = out.shape.dim(2);
    const int64_t out_w = out.shape.dim(3);
    const int64_t planes = out.shape.dim(0) * out.shape.dim(1);

    for (int64_t plane = 0; plane < planes; ++plane) {
        const int64_t base = plane * p.image_h * p.image_w;
        auto load_at = [&](int64_t h, int64_t x) {
            return load<float>(in, base + h * p.image_w + x);
        };

        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                const int64_t offset = argmax_in_window(p, oh, ow, load_at);
                VKML_ASSERT(offset >= 0, "max_pool2d window at ({}, {}) is entirely padding", oh,
                            ow);
                const int64_t h = oh * p.stride_h - p.pad_h + (offset / p.kernel_w) * p.dilation_h;
                const int64_t x = ow * p.stride_w - p.pad_w + (offset % p.kernel_w) * p.dilation_w;
                store<float>(out, (plane * out_h + oh) * out_w + ow, load_at(h, x));
            }
        }
    }
}

/// Routes each window's gradient to the single input position that produced its
/// maximum.
///
/// Walks the OUTPUT -- the input-shaped gradient -- and pulls, for the same
/// reason col2im and scatter_add do: with a stride below the kernel extent one
/// input position can be the maximum of several windows, so a push would need
/// accumulation the target device cannot do deterministically. Pulling makes
/// every destination the property of one iteration, so the fold order is fixed
/// and both backends agree bit-for-bit.
///
/// src[0] is the incoming gradient and src[1] the original input, which is
/// needed because the argmax is recomputed rather than stored. Storing it would
/// need a second output per node, which the graph does not have and which is
/// not worth introducing for one operator.
void k_max_pool2d_backward(Node& out) {
    const Node& grad = *out.src[0];
    const Node& in = *out.src[1];
    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    const auto p = out.params.get<UnfoldParams>();

    const int64_t out_h = grad.shape.dim(2);
    const int64_t out_w = grad.shape.dim(3);
    const int64_t planes = out.shape.dim(0) * out.shape.dim(1);

    for (int64_t plane = 0; plane < planes; ++plane) {
        const int64_t base = plane * p.image_h * p.image_w;
        auto load_at = [&](int64_t h, int64_t x) {
            return load<float>(in, base + h * p.image_w + x);
        };

        for (int64_t h = 0; h < p.image_h; ++h) {
            for (int64_t x = 0; x < p.image_w; ++x) {
                float acc = 0.0F;

                // Only kernel_h * kernel_w window offsets can reach this
                // position, and each determines at most one window -- the same
                // bounded scan col2im uses.
                for (int64_t ki = 0; ki < p.kernel_h; ++ki) {
                    const int64_t top = h + p.pad_h - ki * p.dilation_h;
                    if (top < 0 || top % p.stride_h != 0) {
                        continue;
                    }
                    const int64_t oh = top / p.stride_h;
                    if (oh >= out_h) {
                        continue;
                    }
                    for (int64_t kj = 0; kj < p.kernel_w; ++kj) {
                        const int64_t left = x + p.pad_w - kj * p.dilation_w;
                        if (left < 0 || left % p.stride_w != 0) {
                            continue;
                        }
                        const int64_t ow = left / p.stride_w;
                        if (ow >= out_w) {
                            continue;
                        }
                        // This position receives the window's gradient only if
                        // it is that window's first maximum.
                        if (argmax_in_window(p, oh, ow, load_at) == ki * p.kernel_w + kj) {
                            acc += load<float>(grad, (plane * out_h + oh) * out_w + ow);
                        }
                    }
                }
                store<float>(out, base + h * p.image_w + x, acc);
            }
        }
    }
}

}  // namespace

void register_movement_kernels(KernelTable& t) {
    t[static_cast<size_t>(OpKind::MaxPool2d)] = k_max_pool2d;
    t[static_cast<size_t>(OpKind::MaxPool2dBackward)] = k_max_pool2d_backward;
    t[static_cast<size_t>(OpKind::Im2Col)] = k_im2col;
    t[static_cast<size_t>(OpKind::Col2Im)] = k_col2im;
    t[static_cast<size_t>(OpKind::Cat)] = k_cat;
    t[static_cast<size_t>(OpKind::IndexSelect)] = k_index_select;
    t[static_cast<size_t>(OpKind::ScatterAdd)] = k_scatter_add;
    t[static_cast<size_t>(OpKind::SliceBackward)] = k_slice_backward;
    t[static_cast<size_t>(OpKind::Contiguous)] = k_contiguous;
    t[static_cast<size_t>(OpKind::Cast)] = k_cast;
    t[static_cast<size_t>(OpKind::Full)] = k_full;
    t[static_cast<size_t>(OpKind::Arange)] = k_arange;
    t[static_cast<size_t>(OpKind::Rand)] = k_rand;
}

}  // namespace vkml::cpu
