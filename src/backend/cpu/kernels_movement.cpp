#include "iterate.h"
#include "kernels.h"

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

}  // namespace

void register_movement_kernels(KernelTable& t) {
    t[static_cast<size_t>(OpKind::SliceBackward)] = k_slice_backward;
    t[static_cast<size_t>(OpKind::Contiguous)] = k_contiguous;
    t[static_cast<size_t>(OpKind::Cast)] = k_cast;
    t[static_cast<size_t>(OpKind::Full)] = k_full;
    t[static_cast<size_t>(OpKind::Arange)] = k_arange;
}

}  // namespace vkml::cpu
