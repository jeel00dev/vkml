#include "iterate.h"
#include "kernels.h"

#include "vkml/util/assert.h"

#include <cmath>
#include <limits>
#include <format>

namespace vkml::cpu {
namespace {

// ---------------------------------------------------------------------------
// Activation functions
//
// These are the definitions PyTorch uses, written to match its numerics rather
// than its source. Where a naive formula is numerically poor, the stable form
// is used and the reason is stated -- per the project standard that a slower
// algorithm chosen for stability must justify itself.
// ---------------------------------------------------------------------------

/// Numerically stable logistic function.
///
/// The textbook 1/(1+exp(-x)) overflows exp() for x <= -88 in fp32 and returns
/// inf, then 1/inf = 0, which is the right answer by luck. For x very large
/// positive it is fine. The branch below avoids relying on that luck and keeps
/// the relative error small in both tails: for x < 0 it evaluates
/// e^x / (1 + e^x), whose argument is bounded above by 1.
[[nodiscard]] float sigmoid(float x) noexcept {
    if (x >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-x));
    }
    const float e = std::exp(x);
    return e / (1.0F + e);
}

/// Exact GELU, matching torch.nn.functional.gelu(approximate='none').
///
/// 0.5x(1 + erf(x/sqrt(2))). PyTorch's default is the erf form, not the tanh
/// approximation, and the two differ by up to ~1e-3 -- far outside the 1e-5
/// gate. Using the approximation here would produce a persistent mismatch that
/// looks like a bug.
[[nodiscard]] float gelu(float x) noexcept {
    constexpr float kInvSqrt2 = 0.70710678118654752440F;
    return 0.5F * x * (1.0F + std::erf(x * kInvSqrt2));
}

[[nodiscard]] float silu(float x) noexcept { return x * sigmoid(x); }

/// sign, matching torch on every input including the ones that surprise.
///
/// The fall-through returns +0.0, which covers +0.0, -0.0 AND NaN, because none
/// of them satisfies either comparison. It used to return `x`, with a comment
/// claiming that matched torch. Measured, torch agrees with neither half of that
/// claim: `torch.sign(-0.0)` is +0.0 (bits 0x00000000) and `torch.sign(nan)` is
/// +0.0. numpy differs from torch on NaN but agrees on -0.0, so the old
/// behaviour matched neither reference (issue #27).
[[nodiscard]] float sign(float x) noexcept {
    if (x > 0.0F) {
        return 1.0F;
    }
    if (x < 0.0F) {
        return -1.0F;
    }
    return 0.0F;
}

// ---------------------------------------------------------------------------
// Kernel bodies
// ---------------------------------------------------------------------------

// F16 IS STORAGE, NOT ARITHMETIC.
//
// Both helpers below widen a half to float, run the operation in float, and
// narrow only on the store. `ARCHITECTURE.md` §7.3 fixes this -- the tolerance
// for f16 (1e-3) is derived for "fp16 storage, fp32 accum" and would not hold
// if intermediates were rounded to 16 bits.
//
// The consequence worth stating: every kernel body keeps taking and returning
// `float`, so none of them changed to gain f16 support, and none of them *can*
// accumulate in 16 bits by accident. `Half` deliberately has no arithmetic
// operators (dtype.h), so an attempt to would not compile.

template <typename Op>
void unary_float(Node& out, Op op) {
    switch (out.dtype) {
        case DType::F32: map_unary<float, float>(out, *out.src[0], op); return;
        case DType::F16:
            map_unary<Half, Half>(out, *out.src[0],
                                  [&op](Half x) { return Half(op(x.to_float())); });
            return;
        default: unsupported_dtype(out);
    }
}

template <typename Op>
void binary_float(Node& out, Op op) {
    switch (out.dtype) {
        case DType::F32: map_binary<float, float>(out, *out.src[0], *out.src[1], op); return;
        case DType::F16:
            map_binary<Half, Half>(out, *out.src[0], *out.src[1], [&op](Half x, Half y) {
                return Half(op(x.to_float(), y.to_float()));
            });
            return;
        default: unsupported_dtype(out);
    }
}

template <typename Op>
void compare_float(Node& out, Op op) {
    // Comparisons consume floats and produce Bool, so the output element type
    // differs from the input element type -- and the dtype that has to be
    // checked is therefore the INPUT's.
    //
    // This previously asserted only that the output was Bool and then read the
    // inputs as float regardless. That is not an unsupported-dtype error, it is
    // a wrong answer: an f16 operand had its 2-byte halves read as 4-byte
    // floats, and an i64 operand had two elements read as one. Both returned a
    // plausible mask and neither raised. Found by recording which dtypes the
    // suite actually executes, and fixed here.
    VKML_ASSERT(out.dtype == DType::Bool, "comparison must produce Bool, got {}",
                dtype_name(out.dtype));

    const DType in = out.src[0]->dtype;
    VKML_DEBUG_ASSERT(in == out.src[1]->dtype, "comparison operands must share a dtype");

    switch (in) {
        case DType::F32:
            map_binary<uint8_t, float>(
                out, *out.src[0], *out.src[1],
                [op](float x, float y) -> uint8_t { return op(x, y) ? 1 : 0; });
            return;
        case DType::F16:
            map_binary<uint8_t, Half>(
                out, *out.src[0], *out.src[1],
                [op](Half x, Half y) -> uint8_t { return op(x.to_float(), y.to_float()) ? 1 : 0; });
            return;
        default:
            // Integer comparison would need integer kernels, which do not exist
            // (see the dtype contract in dtype.h). Raising is the honest answer;
            // reading them as floats is what was wrong before.
            unsupported_dtype(out, in);
    }
}

void k_neg(Node& o) {
    unary_float(o, [](float x) { return -x; });
}

void k_abs(Node& o) {
    unary_float(o, [](float x) { return std::fabs(x); });
}

void k_sign(Node& o) {
    unary_float(o, [](float x) { return sign(x); });
}

void k_square(Node& o) {
    unary_float(o, [](float x) { return x * x; });
}

void k_sqrt(Node& o) {
    unary_float(o, [](float x) { return std::sqrt(x); });
}

void k_rsqrt(Node& o) {
    unary_float(o, [](float x) { return 1.0F / std::sqrt(x); });
}

void k_reciprocal(Node& o) {
    unary_float(o, [](float x) { return 1.0F / x; });
}

void k_exp(Node& o) {
    unary_float(o, [](float x) { return std::exp(x); });
}

void k_log(Node& o) {
    unary_float(o, [](float x) { return std::log(x); });
}

void k_erf(Node& o) {
    unary_float(o, [](float x) { return std::erf(x); });
}

void k_sin(Node& o) {
    unary_float(o, [](float x) { return std::sin(x); });
}

void k_cos(Node& o) {
    unary_float(o, [](float x) { return std::cos(x); });
}

void k_tanh(Node& o) {
    unary_float(o, [](float x) { return std::tanh(x); });
}

void k_sigmoid(Node& o) {
    unary_float(o, [](float x) { return sigmoid(x); });
}

void k_relu(Node& o) {
    // `x <= 0 ? 0 : x`, NOT `x > 0 ? x : 0`. The two agree on every number and
    // differ on NaN: NaN fails BOTH comparisons, so the first form falls through
    // to x and propagates it, while the second falls through to 0 and destroys
    // it. torch propagates, and so do maximum(x, 0) and clamp_min(x, 0), which
    // are the same function spelled differently -- relu was the odd one out
    // (issue #27).
    unary_float(o, [](float x) { return x <= 0.0F ? 0.0F : x; });
}

void k_gelu(Node& o) {
    unary_float(o, [](float x) { return gelu(x); });
}

void k_silu(Node& o) {
    unary_float(o, [](float x) { return silu(x); });
}

void k_add(Node& o) {
    binary_float(o, [](float x, float y) { return x + y; });
}

void k_sub(Node& o) {
    binary_float(o, [](float x, float y) { return x - y; });
}

void k_mul(Node& o) {
    binary_float(o, [](float x, float y) { return x * y; });
}

void k_div(Node& o) {
    binary_float(o, [](float x, float y) { return x / y; });
}

void k_pow(Node& o) {
    binary_float(o, [](float x, float y) { return std::pow(x, y); });
}

void k_maximum(Node& o) {
    // std::fmax propagates the non-NaN operand; torch.maximum propagates NaN.
    binary_float(o, [](float x, float y) {
        if (std::isnan(x) || std::isnan(y)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return x > y ? x : y;
    });
}

void k_minimum(Node& o) {
    binary_float(o, [](float x, float y) {
        if (std::isnan(x) || std::isnan(y)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return x < y ? x : y;
    });
}

void k_equal(Node& o) {
    compare_float(o, [](float x, float y) { return x == y; });
}

void k_less(Node& o) {
    compare_float(o, [](float x, float y) { return x < y; });
}

void k_greater(Node& o) {
    compare_float(o, [](float x, float y) { return x > y; });
}

void k_less_equal(Node& o) {
    compare_float(o, [](float x, float y) { return x <= y; });
}

void k_greater_equal(Node& o) {
    compare_float(o, [](float x, float y) { return x >= y; });
}

void k_not_equal(Node& o) {
    compare_float(o, [](float x, float y) { return x != y; });
}

void k_clamp(Node& out) {
    const auto p = out.params.get<ClampParams>();
    unary_float(out, [p](float v) {
        // NaN passes through unchanged, matching torch.clamp.
        if (p.has_lo && v < p.lo) {
            return p.lo;
        }
        if (p.has_hi && v > p.hi) {
            return p.hi;
        }
        return v;
    });
}

/// Triangular mask over the last two axes.
///
/// Not written through map_unary because the predicate depends on *where* an
/// element is, not on its value. The row and column are recovered from the
/// linear index, which is exact because that index walks the logical shape in
/// row-major order regardless of how the operand is strided.
/// `T` is the storage type. A triangular mask selects or zeroes whole elements
/// rather than computing on them, so this moves the stored representation
/// directly instead of widening -- the same reasoning as k_where.
template <typename T>
void tri_impl(Node& out, bool upper) {
    const Node& in = *out.src[0];
    const auto p = out.params.get<TriParams>();

    const int nd = out.shape.ndim();
    const int64_t width = out.shape.dim(nd - 1);
    const int64_t height = out.shape.dim(nd - 2);
    const int64_t n = out.shape.numel();

    for (int64_t i = 0; i < n; ++i) {
        const int64_t col = i % width;
        const int64_t row = (i / width) % height;
        const int64_t offset = col - row;
        const bool keep = upper ? offset >= p.diagonal : offset <= p.diagonal;
        store<T>(out, i, keep ? load<T>(in, i) : T{});
    }
}

void k_tri(Node& out, bool upper) {
    switch (out.dtype) {
        case DType::F32: tri_impl<float>(out, upper); return;
        case DType::F16: tri_impl<Half>(out, upper); return;
        default: unsupported_dtype(out);
    }
}

void k_triu(Node& o) { k_tri(o, true); }

void k_tril(Node& o) { k_tri(o, false); }

void k_where(Node& out) {
    // src0 = condition (Bool), src1 = value if true, src2 = value if false.
    // All three are already broadcast to the output shape.
    const Node& cond = *out.src[0];
    const Node& a = *out.src[1];
    const Node& b = *out.src[2];
    const int64_t n = out.shape.numel();

    // Selection moves whole elements rather than computing on them, so unlike
    // the arithmetic kernels this one does not widen: it copies the stored
    // representation straight across. Doing otherwise would round every
    // selected f16 value through float and back for no reason.
    switch (out.dtype) {
        case DType::F32:
            for (int64_t i = 0; i < n; ++i) {
                store<float>(out, i,
                             load<uint8_t>(cond, i) != 0 ? load<float>(a, i) : load<float>(b, i));
            }
            return;
        case DType::F16:
            for (int64_t i = 0; i < n; ++i) {
                store<Half>(out, i,
                            load<uint8_t>(cond, i) != 0 ? load<Half>(a, i) : load<Half>(b, i));
            }
            return;
        default: unsupported_dtype(out);
    }
}

}  // namespace

void unsupported_dtype(const Node& node) { unsupported_dtype(node, node.dtype); }

void unsupported_dtype(const Node& node, DType dt) {
    throw DTypeError(std::format("cpu backend: op '{}' does not support dtype {}", op_name(node.op),
                                 dtype_name(dt)));
}

void register_elementwise_kernels(KernelTable& t) {
    t[static_cast<size_t>(OpKind::Neg)] = k_neg;
    t[static_cast<size_t>(OpKind::Abs)] = k_abs;
    t[static_cast<size_t>(OpKind::Sign)] = k_sign;
    t[static_cast<size_t>(OpKind::Square)] = k_square;
    t[static_cast<size_t>(OpKind::Sqrt)] = k_sqrt;
    t[static_cast<size_t>(OpKind::Rsqrt)] = k_rsqrt;
    t[static_cast<size_t>(OpKind::Reciprocal)] = k_reciprocal;
    t[static_cast<size_t>(OpKind::Exp)] = k_exp;
    t[static_cast<size_t>(OpKind::Log)] = k_log;
    t[static_cast<size_t>(OpKind::Erf)] = k_erf;
    t[static_cast<size_t>(OpKind::Sin)] = k_sin;
    t[static_cast<size_t>(OpKind::Cos)] = k_cos;
    t[static_cast<size_t>(OpKind::Tanh)] = k_tanh;
    t[static_cast<size_t>(OpKind::Sigmoid)] = k_sigmoid;
    t[static_cast<size_t>(OpKind::Relu)] = k_relu;
    t[static_cast<size_t>(OpKind::Gelu)] = k_gelu;
    t[static_cast<size_t>(OpKind::Silu)] = k_silu;
    t[static_cast<size_t>(OpKind::Clamp)] = k_clamp;

    t[static_cast<size_t>(OpKind::Add)] = k_add;
    t[static_cast<size_t>(OpKind::Sub)] = k_sub;
    t[static_cast<size_t>(OpKind::Mul)] = k_mul;
    t[static_cast<size_t>(OpKind::Div)] = k_div;
    t[static_cast<size_t>(OpKind::Pow)] = k_pow;
    t[static_cast<size_t>(OpKind::Maximum)] = k_maximum;
    t[static_cast<size_t>(OpKind::Minimum)] = k_minimum;

    t[static_cast<size_t>(OpKind::Equal)] = k_equal;
    t[static_cast<size_t>(OpKind::Less)] = k_less;
    t[static_cast<size_t>(OpKind::Greater)] = k_greater;
    t[static_cast<size_t>(OpKind::LessEqual)] = k_less_equal;
    t[static_cast<size_t>(OpKind::GreaterEqual)] = k_greater_equal;
    t[static_cast<size_t>(OpKind::NotEqual)] = k_not_equal;

    t[static_cast<size_t>(OpKind::Where)] = k_where;
    t[static_cast<size_t>(OpKind::Triu)] = k_triu;
    t[static_cast<size_t>(OpKind::Tril)] = k_tril;
}

}  // namespace vkml::cpu
