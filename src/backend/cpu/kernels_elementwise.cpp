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

[[nodiscard]] float sign(float x) noexcept {
    if (x > 0.0F) {
        return 1.0F;
    }
    if (x < 0.0F) {
        return -1.0F;
    }
    return x;  // preserves -0.0 and NaN, as torch.sign does
}

// ---------------------------------------------------------------------------
// Kernel bodies
// ---------------------------------------------------------------------------

template <typename Op>
void unary_f32(Node& out, Op op) {
    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    map_unary<float, float>(out, *out.src[0], op);
}

template <typename Op>
void binary_f32(Node& out, Op op) {
    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    map_binary<float, float>(out, *out.src[0], *out.src[1], op);
}

template <typename Op>
void compare_f32(Node& out, Op op) {
    // Comparisons consume floats and produce Bool, so the output element type
    // differs from the input element type.
    VKML_ASSERT(out.dtype == DType::Bool, "comparison must produce Bool, got {}",
                dtype_name(out.dtype));
    map_binary<uint8_t, float>(out, *out.src[0], *out.src[1],
                               [op](float x, float y) -> uint8_t { return op(x, y) ? 1 : 0; });
}

void k_neg(Node& o) {
    unary_f32(o, [](float x) { return -x; });
}

void k_abs(Node& o) {
    unary_f32(o, [](float x) { return std::fabs(x); });
}

void k_sign(Node& o) {
    unary_f32(o, [](float x) { return sign(x); });
}

void k_square(Node& o) {
    unary_f32(o, [](float x) { return x * x; });
}

void k_sqrt(Node& o) {
    unary_f32(o, [](float x) { return std::sqrt(x); });
}

void k_rsqrt(Node& o) {
    unary_f32(o, [](float x) { return 1.0F / std::sqrt(x); });
}

void k_reciprocal(Node& o) {
    unary_f32(o, [](float x) { return 1.0F / x; });
}

void k_exp(Node& o) {
    unary_f32(o, [](float x) { return std::exp(x); });
}

void k_log(Node& o) {
    unary_f32(o, [](float x) { return std::log(x); });
}

void k_erf(Node& o) {
    unary_f32(o, [](float x) { return std::erf(x); });
}

void k_sin(Node& o) {
    unary_f32(o, [](float x) { return std::sin(x); });
}

void k_cos(Node& o) {
    unary_f32(o, [](float x) { return std::cos(x); });
}

void k_tanh(Node& o) {
    unary_f32(o, [](float x) { return std::tanh(x); });
}

void k_sigmoid(Node& o) {
    unary_f32(o, [](float x) { return sigmoid(x); });
}

void k_relu(Node& o) {
    unary_f32(o, [](float x) { return x > 0.0F ? x : 0.0F; });
}

void k_gelu(Node& o) {
    unary_f32(o, [](float x) { return gelu(x); });
}

void k_silu(Node& o) {
    unary_f32(o, [](float x) { return silu(x); });
}

void k_add(Node& o) {
    binary_f32(o, [](float x, float y) { return x + y; });
}

void k_sub(Node& o) {
    binary_f32(o, [](float x, float y) { return x - y; });
}

void k_mul(Node& o) {
    binary_f32(o, [](float x, float y) { return x * y; });
}

void k_div(Node& o) {
    binary_f32(o, [](float x, float y) { return x / y; });
}

void k_pow(Node& o) {
    binary_f32(o, [](float x, float y) { return std::pow(x, y); });
}

void k_maximum(Node& o) {
    // std::fmax propagates the non-NaN operand; torch.maximum propagates NaN.
    binary_f32(o, [](float x, float y) {
        if (std::isnan(x) || std::isnan(y)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return x > y ? x : y;
    });
}

void k_minimum(Node& o) {
    binary_f32(o, [](float x, float y) {
        if (std::isnan(x) || std::isnan(y)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return x < y ? x : y;
    });
}

void k_equal(Node& o) {
    compare_f32(o, [](float x, float y) { return x == y; });
}

void k_less(Node& o) {
    compare_f32(o, [](float x, float y) { return x < y; });
}

void k_greater(Node& o) {
    compare_f32(o, [](float x, float y) { return x > y; });
}

void k_less_equal(Node& o) {
    compare_f32(o, [](float x, float y) { return x <= y; });
}

void k_greater_equal(Node& o) {
    compare_f32(o, [](float x, float y) { return x >= y; });
}

void k_not_equal(Node& o) {
    compare_f32(o, [](float x, float y) { return x != y; });
}

void k_clamp(Node& out) {
    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    const auto p = out.params.get<ClampParams>();
    map_unary<float, float>(out, *out.src[0], [p](float v) {
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

void k_where(Node& out) {
    // src0 = condition (Bool), src1 = value if true, src2 = value if false.
    // All three are already broadcast to the output shape.
    if (out.dtype != DType::F32) {
        unsupported_dtype(out);
    }
    const Node& cond = *out.src[0];
    const Node& a = *out.src[1];
    const Node& b = *out.src[2];

    const int64_t n = out.shape.numel();
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t c = load<uint8_t>(cond, i);
        store<float>(out, i, c != 0 ? load<float>(a, i) : load<float>(b, i));
    }
}

}  // namespace

void unsupported_dtype(const Node& node) {
    throw DTypeError(std::format("cpu backend: op '{}' does not support dtype {}", op_name(node.op),
                                 dtype_name(node.dtype)));
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
}

}  // namespace vkml::cpu
