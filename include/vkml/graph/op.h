#pragma once

#include "vkml/core/dtype.h"
#include "vkml/core/shape.h"  // kMaxDims, used by PermuteParams

#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>

namespace vkml {

/// Every operation the graph can express.
///
/// This is the inventory from docs/ARCHITECTURE.md §6, stated in full even
/// though M0 only implements a subset. Keeping the enum complete makes it a
/// table of contents for the project: `op_name` and the dispatch table are both
/// exhaustive over it, so an unimplemented op fails with a clear
/// NotImplementedError rather than silently missing.
///
/// Note what is NOT here: there are no `*_backward` entries. Backward passes
/// are built from these same forward ops (docs/ARCHITECTURE.md §3 Fork 2),
/// which is what keeps the kernel count at ~64 instead of ~120. The only
/// exceptions are ScatterAdd, Col2Im, MaxPool2dBackward and SliceBackward,
/// which genuinely cannot be composed and appear here as ordinary forward ops.
/// SliceBackward is the adjoint of a strided narrowing: it scatters a gradient
/// into a zero-filled tensor of the original extent, which no combination of
/// the elementwise/reduction ops can express without an index kernel.
enum class OpKind : uint16_t {
    // -- leaves ------------------------------------------------------------
    Input = 0,  ///< externally supplied storage
    Const,      ///< materialised constant

    // -- creation ----------------------------------------------------------
    Full,
    Arange,

    // -- views: zero-copy, produce a new Shape over the same storage --------
    Reshape,
    Permute,
    Slice,
    Broadcast,
    Squeeze,
    Unsqueeze,

    // -- movement: allocate and copy ---------------------------------------
    Contiguous,
    Cast,
    Cat,

    // -- binary elementwise ------------------------------------------------
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,

    // -- comparison (produce Bool) -----------------------------------------
    Equal,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    NotEqual,

    // -- unary elementwise -------------------------------------------------
    Neg,
    Abs,
    Sign,
    Square,
    Sqrt,
    Rsqrt,
    Reciprocal,
    Exp,
    Log,
    Erf,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Relu,
    Gelu,
    Silu,
    Clamp,

    // -- reductions --------------------------------------------------------
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    ArgMax,
    ArgMin,

    // -- linear algebra ----------------------------------------------------
    Matmul,

    // -- composite ---------------------------------------------------------
    Softmax,
    LogSoftmax,
    BatchNorm,

    // -- convolution / pooling ---------------------------------------------
    Im2Col,
    Col2Im,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    MaxPool2dBackward,

    // -- indexing ----------------------------------------------------------
    IndexSelect,
    ScatterAdd,
    SliceBackward,

    // -- loss --------------------------------------------------------------

    // -- misc --------------------------------------------------------------
    Where,
    Dropout,
    Triu,
    Tril,

    Count
};

inline constexpr int kNumOps = static_cast<int>(OpKind::Count);

/// Maximum inputs to any node. Four covers the widest op in scope (fused
/// attention: q, k, v, mask). ggml uses 10; sizing to actual need keeps Node
/// smaller and more cache-friendly.
inline constexpr int kMaxSrc = 4;

[[nodiscard]] std::string_view op_name(OpKind op) noexcept;

/// True for ops that only reinterpret their input's layout. A view node shares
/// its source's storage and must never be written to.
[[nodiscard]] constexpr bool is_view_op(OpKind op) noexcept {
    switch (op) {
        case OpKind::Reshape:
        case OpKind::Permute:
        case OpKind::Slice:
        case OpKind::Broadcast:
        case OpKind::Squeeze:
        case OpKind::Unsqueeze: return true;
        default: return false;
    }
}

/// True for ops that produce a Bool result regardless of input dtype.
[[nodiscard]] constexpr bool is_comparison_op(OpKind op) noexcept {
    switch (op) {
        case OpKind::Equal:
        case OpKind::Less:
        case OpKind::Greater:
        case OpKind::LessEqual:
        case OpKind::GreaterEqual:
        case OpKind::NotEqual: return true;
        default: return false;
    }
}

// ---------------------------------------------------------------------------
// Op parameters
//
// A fixed 64-byte inline buffer, following ggml's `int32_t op_params[16]`. The
// point is that a Node stays a fixed-size object with no owned heap allocation,
// which is what makes graph construction cheap and Node arrays contiguous.
//
// vkml improves on ggml's raw int32 array by giving each op a named POD struct
// and type-checked accessors, so a mis-indexed parameter is a compile error
// rather than a silently wrong number.
// ---------------------------------------------------------------------------

inline constexpr size_t kOpParamBytes = 64;

struct OpParams {
    alignas(8) std::array<std::byte, kOpParamBytes> raw{};

    template <typename T>
    void set(const T& value) noexcept {
        static_assert(std::is_trivially_copyable_v<T>, "op params must be trivially copyable");
        static_assert(sizeof(T) <= kOpParamBytes, "op params exceed the inline buffer");
        std::memcpy(raw.data(), &value, sizeof(T));
    }

    template <typename T>
    [[nodiscard]] T get() const noexcept {
        static_assert(std::is_trivially_copyable_v<T>, "op params must be trivially copyable");
        static_assert(sizeof(T) <= kOpParamBytes, "op params exceed the inline buffer");
        T value{};
        std::memcpy(&value, raw.data(), sizeof(T));
        return value;
    }

    friend bool operator==(const OpParams& a, const OpParams& b) noexcept { return a.raw == b.raw; }
};

/// Axes are carried as a bitmask rather than a list: rank is at most 4, so 4
/// bits suffice, and a mask makes "is axis i reduced?" a single test inside the
/// kernel loop.
struct ReduceParams {
    uint32_t axes_mask = 0;
    bool keepdim = false;
};

struct SliceParams {
    int32_t axis = 0;
    int64_t start = 0;
    int64_t stop = 0;
    int64_t step = 1;
};

struct PermuteParams {
    std::array<int32_t, kMaxDims> perm{};
};

struct AxisParams {
    int32_t axis = 0;
};

struct CastParams {
    DType target = DType::F32;
};

/// Offset of the retained diagonal for triu/tril, following torch: 0 is the
/// main diagonal, positive moves toward the upper-right, negative toward the
/// lower-left. Which side is kept comes from the OpKind, not from here.
struct TriParams {
    int32_t diagonal = 0;
};

struct ClampParams {
    float lo = 0.0F;
    float hi = 0.0F;
    bool has_lo = false;
    bool has_hi = false;
};

struct FullParams {
    double value = 0.0;  ///< double so an i64 fill value survives the round trip
};

struct ArangeParams {
    double start = 0.0;
    double step = 1.0;
};

struct NormParams {
    int32_t normalized_axes = 1;  ///< number of trailing axes to normalise over
    float eps = 1e-5F;
};

static_assert(sizeof(SliceParams) <= kOpParamBytes);

}  // namespace vkml
