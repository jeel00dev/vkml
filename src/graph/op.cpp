#include "vkml/graph/op.h"

#include <array>

namespace vkml {
namespace {

// Indexed by OpKind. The static_assert below is what keeps this table honest:
// adding an enum entry without a name is a compile error, not a "?" at runtime.
constexpr std::array<std::string_view, kNumOps> kOpNames = {
    "input",
    "const",

    "full",
    "arange",

    "reshape",
    "permute",
    "slice",
    "broadcast",
    "squeeze",
    "unsqueeze",

    "contiguous",
    "cast",
    "cat",

    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "maximum",
    "minimum",

    "eq",
    "lt",
    "gt",
    "le",
    "ge",
    "ne",

    "neg",
    "abs",
    "sign",
    "square",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "exp",
    "log",
    "erf",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "relu",
    "gelu",
    "silu",
    "clamp",

    "sum",
    "mean",
    "max",
    "min",
    "prod",
    "argmax",
    "argmin",

    "matmul",

    "softmax",
    "log_softmax",

    "im2col",
    "col2im",
    "max_pool2d",

    "max_pool2d_backward",

    "index_select",
    "scatter_add",
    "slice_backward",

    "where",
    "dropout",
    "triu",
    "tril",
};

static_assert(kOpNames.size() == static_cast<size_t>(OpKind::Count),
              "kOpNames is out of sync with OpKind -- every op needs a name");

}  // namespace

std::string_view op_name(OpKind op) noexcept {
    const auto i = static_cast<size_t>(op);
    if (i >= kOpNames.size()) {
        return "<invalid>";
    }
    return kOpNames[i];
}

}  // namespace vkml
