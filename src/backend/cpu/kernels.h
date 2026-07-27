#pragma once

#include "vkml/graph/node.h"

#include <array>

namespace vkml::cpu {

/// One CPU kernel.
///
/// A plain function pointer rather than std::function: the table is fixed at
/// startup, and a function pointer keeps the per-node dispatch to one indirect
/// call with no allocation. Whether a backend supports an op is then simply
/// "is this slot non-null", which is what `Backend::supports` reports.
using Kernel = void (*)(Node&);

using KernelTable = std::array<Kernel, kNumOps>;

// Each translation unit fills in the slots it owns. Splitting registration this
// way keeps the kernel files independent of each other and of the backend.
void register_movement_kernels(KernelTable& table);
void register_elementwise_kernels(KernelTable& table);
void register_reduce_kernels(KernelTable& table);
void register_matmul_kernels(KernelTable& table);

/// Throws DTypeError naming the op, for kernels restricted to a dtype subset.
[[noreturn]] void unsupported_dtype(const Node& node);

}  // namespace vkml::cpu
