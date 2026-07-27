#pragma once

#include "vkml/graph/node.h"

#include <span>

namespace vkml {

/// Evaluates `roots` and everything they depend on.
///
/// Three phases, in order:
///   1. schedule  -- topological_order, which skips already-realised subgraphs
///   2. bind      -- give every scheduled node storage; view nodes alias their
///                   base rather than allocating
///   3. compute   -- hand the whole batch to the backend in one call
///
/// Phase 2 is separated from phase 3 so that the backend never decides where a
/// tensor lives. That is what lets the M5 memory planner replace phase 2
/// wholesale -- assigning offsets inside one large buffer instead of one
/// allocation per node -- without any backend changing.
///
/// M0 LIMITATION, deliberate and temporary: phase 2 allocates every
/// intermediate before phase 3 begins, so peak memory is the sum of all
/// intermediates rather than the true high-water mark. Correct, and fine at
/// MNIST scale; replacing it is precisely what M5 exists for.
void realize(std::span<const NodePtr> roots);

void realize(const NodePtr& root);

/// Eager mode: realise after every operation instead of at observation points.
///
/// Required, not a nicety. Under lazy evaluation a bad kernel surfaces at the
/// realize() call, arbitrarily far from the op that caused it; the per-operator
/// validation suite (docs/ARCHITECTURE.md §7.4) runs in eager mode so a failure
/// names the operator directly. Also settable with VKML_EAGER=1.
void set_eager(bool enabled) noexcept;

[[nodiscard]] bool eager() noexcept;

}  // namespace vkml
