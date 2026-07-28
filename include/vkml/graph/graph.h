#pragma once

#include "vkml/graph/node.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vkml {

/// Nodes that must be evaluated to realise `roots`, in dependency order
/// (every node appears after all of its sources).
///
/// Already-realised nodes are treated as leaves: they are not returned and
/// their sources are not traversed. That is what makes repeated `.realize()`
/// calls cheap and what stops a long-lived tensor from dragging its entire
/// construction history along behind it.
///
/// Iterative rather than recursive on purpose. A recursive DFS would overflow
/// the stack on deep graphs, and an unrolled RNN over a long sequence is
/// exactly that -- one of the model families this project targets.
[[nodiscard]] std::vector<Node*> topological_order(std::span<const NodePtr> roots);

/// Convenience overload for a single root.
[[nodiscard]] std::vector<Node*> topological_order(const NodePtr& root);

/// Number of times each node in `order` is consumed by a later node in `order`.
///
/// This is the input the memory planner needs in M5: a buffer becomes reusable
/// the moment its use count reaches zero. ggml computes the same thing as
/// `n_children` in ggml-alloc.c. It is computed here, in the graph layer, so
/// that the planner and any future scheduler share one definition.
[[nodiscard]] std::vector<int> compute_use_counts(std::span<Node* const> order);

/// Renders the graph as Graphviz DOT, for debugging.
[[nodiscard]] std::string to_dot(std::span<const NodePtr> roots);

}  // namespace vkml
