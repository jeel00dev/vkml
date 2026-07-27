#pragma once

#include "vkml/api/tensor.h"
#include "vkml/graph/grad_mode.h"

namespace vkml {

/// Reverse-mode differentiation.
///
/// THE CENTRAL DESIGN CHOICE
/// -------------------------
/// Every backward rule is written in terms of *forward* operations. `d(a*b)/da`
/// is `mul(grad, b)` -- an ordinary Mul node appended to the graph -- not a call
/// into a dedicated `mul_backward` kernel.
///
/// This is ggml's model (`ggml_compute_backward` in ggml.c) and tinygrad's, and
/// it is the reason the project needs ~64 kernels rather than ~120
/// (docs/ARCHITECTURE.md §3 Fork 2). Three consequences follow for free:
///
///   - the backward pass reuses the executor, allocator and every kernel,
///     so a bug fixed in `mul` is fixed in the gradient of `mul`;
///   - higher-order derivatives need no new machinery, since the backward graph
///     is an ordinary graph that can itself be differentiated;
///   - gradient checkpointing becomes "re-emit that subgraph".
///
/// The cost is that a fused backward kernel would be perhaps 10-20 % faster on
/// some ops. That is the right thing to trade away, and individual ops can be
/// fused later as a pure optimisation with no API change.

/// Accumulates gradients into every leaf reachable from `root` that has
/// requires_grad set.
///
/// `root` must be a scalar; the seed gradient is 1. This matches
/// `torch.Tensor.backward()` with no arguments.
///
/// Gradients ACCUMULATE into `.grad` rather than replacing it, as in PyTorch,
/// which is what makes gradient accumulation across micro-batches work. Callers
/// are responsible for zeroing (`optim.zero_grad()`).
void backward(const Tensor& root);

/// As above, with an explicit seed gradient for a non-scalar root.
void backward(const Tensor& root, const Tensor& seed);

/// A tensor sharing the same data but with no gradient history.
[[nodiscard]] Tensor detach(const Tensor& t);

// set_grad_enabled / grad_enabled live in graph/grad_mode.h so that `api` can
// consult them without depending on this layer. Re-exported here by include.

/// RAII form, for scoped use.
class NoGradGuard {
public:
    NoGradGuard() noexcept;
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
    NoGradGuard(NoGradGuard&&) = delete;
    NoGradGuard& operator=(NoGradGuard&&) = delete;

private:
    bool previous_;
};

}  // namespace vkml
