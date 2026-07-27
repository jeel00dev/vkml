#pragma once

namespace vkml {

/// Whether operations should mark their outputs as requiring gradients.
///
/// Lives in `graph` rather than in `autograd` for a layering reason. Backward
/// rules are written in terms of forward ops, so `autograd` depends on `api`;
/// but `api` must consult this flag when building nodes, and cannot therefore
/// depend on `autograd`. Putting the flag at a level both can see keeps the
/// dependency graph acyclic without either layer knowing about the other.
///
/// The equivalent of torch's grad mode. Operations still execute when it is
/// off; they simply do not record that their result is differentiable, so no
/// backward graph accumulates.
void set_grad_enabled(bool enabled) noexcept;

[[nodiscard]] bool grad_enabled() noexcept;

}  // namespace vkml
