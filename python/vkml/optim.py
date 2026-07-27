"""Optimizers.

M0 note on where the update runs
--------------------------------
These implementations apply the update in Python by building tensor ops, which
is correct and easy to validate but costs one graph per parameter per step.

docs/ARCHITECTURE.md 0 (decision 3) calls for the update to become a *graph
node* -- ggml's ``ggml_opt_step_adamw`` model -- so that forward, backward and
update are one submission with no host round-trip. That matters on a discrete
GPU with 256 MiB of host-visible memory, and it lands with ``vkml.compile()`` at
M5. The ``SgdStep``/``AdamStep`` ops are already reserved in OpKind for it.

Keeping the Python version now means the graph-node version has a validated
reference to be checked against, rather than being the first implementation.
"""

from __future__ import annotations

from typing import Iterable

import vkml as V


class Optimizer:
    def __init__(self, params: Iterable[V.Tensor]):
        self.params = [p for p in params]
        if not self.params:
            raise ValueError("optimizer received an empty parameter list")
        for p in self.params:
            if not p.requires_grad:
                raise ValueError("optimizer received a parameter with requires_grad=False")

    def zero_grad(self) -> None:
        """Clear accumulated gradients.

        Gradients accumulate rather than overwrite (PyTorch's rule), so this
        must be called between steps unless accumulation is intended.
        """
        for p in self.params:
            p.grad = V.Tensor()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: list[V.Tensor | None] = [None] * len(self.params)

    def step(self) -> None:
        # The update itself must not be recorded on the autograd tape: it is a
        # mutation of the parameters, not part of the function being
        # differentiated. Detaching the result is what keeps step N's graph from
        # being retained by step N+1.
        with V.no_grad():
            for i, p in enumerate(self.params):
                g = p.grad
                if not g.defined():
                    continue

                if self.weight_decay != 0.0:
                    g = g + p.detach() * self.weight_decay

                if self.momentum != 0.0:
                    v = self._velocity[i]
                    v = g if v is None else (v * self.momentum + g)
                    self._velocity[i] = v.detach().realize()
                    g = self._velocity[i]

                # In place: the Module still holds this exact Tensor, so
                # rebinding self.params[i] would update the optimizer's view and
                # leave the model untouched.
                p.assign_(p.detach() - g * self.lr)


class Adam(Optimizer):
    """Adam with bias correction, matching torch.optim.Adam defaults."""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: list[V.Tensor | None] = [None] * len(self.params)
        self._v: list[V.Tensor | None] = [None] * len(self.params)
        self._t = 0

    def step(self) -> None:
        self._t += 1
        # Bias correction is applied to the step size rather than to m and v
        # individually: algebraically identical, one fewer tensor op per
        # parameter, and it is what torch does.
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        with V.no_grad():
            for i, p in enumerate(self.params):
                g = p.grad
                if not g.defined():
                    continue

                if self.weight_decay != 0.0:
                    g = g + p.detach() * self.weight_decay

                m = self._m[i]
                v = self._v[i]
                m = g * (1.0 - self.beta1) if m is None else (m * self.beta1 + g * (1.0 - self.beta1))
                v = (g * g) * (1.0 - self.beta2) if v is None else (
                    v * self.beta2 + (g * g) * (1.0 - self.beta2)
                )
                self._m[i] = m.detach().realize()
                self._v[i] = v.detach().realize()

                m_hat = self._m[i] / bc1
                v_hat = self._v[i] / bc2
                p.assign_(p.detach() - m_hat * self.lr / (V.sqrt(v_hat) + self.eps))
