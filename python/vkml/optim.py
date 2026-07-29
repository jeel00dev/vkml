"""Optimizers.

Where the update runs
---------------------
The update is built from tensor ops, which are graph nodes, so it already
executes on the device -- an Adam step issues ~16 dispatches per parameter and
no host arithmetic.

What it does not yet achieve is docs/ARCHITECTURE.md 0 (decision 3): the whole
forward, backward and update as ONE submission. The submissions come from the
explicit ``.realize()`` calls below plus ``assign_``, not from the number of
nodes, so fusing the update into a single kernel would not deliver it.

Two halves, and an earlier version of this note only had one of them. The
smaller half is how often Python forces materialisation. The larger half was
that ``assign_`` moved every parameter through HOST memory even though source
and destination were on the same device -- three submissions per assignment
against one for the same arithmetic. That is fixed
(docs/adr/0006-lazy-assign-and-submission-batching.md, stage A): a same-device
assignment is now a device-side copy, worth one submission per parameter per
step, measured 1.6x on an SGD step and 1.5x on Adam.

What remains is that ``assign_`` is still EAGER, so it cuts the graph and caps
how much can be batched. Stage B of that ADR makes it a graph node and realises
every parameter together.

Dedicated ``SgdStep``/``AdamStep`` OpKinds were previously reserved for a fused
version. They were removed: a reservation is a speculative abstraction, the
measurement above shows fusion is not what the goal needs, and re-adding an
enumerator later is a one-line change made with evidence behind it.

The realize calls exist to cut the graph between steps, without which each
step would build on the last and grow without bound. Batching them is a real
and available improvement -- but "so a step is one submission" is the wrong
target and was dropped: ggml-vulkan submits several times per graph on purpose,
to overlap command recording with execution, and caps submission size per device
because a large one can trip a driver timeout on weaker AMD parts. Few
submissions with the GPU kept busy is the goal.
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
    """SGD with optional classical or Nesterov momentum, matching torch.optim.SGD.

    torch also takes `dampening`, which this does not. Nesterov requires
    dampening == 0 there, so the two features never combine, and adding an
    option whose only supported value is its default would be noise. Add it when
    something needs it.
    """

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        super().__init__(params)
        if nesterov and momentum == 0.0:
            # torch rejects this too. Nesterov looks AHEAD along the momentum
            # buffer, so with no momentum there is nothing to look along and
            # the update silently degenerates to plain SGD -- which is worse
            # than an error, because the run appears to work.
            raise ValueError("nesterov momentum requires momentum != 0")
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
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
                    # Classical momentum steps ALONG the buffer. Nesterov steps
                    # along it and then one more momentum-step further, which is
                    # the "look ahead" -- so it uses the current gradient again
                    # rather than replacing it. `g` here is the gradient after
                    # weight decay, which is what torch feeds in too.
                    g = (g + self._velocity[i] * self.momentum) if self.nesterov \
                        else self._velocity[i]

                # In place: the Module still holds this exact Tensor, so
                # rebinding self.params[i] would update the optimizer's view and
                # leave the model untouched.
                p.assign_(p.detach() - g * self.lr)


class RMSProp(Optimizer):
    """RMSProp, matching torch.optim.RMSprop.

    The running average starts at zero, as torch's does, so the first step is
    ``(1 - alpha) * g**2`` rather than ``g**2``. That difference persists for
    many steps through the exponential average, so it is not a detail.
    """

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False):
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self._square_avg: list[V.Tensor | None] = [None] * len(self.params)
        self._grad_avg: list[V.Tensor | None] = [None] * len(self.params)
        self._buffer: list[V.Tensor | None] = [None] * len(self.params)

    def step(self) -> None:
        with V.no_grad():
            for i, p in enumerate(self.params):
                g = p.grad
                if not g.defined():
                    continue

                if self.weight_decay != 0.0:
                    g = g + p.detach() * self.weight_decay

                sq = self._square_avg[i]
                sq = (g * g) * (1.0 - self.alpha) if sq is None else (
                    sq * self.alpha + (g * g) * (1.0 - self.alpha)
                )
                self._square_avg[i] = sq.detach().realize()
                variance = self._square_avg[i]

                if self.centered:
                    # Subtracting the squared mean gives the variance rather
                    # than the second moment, which stops a large constant
                    # gradient from shrinking the step it deserves.
                    ga = self._grad_avg[i]
                    ga = g * (1.0 - self.alpha) if ga is None else (
                        ga * self.alpha + g * (1.0 - self.alpha)
                    )
                    self._grad_avg[i] = ga.detach().realize()
                    variance = variance - self._grad_avg[i] * self._grad_avg[i]

                direction = g / (V.sqrt(variance) + self.eps)

                if self.momentum != 0.0:
                    buf = self._buffer[i]
                    buf = direction if buf is None else (buf * self.momentum + direction)
                    self._buffer[i] = buf.detach().realize()
                    direction = self._buffer[i]

                p.assign_(p.detach() - direction * self.lr)


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


class AdamW(Adam):
    """Adam with *decoupled* weight decay, matching torch.optim.AdamW.

    The only difference from Adam is where the decay is applied, and it is not
    cosmetic. Adam adds ``wd * p`` to the gradient, so the decay then passes
    through the second-moment normalisation and is scaled by ``1/sqrt(v)`` --
    parameters with small gradients get decayed far harder than intended.
    AdamW subtracts ``lr * wd * p`` from the parameter directly, leaving the
    adaptive step to act on the gradient alone.

    torch's default weight_decay is 1e-2 here, not the 0.0 Adam uses; decoupled
    decay is the reason the optimiser exists, so defaulting it off would be a
    trap.
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2):
        # Adam's own weight_decay stays at zero: this class never routes decay
        # through the gradient, which is the entire distinction.
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        self.decoupled_weight_decay = weight_decay

    def step(self) -> None:
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        with V.no_grad():
            for i, p in enumerate(self.params):
                g = p.grad
                if not g.defined():
                    continue

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

                # Decay first, then the adaptive step -- torch's order, and the
                # two do not commute because the decay changes what p is.
                decayed = p.detach()
                if self.decoupled_weight_decay != 0.0:
                    decayed = decayed - decayed * (self.lr * self.decoupled_weight_decay)

                p.assign_(decayed - m_hat * self.lr / (V.sqrt(v_hat) + self.eps))
