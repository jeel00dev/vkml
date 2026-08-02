"""Optimizers.

Where the update runs
---------------------
The update is built from tensor ops, which are graph nodes, so it already
executes on the device -- an Adam step issues ~16 dispatches per parameter and
no host arithmetic.

Why a step is three passes and not one loop
-------------------------------------------
Every parameter's update is independent, so all of them can share a submission.
They did not, because the loop realised each parameter's state and assigned it
before touching the next -- and a submission costs the host about 80 µs against
20 µs of GPU work for the kernels an optimiser runs, measured.

``step()`` is therefore split, as tinygrad's is
(``Tensor.realize(*self.schedule_step())``):

    pass 1   build EVERY parameter's state update lazily, realise them together
    pass 2   build EVERY parameter's new value from that state, realise together
    pass 3   assign

Measured on the CIFAR-100 CNN's 8 parameters, minimum of 40 warm steps:

    per-parameter loop     24 submissions   1.836 ms
    passes 1 and 2         10 submissions   1.146 ms   1.60x

Parameters are **bit-identical** across 60 steps, which is what makes this a
scheduling change rather than a numerical one.

**Fewer submissions is not the same as faster, and this measured it.** Batching
only pass 1 gives 17 submissions and 2.123 ms -- seven fewer submissions and
*slower* than doing nothing. Both passes have to batch before the saving
appears, because pass 2 is where most of the submissions were. Anyone tempted
to treat submission count as the objective should read that line first.

The order matters for a reason that is easy to lose. ``detach()`` realises its
source when the source is not yet computed -- it shares the source's buffer and
an unbound node has none to share -- so detaching BEFORE the batched realise
cuts the graph exactly where the batching was supposed to help
(docs/adr/0006, finding 1). State is detached in pass 2, after realisation, and
that ordering is the whole trick.

What this still does not achieve
--------------------------------
``docs/ARCHITECTURE.md`` 0 (decision 3): the whole forward, backward and update
as ONE submission. ``assign_`` is still EAGER, so pass 3 costs one submission
per parameter and caps the step at 1 + 1 + N. Stage B of
``docs/adr/0006-lazy-assign-and-submission-batching.md`` makes assign a graph
node, which is what removes the N; two ADR-sized changes sit in front of it.

Stage A of that ADR is already in: ``assign_`` was a device-to-host-to-device
round trip and is now a device-side copy, worth one submission per parameter
per step (1.6x on an SGD step, 1.5x on Adam).

Dedicated ``SgdStep``/``AdamStep`` OpKinds were previously reserved for a fused
version. They were removed: a reservation is a speculative abstraction, the
measurement above shows fusion is not what the goal needs, and re-adding an
enumerator later is a one-line change made with evidence behind it.

"A step is one submission" remains the wrong target. ggml-vulkan submits several
times per graph on purpose, to overlap command recording with execution, and
caps submission size per device because a large one can trip a driver timeout on
weaker AMD parts. Few submissions with the GPU kept busy is the goal.
"""

from __future__ import annotations

from typing import Callable, Iterable

import vkml as V

#: What `Optimizer._plan` hands back: the state tensors to realise, and a
#: callable that -- once they are realised -- stores them and returns the
#: parameter's new value.
Plan = tuple[list["V.Tensor"], Callable[[], "V.Tensor"]]


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
        """Apply one update to every parameter, in three batched passes.

        Subclasses implement `_plan` and do not override this. The batching is
        one piece of subtle ordering (see the module docstring), and four copies
        of it would be four chances to get it wrong.
        """
        with V.no_grad():
            self._begin_step()

            plans: list[tuple[V.Tensor, Plan]] = []
            for index, param in enumerate(self.params):
                grad = param.grad
                if not grad.defined():
                    continue
                plans.append((param, self._plan(index, param, self._gradient(param, grad))))

            state = [tensor for _, (tensors, _) in plans for tensor in tensors]
            if state:
                V.realize(*state)

            # Only now: `finish` detaches, and detaching an uncomputed node
            # forces it, which would undo the batching above.
            values = [finish() for _, (_, finish) in plans]
            if values:
                V.realize(*values)

            for (param, _), value in zip(plans, values):
                # In place: the Module still holds this exact Tensor, so
                # rebinding self.params[i] would update the optimizer's view and
                # leave the model untouched.
                param.assign_(value)

    def _begin_step(self) -> None:
        """Per-step bookkeeping that is not per-parameter. Adam's `t` lives here."""

    def _gradient(self, param: V.Tensor, grad: V.Tensor) -> V.Tensor:
        """The gradient the update should use, with coupled weight decay applied.

        One place, because `_plan` needs it and so does the value it builds, and
        applying it in both is how the two would come to disagree.
        """
        decay = getattr(self, "weight_decay", 0.0)
        return grad + param.detach() * decay if decay != 0.0 else grad

    def _plan(self, index: int, param: V.Tensor, grad: V.Tensor) -> Plan:
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

    def _plan(self, index, param, grad):
        # The update itself must not be recorded on the autograd tape: it is a
        # mutation of the parameters, not part of the function being
        # differentiated. Detaching the state is what keeps step N's graph from
        # being retained by step N+1.
        if self.momentum == 0.0:
            return [], lambda: param.detach() - grad * self.lr

        previous = self._velocity[index]
        velocity = grad if previous is None else (previous * self.momentum + grad)

        def finish():
            self._velocity[index] = velocity.detach()
            # Classical momentum steps ALONG the buffer. Nesterov steps along it
            # and then one more momentum-step further, which is the "look
            # ahead" -- so it uses the current gradient again rather than
            # replacing it. `grad` here is the gradient after weight decay,
            # which is what torch feeds in too.
            direction = (grad + self._velocity[index] * self.momentum) if self.nesterov \
                else self._velocity[index]
            return param.detach() - direction * self.lr

        return [velocity], finish


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

    def _plan(self, index, param, grad):
        square = self._square_avg[index]
        square = (grad * grad) * (1.0 - self.alpha) if square is None else (
            square * self.alpha + (grad * grad) * (1.0 - self.alpha)
        )
        pending = [square]

        average = None
        if self.centered:
            # Subtracting the squared mean gives the variance rather than the
            # second moment, which stops a large constant gradient from
            # shrinking the step it deserves.
            average = self._grad_avg[index]
            average = grad * (1.0 - self.alpha) if average is None else (
                average * self.alpha + grad * (1.0 - self.alpha)
            )
            pending.append(average)

        # The momentum buffer depends on the two above. Built here anyway and
        # realised with them: multi-root realize shares the common subgraph, so
        # the dependency costs a dispatch and not a submission.
        buffer = None
        if self.momentum != 0.0:
            variance = square - average * average if self.centered else square
            direction = grad / (V.sqrt(variance) + self.eps)
            previous = self._buffer[index]
            buffer = direction if previous is None else (previous * self.momentum + direction)
            pending.append(buffer)

        def finish():
            self._square_avg[index] = square.detach()
            if self.centered:
                self._grad_avg[index] = average.detach()
            if self.momentum != 0.0:
                self._buffer[index] = buffer.detach()
                return param.detach() - self._buffer[index] * self.lr

            spread = self._square_avg[index]
            if self.centered:
                spread = spread - self._grad_avg[index] * self._grad_avg[index]
            return param.detach() - (grad / (V.sqrt(spread) + self.eps)) * self.lr

        return pending, finish


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

    def _begin_step(self) -> None:
        self._t += 1
        # Bias correction is applied to the step size rather than to m and v
        # individually: algebraically identical, one fewer tensor op per
        # parameter, and it is what torch does.
        self._bc1 = 1.0 - self.beta1 ** self._t
        self._bc2 = 1.0 - self.beta2 ** self._t

    def _moments(self, index, grad):
        """The two running moments, built lazily. Shared with AdamW."""
        m = self._m[index]
        v = self._v[index]
        m = grad * (1.0 - self.beta1) if m is None else (
            m * self.beta1 + grad * (1.0 - self.beta1))
        v = (grad * grad) * (1.0 - self.beta2) if v is None else (
            v * self.beta2 + (grad * grad) * (1.0 - self.beta2)
        )
        return m, v

    def _adaptive_step(self, index):
        """The update direction, from moments that are already realised."""
        m_hat = self._m[index] / self._bc1
        v_hat = self._v[index] / self._bc2
        return m_hat * self.lr / (V.sqrt(v_hat) + self.eps)

    def _plan(self, index, param, grad):
        m, v = self._moments(index, grad)

        def finish():
            self._m[index] = m.detach()
            self._v[index] = v.detach()
            return param.detach() - self._adaptive_step(index)

        return [m, v], finish


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

    def _plan(self, index, param, grad):
        m, v = self._moments(index, grad)

        def finish():
            self._m[index] = m.detach()
            self._v[index] = v.detach()

            # Decay first, then the adaptive step -- torch's order, and the two
            # do not commute because the decay changes what p is.
            decayed = param.detach()
            if self.decoupled_weight_decay != 0.0:
                decayed = decayed - decayed * (self.lr * self.decoupled_weight_decay)
            return decayed - self._adaptive_step(index)

        return [m, v], finish
