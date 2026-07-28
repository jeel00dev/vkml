"""Neural network layers.

Written in Python rather than C++ on purpose (docs/ARCHITECTURE.md 4.1): a
Module holds references and iterates dicts, none of which is hot. The hot part
is the tensor ops it calls, which are already in C++.

The API mirrors ``torch.nn`` closely enough that PyTorch code reads across
unchanged -- ``named_parameters``, ``state_dict``, ``zero_grad``, ``__call__``.
The structure (named children plus named params, recursive with a dotted
prefix) is the same one stable-diffusion.cpp rebuilt on top of ggml as
``GGMLBlock``, and it exists so that parameter names line up with a
``state_dict`` for loading and comparison.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterator

import numpy as _np

import vkml as V

# Weight initialisation draws from here, so that seeding it makes a model
# reproducible. Layers must not call default_rng() themselves: an unseeded
# generator gives every run different weights, which makes a training result
# impossible to reproduce and a divergence impossible to investigate.
#
# Computation is deterministic regardless (docs/ARCHITECTURE.md); this is about
# the one place randomness legitimately enters.
_INIT_RNG = _np.random.default_rng()


def manual_seed(seed: int) -> None:
    """Make subsequent weight initialisation reproducible.

    Mirrors torch.manual_seed in spirit, not in stream: the two libraries draw
    from different generators by design (docs/ARCHITECTURE.md 7.2), so equal
    seeds do not give equal weights. To compare against torch, copy a
    state_dict rather than seeding both.
    """
    global _INIT_RNG
    _INIT_RNG = _np.random.default_rng(seed)


class Module:
    """Base class for layers."""

    def __init__(self):
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    # -- attribute plumbing -------------------------------------------------

    def __setattr__(self, name, value):
        if isinstance(value, V.Tensor):
            self._parameters[name] = value
            return
        if isinstance(value, Module):
            self._modules[name] = value
            return
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # Only called when normal lookup fails, so parameters, buffers and
        # submodules resolve without shadowing real attributes.
        params = self.__dict__.get("_parameters")
        if params is not None and name in params:
            return params[name]
        buffers = self.__dict__.get("_buffers")
        if buffers is not None and name in buffers:
            return buffers[name]
        mods = self.__dict__.get("_modules")
        if mods is not None and name in mods:
            return mods[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def register_buffer(self, name: str, tensor: V.Tensor) -> None:
        """Record persistent state that is NOT trained.

        A buffer travels with the module -- it appears in `state_dict`, so it
        saves, loads and interoperates with a torch checkpoint -- but never in
        `parameters()`, so an optimiser cannot see it. Batch normalisation's
        running statistics are the motivating case: they carry no gradient, and
        letting an optimiser "train" them would destroy the estimate.

        Assigning a Tensor attribute makes a PARAMETER; this is the explicit
        opt-out, exactly as in torch.
        """
        if tensor.requires_grad:
            raise ValueError(
                f"buffer {name!r} has requires_grad=True; buffers are not trained"
            )
        self._buffers[name] = tensor

    # -- traversal ----------------------------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, V.Tensor]]:
        for name, p in self._parameters.items():
            yield (prefix + name, p)
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + ".")

    def parameters(self) -> Iterator[V.Tensor]:
        for _, p in self.named_parameters():
            yield p

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, V.Tensor]]:
        for name, b in self._buffers.items():
            yield (prefix + name, b)
        for name, m in self._modules.items():
            yield from m.named_buffers(prefix + name + ".")

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        yield (prefix.rstrip("."), self)
        for name, m in self._modules.items():
            yield from m.named_modules(prefix + name + ".")

    def state_dict(self) -> dict[str, _np.ndarray]:
        """Parameters and buffers, by dotted name.

        Buffers are included because that is what makes a checkpoint complete:
        a batch-normalised model restored without its running statistics
        evaluates against the wrong distribution while looking perfectly
        healthy. It is also what torch does, so the two round-trip.
        """
        out = {name: p.numpy() for name, p in self.named_parameters()}
        out.update({name: b.numpy() for name, b in self.named_buffers()})
        return out

    def load_state_dict(self, state: dict[str, _np.ndarray]) -> None:
        """Copy values in by name.

        Used heavily by the validation suite: a PyTorch model's ``state_dict``
        is loaded into the vkml model so that a training comparison starts from
        byte-identical weights rather than from a re-derived random init.
        """
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        own = {**params, **buffers}

        missing = set(own) - set(state)
        unexpected = set(state) - set(own)
        if missing or unexpected:
            raise KeyError(
                f"state_dict mismatch; missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )

        for name, existing in own.items():
            # Keep each entry's own dtype rather than forcing f32: a counter
            # buffer is integral, and silently converting it would round-trip
            # wrongly.
            #
            # asarray, NOT ascontiguousarray: the latter promotes a 0-d array to
            # rank 1, which would reject a scalar buffer against its own shape.
            # A 0-d array is always contiguous, so the branch below never fires
            # for one.
            value = _np.asarray(state[name], dtype=existing.numpy().dtype)
            if not value.flags["C_CONTIGUOUS"]:
                value = value.copy(order="C")
            if tuple(value.shape) != tuple(existing.shape):
                raise V.ShapeError(
                    f"'{name}' expects shape {tuple(existing.shape)}, got {tuple(value.shape)}"
                )
            replacement = V.tensor(value, requires_grad=existing.requires_grad)
            if name in params:
                self._replace_param(name, replacement)
            else:
                self._replace_buffer(name, replacement)

    def _resolve(self, dotted: str) -> tuple["Module", str]:
        target = self
        *path, leaf = dotted.split(".")
        for part in path:
            target = target._modules[part]
        return target, leaf

    def _replace_param(self, dotted: str, new: V.Tensor) -> None:
        target, leaf = self._resolve(dotted)
        target._parameters[leaf] = new

    def _replace_buffer(self, dotted: str, new: V.Tensor) -> None:
        target, leaf = self._resolve(dotted)
        target._buffers[leaf] = new

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = V.Tensor()

    def to(self, device) -> "Module":
        """Move every parameter and buffer to `device`, in place.

        CALL THIS BEFORE CONSTRUCTING AN OPTIMISER. The optimiser captures the
        parameter list when it is built, and this replaces each parameter with a
        new tensor -- so an optimiser made first would keep updating the old
        ones while the model used the new. torch has the same ordering
        constraint for the same reason.

        Transfer goes through the host, because that is what a transfer to a
        discrete device is: the data has to cross the bus either way. A direct
        device-to-device path would only matter with a second accelerator
        backend, which does not exist yet.

        Gradients move with their parameters. Dropping them would leave a
        subsequent optimiser step silently updating nothing.
        """
        for name, p in list(self.named_parameters()):
            moved = V.tensor(p.numpy(), device=device, requires_grad=p.requires_grad)
            if p.grad.defined():
                moved.grad = V.tensor(p.grad.numpy(), device=device)
            self._replace_param(name, moved)

        for name, b in list(self.named_buffers()):
            self._replace_buffer(name, V.tensor(b.numpy(), device=device))
        return self

    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # -- interface ----------------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not implement forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        if not self._modules:
            return f"{type(self).__name__}()"
        inner = "\n".join(
            f"  ({k}): " + repr(v).replace("\n", "\n  ") for k, v in self._modules.items()
        )
        return f"{type(self).__name__}(\n{inner}\n)"


class Linear(Module):
    """y = x @ W^T + b, matching torch.nn.Linear.

    Weight is stored as ``(out_features, in_features)`` and transposed in
    forward, exactly as PyTorch does. That layout is not arbitrary: it means a
    PyTorch ``state_dict`` loads without any transposition, which keeps the
    validation comparison honest.

    Initialisation follows torch's Kaiming-uniform default so that an
    independently-initialised vkml model has the same *distribution* as a torch
    one -- though the validation suite copies weights rather than relying on it
    (RNG parity is explicitly not a goal, docs/ARCHITECTURE.md 7.2).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        bound = 1.0 / math.sqrt(in_features)
        rng = _INIT_RNG
        w = rng.uniform(-bound, bound, size=(out_features, in_features)).astype(_np.float32)
        self.weight = V.tensor(w, requires_grad=True)

        if bias:
            b = rng.uniform(-bound, bound, size=(out_features,)).astype(_np.float32)
            self.bias = V.tensor(b, requires_grad=True)
        else:
            self.bias = None

    def __setattr__(self, name, value):
        if name == "bias" and value is None:
            object.__setattr__(self, "_has_bias", False)
            return
        if name == "bias":
            object.__setattr__(self, "_has_bias", True)
        super().__setattr__(name, value)

    def forward(self, x: V.Tensor) -> V.Tensor:
        y = V.matmul(x, self.weight.transpose(-2, -1))
        if getattr(self, "_has_bias", False):
            y = y + self.bias
        return y

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={getattr(self, '_has_bias', False)})"
        )


class ReLU(Module):
    def forward(self, x: V.Tensor) -> V.Tensor:
        return V.relu(x)


class Tanh(Module):
    def forward(self, x: V.Tensor) -> V.Tensor:
        return V.tanh(x)


class Sigmoid(Module):
    def forward(self, x: V.Tensor) -> V.Tensor:
        return V.sigmoid(x)


class GELU(Module):
    def forward(self, x: V.Tensor) -> V.Tensor:
        return V.gelu(x)


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer

    def forward(self, x: V.Tensor) -> V.Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, i: int) -> Module:
        return self._modules[str(i)]


# ---------------------------------------------------------------------------
# Losses
#
# Thin wrappers that translate torch's string `reduction` into the enum the
# C++ API takes. The numerics -- log-sum-exp stability, the tolerance policy,
# the one-hot masking -- live there and are stated once, so these cannot drift
# away from the operators the validation suite actually pins.
# ---------------------------------------------------------------------------

_REDUCTIONS = {
    "mean": V.Reduction.mean,
    "sum": V.Reduction.sum,
    "none": V.Reduction.none,
}


def _reduction(name: str) -> "V.Reduction":
    try:
        return _REDUCTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown reduction {name!r}; expected one of {sorted(_REDUCTIONS)}"
        ) from None


def mse_loss(pred: V.Tensor, target: V.Tensor, reduction: str = "mean") -> V.Tensor:
    return V.mse_loss(pred, target, _reduction(reduction))


def cross_entropy(logits: V.Tensor, target: V.Tensor, reduction: str = "mean") -> V.Tensor:
    """Softmax cross-entropy from logits against I64 class indices.

    TAKES CLASS INDICES, as torch does. An earlier version took a one-hot
    target, because selecting by index needed a gather kernel that did not yet
    exist. It does now, so the signature matches torch and callers no longer
    build an encoding the library can do itself.
    """
    return V.cross_entropy(logits, target, _reduction(reduction))


# ---------------------------------------------------------------------------
# Layers with state that is not a parameter
#
# BatchNorm's running statistics and Dropout's RNG offset are both mutated
# across calls and neither is trained. They live here as ordinary Python
# attributes and are updated with assign_() under no_grad(), which is the same
# arrangement the optimisers use for their moment buffers -- the graph has no
# notion of state that survives a step, and giving it one for two layers would
# be a large change for a small need.
# ---------------------------------------------------------------------------


class BatchNorm2d(Module):
    """Batch normalisation over (N, C, H, W), matching torch.nn.BatchNorm2d.

    Training uses the batch's own statistics and updates a running estimate;
    evaluation uses that estimate. Which one applies is `self.training`, so a
    caller switches behaviour with `.train()` / `.eval()` rather than by passing
    a flag.

    TWO VARIANCE ESTIMATORS, deliberately. The batch is normalised with the
    BIASED variance (divide by N) while the running estimate accumulates the
    UNBIASED one (divide by N-1). That is torch's behaviour, verified, and the
    asymmetry is principled: the biased figure is the right normaliser for the
    batch in hand, the unbiased one the right estimator of the population.
    Using one for both makes evaluation drift away from training as the running
    estimate converges to the wrong value -- which a single-step comparison
    cannot see, so it is pinned by a test that runs many.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = V.tensor(_np.ones(num_features, dtype=_np.float32), requires_grad=True)
            self.bias = V.tensor(_np.zeros(num_features, dtype=_np.float32), requires_grad=True)

        if track_running_stats:
            self.register_buffer("running_mean",
                                 V.tensor(_np.zeros(num_features, dtype=_np.float32)))
            self.register_buffer("running_var",
                                 V.tensor(_np.ones(num_features, dtype=_np.float32)))
            # Present for state_dict interop and DELIBERATELY NOT MAINTAINED.
            #
            # Every torch BatchNorm state_dict carries this key, so the buffer
            # has to exist or load_state_dict rejects the checkpoint. Nothing in
            # vkml reads it: torch uses it only for momentum=None, the
            # cumulative-average mode this layer does not offer.
            #
            # Keeping it accurate would cost a host round-trip per forward pass
            # -- I64 arithmetic is unimplemented on both backends, so the
            # increment cannot happen on the device, and reading the counter
            # back to add one is exactly the per-step synchronisation this
            # project spends effort avoiding. That is a poor trade for a value
            # with no reader.
            #
            # Consequence, stated: a vkml checkpoint loaded into torch reports
            # zero batches. Revisit if momentum=None is ever supported, or if
            # I64 elementwise arithmetic lands and the increment becomes free.
            self.register_buffer("num_batches_tracked", V.tensor(_np.array(0, dtype=_np.int64)))

    def forward(self, x: V.Tensor) -> V.Tensor:
        if x.ndim != 4:
            raise V.ShapeError(f"BatchNorm2d expects (N, C, H, W), got rank {x.ndim}")
        if x.shape[1] != self.num_features:
            raise V.ShapeError(
                f"BatchNorm2d({self.num_features}) received {x.shape[1]} channels"
            )

        weight = self._parameters.get("weight", V.Tensor())
        bias = self._parameters.get("bias", V.Tensor())

        if not self.training and self.track_running_stats:
            return V.batch_norm(x, self.running_mean, self.running_var, weight, bias, self.eps)

        # Everything except the channel axis is reduced over.
        axes = [0, 2, 3]
        mean = V.mean(x, axes)
        centred = x - mean.reshape([1, self.num_features, 1, 1])
        biased_var = V.mean(V.square(centred), axes)

        if self.track_running_stats:
            self._update_running_stats(x, mean, biased_var)

        return V.batch_norm(x, mean, biased_var, weight, bias, self.eps)

    def _update_running_stats(self, x: V.Tensor, mean: V.Tensor, biased_var: V.Tensor) -> None:
        """Exponential average of the batch statistics, torch's convention.

        Detached and assigned in place: this is bookkeeping about the data seen
        so far, not part of the function being differentiated, and letting it
        onto the tape would keep every past batch's graph alive.
        """
        samples = x.size // self.num_features
        # The unbiased estimate, from the biased one: var_unbiased =
        # var_biased * n/(n-1). Undefined for a single sample, where torch
        # leaves the running estimate untouched rather than dividing by zero.
        if samples < 2:
            return
        correction = samples / (samples - 1)

        with V.no_grad():
            m = self.momentum
            # Operators rather than V.mul: the scalar overloads exist in C++ but
            # only the tensor-tensor form is bound, and `*` routes through the
            # scalar path already.
            self.running_mean.assign_(self.running_mean * (1.0 - m) + mean.detach() * m)
            self.running_var.assign_(
                self.running_var * (1.0 - m) + biased_var.detach() * (m * correction)
            )


    def __repr__(self) -> str:
        return (f"BatchNorm2d({self.num_features}, eps={self.eps}, "
                f"momentum={self.momentum}, affine={self.affine})")


class Dropout(Module):
    """Zeroes elements with probability `p` during training, scaling the rest.

    ADVANCES AN OFFSET ON EVERY CALL. The underlying `rand` is a pure function
    of (seed, offset, index), so a module that reused one offset would drop the
    SAME elements at every step -- silently, while the loss curve still looked
    plausible. The counter is what makes successive masks independent, and
    there is a test that two consecutive calls differ.

    Seeding from a module-local counter rather than a global stream keeps the
    whole thing reproducible: the same seed replays the same run.
    """

    def __init__(self, p: float = 0.5, seed: int = 0):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.seed = seed
        object.__setattr__(self, "_offset", 0)

    def forward(self, x: V.Tensor) -> V.Tensor:
        if not self.training or self.p == 0.0:
            return x
        offset = self._offset
        object.__setattr__(self, "_offset", offset + 1)
        return V.dropout(x, self.p, self.seed, offset, True)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


# ---------------------------------------------------------------------------
# Shape and lookup
# ---------------------------------------------------------------------------


class Flatten(Module):
    """Collapses the axes from `start_dim` onward, matching torch.nn.Flatten.

    The default keeps axis 0 as the batch, which is what a convolutional stack
    needs before its first Linear.
    """

    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: V.Tensor) -> V.Tensor:
        dims = list(x.shape)
        head = dims[: self.start_dim]
        tail = 1
        for d in dims[self.start_dim:]:
            tail *= d
        return x.contiguous().reshape(head + [tail])

    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim})"


class Embedding(Module):
    """A lookup table, matching torch.nn.Embedding.

    Forward is a gather; the gradient accumulates every occurrence of a token
    back onto its row, which is what scatter_add exists for. Both come from the
    operator layer, so this module holds only the table.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        rng = _INIT_RNG
        w = rng.normal(0.0, 1.0, size=(num_embeddings, embedding_dim)).astype(_np.float32)
        self.weight = V.tensor(w, requires_grad=True)

    def forward(self, indices: V.Tensor) -> V.Tensor:
        # index_select takes a rank-1 index, so a batched lookup is flattened
        # and the original shape restored afterwards.
        shape = list(indices.shape)
        flat = indices.contiguous().reshape([-1])
        rows = V.index_select(self.weight, 0, flat)
        return rows.reshape(shape + [self.embedding_dim])

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


# ---------------------------------------------------------------------------
# Convolution and pooling
# ---------------------------------------------------------------------------


def _pair(value) -> tuple[int, int]:
    """Accept an int or a pair, as torch's conv and pooling layers do."""
    if isinstance(value, int):
        return (value, value)
    a, b = value
    return (int(a), int(b))


class Conv2d(Module):
    """2D convolution, matching torch.nn.Conv2d.

    Weight layout is (out_channels, in_channels, kh, kw) -- torch's, so a
    state_dict loads without transposition, for the same reason Linear stores
    (out, in).

    Groups are not supported; the operator rejects a mismatched channel count
    rather than reinterpreting it.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        # torch's default: uniform over +-1/sqrt(fan_in), fan_in counting the
        # whole receptive field.
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = 1.0 / math.sqrt(fan_in)
        rng = _INIT_RNG
        w = rng.uniform(-bound, bound,
                        size=(out_channels, in_channels, *self.kernel_size)).astype(_np.float32)
        self.weight = V.tensor(w, requires_grad=True)

        if bias:
            b = rng.uniform(-bound, bound, size=(out_channels,)).astype(_np.float32)
            self.bias = V.tensor(b, requires_grad=True)

    def forward(self, x: V.Tensor) -> V.Tensor:
        return V.conv2d(x, self.weight, self._parameters.get("bias", V.Tensor()),
                        self.stride, self.padding, self.dilation)

    def __repr__(self) -> str:
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding})")


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        # torch: stride defaults to the kernel, which the operator spells as 0.
        self.stride = (0, 0) if stride is None else _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def forward(self, x: V.Tensor) -> V.Tensor:
        return V.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)

    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = (0, 0) if stride is None else _pair(stride)
        self.padding = _pair(padding)

    def forward(self, x: V.Tensor) -> V.Tensor:
        return V.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

    def __repr__(self) -> str:
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


# ---------------------------------------------------------------------------
# Normalisation without batch statistics
# ---------------------------------------------------------------------------


class LayerNorm(Module):
    """Layer normalisation over the trailing axes, matching torch.nn.LayerNorm.

    Stateless: the statistics come from the sample itself, so there is nothing
    to track and no train/eval distinction.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = V.tensor(_np.ones(self.normalized_shape, dtype=_np.float32),
                                   requires_grad=True)
            self.bias = V.tensor(_np.zeros(self.normalized_shape, dtype=_np.float32),
                                 requires_grad=True)

    def forward(self, x: V.Tensor) -> V.Tensor:
        y = V.layer_norm(x, len(self.normalized_shape), self.eps)
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return y

    def __repr__(self) -> str:
        return f"LayerNorm({list(self.normalized_shape)}, eps={self.eps})"


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class MultiheadAttention(Module):
    """Scaled dot-product attention over several heads.

    Parameter layout follows torch.nn.MultiheadAttention exactly -- a packed
    ``in_proj_weight`` of shape (3E, E) plus an ``out_proj`` submodule -- so a
    torch state_dict loads without rearrangement. That is not cosmetic: it is
    what lets the validation suite compare against torch's own implementation
    rather than against a reference written here, which would only prove the two
    agree with each other.

    TWO DELIBERATE DIVERGENCES FROM TORCH, both documented and pinned by tests:

    - ``batch_first`` defaults to True. torch defaults to False, meaning
      (S, B, E), which is a legacy layout almost every caller overrides.
    - Returns the output tensor alone, not ``(output, weights)``. The averaged
      per-head weights torch returns second are a debugging aid, and a tuple
      that is nearly always destructured-and-discarded is worse to use.
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True,
                 batch_first: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        bound = 1.0 / math.sqrt(embed_dim)
        rng = _INIT_RNG
        self.in_proj_weight = V.tensor(
            rng.uniform(-bound, bound, size=(3 * embed_dim, embed_dim)).astype(_np.float32),
            requires_grad=True,
        )
        if bias:
            self.in_proj_bias = V.tensor(_np.zeros(3 * embed_dim, dtype=_np.float32),
                                         requires_grad=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def _split_heads(self, x: V.Tensor) -> V.Tensor:
        """(B, S, E) -> (B, H, S, head_dim), which is what a batched matmul needs.

        Rank 4 is the maximum this library expresses, and this is exactly at it:
        two batch axes (B, H) over a 2-D matmul. Adding a fifth axis anywhere in
        attention would not fit.
        """
        batch, seq, _ = x.shape
        return x.contiguous().reshape([batch, seq, self.num_heads, self.head_dim]) \
                .permute([0, 2, 1, 3])

    def _merge_heads(self, x: V.Tensor) -> V.Tensor:
        batch, _, seq, _ = x.shape
        return x.permute([0, 2, 1, 3]).contiguous().reshape([batch, seq, self.embed_dim])

    def _project(self, x: V.Tensor, index: int) -> V.Tensor:
        """Apply the query, key or value third of the packed input projection.

        Three separate matmuls rather than one packed one followed by a split.
        The FLOP count is identical -- (B,S,E)@(E,E) three times against
        (B,S,E)@(E,3E) once -- so this costs two extra dispatches and buys a
        single path that serves self- and cross-attention alike, with no branch
        on whether the three inputs happen to be the same tensor.
        """
        lo = index * self.embed_dim
        hi = lo + self.embed_dim
        weight = self.in_proj_weight[lo:hi]
        y = V.matmul(x, weight.transpose(-2, -1))
        bias = self._parameters.get("in_proj_bias")
        if bias is not None:
            y = y + bias[lo:hi]
        return y

    def forward(self, query: V.Tensor, key: V.Tensor = None, value: V.Tensor = None,
                attn_mask: V.Tensor = None, is_causal: bool = False) -> V.Tensor:
        key = query if key is None else key
        value = query if value is None else value

        if not self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))

        q = self._split_heads(self._project(query, 0))
        k = self._split_heads(self._project(key, 1))
        v = self._split_heads(self._project(value, 2))

        # 1/sqrt(head_dim), NOT 1/sqrt(embed_dim). Using the model width instead
        # is a plausible-looking error that leaves the model trainable and
        # merely worse, so it would not surface as a failure.
        scores = V.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        mask = attn_mask
        if is_causal:
            if mask is not None:
                raise ValueError("pass either attn_mask or is_causal, not both")
            mask = self._causal_mask(q.shape[2], k.shape[2], query.device)

        if mask is not None:
            # Masked BEFORE the softmax, with -inf rather than after with zero:
            # zeroing afterwards leaves the surviving weights un-normalised, and
            # every row still sums to one only if the masking happens first.
            scores = V.masked_fill(scores, mask, float("-inf"))

        context = V.matmul(V.softmax(scores, -1), v)
        out = self.out_proj(self._merge_heads(context))
        return out if self.batch_first else out.transpose(0, 1)

    @staticmethod
    def _causal_mask(q_len: int, k_len: int, device) -> V.Tensor:
        """True where a position must NOT attend -- torch's convention for a
        boolean mask.

        Strictly above the diagonal, so a position always sees itself. That
        matters beyond correctness: a row masked everywhere would softmax a row
        of -inf into NaN.
        """
        ones = V.full([q_len, k_len], 1.0, device=device)
        return V.greater(V.triu(ones, 1), V.full([], 0.0, device=device))

    def __repr__(self) -> str:
        return f"MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


class TransformerEncoderLayer(Module):
    """Self-attention followed by a feed-forward block, each residual.

    Parameter names match torch.nn.TransformerEncoderLayer (``self_attn``,
    ``linear1``, ``linear2``, ``norm1``, ``norm2``) so a state_dict loads
    unchanged and the comparison is against torch's own layer.

    ``norm_first`` selects pre- or post-normalisation. Post (torch's default) is
    the original formulation; pre is what deep stacks use, because normalising
    inside the residual branch keeps the gradient path to the input clean.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 norm_first: bool = False, seed: int = 0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.norm_first = norm_first
        self.batch_first = batch_first

        if activation not in ("relu", "gelu"):
            raise ValueError(f"unsupported activation {activation!r}; expected relu or gelu")
        self.activation = activation

        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)

        # Three independent dropouts, as torch has: distinct seeds so they do
        # not draw the same mask as each other.
        self.dropout1 = Dropout(dropout, seed=seed + 1)
        self.dropout2 = Dropout(dropout, seed=seed + 2)
        self.dropout = Dropout(dropout, seed=seed + 3)

    def _attention_block(self, x: V.Tensor, mask, is_causal: bool) -> V.Tensor:
        return self.dropout1(self.self_attn(x, x, x, attn_mask=mask, is_causal=is_causal))

    def _feed_forward_block(self, x: V.Tensor) -> V.Tensor:
        hidden = self.linear1(x)
        hidden = V.relu(hidden) if self.activation == "relu" else V.gelu(hidden)
        return self.dropout2(self.linear2(self.dropout(hidden)))

    def forward(self, src: V.Tensor, src_mask: V.Tensor = None,
                is_causal: bool = False) -> V.Tensor:
        x = src
        if self.norm_first:
            x = x + self._attention_block(self.norm1(x), src_mask, is_causal)
            x = x + self._feed_forward_block(self.norm2(x))
        else:
            x = self.norm1(x + self._attention_block(x, src_mask, is_causal))
            x = self.norm2(x + self._feed_forward_block(x))
        return x

    def __repr__(self) -> str:
        return (f"TransformerEncoderLayer(d_model={self.d_model}, nhead={self.nhead}, "
                f"norm_first={self.norm_first})")
