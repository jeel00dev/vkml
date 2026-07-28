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
        rng = _np.random.default_rng()
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
            # Unused here -- it exists for torch's momentum=None cumulative
            # average, which this layer does not offer -- but it is in every
            # torch BatchNorm state_dict, and interop is the point of matching
            # the naming at all.
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
        with V.no_grad():
            self.num_batches_tracked.assign_(
                V.tensor(_np.array(int(self.num_batches_tracked.numpy()) + 1, dtype=_np.int64))
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
        rng = _np.random.default_rng()
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
        rng = _np.random.default_rng()
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
