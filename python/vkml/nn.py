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
        # Only called when normal lookup fails, so parameters and submodules
        # resolve without shadowing real attributes.
        params = self.__dict__.get("_parameters")
        if params is not None and name in params:
            return params[name]
        mods = self.__dict__.get("_modules")
        if mods is not None and name in mods:
            return mods[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    # -- traversal ----------------------------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, V.Tensor]]:
        for name, p in self._parameters.items():
            yield (prefix + name, p)
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + ".")

    def parameters(self) -> Iterator[V.Tensor]:
        for _, p in self.named_parameters():
            yield p

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        yield (prefix.rstrip("."), self)
        for name, m in self._modules.items():
            yield from m.named_modules(prefix + name + ".")

    def state_dict(self) -> dict[str, _np.ndarray]:
        return {name: p.numpy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, _np.ndarray]) -> None:
        """Copy values in by name.

        Used heavily by the validation suite: a PyTorch model's ``state_dict``
        is loaded into the vkml model so that a training comparison starts from
        byte-identical weights rather than from a re-derived random init.
        """
        own = dict(self.named_parameters())
        missing = set(own) - set(state)
        unexpected = set(state) - set(own)
        if missing or unexpected:
            raise KeyError(
                f"state_dict mismatch; missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        for name, param in own.items():
            value = _np.ascontiguousarray(state[name], dtype=_np.float32)
            if tuple(value.shape) != tuple(param.shape):
                raise V.ShapeError(
                    f"'{name}' expects shape {tuple(param.shape)}, got {tuple(value.shape)}"
                )
            self._replace_param(name, V.tensor(value, requires_grad=param.requires_grad))

    def _replace_param(self, dotted: str, new: V.Tensor) -> None:
        target = self
        *path, leaf = dotted.split(".")
        for part in path:
            target = target._modules[part]
        target._parameters[leaf] = new

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
# ---------------------------------------------------------------------------


def mse_loss(pred: V.Tensor, target: V.Tensor, reduction: str = "mean") -> V.Tensor:
    diff = pred - target
    sq = diff * diff
    if reduction == "mean":
        return V.mean(sq)
    if reduction == "sum":
        return V.sum(sq)
    if reduction == "none":
        return sq
    raise ValueError(f"unknown reduction {reduction!r}")


def cross_entropy(logits: V.Tensor, target_onehot: V.Tensor, reduction: str = "mean") -> V.Tensor:
    """Cross-entropy from logits against a one-hot target.

    Computed as ``-sum(target * log_softmax(logits))`` rather than
    ``-log(softmax(logits))``: log_softmax subtracts the max internally and
    never forms the exponential, so it stays finite for logits where softmax
    would underflow to zero and log would then return -inf. This is the same
    reason torch.nn.functional.cross_entropy fuses the two.

    Takes a one-hot target rather than class indices because index-based
    gathering needs an IndexSelect kernel, which is M0-out-of-scope. The
    validation suite compares against torch with the same one-hot encoding.
    """
    logp = V.log_softmax(logits, -1)
    per_sample = V.neg(V.sum(target_onehot * logp, -1))
    if reduction == "mean":
        return V.mean(per_sample)
    if reduction == "sum":
        return V.sum(per_sample)
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction {reduction!r}")
