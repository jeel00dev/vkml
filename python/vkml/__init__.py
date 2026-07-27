"""vkml -- a Vulkan-native deep learning framework.

The C++ core is exposed through ``_vkml_core``; this module is the public
surface. Anything that is more natural to express in Python (array coercion,
convenience wrappers, ``nn`` and ``optim``) lives here rather than in bindings,
per docs/ARCHITECTURE.md 4.1.
"""

from __future__ import annotations

import numpy as _np

from . import _vkml_core as _C

# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

Tensor = _C.Tensor
device = _C.device
dtype = _C.dtype

float32 = _C.dtype.float32
float16 = _C.dtype.float16
int32 = _C.dtype.int32
int64 = _C.dtype.int64
bool_ = _C.dtype.bool

Error = _C.Error
ShapeError = _C.ShapeError
DTypeError = _C.DTypeError
IndexError_ = _C.IndexError
DeviceError = _C.DeviceError
NotImplementedError_ = _C.NotImplementedError
OutOfMemoryError = _C.OutOfMemoryError

__version__ = _C.__version__

# Operators, forwarded verbatim.
add = _C.add
sub = _C.sub
mul = _C.mul
div = _C.div
pow = _C.pow  # noqa: A001 - deliberately shadows the builtin, as in torch
maximum = _C.maximum
minimum = _C.minimum

equal = _C.equal
less = _C.less
greater = _C.greater
less_equal = _C.less_equal
greater_equal = _C.greater_equal
not_equal = _C.not_equal

neg = _C.neg
abs = _C.abs  # noqa: A001
sign = _C.sign
square = _C.square
sqrt = _C.sqrt
rsqrt = _C.rsqrt
reciprocal = _C.reciprocal
exp = _C.exp
log = _C.log
erf = _C.erf
sin = _C.sin
cos = _C.cos
tanh = _C.tanh
sigmoid = _C.sigmoid
relu = _C.relu
gelu = _C.gelu
silu = _C.silu
clamp = _C.clamp
clamp_min = _C.clamp_min
clamp_max = _C.clamp_max
where = _C.where
triu = _C.triu
tril = _C.tril

sum = _C.sum  # noqa: A001
mean = _C.mean
prod = _C.prod
amax = _C.amax
amin = _C.amin
argmax = _C.argmax
argmin = _C.argmin
softmax = _C.softmax
log_softmax = _C.log_softmax
matmul = _C.matmul

zeros = _C.zeros
ones = _C.ones
full = _C.full
arange = _C.arange

backward = _C.backward
detach = _C.detach

set_eager = _C.set_eager
is_eager = _C.is_eager
set_log_level = _C.set_log_level
LogLevel = _C.LogLevel
available_devices = _C.available_devices
cpu = _C.cpu_device()

# Vulkan surface. `has_vulkan` reports whether the backend was COMPILED in;
# `vulkan_available()` whether a device is actually present at runtime. Both
# matter: a CPU-only build and a build with no GPU attached are different
# situations and callers may want to distinguish them.
has_vulkan = _C.has_vulkan
vulkan_available = _C.vulkan_available
vulkan_device_count = _C.vulkan_device_count
if has_vulkan:
    init_vulkan = _C.init_vulkan
    vulkan_device_names = _C.vulkan_device_names
    vulkan_stats = _C.vulkan_stats
    vulkan_capabilities = _C.vulkan_capabilities
    vulkan_set_profiling = _C.vulkan_set_profiling
    vulkan_last_profile = _C.vulkan_last_profile
    vulkan_set_subgroup_override = _C.vulkan_set_subgroup_override
    vulkan_pipeline_stats = _C.vulkan_pipeline_stats


# ---------------------------------------------------------------------------
# Array coercion
# ---------------------------------------------------------------------------

_NUMPY_TO_VKML = {
    _np.dtype("float32"): float32,
    _np.dtype("float16"): float16,
    _np.dtype("int32"): int32,
    _np.dtype("int64"): int64,
    _np.dtype("bool"): bool_,
}

_VKML_TO_NUMPY = {v: k for k, v in _NUMPY_TO_VKML.items()}


def _to_numpy_dtype(dt) -> _np.dtype:
    return _VKML_TO_NUMPY[dt]


def tensor(data, dtype=None, device=None, requires_grad=False) -> Tensor:  # noqa: A002
    """Create a tensor from a nested sequence, scalar or NumPy array.

    The array is made C-contiguous first: the binding deliberately accepts only
    contiguous input so that no stride-interpretation logic is duplicated on the
    C++ side.
    """
    # NOT ascontiguousarray: it is documented to return an "at least 1-D"
    # array, which silently turns a rank-0 scalar into shape (1,). Check
    # contiguity instead and only copy when it is actually needed.
    arr = _np.asarray(data)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = _np.ascontiguousarray(arr)

    if arr.dtype == _np.float64:
        # NumPy defaults Python floats to float64, which vkml does not support
        # (and which no target GPU handles well). Silently narrowing matches
        # torch.tensor's behaviour and avoids a confusing error on `[1.0, 2.0]`.
        arr = arr.astype(_np.float32)
    elif arr.dtype not in _NUMPY_TO_VKML:
        raise DTypeError(f"unsupported numpy dtype {arr.dtype}")

    t = _C.from_numpy(arr, dtype, device if device is not None else cpu)
    if requires_grad:
        t.requires_grad = True
    return t


def from_numpy(arr, dtype=None, device=None) -> Tensor:  # noqa: A002
    """Create a tensor by copying a NumPy array."""
    return tensor(arr, dtype=dtype, device=device)


def asarray(t: Tensor) -> _np.ndarray:
    """Copy a tensor's contents into a NumPy array."""
    return t.numpy()


# ---------------------------------------------------------------------------
# Gradient mode
# ---------------------------------------------------------------------------


class no_grad:  # noqa: N801 - matches torch.no_grad
    """Context manager that suppresses backward-graph construction."""

    def __enter__(self):
        self._prev = _C.is_grad_enabled()
        _C.set_grad_enabled(False)
        return self

    def __exit__(self, *exc):
        _C.set_grad_enabled(self._prev)
        return False


class eager_mode:
    """Context manager forcing evaluation after every operation.

    Values are identical either way; only *when* work happens changes. Used by
    the validation suite so a failure names the offending operator instead of
    surfacing at a distant realize().
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled

    def __enter__(self):
        self._prev = _C.is_eager()
        _C.set_eager(self._enabled)
        return self

    def __exit__(self, *exc):
        _C.set_eager(self._prev)
        return False


from . import nn, optim  # noqa: E402  (placed last: they import from this module)

__all__ = [name for name in dir() if not name.startswith("_")]
