"""vkml -- a Vulkan-native deep learning framework.

The C++ core is exposed through ``_vkml_core``; this module is the public
surface. Anything that is more natural to express in Python (array coercion,
convenience wrappers, ``nn`` and ``optim``) lives here rather than in bindings,
per docs/ARCHITECTURE.md 4.1.
"""

from __future__ import annotations

import builtins as _builtins

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
erfc = _C.erfc
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
cat = _C.cat
Reduction = _C.Reduction
mse_loss = _C.mse_loss
cross_entropy = _C.cross_entropy
binary_cross_entropy_with_logits = _C.binary_cross_entropy_with_logits
kl_div = _C.kl_div
huber_loss = _C.huber_loss
conv2d = _C.conv2d
rand = _C.rand
dropout = _C.dropout
batch_norm = _C.batch_norm
max_pool2d = _C.max_pool2d
avg_pool2d = _C.avg_pool2d
im2col = _C.im2col
col2im = _C.col2im
index_select = _C.index_select
scatter_add = _C.scatter_add
layer_norm = _C.layer_norm
rms_norm = _C.rms_norm
masked_fill = _C.masked_fill
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
zeros_like = _C.zeros_like
ones_like = _C.ones_like
full_like = _C.full_like

backward = _C.backward
detach = _C.detach


def realize(*tensors) -> None:
    """Evaluate several tensors together, as one unit of work.

    ``V.realize(a, b, c)`` gives the same values as realizing each in turn and
    costs less: the whole set is scheduled once and reaches the backend as a
    single submission instead of three. All must be on the same device.

    Variadic here, a list at the C++ boundary, because the call sites read
    better this way -- ``V.realize(*params)`` is what an optimiser wants.
    """
    _C.realize(list(tensors))

set_eager = _C.set_eager
is_eager = _C.is_eager
set_log_level = _C.set_log_level
LogLevel = _C.LogLevel

# The decision recorder. Exported unconditionally and NOT under `has_vulkan`:
# decisions are published from any layer, so a CPU-only build has them too, and
# gating them on Vulkan would misdescribe what they observe.
configuration = _C.configuration
record_decisions = _C.record_decisions
stop_recording_decisions = _C.stop_recording_decisions
decisions = _C.decisions
decisions_published = _C.decisions_published
available_devices = _C.available_devices
cpu = _C.cpu_device()

# Vulkan surface. `has_vulkan` reports whether the backend was COMPILED in;
# `vulkan_available()` whether a device is actually present at runtime. Both
# matter: a CPU-only build and a build with no GPU attached are different
# situations and callers may want to distinguish them.
has_vulkan = _C.has_vulkan
vulkan_available = _C.vulkan_available
vulkan_device_count = _C.vulkan_device_count
# Also defined on a CPU-only build, where it returns []. A hardware report has
# to work on the machines that cannot run the backend -- those are the ones
# worth hearing about.
vulkan_device_reports = _C.vulkan_device_reports
# Likewise []: this is what README.md's post-install check calls, and it is
# called on a CPU-only build as often as on any other (issue #9).
vulkan_device_names = _C.vulkan_device_names
vulkan_unavailable_reason = _C.vulkan_unavailable_reason
if has_vulkan:
    init_vulkan = _C.init_vulkan
    vulkan_stats = _C.vulkan_stats
    vulkan_capabilities = _C.vulkan_capabilities
    vulkan_timestamps_supported = _C.vulkan_timestamps_supported
    vulkan_set_profiling = _C.vulkan_set_profiling
    vulkan_last_profile = _C.vulkan_last_profile

    def vulkan_submit_ms(profile) -> float:
        """GPU milliseconds for one submission, read the only correct way.

        `vulkan_last_profile()` is a footgun without this. Its per-dispatch
        entries end at ALL_COMMANDS, a GLOBAL drain point, so when a submission
        holds INDEPENDENT dispatches every entry's window stretches to the end
        of the whole group and adding them counts the same elapsed time once per
        dispatch. Measured here: split-K's eight partitions each report ~0.84 ms
        and sum to 7.2 ms against a true 0.93 ms.

        The `submit` entry brackets the whole command buffer and is right in
        both cases -- for a single dispatch it equals the sum exactly, measured
        2.720 == 2.720, which is what calibrates it. Falling back to the sum
        keeps this working against a core too old to emit the entry.

        Summing this across SEPARATE submissions is fine and is not what the
        rule forbids; they are serial. See docs/MEASUREMENT-AUDIT.md 3, rule 3.
        """
        for label, ms in profile:
            if label == "submit":
                return ms
        # _builtins.sum, not sum: this module exports vkml's own tensor `sum`,
        # which shadows the builtin here and raises on a generator. Only the
        # fallback branch touches it, so a plain sum() would have looked fine
        # until it ran on a core too old to emit the submit entry.
        return _builtins.sum(ms for _, ms in profile)
    vulkan_set_subgroup_override = _C.vulkan_set_subgroup_override
    vulkan_pipeline_stats = _C.vulkan_pipeline_stats
    vulkan_profile_records = _C.vulkan_profile_records


# ---------------------------------------------------------------------------
# Choosing a device
# ---------------------------------------------------------------------------

# Where to send someone whose GPU did not work out. `vulkan_device_reports()` is
# the right pointer because it exists in every install -- scripts/ is not shipped
# in a wheel, so naming hardware_report.py alone would be advice half the users
# cannot follow.
_NEXT_STEPS = (
    "Call vkml.vulkan_device_reports() to see every device the loader can find "
    "and what each one is missing. README.md's Troubleshooting section covers "
    "installing a driver and the Vulkan loader."
)


def best_device() -> tuple[device, str]:
    """The best usable device, and a plain-English reason for the choice.

    ``V.best_device()`` is for callers who want vkML to decide. It never raises:
    if no GPU is usable it returns the CPU and says why.

    Returns the reason instead of printing it. A library writing to someone's
    stdout is a nuisance, and a caller that wants to log it, show it in a UI, or
    ignore it should be free to::

        device, why = V.best_device()
        print(why)      # "using Vulkan device 0: AMD Radeon RX 5600M ..."

    THIS IS THE ONLY PATH THAT FALLS BACK. A device you NAME is never quietly
    downgraded -- ``V.device("vulkan:1")`` with no second GPU fails, because
    someone who typed that wants that GPU and handing back the CPU would hide
    the thing they asked about. See
    docs/adr/0008-backend-selection-and-cpu-fallback.md.

    A discrete GPU is preferred over an integrated one when both are usable,
    which is an ordinary laptop. The reason names the device picked, so the
    choice is never a mystery.
    """
    if not has_vulkan:
        return cpu, (
            "running on the CPU: this build has no Vulkan backend. It was built "
            "with VKML_VULKAN=OFF; reinstall without that flag to use a GPU."
        )

    reports = vulkan_device_reports()
    if not reports:
        return cpu, f"running on the CPU: {vulkan_unavailable_reason()}. {_NEXT_STEPS}"

    usable = [(i, r) for i, r in enumerate(reports) if not r["missing_requirement"]]
    if not usable:
        # Devices exist but none qualifies -- a different problem from having no
        # driver, and one the user can act on only if we name the feature.
        detail = "; ".join(f"{r['name']} is missing {r['missing_requirement']}" for r in reports)
        return cpu, (
            f"running on the CPU: {len(reports)} Vulkan device(s) found, none meeting "
            f"vkML's requirements ({detail}). vkML needs bufferDeviceAddress, "
            f"scalarBlockLayout and timelineSemaphore; a newer driver sometimes adds "
            f"them. {_NEXT_STEPS}"
        )

    # First discrete if there is one, else first usable. `min` is stable, so
    # among equals this keeps enumeration order.
    index, report = min(usable, key=lambda pair: 0 if pair[1]["device_type"] == "discrete" else 1)
    try:
        init_vulkan(index)
    except Error as exc:
        # The capability probe passed and creating the device still failed. Rare,
        # and worth reporting verbatim rather than swallowing -- it is the one
        # case where the reason is something neither check predicted.
        return cpu, (
            f"running on the CPU: Vulkan device {index} ({report['name']}) reported the "
            f"features vkML needs, but initialising it failed: {exc}. {_NEXT_STEPS}"
        )
    return device(f"vulkan:{index}"), (
        f"using Vulkan device {index}: {report['name']} "
        f"({report['device_type']}, Vulkan {report['api_version']}, "
        f"driver {report['driver_name']})"
    )


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


from . import data, nn, optim, serialize  # noqa: E402  (last: they import from this module)
from .serialize import Checkpoint, load, load_module, save, save_module  # noqa: E402

__all__ = [name for name in dir() if not name.startswith("_")]
