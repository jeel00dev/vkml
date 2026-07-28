"""Reusable CPU <-> Vulkan kernel validation.

Every Vulkan kernel is validated against the CPU backend, which is the project's
correctness oracle (docs/ARCHITECTURE.md 7.1). The CPU backend is in turn
validated against PyTorch by the suite in test_ops_vs_torch.py, so a Vulkan
kernel passing here inherits that guarantee transitively.

Adding a kernel to the suite should be one line. Everything else -- input
generation, shape and stride randomization, tolerance selection, diagnostics --
is handled here.

The generators deliberately produce awkward layouts: transposed views, broadcast
axes with stride 0, sliced views with a non-zero base offset, singleton and
empty extents. Those are where indexing bugs live, and a kernel that only ever
sees dense contiguous input is barely tested at all.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import vkml as V  # noqa: E402
from tolerance import (  # noqa: E402
    FP32_EPS,
    Kind,
    Tolerance,
    backward_error_atol,
    lookup,
    ulp_distance,
)

# ---------------------------------------------------------------------------
# Device availability
# ---------------------------------------------------------------------------

_VULKAN_READY: bool | None = None

# Which physical device the suite runs against. Selectable because a framework
# that claims to run on any Vulkan device has to be run on more than one to make
# that claim mean anything, and a hardcoded 0 makes the second device
# untestable. Devices differ in ways the kernels can depend on -- subgroup size
# and its controllable range, shared memory, the size of the host-visible heap.
#
#     VKML_TEST_DEVICE=1 python -m pytest tests/python -q
VULKAN_DEVICE = int(os.environ.get("VKML_TEST_DEVICE", "0"))


def vulkan_ready() -> bool:
    """True when the selected Vulkan device exists and its backend initialised.

    Cached: initialising the backend creates a device and allocates a staging
    buffer, which should happen once per session, not once per test.
    """
    global _VULKAN_READY
    if _VULKAN_READY is None:
        try:
            _VULKAN_READY = bool(V.has_vulkan) and V.vulkan_available()
            if _VULKAN_READY:
                V.init_vulkan(VULKAN_DEVICE)
        except Exception:  # noqa: BLE001 - absence must never be an error
            _VULKAN_READY = False
    return _VULKAN_READY


requires_vulkan = pytest.mark.skipif(
    not vulkan_ready(), reason="no Vulkan device available"
)


def gpu_device():
    return V.device(f"vulkan:{VULKAN_DEVICE}")


# ---------------------------------------------------------------------------
# Randomized layout generation
# ---------------------------------------------------------------------------


@dataclass
class Layout:
    """A tensor description plus the view transform to apply to it."""

    base_shape: tuple[int, ...]
    transform: str = "contiguous"
    detail: tuple = ()

    def describe(self) -> str:
        if self.transform == "contiguous":
            return f"{self.base_shape} contiguous"
        return f"{self.base_shape} -> {self.transform}{self.detail}"

    def apply(self, t):
        if self.transform == "contiguous":
            return t
        if self.transform == "transpose":
            return t.transpose(*self.detail)
        if self.transform == "permute":
            return t.permute(list(self.detail))
        if self.transform == "slice":
            axis, start, stop, step = self.detail
            key = [slice(None)] * len(self.base_shape)
            key[axis] = slice(start, stop, step)
            return t[tuple(key)]
        if self.transform == "broadcast":
            return t.broadcast_to(list(self.detail))
        raise ValueError(f"unknown transform {self.transform!r}")

    def apply_numpy(self, a: np.ndarray) -> np.ndarray:
        if self.transform == "contiguous":
            return a
        if self.transform == "transpose":
            return np.swapaxes(a, *self.detail)
        if self.transform == "permute":
            return np.transpose(a, self.detail)
        if self.transform == "slice":
            axis, start, stop, step = self.detail
            key = [slice(None)] * a.ndim
            key[axis] = slice(start, stop, step)
            return a[tuple(key)]
        if self.transform == "broadcast":
            return np.broadcast_to(a, self.detail)
        raise ValueError(f"unknown transform {self.transform!r}")


def random_layouts(rng: np.random.Generator, count: int, max_rank: int = 4) -> list[Layout]:
    """Assorted shapes and view transforms, biased toward awkward cases."""
    layouts: list[Layout] = []

    # Fixed cases that must always be covered, regardless of the seed. These are
    # the edges that randomization hits only rarely.
    layouts += [
        Layout((1,)),                       # single element
        Layout((0,)),                       # empty
        Layout((0, 5)),                     # empty with a non-zero axis
        Layout((1, 1, 1, 1)),               # all singleton
        Layout((257,)),                     # one past a workgroup boundary (256)
        Layout((255,)),                     # one below
        Layout((256,)),                     # exactly one workgroup
        Layout((4, 4), "transpose", (0, 1)),
        Layout((3, 1), "broadcast", (3, 4)),           # stride 0
        Layout((8,), "slice", (0, 2, 7, 2)),           # offset + step
        Layout((5, 6), "slice", (1, 1, 5, 2)),         # offset on inner axis
        Layout((2, 3, 4), "permute", (2, 0, 1)),
    ]

    while len(layouts) < count:
        rank = int(rng.integers(1, max_rank + 1))
        shape = tuple(int(rng.integers(1, 9)) for _ in range(rank))
        choice = rng.integers(0, 5)

        if choice == 0 or rank == 1:
            layouts.append(Layout(shape))
        elif choice == 1:
            a, b = rng.choice(rank, size=2, replace=False)
            layouts.append(Layout(shape, "transpose", (int(a), int(b))))
        elif choice == 2:
            perm = tuple(int(x) for x in rng.permutation(rank))
            layouts.append(Layout(shape, "permute", perm))
        elif choice == 3:
            axis = int(rng.integers(0, rank))
            extent = shape[axis]
            start = int(rng.integers(0, extent))
            stop = int(rng.integers(start, extent)) + 1
            step = int(rng.integers(1, 3))
            layouts.append(Layout(shape, "slice", (axis, start, stop, step)))
        else:
            axis = int(rng.integers(0, rank))
            shape = shape[:axis] + (1,) + shape[axis + 1 :]
            target = list(shape)
            target[axis] = int(rng.integers(2, 5))
            layouts.append(Layout(shape, "broadcast", tuple(target)))

    return layouts[:count]


def make_data(rng: np.random.Generator, shape: tuple[int, ...], domain: str) -> np.ndarray:
    lo, hi = {
        "any": (-3.0, 3.0),
        "positive": (0.25, 4.0),
        "nonzero": (0.5, 3.0),
        "bounded": (-3.0, 3.0),
        "unit": (-1.0, 1.0),
    }[domain]
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """Everything a failure report needs, gathered before the comparison."""

    op: str
    layout: Layout
    dtype: str
    seed: int
    inputs: list[np.ndarray] = field(default_factory=list)
    spec_constants: dict = field(default_factory=dict)
    workgroup_size: int = 256
    subgroup_size: int = 0
    extra: str = ""


def _report(ctx: Context, tol: Tolerance, gpu: np.ndarray, cpu: np.ndarray,
            failing: np.ndarray) -> str:
    lines = [
        "",
        "=" * 78,
        f"VULKAN KERNEL MISMATCH: {ctx.op}",
        "=" * 78,
        f"  layout          : {ctx.layout.describe()}",
        f"  result shape    : {gpu.shape}",
        f"  dtype           : {ctx.dtype}",
        f"  seed            : {ctx.seed}",
        f"  backends        : vulkan:0 (actual) vs cpu (expected)",
        f"  workgroup size  : {ctx.workgroup_size}",
        f"  subgroup size   : {ctx.subgroup_size or 'driver-selected'}",
    ]
    if ctx.spec_constants:
        lines.append(f"  spec constants  : {ctx.spec_constants}")
    for i, arr in enumerate(ctx.inputs):
        lines.append(
            f"  input[{i}]        : shape={arr.shape} strides={arr.strides} dtype={arr.dtype}"
        )
    lines.append(f"  tolerance       : {tol.kind.value}"
                 + (f" ulp<={tol.ulp}" if tol.ulp else "")
                 + (f" rtol={tol.rtol:g}" if tol.rtol else "")
                 + (f" atol={tol.atol:g}" if tol.atol else ""))
    lines.append(f"  policy note     : {tol.note or '(none)'}")
    if ctx.extra:
        lines.append(f"  {ctx.extra}")

    if gpu.shape != cpu.shape:
        lines.append(f"  SHAPE MISMATCH  : {gpu.shape} != {cpu.shape}")
        return "\n".join(lines)

    g = gpu.astype(np.float64).ravel()
    c = cpu.astype(np.float64).ravel()
    diff = np.abs(g - c)
    rel = diff / np.maximum(np.abs(c), 1e-30)

    lines += [
        f"  max abs error   : {diff.max():.6g}" if diff.size else "  (empty)",
        f"  max rel error   : {rel.max():.6g}" if diff.size else "",
    ]
    if gpu.dtype == np.float32:
        ulps = ulp_distance(gpu.ravel(), cpu.ravel())
        lines.append(f"  max ULP error   : {int(ulps.max()) if ulps.size else 0}")

    bad_idx = np.flatnonzero(failing.ravel())
    lines.append(f"  failing elements: {bad_idx.size} of {g.size}")
    if bad_idx.size:
        lines.append("  first failures:")
        for flat in bad_idx[:10]:
            multi = np.unravel_index(flat, gpu.shape)
            u = ""
            if gpu.dtype == np.float32:
                u = f"  ulp={int(ulp_distance(gpu.ravel()[flat:flat+1], cpu.ravel()[flat:flat+1])[0])}"
            lines.append(
                f"    {tuple(int(x) for x in multi)}: vulkan={g[flat]!r:>18} "
                f"cpu={c[flat]!r:>18}  abs={diff[flat]:.3g}  rel={rel[flat]:.3g}{u}"
            )
    lines.append("=" * 78)
    return "\n".join(lines)


def compare(ctx: Context, gpu: np.ndarray, cpu: np.ndarray,
            terms_abs_sum: float | None = None, reduction_n: int | None = None) -> None:
    """Compare two results under the policy for `ctx.op`."""
    tol = lookup(ctx.op)

    if gpu.shape != cpu.shape:
        pytest.fail(_report(ctx, tol, gpu, cpu, np.ones(gpu.shape, dtype=bool)))
    if gpu.dtype != cpu.dtype:
        pytest.fail(f"{ctx.op}: dtype mismatch -- vulkan {gpu.dtype} vs cpu {cpu.dtype}")
    if gpu.size == 0:
        return

    if tol.kind is Kind.EXACT:
        failing = gpu != cpu
        # NaN != NaN, so agreement on NaN placement is checked separately rather
        # than counted as a failure.
        if np.issubdtype(gpu.dtype, np.floating):
            both_nan = np.isnan(gpu) & np.isnan(cpu)
            failing = failing & ~both_nan

    elif tol.kind is Kind.ULP:
        diff = np.abs(gpu.astype(np.float64) - cpu.astype(np.float64))
        failing = np.zeros(gpu.shape, dtype=bool)
        if tol.ulp:
            failing |= ulp_distance(gpu, cpu) > tol.ulp
        if tol.atol:
            # An absolute allowance (sin/cos) overrides the ULP test where it is
            # the looser of the two, which is what the Vulkan spec grants.
            failing &= diff > tol.atol

    elif tol.kind is Kind.RELATIVE:
        failing = ~np.isclose(gpu, cpu, rtol=tol.rtol or 1e-5,
                              atol=tol.atol or 1e-7, equal_nan=True)

    elif tol.kind is Kind.BACKWARD:
        if terms_abs_sum is None or reduction_n is None:
            raise ValueError(
                f"'{ctx.op}' uses a backward-error tolerance, so compare() needs "
                f"terms_abs_sum and reduction_n"
            )
        atol = backward_error_atol(terms_abs_sum, reduction_n)
        ctx.extra = (f"backward-error bound: gamma*sum|terms| = {atol:.3g} "
                     f"(n={reduction_n}, sum|terms|={terms_abs_sum:.6g})")
        failing = np.abs(gpu.astype(np.float64) - cpu.astype(np.float64)) > atol

    else:  # pragma: no cover
        raise AssertionError(f"unhandled tolerance kind {tol.kind}")

    if bool(np.any(failing)):
        pytest.fail(_report(ctx, tol, gpu, cpu, failing))


# ---------------------------------------------------------------------------
# Kernel drivers
# ---------------------------------------------------------------------------


def run_unary(op_name: str, fn: Callable, layouts: Sequence[Layout], seed: int,
              domain: str = "any") -> int:
    """Runs `fn` on both backends over every layout and compares.

    Returns the number of cases checked, so a test can assert it actually ran.
    """
    rng = np.random.default_rng(seed)
    checked = 0

    for layout in layouts:
        data = make_data(rng, layout.base_shape, domain)

        cpu_base = V.tensor(data, device=V.cpu)
        gpu_base = V.tensor(data, device=gpu_device())

        cpu_in = layout.apply(cpu_base)
        gpu_in = layout.apply(gpu_base)

        cpu_out = fn(cpu_in).numpy()
        gpu_out = fn(gpu_in).numpy()

        ctx = Context(op=op_name, layout=layout, dtype="f32", seed=seed,
                      inputs=[layout.apply_numpy(data)])
        compare(ctx, gpu_out, cpu_out)
        checked += 1

    return checked


def run_binary(op_name: str, fn: Callable, layouts: Sequence[Layout], seed: int,
               domain: str = "any", domain_b: str | None = None) -> int:
    """Runs `fn` on two operands sharing a layout, on both backends.

    Both operands get the *same* view transform, so the strided and broadcast
    paths are exercised on each side simultaneously -- which is the case a
    kernel is most likely to get wrong, since it must resolve two independent
    stride sets per element. Broadcasting between *different* shapes is a
    separate concern and is covered by run_binary_broadcast.

    `domain_b` defaults to `domain`; div and pow need a right-hand operand that
    avoids zero and negative bases respectively.
    """
    rng = np.random.default_rng(seed)
    checked = 0

    for layout in layouts:
        a = make_data(rng, layout.base_shape, domain)
        b = make_data(rng, layout.base_shape, domain_b or domain)

        cpu_out = fn(layout.apply(V.tensor(a, device=V.cpu)),
                     layout.apply(V.tensor(b, device=V.cpu))).numpy()
        gpu_out = fn(layout.apply(V.tensor(a, device=gpu_device())),
                     layout.apply(V.tensor(b, device=gpu_device()))).numpy()

        ctx = Context(op=op_name, layout=layout, dtype="f32", seed=seed,
                      inputs=[layout.apply_numpy(a), layout.apply_numpy(b)])
        compare(ctx, gpu_out, cpu_out)
        checked += 1

    return checked


# Shape pairs that must broadcast, covering the cases numpy's rules make
# distinct: rank extension, a singleton on either side, and singletons on both
# sides of the same pair of axes (which broadcasts outward in two directions).
BROADCAST_PAIRS: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((3, 4), (4,)),
    ((3, 4), (1, 4)),
    ((3, 4), (3, 1)),
    ((3, 1), (1, 4)),
    ((2, 3, 4), (4,)),
    ((2, 3, 4), (3, 1)),
    ((2, 1, 4), (1, 3, 1)),
    ((5,), (1,)),
    ((1,), (5,)),
    ((2, 3, 4, 5), (1, 3, 1, 5)),
]


def run_binary_broadcast(op_name: str, fn: Callable, seed: int,
                         domain: str = "any", domain_b: str | None = None) -> int:
    """Same operation across mismatched shapes, where one side has stride 0."""
    rng = np.random.default_rng(seed)
    checked = 0

    for shape_a, shape_b in BROADCAST_PAIRS:
        a = make_data(rng, shape_a, domain)
        b = make_data(rng, shape_b, domain_b or domain)

        cpu_out = fn(V.tensor(a, device=V.cpu), V.tensor(b, device=V.cpu)).numpy()
        gpu_out = fn(V.tensor(a, device=gpu_device()),
                     V.tensor(b, device=gpu_device())).numpy()

        ctx = Context(op=op_name, layout=Layout(shape_a, "broadcast", shape_b),
                      dtype="f32", seed=seed, inputs=[a, b])
        compare(ctx, gpu_out, cpu_out)
        checked += 1

    return checked


# ---------------------------------------------------------------------------
# Reduction support
#
# Reductions differ from elementwise ops in two ways the framework has to know
# about: the output shape is not the input shape, and the BACKWARD tolerance
# policy needs sum|terms| and the reduction length, which only the caller can
# compute.
# ---------------------------------------------------------------------------


@dataclass
class ReduceCase:
    """One reduction configuration."""

    shape: tuple[int, ...]
    axes: tuple[int, ...] | None  # None means "every axis"
    keepdim: bool = False

    def describe(self) -> str:
        axes = "all" if self.axes is None else str(self.axes)
        return f"{self.shape} axes={axes} keepdim={self.keepdim}"

    def numpy_axes(self, rank: int):
        if self.axes is None:
            return tuple(range(rank))
        return tuple(a % rank for a in self.axes)

    def reduction_length(self) -> int:
        """Elements folded into each output element."""
        n = 1
        for a in self.numpy_axes(len(self.shape)):
            n *= self.shape[a]
        return n


def reduction_cases(rng: np.random.Generator, count: int) -> list[ReduceCase]:
    """Fixed edges plus randomized configurations."""
    cases: list[ReduceCase] = [
        # Scalar output from every rank.
        ReduceCase((7,), None),
        ReduceCase((3, 4), None),
        ReduceCase((2, 3, 4), None),
        ReduceCase((2, 2, 3, 3), None),
        # Single axis, including the innermost and outermost.
        ReduceCase((3, 4), (0,)),
        ReduceCase((3, 4), (1,)),
        ReduceCase((2, 3, 4), (0,)),
        ReduceCase((2, 3, 4), (2,)),
        # Negative axes.
        ReduceCase((3, 4), (-1,)),
        ReduceCase((2, 3, 4), (-2,)),
        # Multiple axes, contiguous and not.
        ReduceCase((2, 3, 4), (0, 2)),
        ReduceCase((2, 3, 4), (1, 2)),
        ReduceCase((2, 2, 3, 3), (0, 1)),
        # keepdim.
        ReduceCase((3, 4), (1,), keepdim=True),
        ReduceCase((2, 3, 4), None, keepdim=True),
        ReduceCase((2, 3, 4), (0, 2), keepdim=True),
        # Singleton axes: reducing an axis of extent 1 must be a no-op in value.
        ReduceCase((1, 5), (0,)),
        ReduceCase((5, 1), (1,)),
        ReduceCase((1, 1, 1), None),
        # Lengths that straddle the workgroup boundary, where an off-by-one in
        # the tree reduction or the tail handling would show.
        ReduceCase((255,), None),
        ReduceCase((256,), None),
        ReduceCase((257,), None),
        ReduceCase((511,), None),
        ReduceCase((512,), None),
        ReduceCase((513,), None),
        ReduceCase((1024,), None),
        # Long enough that per-lane accumulation error would show if the
        # pairwise structure were wrong.
        ReduceCase((65536,), None),
        ReduceCase((4, 8192), (1,)),
    ]

    while len(cases) < count:
        rank = int(rng.integers(1, 5))
        shape = tuple(int(rng.integers(1, 10)) for _ in range(rank))
        mode = rng.integers(0, 3)
        if mode == 0:
            axes = None
        else:
            k = int(rng.integers(1, rank + 1))
            axes = tuple(sorted(int(a) for a in rng.choice(rank, size=k, replace=False)))
        cases.append(ReduceCase(shape, axes, keepdim=bool(rng.integers(0, 2))))

    return cases[:count]


def run_reduction(op_name: str, vk_fn: Callable, np_fn: Callable,
                  cases: Sequence[ReduceCase], seed: int, domain: str = "any") -> int:
    """Validates a reduction on both backends across every case.

    `vk_fn(tensor, axes, keepdim)` builds the reduction; `np_fn(array, axes,
    keepdim)` is used only to compute sum|terms| for the backward-error bound,
    never as the expected value -- the CPU backend remains the oracle.
    """
    rng = np.random.default_rng(seed)
    checked = 0

    for case in cases:
        data = make_data(rng, case.shape, domain)

        cpu_t = V.tensor(data, device=V.cpu)
        gpu_t = V.tensor(data, device=gpu_device())

        axes = None if case.axes is None else list(case.axes)
        cpu_out = vk_fn(cpu_t, axes, case.keepdim).numpy()
        gpu_out = vk_fn(gpu_t, axes, case.keepdim).numpy()

        ctx = Context(
            op=op_name,
            layout=Layout(case.shape),
            dtype="f32",
            seed=seed,
            inputs=[data],
            extra=f"reduction: {case.describe()}",
        )

        # For a BACKWARD policy the bound needs the worst-case sum of absolute
        # terms over any single output element, which is what bounds that
        # element's error.
        np_axes = case.numpy_axes(len(case.shape))
        abs_sums = np.abs(data.astype(np.float64)).sum(axis=np_axes)
        terms = float(np.max(abs_sums)) if np.size(abs_sums) else 0.0

        compare(ctx, gpu_out, cpu_out,
                terms_abs_sum=terms, reduction_n=case.reduction_length())
        checked += 1

    return checked


# ---------------------------------------------------------------------------
# GEMM support
# ---------------------------------------------------------------------------

# Sizes that break tiled kernels: powers of two, one either side of each, and
# small primes. A kernel with a 16- or 32-wide tile passes on 64 and fails on
# 63 or 65, so those are always covered rather than left to chance.
GEMM_EDGE_SIZES = [1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257]


@dataclass
class GemmCase:
    m: int
    n: int
    k: int
    batch: tuple[int, ...] = ()
    transpose_a: bool = False
    transpose_b: bool = False

    def describe(self) -> str:
        t = ""
        if self.transpose_a:
            t += " Aᵀ"
        if self.transpose_b:
            t += " Bᵀ"
        b = f" batch={self.batch}" if self.batch else ""
        return f"M={self.m} N={self.n} K={self.k}{b}{t}"


def gemm_cases(rng: np.random.Generator, count: int) -> list[GemmCase]:
    cases: list[GemmCase] = []

    # Square at every edge size.
    cases += [GemmCase(s, s, s) for s in GEMM_EDGE_SIZES]
    # Degenerate ranks: a single row, column or inner element.
    cases += [
        GemmCase(1, 1, 1), GemmCase(1, 1, 257), GemmCase(1, 257, 1), GemmCase(257, 1, 1),
        GemmCase(1, 64, 64), GemmCase(64, 1, 64), GemmCase(64, 64, 1),
    ]
    # Tall, wide and rectangular.
    cases += [
        GemmCase(1024, 8, 64), GemmCase(8, 1024, 64), GemmCase(64, 64, 1024),
        GemmCase(127, 63, 31), GemmCase(31, 127, 63), GemmCase(255, 33, 17),
    ]
    # Batched, including a broadcast batch axis.
    cases += [
        GemmCase(16, 16, 16, batch=(2,)), GemmCase(8, 8, 8, batch=(3,)),
        GemmCase(16, 16, 16, batch=(2, 2)),
    ]
    # Transposed operands, which exercise the strided read path.
    cases += [
        GemmCase(32, 32, 32, transpose_a=True), GemmCase(32, 32, 32, transpose_b=True),
        GemmCase(63, 31, 17, transpose_b=True),
    ]

    while len(cases) < count:
        cases.append(GemmCase(int(rng.integers(1, 130)), int(rng.integers(1, 130)),
                              int(rng.integers(1, 130))))
    return cases[:count]


def run_gemm(cases: Sequence[GemmCase], seed: int, torch_too: bool = True) -> int:
    """Validates matmul on the GPU against the CPU oracle, and against PyTorch.

    Uses the existing BACKWARD tolerance policy: the bound is derived from
    sum|a_i*b_i| for the worst output element, which is the only well-posed
    check for a dot product (see tolerance.py).
    """
    rng = np.random.default_rng(seed)
    checked = 0

    for case in cases:
        a_shape = (*case.batch, case.k, case.m) if case.transpose_a else \
                  (*case.batch, case.m, case.k)
        b_shape = (*case.batch, case.n, case.k) if case.transpose_b else \
                  (*case.batch, case.k, case.n)

        a = rng.uniform(-2, 2, size=a_shape).astype(np.float32)
        b = rng.uniform(-2, 2, size=b_shape).astype(np.float32)

        def build(dev):
            ta = V.tensor(a, device=dev)
            tb = V.tensor(b, device=dev)
            if case.transpose_a:
                ta = ta.transpose(-2, -1)
            if case.transpose_b:
                tb = tb.transpose(-2, -1)
            return V.matmul(ta, tb)

        cpu_out = build(V.cpu).numpy()
        gpu_out = build(gpu_device()).numpy()

        an = np.swapaxes(a, -2, -1) if case.transpose_a else a
        bn = np.swapaxes(b, -2, -1) if case.transpose_b else b
        abs_prod = np.abs(an.astype(np.float64)) @ np.abs(bn.astype(np.float64))

        ctx = Context(op="matmul", layout=Layout(a_shape), dtype="f32", seed=seed,
                      inputs=[a, b], extra=f"gemm: {case.describe()}")
        compare(ctx, gpu_out, cpu_out,
                terms_abs_sum=float(abs_prod.max()), reduction_n=case.k)

        if torch_too:
            import torch
            want = torch.matmul(torch.from_numpy(an.copy()),
                                torch.from_numpy(bn.copy())).numpy()
            compare(ctx, gpu_out, want,
                    terms_abs_sum=float(abs_prod.max()), reduction_n=case.k)

        checked += 1
    return checked
