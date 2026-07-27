"""Shared machinery for the PyTorch validation suite.

PyTorch is the ground truth (docs/ARCHITECTURE.md 7). Two rules shape
everything here:

1. INPUTS ARE GENERATED ONCE and the same bytes are fed to both frameworks.
   The specification's phrase "using the same seed" is deliberately *not* read
   as "reproduce PyTorch's RNG bit-for-bit" -- that would be a large amount of
   work validating nothing about our operators. Generating once and sharing
   preserves the intent (identical inputs) exactly, and RNG is tested
   separately for distribution rather than for exact values (7.2).

2. TOLERANCES ARE DERIVED IN ADVANCE from the reduction length, not tuned after
   a failure. See TOLERANCES below and 7.3. A mismatch is a bug until an error
   analysis says otherwise.

The suite runs in eager mode so that a failure names the operator that produced
it rather than surfacing at a distant realize().
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch  # noqa: E402
import vkml as V  # noqa: E402


# ---------------------------------------------------------------------------
# Tolerances (docs/ARCHITECTURE.md 7.3)
#
# fp32 eps = 1.19e-7. Pairwise summation gives a worst-case relative error of
# about (B + log2(n/B))*eps with B = 32, so a K = 4096 reduction lands near
# 4.6e-6 -- inside 1e-5 with roughly 2x margin.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tol:
    atol: float
    rtol: float


TOLERANCES = {
    "elementwise": Tol(1e-6, 1e-6),      # single op, ~1 ulp
    "transcendental": Tol(1e-5, 1e-5),   # exp/log/tanh/gelu: libm ULP differences
    "reduction": Tol(1e-5, 1e-5),        # tree sum, n <= 1e5
    "matmul": Tol(1e-5, 1e-5),           # well-conditioned dot products only
    "fp16": Tol(1e-3, 1e-3),             # fp16 eps ~= 9.8e-4
}


def _describe_failure(name, got, want, tol, inputs, extra=""):
    """Build the diagnostic the specification asks for."""
    got = np.asarray(got)
    want = np.asarray(want)

    lines = [
        "",
        f"operator      : {name}",
        f"vkml shape    : {got.shape}   dtype: {got.dtype}",
        f"torch shape   : {want.shape}   dtype: {want.dtype}",
    ]
    if inputs:
        for i, arr in enumerate(inputs):
            a = np.asarray(arr)
            lines.append(f"input[{i}]      : shape={a.shape} dtype={a.dtype}")
    if extra:
        lines.append(extra)

    if got.shape != want.shape:
        lines.append(f"SHAPE MISMATCH: {got.shape} != {want.shape}")
        return "\n".join(lines)

    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    denom = np.maximum(np.abs(want.astype(np.float64)), 1e-30)
    rel = diff / denom

    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float(rel.max()) if rel.size else 0.0

    lines += [
        f"tolerance     : atol={tol.atol:g} rtol={tol.rtol:g}",
        f"max abs error : {max_abs:.6g}",
        f"max rel error : {max_rel:.6g}",
    ]

    bad = diff > (tol.atol + tol.rtol * np.abs(want.astype(np.float64)))
    n_bad = int(bad.sum())
    lines.append(f"failing elems : {n_bad} of {got.size}")

    if n_bad:
        idx = np.argwhere(bad)[:8]
        lines.append("first failures:")
        for ix in idx:
            key = tuple(int(v) for v in ix)
            lines.append(
                f"    {key}: vkml={got[key]!r:>16}  torch={want[key]!r:>16}  "
                f"abs={diff[key]:.3g}  rel={rel[key]:.3g}"
            )
    return "\n".join(lines)


def assert_close(name, got, want, tol=TOLERANCES["elementwise"], inputs=None, extra=""):
    """Compare a vkml result against a torch result, with full diagnostics."""
    got_np = got.numpy() if isinstance(got, V.Tensor) else np.asarray(got)
    want_np = want.detach().numpy() if isinstance(want, torch.Tensor) else np.asarray(want)

    if got_np.shape != want_np.shape:
        pytest.fail(_describe_failure(name, got_np, want_np, tol, inputs, extra))

    if not np.allclose(got_np, want_np, atol=tol.atol, rtol=tol.rtol, equal_nan=True):
        pytest.fail(_describe_failure(name, got_np, want_np, tol, inputs, extra))


# A dot product's accuracy is bounded in BACKWARD error, not relative to its
# result:
#
#     |computed - exact|  <=  gamma * sum_i |a_i * b_i|
#
# with gamma ~ (B + log2(K/B))*eps for pairwise summation. Comparing relative to
# the *result* is wrong whenever cancellation occurs, because the result can be
# arbitrarily smaller than sum|a_i*b_i|.
#
# This is not academic. For K=4096 with inputs uniform on [-2,2], a measured
# case had result 3.17 against sum|a_i*b_i| = 4051 -- a condition number of 1279.
# The observed error was 4.4e-5, which "fails" a 1e-5 relative check while
# sitting 431x INSIDE the theoretical bound. The kernel was correct; the
# tolerance model was not.
#
# So large-K matmul is checked against a tolerance derived from the actual
# operands. test_matmul_well_conditioned covers the other side: with
# non-cancelling inputs, the same K=4096 reduction holds 1e-5 comfortably,
# which is what demonstrates the summation itself is accurate.


def assert_close_dot(name, got, want, a, b, safety=16.0):
    """Compare a matmul using a backward-error bound derived from the inputs."""
    eps = float(np.finfo(np.float32).eps)
    k = a.shape[-1]
    gamma = (32.0 + np.log2(max(k, 32) / 32.0)) * eps

    abs_prod = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
    atol = float(gamma * abs_prod.max()) * safety

    got_np = got.numpy() if isinstance(got, V.Tensor) else np.asarray(got)
    want_np = want.detach().numpy() if isinstance(want, torch.Tensor) else np.asarray(want)

    tol = Tol(atol, 0.0)
    if not np.allclose(got_np, want_np, atol=atol, rtol=0.0):
        pytest.fail(
            _describe_failure(
                name, got_np, want_np, tol, [a, b],
                extra=(f"K={k}  gamma={gamma:.3g}  max sum|a*b|={abs_prod.max():.6g}\n"
                       f"backward-error bound = gamma*sum|a*b| = {gamma * abs_prod.max():.3g}"),
            )
        )


def assert_dtype(name, got: V.Tensor, want: torch.Tensor):
    mapping = {
        torch.float32: V.float32,
        torch.float16: V.float16,
        torch.int32: V.int32,
        torch.int64: V.int64,
        torch.bool: V.bool_,
    }
    expected = mapping[want.dtype]
    assert got.dtype == expected, (
        f"{name}: dtype mismatch -- vkml {got.dtype} vs torch {want.dtype}"
    )


def assert_shape(name, got: V.Tensor, want: torch.Tensor):
    assert tuple(got.shape) == tuple(want.shape), (
        f"{name}: shape mismatch -- vkml {tuple(got.shape)} vs torch {tuple(want.shape)}"
    )


def assert_strides(name, got: V.Tensor, want: torch.Tensor):
    """Compare strides after converting torch's element strides to bytes.

    torch reports strides in ELEMENTS, vkml (like NumPy) in BYTES. This is the
    one place the two conventions meet, so the conversion is explicit.
    """
    itemsize = want.element_size()
    want_bytes = tuple(s * itemsize for s in want.stride())
    assert tuple(got.strides) == want_bytes, (
        f"{name}: stride mismatch -- vkml {tuple(got.strides)} bytes vs "
        f"torch {want.stride()} elements ({want_bytes} bytes)"
    )


# ---------------------------------------------------------------------------
# Input generation -- once, shared by both frameworks
# ---------------------------------------------------------------------------


def make_input(shape, seed=0, low=-2.0, high=2.0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=shape).astype(dtype)


def pair(arr):
    """The same bytes as a vkml tensor and a torch tensor."""
    return V.tensor(arr), torch.from_numpy(arr.copy())


def pair_grad(arr):
    """As `pair`, with gradient tracking enabled on both."""
    v = V.tensor(arr, requires_grad=True)
    t = torch.from_numpy(arr.copy()).requires_grad_(True)
    return v, t


@pytest.fixture(autouse=True)
def _eager_and_quiet():
    """Run every validation test in eager mode, with logging quietened."""
    V.set_log_level(V.LogLevel.WARN)
    with V.eager_mode(True):
        yield


def pytest_configure(config):
    """Register markers used by tests/python/test_invariants.py.

    `slow` marks property tests that spawn subprocesses -- unavoidable, because
    vkML reads its dispatch switches once into function-local statics, so they
    cannot be varied inside a live process. Run the fast set with
    `-m "not slow"`; CI should run everything.
    """
    config.addinivalue_line(
        "markers", "slow: spawns subprocesses; excluded by -m 'not slow'"
    )
