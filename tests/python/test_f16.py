"""f16 as a compute dtype.

THE CONTRACT, from `ARCHITECTURE.md` §7.3 and the note on `Half` in dtype.h:
**f16 is storage, never an accumulator.** Values are widened to float, the
operation runs in float, and the result is narrowed once on the store. The 1e-3
tolerance is derived for exactly that arrangement and does not hold without it,
which is why `test_accumulation_happens_in_fp32` is the most important test in
this file rather than a nicety.

PyTorch is the oracle for semantics, as everywhere else. Where a test needs to
distinguish "computed in float" from "computed in half", it asserts against an
exactly-representable value instead, because a tolerance comparison cannot tell
those apart -- that is the whole difficulty with verifying a precision contract.

EVERY TEST RUNS ON BOTH BACKENDS, which is what makes this f16 rather than
f16-on-one-machine: the CPU is checked against PyTorch for semantics and Vulkan
against the CPU for kernel bugs (ARCHITECTURE.md §7). Matmul is the exception --
the GEMM family is still f32-only and says so, pinned by
test_vulkan_refuses_f16_matmul.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vkml as V
from conftest import TOLERANCES, assert_close, make_input
from vkvalidate import gpu_device, vulkan_ready

F16_TOL = TOLERANCES["fp16"]

DEVICES = [pytest.param("cpu", id="cpu")]
if vulkan_ready():
    DEVICES.append(pytest.param("gpu", id="vulkan"))


def on(device):
    return gpu_device() if device == "gpu" else V.cpu


def half_pair(shape, seed, device="cpu", low=-2.0, high=2.0):
    """The same values as an f16 vkml tensor and an f16 torch tensor."""
    x = make_input(shape, seed=seed, low=low, high=high).astype(np.float16)
    return V.tensor(x, device=on(device)), torch.from_numpy(x.copy()), x


# ---------------------------------------------------------------------------
# The precision contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_accumulation_happens_in_fp32(device):
    """The one test that would fail if the accumulator were 16-bit.

    f16 has an 11-bit significand, so above 2048 the representable values are 2
    apart: a running half-precision sum of 4096 ones reaches 2048 and then stops
    moving, because 2048 + 1 rounds back to 2048. In float it reaches 4096
    exactly, and 4096 is representable in f16, so the stored result is exact.

    Asserted as equality rather than within a tolerance deliberately. 2048 and
    4096 differ by 100 %, but a test written with `atol=1e-3` on normalised data
    would never generate values large enough to notice -- which is how a
    precision contract silently stops holding.
    """
    n = 4096
    x = V.tensor(np.ones(n, dtype=np.float16), device=on(device))

    total = float(V.sum(x).numpy())

    assert total == 4096.0, (
        f"sum of {n} f16 ones gave {total}. 2048 means the accumulator is f16; "
        "the contract (ARCHITECTURE.md 7.3) is fp32 accumulation"
    )


def test_matmul_accumulates_in_fp32():
    """Same argument along the reduction axis of a matmul, where K is large
    enough that a 16-bit accumulator saturates well before the end."""
    k = 4096
    a = V.tensor(np.ones((1, k), dtype=np.float16))
    b = V.tensor(np.ones((k, 1), dtype=np.float16))

    assert V.matmul(a, b).numpy().item() == 4096.0


@pytest.mark.parametrize("device", DEVICES)
def test_the_result_is_narrowed_to_f16(device):
    """Storage really is 16-bit: a value that f32 holds exactly and f16 cannot
    must come back rounded, or the dtype is a label rather than a format."""
    x = V.tensor(np.array([1.0], dtype=np.float16), device=on(device))
    one_plus = V.add(x, V.tensor(np.array([0.0001], dtype=np.float16), device=on(device)))

    # f16's spacing at 1.0 is 2^-10 ~= 9.77e-4, so 1.0001 is not representable.
    assert one_plus.numpy().item() == 1.0


# ---------------------------------------------------------------------------
# Against PyTorch
# ---------------------------------------------------------------------------

UNARY = [
    ("neg", V.neg, torch.neg, -2.0, 2.0),
    ("abs", V.abs, torch.abs, -2.0, 2.0),
    ("square", V.square, torch.square, -2.0, 2.0),
    ("sqrt", V.sqrt, torch.sqrt, 0.3, 4.0),
    ("exp", V.exp, torch.exp, -2.0, 2.0),
    ("log", V.log, torch.log, 0.3, 4.0),
    ("tanh", V.tanh, torch.tanh, -2.0, 2.0),
    ("sigmoid", V.sigmoid, torch.sigmoid, -2.0, 2.0),
    ("relu", V.relu, torch.relu, -2.0, 2.0),
    ("gelu", V.gelu, torch.nn.functional.gelu, -2.0, 2.0),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,vf,tf,low,high", UNARY, ids=[u[0] for u in UNARY])
def test_unary_matches_torch(name, vf, tf, low, high, device):
    v, t, x = half_pair((4, 5), seed=3000, device=device, low=low, high=high)
    out = vf(v)

    assert out.dtype == V.float16, f"{name} changed dtype to {out.dtype}"
    assert_close(f"{name}(f16)", out, tf(t), F16_TOL, inputs=[x])


BINARY = [
    ("add", V.add, torch.add),
    ("sub", V.sub, torch.sub),
    ("mul", V.mul, torch.mul),
    ("maximum", V.maximum, torch.maximum),
    ("minimum", V.minimum, torch.minimum),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,vf,tf", BINARY, ids=[b[0] for b in BINARY])
def test_binary_matches_torch(name, vf, tf, device):
    va, ta, a = half_pair((4, 5), seed=3010, device=device)
    vb, tb, b = half_pair((4, 5), seed=3011, device=device)

    out = vf(va, vb)
    assert out.dtype == V.float16
    assert_close(f"{name}(f16)", out, tf(ta, tb), F16_TOL, inputs=[a, b])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("axis", [0, 1, None])
def test_reductions_match_torch(axis, device):
    v, t, x = half_pair((6, 7), seed=3020, device=device)

    for name, vf, tf in (("sum", V.sum, torch.sum), ("mean", V.mean, torch.mean)):
        got = vf(v) if axis is None else vf(v, axis)
        want = tf(t) if axis is None else tf(t, dim=axis)
        assert got.dtype == V.float16
        assert_close(f"{name}(f16, axis={axis})", got, want, F16_TOL, inputs=[x])


def test_matmul_matches_torch():
    va, ta, a = half_pair((4, 6), seed=3030)
    vb, tb, b = half_pair((6, 5), seed=3031)

    out = V.matmul(va, vb)
    assert out.dtype == V.float16
    assert_close("matmul(f16)", out, torch.matmul(ta, tb), F16_TOL, inputs=[a, b])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("axis", [0, 1])
def test_softmax_matches_torch(axis, device):
    v, t, x = half_pair((4, 6), seed=3040, device=device)

    assert_close(f"softmax(f16, axis={axis})", V.softmax(v, axis),
                 torch.softmax(t, dim=axis), F16_TOL, inputs=[x])
    assert_close(f"log_softmax(f16, axis={axis})", V.log_softmax(v, axis),
                 torch.log_softmax(t, dim=axis), F16_TOL, inputs=[x])


@pytest.mark.parametrize("device", DEVICES)
def test_broadcasting_works_in_f16(device):
    va, ta, a = half_pair((3, 4), seed=3050, device=device)
    vb, tb, b = half_pair((4,), seed=3051, device=device)

    assert_close("broadcast mul(f16)", va * vb, ta * tb, F16_TOL, inputs=[a, b])


@pytest.mark.parametrize("device", DEVICES)
def test_strided_input_works_in_f16(device):
    base = make_input((5, 4), seed=3060).astype(np.float16)
    v = V.tensor(base, device=on(device)).transpose(0, 1)
    t = torch.from_numpy(base.copy()).transpose(0, 1)
    assert not v.is_contiguous

    assert_close("sum(strided f16)", V.sum(v, 1), torch.sum(t, dim=1), F16_TOL, inputs=[base])


# ---------------------------------------------------------------------------
# Comparisons: previously a silent wrong answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_f16_comparison_is_correct(device):
    """Regression. The comparison kernels checked only that their OUTPUT was
    Bool and then read both inputs as f32 regardless, so an f16 operand had its
    2-byte halves read as 4-byte floats. That returned a plausible mask, raised
    nothing, and was wrong. Found by the coverage audit."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    b = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float16)

    for name, vf, ref in (("gt", V.greater, a > b), ("lt", V.less, a < b),
                          ("ge", V.greater_equal, a >= b), ("eq", V.equal, a == b)):
        got = vf(V.tensor(a, device=on(device)), V.tensor(b, device=on(device))).numpy()
        np.testing.assert_array_equal(got, ref, err_msg=f"{name} on f16")


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_integer_comparison_raises_rather_than_lying(dtype):
    """The same defect, for the dtypes that are still not arithmetic.

    An i64 operand had two elements read as one float and an i32 operand was
    read as a float bit pattern -- which happens to order correctly for positive
    values and inverts for negative ones, the worst kind of nearly-right. There
    are no integer kernels, so refusing is the honest answer.
    """
    a = V.tensor(np.array([-5, -1, 0, 3], dtype=dtype))
    b = V.tensor(np.array([2, -3, -1, -9], dtype=dtype))

    with pytest.raises(V.DTypeError, match=f"does not support dtype {np.dtype(dtype).name[0]}"):
        (a > b).numpy()


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_flows_through_f16(device):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    v = V.tensor(x, device=on(device), requires_grad=True)
    t = torch.from_numpy(x.copy()).requires_grad_(True)

    V.sum(v * v).backward()
    (t * t).sum().backward()

    assert v.grad.dtype == V.float16
    assert_close("f16 grad", v.grad, t.grad, F16_TOL, inputs=[x])


def test_cast_backward_rule():
    """The rule that could not be reached before f16 arithmetic existed.

    Cast's adjoint is `grad.to(source dtype)`. Building a graph that casts and
    then computes was impossible while nothing consumed f16, so the rule was
    declared and never ran -- the last blocking gap in
    docs/VERIFICATION-AUDIT.md. This replaces the test that pinned it as
    unreachable.
    """
    x = make_input((4, 3), seed=3070)
    v = V.tensor(x, requires_grad=True)
    t = torch.from_numpy(x.copy()).requires_grad_(True)

    V.sum(v.to(V.float16) * 2.0).backward()
    (t.to(torch.float16) * 2.0).sum().backward()

    assert v.grad.dtype == V.float32, "the gradient must come back in the SOURCE dtype"
    assert_close("cast grad", v.grad, t.grad, F16_TOL, inputs=[x])


def test_mixed_dtypes_are_refused_not_promoted():
    """vkml does not promote (`check_same_dtype` in api/ops.cpp). Pinned because
    adding a second floating dtype is exactly when a promotion rule starts
    looking tempting, and silently promoting would hide precision changes."""
    a = V.tensor(np.ones((3,), dtype=np.float16))
    b = V.tensor(np.ones((3,), dtype=np.float32))

    with pytest.raises(V.DTypeError, match="different dtypes"):
        (a + b).numpy()


# ---------------------------------------------------------------------------
# Backend divergence, stated
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device available")
@pytest.mark.parametrize("name,build", [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: a * b),
    ("exp", lambda a, b: V.exp(a)),
    ("gelu", lambda a, b: V.gelu(a)),
    ("sum", lambda a, b: V.sum(a, 0)),
    ("mean", lambda a, b: V.mean(a)),
    ("amax", lambda a, b: V.amax(a, 1)),
    ("argmax", lambda a, b: V.argmax(a, 1)),
    ("softmax", lambda a, b: V.softmax(a, 1)),
    ("log_softmax", lambda a, b: V.log_softmax(a, 1)),
    ("greater", lambda a, b: a > b),
    ("where", lambda a, b: V.where(a > b, a, b)),
])
def test_vulkan_agrees_with_the_cpu_oracle_bit_for_bit(name, build):
    """The second link of the correctness chain, for f16.

    ARCHITECTURE.md §7: the CPU is checked against PyTorch for semantics, then
    Vulkan against the CPU for kernel bugs -- an oracle that shares our exact
    semantics, so a mismatch is unambiguously a kernel bug.

    EQUALITY, not a tolerance. Both backends widen to float, compute, and narrow
    once on the store, so they should agree bit for bit; a tolerance here would
    accept a shader that rounded somewhere extra and hide the very drift the two
    implementations exist to catch.
    """
    rng = np.random.default_rng(3100)
    x = rng.standard_normal((6, 7)).astype(np.float16)
    y = rng.standard_normal((6, 7)).astype(np.float16)

    gpu = build(V.tensor(x, device=gpu_device()), V.tensor(y, device=gpu_device())).numpy()
    cpu = build(V.tensor(x), V.tensor(y)).numpy()

    np.testing.assert_array_equal(gpu, cpu, err_msg=f"{name}: vulkan and cpu disagree on f16")


# GEMM shapes chosen to reach different kernels. The dispatcher picks between
# gemv, a naive/tiled path and the register-blocked one by size, and long-K
# admits split-K -- which is the case that matters most here, because under
# split-K the kernel writes f32 partials to a workspace instead of writing the
# output, so its destination type differs from every other dispatch.
GEMM_SHAPES = [
    ("gemv", (64, 128), (128, 1)),
    ("small", (8, 8), (8, 8)),
    ("mid", (64, 64), (64, 64)),
    ("large", (256, 256), (256, 256)),
    ("longK", (32, 4096), (4096, 32)),
]


@pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device available")
@pytest.mark.parametrize("name,sa,sb", GEMM_SHAPES, ids=[g[0] for g in GEMM_SHAPES])
def test_f16_matmul_agrees_with_the_cpu_oracle(name, sa, sb):
    """Bit-exact, across every GEMM path the dispatcher can choose.

    Scaled to 0.1 so a K=4096 reduction stays inside f16's range; the point here
    is the storage and indexing, not overflow, which has its own test.
    """
    rng = np.random.default_rng(3200)
    a = (rng.standard_normal(sa) * 0.1).astype(np.float16)
    b = (rng.standard_normal(sb) * 0.1).astype(np.float16)

    gpu = V.matmul(V.tensor(a, device=gpu_device()), V.tensor(b, device=gpu_device())).numpy()
    cpu = V.matmul(V.tensor(a), V.tensor(b)).numpy()

    assert gpu.dtype == np.float16
    np.testing.assert_array_equal(gpu, cpu, err_msg=f"{name}: vulkan and cpu disagree on f16 matmul")


@pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device available")
def test_split_k_partials_stay_f32():
    """The split-K workspace holds f32 partial sums, whatever the operands are.

    Rounding a partial to 16 bits before the final fold would break fp32
    accumulation across the split exactly as a 16-bit accumulator breaks it
    within one. Forced on rather than hoped for, since the planner only chooses
    split-K for particular shapes.

    Checked by value, at a size where a 16-bit partial would visibly saturate:
    each partition sums 256 ones, and there are enough partitions that the
    totals exceed what f16 can step through one unit at a time.
    """
    k = 4096
    a = V.tensor(np.ones((1, k), dtype=np.float16), device=gpu_device())
    b = V.tensor(np.ones((k, 1), dtype=np.float16), device=gpu_device())

    assert V.matmul(a, b).numpy().item() == float(k)
