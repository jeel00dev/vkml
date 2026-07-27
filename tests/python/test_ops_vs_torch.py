"""Forward-pass validation of every implemented operator against PyTorch."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vkml as V
from conftest import (
    TOLERANCES,
    assert_close,
    assert_close_dot,
    assert_dtype,
    assert_shape,
    assert_strides,
    make_input,
    pair,
)

# ---------------------------------------------------------------------------
# Unary elementwise
#
# `domain` keeps inputs where the function is defined: log and sqrt need
# positives, and feeding them negatives would compare NaN against NaN and prove
# nothing.
# ---------------------------------------------------------------------------

UNARY = [
    ("neg", V.neg, torch.neg, "any", "elementwise"),
    ("abs", V.abs, torch.abs, "any", "elementwise"),
    ("sign", V.sign, torch.sign, "any", "elementwise"),
    ("square", V.square, torch.square, "any", "elementwise"),
    ("sqrt", V.sqrt, torch.sqrt, "positive", "elementwise"),
    ("rsqrt", V.rsqrt, torch.rsqrt, "positive", "transcendental"),
    ("reciprocal", V.reciprocal, torch.reciprocal, "nonzero", "elementwise"),
    ("exp", V.exp, torch.exp, "small", "transcendental"),
    ("log", V.log, torch.log, "positive", "transcendental"),
    ("sin", V.sin, torch.sin, "any", "transcendental"),
    ("cos", V.cos, torch.cos, "any", "transcendental"),
    ("tanh", V.tanh, torch.tanh, "any", "transcendental"),
    ("sigmoid", V.sigmoid, torch.sigmoid, "any", "transcendental"),
    ("relu", V.relu, torch.relu, "any", "elementwise"),
    ("gelu", V.gelu, torch.nn.functional.gelu, "any", "transcendental"),
    ("silu", V.silu, torch.nn.functional.silu, "any", "transcendental"),
]

DOMAINS = {
    "any": (-3.0, 3.0),
    "positive": (0.1, 4.0),
    "nonzero": (0.5, 3.0),
    "small": (-4.0, 4.0),
}

SHAPES = [(), (1,), (7,), (3, 4), (2, 3, 4), (2, 2, 3, 3)]


@pytest.mark.parametrize("name,vf,tf,domain,tol_key", UNARY, ids=[u[0] for u in UNARY])
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_unary(name, vf, tf, domain, tol_key, shape):
    lo, hi = DOMAINS[domain]
    x = make_input(shape, seed=hash((name, shape)) % 2**31, low=lo, high=hi)
    v, t = pair(x)

    got = vf(v)
    want = tf(t)

    assert_shape(name, got, want)
    assert_dtype(name, got, want)
    assert_close(name, got, want, TOLERANCES[tol_key], inputs=[x])


# ---------------------------------------------------------------------------
# Binary elementwise
# ---------------------------------------------------------------------------

BINARY = [
    ("add", V.add, torch.add, "any"),
    ("sub", V.sub, torch.sub, "any"),
    ("mul", V.mul, torch.mul, "any"),
    ("div", V.div, torch.div, "nonzero"),
    ("maximum", V.maximum, torch.maximum, "any"),
    ("minimum", V.minimum, torch.minimum, "any"),
]

BROADCAST_PAIRS = [
    ((3, 4), (3, 4)),      # identical
    ((3, 4), (4,)),        # trailing broadcast
    ((3, 1), (1, 4)),      # mutual broadcast
    ((2, 3, 4), (4,)),
    ((2, 3, 4), (3, 1)),
    ((5,), ()),            # scalar
    ((1,), (6,)),          # singleton stretched
]


@pytest.mark.parametrize("name,vf,tf,domain", BINARY, ids=[b[0] for b in BINARY])
@pytest.mark.parametrize("sa,sb", BROADCAST_PAIRS, ids=[f"{a}x{b}" for a, b in BROADCAST_PAIRS])
def test_binary_broadcast(name, vf, tf, domain, sa, sb):
    lo, hi = DOMAINS[domain]
    a = make_input(sa, seed=1, low=lo, high=hi)
    b = make_input(sb, seed=2, low=lo, high=hi)

    va, ta = pair(a)
    vb, tb = pair(b)

    got = vf(va, vb)
    want = tf(ta, tb)

    assert_shape(name, got, want)
    assert_close(name, got, want, TOLERANCES["elementwise"], inputs=[a, b])


def test_pow():
    base = make_input((4, 3), seed=3, low=0.2, high=3.0)
    expo = make_input((4, 3), seed=4, low=-2.0, high=3.0)
    va, ta = pair(base)
    vb, tb = pair(expo)
    assert_close("pow", V.pow(va, vb), torch.pow(ta, tb), TOLERANCES["transcendental"],
                 inputs=[base, expo])


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

COMPARE = [
    ("less", V.less, torch.lt),
    ("greater", V.greater, torch.gt),
    ("less_equal", V.less_equal, torch.le),
    ("greater_equal", V.greater_equal, torch.ge),
    ("equal", V.equal, torch.eq),
    ("not_equal", V.not_equal, torch.ne),
]


@pytest.mark.parametrize("name,vf,tf", COMPARE, ids=[c[0] for c in COMPARE])
def test_comparison(name, vf, tf):
    # Quantised inputs so that equality actually fires sometimes.
    a = np.round(make_input((5, 4), seed=5) * 2) / 2
    b = np.round(make_input((5, 4), seed=6) * 2) / 2
    va, ta = pair(a)
    vb, tb = pair(b)

    got = vf(va, vb)
    want = tf(ta, tb)

    assert_shape(name, got, want)
    assert_dtype(name, got, want)
    assert np.array_equal(got.numpy(), want.numpy()), f"{name}: boolean mismatch"


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

REDUCE_SHAPES = [(6,), (3, 4), (2, 3, 4), (2, 2, 3, 3)]


@pytest.mark.parametrize("shape", REDUCE_SHAPES, ids=[str(s) for s in REDUCE_SHAPES])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_mean_all_axes(shape, keepdim):
    x = make_input(shape, seed=7)
    v, t = pair(x)

    for name, vf, tf in (("sum", V.sum, torch.sum), ("mean", V.mean, torch.mean)):
        got = vf(v, None, keepdim) if keepdim else vf(v)
        want = tf(t) if not keepdim else tf(t, dim=tuple(range(t.dim())), keepdim=True)
        assert_shape(name, got, want)
        assert_close(name, got, want, TOLERANCES["reduction"], inputs=[x])


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)], ids=["2d", "3d"])
@pytest.mark.parametrize("keepdim", [False, True])
def test_reduce_single_axis(shape, keepdim):
    x = make_input(shape, seed=8)
    v, t = pair(x)

    for axis in range(len(shape)):
        for name, vf, tf in (
            ("sum", V.sum, torch.sum),
            ("mean", V.mean, torch.mean),
            ("amax", V.amax, torch.amax),
            ("amin", V.amin, torch.amin),
        ):
            got = vf(v, axis, keepdim)
            want = tf(t, dim=axis, keepdim=keepdim)
            assert_shape(f"{name}(dim={axis})", got, want)
            assert_close(f"{name}(dim={axis},keepdim={keepdim})", got, want,
                         TOLERANCES["reduction"], inputs=[x])


def test_reduce_multiple_axes():
    x = make_input((2, 3, 4), seed=9)
    v, t = pair(x)
    got = V.sum(v, [0, 2])
    want = torch.sum(t, dim=(0, 2))
    assert_shape("sum(dim=(0,2))", got, want)
    assert_close("sum(dim=(0,2))", got, want, TOLERANCES["reduction"], inputs=[x])


def test_negative_axis():
    x = make_input((2, 3, 4), seed=10)
    v, t = pair(x)
    assert_close("sum(dim=-1)", V.sum(v, -1), torch.sum(t, dim=-1), TOLERANCES["reduction"])


@pytest.mark.parametrize("axis", [0, 1])
def test_argmax_argmin(axis):
    # Distinct values: ties are resolved differently by different frameworks and
    # comparing them would test the tie rule, not the op.
    x = make_input((5, 6), seed=11).astype(np.float32)
    x = x + np.arange(x.size, dtype=np.float32).reshape(x.shape) * 1e-3
    v, t = pair(x)

    for name, vf, tf in (("argmax", V.argmax, torch.argmax), ("argmin", V.argmin, torch.argmin)):
        got = vf(v, axis)
        want = tf(t, dim=axis)
        assert_shape(name, got, want)
        assert_dtype(name, got, want)
        assert np.array_equal(got.numpy(), want.numpy()), f"{name}: index mismatch"


def test_prod():
    x = make_input((3, 4), seed=12, low=0.5, high=1.5)
    v, t = pair(x)
    assert_close("prod", V.prod(v), torch.prod(t), TOLERANCES["reduction"], inputs=[x])


# ---------------------------------------------------------------------------
# Softmax family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,axis", [((5,), 0), ((3, 4), 1), ((3, 4), 0), ((2, 3, 4), -1)])
def test_softmax(shape, axis):
    x = make_input(shape, seed=13, low=-5.0, high=5.0)
    v, t = pair(x)
    assert_close(f"softmax(dim={axis})", V.softmax(v, axis),
                 torch.softmax(t, dim=axis), TOLERANCES["transcendental"], inputs=[x])
    assert_close(f"log_softmax(dim={axis})", V.log_softmax(v, axis),
                 torch.log_softmax(t, dim=axis), TOLERANCES["transcendental"], inputs=[x])


def test_softmax_extreme_logits():
    """Overflow and underflow, where the max-subtraction trick earns its keep."""
    x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]], dtype=np.float32)
    v, t = pair(x)
    assert_close("softmax(extreme)", V.softmax(v, -1), torch.softmax(t, dim=-1),
                 TOLERANCES["transcendental"], inputs=[x])
    assert_close("log_softmax(extreme)", V.log_softmax(v, -1), torch.log_softmax(t, dim=-1),
                 TOLERANCES["transcendental"], inputs=[x])


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

MATMUL_SHAPES = [
    ((3, 4), (4, 5)),
    ((1, 1), (1, 1)),
    ((8, 16), (16, 4)),
    ((2, 3, 4), (2, 4, 5)),      # batched
    ((2, 3, 4), (4, 5)),         # batch against plain matrix
    ((4,), (4,)),                # dot
    ((3, 4), (4,)),              # matrix-vector
    ((3,), (3, 5)),              # vector-matrix
    ((2, 2, 3, 4), (2, 2, 4, 5)),  # two batch axes
]


@pytest.mark.parametrize("sa,sb", MATMUL_SHAPES, ids=[f"{a}@{b}" for a, b in MATMUL_SHAPES])
def test_matmul(sa, sb):
    a = make_input(sa, seed=14)
    b = make_input(sb, seed=15)
    va, ta = pair(a)
    vb, tb = pair(b)

    got = V.matmul(va, vb)
    want = torch.matmul(ta, tb)

    assert_shape("matmul", got, want)
    assert_close("matmul", got, want, TOLERANCES["matmul"], inputs=[a, b])


def test_matmul_large_k_backward_error():
    """K = 4096 with sign-mixed inputs, i.e. heavy cancellation.

    Checked against the backward-error bound rather than a relative tolerance;
    see the analysis above assert_close_dot in conftest.
    """
    a = make_input((4, 4096), seed=16)
    b = make_input((4096, 4), seed=17)
    va, ta = pair(a)
    vb, tb = pair(b)
    assert_close_dot("matmul(K=4096, cancelling)", V.matmul(va, vb), torch.matmul(ta, tb), a, b)


def test_matmul_well_conditioned_large_k():
    """K = 4096 with positive inputs, so nothing cancels.

    This is the test that actually demonstrates the pairwise reduction is
    accurate: with a condition number near 1, the same K holds the 1e-5 gate
    with room to spare. A sequential accumulator would miss it by ~49x.
    """
    a = make_input((4, 4096), seed=16, low=0.5, high=2.0)
    b = make_input((4096, 4), seed=17, low=0.5, high=2.0)
    va, ta = pair(a)
    vb, tb = pair(b)
    assert_close("matmul(K=4096, positive)", V.matmul(va, vb), torch.matmul(ta, tb),
                 TOLERANCES["matmul"], inputs=[a, b])


def test_matmul_transposed_operand():
    """Exercises the strided read path in the kernel."""
    a = make_input((3, 4), seed=18)
    b = make_input((5, 4), seed=19)
    va, ta = pair(a)
    vb, tb = pair(b)
    assert_close("matmul(a, b.T)", V.matmul(va, vb.T), torch.matmul(ta, tb.T),
                 TOLERANCES["matmul"], inputs=[a, b])


# ---------------------------------------------------------------------------
# Views, strides, layout
# ---------------------------------------------------------------------------


def test_view_ops_match_torch_shapes_and_strides():
    x = make_input((2, 3, 4), seed=20)
    v, t = pair(x)

    cases = [
        ("transpose(0,1)", v.transpose(0, 1), t.transpose(0, 1)),
        ("permute(2,0,1)", v.permute([2, 0, 1]), t.permute(2, 0, 1)),
        ("reshape(6,4)", v.reshape([6, 4]), t.reshape(6, 4)),
        ("unsqueeze(1)", v.unsqueeze(1), t.unsqueeze(1)),
        ("slice", v[:, 1:3, :], t[:, 1:3, :]),
    ]
    for name, got, want in cases:
        assert_shape(name, got, want)
        assert_strides(name, got, want)
        assert_close(name, got, want, TOLERANCES["elementwise"], inputs=[x])


def test_arithmetic_on_non_contiguous_views():
    """A strided operand must produce the same answer as a contiguous one."""
    x = make_input((4, 5), seed=21)
    v, t = pair(x)

    vt = v.T
    tt = t.T
    assert not vt.is_contiguous
    assert not tt.is_contiguous()

    assert_close("transposed + 1", vt + 1.0, tt + 1.0, TOLERANCES["elementwise"], inputs=[x])
    assert_close("relu(transposed)", V.relu(vt), torch.relu(tt), TOLERANCES["elementwise"])
    assert_close("sum(transposed, 0)", V.sum(vt, 0), torch.sum(tt, dim=0),
                 TOLERANCES["reduction"])
    assert_close("softmax(transposed)", V.softmax(vt, -1), torch.softmax(tt, dim=-1),
                 TOLERANCES["transcendental"])


def test_squeeze_unsqueeze_roundtrip():
    x = make_input((3, 1, 4), seed=22)
    v, t = pair(x)
    assert_close("squeeze(1)", v.squeeze(1), t.squeeze(1), TOLERANCES["elementwise"])
    assert_close("squeeze->unsqueeze", v.squeeze(1).unsqueeze(1), t.squeeze(1).unsqueeze(1),
                 TOLERANCES["elementwise"])


def test_getitem_matches_numpy_semantics():
    x = make_input((4, 5, 6), seed=23)
    v, t = pair(x)

    cases = [
        ("[0]", v[0], t[0]),
        ("[-1]", v[-1], t[-1]),
        ("[1:3]", v[1:3], t[1:3]),
        ("[:, 2]", v[:, 2], t[:, 2]),
        ("[1, 2]", v[1, 2], t[1, 2]),
        ("[:, 1:4, 2]", v[:, 1:4, 2], t[:, 1:4, 2]),
        ("[::2]", v[::2], t[::2]),
    ]
    for name, got, want in cases:
        assert_shape(name, got, want)
        assert_close(name, got, want, TOLERANCES["elementwise"], inputs=[x])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_tensor():
    x = np.zeros((0, 5), dtype=np.float32)
    v, t = pair(x)
    assert v.size == 0
    assert_shape("empty add", v + 1.0, t + 1.0)
    assert_close("empty add", v + 1.0, t + 1.0)
    assert_close("empty sum", V.sum(v), torch.sum(t), TOLERANCES["reduction"])


def test_scalar_tensor():
    x = np.array(3.5, dtype=np.float32)
    v, t = pair(x)
    assert v.ndim == 0
    assert_close("scalar mul", v * 2.0, t * 2.0)
    assert v.item() == pytest.approx(3.5)


def test_singleton_dims():
    x = make_input((1, 1, 5), seed=24)
    v, t = pair(x)
    assert_close("singleton sum", V.sum(v, 2), torch.sum(t, dim=2), TOLERANCES["reduction"])
    assert_close("singleton squeeze", v.squeeze(0), t.squeeze(0))


def test_large_tensor():
    x = make_input((512, 512), seed=25)
    v, t = pair(x)
    assert_close("large sum", V.sum(v), torch.sum(t), TOLERANCES["reduction"], inputs=[x])
    assert_close("large relu", V.relu(v), torch.relu(t), TOLERANCES["elementwise"])


def test_clamp_and_where():
    x = make_input((4, 5), seed=26)
    v, t = pair(x)
    assert_close("clamp", V.clamp(v, -0.5, 0.5), torch.clamp(t, -0.5, 0.5))
    assert_close("clamp_min", V.clamp_min(v, 0.0), torch.clamp_min(t, 0.0))

    cond_v = V.greater(v, V.zeros([4, 5]))
    cond_t = t > 0
    assert_close("where", V.where(cond_v, v, V.zeros([4, 5])),
                 torch.where(cond_t, t, torch.zeros(4, 5)))


@pytest.mark.parametrize("diagonal", [0, 1, 2, -1, -2, 5, -5])
@pytest.mark.parametrize("shape", [(4, 4), (3, 5), (5, 3), (2, 3, 4), (2, 2, 3, 3), (1, 1)])
def test_triu_tril(shape, diagonal):
    """Non-square shapes and out-of-range diagonals are the cases that catch an
    off-by-one: at diagonal 5 on a 4-wide tensor triu must return all zeros and
    tril must return the input unchanged, and swapping the comparison silently
    inverts both."""
    x = make_input(shape, seed=40)
    v, t = pair(x)
    assert_close(f"triu(d={diagonal})", V.triu(v, diagonal), torch.triu(t, diagonal))
    assert_close(f"tril(d={diagonal})", V.tril(v, diagonal), torch.tril(t, diagonal))


def test_triu_tril_partition_the_tensor():
    """triu(d) and tril(d-1) are complementary: every element belongs to exactly
    one of them, so the two must sum back to the original. Independent of
    PyTorch, and it fails if either side's boundary is off by one."""
    x = make_input((6, 6), seed=41)
    v, _ = pair(x)
    for d in (-2, 0, 3):
        total = V.add(V.triu(v, d), V.tril(v, d - 1)).numpy()
        assert np.allclose(total, x), f"triu(d={d}) + tril(d={d - 1}) != x"


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("shapes", [
    [(2, 3), (2, 3)],
    [(2, 3), (2, 5)],          # unequal along the joined axis
    [(1, 4), (3, 4)],
    [(2, 3), (2, 3), (2, 3)],  # three operands: exercises the left fold
    [(2, 3, 4), (2, 3, 4)],
])
def test_cat(shapes, axis):
    if axis >= len(shapes[0]):
        pytest.skip("axis out of range for this rank")

    arrays = [make_input(s, seed=50 + i) for i, s in enumerate(shapes)]
    try:
        want = torch.cat([torch.from_numpy(a.copy()) for a in arrays], dim=axis)
    except RuntimeError:
        pytest.skip("shape combination is not concatenable on this axis")

    got = V.cat([V.tensor(a) for a in arrays], axis)
    assert_close(f"cat(axis={axis})", got, want, inputs=arrays)


@pytest.mark.parametrize("axis", [0, 1])
def test_cat_gradients_split_back(axis):
    """Concatenation is a permutation, so each operand must take back exactly
    the slice it contributed. Different-sized operands and a non-uniform
    upstream gradient are what make a swapped or mis-offset slice visible."""
    a = make_input((3, 4), seed=60)
    b = make_input((5, 4) if axis == 0 else (3, 2), seed=61)

    va = V.tensor(a, requires_grad=True)
    vb = V.tensor(b, requires_grad=True)
    ta = torch.from_numpy(a.copy()).requires_grad_(True)
    tb = torch.from_numpy(b.copy()).requires_grad_(True)

    # Weight by position so a mis-offset slice cannot cancel out.
    vw = V.cat([va, vb], axis)
    tw = torch.cat([ta, tb], dim=axis)
    V.sum(V.mul(vw, vw)).backward()
    (tw * tw).sum().backward()

    assert_close("cat grad a", va.grad, ta.grad, inputs=[a])
    assert_close("cat grad b", vb.grad, tb.grad, inputs=[b])


def test_cat_rejects_mismatched_non_joined_axis():
    a = V.tensor(make_input((2, 3), seed=62))
    b = V.tensor(make_input((2, 4), seed=63))
    with pytest.raises(V.ShapeError):
        V.cat([a, b], 0)  # differs on axis 1, which is not the joined axis


@pytest.mark.parametrize("mask_shape", [(4, 5), (1, 5), (5,), (4, 1)])
def test_masked_fill(mask_shape):
    """Also covers mask broadcasting, which is where the composition earns its
    keep: `where` already broadcasts all three operands, so masked_fill gets it
    without implementing a rule of its own."""
    x = make_input((4, 5), seed=43)
    m = make_input(mask_shape, seed=44) > 0.0

    v, t = pair(x)
    vm = V.tensor(m)
    tm = torch.from_numpy(m.copy())

    assert_close("masked_fill", V.masked_fill(v, vm, -1e9),
                 torch.masked_fill(t, tm, -1e9), inputs=[x])


def test_masked_fill_gradient_skips_filled_positions():
    """The filled value is a constant, so gradient must reach only the
    unmasked elements. This is the property attention masking depends on."""
    x = make_input((3, 4), seed=45)
    m = np.zeros((3, 4), dtype=bool)
    m[:, 2:] = True

    v = V.tensor(x, requires_grad=True)
    t = torch.from_numpy(x.copy()).requires_grad_(True)

    V.sum(V.masked_fill(v, V.tensor(m), 0.0)).backward()
    torch.masked_fill(t, torch.from_numpy(m.copy()), 0.0).sum().backward()

    assert_close("masked_fill grad", v.grad, t.grad, inputs=[x])
    assert np.allclose(v.grad.numpy()[:, 2:], 0.0), "filled positions received gradient"


def test_masked_fill_rejects_non_bool_mask():
    v = V.tensor(make_input((3, 3), seed=46))
    with pytest.raises(V.DTypeError):
        V.masked_fill(v, V.tensor(make_input((3, 3), seed=47)), 0.0)


def test_triu_requires_rank_two():
    v = V.tensor(make_input((5,), seed=42))
    with pytest.raises(V.ShapeError):
        V.triu(v)


def test_cast_roundtrip():
    x = make_input((3, 4), seed=27)
    v, t = pair(x)
    assert_close("to(int32)", v.to(V.int32).numpy(), t.to(torch.int32).numpy())
    assert_close("to(float16)", v.to(V.float16).to(V.float32),
                 t.to(torch.float16).to(torch.float32), TOLERANCES["fp16"])


def test_nan_and_inf_propagation():
    x = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    v, t = pair(x)
    got = (v + 1.0).numpy()
    want = (t + 1.0).numpy()
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.allclose(got[~np.isnan(got)], want[~np.isnan(want)], equal_nan=True)


def test_dtype_promotion_is_strict():
    """Documented divergence: vkml refuses mixed dtypes rather than promoting.

    Stricter than PyTorch, so any program accepted here behaves identically
    there. See the note in src/api/ops.cpp and docs/adr/0003.
    """
    a = V.tensor(np.ones((3,), dtype=np.float32))
    b = a.to(V.int32)
    with pytest.raises(TypeError):
        _ = a + b
