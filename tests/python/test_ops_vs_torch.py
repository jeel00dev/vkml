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


@pytest.mark.parametrize("shape,axes", [
    ((4, 8), 1),
    ((2, 3, 16), 1),
    ((2, 3, 16), 2),      # normalise over two trailing axes
    ((2, 3, 4, 5), 3),
    ((1, 64), 1),
    ((3, 1), 1),          # width 1: variance is exactly 0, so eps carries it
])
def test_layer_norm(shape, axes):
    x = make_input(shape, seed=70)
    v, t = pair(x)
    normalized_shape = list(shape[len(shape) - axes:])
    assert_close(f"layer_norm(axes={axes})", V.layer_norm(v, axes),
                 torch.nn.functional.layer_norm(t, normalized_shape),
                 TOLERANCES["transcendental"], inputs=[x])


# eps is passed explicitly throughout. torch's rms_norm defaults it to
# finfo(f32).eps = 1.19e-7 while vkml uses 1e-5 (see ops.h), so comparing the
# defaults would compare two different functions -- which is exactly what the
# first version of this test did, and the gradient check caught it.
RMS_EPS = 1e-5


@pytest.mark.parametrize("shape,axes", [((4, 8), 1), ((2, 3, 16), 1), ((2, 3, 4, 5), 2)])
def test_rms_norm(shape, axes):
    x = make_input(shape, seed=71)
    v, t = pair(x)
    normalized_shape = list(shape[len(shape) - axes:])
    assert_close(f"rms_norm(axes={axes})", V.rms_norm(v, axes, RMS_EPS),
                 torch.nn.functional.rms_norm(t, normalized_shape, eps=RMS_EPS),
                 TOLERANCES["transcendental"], inputs=[x])


def test_rms_norm_default_eps_diverges_from_torch_deliberately():
    """Pins the divergence documented in ops.h so it stays a decision rather
    than drifting into a surprise.

    Checked on the gradient, not the forward pass. Forward, the two epsilons
    differ by only ~3e-6 relative and a default-tolerance comparison would call
    them equal. The gradient of an rms-normalised quantity is proportional to
    eps, so an 84x difference in eps shows up as roughly that -- which is how
    this was noticed at all.
    """
    x = make_input((2, 32), seed=76)

    def grad_with(eps):
        v = V.tensor(x, requires_grad=True)
        y = V.rms_norm(v, 1) if eps is None else V.rms_norm(v, 1, eps)
        V.sum(V.mul(y, y)).backward()
        return v.grad.numpy()

    def torch_grad_with(eps):
        t = torch.from_numpy(x.copy()).requires_grad_(True)
        y = torch.nn.functional.rms_norm(t, [32], eps=eps)
        (y * y).sum().backward()
        return t.grad.numpy()

    ours = np.abs(grad_with(None)).max()
    theirs = np.abs(torch_grad_with(None)).max()
    assert ours > 10.0 * theirs, (
        f"defaults now agree (ours {ours:.2e}, torch {theirs:.2e}); "
        f"update ops.h and delete this test"
    )

    # And they agree once eps is stated, which is what the parity tests rely on.
    assert_close("rms_norm grad, explicit eps", grad_with(1e-5), torch_grad_with(1e-5),
                 TOLERANCES["transcendental"], inputs=[x])


def test_layer_norm_output_is_standardised():
    """Independent of torch: the defining property is that each normalised
    group has mean 0 and variance 1. A sign error or a missing rsqrt still
    produces plausible-looking numbers but fails this."""
    x = make_input((6, 128), seed=72, low=-50.0, high=50.0)
    y = V.layer_norm(V.tensor(x), 1).numpy()

    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5), y.mean(axis=-1)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-3), y.var(axis=-1)


def test_layer_norm_survives_a_large_offset():
    """The reason the two-pass form is used. With a mean of 1e6 and a spread of
    1, the one-pass identity var = E[x^2] - E[x]^2 subtracts 1e12 from 1e12 and
    loses every significant digit. This must still standardise."""
    rng = np.random.default_rng(73)
    x = (rng.normal(0.0, 1.0, (4, 256)) + 1.0e6).astype(np.float32)

    y = V.layer_norm(V.tensor(x), 1).numpy()
    assert np.isfinite(y).all(), "non-finite output"
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-2), y.var(axis=-1)


@pytest.mark.parametrize("fn,tf", [
    ("layer_norm", torch.nn.functional.layer_norm),
    ("rms_norm", torch.nn.functional.rms_norm),
])
def test_norm_gradients(fn, tf):
    """The composition differentiates through mean/sub/square/rsqrt with no
    rule of its own, which is the point of graph-based autograd -- but the
    gradient of a normalisation is not obvious (every output depends on every
    input in the group), so it is checked rather than assumed."""
    x = make_input((3, 12), seed=74)
    v = V.tensor(x, requires_grad=True)
    t = torch.from_numpy(x.copy()).requires_grad_(True)

    V.sum(V.mul(getattr(V, fn)(v, 1, RMS_EPS), getattr(V, fn)(v, 1, RMS_EPS))).backward()
    (tf(t, [12], eps=RMS_EPS) * tf(t, [12], eps=RMS_EPS)).sum().backward()

    assert_close(f"{fn} grad", v.grad, t.grad, TOLERANCES["transcendental"], inputs=[x])


def test_norm_rejects_bad_axis_count():
    v = V.tensor(make_input((3, 4), seed=75))
    with pytest.raises(V.ShapeError):
        V.layer_norm(v, 3)
    with pytest.raises(V.ShapeError):
        V.rms_norm(v, 0)


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


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,axis,idx", [
    ((5, 3), 0, [0, 2, 4]),
    ((5, 3), 1, [2, 0]),
    ((5, 3), 0, [1, 1, 1]),        # repeated: legal for a gather
    ((4, 3, 2), 1, [2, 0, 1]),
    ((6,), 0, [5, 0, 3]),
    ((5, 3), 0, []),               # empty index
])
def test_index_select(shape, axis, idx):
    x = make_input(shape, seed=80)
    v, t = pair(x)
    i = np.array(idx, dtype=np.int64)
    assert_close(f"index_select(axis={axis})", V.index_select(v, axis, V.tensor(i)),
                 torch.index_select(t, axis, torch.from_numpy(i.copy())), inputs=[x])


@pytest.mark.parametrize("axis,idx,dim_size", [
    (0, [0, 2, 2, 1], 4),          # index 2 repeated: the accumulating case
    (0, [1, 1, 1, 1], 3),          # all to one row
    (0, [0, 1, 2, 3], 4),          # a permutation, so no accumulation
    (1, [0, 0, 1], 2),
])
def test_scatter_add(axis, idx, dim_size):
    """Repeated indices are the whole point: several source slices land on one
    destination, which is what stops this composing from elementwise ops."""
    i = np.array(idx, dtype=np.int64)
    shape = (len(idx), 3) if axis == 0 else (3, len(idx))
    x = make_input(shape, seed=81)

    v, t = pair(x)
    out_shape = list(shape)
    out_shape[axis] = dim_size

    want = torch.zeros(out_shape).index_add_(axis, torch.from_numpy(i.copy()), t)
    assert_close(f"scatter_add(axis={axis})", V.scatter_add(v, axis, V.tensor(i), dim_size),
                 want, inputs=[x])


def test_scatter_add_and_index_select_are_adjoint():
    """<scatter_add(u), v> == <u, index_select(v)> for all u, v -- the defining
    property of an adjoint pair, and what makes each the other's gradient.
    Checked numerically rather than assumed, because getting it wrong produces
    gradients that are plausible but silently wrong."""
    rng = np.random.default_rng(82)
    i = np.array([0, 2, 2, 1, 0], dtype=np.int64)
    u = rng.normal(size=(5, 4)).astype(np.float32)      # source layout
    w = rng.normal(size=(3, 4)).astype(np.float32)      # destination layout

    lhs = float((V.scatter_add(V.tensor(u), 0, V.tensor(i), 3).numpy() * w).sum())
    rhs = float((u * V.index_select(V.tensor(w), 0, V.tensor(i)).numpy()).sum())
    assert abs(lhs - rhs) <= 1e-4 * max(1.0, abs(lhs)), f"{lhs} vs {rhs}"


def test_embedding_backward_matches_torch():
    """The reason scatter_add exists. An embedding lookup is index_select, and
    its gradient accumulates every occurrence of a token back onto one row."""
    rng = np.random.default_rng(83)
    weight = rng.normal(size=(6, 4)).astype(np.float32)
    tokens = np.array([0, 3, 3, 1, 0, 3], dtype=np.int64)

    v = V.tensor(weight, requires_grad=True)
    t = torch.from_numpy(weight.copy()).requires_grad_(True)

    V.sum(V.mul(V.index_select(v, 0, V.tensor(tokens)),
                V.index_select(v, 0, V.tensor(tokens)))).backward()
    (torch.index_select(t, 0, torch.from_numpy(tokens.copy())) ** 2).sum().backward()

    assert_close("embedding grad", v.grad, t.grad, inputs=[weight])
    # Token 3 appears three times, token 2 never: the gradient must reflect both.
    assert np.allclose(v.grad.numpy()[2], 0.0), "unused row received gradient"
    assert np.abs(v.grad.numpy()[3]).max() > 0.0, "repeated row received none"


def test_index_select_rejects_out_of_range():
    v = V.tensor(make_input((4, 2), seed=84))
    with pytest.raises(V.IndexError_):
        V.index_select(v, 0, V.tensor(np.array([0, 9], dtype=np.int64))).numpy()


def test_index_select_rejects_non_i64_index():
    v = V.tensor(make_input((4, 2), seed=85))
    with pytest.raises(V.DTypeError):
        V.index_select(v, 0, V.tensor(np.array([0.0, 1.0], dtype=np.float32)))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

REDUCTIONS = [("mean", V.Reduction.mean), ("sum", V.Reduction.sum), ("none", V.Reduction.none)]


@pytest.mark.parametrize("name,red", REDUCTIONS, ids=[r[0] for r in REDUCTIONS])
@pytest.mark.parametrize("shape", [(8,), (4, 5), (2, 3, 4)])
def test_mse_loss(shape, name, red):
    a = make_input(shape, seed=90)
    b = make_input(shape, seed=91)
    va, ta = pair(a)
    vb, tb = pair(b)
    assert_close(f"mse_loss({name})", V.mse_loss(va, vb, red),
                 torch.nn.functional.mse_loss(ta, tb, reduction=name),
                 TOLERANCES["reduction"], inputs=[a, b])


@pytest.mark.parametrize("name,red", REDUCTIONS, ids=[r[0] for r in REDUCTIONS])
@pytest.mark.parametrize("n,c", [(4, 3), (1, 2), (16, 10), (3, 100)])
def test_cross_entropy(n, c, name, red):
    logits = make_input((n, c), seed=92)
    labels = np.random.default_rng(93).integers(0, c, size=n).astype(np.int64)

    v, t = pair(logits)
    assert_close(f"cross_entropy({name})", V.cross_entropy(v, V.tensor(labels), red),
                 torch.nn.functional.cross_entropy(
                     t, torch.from_numpy(labels.copy()), reduction=name),
                 TOLERANCES["transcendental"], inputs=[logits])


def test_cross_entropy_survives_confident_logits():
    """The reason this is built on log_softmax rather than log(softmax(x)).

    At a 400-logit gap the losing classes underflow softmax to exactly 0, so
    log(softmax(x)) is -inf and every gradient in the batch becomes NaN. The
    log-sum-exp form must stay finite and give the near-zero loss the correct
    prediction deserves.
    """
    logits = np.array([[400.0, 0.0, -400.0],
                       [-400.0, 400.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    v = V.tensor(logits, requires_grad=True)
    loss = V.cross_entropy(v, V.tensor(labels))
    value = float(loss.numpy())

    assert np.isfinite(value), f"loss is not finite: {value}"
    assert value < 1e-6, f"a confidently correct prediction should cost ~0, got {value}"

    loss.backward()
    assert np.isfinite(v.grad.numpy()).all(), "gradient contains non-finite values"

    # And the naive form really would have failed here, so the test is not
    # asserting something that holds either way.
    assert float(V.softmax(V.tensor(logits), -1).numpy()[0, 2]) == 0.0


def test_cross_entropy_gradients():
    logits = make_input((6, 5), seed=94)
    labels = np.random.default_rng(95).integers(0, 5, size=6).astype(np.int64)

    v = V.tensor(logits, requires_grad=True)
    t = torch.from_numpy(logits.copy()).requires_grad_(True)

    V.cross_entropy(v, V.tensor(labels)).backward()
    torch.nn.functional.cross_entropy(t, torch.from_numpy(labels.copy())).backward()

    assert_close("cross_entropy grad", v.grad, t.grad, TOLERANCES["transcendental"],
                 inputs=[logits])


def test_cross_entropy_gradient_is_softmax_minus_onehot():
    """The closed form, independent of torch: d/dlogits = (softmax - onehot)/N.
    Checking it separately means a bug would have to corrupt both the reference
    and the composition in the same way to pass."""
    logits = make_input((4, 3), seed=96)
    labels = np.array([0, 2, 1, 2], dtype=np.int64)

    v = V.tensor(logits, requires_grad=True)
    V.cross_entropy(v, V.tensor(labels)).backward()

    onehot = np.zeros((4, 3), dtype=np.float32)
    onehot[np.arange(4), labels] = 1.0
    want = (V.softmax(V.tensor(logits), -1).numpy() - onehot) / 4.0

    assert np.allclose(v.grad.numpy(), want, atol=1e-6), f"{v.grad.numpy()}\nvs\n{want}"


def test_cross_entropy_rejects_float_target():
    v = V.tensor(make_input((2, 3), seed=97))
    with pytest.raises(V.DTypeError):
        V.cross_entropy(v, V.tensor(np.array([0.0, 1.0], dtype=np.float32)))


def test_cross_entropy_rejects_label_count_mismatch():
    v = V.tensor(make_input((4, 3), seed=98))
    with pytest.raises(V.ShapeError):
        V.cross_entropy(v, V.tensor(np.array([0, 1], dtype=np.int64)))


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

WINDOWS = [
    # (shape, kernel, stride, padding, dilation)
    ((1, 1, 4, 4), (2, 2), (1, 1), (0, 0), (1, 1)),
    ((2, 3, 5, 5), (3, 3), (1, 1), (1, 1), (1, 1)),   # padded, overlapping
    ((1, 2, 6, 6), (2, 2), (2, 2), (0, 0), (1, 1)),   # strided, no overlap
    ((2, 1, 7, 5), (3, 2), (2, 1), (1, 0), (1, 1)),   # asymmetric everything
    ((1, 2, 5, 5), (2, 2), (1, 1), (0, 0), (2, 2)),   # dilated
    ((1, 1, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),   # kernel == image
]


@pytest.mark.parametrize("shape,kernel,stride,pad,dil", WINDOWS)
def test_im2col(shape, kernel, stride, pad, dil):
    x = make_input(shape, seed=110)
    v, t = pair(x)
    assert_close("im2col", V.im2col(v, kernel, stride, pad, dil),
                 torch.nn.functional.unfold(t, kernel, dilation=dil, padding=pad, stride=stride),
                 inputs=[x])


@pytest.mark.parametrize("shape,kernel,stride,pad,dil", WINDOWS)
def test_col2im(shape, kernel, stride, pad, dil):
    """Overlapping windows are the case that matters: an image position then
    receives several contributions, which is what stops this being a gather."""
    x = make_input(shape, seed=111)
    t = torch.from_numpy(x.copy())
    cols_t = torch.nn.functional.unfold(t, kernel, dilation=dil, padding=pad, stride=stride)
    cols_v = V.im2col(V.tensor(x), kernel, stride, pad, dil)

    assert_close("col2im", V.col2im(cols_v, shape[2:], kernel, stride, pad, dil),
                 torch.nn.functional.fold(cols_t, shape[2:], kernel, dilation=dil,
                                          padding=pad, stride=stride),
                 TOLERANCES["reduction"], inputs=[x])


def test_col2im_counts_overlap():
    """Folding a column tensor of ones gives, at each position, the number of
    windows covering it. Independent of torch, and it fails if contributions
    are overwritten instead of summed -- which a gather-shaped implementation
    would do while still producing plausible numbers."""
    ones = V.full([1, 1 * 3 * 3, 4 * 4], 1.0)
    counts = V.col2im(ones, [4, 4], [3, 3], [1, 1], [1, 1]).numpy()[0, 0]

    # A 3x3 kernel with stride 1 and padding 1 over 4x4: corners are covered by
    # four windows, edges by six, the interior by nine.
    assert counts[0, 0] == 4.0, counts
    assert counts[0, 1] == 6.0, counts
    assert counts[1, 1] == 9.0, counts


@pytest.mark.parametrize("shape,kernel,stride,pad,dil", WINDOWS[:4])
def test_im2col_col2im_are_adjoint(shape, kernel, stride, pad, dil):
    """<im2col(u), v> == <u, col2im(v)>. The defining property of an adjoint
    pair, and what makes each the other's gradient rule."""
    rng = np.random.default_rng(112)
    u = rng.normal(size=shape).astype(np.float32)
    cols = V.im2col(V.tensor(u), kernel, stride, pad, dil)
    v = rng.normal(size=cols.numpy().shape).astype(np.float32)

    lhs = float((cols.numpy() * v).sum())
    rhs = float((u * V.col2im(V.tensor(v), shape[2:], kernel, stride, pad, dil).numpy()).sum())
    assert abs(lhs - rhs) <= 1e-3 * max(1.0, abs(lhs)), f"{lhs} vs {rhs}"


def test_im2col_gradients():
    x = make_input((2, 2, 5, 5), seed=113)
    v = V.tensor(x, requires_grad=True)
    t = torch.from_numpy(x.copy()).requires_grad_(True)

    V.sum(V.mul(V.im2col(v, [3, 3], [1, 1], [1, 1]),
                V.im2col(v, [3, 3], [1, 1], [1, 1]))).backward()
    (torch.nn.functional.unfold(t, [3, 3], padding=[1, 1]) ** 2).sum().backward()

    assert_close("im2col grad", v.grad, t.grad, TOLERANCES["reduction"], inputs=[x])


def test_im2col_rejects_oversized_kernel():
    v = V.tensor(make_input((1, 1, 3, 3), seed=114))
    with pytest.raises(V.ShapeError):
        V.im2col(v, [5, 5])


def test_col2im_rejects_inconsistent_geometry():
    cols = V.im2col(V.tensor(make_input((1, 1, 4, 4), seed=115)), [2, 2])
    with pytest.raises(V.ShapeError):
        V.col2im(cols, [7, 7], [2, 2])


CONVS = [
    # (input, weight, stride, padding, dilation)
    ((1, 1, 5, 5), (2, 1, 3, 3), (1, 1), (0, 0), (1, 1)),
    ((2, 3, 6, 6), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1)),   # padded, same-size out
    ((1, 2, 8, 8), (3, 2, 2, 2), (2, 2), (0, 0), (1, 1)),   # strided
    ((2, 1, 7, 9), (2, 1, 3, 2), (2, 1), (1, 0), (1, 1)),   # asymmetric
    ((1, 2, 7, 7), (2, 2, 3, 3), (1, 1), (2, 2), (2, 2)),   # dilated
    ((1, 3, 4, 4), (5, 3, 4, 4), (1, 1), (0, 0), (1, 1)),   # kernel == image
]


@pytest.mark.parametrize("xs,ws,stride,pad,dil", CONVS)
@pytest.mark.parametrize("use_bias", [False, True])
def test_conv2d(xs, ws, stride, pad, dil, use_bias):
    x = make_input(xs, seed=120)
    w = make_input(ws, seed=121)
    b = make_input((ws[0],), seed=122) if use_bias else None

    vx, tx = pair(x)
    vw, tw = pair(w)
    vb, tb = pair(b) if use_bias else (V.Tensor(), None)

    assert_close("conv2d", V.conv2d(vx, vw, vb, stride, pad, dil),
                 torch.nn.functional.conv2d(tx, tw, tb, stride=stride, padding=pad,
                                            dilation=dil),
                 TOLERANCES["reduction"], inputs=[x, w])


def test_conv2d_gradients():
    """No conv-specific backward rule exists: the gradient comes from im2col,
    matmul and reshape composing. Checked because that composition is the whole
    claim, and a wrong one still produces plausibly-shaped gradients."""
    x = make_input((2, 3, 6, 6), seed=123)
    w = make_input((4, 3, 3, 3), seed=124)
    b = make_input((4,), seed=125)

    vx = V.tensor(x, requires_grad=True)
    vw = V.tensor(w, requires_grad=True)
    vb = V.tensor(b, requires_grad=True)
    tx = torch.from_numpy(x.copy()).requires_grad_(True)
    tw = torch.from_numpy(w.copy()).requires_grad_(True)
    tb = torch.from_numpy(b.copy()).requires_grad_(True)

    V.sum(V.conv2d(vx, vw, vb, padding=(1, 1))).backward()
    torch.nn.functional.conv2d(tx, tw, tb, padding=(1, 1)).sum().backward()

    assert_close("conv2d grad input", vx.grad, tx.grad, TOLERANCES["reduction"], inputs=[x])
    assert_close("conv2d grad weight", vw.grad, tw.grad, TOLERANCES["reduction"], inputs=[w])
    assert_close("conv2d grad bias", vb.grad, tb.grad, TOLERANCES["reduction"], inputs=[b])


def test_conv2d_rejects_grouped_weight():
    """Grouped convolution is unsupported, and the failure must be explicit --
    a mismatched channel count is otherwise a plausible-looking reshape away
    from silently computing something else."""
    x = V.tensor(make_input((1, 4, 5, 5), seed=126))
    w = V.tensor(make_input((4, 2, 3, 3), seed=127))    # would be groups=2
    with pytest.raises(V.ShapeError):
        V.conv2d(x, w)


def test_conv2d_rejects_bad_bias():
    x = V.tensor(make_input((1, 2, 5, 5), seed=128))
    w = V.tensor(make_input((3, 2, 3, 3), seed=129))
    with pytest.raises(V.ShapeError):
        V.conv2d(x, w, V.tensor(make_input((5,), seed=130)))


POOLS = [
    # (shape, kernel, stride, padding)
    ((1, 1, 4, 4), (2, 2), (2, 2), (0, 0)),
    ((2, 3, 6, 6), (2, 2), (2, 2), (0, 0)),
    ((1, 2, 5, 5), (3, 3), (1, 1), (1, 1)),   # padded, overlapping
    ((2, 1, 7, 5), (3, 2), (2, 1), (1, 0)),   # asymmetric
    ((1, 2, 8, 8), (2, 2), (0, 0), (0, 0)),   # stride defaults to kernel
    ((1, 1, 3, 3), (3, 3), (1, 1), (0, 0)),   # kernel == image
]


@pytest.mark.parametrize("shape,kernel,stride,pad", POOLS)
def test_max_pool2d(shape, kernel, stride, pad):
    x = make_input(shape, seed=140)
    v, t = pair(x)
    tstride = kernel if stride == (0, 0) else stride
    assert_close("max_pool2d", V.max_pool2d(v, kernel, stride, pad),
                 torch.nn.functional.max_pool2d(t, kernel, stride=tstride, padding=pad),
                 inputs=[x])


@pytest.mark.parametrize("shape,kernel,stride,pad", POOLS)
def test_avg_pool2d(shape, kernel, stride, pad):
    x = make_input(shape, seed=141)
    v, t = pair(x)
    tstride = kernel if stride == (0, 0) else stride
    assert_close("avg_pool2d", V.avg_pool2d(v, kernel, stride, pad),
                 torch.nn.functional.avg_pool2d(t, kernel, stride=tstride, padding=pad),
                 TOLERANCES["reduction"], inputs=[x])


def test_max_pool2d_pads_with_negative_infinity():
    """The reason max pooling needs its own kernel rather than composing as a
    max over im2col: im2col pads with zero, so an all-negative window would
    report 0 as its maximum. Verified against torch, which does not."""
    x = np.array([[[[-1.0, -2.0], [-3.0, -4.0]]]], dtype=np.float32)
    v, t = pair(x)
    got = V.max_pool2d(v, [2, 2], [1, 1], [1, 1])
    assert_close("max_pool2d(-inf padding)", got,
                 torch.nn.functional.max_pool2d(t, 2, stride=1, padding=1), inputs=[x])
    assert not np.any(got.numpy() == 0.0), f"zero leaked from the padding: {got.numpy()}"


@pytest.mark.parametrize("shape,kernel,stride,pad", POOLS[:4])
def test_max_pool2d_gradients(shape, kernel, stride, pad):
    x = make_input(shape, seed=142)
    v = V.tensor(x, requires_grad=True)
    t = torch.from_numpy(x.copy()).requires_grad_(True)
    tstride = kernel if stride == (0, 0) else stride

    V.sum(V.mul(V.max_pool2d(v, kernel, stride, pad),
                V.max_pool2d(v, kernel, stride, pad))).backward()
    (torch.nn.functional.max_pool2d(t, kernel, stride=tstride, padding=pad) ** 2).sum().backward()

    assert_close("max_pool2d grad", v.grad, t.grad, TOLERANCES["reduction"], inputs=[x])


def test_max_pool2d_gradient_goes_to_one_position_on_a_tie():
    """torch routes a window's gradient to the FIRST maximum only, not split
    among ties. A random sweep never produces a tie, so this pins it directly --
    and a `>=` scan instead of `>` would silently pick the last."""
    x = np.array([[[[5.0, 5.0], [1.0, 2.0]]]], dtype=np.float32)
    v = V.tensor(x, requires_grad=True)
    t = torch.from_numpy(x.copy()).requires_grad_(True)

    V.sum(V.max_pool2d(v, [2, 2])).backward()
    torch.nn.functional.max_pool2d(t, 2).sum().backward()

    assert_close("tie gradient", v.grad, t.grad, inputs=[x])
    assert v.grad.numpy().tolist() == [[[[1.0, 0.0], [0.0, 0.0]]]], v.grad.numpy()


def test_avg_pool2d_counts_padding_in_the_divisor():
    """count_include_pad=True is torch's default, and it is what zero-padded
    im2col plus a mean gives for free -- a corner of a ones tensor averages to
    1/4, not 1."""
    ones = V.full([1, 1, 2, 2], 1.0)
    corner = V.avg_pool2d(ones, [2, 2], [1, 1], [1, 1]).numpy()[0, 0, 0, 0]
    assert corner == pytest.approx(0.25), corner


def test_max_pool2d_rejects_padding_beyond_half_the_kernel():
    v = V.tensor(make_input((1, 1, 5, 5), seed=143))
    with pytest.raises(V.ShapeError):
        V.max_pool2d(v, [2, 2], [1, 1], [2, 2])
