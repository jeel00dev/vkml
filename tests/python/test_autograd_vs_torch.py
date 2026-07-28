"""Gradient validation against PyTorch, plus finite-difference checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vkml as V
from conftest import TOLERANCES, Tol, assert_close, assert_shape, make_input, pair_grad

# Gradients accumulate one more rounding layer than the forward pass, so they
# get the reduction tolerance rather than the elementwise one.
GRAD_TOL = TOLERANCES["reduction"]


def _check_grad(name, vf, tf, x, tol=GRAD_TOL):
    """Forward and backward of a unary op, compared against torch."""
    v, t = pair_grad(x)

    out_v = vf(v)
    out_t = tf(t)
    assert_shape(f"{name} forward", out_v, out_t)
    assert_close(f"{name} forward", out_v, out_t, TOLERANCES["transcendental"], inputs=[x])

    V.sum(out_v).backward()
    out_t.sum().backward()

    assert v.grad.defined(), f"{name}: vkml produced no gradient"
    assert_shape(f"{name} grad", v.grad, t.grad)
    assert_close(f"{name} grad", v.grad, t.grad, tol, inputs=[x])


UNARY_GRAD = [
    ("neg", V.neg, torch.neg, "any"),
    ("abs", V.abs, torch.abs, "any"),
    ("square", V.square, torch.square, "any"),
    ("sqrt", V.sqrt, torch.sqrt, "positive"),
    ("rsqrt", V.rsqrt, torch.rsqrt, "positive"),
    ("reciprocal", V.reciprocal, torch.reciprocal, "nonzero"),
    ("exp", V.exp, torch.exp, "small"),
    ("log", V.log, torch.log, "positive"),
    ("sin", V.sin, torch.sin, "any"),
    ("cos", V.cos, torch.cos, "any"),
    ("tanh", V.tanh, torch.tanh, "any"),
    ("sigmoid", V.sigmoid, torch.sigmoid, "any"),
    ("relu", V.relu, torch.relu, "any"),
    ("gelu", V.gelu, torch.nn.functional.gelu, "any"),
    ("silu", V.silu, torch.nn.functional.silu, "any"),
]

DOMAINS = {
    "any": (-3.0, 3.0),
    "positive": (0.3, 4.0),
    "nonzero": (0.5, 3.0),
    "small": (-3.0, 3.0),
}


@pytest.mark.parametrize("name,vf,tf,domain", UNARY_GRAD, ids=[u[0] for u in UNARY_GRAD])
@pytest.mark.parametrize("shape", [(5,), (3, 4)], ids=["1d", "2d"])
def test_unary_gradients(name, vf, tf, domain, shape):
    lo, hi = DOMAINS[domain]
    # Nudged away from 0 so that abs/relu/sign are not evaluated exactly at their
    # kink, where the subgradient convention differs between frameworks and the
    # comparison would test the convention rather than the rule.
    x = make_input(shape, seed=100 + len(name), low=lo, high=hi)
    x = np.where(np.abs(x) < 0.05, 0.05, x).astype(np.float32)
    _check_grad(name, vf, tf, x)


@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("name,vf,tf", [("triu", V.triu, torch.triu), ("tril", V.tril, torch.tril)])
def test_tri_gradients(name, vf, tf, diagonal):
    """A triangular mask is linear and idempotent, so its own adjoint is
    itself: the masked-out region must receive exactly zero gradient. Getting
    the backward diagonal wrong leaks gradient into elements that contributed
    nothing, which this catches because torch does not."""
    x = make_input((5, 5), seed=140 + diagonal)
    _check_grad(f"{name}(d={diagonal})",
                lambda v: vf(v, diagonal), lambda t: tf(t, diagonal), x)


BINARY_GRAD = [
    ("add", V.add, torch.add, "any"),
    ("sub", V.sub, torch.sub, "any"),
    ("mul", V.mul, torch.mul, "any"),
    ("div", V.div, torch.div, "nonzero"),
    ("maximum", V.maximum, torch.maximum, "any"),
    ("minimum", V.minimum, torch.minimum, "any"),
]


@pytest.mark.parametrize("name,vf,tf,domain", BINARY_GRAD, ids=[b[0] for b in BINARY_GRAD])
def test_binary_gradients(name, vf, tf, domain):
    lo, hi = DOMAINS[domain]
    a = make_input((4, 3), seed=200, low=lo, high=hi)
    b = make_input((4, 3), seed=201, low=lo, high=hi)
    # Separate the operands so maximum/minimum are never evaluated at a tie.
    b = np.where(np.abs(a - b) < 0.05, b + 0.2, b).astype(np.float32)

    va, ta = pair_grad(a)
    vb, tb = pair_grad(b)

    V.sum(vf(va, vb)).backward()
    tf(ta, tb).sum().backward()

    assert_close(f"{name} grad wrt a", va.grad, ta.grad, GRAD_TOL, inputs=[a, b])
    assert_close(f"{name} grad wrt b", vb.grad, tb.grad, GRAD_TOL, inputs=[a, b])


BROADCAST_GRAD = [
    ((3, 4), (4,)),
    ((3, 1), (1, 4)),
    ((2, 3, 4), (4,)),
    ((2, 3, 4), (3, 1)),
    ((5,), ()),
]


@pytest.mark.parametrize("sa,sb", BROADCAST_GRAD, ids=[f"{a}x{b}" for a, b in BROADCAST_GRAD])
def test_broadcast_gradients_reduce_correctly(sa, sb):
    """The adjoint of a broadcast is a sum over the stretched axes.

    This is where a framework most often gets gradients subtly wrong: the
    gradient must be reduced back to each operand's original shape.
    """
    a = make_input(sa, seed=300)
    b = make_input(sb, seed=301)
    va, ta = pair_grad(a)
    vb, tb = pair_grad(b)

    V.sum(va * vb).backward()
    (ta * tb).sum().backward()

    assert_shape("broadcast grad a", va.grad, ta.grad)
    assert_shape("broadcast grad b", vb.grad, tb.grad)
    assert_close("broadcast grad a", va.grad, ta.grad, GRAD_TOL, inputs=[a, b])
    assert_close("broadcast grad b", vb.grad, tb.grad, GRAD_TOL, inputs=[a, b])


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_reduction_gradients(axis, keepdim):
    x = make_input((3, 4), seed=400)

    for name, vf, tf in (("sum", V.sum, torch.sum), ("mean", V.mean, torch.mean)):
        v, t = pair_grad(x)
        if axis is None:
            out_v = vf(v, None, keepdim)
            out_t = tf(t, dim=tuple(range(t.dim())), keepdim=keepdim)
        else:
            out_v = vf(v, axis, keepdim)
            out_t = tf(t, dim=axis, keepdim=keepdim)

        V.sum(out_v).backward()
        out_t.sum().backward()
        assert_shape(f"{name} grad", v.grad, t.grad)
        assert_close(f"{name}(dim={axis},keepdim={keepdim}) grad", v.grad, t.grad, GRAD_TOL,
                     inputs=[x])


@pytest.mark.parametrize("axis", [0, 1])
def test_max_min_gradients(axis):
    # Distinct values: at a tie torch splits the gradient while vkml routes it
    # to every tied element -- a documented divergence, not tested here.
    x = make_input((4, 5), seed=500)
    x = x + np.arange(x.size, dtype=np.float32).reshape(x.shape) * 1e-2

    for name, vf, tf in (("amax", V.amax, torch.amax), ("amin", V.amin, torch.amin)):
        v, t = pair_grad(x)
        V.sum(vf(v, axis)).backward()
        tf(t, dim=axis).sum().backward()
        assert_close(f"{name} grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


def test_matmul_gradients():
    a = make_input((3, 4), seed=600)
    b = make_input((4, 5), seed=601)
    va, ta = pair_grad(a)
    vb, tb = pair_grad(b)

    V.sum(V.matmul(va, vb)).backward()
    torch.matmul(ta, tb).sum().backward()

    assert_shape("matmul grad a", va.grad, ta.grad)
    assert_shape("matmul grad b", vb.grad, tb.grad)
    assert_close("matmul grad a", va.grad, ta.grad, TOLERANCES["matmul"], inputs=[a, b])
    assert_close("matmul grad b", vb.grad, tb.grad, TOLERANCES["matmul"], inputs=[a, b])


def test_batched_matmul_gradients():
    a = make_input((2, 3, 4), seed=610)
    b = make_input((2, 4, 5), seed=611)
    va, ta = pair_grad(a)
    vb, tb = pair_grad(b)

    V.sum(V.matmul(va, vb)).backward()
    torch.matmul(ta, tb).sum().backward()
    assert_close("bmm grad a", va.grad, ta.grad, TOLERANCES["matmul"], inputs=[a, b])
    assert_close("bmm grad b", vb.grad, tb.grad, TOLERANCES["matmul"], inputs=[a, b])


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax_gradients(axis):
    x = make_input((3, 4), seed=700, low=-3.0, high=3.0)

    for name, vf, tf in (
        ("softmax", V.softmax, torch.softmax),
        ("log_softmax", V.log_softmax, torch.log_softmax),
    ):
        v, t = pair_grad(x)
        # Weighted, not a plain sum: softmax's rows sum to 1, so an unweighted
        # sum has zero gradient and would pass no matter what the rule is.
        w = make_input((3, 4), seed=701)
        vw, tw = pair_grad(w)

        V.sum(vf(v, axis) * vw).backward()
        (tf(t, dim=axis) * tw).sum().backward()
        assert_close(f"{name}(dim={axis}) grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


def test_view_gradients():
    """Gradients must flow back through zero-copy views."""
    x = make_input((2, 3, 4), seed=800)

    cases = [
        ("transpose", lambda v: v.transpose(0, 1), lambda t: t.transpose(0, 1)),
        ("permute", lambda v: v.permute([2, 0, 1]), lambda t: t.permute(2, 0, 1)),
        ("reshape", lambda v: v.reshape([6, 4]), lambda t: t.reshape(6, 4)),
        ("unsqueeze", lambda v: v.unsqueeze(1), lambda t: t.unsqueeze(1)),
        ("contiguous", lambda v: v.transpose(0, 1).contiguous(),
         lambda t: t.transpose(0, 1).contiguous()),
    ]
    for name, vf, tf in cases:
        v, t = pair_grad(x)
        # Weighted so the gradient depends on the permutation being inverted
        # correctly; a plain sum would give an all-ones gradient regardless.
        w = make_input(tuple(vf(V.tensor(x)).shape), seed=801)
        vw, _ = pair_grad(w)
        tw = torch.from_numpy(w.copy())

        V.sum(vf(v) * vw).backward()
        (tf(t) * tw).sum().backward()
        assert_shape(f"{name} grad", v.grad, t.grad)
        assert_close(f"{name} grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


def test_chain_of_operations():
    x = make_input((4, 5), seed=900, low=0.5, high=2.0)
    v, t = pair_grad(x)

    out_v = V.sum(V.log(V.relu(V.exp(v * 0.5) + 1.0) * 2.0))
    out_t = torch.log(torch.relu(torch.exp(t * 0.5) + 1.0) * 2.0).sum()

    assert_close("chain forward", out_v, out_t, TOLERANCES["transcendental"], inputs=[x])
    out_v.backward()
    out_t.backward()
    assert_close("chain grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


def test_shared_subgraph_accumulates():
    """A node consumed twice must receive the SUM of both contributions.

    The classic autograd bug is to overwrite instead of accumulate, which halves
    the gradient here and is invisible in any single-use test.
    """
    x = make_input((4,), seed=1000, low=0.5, high=2.0)
    v, t = pair_grad(x)

    shared_v = v * 2.0
    out_v = V.sum(shared_v * shared_v + shared_v)
    shared_t = t * 2.0
    out_t = (shared_t * shared_t + shared_t).sum()

    out_v.backward()
    out_t.backward()
    assert_close("shared subgraph grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


def test_branching_graph():
    """One input, two independent branches recombined."""
    x = make_input((3, 4), seed=1100, low=0.5, high=2.0)
    v, t = pair_grad(x)

    out_v = V.sum(V.relu(v) * 3.0 + V.tanh(v) - V.exp(v * 0.1))
    out_t = (torch.relu(t) * 3.0 + torch.tanh(t) - torch.exp(t * 0.1)).sum()

    out_v.backward()
    out_t.backward()
    assert_close("branching grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


def test_gradient_accumulates_across_backward_calls():
    """Repeated backward adds to .grad, as in PyTorch."""
    x = make_input((4,), seed=1200, low=0.5, high=2.0)
    v, t = pair_grad(x)

    V.sum(v * v).backward()
    first = v.grad.numpy().copy()

    V.sum(v * v).backward()
    second = v.grad.numpy()

    np.testing.assert_allclose(second, first * 2, rtol=1e-6, atol=1e-6)

    t2 = torch.from_numpy(x.copy()).requires_grad_(True)
    (t2 * t2).sum().backward()
    (t2 * t2).sum().backward()
    assert_close("accumulated grad", v.grad, t2.grad, GRAD_TOL, inputs=[x])


def test_no_grad_blocks_tracking():
    x = make_input((4,), seed=1300)
    v = V.tensor(x, requires_grad=True)
    with V.no_grad():
        y = v * 2.0
    assert not y.requires_grad


def test_detach_stops_gradient():
    x = make_input((4,), seed=1400, low=0.5, high=2.0)
    v, t = pair_grad(x)

    V.sum(v * v.detach()).backward()
    (t * t.detach()).sum().backward()
    assert_close("detach grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


# ---------------------------------------------------------------------------
# Finite-difference gradient checking
# ---------------------------------------------------------------------------


def _numeric_grad(fn, x, h=1e-3):
    """Central-difference gradient of a scalar function, evaluated in float64.

    Central differences have O(h^2) truncation error against O(h) for a forward
    difference, and computing in float64 keeps the subtraction from being
    swamped by fp32 cancellation -- with h=1e-3 in fp32, the difference of two
    nearby values loses roughly half the significant digits.
    """
    x64 = x.astype(np.float64)
    grad = np.zeros_like(x64)
    it = np.nditer(x64, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x64[idx]

        x64[idx] = orig + h
        plus = fn(x64.astype(np.float32))
        x64[idx] = orig - h
        minus = fn(x64.astype(np.float32))
        x64[idx] = orig

        grad[idx] = (plus - minus) / (2 * h)
        it.iternext()
    return grad


GRADCHECK_FNS = [
    ("square", lambda v: V.sum(V.square(v)), (0.5, 2.0)),
    ("exp", lambda v: V.sum(V.exp(v)), (-1.0, 1.0)),
    ("log", lambda v: V.sum(V.log(v)), (0.5, 2.0)),
    ("tanh", lambda v: V.sum(V.tanh(v)), (-1.0, 1.0)),
    ("sigmoid", lambda v: V.sum(V.sigmoid(v)), (-1.0, 1.0)),
    ("gelu", lambda v: V.sum(V.gelu(v)), (-1.5, 1.5)),
    ("silu", lambda v: V.sum(V.silu(v)), (-1.5, 1.5)),
    ("sqrt", lambda v: V.sum(V.sqrt(v)), (0.5, 2.0)),
    ("sin", lambda v: V.sum(V.sin(v)), (-1.0, 1.0)),
    ("softmax", lambda v: V.sum(V.softmax(v, -1) * 3.0), (-1.0, 1.0)),
    ("log_softmax", lambda v: V.sum(V.log_softmax(v, -1) * 3.0), (-1.0, 1.0)),
    ("chain", lambda v: V.sum(V.tanh(V.exp(v * 0.5) + 1.0)), (-1.0, 1.0)),
]


@pytest.mark.parametrize("name,fn,domain", GRADCHECK_FNS, ids=[g[0] for g in GRADCHECK_FNS])
def test_finite_difference_gradcheck(name, fn, domain):
    lo, hi = domain
    x = make_input((3, 4), seed=1500, low=lo, high=hi)

    v = V.tensor(x, requires_grad=True)
    fn(v).backward()
    analytic = v.grad.numpy()

    numeric = _numeric_grad(lambda arr: float(fn(V.tensor(arr)).item()), x)

    # 2e-3 absolute: the central difference itself carries O(h^2) = 1e-6
    # truncation plus fp32 evaluation noise amplified by 1/(2h) = 500. This
    # bound is a property of the *check*, not of the gradient rule -- the
    # PyTorch comparisons above are what pin the rules to 1e-5.
    np.testing.assert_allclose(
        analytic, numeric, rtol=2e-2, atol=2e-3,
        err_msg=(f"\n{name}: analytic vs finite-difference gradient\n"
                 f"max abs err = {np.abs(analytic - numeric).max():.3g}"),
    )


# ---------------------------------------------------------------------------
# Slice backward
#
# The adjoint of a strided narrowing needs a real scatter kernel; see
# k_slice_backward. Exercised across axes, offsets and steps because the
# index remapping is easy to get subtly wrong.
# ---------------------------------------------------------------------------

SLICE_CASES = [
    ("[1:3]", lambda v: v[1:3], lambda t: t[1:3]),
    ("[:2]", lambda v: v[:2], lambda t: t[:2]),
    ("[2:]", lambda v: v[2:], lambda t: t[2:]),
    ("[::2]", lambda v: v[::2], lambda t: t[::2]),
    ("[1::2]", lambda v: v[1::2], lambda t: t[1::2]),
    ("[:, 1:4]", lambda v: v[:, 1:4], lambda t: t[:, 1:4]),
    ("[:, ::2]", lambda v: v[:, ::2], lambda t: t[:, ::2]),
    ("[1:4, 1:5]", lambda v: v[1:4, 1:5], lambda t: t[1:4, 1:5]),
    ("[2]", lambda v: v[2], lambda t: t[2]),
    ("[:, 3]", lambda v: v[:, 3], lambda t: t[:, 3]),
]


@pytest.mark.parametrize("name,vf,tf", SLICE_CASES, ids=[c[0] for c in SLICE_CASES])
def test_slice_gradients(name, vf, tf):
    x = make_input((5, 6), seed=1600)
    v, t = pair_grad(x)

    # Weighted, so the gradient is not uniformly 1 and a mis-mapped index shows.
    w = make_input(tuple(vf(V.tensor(x)).shape), seed=1601)
    vw = V.tensor(w)
    tw = torch.from_numpy(w.copy())

    V.sum(vf(v) * vw).backward()
    (tf(t) * tw).sum().backward()

    assert_shape(f"slice {name} grad", v.grad, t.grad)
    assert_close(f"slice {name} grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


def test_slice_gradient_zero_outside_selection():
    """Positions the slice did not select must receive exactly zero."""
    x = make_input((6,), seed=1700)
    v = V.tensor(x, requires_grad=True)
    V.sum(v[2:4] * 3.0).backward()
    g = v.grad.numpy()
    np.testing.assert_array_equal(g[[0, 1, 4, 5]], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(g[2:4], np.full(2, 3.0, dtype=np.float32), rtol=1e-6)


def test_slice_of_slice_gradient():
    x = make_input((8,), seed=1800)
    v, t = pair_grad(x)
    V.sum(v[1:7][::2]).backward()
    t[1:7][::2].sum().backward()
    assert_close("nested slice grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


# ---------------------------------------------------------------------------
# Backward rules the coverage audit found had never fired.
#
# `apply_backward` declares a rule per differentiable operator, and a green
# suite says nothing about which of them ran. Recording execution
# (scripts/coverage_matrix.py) showed five that never had. Four are below; the
# fifth, Cast, is unreachable from Python for a reason that is not a test gap --
# see test_cast_backward_is_unreachable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exponent", [2.0, 3.0, 0.5, -1.0])
def test_pow_gradient_wrt_base(exponent):
    """d/dx x^n = n*x^(n-1). Parametrised over exponents that exercise different
    branches of the identity: even, odd, fractional (needs x > 0) and negative."""
    x = make_input((5, 4), seed=1900, low=0.4, high=2.5)
    v, t = pair_grad(x)

    e_np = np.full(x.shape, exponent, dtype=np.float32)
    V.sum(V.pow(v, V.tensor(e_np))).backward()
    torch.pow(t, torch.from_numpy(e_np.copy())).sum().backward()

    assert_close(f"pow(x,{exponent}) grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


def test_pow_gradient_wrt_exponent_is_refused_not_wrong():
    """The other half of pow's rule is deliberately absent.

    d/dn x^n = x^n * ln(x), and ln(x) is undefined for x <= 0, so the rule
    declines rather than returning a number that is wrong on half its domain
    (autograd.cpp, OpKind::Pow). torch differentiates it and produces NaN there;
    vkml raises instead.

    Pinned as a test because a refusal is a contract: if someone implements the
    exponent gradient later, this failing is how they learn to write the
    comparison against torch rather than silently changing behaviour.
    """
    base = make_input((4, 3), seed=1901, low=0.5, high=2.0)
    expo = make_input((4, 3), seed=1902, low=0.5, high=2.0)

    vb = V.tensor(base, requires_grad=True)
    ve = V.tensor(expo, requires_grad=True)

    with pytest.raises(V.NotImplementedError_, match="exponent of pow"):
        V.sum(V.pow(vb, ve)).backward()


def test_clamp_gradient_is_zero_outside_the_bounds():
    """The whole content of clamp's rule is the mask.

    The input deliberately straddles both bounds: a test drawn entirely from
    inside the range would pass against an implementation that ignored the
    clamping completely.
    """
    x = np.linspace(-3.0, 3.0, 25, dtype=np.float32).reshape(5, 5)
    lo, hi = -1.0, 1.5
    v, t = pair_grad(x)

    V.sum(V.clamp(v, lo, hi)).backward()
    torch.clamp(t, lo, hi).sum().backward()

    assert_close("clamp grad", v.grad, t.grad, GRAD_TOL, inputs=[x])

    # Asserted directly as well, so the test still means something if torch ever
    # changes its boundary convention.
    g = v.grad.numpy()
    assert (g[x < lo] == 0.0).all(), "gradient leaked below the lower bound"
    assert (g[x > hi] == 0.0).all(), "gradient leaked above the upper bound"
    assert (g[(x > lo) & (x < hi)] == 1.0).all(), "gradient lost inside the bounds"


@pytest.mark.parametrize("index", [[0, 2, 2, 1, 0], [1, 1, 1, 1, 1]])
def test_scatter_add_gradient(index):
    """scatter_add's adjoint is index_select, so a source slice used twice must
    collect gradient twice. Repeated indices are the case that distinguishes a
    correct rule from a plausible one."""
    i = np.array(index, dtype=np.int64)
    x = make_input((len(i), 4), seed=1903)
    v, t = pair_grad(x)

    # Weighted, so a mis-mapped row shows up rather than cancelling into a
    # uniform gradient of 1.
    w = make_input((3, 4), seed=1904)

    V.sum(V.scatter_add(v, 0, V.tensor(i), 3) * V.tensor(w)).backward()
    (torch.zeros(3, 4).index_add_(0, torch.from_numpy(i.copy()), t)
     * torch.from_numpy(w.copy())).sum().backward()

    assert_close("scatter_add grad", v.grad, t.grad, GRAD_TOL, inputs=[x])


def test_col2im_gradient():
    """col2im's adjoint is im2col. Overlapping windows are the point: a column
    that lands on several image positions must gather gradient from all of them."""
    image, kernel, stride = (4, 4), (3, 3), (1, 1)
    x = make_input((1, 9, 4), seed=1905)          # (N, C*kh*kw, L)
    v, t = pair_grad(x)

    w = make_input((1, 1, 4, 4), seed=1906)

    V.sum(V.col2im(v, image, kernel, stride) * V.tensor(w)).backward()
    (torch.nn.functional.fold(t, image, kernel, stride=stride)
     * torch.from_numpy(w.copy())).sum().backward()

    assert_close("col2im grad", v.grad, t.grad, GRAD_TOL, inputs=[x])

# Cast's backward rule is now reachable and is tested in test_f16.py, where the
# f16 arithmetic that unblocked it lives.
