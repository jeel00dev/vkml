"""Vulkan kernel validation against the CPU oracle.

Adding a kernel here should be one entry in a list. If it needs more than that,
the framework in vkvalidate.py is missing something.
"""

from __future__ import annotations

import numpy as np
import pytest

import vkml as V
from vkvalidate import (
    Context,
    Layout,
    compare,
    gpu_device,
    make_data,
    random_layouts,
    requires_vulkan,
    run_binary,
    run_binary_broadcast,
    run_unary,
    vulkan_ready,
)

pytestmark = requires_vulkan

# One seed for the session, printed on failure so any case is reproducible.
SEED = 20260726


@pytest.fixture(scope="module")
def layouts():
    rng = np.random.default_rng(SEED)
    return random_layouts(rng, count=40)


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 1.0, -2.5, 3.25e10, -1e-8])
@pytest.mark.parametrize("n", [0, 1, 255, 256, 257, 1023, 4096, 100_000])
def test_fill(n, value):
    """Sizes straddle the 256-invocation workgroup boundary deliberately: an
    off-by-one in the bounds check shows up at 255/256/257 and nowhere else."""
    gpu = V.full([n], value, device=gpu_device()).numpy()
    cpu = V.full([n], value, device=V.cpu).numpy()

    ctx = Context(op="fill", layout=Layout((n,)), dtype="f32", seed=SEED,
                  spec_constants={"workgroup": 256})
    compare(ctx, gpu, cpu)


# ---------------------------------------------------------------------------
# Copy / identity / contiguous
# ---------------------------------------------------------------------------


def test_copy_over_random_layouts(layouts):
    """`contiguous()` on a strided view is the copy kernel's real workload."""
    n = run_unary("copy", lambda t: t.contiguous(), layouts, SEED)
    assert n == len(layouts)


def test_identity_roundtrip(layouts):
    """Upload then download must be bit-exact for every layout."""
    rng = np.random.default_rng(SEED + 1)
    for layout in layouts:
        data = make_data(rng, layout.base_shape, "any")
        gpu = V.tensor(data, device=gpu_device())
        back = layout.apply(gpu).contiguous().numpy()
        want = np.ascontiguousarray(layout.apply_numpy(data))

        ctx = Context(op="identity", layout=layout, dtype="f32", seed=SEED + 1,
                      inputs=[data])
        compare(ctx, back, want)


# ---------------------------------------------------------------------------
# Unary elementwise
# ---------------------------------------------------------------------------

UNARY = [
    ("relu", V.relu, "any"),
    ("neg", V.neg, "any"),
    ("abs", V.abs, "any"),
    ("exp", V.exp, "bounded"),
    ("sign", V.sign, "any"),
    ("square", V.square, "any"),
    ("sqrt", V.sqrt, "positive"),
    ("rsqrt", V.rsqrt, "positive"),
    ("reciprocal", V.reciprocal, "nonzero"),
    ("log", V.log, "positive"),
    ("erf", V.erf, "any"),
    ("sin", V.sin, "any"),
    ("cos", V.cos, "any"),
    ("tanh", V.tanh, "any"),
    ("sigmoid", V.sigmoid, "any"),
    ("gelu", V.gelu, "any"),
    ("silu", V.silu, "any"),
]


@pytest.mark.parametrize("name,fn,domain", UNARY, ids=[u[0] for u in UNARY])
def test_unary(name, fn, domain, layouts):
    n = run_unary(name, fn, layouts, SEED, domain)
    assert n == len(layouts)


CLAMP = [
    ("both", lambda t: V.clamp(t, -1.0, 1.0)),
    ("min-only", lambda t: V.clamp_min(t, 0.0)),
    ("max-only", lambda t: V.clamp_max(t, 0.5)),
]


@pytest.mark.parametrize("name,fn", CLAMP, ids=[c[0] for c in CLAMP])
def test_clamp(name, fn, layouts):
    """Clamp carries its bounds in push constants rather than in a
    specialisation constant, and a one-sided clamp sends the absent bound as an
    infinity -- so the min-only and max-only cases are what exercise that path.
    """
    n = run_unary("clamp", fn, layouts, SEED, "any")
    assert n == len(layouts)


def test_clamp_rejects_inverted_bounds():
    """The shader orders its two comparisons low-then-high, which would resolve
    an inverted range to the upper bound. That case is unreachable because the
    API rejects it first; this pins that guarantee, since the shader's comment
    depends on it."""
    t = V.tensor(np.zeros((4,), dtype=np.float32), device=gpu_device())
    with pytest.raises(V.ShapeError):
        V.clamp(t, 2.0, -2.0)


# ---------------------------------------------------------------------------
# Binary elementwise
# ---------------------------------------------------------------------------

# (name, fn, domain_a, domain_b). The right-hand domain differs where the
# operation is undefined on part of the real line: div by zero, and pow of a
# negative base.
BINARY = [
    ("add", V.add, "any", "any"),
    ("sub", V.sub, "any", "any"),
    ("mul", V.mul, "any", "any"),
    ("div", V.div, "any", "nonzero"),
    ("pow", V.pow, "positive", "unit"),
    ("maximum", V.maximum, "any", "any"),
    ("minimum", V.minimum, "any", "any"),
]

# The first element is the tolerance-policy key, which is the short form; the
# public API spells these out. Kept as a pair rather than derived, because the
# two vocabularies are independent and a mapping would hide that.
COMPARE = [
    ("eq", V.equal),
    ("lt", V.less),
    ("gt", V.greater),
    ("le", V.less_equal),
    ("ge", V.greater_equal),
    ("ne", V.not_equal),
]


@pytest.mark.parametrize("name,fn,da,db", BINARY, ids=[b[0] for b in BINARY])
def test_binary(name, fn, da, db, layouts):
    n = run_binary(name, fn, layouts, SEED, da, db)
    assert n == len(layouts)


@pytest.mark.parametrize("name,fn,da,db", BINARY, ids=[b[0] for b in BINARY])
def test_binary_broadcast(name, fn, da, db):
    n = run_binary_broadcast(name, fn, SEED, da, db)
    assert n == 10


@pytest.mark.parametrize("name,fn", COMPARE, ids=[c[0] for c in COMPARE])
def test_comparison(name, fn, layouts):
    """Comparisons narrow the output to Bool, so the destination is a byte
    buffer while the inputs stay float -- the one place in the binary shader
    where the operand element sizes differ."""
    n = run_binary(name, fn, layouts, SEED)
    assert n == len(layouts)


@pytest.mark.parametrize("name,fn", COMPARE, ids=[c[0] for c in COMPARE])
def test_comparison_broadcast(name, fn):
    n = run_binary_broadcast(name, fn, SEED)
    assert n == 10


def test_comparison_output_is_bool_not_float():
    """Guards the narrowing itself. If the shader wrote through F32Buf the
    values would still compare equal after numpy converts them, so this checks
    the dtype rather than the contents."""
    a = V.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), device=gpu_device())
    b = V.tensor(np.array([2.0, 2.0, 2.0], dtype=np.float32), device=gpu_device())
    out = V.less(a, b).numpy()
    assert out.dtype == np.bool_, f"expected bool output, got {out.dtype}"
    assert out.tolist() == [True, False, False]


def test_equality_of_equal_values_holds_on_gpu():
    """eq is EXACT, and the interesting case is operands that are equal. A
    random sweep almost never produces two identical floats, so this feeds the
    same buffer to both sides."""
    x = np.linspace(-3.0, 3.0, 257, dtype=np.float32)
    t = V.tensor(x, device=gpu_device())
    same = V.equal(t, t).numpy()
    assert same.all(), "x == x must hold for every element"


def test_maximum_propagates_nan_like_torch():
    """torch.maximum returns NaN if either operand is NaN; GLSL's max() is
    undefined for NaN and std::fmax would return the other operand. The policy
    for maximum is EXACT, so a divergence here is a hard failure, and the
    random sweep never generates NaN."""
    a = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
    b = np.array([2.0, 2.0, np.nan, np.nan], dtype=np.float32)

    for name, fn in (("maximum", V.maximum), ("minimum", V.minimum)):
        gpu = fn(V.tensor(a, device=gpu_device()), V.tensor(b, device=gpu_device())).numpy()
        cpu = fn(V.tensor(a, device=V.cpu), V.tensor(b, device=V.cpu)).numpy()
        ctx = Context(op=name, layout=Layout(a.shape), dtype="f32", seed=SEED, inputs=[a, b])
        compare(ctx, gpu, cpu)
        assert np.isnan(gpu[1:]).all(), f"{name} must propagate NaN"


def test_pow_matches_std_pow_for_negative_bases():
    """GLSL's pow(x, y) is undefined for x < 0, but std::pow is defined there
    whenever y is an integer -- std::pow(-2, 3) is -8. The shader peels the
    sign off and reapplies it so the two backends agree on a case the CPU
    answers perfectly well."""
    base = np.array([-2.0, -2.0, -2.0, -3.0, -1.0], dtype=np.float32)
    expo = np.array([2.0, 3.0, 4.0, 3.0, 5.0], dtype=np.float32)

    gpu = V.pow(V.tensor(base, device=gpu_device()), V.tensor(expo, device=gpu_device())).numpy()
    cpu = V.pow(V.tensor(base, device=V.cpu), V.tensor(expo, device=V.cpu)).numpy()

    ctx = Context(op="pow", layout=Layout(base.shape), dtype="f32", seed=SEED,
                  inputs=[base, expo])
    compare(ctx, gpu, cpu)

    # Pin the signs explicitly. An implementation returning |x|^y would satisfy
    # a magnitude check but is wrong for every odd exponent, and that is the
    # bug this test exists to catch. Signs only -- pow carries a 16 ULP
    # allowance (Vulkan spec), so (-3)^3 lands on -27.000002, not -27.
    assert np.sign(gpu).tolist() == [1.0, -1.0, 1.0, -1.0, -1.0]
    assert np.allclose(np.abs(gpu), [4.0, 8.0, 16.0, 27.0, 1.0], rtol=1e-5)


def test_pow_returns_nan_for_negative_base_with_fractional_exponent():
    """The other half of the pow contract: (-2)^0.5 is not real, and std::pow
    returns NaN. GLSL would leave it undefined."""
    base = np.array([-2.0, -4.0], dtype=np.float32)
    expo = np.array([0.5, 1.5], dtype=np.float32)

    gpu = V.pow(V.tensor(base, device=gpu_device()), V.tensor(expo, device=gpu_device())).numpy()
    cpu = V.pow(V.tensor(base, device=V.cpu), V.tensor(expo, device=V.cpu)).numpy()

    assert np.isnan(gpu).all(), f"expected NaN, got {gpu}"
    assert np.isnan(cpu).all(), f"CPU oracle changed behaviour: {cpu}"


# ---------------------------------------------------------------------------
# Triangular masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("diagonal", [0, 1, -1, 3, -3])
@pytest.mark.parametrize("name,fn", [("triu", V.triu), ("tril", V.tril)])
def test_tri_over_random_layouts(name, fn, diagonal, layouts):
    """The predicate is positional, so this is the one kernel where a strided
    or permuted input could disagree with the CPU by masking the wrong element
    rather than by reading the wrong one."""
    rank2 = [lay for lay in layouts if len(lay.base_shape) >= 2]
    assert rank2, "expected the layout sweep to contain rank-2+ cases"

    n = run_unary(name, lambda t: fn(t, diagonal), rank2, SEED, "any")
    assert n == len(rank2)


def test_tri_boundary_is_exact_on_gpu():
    """Pins the diagonal boundary itself. A >= written as > shifts the kept
    region by one column and still passes a shape check."""
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    t = V.tensor(x, device=gpu_device())

    assert V.triu(t, 0).numpy().tolist() == np.triu(x, 0).tolist()
    assert V.triu(t, 1).numpy().tolist() == np.triu(x, 1).tolist()
    assert V.tril(t, 0).numpy().tolist() == np.tril(x, 0).tolist()
    assert V.tril(t, -1).numpy().tolist() == np.tril(x, -1).tolist()


# ---------------------------------------------------------------------------
# Where (ternary select)
# ---------------------------------------------------------------------------


def test_where_over_random_layouts(layouts):
    """Four operands, three of them independently strided, and the condition is
    a byte buffer while the values are floats -- so this exercises the mixed
    element-size path from the opposite side to the comparisons."""
    rng = np.random.default_rng(SEED)
    checked = 0

    for layout in layouts:
        a = make_data(rng, layout.base_shape, "any")
        b = make_data(rng, layout.base_shape, "any")
        # Build the condition through a comparison so it is a genuine Bool
        # tensor produced by the library rather than a host-side construction.
        thresh = make_data(rng, layout.base_shape, "any")

        def run(dev):
            ta = layout.apply(V.tensor(a, device=dev))
            tb = layout.apply(V.tensor(b, device=dev))
            tc = V.greater(ta, layout.apply(V.tensor(thresh, device=dev)))
            return V.where(tc, ta, tb).numpy()

        ctx = Context(op="where", layout=layout, dtype="f32", seed=SEED,
                      inputs=[layout.apply_numpy(a), layout.apply_numpy(b)])
        compare(ctx, run(gpu_device()), run(V.cpu))
        checked += 1

    assert checked == len(layouts)


def test_where_selects_rather_than_blends():
    """The unselected operand may hold NaN or an infinity. A blend written as
    c*a + (1-c)*b would propagate it; a select must not."""
    #        selected   unselected      -> the unselected value must not leak
    #  0:  a=1.0        b=NaN
    #  1:  b=2.0        a=NaN
    #  2:  a=+inf       b=3.0           (selecting an infinity is still fine)
    #  3:  b=-inf       a=4.0
    cond = np.array([True, False, True, False])
    a = np.array([1.0, np.nan, np.inf, 4.0], dtype=np.float32)
    b = np.array([np.nan, 2.0, 3.0, -np.inf], dtype=np.float32)

    def run(dev):
        return V.where(V.tensor(cond, device=dev), V.tensor(a, device=dev),
                       V.tensor(b, device=dev)).numpy()

    gpu = run(gpu_device())
    ctx = Context(op="where", layout=Layout(a.shape), dtype="f32", seed=SEED, inputs=[a, b])
    compare(ctx, gpu, run(V.cpu))

    assert gpu.tolist() == [1.0, 2.0, float("inf"), float("-inf")]
    # The load-bearing part: a NaN sat opposite each of the first two results
    # and neither reached the output.
    assert not np.isnan(gpu[:2]).any()


def test_gelu_negative_tail_is_relatively_accurate():
    """The reason gelu is computed through erfc rather than erf.

    At x = -3, gelu is -4.1e-3, but `1 + erf(x/sqrt2)` forms it by cancelling
    1.0 against -0.9973 and loses most of the significand. Asserted directly
    because the layout sweep draws from a uniform distribution over [-3, 3] and
    would rarely land deep enough in the tail to notice.
    """
    x = np.linspace(-6.0, -3.0, 512, dtype=np.float32)

    gpu = V.gelu(V.tensor(x, device=gpu_device())).numpy()
    cpu = V.gelu(V.tensor(x, device=V.cpu)).numpy()

    ctx = Context(op="gelu", layout=Layout(x.shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, gpu, cpu)

    # Guards against passing vacuously: if the domain were ever widened, these
    # would stop being the small cancelled values the test exists to cover.
    assert np.abs(cpu).max() < 5e-3
    assert np.abs(cpu).max() > 0.0


def test_exp_ulp_bound_is_the_binding_constraint():
    """exp is where the ULP policy actually bites.

    Vulkan permits 3 + 2*|x| ULP for exp; glibc expf() targets under 1. An
    absolute or naive relative check on values of magnitude ~50 reports a
    failure that is really the spec being honoured. This test exists to pin
    that the ULP policy -- not an ad-hoc number -- is what governs.
    """
    x = np.linspace(-3.0, 3.0, 4096, dtype=np.float32)
    gpu = V.exp(V.tensor(x, device=gpu_device())).numpy()
    cpu = V.exp(V.tensor(x, device=V.cpu)).numpy()

    from tolerance import ulp_distance

    ulps = ulp_distance(gpu, cpu)
    assert int(ulps.max()) <= 8, f"exp exceeded its ULP budget: {int(ulps.max())}"

    ctx = Context(op="exp", layout=Layout(x.shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, gpu, cpu)


# ---------------------------------------------------------------------------
# Cast
# ---------------------------------------------------------------------------

CAST_PAIRS = [
    (V.float32, V.float16),
    (V.float32, V.int32),
    (V.float32, V.int64),
    (V.float32, V.bool_),
    (V.float16, V.float32),
    (V.int32, V.float32),
    (V.int64, V.float32),
    (V.bool_, V.float32),
]


@pytest.mark.parametrize("src,dst", CAST_PAIRS,
                         ids=[f"{a}->{b}" for a, b in CAST_PAIRS])
def test_cast(src, dst):
    """IEEE-754 conversions are correctly rounded, so these must be exact."""
    rng = np.random.default_rng(SEED + 2)
    base = rng.uniform(-100.0, 100.0, size=(257,)).astype(np.float32)

    cpu_src = V.tensor(base, device=V.cpu).to(src)
    gpu_src = V.tensor(base, device=gpu_device()).to(src)

    cpu_out = cpu_src.to(dst).numpy()
    gpu_out = gpu_src.to(dst).numpy()

    ctx = Context(op="cast", layout=Layout(base.shape), dtype=f"{src}->{dst}", seed=SEED + 2,
                  inputs=[base], spec_constants={"src": str(src), "dst": str(dst)})
    compare(ctx, gpu_out, cpu_out)


def test_f16_roundtrip_preserves_subnormals():
    """The GPU's native float16_t conversion must handle subnormals, as the
    CPU's hand-written conversion does."""
    values = np.array([5.96e-8, 3.0e-8, 1e-7, 6.1e-5, 65504.0, -65504.0, 0.0, -0.0],
                      dtype=np.float32)
    gpu = V.tensor(values, device=gpu_device()).to(V.float16).to(V.float32).numpy()
    cpu = V.tensor(values, device=V.cpu).to(V.float16).to(V.float32).numpy()

    ctx = Context(op="cast", layout=Layout(values.shape), dtype="f32->f16->f32", seed=SEED,
                  inputs=[values])
    compare(ctx, gpu, cpu)


# ---------------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------------


def test_allocator_reuses_memory_and_does_not_leak():
    """Churn must not grow reserved memory or leave allocations behind."""
    before = V.vulkan_stats(0)

    def churn():
        # Inside a function so the loop variable goes out of scope before the
        # measurement. Reading the counters with a tensor still referenced would
        # report a leak that is really just a live local.
        for _ in range(200):
            t = V.full([4096], 1.0, device=gpu_device())
            _ = V.relu(t).numpy()

    churn()
    after = V.vulkan_stats(0)

    assert after["live_allocations"] == before["live_allocations"], (
        f"leaked {after['live_allocations'] - before['live_allocations']} allocation(s)"
    )
    # The point of suballocation: hundreds of tensors, a handful of device
    # allocations.
    new_device_allocs = after["device_allocations"] - before["device_allocations"]
    assert new_device_allocs <= 2, (
        f"{new_device_allocs} vkAllocateMemory calls for 200 tensors -- suballocation "
        f"is not working"
    )


def test_large_transfer_is_chunked_correctly():
    """Larger than the 32 MiB staging buffer, so the chunking path runs."""
    n = 12 * 1024 * 1024  # 48 MiB of float32
    data = np.arange(n, dtype=np.float32)
    back = V.tensor(data, device=gpu_device()).numpy()
    assert np.array_equal(data, back), "chunked transfer corrupted data"


def test_capabilities_match_measured_hardware():
    """Guards against a driver or query regression silently changing the facts
    the kernels are designed around."""
    caps = V.vulkan_capabilities(0)
    # These three drive real design decisions and must not change unnoticed.
    assert caps["global_float_atomics"] is False, (
        "global float atomicAdd appeared; the deterministic two-pass reduction "
        "design assumes it is absent"
    )
    assert caps["cooperative_matrix"] is False
    assert caps["max_shared_memory_bytes"] >= 32 * 1024
    assert caps["min_subgroup_size"] <= caps["subgroup_size"] <= caps["max_subgroup_size"]


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

from vkvalidate import ReduceCase, reduction_cases, run_reduction  # noqa: E402


@pytest.fixture(scope="module")
def reductions():
    rng = np.random.default_rng(SEED + 10)
    return reduction_cases(rng, count=45)


REDUCE_OPS = [
    ("sum", lambda t, ax, kd: V.sum(t, ax, kd), "any"),
    ("mean", lambda t, ax, kd: V.mean(t, ax, kd), "any"),
    ("max", lambda t, ax, kd: V.amax(t, ax, kd), "any"),
    ("min", lambda t, ax, kd: V.amin(t, ax, kd), "any"),
]


@pytest.mark.parametrize("name,fn,domain", REDUCE_OPS, ids=[r[0] for r in REDUCE_OPS])
def test_reduction(name, fn, domain, reductions):
    n = run_reduction(name, fn, None, reductions, SEED + 11, domain)
    assert n == len(reductions)


ARG_CASES = [
    ((7,), 0), ((3, 4), 0), ((3, 4), 1), ((3, 4), -1),
    ((2, 3, 4), 0), ((2, 3, 4), 2), ((257,), 0), ((1024,), 0),
]


@pytest.mark.parametrize("shape,axis", ARG_CASES, ids=[f"{s}ax{a}" for s, a in ARG_CASES])
def test_argmax_argmin(shape, axis):
    """Values are made distinct: at a tie the tie-break rule is what would be
    under test, not the reduction."""
    rng = np.random.default_rng(SEED + 12)
    data = rng.permutation(int(np.prod(shape))).astype(np.float32).reshape(shape)

    for op, fn in (("argmax", V.argmax), ("argmin", V.argmin)):
        cpu = fn(V.tensor(data, device=V.cpu), axis).numpy()
        gpu = fn(V.tensor(data, device=gpu_device()), axis).numpy()
        ctx = Context(op=op, layout=Layout(shape), dtype="i64", seed=SEED + 12, inputs=[data],
                      extra=f"axis={axis}")
        compare(ctx, gpu, cpu)


def test_reduction_is_deterministic():
    """Same input, repeated: bit-identical.

    This is the property the whole no-atomics, fixed-tree design exists to
    provide. An atomic-based reduction would pass a tolerance check and fail
    this one.
    """
    rng = np.random.default_rng(SEED + 13)
    data = rng.uniform(-5, 5, size=(64, 1024)).astype(np.float32)
    t = V.tensor(data, device=gpu_device())

    first = V.sum(t, [1]).numpy()
    for _ in range(8):
        assert np.array_equal(V.sum(t, [1]).numpy(), first), "reduction is not deterministic"


def test_long_reduction_stays_within_pairwise_bound():
    """65536 terms: a flat sequential accumulator would drift ~780x over budget.

    The per-lane carry stack in reduce.comp is what keeps this inside the bound;
    the framework's BACKWARD policy computes the bound from the actual data.
    """
    rng = np.random.default_rng(SEED + 14)
    data = rng.uniform(0.5, 1.5, size=(65536,)).astype(np.float32)

    cpu = V.sum(V.tensor(data, device=V.cpu)).numpy()
    gpu = V.sum(V.tensor(data, device=gpu_device())).numpy()

    ctx = Context(op="sum", layout=Layout(data.shape), dtype="f32", seed=SEED + 14,
                  inputs=[data])
    compare(ctx, gpu, cpu, terms_abs_sum=float(np.abs(data.astype(np.float64)).sum()),
            reduction_n=data.size)


def test_reduction_with_heavy_cancellation():
    """Alternating signs, so the result is tiny against sum|terms|.

    Relative-to-result would be meaningless here; the backward-error policy is
    what makes the check well posed.
    """
    n = 8192
    data = np.empty(n, dtype=np.float32)
    data[0::2] = 1000.0
    data[1::2] = -1000.0
    data[0] = 1000.5  # leave a small non-zero residue

    cpu = V.sum(V.tensor(data, device=V.cpu)).numpy()
    gpu = V.sum(V.tensor(data, device=gpu_device())).numpy()

    ctx = Context(op="sum", layout=Layout(data.shape), dtype="f32", seed=SEED,
                  inputs=[data])
    compare(ctx, gpu, cpu, terms_abs_sum=float(np.abs(data.astype(np.float64)).sum()),
            reduction_n=n)


# ---------------------------------------------------------------------------
# Softmax / LogSoftmax
# ---------------------------------------------------------------------------

SOFTMAX_CASES = [
    ((5,), 0), ((3, 4), 1), ((3, 4), 0), ((3, 4), -1),
    ((2, 3, 4), -1), ((2, 3, 4), 1), ((2, 3, 4), 0),
    ((1, 257), 1), ((4, 256), 1), ((4, 513), 1), ((2, 4096), 1),
    ((1, 1), 1), ((7, 1), 1),
]


@pytest.mark.parametrize("shape,axis", SOFTMAX_CASES,
                         ids=[f"{s}ax{a}" for s, a in SOFTMAX_CASES])
@pytest.mark.parametrize("op", ["softmax", "log_softmax"])
def test_softmax_random(shape, axis, op):
    rng = np.random.default_rng(SEED + 20)
    data = rng.uniform(-5.0, 5.0, size=shape).astype(np.float32)
    fn = V.softmax if op == "softmax" else V.log_softmax

    cpu = fn(V.tensor(data, device=V.cpu), axis).numpy()
    gpu = fn(V.tensor(data, device=gpu_device()), axis).numpy()

    ctx = Context(op=op, layout=Layout(shape), dtype="f32", seed=SEED + 20,
                  inputs=[data], extra=f"axis={axis}")
    compare(ctx, gpu, cpu)


STABILITY_INPUTS = {
    "large positive": np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32),
    "large negative": np.array([[-1000.0, -1001.0, -1002.0]], dtype=np.float32),
    "mixed sign": np.array([[-500.0, 0.0, 500.0]], dtype=np.float32),
    "identical": np.full((1, 8), 3.5, dtype=np.float32),
    "identical zeros": np.zeros((1, 16), dtype=np.float32),
    "extreme spread": np.array([[0.0, -300.0, -600.0, -900.0]], dtype=np.float32),
    "single element": np.array([[42.0]], dtype=np.float32),
    "tiny magnitudes": np.array([[1e-30, 2e-30, 3e-30]], dtype=np.float32),
}


@pytest.mark.parametrize("name", list(STABILITY_INPUTS), ids=list(STABILITY_INPUTS))
@pytest.mark.parametrize("op", ["softmax", "log_softmax"])
def test_softmax_stability(name, op):
    """The regimes where the max-subtraction trick earns its keep.

    Without it, `large positive` overflows exp() to inf and yields NaN; without
    the (x - max) - log(sum) form, `extreme spread` yields -inf from log(0).
    """
    data = STABILITY_INPUTS[name]
    fn = V.softmax if op == "softmax" else V.log_softmax

    cpu = fn(V.tensor(data, device=V.cpu), -1).numpy()
    gpu = fn(V.tensor(data, device=gpu_device()), -1).numpy()

    assert np.all(np.isfinite(gpu)), f"{op}({name}) produced non-finite values: {gpu}"

    ctx = Context(op=op, layout=Layout(data.shape), dtype="f32", seed=SEED,
                  inputs=[data], extra=f"stability case: {name}")
    compare(ctx, gpu, cpu)

    if op == "softmax":
        rows = gpu.sum(axis=-1)
        np.testing.assert_allclose(rows, np.ones_like(rows), rtol=1e-5,
                                   err_msg=f"softmax rows do not sum to 1 for {name}")


def test_softmax_shift_invariance():
    """softmax(x + c) == softmax(x). Catches a missing or wrong max subtraction
    even when nothing overflows.

    THE SHIFT MUST BE EXACTLY REPRESENTABLE. An earlier version of this test
    used `base + 700.0` and failed at 3.6e-5. That was not the kernel: ulp(702)
    in fp32 is 6.1e-5, so adding 700 quantizes the input by up to 3.0e-5 and the
    two calls are simply given different numbers. Quantizing the base to
    multiples of 2^-10 and shifting by 512 makes the addition exact (verified
    residual 0.0), so any remaining difference really is the kernel's.
    """
    rng = np.random.default_rng(SEED + 21)
    base = rng.uniform(-2, 2, size=(4, 64)).astype(np.float32)
    base = (np.round(base * 1024) / 1024).astype(np.float32)
    shifted = (base + np.float32(512.0)).astype(np.float32)

    # Guard the premise: if this ever stops holding, the test below is
    # measuring input quantization rather than shift invariance.
    assert np.array_equal((shifted - np.float32(512.0)).astype(np.float32), base)

    a = V.softmax(V.tensor(base, device=gpu_device()), -1).numpy()
    b = V.softmax(V.tensor(shifted, device=gpu_device()), -1).numpy()
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_softmax_against_pytorch():
    """The CPU backend is the oracle, but softmax is worth pinning to PyTorch
    directly too -- it is the op most likely to differ in stabilisation
    strategy rather than in arithmetic."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(SEED + 22)
    data = rng.uniform(-8.0, 8.0, size=(6, 129)).astype(np.float32)
    t = torch.from_numpy(data.copy())

    for op, vf, tf in (("softmax", V.softmax, torch.softmax),
                       ("log_softmax", V.log_softmax, torch.log_softmax)):
        gpu = vf(V.tensor(data, device=gpu_device()), -1).numpy()
        want = tf(t, dim=-1).numpy()
        np.testing.assert_allclose(
            gpu, want, rtol=1e-5, atol=1e-6,
            err_msg=f"vulkan {op} disagrees with PyTorch",
        )


# ---------------------------------------------------------------------------
# GEMM (Stage 1: naive reference kernel)
# ---------------------------------------------------------------------------

from vkvalidate import GemmCase, gemm_cases, run_gemm  # noqa: E402


@pytest.fixture(scope="module")
def gemms():
    rng = np.random.default_rng(SEED + 30)
    return gemm_cases(rng, count=60)


def test_gemm_against_cpu_and_pytorch(gemms):
    """Every case is checked against BOTH oracles under the BACKWARD policy."""
    n = run_gemm(gemms, SEED + 31)
    assert n == len(gemms)


@pytest.mark.parametrize("size", [1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255,
                                  256, 257])
def test_gemm_edge_sizes(size):
    """Square GEMM at every awkward size, individually named so a failure
    reports which dimension broke."""
    n = run_gemm([GemmCase(size, size, size)], SEED + 32)
    assert n == 1


def test_gemm_is_deterministic():
    rng = np.random.default_rng(SEED + 33)
    a = rng.uniform(-2, 2, size=(64, 128)).astype(np.float32)
    b = rng.uniform(-2, 2, size=(128, 64)).astype(np.float32)
    ta = V.tensor(a, device=gpu_device())
    tb = V.tensor(b, device=gpu_device())

    first = V.matmul(ta, tb).numpy()
    for _ in range(8):
        assert np.array_equal(V.matmul(ta, tb).numpy(), first), "GEMM is not deterministic"


def test_gemm_large_k_backward_error():
    """K=4096 with sign-mixed inputs: heavy cancellation, where a relative
    check would be meaningless and a sequential accumulator would drift ~50x
    over the pairwise bound."""
    n = run_gemm([GemmCase(4, 4, 4096)], SEED + 34)
    assert n == 1


# ---------------------------------------------------------------------------
# GEMM Stage 4: tile-boundary cases
#
# A tiled kernel fails in ways a naive one cannot: partial tiles on any edge,
# dimensions straddling the tile size, and K not dividing evenly. TILE is 16, so
# every case below is placed relative to 16 and 32.
# ---------------------------------------------------------------------------

TILE = 16

TILE_EDGE_CASES = [
    # Smaller than one tile on each dimension in turn.
    GemmCase(1, 16, 16), GemmCase(16, 1, 16), GemmCase(16, 16, 1),
    GemmCase(15, 15, 15), GemmCase(8, 8, 8),
    # Exactly one tile, and exactly two.
    GemmCase(16, 16, 16), GemmCase(32, 32, 32),
    # One either side of a tile boundary, per dimension.
    GemmCase(15, 16, 16), GemmCase(17, 16, 16),
    GemmCase(16, 15, 16), GemmCase(16, 17, 16),
    GemmCase(16, 16, 15), GemmCase(16, 16, 17),
    GemmCase(31, 32, 32), GemmCase(33, 32, 32),
    GemmCase(32, 31, 32), GemmCase(32, 33, 32),
    GemmCase(32, 32, 31), GemmCase(32, 32, 33),
    # Partial tiles on every edge simultaneously.
    GemmCase(17, 17, 17), GemmCase(33, 33, 33), GemmCase(47, 47, 47),
    # Rectangular, so the M and N tile counts differ.
    GemmCase(48, 17, 33), GemmCase(17, 48, 33), GemmCase(33, 17, 48),
    # Larger, with a partial tile on all three.
    GemmCase(129, 130, 131), GemmCase(255, 257, 129),
    # Batched and transposed with partial tiles.
    GemmCase(17, 19, 23, batch=(2,)), GemmCase(33, 17, 31, transpose_a=True),
    GemmCase(17, 33, 31, transpose_b=True),
    GemmCase(31, 31, 31, batch=(2, 2), transpose_b=True),
]


@pytest.mark.parametrize("case", TILE_EDGE_CASES, ids=[c.describe() for c in TILE_EDGE_CASES])
def test_gemm_tile_boundaries(case):
    """Each case named individually so a failure reports the exact dimensions."""
    assert run_gemm([case], SEED + 40) == 1


def test_gemm_tiled_and_naive_agree_within_policy():
    """The two GPU kernels must agree to within the BACKWARD tolerance.

    NOT bit-for-bit, and the distinction matters. An earlier version of this
    test asserted exact equality and failed at 2 ULP. That was the assertion
    being wrong: the naive kernel folds K in blocks of 32 (matching the CPU's
    kPairwiseBlock) while the tiled kernel folds one block per K-tile, i.e. 16.
    Different block sizes are different summation trees, so different -- both
    valid -- roundings.

    Determinism in this project means: same kernel, same input, same bits, every
    run. It does not mean two different algorithms produce identical bits.
    Per-kernel determinism is covered by test_gemm_is_deterministic.
    """
    import os
    import subprocess
    import sys

    script = (
        "import sys, numpy as np; sys.path.insert(0, 'python'); import vkml as V\n"
        "V.set_log_level(V.LogLevel.WARN); V.init_vulkan(0)\n"
        "rng = np.random.default_rng(7)\n"
        "a = rng.uniform(-2, 2, size=(64, 96)).astype(np.float32)\n"
        "b = rng.uniform(-2, 2, size=(96, 48)).astype(np.float32)\n"
        "d = V.device('vulkan:0')\n"
        "r = V.matmul(V.tensor(a, device=d), V.tensor(b, device=d)).numpy()\n"
        "sys.stdout.buffer.write(r.tobytes())\n"
    )
    root = os.path.join(os.path.dirname(__file__), "..", "..")

    def run(naive: bool) -> np.ndarray:
        env = dict(os.environ)
        env["VKML_GEMM_KERNEL"] = "naive" if naive else "reg"
        out = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                             capture_output=True, check=True)
        return np.frombuffer(out.stdout, dtype=np.float32)

    tiled = run(False)
    naive = run(True)

    rng = np.random.default_rng(7)
    a = rng.uniform(-2, 2, size=(64, 96)).astype(np.float32)
    b = rng.uniform(-2, 2, size=(96, 48)).astype(np.float32)
    abs_prod = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))

    ctx = Context(op="matmul", layout=Layout(a.shape), dtype="f32", seed=7,
                  inputs=[a, b], extra="tiled vs naive GPU kernel")
    compare(ctx, tiled.reshape(64, 48), naive.reshape(64, 48),
            terms_abs_sum=float(abs_prod.max()), reduction_n=96)

    # No additional ULP assertion here. One was tried and removed: it reported
    # 4544 ULP, which sounds alarming but is meaningless -- these outputs pass
    # through zero, and ULP spacing collapses near zero, so a tiny absolute
    # difference becomes a huge ULP count. The BACKWARD policy above is the
    # well-posed check for a dot product and it is sufficient on its own.


# ---------------------------------------------------------------------------
# GEMM Stage 5: register-block boundaries
#
# The block tile is 32x32 with a 2x2 register block, so an invocation owns two
# adjacent rows and two adjacent columns. Partial OUTPUT blocks -- where a
# thread owns fewer than 2x2 valid elements -- are a failure mode the Stage 4
# kernel could not have.
# ---------------------------------------------------------------------------

REG_EDGE_CASES = [
    # Odd extents, so the final thread in a row/column owns exactly one valid
    # element rather than two.
    GemmCase(1, 1, 1), GemmCase(1, 2, 3), GemmCase(3, 1, 2),
    GemmCase(31, 31, 31), GemmCase(33, 33, 33),
    GemmCase(31, 33, 32), GemmCase(33, 31, 32),
    # One either side of the 32-wide block tile, per dimension.
    GemmCase(31, 32, 32), GemmCase(32, 31, 32), GemmCase(32, 32, 31),
    GemmCase(33, 32, 32), GemmCase(32, 33, 32), GemmCase(32, 32, 33),
    # Odd in M only, N only, both -- isolating each partial-block axis.
    GemmCase(65, 64, 64), GemmCase(64, 65, 64), GemmCase(65, 65, 64),
    # K not a multiple of BK=16.
    GemmCase(64, 64, 17), GemmCase(64, 64, 47), GemmCase(64, 64, 129),
    # Non-square, so the M and N block counts differ.
    GemmCase(96, 34, 48), GemmCase(34, 96, 48),
    # Batched and transposed with partial blocks on every axis.
    GemmCase(33, 35, 37, batch=(2,)),
    GemmCase(31, 33, 35, transpose_a=True),
    GemmCase(33, 31, 35, transpose_b=True),
    GemmCase(17, 19, 23, batch=(2, 2), transpose_a=True, transpose_b=True),
]


@pytest.mark.parametrize("case", REG_EDGE_CASES, ids=[c.describe() for c in REG_EDGE_CASES])
def test_gemm_register_block_boundaries(case):
    assert run_gemm([case], SEED + 50) == 1


@pytest.mark.parametrize("kernel", ["naive", "tiled", "reg"])
def test_all_gemm_kernels_agree(kernel):
    """Every kernel variant must satisfy the same BACKWARD policy.

    Run in a subprocess because the kernel choice is read once per process.
    """
    import os
    import subprocess
    import sys

    script = (
        "import sys, numpy as np; sys.path.insert(0, 'python'); import vkml as V\n"
        "V.set_log_level(V.LogLevel.WARN); V.init_vulkan(0)\n"
        "rng = np.random.default_rng(11)\n"
        "a = rng.uniform(-2, 2, size=(70, 90)).astype(np.float32)\n"
        "b = rng.uniform(-2, 2, size=(90, 50)).astype(np.float32)\n"
        "d = V.device('vulkan:0')\n"
        "r = V.matmul(V.tensor(a, device=d), V.tensor(b, device=d)).numpy()\n"
        "sys.stdout.buffer.write(r.tobytes())\n"
    )
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    env = dict(os.environ)
    env["VKML_GEMM_KERNEL"] = kernel
    out = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                         capture_output=True, check=True)
    got = np.frombuffer(out.stdout, dtype=np.float32).reshape(70, 50)

    rng = np.random.default_rng(11)
    a = rng.uniform(-2, 2, size=(70, 90)).astype(np.float32)
    b = rng.uniform(-2, 2, size=(90, 50)).astype(np.float32)
    want = V.matmul(V.tensor(a, device=V.cpu), V.tensor(b, device=V.cpu)).numpy()
    abs_prod = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))

    ctx = Context(op="matmul", layout=Layout(a.shape), dtype="f32", seed=11,
                  inputs=[a, b], extra=f"GEMM kernel = {kernel}")
    compare(ctx, got, want, terms_abs_sum=float(abs_prod.max()), reduction_n=90)
