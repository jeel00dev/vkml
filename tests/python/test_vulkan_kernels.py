"""Vulkan kernel validation against the CPU oracle.

Adding a kernel here should be one entry in a list. If it needs more than that,
the framework in vkvalidate.py is missing something.
"""

from __future__ import annotations

import numpy as np
import pytest

import vkml as V
from vkvalidate import (
    VULKAN_DEVICE,
    Context,
    Layout,
    compare,
    gpu_device,
    make_data,
    random_layouts,
    requires_radv,
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
# Normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,axes", [
    ((4, 8), 1),
    ((2, 3, 64), 1),
    ((2, 3, 16), 2),
    ((1, 257), 1),      # straddles a workgroup boundary inside the reduction
    ((3, 1), 1),        # width 1: variance is exactly 0, eps carries the result
])
@pytest.mark.parametrize("name,fn", [("layer_norm", V.layer_norm), ("rms_norm", V.rms_norm)])
def test_norm_on_gpu(name, fn, shape, axes):
    """These are compositions, not kernels, so this is really a test that the
    whole chain -- mean, sub, square, add, rsqrt, mul -- agrees end to end
    between backends. A reduction that folds differently across workgroups
    would show up here and nowhere in the per-op tests."""
    rng = np.random.default_rng(SEED)
    x = make_data(rng, shape, "any")

    def run(dev):
        return fn(V.tensor(x, device=dev), axes).numpy()

    ctx = Context(op=name, layout=Layout(shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, run(gpu_device()), run(V.cpu))


def test_layer_norm_is_standardised_on_gpu():
    """The defining property, checked on the device rather than inferred from
    the CPU comparison."""
    rng = np.random.default_rng(SEED)
    x = make_data(rng, (6, 512), "any")
    y = V.layer_norm(V.tensor(x, device=gpu_device()), 1).numpy()

    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_a,shape_b,axis", [
    ((2, 3), (2, 3), 0),
    ((2, 3), (2, 3), 1),      # inner axis: the interleaving case
    ((2, 3), (2, 5), 1),      # unequal extents on the joined axis
    ((4, 2), (1, 2), 0),
    ((2, 3, 4), (2, 3, 4), 1),
    ((2, 3, 4), (2, 3, 2), 2),
    ((1,), (7,), 0),
    ((2, 3), (0, 3), 0),      # empty operand
])
def test_cat_on_gpu(shape_a, shape_b, axis):
    """Joining on an inner axis is the case that separates a correct kernel
    from one that just appends buffers: the operands interleave, and each
    source index must be rebuilt with its own extent rather than the output's."""
    rng = np.random.default_rng(SEED)
    a = make_data(rng, shape_a, "any")
    b = make_data(rng, shape_b, "any")

    def run(dev):
        return V.cat([V.tensor(a, device=dev), V.tensor(b, device=dev)], axis).numpy()

    ctx = Context(op="cat", layout=Layout(shape_a, "broadcast", shape_b),
                  dtype="f32", seed=SEED, inputs=[a, b])
    compare(ctx, run(gpu_device()), run(V.cpu))


def test_cat_of_strided_operands_on_gpu():
    """Both operands transposed, so the kernel resolves two independent stride
    sets while also remapping the index across the join."""
    rng = np.random.default_rng(SEED)
    a = make_data(rng, (4, 3), "any")
    b = make_data(rng, (4, 5), "any")

    def run(dev):
        ta = V.tensor(a, device=dev).transpose(0, 1)   # (3, 4)
        tb = V.tensor(b, device=dev).transpose(0, 1)   # (5, 4)
        return V.cat([ta, tb], 0).numpy()

    ctx = Context(op="cat", layout=Layout((4, 3), "transpose", (0, 1)),
                  dtype="f32", seed=SEED, inputs=[a, b])
    compare(ctx, run(gpu_device()), run(V.cpu))


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


def test_device_meets_what_the_kernels_require():
    """What vkML actually needs of any device, as opposed to what this one has.

    Split out from a test that asserted `global_float_atomics is False`. That
    was a fact about the development GPU, not a requirement: the reductions do
    not use atomics -- because their ordering is not reproducible, and only
    incidentally because RADV lacks them -- so a device that offers them is no
    threat to a design that never calls them. lavapipe offers them, and vkML
    runs there unchanged.
    """
    caps = V.vulkan_capabilities(VULKAN_DEVICE)

    # The GEMM tiles are sized against this; below it they would not fit.
    assert caps["max_shared_memory_bytes"] >= 32 * 1024
    assert caps["min_subgroup_size"] <= caps["subgroup_size"] <= caps["max_subgroup_size"]
    assert caps["max_workgroup_invocations"] >= 256, "the default workgroup size"


@requires_radv
def test_reference_gpu_assumptions_still_hold():
    """The facts the TUNING assumes, on the GPU it was tuned for.

    These are not portability requirements -- they are the properties that made
    particular design choices right on RADV/Navi10, and a driver update changing
    one silently is what this catches. Skipped elsewhere, because elsewhere they
    are simply different.
    """
    caps = V.vulkan_capabilities(VULKAN_DEVICE)
    assert caps["global_float_atomics"] is False, (
        "global float atomicAdd appeared on RADV; the deterministic two-pass "
        "reduction was chosen partly because it was absent"
    )
    assert caps["cooperative_matrix"] is False
    assert caps["subgroup_size"] == 64


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


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,axis,idx", [
    ((5, 3), 0, [0, 2, 4]),
    ((5, 3), 1, [2, 0]),
    ((5, 3), 0, [1, 1, 1]),
    ((4, 3, 2), 1, [2, 0, 1]),
    ((6,), 0, [5, 0, 3]),
    ((257, 2), 0, list(range(0, 257, 3))),   # crosses a workgroup boundary
])
def test_index_select_on_gpu(shape, axis, idx):
    rng = np.random.default_rng(SEED)
    x = make_data(rng, shape, "any")
    i = np.array(idx, dtype=np.int64)

    def run(dev):
        return V.index_select(V.tensor(x, device=dev), axis, V.tensor(i, device=dev)).numpy()

    ctx = Context(op="index_select", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, run(gpu_device()), run(V.cpu))


@pytest.mark.parametrize("axis,idx,dim_size", [
    (0, [0, 2, 2, 1], 4),
    (0, [1, 1, 1, 1], 3),
    (0, [0, 1, 2, 3], 4),
    (1, [0, 0, 1], 2),
    (0, [0] * 64, 3),        # heavy contention on one row
])
def test_scatter_add_on_gpu(axis, idx, dim_size):
    """The kernel inverts the loop -- one thread per destination, scanning the
    index -- precisely so that repeated indices need no atomic. Contention is
    therefore the case to test, not the case to avoid."""
    rng = np.random.default_rng(SEED)
    i = np.array(idx, dtype=np.int64)
    shape = (len(idx), 3) if axis == 0 else (3, len(idx))
    x = make_data(rng, shape, "any")

    def run(dev):
        return V.scatter_add(V.tensor(x, device=dev), axis, V.tensor(i, device=dev),
                             dim_size).numpy()

    ctx = Context(op="scatter_add", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, run(gpu_device()), run(V.cpu))


def test_scatter_add_is_bit_reproducible():
    """The policy entry claims EXACT on the grounds that both backends fold in
    ascending index order. This is that claim under test.

    Float addition is not associative, so a different fold order shows up as a
    different result -- values are chosen with widely separated magnitudes so a
    reordering cannot hide inside the rounding.
    """
    idx = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
    x = np.array([[1e8], [1.0], [-1e8], [1.0], [1e-8], [1.0]], dtype=np.float32)

    gpu = V.scatter_add(V.tensor(x, device=gpu_device()), 0,
                        V.tensor(idx, device=gpu_device()), 1).numpy()
    cpu = V.scatter_add(V.tensor(x, device=V.cpu), 0, V.tensor(idx, device=V.cpu), 1).numpy()

    assert gpu.tobytes() == cpu.tobytes(), (
        f"fold order differs between backends: gpu={gpu.tolist()} cpu={cpu.tolist()}"
    )
    # Repeating on the GPU must give the identical bytes, not merely close ones.
    again = V.scatter_add(V.tensor(x, device=gpu_device()), 0,
                          V.tensor(idx, device=gpu_device()), 1).numpy()
    assert gpu.tobytes() == again.tobytes(), "GPU result is not reproducible run to run"


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,c", [(4, 3), (1, 2), (16, 10), (3, 100), (2, 257)])
def test_cross_entropy_on_gpu(n, c):
    """The full chain -- arange, cast, equal, log_softmax, mul, sum, neg, mean
    -- on the device. Until arange had a kernel this raised NotImplementedError
    rather than falling back, so this is also the regression test for that."""
    rng = np.random.default_rng(SEED)
    logits = make_data(rng, (n, c), "any")
    labels = rng.integers(0, c, size=n).astype(np.int64)

    def run(dev):
        return V.cross_entropy(V.tensor(logits, device=dev),
                               V.tensor(labels, device=dev)).numpy()

    ctx = Context(op="cross_entropy", layout=Layout((n, c)), dtype="f32", seed=SEED,
                  inputs=[logits])
    compare(ctx, run(gpu_device()), run(V.cpu))


@pytest.mark.parametrize("shape", [(8,), (4, 5), (2, 3, 4)])
def test_mse_loss_on_gpu(shape):
    rng = np.random.default_rng(SEED)
    a = make_data(rng, shape, "any")
    b = make_data(rng, shape, "any")

    def run(dev):
        return V.mse_loss(V.tensor(a, device=dev), V.tensor(b, device=dev)).numpy()

    ctx = Context(op="mse_loss", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[a, b])
    compare(ctx, run(gpu_device()), run(V.cpu))


@pytest.mark.parametrize("shape", [(8,), (4, 5), (2, 3, 4)])
def test_binary_cross_entropy_with_logits_on_gpu(shape):
    """maximum, abs, neg, exp, add-scalar, log, sub, mul and a mean, composed."""
    rng = np.random.default_rng(SEED)
    logits = make_data(rng, shape, "any")
    target = rng.random(shape).astype(np.float32)

    def run(dev):
        return V.binary_cross_entropy_with_logits(V.tensor(logits, device=dev),
                                                  V.tensor(target, device=dev)).numpy()

    ctx = Context(op="binary_cross_entropy_with_logits", layout=Layout(shape), dtype="f32",
                  seed=SEED, inputs=[logits, target])
    compare(ctx, run(gpu_device()), run(V.cpu))


@pytest.mark.parametrize("shape", [(8,), (4, 5)])
def test_kl_div_on_gpu(shape):
    """Exercises the `where` that keeps 0 * log(0) out of the result.

    The target deliberately contains exact zeros: on the device both branches
    of the selection are evaluated, so log(0) really does produce -inf here and
    the test is that it gets discarded rather than multiplied.
    """
    rng = np.random.default_rng(SEED)
    log_input = np.log(rng.random(shape).astype(np.float32) + 1e-3)
    target = rng.random(shape).astype(np.float32)
    target.reshape(-1)[::3] = 0.0

    def run(dev):
        return V.kl_div(V.tensor(log_input, device=dev),
                        V.tensor(target, device=dev)).numpy()

    ctx = Context(op="kl_div", layout=Layout(shape), dtype="f32", seed=SEED,
                  inputs=[log_input, target])
    compare(ctx, run(gpu_device()), run(V.cpu))


@pytest.mark.parametrize("shape", [(8,), (4, 5), (2, 3, 4)])
def test_huber_loss_on_gpu(shape):
    """Inputs are scaled so both sides of the delta join occur in every shape."""
    rng = np.random.default_rng(SEED)
    a = make_data(rng, shape, "any") * 3.0
    b = make_data(rng, shape, "any") * 3.0

    def run(dev):
        return V.huber_loss(V.tensor(a, device=dev), V.tensor(b, device=dev)).numpy()

    ctx = Context(op="huber_loss", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[a, b])
    compare(ctx, run(gpu_device()), run(V.cpu))


def test_arange_on_gpu():
    """fill and arange share a kernel -- a fill is an arange with slope 0 --
    so this covers the slope path that fill never exercises."""
    for start, stop, step in ((0.0, 10.0, 1.0), (-5.0, 5.0, 0.5), (2.0, 300.0, 3.0)):
        gpu = V.arange(start, stop, step, device=gpu_device()).numpy()
        cpu = V.arange(start, stop, step, device=V.cpu).numpy()
        ctx = Context(op="arange", layout=Layout(gpu.shape), dtype="f32", seed=SEED)
        compare(ctx, gpu, cpu)


def test_cross_entropy_gradient_on_gpu():
    """A loss is only useful if it produces a gradient, and this is the first
    point in the project where a full forward and backward pass runs on the
    device."""
    rng = np.random.default_rng(SEED)
    logits = make_data(rng, (6, 5), "any")
    labels = rng.integers(0, 5, size=6).astype(np.int64)

    def grad(dev):
        v = V.tensor(logits, device=dev, requires_grad=True)
        V.cross_entropy(v, V.tensor(labels, device=dev)).backward()
        return v.grad.numpy()

    ctx = Context(op="cross_entropy", layout=Layout((6, 5)), dtype="f32", seed=SEED,
                  inputs=[logits])
    compare(ctx, grad(gpu_device()), grad(V.cpu))


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

WINDOWS = [
    ((1, 1, 4, 4), (2, 2), (1, 1), (0, 0), (1, 1)),
    ((2, 3, 5, 5), (3, 3), (1, 1), (1, 1), (1, 1)),   # padded, overlapping
    ((1, 2, 6, 6), (2, 2), (2, 2), (0, 0), (1, 1)),   # strided, no overlap
    ((2, 1, 7, 5), (3, 2), (2, 1), (1, 0), (1, 1)),   # asymmetric everything
    ((1, 2, 5, 5), (2, 2), (1, 1), (0, 0), (2, 2)),   # dilated
    ((1, 1, 17, 17), (3, 3), (1, 1), (1, 1), (1, 1)),  # crosses a workgroup
]


@pytest.mark.parametrize("shape,kernel,stride,pad,dil", WINDOWS)
def test_im2col_on_gpu(shape, kernel, stride, pad, dil):
    rng = np.random.default_rng(SEED)
    x = make_data(rng, shape, "any")

    def run(dev):
        return V.im2col(V.tensor(x, device=dev), kernel, stride, pad, dil).numpy()

    ctx = Context(op="im2col", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, run(gpu_device()), run(V.cpu))


@pytest.mark.parametrize("shape,kernel,stride,pad,dil", WINDOWS)
def test_col2im_on_gpu(shape, kernel, stride, pad, dil):
    """Overlapping geometries are the ones that exercise the accumulation the
    inverted loop exists to make deterministic."""
    rng = np.random.default_rng(SEED)
    x = make_data(rng, shape, "any")

    def run(dev):
        cols = V.im2col(V.tensor(x, device=dev), kernel, stride, pad, dil)
        return V.col2im(cols, shape[2:], kernel, stride, pad, dil).numpy()

    ctx = Context(op="col2im", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, run(gpu_device()), run(V.cpu))


def test_col2im_fold_is_bit_reproducible():
    """The policy claims EXACT on the grounds that both backends fold in the
    same kernel order. Magnitudes are spread widely so a reordering cannot hide
    inside the rounding, and the geometry gives each interior position nine
    overlapping contributions."""
    cols = np.zeros((1, 9, 16), dtype=np.float32)
    cols[0, :, :] = np.array([1e8, 1.0, -1e8, 1e-8, 1.0, -1.0, 1e7, 1.0, -1e7],
                             dtype=np.float32)[:, None]

    def run(dev):
        return V.col2im(V.tensor(cols, device=dev), [4, 4], [3, 3], [1, 1], [1, 1]).numpy()

    gpu, cpu = run(gpu_device()), run(V.cpu)
    assert gpu.tobytes() == cpu.tobytes(), f"fold order differs:\n{gpu}\nvs\n{cpu}"
    assert gpu.tobytes() == run(gpu_device()).tobytes(), "GPU result is not reproducible"


@pytest.mark.parametrize("xs,ws,stride,pad,dil", [
    ((1, 1, 5, 5), (2, 1, 3, 3), (1, 1), (0, 0), (1, 1)),
    ((2, 3, 6, 6), (4, 3, 3, 3), (1, 1), (1, 1), (1, 1)),
    ((1, 2, 8, 8), (3, 2, 2, 2), (2, 2), (0, 0), (1, 1)),
    ((1, 2, 7, 7), (2, 2, 3, 3), (1, 1), (2, 2), (2, 2)),
])
def test_conv2d_on_gpu(xs, ws, stride, pad, dil):
    """conv2d is a composition, so this checks the whole chain -- im2col, the
    tuned matmul, reshape and the bias broadcast -- agrees between backends."""
    rng = np.random.default_rng(SEED)
    x = make_data(rng, xs, "any")
    w = make_data(rng, ws, "any")
    b = make_data(rng, (ws[0],), "any")

    def run(dev):
        return V.conv2d(V.tensor(x, device=dev), V.tensor(w, device=dev),
                        V.tensor(b, device=dev), stride, pad, dil).numpy()

    ctx = Context(op="matmul", layout=Layout(xs), dtype="f32", seed=SEED, inputs=[x, w])
    gpu, cpu = run(gpu_device()), run(V.cpu)
    # matmul carries a BACKWARD bound; the reduction length is the flattened
    # patch, and the terms are the products summed to make one output element.
    k = ws[1] * ws[2] * ws[3]
    compare(ctx, gpu, cpu, terms_abs_sum=float(np.abs(x).max() * np.abs(w).max() * k),
            reduction_n=k)


POOLS = [
    ((1, 1, 4, 4), (2, 2), (2, 2), (0, 0)),
    ((2, 3, 6, 6), (2, 2), (2, 2), (0, 0)),
    ((1, 2, 5, 5), (3, 3), (1, 1), (1, 1)),   # padded, overlapping
    ((2, 1, 7, 5), (3, 2), (2, 1), (1, 0)),   # asymmetric
    ((1, 1, 17, 17), (3, 3), (2, 2), (1, 1)),  # crosses a workgroup boundary
]


@pytest.mark.parametrize("shape,kernel,stride,pad", POOLS)
def test_max_pool2d_on_gpu(shape, kernel, stride, pad):
    rng = np.random.default_rng(SEED)
    x = make_data(rng, shape, "any")

    def run(dev):
        return V.max_pool2d(V.tensor(x, device=dev), kernel, stride, pad).numpy()

    ctx = Context(op="max_pool2d", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, run(gpu_device()), run(V.cpu))


@pytest.mark.parametrize("shape,kernel,stride,pad", POOLS)
def test_max_pool2d_backward_on_gpu(shape, kernel, stride, pad):
    """Overlapping windows are the case the inverted loop exists for: one input
    position can be the maximum of several windows and so receives several
    contributions."""
    rng = np.random.default_rng(SEED)
    x = make_data(rng, shape, "any")

    def grad(dev):
        v = V.tensor(x, device=dev, requires_grad=True)
        pooled = V.max_pool2d(v, kernel, stride, pad)
        V.sum(V.mul(pooled, pooled)).backward()
        return v.grad.numpy()

    ctx = Context(op="max_pool2d_backward", layout=Layout(shape), dtype="f32", seed=SEED,
                  inputs=[x])
    compare(ctx, grad(gpu_device()), grad(V.cpu))


def test_max_pool2d_negative_padding_on_gpu():
    """The -infinity padding, on the device. A shader padding with zero would
    return 0 here and pass every shape check."""
    x = -np.abs(make_data(np.random.default_rng(SEED), (1, 2, 5, 5), "any")) - 1.0

    def run(dev):
        return V.max_pool2d(V.tensor(x, device=dev), [3, 3], [1, 1], [1, 1]).numpy()

    gpu = run(gpu_device())
    ctx = Context(op="max_pool2d", layout=Layout(x.shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, gpu, run(V.cpu))
    assert not np.any(gpu == 0.0), f"zero leaked from the padding: {gpu}"


def test_max_pool2d_tie_resolves_identically_on_gpu():
    """Both backends must pick the same maximum when a window contains a tie,
    or their gradients diverge while their forward outputs agree."""
    x = np.array([[[[5.0, 5.0], [1.0, 2.0]]]], dtype=np.float32)

    def grad(dev):
        v = V.tensor(x, device=dev, requires_grad=True)
        V.sum(V.max_pool2d(v, [2, 2])).backward()
        return v.grad.numpy()

    gpu, cpu = grad(gpu_device()), grad(V.cpu)
    assert gpu.tobytes() == cpu.tobytes(), f"tie broken differently:\n{gpu}\nvs\n{cpu}"


@pytest.mark.parametrize("shape,kernel,stride,pad", POOLS[:3])
def test_avg_pool2d_on_gpu(shape, kernel, stride, pad):
    rng = np.random.default_rng(SEED)
    x = make_data(rng, shape, "any")

    def run(dev):
        return V.avg_pool2d(V.tensor(x, device=dev), kernel, stride, pad).numpy()

    ctx = Context(op="mean", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[x])
    n = kernel[0] * kernel[1]
    compare(ctx, run(gpu_device()), run(V.cpu),
            terms_abs_sum=float(np.abs(x).max() * n), reduction_n=n)


# ---------------------------------------------------------------------------
# Random number generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 255, 256, 257, 4096, 100_000])
def test_rand_is_bit_identical_across_backends(n):
    """The two Philox implementations must agree EXACTLY, not merely be
    similarly uniform. Each would look perfectly random on its own while
    disagreeing on every value, so a distributional test cannot catch a
    divergence in the round structure -- only a byte comparison can.

    Sizes straddle the workgroup boundary because the counter is the global
    invocation index: a shader seeding per workgroup instead would agree at
    n <= 256 and diverge above it.
    """
    gpu = V.rand([n], 20260728, 5, device=gpu_device()).numpy()
    cpu = V.rand([n], 20260728, 5, device=V.cpu).numpy()
    assert gpu.tobytes() == cpu.tobytes(), (
        f"philox diverged at n={n}: first mismatch at "
        f"{int(np.argmax(gpu != cpu))}, gpu={gpu[gpu != cpu][:3]} cpu={cpu[gpu != cpu][:3]}"
    )


def test_rand_is_reproducible_across_runs_on_gpu():
    a = V.rand([10_000], 3, 7, device=gpu_device()).numpy()
    b = V.rand([10_000], 3, 7, device=gpu_device()).numpy()
    assert a.tobytes() == b.tobytes(), "GPU generator is not reproducible run to run"


@pytest.mark.parametrize("p", [0.25, 0.5, 0.9])
def test_dropout_mask_matches_across_backends(p):
    """Dropout composes from rand, comparison and select, so an identical
    generator must give an identical mask -- the same elements dropped, not
    merely the same fraction."""
    rng = np.random.default_rng(SEED)
    x = make_data(rng, (64, 64), "any")

    def run(dev):
        return V.dropout(V.tensor(x, device=dev), p, seed=1234, offset=9).numpy()

    ctx = Context(op="dropout", layout=Layout((64, 64)), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, run(gpu_device()), run(V.cpu))


# ---------------------------------------------------------------------------
# Module placement and training on the device
# ---------------------------------------------------------------------------


def test_module_to_moves_parameters_and_buffers():
    model = V.nn.Sequential(V.nn.Conv2d(1, 2, 3, padding=1), V.nn.BatchNorm2d(2))
    before = {name: a.copy() for name, a in model.state_dict().items()}

    model.to(gpu_device())

    expected = str(gpu_device())
    for _, p in model.named_parameters():
        assert str(p.device) == expected, p.device
    for _, b in model.named_buffers():
        assert str(b.device) == expected, b.device

    # Values survive the move unchanged; only their residence differs.
    after = model.state_dict()
    for name, value in before.items():
        assert np.array_equal(after[name], value), name


def test_module_to_carries_gradients():
    """Dropping gradients would leave a subsequent optimiser step silently
    updating nothing, which is worse than failing."""
    model = V.nn.Linear(4, 3)
    x = V.tensor(np.ones((2, 4), dtype=np.float32))
    V.sum(model(x)).backward()
    grad_before = model.weight.grad.numpy().copy()

    model.to(gpu_device())
    assert model.weight.grad.defined(), "gradient was dropped by to()"
    assert np.allclose(model.weight.grad.numpy(), grad_before)


def test_cnn_trains_on_the_device():
    """The whole stack on Vulkan: conv, batch norm, pooling, flatten, linear,
    cross-entropy, backward and an optimiser step. Asserts the loss actually
    falls -- a model that ran without error but learned nothing would pass a
    shape check."""
    model = V.nn.Sequential(
        V.nn.Conv2d(1, 4, 3, padding=1), V.nn.BatchNorm2d(4), V.nn.ReLU(),
        V.nn.MaxPool2d(2), V.nn.Flatten(), V.nn.Linear(4 * 4 * 4, 3)).to(gpu_device())
    opt = V.optim.SGD(model.parameters(), lr=0.05)

    rng = np.random.default_rng(SEED)
    x = V.tensor(rng.normal(size=(8, 1, 8, 8)).astype(np.float32), device=gpu_device())
    y = V.tensor(rng.integers(0, 3, size=8).astype(np.int64), device=gpu_device())

    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = V.nn.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0] * 0.75, f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_device_and_cpu_training_agree():
    """The same model, data and seed on both backends must follow the same
    trajectory -- the oracle chain applied to a whole training loop rather than
    to one operator."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(6, 5)).astype(np.float32)
    y = rng.integers(0, 4, size=6).astype(np.int64)
    init = {"0.weight": rng.normal(size=(4, 5)).astype(np.float32),
            "0.bias": rng.normal(size=(4,)).astype(np.float32)}

    def train(device):
        model = V.nn.Sequential(V.nn.Linear(5, 4))
        model.load_state_dict(init)
        model.to(device)
        opt = V.optim.SGD(model.parameters(), lr=0.1)
        tx = V.tensor(x, device=device)
        ty = V.tensor(y, device=device)
        out = []
        for _ in range(10):
            opt.zero_grad()
            loss = V.nn.cross_entropy(model(tx), ty)
            loss.backward()
            opt.step()
            out.append(loss.item())
        return np.array(out, dtype=np.float32)

    gpu, cpu = train(gpu_device()), train(V.cpu)
    ctx = Context(op="cross_entropy", layout=Layout((6, 5)), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, gpu, cpu)


@pytest.mark.parametrize("shape,axis,start,stop,step", [
    ((8,), 0, 2, 6, 1),
    ((8,), 0, 1, 8, 3),          # step > 1: skipped positions inside the range
    ((4, 6), 1, 1, 5, 2),
    ((4, 6), 0, 0, 4, 1),        # whole axis
    ((2, 3, 5), 2, 3, 5, 1),
    ((257,), 0, 5, 250, 7),      # crosses a workgroup boundary
])
def test_slice_backward_on_gpu(shape, axis, start, stop, step):
    """Positions the slice skipped -- including those INSIDE its range when the
    step exceeds one -- must receive exactly zero, not a neighbour's gradient."""
    rng = np.random.default_rng(SEED)
    x = make_data(rng, shape, "any")

    def grad(dev):
        v = V.tensor(x, device=dev, requires_grad=True)
        key = [slice(None)] * len(shape)
        key[axis] = slice(start, stop, step)
        V.sum(v[tuple(key)] * 2.0).backward()
        return v.grad.numpy()

    ctx = Context(op="copy", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, grad(gpu_device()), grad(V.cpu))


def test_transformer_trains_on_the_device():
    """Embedding, attention, feed-forward, residuals and normalisation on
    Vulkan, with an optimiser step.

    This is what needed slice_backward: attention slices the packed
    in_proj_weight, so its gradient scatters back through a narrowing.
    """
    vocab, d_model, seq = 20, 8, 5
    model = V.nn.Sequential(
        V.nn.Embedding(vocab, d_model),
        V.nn.TransformerEncoderLayer(d_model, 2, dim_feedforward=16, dropout=0.0),
        V.nn.Flatten(),
        V.nn.Linear(seq * d_model, 3),
    ).to(gpu_device())
    model.train()
    opt = V.optim.Adam(model.parameters(), lr=1e-2)

    rng = np.random.default_rng(SEED)
    tokens = V.tensor(rng.integers(0, vocab, size=(4, seq)).astype(np.int64),
                      device=gpu_device())
    labels = V.tensor(rng.integers(0, 3, size=4).astype(np.int64), device=gpu_device())

    losses = []
    for _ in range(25):
        opt.zero_grad()
        loss = V.nn.cross_entropy(model(tokens), labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0] * 0.5, f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_attention_agrees_across_backends():
    """The whole attention block compared between backends -- projection,
    head split, scaled scores, softmax, context and output projection."""
    rng = np.random.default_rng(SEED)
    x = make_data(rng, (2, 6, 8), "any")
    init = {
        "in_proj_weight": make_data(rng, (24, 8), "any"),
        "in_proj_bias": make_data(rng, (24,), "any"),
        "out_proj.weight": make_data(rng, (8, 8), "any"),
        "out_proj.bias": make_data(rng, (8,), "any"),
    }

    def run(dev):
        mha = V.nn.MultiheadAttention(8, 2)
        mha.load_state_dict(init)
        mha.to(dev)
        return mha(V.tensor(x, device=dev), is_causal=True).numpy()

    ctx = Context(op="softmax", layout=Layout((2, 6, 8)), dtype="f32", seed=SEED, inputs=[x])
    compare(ctx, run(gpu_device()), run(V.cpu))


# ---------------------------------------------------------------------------
# assign_
#
# The in-place write behind every optimiser update and BatchNorm's running
# statistics. It had NO test until now, which is how it went unnoticed that it
# moved every assignment through host memory even when source and destination
# were on the same device (docs/adr/0006-lazy-assign-and-submission-batching.md).
#
# These pin BEHAVIOUR, not the transfer path: the point of the device-side copy
# is that nothing observable changes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1,), (7,), (64, 64), (3, 4, 5)])
def test_assign_writes_the_source_values_on_both_backends(shape):
    """assign_ must leave the destination holding exactly the source bytes."""
    rng = np.random.default_rng(SEED)
    dst = make_data(rng, shape, "any")
    src = make_data(rng, shape, "any")

    def run(dev):
        d = V.tensor(dst, device=dev)
        d.assign_(V.tensor(src, device=dev))
        return d.numpy()

    ctx = Context(op="assign_", layout=Layout(shape), dtype="f32", seed=SEED, inputs=[dst, src])
    compare(ctx, run(gpu_device()), run(V.cpu))
    # Against numpy too, so a bug identical on both backends cannot pass.
    np.testing.assert_array_equal(run(gpu_device()), src)


def test_assign_is_visible_to_an_existing_alias():
    """The Module holds the same Tensor object, so the write must be in place.

    `optim.py` relies on this: it assigns THROUGH the parameter rather than
    rebinding it, because rebinding would update the optimiser's list and leave
    the model looking at the old tensor.
    """
    for dev in (gpu_device(), V.cpu):
        t = V.tensor(np.zeros((4,), dtype=np.float32), device=dev)
        alias = t
        t.assign_(V.tensor(np.arange(4, dtype=np.float32), device=dev))
        np.testing.assert_array_equal(alias.numpy(), np.arange(4, dtype=np.float32))


def test_assign_from_a_computed_expression():
    """The optimiser's actual shape: assign the result of arithmetic on the
    destination itself, which is a read of `p` feeding a write to `p`."""
    rng = np.random.default_rng(SEED)
    p0 = make_data(rng, (32, 32), "any")
    g = make_data(rng, (32, 32), "any")
    lr = 0.1

    def run(dev):
        p = V.tensor(p0, device=dev)
        grad = V.tensor(g, device=dev)
        p.assign_(p.detach() - grad * lr)
        return p.numpy()

    ctx = Context(op="assign_", layout=Layout((32, 32)), dtype="f32", seed=SEED, inputs=[p0, g])
    compare(ctx, run(gpu_device()), run(V.cpu))
    np.testing.assert_allclose(run(V.cpu), p0 - g * lr, rtol=1e-6, atol=1e-6)


def test_assign_to_itself_is_a_no_op():
    """`t.assign_(t)` is the degenerate overlap: source and destination are the
    same bytes. vkCmdCopyBuffer forbids overlapping regions in one buffer, so
    this is the case the device-side path must NOT take."""
    rng = np.random.default_rng(SEED)
    values = make_data(rng, (16,), "any")

    for dev in (gpu_device(), V.cpu):
        t = V.tensor(values, device=dev)
        t.assign_(t)
        np.testing.assert_array_equal(t.numpy(), values)


def test_assign_between_overlapping_slices_of_one_storage():
    """Partial overlap within a single storage, which slicing makes reachable.

    Same-shape views at different offsets of the same buffer. The device copy
    cannot express this, so assign_ must fall back to staging through the host
    -- and the result must still be the plain copy semantics numpy gives.
    """
    rng = np.random.default_rng(SEED)
    values = make_data(rng, (10,), "any")

    for dev in (gpu_device(), V.cpu):
        t = V.tensor(values, device=dev)
        t[0:5].assign_(t[2:7])
        expected = values.copy()
        expected[0:5] = values[2:7]
        np.testing.assert_array_equal(t.numpy(), expected), f"on {dev}"
