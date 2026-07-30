"""The two backends must agree at the EDGES of the float range, not just the middle.

WHY THIS FILE EXISTS
--------------------
`tanh(89.0)` returned NaN on Vulkan and 1.0 on the CPU (issue #26). A finite
input, a categorically wrong answer, and nothing caught it -- because every
operator test drew its values from a normal distribution, where |x| never
approaches the range the defect lives in.

The cause generalises past tanh. GLSL defines tanh as
(e^x - e^-x) / (e^x + e^-x), and an implementation evaluating that literally
gives inf/inf = NaN once |x| passes ln(FLT_MAX) = 88.7228. Whether it does is
left to the implementation: RADV does not, AMD's Windows driver does -- the same
shape as issue #3, where f32 -> f16 narrowing rounded differently on the two.

So this sweeps EVERY unary operator over the magnitudes where a naive
implementation breaks, and holds the CPU kernel as the oracle
(docs/ARCHITECTURE.md 7).

WHAT IT DOES NOT ASSERT, and why that matters more than what it does. An earlier
version demanded bit-for-bit agreement and failed on 16 of 17 operators. It was
the test that was wrong, and being wrong found something: the two backends
diverge in two separate, legitimate ways, and neither was written down anywhere.

  1. SUBNORMALS. Vulkan permits flush-to-zero for f32 denormals unless
     shaderDenormPreserveFloat32 is requested, and vkml does not request it. So
     relu(1e-45) is 0 on the GPU and 1e-45 on the CPU, and exp(-89) is 0 rather
     than the subnormal 2.227e-39. Pinned below rather than papered over.
  2. TRANSCENDENTAL ULP. exp, sin, cos, tanh differ by ~1 ULP, which is what the
     1e-5 gate in ARCHITECTURE.md 7.3 exists to permit.

The contract that DOES hold at the extremes is categorical, and it is the one
that catches #26: a finite input must not produce NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

import vkml as V
from vkvalidate import gpu_device, vulkan_ready

pytestmark = pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device")

#: Smallest positive NORMAL f32. Anything below this is subnormal, and a Vulkan
#: device is free to flush it to zero, so it is also the absolute tolerance
#: below which a disagreement carries no information.
FLT_MIN = np.float32(1.1754944e-38)

#: The transcendental gate from docs/ARCHITECTURE.md 7.3, which gives
#: atol = rtol = 1e-5 for exp / log / tanh / gelu. Applied as written: an
#: earlier draft used atol = FLT_MIN, which is far STRICTER than the project's
#: own policy and failed on differences of order 1e-20 that the policy permits.
RTOL = 1e-5
ATOL = 1e-5

# Every unary operator on the Python surface. Listed explicitly rather than
# discovered, so adding an operator without extending this list is a visible
# omission rather than a silent gap.
UNARY_OPS = [
    "relu", "neg", "abs", "exp", "sign", "square", "sqrt", "rsqrt",
    "reciprocal", "log", "erf", "sin", "cos", "tanh", "sigmoid", "gelu", "silu",
]

#: Operators whose result is NaN for negative input on every conforming
#: implementation. Exempted BY NAME so the exemption is a decision, not a hole.
UNDEFINED_FOR_NEGATIVES = {"sqrt", "rsqrt", "log"}

#: Magnitude past which GLSL does not require sin/cos to be accurate. The
#: driver's argument reduction gives up long before FLT_MAX -- measured here,
#: sin(3.4e38) is 0.0 on the GPU against -0.522 on the CPU -- and no amount of
#: care in vkml changes that, because the reduction happens inside the built-in.
TRIG_ACCURATE_BELOW = 1e6

#: Extreme-magnitude cases with a KNOWN cause, excluded from the numeric
#: comparison and named so each exclusion is a decision rather than a hole.
#: Anything NOT listed here must agree with the CPU within the gate.
KNOWN_DIVERGENCES = {
    "sin": "argument reduction above 1e6 is not required to be accurate",
    "cos": "argument reduction above 1e6 is not required to be accurate",
}


def _comparable_mask(op: str, x: np.ndarray) -> np.ndarray:
    """Which inputs the numeric comparison applies to, and why the rest are out."""
    keep = np.isfinite(x)
    if op in ("sin", "cos"):
        keep &= np.abs(x) <= TRIG_ACCURATE_BELOW
    return keep


def _normal_extremes() -> np.ndarray:
    """Magnitudes where a naive implementation stops agreeing with a careful one.

    Chosen, not sampled: the exp thresholds are the entire point. 88.7228 is
    ln(FLT_MAX), past which e^x is no longer representable, and every operator
    built on exp has its edge case within a few ulp of it. A random sweep of the
    same size lands here essentially never, which is how #26 survived.

    Subnormals are deliberately absent -- they get their own test below.
    """
    return np.array(
        [
            0.0, -0.0, 1.0, -1.0,
            # Around ln(FLT_MAX): where tanh, sigmoid and silu break if written
            # as a ratio of exponentials.
            88.0, 88.7228, 88.73, 89.0, 90.0,
            -88.0, -88.7228, -88.73, -89.0, -90.0,
            # Where tanh saturates exactly in f32, and either side of it.
            9.5, 10.0, 10.5, -9.5, -10.0, -10.5,
            # The f32 extremes and the smallest normals.
            3.4028235e38, -3.4028235e38,
            1.1754944e-38, -1.1754944e-38,
            np.inf, -np.inf,
        ],
        dtype=np.float32,
    )


def _finite_inputs_for(op: str) -> np.ndarray:
    x = _normal_extremes()
    x = x[np.isfinite(x)]
    return x[x > 0] if op in UNDEFINED_FOR_NEGATIVES else x


@pytest.mark.parametrize("op", UNARY_OPS)
def test_a_finite_input_never_produces_nan(op):
    """The rule issue #26 broke, stated as a rule rather than as one value.

    A NaN produced from a finite argument is the worst failure shape available:
    silent, self-propagating through every later operation, and mid-training it
    reads as a diverged model rather than a kernel defect. There is no tolerance
    involved -- this is categorical, so it holds on any driver.
    """
    x = _finite_inputs_for(op)
    got = getattr(V, op)(V.tensor(x, device=gpu_device())).numpy()

    bad = np.isnan(got)
    assert not bad.any(), (
        f"{op} produced NaN from finite input(s) {x[bad].tolist()} -- "
        "a finite argument must never yield NaN"
    )


@pytest.mark.parametrize("op", UNARY_OPS)
def test_nan_and_infinity_classification_matches_the_cpu(op):
    """Whether a result is NaN, infinite or finite must not depend on the backend.

    Checked separately from the numeric comparison because it is the categorical
    half of the contract, and the half a tolerance cannot express: NaN is not
    "far from" 1.0, it is a different kind of answer. This is what would have
    caught #26 on the reporting machine.
    """
    x = _normal_extremes()
    got = getattr(V, op)(V.tensor(x, device=gpu_device())).numpy()
    expected = getattr(V, op)(V.tensor(x, device=V.cpu)).numpy()

    for i, value in enumerate(x):
        assert np.isnan(got[i]) == np.isnan(expected[i]), (
            f"{op}({value!r}): vulkan={got[i]!r} cpu={expected[i]!r} -- "
            "the backends disagree about whether this is NaN"
        )
        assert np.isinf(got[i]) == np.isinf(expected[i]), (
            f"{op}({value!r}): vulkan={got[i]!r} cpu={expected[i]!r} -- "
            "the backends disagree about whether this is infinite"
        )


@pytest.mark.parametrize("op", UNARY_OPS)
def test_values_agree_with_the_cpu_within_the_documented_gate(op):
    """Numeric agreement on NORMAL magnitudes.

    The gate is the project's own, from ARCHITECTURE.md 7.3: atol = rtol = 1e-5.
    Not tighter -- an earlier draft demanded agreement to within the smallest
    normal and failed on differences of order 1e-20, which that policy expressly
    permits. The categorical tests above are what carry the weight here; this one
    only asserts that no operator has drifted outside its stated bound.
    """
    x = _normal_extremes()
    got = getattr(V, op)(V.tensor(x, device=gpu_device())).numpy()
    expected = getattr(V, op)(V.tensor(x, device=V.cpu)).numpy()

    keep = _comparable_mask(op, x) & np.isfinite(got) & np.isfinite(expected)
    np.testing.assert_allclose(
        got[keep], expected[keep], rtol=RTOL, atol=ATOL,
        err_msg=(f"{op} disagrees with the CPU oracle beyond the documented gate"
                 + (f" (known divergences excluded: {KNOWN_DIVERGENCES[op]})"
                    if op in KNOWN_DIVERGENCES else "")),
    )


def test_the_gpu_may_flush_subnormals_and_the_cpu_does_not():
    """Pins a real divergence rather than pretending it is not there.

    Vulkan permits flush-to-zero for f32 denormals unless
    shaderDenormPreserveFloat32 is requested; vkml does not request it, so on
    this backend a subnormal may arrive as zero. That is a legitimate platform
    difference, but an UNDOCUMENTED one is a trap -- someone eventually debugs a
    vanishing gradient and finds it here.

    Asserted as a bound, not as an expectation of which way it goes: a device
    that preserves subnormals passes this too, because then the two agree
    exactly.
    """
    tiny = np.array([1e-45, -1e-45, 5e-44, 1.1e-38], dtype=np.float32)

    got = V.relu(V.tensor(tiny, device=gpu_device())).numpy()
    expected = V.relu(V.tensor(tiny, device=V.cpu)).numpy()

    for i, value in enumerate(tiny):
        assert got[i] == expected[i] or abs(float(got[i]) - float(expected[i])) < float(FLT_MIN), (
            f"relu({value!r}): vulkan={got[i]!r} cpu={expected[i]!r} -- a disagreement "
            "larger than the subnormal range is not flush-to-zero, it is a bug"
        )


def test_sign_of_a_subnormal_is_a_known_divergence():
    """The one place flush-to-zero changes a result CATEGORICALLY, not just in scale.

    `sign` branches on `x > 0.0`. A device that flushes subnormals compares the
    input as zero, takes neither branch, and returns the argument unchanged --
    so sign(1e-45) is 1e-45 on the GPU and 1.0 on the CPU. Every other operator
    merely loses a subnormal-sized quantity; this one returns a different KIND
    of answer.

    Recorded here so that it is a known, bounded divergence with a named cause
    rather than a surprise. If a future device preserves subnormals it will
    agree with the CPU and this still passes.
    """
    tiny = np.array([1e-45, -1e-45], dtype=np.float32)

    got = V.sign(V.tensor(tiny, device=gpu_device())).numpy()
    expected = V.sign(V.tensor(tiny, device=V.cpu)).numpy()

    for i, value in enumerate(tiny):
        agrees = got[i] == expected[i]
        flushed = abs(float(got[i])) < float(FLT_MIN)
        assert agrees or flushed, (
            f"sign({value!r}): vulkan={got[i]!r} cpu={expected[i]!r} -- neither agreement "
            "nor flush-to-zero explains this"
        )


def test_tanh_saturates_instead_of_overflowing():
    """Issue #26, pinned directly.

    Kept beside the swept cases because it is the value that was actually
    reported, and a named regression test is what says the report was answered.

    VACUOUS ON RADV, and said here rather than left to be discovered: RADV's own
    tanh already saturates instead of overflowing, so this passes with or without
    the clamp in unary.comp. Its value is on drivers that do not -- which is the
    whole point of the bug. The clamp is checked from the other side by the
    precision test below, which CAN fail on any driver.
    """
    x = np.array([88.0, 89.0, 100.0, 1e10, 3.4028235e38], dtype=np.float32)

    assert np.array_equal(
        V.tanh(V.tensor(x, device=gpu_device())).numpy(), np.ones_like(x)
    ), "tanh must saturate to 1.0 above the exp overflow threshold, not produce NaN"
    assert np.array_equal(
        V.tanh(V.tensor(-x, device=gpu_device())).numpy(), -np.ones_like(x)
    )


def test_tanh_keeps_full_precision_below_saturation():
    """The saturation clamp must not cost accuracy where the built-in was right.

    A clamp placed at 9 rather than 10 would round tanh(9.5) -- still
    distinguishable from 1.0 in f32 on the CPU -- up to 1.0. That is the failure
    mode of fixing an overflow by clamping too early, and a test that only
    checked the saturated tail would never see it.

    Compared against the CPU within the gate rather than bit-for-bit, and NOT
    asserted to be below 1.0: below the clamp the value still comes from the
    driver's built-in, and RADV's already returns exactly 1.0 at 9.5. Asserting
    otherwise would test the driver rather than this clamp.
    """
    # 5.0, 5.5 and 6.0 are the load-bearing samples: they are the last points
    # where tanh is still further from 1.0 than the 1e-5 gate (9.1e-5, 3.3e-5 and
    # 1.2e-5 respectively), so a clamp placed at or below 6 is DETECTABLE here.
    # An earlier version sampled only 0.5, 1, 5, 9, 9.5, 9.9 and could not fail:
    # moving the clamp to 5.0 left every one of those either untouched or already
    # within the gate of 1.0. Above about 6.5 tanh is inside the gate of 1.0
    # anyway, so a clamp there is genuinely undetectable and genuinely harmless.
    x = np.array([0.5, 1.0, 5.0, 5.5, 6.0, 9.0, 9.5, 9.9], dtype=np.float32)

    got = V.tanh(V.tensor(x, device=gpu_device())).numpy()
    expected = V.tanh(V.tensor(x, device=V.cpu)).numpy()

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=float(FLT_MIN))
