"""Centralized numerical tolerance policy.

WHY THIS EXISTS
---------------
Individual tests must not choose their own tolerances. Twice during this
project a test "failed" because its tolerance model was wrong rather than
because the code was:

  1. matmul at K=4096 missed a 1e-5 *relative* check while sitting 431x inside
     its backward-error bound -- the dot product was ill-conditioned
     (Sigma|a_i*b_i| = 4051 against a result of 3.17), so relative-to-result was
     meaningless.
  2. Vulkan exp() differed from glibc expf() by 1.14e-5 absolute, which is
     3 ULP at magnitude 46 -- inside the Vulkan specification's own allowance.

Both were the *check* being wrong, and both cost real debugging time. So the
tolerance for an operation is now a property of the operation, derived from a
citable source, and stated once here.

SOURCES
-------
* Vulkan 1.3 specification, "Precision and Operation of SPIR-V Instructions",
  which gives per-instruction ULP allowances for 32-bit floats. These are
  *permissions granted to the driver*, so a conforming implementation may
  legitimately differ from libm by that much.
* IEEE-754: correctly-rounded operations (+, -, *, /, sqrt, fma) are exact to
  0.5 ULP, so anything built only from those must agree bit-for-bit.
* Higham, "Accuracy and Stability of Numerical Algorithms": summation and dot
  products are bounded in BACKWARD error, |computed - exact| <= gamma * sum|terms|,
  with gamma ~ n*eps sequential and ~(B + log2(n/B))*eps for pairwise with base
  block B.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

FP32_EPS = float(np.finfo(np.float32).eps)  # 1.1920929e-07
FP16_EPS = float(np.finfo(np.float16).eps)  # 9.7656250e-04

# Base size of the sequential block in the pairwise summation both backends use.
# Must match kPairwiseBlock in src/backend/cpu/reduce.h.
PAIRWISE_BLOCK = 32


class Kind(Enum):
    """How an operation's error is bounded."""

    EXACT = "exact"
    """Bit-for-bit identical. Only for operations that move or select bits, or
    that are built solely from correctly-rounded IEEE-754 primitives."""

    ULP = "ulp"
    """Within N units in the last place. For transcendentals, where the Vulkan
    spec explicitly permits the driver to differ from a correctly-rounded
    result."""

    RELATIVE = "relative"
    """Within a relative bound. For composites whose error is dominated by a
    small number of rounding steps."""

    BACKWARD = "backward"
    """|computed - exact| <= gamma * sum|terms|. For summation, dot products and
    anything built on them, where the result may be far smaller than the terms
    that produced it."""


@dataclass(frozen=True)
class Tolerance:
    kind: Kind
    ulp: int = 0
    rtol: float = 0.0
    atol: float = 0.0
    note: str = ""


# ---------------------------------------------------------------------------
# The policy.
#
# ULP figures are the Vulkan 1.3 specification's allowances for 32-bit float,
# with a small margin where the spec expresses the bound as a function of the
# input (exp is "3 + 2*|x|" ULP, so a fixed number only holds over a bounded
# domain -- the generators keep inputs inside one).
# ---------------------------------------------------------------------------

POLICY: dict[str, Tolerance] = {
    # -- exact: bit movement or selection only ------------------------------
    "fill": Tolerance(Kind.EXACT, note="writes a constant; no arithmetic"),
    # arange shares fill's kernel (fill is arange with slope 0). The CPU
    # evaluates start + i*step in double and rounds once; the shader has no
    # doubles and rounds twice. Integer parameters -- every use inside the
    # library, including cross_entropy's class indices -- are exact on both.
    "arange": Tolerance(Kind.ULP, ulp=4, note="double vs float evaluation of start + i*step"),
    # Both backends run the identical Philox rounds over the identical
    # counter, all in integer arithmetic, so the bits agree exactly. A
    # tolerance here would hide a divergence in the round structure, which is
    # the only failure mode that matters.
    "rand": Tolerance(Kind.EXACT, note="identical integer rounds on both backends"),
    "dropout": Tolerance(Kind.EXACT, note="a select and one multiply over an exact mask"),
    "copy": Tolerance(Kind.EXACT, note="moves bits"),
    "cat": Tolerance(Kind.EXACT, note="moves bits; no arithmetic"),
    "index_select": Tolerance(Kind.EXACT, note="gathers elements; no arithmetic"),
    "im2col": Tolerance(Kind.EXACT, note="gathers elements or writes zero padding"),
    "max_pool2d": Tolerance(Kind.EXACT, note="selects an existing element"),
    # EXACT for the same reason as col2im: both backends walk the output and
    # pull, so contributions to an input position arrive in the same window
    # order and the two perform an identical sequence of additions.
    "max_pool2d_backward": Tolerance(Kind.EXACT, note="same window-order fold on both backends"),
    # EXACT for the same reason as scatter_add: both backends walk the output
    # and pull, so the contributions to an image position arrive in the same
    # kernel order and the two perform an identical sequence of additions.
    "col2im": Tolerance(Kind.EXACT, note="same kernel-order fold on both backends"),
    # EXACT despite being a sum, and that is a claim about the fold rather
    # than about the magnitudes: both backends accumulate into a destination
    # in ascending index order, so the two perform the identical sequence of
    # IEEE additions. A tolerance here would be weaker than the truth and
    # would stop detecting an ordering change, which is the failure that
    # matters -- see test_scatter_add_is_bit_reproducible.
    "scatter_add": Tolerance(Kind.EXACT, note="same ascending-index fold on both backends"),
    "contiguous": Tolerance(Kind.EXACT, note="moves bits"),
    "identity": Tolerance(Kind.EXACT),
    "relu": Tolerance(Kind.EXACT, note="max(x, 0): a select, not an operation"),
    "neg": Tolerance(Kind.EXACT, note="sign bit flip"),
    "abs": Tolerance(Kind.EXACT, note="sign bit clear"),
    "sign": Tolerance(Kind.EXACT),
    "clamp": Tolerance(Kind.EXACT, note="two selects"),
    "maximum": Tolerance(Kind.EXACT),
    "minimum": Tolerance(Kind.EXACT),
    "where": Tolerance(Kind.EXACT, note="select"),
    "triu": Tolerance(Kind.EXACT, note="keeps an element or writes zero"),
    "tril": Tolerance(Kind.EXACT, note="keeps an element or writes zero"),
    "cast": Tolerance(Kind.EXACT, note="IEEE-754 conversions are correctly rounded"),
    "eq": Tolerance(Kind.EXACT),
    "lt": Tolerance(Kind.EXACT),
    "gt": Tolerance(Kind.EXACT),
    "le": Tolerance(Kind.EXACT),
    "ge": Tolerance(Kind.EXACT),
    "ne": Tolerance(Kind.EXACT),
    "argmax": Tolerance(Kind.EXACT, note="returns an index"),
    "argmin": Tolerance(Kind.EXACT),
    "max": Tolerance(Kind.EXACT, note="selects an existing element"),
    "min": Tolerance(Kind.EXACT),
    # IEEE-754 requires these to be correctly rounded, so a conforming GPU and
    # a conforming CPU must agree exactly.
    "add": Tolerance(Kind.EXACT, note="IEEE-754 correctly rounded"),
    "sub": Tolerance(Kind.EXACT, note="IEEE-754 correctly rounded"),
    "mul": Tolerance(Kind.EXACT, note="IEEE-754 correctly rounded"),
    "square": Tolerance(Kind.EXACT, note="x*x is a single IEEE-754 multiply"),
    # -- ULP: Vulkan spec allowances ----------------------------------------
    "div": Tolerance(Kind.ULP, ulp=3, note="Vulkan: 2.5 ULP for OpFDiv"),
    "sqrt": Tolerance(Kind.ULP, ulp=3, note="Vulkan: 2.5 ULP"),
    "rsqrt": Tolerance(Kind.ULP, ulp=3, note="Vulkan: 2 ULP for InverseSqrt"),
    "exp": Tolerance(Kind.ULP, ulp=8, note="Vulkan: 3 + 2*|x| ULP; inputs bounded to |x|<=3"),
    "log": Tolerance(Kind.ULP, ulp=4, note="Vulkan: 3 ULP outside [0.5, 2]"),
    "sin": Tolerance(Kind.ULP, ulp=0, atol=2.0**-11,
                     note="Vulkan gives an ABSOLUTE bound 2^-11 inside [-pi, pi], not ULP"),
    "cos": Tolerance(Kind.ULP, ulp=0, atol=2.0**-11, note="as sin"),
    # No Vulkan built-in exists, so the GPU evaluates its own approximation:
    # a Maclaurin series below |x|=0.5 and 1-erfc above it, where erfc is the
    # Numerical Recipes 6.2 form with relative error < 1.2e-7. glibc's erf is
    # near-correctly-rounded, so the difference is dominated by our truncation
    # rather than by either library. 1e-6 leaves an order of magnitude of
    # margin over the 1.2e-7 bound.
    "erf": Tolerance(Kind.RELATIVE, rtol=1e-6, atol=1e-7,
                     note="NR 6.2 erfc, rel err < 1.2e-7; series below |x|=0.5"),
    "tanh": Tolerance(Kind.RELATIVE, rtol=1e-5, note="composite of exp and div"),
    "sigmoid": Tolerance(Kind.RELATIVE, rtol=1e-5, note="composite of exp and div"),
    "gelu": Tolerance(Kind.RELATIVE, rtol=1e-5, note="erf-based; libm and GPU erf differ"),
    "silu": Tolerance(Kind.RELATIVE, rtol=1e-5, note="x * sigmoid(x)"),
    "pow": Tolerance(Kind.ULP, ulp=16, note="Vulkan: 16 ULP for OpPow"),
    "reciprocal": Tolerance(Kind.ULP, ulp=3, note="Vulkan: 2.5 ULP for OpFDiv"),
    # -- backward error: reductions and products ----------------------------
    "sum": Tolerance(Kind.BACKWARD, note="pairwise summation"),
    "mean": Tolerance(Kind.BACKWARD, note="pairwise sum then one divide"),
    "prod": Tolerance(Kind.BACKWARD, note="sequential product"),
    "matmul": Tolerance(Kind.BACKWARD, note="pairwise dot product"),
    "softmax": Tolerance(Kind.RELATIVE, rtol=1e-5,
                         note="max-subtract, exp, pairwise sum, divide"),
    "log_softmax": Tolerance(Kind.RELATIVE, rtol=1e-5, atol=1e-5,
                             note="as softmax; atol covers results near zero"),
    # Derived rather than picked. y_i = (x_i - mu)/sqrt(var + eps), and both mu
    # and var are pairwise means carrying backward error gamma_N. Propagating:
    # the absolute floor on y is gamma_N * mean|x| / sigma, and the relative
    # term is gamma_N/2 (sqrt halves it) plus 2 ULP for rsqrt. For x ~ N(0,1)
    # at N = 16384 that is atol 3.9e-6 and rtol 2.7e-6, so these leave 2.6x and
    # 3.7x margin. Both grow only as log2(N) -- the pairwise fold is what keeps
    # this from degrading with width.
    # atol is load-bearing, not decoration: an output element is near zero
    # whenever x_i is near the mean, and a purely relative check is meaningless
    # there. Same reason log_softmax carries one.
    # mse: sub and square are exact, so the error is the mean's backward
    # bound. Every term is non-negative, so sum|terms| == N*result and the
    # backward bound collapses to a RELATIVE gamma_N, about 4.4e-6 at N=1024.
    "mse_loss": Tolerance(Kind.RELATIVE, rtol=1e-5, atol=1e-5,
                          note="mean of non-negative terms; gamma_N relative"),
    # cross_entropy: the masked row sum adds NO error -- exactly one term is
    # non-zero and adding zeros is exact in IEEE-754 -- so the per-sample
    # error is log_softmax's, and the batch mean contributes gamma_N on top.
    "cross_entropy": Tolerance(Kind.RELATIVE, rtol=1e-5, atol=1e-5,
                               note="log_softmax error plus the batch mean's gamma_N"),
    # Same shape as layer_norm, one step shorter: the statistics arrive
    # already computed, so the only error here is the rsqrt and the affine.
    "batch_norm": Tolerance(Kind.RELATIVE, rtol=1e-5, atol=1e-5,
                            note="rsqrt then affine over supplied statistics"),
    "layer_norm": Tolerance(Kind.RELATIVE, rtol=1e-5, atol=1e-5,
                            note="two pairwise means then rsqrt; see derivation above"),
    "rms_norm": Tolerance(Kind.RELATIVE, rtol=1e-5, atol=1e-5,
                          note="one pairwise mean then rsqrt; bound as layer_norm"),
}


def lookup(op: str) -> Tolerance:
    if op not in POLICY:
        raise KeyError(
            f"no tolerance policy for '{op}'. Add one to tests/python/tolerance.py with a "
            f"citable justification rather than picking a number at the call site."
        )
    return POLICY[op]


# ---------------------------------------------------------------------------
# Error measures
# ---------------------------------------------------------------------------


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in representable float32 steps.

    Reinterpreting a float's bits as a sign-magnitude integer makes adjacent
    representable values adjacent integers, so subtracting gives the count of
    steps between them. Negative values are folded so the mapping stays
    monotonic across zero.
    """
    ai = a.astype(np.float32).view(np.int32).astype(np.int64)
    bi = b.astype(np.float32).view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(0x80000000) - ai, ai)
    bi = np.where(bi < 0, np.int64(0x80000000) - bi, bi)
    return np.abs(ai - bi)


def pairwise_gamma(n: int) -> float:
    """Relative error factor for pairwise summation of n terms."""
    if n <= 1:
        return 0.0
    blocks = max(n / PAIRWISE_BLOCK, 1.0)
    return (PAIRWISE_BLOCK + np.log2(blocks)) * FP32_EPS


def backward_error_atol(terms_abs_sum: float, n: int, safety: float = 4.0) -> float:
    """Absolute tolerance for a sum or dot product.

    `terms_abs_sum` is sum|terms| -- for a plain sum that is sum|x_i|, for a dot
    product sum|a_i*b_i|. The safety factor covers the fact that BOTH
    implementations carry error, so their difference can be up to twice either
    bound.
    """
    return float(pairwise_gamma(n) * terms_abs_sum * safety)
