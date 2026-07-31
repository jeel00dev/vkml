"""Element-wise operators.

WHAT IS WRITTEN HERE AND WHAT IS NOT. Seven of these -- sign, tanh, sigmoid,
erf, erfc, relu, gelu -- have a dedicated GLSL function carrying a `///` block
that explains the numerical choice behind it. That prose is rendered on the page
straight from `shaders/unary.comp` and is deliberately NOT duplicated here: the
version next to the code is the one that stays true.

The remaining fourteen are a single `case` arm in `apply_op`'s switch, calling a
GLSL built-in directly. There is nothing to extract, so what appears here was
written after reading both the shader arm and the matching `k_*` CPU kernel.
"""
from __future__ import annotations

E: dict[str, dict] = {}


def _simple(name, summary, *, detail=None, note=None, warning=None,
            example=None, see=(), pos_only=False):
    e = {"summary": summary,
         "params": [("input", "Tensor",
                     "Any shape. float32 or float16."
                     + (" Negative values give NaN." if pos_only else ""))],
         "returns": "A tensor of the same shape and dtype."}
    if detail:
        e["detail"] = detail
    if note:
        e["note"] = note
    if warning:
        e["warning"] = warning
    if example:
        e["example"] = example
    if see:
        e["see"] = list(see)
    E[name] = e


_simple("abs", "Absolute value, element-wise.",
        detail="Dispatches GLSL's `abs()` on the GPU and `std::fabs` on the CPU. "
               "`abs(-0.0)` is `+0.0` and `abs(NaN)` is NaN on both.",
        example=""">>> x = vkml.tensor(np.array([-2.0, -0.0, 3.0], dtype=np.float32))
>>> vkml.abs(x).numpy()
array([2., 0., 3.], dtype=float32)""",
        see=["sign", "neg", "square"])

_simple("neg", "Arithmetic negation, element-wise.",
        detail="Compiled as unary minus rather than a multiply by −1, so the sign bit is "
               "flipped and nothing else: `neg(0.0)` is `-0.0`, and `neg(NaN)` keeps the "
               "payload.",
        example=""">>> vkml.neg(vkml.tensor(np.array([1.0, -2.5], dtype=np.float32))).numpy()
array([-1. ,  2.5], dtype=float32)""",
        see=["abs", "sub"])

_simple("exp", "Natural exponential, element-wise.",
        detail="`exp(x)` overflows to `+inf` above about 88.72 in float32 — that bound, "
               "`ln(FLT_MAX)`, is the same one that makes a naive `tanh` return NaN. "
               "Operators built on `exp` here subtract a maximum first for exactly that "
               "reason; see `softmax` and `sigmoid`.",
        note="A subnormal result may be flushed to zero on the GPU. Vulkan permits "
             "flush-to-zero for float32 denormals unless `shaderDenormPreserveFloat32` is "
             "requested, and vkML does not request it — so `exp(-89)` is `0.0` on the GPU "
             "and the subnormal `2.227e-39` on the CPU.",
        example=""">>> vkml.exp(vkml.tensor(np.array([0.0, 1.0, 88.0], dtype=np.float32))).numpy()
array([1.0000000e+00, 2.7182817e+00, 1.6516363e+38], dtype=float32)""",
        see=["log", "sigmoid", "softmax", "tanh"])

_simple("log", "Natural logarithm, element-wise.", pos_only=True,
        detail="`log(0)` is `-inf` and `log(x)` for `x < 0` is NaN, matching IEEE and torch.",
        warning="Taking the log of a softmax output underflows: a probability reaches 0 in "
                "float32 once the logit gap passes about 90, and `log(0)` then poisons every "
                "gradient in the batch. Use `log_softmax`, which never forms the probability.",
        example=""">>> vkml.log(vkml.tensor(np.array([1.0, np.e], dtype=np.float32))).numpy()
array([0.        , 0.99999994], dtype=float32)""",
        see=["exp", "log_softmax", "cross_entropy"])

_simple("sqrt", "Square root, element-wise.", pos_only=True,
        detail="`sqrt(-0.0)` is `-0.0` and `sqrt(x)` for `x < 0` is NaN, both matching IEEE.",
        example=""">>> vkml.sqrt(vkml.tensor(np.array([4.0, 9.0], dtype=np.float32))).numpy()
array([2., 3.], dtype=float32)""",
        see=["rsqrt", "square"])

_simple("rsqrt", "Reciprocal square root, `1/sqrt(x)`, element-wise.", pos_only=True,
        detail="Compiled to GLSL's `inversesqrt()` rather than a divide after a `sqrt`. It is "
               "one instruction on the GPU, and it is the form normalisation layers want — "
               "`rms_norm` and `layer_norm` both scale by an inverse root rather than dividing.",
        example=""">>> vkml.rsqrt(vkml.tensor(np.array([4.0, 16.0], dtype=np.float32))).numpy()
array([0.5 , 0.25], dtype=float32)""",
        see=["sqrt", "reciprocal", "rms_norm"])

_simple("reciprocal", "`1/x`, element-wise.",
        detail="`reciprocal(0.0)` is `+inf` and `reciprocal(-0.0)` is `-inf`, matching IEEE "
               "division rather than raising.",
        example=""">>> vkml.reciprocal(vkml.tensor(np.array([2.0, 4.0], dtype=np.float32))).numpy()
array([0.5 , 0.25], dtype=float32)""",
        see=["div", "rsqrt"])

_simple("square", "`x * x`, element-wise.",
        detail="A multiply, not `pow(x, 2)`: exact for every representable input, one "
               "instruction, and its gradient is `2x` rather than the general power rule.",
        example=""">>> vkml.square(vkml.tensor(np.array([-3.0, 4.0], dtype=np.float32))).numpy()
array([ 9., 16.], dtype=float32)""",
        see=["pow", "sqrt", "mse_loss"])

_simple("sin", "Sine, element-wise, in radians.",
        warning="GLSL does not require `sin`/`cos` to be accurate for large arguments, and "
                "the driver's argument reduction gives up long before FLT_MAX. Measured on "
                "RADV, `sin(3.4e38)` returns 0.0 against −0.522 on the CPU. The suite treats "
                "magnitudes above 1e6 as outside the comparable range; nothing vkML does "
                "changes this, because the reduction happens inside the built-in.",
        example=""">>> vkml.sin(vkml.tensor(np.array([0.0, np.pi / 2], dtype=np.float32))).numpy()
array([0., 1.], dtype=float32)""",
        see=["cos"])

_simple("cos", "Cosine, element-wise, in radians.",
        warning="The same argument-reduction limit as `sin`: not required to be accurate "
                "above roughly 1e6, and the inaccuracy is the driver's, not vkML's.",
        example=""">>> vkml.cos(vkml.tensor(np.array([0.0, np.pi], dtype=np.float32))).numpy()
array([ 1., -1.], dtype=float32)""",
        see=["sin"])

_simple("silu", "SiLU / Swish: `x · sigmoid(x)`, element-wise.",
        detail="Composed in the kernel as a multiply against `sigmoid_op`, so it inherits "
               "sigmoid's two-branch form and its overflow safety rather than repeating them.",
        example=""">>> vkml.silu(vkml.tensor(np.array([-1.0, 0.0, 1.0], dtype=np.float32))).numpy()
array([-0.26894143,  0.        ,  0.7310586 ], dtype=float32)""",
        see=["sigmoid", "gelu", "relu"])

E["clamp"] = {
    "summary": "Limit every element to `[min, max]`.",
    "detail": "Implemented as two nested comparisons, `x < lo ? lo : (x > hi ? hi : x)`, "
              "rather than `min(max(x, lo), hi)`. The bounds arrive as push constants, so "
              "no extra tensor is allocated for them.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("min", "float", "Lower bound, inclusive."),
               ("max", "float", "Upper bound, inclusive.")],
    "returns": "A tensor of the same shape and dtype.",
    "note": "NaN fails both comparisons and falls through unchanged, so it propagates.",
    "example": """>>> x = vkml.tensor(np.array([-3.0, 0.5, 7.0], dtype=np.float32))
>>> vkml.clamp(x, -1.0, 1.0).numpy()
array([-1. ,  0.5,  1. ], dtype=float32)""",
    "see": ["clamp_min", "clamp_max", "maximum", "minimum", "relu"],
}

E["clamp_min"] = {
    "summary": "Raise every element to at least `min`.",
    "detail": "`clamp_min(x, 0)` is `relu` computed a different way, and the two agree on "
              "every input including NaN — which is the point of `relu`'s comparison form.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("min", "float", "Lower bound, inclusive.")],
    "returns": "A tensor of the same shape and dtype.",
    "example": """>>> vkml.clamp_min(vkml.tensor(np.array([-2.0, 3.0], dtype=np.float32)), 0.0).numpy()
array([0., 3.], dtype=float32)""",
    "see": ["clamp", "clamp_max", "relu", "maximum"],
}

E["clamp_max"] = {
    "summary": "Lower every element to at most `max`.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("max", "float", "Upper bound, inclusive.")],
    "returns": "A tensor of the same shape and dtype.",
    "example": """>>> vkml.clamp_max(vkml.tensor(np.array([-2.0, 3.0], dtype=np.float32)), 0.0).numpy()
array([-2.,  0.], dtype=float32)""",
    "see": ["clamp", "clamp_min", "minimum"],
}

# The seven with dedicated GLSL functions get a summary and an example here; the
# reasoning is rendered from the shader itself and is not repeated.
_simple("sign", "The sign of each element: −1, 0 or +1.",
        example=""">>> x = vkml.tensor(np.array([-2.0, -0.0, 0.0, 3.0], dtype=np.float32))
>>> vkml.sign(x).numpy()
array([-1.,  0.,  0.,  1.], dtype=float32)""",
        see=["abs", "relu"])

_simple("erf", "The error function, element-wise.",
        example=""">>> vkml.erf(vkml.tensor(np.array([0.0, 1.0], dtype=np.float32))).numpy()
array([0.       , 0.8427008], dtype=float32)""",
        see=["erfc", "gelu"])

_simple("erfc", "The complementary error function, `1 − erf(x)`, element-wise.",
        detail="Computed directly rather than as `1 − erf(x)`, which cancels catastrophically "
               "in the positive tail — the tail this function exists to serve. `gelu` is built "
               "on it for the same reason.",
        example=""">>> vkml.erfc(vkml.tensor(np.array([0.0, 1.0], dtype=np.float32))).numpy()
array([1.       , 0.1572992], dtype=float32)""",
        see=["erf", "gelu"])

_simple("sigmoid", "The logistic function, `1/(1 + exp(−x))`, element-wise.",
        example=""">>> vkml.sigmoid(vkml.tensor(np.array([-1.0, 0.0, 1.0], dtype=np.float32))).numpy()
array([0.26894143, 0.5       , 0.7310586 ], dtype=float32)""",
        see=["tanh", "silu", "binary_cross_entropy_with_logits"])
