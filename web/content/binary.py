"""Arithmetic and comparison operators.

All fourteen share ONE shader, shaders/binary.comp, and one CPU kernel family.
Written after reading that shader in full, kernels_elementwise.cpp, and
include/vkml/core/shape.h for how broadcasting reaches a kernel.
"""
from __future__ import annotations

B: dict[str, dict] = {}

_BROADCAST = ("Shapes are broadcast by NumPy rules. Broadcasting is implemented by giving the "
              "expanded axes **stride 0** rather than by materialising a copy — `Shape` holds "
              "strides in bytes and permits 0, so a broadcast operand re-reads the same "
              "element instead of allocating an expanded one.")

_ONE_SHADER = ("Arithmetic and comparison share a single shader. The operation is a "
               "specialisation constant, so the driver folds the switch away at pipeline "
               "creation and each variant is as tight as a dedicated shader. Comparisons live "
               "there too because everything except the final store is identical — same "
               "broadcast indexing, same operand layout, same bounds check — and only the "
               "destination element type differs, which is also decided at pipeline creation.")


def _arith(name, summary, *, detail=None, note=None, warning=None, example=None, see=()):
    e = {"summary": summary,
         "detail": (detail + "\n\n" if detail else "") + _BROADCAST,
         "params": [("input", "Tensor", "The left operand."),
                    ("other", "Tensor", "The right operand, broadcastable against `input`.")],
         "returns": "A tensor of the broadcast shape."}
    if note:
        e["note"] = note
    if warning:
        e["warning"] = warning
    if example:
        e["example"] = example
    if see:
        e["see"] = list(see)
    B[name] = e


def _cmp(name, summary, *, note=None, example=None, see=()):
    e = {"summary": summary,
         "detail": _BROADCAST + "\n\n" + _ONE_SHADER,
         "params": [("input", "Tensor", "The left operand."),
                    ("other", "Tensor", "The right operand, broadcastable against `input`.")],
         "returns": "A **bool** tensor of the broadcast shape.",
         "note": note or ("Every comparison against NaN is false, including `NaN == NaN`, "
                          "which is IEEE-754 and matches torch. The input dtype and the "
                          "output dtype are tracked separately — a comparison's output is "
                          "always bool, and the shader reads the *input* dtype to decide how "
                          "to load, a distinction the CPU kernel originally got wrong by "
                          "checking the output dtype and reading f32 regardless.")}
    if example:
        e["example"] = example
    if see:
        e["see"] = list(see)
    B[name] = e


# ------------------------------------------------------------- arithmetic --

_arith("add", "Element-wise addition.",
       note="Exact: the two backends agree bit for bit, so the test suite compares them with "
            "no tolerance at all.",
       example=""">>> a = vkml.tensor(np.array([1.0, 2.0], dtype=np.float32))
>>> b = vkml.tensor(np.array([[10.0], [20.0]], dtype=np.float32))
>>> vkml.add(a, b).numpy()
array([[11., 12.],
       [21., 22.]], dtype=float32)""",
       see=["sub", "mul", "maximum"])

_arith("sub", "Element-wise subtraction, `input − other`.",
       note="Exact on both backends.",
       example=""">>> vkml.sub(vkml.tensor(np.array([5.0, 3.0], dtype=np.float32)),
...           vkml.tensor(np.array([2.0, 1.0], dtype=np.float32))).numpy()
array([3., 2.], dtype=float32)""",
       see=["add", "neg"])

_arith("mul", "Element-wise multiplication.",
       note="Exact on both backends.",
       example=""">>> vkml.mul(vkml.tensor(np.array([2.0, 3.0], dtype=np.float32)),
...           vkml.tensor(np.array([4.0, 5.0], dtype=np.float32))).numpy()
array([ 8., 15.], dtype=float32)""",
       see=["div", "add", "square"])

B["scaled_add"] = {
    "summary": "`input * alpha + other * beta`, as one operation.",
    "detail": "Both coefficients are plain numbers and travel in the kernel's push "
              "constants, so nothing is materialised for them.\n\n"
              "The composed form costs FOUR nodes rather than one: a scalar operand "
              "becomes a rank-0 tensor, so `a * 0.9` is a `full` and a `mul`. Measured "
              "on an SGD-with-momentum step over an MNIST MLP, the composed form issued "
              "24 dispatches against 8 here." + "\n\n" + _BROADCAST,
    "params": [("input", "Tensor", "The left operand."),
               ("other", "Tensor", "The right operand, broadcastable against `input`."),
               ("alpha", "float", "Coefficient applied to `input`."),
               ("beta", "float", "Coefficient applied to `other`.")],
    "returns": "A tensor of the broadcast shape.",
    "note": "Bit-identical to `input * alpha + other * beta`, on both backends, checked "
            "byte for byte against the composed form and an independent f32 reference. "
            "It is a cost change, not a numerical one.",
    "example": """>>> a = vkml.tensor(np.array([1.0, 2.0], dtype=np.float32))
>>> b = vkml.tensor(np.array([10.0, 20.0], dtype=np.float32))
>>> vkml.scaled_add(a, b, 0.5, 2.0).numpy()
array([20.5, 41. ], dtype=float32)""",
    "see": ["add", "mul"],
}

_arith("div", "Element-wise division, `input / other`.",
       detail="IEEE division throughout: `x/0` is `±inf` and `0/0` is NaN, rather than "
              "raising.",
       example=""">>> vkml.div(vkml.tensor(np.array([1.0, 0.0], dtype=np.float32)),
...           vkml.tensor(np.array([2.0, 0.0], dtype=np.float32))).numpy()
array([0.5, nan], dtype=float32)""",
       see=["mul", "reciprocal"])

_arith("pow", "Element-wise power, `input ** other`.",
       detail="**Matches `std::pow`, not GLSL's `pow`.** GLSL leaves `pow(x, y)` explicitly "
              "undefined for `x < 0`, while `std::pow` is defined there whenever `y` is an "
              "integer — `std::pow(-2, 3)` is `-8`, not NaN. Leaving the built-in to handle it "
              "would make the two backends disagree on a case the CPU answers perfectly well, "
              "so the shader peels the sign off and reapplies it.",
       example=""">>> vkml.pow(vkml.tensor(np.array([-2.0, 2.0], dtype=np.float32)),
...           vkml.tensor(np.array([3.0, 3.0], dtype=np.float32))).numpy()
array([-8.,  8.], dtype=float32)""",
       see=["square", "sqrt", "exp"])

_arith("maximum", "Element-wise maximum of two tensors.",
       detail="**NaN propagates**, matching `torch.maximum`. GLSL's `max()` is undefined for "
              "NaN and `std::fmax` returns the non-NaN operand, so the shader tests for NaN "
              "explicitly and returns a quiet NaN written from its bit pattern — not `0.0/0.0`, "
              "which the compiler is free to constant-fold or treat as undefined.\n\n"
              "This is tested as *exact*, so the difference is not academic.",
       example=""">>> vkml.maximum(vkml.tensor(np.array([1.0, float('nan')], dtype=np.float32)),
...               vkml.tensor(np.array([2.0, 5.0], dtype=np.float32))).numpy()
array([ 2., nan], dtype=float32)""",
       see=["minimum", "clamp_min", "amax", "relu"])

_arith("minimum", "Element-wise minimum of two tensors.",
       detail="**NaN propagates**, by the same explicit test as `maximum`.",
       example=""">>> vkml.minimum(vkml.tensor(np.array([1.0, float('nan')], dtype=np.float32)),
...               vkml.tensor(np.array([2.0, 5.0], dtype=np.float32))).numpy()
array([ 1., nan], dtype=float32)""",
       see=["maximum", "clamp_max", "amin"])

# ------------------------------------------------------------- comparison --

_cmp("equal", "Element-wise `input == other`.",
     example=""">>> vkml.equal(vkml.tensor(np.array([1.0, 2.0], dtype=np.float32)),
...            vkml.tensor(np.array([1.0, 3.0], dtype=np.float32))).numpy()
array([ True, False])""",
     see=["not_equal", "less", "where"])

_cmp("not_equal", "Element-wise `input != other`.",
     note="`NaN != NaN` is **true** — the only comparison a NaN satisfies, since every other "
          "comparison against NaN is false.",
     example=""">>> vkml.not_equal(vkml.tensor(np.array([1.0, float('nan')], dtype=np.float32)),
...                vkml.tensor(np.array([1.0, float('nan')], dtype=np.float32))).numpy()
array([False,  True])""",
     see=["equal", "where"])

_cmp("less", "Element-wise `input < other`.",
     example=""">>> vkml.less(vkml.tensor(np.array([1.0, 3.0], dtype=np.float32)),
...           vkml.tensor(np.array([2.0, 2.0], dtype=np.float32))).numpy()
array([ True, False])""",
     see=["less_equal", "greater", "where"])

_cmp("less_equal", "Element-wise `input <= other`.",
     example=""">>> vkml.less_equal(vkml.tensor(np.array([2.0, 3.0], dtype=np.float32)),
...                 vkml.tensor(np.array([2.0, 2.0], dtype=np.float32))).numpy()
array([ True, False])""",
     see=["less", "greater_equal", "relu"])

_cmp("greater", "Element-wise `input > other`.",
     example=""">>> vkml.greater(vkml.tensor(np.array([3.0, 1.0], dtype=np.float32)),
...              vkml.tensor(np.array([2.0, 2.0], dtype=np.float32))).numpy()
array([ True, False])""",
     see=["greater_equal", "less", "where"])

_cmp("greater_equal", "Element-wise `input >= other`.",
     example=""">>> vkml.greater_equal(vkml.tensor(np.array([2.0, 1.0], dtype=np.float32)),
...                    vkml.tensor(np.array([2.0, 2.0], dtype=np.float32))).numpy()
array([ True, False])""",
     see=["greater", "less_equal"])

B["where"] = {
    "summary": "Select element-wise from two tensors according to a condition.",
    "detail": "`condition ? a : b`, broadcast across all three operands.\n\n"
              "**Both branches are always evaluated.** That is what element-wise selection "
              "means: `where` chooses between two values that already exist, it does not "
              "avoid computing one. Where a branch would produce `inf` or NaN, the arithmetic "
              "still happens and the result is discarded — `kl_div` relies on exactly this, "
              "computing `log(0) = -inf` and throwing the poisoned product away.\n\n"
              + _BROADCAST,
    "params": [("condition", "Tensor", "A bool tensor."),
               ("input", "Tensor", "Taken where `condition` is true."),
               ("other", "Tensor", "Taken where `condition` is false.")],
    "returns": "A tensor of the broadcast shape, with `input`'s dtype.",
    "note": "Its push-constant block was repacked to fit the 128-byte guarantee by storing "
            "shared extents once instead of per operand — the three operands of a `where` "
            "usually agree on shape, which is what made that possible.",
    "example": """
>>> c = vkml.greater(vkml.tensor(np.array([1.0, 3.0], dtype=np.float32)),
...                  vkml.tensor(np.array([2.0, 2.0], dtype=np.float32)))
>>> vkml.where(c, vkml.tensor(np.array([10.0, 20.0], dtype=np.float32)),
...            vkml.tensor(np.array([-1.0, -2.0], dtype=np.float32))).numpy()
array([-1., 20.], dtype=float32)
""",
    "see": ["greater", "masked_fill", "clamp", "kl_div"],
}
