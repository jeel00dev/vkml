"""Loss operators.

Every one of these is COMPOSED from operators that already exist -- none has a
kernel of its own. Written after tracing each composition in src/api/ops.cpp and
reading the /// blocks in include/vkml/api/ops.h.
"""
from __future__ import annotations

LO: dict[str, dict] = {}

LO["cross_entropy"] = {
    "summary": "Cross-entropy between unnormalised logits and integer class targets.",
    "detail": "Takes **logits, not probabilities**. The log-softmax is applied internally; "
              "passing an already-softmaxed tensor applies it twice and produces a quietly "
              "wrong, still-plausible loss.\n\n"
              "Composed rather than kernelled, in one line of `ops.cpp`:\n\n"
              "- `log_softmax(logits, -1)` — never forms the probability, so it cannot "
              "underflow. `log(softmax(x))` reaches `log(0) = -inf` once the logit gap passes "
              "about 90 in float32, and that poisons every gradient in the batch.\n"
              "- multiplied by a **one-hot mask** built from the labels, rather than gathered. "
              "That composes from operators both backends already have, so cross-entropy "
              "needed no new kernel on either.\n"
              "- summed along the class axis, negated, then reduced.\n\n"
              "Exactly one term per row survives the mask, so summing the masked row recovers "
              "that term **exactly** — adding zeros is exact in IEEE-754. The row sum is "
              "therefore not a source of error, which is why the tolerance is inherited from "
              "`log_softmax` rather than widened for a reduction over C.",
    "params": [("input", "Tensor", "Logits, `(N, C)` or `(C,)` for a single sample."),
               ("target", "Tensor", "Class indices, `(N,)`, int64, each in `[0, C)`."),
               ("reduction", "Reduction = mean", "`mean`, `sum` or `none`.")],
    "returns": "A 0-d tensor under `mean` or `sum`; `(N,)` under `none`.",
    "note": "A rank-1 input is lifted with `unsqueeze(0)` so one code path handles the "
            "batched and unbatched cases.",
    "example": """
>>> logits = vkml.tensor(np.random.rand(4, 10).astype(np.float32))
>>> target = vkml.tensor(np.array([1, 0, 4, 9], dtype=np.int64))
>>> vkml.cross_entropy(logits, target).shape
()
""",
    "see": ["log_softmax", "softmax", "binary_cross_entropy_with_logits", "kl_div"],
}

LO["mse_loss"] = {
    "summary": "Mean squared error between two tensors.",
    "detail": "`(input − target)²`, then reduced. Composed from `sub`, `square` and the "
              "reduction — no kernel of its own.",
    "params": [("input", "Tensor", "Predictions."),
               ("target", "Tensor", "Targets, the same shape as `input`."),
               ("reduction", "Reduction = mean", "`mean`, `sum` or `none`.")],
    "returns": "A 0-d tensor under `mean` or `sum`; the elementwise shape under `none`.",
    "example": """
>>> a = vkml.tensor(np.array([1.0, 2.0], dtype=np.float32))
>>> b = vkml.tensor(np.array([1.5, 2.5], dtype=np.float32))
>>> vkml.mse_loss(a, b).item()
0.25
""",
    "see": ["huber_loss", "square"],
}

LO["huber_loss"] = {
    "summary": "Squared error near zero, absolute error beyond `delta`.",
    "detail": "Quadratic inside `delta` and linear outside it — the two agree in **value and "
              "slope** at the join, which is the whole point of the loss: it is less sensitive "
              "to outliers than MSE without the gradient discontinuity of absolute error.\n\n"
              "Composed as `where(|error| < delta, 0.5·error², delta·(|error| − 0.5·delta))`. "
              "Both branches of `where` are evaluated — that is what elementwise selection "
              "means — and both are finite here, so nothing is lost by the discarded one.",
    "params": [("input", "Tensor", "Predictions."),
               ("target", "Tensor", "Targets, the same shape as `input`."),
               ("reduction", "Reduction = mean", "`mean`, `sum` or `none`."),
               ("delta", "float = 1.0", "Where the quadratic region ends. Must be positive.")],
    "returns": "A 0-d tensor under `mean` or `sum`; the elementwise shape under `none`.",
    "note": "A non-positive `delta` raises `ShapeError`, which is the type that maps to "
            "Python's `ValueError` — following `dropout`'s probability check, since a bad "
            "scalar argument is what that is.",
    "example": """
>>> a = vkml.tensor(np.array([0.0, 5.0], dtype=np.float32))
>>> b = vkml.tensor(np.array([0.5, 0.0], dtype=np.float32))
>>> vkml.huber_loss(a, b, vkml.Reduction.none, 1.0).numpy()
array([0.125, 4.5  ], dtype=float32)
""",
    "see": ["mse_loss", "where", "clamp"],
}

LO["kl_div"] = {
    "summary": "Kullback–Leibler divergence between a log-probability input and a target.",
    "detail": "Takes **log-probabilities** as `input`, matching `torch.nn.functional.kl_div`. "
              "Pair it with `log_softmax`, not `softmax`.\n\n"
              "The target may be probabilities or log-probabilities, selected by `log_target`, "
              "and the two branches are genuinely different computations:\n\n"
              "- `log_target=True` — `exp(t)·(t − input)`. Both sides are already logs, so "
              "there is no `log(0)` to guard.\n"
              "- `log_target=False` — `t·(log t − input)`, which is 0 at `t = 0` by the limit "
              "of `t log t`. **The arithmetic gets there as `0 · -inf = NaN` instead**, so the "
              "zero is *selected* rather than computed, with a `where` on `t > 0`. `log(0)` "
              "still happens and still gives `-inf`; it is the multiply that would poison the "
              "result, and that product is the one discarded.",
    "params": [("input", "Tensor", "Log-probabilities."),
               ("target", "Tensor", "Probabilities, or log-probabilities if `log_target`."),
               ("reduction", "Reduction = mean", "`mean`, `sum` or `none`."),
               ("log_target", "bool = False", "Whether `target` is already logged.")],
    "returns": "A 0-d tensor under `mean` or `sum`; the elementwise shape under `none`.",
    "warning": "`mean` divides by the **total element count**, matching `torch`'s `'mean'` "
               "rather than its `'batchmean'`. Torch itself warns that `'mean'` does not match "
               "the mathematical definition and plans to change it; vkML matches torch's "
               "current behaviour, so a comparison against a future torch may diverge here.",
    "see": ["log_softmax", "cross_entropy", "where"],
}

LO["binary_cross_entropy_with_logits"] = {
    "summary": "Binary cross-entropy applied directly to logits.",
    "detail": "Takes **logits, not probabilities** — the sigmoid is folded in. Doing it in one "
              "step is what makes it stable: `log(sigmoid(x))` underflows for large negative "
              "`x` in exactly the way `log(softmax(x))` does, and the fused form never builds "
              "the probability.",
    "params": [("input", "Tensor", "Logits, any shape."),
               ("target", "Tensor", "Targets in `[0, 1]`, the same shape as `input`."),
               ("reduction", "Reduction = mean", "`mean`, `sum` or `none`.")],
    "returns": "A 0-d tensor under `mean` or `sum`; the elementwise shape under `none`.",
    "see": ["sigmoid", "cross_entropy"],
}
