"""Why each gap in the surface exists, keyed to what the build extracts.

THE DIVISION OF LABOUR. The build already knows WHAT is true: which operators
run on Vulkan, which OpKinds have a gradient rule, which shader and CPU kernel
back each one, how many tests cover it. None of that is written down here, and
none of it should be -- it is extracted, so it cannot drift.

What the build cannot know is WHY. `has_grad = False` is a fact; whether that
means "not implemented yet", "mathematically undefined" or "this op IS a
gradient" is a judgement, and getting it wrong is worse than silence. A table
generated without this file would state, in a machine-checked-looking format,
that vkML is missing 19 gradient rules. Three of those are real.

So: facts from the code, reasons from here, and `check_docs_references.py`
fails if the two disagree in either direction -- an undeclared gap, or a
declared reason for a gap that no longer exists. The second direction is the
one that matters over time: it is what stops the site claiming something is
unsupported after somebody implemented it.

REASON CATEGORIES. Chosen so a reader can tell "we decided not to" from "we
have not got to it yet", which are very different signals about a project.

    by-design    the question does not apply -- a leaf has no input to
                 differentiate, a comparison returns Bool
    guarantee    supporting it would break a guarantee the project makes, and
                 the guarantee is worth more
    not-yet      wanted, not implemented, nothing blocking it
    measure      deferred until a measurement justifies the work
    architecture blocked on architectural work that is itself tracked
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Gradient rules. 66 OpKinds, 47 with a rule.
#
# Every entry is one of the 19 without, and the extraction is checked against
# this mapping at build time.
# --------------------------------------------------------------------------

GRADIENT_REASONS: dict[str, tuple[str, str]] = {
    # Leaves. There is no input to propagate a gradient to -- these produce a
    # tensor from shape and dtype alone. `requires_grad` on a leaf is where a
    # gradient ACCUMULATES; it is not a rule these ops are missing.
    "Input":  ("by-design", "A leaf. Gradients accumulate here rather than flow through."),
    "Const":  ("by-design", "A constant has no input to differentiate with respect to."),
    "Full":   ("by-design", "Produces a tensor from a shape and a scalar; no tensor input."),
    "Arange": ("by-design", "Produces a tensor from start, stop and step; no tensor input."),
    "Rand":   ("by-design", "Produces a tensor from a seed; no tensor input."),

    # Boolean and integer results. The derivative is zero wherever it is
    # defined and undefined on the boundary, so there is nothing to propagate.
    "Equal":        ("by-design", "Returns Bool. Zero derivative off the boundary, undefined on it."),
    "NotEqual":     ("by-design", "Returns Bool. Zero derivative off the boundary, undefined on it."),
    "Less":         ("by-design", "Returns Bool. Zero derivative off the boundary, undefined on it."),
    "LessEqual":    ("by-design", "Returns Bool. Zero derivative off the boundary, undefined on it."),
    "Greater":      ("by-design", "Returns Bool. Zero derivative off the boundary, undefined on it."),
    "GreaterEqual": ("by-design", "Returns Bool. Zero derivative off the boundary, undefined on it."),
    "ArgMax":       ("by-design", "Returns integer indices, which are not a differentiable function of the input."),
    "ArgMin":       ("by-design", "Returns integer indices, which are not a differentiable function of the input."),
    "Sign":         ("by-design", "Piecewise constant: derivative zero almost everywhere, undefined at zero."),

    # These ARE gradients. Giving them their own rule would mean
    # differentiating a backward pass, which is second-order and not something
    # vkML claims.
    "MaxPool2dBackward": ("by-design", "Already a backward op. A rule for it would be a second derivative."),
    "SliceBackward":     ("by-design", "Already a backward op. A rule for it would be a second derivative."),

    # The three that are genuinely absent.
    "Erf":  ("not-yet", "The forward op exists on both backends. The rule is "
                        "d/dx erf(x) = 2/sqrt(pi) * exp(-x^2) and nothing blocks it."),
    "Erfc": ("not-yet", "As `Erf`, negated. `gelu` reaches its gradient through a "
                        "different path, so nothing on a training path is waiting on this."),
    "Prod": ("not-yet", "Reachable as prod(x) * sum(grad / x), which needs care where "
                        "x contains a zero. `prod` is CPU-only for a separate reason -- "
                        "see the backend table."),
}

# --------------------------------------------------------------------------
# Backend coverage. Extracted per operator; reasons for anything CPU-only.
# --------------------------------------------------------------------------

BACKEND_REASONS: dict[str, tuple[str, str]] = {
    "prod": ("guarantee",
             "A parallel reduction reorders the fold, and for a product that is a "
             "different answer rather than a rounding difference: multiplying 1e20 "
             "and 1e-20 alternately gives 1.0 in the CPU's index order and inf once "
             "the large values are grouped, which lane-striding does immediately "
             "(measured). Matching the CPU would mean one lane multiplying in index "
             "order -- a kernel with no parallelism, slower than the CPU at the only "
             "thing it would be correct for. The CPU backend is the oracle the GPU is "
             "checked against, so a GPU `prod` that legitimately disagreed would break "
             "that chain rather than extend it. Nothing in nn, the losses or the "
             "optimisers calls it."),
}

# --------------------------------------------------------------------------
# Everything else a reader might reasonably expect and not find. Unlike the two
# tables above these are not extracted -- there is no code to point at for a
# feature that does not exist -- so each carries its own evidence.
# --------------------------------------------------------------------------

FEATURE_NOTES: list[dict] = [
    {
        "title": "Mixed CPU and Vulkan execution in one graph",
        "reason": "by-design",
        "text": "A graph runs entirely on one backend. There is no automatic "
                "per-node fallback, so an operator the GPU cannot run is an error "
                "rather than a silent transfer. Splitting a graph across devices "
                "means inserting transfers the author did not write, and a "
                "transfer is the most expensive thing in this system -- a "
                "submission costs about 105us against 9us for a dispatch. The "
                "decision and the alternatives are recorded in ADR 0008.",
        "see": ("docs/adr/0008-backend-selection-and-cpu-fallback.md", None),
    },
    {
        "title": "Tensors of rank above 4",
        "reason": "by-design",
        "text": "`kMaxDims` is 4. Every push-constant block carries extents and "
                "strides inline, and Vulkan guarantees only 128 bytes of push "
                "constants -- the budget that rank cap buys is what keeps every "
                "shader inside the guaranteed minimum on any conformant device.",
        "see": ("docs/adr/0009-operand-metadata-out-of-push-constants.md", None),
    },
    {
        "title": "float64",
        "reason": "by-design",
        "text": "The dtypes are f32, f16, i32, i64 and bool. Double-precision "
                "compute is optional in Vulkan and absent on most consumer GPUs, "
                "so supporting it would mean a CPU-only dtype -- which is the "
                "backend divergence the oracle design exists to avoid.",
        "see": None,
    },
    {
        "title": "Distributed or multi-GPU training",
        "reason": "not-yet",
        "text": "One device per graph today. The device is already a first-class "
                "part of every tensor and the allocator is per-device, so the "
                "groundwork is there, but nothing coordinates two of them.",
        "see": None,
    },
    {
        "title": "Operator fusion",
        "reason": "measure",
        "text": "`layer_norm` and `rms_norm` are composed from smaller operators "
                "rather than fused. Fusing them would save bandwidth, not "
                "accuracy, and the measurement that would justify it has not been "
                "taken -- on the profile so far the time goes to submission "
                "overhead rather than to kernel bandwidth, so fusing first would "
                "be optimising the part that is not the constraint.",
        "see": None,
    },
    {
        "title": "A lazy `detach()`",
        "reason": "architecture",
        "text": "`detach()` shares its source's buffer, so an unrealized source "
                "has nothing to share and the call forces evaluation -- which cuts "
                "the graph. Every optimiser calls it on its intermediates, so this "
                "caps how much work any batching can combine. Measured: a "
                "prototype SGD that builds all updates lazily still submits 13 "
                "times rather than the predicted 9. Fixing it is a change to "
                "autograd, not to the optimiser.",
        "see": ("docs/adr/0006-lazy-assign-and-submission-batching.md", None),
    },
]

REASON_LABELS = {
    "by-design":    ("By design", "The question does not apply, or supporting it would cost more than it gives."),
    "guarantee":    ("Would break a guarantee", "Possible, but not without giving up something the project promises."),
    "not-yet":      ("Not yet implemented", "Wanted; nothing blocking it."),
    "measure":      ("Waiting on a measurement", "Deferred until a number justifies the work."),
    "architecture": ("Waiting on architectural work", "Blocked on a change tracked elsewhere."),
}
