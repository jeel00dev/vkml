"""Class references.

CLASSES is keyed by class name. Each entry describes the type as a whole; the
member table beneath it is generated from the header or the Python source, so
signatures cannot drift.

The `members` mapping documents individual members where there is something to
say. A member with no entry still appears in the generated table with its
signature and a source link — the mapping adds meaning, it does not gate
visibility.
"""
from __future__ import annotations

CLASSES: dict[str, dict] = {}

CLASSES["Tensor"] = {
    "lang": "cpp",
    "summary": "A handle to a value in the computation graph.",
    "detail": "**Cheap to copy.** Two `Tensor`s sharing a node are two names for one value, as "
              "in PyTorch — copying a `Tensor` copies a handle, never the data.\n\n"
              "A `Tensor` is not necessarily computed yet. Operations build graph nodes and "
              "evaluation is deferred until something observes the data (`to_host`, `item`, "
              "`backward`) or `realize()` is called. Eager mode collapses that distinction "
              "while debugging.\n\n"
              "The graph node type is **not visible through the public API**: `Node` is a "
              "forward declaration only, so callers and the binding layer see an opaque "
              "handle and the internal representation can change without breaking either. "
              "That is a recorded guardrail, not an implementation accident.\n\n"
              "**Layout is row-major**: `shape()[0]` is the outermost axis, matching NumPy, "
              "PyTorch and DLPack. `strides()` reports **bytes**, as NumPy does — the DLPack "
              "bridge converts to element strides at the boundary and nowhere else.",
    "groups": [
        ("Construction", ["Tensor", "full", "zeros", "ones", "arange", "from_host"]),
        ("Shape and layout", ["shape", "strides", "size", "ndim", "numel", "dtype", "device",
                              "is_contiguous", "defined"]),
        ("Views — no copy", ["reshape", "permute", "transpose", "squeeze", "unsqueeze",
                             "slice", "broadcast_to"]),
        ("Copies", ["contiguous", "to", "assign_"]),
        ("Observation", ["to_host", "item", "str"]),
        ("Autograd", ["requires_grad", "set_requires_grad", "grad", "set_grad", "realize"]),
        ("Internals", ["node"]),
    ],
    "members": {
        "reshape": "A **view**. Reinterprets the same storage with new extents; the element "
                   "count must be unchanged. No copy, so mutating either handle's storage is "
                   "visible through both.",
        "permute": "A **view**. Reorders axes by permuting the stride vector, so the data is "
                   "untouched and the result is usually non-contiguous.",
        "transpose": "A **view**. `permute` restricted to swapping two axes.",
        "squeeze": "A **view**. Drops an axis of extent 1.",
        "unsqueeze": "A **view**. Inserts an axis of extent 1.",
        "slice": "A **view**. Adjusts the offset and extents, and the stride when `step` is "
                 "greater than 1. The result shares storage with the original.",
        "broadcast_to": "A **view**. Expanded axes are given **stride 0**, so the same element "
                        "is re-read rather than duplicated — which is why broadcasting costs "
                        "no memory anywhere in vkML.",
        "contiguous": "Materialises a contiguous **copy**, and returns `*this` unchanged when "
                      "the tensor already is contiguous — so calling it defensively is free on "
                      "the common path.",
        "to": "Converts dtype, allocating a new tensor.",
        "assign_": "**Overwrites this tensor's storage in place** — the one deliberate escape "
                   "from the otherwise functional graph.\n\n"
                   "It exists for exactly one reason: optimisers must update parameters that "
                   "modules already hold references to. Rebinding a new `Tensor` would leave "
                   "every `Module` still pointing at the old one, so PyTorch mutates in place "
                   "and so does this.\n\n"
                   "**Hazard**: any already-computed node that read this tensor keeps its old "
                   "result, while any node computed afterwards sees the new values. That is "
                   "harmless in the intended use — the training graph is rebuilt each step — "
                   "but `assign_` must not be used mid-graph. Requires matching shape, dtype "
                   "and device, and a contiguous destination.",
        "item": "The value of a single-element tensor. Forces realization and a device-to-host "
                "copy.",
        "to_host": "Copies the contents out to host memory. Forces realization.",
        "realize": "Runs the graph this tensor is the root of. See `realize`.",
        "grad": "The accumulated gradient, or an undefined tensor when none has been computed. "
                "Check `defined()` before using it.",
        "defined": "Whether this handle refers to a value at all. A default-constructed "
                   "`Tensor` and an uncomputed `.grad` are both undefined.",
        "strides": "In **bytes**, not elements, matching NumPy. A stride may be 0, which is "
                   "how broadcasting is represented.",
    },
    "see": ["tensor", "realize", "backward", "detach"],
}


# ------------------------------------------------------------- nn modules --

CLASSES["Module"] = {
    "lang": "python",
    "summary": "Base class for layers: named parameters, buffers and children.",
    "detail": "Written in Python rather than C++ **on purpose**. A `Module` holds references "
              "and iterates dicts, none of which is hot; the hot part is the tensor operators "
              "it calls, and those are already in C++.\n\n"
              "The structure — named children plus named parameters, recursive with a dotted "
              "prefix — exists so parameter names line up with a `state_dict` for loading and "
              "comparison. It mirrors `torch.nn` closely enough that PyTorch code reads across "
              "unchanged.\n\n"
              "**Assigning a `Tensor` attribute makes a parameter**, and assigning a `Module` "
              "makes a child. That is done in `__setattr__`, so `self.weight = ...` registers "
              "without any explicit call. `register_buffer` is the explicit opt-out for state "
              "that must travel with the module but must not be trained.",
    "groups": [
        ("Construction", ["__init__", "register_buffer"]),
        ("Traversal", ["named_parameters", "parameters", "named_buffers", "named_modules"]),
        ("State", ["state_dict", "load_state_dict"]),
        ("Placement and mode", ["to", "train", "eval", "zero_grad"]),
        ("Interface", ["forward", "__call__", "__repr__"]),
        ("Attribute plumbing", ["__setattr__", "__getattr__"]),
    ],
    "members": {
        "register_buffer": "Records persistent state that is **not trained**. A buffer appears "
                           "in `state_dict`, so it saves, loads and interoperates with a torch "
                           "checkpoint — but never in `parameters()`, so an optimiser cannot "
                           "see it. Batch normalisation's running statistics are the "
                           "motivating case: they carry no gradient, and letting an optimiser "
                           "\"train\" them would destroy the estimate. Raises if the tensor "
                           "has `requires_grad`.",
        "state_dict": "Parameters **and buffers**, by dotted name. Buffers are included "
                      "because that is what makes a checkpoint complete: a batch-normalised "
                      "model restored without its running statistics evaluates against the "
                      "wrong distribution while looking perfectly healthy.",
        "load_state_dict": "Copies values in by name, **in place**, preserving each entry's "
                           "device, dtype and `requires_grad`. Every key must match — a "
                           "missing or unexpected one raises rather than being ignored.",
        "to": "Moves every parameter and buffer to a device, in place.\n\n"
              "**Call this before constructing an optimiser.** The optimiser captures the "
              "parameter list when it is built, and `to` replaces each parameter with a new "
              "tensor — so an optimiser made first would keep updating the old ones while the "
              "model used the new. torch has the same ordering constraint for the same "
              "reason.\n\n"
              "Transfer goes through the host, because that is what a transfer to a discrete "
              "device is. Gradients move with their parameters; dropping them would leave a "
              "subsequent optimiser step silently updating nothing.",
        "train": "Sets training mode recursively. Only `Dropout` and `BatchNorm2d` behave "
                 "differently between modes.",
        "eval": "`train(False)`.",
        "zero_grad": "Clears every parameter's gradient by assigning an undefined tensor. "
                     "Required between steps, because `backward` **accumulates**.",
        "named_modules": "Yields `self` first, then children depth-first, so a caller can "
                         "match on the root as well as the leaves.",
        "__setattr__": "Routes a `Tensor` into `_parameters` and a `Module` into `_modules`, "
                       "which is what makes `self.weight = ...` register without a call.",
        "__getattr__": "Only invoked when normal lookup fails, so parameters, buffers and "
                       "children resolve without shadowing real attributes.",
    },
    "see": ["backward", "save_module", "load_module"],
}

CLASSES["Linear"] = {
    "lang": "python",
    "summary": "`y = x @ Wᵀ + b`, matching `torch.nn.Linear`.",
    "detail": "The weight is stored as `(out_features, in_features)` and transposed in "
              "`forward`, exactly as PyTorch does. That layout is not arbitrary — it means a "
              "PyTorch `state_dict` loads **without any transposition**, which keeps the "
              "validation comparison honest.\n\n"
              "**Initialisation** draws `U(−1/√fan_in, +1/√fan_in)` for both weight and bias. "
              "That closed form is exactly torch's default `kaiming_uniform_(w, a=√5)`: with "
              "`a=√5` the gain is `√(2/6)`, so the bound reduces to `1/√fan_in`. Verified "
              "against `torch.nn.Linear` — both give ±0.0357142857 at `fan_in=784`.\n\n"
              "Weights are drawn from a module-level generator seeded by `nn.manual_seed`, not "
              "from `default_rng()` per layer. An unseeded generator would give every run "
              "different weights, making a training result impossible to reproduce and a "
              "divergence impossible to investigate.",
    "note": "`manual_seed` mirrors `torch.manual_seed` in spirit, not in stream — the two "
            "libraries draw from different generators by design, so equal seeds do **not** "
            "give equal weights. To compare against torch, copy a `state_dict` rather than "
            "seeding both.",
    "groups": [("Construction", ["__init__"]),
               ("Forward", ["forward"]),
               ("Internals", ["__setattr__", "__repr__"])],
    "see": ["matmul", "Module"],
}

CLASSES["BatchNorm2d"] = {
    "lang": "python",
    "summary": "Normalise each channel over the batch and spatial axes.",
    "detail": "**Two variance estimators, deliberately.** The batch is normalised with the "
              "*biased* variance (divide by N) while the running estimate accumulates the "
              "*unbiased* one (divide by N−1). That is torch's behaviour, verified, and the "
              "asymmetry is principled: the biased figure is the right normaliser for the "
              "batch in hand, the unbiased one the right estimator of the population.\n\n"
              "Using one for both makes evaluation drift away from training as the running "
              "estimate converges to the wrong value — which a single-step comparison cannot "
              "see, so it is pinned by a test that runs many.\n\n"
              "The running statistics are updated **under `no_grad` and assigned in place**. "
              "They are bookkeeping about the data seen so far, not part of the function being "
              "differentiated, and letting them onto the tape would keep every past batch's "
              "graph alive.",
    "warning": "`num_batches_tracked` exists **and is deliberately not maintained**. Every "
               "torch BatchNorm `state_dict` carries the key, so the buffer must exist or "
               "`load_state_dict` rejects the checkpoint — but nothing in vkML reads it, since "
               "torch uses it only for `momentum=None`, a mode this layer does not offer.\n\n"
               "Keeping it accurate would cost a host round-trip per forward pass: int64 "
               "arithmetic is unimplemented on both backends, so the increment cannot happen "
               "on the device, and reading the counter back is exactly the per-step "
               "synchronisation this project spends effort avoiding.\n\n"
               "**Stated consequence**: a vkML checkpoint loaded into torch reports zero "
               "batches tracked.",
    "note": "A single-sample batch leaves the running estimate untouched rather than dividing "
            "by zero, matching torch.",
    "groups": [("Construction", ["__init__"]),
               ("Forward", ["forward"]),
               ("Internals", ["_update_running_stats", "__repr__"])],
    "see": ["batch_norm", "LayerNorm", "Module"],
}

CLASSES["Dropout"] = {
    "lang": "python",
    "summary": "Zero elements with probability `p` during training, scaling the rest.",
    "detail": "**Advances an offset on every call.** The underlying `rand` is a pure function "
              "of `(seed, offset, index)`, so a module reusing one offset would drop the "
              "*same* elements at every step — silently, while the loss curve still looked "
              "plausible. The counter is what makes successive masks independent, and there is "
              "a test that two consecutive calls differ.\n\n"
              "Seeding from a module-local counter rather than a global stream keeps the whole "
              "thing reproducible: the same seed replays the same run.",
    "note": "`p` must be in `[0, 1)`. `p=0.0` short-circuits to the identity, as does "
            "evaluation mode.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"]),
               ("Internals", ["__repr__"])],
    "see": ["dropout", "rand", "Module"],
}

CLASSES["MultiheadAttention"] = {
    "lang": "python",
    "summary": "Scaled dot-product attention over several heads.",
    "detail": "Parameter layout follows `torch.nn.MultiheadAttention` exactly — a packed "
              "`in_proj_weight` of shape `(3E, E)` plus an `out_proj` submodule — so a torch "
              "`state_dict` loads without rearrangement.\n\n"
              "That is not cosmetic. It is what lets the validation suite compare against "
              "torch's own implementation rather than against a reference written here, which "
              "would only prove the two agree with each other.",
    "warning": "**Two deliberate divergences from torch**, both pinned by tests:\n\n"
               "- `batch_first` defaults to **True**. torch defaults to False, meaning "
               "`(S, B, E)`, a legacy layout almost every caller overrides.\n"
               "- It returns the **output tensor alone**, not `(output, weights)`. The "
               "averaged per-head weights torch returns second are a debugging aid, and a "
               "tuple that is nearly always destructured-and-discarded is worse to use.",
    "note": "`embed_dim` must divide by `num_heads`; the constructor raises otherwise. Causal "
            "masking is built from `triu` plus a comparison, not a dedicated kernel.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"]),
               ("Internals", ["__repr__"])],
    "see": ["softmax", "matmul", "triu", "TransformerEncoderLayer"],
}

CLASSES["TransformerEncoderLayer"] = {
    "lang": "python",
    "summary": "Self-attention followed by a feed-forward block, each residual.",
    "detail": "Parameter names match `torch.nn.TransformerEncoderLayer` — `self_attn`, "
              "`linear1`, `linear2`, `norm1`, `norm2` — so a `state_dict` loads unchanged and "
              "the comparison is against torch's own layer.\n\n"
              "`norm_first` selects pre- or post-normalisation. Post (torch's default) is the "
              "original formulation; pre is what deep stacks use, because normalising inside "
              "the residual branch keeps the gradient path to the input clean.",
    "note": "`activation` accepts `\"relu\"` or `\"gelu\"`; anything else raises at "
            "construction rather than at the first forward pass.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"]),
               ("Internals", ["__repr__"])],
    "see": ["MultiheadAttention", "LayerNorm", "Linear", "gelu"],
}

CLASSES["Sequential"] = {
    "lang": "python",
    "summary": "Chain modules, calling each on the previous one's output.",
    "detail": "Children are registered under their positional index, so a `state_dict` key "
              "reads `0.weight`, `1.bias` and so on — the same convention as torch, which is "
              "what lets a torch `Sequential` checkpoint load here.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"]),
               ("Access", ["__getitem__", "__len__", "__iter__"])],
    "see": ["Module", "Linear"],
}

CLASSES["Embedding"] = {
    "lang": "python",
    "summary": "A lookup table mapping integer indices to dense vectors.",
    "detail": "Implemented as an `index_select` over the weight, so its gradient is a "
              "`scatter_add` back into a zero tensor — which is why `scatter_add`'s "
              "determinism matters for training a model with an embedding, not only for "
              "inference.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"]),
               ("Internals", ["__repr__"])],
    "see": ["index_select", "scatter_add", "Module"],
}

CLASSES["Conv2d"] = {
    "lang": "python",
    "summary": "2-D convolution, matching `torch.nn.Conv2d`.",
    "detail": "Calls `conv2d`, which lowers to `im2col` followed by `matmul` — so every GEMM "
              "improvement reaches this layer for free, and so does the im2col memory "
              "expansion.",
    "warning": "Grouped and depthwise convolution are not supported; the weight's input "
               "channels must equal the input's.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"]),
               ("Internals", ["__repr__"])],
    "see": ["conv2d", "im2col", "matmul", "Module"],
}

CLASSES["LayerNorm"] = {
    "lang": "python",
    "summary": "Layer normalisation with an optional affine transform.",
    "detail": "Calls `layer_norm` for the normalisation, then applies weight and bias itself — "
              "the operator is the normalisation alone, and this layer owns the affine part.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"]),
               ("Internals", ["__repr__"])],
    "see": ["layer_norm", "rms_norm", "BatchNorm2d", "Module"],
}


# The activation and shape wrappers. Each is a one-line `forward` calling the
# operator of the same name, so the entry is short on purpose -- padding it
# would imply there is more happening than there is. The operator page carries
# the numerical detail.
for _name, _op, _what in [
    ("ReLU", "relu", "the rectified linear unit"),
    ("GELU", "gelu", "the Gaussian error linear unit"),
    ("Sigmoid", "sigmoid", "the logistic function"),
    ("Tanh", "tanh", "the hyperbolic tangent"),
]:
    CLASSES[_name] = {
        "lang": "python",
        "summary": f"Apply {_what}, element-wise.",
        "detail": f"A stateless wrapper: `forward` calls `{_op}` and nothing else. It exists "
                  f"so the activation can sit inside a `Sequential`. See `{_op}` for the "
                  f"numerical behaviour, which is where the substance is.",
        "groups": [("Forward", ["forward"])],
        "see": [_op, "Sequential"],
    }

CLASSES["Flatten"] = {
    "lang": "python",
    "summary": "Collapse every axis after the first into one.",
    "detail": "A **view**, not a copy — it reshapes, so no data moves. Typically the join "
              "between a convolutional stack and a linear head.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"])],
    "see": ["Sequential", "Linear"],
}

CLASSES["MaxPool2d"] = {
    "lang": "python",
    "summary": "Maximum over each sliding window, per channel.",
    "detail": "A stateless wrapper over `max_pool2d`, holding only the window geometry.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"]),
               ("Internals", ["__repr__"])],
    "see": ["max_pool2d", "AvgPool2d"],
}

CLASSES["AvgPool2d"] = {
    "lang": "python",
    "summary": "Mean over each sliding window, per channel.",
    "detail": "A stateless wrapper over `avg_pool2d`, holding only the window geometry.",
    "groups": [("Construction", ["__init__"]), ("Forward", ["forward"]),
               ("Internals", ["__repr__"])],
    "see": ["avg_pool2d", "MaxPool2d"],
}


# ------------------------------------------------- optimisers and data ----

CLASSES["Optimizer"] = {
    "lang": "python",
    "summary": "Base class for optimisers: holds a parameter list and clears gradients.",
    "detail": "**Captures the parameter list at construction.** Every optimiser here iterates "
              "that captured list, which is why `model.to(device)` must happen *before* the "
              "optimiser is built — `to` replaces each parameter with a new tensor, and an "
              "optimiser made first would keep updating the old ones while the model used the "
              "new.\n\n"
              "Every `step` runs under `no_grad` and writes through `assign_`. Both matter: "
              "the update is a mutation of the parameters, not part of the function being "
              "differentiated, so recording it would keep step N's graph alive into step N+1; "
              "and rebinding `self.params[i]` instead of assigning in place would update the "
              "optimiser's view and leave the model untouched.",
    "groups": [("Construction", ["__init__"]), ("Interface", ["step", "zero_grad"])],
    "see": ["SGD", "Adam", "AdamW", "RMSProp", "backward"],
}

CLASSES["SGD"] = {
    "lang": "python",
    "summary": "Stochastic gradient descent, with optional momentum and Nesterov look-ahead.",
    "detail": "The update as implemented, in order:\n\n"
              "- `g = grad`, plus `weight_decay · p` when weight decay is set.\n"
              "- With momentum: `v ← v·momentum + g`, and on the first step `v = g` rather "
              "than a zero-initialised buffer.\n"
              "- Classical momentum then steps **along the buffer**: `g ← v`.\n"
              "- **Nesterov** steps along the buffer and then one more momentum-step further — "
              "`g ← g + v·momentum` — which is the look-ahead. It uses the current gradient "
              "*again* rather than replacing it, and that `g` is the gradient after weight "
              "decay, which is what torch feeds in too.\n"
              "- `p ← p − lr·g`, assigned in place.\n\n"
              "State: one velocity tensor per parameter, and only when momentum is non-zero — "
              "so plain SGD costs no extra memory.",
    "groups": [("Construction", ["__init__"]), ("Update", ["step"])],
    "see": ["Optimizer", "Adam", "backward"],
}

CLASSES["Adam"] = {
    "lang": "python",
    "summary": "Adam with bias correction, matching `torch.optim.Adam` defaults.",
    "detail": "The update as implemented:\n\n"
              "- `g = grad`, plus `weight_decay · p` when set — **coupled** decay, added to "
              "the gradient.\n"
              "- `m ← m·β₁ + g·(1−β₁)`, and on the first step `m = g·(1−β₁)` rather than a "
              "zero buffer.\n"
              "- `v ← v·β₂ + g²·(1−β₂)`, likewise.\n"
              "- `p ← p − lr·m̂ / (√v̂ + ε)`, assigned in place.\n\n"
              "**Bias correction is applied to the step size, not to `m` and `v` "
              "individually.** Algebraically identical, one fewer tensor operation per "
              "parameter, and it is what torch does.\n\n"
              "State: two tensors per parameter, so Adam costs **2× the model size** in "
              "optimiser state — worth knowing before choosing it on a device with 5.75 GiB.",
    "groups": [("Construction", ["__init__"]), ("Update", ["step"])],
    "see": ["AdamW", "SGD", "Optimizer"],
}

CLASSES["AdamW"] = {
    "lang": "python",
    "summary": "Adam with **decoupled** weight decay, matching `torch.optim.AdamW`.",
    "detail": "The only difference from `Adam` is *where* the decay is applied, and it is not "
              "cosmetic.\n\n"
              "`Adam` adds `wd·p` to the gradient, so the decay then passes through the "
              "second-moment normalisation and is scaled by `1/√v` — meaning **parameters with "
              "small gradients get decayed far harder than intended**.\n\n"
              "`AdamW` subtracts `lr·wd·p` from the parameter directly, leaving the adaptive "
              "step to act on the gradient alone. Everything else — the moments, the bias "
              "correction, the state cost — is inherited unchanged.",
    "groups": [("Construction", ["__init__"]), ("Update", ["step"])],
    "see": ["Adam", "SGD", "Optimizer"],
}

CLASSES["RMSProp"] = {
    "lang": "python",
    "summary": "RMSProp, matching `torch.optim.RMSprop`.",
    "detail": "Divides the gradient by a running root-mean-square of recent gradients.\n\n"
              "**The running average starts at zero, as torch's does**, so the first step is "
              "`(1−α)·g²` rather than `g²`. That difference persists for many steps through "
              "the exponential average, so it is not a detail — a reimplementation that "
              "initialises from the first gradient diverges from torch for a long time.",
    "groups": [("Construction", ["__init__"]), ("Update", ["step"])],
    "see": ["Adam", "SGD", "Optimizer"],
}

CLASSES["DataLoader"] = {
    "lang": "python",
    "summary": "Iterate a dataset in batches, optionally shuffled.",
    "detail": "Single-process and synchronous: batches are assembled on the calling thread. "
              "Prefetching and worker processes are tracked as future work, not implemented.",
    "note": "Shuffling is seeded, so a run replays. The final batch is smaller when the "
            "dataset size is not a multiple of the batch size — it is **not** dropped, so "
            "every sample is seen exactly once per epoch.",
    "groups": [("Construction", ["__init__"]), ("Iteration", ["__iter__", "__len__"])],
    "see": ["ArrayDataset", "rand"],
}

CLASSES["ArrayDataset"] = {
    "lang": "python",
    "summary": "A dataset over one or more equal-length NumPy arrays.",
    "detail": "Indexing yields a tuple with one element per array, so features and labels stay "
              "aligned by construction rather than by convention.",
    "groups": [("Construction", ["__init__"]), ("Access", ["__len__", "__getitem__"])],
    "see": ["DataLoader"],
}

CLASSES["Checkpoint"] = {
    "lang": "python",
    "summary": "What `load` returns: the arrays, the metadata and the format version.",
    "detail": "Kept as **separate fields rather than one merged mapping**, so a metadata key "
              "can never collide with a tensor name.\n\n"
              "`version` is the checkpoint's `FORMAT_VERSION`, which increments when the "
              "layout changes in a way an older reader would misread. Loading a newer "
              "checkpoint fails with a message naming both versions instead of "
              "misinterpreting the bytes.",
    "groups": [("Fields", ["tensors", "metadata", "version"]),
               ("Internals", ["__repr__"])],
    "see": ["load", "save", "load_module"],
}


# The Python Tensor is a SEPARATE page from the C++ one, not a duplicate of it.
# The two surfaces are not the same class with different syntax: `.size` is the
# element count here and a per-axis extent there, `to()` is dtype-only, and
# `numel`/`size(axis)` are not bound at all. One page covering both would have
# to be wrong about at least one of those.
CLASSES["vkml.Tensor"] = {
    "lang": "native",
    "symbol": "Tensor",
    "extra_members": ("__getitem__",),
    "summary": "The Python handle to a value in the computation graph.",
    "detail": "Defined in C++ and exposed through nanobind, so this page is generated by "
              "introspecting the built extension and every signature below is the one the "
              "installed module actually has. Member links point at "
              "`bindings/module.cpp` — the binding site, which is where the Python name is "
              "chosen and therefore the only place a rename is visible.\n\n"
              "The semantics — lazy evaluation, views sharing storage, cheap handle "
              "copies — are the C++ ones, and are described on the [`Tensor`](class-tensor.html) "
              "page. What follows is the surface as **Python** sees it, and the places where "
              "the two disagree.",
    "warning": "**`.size` is the element count, not the shape**, following NumPy rather than "
               "PyTorch. It is a property, so `x.size()` raises "
               "`TypeError: 'int' object is not callable`, and `x.size(0)` is not available. "
               "The C++ class has a *different* function of the same name — "
               "`Tensor::size(int axis)`, a per-axis extent — which is not bound here. Use "
               "`x.shape` for the shape and `x.shape[0]` for one axis. The rename happens at "
               "`bindings/module.cpp:338`, where Python's `size` is bound to C++ `numel`; "
               "`numel` itself is not a Python name.",
    "note": "**Shape arguments are sequences, not varargs.** `x.reshape([3, 2])` and "
            "`x.view([3, 2])` are correct; the PyTorch spelling `x.reshape(3, 2)` raises "
            "`TypeError`. This applies to `reshape`, `view`, `broadcast_to` and `permute`.\n\n"
            "**`to()` converts dtype, not device.** There is no device-transfer method on "
            "either surface — `Tensor::to` has a single `DType` overload. Place a tensor by "
            "passing `device=` when you create it, as the MNIST example does: "
            "`V.tensor(xb_np, device=device)`. `astype` is bound to the same C++ function as "
            "`to` and is an exact alias.",
    "groups": [
        ("Shape and layout", ["shape", "strides", "size", "ndim", "dtype", "device",
                              "is_contiguous", "defined"]),
        ("Views — no copy", ["reshape", "view", "T", "permute", "transpose", "squeeze",
                             "unsqueeze", "broadcast_to", "__getitem__"]),
        ("Copies and conversion", ["contiguous", "to", "astype", "assign_"]),
        ("Observation", ["numpy", "item"]),
        ("Autograd", ["requires_grad", "grad", "backward", "detach", "realize"]),
        ("Maths, in method form", ["abs", "exp", "log", "sqrt", "matmul", "relu", "gelu",
                                   "sigmoid", "silu", "tanh", "softmax", "log_softmax"]),
        ("Reductions, in method form", ["sum", "mean", "max", "min"]),
    ],
    "members": {
        "size": "The **total number of elements**, as `numpy.ndarray.size` — not the shape, "
                "and not a method. Bound to C++ `numel()`.",
        "shape": "Extents as a `tuple`, outermost axis first. This is the member to reach "
                 "for when porting PyTorch code that calls `x.size()` or `x.size(0)`.",
        "strides": "Strides in **bytes**, as NumPy reports them. A `list`, not a tuple.",
        "T": "A **view** with the last two axes exchanged — `transpose(-2, -1)`. No copy, "
             "and the result is non-contiguous.",
        "view": "A **view**, and an exact alias of `reshape`: both are bound to C++ "
                "`reshape`, so unlike PyTorch there is no contiguity precondition that "
                "distinguishes them. Takes a sequence.",
        "reshape": "A **view**. Reinterprets the same storage with new extents; the element "
                   "count must be unchanged. Takes a sequence, not varargs.",
        "numpy": "Copies to a new `numpy.ndarray`, realizing the graph first. This is the "
                 "principal way data leaves the framework, and it is always a copy — the "
                 "array does not alias tensor storage, so writing to it is safe and has no "
                 "effect on the tensor.",
        "astype": "Converts dtype. Bound to the same C++ function as `to`, so the two are "
                  "interchangeable; `astype` is the NumPy spelling.",
        "to": "Converts **dtype only**. Does not move between devices.",
        "max": "Maximum over `dim`, or over every element when `dim` is `None`. Returns a "
               "`Tensor` — unlike `torch.max`, it does **not** return a `(values, indices)` "
               "pair, and there is no `argmax` on this surface. Equivalent to the free "
               "function `amax`.",
        "min": "Minimum over `dim`, or over every element when `dim` is `None`. Returns a "
               "`Tensor`, with no indices — see `max`. Equivalent to the free function "
               "`amin`.",
        "item": "The single element of a scalar tensor, as a Python `float`. Realizes the "
                "graph.",
        "__getitem__": "Basic slicing, producing a **view**. This is the Python spelling of "
                       "C++ `slice`.",
    },
    "example": """>>> import vkml as V
>>> x = V.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
>>> x.shape
(2, 3)
>>> x.size          # element count, NumPy-style -- not the shape
6
>>> x.ndim
2
>>> x.strides       # BYTES
[12, 4]
>>> x.numpy()
array([[1., 5., 3.],
       [4., 2., 6.]], dtype=float32)
>>> x.T.shape
(3, 2)
>>> x.view([3, 2]).numpy()
array([[1., 5.],
       [3., 4.],
       [2., 6.]], dtype=float32)
>>> x.max(dim=1)
Tensor(shape=(2,), dtype=f32, device=cpu)
>>> x.astype(V.dtype.float16).dtype
dtype.float16""",
    "see": ["Tensor", "tensor", "realize", "backward", "detach"],
}
