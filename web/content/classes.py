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
