"""Reduction and creation operators.

Written after reading shaders/reduce.comp, src/backend/cpu/kernels_reduce.cpp,
src/backend/cpu/philox.h and iterate.h. Where the source states a reason -- and
for this family it usually does -- the reason is repeated here rather than
paraphrased into something vaguer.
"""
from __future__ import annotations

R: dict[str, dict] = {}

# ------------------------------------------------------------- reductions --

R["sum"] = {
    "summary": "Sum every element of a tensor.",
    "detail": "Reduces to a 0-d tensor.\n\n"
              "The fold is **pairwise**, not sequential: `pairwise_sum` in "
              "`src/backend/cpu/iterate.h` recurses until a run is at most "
              "`kPairwiseBlock = 32` elements and only then adds in order. The error grows "
              "as `O(log n)` in the element count rather than `O(n)`, which is what keeps a "
              "large reduction usable in float32 at all.\n\n"
              "On the GPU the same shape is achieved differently: each invocation folds its "
              "own strided slice, then the workgroup combines through a shared-memory tree. "
              "Both trees are determined by the tensor's shape, never by which invocation "
              "finished first.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A 0-d tensor holding the total.",
    "note": "Bit-identical across runs on the same device, and that is a consequence of the "
            "fixed tree rather than a coincidence — an atomic-accumulation reduction would "
            "give a different answer each run, which is why vkML does not use one.",
    "example": """
>>> x = vkml.tensor(np.ones((1000,), dtype=np.float32))
>>> float(vkml.sum(x).item())
1000.0
""",
    "see": ["mean", "prod", "amax", "amin"],
}

R["mean"] = {
    "summary": "Arithmetic mean of every element.",
    "detail": "Computed as `pairwise_sum(x) / n` — the division happens once, at the end, "
              "rather than accumulating `x/n` per element. Dividing first would lose the "
              "low bits of every term before adding them.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A 0-d tensor holding the mean.",
    "example": """
>>> vkml.mean(vkml.tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))).item()
2.5
""",
    "see": ["sum", "amax"],
}

R["prod"] = {
    "summary": "The product of every element.",
    "detail": "Folded **sequentially, in index order**, and deliberately so. Every other "
              "reduction here folds pairwise because that improves a sum's error bound. A "
              "product gains nothing from it — relative errors compose multiplicatively "
              "whatever the order — and reordering costs something real: it changes when the "
              "fold overflows.\n\n"
              "Multiplying `1e20` and `1e-20` alternately stays at `1.0` in index order and "
              "reaches `inf` if the large values are grouped together first.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A 0-d tensor holding the product.",
    "warning": "**CPU only, and the reason is numerical rather than an omission.** A GPU "
               "reduction is a tree, and a tree reassociates the fold — which is exactly what "
               "changes the overflow point. Rather than ship a kernel that disagrees with the "
               "oracle on inputs like the one above, `prod` raises `NotImplementedError` on a "
               "Vulkan tensor. The rationale is recorded above `k_prod` in "
               "`kernels_reduce.cpp`.",
    "note": "It also has no gradient rule, so `backward` through a `prod` raises.",
    "example": """
>>> vkml.prod(vkml.tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))).item()
24.0
""",
    "see": ["sum", "mean"],
}

R["amax"] = {
    "summary": "The largest element of a tensor.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A 0-d tensor holding the maximum.",
    "warning": "**NaN propagates**, matching `torch.amax`. The check is written twice in the "
               "Vulkan kernel — once in the per-invocation fold and once in the shared-memory "
               "tree — because a reduction built only from `>` comparisons drops NaN at "
               "whichever stage it is missing, and a NaN silently disappearing from a "
               "reduction hides a diverged model.",
    "example": """
>>> x = vkml.tensor(np.array([1.0, float('nan'), 3.0], dtype=np.float32))
>>> vkml.amax(x).item()
nan
""",
    "see": ["amin", "argmax", "maximum", "sum"],
}

R["amin"] = {
    "summary": "The smallest element of a tensor.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A 0-d tensor holding the minimum.",
    "warning": "**NaN propagates**, checked in both fold stages, exactly as for `amax`.",
    "example": """
>>> vkml.amin(vkml.tensor(np.array([3.0, 1.0, 2.0], dtype=np.float32))).item()
1.0
""",
    "see": ["amax", "argmin", "minimum"],
}

R["argmax"] = {
    "summary": "The index of the largest element along an axis.",
    "detail": "Ties keep the **first** maximum. The comparison is a strict `>`, which the CPU "
              "kernel notes is what `torch.argmax` documents — `>=` would silently return the "
              "last one instead.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("dim", "int", "Axis to search along."),
               ("keepdim", "bool = False", "Keep the reduced axis with extent 1.")],
    "returns": "An int64 tensor of indices.",
    "example": """
>>> x = vkml.tensor(np.array([[1.0, 5.0, 5.0]], dtype=np.float32))
>>> vkml.argmax(x, 1).numpy()
array([1], dtype=int64)
""",
    "see": ["amax", "argmin"],
}

R["argmin"] = {
    "summary": "The index of the smallest element along an axis.",
    "detail": "Ties keep the **first** minimum, mirroring `argmax`.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("dim", "int", "Axis to search along."),
               ("keepdim", "bool = False", "Keep the reduced axis with extent 1.")],
    "returns": "An int64 tensor of indices.",
    "example": """
>>> x = vkml.tensor(np.array([[3.0, 1.0, 1.0]], dtype=np.float32))
>>> vkml.argmin(x, 1).numpy()
array([1], dtype=int64)
""",
    "see": ["amin", "argmax"],
}

# --------------------------------------------------------------- creation --

R["ones"] = {
    "summary": "Create a tensor of the given shape filled with ones.",
    "params": [("shape", "Sequence[int]", "Extent of each axis."),
               ("dtype", "dtype = float32", "Element type."),
               ("device", "device = cpu", "Where to allocate.")],
    "returns": "A new tensor of `shape`, every element one.",
    "example": """
>>> vkml.ones([2, 2]).numpy()
array([[1., 1.],
       [1., 1.]], dtype=float32)
""",
    "see": ["zeros", "full", "tensor"],
}

R["full"] = {
    "summary": "Create a tensor of the given shape filled with one value.",
    "detail": "The fill value travels as a push constant rather than as a staged buffer, so "
              "no host-to-device copy is involved on the GPU path.",
    "params": [("shape", "Sequence[int]", "Extent of each axis."),
               ("value", "float", "The value every element takes."),
               ("dtype", "dtype = float32", "Element type."),
               ("device", "device = cpu", "Where to allocate.")],
    "returns": "A new tensor of `shape`.",
    "example": """
>>> vkml.full([2, 2], 2.5).numpy()
array([[2.5, 2.5],
       [2.5, 2.5]], dtype=float32)
""",
    "see": ["zeros", "ones"],
}

R["rand"] = {
    "summary": "Uniform random values in `[0, 1)`, from a counter-based generator.",
    "detail": "Uses **Philox4x32-10** (`src/backend/cpu/philox.h`), a counter-based generator "
              "rather than a stateful one. The value at each index is a pure function of "
              "`(seed, offset, index)`, so it does not matter which invocation computes which "
              "element or in what order — the GPU produces the same draw as the CPU without "
              "any sequencing between threads.\n\n"
              "That is why the signature takes a seed and an offset rather than carrying "
              "hidden state: the offset is how you advance the stream between calls.\n\n"
              "Each value takes the top 24 bits of a 32-bit output, which is the float "
              "significand's width, so every result is exactly representable.",
    "params": [("shape", "Sequence[int]", "Extent of each axis."),
               ("seed", "int", "Identifies the stream."),
               ("offset", "int = 0", "Position within the stream. Advance it between draws."),
               ("device", "device = cpu", "Where to allocate.")],
    "returns": "A new tensor of `shape` with values in `[0, 1)`.",
    "note": "Same `(seed, offset, shape)` gives the same values on every device and every run "
            "— this is part of the determinism contract, not a convenience.",
    "example": """
>>> a = vkml.rand([4], 42, 0).numpy()
>>> b = vkml.rand([4], 42, 0).numpy()
>>> bool((a == b).all())
True
""",
    "see": ["zeros", "dropout"],
}

R["from_numpy"] = {
    "summary": "Create a CPU tensor from a NumPy array.",
    "detail": "The data is copied. Use `tensor` when a device or `requires_grad` is wanted.",
    "params": [("array", "numpy.ndarray", "Values to copy.")],
    "returns": "A new CPU tensor.",
    "example": """
>>> vkml.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).numpy()
array([1., 2.], dtype=float32)
""",
    "see": ["tensor", "asarray"],
}

R["asarray"] = {
    "summary": "Copy a tensor's contents out into a NumPy array.",
    "detail": "The inverse direction to `from_numpy`. Forces realization: a lazily built "
              "graph has to run before there are values to copy.",
    "params": [("tensor", "Tensor", "The tensor to read.")],
    "returns": "A new NumPy array holding a copy.",
    "example": """
>>> vkml.asarray(vkml.ones([3]))
array([1., 1., 1.], dtype=float32)
""",
    "see": ["from_numpy", "tensor"],
}
