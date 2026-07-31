"""Shape and indexing operators.

Written after reading shaders/cat.comp, shaders/tri.comp,
shaders/index_select.comp, shaders/scatter_add.comp, shaders/im2col.comp,
src/backend/cpu/kernels_movement.cpp and the ops.cpp compositions.
"""
from __future__ import annotations

S: dict[str, dict] = {}

S["cat"] = {
    "summary": "Join tensors along an existing axis.",
    "detail": "The output index is split into `(outer, along, inner)` around the joined axis: "
              "`inner` is the product of the extents after it, `outer` everything before, so "
              "an element's position along the axis is `(i / inner) % extent`. Positions below "
              "the first operand's extent come from it; the rest from the second with the "
              "offset subtracted.\n\n"
              "**Each source index is rebuilt with that source's own axis extent.** Reusing "
              "the output's extent would read past the end of the shorter operand — a trap "
              "documented in both kernels, and the reason they compute it the same way.\n\n"
              "That property is also what brought `cat`'s push-constant block inside the "
              "128-byte guarantee. The operands' extents were 32 bytes restating what the "
              "output's extents plus the joined axis already said, so the shader now "
              "reconstructs them and the block dropped from 144 to 112 bytes.",
    "params": [("tensors", "Sequence[Tensor]", "Tensors to join. All must agree on every axis "
                                               "except the joined one."),
               ("axis", "int", "Axis to join along.")],
    "returns": "A new tensor whose extent along `axis` is the sum of the inputs'.",
    "warning": "The axis packed into the push constants is the index into the **padded** "
               "extent array — `to_gpu_operand` right-pads to rank 4, so a rank-2 tensor's "
               "axis 0 lives at index 2. Packing the tensor-space axis instead would address "
               "the wrong component for every operand of rank below 4, and would do it "
               "silently.",
    "example": """
>>> a = vkml.tensor(np.array([[1.0, 2.0]], dtype=np.float32))
>>> b = vkml.tensor(np.array([[3.0, 4.0]], dtype=np.float32))
>>> vkml.cat([a, b], 0).numpy()
array([[1., 2.],
       [3., 4.]], dtype=float32)
""",
    "see": ["index_select", "where"],
}

S["tril"] = {
    "summary": "Zero everything above the k-th diagonal.",
    "detail": "Keeps the lower triangle. `k=0` is the main diagonal, positive `k` keeps more "
              "above it, negative less.",
    "params": [("input", "Tensor", "Rank 2 or higher; the last two axes are the matrix."),
               ("k", "int = 0", "Diagonal offset.")],
    "returns": "A tensor of the same shape and dtype.",
    "example": """
>>> x = vkml.tensor(np.ones((3, 3), dtype=np.float32))
>>> vkml.tril(x, 0).numpy()
array([[1., 0., 0.],
       [1., 1., 0.],
       [1., 1., 1.]], dtype=float32)
""",
    "see": ["triu", "masked_fill"],
}

S["triu"] = {
    "summary": "Zero everything below the k-th diagonal.",
    "detail": "Keeps the upper triangle. Combined with `masked_fill` this is how causal "
              "attention masks are built — `nn.MultiheadAttention`'s `is_causal` path uses "
              "exactly that pair.",
    "params": [("input", "Tensor", "Rank 2 or higher; the last two axes are the matrix."),
               ("k", "int = 0", "Diagonal offset.")],
    "returns": "A tensor of the same shape and dtype.",
    "example": """
>>> x = vkml.tensor(np.ones((3, 3), dtype=np.float32))
>>> vkml.triu(x, 1).numpy()
array([[0., 1., 1.],
       [0., 0., 1.],
       [0., 0., 0.]], dtype=float32)
""",
    "see": ["tril", "masked_fill", "where"],
}

S["masked_fill"] = {
    "summary": "Replace elements where a bool mask is true with a constant.",
    "detail": "The fill value travels as a push constant, so no tensor is allocated for it — "
              "which is the difference from `where`, where both branches are tensors that "
              "already exist.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("mask", "Tensor", "A bool tensor, broadcastable against `input`."),
               ("value", "float", "What to write where the mask is true.")],
    "returns": "A tensor of the same shape and dtype as `input`.",
    "example": """
>>> x = vkml.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
>>> m = vkml.greater(x, vkml.tensor(np.array([1.5, 1.5, 1.5], dtype=np.float32)))
>>> vkml.masked_fill(x, m, 0.0).numpy()
array([1., 0., 0.], dtype=float32)
""",
    "see": ["where", "triu", "clamp"],
}

S["index_select"] = {
    "summary": "Gather slices along one axis by index.",
    "detail": "The index tensor must be rank 1 and int64. The output takes the index's length "
              "along the selected axis and keeps every other axis unchanged.",
    "params": [("input", "Tensor", "Source tensor."),
               ("axis", "int", "Axis to index along."),
               ("index", "Tensor", "Rank-1 int64 indices.")],
    "returns": "A new tensor with `index.numel()` entries along `axis`.",
    "note": "Its gradient is a `scatter_add` back into a zero tensor, which is why the two "
            "are implemented as a pair and why `scatter_add`'s determinism matters for "
            "training rather than only for inference.",
    "example": """
>>> x = vkml.tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
>>> idx = vkml.tensor(np.array([2, 0], dtype=np.int64))
>>> vkml.index_select(x, 0, idx).numpy()
array([[5., 6.],
       [1., 2.]], dtype=float32)
""",
    "see": ["scatter_add", "cat"],
}

S["scatter_add"] = {
    "summary": "Accumulate rows of a source into a zero tensor at given indices.",
    "detail": "The inverse of `index_select`, and the operation its gradient needs. Several "
              "source rows may target the same destination, so the contributions must be "
              "*added*, not written — which is what makes this irreducible to the element-wise "
              "and reduction operators.\n\n"
              "**Both backends walk the source in ascending linear order**, so for any "
              "destination the contributions arrive in ascending index order. That fixed order "
              "is what makes the result bit-reproducible, and the two backends agree exactly "
              "rather than merely within a tolerance.",
    "params": [("src", "Tensor", "Source rows."),
               ("axis", "int", "Axis to scatter along."),
               ("index", "Tensor", "Rank-1 int64 indices, one per source row."),
               ("dim_size", "int", "Extent of the output along `axis`.")],
    "returns": "A new tensor with `dim_size` entries along `axis`.",
    "warning": "**The GPU has no global float `atomicAdd`** on this hardware, and the ordering "
               "above is the only way to have determinism at all without one. The consequence "
               "is a scan: the kernel is `O(dim_size × index_len)` rather than "
               "`O(index_len)`. Replacing it with a sort-based segmented reduction is tracked "
               "as future work, and any replacement has to preserve the ascending order or it "
               "trades determinism for speed.",
    "example": """
>>> src = vkml.tensor(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
>>> idx = vkml.tensor(np.array([0, 0, 1], dtype=np.int64))
>>> vkml.scatter_add(src, 0, idx, 2).numpy()
array([[3.],
       [3.]], dtype=float32)
""",
    "see": ["index_select"],
}

S["im2col"] = {
    "summary": "Expand sliding windows into columns, the lowering behind `conv2d`.",
    "detail": "`(N, C, H, W)` becomes `(N, C·kh·kw, L)`, where `L` is the number of window "
              "positions. Every element of every window is written out, so a convolution "
              "becomes a single matrix multiply.",
    "params": [("input", "Tensor", "`(N, C, H, W)`."),
               ("kernel", "Sequence[int]", "Window size in `(H, W)`."),
               ("stride", "Sequence[int] = [1, 1]", "Step."),
               ("padding", "Sequence[int] = [0, 0]", "Zero padding on both sides."),
               ("dilation", "Sequence[int] = [1, 1]", "Spacing between window elements.")],
    "returns": "`(N, C·kh·kw, L)`.",
    "warning": "The expansion is **materialised in memory**. A 3×3 kernel makes the "
               "intermediate roughly nine times the input, and that memory is the price "
               "`conv2d` pays for reaching the tuned GEMM path. Implicit GEMM — folding the "
               "addressing into the GEMM's operand load so the expansion is never written — "
               "is the standard fix and is not implemented.",
    "example": """
>>> x = vkml.tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
>>> vkml.im2col(x, [2, 2], [2, 2], [0, 0]).shape
(1, 4, 4)
""",
    "see": ["col2im", "conv2d", "matmul"],
}

S["col2im"] = {
    "summary": "Fold columns back into an image, accumulating overlaps.",
    "detail": "The adjoint of `im2col`, and the operation `conv2d`'s gradient with respect to "
              "its input needs. Where windows overlap, the contributions are **summed** — "
              "which is what makes it the adjoint rather than merely the inverse.",
    "params": [("cols", "Tensor", "`(N, C·kh·kw, L)`."),
               ("image", "Sequence[int]", "Spatial size `(H, W)` to fold back into."),
               ("kernel", "Sequence[int]", "Window size in `(H, W)`."),
               ("stride", "Sequence[int] = [1, 1]", "Step."),
               ("padding", "Sequence[int] = [0, 0]", "Padding used by the forward `im2col`."),
               ("dilation", "Sequence[int] = [1, 1]", "Spacing between window elements.")],
    "returns": "`(N, C, H, W)`.",
    "note": "Because overlaps accumulate, this is one of the few operators where the two "
            "backends agree within a tolerance rather than bit-exactly — the tolerance table "
            "lists it alongside the transcendentals for that reason.",
    "see": ["im2col", "conv2d", "scatter_add"],
}

S["detach"] = {
    "summary": "A tensor sharing the same values but outside the autograd graph.",
    "detail": "Returns a tensor with `requires_grad` false and no recorded history, so a "
              "gradient cannot flow back through it. Used to freeze part of a model, or to "
              "stop a running statistic from being differentiated.",
    "params": [("input", "Tensor", "The tensor to detach.")],
    "returns": "A tensor with the same values and no graph history.",
    "warning": "**`detach` currently forces realization.** A lazily built graph runs at the "
               "point of the call rather than staying deferred, which costs a submission "
               "wherever it appears in a hot loop. Making it lazy is tracked as an "
               "architectural change, not a local fix — it needs the graph to represent "
               "'same values, no history' as a node rather than as an evaluated result.",
    "example": """
>>> x = vkml.tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
>>> vkml.detach(x).requires_grad
False
""",
    "see": ["backward", "realize"],
}
