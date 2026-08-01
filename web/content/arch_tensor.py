"""Architecture: tensors, storage and views.

Written after reading include/vkml/core/{shape,storage,allocator,device,dtype}.h
and include/vkml/api/tensor.h in full, and verifying the rank limit by executing
it rather than trusting the constant.
"""
from __future__ import annotations

PAGE = """
<h1>Tensors, storage and views</h1>
<p class="lede">What a tensor owns, what it merely refers to, and where the boundary between
the two is drawn.</p>

<h2 id="three-types">Three types, one value</h2>

<p>A value in vkML is described by three objects with sharply separated jobs. Keeping them
separate is what makes a view cost nothing and a layering rule enforceable.</p>

<div class="table-scroll">
<table>
<thead><tr><th>Type</th><th>Owns</th><th>Declared in</th></tr></thead>
<tbody>
<tr><td><code>Tensor</code></td><td>A handle to a graph node. Nothing else.</td>
    <td><code>include/vkml/api/tensor.h</code></td></tr>
<tr><td><code>Shape</code></td><td>Extents and strides. No data.</td>
    <td><code>include/vkml/core/shape.h</code></td></tr>
<tr><td><code>Storage</code></td><td>A refcounted block of device memory.</td>
    <td><code>include/vkml/core/storage.h</code></td></tr>
</tbody>
</table>
</div>

<p><code>Tensor</code> is cheap to copy — two of them sharing a node are two names for one
value, as in PyTorch. A <code>Storage</code> is always held through
<code>shared_ptr</code>, because several tensors may view one block and a view must keep the
underlying memory alive. ggml solves the same problem by tracking a
<code>view_src</code> pointer on each tensor.</p>

<h2 id="storage-deleter">Storage does not know how to allocate</h2>

<p>A <code>Storage</code> holds a <strong>deleter supplied by whoever created it</strong>. That
is not a generalisation for its own sake — it is what lets <code>core</code> (layer 1) own the
type while <code>backend/vulkan</code> (layer 4) supplies Vulkan-specific freeing. Without the
inversion, <code>core</code> would have to know about Vulkan, which the layering check
rejects.</p>

<p>The <code>Allocator</code> interface is deliberately minimal: allocate, identify, report a
device. It models <em>"ask this thing for memory"</em> rather than a general allocator
framework, so alignment policies, memory kinds and async free lists can be added to a concrete
allocator without touching the interface.</p>

<div class="admon note"><span class="label">ⓘ Note</span><div class="body">
<p>That seam is not hypothetical. The development GPU exposes only <strong>256 MiB of
host-visible device-local memory</strong> against 5.75 GiB of device-local total, so every
upload goes through a separate staging allocator on the same device. Memory has to be
requestable independently of who computes on it, or that case cannot be expressed.</p>
</div></div>

<p>CPU allocations are aligned to <strong>64 bytes</strong> — one cache line on every CPU this
runs on, and the alignment AVX-512 wants. The CPU backend is a correctness oracle and will not
be hand-vectorised, but aligning costs nothing and removes a variable if the compiler ever
vectorises part of it.</p>

<h2 id="layout">Layout: row-major, strides in bytes</h2>

<p><code>shape()[0]</code> is the outermost axis, matching NumPy, PyTorch and DLPack — so
zero-copy interop needs no axis reversal. ggml reverses the order, which its own documentation
records as a recurring source of confusion for people arriving from PyTorch. The cost of
agreeing with NumPy is that anyone reading ggml kernels alongside vkML kernels must mentally
reverse the indices, and that cost is paid once by the maintainers rather than continuously by
every user of the Python API.</p>

<p><strong>Strides are in bytes</strong>, following ggml and NumPy — <code>ndarray.strides</code>
is also in bytes. Bytes rather than elements makes broadcasting (stride 0) and future
mixed-dtype views expressible without special-casing.</p>

<h2 id="rank-limit">Rank is capped at 4</h2>

<pre><code>&gt;&gt;&gt; vkml.tensor(np.zeros((2, 2, 2, 2, 2), dtype=np.float32))
ShapeError: rank 5 exceeds kMaxDims=4</code></pre>

<p>The reasoning is concrete rather than aesthetic. Every model in scope is rank ≤ 4 — CNNs
are <code>[N, C, H, W]</code>, transformers <code>[B, H, S, D]</code>, RNNs
<code>[T, B, F]</code> — and the push-constant budget decides the rest: three tensors ×
(dims + strides) costs <strong>96 bytes at rank 4 but 192 at rank 8</strong>, which would force
shape metadata into a uniform buffer and add an indirection to every kernel.</p>

<p>Raising it later is a contained change — the constant, plus the push-constant layout — but
it taxes every kernel, so the header records that it should be a deliberate decision rather
than a drift.</p>

<h2 id="views">Which operations alias, and which copy</h2>

<p>This is the distinction that decides whether an operation costs memory, and it is worth
knowing precisely rather than by intuition.</p>

<div class="table-scroll">
<table>
<thead><tr><th>Operation</th><th>Result</th><th>How</th></tr></thead>
<tbody>
<tr><td><code>reshape</code></td><td>view</td><td>New extents over the same storage.</td></tr>
<tr><td><code>permute</code>, <code>transpose</code></td><td>view</td>
    <td>Reorders the stride vector; data untouched. Usually non-contiguous after.</td></tr>
<tr><td><code>squeeze</code>, <code>unsqueeze</code></td><td>view</td>
    <td>Adds or drops an axis of extent 1.</td></tr>
<tr><td><code>slice</code></td><td>view</td>
    <td>Adjusts offset and extents, and the stride when <code>step &gt; 1</code>.</td></tr>
<tr><td><code>broadcast_to</code></td><td>view</td>
    <td><strong>Stride 0</strong> on expanded axes — the same element is re-read.</td></tr>
<tr><td><code>contiguous</code></td><td>copy</td>
    <td>Materialises. Returns <code>*this</code> unchanged when already contiguous.</td></tr>
<tr><td><code>to(dtype)</code></td><td>copy</td><td>Converts, allocating.</td></tr>
<tr><td><code>detach</code></td><td>copy today</td>
    <td>Forces realization; making it lazy is tracked architectural work.</td></tr>
</tbody>
</table>
</div>

<p>Stride-0 broadcasting is why a broadcast costs no memory <em>anywhere</em> in vkML. It is
also why <code>conv2d</code> can reshape a bias to <code>(C_out, 1, 1)</code> and add it across
batch and space without materialising anything.</p>

<h2 id="assign">The one mutation</h2>

<p><code>assign_</code> overwrites a tensor's storage in place, and it is the single deliberate
escape from an otherwise functional graph. It exists for exactly one reason: <strong>optimisers
must update parameters that modules already hold references to</strong>. Rebinding a new
<code>Tensor</code> would leave every <code>Module</code> pointing at the old one, so PyTorch
mutates in place and so does this.</p>

<div class="admon warn"><span class="label">⚠ Warning</span><div class="body">
<p>Any already-computed node that read the tensor keeps its old result, while any node computed
afterwards sees the new values. That is harmless in the intended use — the training graph is
rebuilt every step — but <code>assign_</code> must not be used mid-graph. It requires matching
shape, dtype and device, and a contiguous destination.</p>
</div></div>

<h2 id="opaque">The graph node is not visible</h2>

<p><code>Node</code> is a forward declaration in the public header and nothing more. Callers and
the binding layer see an opaque handle, so the internal representation can change without
breaking either. That is a recorded guardrail, which is why <code>Tensor</code>'s special
members are declared in the header and defined in <code>tensor.cpp</code>, where
<code>Node</code> is complete.</p>
"""
