"""Architecture: the lazy graph and execution.

Written after reading include/vkml/graph/{node,graph,op,grad_mode}.h,
src/graph/graph.cpp, src/dispatch/executor.cpp and the compute() entry point in
src/backend/vulkan/vulkan_backend.cpp.
"""
from __future__ import annotations

PAGE = """
<h1>The lazy graph and execution</h1>
<p class="lede">What happens between writing <code>a + b</code> and a kernel running, traced
through the code that does it.</p>

<h2 id="nothing-runs">Nothing runs when you write it</h2>

<p>An operator builds a <code>Node</code> and returns a <code>Tensor</code> handle to it. No
memory is allocated, no kernel is dispatched, nothing is computed. Work happens only when
something observes a value — <code>to_host()</code>, <code>item()</code>, a backward pass — or
when <code>realize()</code> is called explicitly.</p>

<p>That deferral is a performance mechanism, not an API style. Batching lets many operations
share one GPU submission, and on the development hardware a submission costs about
<strong>105 µs against 9 µs for a dispatch</strong>. Reducing submissions is worth far more
than making any single kernel faster.</p>

<h2 id="node">What a Node holds</h2>

<p>A <code>Node</code> is <strong>immutable once constructed</strong>, with exactly one
exception: its realisation state. That exception is two separate fields for two separate
events — <code>storage</code>, filled in when the node is <em>bound</em>, and a
<code>kFlagComputed</code> bit set when it has been <em>evaluated</em>.</p>

<p>Immutability is not a stylistic preference. It is what makes three things sound:</p>
<ul>
<li>two <code>Tensor</code>s may share a node, so a mutation would be action at a distance;</li>
<li>an optimisation pass may assume a node never changes under it, and can therefore rewrite by
building new nodes rather than patching old ones;</li>
<li>lowering to an execution graph is a pure function of the graph, and hence cacheable.</li>
</ul>

<p>Pass results consequently live in side-tables keyed by position in the topological order,
never as new mutable fields.</p>

<h2 id="ownership">Why sources are shared_ptr, not raw pointers</h2>

<p>The architecture sketch called for <code>std::array&lt;Node*, 4&gt;</code> with nodes
arena-allocated, mirroring ggml — where a context owns every node and raw pointers are safe
because the context outlives them all.</p>

<p>vkML uses <code>shared_ptr</code> instead, because it has no equivalent arena. Graphs here
are built incrementally from Python, node by node, with <strong>no natural scope that owns
them</strong>: a Python-held <code>Tensor</code> must keep its whole producing subgraph alive on
its own. Raw pointers would need either an arena tied to some lifetime — there is nothing to tie
it to — or manual refcounting, which is what <code>shared_ptr</code> already is.</p>

<p>This cannot create reference cycles: the DAG is built strictly bottom-up, so a node's sources
always predate it and can never point back.</p>

<div class="admon note"><span class="label">ⓘ Note</span><div class="body">
<p>The cost was measured rather than assumed: <strong>~410 ns/node to build and ~460 ns/node to
traverse</strong>, against ~1000 ns of actual compute for a modest element-wise operation —
under 1% of step time at current graph sizes. An arena is 20–64× faster on both counts but
cannot express Python's unpredictable object lifetimes without reintroducing refcounting. The
recorded resolution is to lower into a flat arena-backed execution graph later, where planning
and execution get the locality and this layer keeps its safety.</p>
</div></div>

<h2 id="deep-graphs">The destructor is iterative, and has to be</h2>

<p>Default destruction would recurse: destroying a node releases its source
<code>shared_ptr</code>s, which destroys those nodes, which release theirs. On a deep graph that
is a stack overflow — and a deep graph is not hypothetical, since an unrolled RNN over a long
sequence is a chain of order 10⁵ nodes.</p>

<p><code>Node</code>'s destructor therefore tears the source chain down with an explicit
worklist. There is a deep-chain case in <code>tests/cpp/test_graph.cpp</code> that segfaults
without it. It costs nothing for leaves, because the worklist never allocates unless there is
actually a source to release.</p>

<h2 id="views">Views keep two edges, and collapsing the wrong one is a silent bug</h2>

<p>A view node carries <em>two</em> references to what it aliases:</p>

<div class="table-scroll">
<table>
<thead><tr><th>Field</th><th>Points at</th><th>Why</th></tr></thead>
<tbody>
<tr><td><code>view_src</code></td><td>the <strong>root</strong> storage owner</td>
    <td>Collapsed through chains, so binding is one hop.</td></tr>
<tr><td><code>src[0]</code></td><td>the <strong>immediate</strong> base</td>
    <td>Scheduling and autograd both need the real chain.</td></tr>
</tbody>
</table>
</div>

<p>Collapsing <code>src[0]</code> as well is a bug <strong>forward execution cannot see</strong>.
The view's own <code>Shape</code> already encodes the whole transformation, so values come out
right. Backward then breaks, because a gradient rule reads <code>src[0]-&gt;shape</code> to know
what shape to produce a gradient in — for <code>W.transpose().broadcast_to(...)</code> it would
see <code>W</code>'s <code>(out, in)</code> instead of the transposed <code>(in, out)</code> and
emit a correctly-valued but <strong>transposed</strong> gradient. That is exactly how it was
found, through a linear-layer forward-and-backward test.</p>

<h2 id="realize">What realize() does, step by step</h2>

<p>Traced through <code>src/dispatch/executor.cpp</code>:</p>

<ol>
<li><strong>Topological order.</strong> Roots are walked to a flat schedule. An already-realised
node is treated as a leaf and not re-emitted, so nothing runs twice.</li>
<li><strong>One device per graph.</strong> Every node must agree; a graph spanning devices raises
<code>DeviceError</code> naming both.</li>
<li><strong>Support check, per node.</strong> If the backend cannot evaluate an operator, the
error says so and names the remedy.</li>
<li><strong>Coverage recording</strong>, when enabled.</li>
<li><strong>Bind storage.</strong> A view takes its base's storage plus an offset — the
topological order guarantees the base was bound first. Anything computed gets a fresh
allocation, and is asserted contiguous.</li>
<li><strong>One <code>compute()</code> call</strong> with the whole schedule. This is where
batching pays: the backend sees the entire graph, not one node at a time.</li>
<li><strong>Mark computed</strong>, after <code>compute()</code> returns — the only place that
bit is set.</li>
</ol>

<div class="admon warn"><span class="label">⚠ Warning</span><div class="body">
<p><strong>There is no automatic fallback.</strong> When a backend cannot evaluate an operator,
vkML will not split the graph to run it elsewhere. Doing so moves data through host memory at
every split — measured at roughly three times the cost of the arithmetic it carries — and would
do it silently. The error names the explicit remedy instead:</p>
<pre><code>vkml.tensor(t.numpy(), device=vkml.cpu)</code></pre>
</div></div>

<h2 id="bound-vs-computed">Bound is not the same as computed</h2>

<p>These are two separate events with two separate fields, and the split is deliberate. A node is
<strong>bound</strong> when it has memory and <strong>computed</strong> when that memory has been
written. Binding happens during scheduling; the computed bit is set only after
<code>compute()</code> returns.</p>

<p>Merging them would make a node that has been allocated but not yet written indistinguishable
from one holding a value — which is precisely the state every node is in between step 5 and
step 7 above.</p>

<h2 id="eager">Eager mode</h2>

<p><code>set_eager(True)</code>, or <code>VKML_EAGER=1</code>, realizes after every operation. A
failure then surfaces <em>at the operation that caused it</em> rather than at the next
realize — which is the point. It is a debugging aid and it is slower, because every operation
becomes its own submission and the batching above is given up entirely.</p>

<p>The flag is an <code>atomic&lt;bool&gt;</code> read with relaxed ordering, initialised once
from the environment at first use.</p>
"""
