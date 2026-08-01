"""The architecture entry point: one page, three depths.

The Architecture section had seven pages and no way in. A reader who wanted "how
does this work" had to pick one of the seven and hope. This answers the question
at whatever depth they need and hands off to the rest.

The layer count, file count and edge count are NOT written here.
`{{diagram:layer_stack}}` is replaced at build time from the same include scan
`scripts/check_layering.py` uses, so the page cannot describe a structure the
tree does not have.
"""

PAGE = """
<p class="lede">How a Python call becomes a dispatch on the GPU, in three
depths. Read the first section and stop, or keep going &mdash; each one assumes
the one above it and nothing below it.</p>

<h2 id="overview">In thirty seconds</h2>

<p>vkML is a stack of layers, and the dependency direction is <em>enforced</em>
rather than agreed. A layer may include only from a layer below it, so every
arrow in this diagram points down &mdash; if one curved upward, the build would
be failing.</p>

{{diagram:layer_stack}}

<p class="dia-note">Generated from the same include scan that
<code>scripts/check_layering.py</code> runs in CI, so the picture and the gate
cannot disagree. <code>plan</code> is declared in the order and holds no files
yet, so it is drawn dashed rather than hidden.</p>

<h2 id="flow">In five minutes</h2>

<p><strong>Nothing runs when you write it.</strong> An operator builds a graph
node and returns a <code>Tensor</code> handle. No memory is allocated, no kernel
is dispatched, nothing is computed. Work happens only when something observes a
value &mdash; <code>numpy()</code>, <code>item()</code>, a backward pass &mdash;
or when <code>realize()</code> is called explicitly.</p>

<p>That deferral is a performance mechanism, not an API style. Batching lets many
operations share one GPU submission, and on the development hardware a submission
costs about <strong>105&nbsp;&micro;s against 9&nbsp;&micro;s for a
dispatch</strong>. Reducing submissions is worth far more than making any single
kernel faster, which is why the graph exists at all.</p>

<p>When a value is observed, the dispatcher walks the graph in topological order,
asks a backend whether it supports each node, and submits. A graph runs entirely
on <em>one</em> backend: there is no per-node fallback, so an operator the GPU
cannot run is an error rather than a silent host transfer &mdash;
<a href="limitations.html#features">why that is deliberate</a>.</p>

<p><strong>Two backends, and they are not peers.</strong> The CPU backend is the
<em>oracle</em>: its job is to be right, and every Vulkan result is checked
against it. The Vulkan backend is 24 compute shaders compiled to SPIR-V and
dispatched with operand addresses in push constants &mdash; no descriptor sets at
all.</p>

<h2 id="deep">Going deeper</h2>

<p>Each of these traces one subsystem through the code, naming the file and line
of everything it describes.</p>

<div class="cards">
  <a class="card" href="arch-tensor.html"><h3>Tensors, storage and views</h3>
  <p>What a handle owns, how a view shares storage, and why strides are in bytes.</p></a>
  <a class="card" href="arch-graph.html"><h3>The lazy graph and execution</h3>
  <p>Node construction, the topological walk, and how work reaches a backend.</p></a>
  <a class="card" href="arch-autograd.html"><h3>Autograd</h3>
  <p>How the backward pass is built, and which operations carry a rule.</p></a>
  <a class="card" href="arch-cpu.html"><h3>The CPU backend</h3>
  <p>The oracle: pairwise summation, the tolerance policy, and why it is not a
  fallback.</p></a>
  <a class="card" href="arch-vulkan.html"><h3>The Vulkan backend</h3>
  <p>Pipelines, push constants, memory, and the six kernels behind one matmul.</p></a>
  <a class="card" href="arch-shaders.html"><h3>Shaders and the GLSL layer</h3>
  <p>The 24 compute shaders, their specialisation constants and shared helpers.</p></a>
  <a class="card" href="arch-numerics.html"><h3>Dtypes, devices and numerics</h3>
  <p>Five dtypes, software f16 narrowing, and what determinism actually
  guarantees.</p></a>
</div>

<h2 id="verify">How the architecture stays true</h2>

<p>The layer order above is not a diagram of intent. It is checked on every
build, and it has already caught a real violation &mdash; autograd reaching into
api. Alongside it: push-constant blocks against the 128&nbsp;bytes Vulkan
guarantees, the GEMM contraction contract from ADR 0005, and a mutation campaign
that breaks each kernel on purpose to confirm the tests notice.</p>

<p>That last one matters more than it sounds. A green suite proves the tests ran,
not that they can fail &mdash; see
<a href="testing.html">Testing and verification</a>.</p>
"""
