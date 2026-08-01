"""Architecture: autograd.

Written after reading include/vkml/autograd/autograd.h, src/autograd/autograd.cpp
in full (all 47 gradient rules), include/vkml/graph/op.h for the named
exceptions, and include/vkml/graph/grad_mode.h.
"""
from __future__ import annotations

PAGE = """
<h1>Autograd</h1>
<p class="lede">Reverse-mode differentiation built entirely from forward operators — and what
that buys.</p>

<h2 id="central-choice">Every backward rule is a forward operation</h2>

<p><code>d(a·b)/da</code> is <code>mul(grad, b)</code> — an ordinary <code>Mul</code> node
appended to the graph, not a call into a dedicated <code>mul_backward</code> kernel.</p>

<p>This is ggml's model and tinygrad's, and it is the reason the project needs roughly
<strong>64 kernels rather than 120</strong>. Three consequences follow for free:</p>

<ul>
<li>the backward pass reuses the executor, the allocator and every kernel — so a bug fixed in
<code>mul</code> is fixed in the gradient of <code>mul</code>;</li>
<li>higher-order derivatives need no new machinery, because the backward graph is an ordinary
graph that can itself be differentiated;</li>
<li>gradient checkpointing becomes "re-emit that subgraph".</li>
</ul>

<p>The cost is that a fused backward kernel would be perhaps 10–20% faster on some operators.
That is recorded as the right thing to trade away, and individual operators can be fused later
as a pure optimisation with no API change.</p>

<h2 id="exceptions">Four exceptions, each earned</h2>

<p><code>OpKind</code> contains no <code>*_backward</code> entries — with four deliberate
exceptions, which appear as ordinary <em>forward</em> operators because they genuinely cannot be
composed from the element-wise and reduction set:</p>

<div class="table-scroll">
<table>
<thead><tr><th>Operator</th><th>Adjoint of</th><th>Why it cannot be composed</th></tr></thead>
<tbody>
<tr><td><code>ScatterAdd</code></td><td><code>index_select</code></td>
    <td>Several sources may target one destination, so contributions must accumulate.</td></tr>
<tr><td><code>Col2Im</code></td><td><code>im2col</code></td>
    <td>Overlapping windows must sum where they overlap.</td></tr>
<tr><td><code>MaxPool2dBackward</code></td><td><code>max_pool2d</code></td>
    <td>The gradient goes only to the argmax of each window.</td></tr>
<tr><td><code>SliceBackward</code></td><td><code>slice</code></td>
    <td>Scatters into a zero-filled tensor of the original extent — no combination of the
        element-wise or reduction operators expresses that without an index kernel.</td></tr>
</tbody>
</table>
</div>

<p>All four are scatters. That is the shape of thing this design cannot express, and naming them
as forward operators rather than as backward special cases keeps the rule intact: there is one
kind of node, and gradients are built from it.</p>

<h2 id="traversal">How backward() walks the graph</h2>

<ol>
<li><strong>Checks first.</strong> The root must be defined, must require grad, and must be a
floating dtype — an integer root raises <code>DTypeError</code> rather than producing zeros.</li>
<li><strong>Autograd order.</strong> The subgraph reachable from the root is ordered.</li>
<li><strong>Seed.</strong> The root's gradient is the supplied seed; the no-argument form uses
ones and requires a scalar, matching <code>torch.Tensor.backward()</code>.</li>
<li><strong>Reverse iteration.</strong> A node's gradient is complete only once every consumer
has contributed, and consumers all appear later in the order — so walking it backwards is what
makes each gradient final when it is used. Nodes with no gradient entry are unreachable from the
root and skipped; leaves receive but do not propagate.</li>
<li><strong>Deposit into leaves</strong>, <strong>accumulating</strong> rather than replacing.</li>
</ol>

<div class="admon note"><span class="label">ⓘ Note</span><div class="body">
<p>Accumulation is PyTorch's rule and is what makes gradient accumulation across micro-batches
work. It also means a training loop <em>must</em> call <code>zero_grad()</code> between steps —
vkML will not do it, because it cannot tell an intentional accumulation from a forgotten
reset.</p>
</div></div>

<h2 id="who-retains">Which tensors keep a gradient</h2>

<p>Only leaves marked as parameters retain one. Intermediate <code>.grad</code> is dropped,
matching PyTorch, where it is discarded unless explicitly retained. The gradient is realized as
it is deposited, so <code>.grad</code> holds a value rather than an unevaluated graph — which
matters because an optimiser reads every gradient immediately afterwards.</p>

<h2 id="coverage">47 rules of 66 operators</h2>

<p>19 operators have no gradient rule. Calling <code>backward</code> through one raises
<code>NotImplementedError</code> naming the operator rather than silently producing a zero. They
fall into four groups, and the distinction matters — only one of them is a gap:</p>

<div class="table-scroll">
<table>
<thead><tr><th>Group</th><th>Operators</th><th>Status</th></tr></thead>
<tbody>
<tr><td>Leaves and creation</td>
    <td><code>Input</code> <code>Const</code> <code>Full</code> <code>Arange</code>
        <code>Rand</code></td>
    <td>Nothing to propagate to — they have no inputs.</td></tr>
<tr><td>Boolean results</td>
    <td><code>Equal</code> <code>NotEqual</code> <code>Less</code> <code>LessEqual</code>
        <code>Greater</code> <code>GreaterEqual</code></td>
    <td>Not differentiable — the output is not a float.</td></tr>
<tr><td>Discontinuous or integral</td>
    <td><code>ArgMax</code> <code>ArgMin</code> <code>Sign</code></td>
    <td>Derivative is zero almost everywhere and undefined at the steps.</td></tr>
<tr><td>Already a backward</td>
    <td><code>MaxPool2dBackward</code> <code>SliceBackward</code></td>
    <td>Second-order through them is not implemented.</td></tr>
<tr><td><strong>Genuinely missing</strong></td>
    <td><code>Prod</code> <code>Erf</code> <code>Erfc</code></td>
    <td>Differentiable, simply not written.</td></tr>
</tbody>
</table>
</div>

<p>Only the last row is a gap, and a small one: <code>erfc</code> exists so that
<code>gelu</code>'s gradient can be built on it, and the gradient of <code>erfc</code> itself has
no caller yet. <code>prod</code> has no Vulkan kernel either, for the separate numerical reason
described on its own page.</p>

<p>Each operator's page states which case it is in, extracted from the dispatch rather than
listed by hand.</p>

<h2 id="no-grad">Turning recording off</h2>

<p><code>no_grad()</code> suppresses graph recording for its scope. Two places rely on it, and
for the same reason: an update is a mutation of state, not part of the function being
differentiated.</p>

<ul>
<li><strong>Optimiser steps.</strong> Recording the update would keep step N's graph alive into
step N+1.</li>
<li><strong>BatchNorm's running statistics.</strong> They are bookkeeping about the data seen so
far; letting them onto the tape would retain every past batch's graph.</li>
</ul>

<p><code>detach</code> is the per-tensor equivalent: same values, no history.</p>
"""
