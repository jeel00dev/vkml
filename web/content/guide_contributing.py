"""Guide: adding an operator.

The file list is derived from commit 103e513, which added `erfc` -- the cheapest
possible operator, a unary function slotting into an existing shader's switch.
Anything harder touches strictly more.
"""
from __future__ import annotations

PAGE = """
<h1>Adding an operator</h1>
<p class="lede">Every file that has to change, derived from the commit that added
<code>erfc</code> — the cheapest operator this project can have.</p>

<h2 id="cost">What it costs</h2>

<p><code>erfc</code> is a unary function that slots into an existing shader's <code>switch</code>.
It needs no new dispatch path, no new push-constant block and no new memory pattern. It touched
<strong>15 files and 145 lines</strong>. Anything harder touches strictly more.</p>

<p>That number is worth knowing before starting, and it is the reason the project's roadmap puts
reducing this cost ahead of adding features.</p>

<h2 id="files">The fifteen files</h2>

<h3 id="identity">Identity — the graph's vocabulary</h3>
<div class="table-scroll">
<table>
<thead><tr><th>File</th><th>What to add</th></tr></thead>
<tbody>
<tr><td><code>include/vkml/graph/op.h</code></td>
    <td>An <code>OpKind</code> enumerator. <strong>Append it</strong> — do not insert. The value
        reaches the shader as a specialisation constant, so renumbering changes what an already
        cached pipeline computes.</td></tr>
<tr><td><code>src/graph/op.cpp</code></td><td>Its name, for errors and the coverage report.</td></tr>
</tbody>
</table>
</div>

<h3 id="api">Public API</h3>
<div class="table-scroll">
<table>
<tbody>
<tr><td><code>include/vkml/api/ops.h</code></td>
    <td>The declaration, and a <code>///</code> block explaining anything non-obvious about it.
        <strong>That block is rendered on this site</strong>, so it is documentation rather than
        a comment.</td></tr>
<tr><td><code>src/api/ops.cpp</code></td>
    <td>Shape and dtype checks, then the node. Compose from existing operators if you can — most
        losses do, and they needed no new kernel on either backend as a result.</td></tr>
</tbody>
</table>
</div>

<h3 id="kernels">Kernels — both backends</h3>
<div class="table-scroll">
<table>
<tbody>
<tr><td><code>src/backend/cpu/kernels_*.cpp</code></td>
    <td>The CPU implementation. <strong>Required.</strong> CPU support must be a superset of
        Vulkan support, and a test enforces it — a GPU-only operator has no oracle.</td></tr>
<tr><td><code>shaders/*.comp</code></td>
    <td>The GLSL. Add an <code>OP_</code> constant and a <code>case</code>, or a new shader if the
        memory pattern differs.</td></tr>
<tr><td><code>src/backend/vulkan/vulkan_backend.cpp</code></td>
    <td>The <code>supports()</code> entry and the dispatch. If the operator cannot run on the GPU,
        leave it out of <code>supports()</code> — it will then raise
        <code>NotImplementedError</code>, which is the designed behaviour rather than a
        gap.</td></tr>
</tbody>
</table>
</div>

<h3 id="grad">Gradient</h3>
<div class="table-scroll">
<table>
<tbody>
<tr><td><code>src/autograd/autograd.cpp</code></td>
    <td>A <code>case</code> building the gradient <strong>from forward operators</strong>. A
        dedicated backward kernel needs an argument for why the gradient genuinely cannot be
        composed — only four operators have earned one, and all four are scatters.</td></tr>
</tbody>
</table>
</div>

<h3 id="python">Python surface</h3>
<div class="table-scroll">
<table>
<tbody>
<tr><td><code>bindings/module.cpp</code></td><td>The nanobind binding.</td></tr>
<tr><td><code>python/vkml/__init__.py</code></td><td>The re-export.</td></tr>
</tbody>
</table>
</div>

<h3 id="tests">Tests and policy</h3>
<div class="table-scroll">
<table>
<tbody>
<tr><td><code>tests/python/tolerance.py</code></td>
    <td>A tolerance entry with a <strong>citable justification</strong>, not a number that makes
        the test pass. Which kind — exact, ULP, relative or backward — is itself the
        argument.</td></tr>
<tr><td><code>tests/python/test_ops_vs_torch.py</code></td><td>Agreement with PyTorch.</td></tr>
<tr><td><code>tests/python/test_vulkan_kernels.py</code></td><td>Agreement with the CPU oracle.</td></tr>
<tr><td><code>tests/python/test_invariants.py</code></td>
    <td>Anything the operator promises that a value comparison cannot express — NaN behaviour, a
        bound, an exactness claim.</td></tr>
<tr><td><code>docs/coverage-baseline.json</code></td>
    <td>Regenerate <strong>only after</strong> confirming the run has no new gaps. Writing a
        baseline from a failing run accepts the regression it was meant to catch.</td></tr>
</tbody>
</table>
</div>

<h2 id="documentation">Documentation</h2>

<p>The site's implementation table, backend chips and cross-links are generated, so a new
operator appears automatically. What is not generated is what it <em>does</em>: add an entry to
the matching file in <code>web/content/</code>, and the build will report the coverage
percentage.</p>

<p>Examples are executed by a gate. <strong>Paste what the interpreter printed</strong> — that
gate's first run found 17 invented outputs, including <code>.shape</code> written as a list when
it returns a tuple.</p>

<h2 id="before-pushing">Before pushing</h2>

<pre><code>ctest --preset release
python -m pytest tests/python -q
VKML_MIN_SPEC=1 python -m pytest tests/python -q

python scripts/check_layering.py
python scripts/check_push_constants.py
python scripts/check_cpu_only_build.py
python scripts/check_docs_examples.py
python scripts/check_docs_references.py
python scripts/check_versions.py
python scripts/coverage_matrix.py</code></pre>

<p><code>docs/PRE-COMMIT-CHECKLIST.md</code> is the authority, and every item on it exists
because something got through without it.</p>

<h2 id="standard">Two habits the project treats as standard</h2>

<ul>
<li><strong>Red-verify.</strong> Break the thing your new test guards and watch it fail. A test
that has never been seen to fail is not yet evidence.</li>
<li><strong>Say what you did not verify.</strong> A commit message here records what was
measured, what was assumed, and what could not be checked — a claim without that is harder to
trust than an admission.</li>
</ul>
"""
