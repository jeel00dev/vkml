"""Guide: testing and verification.

Written after reading docs/TESTING-STRATEGY.md, docs/MEASUREMENT-AUDIT.md and
every script in scripts/, and running each of the gates to confirm what it
reports.
"""
from __future__ import annotations

PAGE = """
<h1>Testing and verification</h1>
<p class="lede">How correctness is established here, and why a green suite is treated as
necessary rather than sufficient.</p>

<h2 id="chain">The correctness chain</h2>

<p>Correctness is a chain of two links, and each one only works because of the other:</p>

<ol>
<li><strong>CPU against PyTorch</strong>, for <em>semantics</em>. Does <code>relu</code> mean
what everyone else means by <code>relu</code>?</li>
<li><strong>Vulkan against the CPU</strong>, for <em>kernel bugs</em>. The CPU backend shares
vkML's exact semantics, so a mismatch here is unambiguously a kernel defect rather than a
difference of convention.</li>
</ol>

<p>The second link is the valuable one, and it requires the first: comparing a GPU kernel
directly against torch would confound a kernel bug with a semantic difference, and the
investigation would start in the wrong place.</p>

<p>It also forces a rule that is enforced by a test rather than by convention: <strong>CPU
support must be a superset of Vulkan support</strong>. That broke once, when widening the Vulkan
<code>supports()</code> gates for float16 made the GPU accept operators the CPU still rejected —
every one of those had a GPU result no oracle could check, and the suite stayed green
throughout.</p>

<h2 id="suites">The two suites</h2>

<pre><code>ctest --preset release                 # C++: 115 cases, 2677 assertions
python -m pytest tests/python -q       # Python + PyTorch: 1413 tests</code></pre>

<p>The Python suite is where operator agreement is checked, because that is where PyTorch is.
The C++ suite covers what has no Python surface — graph ownership, the dispatch grid, shape
arithmetic, the environment helpers.</p>

<h2 id="gates">Gates beyond the suites</h2>

<div class="table-scroll">
<table>
<thead><tr><th>Gate</th><th>What it prevents</th></tr></thead>
<tbody>
<tr><td><code>check_layering.py</code></td>
    <td>A lower layer including a higher one. Has caught a real violation — autograd reaching
        into api.</td></tr>
<tr><td><code>check_push_constants.py</code></td>
    <td>A push-constant block exceeding the 128 bytes Vulkan guarantees. Guards the defect class
        that produced 19 failing tests on a driver reporting exactly 128 and nothing at all on a
        development GPU reporting 256.</td></tr>
<tr><td><code>check_cpu_only_build.py</code></td>
    <td>A test that assumes a GPU. Three CI jobs build CPU-only, so this fails locally rather
        than in three jobs at once.</td></tr>
<tr><td><code>coverage_matrix.py</code></td>
    <td>An operator quietly losing coverage. Compares against a recorded baseline of accepted
        gaps; a new gap fails, a closed one only warns.</td></tr>
<tr><td><code>mutation_check.py</code></td>
    <td>Tests that cannot fail.</td></tr>
<tr><td><code>check_docs_examples.py</code></td>
    <td>A documented example whose output was never run.</td></tr>
<tr><td><code>check_docs_references.py</code></td>
    <td>A cited file, line or constant that no longer exists.</td></tr>
<tr><td><code>check_docs_links.py</code></td>
    <td>A broken internal link or anchor.</td></tr>
<tr><td><code>check_versions.py</code></td>
    <td>A version stated in prose disagreeing with the file that decides it.</td></tr>
</tbody>
</table>
</div>

<h2 id="green-is-not-enough">A green suite proves the tests ran</h2>

<p>It does not prove they <em>can fail</em>. That distinction is not academic here: a green suite
missed a documented list of real bugs, which is why the project's testing strategy exists as a
document.</p>

<p>Two habits follow, and both are treated as standard rather than as extras:</p>

<ul>
<li><strong>Red-verify every new gate.</strong> Break the thing it guards and watch it fail. A
gate that has never been seen to fail is a script.</li>
<li><strong>Check for vacuity.</strong> A probe that reports "0 differences" may be reporting
that it found nothing, or that it ran nothing. One in this project printed nothing because pytest
captures stderr; another compared against instrumentation that had been removed. Both looked like
passes.</li>
</ul>

<h2 id="min-spec">Testing against limits you do not own</h2>

<pre><code>VKML_MIN_SPEC=1 python -m pytest tests/python -q</code></pre>

<p>This makes any device report the Vulkan 1.3 Required Limits. It only ever reports limits
<em>smaller</em> than the hardware has, so it can make vkML more conservative and never less.</p>

<p>It exists because most of this project's portability bugs were the same shape: a limit
asserted against what the development GPU reports rather than what Vulkan guarantees. Push-constant
budgets, workgroup counts and subgroup ranges all failed that way, and every one was invisible
locally and fatal elsewhere.</p>

<div class="admon note"><span class="label">ⓘ Note</span><div class="body">
<p>Run it before claiming a limit is satisfied. It is the cheapest way to find the next instance
of the most common bug this project has.</p>
</div></div>

<h2 id="second-driver">A second driver</h2>

<p>The suite also runs against <strong>lavapipe</strong>, a software Vulkan implementation with a
different SPIR-V compiler, a subgroup size of 8 against RADV's 64, and no real memory hierarchy.
A different <em>driver</em> is where portability actually breaks — not a different device.</p>

<p>It earns its place: it is what surfaced a case where <code>sign</code> returns
<code>-0.0</code> for <code>+0.0</code> and NaN on one Mesa version and <code>+0.0</code> on
another, from a shader whose source reads <code>return 0.0</code>.</p>

<h2 id="measurement">Measuring anything</h2>

<p>The project keeps a separate document listing the instruments that lie, and it is not
optional reading before claiming a speedup. The short version:</p>

<ul>
<li><strong>Minimum, never mean.</strong> Noise is one-sided.</li>
<li><strong>Minimum across <em>process</em> runs</strong>, not iterations within one — pipeline
caches and allocator state persist.</li>
<li><strong>Warm the pipelines</strong>, and say whether validation was on.</li>
<li><strong>Run a same-binary control.</strong> Twice in this project a measured improvement
turned out to be inside the noise floor of an identical binary, and once a padding-only change
"improved" by −6.5%, which no padding can cause.</li>
</ul>
"""
