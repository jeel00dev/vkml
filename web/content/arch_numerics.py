"""Architecture: dtypes, devices and numerics.

Written after reading include/vkml/core/{dtype,device}.h, shaders/common.glsl's
conversion helpers, tests/python/tolerance.py in full, and ARCHITECTURE.md 7.3.
The dtype support table is generated from the tolerance policy and the backend's
supports() rather than typed.
"""
from __future__ import annotations

PAGE = """
<h1>Dtypes, devices and numerics</h1>
<p class="lede">Five element types, two backends, and a tolerance policy derived up front rather
than tuned until the tests pass.</p>

<h2 id="dtypes">Five types, deliberately</h2>

<p>ggml carries around 30 types because quantised inference needs them. Training does not, and
every extra type multiplies the kernel matrix — quantisation is an explicit non-goal here.
<strong>BF16 is absent because the target GPU does not support it</strong>, measured rather than
assumed.</p>

<div class="table-scroll">
<table>
<thead><tr><th>Type</th><th>Bytes</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>F32</code></td><td>4</td><td>Everything.</td></tr>
<tr><td><code>F16</code></td><td>2</td>
    <td>The same operators as F32 on both backends, with one exception —
        <code>prod</code> is CPU-only for every dtype. <strong>Storage only, never an
        accumulator.</strong></td></tr>
<tr><td><code>I32</code></td><td>4</td><td>Storage and cast only.</td></tr>
<tr><td><code>I64</code></td><td>8</td>
    <td>Storage, cast, and <em>indexing</em> — <code>index_select</code>,
        <code>scatter_add</code>, and the results of <code>argmax</code>/<code>argmin</code>.</td></tr>
<tr><td><code>Bool</code></td><td>1</td>
    <td>Masks: comparison results, and <code>where</code>'s condition.</td></tr>
</tbody>
</table>
</div>

<div class="admon warn"><span class="label">⚠ Warning</span><div class="body">
<p><strong>Neither integer type is an arithmetic type.</strong> There are no integer kernels, and
every operator that would need one raises rather than reinterpreting the bytes. That is why
<code>BatchNorm2d</code>'s <code>num_batches_tracked</code> counter cannot be incremented on the
device — the increment has no kernel to run in.</p>
</div></div>

<p>Only floating tensors can carry gradients, matching PyTorch. <code>backward</code> on an
integer root raises <code>DTypeError</code> rather than producing zeros.</p>

<h2 id="f16-storage">f16 is storage, never an accumulator</h2>

<p>Values widen to <code>float</code> at the memory boundary and narrow once on the store. Both
backends implement it the same way and the source says so — the CPU's <code>widen</code> and the
shader's dtype-switched load are deliberately written to look alike.</p>

<p>The C++ side goes further: <code>Half</code> is a <strong>trivial storage wrapper with no
arithmetic operators</strong>, on purpose. An implicit-conversion half type makes it far too easy
to accumulate in 16 bits by accident, and the whole contract is that you cannot.</p>

<p>It matters most in <code>matmul</code>: an f16 accumulator over K = 784 would lose roughly
three decimal digits, well outside the 1e-3 that the f16 tolerance allows.</p>

<h2 id="conversion">The conversion itself</h2>

<p>IEEE-754 binary16, handling subnormals, infinities and NaN correctly. The naive bit-shuffle
most tutorials show <strong>silently flushes subnormals to zero</strong>, which would surface as
a tolerance failure against PyTorch only for very small values and would be miserable to track
down later. The branch-free approach follows Fabian Giesen's
<code>float_to_half_fast3</code>, the same lineage ggml's implementation comes from.</p>

<p>On the GPU the narrowing is done in the integer domain rather than with
<code>float16_t(value)</code>, because SPIR-V leaves <code>OpFConvert</code>'s rounding mode
implementation-defined — see the shader page for that story.</p>

<h2 id="tolerance">Tolerance is a property of the operation</h2>

<p>Individual tests do not choose their own tolerances. Twice during the project a test "failed"
because its tolerance model was wrong rather than because the code was:</p>

<ul>
<li><code>matmul</code> at K = 4096 missed a 1e-5 <em>relative</em> check while sitting
<strong>431× inside its backward-error bound</strong> — the dot product was ill-conditioned
(Σ|aᵢbᵢ| = 4051 against a result of 3.17), so relative-to-result was meaningless.</li>
<li>Vulkan <code>exp()</code> differed from glibc's by 1.14e-5 absolute, which is 3 ULP at
magnitude 46 — <strong>inside the Vulkan specification's own allowance</strong>.</li>
</ul>

<p>Both were the check being wrong, and both cost real debugging time. The tolerance for an
operation is therefore derived from a citable source and stated once, in
<code>tests/python/tolerance.py</code>, which carries <strong>69 entries across four
kinds</strong> — 39 exact, 16 relative, 10 ULP and 4 backward:</p>

<div class="table-scroll">
<table>
<thead><tr><th>Kind</th><th>Bound</th><th>Used for</th></tr></thead>
<tbody>
<tr><td><code>EXACT</code></td><td>bit-for-bit</td>
    <td>Operations that move or select bits, or are built solely from correctly-rounded IEEE-754
        primitives — which are exact to 0.5 ULP, so anything built only from them must agree
        exactly.</td></tr>
<tr><td><code>ULP</code></td><td>within N units in the last place</td>
    <td>Transcendentals, where the Vulkan specification explicitly <em>permits</em> the driver to
        differ from a correctly-rounded result.</td></tr>
<tr><td><code>RELATIVE</code></td><td>relative to the result</td>
    <td>Composites whose error is dominated by a few rounding steps.</td></tr>
<tr><td><code>BACKWARD</code></td><td>|computed − exact| ≤ γ·Σ|terms|</td>
    <td>Summation, dot products and anything built on them, where the result may be far smaller
        than the terms that produced it.</td></tr>
</tbody>
</table>
</div>

<p>The sources are named: the Vulkan 1.3 specification's per-instruction ULP allowances (which
are <em>permissions granted to the driver</em>, so a conforming implementation may legitimately
differ from libm by that much); IEEE-754 for the correctly-rounded operations; and Higham for the
backward-error bounds on summation.</p>

<p>Above that sits the class-level policy decided in advance — 1e-6 for element-wise f32, 1e-5
for reductions and matmul within their K bounds, 1e-3 for f16 storage with f32 accumulation. Any
failure is investigated as a bug first, and a tolerance that genuinely needs to change must come
with the error analysis that justifies it.</p>

<h2 id="determinism">Determinism</h2>

<p>Identical inputs give bit-identical outputs on the same device, and across drivers wherever
the contract claims it. Two mechanisms carry it:</p>

<ul>
<li><strong>Fixed reduction trees.</strong> Determined by shape, never by which workgroup
finished first. An atomic-accumulation reduction would give a different answer each run — which
is also why the absence of a global float <code>atomicAdd</code> on this hardware costs less than
it appears to.</li>
<li><strong>Software f32→f16 narrowing</strong>, because the hardware instruction's rounding mode
is implementation-defined.</li>
</ul>

<p>Verified across a discrete RX 5600M and an integrated Renoir: MNIST and CIFAR-100 both produce
the same accuracy and the same loss to the last digit on the two.</p>

<h2 id="nan">NaN follows PyTorch, on both backends</h2>

<p>This was unwritten until it drifted: <code>relu(nan)</code> returned 0 while
<code>maximum(x, 0)</code> and <code>clamp_min(x, 0)</code> — the same function spelled
differently — returned NaN, and the Vulkan <code>amax</code>/<code>amin</code> reductions dropped
NaN where the CPU propagated it.</p>

<p><strong>A tolerance cannot express any of this.</strong> NaN is not <em>far from</em> a
number, it is a different kind of answer.</p>

<p>The rule is that torch is the reference, because a user should get the same answer from vkML
as from torch and the same answer from either backend. Where torch and NumPy disagree —
<code>sign(nan)</code> is +0.0 in torch and NaN in NumPy — torch wins.</p>

<p>One mechanism explains every case: <strong>every comparison against NaN is false.</strong> So
<code>x &gt; 0 ? x : 0</code> falls through to 0 and destroys a NaN, while
<code>x &lt;= 0 ? 0 : x</code> falls through to <code>x</code> and keeps it. The two are identical
on numbers, so choosing between them is not a matter of style — and the same choice appears in
<code>relu</code>'s gradient, whose mask is <code>x &lt;= 0</code> for exactly this reason.</p>

<p>Comparison alone cannot make a min/max reduction propagate NaN at all, which is why those
kernels test <code>isnan</code> explicitly at both fold stages.</p>

<h2 id="subnormals">One documented divergence: subnormals</h2>

<p>Vulkan permits flush-to-zero for float32 denormals unless
<code>shaderDenormPreserveFloat32</code> is requested, and vkML does not request it. So
<code>relu(1e-45)</code> is 0 on the GPU and 1e-45 on the CPU, and <code>exp(-89)</code> is 0
rather than the subnormal 2.227e-39.</p>

<p>This is pinned by tests rather than papered over, and the tolerance below which a disagreement
carries no information is <code>FLT_MIN</code> — the smallest positive <em>normal</em> float.</p>
"""
