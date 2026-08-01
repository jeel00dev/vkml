"""Architecture: the CPU backend.

Written after reading every file in src/backend/cpu/ -- cpu_backend.cpp,
iterate.h, kernels.h, kernels_elementwise.cpp, kernels_matmul.cpp,
kernels_movement.cpp, kernels_reduce.cpp, reduce.h, reduce.cpp and philox.h --
and checking the error bounds it states.
"""
from __future__ import annotations

PAGE = """
<h1>The CPU backend</h1>
<p class="lede">The correctness oracle. Deliberately naive, deliberately slow, and the reason
the Vulkan backend can be trusted at all.</p>

<h2 id="role">It is a reference, not a fallback</h2>

<p>Correctness here is a chain: the CPU backend is checked against PyTorch for semantics, then
the Vulkan backend against the CPU for kernel bugs. That second link only means anything because
the CPU backend shares vkML's <em>exact</em> semantics — so a mismatch is unambiguously a kernel
defect rather than a difference of convention.</p>

<p>Two consequences follow, and both are enforced rather than hoped for:</p>

<ul>
<li><strong>CPU support must be a superset of Vulkan support.</strong> A test asserts it
directly, because it broke once: widening the Vulkan <code>supports()</code> gates for float16
made the GPU accept operators the CPU still rejected, and every one of those had a GPU result no
oracle could check while the suite stayed green.</li>
<li><strong>Performance comes last here.</strong> The matmul is a naive triple loop with no
blocking and no vectorisation, and the source says so where someone might otherwise
"fix" it.</li>
</ul>

<div class="admon note"><span class="label">ⓘ Note</span><div class="body">
<p>Measured on this machine, the CPU backend is roughly <strong>116× slower than PyTorch</strong>
on the same work. That is the cost of the trade, not a defect — and it is why the README's
CPU-only install path is for checking answers, not for getting them.</p>
</div></div>

<h2 id="pairwise">Pairwise summation, and why it is not optional</h2>

<p>This is the single most important numerical decision in the backend, and it is a correctness
requirement rather than a refinement.</p>

<p>Sequential summation of <code>n</code> values in float32 has a worst-case relative error of
<code>n·ε</code>, with <code>ε = 2⁻²³ ≈ 1.19e-7</code>. The project's gate against PyTorch is
<code>atol = rtol = 1e-5</code>. So:</p>

<div class="table-scroll">
<table>
<thead><tr><th>n</th><th>What it is</th><th>Sequential error</th><th>Against the 1e-5 gate</th></tr></thead>
<tbody>
<tr><td>784</td><td>MNIST input features</td><td>≈ 9.3e-5</td>
    <td><strong>fails by ~9×</strong></td></tr>
<tr><td>4096</td><td>transformer hidden</td><td>≈ 4.9e-4</td>
    <td><strong>fails by ~49×</strong></td></tr>
</tbody>
</table>
</div>

<p>A naive accumulator does not merely lose a little precision — it misses the acceptance
criterion outright, and does so in a way that <em>looks like a kernel bug</em>. Pairwise
summation splits the range recursively, giving a bound of about
<code>(B + log₂(n/B))·ε</code> for a sequential base case of size <code>B</code>:</p>

<div class="table-scroll">
<table>
<thead><tr><th>B = 32, n</th><th>Pairwise error</th><th>Margin</th></tr></thead>
<tbody>
<tr><td>784</td><td>≈ 4.3e-6</td><td>2.3×</td></tr>
<tr><td>4096</td><td>≈ 4.6e-6</td><td>2.2×</td></tr>
<tr><td>16384</td><td>≈ 4.9e-6</td><td>2.0×</td></tr>
</tbody>
</table>
</div>

<p><code>kPairwiseBlock = 32</code> is chosen to keep at least 2× margin out to n = 16384, which
covers every reduction length these models produce. NumPy uses 128 and would still pass in
practice, because real rounding errors random-walk rather than aligning — but designing to the
worst-case bound costs nothing measurable and removes a whole class of "why is this test flaky
at large K" investigation later.</p>

<p>It also <strong>mirrors what the GPU has to do anyway</strong>. That device has no global
float <code>atomicAdd</code>, so its reductions must be tree-shaped regardless — subgroup
reduction, then shared memory, then a deterministic second pass. Matching the CPU reference to
that structure is what keeps the two comparable, and it is why the GEMM shaders use
<code>BK = 32</code>: the same block size, so both backends fold K identically.</p>

<h2 id="f16">float16 is a storage format, never an accumulator</h2>

<p>Every kernel computes in <code>float</code> whatever it stores. The widening lives in exactly
two overloads in <code>iterate.h</code>, which is what keeps the f32 and f16 paths from drifting
apart.</p>

<p>It matters most in matmul: an f16 accumulator over K = 784 would lose roughly three decimal
digits, well outside the 1e-3 the f16 tolerance allows. The dot product accumulates in float
regardless of the storage type.</p>

<h2 id="iterate">Strided iteration</h2>

<p><code>iterate.h</code> converts a flat logical index into a byte offset per operand.
Broadcasting is handled <strong>for free</strong>: a stride of 0 contributes nothing to the
offset, so every index along that axis reads the same element — no special case, no branch.</p>

<p>That is why a broadcast operand costs no memory and no separate code path anywhere in the
backend.</p>

<h2 id="philox">Randomness is counter-based</h2>

<p><code>rand</code> and <code>dropout</code> use <strong>Philox4x32-10</strong>, a counter-based
generator rather than a stateful one. The value at each index is a pure function of
<code>(seed, offset, index)</code>.</p>

<p>That property is what lets the GPU produce the same draw as the CPU without any sequencing
between invocations — there is no stream to keep in step, because there is no stream. It is also
why the signatures take a seed and an offset rather than carrying hidden state.</p>

<p>Each value takes the top 24 bits of a 32-bit output, which is the float significand's width,
so every result is exactly representable. The round keys are the fractional parts of the golden
ratio and of √3, the paper's own "nothing up my sleeve" construction.</p>

<h2 id="alignment">Allocation</h2>

<p>CPU allocations are aligned to <strong>64 bytes</strong> — one cache line on every CPU this
will run on, and the alignment AVX-512 wants. The backend will not be hand-vectorised, but
aligning costs nothing and removes a variable if the compiler ever vectorises part of it.</p>
"""
