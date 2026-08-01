"""Architecture: the Vulkan backend.

Written after reading src/backend/vulkan/{vk_device,vk_command,vk_pipeline,
vk_allocator}.{h,cpp}, vk_dispatch_grid.h and vulkan_backend.cpp. The numeric
facts -- push-block sizes, shader counts, barrier counts -- come from
web/research.py rather than being typed here, so they cannot go stale.
"""
from __future__ import annotations

PAGE = """
<h1>The Vulkan backend</h1>
<p class="lede">How a graph node becomes a dispatch, and the device limits that shape every
decision along the way.</p>

<h2 id="required">Three required features, and what each buys</h2>

<p>A device missing any of these cannot run vkML at all, and
<code>vulkan_unavailable_reason()</code> names the first one found:</p>

<div class="table-scroll">
<table>
<thead><tr><th>Feature</th><th>What depends on it</th></tr></thead>
<tbody>
<tr><td><code>bufferDeviceAddress</code></td>
    <td><strong>Every buffer in every shader.</strong> vkML binds no descriptor sets at all —
        all 24 shaders take their operands as <code>uint64_t</code> addresses inside the
        push-constant block.</td></tr>
<tr><td><code>scalarBlockLayout</code></td>
    <td>Lets the push block use C-like scalar packing, so the GLSL struct and the C++ struct
        agree without padding rules diverging.</td></tr>
<tr><td><code>timelineSemaphore</code></td>
    <td>Host-side completion. One monotonically increasing counter per stream — waiting for
        value N means "everything up to submit N has finished", which replaces fences
        entirely: no fence pool, no reset, no per-submit object.</td></tr>
</tbody>
</table>
</div>

<div class="admon note"><span class="label">ⓘ Note</span><div class="body">
<p>Zero descriptor sets is unusual enough to be worth stating plainly, and it explains why the
push-constant budget is the recurring constraint in this backend: every operand costs 8 bytes of
the 128 Vulkan guarantees before any shape metadata is packed alongside it.</p>
</div></div>

<h2 id="push">The push-constant budget</h2>

<p><code>maxPushConstantsSize</code> is guaranteed to be only <strong>128 bytes</strong>. The
development GPU reports 256, which is exactly why this went wrong once: three blocks exceeded
the guarantee, producing 19 failing tests on an AMD Windows driver reporting 128 and nothing at
all locally.</p>

<p>Every block now fits, and the constraint is enforced from both sides:</p>

<ul>
<li><strong>C++</strong> — all 16 push structs carry a
<code>static_assert(sizeof(X) &lt;= kGuaranteedPushConstantBytes)</code>, so the build fails on
the machine that writes the change.</li>
<li><strong>GLSL</strong> — <code>scripts/check_push_constants.py</code> reads the shader
declarations and fails if any block exceeds 128 bytes.</li>
</ul>

<p>Two of those assertions were missing until recently, on the two smallest blocks — which is
exactly where a missing guard hides, since they are the least likely to grow.</p>

<p>The largest blocks today are <code>binary</code> at 124 B, <code>softmax</code> at 120 B and
<code>where</code> at 116 B. Fitting them took per-operator work rather than one general fix:
<code>where</code> and <code>softmax</code> store shared extents once instead of per operand,
and <code>cat</code> <em>derives</em> its operands' extents from the output's, which brought it
from 144 down to 112.</p>

<h2 id="workgroup">Workgroup width adapts to the device</h2>

<p>No shader declares a literal workgroup size. <code>common.glsl</code> declares
<code>layout(local_size_x_id = 0) in;</code> once and all 24 inherit it, so the width is a
<strong>specialisation constant</strong> resolved at pipeline creation.</p>

<p>That is the mechanism behind the clamp: the general width is
<code>min(256, maxComputeWorkGroupInvocations)</code>. Vulkan guarantees only 128, and this
asked for 256 unconditionally — so a conformant minimum-spec device could create almost no
pipeline at all. Devices allowing 256 keep it and are unaffected.</p>

<p>Varying it is safe because the width reaches the shader as a constant and the grid is derived
from it, so every kernel on that path adapts, and shared-memory requests scale with it — a
narrower workgroup asks for less, never more.</p>

<div class="admon warn"><span class="label">⚠ Warning</span><div class="body">
<p>Clamping alone would not have been enough. Both blocked GEMM kernels hardcode 256 invocations
— deliberately, so the three variants stay comparable — so a floor device would have had every
element-wise, reduction and movement operator and still no matmul: the appearance of support
without the ability to train. <code>matmul</code> therefore falls back to the naive kernel,
which takes the clamped width and fits by construction.</p>
</div></div>

<h2 id="grid">The dispatch grid, and the 65535 ceiling</h2>

<p><code>maxComputeWorkGroupCount[x]</code> is guaranteed to be only <strong>65535</strong>. The
development GPU reports 2³²−1, so every elementwise operation above 64 MiB failed on a driver
reporting the floor and nothing showed locally.</p>

<p><code>choose_dispatch_grid</code> folds an oversized grid into the second dimension. It takes
the limits <strong>as parameters</strong> rather than reading them from a device, which is what
makes the guaranteed floor testable on any machine — the test for it compiles unconditionally,
including in the three CPU-only CI jobs.</p>

<p>The reduction path hits that ceiling far sooner than the element-wise one, because it
dispatches one workgroup per output <em>row</em>: 32 images of 3×64×64 average-pooled is 98,304
rows against a guaranteed 65,535, an ordinary batch — where the element-wise path needs 16.7
million elements to get there.</p>

<h2 id="sync">Synchronisation: two primitives, both the simplest correct choice</h2>

<p><strong>A global memory barrier between dependent dispatches</strong> — a single
<code>vkCmdPipelineBarrier</code> with <code>shaderRead|shaderWrite</code> on both sides, not a
per-buffer barrier. Tracking per-buffer hazards costs more CPU time than the barrier costs GPU
time, for a graph where almost every node depends on its predecessor anyway. It is conservative,
ordering more than strictly necessary, which is the right default when correctness comes
first.</p>

<p>The barrier is <strong>not optional</strong>: without it a dispatch may read a buffer another
is still writing, because the GPU does not serialise dispatches on its own. Making it selective
requires a planner that knows which nodes alias, and that is the one place skipping it will be
justified.</p>

<p><strong>A timeline semaphore for host completion</strong>, as described above.</p>

<h2 id="recorder">The recorder makes no decisions</h2>

<p><code>Recorder</code> records and submits. It does not choose an order, does not allocate,
and does not decide which kernel runs — those belong to the executor above it. Keeping the split
means a future lowered execution graph can drive the same recorder unchanged, simply calling
<code>dispatch</code> in a different order.</p>

<h2 id="pipelines">Pipelines and specialisation</h2>

<p><code>KernelConfig</code> exists so the runtime never hardcodes a workgroup or subgroup
decision. llama.cpp demonstrates why that matters on this exact GPU: its RDNA1 table pins wave64
for softmax, argmax and matrix-vector work and wave32 everywhere else, because the right width
differs per kernel and the device's <em>default</em> — 64 here — is not the right answer for
most of them.</p>

<p>A subgroup size of 0 means "let the driver choose"; anything else creates the pipeline with
<code>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo</code>, which needs
<code>subgroupSizeControl</code>. Shared-memory requests are validated against
<code>maxComputeSharedMemorySize</code> before creation rather than failing at dispatch.</p>

<p>Autotuning is explicitly not implemented — but when it is, it becomes a search over these
fields rather than a redesign, because nothing above reads a hardcoded constant.</p>

<h2 id="min-spec">Testing against the floor</h2>

<p><code>VKML_MIN_SPEC=1</code> makes any device report the Vulkan 1.3 Required Limits. It only
ever reports limits <em>smaller</em> than the hardware has, so it can make vkML more
conservative and never less — which is what makes it a testing facility rather than a tuning
knob.</p>

<p>It exists because the alternative was hand-editing <code>vk_device.cpp</code>, which CI
cannot run and a contributor cannot reproduce. Run it before claiming a limit is satisfied: it
is the cheapest way to find the next instance of this project's most common bug.</p>
"""
