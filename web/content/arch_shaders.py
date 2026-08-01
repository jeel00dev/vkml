"""Architecture: shaders and the GLSL layer.

Written after reading shaders/common.glsl in full and every shader's header
comment. The shader inventory table is GENERATED from web/research.py at build
time rather than typed, so adding a shader adds a row and changing one changes
its numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research as R  # noqa: E402


def _inventory() -> str:
    """The shader table, built from the tree."""
    d = R.all_shader_details()
    rows = []
    for stem in sorted(d):
        s = d[stem]
        push = s["push"]["bytes"] if s["push"] else 0
        shared = "yes" if s["shared_mem"] else "—"
        ops = len(s["ops"])
        rows.append(
            f"<tr><td><a href=\"{s['url']}\"><code>{stem}.comp</code></a></td>"
            f"<td>{s['lines']}</td>"
            f"<td>{push} B</td>"
            f"<td>{len(s['spec_constants'])}</td>"
            f"<td>{shared}</td>"
            f"<td>{s['barriers']}</td>"
            f"<td>{ops if ops else '—'}</td></tr>")
    return (
        '<div class="table-scroll"><table>'
        "<thead><tr><th>Shader</th><th>Lines</th><th>Push</th><th>Spec</th>"
        "<th>Shared</th><th>Barriers</th><th>Ops</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>")


PAGE = """
<h1>Shaders and the GLSL layer</h1>
<p class="lede">24 compute shaders, one shared preamble, and the conventions that let one module
serve many operators.</p>

<h2 id="descriptorless">Descriptor-less binding</h2>

<p>Buffers arrive as <strong>64-bit device addresses in push constants</strong>, not through
descriptor sets. Using <code>bufferDeviceAddress</code> deletes descriptor pools, set layouts,
per-dispatch <code>vkUpdateDescriptorSets</code> and the pool-growth logic that ggml-vulkan
needs — roughly 200–500 lines of the most error-prone code in a Vulkan backend, plus real
per-dispatch CPU cost.</p>

<p><code>scalarBlockLayout</code> means these structs lay out identically in GLSL and C++, so a
push block is declared once per operator and mirrored by a plain C++ struct with no
<code>std140</code> padding rules to get wrong.</p>

<h2 id="one-module">One module, many pipelines</h2>

<p>The operation is a <strong>specialisation constant</strong>, so the driver folds the switch
away at pipeline creation and each variant is as tight as a dedicated shader. That is why
<code>unary.comp</code> serves 20 operators and <code>binary.comp</code> serves 13 without any
of them paying for the others.</p>

<p>Comparisons live in <code>binary.comp</code> alongside arithmetic because everything except
the final store is identical — same broadcast indexing, same operand layout, same bounds check.
Only the destination element type differs, and that is decided at pipeline creation too.</p>

<h2 id="f16">Why f32→f16 narrowing is done in software</h2>

<p>A <em>load</em> from an f16 buffer is a hardware widening, which is exact and needs no help. A
<em>store</em> goes through <code>f32_to_f16_bits</code>, an integer-domain round-to-nearest-even
routine, rather than <code>float16_t(value)</code>.</p>

<div class="admon warn"><span class="label">⚠ Warning</span><div class="body">
<p><strong>SPIR-V leaves <code>OpFConvert</code>'s rounding mode implementation-defined.</strong>
RADV rounds to nearest even; AMD's Windows compiler rounds toward zero. The same program
therefore produced different f16 results on the two, and the cross-backend oracle — which
compares bit for bit against the CPU's round-to-nearest-even routine — failed on Windows.</p>
</div></div>

<p>Determinism across drivers is a project invariant, so the fix could not be a tolerance. It
could have been the <code>RoundingModeRTE</code> execution mode from
<code>VK_KHR_shader_float_controls</code>, but that needs a device capability, a fallback for
devices without it, and trust that the driver honours it — three things not verifiable from
inside the shader. The routine used instead contains <strong>no floating-point operation whose
rounding a driver could choose</strong>: integer shifts and comparisons only.</p>

<p>It is bit-for-bit the same function as <code>vkml::fp32_to_fp16</code> in
<code>src/core/dtype.cpp</code>, which reaches the same result by a different route, and the two
are checked against each other over the whole f32 exponent range.</p>

<h2 id="f16-storage">f16 is storage, never arithmetic</h2>

<p>Both conversion helpers sit at the memory boundary and everything between them is
<code>float</code>. The dtype is a specialisation constant at every call site, so the branch is
folded away at pipeline creation and an f32 kernel compiles to exactly what it did before f16
existed.</p>

<p>This is deliberately written to look like the CPU backend's <code>widen</code>, because it
implements the same half of the numerical contract.</p>

<h2 id="rank">Rank 4, mirrored from the C++ side</h2>

<p><code>kMaxDims = 4</code> keeps a three-operand shape/stride block at 96 bytes, comfortably
inside the budget; rank 8 would need 192 and force the metadata into a uniform buffer, adding an
indirection to every kernel.</p>

<h2 id="vec4">Vector loads are not automatically one instruction</h2>

<p>Tile loading uses 4-wide access with <code>buffer_reference_align = 4</code> rather than 16.
<code>scalarBlockLayout</code> permits a <code>vec4</code> at 4-byte alignment, so a tile row not
starting on a 16-byte boundary is still legal — but the driver splits the access when it cannot
prove alignment, which is the correct fallback and also means <strong>a vec4 load is not
automatically a single instruction</strong>.</p>

<h2 id="helpers">Shared helpers</h2>

<div class="table-scroll">
<table>
<thead><tr><th>Helper</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>global_index()</code></td>
    <td>Flat invocation index, folding y into x. An identity while y holds a single group, so
        the ordinary case is unchanged.</td></tr>
<tr><td><code>global_group_index()</code></td>
    <td>The same at workgroup granularity, for kernels that index by group — reductions,
        softmax, GEMV and the three GEMM variants.</td></tr>
<tr><td><code>offset_from()</code></td>
    <td>Byte offset of a logical index under arbitrary strides. Broadcasting is free: a stride
        of 0 contributes nothing.</td></tr>
<tr><td><code>f32_to_f16_bits()</code></td>
    <td>Software round-to-nearest-even narrowing, as above.</td></tr>
</tbody>
</table>
</div>

<h2 id="inventory">The shaders</h2>

<p>Generated from the tree — adding a shader adds a row, and changing one changes its numbers.
"Ops" counts the <code>OP_</code> constants a module dispatches on; a dash means the shader
serves one operation.</p>

__INVENTORY__

<h2 id="compilation">Compilation</h2>

<p>Shaders are compiled to SPIR-V at build time by <code>glslc</code> or
<code>glslangValidator</code> from the Vulkan SDK, and the result is embedded in the binary — so
a built vkML has no runtime dependency on a shader compiler and cannot fail at first dispatch
because a <code>.comp</code> file moved.</p>
"""

PAGE = PAGE.replace("__INVENTORY__", _inventory())
