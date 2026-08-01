"""Reference: environment switches and build options.

Both tables are GENERATED. The environment table comes from the actual
`env_flag`/`env_int`/`env_value` call sites in src/, so a new switch appears the
moment it is added and its default is quoted from the call rather than
paraphrased. The build table comes from CMakeLists.txt's option() declarations.

What cannot be generated is what a switch is FOR, so that is written here and
keyed by name. A switch with no note still appears in the table -- the note adds
meaning, it does not decide visibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research as R  # noqa: E402

# What each switch is for. Written after reading its call site.
NOTES: dict[str, tuple[str, str]] = {
    # name: (category, description)
    "VKML_EAGER": ("debugging",
                   "Realize after every operation instead of building a graph. A failure then "
                   "surfaces at the operation that caused it. Slower — every operation becomes "
                   "its own submission."),
    "VKML_VULKAN_DEBUG": ("debugging",
                          "Trace every dispatch with its shape and grid."),
    "VKML_VULKAN_VALIDATION": ("debugging",
                               "Vulkan validation layers. <strong>On by default</strong> — this "
                               "is a correctness-first project, and the cost is paid on every "
                               "run unless explicitly disabled."),
    "VKML_VULKAN_DUMP": ("debugging",
                         "Print the first N elements of each dispatch's output. Clamped to 256."),
    "VKML_VULKAN_NO_PIPELINE_STATS": ("debugging",
                                      "Stop collecting compiler statistics per pipeline."),
    "VKML_COVERAGE": ("testing",
                      "Write an operator-coverage table to this path. Presence-and-non-empty "
                      "rather than a flag, because it names a <em>file</em> — so "
                      "<code>0</code> is a legitimate filename, not \"off\"."),
    "VKML_MIN_SPEC": ("testing",
                      "Report the Vulkan 1.3 Required Limits on any device. Only ever reports "
                      "limits <em>smaller</em> than the hardware has, so it can make vkML more "
                      "conservative and never less. A testing facility, not a tuning knob."),
    "VKML_GEMV": ("GEMM experiments",
                  "Force or disable the matrix–vector kernel."),
    "VKML_GEMM_KERNEL": ("GEMM experiments",
                         "Select the matmul kernel: naive, tiled, or register-blocked "
                         "(the default)."),
    "VKML_GEMM_BLOCK": ("GEMM experiments",
                        "Register-block geometry. Most values exist to <em>test the register "
                        "model</em> rather than as performance candidates — 2x8 and 8x2 are the "
                        "discriminating pair for a model symmetric in RM and RN."),
    "VKML_GEMM_TILE": ("GEMM experiments",
                       "Threadblock tile size. The larger tiles were measured and <em>rejected</em> "
                       "— arithmetic intensity doubled and it was still slower, because "
                       "concurrent workgroups per CU halve at each step."),
    "VKML_GEMM_DB": ("GEMM experiments", "Use the double-buffered GEMM variant."),
    "VKML_GEMM_NOVEC": ("GEMM experiments", "Disable vectorised global loads, as a control."),
    "VKML_GEMM_NOLDSVEC": ("GEMM experiments", "Disable vectorised shared-memory loads."),
    "VKML_GEMM_SPLITK": ("GEMM experiments",
                         "Force or disable split-K. Bit-identical to the unsplit kernel by "
                         "construction, not by measurement."),
    "VKML_GEMM_SPLITK_SPLITS": ("GEMM experiments",
                                "How many K partitions when split-K is forced. Values of 1 or "
                                "less fall back to the default."),
    "VKML_SOFTMAX_PAD_KB": ("GEMM experiments",
                            "Shared-memory padding for softmax, in KiB. A discriminator for "
                            "bank-conflict experiments; it changes no result."),
    "VKML_GEMV_PAD_KB": ("GEMM experiments", "The same, for GEMV."),
}

CATEGORY_ORDER = ["debugging", "testing", "GEMM experiments"]


def _env_table() -> str:
    switches = R.env_switches()
    by_cat: dict[str, list] = {}
    for name in sorted(switches):
        cat = NOTES.get(name, ("other", ""))[0]
        by_cat.setdefault(cat, []).append(name)

    out = []
    for cat in CATEGORY_ORDER + [c for c in sorted(by_cat) if c not in CATEGORY_ORDER]:
        if cat not in by_cat:
            continue
        out.append(f'<h3 id="env-{cat.replace(" ", "-")}">{cat.title()}</h3>')
        rows = []
        for name in by_cat[cat]:
            s = switches[name]
            note = NOTES.get(name, ("", ""))[1] or (
                '<span class="none">no description written yet</span>')
            site = s["sites"][0]
            default = s["default"] or "unset"
            rows.append(
                f"<tr><td><code>{name}</code></td>"
                f"<td>{s['kind']}</td>"
                f"<td><code>{default}</code></td>"
                f"<td>{note}<br><a href=\"{site['url']}\"><code>{site['path']}:"
                f"{site['line']}</code></a></td></tr>")
        out.append('<div class="table-scroll"><table>'
                   "<thead><tr><th>Variable</th><th>Reads as</th><th>Default</th>"
                   "<th>What it does · where it is read</th></tr></thead>"
                   "<tbody>" + "".join(rows) + "</tbody></table></div>")
    return "".join(out)


def _cmake_table() -> str:
    rows = "".join(
        f"<tr><td><code>{o['name']}</code></td><td><code>{o['default']}</code></td>"
        f"<td>{o['help']}</td></tr>"
        for o in R.cmake_options())
    return ('<div class="table-scroll"><table>'
            "<thead><tr><th>Option</th><th>Default</th><th>What it does</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


PAGE = """
<h1>Environment switches and build options</h1>
<p class="lede">Every <code>VKML_*</code> variable the library reads, generated from its call
site — so a new switch appears here the moment it is added.</p>

<div class="admon warn"><span class="label">⚠ Warning</span><div class="body">
<p><strong>These are configuration, not a control interface.</strong> Every switch is read once,
at first use, and cached for the life of the process. Changing one after vkML has started has no
defined effect, because pipelines are already selected and cached from the earlier value. Set
them before launching.</p>
</div></div>

<h2 id="env">Environment switches</h2>

<p>The <em>reads as</em> column is how the value is parsed. A <code>flag</code> is off when the
value starts with <code>0</code> and on for anything else non-empty; an <code>int</code> parses
base-10 and falls back when unset or empty; a <code>value</code> is used as a string, and the
call site decides what it means.</p>

__ENV__

<h2 id="build">Build options</h2>

<p>Passed to CMake with <code>-D</code>. These are compile-time, unlike everything above.</p>

__CMAKE__

<div class="admon warn"><span class="label">⚠ Warning</span><div class="body">
<p><code>cmake --preset release</code> does <strong>not</strong> enable Vulkan. The presets set
the build type and warning flags only, and <code>VKML_VULKAN</code> defaults to
<code>OFF</code> — so without <code>-DVKML_VULKAN=ON</code> you get a CPU-only build in which
every GPU test silently skips.</p>
</div></div>

<h2 id="one-getenv">One getenv in the whole project</h2>

<p>All of these go through a single helper in <code>src/util/env.cpp</code>, which is the only
place <code>std::getenv</code> is called. That is not tidiness: <code>getenv</code> returns a
pointer into the environment block, which a <code>putenv</code> from any thread may invalidate —
and vkML is embedded in Python, where <code>os.environ[...] = ...</code> does exactly that. The
helper returns an owned <code>std::string</code>, which makes the question structurally
impossible rather than merely unlikely.</p>

<p>It is also the single place MSVC's C4996 deprecation has to be answered, rather than in
eighteen files.</p>
"""

PAGE = PAGE.replace("__ENV__", _env_table()).replace("__CMAKE__", _cmake_table())
