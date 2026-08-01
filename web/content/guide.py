"""The narrative pages: landing, install, first model, concepts.

PAGES is a list of (slug, title, html). Written as HTML rather than markdown
because there are four of them and adding a markdown parser to render four pages
would be more machinery than the pages are worth.
"""
from __future__ import annotations

LANDING = """
<div class="lhero">
  <img src="assets/logo-256.png" alt="vkML" width="256" height="171">
  <h1>Machine learning on the GPU you already have</h1>
  <p class="lede">A tensor library and neural-network framework in C++20 with a Python
  API. Compute shaders only &mdash; no CUDA, no ROCm, no vendor runtime.</p>
  <p class="cta">
    <a class="btn solid" href="get-started.html">Get started</a>
    <a class="btn" href="api.html">API reference</a>
    <a class="btn" href="https://github.com/jeel00dev/vkml">GitHub</a>
  </p>
  <div class="lstats">
    <div class="lstat"><b>102</b><span>operators</span></div>
    <div class="lstat"><b>24</b><span>compute shaders</span></div>
    <div class="lstat"><b>2</b><span>backends, one an oracle</span></div>
    <div class="lstat"><b>Vulkan&nbsp;1.3</b><span>and nothing else</span></div>
  </div>
</div>

<div class="lsec">
<h2 id="quick-start">Train something in twenty lines</h2>
<p>Install, pick a device, and run a training loop. The whole API is NumPy-shaped
and PyTorch-shaped on purpose &mdash; a <code>state_dict</code> from torch loads
without translation.</p>
<pre><code>pip install vkml</code></pre>
<pre><code>import vkml
from vkml import nn, optim

vkml.init_vulkan(0)
dev, why = vkml.best_device()
print(why)          # using Vulkan device 0: AMD Radeon RX 5600M (discrete, ...)

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(dev)
opt = optim.Adam(model.parameters(), lr=1e-3)

for x, y in loader:
    opt.zero_grad()
    vkml.backward(nn.cross_entropy(model(x), y))
    opt.step()</code></pre>
</div>

<div class="lsec">
<h2 id="any-gpu">Runs on any Vulkan GPU</h2>
<p>Vulkan 1.3 compute is the only requirement. That covers AMD, Intel, NVIDIA,
Apple through MoltenVK, and the integrated GPU in a laptop &mdash; the same
binary, with no vendor toolkit to install and no per-vendor code path to
maintain.</p>
<div class="lsplit">
  <div>
    <h3>No vendor runtime</h3>
    <p>Twenty-four GLSL compute shaders compiled to SPIR-V at build time. There is
    no CUDA, no ROCm and no vendor SDK anywhere in the build.</p>
    <h3>Written against the guarantee, not the machine</h3>
    <p>Push-constant budgets, workgroup widths and subgroup ranges are checked
    against what Vulkan 1.3 <em>requires</em>, not what the development GPU happens
    to report. <code>VKML_MIN_SPEC=1</code> makes any device report the guaranteed
    floor so that is testable.</p>
  </div>
  <div>
    <h3>Deterministic by contract</h3>
    <p>Identical inputs give identical bytes on the same device, and across
    drivers where the contract says so. f32&rarr;f16 narrowing is done in software
    rather than left to <code>OpFConvert</code>, whose rounding mode SPIR-V leaves
    implementation-defined.</p>
    <h3>Lazy by default</h3>
    <p>Operations build a graph; evaluation waits until something observes the
    result. Eager mode collapses that while debugging.</p>
  </div>
</div>
</div>

<div class="lsec">
<h2 id="correctness">The CPU backend is the oracle, not a fallback</h2>
<p>Every operator has a CPU implementation whose job is to be <em>right</em>, and
the Vulkan one is checked against it. Both are checked against PyTorch. Where the
two backends disagree, the difference is a documented tolerance with a stated
kind &mdash; exact, ULP, relative or backward &mdash; not a fudge factor.</p>
<div class="lsplit">
  <div>
    <h3>Tested against two references</h3>
    <p>1,434 Python tests compare against PyTorch and against the CPU oracle,
    including strided inputs, empty tensors, broadcasting, mixed precision and
    extreme magnitudes.</p>
  </div>
  <div>
    <h3>The tests are checked too</h3>
    <p>A mutation campaign breaks each kernel on purpose and confirms the suite
    notices. A green suite proves the tests ran, not that they can fail.</p>
  </div>
</div>
</div>

<div class="lsec">
<h2 id="architecture">How it fits together</h2>
<p>Nine layers, and the dependency direction is enforced by a build gate rather
than by convention.</p>
<pre><code>util 0 &middot; core 1 &middot; graph 2 &middot; backend/api 3 &middot; backend/cpu | backend/vulkan 4
dispatch 5 &middot; plan 5 &middot; api 6 &middot; autograd 7</code></pre>
<p>A Python call enters the API layer, builds graph nodes, and returns. When a
result is observed the dispatcher walks the graph, picks a backend per node, and
submits. On Vulkan that means selecting one of six matmul pipelines, packing
push constants, and recording a dispatch &mdash; all described in
<a href="concepts.html">Concepts</a> and the architecture pages.</p>
</div>

<div class="lsec">
<h2 id="explore">Explore</h2>
<div class="cards">
  <a class="card" href="get-started.html">
    <h3>Get started</h3>
    <p>Build it, check your device is usable, put a tensor on the GPU and train
    MNIST end to end.</p>
  </a>
  <a class="card" href="concepts.html">
    <h3>Concepts</h3>
    <p>Lazy graphs, the two backends, dtypes and devices, and why the CPU one is
    the oracle.</p>
  </a>
  <a class="card" href="api.html">
    <h3>API reference</h3>
    <p>102 operators and 27 classes, with signatures generated from the installed
    module so they cannot go stale.</p>
  </a>
  <a class="card" href="arch-tensor.html">
    <h3>Architecture</h3>
    <p>Tensors and views, the graph, autograd, both backends, the shader layer
    and the numerics.</p>
  </a>
  <a class="card" href="performance.html">
    <h3>Performance</h3>
    <p>Where the time actually goes, measured with timestamp queries, and what is
    being done about it.</p>
  </a>
  <a class="card" href="https://github.com/jeel00dev/vkml">
    <h3>Source</h3>
    <p>Apache-2.0. Issues, design records and the engineering handbook.</p>
  </a>
</div>
</div>
"""

GET_STARTED = """
<h1>Get started</h1>
<p class="lede">Build vkML, check your GPU is usable, and train a model.</p>

<h2>Requirements</h2>
<ul>
  <li><strong>A C++20 compiler with <code>&lt;format&gt;</code></strong>. 19 source files use
  <code>std::format</code>, which is the strictest library feature the project needs — it
  shipped in libstdc++ with GCC 13, in libc++ with LLVM 17, and in MSVC 19.29.</li>
  <li><strong>CMake 3.25 or newer</strong> — the figure in
  <code>cmake_minimum_required</code>, not an approximation of it — and Ninja.</li>
  <li>The Vulkan SDK, for <code>glslc</code> or <code>glslangValidator</code>.</li>
  <li>Python 3.10+ with NumPy, matching <code>requires-python</code> in
  <code>pyproject.toml</code>.</li>
</ul>

<div class="admon note"><span class="label">ⓘ Note</span><div class="body">
<p><strong>Only some of that is verified.</strong> The compiler versions above are the releases
that shipped <code>&lt;format&gt;</code>, not versions this project has been built with. What is
actually known to work is GCC 14.2.1 (the development machine), MSVC 19.51 (a Windows build
with warnings as errors), and whatever <code>g++</code> and <code>clang++</code> the CI image
ships — which is unpinned. If you build on an older toolchain, a report either way is
useful.</p>
</div></div>

<p>PyTorch is a <strong>test-only</strong> dependency. It is the correctness oracle and is
never needed at runtime.</p>

<h2>Build</h2>
<pre><code># Linux and macOS
cmake --preset release -DVKML_VULKAN=ON
cmake --build build/release -j$(nproc)
pip install -e .</code></pre>

<div class="admon warn">
  <span class="label">warning</span>
  <p><code>cmake --preset release</code> on its own does <strong>not</strong> enable Vulkan —
  the presets set the build type and warning flags only. Without
  <code>-DVKML_VULKAN=ON</code> you get a CPU-only build and every GPU test silently
  skips.</p>
</div>

<p>On Windows the presets set <code>CMAKE_BUILD_TYPE</code>, which a multi-config generator
ignores, so pass the configuration to the build instead:</p>
<pre><code>cmake -B build/msvc -DVKML_VULKAN=ON -DVKML_BUILD_PYTHON=ON
cmake --build build/msvc --config Release --parallel</code></pre>

<h2>Check your device</h2>
<pre><code>python scripts/hardware_report.py</code></pre>
<p>This prints what each Vulkan device reports — required and optional features, subgroup
size, workgroup limits, memory heaps. Paste it into an issue when reporting a bug; almost
every portability defect in this project was found because somebody ran it on hardware the
maintainers do not have.</p>

<h2>Your first tensor</h2>
<pre class="repl"><code><span class="p">&gt;&gt;&gt; </span>import numpy as np, vkml
<span class="p">&gt;&gt;&gt; </span>vkml.init_vulkan(0)
<span class="o">'vulkan:0'</span>
<span class="p">&gt;&gt;&gt; </span>dev, why = vkml.best_device()
<span class="p">&gt;&gt;&gt; </span>print(why)
<span class="o">using Vulkan device 0: AMD Radeon RX 5600M (RADV NAVI10) (discrete, Vulkan 1.4.354, driver radv)   # differs per machine</span>
<span class="p">&gt;&gt;&gt; </span>x = vkml.tensor(np.random.rand(1024, 1024).astype(np.float32), device=dev)
<span class="p">&gt;&gt;&gt; </span>vkml.matmul(x, x).shape
<span class="o">(1024, 1024)</span></code></pre>

<h2>Train something</h2>
<p>Both examples train end to end and compare against a PyTorch model step for step:</p>
<pre><code>python examples/mnist/train.py     --device vulkan:0 --epochs 10
python examples/cifar100/train.py  --device vulkan:0 --epochs 10</code></pre>

<div class="admon">
  <span class="label">note</span>
  <p>Batch size matters more than it looks. Measured on an RX 5600M, per-sample cost falls
  roughly 4× going from batch 64 to batch 512 — at small batches the GPU spends most of a step
  waiting for work rather than doing it. If you are benchmarking, say which batch size you
  used.</p>
</div>

<h2>Run the tests</h2>
<pre><code>ctest --preset release                 # C++ suite
python -m pytest tests/python -q       # Python + PyTorch validation
VKML_MIN_SPEC=1 python -m pytest tests/python -q   # against Vulkan's guaranteed floor</code></pre>
<p><code>VKML_MIN_SPEC=1</code> makes any device report the Vulkan 1.3 Required Limits. It only
ever reports limits <em>smaller</em> than the hardware has, so it can make vkML more
conservative and never less — run it before claiming a limit is satisfied.</p>
"""


CONCEPTS = """
<h1>Concepts</h1>
<p class="lede">The four ideas that explain most of vkML's behaviour.</p>

<h2>Tensors are lazy</h2>
<p>An operation records a node; it does not run. Work happens when a result is needed —
<code>realize()</code>, <code>.numpy()</code>, <code>.item()</code>, or a backward pass.</p>
<p>This is not only an optimisation. Batching lets many operations share a single GPU
submission, and a submission costs roughly <strong>105 µs against 9 µs for a dispatch</strong>
on the development hardware. Reducing submissions is worth far more than making any one
kernel faster.</p>
<p><code>set_eager(True)</code>, or <code>VKML_EAGER=1</code>, runs everything immediately. It
is slower, and it is the right setting while debugging because a failure surfaces at the
operation that caused it.</p>

<h2>Two backends, one of which is a reference</h2>
<div class="table-scroll">
<table>
  <tr><th>Backend</th><th>Role</th><th>Optimised</th></tr>
  <tr><td><code>cpu</code></td><td>Correctness oracle</td>
      <td>No — deliberately a naive triple loop, so it stays simple enough to trust</td></tr>
  <tr><td><code>vulkan:N</code></td><td>Execution</td>
      <td>Yes — hand-written compute shaders</td></tr>
</table>
</div>
<p>Because the CPU backend is the reference, <strong>CPU support must be a superset of Vulkan
support</strong>, and that is enforced by a test rather than by convention. It also means the
CPU backend is slow by design: roughly 116× slower than PyTorch on the same machine. Use it to
check answers, not to get them.</p>

<h2>Devices are explicit</h2>
<p>Tensors do not move on their own. An operation whose operands are on different devices
raises rather than inserting a transfer, because an implicit copy across PCIe is the kind of
cost that should appear in your code and not in a profile.</p>
<pre class="repl"><code><span class="p">&gt;&gt;&gt; </span>a = vkml.tensor(np.zeros((2, 2), dtype=np.float32))
<span class="p">&gt;&gt;&gt; </span>b = vkml.tensor(np.zeros((2, 2), dtype=np.float32), device=vkml.device("vulkan:0"))
<span class="p">&gt;&gt;&gt; </span>vkml.matmul(a, b)
<span class="o">DeviceError: 'matmul' operands are on different devices: cpu and vulkan:0</span></code></pre>
<p>Device <em>indices</em> are not stable across environments — the same machine can report a
discrete GPU at index 0 natively and at index 1 inside a container. Prefer
<code>best_device()</code>, or select on <code>device_type</code> from
<code>vulkan_device_reports()</code>.</p>

<h2>Determinism is a hard invariant</h2>
<p>Identical inputs give bit-identical outputs on the same device, and across drivers wherever
the contract claims it. Two consequences show up in the API:</p>
<ul>
  <li>Reductions use a fixed pairwise tree determined by shape, not by whichever workgroup
  finished first.</li>
  <li>float32→float16 narrowing is implemented in software, because SPIR-V leaves
  <code>OpFConvert</code>'s rounding mode implementation-defined and two drivers disagreed.</li>
</ul>
<p>Any change that would trade this for speed needs a re-derived error bound first. It is not
a preference.</p>

<h2>NaN follows PyTorch</h2>
<p>vkML matches PyTorch's NaN semantics wherever a Vulkan primitive allows it, and documents
the cases where it cannot. A finite input never produces NaN; NaN propagates through
reductions; <code>relu</code> and <code>amax</code> do not silently swallow it. Where a
deliberate divergence exists it is stated on the operator's own page.</p>
"""


from .arch_overview import PAGE as ARCH_OVERVIEW  # noqa: E402
from .arch_autograd import PAGE as ARCH_AUTOGRAD  # noqa: E402
from .arch_cpu import PAGE as ARCH_CPU  # noqa: E402
from .arch_numerics import PAGE as ARCH_NUMERICS  # noqa: E402
from .guide_contributing import PAGE as GUIDE_CONTRIBUTING  # noqa: E402
from .guide_perf import PAGE as GUIDE_PERF  # noqa: E402
from .guide_testing import PAGE as GUIDE_TESTING  # noqa: E402
from .reference_env import PAGE as REFERENCE_ENV  # noqa: E402
from .arch_shaders import PAGE as ARCH_SHADERS  # noqa: E402
from .arch_vulkan import PAGE as ARCH_VULKAN  # noqa: E402
from .arch_graph import PAGE as ARCH_GRAPH  # noqa: E402
from .arch_tensor import PAGE as ARCH_TENSOR  # noqa: E402

PAGES: list[tuple[str, str, str]] = [
    ("index", "vkML — machine learning on Vulkan", LANDING),
    ("get-started", "Get started", GET_STARTED),
    ("concepts", "Concepts", CONCEPTS),
    ("performance", "Performance", GUIDE_PERF),
    ("testing", "Testing and verification", GUIDE_TESTING),
    ("contributing", "Adding an operator", GUIDE_CONTRIBUTING),
    ("arch-overview", "Architecture overview", ARCH_OVERVIEW),
    ("arch-tensor", "Tensors, storage and views", ARCH_TENSOR),
    ("arch-graph", "The lazy graph and execution", ARCH_GRAPH),
    ("arch-autograd", "Autograd", ARCH_AUTOGRAD),
    ("arch-cpu", "The CPU backend", ARCH_CPU),
    ("arch-vulkan", "The Vulkan backend", ARCH_VULKAN),
    ("arch-shaders", "Shaders and the GLSL layer", ARCH_SHADERS),
    ("arch-numerics", "Dtypes, devices and numerics", ARCH_NUMERICS),
    ("reference-env", "Environment switches and build options", REFERENCE_ENV),
]
