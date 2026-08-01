"""The narrative pages: landing, install, first model, concepts.

PAGES is a list of (slug, title, html). Written as HTML rather than markdown
because there are four of them and adding a markdown parser to render four pages
would be more machinery than the pages are worth.
"""
from __future__ import annotations

LANDING = """
<div class="hero">
  <img src="assets/vkml_logo.png" alt="vkML">
  <h1>Machine learning on Vulkan</h1>
  <p class="lede">A tensor library and neural-network framework in C++20 with a Python API.
  Compute shaders only, no vendor runtime — it runs on the GPU you already have.</p>
  <p>
    <a class="btn solid" href="get-started.html">Get started</a>
    <a class="btn" href="api.html">API reference</a>
  </p>
</div>

<div class="cards">
  <a class="card" href="get-started.html">
    <h3>Install and train</h3>
    <p>Build it, put a tensor on the GPU, and train MNIST end to end.</p>
  </a>
  <a class="card" href="concepts.html">
    <h3>How it works</h3>
    <p>Lazy graphs, the two backends, and why the CPU one is the oracle.</p>
  </a>
  <a class="card" href="api.html">
    <h3>API reference</h3>
    <p>Every public function, with signatures generated from the module.</p>
  </a>
  <a class="card" href="https://github.com/jeel00dev/vkml">
    <h3>Source</h3>
    <p>Apache-2.0. Issues, design records and the engineering handbook.</p>
  </a>
</div>

<h2>What it looks like</h2>
<pre><code>import numpy as np, vkml
from vkml import nn, optim

vkml.init_vulkan(0)
dev, why = vkml.best_device()
print(why)          # using Vulkan device 0: AMD Radeon RX 5600M (discrete, …)

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(dev)
opt = optim.Adam(model.parameters(), lr=1e-3)

for x, y in loader:
    opt.zero_grad()
    vkml.backward(nn.cross_entropy(model(x), y))
    opt.step()</code></pre>

<h2>What makes it different</h2>

<h3>Portable by construction</h3>
<p>No CUDA, no ROCm, no vendor SDK. Vulkan compute shaders run on AMD, Intel, NVIDIA and
software rasterisers alike. The library asserts against the limits Vulkan <em>guarantees</em>
rather than the ones the development machine happens to report — every push-constant block
fits the guaranteed 128 bytes, and the workgroup width adapts to the device.</p>

<h3>The CPU backend is an oracle, not a fallback</h3>
<p>Correctness is a chain: the CPU backend is checked against PyTorch for semantics, then the
Vulkan backend against the CPU for kernel bugs. That second link only means anything because
the CPU backend shares vkML's exact semantics, so a mismatch is unambiguously a kernel
defect.</p>
<p>It follows that the Vulkan backend is <strong>all or nothing</strong>. An operation it does
not implement raises <code>NotImplementedError</code> instead of quietly moving your data to
the host — a silent fallback would make a GPU run indistinguishable from a CPU one.</p>

<h3>Deterministic on purpose</h3>
<p>Identical inputs give bit-identical outputs on the same device, and across drivers where
the contract says so. Reduction trees are fixed by shape rather than by scheduling, and
float32→float16 narrowing is done in software because SPIR-V leaves
<code>OpFConvert</code>'s rounding mode implementation-defined.</p>

<div class="admon warn">
  <span class="label">alpha</span>
  <p>vkML is under active development. The core tensor and autograd layers are stable and
  heavily tested; performance work is ongoing and the API may still change. See
  <a href="https://github.com/jeel00dev/vkml/issues">open issues</a> for what is known.</p>
</div>
"""


GET_STARTED = """
<h1>Get started</h1>
<p class="lede">Build vkML, check your GPU is usable, and train a model.</p>

<h2>Requirements</h2>
<ul>
  <li>A C++20 compiler — GCC 12+, Clang 15+, or MSVC 19.3+</li>
  <li>CMake 3.24+ and Ninja</li>
  <li>The Vulkan SDK, for <code>glslc</code> or <code>glslangValidator</code></li>
  <li>Python 3.10+ with NumPy</li>
</ul>
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


from .arch_autograd import PAGE as ARCH_AUTOGRAD  # noqa: E402
from .arch_graph import PAGE as ARCH_GRAPH  # noqa: E402
from .arch_tensor import PAGE as ARCH_TENSOR  # noqa: E402

PAGES: list[tuple[str, str, str]] = [
    ("index", "vkML — machine learning on Vulkan", LANDING),
    ("get-started", "Get started", GET_STARTED),
    ("concepts", "Concepts", CONCEPTS),
    ("arch-tensor", "Tensors, storage and views", ARCH_TENSOR),
    ("arch-graph", "The lazy graph and execution", ARCH_GRAPH),
    ("arch-autograd", "Autograd", ARCH_AUTOGRAD),
]
