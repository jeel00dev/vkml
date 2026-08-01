"""Guide: performance.

Every number here was measured during this project's own benchmarking sessions,
on the machine described in the table. Nothing is quoted from another project or
extrapolated from a partial run. Where a figure could not be obtained honestly,
the page says so rather than estimating.
"""
from __future__ import annotations

PAGE = """
<h1>Performance</h1>
<p class="lede">What vkML currently costs, where the time actually goes, and which of those
numbers are trustworthy.</p>

<div class="admon warn"><span class="label">⚠ Warning</span><div class="body">
<p><strong>vkML is not fast yet, and this page does not pretend otherwise.</strong> The core is
correct and heavily tested; the performance work is largely ahead. Everything below is a
measurement, not a target.</p>
</div></div>

<h2 id="machine">The machine</h2>

<div class="table-scroll">
<table>
<thead><tr><th>Component</th><th>What it is</th></tr></thead>
<tbody>
<tr><td>Discrete GPU</td><td>AMD Radeon RX 5600M (RADV NAVI10), 36 CUs, 5.75 GiB device-local,
    256 MiB host-visible</td></tr>
<tr><td>Integrated GPU</td><td>AMD Radeon Graphics (RADV RENOIR), 6 CUs</td></tr>
<tr><td>Driver</td><td>RADV (Mesa)</td></tr>
</tbody>
</table>
</div>

<p>Validation layers are <strong>on by default</strong>. Every figure below includes that cost,
which is the honest default to measure — but it is worth knowing before comparing against
anything else.</p>

<h2 id="headline">The headline number</h2>

<p>Same CNN architecture, same batch, same optimiser, CIFAR-100:</p>

<div class="table-scroll">
<table>
<thead><tr><th>Configuration</th><th>ms/step</th></tr></thead>
<tbody>
<tr><td>PyTorch, <strong>CPU</strong>, 8 threads</td><td>34.26</td></tr>
<tr><td>vkML, <strong>discrete GPU</strong>, 36 CUs</td><td>35.77</td></tr>
</tbody>
</table>
</div>

<p>A 36-compute-unit discrete GPU is 4% <em>slower</em> than PyTorch on a CPU. Since torch on a
GPU would be many times faster than torch on a CPU, the real distance to parity is not the 1.04×
this shows — it is that whole further multiple.</p>

<h2 id="where">Where the time goes: starvation, not slow kernels</h2>

<p>Batch scaling separates fixed cost from arithmetic. If per-sample time falls as the batch
grows, the device was idle waiting for work:</p>

<div class="table-scroll">
<table>
<thead><tr><th>Batch</th><th>ms/step</th><th>ms/sample</th><th>vs batch 64</th></tr></thead>
<tbody>
<tr><td>64</td><td>34.45</td><td>0.538</td><td>1.00×</td></tr>
<tr><td>128</td><td><strong>26.23</strong></td><td>0.205</td><td>0.38×</td></tr>
<tr><td>256</td><td>39.39</td><td>0.154</td><td>0.29×</td></tr>
<tr><td>512</td><td>70.29</td><td>0.137</td><td><strong>0.26×</strong></td></tr>
</tbody>
</table>
</div>

<p>Per-sample cost falls <strong>3.9×</strong>. The strongest single line is batch 128: it does
<em>twice the work</em> of batch 64 in <em>less wall time</em>, which only happens when fixed
per-step cost dominates arithmetic.</p>

<p><strong>At the batch size the examples use, roughly three quarters of a training step is
overhead.</strong></p>

<h2 id="corroboration">Three independent observations agree</h2>

<ul>
<li>MNIST trains at <strong>4.41 s/epoch on the 36-CU discrete card and 4.35 s/epoch on the 6-CU
integrated one</strong>. Six times the compute, no difference — the workload never reaches the
arithmetic.</li>
<li>A submission costs about <strong>105 µs against 9 µs for a dispatch</strong>.</li>
<li>The optimiser is <strong>62.7% of an MLP step</strong>, across 12 submissions.</li>
</ul>

<p>CIFAR-100's CNN behaves differently — its own breakdown reports <strong>96.3% compute</strong>
— so the ceiling is specific to small models and short steps, which is exactly where an LLM
decode step would live.</p>

<h2 id="devices">Two GPUs, identical results</h2>

<div class="table-scroll">
<table>
<thead><tr><th>Workload</th><th>Discrete (36 CU)</th><th>Integrated (6 CU)</th><th>Test accuracy</th></tr></thead>
<tbody>
<tr><td>MNIST MLP, 10 epochs</td><td>4.41 s/epoch</td><td>4.35 s/epoch</td>
    <td><strong>97.47% on both</strong></td></tr>
<tr><td>CIFAR-100 CNN, 10 epochs</td><td>17.48 s/epoch</td><td>60.45 s/epoch</td>
    <td><strong>28.90% on both</strong></td></tr>
</tbody>
</table>
</div>

<p>The accuracies are identical to the last digit on two different GPUs. That is the determinism
contract holding across hardware, not a coincidence.</p>

<p>The timings show the same split as the batch scaling: on the compute-bound CNN the 6-CU part
is 3.5× slower, as its compute-unit count predicts; on the submission-bound MLP it <em>ties the
discrete card</em>.</p>

<h2 id="torch-parity">Against PyTorch, on accuracy</h2>

<div class="table-scroll">
<table>
<thead><tr><th>Workload</th><th>vkML</th><th>PyTorch</th><th>Difference</th></tr></thead>
<tbody>
<tr><td>MNIST MLP, GPU</td><td>97.47%</td><td>97.50%</td><td>−0.03 pp</td></tr>
<tr><td>MNIST MLP, CPU backend</td><td>97.75%</td><td>97.50%</td><td>+0.25 pp</td></tr>
<tr><td>CIFAR-100 CNN</td><td>28.90%</td><td>30.10%</td><td>−1.20 pp</td></tr>
</tbody>
</table>
</div>

<h2 id="cpu-backend">The CPU backend is not for training</h2>

<p>Measured: 4 CIFAR steps took 12.66 s of compute, about <strong>3.17 s/step</strong>. The full
set is 782 steps per epoch, so one epoch is roughly <strong>41 minutes</strong> and ten are about
<strong>6.9 hours</strong>. It did not complete a single epoch on even 2,000 examples within a
550-second budget.</p>

<p>That is the cost of the backend being a deliberately naive correctness oracle. Use it to check
answers, not to get them.</p>

<h2 id="honest">What is not measured</h2>

<p><strong>There is no per-kernel attribution today.</strong>
<code>vulkan_last_profile</code> returns submission-level <code>('submit', ms)</code> pairs, so
none of the evidence above identifies <em>which kernel</em> dominates a step. Everything here is
indirect — batch scaling, device substitution, submission counting. That is enough to locate the
problem and not enough to close it.</p>

<p>Timestamps are supported by the device; what is missing is recording them around each dispatch
and aggregating by kernel name. Until that exists, treat any claim about a specific kernel's
share of a step as unproven.</p>

<h2 id="benchmarking">If you benchmark this yourself</h2>

<ul>
<li><strong>State the batch size.</strong> A 4× difference in per-sample cost sits between batch
64 and 512, so a figure without one is not comparable.</li>
<li><strong>Say whether validation layers were on.</strong> They are on by default.</li>
<li><strong>Report the minimum across process runs, never the mean.</strong> Noise here is
one-sided — something slowing a run down is a cause, something speeding it up is not.</li>
<li><strong>Run a same-binary control.</strong> A previous measurement in this project showed
−19% for a change whose control showed a ±18.9% noise floor, and another showed +3.7% for a
padding-only change whose control moved −6.5% — which no padding can cause.</li>
</ul>
"""
