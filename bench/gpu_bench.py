#!/usr/bin/env python3
"""GPU benchmark and performance regression suite.

METHODOLOGY
-----------
GPU execution is measured with Vulkan timestamp queries, never with wall clock.
That distinction is not pedantic: before timestamps existed, wall-clock timing
reported softmax as 50x slower than sum, when the true GPU ratio is 2.4x and the
rest was download time. Every number here separates:

    gpu_ms    kernel execution, from timestamp queries
    upload    host -> device, wall clock (no GPU work to time)
    download  device -> host, wall clock
    wall      end to end

Minimum is the headline figure -- for a deterministic workload it is the sample
least polluted by scheduler noise -- with mean and standard deviation reported
alongside so a wide spread is visible rather than hidden.

REGRESSION MODE
---------------
`--check baseline.json` compares against a recorded baseline and WARNS; it never
fails. Kernel time and wall-clock time are compared separately, so a transfer
change and a kernel change stay distinguishable.

A threshold alone cannot say whether a change is real. Measured over ten runs on
an RX 5600M, 22 of 34 benchmarks moved by more than the 15% threshold WITHOUT
any code change, and the within-run standard deviation does not predict that --
it understates the between-run spread by a median of 5.4x. So `--runs N` records
each benchmark's own run-to-run spread into the baseline, and the comparison
reports anything inside three of those as noise rather than as a warning. The
one benchmark whose within-run figure did predict its spread, `dispatch 1
element`, is also the one whose reported regression survived the control.

Baselines carry GPU model, driver, subgroup configuration and timestamp period,
because a number without that metadata is not interpretable on another machine.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vkml as V  # noqa: E402

DEFAULT_THRESHOLD = 0.15  # warn beyond 15% slower
# How many multiples of a benchmark's measured run-to-run spread a change must
# exceed before it is called a regression rather than noise. Three is the usual
# convention and is deliberately conservative: this tool warns, so the cost of
# a missed regression is a later investigation, while the cost of crying wolf
# is that the whole report stops being read.
NOISE_SIGMA = 3.0


@dataclass
class Sample:
    name: str
    category: str
    gpu_min: float = 0.0
    gpu_mean: float = 0.0
    gpu_sd: float = 0.0
    # Dispersion of gpu_min ACROSS separate runs of this program, as a fraction
    # of the median. Zero when only one run was taken. See noise_floor() for why
    # this exists and why gpu_sd cannot stand in for it.
    gpu_run_sd: float = 0.0
    wall_min: float = 0.0
    wall_mean: float = 0.0
    # The same between-run measure for wall clock. Kept separate because wall
    # time carries transfer and host cost that kernel time does not, and is
    # consistently the noisier of the two -- so one floor cannot serve both.
    wall_run_sd: float = 0.0
    transfer_ms: float = 0.0
    bytes_moved: float = 0.0
    gbps: float = 0.0


def _stats(values):
    if not values:
        return 0.0, 0.0, 0.0
    return (min(values), statistics.fmean(values),
            statistics.pstdev(values) if len(values) > 1 else 0.0)


# One home for the rule, not two. This lived here first; it moved into the
# library when examples/cifar100/train.py needed it as well, because the whole
# hazard is that a second copy drifts and starts silently reporting
# multiply-counted numbers. See vkml.vulkan_submit_ms and
# docs/MEASUREMENT-AUDIT.md 3.
_gpu_time = V.vulkan_submit_ms


def measure(name, category, fn, reps=25, bytes_moved=0.0, device=None) -> Sample:
    """Times `fn`, separating GPU execution from everything else."""
    if device is not None:
        warm_up(device)     # raise the clock, immediately before timing
    fn()  # warm up: pipeline creation and first-touch belong to setup

    gpu, wall = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wall.append((time.perf_counter() - t0) * 1e3)
        profile = V.vulkan_last_profile()
        if profile:
            gpu.append(_gpu_time(profile))

    g_min, g_mean, g_sd = _stats(gpu)
    w_min, w_mean, _ = _stats(wall)

    s = Sample(name=name, category=category, gpu_min=g_min, gpu_mean=g_mean, gpu_sd=g_sd,
               wall_min=w_min, wall_mean=w_mean, transfer_ms=max(w_min - g_min, 0.0),
               bytes_moved=bytes_moved)
    if bytes_moved and s.transfer_ms > 0:
        s.gbps = bytes_moved / 1e9 / (s.transfer_ms / 1e3)
    return s


_WARM: dict = {}


def warm_up(device, rounds: int = 12) -> None:
    """Drive the GPU hard IMMEDIATELY BEFORE a measurement.

    WHY THIS IS NOT OPTIONAL HERE. This machine parks at 400 MHz of a possible
    1500 and raises clocks in response to load, so an unwarmed benchmark
    measures the clock policy rather than the code. Measured on `dispatch 1
    element` in one process: 0.01052 ms cold against 0.00572 ms after sixty
    1024-cubed matmuls -- a factor of 1.8 from clock state alone, with the same
    binary.

    That is not a subtlety, it is the whole signal for the small benchmarks. An
    apparent 38.6% regression against a recorded baseline turned out to be
    entirely inside this range, and the baseline recorded no clock or driver
    state to rule it out.

    WARM-UP IS A PRECONDITION OF EACH MEASUREMENT, NOT A PHASE. Warming once at
    the top of the run does not work, and that was measured rather than assumed:

        warm -> measure                 0.00612 ms
        warm -> transfers -> measure    0.00932    the transfers undo it
        transfers -> warm -> measure    0.00560

    The suite runs 1/4/16 MiB uploads and downloads before it reaches the
    dispatch case. Those move memory without raising the compute clock and take
    long enough for it to fall back, so a warm-up at the start of the run had no
    effect on the number by the time it mattered -- verified by adding one and
    watching the figure not move.

    The operands are cached per device, because allocating two megabyte tensors
    before every measurement would itself become the thing being measured.
    """
    ab = _WARM.get(str(device))
    if ab is None:
        ab = (V.tensor(np.random.rand(1024, 1024).astype(np.float32), device=device),
              V.tensor(np.random.rand(1024, 1024).astype(np.float32), device=device))
        ab[0].realize()
        ab[1].realize()
        _WARM[str(device)] = ab
    for _ in range(rounds):
        V.matmul(ab[0], ab[1]).realize()


def run_all(device) -> list[Sample]:
    rng = np.random.default_rng(0)
    out: list[Sample] = []

    # -- transfers ----------------------------------------------------------
    for mib in (1, 4, 16):
        n = mib * 1024 * 1024 // 4
        host = rng.random(n, dtype=np.float32)
        nbytes = float(n * 4)

        t0 = time.perf_counter()
        for _ in range(10):
            V.tensor(host, device=device)
        up = (time.perf_counter() - t0) * 1e2  # ms per iteration
        out.append(Sample(f"upload {mib} MiB", "transfer", wall_min=up,
                          bytes_moved=nbytes, gbps=nbytes / 1e9 / (up / 1e3)))

        dev = V.tensor(host, device=device)
        t0 = time.perf_counter()
        for _ in range(10):
            dev.numpy()
        down = (time.perf_counter() - t0) * 1e2
        out.append(Sample(f"download {mib} MiB", "transfer", wall_min=down,
                          bytes_moved=nbytes, gbps=nbytes / 1e9 / (down / 1e3)))

    # -- dispatch overhead --------------------------------------------------
    tiny = V.tensor(np.ones((1,), dtype=np.float32), device=device)
    out.append(measure("dispatch 1 element", "overhead",
                       lambda: V.relu(tiny).numpy(), device=device))

    # -- elementwise --------------------------------------------------------
    for n in (256, 1024, 2048):
        x = V.tensor(rng.random((n, n), dtype=np.float32), device=device)
        nbytes = float(n * n * 4)
        # Elementwise reads once and writes once.
        out.append(measure(f"relu {n}x{n}", "elementwise",
                           lambda x=x: V.relu(x).numpy(), bytes_moved=nbytes, device=device))
        out.append(measure(f"exp {n}x{n}", "elementwise",
                           lambda x=x: V.exp(x).numpy(), bytes_moved=nbytes, device=device))

    # -- f16 against f32, at a size that saturates memory bandwidth ----------
    #
    # The point of f16 is traffic, so it is measured where traffic is the
    # constraint: below saturation the two dtypes sit at different places on the
    # bandwidth curve and the ratio says more about occupancy than about dtype.
    # Both operands stay resident and nothing is downloaded, so the kernel time
    # is the whole measurement.
    #
    # bytes_moved is deliberately left unset: this table's GB/s column is
    # TRANSFER bandwidth, derived from wall minus GPU time, and these rows
    # transfer nothing. Passing the traffic the kernel moves would print a
    # number computed from bytes that never crossed the bus.
    for label, npdt in (("f32", np.float32), ("f16", np.float16)):
        n = 1 << 24
        a = V.tensor(rng.standard_normal(n).astype(npdt), device=device)
        b = V.tensor(rng.standard_normal(n).astype(npdt), device=device)
        a.realize()
        b.realize()
        out.append(measure(f"mul {label} 2^24", "elementwise",
                           lambda a=a, b=b: (a * b).realize(), device=device))

    # -- f16 against f32 in a GEMM ------------------------------------------
    #
    # Tracked because f16 is currently SLOWER here, not faster: the vectorised
    # tile load is f32-only, so f16 falls back to scalar loads. Measured at
    # 2048^3, f32 runs 6.74 ms with vec4 and 9.89 ms without, and f16 runs
    # 9.74 ms -- so the gap is the vectorisation, and at equal vectorisation
    # f16 gains only a few per cent, because a tiled GEMM is compute-bound
    # rather than bandwidth-bound. This pair is here to notice if that changes.
    for label, npdt in (("f32", np.float32), ("f16", np.float16)):
        sz = 1024
        a = V.tensor((rng.standard_normal((sz, sz)) * 0.1).astype(npdt), device=device)
        b = V.tensor((rng.standard_normal((sz, sz)) * 0.1).astype(npdt), device=device)
        a.realize()
        b.realize()
        out.append(measure(f"matmul {label} 1024^3", "gemm",
                           lambda a=a, b=b: V.matmul(a, b).realize(), device=device))

    # -- reductions ---------------------------------------------------------
    for rows, cols in ((64, 4096), (1024, 1024), (4096, 256), (1, 1048576)):
        x = V.tensor(rng.random((rows, cols), dtype=np.float32), device=device)
        label = f"{rows}x{cols}"
        out.append(measure(f"sum {label}", "reduction",
                           lambda x=x: V.sum(x, [1]).numpy(), device=device))
        out.append(measure(f"max {label}", "reduction",
                           lambda x=x: V.amax(x, [1]).numpy(), device=device))
        out.append(measure(f"argmax {label}", "reduction",
                           lambda x=x: V.argmax(x, 1).numpy(), device=device))

    # -- softmax ------------------------------------------------------------
    for rows, cols in ((64, 4096), (1024, 1024), (4096, 256)):
        x = V.tensor(rng.random((rows, cols), dtype=np.float32), device=device)
        out.append(measure(f"softmax {rows}x{cols}", "softmax",
                           lambda x=x: V.softmax(x, -1).numpy(), device=device))
        out.append(measure(f"log_softmax {rows}x{cols}", "softmax",
                           lambda x=x: V.log_softmax(x, -1).numpy(), device=device))

    # -- GEMM, the current optimization target ------------------------------
    for n in (512, 1024):
        a = V.tensor(rng.random((n, n), dtype=np.float32), device=device)
        b = V.tensor(rng.random((n, n), dtype=np.float32), device=device)
        out.append(measure(f"gemm {n}x{n}x{n}", "gemm",
                           lambda a=a, b=b: V.matmul(a, b).numpy(), reps=12, device=device))

    # -- subgroup scaling ---------------------------------------------------
    # A width the device cannot pin is not a slow arm, it is an absent one. The
    # backend rejects an out-of-range size, and a device may refuse pinning for
    # compute outright (RADV RENOIR advertises subgroupSizeControl with an empty
    # requiredSubgroupSizeStages). Both used to be reported as ordinary results:
    # before the backend gained that second check the driver ignored the request
    # and the arm silently measured an UNPINNED run under a "wave32" label, which
    # is worse than a missing row. Skip what cannot be honoured, and say so.
    x = V.tensor(rng.random((1024, 1024), dtype=np.float32), device=device)
    caps = V.vulkan_capabilities(device.index)
    for sg, tag in ((0, "driver"), (32, "wave32"), (64, "wave64")):
        pinnable = caps["can_pin_subgroup_size"] and (
            caps["min_subgroup_size"] <= sg <= caps["max_subgroup_size"])
        if sg != 0 and not pinnable:
            print(f"  skipping [{tag}]: this device cannot pin a compute subgroup size of {sg}")
            continue
        V.vulkan_set_subgroup_override(sg)
        out.append(measure(f"sum 1024x1024 [{tag}]", "subgroup",
                           lambda x=x: V.sum(x, [1]).numpy(), device=device))
    V.vulkan_set_subgroup_override(0)

    return out


def pipeline_resources() -> list[dict]:
    """Compiler-reported resource usage, recorded alongside timings.

    These are first-class benchmark metadata, not diagnostics. They are also
    considerably more reproducible than timings: on this machine a 1024^3 GEMM
    varies by ~13% run to run while VGPR count is bit-exact every time, so a
    resource regression is detectable where a timing regression would be noise.
    """
    if not hasattr(V, "vulkan_pipeline_stats"):
        return []
    return [p for p in V.vulkan_pipeline_stats(0) if p.get("available")]


def metadata() -> dict:
    caps = V.vulkan_capabilities(0)
    return {
        "gpu": V.vulkan_device_names()[0],
        "subgroup_size": caps["subgroup_size"],
        "min_subgroup_size": caps["min_subgroup_size"],
        "max_subgroup_size": caps["max_subgroup_size"],
        "subgroup_policy": "driver-selected (override available)",
        "total_memory_bytes": caps["total_memory_bytes"],
        "max_shared_memory_bytes": caps["max_shared_memory_bytes"],
        "global_float_atomics": caps["global_float_atomics"],
        # Validation layers cost real time on every submission and are ON BY
        # DEFAULT. Recording the setting is what makes a wall-clock comparison
        # against this baseline interpretable: measured on an RX 5600M, the same
        # suite reports 14 regressions past 15% with layers on and 2 with them
        # off, and twelve of those fourteen are wall-clock only.
        "validation_layers": os.environ.get("VKML_VULKAN_VALIDATION", "1") != "0",
    }


def print_table(samples: list[Sample]) -> None:
    print(f"\n{'category':<12} {'benchmark':<28} {'GPU min':>10} {'sd':>7} "
          f"{'wall':>9} {'transfer':>9} {'GB/s':>8}")
    print("-" * 90)
    for s in samples:
        gpu = f"{s.gpu_min:.3f}ms" if s.gpu_min else "-"
        sd = f"{s.gpu_sd:.3f}" if s.gpu_sd else "-"
        transfer = f"{s.transfer_ms:.3f}ms" if s.transfer_ms else "-"
        gbps = f"{s.gbps:.2f}" if s.gbps else "-"
        print(f"{s.category:<12} {s.name:<28} {gpu:>10} {sd:>7} "
              f"{s.wall_min:8.3f}ms {transfer:>9} {gbps:>8}")
    print()


def noise_floor(runs: int) -> dict[str, float]:
    """Re-run this program `runs` times and report each benchmark's own spread.

    WHY A SEPARATE PROCESS PER RUN, and why `gpu_sd` is not a substitute.

    `gpu_sd` is the dispersion of the repeats taken inside ONE run, with the
    device already open, the pipelines already built and the clocks already
    where they settled. It says nothing about what changes between runs, and
    that turns out to be almost everything: measured over ten runs on an
    RX 5600M with the validation layers off, between-run dispersion exceeded
    the within-run figure by a median of 5.4x and by up to 222x, and was more
    than twice as large for 23 of 34 benchmarks. `exp 1024x1024` reports a
    within-run sd near 4% and moves 165% between runs.

    Using `gpu_sd` to judge a regression would therefore call a benchmark
    quiet precisely where it is loudest. Only repeated runs measure the
    quantity a comparison actually needs.

    Not every benchmark is noisy, which is the point of measuring rather than
    assuming: `dispatch 1 element` matches its within-run figure almost exactly
    (3.1% against 3.3%), so a change there means something.
    """
    import subprocess
    import tempfile

    per_name: dict[tuple[str, str], list[float]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(runs):
            path = os.path.join(tmp, f"run{i}.json")
            # `--runs` is deliberately NOT forwarded: the child takes one
            # measurement, and this loop is the repetition.
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--json", path],
                capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not os.path.exists(path):
                print(f"  run {i + 1}/{runs} failed:\n{proc.stderr.strip()[-500:]}")
                continue
            with open(path) as f:
                for s in json.load(f)["samples"]:
                    for field in ("gpu_min", "wall_min"):
                        if s.get(field):
                            per_name.setdefault((s["name"], field), []).append(s[field])
            print(f"  run {i + 1}/{runs} done")

    out: dict[tuple[str, str], float] = {}
    for key, values in per_name.items():
        if len(values) >= 2:
            med = statistics.median(values)
            if med > 0:
                out[key] = statistics.pstdev(values) / med
    return out


def check(samples: list[Sample], baseline_path: str, threshold: float) -> int:
    with open(baseline_path) as f:
        baseline = json.load(f)
    prior = {s["name"]: s for s in baseline["samples"]}

    if baseline.get("metadata", {}).get("gpu") != metadata()["gpu"]:
        print(f"NOTE: baseline was recorded on {baseline['metadata'].get('gpu')!r}, "
              f"this is {metadata()['gpu']!r}; comparison is not meaningful.")
        return 0

    # A baseline recorded without validation layers cannot be compared against a
    # run with them, and the difference is not small: twelve of fourteen
    # warnings in one measured run were this and nothing else. Older baselines
    # predate the field, so an absent value is treated as unknown and reported
    # rather than assumed either way.
    here = metadata()["validation_layers"]
    there = baseline.get("metadata", {}).get("validation_layers")
    if there is None:
        print("NOTE: this baseline predates the validation_layers field, so wall-clock "
              f"comparisons may be measuring layer overhead. This run has them "
              f"{'ON' if here else 'OFF'}; re-record the baseline to remove the doubt.")
    elif there != here:
        print(f"NOTE: baseline recorded with validation layers "
              f"{'ON' if there else 'OFF'}, this run has them "
              f"{'ON' if here else 'OFF'}. WALL-CLOCK COMPARISONS ARE NOT MEANINGFUL "
              f"across that difference; kernel times are.")

    warnings = 0
    suppressed = 0

    # Resource regressions are checked FIRST and with a tight threshold,
    # because these counters are exact. A VGPR increase is a real change in the
    # compiled kernel, never measurement noise -- Stage 5's regression (an
    # optimisation that made the kernel slower) was exactly this, and would have
    # been caught here rather than by benchmarking.
    prior_pipes = {p["name"]: p for p in baseline.get("pipelines", [])}
    for r in pipeline_resources():
        old = prior_pipes.get(r["name"])
        if old is None:
            continue
        for field in ("vgprs", "sgprs", "lds_bytes", "instructions", "scratch_bytes"):
            before, now = old.get(field, 0), r.get(field, 0)
            if before and now != before:
                direction = "increased" if now > before else "decreased"
                print(f"RESOURCE  {r['name']:<40} {field:<14} {before} -> {now} ({direction})")
                warnings += 1
        # Occupancy falling is always worth a warning; rising never is.
        b_w, n_w = old.get("waves_per_simd", 0), r.get("waves_per_simd", 0)
        if b_w and n_w and n_w < b_w:
            print(f"RESOURCE  {r['name']:<40} {'occupancy':<14} {b_w} -> {n_w} waves/SIMD")
            warnings += 1

    for s in samples:
        old = prior.get(s.name)
        if old is None:
            continue
        # Kernel and wall clock are compared separately so a transfer-path
        # change is never mistaken for a kernel regression.
        # A benchmark's own run-to-run spread, if the baseline recorded one.
        # Anything within NOISE_SIGMA of it is reported but not counted: a
        # warning nobody can act on trains people to ignore all of them, and
        # 22 of 34 benchmarks here exceed the flat 15% threshold on noise alone.
        for field, label, sd_field in (("gpu_min", "kernel", "gpu_run_sd"),
                                       ("wall_min", "wall", "wall_run_sd")):
            before, now = old.get(field, 0.0), getattr(s, field)
            if before <= 0 or now <= 0:
                continue
            delta = (now - before) / before
            if delta <= threshold:
                continue
            # Each measure is judged against ITS OWN floor: wall time carries
            # transfer and host cost that kernel time does not, and is reliably
            # the noisier, so borrowing the kernel figure would under-report it.
            run_sd = float(old.get(sd_field, 0.0) or 0.0)
            noisy = run_sd > 0 and delta <= NOISE_SIGMA * run_sd
            tag = "noise " if noisy else "WARN  "
            detail = (f"  [within {NOISE_SIGMA:g}x this benchmark's own "
                      f"{run_sd:.1%} run-to-run spread]" if noisy else "")
            print(f"{tag}{s.name:<28} {label:<7} {before:.3f} -> {now:.3f} ms "
                  f"({delta:+.1%}){detail}")
            if noisy:
                suppressed += 1
            else:
                warnings += 1

    if not any(s.get("gpu_run_sd") or s.get("wall_run_sd") for s in prior.values()):
        print("\nNOTE: this baseline carries no run-to-run noise floor, so every "
              "change is judged against the flat threshold alone. Re-record it "
              "with --runs N to make the warnings interpretable.")
    print(f"\nregression check: {warnings} warning(s) beyond {threshold:.0%}"
          + (f", {suppressed} within measured noise" if suppressed else ""))
    # Deliberately always 0: performance is noisy on shared machines and must
    # never block a merge. The signal is the warning, not the exit code.
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH", help="write results as JSON")
    ap.add_argument("--check", metavar="BASELINE", help="compare against a baseline and warn")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--runs", type=int, default=1, metavar="N",
                    help="re-run N times in separate processes and record each "
                         "benchmark's run-to-run spread; use when recording a "
                         "baseline, so later comparisons can tell a regression "
                         "from noise")
    args = ap.parse_args()

    if not V.has_vulkan or not V.vulkan_available():
        print("no Vulkan device available")
        return 0

    V.set_log_level(V.LogLevel.WARN)
    V.init_vulkan(0)
    V.vulkan_set_profiling(True)
    device = V.device("vulkan:0")

    meta = metadata()
    print(f"vkml GPU baseline\ndevice: {meta['gpu']}")
    print(f"subgroup: {meta['subgroup_size']} (range {meta['min_subgroup_size']}"
          f"-{meta['max_subgroup_size']}, policy: {meta['subgroup_policy']})")

    samples = run_all(device)
    print_table(samples)

    if args.runs > 1:
        print(f"\nmeasuring the run-to-run noise floor over {args.runs} runs")
        floor = noise_floor(args.runs)
        for s in samples:
            s.gpu_run_sd = floor.get((s.name, "gpu_min"), 0.0)
            s.wall_run_sd = floor.get((s.name, "wall_min"), 0.0)
        loud = sorted(((v, k) for k, v in floor.items()), reverse=True)[:6]
        print("\n  noisiest measurements (run-to-run sd, relative to median):")
        for v, (name, field) in loud:
            print(f"    {name:<30} {field:<9} {v:>7.1%}")

    resources = pipeline_resources()
    if resources:
        print(f"{'pipeline':<44} {'VGPR':>5} {'SGPR':>5} {'LDS':>7} {'waves':>6} "
              f"{'instr':>7} {'scratch':>8}")
        print("-" * 88)
        for r in sorted(resources, key=lambda x: x["name"]):
            print(f"{r['name']:<44} {r['vgprs']:>5} {r['sgprs']:>5} {r['lds_bytes']:>7} "
                  f"{r['waves_per_simd']:>6} {r['instructions']:>7} {r['scratch_bytes']:>8}")
        print()

    stats = V.vulkan_stats(0)
    print(f"allocator: {stats['reserved_bytes']/2**20:.0f} MiB reserved in "
          f"{stats['block_count']} block(s), peak {stats['peak_in_use_bytes']/2**20:.1f} MiB, "
          f"{stats['device_allocations']} device allocations for "
          f"{stats['total_allocations']} tensors")
    print(f"execution: {stats['submissions']} submissions, {stats['dispatches']} dispatches, "
          f"{stats['pipelines']} pipelines")

    if args.json:
        with open(args.json, "w") as f:
            # The conditions, not just the numbers. A baseline that records a
            # GPU model and no clock state, driver or commit cannot answer "is
            # this still comparable" -- which is how a 38.6% clock artefact
            # became a tracked regression.
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
            from check_baselines import stamp
            recorded = stamp()
            recorded["driver"] = meta.get("gpu", "unknown")
            recorded["warmed"] = True     # measure() warms before every timing
            json.dump({"recorded": recorded, "metadata": meta, "allocator": stats,
                       "samples": [asdict(s) for s in samples],
                       "pipelines": resources}, f, indent=2)
        print(f"\nwrote {args.json}")

    if args.check:
        return check(samples, args.check, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
