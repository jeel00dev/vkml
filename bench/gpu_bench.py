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


@dataclass
class Sample:
    name: str
    category: str
    gpu_min: float = 0.0
    gpu_mean: float = 0.0
    gpu_sd: float = 0.0
    wall_min: float = 0.0
    wall_mean: float = 0.0
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


def measure(name, category, fn, reps=25, bytes_moved=0.0) -> Sample:
    """Times `fn`, separating GPU execution from everything else."""
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
                       lambda: V.relu(tiny).numpy()))

    # -- elementwise --------------------------------------------------------
    for n in (256, 1024, 2048):
        x = V.tensor(rng.random((n, n), dtype=np.float32), device=device)
        nbytes = float(n * n * 4)
        # Elementwise reads once and writes once.
        out.append(measure(f"relu {n}x{n}", "elementwise",
                           lambda x=x: V.relu(x).numpy(), bytes_moved=nbytes))
        out.append(measure(f"exp {n}x{n}", "elementwise",
                           lambda x=x: V.exp(x).numpy(), bytes_moved=nbytes))

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
                           lambda a=a, b=b: (a * b).realize()))

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
                           lambda a=a, b=b: V.matmul(a, b).realize()))

    # -- reductions ---------------------------------------------------------
    for rows, cols in ((64, 4096), (1024, 1024), (4096, 256), (1, 1048576)):
        x = V.tensor(rng.random((rows, cols), dtype=np.float32), device=device)
        label = f"{rows}x{cols}"
        out.append(measure(f"sum {label}", "reduction",
                           lambda x=x: V.sum(x, [1]).numpy()))
        out.append(measure(f"max {label}", "reduction",
                           lambda x=x: V.amax(x, [1]).numpy()))
        out.append(measure(f"argmax {label}", "reduction",
                           lambda x=x: V.argmax(x, 1).numpy()))

    # -- softmax ------------------------------------------------------------
    for rows, cols in ((64, 4096), (1024, 1024), (4096, 256)):
        x = V.tensor(rng.random((rows, cols), dtype=np.float32), device=device)
        out.append(measure(f"softmax {rows}x{cols}", "softmax",
                           lambda x=x: V.softmax(x, -1).numpy()))
        out.append(measure(f"log_softmax {rows}x{cols}", "softmax",
                           lambda x=x: V.log_softmax(x, -1).numpy()))

    # -- GEMM, the current optimization target ------------------------------
    for n in (512, 1024):
        a = V.tensor(rng.random((n, n), dtype=np.float32), device=device)
        b = V.tensor(rng.random((n, n), dtype=np.float32), device=device)
        out.append(measure(f"gemm {n}x{n}x{n}", "gemm",
                           lambda a=a, b=b: V.matmul(a, b).numpy(), reps=12))

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
                           lambda x=x: V.sum(x, [1]).numpy()))
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


def check(samples: list[Sample], baseline_path: str, threshold: float) -> int:
    with open(baseline_path) as f:
        baseline = json.load(f)
    prior = {s["name"]: s for s in baseline["samples"]}

    if baseline.get("metadata", {}).get("gpu") != metadata()["gpu"]:
        print(f"NOTE: baseline was recorded on {baseline['metadata'].get('gpu')!r}, "
              f"this is {metadata()['gpu']!r}; comparison is not meaningful.")
        return 0

    warnings = 0

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
        for field, label in (("gpu_min", "kernel"), ("wall_min", "wall")):
            before, now = old.get(field, 0.0), getattr(s, field)
            if before <= 0 or now <= 0:
                continue
            delta = (now - before) / before
            if delta > threshold:
                print(f"WARN  {s.name:<28} {label:<7} {before:.3f} -> {now:.3f} ms "
                      f"({delta:+.1%})")
                warnings += 1
    print(f"\nregression check: {warnings} warning(s) beyond {threshold:.0%}")
    # Deliberately always 0: performance is noisy on shared machines and must
    # never block a merge. The signal is the warning, not the exit code.
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH", help="write results as JSON")
    ap.add_argument("--check", metavar="BASELINE", help="compare against a baseline and warn")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
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
            json.dump({"metadata": meta, "allocator": stats,
                       "samples": [asdict(s) for s in samples],
                       "pipelines": resources}, f, indent=2)
        print(f"\nwrote {args.json}")

    if args.check:
        return check(samples, args.check, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
