#!/usr/bin/env python3
"""Submission-model latency: what one realise costs, and where that cost sits.

WHY THIS EXISTS
---------------
`docs/SMALL-STEP-LATENCY.md` decomposed the per-realise cost by hand, in a
session, with nothing committed. Two of its conclusions were wrong and neither
could be re-run to find out:

  1. It was measured with the Vulkan VALIDATION LAYERS ENABLED. They are on by
     default -- `vulkan_backend()` reads VKML_VULKAN_VALIDATION defaulting to
     true -- so a measurement takes them unless it acts. MEASUREMENT-AUDIT rule
     5 says never to. Measured here: they cost 39% of a one-node realise.

  2. It attributed the marginal per-node cost to the HOST, by subtracting a
     graph of views from a graph of compute nodes. A view graph issues no
     dispatches, so that subtraction does not cancel GPU time -- there was none
     to cancel -- and the difference carried every GPU cost with it.

Both are methodology, not arithmetic, which is why they survived review and why
the fix is a committed tool rather than a corrected paragraph.

WHAT IT MEASURES
----------------
A dependent chain of N trivial kernels is realised as one submission. Sweeping N
and regressing against it separates the two costs that matter, and the FIXED
cost cancels in the slope:

    wall(N) = fixed + marginal * N

Run twice, once per arm, and the same regression on the GPU `submit` window says
how much of `marginal` is the GPU rather than the host. Rule 4 forbids comparing
a profiled wall clock against an unprofiled one, so the arms are never
subtracted from each other -- each is regressed on its own and only the SLOPES
are compared.

    python bench/latency_bench.py
    python bench/latency_bench.py --json latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Set BEFORE vkml is imported, and before any backend is created: the switch is
# read when `vulkan_backend()` first builds the device, and every caller caches
# what it read. Setting it afterwards has no effect and would silently leave the
# layers on -- which is the failure this file exists to prevent.
#
# setdefault, not assignment: someone deliberately measuring the cost of the
# layers must be able to ask for them, and the report says which arm ran.
os.environ.setdefault("VKML_VULKAN_VALIDATION", "0")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vkml as V  # noqa: E402

#: Chain lengths swept. Spread over a decade so the regression has leverage, and
#: starting at 1 because the intercept is the number the submission model is
#: judged on.
NODE_COUNTS = (1, 2, 4, 8, 16, 32)

#: Elements per tensor. Deliberately tiny: this measures the cost of GETTING to
#: a kernel, so the kernel itself must not be part of the answer. 64 floats is
#: one workgroup and below any bandwidth effect.
NUMEL = 64


def validation_layers_observed() -> bool | None:
    """Whether the backend actually consulted the switch, and what it saw.

    Read from `vkml.configuration()` rather than `os.environ`, because those are
    different questions. The environment says what was ASKED FOR; configuration
    says what the library READ, at the moment it read it. `bench/gpu_bench.py`
    records the former in its baselines, so a baseline taken by a process that
    set the variable after import claims an arm it did not run.

    None when the backend never consulted it -- which means no Vulkan device was
    created, and the caller should not report an arm at all.
    """
    for entry in V.configuration():
        if entry["name"] == "VKML_VULKAN_VALIDATION":
            return entry["value"][:1] != "0" if entry["set"] else True
    return None


def warm(device) -> None:
    """Raise the GPU clock immediately before timing.

    Not once at the start of the run: this machine parks at 400 MHz and falls
    back within the time a few transfers take, so anything measured after an
    unrelated phase is measured cold. MEASUREMENT-AUDIT's closing section has
    the sweep that established this -- a factor of 1.8 on a small dispatch.
    """
    a = V.tensor(np.ones((512, 512), dtype=np.float32), device=device)
    for _ in range(30):
        V.realize(a @ a)
    V.vulkan_synchronize()


def chain(x, n_ops: int):
    """A dependent chain of `n_ops` compute nodes.

    `relu` because it is the cheapest kernel with trivial addressing -- the same
    control ADR 0011 used to show three kernels were addressing-bound rather
    than memory-bound. Dependent rather than independent, so the barrier the
    executor emits between every pair is doing real ordering work and the
    dispatches cannot overlap; that is the case the current design always pays.
    """
    for _ in range(n_ops):
        x = x.relu()
    return x


def measure_wall(x, n_ops: int, reps: int) -> float:
    """Minimum wall time, in microseconds, of one realise of an `n_ops` chain."""
    best = float("inf")
    for _ in range(reps):
        root = chain(x, n_ops)  # built outside the timed region on purpose
        t0 = time.perf_counter()
        V.realize(root)
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def measure_gpu(x, n_ops: int, reps: int) -> float:
    """Minimum GPU time, in microseconds, for the same chain.

    The `submit` window, never the sum of the per-dispatch entries: those end at
    ALL_COMMANDS, a global drain point, so summing them multiply-counts any
    dispatches that overlapped (MEASUREMENT-AUDIT rule 3). `vulkan_submit_ms`
    is the accessor that knows this.

    The synchronise is NOT optional and is not there to steady the clock.
    Submission is asynchronous (ADR 0012), so when `realize()` returns, the work
    it queued has not necessarily run and its timestamps have not been read
    back. Reading the profile here without waiting returns the PREVIOUS
    submission's intervals -- which, in a sweep that changes the chain length,
    silently reports the wrong N and understated this by 3x when it was first
    written that way.
    """
    best = float("inf")
    for _ in range(reps):
        root = chain(x, n_ops)
        V.realize(root)
        V.vulkan_synchronize()
        ms = V.vulkan_submit_ms(V.vulkan_last_profile())
        if ms > 0:
            best = min(best, ms)
    return best * 1e3


def fit(xs, ys) -> tuple[float, float]:
    """Least-squares slope and intercept of ys against xs."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denominator = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator
    return slope, my - slope * mx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="Vulkan device index")
    parser.add_argument("--reps", type=int, default=40, help="repetitions per point")
    parser.add_argument("--json", help="also write the table here")
    args = parser.parse_args()

    if not V.has_vulkan or not V.vulkan_available():
        print("no Vulkan device: this benchmark measures the Vulkan submission model")
        return 1

    V.init_vulkan(args.device)
    device = V.device(f"vulkan:{args.device}")

    validation = validation_layers_observed()
    if validation:
        print("WARNING: validation layers are ON. MEASUREMENT-AUDIT rule 5 forbids "
              "benchmarking this way; the numbers below are inflated by roughly a third.")

    x = V.tensor(np.ones((NUMEL,), dtype=np.float32), device=device)
    V.realize(x)

    warm(device)
    V.vulkan_set_profiling(False)
    wall = [measure_wall(x, n, args.reps) for n in NODE_COUNTS]

    warm(device)
    V.vulkan_set_profiling(True)
    gpu = [measure_gpu(x, n, args.reps) for n in NODE_COUNTS]
    V.vulkan_set_profiling(False)

    wall_marginal, wall_fixed = fit(NODE_COUNTS, wall)
    gpu_marginal, gpu_fixed = fit(NODE_COUNTS, gpu)

    print(f"\n  {V.vulkan_device_names()[args.device]}")
    print(f"  validation layers: {'ON' if validation else 'off'}   "
          f"chain of relu, {NUMEL} elements, best of {args.reps}\n")
    print(f"  {'nodes':>6} {'wall us':>10} {'GPU us':>10} {'GPU/wall':>9}")
    for n, w, g in zip(NODE_COUNTS, wall, gpu):
        print(f"  {n:>6} {w:>10.2f} {g:>10.2f} {g / w:>9.2f}")

    # The GPU intercept is reported only when it is positive. A submission has
    # no fixed GPU cost to speak of -- just the timestamp bracket -- so the fit
    # regularly places it at or below zero, and printing "-7.74 us of GPU per
    # submission" invites a reader to believe a quantity that does not exist.
    gpu_fixed_note = f"{gpu_fixed:6.2f} us GPU" if gpu_fixed > 0 else "no fixed GPU cost"
    print(f"\n  per submission (fixed)   {wall_fixed:8.2f} us wall, {gpu_fixed_note}")
    print(f"  per node (marginal)      {wall_marginal:8.2f} us wall, {gpu_marginal:6.2f} us GPU")

    # Stated as a BOUND, not as a subtraction. The profiled arm writes two extra
    # timestamps per dispatch, so the profiler's own cost is per NODE and lands
    # in the slope -- subtracting the slopes would charge the host for it and can
    # (and did) produce a negative "host cost". The inequality survives that
    # contamination in the safe direction: the GPU slope is an upper bound on GPU
    # time, so if it still covers the whole wall slope, the host cannot be
    # contributing measurably.
    #
    # The direct measurement, from timing begin()..submit() with no timestamps
    # anywhere, agrees: 0.24 us of host per dispatch (docs/SMALL-STEP-LATENCY.md).
    if gpu_marginal >= wall_marginal:
        print("  => the marginal cost is GPU-side. An upper bound on GPU time already "
              "covers\n     the whole wall slope, so host work per node is not measurable here.")
    else:
        print(f"  => at most {wall_marginal - gpu_marginal:.2f} us/node is host work; "
              "the GPU slope is an\n     upper bound, so the true host share is smaller still.")

    print("\n  The fixed cost is the submission: a host round trip to the GPU and back.")
    print("  The marginal cost is one barrier-separated dispatch.\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "device": V.vulkan_device_names()[args.device],
                "validation_layers": validation,
                "numel": NUMEL,
                "node_counts": list(NODE_COUNTS),
                "wall_us": wall,
                "gpu_us": gpu,
                "fixed_us_wall": wall_fixed,
                "marginal_us_wall": wall_marginal,
                "marginal_us_gpu": gpu_marginal,
            }, f, indent=2)
        print(f"  wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
