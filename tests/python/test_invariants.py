"""Property-based invariants and permanent regression cases.

Every test here encodes something the project LEARNED, usually the hard way.
The distinction from the rest of the suite matters:

  test_ops_vs_torch.py     checks VALUES against an oracle
  test_vulkan_kernels.py   checks KERNELS against the CPU backend
  this file                checks PROPERTIES that must hold for every input,
                           and pins every past bug so it cannot return silently

A property test states a law -- "changing the tile geometry cannot change a
bit" -- and then tries to break it with many inputs. That catches whole classes
of error an example-based test cannot, because the failures this project has
actually hit were never the inputs anyone thought to write down.

Sources are named per test. When one fails, the comment tells you what was
believed, what turned out to be true, and where it is written up.
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import vkml as V  # noqa: E402
from conftest import TOLERANCES  # noqa: E402
from vkvalidate import (VULKAN_DEVICE, gpu_device, requires_radv,  # noqa: E402
                        requires_vulkan)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Every dispatch switch, pinned to its default.
#
# These tests spawn subprocesses and inherit os.environ, so an ambient
# VKML_* set by the developer or by another test run would silently change
# what the CONTROL arm computes -- and a comparison against a contaminated
# control is worthless. Stage 6.5 produced a spurious 1.4x speedup exactly this
# way; forcing the baseline here is the same lesson applied to the test suite.
# A test that varies a switch overrides its own entry.
_NEUTRAL_BASE = {
    "VKML_GEMV": "OFF",
    "VKML_GEMM_SPLITK": "OFF",
    "VKML_GEMM_KERNEL": "",
    "VKML_GEMM_TILE": "",
    "VKML_GEMM_NOVEC": "",
    "VKML_GEMM_NOLDSVEC": "",
    "VKML_GEMM_DB": "",
    "VKML_VULKAN_NO_PIPELINE_STATS": "",
}


def _env(overrides: dict[str, str]) -> dict[str, str]:
    """Ambient environment, with every dispatch switch pinned, then overrides."""
    e = {**os.environ, **_NEUTRAL_BASE}
    # VKML_GEMM_BLOCK is absent rather than empty by default: the backend keys
    # on its PRESENCE, not its value.
    e.pop("VKML_GEMM_BLOCK", None)
    e.pop("VKML_GEMM_SPLITK_SPLITS", None)
    e.update(overrides)
    return e
CPU = V.device("cpu")


# ---------------------------------------------------------------------------
# 1. The carry-stack equivalence theorem (docs/SPLIT_K_DESIGN.md 2.3-2.4)
#
# Pure host-side arithmetic, no GPU. This is the PROOF CHECK for the theorem
# split-K depends on: folding aligned power-of-two chunks and then folding the
# partials with the same algorithm reproduces the unsplit tree exactly.
#
# It lives in the permanent suite rather than a scratch script because the
# theorem is load-bearing -- if it ever stops holding, split-K silently stops
# being bit-identical and every gate downstream of it becomes meaningless.
# ---------------------------------------------------------------------------


def _carry_fold(vals: np.ndarray) -> np.float32:
    """vkML's binary-counter push/drain, mirroring shaders/gemm_reg.comp."""
    stack: dict[int, np.float32] = {}
    count = 0
    for v in vals:
        block = np.float32(v)
        c, level = count, 0
        while c & 1:
            block = np.float32(block + stack[level])
            c >>= 1
            level += 1
        stack[level] = block
        count += 1
    total = np.float32(0.0)
    level = 0
    while (count >> level) != 0:
        if (count >> level) & 1:
            total = np.float32(total + stack[level])
        level += 1
    return total


def _split_fold(vals: np.ndarray, chunk: int) -> np.float32:
    parts = [_carry_fold(vals[i : i + chunk]) for i in range(0, len(vals), chunk)]
    return _carry_fold(np.array(parts, dtype=np.float32))


@pytest.mark.parametrize("tiles", [1, 2, 3, 5, 7, 8, 12, 15, 16, 17, 31, 33, 64, 129, 256])
def test_carry_stack_equivalence_holds_for_power_of_two_chunks(tiles):
    """Aligned chunking reproduces the unsplit fold BIT-EXACTLY, for any tile count.

    Note `tiles` deliberately includes non-powers-of-two: naive intuition says
    those should break, and the theorem says they do not. That is the case
    worth pinning.
    """
    rng = np.random.default_rng(tiles)
    vals = (rng.standard_normal(tiles) * 1e3).astype(np.float32)
    reference = _carry_fold(vals)
    q = 0
    while (1 << q) <= tiles:
        chunk = 1 << q
        got = _split_fold(vals, chunk)
        assert got.tobytes() == reference.tobytes(), (
            f"chunk={chunk} tiles={tiles}: {got!r} != {reference!r}"
        )
        q += 1


@pytest.mark.parametrize("chunk", [3, 5, 6, 7, 12])
def test_carry_stack_equivalence_fails_for_non_power_of_two_chunks(chunk):
    """The NEGATIVE control for the test above.

    Without this, the positive test could pass because the fold is insensitive
    to chunking altogether, which would make it vacuous. The alignment
    condition has to be shown to matter.

    Stated STATISTICALLY, and that is not a stylistic choice. Two different
    associations of the same floating-point sum frequently agree by chance on
    any given input -- an early single-input version of this test failed for
    chunk=5 for exactly that reason. A negative control asserting "these differ"
    must therefore quantify over inputs: a misaligned chunking must break
    bit-identity for a substantial FRACTION of random draws, not for one
    hand-picked vector.
    """
    rng = np.random.default_rng(7 + chunk)
    trials, mismatches = 60, 0
    for _ in range(trials):
        vals = (rng.standard_normal(97) * 1e3).astype(np.float32)
        if _split_fold(vals, chunk).tobytes() != _carry_fold(vals).tobytes():
            mismatches += 1
    assert mismatches > trials // 4, (
        f"chunk={chunk} matched the aligned fold in {trials - mismatches}/{trials} draws; "
        "the alignment condition looks unnecessary, which contradicts the theorem"
    )


def test_flat_sequential_reduction_would_break_bit_identity():
    """Why the reduce shader must NOT be a flat sum.

    llama.cpp, CUTLASS and Tensile all reduce split-K partials sequentially.
    Copying that design would have cost bit-identity, and nothing in the
    value-based suite would have caught it -- a sequential fold is still
    accurate to well within tolerance. Pinned so the shader is never
    "simplified" into one.

    Statistical for the same reason as the test above.
    """
    rng = np.random.default_rng(11)
    trials, mismatches = 60, 0
    for _ in range(trials):
        vals = (rng.standard_normal(64) * 1e3).astype(np.float32)
        parts = [_carry_fold(vals[i : i + 8]) for i in range(0, 64, 8)]
        flat = np.float32(0.0)
        for part in parts:
            flat = np.float32(flat + part)
        if flat.tobytes() != _carry_fold(vals).tobytes():
            mismatches += 1
    assert mismatches > trials // 4, (
        f"a flat sequential reduction matched the carry-stack fold in "
        f"{trials - mismatches}/{trials} draws -- the reduce shader's algorithm "
        "would then be unconstrained, which contradicts docs/SPLIT_K_DESIGN.md 3"
    )


# ---------------------------------------------------------------------------
# 2. Dispatch-path independence
#
# vkML has accumulated many switches that change HOW a result is computed while
# promising not to change WHAT is computed: kernel choice, tile geometry,
# register block, vectorisation, LDS widening, double buffering, split-K.
#
# Each is individually tested elsewhere. The property here is the one that
# matters and that no single test states: ALL of them, together, are
# bit-neutral. Stage 6.5 shipped a 1.4x speedup that turned out to be a damaged
# control arm; a property like this is what makes that class of error loud.
# ---------------------------------------------------------------------------

_SHAPES = [(1, 1, 1), (7, 33, 5), (32, 32, 32), (64, 128, 64), (127, 129, 31), (256, 512, 128)]


@functools.lru_cache(maxsize=None)
def _hash_matmuls_cached(env_items: tuple[tuple[str, str], ...]) -> str:
    return _hash_matmuls(dict(env_items))


def _hash_matmuls(env: dict[str, str]) -> str:
    """Runs the same matmuls in a SUBPROCESS with `env` applied.

    A subprocess is required, not a convenience: every one of these switches is
    read once into a function-local static on first use, so they cannot be
    changed within a live process.
    """
    script = "\n".join([
        "import sys,hashlib",
        "sys.path.insert(0,'python')",
        "import numpy as np, vkml as V",
        "V.set_log_level(V.LogLevel.ERROR)",
        "V.init_vulkan(0)",
        "d=V.device('vulkan:0')",
        "rng=np.random.default_rng(4242)",
        "h=hashlib.sha256()",
        f"shapes={_SHAPES!r}",
        "for (m,k,n) in shapes:",
        "    A=rng.standard_normal((m,k),dtype=np.float32)",
        "    B=rng.standard_normal((k,n),dtype=np.float32)",
        "    h.update(V.matmul(V.tensor(A,device=d),V.tensor(B,device=d)).numpy().tobytes())",
        "print(h.hexdigest())",
    ])
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, env=_env(env),
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, f"env={env} failed:\n{out.stderr[-2000:]}"
    return out.stdout.strip().splitlines()[-1]


# Every switch that must be bit-neutral, with the stage that introduced it.
_NEUTRAL_ENVS = [
    pytest.param({}, id="default"),
    pytest.param({"VKML_GEMM_KERNEL": "naive"}, id="naive-stage1"),
    pytest.param({"VKML_GEMM_NOVEC": "1"}, id="no-vec4-stage6"),
    pytest.param({"VKML_GEMM_NOLDSVEC": "1"}, id="no-lds-vec-stage6.5"),
    pytest.param({"VKML_GEMM_DB": "1"}, id="double-buffered-stage7"),
    pytest.param({"VKML_GEMM_BLOCK": "4x2"}, id="reg-block-4x2-stage8"),
    pytest.param({"VKML_GEMM_BLOCK": "4x4"}, id="reg-block-4x4-stage8"),
    pytest.param({"VKML_GEMM_TILE": "m"}, id="tile-64x32-m3.01"),
    pytest.param({"VKML_GEMM_TILE": "l"}, id="tile-64x64-m3.01"),
    pytest.param({"VKML_GEMM_SPLITK": "OFF"}, id="splitk-off-m3.03"),
    pytest.param({"VKML_GEMM_SPLITK": "FORCED", "VKML_GEMM_SPLITK_SPLITS": "2"}, id="splitk-2"),
    pytest.param({"VKML_GEMM_SPLITK": "FORCED", "VKML_GEMM_SPLITK_SPLITS": "5"}, id="splitk-5"),
    pytest.param({"VKML_GEMM_SPLITK": "FORCED", "VKML_GEMM_SPLITK_SPLITS": "16"}, id="splitk-16"),
    pytest.param({"VKML_VULKAN_NO_PIPELINE_STATS": "1"}, id="no-capture-statistics-m3.r1"),
]


@requires_vulkan
@pytest.mark.slow
@pytest.mark.parametrize("env", _NEUTRAL_ENVS)
def test_dispatch_path_is_bit_neutral(env):
    """Changing HOW a matmul is computed must not change WHAT it computes.

    THE EXACT SCOPE OF THIS LAW: bit-identity holds among kernels that fold K in
    blocks of `kPairwiseBlock` (= 32) with the carry stack. `gemm_naive` and
    `gemm_reg` both do, so every switch between them is bit-neutral. `gemm_tiled`
    does NOT -- it folds in blocks of TILE = 16 -- and is covered separately by
    test_tiled_kernel_has_a_different_fold_tree below.

    An early version of this test asserted bit-identity for `tiled` too and
    failed, correctly. If this test fails for a newly added kernel, establish
    which fold tree that kernel implements BEFORE weakening the assertion: a
    kernel that folds in 32-blocks and still disagrees is a real bug.
    """
    baseline = _hash_matmuls_cached(())
    assert _hash_matmuls(env) == baseline, (
        f"dispatch switch {env} changed the result; it must not"
    )


@requires_vulkan
@pytest.mark.slow
def test_split_k_bit_identity_over_randomised_shapes():
    """Property: split-K is bit-identical for ARBITRARY shapes, not just nice ones.

    Gate 1 of M3-03 used a fixed matrix of shapes. This draws shapes at random
    including prime and non-power-of-two K, which is where the alignment lemma
    is least obvious.
    """
    rng = np.random.default_rng(20260727)
    shapes = [
        (int(rng.integers(1, 200)), int(rng.integers(1, 3000)), int(rng.integers(1, 200)))
        for _ in range(25)
    ]
    script = (
        "import sys,hashlib;sys.path.insert(0,'python');"
        "import numpy as np,vkml as V;"
        "V.set_log_level(V.LogLevel.ERROR);V.init_vulkan(0);d=V.device('vulkan:0');"
        "rng=np.random.default_rng(99);h=hashlib.sha256();"
        f"shapes={shapes!r};"
        "\nfor (m,k,n) in shapes:\n"
        "    A=rng.standard_normal((m,k),dtype=np.float32)\n"
        "    B=rng.standard_normal((k,n),dtype=np.float32)\n"
        "    h.update(V.matmul(V.tensor(A,device=d),V.tensor(B,device=d)).numpy().tobytes())\n"
        "print(h.hexdigest())"
    )

    def run(env):
        out = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                             env=_env(env), capture_output=True,
                             text=True, timeout=900)
        assert out.returncode == 0, out.stderr[-2000:]
        return out.stdout.strip().splitlines()[-1]

    reference = run({"VKML_GEMM_SPLITK": "OFF"})
    for splits in ("2", "3", "8", "32"):
        got = run({"VKML_GEMM_SPLITK": "FORCED", "VKML_GEMM_SPLITK_SPLITS": splits})
        assert got == reference, f"split-K with {splits} partitions changed the result"


@requires_vulkan
@pytest.mark.slow
def test_tiled_kernel_has_a_different_fold_tree():
    """`gemm_tiled` is NOT bit-compatible, and that is by design.

    Stage 4's kernel folds K in blocks of TILE = 16; every other kernel uses
    blocks of 32 to match `kPairwiseBlock` in src/backend/cpu/reduce.h. A
    different block size is a different reduction tree, so different bytes are
    correct behaviour, not a defect.

    Pinned in both directions. If it ever starts matching, someone has changed
    TILE to 32 -- fine, but the comparison arm it provides for Stage 4 would no
    longer be the kernel that was measured. If it starts disagreeing with the
    ORACLE, that is a real bug.
    """
    assert _hash_matmuls({"VKML_GEMM_KERNEL": "tiled"}) != _hash_matmuls_cached(())

    rng = np.random.default_rng(5150)
    A = rng.standard_normal((96, 320), dtype=np.float32)
    B = rng.standard_normal((320, 96), dtype=np.float32)
    cpu = V.matmul(V.tensor(A, device=CPU), V.tensor(B, device=CPU)).numpy()
    gpu = V.matmul(V.tensor(A, device=gpu_device()),
                   V.tensor(B, device=gpu_device())).numpy()
    scale = np.abs(A).sum(axis=1, keepdims=True) @ np.abs(B).sum(axis=0, keepdims=True)
    gamma = (32 + max(1, 320 // 16).bit_length()) * np.finfo(np.float32).eps
    assert np.all(np.abs(gpu - cpu) <= gamma * scale + np.finfo(np.float32).tiny)


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------


@requires_vulkan
@pytest.mark.parametrize("shape", [(64, 512, 64), (128, 1031, 96)])
def test_repeated_execution_is_byte_identical(shape):
    """Same inputs, same process, many runs -> identical bytes.

    Guards against any future use of atomics, non-deterministic work
    distribution, or uninitialised memory in a reduction.
    """
    m, k, n = shape
    dev = gpu_device()
    rng = np.random.default_rng(3)
    a = V.tensor(rng.standard_normal((m, k), dtype=np.float32), device=dev)
    b = V.tensor(rng.standard_normal((k, n), dtype=np.float32), device=dev)
    digests = {hashlib.sha256(V.matmul(a, b).numpy().tobytes()).hexdigest() for _ in range(25)}
    assert len(digests) == 1, f"non-deterministic: {len(digests)} distinct results"


@requires_vulkan
def test_long_running_stability():
    """400 consecutive matmuls of varying shape stay bit-stable and leak nothing.

    Catches state that accumulates across dispatches -- a workspace that is
    grown but never reused correctly, a pipeline cache key collision, a query
    pool that wraps.
    """
    dev = gpu_device()
    rng = np.random.default_rng(17)
    a = V.tensor(rng.standard_normal((96, 640), dtype=np.float32), device=dev)
    b = V.tensor(rng.standard_normal((640, 96), dtype=np.float32), device=dev)
    first = hashlib.sha256(V.matmul(a, b).numpy().tobytes()).hexdigest()
    for i in range(400):
        # Interleave other shapes so the workspace and pipeline cache churn.
        s = 32 + (i % 7) * 16
        x = V.tensor(rng.standard_normal((s, 128), dtype=np.float32), device=dev)
        y = V.tensor(rng.standard_normal((128, s), dtype=np.float32), device=dev)
        V.matmul(x, y).numpy()
        if i % 50 == 0:
            again = hashlib.sha256(V.matmul(a, b).numpy().tobytes()).hexdigest()
            assert again == first, f"drifted after {i} interleaved matmuls"


# ---------------------------------------------------------------------------
# 4. Boundary and pathological shapes
# ---------------------------------------------------------------------------


@requires_vulkan
@pytest.mark.parametrize(
    "m,k,n",
    [
        (1, 1, 1),        # scalar
        (1, 32, 1),       # exactly one K-tile
        (1, 33, 1),       # one tile plus one element -- partial-tile path
        (1, 31, 1),       # less than one K-tile
        (32, 32, 32),     # exactly one threadblock tile
        (33, 32, 33),     # one tile plus one row/col -- bounds-check path
        (31, 32, 31),     # under-full tile
        (1, 4096, 1),     # extreme K, minimal MN: deepest carry stack
        (4096, 1, 4096),  # K=1: shallowest possible fold
        (1, 1, 4096),     # outer product
        (4096, 1, 1),
        (257, 257, 257),  # prime-ish, nothing divides
    ],
)
def test_boundary_shapes_match_cpu_oracle(m, k, n):
    """Boundary shapes against the CPU oracle under the standard policy.

    K=1 and K=31 matter because they exercise a carry stack that never folds;
    K=33 because it is the first size with a partial trailing tile; 33 and 31 in
    M/N because they straddle the 32-wide threadblock tile in both directions.
    """
    rng = np.random.default_rng(m * 1000 + k * 10 + n)
    A = rng.standard_normal((m, k), dtype=np.float32)
    B = rng.standard_normal((k, n), dtype=np.float32)
    gpu = V.matmul(V.tensor(A, device=gpu_device()),
                   V.tensor(B, device=gpu_device())).numpy()
    cpu = V.matmul(V.tensor(A, device=CPU),
                   V.tensor(B, device=CPU)).numpy()
    assert gpu.shape == cpu.shape == (m, n)
    assert gpu.dtype == cpu.dtype == np.float32
    # Backward-error bound, not a relative tolerance: see docs/ARCHITECTURE.md 7.3
    # and the K=4096 investigation that showed a relative bound is the wrong model.
    scale = np.abs(A).sum(axis=1, keepdims=True) @ np.abs(B).sum(axis=0, keepdims=True).clip(
        min=np.finfo(np.float32).tiny
    )
    gamma = (32 + max(1, k // 32).bit_length()) * np.finfo(np.float32).eps
    assert np.all(np.abs(gpu - cpu) <= gamma * np.abs(scale) + np.finfo(np.float32).tiny)


# ---------------------------------------------------------------------------
# 5. Adversarial numerical inputs
# ---------------------------------------------------------------------------


@requires_vulkan
@pytest.mark.parametrize(
    "name,make",
    [
        ("cancelling", lambda r, s: np.where(
            (np.arange(s[0] * s[1]).reshape(s) % 2) == 0, 1e7, -1e7).astype(np.float32)),
        ("tiny", lambda r, s: (r.standard_normal(s) * 1e-30).astype(np.float32)),
        ("huge", lambda r, s: (r.standard_normal(s) * 1e20).astype(np.float32)),
        ("mixed-magnitude", lambda r, s: (
            r.standard_normal(s) * 10.0 ** r.integers(-20, 20, s)).astype(np.float32)),
        ("subnormal", lambda r, s: (r.standard_normal(s) * 1e-42).astype(np.float32)),
        ("exact-halves", lambda r, s: (r.integers(-8, 8, s) * 0.5).astype(np.float32)),
    ],
)
def test_adversarial_inputs_agree_with_cpu_ordering(name, make):
    """Pathological magnitudes must not make GPU and CPU DISAGREE STRUCTURALLY.

    These inputs are chosen to stress catastrophic cancellation, gradual
    underflow and overflow. The assertion is deliberately weak on value -- with
    1e7 - 1e7 cancellation no tolerance is meaningful -- and strong on
    structure: no NaN or infinity may appear that the CPU backend does not also
    produce. A GPU-only NaN means a real divergence (uninitialised shared
    memory, a wrong bound check), which is what this is looking for.
    """
    rng = np.random.default_rng(hash(name) % 2**31)
    A = make(rng, (64, 96))
    B = make(rng, (96, 64))
    gpu = V.matmul(V.tensor(A, device=gpu_device()),
                   V.tensor(B, device=gpu_device())).numpy()
    cpu = V.matmul(V.tensor(A, device=CPU),
                   V.tensor(B, device=CPU)).numpy()
    assert np.array_equal(np.isnan(gpu), np.isnan(cpu)), f"{name}: NaN pattern differs"
    assert np.array_equal(np.isinf(gpu), np.isinf(cpu)), f"{name}: infinity pattern differs"
    assert np.array_equal(np.signbit(gpu), np.signbit(cpu)) or not np.all(
        np.isfinite(gpu)
    ), f"{name}: sign pattern differs"


# ---------------------------------------------------------------------------
# 6. Measurement-infrastructure invariants (M3-R1)
#
# The profiler is part of the evidence chain, so its correctness is a testable
# property rather than an assumption. Both cases below are bugs that HAPPENED.
# ---------------------------------------------------------------------------


@requires_vulkan
def test_profiler_reports_nonzero_for_a_real_dispatch():
    """Regression: resolve_timestamps() once cleared the profile before checking
    whether anything was pending, so a download wiped the compute profile and
    every kernel reported 0.000 ms."""
    # The device index matters: profiling is per backend instance and these
    # default to 0, so asking device 0 for a profile of work that ran on device 1
    # returns an empty list. Found by running the suite on a second GPU.
    # A queue family may legitimately report no timestamp bits, or a zero
    # period, and then every dispatch reads 0.000 ms no matter how much work
    # ran. Skipping is the honest outcome: the regression this guards is
    # unobservable there, not absent. Checked rather than assumed -- vkml never
    # queried either until a driver that fails them turned up in CI.
    if not V.vulkan_timestamps_supported(VULKAN_DEVICE):
        pytest.skip("device cannot produce timestamps (no valid bits, or a zero period)")

    # KNOWN TO FAIL on the GitHub macOS runner, and deliberately not skipped
    # there. That device passes both checks above and still returns 0.000 ms
    # because its timestamps never advance -- an "Apple Paravirtual device",
    # where counter sampling is commonly unavailable. Validation, once actually
    # loaded, had nothing to say about vkCmdWriteTimestamp, the query pool or
    # vkGetQueryPoolResults, so the usage is spec-valid and the device simply
    # cannot measure.
    #
    # No automatic skip exists for that, and adding one would blind this test to
    # its own regression: resolve_timestamps() clearing the profile early
    # produced 0.000 ms too, which is indistinguishable from a device that
    # cannot count. Losing the distinction to gain a green tick is the wrong
    # trade, so macOS stays continue-on-error until someone decides otherwise.

    V.vulkan_set_profiling(True, VULKAN_DEVICE)
    try:
        dev = gpu_device()
        rng = np.random.default_rng(1)
        a = V.tensor(rng.standard_normal((256, 256), dtype=np.float32), device=dev)
        b = V.tensor(rng.standard_normal((256, 256), dtype=np.float32), device=dev)
        V.matmul(a, b).numpy()          # .numpy() submits a DOWNLOAD after the compute
        profile = V.vulkan_last_profile(VULKAN_DEVICE)
        assert profile, "profile empty after a real dispatch"
        assert sum(ms for _, ms in profile) > 0.0, "profile reports 0.000 ms"
    finally:
        V.vulkan_set_profiling(False, VULKAN_DEVICE)


# Needs hardware that actually overlaps independent dispatches. lavapipe does
# not -- it is a software rasteriser reporting zero compute units, and it runs
# the eight split-K partitions one after another, so the submit window equals
# the summed dispatch time instead of beating it. Marked for the GPU the
# behaviour was measured on; another discrete GPU would very likely satisfy it
# too, but that is untested and I will not claim it.
@requires_radv
@requires_vulkan
def test_submit_window_bounds_concurrent_dispatches():
    """The submit window must be a TRUE upper bound on concurrent execution.

    Per-dispatch timestamps end at ALL_COMMANDS, a global drain point, so with
    independent dispatches each one's window stretches to the end of the whole
    group and the sum multiply-counts. Split-K's eight partitions each report
    ~0.84 ms and sum to ~7.2 ms against a real 0.93 ms.

    The property: the submit window is never larger than the sum of the parts,
    and for a single dispatch the two agree. If a future change made the submit
    window merely another per-dispatch entry, this would catch it.
    """
    script = (
        "import sys;sys.path.insert(0,'python');import numpy as np,vkml as V;"
        "V.set_log_level(V.LogLevel.ERROR);V.init_vulkan(0);V.vulkan_set_profiling(True);"
        "d=V.device('vulkan:0');rng=np.random.default_rng(0);"
        "a=V.tensor(rng.random((64,16384),dtype=np.float32),device=d);"
        "b=V.tensor(rng.random((16384,64),dtype=np.float32),device=d);"
        "V.matmul(a,b).numpy();V.matmul(a,b).numpy();"
        "p=V.vulkan_last_profile();"
        # Everything that is not the submit entry is a dispatch. Selected by
        # exclusion, not by matching a literal label: dispatches are named
        # after their op when the backend labels them, so an allow-list here
        # would silently select NOTHING and pass a sum of zero.
        "sub=[ms for nm,ms in p if nm=='submit'];"
        "dis=[ms for nm,ms in p if nm!='submit'];"
        "print(sub[0], sum(dis), len(dis))"
    )

    def run(env):
        out = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                             env=_env(env), capture_output=True,
                             text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        window, total, count = out.stdout.strip().splitlines()[-1].split()
        return float(window), float(total), int(count)

    # Single dispatch: the window and the sum must agree closely.
    w1, t1, c1 = run({"VKML_GEMM_SPLITK": "OFF"})
    assert c1 == 1
    assert abs(w1 - t1) <= 0.05 * max(w1, t1), f"single dispatch: window {w1} vs sum {t1}"

    # Concurrent dispatches: the window must be a strict lower bound on the sum,
    # and must not itself grow with the partition count.
    w8, t8, c8 = run({"VKML_GEMM_SPLITK": "FORCED", "VKML_GEMM_SPLITK_SPLITS": "8"})
    assert c8 >= 8, f"expected >= 8 dispatches, saw {c8}"
    assert w8 < t8, f"window {w8} should be below the summed {t8} when dispatches overlap"
    assert w8 < w1, f"split-K window {w8} should beat the unsplit {w1}"


@requires_vulkan
def test_cumulative_gpu_time_accumulates_across_submissions():
    """`vulkan_stats()['gpu_ms']` must total the whole-submit windows.

    `vulkan_last_profile()` holds only the LAST submission, so a workload that
    submits repeatedly -- any training step -- had no admissible GPU total, and
    MEASUREMENT-AUDIT 7 rule 1b could not be checked for it. This counter is
    what makes the stage split in docs/BACKWARD-PERF-INVESTIGATION.md 1
    measurable at all.

    The law: over N separate submissions the counter grows by the sum of their
    windows. Asserted against the profile's own `submit` entry for the last
    one, so the two instruments have to agree, and summing across SEPARATE
    submissions is what rule 3 permits -- they are serial.

    In a subprocess: the counter is inert unless profiling is on, and switching
    that on is process-global -- it would follow every later test in the file.
    """
    if not V.vulkan_timestamps_supported(VULKAN_DEVICE):
        pytest.skip("device cannot produce timestamps (no valid bits, or a zero period)")

    script = (
        "import sys;sys.path.insert(0,'python');import numpy as np,vkml as V;"
        "V.set_log_level(V.LogLevel.ERROR);V.init_vulkan(0);V.vulkan_set_profiling(True);"
        "d=V.device('vulkan:0');"
        "t=V.tensor(np.ones((4096,),dtype=np.float32),device=d);"
        "V.relu(t).realize();"                       # rule 6: warm
        "b=V.vulkan_stats(0)['gpu_ms'];"
        "[V.relu(t).realize() for _ in range(8)];"
        "print(V.vulkan_stats(0)['gpu_ms']-b, V.vulkan_submit_ms(V.vulkan_last_profile()))"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=_env({}),
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    grew, last = (float(v) for v in out.stdout.strip().splitlines()[-1].split())

    # The device says it supports timestamps and they do not advance. That is
    # real: the macOS runner's "Apple Paravirtual device" reports valid bits and
    # a non-zero period, and still counts nothing, so `timestamps_supported()`
    # above cannot screen it out. PROBE, then decide.
    #
    # Skipping here does NOT lose the coverage that a zero profile would
    # otherwise buy. test_profiler_reports_nonzero_for_a_real_dispatch exists
    # for exactly that regression and deliberately does not skip -- see its
    # comment. This test is about ACCUMULATION, which an instrument reading zero
    # cannot demonstrate either way, and duplicating the other test's job here
    # only turned one known macOS failure into two.
    if last == 0.0:
        pytest.skip("this device's timestamps do not advance, so nothing can be accumulated")

    assert grew > 0, "cumulative gpu_ms did not move over 8 submissions"
    assert grew >= last, (
        f"cumulative gpu_ms grew {grew} ms over 8 submissions but the LAST one "
        f"alone reports {last} ms -- the counter is not accumulating"
    )
    # Eight comparable submissions: the total must look like several of them,
    # not like one. Deliberately loose -- this pins accumulation, not timing.
    assert grew > 2 * last, (
        f"cumulative gpu_ms grew only {grew} ms over 8 submissions of ~{last} ms; "
        f"it looks like it is being overwritten rather than summed"
    )


# ---------------------------------------------------------------------------
# 7. Compiler-behaviour invariants (M3-R2)
#
# The register model below was derived from a systematic sweep of
# (RM, RN, STACK_LEVELS) -> compiler resources, not from a single observation.
# It is PREDICTIVE: it correctly predicted, before measurement, that split-K
# would take 4x4 at K=16384 from 40 KiB of scratch to none.
#
# These tests pin the model. If one fails, the driver's allocator has changed
# behaviour and every register-pressure conclusion in the project needs
# revisiting -- which is exactly the signal worth having.
# ---------------------------------------------------------------------------

_STACK_CONST = {(2, 2): 17, (4, 2): 25, (4, 4): 35}


def _gemm_stats(env: dict[str, str], m: int, k: int, n: int) -> dict:
    script = "\n".join([
        "import sys,json;sys.path.insert(0,'python')",
        "import numpy as np, vkml as V",
        "V.set_log_level(V.LogLevel.ERROR);V.init_vulkan(0)",
        "d=V.device('vulkan:0');rng=np.random.default_rng(0)",
        f"a=V.tensor(rng.random(({m},{k}),dtype=np.float32),device=d)",
        f"b=V.tensor(rng.random(({k},{n}),dtype=np.float32),device=d)",
        "V.matmul(a,b).numpy()",
        "out=[r for r in V.vulkan_pipeline_stats(0)"
        "     if r.get('available') and 'gemm_reg' in r['name']]",
        "print(json.dumps(out[0]) if out else '{}')",
    ])
    res = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                         env=_env(env), capture_output=True,
                         text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-2000:]
    import json as _json
    return _json.loads(res.stdout.strip().splitlines()[-1])


@requires_radv
@requires_vulkan
@pytest.mark.slow
@pytest.mark.parametrize("block,k,levels", [("2x2", 256, 4), ("2x2", 512, 6), ("2x2", 2048, 8)])
def test_register_model_predicts_vgpr_exactly(block, k, levels):
    """VGPR = RM*RN*STACK_LEVELS + C(RM,RN), with slope EXACTLY 1.0.

    Derived from 10 spill-free points across three register blocks; the slope is
    over-determined (six independent points for 2x2 alone) and fitted exactly.
    Each carry-stack float costs precisely one VGPR: the compiler keeps the
    whole private array live in registers with no packing or reuse.

    This is the mechanism behind the pairwise bound -- the carry stack's
    register cost is not approximately RM*RN*L, it is exactly that.
    """
    rm, rn = (int(block[0]), int(block[2]))
    st = _gemm_stats({"VKML_GEMM_BLOCK": block}, 64, k, 64)
    assert st, "no gemm_reg pipeline statistics"
    assert st["scratch_bytes"] == 0, "model applies only to spill-free configurations"
    predicted = rm * rn * levels + _STACK_CONST[(rm, rn)]
    assert st["vgprs"] == predicted, (
        f"{block} K={k} L={levels}: predicted VGPR {predicted}, driver reported "
        f"{st['vgprs']}. The register allocator's behaviour has changed."
    )


@requires_radv
@requires_vulkan
@pytest.mark.slow
def test_high_occupancy_can_mean_spilling_not_efficiency():
    """The trap that makes `max_waves` unsafe to rank on.

    At 4x4 with K=16384 the allocator CAPS registers and spills: it reports
    VGPR=35 and waves=16, which looks better than the spill-free VGPR=99 /
    waves=8 -- while writing 40 KiB to scratch. A tuner ranking candidates by
    occupancy would choose the spilling one.

    Pinned because any future autotuner MUST reject non-zero scratch before it
    looks at occupancy (docs/PERFORMANCE-MODEL.md 5d).
    """
    spilling = _gemm_stats({"VKML_GEMM_BLOCK": "4x4"}, 64, 16384, 64)
    assert spilling["scratch_bytes"] > 0, "expected 4x4 at K=16384 to spill"
    assert spilling["vgprs"] < 50, "spilling configuration should report LOW vgpr usage"
    assert spilling["waves_per_simd"] >= 16, (
        "the trap is that the spilling configuration reports HIGH occupancy; "
        "if that is no longer true the warning can be relaxed"
    )


@requires_radv
@requires_vulkan
@pytest.mark.slow
def test_split_k_removes_the_4x4_spill():
    """Split-K shortens STACK_LEVELS enough to unlock the 4x4 register block.

    M3-02 listed this as an open question and explicitly declined to assume it.
    The register model then PREDICTED it quantitatively -- 4x4 at L=4 needs
    64 + 35 = 99 VGPRs, under the ~100 spill cliff -- and measurement confirmed
    it exactly: 40960 B of scratch becomes 0.

    The block is still not worth using (0.884 ms vs 0.763 ms for 2x2, because
    occupancy falls to 8 waves), which is the second, independent reason 4x4
    loses. Pinned so that story cannot silently change.
    """
    env = {"VKML_GEMM_BLOCK": "4x4", "VKML_GEMM_SPLITK": "FORCED",
           "VKML_GEMM_SPLITK_SPLITS": "128"}
    st = _gemm_stats(env, 64, 16384, 64)
    assert st["scratch_bytes"] == 0, f"split-K should remove the spill, saw {st['scratch_bytes']}"
    assert st["vgprs"] == 99, f"model predicts 99 VGPRs at 4x4/L=4, saw {st['vgprs']}"


# ---------------------------------------------------------------------------
# 8. Corrected register laws (M3-R3)
#
# M3-R2 believed the allocator spilled past ~100 VGPRs. The 2x8 / 8x2 pair --
# identical RM*RN and identical stacks, deliberately built to separate array
# size from register pressure -- refuted that. All figures at lds_vec=0.
# ---------------------------------------------------------------------------

_C = {(2, 2): 18, (4, 2): 26, (2, 4): 23, (4, 4): 35, (2, 8): 35, (8, 2): 42}


@requires_radv
@requires_vulkan
@pytest.mark.slow
@pytest.mark.parametrize("block,k,levels", [
    ("2x4", 256, 4), ("2x4", 2048, 8), ("2x8", 256, 4), ("8x2", 256, 4),
])
def test_law1_vgpr_equals_stack_plus_constant(block, k, levels):
    """VGPR = RM*RN*STACK_LEVELS + C(RM,RN), slope exactly 1.000.

    Exact on every spill-free point of six geometries. One carry-stack float
    costs precisely one VGPR.
    """
    rm, rn = int(block[0]), int(block[2])
    st = _gemm_stats({"VKML_GEMM_BLOCK": block, "VKML_GEMM_NOLDSVEC": "1"}, 64, k, 64)
    assert st["scratch_bytes"] == 0, "Law 1 applies to spill-free configurations only"
    assert st["vgprs"] == rm * rn * levels + _C[(rm, rn)]


@requires_radv
@requires_vulkan
@pytest.mark.slow
def test_law4_spill_threshold_is_array_size_not_register_pressure():
    """The decisive pair. Same VGPR count, opposite spill outcome.

        8x2 L=4 : stack 64 floats, VGPR 106 -> NO spill
        4x2 L=10: stack 80 floats, VGPR 106 -> spills 20480 B

    A total-register-pressure threshold cannot express this; a private-array-size
    threshold can. If this test ever fails, the driver's allocator has changed
    its strategy and docs/PERFORMANCE-MODEL.md 5h must be re-derived.
    """
    clean = _gemm_stats({"VKML_GEMM_BLOCK": "8x2", "VKML_GEMM_NOLDSVEC": "1"}, 64, 256, 64)
    assert clean["scratch_bytes"] == 0, "8x2 L=4 (64-float stack) must not spill"
    assert clean["vgprs"] == 106

    spilled = _gemm_stats({"VKML_GEMM_BLOCK": "4x2", "VKML_GEMM_NOLDSVEC": "1"}, 64, 8192, 64)
    assert spilled["scratch_bytes"] > 0, "4x2 L=10 (80-float stack) must spill"
    # Law 2: a spilled kernel's VGPR count collapses to exactly C(RM,RN).
    assert spilled["vgprs"] == _C[(4, 2)]
    # Law 3: scratch is the whole array, at one byte per float per invocation.
    assert spilled["scratch_bytes"] == 4 * 2 * 10 * 256


@requires_radv
@requires_vulkan
@pytest.mark.slow
def test_register_cost_is_asymmetric_in_rm_and_rn():
    """RM costs materially more registers than RN.

    A is read with stride BK, so each of RM rows needs its own address register;
    B is contiguous. C(2,8)=35 against C(8,2)=42 -- a 7-VGPR gap at identical
    RM*RN. Any future model of C that is symmetric in RM and RN is wrong.
    """
    wide = _gemm_stats({"VKML_GEMM_BLOCK": "2x8", "VKML_GEMM_NOLDSVEC": "1"}, 64, 256, 64)
    tall = _gemm_stats({"VKML_GEMM_BLOCK": "8x2", "VKML_GEMM_NOLDSVEC": "1"}, 64, 256, 64)
    assert wide["scratch_bytes"] == 0 and tall["scratch_bytes"] == 0
    assert tall["vgprs"] > wide["vgprs"], (
        f"expected the tall block to cost more; 2x8={wide['vgprs']} 8x2={tall['vgprs']}"
    )
    assert tall["vgprs"] - wide["vgprs"] == 7


# ---------------------------------------------------------------------------
# 9. Production dispatch decisions (I1-R1)
#
# The split-K heuristic is `ktiles >= tiles`, plus a chunk floor and a workspace
# cap. It was fitted to 16 measured shape/K combinations and classifies every
# one correctly. These tests pin the DECISIONS, not the timings: a heuristic
# that silently changes which shapes it accepts is a regression even when every
# result is still bit-identical.
# ---------------------------------------------------------------------------


def _split_decisions(shapes: list[tuple[int, int, int]]) -> dict[tuple, int]:
    """Runs each shape and reports the partition count the backend chose."""
    body = "\n".join(
        f"    V.matmul(V.tensor(rng.random(({m},{k}),dtype=np.float32),device=d),"
        f" V.tensor(rng.random(({k},{n}),dtype=np.float32),device=d)).numpy()"
        for (m, k, n) in shapes
    )
    script = "\n".join([
        "import sys; sys.path.insert(0,'python')",
        "import numpy as np, vkml as V",
        "V.set_log_level(V.LogLevel.INFO); V.init_vulkan(0)",
        "d=V.device('vulkan:0'); rng=np.random.default_rng(0)",
        "if True:",
        body,
    ])
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                         # AUTO explicitly: _NEUTRAL_BASE pins split-K OFF for
                         # every other test, and these tests are about what AUTO
                         # decides.
                         env=_env({"VKML_VULKAN_DEBUG": "1", "VKML_GEMM_SPLITK": ""}),
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    import re
    chosen: dict[tuple, int] = {}
    current = None
    for line in out.stderr.splitlines() + out.stdout.splitlines():
        g = re.search(r"gemm M=(\d+) N=(\d+) K=(\d+)", line)
        if g:
            if current is not None:
                chosen.setdefault(current, 1)
            current = (int(g.group(1)), int(g.group(3)), int(g.group(2)))
            continue
        s = re.search(r"split-k splits=(\d+)", line)
        if s and current is not None:
            chosen[current] = int(s.group(1))
            current = None
    if current is not None:
        chosen.setdefault(current, 1)
    return chosen


@requires_radv
@requires_vulkan
@pytest.mark.slow
def test_split_k_heuristic_accepts_only_measured_wins():
    """Every shape split-K is enabled for measured >= 1.5x; every shape it
    declines measured <= 1.23x, including two that would have LOST (0.59x and
    0.84x). A tiles-only rule cannot separate these: at tiles=64 the outcome
    spans 0.59x to 1.99x depending on K alone.
    """
    should_split = [(64, 512, 64), (64, 2048, 64), (128, 512, 128),
                    (128, 1024, 128), (256, 2048, 256), (256, 4096, 256)]
    should_not = [(256, 256, 256), (256, 512, 256), (256, 1024, 256),
                  (1024, 1024, 1024), (1536, 4096, 1536)]
    got = _split_decisions(should_split + should_not)
    for shape in should_split:
        assert got.get(shape, 1) > 1, (
            f"{shape} measured a >=1.5x win from split-K but the heuristic declined it"
        )
    for shape in should_not:
        assert got.get(shape, 1) == 1, (
            f"{shape} measured <=1.23x (or a loss) from split-K but the heuristic "
            f"enabled {got.get(shape)} partitions"
        )


@requires_vulkan
@pytest.mark.slow
def test_split_k_partition_count_is_always_a_power_of_two_chunk():
    """The correctness constraint, checked on the shapes AUTO actually accepts.

    `chunk` must be a power of two or bit-identity is lost (docs/SPLIT_K_DESIGN.md
    2.5). splits = ceil(ktiles/chunk), so this asserts the derived relationship
    holds rather than trusting the planner's arithmetic.
    """
    shapes = [(64, 512, 64), (64, 2048, 64), (128, 1024, 128), (256, 4096, 256)]
    for shape, splits in _split_decisions(shapes).items():
        if splits == 1:
            continue
        m, k, n = shape
        ktiles = (k + 31) // 32
        chunk = (ktiles + splits - 1) // splits
        # Recover the chunk the backend must have used and check it is 2^q.
        candidates = [c for c in (1 << q for q in range(0, 16))
                      if (ktiles + c - 1) // c == splits]
        assert candidates, (
            f"{shape}: {splits} partitions over {ktiles} k-tiles is not reachable "
            f"with any power-of-two chunk (nearest non-power-of-two would be {chunk})"
        )



# ---------------------------------------------------------------------------
# 10. The graph must contain no work that does nothing
#     (docs/BACKWARD-PERF-INVESTIGATION.md)
#
# `reduce_to_shape()` undoes broadcasting in the backward pass by summing the
# axes that were stretched. It used to reduce every LEADING axis without
# checking its extent -- and matmul promotes its operands to batched 4-D, so
# every gradient flowing back through one arrives with leading axes of extent 1.
#
# The result was a reduction that reduced NOTHING. Because the reduce kernel
# launches one workgroup per OUTPUT element, a (1, 1, 4096, 4096) gradient
# became 16,777,216 workgroups -- 4.29 billion invocations to copy 16.7 M
# floats -- and took 167 ms of a 170 ms backward pass.
#
# Ten hypotheses missed it, and no test caught it, for one reason worth
# remembering: THE VALUES WERE ALWAYS CORRECT. Summing a single element returns
# that element. Only the dispatch COUNT ever showed it.
#
# The test below runs in a subprocess, because the suite's autouse fixture
# forces EAGER mode: that realises every operation separately, which inflates
# the dispatch count and makes it vary with shape for reasons unrelated to the
# property being asserted. The graph this test is about is the lazy one.
#
# ONE THING THAT DOES NOT WORK, so nobody spends an afternoon on it again: an
# obvious-looking test that the dispatch count is independent of tensor size
# does NOT catch this. It was written, and it PASSED against the bug. The
# degenerate reduction adds exactly one dispatch whatever the shape -- what
# scaled with size was that dispatch's cost, not the count. Only an absolute
# budget catches it.
# ---------------------------------------------------------------------------

_BACKWARD_DISPATCHES = (
    "import sys;sys.path.insert(0,'python');import numpy as np,vkml as V;"
    "V.set_log_level(V.LogLevel.ERROR);V.init_vulkan(0);"
    "d=V.device('vulkan:0');rng=np.random.default_rng(0);"
    "n=int(sys.argv[1]);"
    "a=V.tensor(rng.random((32,n),dtype=np.float32),device=d);"
    "b=V.tensor(rng.random((n,32),dtype=np.float32),device=d);"
    "a.requires_grad=True;b.requires_grad=True;"
    "step=lambda:(setattr(a,'grad',V.Tensor()),setattr(b,'grad',V.Tensor()),"
    "V.backward(V.sum(V.matmul(a,b))));"
    "step();"                                  # warm: first run compiles pipelines
    "c=V.vulkan_stats(0)['dispatches'];step();"
    "print(V.vulkan_stats(0)['dispatches']-c)"
)


def _backward_dispatches(k: int) -> int:
    """Dispatches issued by one `sum(a @ b)` backward, in lazy mode."""
    out = subprocess.run([sys.executable, "-c", _BACKWARD_DISPATCHES, str(k)],
                         cwd=REPO, env=_env({}), capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return int(out.stdout.strip().splitlines()[-1])


@requires_vulkan
def test_backward_emits_no_degenerate_reductions():
    """A backward pass must not spend a dispatch on a no-op reduction.

    Budget for `sum(a @ b)` with 2-D operands, verified against the trace:

        1  fill        the seed gradient
        1  contiguous  materialising the broadcast seed
        1  matmul      db = a.T @ g
        1  matmul      da = g @ b.T

    Four, and no reduction at all -- nothing here was broadcast along an axis
    that carries more than one value. The bug added a fifth dispatch that
    computed an identity over 16.7 M elements.

    An upper BOUND rather than an equality, so a future change that fuses or
    elides work does not fail here; one that reintroduces a no-op will.
    """
    used = _backward_dispatches(256)
    assert used <= 4, (
        f"{used} dispatches for a matmul-sum backward, expected at most 4. A "
        f"reduction over an axis of extent 1 is the known cause -- re-run with "
        f"VKML_VULKAN_DEBUG=1 and look for 'reduce ... n_red=1'."
    )




@requires_vulkan
def test_multi_root_realize_uses_one_submission():
    """Four independent tensors realized together must cost ONE submission.

    In a subprocess, and that is not incidental: the suite's autouse fixture
    forces EAGER mode, which realises every operation as it is built, so by the
    time V.realize() is reached there is nothing left to schedule. Measured
    rather than assumed -- written in-process first, this reported 4 either way
    and would have passed against a V.realize() that did nothing at all.

    An upper BOUND, not an equality. Splitting a large graph over several
    submissions is legitimate and deliberate elsewhere -- ggml-vulkan does it to
    overlap command recording with execution, and caps submission size per
    device to avoid driver timeouts -- so what is pinned here is "far fewer than
    one each", never "exactly one".
    """
    script = (
        "import sys;sys.path.insert(0,'python');import numpy as np,vkml as V;"
        "V.set_log_level(V.LogLevel.ERROR);V.init_vulkan(0);"
        "d=V.device('vulkan:0');rng=np.random.default_rng(0);"
        "ts=[V.tensor(rng.random((64,64),dtype=np.float32),device=d) for _ in range(4)];"
        "V.realize(*[V.relu(t) for t in ts]);"  # warm
        "c=V.vulkan_stats(0)['submissions'];"
        "V.realize(*[V.relu(t) for t in ts]);"
        "print(V.vulkan_stats(0)['submissions']-c)"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=_env({}),
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    used = int(out.stdout.strip().splitlines()[-1])
    assert used <= 2, (
        f"{used} submissions for 4 tensors realized together; the point of "
        f"multi-root realize is that they share one"
    )


def test_gelu_keeps_relative_accuracy_in_its_negative_tail():
    """gelu's tail has to survive in RELATIVE terms, on the CPU, with no GPU here.

    WHAT WAS BELIEVED. That gelu's accuracy was pinned, because three separate
    tests covered it and all three passed.

    WHAT TURNED OUT TO BE TRUE. All three were vacuous on this domain, each for
    a different reason, while the CPU kernel was catastrophically wrong (#28):

      - test_vulkan_kernels.py compares the GPU against the CPU. Both formed
        gelu as 0.5x(1 + erf(x/sqrt2)), so they cancelled identically and
        agreed with each other about the wrong answer.
      - test_extreme_values.py holds the CPU kernel as the oracle by
        construction, so a wrong CPU kernel is its definition of correct.
      - test_ops_vs_torch.py compares against torch, which forms gelu the same
        way and flushes 78 of these same 512 samples to zero itself. Measured,
        not assumed: torch's own max relative error here is 1.0.

    And the tolerance shape hid what was left. The central transcendental gate
    is Tol(atol=1e-5, rtol=1e-5), applied as atol + rtol*|want|. gelu(-5.5) is
    1e-7, so the absolute term alone admits ANY answer on the deep tail --
    including exactly zero, which is what the kernel returned for 102 of these
    512 inputs.

    So the bound below is the central RELATIVE tolerance with atol pinned to
    zero. That is not a new number chosen at a call site: it is the policy's own
    rtol, applied in the only form that says anything where every value is two
    orders of magnitude below the policy's atol.

    And the reference is neither backend, and not torch: it is the exact value
    in double precision, which is the only thing on this domain that is right.

    Deliberately CPU-only. The defect was in the oracle, and every existing test
    that could have caught it needed a GPU -- so it could not fail in the
    CPU-only configuration three CI jobs build.
    """
    x = np.linspace(-6.0, -3.0, 512, dtype=np.float32)
    got = V.gelu(V.tensor(x, device=V.cpu)).numpy().astype(np.float64)

    # 0.5x*erfc(-x/sqrt2) in double precision. erfc rather than 1 + erf for the
    # same reason the kernel now uses it: the sum is the thing that cancels.
    exact = np.array([0.5 * float(v) * math.erfc(-float(v) / math.sqrt(2.0)) for v in x])

    rtol = TOLERANCES["transcendental"].rtol

    # The true value never reaches zero here, so a returned zero is total loss
    # of significance rather than rounding. Asserted rather than recalled,
    # because it is what makes the relative bound below well posed at all.
    assert np.abs(exact).min() > 0.0, "domain reaches a true zero; rel. error is undefined"
    zeros = int(np.count_nonzero(got == 0.0))
    assert zeros == 0, (
        f"gelu returned exactly zero for {zeros} of {x.size} inputs on [-6, -3], "
        f"where the smallest true magnitude is {np.abs(exact).min():.3e}"
    )

    rel = np.abs(got - exact) / np.abs(exact)
    worst = int(np.argmax(rel))
    assert rel.max() <= rtol, (
        f"gelu lost the tail: max relative error {rel.max():.3e} > {rtol:.0e} at "
        f"x = {x[worst]:.6f} (got {got[worst]:.6e}, exact {exact[worst]:.6e})"
    )

    # This test carries its own mutant, because a test that cannot fail is worse
    # than none (skill 10) and this one replaces three that could not. Below is
    # exactly the expression the kernel used before #28, in fp32. It must NOT
    # clear the bound above -- if it ever does, the domain has drifted out of
    # the cancelling regime and the test has stopped covering anything.
    inv_sqrt2 = np.float32(0.70710678118654752440)
    erf_f32 = np.array([math.erf(float(v)) for v in (x * inv_sqrt2)], dtype=np.float32)
    cancelling = (np.float32(0.5) * x * (np.float32(1.0) + erf_f32)).astype(np.float64)
    mutant = np.abs(cancelling - exact) / np.abs(exact)
    assert mutant.max() > rtol, (
        f"the pre-#28 cancelling form now passes at {mutant.max():.3e}, so this "
        f"domain no longer exercises the loss of significance the test exists for"
    )


def test_gelu_gradient_keeps_relative_accuracy_in_its_negative_tail():
    """The same cancellation as the forward, one derivative up -- and torch has it.

    gelu's gradient is Phi(x) + x*phi(x). The cdf term was built as
    0.5(1 + erf(x/sqrt2)), which cancels for negative x exactly as the forward
    kernel did before #28. It survived that fix because it lives in autograd.cpp
    rather than in a kernel, and because it is much less visible: x*phi(x)
    dominates the tail and is computed accurately, so the error is 2.8% at
    x = -6 rather than the forward's 100%.

    WHY THE REFERENCE IS NOT TORCH. Measured on this domain, torch's own gelu
    gradient is 16% wrong at x = -6 and 0.34% at x = -5 -- it forms the cdf the
    same cancelling way. vkml is now roughly seven orders of magnitude closer to
    the true value there than torch is, so asserting against torch would pin
    vkml to the less accurate answer and fail the moment it got better. This is
    the one place in the suite where the established framework is NOT the
    oracle, and the reason is written here rather than assumed.

    The bound is the central RELATIVE tolerance with atol pinned to zero, for
    the same reason as the forward test: every gradient here is far below the
    policy's absolute term, so atol + rtol*|want| would admit anything.
    """
    x = np.linspace(-6.0, -3.0, 256, dtype=np.float32)

    v = V.tensor(x, device=V.cpu, requires_grad=True)
    V.sum(V.gelu(v)).backward()
    got = v.grad.numpy().astype(np.float64)

    # Phi(x) + x*phi(x) in double precision, with Phi through erfc so the
    # reference itself does not carry the defect under test.
    exact = np.array([
        0.5 * math.erfc(-float(t) / math.sqrt(2.0))
        + float(t) * math.exp(-0.5 * float(t) * float(t)) / math.sqrt(2.0 * math.pi)
        for t in x
    ])

    rtol = TOLERANCES["transcendental"].rtol
    assert np.abs(exact).min() > 0.0, "domain reaches a true zero; rel. error is undefined"

    rel = np.abs(got - exact) / np.abs(exact)
    worst = int(np.argmax(rel))
    assert rel.max() <= rtol, (
        f"gelu gradient lost the tail: max relative error {rel.max():.3e} > {rtol:.0e} "
        f"at x = {x[worst]:.6f} (got {got[worst]:.6e}, exact {exact[worst]:.6e})"
    )

    # The mutant, as in the forward test: the pre-fix cdf in fp32, which must
    # still fail the bound. Without this the test would go quiet if the domain
    # ever moved out of the cancelling regime.
    inv_sqrt2 = np.float32(0.70710678118654752440)
    erf_f32 = np.array([math.erf(float(t)) for t in (x * inv_sqrt2)], dtype=np.float32)
    cdf_cancelling = np.float32(0.5) * (np.float32(1.0) + erf_f32)
    pdf = np.exp(np.float32(-0.5) * x * x, dtype=np.float32) * np.float32(0.39894228040143267794)
    mutant_rel = np.abs((cdf_cancelling + x * pdf).astype(np.float64) - exact) / np.abs(exact)
    assert mutant_rel.max() > rtol, (
        f"the pre-fix cancelling cdf now passes at {mutant_rel.max():.3e}, so this "
        f"domain no longer exercises the loss of significance the test exists for"
    )
