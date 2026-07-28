#!/usr/bin/env python3
"""Mutation campaign: break each implementation, confirm the suite notices.

docs/MEASUREMENT-AUDIT.md rule 10 says to check every gate for vacuity before
trusting a pass. A green suite proves the tests ran, not that they can fail --
and a test that cannot fail is worse than none, because it manufactures
confidence. This is how that rule is checked rather than asserted.

Each entry applies one semantically MEANINGFUL mutation -- an off-by-one, a
dropped guard, a reversed fold -- rebuilds if the source is compiled, runs the
tests that should catch it, and reports:

    KILLED    the suite failed. The tests detect this defect.
    SURVIVED  the suite passed. Something is untested; investigate.

A syntax error would prove nothing, so every mutation compiles (or, in Python,
imports).

Kernels are not the only thing worth mutating. The data pipeline and the
checkpoint format carry no numerics at all, but they fail SILENTLY -- a shuffle
that unpairs inputs from labels, a checkpoint that loads with a key missing --
and a silent failure is exactly what a vacuous test hides.

MAINTENANCE COST, stated plainly: each mutation is a literal string from the
source, so a refactor silently stops it applying. That failure is reported as
PATTERN-MISSING rather than passing quietly, but it does mean this file has to
be updated alongside the kernels it targets. Adding a kernel without adding a
mutation here leaves a gap this script will not notice.

Usage:  python scripts/mutation_check.py [path-substring]
Exit:   0 if every mutation was killed, 1 otherwise.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/jeel/Projects/vkml")
PY = ROOT / ".venv/bin/python"

# (label, file, find, replace, test selector)
# Each mutation changes MEANING, not syntax -- a compile error would prove
# nothing about the tests.
MUTATIONS = [
    # --- shaders -----------------------------------------------------------
    ("tri: diagonal off-by-one", "shaders/tri.comp",
     "offset >= p.diagonal : offset <= p.diagonal",
     "offset > p.diagonal : offset <= p.diagonal", "test_tri"),

    ("cat: reuse output extent for the source index", "shaders/cat.comp",
     "const uint extent = from_a ? p.a_extent : p.b_extent;",
     "const uint extent = p.out_extent;", "test_cat"),

    ("index_select: ignore the index vector", "shaders/index_select.comp",
     "const uint j = uint(clamp(raw, int64_t(0), int64_t(p.src_extent) - 1));",
     "const uint j = k % p.src_extent;", "test_index_select"),

    ("scatter_add: reverse the fold order", "shaders/scatter_add.comp",
     "for (uint k = 0; k < p.src_extent; ++k) {",
     "for (uint k = p.src_extent; k-- > 0;) {", "test_scatter_add"),

    ("im2col: drop the padding bounds check", "shaders/im2col.comp",
     "if (h >= 0 && h < p.image_h && x >= 0 && x < p.image_w) {",
     "if (h >= 0 && h < p.image_h && x >= 0 && x < p.image_w || true) {", "test_im2col"),

    ("col2im: drop the stride-boundary test", "shaders/col2im.comp",
     "if (top < 0 || (top % p.stride_h) != 0) { continue; }",
     "if (top < 0) { continue; }", "test_col2im"),

    ("max_pool2d: tie rule picks the last maximum", "shaders/max_pool2d.comp",
     "if (v > best) {", "if (v >= best) {", "test_max_pool2d or test_conv2d"),

    ("rand: nine Philox rounds instead of ten", "shaders/rand.comp",
     "const int  kRounds = 10;", "const int  kRounds = 9;", "test_rand or test_dropout"),

    # Strided indexing. Only reachable at all since the coverage audit gave
    # these operators non-contiguous inputs; before that, every source was
    # contiguous and this arithmetic was dead in every test.
    ("operand_offset: ignore strides, walk flat", "shaders/common.glsl",
     "        off += idx * op.nb[i];",
     "        off += idx * op.nb[VKML_MAX_DIMS - 1];",
     "test_layout_and_scale"),

    # The shader half of the f16 contract. DTYPE is a specialisation constant,
    # so an f16 kernel that ignored it would read halves through F32Buf -- the
    # same defect the CPU comparisons carried, on the other backend.
    ("shader load_f: ignore DTYPE and read every operand as f32", "shaders/common.glsl",
     "    if (dtype == T_F16) {\n        return float(F16Buf(buf).v[idx]);\n    }\n"
     "    return F32Buf(buf).v[idx];",
     "    return F32Buf(buf).v[idx];",
     "test_vulkan_agrees_with_the_cpu_oracle_bit_for_bit"),

    ("shader store_f: narrow to f16 twice", "shaders/common.glsl",
     "        F16Buf(buf).v[idx] = float16_t(value);",
     "        F16Buf(buf).v[idx] = float16_t(float(float16_t(value * 1.0009765625)));",
     "test_vulkan_agrees_with_the_cpu_oracle_bit_for_bit"),

    # --- CPU oracle --------------------------------------------------------
    ("philox(cpu): nine rounds instead of ten", "src/backend/cpu/philox.h",
     "inline constexpr int kPhiloxRounds = 10;",
     "inline constexpr int kPhiloxRounds = 9;", "test_rand or test_dropout"),

    ("k_col2im: drop the stride-boundary test", "src/backend/cpu/kernels_movement.cpp",
     "if (top < 0 || top % p.stride_h != 0) {\n                continue;\n            }",
     "if (top < 0) {\n                continue;\n            }", "test_col2im"),

    ("k_max_pool2d: tie rule picks the last maximum", "src/backend/cpu/kernels_movement.cpp",
     "            if (v > best) {", "            if (v >= best) {",
     "test_max_pool2d or test_conv2d"),

    ("k_cat: reuse output extent for the source index", "src/backend/cpu/kernels_movement.cpp",
     "const int64_t extent = from_a ? a_extent : b_extent;",
     "const int64_t extent = out_extent;", "test_cat"),

    # --- the f16 precision contract ----------------------------------------
    # ARCHITECTURE.md 7.3 says f16 is storage and fp32 is the accumulator. That
    # is a claim about precision, which a tolerance comparison cannot check --
    # only a value chosen so the two answers differ by 100 % can.
    ("reduction: accumulate in f16 instead of fp32", "src/backend/cpu/kernels_reduce.cpp",
     "    reduce_float(out, [](const auto& read, int64_t n) "
     "{ return pairwise_sum<float>(read, 0, n); });",
     "    reduce_float(out, [](const auto& read, int64_t n) {\n"
     "        Half acc{0.0F};\n"
     "        for (int64_t i = 0; i < n; ++i) { acc = Half(acc.to_float() + read(i)); }\n"
     "        return acc.to_float();\n"
     "    });",
     "test_accumulation_happens_in_fp32"),

    ("matmul: accumulate the dot product in f16", "src/backend/cpu/kernels_matmul.cpp",
     "                    const float dot = pairwise_sum<float>(",
     "                    Half dot_acc{0.0F};\n"
     "                    for (int64_t q = 0; q < k; ++q) {\n"
     "                        T qa{};\n"
     "                        T qb{};\n"
     "                        std::memcpy(&qa, a_bytes + a_row + q * a.shape.stride(3), sizeof(T));\n"
     "                        std::memcpy(&qb, b_bytes + b_col + q * b.shape.stride(2), sizeof(T));\n"
     "                        dot_acc = Half(dot_acc.to_float() + widen(qa) * widen(qb));\n"
     "                    }\n"
     "                    const float dot = dot_acc.to_float();\n"
     "                    [[maybe_unused]] const float unused_dot = pairwise_sum<float>(",
     "test_matmul_accumulates_in_fp32"),

    ("compare: read every input as f32, whatever it stores",
     "src/backend/cpu/kernels_elementwise.cpp",
     "    const DType in = out.src[0]->dtype;",
     "    const DType in = DType::F32;",
     "test_f16_comparison_is_correct"),

    # --- python: data pipeline ---------------------------------------------
    # No kernel here and no numerics -- what these guard is bookkeeping, which
    # fails silently. A shuffle that drops samples or unpairs inputs from labels
    # trains a model that looks healthy and is not.
    ("loader: sample with replacement instead of permuting", "python/vkml/data.py",
     "order = _np.random.default_rng(self.seed + self._epoch).permutation(n)",
     "order = _np.random.default_rng(self.seed + self._epoch).integers(0, n, n)",
     "test_epoch_is_a_permutation_not_a_sample"),

    ("dataset: permute each array independently", "python/vkml/data.py",
     "return tuple(array[indices] for array in self.arrays)",
     "return tuple(_np.random.default_rng(i).permutation(array[indices])\n"
     "                     for i, array in enumerate(self.arrays))",
     "test_shuffle_keeps_paired_arrays_aligned"),

    ("loader: reseed identically every epoch", "python/vkml/data.py",
     "self._epoch += 1", "pass", "test_successive_epochs_differ"),

    ("loader: drop_last off-by-one", "python/vkml/data.py",
     "limit = n - self.batch_size + 1 if self.drop_last else n",
     "limit = n - self.batch_size + 2 if self.drop_last else n",
     "test_drop_last_keeps_every_batch_the_same_shape"),

    ("split: cut without shuffling first", "python/vkml/data.py",
     "order = _np.random.default_rng(seed).permutation(n)\n    cut",
     "order = _np.arange(n)\n    cut", "test_split_shuffles_before_cutting"),

    # --- python: checkpoints -----------------------------------------------
    ("checkpoint: allow pickle on load", "python/vkml/serialize.py",
     "tensors[key] = _npy.read_array(data, allow_pickle=False)",
     "tensors[key] = _npy.read_array(data, allow_pickle=True)",
     "test_a_pickle_payload_is_refused_rather_than_executed"),

    ("checkpoint: write straight to the destination", "python/vkml/serialize.py",
     'temp_path = path.with_name(f"{path.name}.{os.getpid()}.partial")',
     "temp_path = path", "test_a_save_interrupted_mid_write"),

    ("checkpoint: catch Exception rather than BaseException", "python/vkml/serialize.py",
     "except BaseException:", "except Exception:", "test_a_save_interrupted_mid_write"),

    ("checkpoint: leave the partial file behind", "python/vkml/serialize.py",
     "temp_path.unlink(missing_ok=True)", "pass", "test_a_save_interrupted_mid_write"),

    ("checkpoint: accept any format version", "python/vkml/serialize.py",
     "if version > FORMAT_VERSION:", "if False:", "test_rejects_a_newer_format_version"),

    ("checkpoint: skip the missing-member check", "python/vkml/serialize.py",
     "if member not in names:", "if False and member not in names:",
     "test_rejects_a_missing_member"),
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def build() -> bool:
    r = run(["cmake", "--build", "build/release", "-j8"])
    return r.returncode == 0


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = []

    for label, rel, find, replace, selector in MUTATIONS:
        if only and only not in rel:
            continue
        path = ROOT / rel
        original = path.read_text()

        if find not in original:
            results.append((label, "PATTERN-MISSING"))
            print(f"  !! {label}: pattern not found -- mutation not applied", flush=True)
            continue

        try:
            path.write_text(original.replace(find, replace, 1))
            # Python mutations need no rebuild, and skipping it turns a minute
            # per mutation into a second. Only compiled sources pay for it.
            if rel.startswith(("shaders/", "src/")) and not build():
                results.append((label, "BUILD-FAILED"))
                print(f"  !! {label}: did not compile", flush=True)
                continue

            r = run([str(PY), "-m", "pytest", "tests/python", "-x", "-q", "-k", selector])
            killed = r.returncode != 0
            results.append((label, "KILLED" if killed else "SURVIVED"))
            mark = "ok " if killed else "!! "
            print(f"  {mark}{label}: {'KILLED' if killed else 'SURVIVED'}", flush=True)
        finally:
            path.write_text(original)

    build()  # leave the tree as we found it

    print()
    survived = [r for r in results if r[1] != "KILLED"]
    print(f"{len(results) - len(survived)}/{len(results)} mutations killed")
    for label, status in survived:
        print(f"  SURVIVING: {label} [{status}]")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
