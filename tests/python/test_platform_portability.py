"""Behaviour that differs between Windows and POSIX, tested on both.

WHY THIS FILE EXISTS
--------------------
A full green suite on Linux missed seven bugs that a single run on Windows
found. Reviewing them afterwards, they were not seven unrelated defects: with
one exception they were all the same mistake, which is that a property of the
DEVELOPMENT MACHINE had been written down as if it were the contract.

    issue #2   push-constant blocks asserted against 256, the limit this GPU
               reports, rather than the 128 Vulkan guarantees
    issue #3   f32 -> f16 narrowing left to the driver, which is
               implementation-defined and differs between RADV and Windows AMD
    issue #4   a push-constant range smaller than the shader's block, which
               RADV tolerated and AMD's Windows compiler did not
    issue #6   a test looping over subgroup widths this device happens to offer
    issue #10  a build step that overwrote a file another process had mapped,
               which POSIX permits and Windows does not
    issue #13  inline asm, which MSVC does not have

The Vulkan ones are gated at compile time now (see the push-constant assertions
in vulkan_backend.cpp) or fixed by construction. What was left ungated is the
HOST half: file and path handling, where Linux is permissive and Windows is not,
and where a Linux-only run cannot tell the difference.

These tests need no GPU, so they run in every CI job on both platforms. They are
written to assert the CONTRACT rather than one platform's behaviour -- a test
that encodes "os.replace always succeeds" would be the original mistake again,
one layer up.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import vkml as V


def _tensors() -> dict[str, np.ndarray]:
    return {
        "w": np.arange(12, dtype=np.float32).reshape(3, 4),
        "b": np.array([0.5, -0.25], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "plain.vkml",
        "with space.vkml",
        "with-dash_and.dots.vkml",
        "ünïcødé.vkml",           # non-ASCII: cp1252 vs utf-8 on Windows
        "日本語.vkml",             # outside latin-1 entirely
        "UPPER.VKML",             # case-insensitive filesystems
    ],
)
def test_checkpoint_round_trips_through_awkward_filenames(tmp_path, name):
    """A checkpoint path is whatever the user's filesystem allows.

    Encoding is the trap here, not the characters. Windows filesystem APIs are
    UTF-16 and the legacy code page is not UTF-8, so a name that survives a
    round trip on Linux can fail to open, or open a DIFFERENT file, on Windows.
    """
    path = tmp_path / name
    V.save(path, _tensors())
    loaded = V.load(path)

    for key, array in _tensors().items():
        assert np.array_equal(loaded.tensors[key], array)


def test_checkpoint_round_trips_through_a_nested_directory(tmp_path):
    """Separators must come from the OS, not be spelled into the path."""
    nested = tmp_path / "runs" / "exp-01" / "checkpoints"
    nested.mkdir(parents=True)
    path = nested / "epoch_3.vkml"

    V.save(path, _tensors())
    assert np.array_equal(V.load(path).tensors["w"], _tensors()["w"])


def test_a_str_path_works_as_well_as_a_Path(tmp_path):
    """Callers pass both, and `str` is what argparse and config files give."""
    path = tmp_path / "as_str.vkml"
    V.save(str(path), _tensors())
    assert np.array_equal(V.load(str(path)).tensors["b"], _tensors()["b"])


# ---------------------------------------------------------------------------
# Bytes on disk
# ---------------------------------------------------------------------------


def test_the_checkpoint_is_written_as_binary(tmp_path):
    """Guards against a text-mode write mangling bytes.

    Opening a file in text mode on Windows turns every \\n into \\r\\n. In a zip
    of float32 arrays that is silent corruption of any byte that happens to be
    0x0A -- which, in real weights, is most files. A checkpoint that only fails
    to load on Windows, and only sometimes, is about the worst failure shape
    available, so assert the exact byte count instead of waiting for it.
    """
    path = tmp_path / "binary.vkml"
    V.save(path, _tensors())

    on_disk = path.read_bytes()
    assert on_disk[:2] == b"PK", "not a zip: the header was rewritten"
    assert len(on_disk) == path.stat().st_size

    # Every member's stored size must match what the zip directory claims. A
    # newline translation anywhere would desynchronise the two.
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            assert len(archive.read(info.filename)) == info.file_size


def test_values_survive_the_round_trip_bit_for_bit(tmp_path):
    """Not almost-equal. A checkpoint is storage, so it is exact or it is broken."""
    rng = np.random.default_rng(0)
    tensors = {
        "f32": rng.standard_normal((7, 5)).astype(np.float32),
        "f16": rng.standard_normal(31).astype(np.float16),
        "i32": rng.integers(-2**31, 2**31 - 1, size=17, dtype=np.int32),
        "i64": rng.integers(-2**62, 2**62, size=9, dtype=np.int64),
        "edge": np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32),
    }
    path = tmp_path / "exact.vkml"
    V.save(path, tensors)
    loaded = V.load(path)

    for key, array in tensors.items():
        got = loaded.tensors[key]
        assert got.dtype == array.dtype, key
        assert np.array_equal(got, array, equal_nan=True), key


# ---------------------------------------------------------------------------
# Replacing a file that already exists
# ---------------------------------------------------------------------------


def test_overwriting_a_checkpoint_leaves_the_new_contents(tmp_path):
    """The ordinary case: a training loop saving every epoch to one path."""
    path = tmp_path / "rolling.vkml"
    V.save(path, {"w": np.zeros(4, dtype=np.float32)})
    V.save(path, {"w": np.ones(4, dtype=np.float32)})

    assert np.array_equal(V.load(path).tensors["w"], np.ones(4, dtype=np.float32))


def test_a_save_leaves_no_partial_file_behind(tmp_path):
    """`.partial` litter means an interrupted save was not cleaned up.

    The temp file carries the pid so concurrent savers cannot delete each
    other's, which makes leftovers cheap to detect and worth detecting: they
    accumulate silently in a checkpoint directory, and on Windows a leftover
    holding a handle is what makes the NEXT save fail.
    """
    path = tmp_path / "clean.vkml"
    V.save(path, _tensors())

    leftovers = [p.name for p in tmp_path.iterdir() if ".partial" in p.name]
    assert leftovers == []


def test_an_interrupted_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """The invariant a training run actually depends on.

    A save that dies partway -- Ctrl-C, a full disk, a killed job -- must not
    take the previous epoch's weights with it. That is why `save` writes to a
    pid-suffixed temp file and only then calls `os.replace`, which is atomic.

    Interrupting the write is the only way to test that. Simply saving twice
    passes whether or not the temp file exists, because a direct write to the
    destination also completes and also produces a valid checkpoint -- an
    earlier version of this test did exactly that and could not fail.

    The failure is injected on the SECOND array so the archive is genuinely
    half-written when it dies.
    """
    path = tmp_path / "precious.vkml"
    original = {"a": np.full(4, 7.0, dtype=np.float32), "b": np.full(4, 8.0, dtype=np.float32)}
    V.save(path, original)
    before = path.read_bytes()

    from numpy.lib import format as npy_format

    real_write = npy_format.write_array
    calls = {"n": 0}

    def fail_on_the_second_array(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated interruption mid-save")
        return real_write(*args, **kwargs)

    monkeypatch.setattr("vkml.serialize._npy.write_array", fail_on_the_second_array)

    with pytest.raises(KeyboardInterrupt):
        V.save(path, {"a": np.zeros(4, dtype=np.float32), "b": np.zeros(4, dtype=np.float32)})

    assert calls["n"] == 2, "the interruption did not land mid-archive"
    assert path.read_bytes() == before, "the previous checkpoint was modified by a failed save"
    assert np.array_equal(V.load(path).tensors["a"], original["a"])
    assert [p.name for p in tmp_path.iterdir() if ".partial" in p.name] == []


# ---------------------------------------------------------------------------
# Interpreter layout
# ---------------------------------------------------------------------------


def test_the_venv_interpreter_is_looked_for_under_both_layouts():
    """POSIX puts it in bin/python, Windows in Scripts/python.exe.

    Two maintenance scripts resolve this, and both used to check only the POSIX
    path -- silently falling through to whatever interpreter was running, which
    on a machine with a virtualenv is usually the wrong one. That is how the
    CPU-only gate came to fail with "nanobind not found" instead of anything to
    do with what it checks.

    Asserted by reading the scripts rather than by importing them, because they
    run as programs and have side effects at import.
    """
    repo = Path(__file__).resolve().parents[2]
    for name in ("check_cpu_only_build.py", "mutation_check.py"):
        source = (repo / "scripts" / name).read_text(encoding="utf-8")
        assert ".venv/bin/python" in source, f"{name}: no POSIX venv path"
        assert ".venv/Scripts/python.exe" in source, f"{name}: no Windows venv path"


def test_this_platform_is_one_the_suite_knows_about():
    """A canary, not a constraint.

    If vkml is ever run somewhere that is neither POSIX nor Windows, the
    assumptions in this file need revisiting rather than silently holding.
    """
    assert os.name in ("posix", "nt"), f"unrecognised os.name {os.name!r} on {sys.platform}"
