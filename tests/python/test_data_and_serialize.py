"""Data pipeline and checkpoint format.

Neither of these needs PyTorch as an oracle: there is no numerical contract
here, only bookkeeping. What can go wrong is bookkeeping going wrong silently --
a shuffle that drops samples, two arrays permuted apart from each other, a
checkpoint that loads with a key missing. Each of those trains a model that
looks fine and is not, so the tests below are written to catch exactly them.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

import vkml as V
from vkml.data import (ArrayDataset, Compose, DataLoader, RandomCrop,
                       RandomHorizontalFlip, split)
from vkvalidate import gpu_device, vulkan_ready

#: Applied per test rather than to the file: the CPU direction of the
#: device-preservation rule must still run in the three CPU-only CI jobs.
requires_vulkan = pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def test_array_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="first axis"):
        ArrayDataset(np.zeros((10, 3)), np.zeros(9))


def test_array_dataset_rejects_empty():
    with pytest.raises(ValueError, match="at least one array"):
        ArrayDataset()


def test_array_dataset_indexes_every_array():
    x = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    dataset = ArrayDataset(x, y)

    xb, yb = dataset[np.array([3, 1])]

    assert np.array_equal(xb, x[[3, 1]])
    assert np.array_equal(yb, y[[3, 1]])


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_unshuffled_loader_preserves_order():
    x = np.arange(10)
    loader = DataLoader(ArrayDataset(x), batch_size=3)

    seen = np.concatenate([batch[0] for batch in loader])

    assert np.array_equal(seen, x)


@pytest.mark.parametrize("n,batch_size,drop_last,expected", [
    (10, 3, False, 4),   # 3 + 3 + 3 + 1
    (10, 3, True, 3),    # the trailing 1 is dropped
    (9, 3, True, 3),     # exact division drops nothing
    (9, 3, False, 3),
    (2, 3, True, 0),     # fewer samples than a batch
    (2, 3, False, 1),
])
def test_len_matches_batches_yielded(n, batch_size, drop_last, expected):
    loader = DataLoader(ArrayDataset(np.arange(n)), batch_size=batch_size,
                        drop_last=drop_last)

    assert len(loader) == expected
    assert len(list(loader)) == expected


@pytest.mark.parametrize("n", [7, 8, 9, 10, 11])
def test_drop_last_keeps_every_batch_the_same_shape(n):
    """Parametrised over n because the boundary is n = -1 (mod batch_size).

    That is the only length at which an off-by-one in the stop bound admits a
    short batch; at every other length the two agree, so a single fixed n can
    pass while the bound is wrong.
    """
    loader = DataLoader(ArrayDataset(np.arange(n)), batch_size=4, drop_last=True)

    shapes = {batch[0].shape for batch in loader}

    assert shapes <= {(4,)}


def test_epoch_is_a_permutation_not_a_sample():
    """Every sample exactly once.

    A loader that drew random indices instead of permuting would pass a test
    that only checked the batch count, while showing some samples twice per
    epoch and others never. That biases training in a way no loss curve reveals.
    """
    n = 100
    loader = DataLoader(ArrayDataset(np.arange(n)), batch_size=10, shuffle=True, seed=7)

    seen = np.concatenate([batch[0] for batch in loader])

    assert np.array_equal(np.sort(seen), np.arange(n))


def test_shuffle_keeps_paired_arrays_aligned():
    """The defect this exists for: permuting inputs and labels independently.

    y is a function of x, so a misalignment is detectable rather than merely
    suspicious. Training with shuffled-apart labels converges to nothing and
    looks like a bad hyperparameter.
    """
    x = np.arange(60).reshape(20, 3)
    y = x.sum(axis=1)
    loader = DataLoader(ArrayDataset(x, y), batch_size=4, shuffle=True, seed=3)

    for xb, yb in loader:
        assert np.array_equal(yb, xb.sum(axis=1))


def test_same_seed_reproduces_the_run():
    def epochs(loader, count):
        return [np.concatenate([b[0] for b in loader]) for _ in range(count)]

    a = DataLoader(ArrayDataset(np.arange(50)), batch_size=5, shuffle=True, seed=11)
    b = DataLoader(ArrayDataset(np.arange(50)), batch_size=5, shuffle=True, seed=11)

    for left, right in zip(epochs(a, 3), epochs(b, 3)):
        assert np.array_equal(left, right)


def test_different_seeds_give_different_orders():
    a = DataLoader(ArrayDataset(np.arange(50)), batch_size=5, shuffle=True, seed=1)
    b = DataLoader(ArrayDataset(np.arange(50)), batch_size=5, shuffle=True, seed=2)

    first = np.concatenate([batch[0] for batch in a])
    second = np.concatenate([batch[0] for batch in b])

    assert not np.array_equal(first, second)


def test_successive_epochs_differ():
    """A loader that reshuffled from the same seed every epoch would repeat one
    order forever -- reproducible, and useless as a shuffle."""
    loader = DataLoader(ArrayDataset(np.arange(50)), batch_size=5, shuffle=True, seed=0)

    first = np.concatenate([batch[0] for batch in loader])
    second = np.concatenate([batch[0] for batch in loader])

    assert not np.array_equal(first, second)


def test_the_order_is_fixed_when_iteration_starts_not_when_it_is_consumed():
    """Two iterators taken before either is read must see different epochs.

    If __iter__ were a generator function the draw would be deferred to the
    first next(), and both would see whatever epoch happened to be current when
    they were first pulled -- the same one, in the natural usage.
    """
    loader = DataLoader(ArrayDataset(np.arange(40)), batch_size=8, shuffle=True, seed=6)

    first, second = iter(loader), iter(loader)

    assert not np.array_equal(
        np.concatenate([b[0] for b in first]), np.concatenate([b[0] for b in second]))


def test_set_epoch_pins_the_order():
    """What makes resuming a run possible: epoch 3 is epoch 3 either way."""
    fresh = DataLoader(ArrayDataset(np.arange(40)), batch_size=8, shuffle=True, seed=5)
    for _ in range(3):
        list(fresh)
    expected = np.concatenate([batch[0] for batch in fresh])

    resumed = DataLoader(ArrayDataset(np.arange(40)), batch_size=8, shuffle=True, seed=5)
    resumed.set_epoch(3)
    got = np.concatenate([batch[0] for batch in resumed])

    assert np.array_equal(got, expected)


def test_unshuffled_loader_repeats_exactly():
    loader = DataLoader(ArrayDataset(np.arange(20)), batch_size=5)

    first = np.concatenate([batch[0] for batch in loader])
    second = np.concatenate([batch[0] for batch in loader])

    assert np.array_equal(first, second)


def test_rejects_nonpositive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        DataLoader(ArrayDataset(np.arange(4)), batch_size=0)


def test_device_yields_tensors_with_the_same_values():
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    plain = DataLoader(ArrayDataset(x), batch_size=4)
    placed = DataLoader(ArrayDataset(x), batch_size=4, device=V.cpu)

    for (want,), (got,) in zip(plain, placed):
        assert isinstance(got, V.Tensor)
        assert np.array_equal(got.numpy(), want)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_split_is_disjoint_and_complete():
    x = np.arange(100)
    left, right = split(ArrayDataset(x), 0.2, seed=4)

    assert len(left) == 20
    assert len(right) == 80
    assert np.array_equal(np.sort(np.concatenate([left.arrays[0], right.arrays[0]])), x)


def test_split_keeps_pairs_together():
    x = np.arange(100)
    left, right = split(ArrayDataset(x, x * 3), 0.3, seed=9)

    for part in (left, right):
        assert np.array_equal(part.arrays[1], part.arrays[0] * 3)


def test_split_is_deterministic():
    x = np.arange(100)
    first, _ = split(ArrayDataset(x), 0.25, seed=2)
    second, _ = split(ArrayDataset(x), 0.25, seed=2)

    assert np.array_equal(first.arrays[0], second.arrays[0])


def test_split_shuffles_before_cutting():
    """A dataset ordered by class must not split into one class per side."""
    x = np.arange(100)
    left, _ = split(ArrayDataset(x), 0.5, seed=1)

    assert not np.array_equal(np.sort(left.arrays[0]), np.arange(50))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_degenerate_fractions(fraction):
    with pytest.raises(ValueError, match="fraction"):
        split(ArrayDataset(np.arange(10)), fraction)


def test_split_rejects_a_fraction_that_rounds_to_empty():
    with pytest.raises(ValueError, match="empty half"):
        split(ArrayDataset(np.arange(10)), 0.01)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_round_trip_preserves_values_dtypes_and_shapes(tmp_path):
    tensors = {
        "weight": np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32),
        "bias": np.zeros(3, dtype=np.float32),
        "count": np.array(7, dtype=np.int64),          # 0-d, and integral
        "flags": np.array([True, False], dtype=bool),
    }
    path = tmp_path / "ckpt.vkml"

    V.save(path, tensors)
    got = V.load(path)

    assert set(got.tensors) == set(tensors)
    for key, want in tensors.items():
        assert got.tensors[key].dtype == want.dtype
        assert got.tensors[key].shape == want.shape
        assert np.array_equal(got.tensors[key], want)   # exact: this is a copy, not a computation


def test_round_trip_preserves_key_order(tmp_path):
    keys = ["z", "a", "m", "b"]
    path = tmp_path / "ordered.vkml"

    V.save(path, {k: np.zeros(1) for k in keys})

    assert list(V.load(path).tensors) == keys


def test_metadata_round_trips(tmp_path):
    path = tmp_path / "meta.vkml"
    metadata = {"model": "cnn", "epoch": 3, "accuracy": 0.987, "layers": [784, 128, 10]}

    V.save(path, {"w": np.ones(2)}, metadata=metadata)

    assert V.load(path).metadata == metadata


def test_metadata_defaults_to_empty(tmp_path):
    path = tmp_path / "bare.vkml"
    V.save(path, {"w": np.ones(2)})

    assert V.load(path).metadata == {}


def test_compressed_and_stored_read_back_identically(tmp_path):
    tensors = {"w": np.zeros((64, 64), dtype=np.float32)}
    stored, deflated = tmp_path / "s.vkml", tmp_path / "d.vkml"

    V.save(stored, tensors)
    V.save(deflated, tensors, compress=True)

    assert np.array_equal(V.load(stored).tensors["w"], V.load(deflated).tensors["w"])
    assert deflated.stat().st_size < stored.stat().st_size   # zeros do compress


def test_module_round_trip_includes_buffers(tmp_path):
    """Buffers are what makes a checkpoint complete.

    BatchNorm restored without its running statistics evaluates against the
    wrong distribution while every weight is correct.
    """
    model = V.nn.BatchNorm2d(3)
    model.running_mean = V.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    path = tmp_path / "model.vkml"

    V.save_module(path, model, metadata={"model": "bn"})

    restored = V.nn.BatchNorm2d(3)
    checkpoint = V.load_module(path, restored)

    assert checkpoint.metadata == {"model": "bn"}
    for key, want in model.state_dict().items():
        assert np.array_equal(restored.state_dict()[key], want), key


# -- rejection --------------------------------------------------------------


def test_rejects_a_file_that_is_not_a_zip(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not a checkpoint")

    with pytest.raises(ValueError, match="not a zip archive"):
        V.load(path)


def test_rejects_a_plain_npz(tmp_path):
    """The format this replaces. It is a zip, so the error has to be specific."""
    path = tmp_path / "weights.npz"
    np.savez(path, w=np.ones(3))

    with pytest.raises(ValueError, match="vkml.json"):
        V.load(path)


def test_rejects_a_newer_format_version(tmp_path):
    path = tmp_path / "future.vkml"
    V.save(path, {"w": np.ones(2)})
    _rewrite_header(path, lambda h: {**h, "version": V.serialize.FORMAT_VERSION + 1})

    with pytest.raises(ValueError, match="Upgrade vkml"):
        V.load(path)


def test_rejects_a_missing_member(tmp_path):
    """A key listed but absent is a truncated or edited archive, not an empty
    tensor -- loading it would leave a parameter at its random init."""
    path = tmp_path / "partial.vkml"
    V.save(path, {"w": np.ones(2)})
    _rewrite_header(path, lambda h: {**h, "keys": h["keys"] + ["missing"]})

    with pytest.raises(ValueError, match="missing"):
        V.load(path)


def test_rejects_object_dtype_on_save(tmp_path):
    with pytest.raises(TypeError, match="pickle"):
        V.save(tmp_path / "obj.vkml", {"w": np.array([{"a": 1}], dtype=object)})


def test_rejects_non_string_keys(tmp_path):
    with pytest.raises(TypeError, match="keys must be strings"):
        V.save(tmp_path / "bad.vkml", {0: np.ones(2)})


class _Payload:
    """Stands in for a hostile checkpoint. Its `__reduce__` runs on unpickle."""

    executed = False

    def __reduce__(self):
        return (_mark_executed, ())


def _mark_executed():
    _Payload.executed = True
    return None


def test_a_pickle_payload_is_refused_rather_than_executed(tmp_path):
    """The guarantee the format exists for.

    Builds the attack rather than assuming the flag works: an array whose
    elements unpickle by *calling* something, written with allow_pickle=True and
    listed as a normal tensor. Load must refuse it, and the call must not have
    happened.
    """
    _Payload.executed = False

    hostile = io.BytesIO()
    np.lib.format.write_array(hostile, np.array([_Payload()], dtype=object), allow_pickle=True)

    path = tmp_path / "hostile.vkml"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("vkml.json", json.dumps({
            "format": "vkml-checkpoint",
            "version": V.serialize.FORMAT_VERSION,
            "keys": ["payload"],
            "metadata": {},
        }))
        archive.writestr("tensors/payload.npy", hostile.getvalue())

    with pytest.raises(ValueError, match="[Oo]bject arrays"):
        V.load(path)

    assert _Payload.executed is False


# -- durability -------------------------------------------------------------


def test_unserialisable_metadata_is_rejected_before_anything_is_written(tmp_path):
    """Cheapest of the two guards: the header is built before the file is
    opened, so a bad metadata value cannot reach the destination at all."""
    path = tmp_path / "epoch.vkml"
    V.save(path, {"w": np.ones(3)}, metadata={"epoch": 1})

    with pytest.raises(TypeError):
        V.save(path, {"w": np.zeros(3)}, metadata={"bad": {1, 2}})   # a set is not JSON

    assert V.load(path).metadata == {"epoch": 1}
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("failure", [RuntimeError("disk full"), KeyboardInterrupt()])
def test_a_save_interrupted_mid_write_leaves_the_previous_checkpoint_intact(
        tmp_path, monkeypatch, failure):
    """The case atomic writing exists for.

    Saving each epoch over one path is the normal pattern, so a write that dies
    partway must not take the last good checkpoint with it. The failure is
    injected into the array writer -- after the archive is open and the first
    bytes are committed -- because a guard that only runs before the file opens
    would pass a test built on unserialisable metadata while leaving a truncated
    file behind here.

    KeyboardInterrupt is parametrised in on its own: it is how a training run
    usually dies, and it is not an Exception, so cleanup that caught the narrower
    type would leak a .partial file for the one failure that happens most.
    """
    path = tmp_path / "epoch.vkml"
    V.save(path, {"w": np.ones(3)}, metadata={"epoch": 1})

    def explode(*args, **kwargs):
        raise failure

    monkeypatch.setattr(np.lib.format, "write_array", explode)

    with pytest.raises(type(failure)):
        V.save(path, {"w": np.zeros(3)}, metadata={"epoch": 2})

    monkeypatch.undo()

    checkpoint = V.load(path)
    assert checkpoint.metadata == {"epoch": 1}
    assert np.array_equal(checkpoint.tensors["w"], np.ones(3))
    assert not list(tmp_path.glob("*.partial"))


def test_overwriting_replaces_the_whole_checkpoint(tmp_path):
    """Not merges with it: a stale key surviving a rename would restore a layer
    that no longer exists."""
    path = tmp_path / "ckpt.vkml"
    V.save(path, {"old": np.ones(2), "shared": np.ones(2)})
    V.save(path, {"shared": np.zeros(2)})

    assert set(V.load(path).tensors) == {"shared"}


def _rewrite_header(path, transform):
    """Rebuild an archive with a modified vkml.json, to forge a bad checkpoint."""
    with zipfile.ZipFile(path, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}

    members["vkml.json"] = json.dumps(transform(json.loads(members["vkml.json"]))).encode()

    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


# ---------------------------------------------------------------------------
# Decompression bombs
#
# A small archive that expands enormously. `load` rejects it from the zip
# DIRECTORY, before reading a byte, so the allocation never happens. Bounded by
# expansion RATIO rather than absolute size -- see serialize.py's module
# docstring for the measurements behind that choice.
# ---------------------------------------------------------------------------


def _bomb(path, uncompressed_bytes: int = 64 * 1024 * 1024):
    """A valid vkml checkpoint whose tensor member is highly compressible."""
    payload = io.BytesIO()
    zeros = np.zeros((uncompressed_bytes // 4,), dtype=np.float32)
    np.lib.format.write_array(payload, zeros, allow_pickle=False)

    header = json.dumps({
        "format": "vkml-checkpoint", "version": 1, "keys": ["w"], "metadata": {},
    }).encode()

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("vkml.json", header)
        archive.writestr("tensors/w.npy", payload.getvalue())
    return path


def test_a_decompression_bomb_is_rejected(tmp_path):
    """The core case: tiny on disk, enormous in memory."""
    path = _bomb(tmp_path / "bomb.vkml")
    assert path.stat().st_size < 1_000_000, "test bomb should be small on disk"

    with pytest.raises(ValueError) as excinfo:
        V.load(path)
    message = str(excinfo.value)
    assert "expands" in message, message
    # Actionable: names the limit AND how to raise it.
    assert "max_expansion_ratio" in message, message


def test_the_bomb_is_rejected_without_being_read(tmp_path):
    """Rejection must cost nothing -- the point is that the allocation this
    guards against never happens.

    Asserted by making every member unreadable after the directory is written:
    the archive's compressed payload is overwritten with garbage, so any attempt
    to actually decompress raises BadZipFile instead. A guard that reads first
    and checks afterwards would surface that error instead of the size one.
    """
    path = _bomb(tmp_path / "bomb.vkml")
    raw = bytearray(path.read_bytes())
    # Corrupt the deflate stream of the first member, leaving the directory
    # intact so the declared sizes still parse.
    start = raw.find(b"PK\x03\x04")
    raw[start + 200:start + 400] = b"\x00" * 200
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="expands"):
        V.load(path)


def test_raising_the_limit_lets_a_genuine_file_through(tmp_path):
    """The escape hatch the error message promises must actually work -- a
    pruned model stored densely can compress this well and is not an attack."""
    path = _bomb(tmp_path / "sparse.vkml")
    checkpoint = V.load(path, max_expansion_ratio=float("inf"))
    assert "w" in checkpoint.tensors
    assert not checkpoint.tensors["w"].any(), "the payload is zeros"


def test_ordinary_checkpoints_are_nowhere_near_the_limit(tmp_path):
    """The guard must not fire on real data, compressed or not.

    This is the measurement the threshold was chosen from, kept executable: real
    weights are high-entropy and barely compress, so they sit ~100x below the
    limit rather than near it.
    """
    rng = np.random.default_rng(0)
    state = {"w": rng.standard_normal((256, 256)).astype(np.float32),
             "b": rng.standard_normal((256,)).astype(np.float32)}

    for compress in (False, True):
        path = tmp_path / f"real-{compress}.vkml"
        V.save(path, state, compress=compress)
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            ratio = sum(e.file_size for e in entries) / sum(e.compress_size for e in entries)
        assert ratio < 2.0, f"real weights expanded {ratio:.1f}x with compress={compress}"
        V.load(path)  # and it loads


def test_a_stored_uncompressed_archive_is_never_rejected(tmp_path):
    """compress=False is the default, and cannot expand at all. The ratio is
    exactly 1, which must not trip a boundary bug in the comparison."""
    state = {"w": np.zeros((1024,), dtype=np.float32)}   # zeros, but STORED
    path = tmp_path / "stored.vkml"
    V.save(path, state, compress=False)
    V.load(path)


# -- device preservation ----------------------------------------------------


@requires_vulkan
def test_load_state_dict_keeps_parameters_on_the_module_s_device(tmp_path):
    """Loading into a module already on a device must not move it off.

    THE ORDER IS THE TEST. Every other `load_state_dict` call in this suite
    builds the module, loads, and *then* moves it to a device -- and that order
    hid the defect completely, because the parameters were on the CPU at the
    moment of the load anyway. The order here is the opposite one, which is what
    resuming from a checkpoint actually looks like: you build the model, place
    it, and only then restore the weights into it. `load_module` can only do it
    this way round.

    The replacement tensor was built without a device, so it landed on the
    default one and silently relocated the whole model to the CPU. Nothing threw
    at the load. The next forward pass did, from matmul, reporting operands on
    different devices -- a message that names neither the load nor the parameter
    that moved.

    The forward pass at the end is the half that matters most: asserting on
    `.device` alone would still pass if the tensors were on the right device but
    unusable, and the cross-device error is the symptom a user actually meets.
    """
    dev = gpu_device()
    model = V.nn.Sequential(V.nn.Linear(6, 4), V.nn.ReLU()).to(dev)
    path = tmp_path / "m.vkml"
    V.save_module(path, model)

    restored = V.nn.Sequential(V.nn.Linear(6, 4), V.nn.ReLU()).to(dev)
    V.load_module(path, restored)

    for name, p in restored.named_parameters():
        assert str(p.device) == str(dev), f"parameter '{name}' moved to {p.device}"
    for name, b in restored.named_buffers():
        assert str(b.device) == str(dev), f"buffer '{name}' moved to {b.device}"

    x = V.tensor(np.arange(12, dtype=np.float32).reshape(2, 6), device=dev)
    got = restored(x).numpy()
    want = model(x).numpy()
    assert np.array_equal(got, want), "restored module computes a different result"


def test_load_state_dict_keeps_parameters_on_the_cpu(tmp_path):
    """The inverse direction, so the fix cannot become a GPU special case.

    THIS ONE DOES NOT FAIL AGAINST THE ORIGINAL DEFECT, and saying so is the
    point: the missing argument sent parameters to the DEFAULT device, which on
    CPU is where they already belonged, so the bug was invisible here. Checked by
    reverting the fix -- the two GPU cases go red and this stays green.

    It is kept as a directional guard rather than a regression test. The fix
    could be wrong in the other direction -- pinning a device rather than
    carrying the existing one -- and that mistake would pass both GPU cases while
    breaking every CPU user. This is also the only one of the three that runs in
    the three CPU-only CI jobs, which build without Vulkan at all.
    """
    model = V.nn.Sequential(V.nn.Linear(6, 4), V.nn.ReLU())
    path = tmp_path / "m.vkml"
    V.save_module(path, model)

    restored = V.nn.Sequential(V.nn.Linear(6, 4), V.nn.ReLU())
    V.load_module(path, restored)

    for name, p in restored.named_parameters():
        assert str(p.device) == "cpu", f"parameter '{name}' moved to {p.device}"

    x = V.tensor(np.arange(12, dtype=np.float32).reshape(2, 6))
    assert np.array_equal(restored(x).numpy(), model(x).numpy())


@requires_vulkan
def test_load_state_dict_preserves_dtype_and_requires_grad_with_the_device(tmp_path):
    """All three properties belong to the entry being replaced, not to the array.

    device was the one that was dropped; dtype and requires_grad were already
    handled. Pinned together because they are one rule -- the next person adding
    a fourth property should see the whole set in one place rather than
    discovering it the way this was discovered.

    BatchNorm2d carries the awkward cases: a trained affine pair that must keep
    requires_grad, running statistics that must not gain it, and an integral
    `num_batches_tracked` counter that must stay int64 rather than becoming f32.
    """
    dev = gpu_device()
    model = V.nn.BatchNorm2d(3).to(dev)
    path = tmp_path / "bn.vkml"
    V.save_module(path, model)

    restored = V.nn.BatchNorm2d(3).to(dev)
    before = {n: (str(t.device), t.dtype, t.requires_grad)
              for n, t in [*restored.named_parameters(), *restored.named_buffers()]}
    V.load_module(path, restored)
    after = {n: (str(t.device), t.dtype, t.requires_grad)
             for n, t in [*restored.named_parameters(), *restored.named_buffers()]}

    assert after == before, f"a property changed across the load: {before} -> {after}"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
#
# The failure these are written against is not a crash. An augmentation that
# draws one coin for the whole batch, or that draws the same coin every epoch,
# still runs, still trains, and quietly does something other than augmenting —
# it correlates a step's samples, or doubles the dataset instead of multiplying
# it. Every test below is aimed at that class rather than at shapes.


def _images(n=8, c=3, h=6, w=6, seed=0):
    return np.random.default_rng(seed).random((n, c, h, w)).astype(np.float32)


def test_a_transform_sees_the_whole_batch_and_can_change_it():
    seen = []

    def record(rng, arrays):
        seen.append(arrays[0].shape)
        return (arrays[0] * 0.0,) + tuple(arrays[1:])

    x, y = _images(8), np.arange(8, dtype=np.int64)
    loader = DataLoader(ArrayDataset(x, y), batch_size=4, transform=record)
    batches = list(loader)

    assert seen == [(4, 3, 6, 6), (4, 3, 6, 6)], "the transform did not receive whole batches"
    assert all(np.all(bx == 0.0) for bx, _ in batches), "the transform's output was discarded"
    assert all(len(b) == 2 for b in batches), "the label was dropped"


def test_a_transform_is_reproducible_from_the_seed():
    """The property the rng-as-argument contract exists to guarantee."""
    x = _images(16)

    def run(seed):
        loader = DataLoader(ArrayDataset(x), batch_size=4, shuffle=True, seed=seed,
                            transform=RandomHorizontalFlip(p=0.5))
        return [b[0].copy() for b in loader]

    a, b = run(7), run(7)
    assert all(np.array_equal(p, q) for p, q in zip(a, b)), (
        "the same seed produced different augmentation"
    )
    c = run(8)
    assert any(not np.array_equal(p, q) for p, q in zip(a, c)), (
        "a different seed produced identical augmentation; the seed is being ignored"
    )


def test_augmentation_differs_between_epochs():
    """Augmentation that repeats each epoch is a bigger fixed dataset, not
    augmentation."""
    x = _images(16)
    loader = DataLoader(ArrayDataset(x), batch_size=16,
                        transform=RandomHorizontalFlip(p=0.5))
    first = next(iter(loader))[0].copy()
    second = next(iter(loader))[0].copy()
    assert not np.array_equal(first, second), (
        "two epochs produced identical augmentation; the generator is not advancing"
    )


def test_the_transform_draws_from_its_own_stream_not_the_shuffle():
    """The two generators are separate on purpose.

    Sharing one between the shuffle and the transform would make the same seed
    give different augmentation depending on whether shuffling was on — a
    difference nobody asked for and nobody would look for.

    Checked by recording what the transform DRAWS, which is the thing that must
    not move. Comparing output arrays cannot do it: the samples arrive in a
    different order, so equal draws produce unequal batches either way.
    """
    def run(shuffle):
        drawn = []
        loader = DataLoader(ArrayDataset(_images(8)), batch_size=4, shuffle=shuffle, seed=3,
                            transform=lambda rng, a: (drawn.append(float(rng.random())), a)[1])
        list(loader)
        return drawn

    assert run(True) == run(False), (
        "the transform's random draws changed when shuffling was turned on; the two "
        "streams are not independent"
    )


def test_random_horizontal_flip_decides_per_sample():
    """One coin for the whole batch is the classic way this goes wrong."""
    x = _images(64, seed=1)
    loader = DataLoader(ArrayDataset(x), batch_size=64,
                        transform=RandomHorizontalFlip(p=0.5))
    out = next(iter(loader))[0]

    flipped = [np.array_equal(out[i], x[i][:, :, ::-1]) for i in range(64)]
    kept = [np.array_equal(out[i], x[i]) for i in range(64)]
    assert all(f or k for f, k in zip(flipped, kept)), "a sample is neither flipped nor kept"
    assert any(flipped) and any(kept), (
        f"{sum(flipped)} of 64 flipped; a per-sample decision should give both outcomes "
        f"in a batch this size, and all-or-nothing means one coin was drawn for the batch"
    )


@pytest.mark.parametrize("p,want", [(0.0, False), (1.0, True)])
def test_random_horizontal_flip_endpoints(p, want):
    x = _images(4, seed=2)
    out = RandomHorizontalFlip(p=p)(np.random.default_rng(0), (x,))[0]
    assert np.array_equal(out, x[:, :, :, ::-1] if want else x)


def test_random_crop_with_no_padding_and_a_full_size_window_is_the_identity():
    x = _images(4, h=8, w=8, seed=3)
    out = RandomCrop(8, padding=0)(np.random.default_rng(0), (x,))[0]
    assert np.array_equal(out, x)


def test_random_crop_translates_each_sample_independently():
    """The point of the crop: a sample must be able to move differently from its
    neighbour, or the batch is translated as a block and a convolution still
    sees a fixed relationship.

    The offset is RECOVERED rather than guessed at. Each image is filled with
    `row * W + col`, so the padded window's top-left value names the offset
    exactly — and the assertion is that several distinct offsets occur, which is
    a statement about the batch rather than about a particular seed. An earlier
    version asked whether any sample landed on the identity crop; with 25
    possible offsets that is true 92% of the time, so it was a test that failed
    one run in thirteen for no reason.
    """
    h = w = 8
    grid = (np.arange(h * w, dtype=np.float32).reshape(1, 1, h, w))
    x = np.repeat(grid, 64, axis=0)
    out = RandomCrop(8, padding=2, fill=-1.0)(np.random.default_rng(5), (x,))[0]
    assert out.shape == x.shape

    corners = {float(out[i, 0, 0, 0]) for i in range(64)}
    assert len(corners) > 1, (
        f"every sample in the batch was cropped at the same offset ({corners}); "
        f"the offset is drawn once per batch rather than once per sample"
    )


def test_random_crop_keeps_the_layout_and_contiguity():
    """The gather reorders axes internally; (N, C, H, W) has to come back out."""
    x = _images(4, c=3, h=10, w=12, seed=6)
    out = RandomCrop((6, 8), padding=1)(np.random.default_rng(0), (x,))[0]
    assert out.shape == (4, 3, 6, 8)
    assert out.dtype == x.dtype
    assert out.flags["C_CONTIGUOUS"], "a non-contiguous batch would copy again at upload"


def test_random_crop_pads_with_fill():
    """With enough padding the window can land entirely outside the image."""
    x = np.ones((1, 1, 2, 2), dtype=np.float32)
    out = RandomCrop(2, padding=2, fill=0.0)(np.random.default_rng(0), (x,))[0]
    assert out.shape == (1, 1, 2, 2)
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_compose_runs_in_order_and_shares_one_generator():
    order = []

    class Mark:
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, rng, arrays):
            order.append((self.tag, float(rng.random())))
            return arrays

    Compose(Mark("a"), Mark("b"))(np.random.default_rng(0), (np.zeros((1, 1, 1, 1)),))
    assert [tag for tag, _ in order] == ["a", "b"]
    assert order[0][1] != order[1][1], (
        "both transforms drew the same number; they are not sharing one generator"
    )


def test_transforms_pass_the_label_through_untouched():
    x, y = _images(8), np.arange(8, dtype=np.int64)
    pipeline = Compose(RandomHorizontalFlip(p=1.0), RandomCrop(6, padding=1))
    out_x, out_y = pipeline(np.random.default_rng(0), (x, y))
    assert np.array_equal(out_y, y), "a spatial transform altered the labels"
    assert out_x.shape[0] == 8


def test_transforms_reject_what_they_cannot_operate_on():
    rng = np.random.default_rng(0)
    flat = np.zeros((4, 16), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(N, C, H, W\)"):
        RandomHorizontalFlip()(rng, (flat,))
    with pytest.raises(ValueError, match=r"\(N, C, H, W\)"):
        RandomCrop(4)(rng, (flat,))
    with pytest.raises(ValueError, match="does not fit"):
        RandomCrop(9)(rng, (_images(2, h=6, w=6),))
    with pytest.raises(ValueError):
        RandomHorizontalFlip(p=1.5)
    with pytest.raises(ValueError):
        RandomCrop(4, padding=-1)
    with pytest.raises(TypeError):
        Compose(object())


def test_a_transform_returning_a_bare_array_is_rejected():
    """It would silently change the batch's arity, and the unpack error would
    surface at the caller rather than here."""
    loader = DataLoader(ArrayDataset(_images(4)), batch_size=4,
                        transform=lambda rng, arrays: arrays[0])
    with pytest.raises(TypeError, match="tuple of arrays"):
        list(loader)


def test_a_non_callable_transform_is_rejected_at_construction():
    with pytest.raises(TypeError):
        DataLoader(ArrayDataset(_images(4)), batch_size=2, transform="flip")


@pytest.mark.parametrize("device_arg", [False, True])
def test_the_transform_runs_before_the_device_transfer(device_arg):
    """Augmenting after the upload would need every transform written twice."""
    if device_arg and not V.vulkan_available():
        pytest.skip("no Vulkan device")
    x = _images(4)
    device = V.device("vulkan:0") if device_arg else None
    loader = DataLoader(ArrayDataset(x), batch_size=4, device=device,
                        transform=RandomHorizontalFlip(p=1.0))
    got = next(iter(loader))[0]
    got = got.numpy() if device_arg else got
    assert np.allclose(got, x[:, :, :, ::-1])
