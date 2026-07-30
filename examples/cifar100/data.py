"""CIFAR-100 loading.

Same geometry as CIFAR-10 -- 32x32 RGB, 50,000 train and 10,000 test -- with 100
fine classes instead of 10. For the purpose this example serves, exercising the
conv/pool path on something MNIST is too small to stress, the two are
interchangeable, and the wider classifier head is if anything the better test.

THE PICKLE PROBLEM, AND WHAT IS DONE ABOUT IT
---------------------------------------------
CIFAR-100 is distributed only as Python pickles. A pickle names objects to
import and call while it loads, which is why vkml's own checkpoint format
refuses pickle outright (python/vkml/serialize.py) -- a model file must not be
able to run a program.

That stance cannot be met here by choosing a different distribution, because
there is no other distribution. It is met in the loader instead. `find_class`
is the single door through which a pickle reaches code, and _NumpyOnlyUnpickler
opens it for a handful of numpy names and refuses everything else.

The allowlist was not guessed. pickletools.genops parses a pickle's opcodes
WITHOUT executing them, and run over both data files it reports exactly three
globals -- numpy.core.multiarray._reconstruct, numpy.ndarray and numpy.dtype --
which is the standard array-rebuilding triple and nothing more. The allowlist is
enforced regardless of that finding, so a substituted archive gains nothing by
naming something else.

Check it yourself rather than trusting this docstring:

    python examples/cifar100/data.py --audit
"""

from __future__ import annotations

import gzip
import io
import pickle
import pickletools
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"

# Whichever container is present. The canonical download is a .tar.gz; the copy
# that prompted this example was a .zip of the same three members, so both are
# accepted rather than making anyone repackage 168 MB.
CONTAINERS = ("cifar-100-python.zip", "cifar-100-python.tar.gz")
SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

MEMBERS = {
    "train": "cifar-100-python/train",
    "test": "cifar-100-python/test",
    "meta": "cifar-100-python/meta",
}

EXPECTED_ROWS = {"train": 50000, "test": 10000}
PIXELS_PER_IMAGE = 3 * 32 * 32
FINE_CLASSES = 100


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """A pickle reader that can rebuild a numpy array and do nothing else.

    Every other global -- os.system, builtins.eval, anything at all -- raises
    instead of importing. This is the whole security boundary for this file.
    """

    ALLOWED = frozenset({
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        # numpy 2 moved the private module and keeps the old path as an alias,
        # so a pickle written by either numpy must load under either.
        ("numpy._core.multiarray", "_reconstruct"),
    })

    def find_class(self, module: str, name: str):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(
                f"refusing to import '{module}.{name}' while unpickling CIFAR-100. "
                f"This loader permits only numpy array reconstruction; anything else "
                f"means the archive is not the dataset it claims to be."
            )
        return super().find_class(module, name)


def _archive_path() -> Path:
    for name in CONTAINERS:
        path = CACHE / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no CIFAR-100 archive in {CACHE}. Download {SOURCE_URL} and put it there "
        f"under one of these names: {list(CONTAINERS)}."
    )


def _member_bytes() -> dict[str, bytes]:
    """The three pickles, read out of whichever container is present.

    Members are read BY NAME. extractall() is never used: a crafted entry can
    carry a path that escapes the destination, and naming what you want means a
    doctored archive has nowhere to write.
    """
    path = _archive_path()
    try:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                return {key: archive.read(member) for key, member in MEMBERS.items()}

        with tarfile.open(path, "r:gz") as archive:
            out = {}
            for key, member in MEMBERS.items():
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"{member} is missing from {path.name}")
                out[key] = handle.read()
            return out
    except (EOFError, tarfile.ReadError, zipfile.BadZipFile, gzip.BadGzipFile) as exc:
        # An INCOMPLETE archive, which is likelier here than a missing one: the
        # download is manual precisely because the host dislikes being scripted
        # at, and it throttles hard enough that interrupting it is the normal
        # outcome. Untranslated, this surfaced as a twenty-line traceback ending
        # in `EOFError: Compressed file ended before the end-of-stream marker was
        # reached` -- which never names CIFAR-100, this file, or the remedy
        # (issue #18).
        #
        # _archive_path() already says the right thing when the file is absent;
        # this is the same courtesy for a file that is present and unusable.
        raise ValueError(
            f"{path} is not a complete archive ({path.stat().st_size:,} bytes): "
            f"{type(exc).__name__}: {exc}. The download was probably interrupted -- "
            f"{SOURCE_URL} is slow and throttles. Delete the file and fetch it again."
        ) from exc


def _unpickle(payload: bytes) -> dict:
    # encoding="bytes" because these were written by Python 2: its str is a byte
    # string, and the default ascii decoding dies on the first pixel above 0x7f.
    # It is also why every key in the result is bytes rather than str.
    #
    # KNOWN FUTURE BREAK, left visible rather than silenced: these pickles store
    # dtype alignment as the integer 0, which numpy 2.4 deprecates in favour of
    # a bool, so loading emits a VisibleDeprecationWarning and will eventually
    # fail outright. Suppressing it would hide the deadline. When numpy removes
    # the shim, the fix is to convert the archive once into .npy files -- which
    # is also what vkml's own serialisation format would have done from the
    # start. Revisit when the warning becomes an error.
    return _NumpyOnlyUnpickler(io.BytesIO(payload), encoding="bytes").load()


def audit() -> None:
    """Report every global the archive's pickles name, without executing them.

    The design of this file rests on a claim about what is in those pickles.
    genops walks the opcode stream and calls nothing, so the claim is checkable
    rather than something to take on faith.
    """
    allowed = {f"{module} {name}" for module, name in _NumpyOnlyUnpickler.ALLOWED}
    risky = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "INST", "OBJ",
             "BUILD", "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4"}

    for key, payload in _member_bytes().items():
        named = sorted({arg for op, arg, _ in pickletools.genops(io.BytesIO(payload))
                        if op.name in risky and op.name == "GLOBAL"})
        outside = [name for name in named if name not in allowed]
        print(f"{key:6} {len(payload):>12,} bytes")
        print(f"       globals named:         {named or 'none'}")
        print(f"       outside the allowlist: {outside or 'none'}")


def load(normalise: bool = True, coarse: bool = False) -> dict[str, np.ndarray]:
    """The four arrays, read from the cached archive.

    Images come back float32 shaped (N, 3, 32, 32); labels int64, which is what
    cross_entropy takes. `coarse` selects the 20 superclasses instead of the 100
    fine ones -- the dataset ships both and the choice belongs to the caller.

    `normalise` scales to [0, 1]. Per-channel standardisation is deliberately
    left out: it needs statistics this function has no opinion about.
    """
    raw = _member_bytes()
    label_key = b"coarse_labels" if coarse else b"fine_labels"

    def split(key: str) -> tuple[np.ndarray, np.ndarray]:
        record = _unpickle(raw[key])
        if label_key not in record or b"data" not in record:
            raise ValueError(
                f"{MEMBERS[key]}: expected keys {label_key!r} and b'data', "
                f"found {sorted(record)!r}"
            )

        images = np.asarray(record[b"data"], dtype=np.uint8)
        labels = np.asarray(record[label_key], dtype=np.int64)

        rows = EXPECTED_ROWS[key]
        if images.shape != (rows, PIXELS_PER_IMAGE):
            raise ValueError(
                f"{MEMBERS[key]}: expected data of shape {(rows, PIXELS_PER_IMAGE)}, "
                f"got {images.shape}"
            )
        if labels.shape != (rows,):
            raise ValueError(f"{MEMBERS[key]}: expected {rows} labels, got {labels.shape}")

        # The 3072 values are three 1024-byte planes, so this reshape yields the
        # (C, H, W) the conv layers want rather than a transpose of it.
        return images.reshape(rows, 3, 32, 32), labels

    train_x, train_y = split("train")
    test_x, test_y = split("test")

    def pixels(images: np.ndarray) -> np.ndarray:
        out = images.astype(np.float32)
        return out / 255.0 if normalise else out

    return {
        "train_x": pixels(train_x),
        "train_y": train_y,
        "test_x": pixels(test_x),
        "test_y": test_y,
    }


def class_names(coarse: bool = False) -> list[str]:
    """Human-readable label names, from the archive's own meta pickle."""
    meta = _unpickle(_member_bytes()["meta"])
    key = b"coarse_label_names" if coarse else b"fine_label_names"
    return [name.decode() if isinstance(name, bytes) else name for name in meta[key]]


if __name__ == "__main__":
    if "--audit" in sys.argv:
        audit()
        raise SystemExit(0)

    d = load()
    for key, value in d.items():
        print(f"{key:9} {str(value.shape):22} {value.dtype}  "
              f"range [{value.min()}, {value.max()}]")
    counts = np.bincount(d["train_y"], minlength=FINE_CLASSES)
    print(f"classes: {FINE_CLASSES}, per-class train count "
          f"min {counts.min()} max {counts.max()}")
    print("first five names:", class_names()[:5])
