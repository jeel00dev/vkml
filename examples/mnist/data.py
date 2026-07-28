"""MNIST loading.

The IDX files are fetched once into a cache directory and read directly. Two
reasons not to reach for a dataset library: it would add a dependency for four
files with a format that fits on a page, and vkml has no DataLoader yet -- the
point of running this example is to find out what one should look like, so
inventing the API first would be guessing.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

# Mirrors, tried in order. The canonical yann.lecun.com host has been
# unreliable for years; these two carry byte-identical copies.
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

# Expected element count per file, checked after parsing so a truncated or
# swapped download fails loudly rather than training on half a dataset.
FILES = {
    "train-images-idx3-ubyte.gz": 60000,
    "train-labels-idx1-ubyte.gz": 60000,
    "t10k-images-idx3-ubyte.gz": 10000,
    "t10k-labels-idx1-ubyte.gz": 10000,
}

CACHE = Path(__file__).resolve().parent / "cache"


def _download(name: str) -> Path:
    path = CACHE / name
    if path.exists():
        return path

    CACHE.mkdir(parents=True, exist_ok=True)
    errors = []
    for mirror in MIRRORS:
        try:
            with urllib.request.urlopen(mirror + name, timeout=60) as response:
                payload = response.read()
            # Write via a temporary and rename, so an interrupted download
            # cannot leave a truncated file that looks cached next time.
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.rename(path)
            return path
        except Exception as exc:  # noqa: BLE001 - report every mirror's failure
            errors.append(f"{mirror}: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"could not fetch {name}\n  " + "\n  ".join(errors))


# IDX header: two zero bytes, a type code, then the rank -- followed by one
# 32-bit big-endian extent per axis. 0x08 is unsigned byte, which is what every
# MNIST file uses for both pixels and labels.
IDX_UNSIGNED_BYTE = 0x08


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        zero, type_code, ndim = struct.unpack(">HBB", handle.read(4))
        if zero != 0 or type_code != IDX_UNSIGNED_BYTE:
            raise ValueError(
                f"{path.name}: not an unsigned-byte IDX file "
                f"(leading {zero}, type 0x{type_code:02x})"
            )
        shape = struct.unpack(f">{ndim}I", handle.read(4 * ndim))
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(shape)


def load(normalise: bool = True) -> dict[str, np.ndarray]:
    """The four arrays, cached on disk after the first call.

    Images come back as float32 in [0, 1] shaped (N, 1, 28, 28) -- the layout a
    conv layer wants, and one an MLP can flatten. Labels are int64, which is
    what cross_entropy takes.
    """
    raw = {name: _read_idx(_download(name)) for name in FILES}

    for name, expected in FILES.items():
        got = len(raw[name])
        if got != expected:
            raise ValueError(f"{name}: expected {expected} items, read {got}")

    def images(key: str) -> np.ndarray:
        pixels = raw[key].astype(np.float32).reshape(-1, 1, 28, 28)
        return pixels / 255.0 if normalise else pixels

    return {
        "train_x": images("train-images-idx3-ubyte.gz"),
        "train_y": raw["train-labels-idx1-ubyte.gz"].astype(np.int64),
        "test_x": images("t10k-images-idx3-ubyte.gz"),
        "test_y": raw["t10k-labels-idx1-ubyte.gz"].astype(np.int64),
    }


if __name__ == "__main__":
    d = load()
    for key, value in d.items():
        print(f"{key:9} {str(value.shape):20} {value.dtype}  "
              f"range [{value.min()}, {value.max()}]")
