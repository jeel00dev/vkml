"""Saving and loading model state.

THE FORMAT IS A ZIP CONTAINING ONLY DATA. Never code.

That is the whole design, and it is a security decision before it is an
engineering one. The well-known failure in this field is a model format that
deserialises into code execution: Python's ``pickle`` reconstructs objects by
*calling* what the stream names, so loading a checkpoint from anywhere but your
own disk is running a program someone else wrote. Formats built on it inherit
that, which is why the ecosystem has been migrating away from them.

A vkml checkpoint has no mechanism for it. Every member is either a ``.npy``
array read with ``allow_pickle=False``, or one JSON document. Neither can name a
callable, so a malicious file has nothing to reach for. The parsing of the
array bytes is delegated to numpy's own reader rather than hand-written here --
a hand-rolled binary parser is exactly where a memory-safety bug would go, and
numpy's has far more scrutiny than this file will ever get.

LAYOUT

    vkml.json          format identifier, version, key list, user metadata
    tensors/<key>.npy  one array per state_dict entry

WHAT THIS DOES NOT DEFEND AGAINST. A decompression bomb: a small archive can
expand to a large allocation, and load will attempt it. That costs memory, not
control, so it is a denial of service against a process that already chose to
open the file. Recorded rather than fixed because a size cap needs a threshold,
and there is no evidence yet for what it should be.

VERSIONING. `FORMAT_VERSION` increments when the layout changes in a way an
older reader would misread. Loading a newer checkpoint fails with a message
naming both versions instead of misinterpreting the bytes.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as _np
from numpy.lib import format as _npy   # the .npy reader/writer, imported explicitly

FORMAT_VERSION = 1

_MAGIC = "vkml-checkpoint"
_METADATA_MEMBER = "vkml.json"
_TENSOR_PREFIX = "tensors/"


@dataclass
class Checkpoint:
    """What `load` returns: the arrays, plus whatever was recorded alongside.

    Kept as two fields rather than one merged mapping so a metadata key can
    never collide with a parameter name.
    """

    tensors: dict[str, _np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = FORMAT_VERSION

    def __repr__(self) -> str:
        return (f"Checkpoint({len(self.tensors)} tensors, "
                f"metadata={sorted(self.metadata)}, version={self.version})")


def save(path, tensors: Mapping[str, _np.ndarray], metadata: Mapping[str, Any] | None = None,
         compress: bool = False) -> None:
    """Write a state dict, and optionally metadata describing it.

    `metadata` must be JSON-representable -- the architecture name, the epoch,
    the hyperparameters. It is not a general object store, deliberately: the
    moment a format can carry arbitrary objects it needs an arbitrary
    deserialiser, which is the thing this format exists to avoid.

    WRITES ATOMICALLY. A checkpoint is written to a temporary file beside the
    destination and renamed over it, so an interrupt during the write leaves the
    previous checkpoint intact. Saving every epoch over one path is the normal
    pattern, and a truncated file there costs the whole run.
    """
    path = Path(path)

    # Convert and validate everything up front, so a bad entry is rejected
    # before the file is opened rather than partway through writing it.
    arrays: dict[str, _np.ndarray] = {}
    for key, value in tensors.items():
        if not isinstance(key, str):
            raise TypeError(f"tensor keys must be strings, got {type(key).__name__}")
        array = _np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(
                f"'{key}' has dtype {array.dtype}, which can only be serialised through "
                "pickle; vkml checkpoints store plain numeric arrays only"
            )
        arrays[key] = array

    metadata = dict(metadata or {})
    header = {
        "format": _MAGIC,
        "version": FORMAT_VERSION,
        "keys": list(arrays),
        "metadata": metadata,
    }
    # Serialise the header before opening the file: a metadata value that is not
    # JSON-representable should fail before anything is written, not leave a
    # half-built archive behind.
    header_bytes = json.dumps(header, indent=2, sort_keys=False).encode("utf-8")

    # ZIP_STORED by default, because trained float32 weights barely compress:
    # they are not sparse and their low mantissa bits are effectively noise. On
    # the MNIST MLP checkpoint deflate returned 93.5 % of the stored size for
    # 7x the write time (measured 2026-07-28). The same arrays zeroed compress
    # to 0.3 %, which is the case `compress` exists for -- a checkpoint
    # dominated by zero-initialised or quantised buffers.
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED

    # The pid is in the temp name so two processes saving to one path cannot
    # delete each other's half-written file. Which one wins the rename is then
    # the only race left, and os.replace makes that outcome well defined.
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        with zipfile.ZipFile(temp_path, "w", compression=mode) as archive:
            archive.writestr(_METADATA_MEMBER, header_bytes)
            for key, array in arrays.items():
                buffer = io.BytesIO()
                _npy.write_array(buffer, array, allow_pickle=False)
                archive.writestr(_TENSOR_PREFIX + key + ".npy", buffer.getvalue())
        os.replace(temp_path, path)
    except BaseException:
        # Includes KeyboardInterrupt, which is how a training run usually dies.
        temp_path.unlink(missing_ok=True)
        raise


def load(path) -> Checkpoint:
    """Read a checkpoint written by `save`.

    Every rejection names what was wrong with the file. A checkpoint that fails
    to load is normally a mismatch between what someone saved and what they
    think they saved, and a message that says which key or which version is what
    turns that into a two-minute problem.
    """
    path = Path(path)

    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a vkml checkpoint (not a zip archive)")

    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())

        if _METADATA_MEMBER not in names:
            raise ValueError(
                f"{path} has no {_METADATA_MEMBER}, so it is not a vkml checkpoint. "
                "A plain .npz of the same arrays is not one either -- it carries no "
                "version and no key list."
            )

        try:
            header = json.loads(archive.read(_METADATA_MEMBER))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: {_METADATA_MEMBER} is not valid JSON: {error}") from error

        if not isinstance(header, dict) or header.get("format") != _MAGIC:
            raise ValueError(f"{path}: not a vkml checkpoint (format field is "
                             f"{header.get('format') if isinstance(header, dict) else header!r})")

        version = header.get("version")
        if not isinstance(version, int):
            raise ValueError(f"{path}: version field is {version!r}, expected an integer")
        if version > FORMAT_VERSION:
            raise ValueError(
                f"{path} was written in checkpoint format v{version}; this build reads up to "
                f"v{FORMAT_VERSION}. Upgrade vkml to read it."
            )

        keys = header.get("keys")
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError(f"{path}: the key list is missing or malformed")

        # The key list is the authority on both contents and order, so a member
        # added to the archive after the fact is never loaded and a missing one
        # is an error rather than a silently short state dict.
        tensors: dict[str, _np.ndarray] = {}
        for key in keys:
            member = _TENSOR_PREFIX + key + ".npy"
            if member not in names:
                raise ValueError(f"{path}: '{key}' is listed in {_METADATA_MEMBER} "
                                 f"but {member} is missing from the archive")
            with archive.open(member) as stream:
                # Read through BytesIO: numpy's reader seeks, and a zip member
                # stream is not reliably seekable.
                data = io.BytesIO(stream.read())
            tensors[key] = _npy.read_array(data, allow_pickle=False)

        metadata = header.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: metadata is {type(metadata).__name__}, expected an object")

    return Checkpoint(tensors=tensors, metadata=metadata, version=version)


def save_module(path, module, metadata: Mapping[str, Any] | None = None,
                compress: bool = False) -> None:
    """Save a module's `state_dict`. The common case, spelled once."""
    save(path, module.state_dict(), metadata=metadata, compress=compress)


def load_module(path, module) -> Checkpoint:
    """Load into a module and return the checkpoint, for its metadata.

    Returns rather than discards because the metadata is usually why the caller
    reached for a checkpoint over a bare state dict -- which epoch this was, what
    it scored.

    THE METADATA ARRIVES AFTER THE STATE DICT IS INSTALLED, so a check written
    against it cannot guard the load. If the metadata decides whether the load
    should happen at all -- does this file even hold the architecture I am
    building? -- use `load`, check, then `load_state_dict` yourself. Written down
    because getting this backwards leaves a guard that looks present and never
    runs.
    """
    checkpoint = load(path)
    module.load_state_dict(checkpoint.tensors)
    return checkpoint
