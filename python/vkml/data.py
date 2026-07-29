"""Datasets and batching.

Designed against a real caller rather than ahead of one: examples/mnist trained
end to end with a hand-written batching loop first, and this is that loop's
requirements made reusable. What it needed was shuffled fixed-size batches with
a reproducible order, and that is what is here.

DELIBERATELY ABSENT, recorded so the omissions are decisions rather than gaps:

  Prefetching and worker processes. The datasets in scope fit in RAM as numpy
  arrays, so a batch is a slice and there is nothing to overlap. Adding workers
  would mean process management, serialisation across a pipe and a shutdown
  path, for a problem nobody has yet. Revisit when a dataset does not fit in
  memory or when a profile shows the training loop waiting on data.

  MEASURED, so this is no longer only an argument. A CIFAR-100 CNN over 2343
  steps on the GPU (examples/cifar100/train.py, which times the loader, the
  host-to-device upload and the compute separately) spends 0.2% of each step
  producing batches, 0.8% uploading them and 99.0% computing. Prefetching hides
  only the first of those, so its entire ceiling here is two tenths of one
  percent. The trigger above stands unchanged and remains unmet.

  A transform pipeline. Every caller so far normalises once, up front, over the
  whole array -- which is faster than per-sample transforms and simpler to
  reproduce. Revisit when augmentation is wanted, since that genuinely has to
  happen per epoch.
"""

from __future__ import annotations

from typing import Iterator

import numpy as _np

import vkml as V


class Dataset:
    """Indexable collection of samples.

    Subclasses implement `__len__` and `__getitem__(indices)`, where indices is
    an array -- batched rather than per-sample, because the arrays in scope
    slice far faster in one call than in a Python loop.
    """

    def __len__(self) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not implement __len__")

    def __getitem__(self, indices: _np.ndarray) -> tuple[_np.ndarray, ...]:
        raise NotImplementedError(f"{type(self).__name__} does not implement __getitem__")


class ArrayDataset(Dataset):
    """Several arrays sharing a first axis -- typically inputs and labels.

    Holds references, not copies: MNIST is ~180 MB as float32 and duplicating it
    to wrap it would be a poor trade. The caller therefore must not mutate the
    arrays afterwards, which is the usual bargain for a view.
    """

    def __init__(self, *arrays: _np.ndarray):
        if not arrays:
            raise ValueError("ArrayDataset needs at least one array")
        lengths = {len(a) for a in arrays}
        if len(lengths) != 1:
            raise ValueError(
                f"arrays disagree on their first axis: {[len(a) for a in arrays]}"
            )
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, indices: _np.ndarray) -> tuple[_np.ndarray, ...]:
        return tuple(array[indices] for array in self.arrays)


class DataLoader:
    """Iterates a dataset in batches.

    REPRODUCIBLE BY CONSTRUCTION. The shuffle is driven by `seed` plus an epoch
    counter that advances each time iteration starts, so successive epochs see
    different orders while the whole run replays from the seed alone. A loader
    seeded from the clock would make a training result impossible to reproduce,
    which is the same defect that made an earlier divergence impossible to
    investigate.

    `device` is opt-in. Without it batches come back as numpy arrays and the
    caller places them; with it they arrive as tensors already resident. Either
    way the transfer is visible at the call site rather than hidden in an
    iterator.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                 drop_last: bool = False, seed: int = 0, device=None):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.device = device
        self._epoch = 0

    def __len__(self) -> int:
        """Number of batches one pass yields."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Pin the shuffle to a specific epoch.

        Iteration advances this on its own; setting it explicitly is what lets a
        run resume mid-training and see the order it would have seen.
        """
        self._epoch = epoch

    def __iter__(self) -> Iterator[tuple]:
        # Not itself a generator: the order is drawn and the epoch advances when
        # iter() is called, not when the first batch is pulled. A generator
        # function would defer both, so a loader that was iterated but never
        # consumed would silently repeat its previous order.
        n = len(self.dataset)
        if self.shuffle:
            order = _np.random.default_rng(self.seed + self._epoch).permutation(n)
        else:
            order = _np.arange(n)
        self._epoch += 1
        return self._batches(order, n)

    def _batches(self, order: _np.ndarray, n: int) -> Iterator[tuple]:
        # drop_last keeps every batch the same shape. That matters more than the
        # handful of samples it discards: a trailing short batch changes the
        # graph shape, which costs a re-plan, and it silently reweights the last
        # step of each epoch because the loss is a mean over the batch.
        limit = n - self.batch_size + 1 if self.drop_last else n

        for start in range(0, limit, self.batch_size):
            indices = order[start:start + self.batch_size]
            batch = self.dataset[indices]
            if self.device is None:
                yield batch
            else:
                yield tuple(V.tensor(part, device=self.device) for part in batch)

    def __repr__(self) -> str:
        return (f"DataLoader(n={len(self.dataset)}, batch_size={self.batch_size}, "
                f"shuffle={self.shuffle}, drop_last={self.drop_last})")


def split(dataset: ArrayDataset, fraction: float, seed: int = 0
          ) -> tuple[ArrayDataset, ArrayDataset]:
    """Partition into two disjoint datasets -- a validation split.

    Shuffles before cutting, because a dataset ordered by class would otherwise
    put whole classes on one side. Deterministic given the seed.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")

    n = len(dataset)
    order = _np.random.default_rng(seed).permutation(n)
    cut = int(round(n * fraction))
    if cut == 0 or cut == n:
        raise ValueError(f"fraction {fraction} splits {n} samples into an empty half")

    left = tuple(array[order[:cut]] for array in dataset.arrays)
    right = tuple(array[order[cut:]] for array in dataset.arrays)
    return ArrayDataset(*left), ArrayDataset(*right)
