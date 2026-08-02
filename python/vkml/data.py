"""Datasets and batching.

Designed against a real caller rather than ahead of one: examples/mnist trained
end to end with a hand-written batching loop first, and this is that loop's
requirements made reusable. What it needed was shuffled fixed-size batches with
a reproducible order, and that is what is here.

DELIBERATELY ABSENT, recorded so the omissions are decisions rather than gaps:

  Prefetching and worker processes. The datasets in scope fit in RAM as numpy
  arrays, so an unaugmented batch is a slice and there is nothing to overlap.
  Adding workers would mean process management, serialisation across a pipe and
  a shutdown path. Revisit when a dataset does not fit in memory or when a
  profile shows the training loop waiting on data.

  MEASURED, and the number MOVED when transforms arrived -- which is the
  honest reason to keep re-reading a deferral rather than citing it. On
  examples/cifar100, which times the loader, the upload and the compute
  separately:

      no augmentation      batch  1.7%   transfer 5.8%   compute 92.6%
      --augment            batch 21.2%   transfer 4.9%   compute 73.9%

  Prefetching hides the first column only. Its ceiling was two tenths of one
  percent when this paragraph was written and is TWENTY-ONE PERCENT for an
  augmented run, because the crop and the flip are real work on the host.

  STILL DEFERRED, and now for a different reason than "there is nothing to
  hide". Overlapping 21% needs a thread the GIL does not serialise -- numpy
  releases it inside a vectorised call, so a thread rather than a process may
  well be enough here, which is a much smaller change than the worker pool this
  paragraph assumed. That is worth measuring before designing, and neither
  example trains long enough for 21% of a 4.7 s epoch to be the thing standing
  between vkML and anything. The trigger is met; the work is queued rather than
  refused, which is a different state and is recorded as one.

PRESENT, and here for a reason the docstring above once used to defer it:

  A transform hook. This said "revisit when augmentation is wanted, since that
  genuinely has to happen per epoch", and augmentation is now wanted --
  examples/cifar100 states its absence as a deliberate omission that costs
  accuracy, and PHASE2-MANIFESTO names transforms as P1 completeness.

  BATCHED, NOT PER-SAMPLE, because Dataset.__getitem__ already is. A transform
  receives the whole batch and returns it, so a flip is one vectorised numpy
  operation over 64 images rather than 64 Python calls. That is the same
  argument the Dataset docstring makes for its own batched indexing.

  RNG PASSED IN, NEVER REACHED FOR. A transform's signature is
  `f(rng, arrays) -> arrays`, and the loader owns the generator. Augmentation
  that draws from numpy's global state is irreproducible, and this project has
  already paid for exactly that once -- `nn.manual_seed` exists because layers
  called an unseeded `default_rng()` and a divergence could not be
  re-observed. Making the generator an argument means a transform cannot
  quietly become non-deterministic; it would have to import numpy and ignore
  what it was handed.
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
                 drop_last: bool = False, seed: int = 0, device=None, transform=None):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if transform is not None and not callable(transform):
            raise TypeError(
                f"transform must be callable as f(rng, arrays) -> arrays, got "
                f"{type(transform).__name__}"
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.device = device
        self.transform = transform
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
        # A SEPARATE STREAM from the shuffle, drawn from the same seed and
        # epoch. Sharing one generator would make the augmentation depend on
        # whether shuffling was on, so the same seed would give different
        # augmentations for `shuffle=True` and `shuffle=False` -- which is a
        # difference nobody asked for and nobody would look for.
        rng = _np.random.default_rng([self.seed, self._epoch, 0x7A11])
        self._epoch += 1
        return self._batches(order, n, rng)

    def _batches(self, order: _np.ndarray, n: int, rng) -> Iterator[tuple]:
        # drop_last keeps every batch the same shape. That matters more than the
        # handful of samples it discards: a trailing short batch changes the
        # graph shape, which costs a re-plan, and it silently reweights the last
        # step of each epoch because the loss is a mean over the batch.
        limit = n - self.batch_size + 1 if self.drop_last else n

        for start in range(0, limit, self.batch_size):
            indices = order[start:start + self.batch_size]
            batch = self.dataset[indices]
            if self.transform is not None:
                batch = self.transform(rng, batch)
                if not isinstance(batch, tuple):
                    raise TypeError(
                        f"transform must return a tuple of arrays, got "
                        f"{type(batch).__name__}. A transform that returns one array "
                        f"silently changes the batch's arity, which shows up as a "
                        f"confusing unpack error at the call site rather than here"
                    )
            # AFTER the transform, deliberately. Augmenting on the device would
            # need every transform written twice, and the arrays are on the host
            # at this point anyway -- moving them first and back would cost two
            # transfers to save none.
            if self.device is None:
                yield batch
            else:
                yield tuple(V.tensor(part, device=self.device) for part in batch)

    def __repr__(self) -> str:
        transform = "" if self.transform is None else ", transform=on"
        return (f"DataLoader(n={len(self.dataset)}, batch_size={self.batch_size}, "
                f"shuffle={self.shuffle}, drop_last={self.drop_last}{transform})")


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


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
#
# THREE, NOT A LIBRARY. `Compose`, `RandomHorizontalFlip` and `RandomCrop` are
# the standard CIFAR augmentation and between them they exercise every part of
# the contract: composition, a per-sample random decision, and a shape-changing
# spatial operation. Anything else a caller wants is a five-line function with
# the same signature, and a transform zoo would be a maintenance surface built
# ahead of a user (P6, and the handbook's YAGNI).
#
# EVERY ONE IS BATCHED AND PER-SAMPLE AT THE SAME TIME, which is the part worth
# reading carefully. A flip that draws ONE coin for the whole batch is not
# augmentation -- it doubles the dataset instead of multiplying it, and it
# correlates every sample in a step. So the decision is drawn per sample and
# applied with a vectorised select, never with a Python loop.
#
# THEY TRANSFORM THE FIRST ARRAY ONLY. A batch is `(inputs, labels)`, and a
# spatial augmentation applies to the images and not to the integers beside
# them. Passing the rest through unchanged is what makes them composable with
# `ArrayDataset(x, y)` without every caller writing an adapter.


class Compose:
    """Run several transforms in order, threading the same generator through.

    One generator, not one each: two transforms seeded independently would draw
    the same numbers whenever their call counts happened to line up, which is
    the classic way correlated "randomness" gets into an augmentation pipeline.
    """

    def __init__(self, *transforms):
        for t in transforms:
            if not callable(t):
                raise TypeError(f"Compose takes callables, got {type(t).__name__}")
        self.transforms = transforms

    def __call__(self, rng, arrays: tuple) -> tuple:
        for t in self.transforms:
            arrays = t(rng, arrays)
        return arrays

    def __repr__(self) -> str:
        return "Compose(" + ", ".join(repr(t) for t in self.transforms) + ")"


class RandomHorizontalFlip:
    """Mirror each image left-to-right with probability `p`, independently.

    Expects the first array as (N, C, H, W) -- vkML's layout everywhere -- and
    flips the last axis.
    """

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p

    def __call__(self, rng, arrays: tuple) -> tuple:
        images = arrays[0]
        if images.ndim != 4:
            raise ValueError(
                f"RandomHorizontalFlip expects (N, C, H, W), got shape {images.shape}"
            )
        # One coin per SAMPLE. `np.where` over the whole batch rather than a
        # loop: the flipped copy costs one vectorised reverse, and selecting
        # between two arrays is a single pass.
        flip = rng.random(len(images)) < self.p
        flipped = images[:, :, :, ::-1]
        out = _np.where(flip[:, None, None, None], flipped, images)
        return (_np.ascontiguousarray(out),) + tuple(arrays[1:])

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomCrop:
    """Pad by `padding` on every side, then take a random `size` window per sample.

    The standard CIFAR augmentation: pad 4, crop back to 32. Translating the
    image is what stops a convolution memorising absolute position.
    """

    def __init__(self, size, padding: int = 0, fill: float = 0.0):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2 or min(self.size) < 1:
            raise ValueError(f"size must be one or two positive ints, got {size}")
        if padding < 0:
            raise ValueError(f"padding must not be negative, got {padding}")
        self.padding = padding
        self.fill = fill

    def __call__(self, rng, arrays: tuple) -> tuple:
        images = arrays[0]
        if images.ndim != 4:
            raise ValueError(f"RandomCrop expects (N, C, H, W), got shape {images.shape}")

        if self.padding:
            pad = ((0, 0), (0, 0), (self.padding,) * 2, (self.padding,) * 2)
            images = _np.pad(images, pad, mode="constant", constant_values=self.fill)

        n, _, height, width = images.shape
        out_h, out_w = self.size
        if out_h > height or out_w > width:
            raise ValueError(
                f"crop {self.size} does not fit a padded image of {height}x{width}"
            )

        # One offset per sample, then gathered with fancy indexing. Building the
        # index arrays costs O(N x H x W) integers, which for a CIFAR batch is
        # 64 x 32 x 32 -- cheaper than 64 Python-level slices and, unlike them,
        # one pass over the data.
        top = rng.integers(0, height - out_h + 1, size=n)
        left = rng.integers(0, width - out_w + 1, size=n)
        rows = top[:, None] + _np.arange(out_h)[None, :]
        cols = left[:, None] + _np.arange(out_w)[None, :]
        sample = _np.arange(n)[:, None, None]
        out = images[sample, :, rows[:, :, None], cols[:, None, :]]
        # The gather above puts the two indexed spatial axes first, so the
        # result is (N, H, W, C) and has to come back to vkML's (N, C, H, W).
        out = _np.ascontiguousarray(out.transpose(0, 3, 1, 2))
        return (out,) + tuple(arrays[1:])

    def __repr__(self) -> str:
        return f"RandomCrop(size={self.size}, padding={self.padding})"
