# Milestone D — Data Pipeline and Serialisation

**Status:** complete. **Date:** 2026-07-28.
**Objective (`PHASE2-MANIFESTO.md`, P1):** data utilities and model save/load.

The two remaining subsystems in P1's list. Both were built *after* MNIST trained
end to end rather than before, which changed what they are.

---

## 1. What shipped

| | |
|---|---|
| `python/vkml/data.py` | `Dataset`, `ArrayDataset`, `DataLoader`, `split` |
| `python/vkml/serialize.py` | `save`, `load`, `save_module`, `load_module`, `Checkpoint` |
| Tests | 1009 Python (was 958), 51 of them new |
| Mutations | 11 new, **11 killed** |

`examples/mnist` was migrated onto both, which is the only thing that proves the
API fits. It deleted its hand-rolled `batches()` generator, its two open-coded
evaluation loops, and its `np.savez` call.

---

## 2. Sequencing was the design method, and it worked

Milestone E (MNIST) ran *before* D on the argument that a data pipeline written
without a caller is a guess. The record of what that bought:

**It specified what to build.** `train.py`'s `batches()` was 4 lines: permute,
walk in strides, drop the tail. `DataLoader` is that, plus the epoch counter the
caller was passing in by hand as `default_rng(seed + epoch)`.

**It specified what NOT to build.** P1 lists "prefetch, transforms" among the
data utilities. Neither has a caller:

| Deferred | Why | Revisit when |
|---|---|---|
| Prefetch / worker processes | The datasets in scope are numpy arrays in RAM. A batch is a slice; there is nothing to overlap. Building it means process management, pipe serialisation and a shutdown path | A dataset does not fit in memory, or a profile shows the training loop waiting on data |
| Transform pipeline | Every caller normalises once, up front, over the whole array — faster than per-sample and simpler to reproduce | Augmentation is wanted; that genuinely must happen per epoch |

Recorded per P7 rather than silently skipped. Both are in `data.py`'s module
docstring, where the next person to want them will look.

**It surfaced a requirement no design document had.** `train.py` feeds *two*
frameworks and they must see the same rows. The obvious loader — one that yields
device tensors — cannot serve that caller at all. So `device` is opt-in, and with
it unset the loader yields numpy, which both frameworks are built from. That is a
stronger guarantee than handing each one matching indices: there is only one set
of bytes, so the two runs cannot diverge on which rows they saw.

---

## 3. The checkpoint format

`np.savez` was honest for an example and is not a format: no version, no key
list, nothing saying what model the arrays belong to.

**The format is a zip containing only data.** Every member is either a `.npy`
array read with `allow_pickle=False`, or one JSON document. This is the
manifesto's security requirement — *never deserialise into code execution* — and
it is a design property rather than a check: neither member type can name a
callable, so a hostile file has nothing to reach for.

Parsing the array bytes is delegated to numpy's reader rather than hand-written.
A hand-rolled binary parser is where a memory-safety bug would go, and numpy's
has far more scrutiny than this file will get.

```
vkml.json          format identifier, version, key list, user metadata
tensors/<key>.npy  one array per state_dict entry
```

Three properties beyond round-tripping:

- **Atomic.** Written to `<name>.<pid>.partial` and renamed. Saving each epoch
  over one path is the normal pattern, and a write that dies partway must not
  take the last good checkpoint with it. The cleanup catches `BaseException`,
  not `Exception`, because `KeyboardInterrupt` is how a training run usually
  dies and is not an `Exception`.
- **Versioned.** A newer checkpoint is refused by name instead of misread.
- **Self-describing.** The metadata is what makes the GUI's architecture check
  possible: `mlp.vkml` loaded as a CNN now says so, rather than emitting a list
  of unexpected keys.

**Stated limit.** A decompression bomb will be attempted. That costs memory, not
control — a denial of service against a process that already chose to open the
file. Not fixed because a size cap needs a threshold and there is no evidence for
what it should be.

**Measured, not assumed:** deflate returns 93.5 % of stored size on the trained
MLP checkpoint for 7× the write time, so `ZIP_STORED` is the default. The same
arrays zeroed compress to 0.3 %, which is what the `compress` flag is for.

---

## 4. Verification

### 4.1 The refactor is behaviour-preserving, and that is checked

The migration could have silently changed the training run. It did not: the new
loader and the deleted `batches()` produce **batch-for-batch identical output**
at the parameters the reported run used (n=60000, batch 64, seed 20260728, 3
epochs) — verified by comparing every batch of every epoch, not by reading the
code. Both derive the order the same way, `default_rng(seed + epoch).permutation(n)`.

Confirmed end to end: a fresh epoch 1 after the refactor reproduces
`train acc 97.24% / test acc 96.12% / test loss 0.1251` exactly.

**The previously reported MNIST numbers therefore still stand.** Full re-run:
**97.25 %** test accuracy vs torch's 97.38 % (0.13 pp), 23.8 s for 3 epochs on
`vulkan:0`.

### 4.2 Mutation campaign

`scripts/mutation_check.py` gained a Python section. There are no numerics here,
which is exactly why it matters: these subsystems fail *silently*. A shuffle that
unpairs inputs from labels trains a model that looks healthy and is not.

**11 of 11 killed.**

| Mutation | Result |
|---|---|
| Sample with replacement instead of permuting | KILLED |
| Permute each array independently | KILLED |
| Reseed identically every epoch | KILLED |
| `drop_last` off-by-one | KILLED |
| Split without shuffling first | KILLED |
| Allow pickle on load | KILLED |
| Write straight to the destination | KILLED |
| Catch `Exception` rather than `BaseException` | KILLED |
| Leave the partial file behind | KILLED |
| Accept any format version | KILLED |
| Skip the missing-member check | KILLED |

Two survived the first pass. Both were **defects in the tests**, and neither
would have been found by reading them:

- *`drop_last` off-by-one.* The stop bound only admits a short batch when
  n ≡ −1 (mod batch_size). The test used n=10, batch 4 — a length where the
  correct and mutated bounds agree. Fixed by parametrising over the boundary.
- *Atomic write.* The test triggered failure with unserialisable metadata, which
  is rejected *before* the file is opened, so it never exercised the rename at
  all. Fixed by injecting the failure into the array writer, mid-archive, and
  parametrising over `KeyboardInterrupt`.

The pickle test builds the attack rather than trusting the flag: an array whose
elements unpickle by *calling* something, written with `allow_pickle=True` and
listed as a normal tensor. Load must refuse it **and** the call must not have
happened. Both are asserted.

### 4.3 Gates

All eight green: layering (54 files) · clang-format · debug `-Werror` · release ·
ASan build and suite · ctest · **1009 Python tests, 3 skipped** · validation
layers clean on the refactored training path.

---

## 5. Findings

**A convenience wrapper can make a guard unreachable.** `load_module` loads and
installs in one call, so the state dict is applied *before* the metadata is
returned. The GUI's architecture check was written against that metadata and
therefore never ran — the key-set mismatch fired first, which is precisely the
confusing error the check existed to replace. Found by running the failure path,
not by review. The fix is ordering at the call site (`load` → check → install);
`load_module`'s docstring now says so, because a guard that looks present and
never runs is worse than no guard.

**`__iter__` as a generator function defers more than it looks.** The order draw
and the epoch increment would not happen until the first `next()`, so two
iterators taken before either was read would see the same epoch. Split into a
plain `__iter__` that draws and advances, returning a generator over a fixed
order. Has its own test.

**Padding the temp filename with the pid costs nothing and removes a race.** Two
processes saving to one path would otherwise unlink each other's half-written
file. Which one wins the rename is then the only race left, and `os.replace`
makes that outcome well defined.

---

## 6. Carried forward

| # | Item | Trigger |
|---|---|---|
| 15 | Run the Python suite under a sanitizer in CI | Unchanged from Milestone B |
| 16 | CPU fallback via graph splitting, or state that Vulkan is all-or-nothing | Unchanged |
| 17–19 | `layer_norm` fusion · `scatter_add` scan · batched optimiser submission | Unchanged; all await a profile |
| 20 | Bind the scalar overloads of `add`/`sub`/`mul`/`div`/`pow` | Unchanged |
| 21 | Prefetch / worker processes in `DataLoader` | §2 |
| 22 | Transform pipeline in `DataLoader` | §2 |
| 23 | A decompression-size cap on `load` | Evidence for what the threshold should be |
| 24 | No `LICENSE` file exists | Blocks outside contribution |

---

## 7. Gate

**Gate for Milestone D specifically — a data pipeline and a checkpoint format,
both proven against a real caller — is met.**

### Correction: P1 is not closed

An earlier draft of this section claimed that with D complete "the list is
closed." That was wrong, and it is corrected here rather than quietly dropped,
because a phase marked done is a phase nobody re-reads. Checking P1's list item
by item against the code:

| P1 item | State |
|---|---|
| Tensor system, runtime, all core operators | done (A, B) |
| Optimisers — SGD, Momentum, Adam, AdamW, RMSProp | done (C) |
| Data utilities, model save/load/checkpoint | done (D), less prefetch and transforms |
| `nn` — Linear, Embedding, Attention, MHA, TransformerBlock (encoder), residuals | done (C) |
| **`nn` — Conv1d, Conv3d** | **not built.** Only `Conv2d` exists |
| **`nn` — FeedForward, PositionalEncoding, decoder block** | **not built** |
| **Losses — BCE, KL, Huber** | **not built.** Only `mse_loss`, `cross_entropy` |
| **Autograd — checkpointing, custom gradients** | **not built.** `autograd.h` describes checkpointing as an affordance of the graph design; no implementation and no user-facing API |

Six subsystems from P1's own sentence, not two. The error came from reasoning
about milestones B–E — which did close what *they* set out to do — instead of
re-reading the phase definition. It is the same failure the Milestone B ledger
warned about: a list written in advance is a prediction, and checking work
against the milestone rather than against the specification lets the gap survive.

`Conv3d` additionally needs rank 5, which `kMaxDims = 4` forbids. That is a
push-constant budget consequence (`ARCHITECTURE.md`), so it is a design change,
not a missing layer.
