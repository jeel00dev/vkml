# vkML

A deep learning framework built on Vulkan compute, in C++20 with a Python API.

Tensors, autograd, `nn` layers, optimisers and a data pipeline, running on the
GPU through Vulkan rather than CUDA or ROCm — so the same build works on any
device with a Vulkan driver, and on the CPU when there is none.

```python
import numpy as np, vkml as V

gpu = V.device("vulkan:0") if V.vulkan_available() else V.cpu
V.init_vulkan(0)

model = V.nn.Sequential(
    V.nn.Flatten(),
    V.nn.Linear(784, 128), V.nn.ReLU(),
    V.nn.Linear(128, 10),
).to(gpu)

optimiser = V.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

x = V.tensor(np.random.randn(64, 1, 28, 28).astype(np.float32), device=gpu)
y = V.tensor(np.random.randint(0, 10, 64), device=gpu)

loss = V.nn.cross_entropy(model(x), y)
loss.backward()
optimiser.step()
```

## Install

```sh
pip install .
```

Needs a C++20 compiler, CMake ≥ 3.25 and the Vulkan SDK. The SPIR-V is compiled
into the extension, so there is nothing to locate at run time.

Without the Vulkan SDK the build fails deliberately rather than quietly
producing a CPU-only library. If that is what you want, ask for it:

```sh
pip install . -C cmake.define.VKML_VULKAN=OFF
```

## What works

A worked example lives in `examples/mnist` — it trains, and checks itself
against PyTorch step by step:

```sh
python examples/mnist/train.py            # GPU if one is present
python examples/mnist/gui.py              # draw a digit, watch it predict
```

- **Operators**: elementwise, comparisons, reductions, softmax, matmul,
  im2col/col2im, pooling, indexing and scatter — on both backends.
- **Autograd**: reverse mode, with backward rules written in terms of forward
  operations, so gradients reuse the same kernels.
- **`nn`**: Linear, Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d, LayerNorm,
  Dropout, Embedding, MultiheadAttention, TransformerEncoderLayer, activations.
- **Optimisers**: SGD (with momentum), Adam, AdamW, RMSProp.
- **dtypes**: f32 and f16 compute; i32/i64 are index and storage types.
- **Checkpoints**: a zip of `.npy` arrays plus JSON metadata, read with
  `allow_pickle=False` — a model file cannot execute code on load.

## What does not

Stated plainly, because finding out later is worse:

- **Tested on one GPU.** An RX 5600M (RDNA1, RADV) on Linux. Nothing proves it
  runs on NVIDIA, Intel, Windows or macOS yet.
- **Vulkan is all-or-nothing.** An operator the GPU cannot run raises rather
  than falling back to the CPU. Two cases exist: `prod`, and `max_pool2d` given
  a non-contiguous input.
- **No Conv1d/Conv3d**, no BCE/KL/Huber losses, no gradient checkpointing.
- **Rank ≤ 4.** A push-constant budget decision, not an oversight.
- f16 matmul is correct but currently *slower* than f32 on the GPU, because the
  vectorised tile load is f32-only.

## How it is checked

PyTorch is the reference. The chain is CPU-against-PyTorch for semantics, then
Vulkan-against-CPU for kernel bugs — an oracle with identical semantics, so a
mismatch is unambiguously a kernel bug.

Tolerances are derived from the reduction length in advance, never widened after
a failure. Beyond the suite passing, three gates check that the suite could
fail at all: a mutation campaign that breaks each kernel and confirms a test
notices, a coverage recording that reports which operator × dtype × layout
combinations are never exercised, and the Python suite run under
AddressSanitizer.

```sh
python -m pytest tests/python -q          # the validation suite
python scripts/mutation_check.py          # can the suite fail?
python scripts/asan_python.py             # the suite, instrumented
```

Design decisions and their rejected alternatives are in `docs/` —
`ARCHITECTURE.md` for the structure, `adr/` for the choices that were expensive
to reverse, `THEORY.md` and `MEASUREMENT-AUDIT.md` for the numerics and how
things are measured.

## License

Apache-2.0. See [LICENSE](LICENSE).
