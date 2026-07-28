<div align="center">

<img src="assets/vkml_logo.png" alt="vkML — Vulkan based machine learning library" width="440">

**A deep learning framework built on Vulkan compute.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://en.cppreference.com/w/cpp/20)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.3-A41E22.svg)](https://www.vulkan.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)

</div>

---

## What vkML is

vkML is a tensor library with reverse-mode autograd, neural network layers,
optimisers and a data pipeline — the shape of API you would expect from PyTorch
— running on the GPU through **Vulkan compute shaders** instead of CUDA or ROCm.

It is written in C++20 (~13k lines of core and shader code) with a Python API
through [nanobind](https://github.com/wjakob/nanobind). Tensors are lazy: an
operation appends a node to a graph, and work happens when a result is actually
needed, so a whole expression reaches the GPU as one submission.

### Why Vulkan

CUDA is the reason most GPU deep learning runs on NVIDIA hardware. ROCm covers
some AMD cards and not others — this project began because the author's GPU was
one of the ones it does not cover.

Vulkan is a different bet: it is a **vendor-neutral compute API** with drivers
from AMD, NVIDIA, Intel, Qualcomm, ARM and Apple (through MoltenVK). A compute
shader compiled to SPIR-V runs on all of them. That trades away CUDA's mature
libraries — no cuBLAS, no cuDNN, every kernel is written here — for the
possibility of a single build that runs anywhere.

Concretely, the design leans on Vulkan features that make that trade cheaper:

| Feature | What it buys |
|---|---|
| `bufferDeviceAddress` | Buffers are 64-bit pointers in push constants — no descriptor sets, pools, or per-dispatch updates |
| Specialisation constants | One SPIR-V module per operator family, specialised per op, dtype and layout at pipeline creation |
| `scalar_block_layout` | Push-constant structs lay out identically in GLSL and C++, so shape metadata is mirrored, not translated |
| Timeline semaphores | Submission and readback synchronise without fences per dispatch |

---

## Quick start

```python
import numpy as np
import vkml as V

V.init_vulkan(0)
device = V.device("vulkan:0") if V.vulkan_available() else V.cpu

model = V.nn.Sequential(
    V.nn.Flatten(),
    V.nn.Linear(784, 128), V.nn.ReLU(),
    V.nn.Linear(128, 10),
).to(device)

optimiser = V.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for images, labels in loader:                     # vkml.data.DataLoader
    x = V.tensor(images, device=device)
    y = V.tensor(labels, device=device)

    optimiser.zero_grad()
    loss = V.nn.cross_entropy(model(x), y)
    loss.backward()
    optimiser.step()
```

A complete, runnable version — which also checks itself against PyTorch step by
step — is in [`examples/mnist`](examples/mnist):

```sh
python examples/mnist/train.py          # trains on the GPU, compares to PyTorch
python examples/mnist/gui.py            # draw a digit and watch it predict
```

---

## Installation

```sh
pip install .
```

Requires a C++20 compiler, CMake ≥ 3.25 and the Vulkan SDK. The SPIR-V for every
shader is compiled into the extension, so an installed vkML has no data files to
find at run time.

Without the Vulkan SDK the build **fails deliberately** rather than quietly
producing a CPU-only library. If that is genuinely what you want:

```sh
pip install . -C cmake.define.VKML_VULKAN=OFF
```

---

## Components

| Component | What it provides |
|---|---|
| `vkml` | Tensors, 65 operators, dtypes, devices, lazy evaluation |
| `vkml.nn` | `Module`, Linear, Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d, LayerNorm, Dropout, Embedding, MultiheadAttention, TransformerEncoderLayer, activations, losses |
| `vkml.optim` | SGD (with momentum), Adam, AdamW, RMSProp |
| `vkml.data` | `Dataset`, `ArrayDataset`, `DataLoader`, reproducible shuffling, `split` |
| `vkml.serialize` | Checkpoints as a zip of `.npy` arrays plus JSON metadata |

Autograd is reverse-mode with 47 backward rules, each written in terms of
*forward* operations — so a gradient reuses the same kernels as the forward
pass, and a bug fixed in `mul` is fixed in the gradient of `mul`.

Both **f32 and f16** compute on either backend. f16 is storage, never an
accumulator: values widen to float at the memory boundary and narrow once on the
store, so a reduction of any length keeps fp32 accumulation.

### Checkpoints do not execute code

A vkML checkpoint is a zip containing only data — `.npy` members read with
`allow_pickle=False`, plus one JSON document. Neither can name a callable, so
loading a model file cannot run a program. The well-known failure in this field
is a format built on `pickle`, which reconstructs objects by *calling* what the
stream names.

---

## How correctness is established

Every operator is checked against PyTorch, in a chain:

```
vkml-cpu  vs  PyTorch     catches wrong formulas, axes and broadcast rules
vkml-vulkan  vs  vkml-cpu catches kernel bugs, against an oracle with identical semantics
```

Tolerances are derived from the reduction length **in advance** and never
widened after a failure — a mismatch is a bug until an error analysis says
otherwise.

A green test suite only proves the tests ran, so three further gates check that
the suite *could* fail:

| Gate | Question it answers |
|---|---|
| `scripts/mutation_check.py` | Break each kernel deliberately — does a test notice? |
| `scripts/coverage_matrix.py` | Which operator × dtype × layout combinations are never exercised? |
| `scripts/asan_python.py` | The Python suite against an AddressSanitizer build |

```sh
python -m pytest tests/python -q        # the validation suite
python scripts/mutation_check.py        # can the suite fail?
python scripts/asan_python.py           # the suite, instrumented
```

Results are reproducible by construction: no global atomics, fixed reduction
trees, and a counter-based RNG that is a pure function of seed and index.

---

## Project status

**Alpha.** It trains real models and is tested hard, but it has been run on one
machine and does not cover everything a mature framework does.

**What works** — the MNIST MLP and CNN train end to end on the GPU and agree
with PyTorch to well inside tolerance; 65 operators across both backends;
1,176 tests.

**What does not, stated plainly:**

- **Tested on one GPU.** An AMD RX 5600M (RDNA1) with RADV on Linux. Nothing yet
  proves it runs on NVIDIA, Intel, Windows or macOS.
- **Vulkan is all-or-nothing.** An operator the GPU cannot run raises rather than
  falling back to the CPU. Two cases exist: `prod`, and `max_pool2d` given a
  non-contiguous input.
- **Missing layers:** Conv1d/Conv3d, BCE/KL/Huber losses, gradient checkpointing.
- **Rank ≤ 4.** A push-constant budget decision, not an oversight.
- **f16 matmul is correct but slower than f32**, because the vectorised tile load
  is f32-only.
- No distributed training, no quantisation, no ONNX.

---

## Building from source

```sh
cmake --preset release -DVKML_VULKAN=ON
cmake --build build/release -j$(nproc)
python -m pytest tests/python -q
```

Presets: `release`, `debug` (warnings as errors), `asan`, `relwithdebinfo`.

---

## Documentation

Design decisions live with their reasoning, including the alternatives that were
rejected and why:

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layering, the graph IR, backend interface, the forks taken and their trade-offs |
| [`docs/THEORY.md`](docs/THEORY.md) | The numerics: error bounds, pairwise summation, stability |
| [`docs/adr/`](docs/adr) | Decisions expensive to reverse — graph ownership, allocator, dtype promotion |
| [`docs/MEASUREMENT-AUDIT.md`](docs/MEASUREMENT-AUDIT.md) | How performance is measured, and which instruments lie |
| [`docs/PERFORMANCE-MODEL.md`](docs/PERFORMANCE-MODEL.md) | What the hardware can do, and what the kernels achieve |

---

## License

vkML is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
