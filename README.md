<div align="center">

<img src="https://raw.githubusercontent.com/jeel00dev/vkml/main/assets/vkml_logo.png" alt="vkML — Vulkan based machine learning library" width="440">

**A deep learning framework that runs on Vulkan compute.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://en.cppreference.com/w/cpp/20)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.3-A41E22.svg)](https://www.vulkan.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)

</div>

---

## About vkML

vkML is a deep learning library. It gives you tensors, automatic differentiation,
neural network layers, optimisers and a data loader — roughly the same API you
already know from PyTorch — and it runs them on your GPU using **Vulkan compute
shaders** instead of CUDA or ROCm.

The core is written in C++20 (about 16k lines, including the shaders), and the
Python API is built with [nanobind](https://github.com/wjakob/nanobind).

Tensors are lazy. When you write `a + b`, nothing runs yet — vkML just records
the operation in a graph. The work is sent to the GPU when you actually need a
result, so a whole expression can go over in a single submission.

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

### Why Vulkan?

Most GPU deep learning runs on NVIDIA cards because it runs on CUDA. AMD has
ROCm, but it only supports some cards — and mine is not one of them. That is the
problem that started this project.

Vulkan is a different approach. It is a **vendor-neutral** API, and drivers exist
for AMD, NVIDIA, Intel, Qualcomm, ARM and Apple (through MoltenVK). A compute
shader compiled to SPIR-V can run on all of them.

The trade-off is real, and worth knowing before you start: there is no cuBLAS and
no cuDNN here, so every kernel in vkML is one that had to be written by hand. In
return, one build can potentially run anywhere.

A few Vulkan features make that trade cheaper than it sounds:

| Feature | What it gives us |
|---|---|
| `bufferDeviceAddress` | Buffers become 64-bit pointers passed in push constants, so there are no descriptor sets or pools to manage per dispatch |
| Specialisation constants | One SPIR-V module per operator family, specialised for the op, dtype and layout when the pipeline is created |
| `scalarBlockLayout` | Push-constant structs have the same layout in GLSL and C++, so shape metadata is mirrored instead of translated |
| Timeline semaphores | Submission and readback stay in sync without a fence for every dispatch |

---

## Installation

### Step 1 — install the dependencies

vkML is compiled when you install it, so you need a few tools on your machine
first. Pick your platform below.

<details open>
<summary><b>Linux (Debian / Ubuntu)</b></summary>

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build \
                        libvulkan-dev glslang-tools
```

That gives you a C++ compiler, CMake, the Vulkan development files and
`glslangValidator` to compile the shaders.

If you do not have a GPU (or your driver is not working), you can install a
software Vulkan driver and still use the library:

```sh
sudo apt-get install -y mesa-vulkan-drivers vulkan-tools
```

</details>

<details>
<summary><b>Linux (Fedora / RHEL)</b></summary>

```sh
sudo dnf install -y gcc-c++ cmake ninja-build \
                    vulkan-headers vulkan-loader-devel vulkan-loader
```

You also need a GLSL compiler, which provides either `glslc` or
`glslangValidator`. We have not verified which package supplies it on Fedora, so
we would rather not send you after the wrong name — if the build stops with "no
GLSL compiler found", search your package manager for `glslang` or `shaderc`.
Please open an issue with what worked and we will put it here.

</details>

<details>
<summary><b>Linux (Arch)</b></summary>

```sh
sudo pacman -S base-devel cmake ninja vulkan-headers vulkan-icd-loader glslang
```

`glslang` provides `glslangValidator`. If you would rather have `glslc`, install
`shaderc` instead. Either one works.

</details>

<details>
<summary><b>macOS</b></summary>

Vulkan is not native on macOS, so you also need MoltenVK, which translates
Vulkan to Metal. Homebrew has everything:

```sh
brew install cmake ninja molten-vk vulkan-loader vulkan-headers vulkan-tools glslang
```

The Vulkan loader has to be told where MoltenVK is, because Homebrew does not
install it where the loader looks by default. Rather than guess the path, ask
Homebrew for it:

```sh
export VK_ICD_FILENAMES="$(brew list molten-vk | grep -m1 'MoltenVK_icd.json')"
```

Add that line to your `~/.zshrc` so it survives a new terminal.

Do not be tempted to write the path out by hand. Our own CI got it wrong three
times: the file lives under `etc/` on some versions and `share/` on others, and
`brew --prefix` does not point where you would expect. Asking `brew list` costs
nothing and is always right.

Check it worked — this should list a device:

```sh
vulkaninfo --summary
```

</details>

<details>
<summary><b>Windows</b></summary>

You need three things:

1. **Visual Studio 2022** with the "Desktop development with C++" workload
   ([download](https://visualstudio.microsoft.com/downloads/)). The free
   Community edition is fine.
2. **CMake 3.25 or newer** ([download](https://cmake.org/download/)). Tick
   "Add CMake to the system PATH" during setup.
3. **The Vulkan SDK** from LunarG ([download](https://vulkan.lunarg.com/sdk/home)).
   This includes the headers, the loader and `glslc`, so it covers everything
   else vkML needs.

The Vulkan SDK installer sets the `VULKAN_SDK` environment variable, which is how
CMake finds it. **Open a new terminal after installing**, or the variable will
not be set in your current one.

Then run the install from a *Developer Command Prompt for VS 2022* — that is the
shell that has the MSVC compiler on its PATH.

> **Worth knowing:** Windows is built and tested on every commit, but only as a
> CPU-only build, because the CI runners have no GPU. That means the Windows
> build *with Vulkan enabled* has never run anywhere we can see. It should work,
> and we would very much like to hear whether it does — see
> [If you have hardware we do not](#if-you-have-hardware-we-do-not).

</details>

### Step 2 — install vkML

You need Python 3.10 or newer. Clone the repository and install it:

```sh
git clone https://github.com/jeel00dev/vkml.git
cd vkml
pip install .
```

That one command builds the C++ core, compiles the shaders into the extension
and installs the Python package. It takes a few minutes the first time.

vkML is not on PyPI yet, so installing from source is the only way for now.

### Step 3 — check that it worked

```sh
python -c "import vkml; print(vkml.__version__); print(vkml.vulkan_device_names())"
```

You should see a version number and a list of your GPUs. If the list is empty,
vkML installed correctly but cannot see a Vulkan device — see
[Troubleshooting](#troubleshooting) below.

### Installing without Vulkan

vkML also has a CPU backend, mostly used to check the GPU results against. If
you want to build only that — for example on a machine with no Vulkan SDK — ask
for it explicitly:

```sh
pip install . -C cmake.define.VKML_VULKAN=OFF
```

By default the build **fails** when the Vulkan SDK is missing rather than quietly
giving you a CPU-only library, because that would not be the thing you asked to
install.

### One runtime dependency pip cannot install for you

vkML links against the **Vulkan loader** — `libvulkan.so.1` on Linux,
`libvulkan.1.dylib` on macOS, `vulkan-1.dll` on Windows. This is the small system
library that finds your graphics driver. Without it, `import vkml` fails before
you reach any Python code, and pip cannot fix that for you.

Most machines with working graphics already have it. On a headless server you may
need to install it:

| System | Package |
|---|---|
| Debian / Ubuntu | `libvulkan1` |
| Fedora / RHEL | `vulkan-loader` |
| Arch | `vulkan-icd-loader` |
| macOS | `brew install vulkan-loader` |
| Windows | included with the Vulkan SDK and with most GPU drivers |

The loader is not the same thing as the driver. If you have the loader but no
driver, vkML imports fine and `vulkan_device_count()` returns `0` — you can
still use the CPU backend, but you have to ask for it.

---

## Choosing a device

**Using the GPU is an explicit choice.** vkML never transparently falls back to
the CPU, and this is deliberate rather than unfinished — a hidden fallback makes
performance impossible to reason about, because the same code silently runs at
very different speeds depending on the machine.

There are two ways to pick a device, and they behave differently on purpose.

### Ask vkML to choose

```python
device, why = vkml.best_device()
print(why)
# using Vulkan device 0: AMD Radeon RX 5600M (RADV NAVI10) (discrete, Vulkan
# 1.4.354, driver radv)
```

`best_device()` never raises. If no GPU is usable it returns the CPU **and tells
you why**:

```
running on the CPU: the Vulkan loader could not create an instance
(VK_ERROR_INCOMPATIBLE_DRIVER); either no driver is installed, or none of the
installed drivers supports Vulkan 1.3. Call vkml.vulkan_device_reports() to see
every device the loader can find and what each one is missing. README.md's
Troubleshooting section covers installing a driver and the Vulkan loader.
```

It returns the reason rather than printing it, so you can log it, show it in a
UI, or ignore it. It prefers a discrete GPU over an integrated one when both
work, and names the one it picked.

**This is the only path that falls back.**

### Name a device yourself

```python
vkml.init_vulkan(0)
device = vkml.device("vulkan:0")
```

Naming a device is a request, and it is **never quietly downgraded**. If that
GPU is not usable, `init_vulkan` raises with the reason — because someone who
typed `vulkan:1` wants *that* GPU, and handing back the CPU would hide the very
thing they were asking about behind a number that merely looks slow.

### If an operator has no GPU kernel

vkML raises rather than moving that one operator to the CPU behind your back:

```
NotImplementedError: backend 'vulkan:0' cannot evaluate op 'prod'. vkML does not
fall back to another device automatically -- doing so would move data through
host memory on every use and be far slower without saying so. Move this part of
the computation across explicitly, with `vkml.tensor(t.numpy(),
device=vkml.cpu)`, or open an issue if you need 'prod' on this backend.
```

Splitting a graph across devices means copying intermediates through host memory
at every split, which we measured at roughly **three times the cost of the
arithmetic being carried**. Doing that silently would turn a working model into
a mysteriously slow one. Two operators are affected today: `prod`, and
`max_pool2d` given a non-contiguous input.

The reasoning is written up in
[`docs/adr/0008-backend-selection-and-cpu-fallback.md`](docs/adr/0008-backend-selection-and-cpu-fallback.md).

---

## Building from source

If you want to work on vkML rather than just use it, build it with CMake
directly. This gives you the C++ tests and the benchmarks too.

### Linux and macOS

```sh
cmake --preset release -DVKML_VULKAN=ON
cmake --build build/release -j$(nproc)          # macOS: -j$(sysctl -n hw.ncpu)
```

The Python extension is written straight into `python/vkml/`, so you can use it
by putting that folder on your path:

```sh
PYTHONPATH=python python -c "import vkml; print(vkml.__version__)"
```

### Windows

Windows uses a multi-config generator, which ignores `CMAKE_BUILD_TYPE`. Do not
use the presets there — pass the config to the build instead:

```sh
cmake -B build/msvc -DVKML_VULKAN=ON -DVKML_BUILD_PYTHON=ON
cmake --build build/msvc --config Release --parallel
```

### Build options

| Option | Default | What it does |
|---|---|---|
| `VKML_VULKAN` | `OFF` | Build the Vulkan backend. `pip install` turns this on for you; with CMake you pass it yourself, as above |
| `VKML_BUILD_TESTS` | `ON` | Build the C++ test suite |
| `VKML_BUILD_PYTHON` | `ON` | Build the Python extension |
| `VKML_BUILD_BENCH` | `ON` | Build the benchmarks |
| `VKML_WERROR` | `OFF` | Treat warnings as errors |
| `VKML_SANITIZE` | `OFF` | Enable ASan and UBSan |

Presets: `release`, `debug` (warnings as errors), `asan`, `relwithdebinfo`.

### Running the tests

The Python test suite compares every operator against PyTorch, so you need
PyTorch installed. The CPU build is much smaller and is all the tests need:

```sh
pip install numpy pytest nanobind
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then:

```sh
python -m pytest tests/python -q      # Python suite, checked against PyTorch
ctest --preset release                # C++ suite
```

---

## Troubleshooting

**`import vkml` fails with "libvulkan.so.1: cannot open shared object file"**
The Vulkan loader is missing. Install it from the table
[above](#one-runtime-dependency-pip-cannot-install-for-you).

**The build stops with "no GLSL compiler found"**
vkML needs either `glslc` or `glslangValidator` to compile its shaders. Install
`glslang-tools` on Debian/Ubuntu, `glslang` on Fedora or Arch, or `shaderc` on
Arch if you prefer `glslc`. On Windows and macOS the Vulkan SDK and Homebrew
packages above already include one.

**`vulkan_device_names()` returns an empty list**
vkML installed fine, but the loader found no usable device. Run
`vulkaninfo --summary` to see whether *anything* can find your GPU. If that is
also empty, the problem is your driver, not vkML. On macOS, check that
`VK_ICD_FILENAMES` points at a file that exists.

**Your GPU is listed but vkML refuses it**
The backend needs exactly three Vulkan features: `bufferDeviceAddress`,
`scalarBlockLayout` and `timelineSemaphore`. If one is missing, vkML says which
one instead of crashing. Run `python scripts/hardware_report.py` for the details.

**CMake cannot find Vulkan on Windows**
The `VULKAN_SDK` environment variable is set by the SDK installer, but only for
terminals opened afterwards. Close your terminal, open a new one and try again.

---

## Examples

Two complete training scripts, both of which check themselves against PyTorch as
they run.

**MNIST** downloads the dataset for you the first time:

```sh
python examples/mnist/train.py          # trains on the GPU, compared to PyTorch
python examples/mnist/gui.py            # draw a digit and watch it predict
```

**CIFAR-100** needs the dataset downloaded by hand, because the site it comes
from does not like being scripted at:

```sh
mkdir -p examples/cifar100/cache
curl -L https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz \
     -o examples/cifar100/cache/cifar-100-python.tar.gz

python examples/cifar100/train.py
```

If the archive is missing, the script tells you exactly where to put it.

---

## What is included

| Module | What it provides |
|---|---|
| `vkml` | Tensors, 65 operators, dtypes, devices, lazy evaluation |
| `vkml.nn` | `Module`, Linear, Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d, LayerNorm, Dropout, Embedding, MultiheadAttention, TransformerEncoderLayer, activations, and the MSE, cross-entropy, BCE-with-logits, KL-divergence and Huber losses |
| `vkml.optim` | SGD (with momentum), Adam, AdamW, RMSProp |
| `vkml.data` | `Dataset`, `ArrayDataset`, `DataLoader`, reproducible shuffling, `split` |
| `vkml.serialize` | Checkpoints stored as a zip of `.npy` arrays plus JSON metadata |

Autograd is reverse-mode, with 47 backward rules. Each rule is written using
*forward* operations, which means gradients reuse the same kernels as the forward
pass — so a bug fixed in `mul` is fixed in the gradient of `mul` at the same
time.

Both **f32 and f16** work on either backend. f16 is used for storage only, never
for accumulation: values widen to float when they are read and narrow once when
written, so a reduction of any length still accumulates in fp32.

### Checkpoints cannot execute code

A vkML checkpoint is a zip file containing only data — `.npy` members read with
`allow_pickle=False`, plus a single JSON document. Neither format can name a
function, so loading a model file cannot run a program.

This is a deliberate choice. The well-known problem in this area is model
formats built on `pickle`, which rebuild objects by *calling* whatever the file
names, and we did not want to repeat that.

**Decompression bombs are rejected too**, before a byte is read. A small archive
that expands to an enormous allocation is caught from the zip directory:

```python
vkml.load("suspicious.vkml")
# ValueError: suspicious.vkml expands 1028x (203,998 bytes on disk to
# 209,715,402 in memory), over the 100x limit, and was rejected without being
# read. ...
```

The limit is an expansion *ratio* rather than a byte count, because the two
populations do not overlap: real checkpoints expand about 1×, real weights asked
to compress reach 1.1× — trained weights are high-entropy and barely compress —
while an all-zeros bomb reaches over 1000×. A byte cap has no non-arbitrary
value, since a 28 GB checkpoint and a 200 KB bomb are both things someone might
load.

A pruned model stored densely is mostly zeros and could exceed the limit
honestly. Raise it with `vkml.load(path, max_expansion_ratio=...)`; the error
says so.

---

## How we check correctness

Every operator is tested against PyTorch, in two stages:

```
vkml-cpu     vs  PyTorch      catches wrong formulas, axes and broadcast rules
vkml-vulkan  vs  vkml-cpu     catches kernel bugs, against an oracle with identical semantics
```

Tolerances are worked out from the reduction length **in advance**, and never
widened after a failure. A mismatch is treated as a bug until an error analysis
says otherwise.

A green test suite only proves the tests ran, so there are three more checks that
ask whether the suite *could* have failed:

| Check | The question it answers |
|---|---|
| `scripts/mutation_check.py` | If we break a kernel on purpose, does a test notice? |
| `scripts/coverage_matrix.py` | Which operator × dtype × layout combinations are never exercised? |
| `scripts/asan_python.py` | Does the Python suite pass against an AddressSanitizer build? |

```sh
python -m pytest tests/python -q        # the validation suite
python scripts/mutation_check.py        # can the suite fail?
python scripts/asan_python.py           # the suite, instrumented
```

Results are reproducible by design, which we treat as part of correctness rather
than a nice extra: no global atomics, fixed reduction trees, and a counter-based
RNG that is a pure function of its seed and index.

---

## Project status

**Alpha.** vkML trains real models and is tested hard, but the GPU evidence comes
from two cards plus whatever CI can reach, and it does not cover everything a
mature framework does.

**What works.** The MNIST MLP and CNN train end to end on the GPU and agree with
PyTorch well inside tolerance. 65 operators across both backends. 1,219 Python
tests and 96 C++ cases, run against three Vulkan drivers, with CI on Linux and
Windows. macOS is not in the push pipeline — see below.

**What does not.** Listed plainly, because you should find out here rather than
half an hour in:

- **Three drivers, and one of them is software.** A discrete RX 5600M (RDNA1) and
  an integrated Renoir iGPU, both under RADV; lavapipe, Mesa's software
  rasteriser, which has a subgroup size of 8 against RADV's 64 and half the
  shared memory; and MoltenVK on Apple Silicon. The full suite passes on both
  RADV GPUs and on lavapipe, and everything except one profiler test passes on
  MoltenVK. MNIST trains to the same accuracy and the same maximum divergence
  from PyTorch on both GPUs. There is **no evidence at all about NVIDIA, Intel,
  Qualcomm or ARM** — those are the reports we would most like to see.
- **macOS is not a supported platform yet.** It has been *probed*, not adopted:
  the job reached 1,233 passing tests on a GitHub runner, and one test cannot
  pass there because that runner's virtualised GPU reports working timestamps
  and never advances them. Rather than leave a permanently red job on every
  push, it now lives in `.github/workflows/macos-moltenvk.yml` and runs only on
  request (`gh workflow run macos-moltenvk.yml`). Treat macOS as untested until
  it comes back into the main pipeline.
- **What each platform actually proves.** Linux is the real test; every job runs
  there. Windows compiles under MSVC and passes the C++ suite, but the runner has
  no GPU, so nothing Vulkan is exercised there. The macOS job, when run by hand,
  exercises MoltenVK on an *Apple Paravirtual device* in a VM rather than
  physical Apple hardware; GPU timestamps do not advance there, so the profiler
  cannot be used on that runner. Validation layers are clean on it.
- **Vulkan is all-or-nothing**, by design rather than by omission — see
  [Choosing a device](#choosing-a-device). If the GPU cannot run an operator,
  vkML raises instead of silently moving that work to the CPU. Two cases exist
  today: `prod`, and `max_pool2d` with a non-contiguous input. It is listed here
  as well as there because it is a real limit on what vkML can run, whatever the
  reasoning behind it.
- **Missing layers:** Conv1d, Conv3d, gradient checkpointing.
- **Tensors are limited to rank 4.** This is a deliberate push-constant budget
  decision, not an oversight.
- **f16 matmul is correct but slower than f32**, because the vectorised tile load
  is f32-only for now.
- No distributed training, no quantisation, no ONNX.

### If you have hardware we do not

That list is short because it was written by someone with two GPUs. If you have
anything else — NVIDIA, Intel, Qualcomm, ARM, Apple, or Windows — a single
command tells us both whether vkML runs on it:

```sh
python scripts/hardware_report.py --run-tests
```

It describes every Vulkan device it can find and runs the validation suite
against the ones it can use. It is written to work even on a device vkML
**cannot** use, which is the report we most want: if your device is rejected, it
tells you which of the three required features is missing instead of crashing.
Please paste the output into an issue.

---

## Documentation

Design decisions are kept together with the reasoning behind them, including the
alternatives that were rejected and why:

| Document | What is in it |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layering, the graph IR, the backend interface, and the forks taken along the way |
| [`docs/THEORY.md`](docs/THEORY.md) | The numerics: error bounds, pairwise summation, stability |
| [`docs/adr/`](docs/adr) | Decisions that are expensive to reverse — graph ownership, the allocator, dtype promotion |
| [`docs/MEASUREMENT-AUDIT.md`](docs/MEASUREMENT-AUDIT.md) | How performance is measured, and which instruments lie |
| [`docs/PERFORMANCE-MODEL.md`](docs/PERFORMANCE-MODEL.md) | What the hardware can do, and what the kernels achieve |

---

## License

vkML is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
