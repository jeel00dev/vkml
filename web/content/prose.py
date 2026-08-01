"""Per-operator documentation.

Written against the actual behaviour of this library, not adapted from torch's
wording. Where vkML and torch differ, the difference is stated -- those are the
sentences that stop a reader debugging the wrong thing.
"""
from __future__ import annotations

PROSE: dict[str, dict] = {}


# --------------------------------------------------------------- creation --

PROSE["tensor"] = {
    "summary": "Create a tensor from a NumPy array.",
    "detail": "The data is **copied**, never aliased, so mutating the source array afterwards "
              "cannot change the tensor. Dtype follows the array unless one is given; a "
              "float64 array is narrowed to float32, because no backend implements float64.",
    "params": [
        ("data", "numpy.ndarray", "Values to copy. Any shape, including 0-d."),
        ("device", "device = cpu", "Where the tensor lives. Vulkan devices must be "
                                   "initialised with `init_vulkan` first."),
        ("requires_grad", "bool = False", "Whether autograd should track operations on it."),
    ],
    "returns": "A new tensor holding a copy of `data`.",
    "example": """
>>> import numpy as np, vkml
>>> x = vkml.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
>>> x.shape
(2, 2)
>>> vkml.init_vulkan(0)
'vulkan:0'
>>> g = vkml.tensor(np.zeros((4, 4), dtype=np.float32), device=vkml.device("vulkan:0"))
>>> g.device
device('vulkan:0')
""",
    "see": ["zeros", "ones", "from_numpy", "arange"],
}

PROSE["zeros"] = {
    "summary": "Create a tensor of the given shape filled with zeros.",
    "params": [
        ("shape", "Sequence[int]", "Extent of each axis."),
        ("dtype", "dtype = float32", "Element type."),
        ("device", "device = cpu", "Where to allocate."),
    ],
    "returns": "A new tensor of `shape`, every element zero.",
    "example": """
>>> vkml.zeros([2, 3]).numpy()
array([[0., 0., 0.],
       [0., 0., 0.]], dtype=float32)
""",
    "see": ["ones", "full", "tensor"],
}

PROSE["zeros_like"] = {
    "summary": "Create a tensor of zeros with the same shape, dtype and device as the input.",
    "detail": "Shorthand for `zeros(x.shape, dtype=x.dtype, device=x.device)`. Every "
              "property is taken from the input, so an f16 input gives an f16 result — "
              "nothing falls back to the `float32` default that the explicit form would "
              "apply if you forgot to pass `dtype`.\n\n"
              "The shape is the **logical** one, so a transposed view gives the "
              "transposed shape.",
    "params": [("input", "Tensor", "Tensor whose shape, dtype and device to copy.")],
    "returns": "A new tensor shaped like `input`, every element zero.",
    "note": "The input's *values* are not read and its graph is not realized — only its "
            "shape, dtype and device are used.",
    "example": """
>>> x = vkml.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> vkml.zeros_like(x).numpy()
array([[0., 0., 0.],
       [0., 0., 0.]], dtype=float32)
>>> vkml.zeros_like(x.astype(vkml.dtype.float16)).dtype
dtype.float16
""",
    "see": ["zeros", "ones_like", "full_like"],
}

PROSE["ones_like"] = {
    "summary": "Create a tensor of ones with the same shape, dtype and device as the input.",
    "detail": "Shorthand for `ones(x.shape, dtype=x.dtype, device=x.device)`. See "
              "`zeros_like` for how the input's properties are inherited.",
    "params": [("input", "Tensor", "Tensor whose shape, dtype and device to copy.")],
    "returns": "A new tensor shaped like `input`, every element one.",
    "example": """
>>> x = vkml.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> vkml.ones_like(x).numpy()
array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)
""",
    "see": ["ones", "zeros_like", "full_like"],
}

PROSE["full_like"] = {
    "summary": "Create a tensor filled with a value, shaped like the input.",
    "detail": "Shorthand for `full(x.shape, value, dtype=x.dtype, device=x.device)`. The "
              "value is given as a `float` and converted to the input's dtype, so a value "
              "outside that type's range will not survive the conversion.",
    "params": [
        ("input", "Tensor", "Tensor whose shape, dtype and device to copy."),
        ("value", "float", "The fill value, converted to `input`'s dtype."),
    ],
    "returns": "A new tensor shaped like `input`, every element `value`.",
    "example": """
>>> x = vkml.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> vkml.full_like(x, 7.0).numpy()
array([[7., 7., 7.],
       [7., 7., 7.]], dtype=float32)
""",
    "see": ["full", "zeros_like", "ones_like"],
}

PROSE["arange"] = {
    "summary": "Create a 1-D tensor of evenly spaced values over a half-open interval.",
    "detail": "The interval is `[start, stop)` — `stop` is excluded, matching "
              "`numpy.arange` and `torch.arange`.",
    "params": [
        ("start", "float", "First value."),
        ("stop", "float", "Exclusive upper bound."),
        ("step", "float = 1", "Spacing between values."),
        ("device", "device = cpu", "Where to allocate."),
    ],
    "returns": "A 1-D tensor with `ceil((stop - start) / step)` elements.",
    "example": """
>>> vkml.arange(0, 5, 1).numpy()
array([0., 1., 2., 3., 4.], dtype=float32)
""",
}


# ------------------------------------------------------------ element-wise --

PROSE["relu"] = {
    "summary": "Apply the rectified linear unit, element-wise.",
    "detail": "Computes `x <= 0 ? 0 : x`.\n\n"
              "The comparison is written against zero rather than as `max(x, 0)` so that "
              "**NaN propagates**: `max` would return the non-NaN operand on some drivers and "
              "silently swallow it.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A tensor of the same shape and dtype.",
    "note": "`relu(-0.0)` returns `+0.0`, matching PyTorch.",
    "example": """
>>> import numpy as np, vkml
>>> x = vkml.tensor(np.array([-2.0, -0.0, 0.0, 3.0], dtype=np.float32))
>>> vkml.relu(x).numpy()
array([0., 0., 0., 3.], dtype=float32)
""",
    "see": ["gelu", "silu", "sigmoid", "clamp_min"],
}

PROSE["gelu"] = {
    "summary": "Apply the Gaussian error linear unit, element-wise.",
    "detail": "Computes the exact form `x · Φ(x)`, where `Φ` is the standard normal CDF — "
              "not the tanh approximation.\n\n"
              "Both backends evaluate `Φ` through `erfc` rather than as `0.5(1 + erf(x/√2))`. "
              "The second form cancels catastrophically in the negative tail: measured over 512 "
              "points on `[-6, -3]` it reached a relative error of 1.0, returning exactly zero "
              "on a domain where the true value never is.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A tensor of the same shape and dtype.",
    "warning": "`gelu(-inf)` is NaN, matching PyTorch. A large finite negative input is not — "
               "it underflows smoothly to zero.",
    "example": """
>>> x = vkml.tensor(np.array([-6.0, -1.0, 0.0, 1.0], dtype=np.float32))
>>> vkml.gelu(x).numpy()
array([-5.9195355e-09, -1.5865526e-01,  0.0000000e+00,  8.4134471e-01],
      dtype=float32)
""",
    "see": ["erf", "erfc", "silu", "relu"],
}

PROSE["tanh"] = {
    "summary": "Apply the hyperbolic tangent, element-wise.",
    "detail": "Saturates to ±1 outside roughly `|x| > 10`, and the Vulkan kernel clamps there "
              "explicitly.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A tensor of the same shape and dtype.",
    "warning": "GLSL defines `tanh` as `(eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)`, which evaluates to "
               "`inf/inf = NaN` once `|x|` passes `ln(FLT_MAX) ≈ 88.72`. Whether a driver does "
               "that is left to the driver — one AMD driver did. vkML does not use the "
               "built-in, so a finite input never yields NaN here.",
    "example": """
>>> x = vkml.tensor(np.array([-89.0, -1.0, 0.0, 89.0], dtype=np.float32))
>>> vkml.tanh(x).numpy()
array([-1.       , -0.7615942,  0.       ,  1.       ], dtype=float32)
""",
    "see": ["sigmoid", "gelu"],
}

PROSE["clamp"] = {
    "summary": "Limit every element to the range `[min, max]`.",
    "params": [
        ("input", "Tensor", "Any shape, float dtype."),
        ("min", "float", "Lower bound, inclusive."),
        ("max", "float", "Upper bound, inclusive."),
    ],
    "returns": "A tensor of the same shape and dtype.",
    "example": """
>>> x = vkml.tensor(np.array([-3.0, 0.5, 7.0], dtype=np.float32))
>>> vkml.clamp(x, -1.0, 1.0).numpy()
array([-1. ,  0.5,  1. ], dtype=float32)
""",
    "see": ["clamp_min", "clamp_max", "maximum", "minimum"],
}


# ---------------------------------------------------------------- reduction --

PROSE["sum"] = {
    "summary": "Sum every element of a tensor.",
    "detail": "Reduces to a 0-d tensor. Summation is **pairwise**, not sequential: the error "
              "grows as `O(log n)` in the element count rather than `O(n)`, which is what keeps "
              "a large reduction usable in float32.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A 0-d tensor holding the total.",
    "note": "The result is bit-identical across runs on the same device, because the reduction "
            "tree is fixed by shape rather than by how work happened to be scheduled.",
    "example": """
>>> x = vkml.tensor(np.ones((1000,), dtype=np.float32))
>>> float(vkml.sum(x).item())
1000.0
""",
    "see": ["mean", "amax", "amin", "prod"],
}

PROSE["amax"] = {
    "summary": "The largest element of a tensor.",
    "detail": "Reduces to a 0-d tensor.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A 0-d tensor holding the maximum.",
    "warning": "**NaN propagates.** If any element is NaN the result is NaN, matching "
               "`torch.amax`. This is checked in both the per-lane fold and the shared-memory "
               "tree, because a comparison-only reduction would otherwise drop it.",
    "example": """
>>> x = vkml.tensor(np.array([1.0, float('nan'), 3.0], dtype=np.float32))
>>> vkml.amax(x).item()
nan
""",
    "see": ["amin", "argmax", "maximum"],
}

PROSE["prod"] = {
    "summary": "The product of every element of a tensor.",
    "detail": "Reduces to a 0-d tensor.",
    "params": [("input", "Tensor", "Any shape, float dtype.")],
    "returns": "A 0-d tensor holding the product.",
    "warning": "**CPU only, by decision.** Calling `prod` on a Vulkan tensor raises "
               "`NotImplementedError` rather than moving the data to the CPU. vkML's Vulkan "
               "backend is all-or-nothing: an operation either runs on the device or fails "
               "loudly, so a silent host round-trip can never be mistaken for GPU execution.",
    "example": """
>>> x = vkml.tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
>>> vkml.prod(x).item()
24.0
""",
    "see": ["sum", "mean"],
}


# --------------------------------------------------------- linear algebra --

PROSE["matmul"] = {
    "summary": "Matrix product of two tensors.",
    "detail": "The behaviour depends on the dimensionality of the arguments:\n\n"
              "- If both are 2-D, the ordinary matrix product.\n"
              "- If either is more than 2-D, the leading axes are treated as batch and are "
              "broadcast against each other; the last two axes are multiplied.\n"
              "- The inner dimensions must agree: `(…, n, k)` against `(…, k, m)` gives "
              "`(…, n, m)`.\n\n"
              "The kernel used is chosen from the shape. A matrix–vector product dispatches a "
              "dedicated GEMV kernel; larger products use a blocked kernel, falling back to a "
              "naive one on devices whose workgroup limit cannot fit the blocked geometry.",
    "params": [
        ("input", "Tensor", "The left operand."),
        ("other", "Tensor", "The right operand. Its second-to-last axis must match `input`'s "
                            "last axis."),
    ],
    "returns": "The product, with batch axes broadcast.",
    "note": "Accumulation is always float32, including when the inputs are float16. vkML does "
            "not offer float16 accumulation at any tile size — it is a common way to buy "
            "throughput and it is incompatible with the numerical contract.",
    "example": """
>>> a = vkml.tensor(np.random.rand(64, 128).astype(np.float32))
>>> b = vkml.tensor(np.random.rand(128, 32).astype(np.float32))
>>> vkml.matmul(a, b).shape
(64, 32)
""",
    "see": ["conv2d", "im2col"],
}

PROSE["softmax"] = {
    "summary": "Apply the softmax function along one axis.",
    "detail": "Computes `exp(xᵢ − max(x)) / Σ exp(xⱼ − max(x))` along `dim`. The maximum is "
              "subtracted first so that a large input cannot overflow `exp`.",
    "params": [
        ("input", "Tensor", "Any shape, float dtype."),
        ("dim", "int = -1", "Axis to normalise over. Negative counts from the end."),
    ],
    "returns": "A tensor of the same shape whose values along `dim` sum to 1.",
    "example": """
>>> x = vkml.tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
>>> vkml.softmax(x, -1).numpy()
array([[0.09003057, 0.24472848, 0.66524094]], dtype=float32)
""",
    "see": ["log_softmax", "cross_entropy"],
}

PROSE["conv2d"] = {
    "summary": "Apply a 2-D convolution over a batch of images.",
    "detail": "Input is `(N, C_in, H, W)` and the weight is `(C_out, C_in, kH, kW)`, giving "
              "`(N, C_out, H_out, W_out)`.",
    "params": [
        ("input", "Tensor", "Batch of images, `(N, C_in, H, W)`."),
        ("weight", "Tensor", "Kernels, `(C_out, C_in, kH, kW)`."),
        ("bias", "Tensor = undefined", "Per-output-channel bias, `(C_out,)`. Omit for none."),
        ("stride", "Sequence[int] = [1, 1]", "Step in `(H, W)`."),
        ("padding", "Sequence[int] = [0, 0]", "Zero padding added to both sides of `(H, W)`."),
        ("dilation", "Sequence[int] = [1, 1]", "Spacing between kernel elements."),
    ],
    "returns": "The convolved batch.",
    "example": """
>>> x = vkml.tensor(np.random.rand(8, 3, 32, 32).astype(np.float32))
>>> w = vkml.tensor(np.random.rand(16, 3, 3, 3).astype(np.float32))
>>> vkml.conv2d(x, w, stride=[1, 1], padding=[1, 1]).shape
(8, 16, 32, 32)
""",
    "see": ["max_pool2d", "avg_pool2d", "im2col", "matmul"],
}


# ------------------------------------------------------------------ losses --

PROSE["cross_entropy"] = {
    "summary": "Cross-entropy between unnormalised logits and integer class targets.",
    "detail": "Takes **logits**, not probabilities — the log-softmax is applied internally, in "
              "one numerically stable pass. Passing an already-softmaxed tensor is a common "
              "mistake and produces a quietly wrong, still-plausible loss.",
    "params": [
        ("input", "Tensor", "Logits, `(N, C)`."),
        ("target", "Tensor", "Class indices, `(N,)`, int64, each in `[0, C)`."),
        ("reduction", "Reduction = mean", "`mean`, `sum` or `none`."),
    ],
    "returns": "A 0-d tensor under `mean` or `sum`; `(N,)` under `none`.",
    "example": """
>>> logits = vkml.tensor(np.random.rand(4, 10).astype(np.float32))
>>> target = vkml.tensor(np.array([1, 0, 4, 9], dtype=np.int64))
>>> vkml.cross_entropy(logits, target).shape
()
""",
    "see": ["log_softmax", "binary_cross_entropy_with_logits", "kl_div"],
}


# ---------------------------------------------------- autograd & execution --

PROSE["backward"] = {
    "summary": "Compute gradients of a scalar with respect to every leaf that requires them.",
    "detail": "Walks the recorded graph in reverse and accumulates into each leaf's `.grad`. "
              "**Accumulates** — it does not overwrite — so a training loop must call "
              "`optimizer.zero_grad()` between steps, exactly as in PyTorch.",
    "params": [("root", "Tensor", "A 0-d tensor. Calling on a non-scalar raises.")],
    "note": "Backward passes are built from forward operators wherever they can be, rather "
            "than each having a hand-written kernel. That is why the kernel count stays near "
            "64 rather than doubling.",
    "example": """
>>> x = vkml.tensor(np.array([3.0], dtype=np.float32), requires_grad=True)
>>> vkml.backward(vkml.sum(vkml.mul(x, x)))
>>> x.grad.numpy()
array([6.], dtype=float32)
""",
    "see": ["realize", "detach"],
}

PROSE["realize"] = {
    "summary": "Force a lazily built graph to execute.",
    "detail": "Operations are recorded, not run, until a result is needed. `realize` is the "
              "explicit trigger; reading `.numpy()` or `.item()` triggers it implicitly.\n\n"
              "Batching work this way is what lets several operations share one submission, "
              "which matters because a submission costs an order of magnitude more than a "
              "dispatch.",
    "params": [("tensor", "Tensor", "The graph root to evaluate.")],
    "see": ["set_eager", "is_eager", "backward"],
}

PROSE["set_eager"] = {
    "summary": "Run every operation immediately instead of building a graph.",
    "detail": "Off by default. Turning it on makes a failure surface at the operation that "
              "caused it rather than at the next realize, which is the point — it is a "
              "debugging aid, not a performance mode, and it is slower.",
    "params": [("enabled", "bool", "Whether to execute eagerly.")],
    "note": "`VKML_EAGER=1` in the environment does the same thing without a code change. It is "
            "read once during initialisation and must not be changed while the process runs.",
    "see": ["is_eager", "realize"],
}


# ----------------------------------------------------------- serialization --

PROSE["save"] = {
    "summary": "Write named arrays to a vkML checkpoint file.",
    "params": [
        ("path", "str | Path", "Destination file."),
        ("tensors", "Mapping[str, numpy.ndarray]", "Arrays to store, by name."),
        ("metadata", "Mapping[str, Any] = None", "JSON-serialisable extras — epoch, score, "
                                                 "whatever the caller wants back."),
        ("compress", "bool = False", "Deflate the payload."),
    ],
    "note": "The path comes first, and the payload is NumPy arrays rather than tensors.",
    "example": """
>>> w = vkml.tensor(np.zeros((4, 4), dtype=np.float32))
>>> vkml.save("ckpt.vkml", {"w": w.numpy()}, metadata={"epoch": 3})
>>> ck = vkml.load("ckpt.vkml")
>>> ck.metadata["epoch"]
3
""",
    "see": ["load", "save_module", "load_module"],
}

PROSE["load_module"] = {
    "summary": "Load a checkpoint into an existing module and return it for its metadata.",
    "detail": "Parameters are restored **in place**, keeping each entry's device, dtype and "
              "`requires_grad`. A module already moved to a GPU stays there.",
    "params": [
        ("path", "str | Path", "Checkpoint to read."),
        ("module", "Module", "Module to load into. Its state dict must match exactly — a "
                             "missing or unexpected key raises."),
    ],
    "returns": "The `Checkpoint`, for its metadata.",
    "warning": "The metadata arrives **after** the state dict is installed, so a check written "
               "against it cannot guard the load. To decide whether to load at all, call "
               "`load` first, inspect, then `load_state_dict` yourself.",
    "example": """
>>> dev = vkml.device("vulkan:0")
>>> model = vkml.nn.Linear(16, 8).to(dev)
>>> vkml.save_module("m.vkml", model)
>>> ck = vkml.load_module("m.vkml", model)
>>> next(iter(model.named_parameters()))[1].device      # unchanged by the load
device('vulkan:0')
""",
    "see": ["save_module", "load", "save"],
}


# ------------------------------------------------------- devices & runtime --

PROSE["init_vulkan"] = {
    "summary": "Initialise a Vulkan device and make it available for allocation.",
    "detail": "Must be called before any tensor is placed on `vulkan:N`. Creates the device, "
              "the allocator and a staging buffer, so it is a once-per-process cost rather "
              "than a per-tensor one.",
    "params": [("index", "int = 0", "Which physical device, in enumeration order.")],
    "returns": "A description of the device that was selected.",
    "warning": "Enumeration order is **not** stable across environments — the same machine can "
               "report its discrete GPU at index 0 natively and at index 1 inside a container. "
               "Use `best_device` when the intent is \"the fastest one\", and select by "
               "`device_type` from `vulkan_device_reports` when the intent is a specific class "
               "of device.",
    "example": """
>>> vkml.init_vulkan(0)
'vulkan:0'
>>> vkml.vulkan_device_names()
['AMD Radeon RX 5600M (RADV NAVI10)', 'AMD Radeon Graphics (RADV RENOIR)']
""",
    "see": ["best_device", "vulkan_device_reports", "available_devices"],
}

PROSE["best_device"] = {
    "summary": "Pick the most capable available device, preferring discrete Vulkan hardware.",
    "returns": "A `(device, reason)` pair. The reason names the device and why it was chosen, "
               "so it can be logged rather than guessed at.",
    # No literal output: the device index, name and driver version all differ per
    # machine, which is the very instability this function exists to absorb.
    "example": """
>>> dev, why = vkml.best_device()
>>> print(why)                          # doctest: +SKIP
using Vulkan device 0: AMD Radeon RX 5600M (RADV NAVI10) (discrete, Vulkan 1.4.354, driver radv)
""",
    "see": ["init_vulkan", "available_devices", "vulkan_device_reports"],
}
