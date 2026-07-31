"""Linear algebra and neural-network operators.

Written after reading shaders/softmax.comp, the five GEMM-family shaders,
VulkanBackend::compute's Matmul case in src/backend/vulkan/vulkan_backend.cpp
(lines 1748-2100, which is where every kernel-selection decision is made), and
src/api/ops.cpp for the operators that are composed rather than dispatched.
"""
from __future__ import annotations

L: dict[str, dict] = {}

L["matmul"] = {
    "summary": "Matrix product of two tensors.",
    "detail": "The behaviour depends on the dimensionality of the arguments:\n\n"
              "- If both are 2-D, the ordinary matrix product.\n"
              "- If either is more than 2-D, the leading axes are batch and broadcast against "
              "each other; the last two are multiplied.\n"
              "- The inner dimensions must agree: `(…, n, k)` against `(…, k, m)` gives "
              "`(…, n, m)`.\n\n"
              "**Six pipelines sit behind this one operator** — `gemv`, `gemm_naive`, "
              "`gemm_tiled`, `gemm_db`, `gemm_reg` and `gemm_split_k_reduce` — and it is the "
              "only operator in vkML that creates more than one. Which runs is decided per "
              "dispatch, in this order:\n\n"
              "- **GEMV**, when explicitly selected. One workgroup per *output element*. The "
              "tiled kernel collapses to `ceil(M/32)` workgroups when N=1, which on the "
              "development GPU is 128 against 288 concurrent slots — 44% occupancy, where "
              "throughput falls off a cliff. GEMV restores the grid to M·N.\n"
              "- **Naive, tiled or register-blocked**, from `VKML_GEMM_KERNEL`; the "
              "register-blocked kernel is the default.\n"
              "- **A forced fall back to naive** when the device cannot run the blocked "
              "kernel. Both blocked kernels hardcode 256 invocations — deliberately, so the "
              "three variants stay comparable — and Vulkan guarantees only 128.\n"
              "- **Split-K**, for the register-blocked kernel only, when the K dimension is "
              "long relative to the output.\n\n"
              "The register block is `BM=32, BN=32, RM=2, RN=2` with `BK=32`. `BK` is 32 "
              "rather than 16 for two measured reasons: it halves the K-tile count, which "
              "costs one fewer carry-stack level and therefore fewer registers; and it makes "
              "each block a 32-element sequential sum, **exactly matching `kPairwiseBlock` in "
              "`src/backend/cpu/reduce.h`**, so the two backends fold K with the same "
              "structure.",
    "params": [("input", "Tensor", "The left operand."),
               ("other", "Tensor",
                "The right operand. Its second-to-last axis must match `input`'s last axis.")],
    "returns": "The product, with batch axes broadcast.",
    "note": "Accumulation is always float32, including for float16 inputs. vkML does not "
            "offer float16 accumulation at any tile size — it is a common way to buy "
            "throughput and it is incompatible with the numerical contract.",
    "tip": "Split-K is bit-identical to the unsplit kernel, and that is a proof rather than a "
           "measurement: the chunk is always a power-of-two number of K-tiles, so no fold "
           "inside a partition crosses a boundary whose tile index has `q` low zero bits, "
           "which makes every partial exactly a subtree of the unsplit carry stack. GEMV is "
           "bit-identical to the tiled kernel by the same argument applied across lanes "
           "instead of across workgroups.",
    "warning": "The fallback is decided on **what the device permits, not what you asked "
               "for**. An explicit `VKML_GEMM_KERNEL` request for a kernel the device cannot "
               "run is still overridden, because the alternative is a `DeviceError` the "
               "caller can do nothing about. It is logged once per device, not per dispatch.",
    "example": """
>>> a = vkml.tensor(np.random.rand(64, 128).astype(np.float32))
>>> b = vkml.tensor(np.random.rand(128, 32).astype(np.float32))
>>> vkml.matmul(a, b).shape
(64, 32)
""",
    "see": ["conv2d", "im2col"],
}

L["softmax"] = {
    "summary": "Apply the softmax function along one axis.",
    "detail": "Computes `exp(xᵢ − max(x)) / Σ exp(xⱼ − max(x))` along `dim`.\n\n"
              "**One workgroup per row, three passes inside it**: the maximum over the axis, "
              "the sum of `exp(x − max)`, then the normalised write. All three share a single "
              "workgroup, so the two reductions synchronise with `barrier()` rather than "
              "needing separate dispatches and a global barrier — which is the whole reason "
              "softmax is one kernel and not three. The shader carries seven barriers, more "
              "than any other in the project.\n\n"
              "The maximum is subtracted before exponentiating so a large logit cannot "
              "overflow: `exp` reaches `+inf` above about 88.72 in float32.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("dim", "int = -1", "Axis to normalise over. Negative counts from the end.")],
    "returns": "A tensor of the same shape whose values along `dim` sum to 1.",
    "note": "The sum pass folds pairwise, matching the reduction family, so a long axis keeps "
            "the `O(log n)` error bound rather than accumulating sequentially.",
    "example": """
>>> x = vkml.tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
>>> vkml.softmax(x, -1).numpy()
array([[0.09003057, 0.24472848, 0.66524094]], dtype=float32)
""",
    "see": ["log_softmax", "cross_entropy", "sum"],
}

L["log_softmax"] = {
    "summary": "The logarithm of softmax, computed without forming the softmax.",
    "detail": "`x − max(x) − log(Σ exp(x − max(x)))`, sharing the softmax kernel and its "
              "three-pass structure — the difference is only what the final pass writes.\n\n"
              "Computing it this way is not an optimisation. `log(softmax(x))` underflows: a "
              "probability reaches exactly 0 in float32 once the logit gap passes about 90, "
              "and `log(0)` is `-inf`, which then poisons every gradient in the batch.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("dim", "int = -1", "Axis to normalise over.")],
    "returns": "A tensor of the same shape.",
    "example": """
>>> x = vkml.tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
>>> vkml.log_softmax(x, -1).numpy()
array([[-2.407606  , -1.4076059 , -0.40760595]], dtype=float32)
""",
    "see": ["softmax", "cross_entropy", "kl_div"],
}

L["layer_norm"] = {
    "summary": "Normalise over the trailing axes to zero mean and unit variance.",
    "detail": "Takes a **count of trailing axes** rather than a shape: `normalized_axes=1` "
              "normalises over the last axis, `2` over the last two. No weight or bias — "
              "vkML's `layer_norm` is the normalisation alone, and `vkml.nn.LayerNorm` "
              "applies the affine transform on top of it.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("normalized_axes", "int = 1", "How many trailing axes to normalise over."),
               ("eps", "float = 1e-5", "Added to the variance before the square root.")],
    "returns": "A tensor of the same shape and dtype.",
    "example": """
>>> x = vkml.tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
>>> vkml.layer_norm(x, 1, 1e-5).numpy()
array([[-1.2247356,  0.       ,  1.2247356]], dtype=float32)
""",
    "see": ["rms_norm", "batch_norm", "rsqrt"],
}

L["rms_norm"] = {
    "summary": "Normalise over the trailing axes by root-mean-square, without centring.",
    "detail": "`x / sqrt(mean(x²) + eps)`. Unlike `layer_norm` it does not subtract the mean, "
              "which makes it cheaper — one reduction instead of two — and is why "
              "transformer implementations increasingly prefer it.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("normalized_axes", "int = 1", "How many trailing axes to normalise over."),
               ("eps", "float = 1e-5", "Added to the mean square before the root.")],
    "returns": "A tensor of the same shape and dtype.",
    "example": """
>>> x = vkml.tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
>>> vkml.rms_norm(x, 1, 1e-5).numpy()
array([[0.46290955, 0.9258191 , 1.3887286 ]], dtype=float32)
""",
    "see": ["layer_norm", "rsqrt"],
}

L["batch_norm"] = {
    "summary": "Normalise each channel by supplied statistics, then scale and shift.",
    "detail": "`(x − mean) / sqrt(variance + eps) · weight + bias`, with the statistics passed "
              "in rather than computed. That is deliberate: this operator does not decide "
              "between batch statistics and running statistics, so it has no train/eval mode "
              "and no hidden state. `vkml.nn.BatchNorm2d` owns that decision and the running "
              "buffers, and calls this.",
    "params": [("input", "Tensor", "`(N, C, …)` — channels on axis 1."),
               ("mean", "Tensor", "Per-channel mean, `(C,)`."),
               ("variance", "Tensor", "Per-channel variance, `(C,)`."),
               ("weight", "Tensor = undefined", "Per-channel scale, `(C,)`. Omit for none."),
               ("bias", "Tensor = undefined", "Per-channel shift, `(C,)`. Omit for none."),
               ("eps", "float = 1e-5", "Added to the variance before the root.")],
    "returns": "A tensor of the same shape as `input`.",
    "see": ["layer_norm", "rms_norm"],
}

L["dropout"] = {
    "summary": "Randomly zero elements with probability `p`, scaling the rest.",
    "detail": "Inverted dropout: survivors are scaled by `1/(1−p)` during training so the "
              "expected value is unchanged and inference needs no compensation.\n\n"
              "The mask comes from the same **counter-based Philox** generator as `rand`, "
              "which is why the signature takes `(seed, offset)` rather than reading hidden "
              "state. The mask is a pure function of those, so it is reproducible across runs "
              "and identical on both backends — and you must advance `offset` between steps "
              "or every step drops the same elements.",
    "params": [("input", "Tensor", "Any shape, float dtype."),
               ("p", "float", "Probability of zeroing an element. `0.0` is the identity."),
               ("seed", "int", "Identifies the random stream."),
               ("offset", "int = 0", "Position in the stream. Advance between steps."),
               ("training", "bool = True", "When False, returns the input unchanged.")],
    "returns": "A tensor of the same shape and dtype.",
    "warning": "There is no global RNG state. Passing the same `(seed, offset)` on every step "
               "produces the same mask every step, which trains a model with a fixed set of "
               "dead units rather than with dropout.",
    "see": ["rand"],
}

L["conv2d"] = {
    "summary": "Apply a 2-D convolution over a batch of images.",
    "detail": "**Lowered to `im2col` followed by `matmul`**, not implemented as a direct "
              "convolution. `src/api/ops.cpp` builds it in four steps:\n\n"
              "- `im2col` expands `(N, C, H, W)` into `(N, C·kh·kw, L)`, where `L` is the "
              "number of window positions.\n"
              "- The weight is reshaped from `(C_out, C, kh, kw)` to `(C_out, C·kh·kw)`.\n"
              "- `matmul` broadcasts the batch axis, so the weights are shared across the "
              "batch without a copy.\n"
              "- The bias is reshaped to `(C_out, 1, 1)` so it broadcasts across batch and "
              "space with stride 0 rather than being materialised.\n\n"
              "The consequence is that every GEMM improvement reaches convolution for free — "
              "and so does every GEMM limitation.",
    "params": [("input", "Tensor", "`(N, C_in, H, W)`."),
               ("weight", "Tensor", "`(C_out, C_in, kh, kw)`."),
               ("bias", "Tensor = undefined", "`(C_out,)`. Omit for none."),
               ("stride", "Sequence[int] = [1, 1]", "Step in `(H, W)`."),
               ("padding", "Sequence[int] = [0, 0]", "Zero padding on both sides of `(H, W)`."),
               ("dilation", "Sequence[int] = [1, 1]", "Spacing between kernel elements.")],
    "returns": "`(N, C_out, H_out, W_out)`.",
    "warning": "**Grouped and depthwise convolution are not supported.** The weight's input "
               "channels must equal the input's, and `ops.cpp` rejects anything else by name "
               "rather than computing something wrong.",
    "note": "The im2col expansion is materialised in memory: a 3×3 kernel makes the "
            "intermediate roughly nine times the input. Implicit GEMM — folding the im2col "
            "addressing into the GEMM's operand load so the expansion is never written — is "
            "the standard fix and is recorded in the extensibility roadmap, not implemented.",
    "example": """
>>> x = vkml.tensor(np.random.rand(8, 3, 32, 32).astype(np.float32))
>>> w = vkml.tensor(np.random.rand(16, 3, 3, 3).astype(np.float32))
>>> vkml.conv2d(x, w, stride=[1, 1], padding=[1, 1]).shape
(8, 16, 32, 32)
""",
    "see": ["im2col", "col2im", "matmul", "max_pool2d"],
}

L["max_pool2d"] = {
    "summary": "Maximum over each sliding window, per channel.",
    "detail": "Accepts a **non-contiguous input**. The kernel addresses planes through the "
              "same `Operand` stride machinery every other kernel uses, rather than assuming "
              "a packed layout — so a transposed or sliced tensor pools correctly without a "
              "materialising copy first. A `CONTIGUOUS` specialisation constant lets the "
              "packed case skip the stride arithmetic entirely.",
    "params": [("input", "Tensor", "`(N, C, H, W)`."),
               ("kernel", "Sequence[int]", "Window size in `(H, W)`."),
               ("stride", "Sequence[int] = [0, 0]", "Step. Defaults to the kernel size."),
               ("padding", "Sequence[int] = [0, 0]", "Padding on both sides of `(H, W)`."),
               ("dilation", "Sequence[int] = [1, 1]", "Spacing between window elements.")],
    "returns": "`(N, C, H_out, W_out)`.",
    "note": "`max_pool2d_backward` still requires a contiguous input — the forward pass was "
            "extended and the backward one was not.",
    "example": """
>>> x = vkml.tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
>>> vkml.max_pool2d(x, [2, 2], [2, 2]).numpy()
array([[[[ 5.,  7.],
         [13., 15.]]]], dtype=float32)
""",
    "see": ["avg_pool2d", "conv2d", "amax"],
}

L["avg_pool2d"] = {
    "summary": "Mean over each sliding window, per channel.",
    "params": [("input", "Tensor", "`(N, C, H, W)`."),
               ("kernel", "Sequence[int]", "Window size in `(H, W)`."),
               ("stride", "Sequence[int] = [0, 0]", "Step. Defaults to the kernel size."),
               ("padding", "Sequence[int] = [0, 0]", "Padding on both sides of `(H, W)`.")],
    "returns": "`(N, C, H_out, W_out)`.",
    "note": "This is the operator that exposed the reduction dispatch ceiling: it issues one "
            "workgroup per output *row*, so 32 images of 3×64×64 reach 98,304 rows against "
            "the 65,535 Vulkan guarantees — an ordinary batch, where the elementwise path "
            "needs 16.7 million elements to get there.",
    "example": """
>>> x = vkml.tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
>>> vkml.avg_pool2d(x, [2, 2], [2, 2]).numpy()
array([[[[ 2.5,  4.5],
         [10.5, 12.5]]]], dtype=float32)
""",
    "see": ["max_pool2d", "mean", "conv2d"],
}
