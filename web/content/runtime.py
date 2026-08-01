"""Autograd control, serialization, and device introspection.

Written after reading python/vkml/serialize.py in full, the device-report
plumbing in src/backend/vulkan/vk_device.cpp, and the profiling path in
vk_command.cpp and vulkan_backend.cpp.
"""
from __future__ import annotations

RT: dict[str, dict] = {}

# ------------------------------------------------- autograd and execution --

RT["backward"] = {
    "summary": "Compute gradients of a scalar with respect to every leaf that requires them.",
    "detail": "Walks the recorded graph in reverse and **accumulates** into each leaf's "
              "`.grad` — it does not overwrite. A training loop must therefore call "
              "`optimizer.zero_grad()` between steps, exactly as in PyTorch.\n\n"
              "Backward rules are built from **forward operators** wherever they can be, "
              "rather than each having a hand-written kernel. That is why the kernel count "
              "stays near 64 instead of roughly doubling, and it is a recorded design rule "
              "rather than an accident: a new operator earns a dedicated backward kernel only "
              "if its gradient genuinely cannot be composed.",
    "params": [("root", "Tensor", "A 0-d tensor. Calling on a non-scalar raises.")],
    "note": "47 of the 66 OpKinds have a gradient rule. Calling `backward` through one that "
            "does not — `prod`, `erf`, `erfc` among them — raises `NotImplementedError` "
            "naming the operator, rather than silently producing a zero gradient.",
    "example": """
>>> x = vkml.tensor(np.array([3.0], dtype=np.float32), requires_grad=True)
>>> vkml.backward(vkml.sum(vkml.mul(x, x)))
>>> x.grad.numpy()
array([6.], dtype=float32)
""",
    "see": ["realize", "detach", "set_eager"],
}

RT["realize"] = {
    "summary": "Force a lazily built graph to execute.",
    "detail": "Operations are recorded, not run, until a result is needed. `realize` is the "
              "explicit trigger; `.numpy()`, `.item()` and a backward pass trigger it "
              "implicitly.\n\n"
              "Batching work this way is what lets many operations share one submission, and "
              "on this hardware a submission costs about **105 µs against 9 µs for a "
              "dispatch**. Reducing submissions is worth far more than making any single "
              "kernel faster, which is why the laziness is a performance mechanism and not "
              "just an API style.",
    "params": [("tensor", "Tensor", "The graph root to evaluate.")],
    "see": ["set_eager", "is_eager", "backward"],
}

RT["set_eager"] = {
    "summary": "Run every operation immediately instead of building a graph.",
    "detail": "Off by default. Turning it on makes a failure surface **at the operation that "
              "caused it** rather than at the next realize, which is the point — it is a "
              "debugging aid, not a performance mode, and it is slower because every "
              "operation becomes its own submission.",
    "params": [("enabled", "bool", "Whether to execute eagerly.")],
    "note": "`VKML_EAGER=1` does the same without a code change. Like every `VKML_*` switch it "
            "is read once during initialisation and is not intended to change while the "
            "process runs.",
    "see": ["is_eager", "realize"],
}

RT["is_eager"] = {
    "summary": "Whether eager execution is currently on.",
    "returns": "`True` if operations run immediately.",
    "see": ["set_eager", "realize"],
}

# ---------------------------------------------------------- serialization --

RT["save"] = {
    "summary": "Write named arrays to a vkML checkpoint.",
    "detail": "The file is a **ZIP archive** containing `vkml.json` — format identifier, "
              "version, key list and user metadata — and one `tensors/<key>.npy` per entry.\n\n"
              "That layout is a **security decision before an engineering one**. The "
              "well-known failure in this field is a model format that deserialises into code "
              "execution: Python's `pickle` reconstructs objects by *calling* what the stream "
              "names, so loading a checkpoint from anywhere but your own disk runs a program "
              "someone else wrote. A vkML checkpoint has no mechanism for it — every member is "
              "either a `.npy` array read with `allow_pickle=False` or one JSON document, and "
              "neither can name a callable.\n\n"
              "Parsing the array bytes is delegated to NumPy's own reader rather than "
              "hand-written, because a hand-rolled binary parser is exactly where a "
              "memory-safety bug would go.",
    "params": [("path", "str | Path", "Destination file."),
               ("tensors", "Mapping[str, numpy.ndarray]", "Arrays to store, by name."),
               ("metadata", "Mapping[str, Any] = None",
                "JSON-representable extras — epoch, score, architecture name."),
               ("compress", "bool = False", "Deflate the payload. Off by default.")],
    "note": "The path comes **first**, and the payload is NumPy arrays rather than tensors.",
    "example": """
>>> import tempfile, os
>>> path = os.path.join(tempfile.mkdtemp(), "ckpt.vkml")
>>> w = vkml.tensor(np.zeros((4, 4), dtype=np.float32))
>>> vkml.save(path, {"w": w.numpy()}, metadata={"epoch": 3})
>>> ck = vkml.load(path)
>>> ck.metadata["epoch"]
3
""",
    "see": ["load", "save_module", "load_module"],
}

RT["load"] = {
    "summary": "Read a vkML checkpoint and return its arrays and metadata.",
    "detail": "Returns a `Checkpoint` with `.tensors`, `.metadata` and `.version`, kept as "
              "separate fields so a metadata key can never collide with a tensor name.",
    "params": [("path", "str | Path", "Checkpoint to read."),
               ("max_expansion_ratio", "float = 100.0",
                "Reject the file if its members expand by more than this factor.")],
    "returns": "A `Checkpoint`.",
    "warning": "**Decompression bombs are rejected by expansion ratio, not by absolute size**, "
               "and the reason is measured rather than assumed: every checkpoint in this "
               "repository expands 1.00× (the default is stored, not deflated), real float32 "
               "weights asked to compress reach 1.1× — weights are high-entropy and barely "
               "compress — and an all-zeros bomb reaches 1017×. Three orders of magnitude with "
               "nothing in between.\n\n"
               "A byte cap has no non-arbitrary value: set low it breaks a legitimate large "
               "model, set high it stops nothing, and a 28 GB checkpoint and a 200 KB bomb are "
               "both things someone might load. The ratio makes an attacker's cost scale with "
               "the damage — forcing an N-byte allocation requires shipping N/100 bytes. It "
               "does not bound memory absolutely and is not meant to.",
    "note": "Known false positive: a pruned or sparse model stored densely is mostly zeros and "
            "can exceed the ratio legitimately. The error names the file, the ratio and the "
            "argument that raises it.",
    "see": ["save", "load_module"],
}

RT["save_module"] = {
    "summary": "Write a module's state dict to a checkpoint.",
    "params": [("path", "str | Path", "Destination file."),
               ("module", "Module", "The module whose state to save."),
               ("metadata", "Mapping[str, Any] = None", "JSON-representable extras."),
               ("compress", "bool = False", "Deflate the payload.")],
    "see": ["load_module", "save"],
}

RT["load_module"] = {
    "summary": "Load a checkpoint into an existing module and return it for its metadata.",
    "detail": "Parameters are restored **in place**, keeping each entry's device, dtype and "
              "`requires_grad`. A module already moved to a GPU stays there.\n\n"
              "The state dict must match exactly — a missing or unexpected key raises rather "
              "than being ignored, so a checkpoint from a different architecture fails at the "
              "load instead of producing a partly-initialised model.",
    "params": [("path", "str | Path", "Checkpoint to read."),
               ("module", "Module", "Module to load into.")],
    "returns": "The `Checkpoint`, for its metadata.",
    "warning": "**The metadata arrives after the state dict is installed**, so a check written "
               "against it cannot guard the load. To decide whether to load at all, call "
               "`load` first, inspect, then `load_state_dict` yourself.",
    "example": """
>>> import tempfile, os
>>> path = os.path.join(tempfile.mkdtemp(), "m.vkml")
>>> dev = vkml.device("vulkan:0")
>>> model = vkml.nn.Linear(16, 8).to(dev)
>>> vkml.save_module(path, model)
>>> ck = vkml.load_module(path, model)
>>> next(iter(model.named_parameters()))[1].device      # unchanged by the load
device('vulkan:0')
""",
    "see": ["save_module", "load", "save"],
}

# --------------------------------------------------- devices and profiling --

RT["init_vulkan"] = {
    "summary": "Initialise a Vulkan device and make it available for allocation.",
    "detail": "Must be called before any tensor is placed on `vulkan:N`. Creates the device, "
              "the allocator and a staging buffer, so it is a once-per-process cost rather "
              "than a per-tensor one.",
    "params": [("index", "int = 0", "Which physical device, in enumeration order.")],
    "returns": "The device string that was initialised.",
    "warning": "**Enumeration order is not stable across environments.** The same machine can "
               "report its discrete GPU at index 0 natively and at index 1 inside a container, "
               "with a software rasteriser appearing as a third device. Use `best_device` when "
               "the intent is \"the fastest one\", and select on `device_type` from "
               "`vulkan_device_reports` when the intent is a specific class of device.",
    "example": """
>>> vkml.init_vulkan(0)
'vulkan:0'
""",
    "see": ["best_device", "vulkan_device_reports", "available_devices"],
}

RT["available_devices"] = {
    "summary": "Every device that can currently hold a tensor.",
    "returns": "A list of `device` objects, always including `cpu`.",
    "see": ["best_device", "init_vulkan"],
}

RT["vulkan_available"] = {
    "summary": "Whether a usable Vulkan device was found.",
    "returns": "`True` if at least one device passed the required-feature check.",
    "see": ["vulkan_unavailable_reason", "vulkan_capabilities"],
}

RT["vulkan_unavailable_reason"] = {
    "summary": "Why Vulkan is unavailable, when it is.",
    "detail": "Names the first required feature that was missing, rather than reporting a "
              "generic failure — the required set is `bufferDeviceAddress`, "
              "`scalarBlockLayout` and `timelineSemaphore`, and a device lacking any of them "
              "cannot run vkML's kernels at all.",
    "returns": "A string, empty when Vulkan is available.",
    "see": ["vulkan_available", "vulkan_capabilities"],
}

RT["vulkan_device_count"] = {
    "summary": "How many Vulkan devices were enumerated.",
    "returns": "An integer, including devices that failed the feature check.",
    "see": ["vulkan_device_names", "vulkan_device_reports"],
}

RT["vulkan_device_names"] = {
    "summary": "The name of each enumerated Vulkan device.",
    "returns": "A list of strings, in enumeration order.",
    "see": ["vulkan_device_reports", "init_vulkan"],
}

RT["vulkan_device_reports"] = {
    "summary": "A full capability report per device.",
    "detail": "Each entry carries the device name, `device_type` (`discrete`, `integrated` or "
              "`cpu`), `driver_name`, the required and optional features, and the limits that "
              "decide which kernels can run — subgroup size and its controllable range, "
              "workgroup invocations, shared memory, push-constant bytes, memory heaps.\n\n"
              "**Select on `device_type` rather than on index** when a specific class of "
              "device is wanted; the index is not stable across environments.",
    "returns": "A list of dicts, one per device.",
    "see": ["vulkan_capabilities", "best_device", "init_vulkan"],
}

RT["vulkan_capabilities"] = {
    "summary": "The capability report for one initialised device.",
    "params": [("index", "int = 0", "Which device.")],
    "returns": "A dict of features and limits.",
    "see": ["vulkan_device_reports"],
}

RT["vulkan_stats"] = {
    "summary": "Allocation and dispatch counters for one device.",
    "params": [("index", "int = 0", "Which device.")],
    "returns": "A dict of counters.",
    "see": ["vulkan_pipeline_stats", "vulkan_last_profile"],
}

RT["vulkan_pipeline_stats"] = {
    "summary": "Compiler statistics for every pipeline created so far.",
    "detail": "Reports what the shader compiler produced per pipeline — register counts, "
              "spilled scratch, occupancy — which is how the register model in the GEMM work "
              "was fitted and tested. Because each geometry compiles to an independent "
              "pipeline through specialisation constants, they benchmark and report "
              "separately.",
    "params": [("index", "int = 0", "Which device.")],
    "returns": "A list of dicts, one per pipeline.",
    "note": "Only populated when the driver supports pipeline executable properties. "
            "`VKML_VULKAN_NO_PIPELINE_STATS=1` disables collection.",
    "see": ["vulkan_stats", "vulkan_last_profile"],
}

RT["vulkan_timestamps_supported"] = {
    "summary": "Whether the device can timestamp GPU work.",
    "params": [("index", "int = 0", "Which device.")],
    "returns": "`True` if timestamp queries are usable.",
    "note": "A device may report valid timestamp bits and then never advance the counter — "
            "one virtualised device does exactly that — so a `True` here is necessary but not "
            "sufficient for a meaningful profile.",
    "see": ["vulkan_last_profile", "vulkan_set_profiling"],
}

RT["vulkan_set_profiling"] = {
    "summary": "Turn GPU timing on or off for a device.",
    "params": [("enabled", "bool", "Whether to record timings."),
               ("index", "int = 0", "Which device.")],
    "see": ["vulkan_last_profile", "vulkan_timestamps_supported"],
}

RT["vulkan_last_profile"] = {
    "summary": "Timings from the most recent profiled run.",
    "returns": "A list of `(label, milliseconds)` pairs.",
    "params": [("index", "int = 0", "Which device.")],
    "warning": "**Labelled by operator, not by kernel.** The entries are real per-operation GPU "
               "time, not submission totals — timestamps are written and resolved per node. What "
               "they cannot tell you is which KERNEL ran: a `matmul` entry does not distinguish "
               "the naive kernel from the register-blocked one, nor show split-K's partitions. "
               "Joining cost to kernel choice needs a shared dispatch identity, which is "
               "designed but not yet implemented. Until then, attribution comes from indirect "
               "evidence such "
               "as batch scaling or device substitution.",
    "see": ["vulkan_submit_ms", "vulkan_set_profiling", "vulkan_pipeline_stats"],
}

RT["vulkan_submit_ms"] = {
    "summary": "Total GPU milliseconds across a profile's submissions.",
    "params": [("profile", "list", "The result of `vulkan_last_profile`.")],
    "returns": "A float.",
    "see": ["vulkan_last_profile"],
}

RT["vulkan_set_subgroup_override"] = {
    "summary": "Force a subgroup width, for testing.",
    "detail": "A testing facility rather than a tuning knob. The device reports a controllable "
              "range — 32 to 64 on the development GPU — and pinning a width inside it is how "
              "a kernel's dependence on subgroup size is tested without different hardware.",
    "params": [("index", "int = 0", "Which device."),
               ("size", "int", "The width to force. 0 restores the driver's choice.")],
    "see": ["vulkan_capabilities"],
}

RT["set_log_level"] = {
    "summary": "Set the minimum severity vkML logs.",
    "params": [("level", "LogLevel", "`OFF`, `ERROR`, `WARN`, `INFO`, `DEBUG` or `TRACE`.")],
    "note": "`VKML_VULKAN_DEBUG=1` raises Vulkan-specific tracing separately, and logs every "
            "dispatch with its shape and grid.",
    "see": ["vulkan_stats"],
}


# ---------------------------------------------------------------------------
# The decision recorder (docs/OBSERVABILITY-ARCHITECTURE.md)
# ---------------------------------------------------------------------------
#
# Not prefixed `vulkan_`, and that is deliberate: decisions are published from
# any layer, so a CPU-only build has them too.

RT["record_decisions"] = {
    "summary": "Begin recording what the engine chooses, and instead of what.",
    "detail": "vkML makes choices a caller cannot see — which matmul kernel fits the device, "
              "whether split-K participates, which memory kind an allocation really got. Those "
              "used to exist only as log text, which nothing could assert on and which pytest "
              "captures away entirely. Recording turns them into structured facts you can read.\n\n"
              "The window is bounded and the oldest entry is dropped, because unbounded history "
              "would make this a tracing system rather than an answer to *why did that happen*.",
    "params": [("capacity", "int = 256", "How many decisions to keep.")],
    "note": "Recording does not silence the log — the two are independent renderings of the "
            "same fact.",
    "tip": "Measured cost: 58.8 ns to publish a decision with nobody recording, 131.7 ns while "
           "recording, so the recorder itself adds about 73 ns per decision. Against a "
           "millisecond-scale operation that is unmeasurable; against a microsecond-scale one it "
           "would not be. Decisions are published for coarse choices, which is why recording is "
           "safe to leave on while you investigate.",
    "see": ["decisions", "decisions_published", "stop_recording_decisions",
            "vulkan_pipeline_stats"],
}

RT["decisions"] = {
    "summary": "What the engine recently chose, oldest first.",
    "detail": "Each entry carries `site` (where the choice was made), `op` (the operation it was "
              "made for), `chose`, `instead_of`, `because`, and the numbers that forced it — "
              "`required` against `available`. The numbers are the point: they can be "
              "contradicted by what the driver independently reports about the pipeline that was "
              "actually created, which a prose reason cannot.",
    "returns": "A list of dicts, oldest first.",
    "note": "A decision is a claim the engine makes about itself. It is checked against facts "
            "with a different owner — `vulkan_pipeline_stats` reports what the *driver* saw in "
            "the compiled pipeline, and `vulkan_last_profile` counts dispatches — because two "
            "accessors over the same decision could never disagree.",
    "see": ["record_decisions", "decisions_published", "vulkan_pipeline_stats",
            "vulkan_last_profile"],
}

RT["decisions_published"] = {
    "summary": "How many decisions were published since recording began, including evicted ones.",
    "detail": "Compare with `len(decisions())` to tell a full window from a complete history. A "
              "reader who cannot make that distinction will draw conclusions from a truncated "
              "record without knowing it.",
    "returns": "An integer count.",
    "see": ["decisions", "record_decisions"],
}

RT["stop_recording_decisions"] = {
    "summary": "Stop recording and release the window.",
    "see": ["record_decisions"],
}


RT["configuration"] = {
    "summary": "Every environment switch this process has consulted, and what it saw.",
    "detail": "Eighteen `VKML_*` variables change what vkML does, and none of them used to be "
              "visible in a running process. `VKML_GEMM_NOVEC=1` costs a measured 14.6% on a "
              "1024-cubed matmul and left no trace anywhere — the same signature as a code "
              "regression, and impossible to rule out after the fact.\n\n"
              "Entries are OBSERVED, not declared: one appears because something read it, "
              "recorded inside the project's single `getenv`. A hand-written list of switches "
              "would be a second model of vkML's configuration and would go stale the first time "
              "somebody added one.",
    "returns": "A list of dicts with `name`, `value` and `set`.",
    "note": "`set` distinguishes unset from set-to-empty, which is the difference between "
            "*the default applied* and *nobody looked*.",
    "warning": "A switch appears only once it has been consulted. Several are read lazily on "
               "first use, so ask after the work you care about rather than at startup — a "
               "matmul is what makes the GEMM switches appear.",
    "see": ["decisions", "record_decisions"],
}


RT["vulkan_profile_records"] = {
    "summary": "Measured GPU intervals carrying the identity of what they measured.",
    "detail": "The same intervals `vulkan_last_profile` returns, plus a `dispatch` field. Join it "
              "against `decisions()` on the same field to answer *what did this kernel cost* — "
              "a question neither side can answer alone.\n\n"
              "The profiler deliberately does not know which kernel ran, and the planner "
              "deliberately does not know what anything cost. Identity is a third fact, owned by "
              "the recorder, describing nothing. That is what stops kernel selection acquiring a "
              "second owner.",
    "params": [("index", "int = 0", "Which device.")],
    "returns": "A list of dicts with `label`, `gpu_ms` and `dispatch`.",
    "note": "`dispatch` is 0 when the interval is not a dispatch — the whole-submission entry is "
            "the case that exists today. Compare ids for equality only; they are opaque and will "
            "widen when multiple queues arrive.",
    "see": ["vulkan_last_profile", "decisions", "record_decisions"],
}
