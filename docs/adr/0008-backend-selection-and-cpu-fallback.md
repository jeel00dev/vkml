# ADR 0008 — Backend selection, and why there is no automatic CPU fallback

**Status:** accepted; both decisions implemented (7 below)
**Date:** 2026-07-29
**Covers:** task #16 ("implement CPU fallback via graph splitting, or state that
Vulkan is all-or-nothing"), and the backend-selection experience around it.

---

## 1. What happens today, measured

The task and the UX question around it both assume there may be a silent
fallback. **There is none.** Every path throws. Run on a machine made
driver-less with `VK_ICD_FILENAMES=/nonexistent/none.json`:

| Situation | Today |
|---|---|
| `V.device("vulkan:0")`, no driver | **Succeeds.** It only constructs a value; nothing is validated |
| a tensor on that device | `DeviceError: no backend registered for device 'vulkan:0'` |
| `V.init_vulkan(0)`, no driver | `DeviceError: vkCreateInstance failed: VK_ERROR_INCOMPATIBLE_DRIVER` |
| an op Vulkan cannot run | `NotImplementedError: backend 'vulkan:0' cannot evaluate op 'prod'` |

So the honest summary is: **nothing is silent, and nothing is helpful.** Three of
the four messages state what failed and none states *why* or *what to do*. The
raw `VK_ERROR_INCOMPATIBLE_DRIVER` in particular is a Vulkan enum shown to a
user who may not know Vulkan exists.

`vulkan_unavailable_reason()` already produces exactly the right sentence --
*"the Vulkan loader could not create an instance (VK_ERROR_INCOMPATIBLE_DRIVER);
either no driver is installed, or none of the installed drivers supports Vulkan
1.3"* -- and **nothing in the library's own error paths calls it.** Only
`scripts/hardware_report.py` does.

There is also real duplication: `examples/mnist/train.py` and
`examples/cifar100/train.py` both hand-roll the same `resolve_device`, down to
the same printed sentence.

---

## 2. Decision 1 -- no automatic per-operator fallback. Vulkan stays all-or-nothing.

The literal #16 asks whether to split a graph so an operator the GPU cannot run
executes on the CPU instead. **Recommendation: no, and document it.**

Three reasons, the second decisive:

**It would be very slow, and we now know by how much.** A split point means
moving intermediates device -> host -> device. Stage A measured that exact
transfer: a host round trip cost three times the arithmetic it carried
(`docs/adr/0006`, 7). A model using one unsupported operator in its inner loop
would run correct and catastrophically slow.

**It would be silent, which is the failure mode we are trying to remove.** A
user whose model contains `prod` would get right answers at a fraction of the
speed with nothing to indicate why. An automatic fallback *is* the silent
degradation this ADR exists to avoid -- so implementing it to improve the
experience would defeat the goal.

**The surface is two narrow cases**, not a general gap: `prod`, and
`max_pool2d` given a non-contiguous input. Both are better addressed directly --
implement the kernel, or make the caller contiguous -- than by a scheduling
mechanism that applies to everything.

**Cost of saying no.** A user who needs one unsupported operator has to move
that part of their model to the CPU by hand, and vkML cannot run models it
almost supports. That is a real cost and the reason to revisit if the list of
unsupported operators ever grows past a handful.

What changes instead: the error names the remedy.

```
NotImplementedError: backend 'vulkan:0' cannot evaluate op 'prod'.
vkML does not fall back to the CPU automatically -- a fallback would move data
through host memory on every use and be far slower without saying so. Run this
part on the CPU explicitly (`t.to(vkml.cpu)`), or open an issue if you need it
on the GPU.
```

---

## 3. Decision 2 -- selection is explicit, and always explains itself

Two rules, and the first is already the project's recorded judgement, taken from
the examples' own comment:

> **A device the user NAMED is never silently downgraded.** Someone who typed
> `vulkan:1` wants that GPU; handing back the CPU hides the thing they asked
> about.

> **A device the user asked vkML to CHOOSE always comes with the reason.**

That gives one new function, replacing the block both examples copy:

```python
def best_device(*, prefer: str = "vulkan") -> tuple[Device, str]:
    """The best usable device, and a one-line explanation of the choice.

    Returns the reason rather than printing it: a library should not write to
    anyone's stdout, and a caller that wants to log it, show it in a UI or
    ignore it should be free to.
    """
```

Two outcomes, both explaining themselves:

```python
dev, why = V.best_device()
# ("vulkan:0", "using Vulkan device 0: AMD Radeon RX 5600M (RADV NAVI10)")
# (cpu,        "running on the CPU: the Vulkan loader could not create an
#               instance (VK_ERROR_INCOMPATIBLE_DRIVER); either no driver is
#               installed, or none of the installed drivers supports Vulkan 1.3")
```

The CPU sentence is `vulkan_unavailable_reason()`, which already exists and is
already right. This is wiring, not new diagnosis.

**The four things asked for**, mapped:

| Wanted | Where it comes from |
|---|---|
| which backend was selected | the returned `Device`, and it is named in the reason |
| whether a fallback occurred | `dev == V.cpu` while `prefer="vulkan"`, and the reason says "running on the CPU" |
| why | `vulkan_unavailable_reason()`, or the per-device `missing_requirement` when a GPU is present but unusable |
| how to enable Vulkan | the reason names the cause; the README's Troubleshooting section covers each cause |

---

## 4. Where the improved messages can live -- a layering constraint

`backend_for()` produces `no backend registered for device 'vulkan:0'` and lives
in `backend/api`, **layer 3**. `backend/vulkan` is **layer 4**
(`scripts/check_layering.py`). A lower layer cannot include a higher one, so
that message *cannot* pull in `vulkan_unavailable_reason()`.

Two ways round it, and the cheap one is right:

* **Enrich it in Python**, where both `backend_for`'s failure and
  `vulkan_unavailable_reason()` are already visible. `best_device()` lives there
  anyway. **Recommended.**
* Add a registration-hint mechanism to `backend/api` so a backend can publish
  why it failed to register. That is a general solution to a problem with two
  instances, and would be speculative (P6).

`backend_for`'s own message gains only what its layer legitimately knows: that
the device was never initialised.

---

## 5. Cost of this proposal

| | |
|---|---|
| **Benefit** | Every failure states the cause and the remedy. One helper replaces a block copied into two examples and partly into `hardware_report.py`. No behaviour becomes silent |
| **Cost** | One new public function to keep. Longer error strings. And the honest one: choosing NOT to build graph splitting means vkML still cannot run a model that uses one unsupported operator |
| **Worthwhile when** | The unsupported set stays small and explicit failure is preferable to silent slowness -- true now, with two narrow cases |
| **Not worthwhile when** | The unsupported set grows enough that "use the CPU by hand" stops being reasonable advice. Then graph splitting earns its complexity, and it should arrive with a loud warning rather than silently |

**Revisit trigger for decision 1:** more than a handful of operators unsupported
on Vulkan, or a real model blocked by one.

---

## 6. What this does NOT change

* `V.device("vulkan:0")` still constructs without validating. Making it throw
  would mean device construction touching the driver, and the failure already
  arrives at first use with a message this ADR improves.
* No existing call gains a fallback. `init_vulkan()` still throws when the
  device it was asked for is unusable, which is decision 2's first rule.

---

## 7. As built

### Decision 1 -- accepted. Vulkan stays all-or-nothing, and now says so.

No graph splitting. The unsupported-operator error states the absence of a
fallback and the reason for it, rather than leaving a reader to infer either:

```
NotImplementedError: backend 'vulkan:0' cannot evaluate op 'prod'. vkML does not
fall back to another device automatically -- doing so would move data through
host memory on every use and be far slower without saying so. Move this part of
the computation to the CPU explicitly, or open an issue if you need 'prod' on
this backend.
```

README.md gains a **Choosing a device** section stating the design directly:
using the GPU is an explicit choice, and the reasoning is a hidden fallback
makes performance impossible to reason about. The limitation stays listed under
"What does not" as well -- it is a real limit on what vkML can run whatever the
reasoning behind it, and a reader scanning limitations should not have to find
the rationale first.

### Decision 2 -- accepted. `best_device()`.

```python
device, why = vkml.best_device()
```

Never raises. Returns the reason rather than printing it. Prefers a discrete GPU
over an integrated one when both qualify, and names the one it picked.

Verified on this machine and on a simulated driver-less one
(`VK_ICD_FILENAMES` pointing at a file that does not exist):

```
using Vulkan device 0: AMD Radeon RX 5600M (RADV NAVI10) (discrete,
Vulkan 1.4.354, driver radv)

running on the CPU: the Vulkan loader could not create an instance
(VK_ERROR_INCOMPATIBLE_DRIVER); either no driver is installed, or none of the
installed drivers supports Vulkan 1.3. Call vkml.vulkan_device_reports() ...
```

Both examples now call it for `--device auto`, replacing a block copied into
each that said only "no Vulkan device found" -- so they report *which* GPU.

### The layering constraint held

4 predicted that `backend_for()` could not reach the Vulkan reason, and that is
how it was built: its message says only that a device must be initialised, which
is all layer 3 legitimately knows. The Vulkan-specific explanation is added by
`init_vulkan`'s own error (layer 4, where `unavailable_reason()` lives) and by
`best_device()` in Python. No registration-hint mechanism was added.

One thing 4 did not anticipate: `init_vulkan`'s message would have quoted the
result code TWICE, once directly and once inside `unavailable_reason()`. It now
reports whichever actually explains the failure. An empty probe reason is
reported distinctly, because it means a minimal instance works and the real one
failed on something vkML asked for -- validation layers being the usual cause.

### What is still true from 6

`V.device("vulkan:0")` still constructs without validating; the failure arrives
at first use, with the improved message. No existing call gained a fallback.

### Two places the proposal was wrong, corrected in the build

Sections 2 and 3 are the proposal and are left as written. Two details did not
survive contact:

**`best_device()` has no `prefer` parameter.** 3 sketched
`best_device(*, prefer="vulkan")`. `prefer="cpu"` would only ever mean "return
`V.cpu`", which callers can already write, so the parameter was speculative
(P6) and was dropped. The shipped signature is `best_device() -> (device, str)`.

**2's draft error told users to run `t.to(vkml.cpu)`, and that API does not
exist.** `Tensor::to()` takes a DType, not a Device -- checked, not assumed --
so there is no direct device-move method at all. Advice naming a function that
does not exist is worse than no advice, and it was caught only by trying it.

The shipped message names a route that was RUN first:

```python
vkml.tensor(t.numpy(), device=vkml.cpu)
```

**That gap is worth recording separately:** moving a tensor between devices has
no first-class API and goes through host memory via numpy. It is adequate for
the case this error covers -- an occasional unsupported operator -- and would be
poor as a general mechanism. Not filed as a task, because nothing needs it yet;
noted here so the next person to want `t.to(device)` knows it is absent by
omission rather than by decision.
