"""The device report: what a user on unknown hardware sends back.

This surface exists to be run on machines the project has never seen, so its
contract is unusually strict: it must produce an answer on a device the backend
cannot use, on a machine with no GPU, and on a build with no Vulkan at all.
Every one of those is a case where the ordinary capability query throws.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import vkml as V  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Keys every report carries. Pinned as a set rather than checked one by one so
# that a field disappearing from the binding fails here rather than silently
# producing a thinner report from someone else's machine, which is unrepeatable.
REQUIRED_KEYS = {
    "name",
    "driver_name",
    "device_type",
    "api_version",
    "driver_version",
    "vendor_id",
    "device_id",
    "supported",
    "missing_requirement",
    "buffer_device_address",
    "scalar_block_layout",
    "timeline_semaphore",
    "synchronization2",
    "subgroup_size_control",
    "shader_float16",
    "shader_int8",
    "shader_int16",
    "storage_buffer_16bit",
    "global_float_atomic_add",
    "shared_float_atomic_add",
    "cooperative_matrix",
    "subgroup_size",
    "min_subgroup_size",
    "max_subgroup_size",
    "max_workgroup_invocations",
    "max_workgroup_count_x",
    "shader_core_count",
    "max_shared_memory",
    "max_push_constants",
    "max_allocation_size",
    "device_local_bytes",
    "host_visible_device_local_bytes",
    "timestamp_period",
}


def test_reports_exist_on_any_build():
    """Never raises: a CPU-only build reports [], it does not fail."""
    reports = V.vulkan_device_reports()
    assert isinstance(reports, list)
    if not V.has_vulkan:
        assert reports == []


def test_device_names_exist_on_any_build():
    """The README's post-install check must answer on a CPU-only build too.

    `python -c "import vkml; print(vkml.vulkan_device_names())"` is Step 3 of the
    installation instructions, and the README documents a CPU-only install two
    sections earlier. Binding this name only when `has_vulkan` made that command
    raise AttributeError, which reads as a broken install rather than as a
    machine with no GPU (issue #9). An empty list is the honest answer.
    """
    names = V.vulkan_device_names()
    assert isinstance(names, list)
    assert len(names) == V.vulkan_device_count()
    if not V.has_vulkan:
        assert names == []


def test_the_observability_surface_exists_on_any_build():
    """Configuration and Decision are layer-0 facts, so a CPU-only build has them.

    They were bound inside `#ifdef VKML_HAS_VULKAN` while
    `python/vkml/__init__.py` exported all five unconditionally with a comment
    saying why -- so `import vkml` on a CPU-only build died at
    `_C.configuration` before it could report anything at all. The comment was
    right and the C++ did not match it, which is the failure this names.

    A test rather than a bare gate because the reason is a rule --
    observability is not a Vulkan feature -- and a rule is worth stating where
    someone will read it.
    """
    for name in ("configuration", "record_decisions", "stop_recording_decisions",
                 "decisions", "decisions_published"):
        assert hasattr(V, name), (
            f"vkml.{name} is missing. It observes facts published from layer 0 and must "
            f"not be compiled out with the Vulkan backend")

    V.record_decisions(4)
    try:
        assert isinstance(V.decisions(), list)
        assert isinstance(V.decisions_published(), int)
        assert isinstance(V.configuration(), list)
    finally:
        V.stop_recording_decisions()


def test_report_carries_every_field():
    for report in V.vulkan_device_reports():
        assert set(report) == REQUIRED_KEYS


def test_one_report_per_device():
    assert len(V.vulkan_device_reports()) == V.vulkan_device_count()


def test_supported_agrees_with_missing_requirement():
    """The two can never disagree -- `supported` is derived from the reason."""
    for report in V.vulkan_device_reports():
        assert report["supported"] == (report["missing_requirement"] == "")
        if report["supported"]:
            assert report["buffer_device_address"]
            assert report["scalar_block_layout"]
            assert report["timeline_semaphore"]
        else:
            assert report["missing_requirement"] in {
                "bufferDeviceAddress",
                "scalarBlockLayout",
                "timelineSemaphore",
            }


def test_identity_fields_are_populated():
    for report in V.vulkan_device_reports():
        assert report["name"]
        assert report["api_version"].count(".") == 2
        assert report["device_type"] in {
            "discrete",
            "integrated",
            "virtual",
            "cpu",
            "other",
        }
        assert report["max_workgroup_invocations"] > 0


def test_unavailable_reason_is_empty_exactly_when_a_device_exists():
    reason = V.vulkan_unavailable_reason()
    assert bool(reason) == (len(V.vulkan_device_reports()) == 0)


def test_unavailable_reason_names_the_loader_failure():
    """A machine with no usable driver must say WHY, not just "none".

    Driven with the loader pointed at a file that does not exist, because the
    two causes of zero devices -- no instance, versus an instance that
    enumerates nothing -- need different advice and previously looked identical.
    """
    source = """
import sys
sys.path.insert(0, sys.argv[1])
import vkml as V
assert V.vulkan_device_reports() == [], "expected no devices with a bogus ICD"
print(V.vulkan_unavailable_reason())
"""
    python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
    result = subprocess.run(
        [sys.executable, "-c", source, python_dir],
        env=dict(os.environ, VK_ICD_FILENAMES="/nonexistent-vkml-test.json"),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # A build without Vulkan, or a loader that ignores the override.
        return
    assert result.stdout.strip(), "no devices and no explanation is the old behaviour"


def test_unusable_reason_distinguishes_the_three_outcomes():
    """The CI gate's logic, on inputs no machine here can produce.

    A rejected device cannot be created locally, and "no device at all" is the
    state that already went green once while proving nothing -- so both are
    tested against synthesised reports rather than hardware.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    from hardware_report import unusable_reason  # noqa: PLC0415

    usable = {"name": "gpu", "supported": True, "missing_requirement": ""}
    rejected = {
        "name": "old gpu",
        "supported": False,
        "missing_requirement": "bufferDeviceAddress",
    }

    assert unusable_reason([]) == "no Vulkan device is visible"
    assert unusable_reason([usable]) is None
    assert "bufferDeviceAddress" in unusable_reason([rejected])
    # One usable device is enough, even beside a rejected one: that is a
    # working machine, not a failure.
    assert unusable_reason([rejected, usable]) is None


def test_require_device_catches_a_backend_that_will_not_start():
    """Seeing a device is not the same as being able to use it.

    The report runs on a bare probe instance; the real backend asks for more,
    validation layers among them. When those disagree -- a broken layer install
    is enough -- the gate passed, every Vulkan test then skipped for want of a
    backend, and the job went green having exercised nothing. That happened on
    macOS, so the gate now starts the backend too.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    from hardware_report import backend_start_error  # noqa: PLC0415

    if V.vulkan_device_count() > 0:
        assert backend_start_error(0) is None

    # An index no device occupies stands in for a backend that refuses to
    # start: the real cause needs a broken driver install, which cannot be
    # staged here, but the path from exception to message is the same one.
    refused = backend_start_error(V.vulkan_device_count() + 50)
    assert refused and "will not start" in refused


def test_reporting_creates_no_backend():
    """The load-bearing property, checked in a clean process.

    `available_devices()` lists backends that have actually been created. If
    reporting created one, a report on unusable hardware would throw during the
    very call meant to explain why the hardware is unusable.

    A subprocess because any earlier test in this session may already have
    initialised Vulkan, which would make the assertion vacuous.
    """
    source = """
import sys
sys.path.insert(0, sys.argv[1])
import vkml as V
before = [str(d) for d in V.available_devices()]
count = len(V.vulkan_device_reports())
after = [str(d) for d in V.available_devices()]
assert before == after, f"reporting registered a backend: {before} -> {after}"
print(count)
"""
    python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
    result = subprocess.run(
        [sys.executable, "-c", source, python_dir],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) == V.vulkan_device_count()


# ---------------------------------------------------------------------------
# best_device()
#
# The ONE path that falls back to the CPU, and it always says why. A device the
# caller NAMES is never quietly downgraded.
# docs/adr/0008-backend-selection-and-cpu-fallback.md.
# ---------------------------------------------------------------------------

# A driver manifest that does not exist. The Vulkan loader reads
# VK_ICD_FILENAMES to find drivers, so pointing it at nothing is how a machine
# with no GPU is simulated without needing one.
_NO_DRIVER = {"VK_ICD_FILENAMES": "/nonexistent/vkml-test-no-such-icd.json"}


def _run(snippet, env=None):
    """Runs `snippet` in a fresh interpreter, returning stdout."""
    full_env = {**os.environ, **(env or {})}
    out = subprocess.run([sys.executable, "-c", snippet], cwd=REPO, env=full_env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


_BEST_DEVICE = (
    "import sys;sys.path.insert(0,'python');import vkml as V;"
    "V.set_log_level(V.LogLevel.ERROR);"
    "d,why=V.best_device();print(repr(str(d)));print(repr(why))"
)


def test_best_device_returns_a_device_and_a_reason():
    lines = _run(_BEST_DEVICE).strip().splitlines()
    name, why = eval(lines[-2]), eval(lines[-1])  # noqa: S307 - our own repr
    assert name in ("cpu",) or name.startswith("vulkan:"), name
    assert why, "best_device must always explain its choice"
    # Whichever it picked, the reason must name it.
    assert ("CPU" in why) if name == "cpu" else ("Vulkan device" in why), why


def test_best_device_falls_back_to_cpu_with_an_actionable_reason():
    """No usable GPU: the CPU, and a reason a user can act on.

    This is the case the whole ADR is about. It must not raise, must not be
    silent, and must say something better than a bare Vulkan enum.

    THERE ARE TWO WAYS TO HAVE NO GPU and they need different advice, which is
    why this branches instead of asserting one sentence. A Vulkan-enabled build
    with no driver should talk about drivers; a build compiled without the
    backend at all should talk about the build flag, because telling that user
    to check their driver would send them after the wrong thing.

    The first version asserted only the driver wording and passed here while
    failing on three CI jobs, which build CPU-only -- the presets do not set
    VKML_VULKAN. Reproduced locally afterwards by doing the same.
    """
    lines = _run(_BEST_DEVICE, _NO_DRIVER).strip().splitlines()
    name, why = eval(lines[-2]), eval(lines[-1])  # noqa: S307
    assert name == "cpu", f"expected the CPU with no usable GPU, got {name}"
    assert "running on the CPU" in why, why

    if V.has_vulkan:
        # Vulkan compiled in, but the loader was pointed at nothing.
        assert "driver" in why.lower(), why
        assert "vulkan_device_reports" in why, why
    else:
        # No backend in this build at all. The remedy is a rebuild, not a driver.
        assert "VKML_VULKAN" in why, why
        assert "reinstall" in why.lower(), why


@pytest.mark.skipif(not V.has_vulkan,
                    reason="init_vulkan does not exist in a build without the Vulkan backend")
def test_a_named_device_is_never_silently_downgraded():
    """`init_vulkan` must RAISE when the device asked for is unusable.

    The rule best_device() exists to preserve: someone who names a device wants
    that device, and quietly handing back the CPU hides what they asked about.

    Skipped, not adapted, when the backend is absent: `init_vulkan` is only
    bound when `has_vulkan` (see vkml/__init__.py), so there is no named device
    to downgrade and the property is vacuous rather than violated.
    """
    snippet = (
        "import sys;sys.path.insert(0,'python');import vkml as V;"
        "V.set_log_level(V.LogLevel.ERROR);"
        "\ntry:\n"
        "    V.init_vulkan(0)\n"
        "    print('NO_ERROR')\n"
        "except V.DeviceError as e:\n"
        "    print(repr(str(e)))\n"
    )
    out = _run(snippet, _NO_DRIVER).strip().splitlines()[-1]
    assert out != "NO_ERROR", "init_vulkan silently succeeded with no driver"
    message = eval(out)  # noqa: S307
    assert "does not fall back" in message, message
    assert "best_device" in message, message


def test_unsupported_op_error_states_there_is_no_fallback():
    """`prod` is CPU-only. The error must say vkML will not silently move the
    work, and name what to do instead -- an all-or-nothing backend is only a
    good design if it explains itself."""
    if not (V.has_vulkan and V.vulkan_available()):
        pytest.skip("no Vulkan device present")
    snippet = (
        "import sys;sys.path.insert(0,'python');import numpy as np,vkml as V;"
        "V.set_log_level(V.LogLevel.ERROR);V.init_vulkan(0);"
        "t=V.tensor(np.ones((4,),dtype=np.float32),device=V.device('vulkan:0'))\n"
        "try:\n"
        "    V.prod(t).numpy()\n"
        "    print('NO_ERROR')\n"
        "except V.NotImplementedError_ as e:\n"
        "    print(repr(str(e)))\n"
    )
    out = _run(snippet, {"VKML_VULKAN_VALIDATION": "0"}).strip().splitlines()[-1]
    assert out != "NO_ERROR", "prod unexpectedly ran on Vulkan"
    message = eval(out)  # noqa: S307
    assert "does not fall back" in message, message
    # Names a REMEDY the reader can run, not just a direction to go in. The
    # first version of this message said "move it to the CPU" and gave no API --
    # and the ADR's draft suggested `t.to(vkml.cpu)`, which does not exist.
    assert "vkml.tensor(" in message, message
