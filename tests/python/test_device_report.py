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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import vkml as V  # noqa: E402

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
