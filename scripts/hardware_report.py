#!/usr/bin/env python3
"""Describe this machine's Vulkan devices, and optionally validate them.

WHY THIS EXISTS
---------------
vkML claims to run on any Vulkan 1.3 device. That claim is currently supported
by two GPUs and two drivers on one operating system — everything else is
untested, and no amount of local work changes that. The only way to widen it is
for people on other hardware to run something and send back the result.

So this is the thing they run. One command, no build knowledge required, and
output that is complete enough to diagnose from without a follow-up question.

WHAT IT REPORTS, AND WHY IT CANNOT SIMPLY CREATE A DEVICE
---------------------------------------------------------
The backend requires exactly three features — bufferDeviceAddress,
scalarBlockLayout and timelineSemaphore — and refuses a device that lacks any of
them. Everything else it detects and adapts to.

That makes the interesting report the one from a device vkML *rejects*, and it
is precisely the report the ordinary capability query cannot produce: obtaining
DeviceCapabilities goes through backend creation, which throws on such a device.
`vulkan_device_reports()` exists for this reason — it queries physical devices
only, so it answers for hardware the library cannot use, for a machine with no
GPU, and for a build compiled without Vulkan at all.

Usage:
    python scripts/hardware_report.py              # describe this machine
    python scripts/hardware_report.py --run-tests  # ... and run the suite on it
    python scripts/hardware_report.py --json       # machine-readable

Paste the output into an issue at https://github.com/jeel00dev/vkml/issues.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

try:
    import vkml as V
except ImportError as exc:  # pragma: no cover - depends on how vkml was installed
    print(f"cannot import vkml: {exc}\n", file=sys.stderr)
    print("Build it first (see README), or `pip install .`", file=sys.stderr)
    raise SystemExit(2) from exc

# Loader variables that change which driver answers. Worth reporting: a
# surprising result is sometimes just a driver override left in the environment.
LOADER_VARS = (
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "VK_LAYER_PATH",
    "VK_INSTANCE_LAYERS",
    "VK_LOADER_DRIVERS_SELECT",
    "VKML_TEST_DEVICE",
)


def environment() -> dict:
    """Everything about the machine that is not a Vulkan device."""
    return {
        "vkml_version": getattr(V, "__version__", "unknown"),
        "vulkan_compiled_in": bool(V.has_vulkan),
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "loader_env": {k: os.environ[k] for k in LOADER_VARS if k in os.environ},
    }


def gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def print_environment(env: dict) -> None:
    print("=" * 72)
    print("vkML hardware report")
    print("=" * 72)
    print(f"  vkml ............ {env['vkml_version']}")
    print(f"  vulkan built in . {env['vulkan_compiled_in']}")
    print(f"  python .......... {env['python']}")
    print(f"  platform ........ {env['platform']}")
    print(f"  machine ......... {env['machine']}")
    for key, value in env["loader_env"].items():
        print(f"  {key} = {value}")


def print_device(index: int, d: dict) -> None:
    verdict = "USABLE" if d["supported"] else f"REJECTED ({d['missing_requirement']})"
    print()
    print("-" * 72)
    print(f"device {index}: {d['name']}   [{verdict}]")
    print("-" * 72)
    print(f"  driver .............. {d['driver_name']} (version {d['driver_version']})")
    print(f"  type ................ {d['device_type']}")
    print(f"  vulkan .............. {d['api_version']}")
    print(f"  ids ................. vendor 0x{d['vendor_id']:04x} device 0x{d['device_id']:04x}")

    # The three that decide whether vkML runs at all, always shown together and
    # always first: on a rejected device this is the whole answer.
    print("  required:")
    for key in ("buffer_device_address", "scalar_block_layout", "timeline_semaphore"):
        print(f"    {'ok  ' if d[key] else 'MISSING'} {key}")

    print("  optional:")
    for key in (
        "shader_float16",
        "shader_int8",
        "shader_int16",
        "storage_buffer_16bit",
        "synchronization2",
        "subgroup_size_control",
        "global_float_atomic_add",
        "shared_float_atomic_add",
        "cooperative_matrix",
    ):
        print(f"    {'yes' if d[key] else 'no '} {key}")

    cores = d["shader_core_count"]
    print("  limits:")
    print(
        f"    subgroup .......... {d['subgroup_size']} "
        f"(controllable {d['min_subgroup_size']}..{d['max_subgroup_size']})"
    )
    print(f"    workgroup ......... {d['max_workgroup_invocations']} invocations")
    print(f"    shared memory ..... {d['max_shared_memory'] // 1024} KiB")
    print(f"    push constants .... {d['max_push_constants']} bytes")
    print(f"    max allocation .... {gib(d['max_allocation_size'])}")
    print(f"    device-local ...... {gib(d['device_local_bytes'])}")
    print(f"    host-visible ...... {gib(d['host_visible_device_local_bytes'])}")
    print(f"    compute units ..... {cores if cores else 'not reported'}")


def unusable_reason(devices: list, no_device_detail: str = "") -> str | None:
    """Why this machine cannot run vkML, or None if it can.

    Pure: `no_device_detail` is passed in rather than queried, so the decision
    can be tested against device lists no machine here can produce.

    Exists for CI. A job that finds no device and then reports success is
    indistinguishable from one that proved the backend works, and the second is
    the reading people take -- the suite passes either way, because the Vulkan
    tests skip themselves when no device is present.
    """
    if not devices:
        return no_device_detail or "no Vulkan device is visible"
    rejected = [
        f"{d['name']} (missing {d['missing_requirement']})" for d in devices if not d["supported"]
    ]
    if len(rejected) == len(devices):
        return "no visible device meets the requirements: " + "; ".join(rejected)
    return None


def run_suite(index: int) -> dict:
    """Runs the validation suite against one device, in a child process.

    A subprocess because the device is selected by environment variable and the
    backend is created once per process; there is no supported way to switch
    devices inside a running interpreter.
    """
    tests = ROOT / "tests" / "python"
    if not tests.is_dir():
        # Someone who pip-installed vkML and downloaded only this file. The
        # description above is still worth having, so say what is missing
        # rather than letting pytest fail with a path error.
        print(f"\ncannot find the suite at {tests} -- run this from a clone of the")
        print("repository if you want --run-tests. The report above is unaffected.")
        return {"device": index, "returncode": 0, "summary": "suite not present"}

    env = dict(os.environ, VKML_TEST_DEVICE=str(index))
    print(f"\nrunning the validation suite on device {index} ...", flush=True)

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests), "-q"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    # The last non-empty line is pytest's summary ("N passed, M skipped in Xs").
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    summary = lines[-1] if lines else "no output"
    print(f"  {summary}")
    if completed.returncode != 0:
        print("\n--- failures ------------------------------------------------")
        print(completed.stdout[-4000:])
        print(completed.stderr[-2000:])
    return {"device": index, "returncode": completed.returncode, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="also run the validation suite against every usable device",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--require-device",
        action="store_true",
        help="exit non-zero unless at least one visible device can run vkML",
    )
    args = parser.parse_args()

    env = environment()
    devices = list(V.vulkan_device_reports())

    if not args.json:
        print_environment(env)
        if not devices:
            print("\n  no Vulkan devices found.")
            print(f"  why: {V.vulkan_unavailable_reason()}")
        for index, device in enumerate(devices):
            print_device(index, device)

    results = []
    if args.run_tests:
        usable = [i for i, d in enumerate(devices) if d["supported"]]
        if not usable:
            print("\nno usable device; the suite would only exercise the CPU backend.")
        for index in usable:
            results.append(run_suite(index))

    if args.json:
        json.dump(
            {"environment": env, "devices": devices, "suite": results},
            sys.stdout,
            indent=2,
        )
        print()
    else:
        print()
        print("=" * 72)
        print("Please paste everything above into an issue:")
        print("  https://github.com/jeel00dev/vkml/issues")
        print("=" * 72)

    if args.require_device:
        reason = unusable_reason(devices, V.vulkan_unavailable_reason())
        if reason:
            print(f"\nrequired a usable device: {reason}", file=sys.stderr)
            return 2

    # A failing suite is a failing report. Describing the machine is not.
    return 1 if any(r["returncode"] != 0 for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
