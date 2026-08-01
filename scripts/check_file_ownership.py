"""Fail if anything in the working tree is owned by another user.

WHY THIS EXISTS. The CI container bind-mounts the workspace, and while it ran as
root every artifact it wrote -- `build/`, `__pycache__/`, CMake's caches -- came
back owned by root. 551 such files accumulated before anyone noticed, because
nothing goes wrong until a host-side command tries to write one of them.

When it does go wrong, the error does not describe the problem. The first
symptom was `pip install -e .` failing with

    CMake Error: Could not open file for write in copy operation
                 build/wheel/CMakeFiles/4.2.2/CMakeSystem.cmake.tmp
    CMake Error: : System Error: No such file or directory

which reports a permission failure as a missing file and never mentions
ownership, root, or the container. Recovering from that costs more than the
check, and the check is a stat.

Run at the end of a containerised job it proves the `--user` mapping held; run
locally it names the files to clean and the command to do it.

Not a git-tracked-files check: everything this has ever caught was untracked
build output, which is exactly what `git status` will not show you.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories with nothing to protect and a lot to walk.
SKIP = {".git", ".venv", "node_modules", "_site"}


def main() -> int:
    me = os.getuid()
    foreign: list[tuple[Path, int]] = []

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            try:
                uid = p.lstat().st_uid
            except OSError:
                continue          # vanished mid-walk; not our problem
            if uid != me:
                foreign.append((p, uid))

    if not foreign:
        print(f"  every file belongs to uid {me}")
        return 0

    by_uid: dict[int, int] = {}
    for _, uid in foreign:
        by_uid[uid] = by_uid.get(uid, 0) + 1

    print(f"  {len(foreign)} files are not owned by uid {me}:")
    for uid, count in sorted(by_uid.items()):
        who = "root" if uid == 0 else f"uid {uid}"
        print(f"    {count:5} owned by {who}")
    print("\n  a sample:")
    for p, uid in foreign[:5]:
        print(f"    {p.relative_to(ROOT)}  (uid {uid})")

    print("\n  These are almost certainly artifacts of a container that ran as root.")
    print("  The container now runs with `--user`, so this should not recur; clean")
    print("  up what is already there with:")
    print(f"\n      sudo chown -R {me}:{os.getgid()} {ROOT}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
