#!/usr/bin/env python3
"""Stop hook: when Python files have uncommitted changes, run `make test-fast` before Claude stops.

Blocks at most once per stop attempt (honours stop_hook_active), so Claude can still stop after explaining.
"""
import json
import os
import subprocess
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if data.get("stop_hook_active"):
        return 0
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    if not (os.path.isfile(os.path.join(root, "Makefile")) and os.path.isdir(os.path.join(root, "src", "qet"))):
        return 0
    try:
        st = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=root,
                            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if st.returncode != 0 or not st.stdout.strip():
        return 0
    try:
        r = subprocess.run(["make", "test-fast"], cwd=root, capture_output=True, text=True, timeout=280)
        ok, out = r.returncode == 0, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        ok, out = False, "make test-fast timed out after 280 s (it should take under 2 minutes)."
    except OSError:  # make is missing: nothing to check
        return 0
    if ok:
        return 0
    tail = "\n".join(out.strip().splitlines()[-30:])
    print(json.dumps({"decision": "block",
                      "reason": "make test-fast fails with uncommitted Python changes. Fix the failures before "
                                "ending the turn, or state plainly why they cannot be fixed now.\n\n" + tail}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
