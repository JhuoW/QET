#!/usr/bin/env python3
"""PreToolUse hook (Edit|Write|NotebookEdit): block edits to human-owned and read-only paths.

Override for a maintenance session started by the human: QET_ALLOW_PROTECTED_EDITS=1 claude
"""
import json
import os
import sys

PROTECTED_PREFIXES = ("reference/", "third_party/", "results/raw/", ".claude/hooks/")
PROTECTED_FILES = ("docs/SPEC.md", "docs/EVAL_PLAN.md", ".claude/settings.json")


def main() -> int:
    if os.environ.get("QET_ALLOW_PROTECTED_EDITS") == "1":
        return 0
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    ti = data.get("tool_input") or {}
    path = ti.get("file_path") or ti.get("notebook_path") or ""
    if not path:
        return 0
    root = os.path.realpath(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd())
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
    if not (full == root or full.startswith(root + os.sep)):
        return 0
    rel = os.path.relpath(full, root).replace(os.sep, "/")
    if rel in PROTECTED_FILES or rel.startswith(PROTECTED_PREFIXES):
        print(f"Blocked: {rel} is protected (AGENTS.md, 'Sources of truth'). Propose the change in "
              f"docs/DECISIONS.md and ask the human; do not work around this block.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
