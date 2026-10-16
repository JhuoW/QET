#!/usr/bin/env python3
"""PreToolUse hook (Bash): a speed bump against modifying protected paths through the shell.

Heuristic by design (permission rules and code review are the real protection). It blocks shell segments that
name a protected path together with a writing command, and fixture regeneration without --check.
Override for a maintenance session started by the human: QET_ALLOW_PROTECTED_EDITS=1 claude
"""
import json
import os
import re
import sys

PROT = r"(reference/|third_party/|results/raw/|docs/SPEC\.md|docs/EVAL_PLAN\.md|\.claude/settings\.json|\.claude/hooks/)"
WRITERS = [
    r"(^|[\s(])(cp|mv|rm|ln|truncate|chmod|chown|install|rsync|dd|touch)\s",
    r"\bsed\b[^\n]*\s-[a-zA-Z]*i",
    r"\bperl\b[^\n]*\s-[a-zA-Z]*i",
    r"\bgit\s+(checkout|restore|rm|mv|apply|stash)\b",
    r"\btee\b",
    r">>?\s*['\"]?[^\s'\"]*" + PROT,
    r"open\([^)]*" + PROT + r"[^)]*,\s*['\"][wax]",
    r"(write_text|write_bytes|shutil\.|os\.remove|os\.unlink|\.unlink\(|os\.rename|\.rename\()",
]


def main() -> int:
    if os.environ.get("QET_ALLOW_PROTECTED_EDITS") == "1":
        return 0
    try:
        cmd = (json.load(sys.stdin).get("tool_input") or {}).get("command", "")
    except json.JSONDecodeError:
        return 0
    if re.search(r"make_fixtures\.py", cmd) and "--check" not in cmd:
        print("Blocked: only the human regenerates fixtures. Use `python reference/make_fixtures.py --check` "
              "to verify them.", file=sys.stderr)
        return 2
    for seg in re.split(r"\|\||&&|;|\||\n", cmd):
        if re.search(PROT, seg) and any(re.search(w, seg) for w in WRITERS):
            print("Blocked: this command appears to modify a protected path (reference/, third_party/, "
                  "results/raw/, docs/SPEC.md, docs/EVAL_PLAN.md, .claude config). Use the Read tool to inspect "
                  "files there; ask the human for changes.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
