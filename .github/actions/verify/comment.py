#!/usr/bin/env python3
"""Prepare a cvconform PR comment body from a JSON report."""

import json
import os
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: comment.py <report.json> <os_name> [workspace]", file=sys.stderr)
        return 1

    report_path = sys.argv[1]
    os_name = sys.argv[2]
    workspace = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("GITHUB_WORKSPACE", ".")

    if not os.path.isfile(report_path):
        print("::warning::cvconform report not found; skipping comment")
        return 0

    with open(report_path, encoding="utf-8") as f:
        r = json.load(f)

    conformant = all(s == "conformant" for s in r.get("target_status", {}).values())
    badge = ":white_check_mark:" if conformant else ":x:"
    lines = [f"## cvconform conformance ({os_name}) {badge}"]

    for t, status in (r.get("target_status") or {}).items():
        score = (r.get("targets", {}).get(t) or {}).get("overall_score", 0.0)
        lines.append(f"- **{t}**: {score:.2f}% · {status}")

    if not conformant:
        lines.append("\n### Findings")
        for f in r.get("findings", []):
            backend = f.get("backend", "")
            cause = f.get("cause", "")
            confidence = f.get("confidence", "")
            lines.append(f'- `{backend}` · {cause} (conf {confidence})')

    body = "\n".join(lines)
    comment_path = os.path.join(workspace, "cvconform-pr-body.md")
    with open(comment_path, "w", encoding="utf-8") as out:
        out.write(body)

    print("PR comment body prepared in", comment_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
