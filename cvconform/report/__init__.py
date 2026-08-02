"""Report package — human + machine-readable conformance reports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def render_human(report: Dict[str, Any]) -> str:
    """Render the machine-readable report dict as the human terminal report."""
    L = []
    L.append("=" * 62)
    L.append("VISION CONFORM REPORT")
    L.append("=" * 62)
    L.append(f"Model:       {report.get('model', 'unnamed')}")
    L.append(f"Reference:   {report.get('reference', '')}")
    L.append(f"Dataset:     {report.get('dataset', 'synthetic-seeded')}  (seed={report.get('seed', 0)})")
    L.append(f"Environment: {report.get('environment', '')}")
    L.append("")

    targets = report.get("targets", {})
    L.append("TARGETS")
    L.append("-" * 62)
    status_by_t = report.get("target_status", {})
    for tname, tres in targets.items():
        conf = status_by_t.get(tname, "?")
        L.append(f"  {'PASS' if conf == 'conformant' else 'FAIL' if conf == 'degraded' else 'ERROR':<6} "
                 f"{tname}")
    L.append("")

    L.append("CONFORMANCE SCORES")
    L.append("-" * 62)
    for tname, tres in targets.items():
        overall = tres.get("overall_score", 0.0)
        L.append(f"  {tname:<14}{overall:>8.2f}%")
    L.append("")

    # Findings
    findings = report.get("findings", [])
    if findings:
        L.append("FINDINGS")
        L.append("-" * 62)
        for f in findings:
            L.append(f"  Backend:     {f.get('backend')}")
            L.append(f"  Affected:    {f.get('affected')}")
            L.append(f"  Cause:       {f.get('cause')}")
            L.append(f"  Mechanism:   {f.get('mechanism')}")
            L.append(f"  Impact:      {f.get('impact')}")
            L.append(f"  Confidence:  {f.get('confidence')}")
            fixes = f.get("fixes") or []
            if fixes:
                L.append("  Recommended fixes:")
                for i, fix in enumerate(fixes, 1):
                    L.append(f"    {i}. {fix}")
            L.append("")
    else:
        L.append("FINDINGS: none — all targets conformant within policy.")
    return "\n".join(L)
