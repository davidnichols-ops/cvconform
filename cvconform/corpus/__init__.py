"""Regression Memory — the structured corpus where cvconform 'gets smarter forever'.

Every discovered failure (from Fault Injection, Failure Discovery, or a user's
real deployment) is stamped into a versioned, machine-readable corpus entry.
The corpus is the product's moat: each entry is a permanent, re-runnable test
that pins the environment + exact versions so future runs can detect regressions
(reappearance, worsening, or fixes across runtime releases).

Layout:
    corpus/
      models/<hash>/            # clean reference + target artifacts
      inputs/<hash>/            # repro inputs + generator recipe
      failures/<backend>/<slug>/  # evidence + manifest + verdict
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from cvconform.engines.comparison import ComparisonResult, Divergence


def stable_id(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def materialize_manifest(clean_state: Dict[str, Any], fault: str,
                         backend: str, versions: Dict[str, str]) -> Dict[str, Any]:
    """Build the JSON manifest (evidence packet) for a corpus entry."""
    return {
        "id": stable_id(f"{backend}-{fault}-{versions.get('runtime','?')}-{_utc_ts()}"),
        "created": _now(),
        "kind": "failure",
        "backend": backend,
        "fault": fault,
        "signal": clean_state,
        "environment": versions,
    }


def _utc_ts() -> str:
    return str(int(datetime.now(timezone.utc).timestamp()))


class ConformanceCorpus:
    """Store + query conformance regression entries."""

    def __init__(self, root: str = "corpus"):
        self.root = root
        self.failures_dir = os.path.join(root, "failures")

    # -- path helpers ----------------------------------------------------
    def failure_path(self, backend: str, slug: str) -> str:
        return os.path.join(self.failures_dir, backend, slug)

    # -- write -----------------------------------------------------------
    def record_failure(self, backend: str, slug: str, entry: Dict[str, Any]) -> str:
        """Persist a failure entry + its manifest. Returns the entry path."""
        d = self.failure_path(backend, slug)
        os.makedirs(d, exist_ok=True)
        manifest_path = os.path.join(d, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(entry, f, indent=2, default=_json_default)
        return d

    # -- read ------------------------------------------------------------
    def list_failures(self, backend: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = []
        if backend:
            # single backend dir: failures/<backend>/<slug>/manifest.json
            base = os.path.join(self.failures_dir, backend)
            if os.path.isdir(base):
                for slug in sorted(os.listdir(base)):
                    mp = os.path.join(base, slug, "manifest.json")
                    if os.path.exists(mp):
                        with open(mp) as f:
                            entries.append(json.load(f))
        else:
            # all backends: failures/<backend>/<slug>/manifest.json
            if os.path.isdir(self.failures_dir):
                for be in sorted(os.listdir(self.failures_dir)):
                    be_dir = os.path.join(self.failures_dir, be)
                    if not os.path.isdir(be_dir):
                        continue
                    for slug in sorted(os.listdir(be_dir)):
                        mp = os.path.join(be_dir, slug, "manifest.json")
                        if os.path.exists(mp):
                            with open(mp) as f:
                                entries.append(json.load(f))
        return entries

    def summarize(self) -> Dict[str, Any]:
        by_backend: Dict[str, int] = {}
        total = 0
        for e in self.list_failures():
            by_backend[e.get("backend", "?")] = by_backend.get(e.get("backend", "?"), 0) + 1
            total += 1
        return {"total_entries": total, "by_backend": by_backend}


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)
