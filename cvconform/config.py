"""Configuration — read/write `.cvconform.yaml`.

Config is an *optimization*, never a prerequisite. Everything here works with
no config file; when one exists it can tune the detected contract, reference,
targets, calibration, and policy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml

CONFIG_FILENAMES = [".cvconform.yaml", ".cvconform.yml", "cvconform.yaml"]


@dataclass
class CVConformConfig:
    version: int = 1
    model: Dict[str, Any] = field(default_factory=dict)
    reference: Optional[str] = None
    targets: Optional[List[str]] = None
    calibration: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    policy: Dict[str, float] = field(default_factory=dict)
    ci: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"version": self.version}
        for k, v in asdict(self).items():
            if k == "version":
                continue
            if v is not None and v != {} and v != []:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CVConformConfig":
        c = cls()
        c.version = d.get("version", 1)
        c.model = d.get("model", {})
        c.reference = d.get("reference")
        c.targets = d.get("targets")
        c.calibration = d.get("calibration", {})
        c.seed = d.get("seed", 0)
        c.policy = d.get("policy", {})
        c.ci = d.get("ci", {})
        return c


def find_config(start: str = ".") -> Optional[str]:
    """Search up the tree for a .cvconform config file."""
    cur = os.path.abspath(start)
    while True:
        for name in CONFIG_FILENAMES:
            p = os.path.join(cur, name)
            if os.path.exists(p):
                return p
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def load_config(path: Optional[str] = None) -> Optional[CVConformConfig]:
    if path is None:
        path = find_config()
    if path is None:
        return None
    try:
        with open(path) as f:
            return CVConformConfig.from_dict(yaml.safe_load(f) or {})
    except Exception:
        return None


def write_config(path: str, cfg: CVConformConfig) -> str:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    return path
