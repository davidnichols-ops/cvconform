# Report Schema — the machine-readable CI/CD contract

`cvconform verify` produces a report. The human view is for people; the JSON
view (`--json report.json`, or `verify_to_json`) is the contract for CI/CD,
dashboards, and the Regression Memory corpus. It is stable and versioned.

## Top-level

```json
{
  "model": "yolo11.pt",
  "source": "pytorch",
  "reference": "pytorch",
  "dataset": "synthetic",
  "seed": 0,
  "input_shape": [1, 3, 224, 224],
  "environment": { "runtime": "pytorch", "runtime_version": "2.13.0", "...": "..." },
  "compilation": { "onnx": {"method": "export_onnx", "from": "pytorch"}, "coreml": { ... } },
  "targets": { "<target>": { "overall_score": 96.36, "is_conformant": true,
                             "scores": { "<output>": 96.4 }, "divergences": [] } },
  "findings": [ { "backend": "coreml", "affected": "...", "cause": "...",
                  "mechanism": "precision", "impact": "...", "confidence": 0.9,
                  "fixes": ["Force FP32 accumulation", "..."] } ],
  "research":   [ { "backend": "coreml", "cause": "...", "matched_issue": "...",
                    "recommendation": "...", "sources": [ {...} ] } ],
  "target_status": { "onnx": "conformant", "coreml": "degraded" }
}
```

## Key fields for CI gating

- **`target_status`** — per target: `conformant` | `degraded` | `error`.
  A CI gate should fail when any target is not `conformant`.
- **`targets.<t>.overall_score`** — 0..100 conformance score.
- **`targets.<t>.divergences`** — array of divergence evidence:
  `{output, kind, metric, observed, threshold, magnitude, first_divergence,
    mechanism, confidence}`.
- **`findings`** — root-cause evidence per failing target (`mechanism` is a
  stable id: `fusion`, `precision`, `quant_scale`, `fallback`, `padding`,
  `nms`, `output_shape`, `nan_inf`, `execution_failed`).
- **`research`** — AI-research summaries per finding.

## Mechanism ids (stable)

`fusion`, `precision`, `quant_scale`, `fallback`, `padding`, `nms`,
`output_shape`, `nan_inf`, `execution_failed`, `comparison_error`.

## Determinism

Same `model` + same `environment` (exact backend versions) + same `seed`
=> identical report (scores + divergences). Use this to build regression gates
that alert on *any* conformance change, not just pass/fail.

## Interpreting

- A target is **degraded** (not conformant) when `overall_score < 95` or it has
  divergences. The `findings`/`research` fields explain *why* with evidence and
  recommended fixes.
- `cvconform verify --require-conformant` exits non-zero when any target is
  non-conformant — drop-in for a CI gate.
