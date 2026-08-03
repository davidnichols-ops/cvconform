# Changelog

## [0.1.0] — 2026-08-02

First working vertical slice + the Phase 4-6 systems that make it a real
correctness layer, not a demo.

**Added**
- Vision IR (`cvconform.ir`): `VisionGraph`, `Operator`, `TensorSpec`, precision
  model, canonical operator registry, schema hashing.
- ONNX loader (`cvconform.loaders.onnx_loader`): ONNX -> VisionGraph lowering.
- Runtimes: onnxruntime / pytorch / coreml wrappers with exact version reporting
  and output normalization (CPU-focused, CoreML on Apple Silicon).
- Differential Execution Engine (`cvconform.engines.differential`): same seedable
  inputs to reference + targets; name-aware + positional output alignment.
- Comparison engine (`cvconform.engines.comparison`): per-kind metrics
  (tensor / boxes / classes / scores), tolerance policies, scored conformance.
- Root-Cause Analysis (`cvconform.engines.analyze`): Evidence objects with
  mechanism, confidence, and fixes.
- Fault Injection (`cvconform.faults`): 7 reproducible failure archetypes.
- Failure Discovery (`cvconform.discovery`): deterministic perturbation families.
- Automatic Failure Reduction (`cvconform.reduce`): spatial + value minimization.
- Regression Memory (`cvconform.corpus`): versioned, backend-keyed manifest store.
- AI Research Agent (`cvconform.research`): evidence -> research summary,
  offline fallback.
- Report renderer + machine-readable JSON schema (`docs/report-schema.md`).
- CLI: `verify` (with `--json`, `--corpus`, `--require-conformant`) and
  `discover`.
- Python API: `from cvconform import verify`.
- Docs: `README`, `docs/report-schema.md`, CI workflow.

**Measured on the demo (M4, seed=0):** ONNX 100.00% conformant, CoreML 96.36%.

**Tests:** 36 passing (unit, IR, comparison, discovery/reduce, fault injection,
research/corpus, integration/CI-gate).

### Infrastructure / notable lessons
- Project venv pinned to Python 3.12 — native wheels for torch/coremltools/onnx
  (the system Python 3.14 lacks them).
- coremltools 9: no standalone ONNX converter, `predict()` takes no
  `useCPUOnly`; models must be traced (not scripted) with static reshapes to
  convert cleanly.
- torch 2.13 legacy exporter (`dynamo=False`) required to export ScriptModules
  and to name outputs for clean comparison.
