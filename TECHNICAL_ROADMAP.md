# TECHNICAL_ROADMAP — cvconform

Milestones. Each ends with something concrete, testable, and evidence-backed.

## M0 — Foundations (Phase 2, ≈ done)
- [x] Environment recon (CURRENT_STATE.md)
- [x] VISION / ARCHITECTURE / PRODUCT_REQUIREMENTS / this roadmap
- [x] Project skeleton, Python 3.12 venv, git, conventional commits.

## M1 — Vertical slice (Phase 3) — "real conformance report, end to end"
Goal: a real model, 2-3 real runtimes, a real report.
- **IR**: `VisionGraph`, `Operator`, `TensorSpec`, precision model, canonical op
  registry + ONNX loader lowering to IR. (Unit-tested.)
- **Runtimes**: `run_onnx` (onnxruntime, real), `run_pytorch` (torchscript/torch,
  real), `run_coreml` (coremltools, real). Each: version-reporting wrapper +
  output normalization -> normalized output schema.
- **Differential engine**: seedable input generation; comparison for tensor /
  boxes / classes / scores; tolerance policy; per-target score.
- **Report**: human text + JSON.
- **Slice demo**: a tiny YOLO-style detector (conv backbone + detection head, and
  a real NMS) exported to ONNX and CoreML, verified on a handful of seeded
  synthetic inputs across pytorch + onnx + coreml.
- **Exit criteria:** `cvconform verify <model.pt>` prints a real scored report
  for onnx + coreml; JSON is machine-readable; tests green.

## M2 — Break it (Phase 4)
- Build the failure injector / test rig:
  - bad exports (wrong batch, wrong input names, dropped outputs)
  - degraded precision (fp16 export, quantized int8)
  - unsupported operators (forcing fallback)
  - corrupted models (truncated weights, bad scales)
  - NMS differences, padding differences
- Prove the engines *detect and explain* each injected failure (root-cause
  hypotheses fire with high confidence).
- **Exit criteria:** a battery of fault-injection tests, each pinned to expected
  Evidence output. Regression memory ingests the injected archetypes.

## M3 — Expand (Phase 5)
- Runtimes: `run_openvino`, `run_tflite`, `run_mlx`, `run_tensorflow`,
  `run_tensorrt` (CI-only).
- IR passes: `canonicalize`, `partition`, `substitute`, `annotate`.
- Failure Discovery v1 (mutate / fuzz / synthesize) feeding Reduce.
- Automatic Failure Reduction (spatial + value minimization).
- AI Research Agent integration stub (evidence -> search -> summary).
- **Exit criteria:** 6+ runtimes behind one verify; discovery finds a divergence
  automatically; a reduced repro < 15% of original size; research summary
  produced.

## M4 — Production hardening (Phase 6)
- Full test suite (unit + integration + fault-injection + perf smoke).
- Docs (install, verify, CI integration, report schema, contributing).
- CI (GitHub Actions: full on-arm64 runners; GPU/TensorRT job optional/skipped).
- Packaging (`pyproject.toml`, extras, `pip install cvconform` verified).
- Examples + benchmark/model corpus.
- Security review (no secrets in config; sandboxed remote code exec; corpus I/O
  validation).
- **Exit criteria:** a fresh engineer follows README to a useful report, and the
  CI gate passes.

## M5 — Moat (onward)
- Open the corpus as the structured "how vision models fail" database.
- Publically-citable failure entries; contributor flow for new archetypes.
- Versioned conformance feed per runtime release ("ONNX 1.17 broke Resize"):
  the automated discoverer becomes the market signal.

## Sequencing notes
- M1 is the only prerequisite for anything user-visible; land it first and make
  it real.
- Evidence schema is designed up front (used by differential, root-cause,
  corpus) so nothing is reworked later.
- GPU-only runtimes are designed for, not blocked on — code stays tested on
  CPU/Metal.
