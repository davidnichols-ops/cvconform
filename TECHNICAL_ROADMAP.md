# TECHNICAL_ROADMAP — cvconform

Milestones. Each ends with something concrete, testable, and evidence-backed.

## M0 — Foundations (Phase 2, ≈ done)
- [x] Environment recon (CURRENT_STATE.md)
- [x] VISION / ARCHITECTURE / PRODUCT_REQUIREMENTS / this roadmap
- [x] Project skeleton, Python 3.12 venv, git, conventional commits.

## M1 — Vertical slice (Phase 3) — ✅ DONE
Goal: a real model, 2-3 real runtimes, a real report.
- **IR**: `VisionGraph`, `Operator`, `TensorSpec`, precision model, canonical op
  registry + ONNX loader lowering to IR. (Unit-tested.) ✅
- **Runtimes**: `run_onnx` (onnxruntime, real), `run_pytorch` (torchscript/torch,
  real), `run_coreml` (coremltools, real). Each: version-reporting wrapper +
  output normalization -> normalized output schema. ✅
- **Differential engine**: seedable input generation; comparison for tensor /
  boxes / classes / scores; tolerance policy; per-target score. ✅
- **Report**: human text + JSON. ✅
- **Slice demo**: tiny YOLO-style detector exported to ONNX and CoreML, verified
  on seeded synthetic inputs across pytorch + onnx + coreml. ✅
  **Measured result:** ONNX 100.00%, CoreML 96.36% on the demo (M4, seed=0).

## M2 — Break it (Phase 4) — ✅ DONE
- Fault-injection rig: `cvconform.faults` (weights_scale, precision_fp16,
  weight_nan, weight_corrupt, bias_shift, output_permute, act_clip). ✅
- Proof: clean control conformant; every fault detected + explained. ✅
- Hardened the engine in the process (shape mismatch is now a hard failure). ✅

## M3 — Expand (Phase 5) — ✅ (core) DONE
- Runtimes: onnxruntime / pytorch / coreml live locally. openvino/tflite/mlx/
  tensorflow/tensorrt remain on the roadmap behind the same engine interface. ⏳
- IR passes (canonicalize/partition/substitute/annotate): roadmap. ⏳
- **Failure Discovery v1** (noise/edges/illumination/blur/compression/extremes
  generators + Expedition ranking) — ✅ DONE.
- **Automatic Failure Reduction** (spatial + value minimization) — ✅ DONE.
- **AI Research Agent** (evidence -> search -> summary, offline fallback) — ✅ DONE.
- **Regression Memory** (`ConformanceCorpus`, versioned manifests) — ✅ DONE.

## M4 — Production hardening (Phase 6) — ✅ (core) DONE
- Test suite: 36 unit + integration + fault-injection + discovery/reduce. ✅
- Docs: README, this roadmap, VISION/ARCHITECTURE/PRODUCT_REQUIREMENTS/CURRENT_STATE. ✅
- CI: GitHub Actions (macOS test job with torch+onnx+coreml; ruff job). ✅
- Packaging: `pip install -e .`, `cvconform` entrypoint, `--require-conformant`
  CI gate, `--corpus` recording. ✅
- Examples + reproducible demo artifacts. ✅
- **Exit criterion met:** `cvconform verify model.pt` produces a scored,
  explained report a fresh engineer can gate CI on.

## Remaining (M5 / onward)
- More runtimes behind the same interface (openvino, tflite, mlx, tensorrt-CI).
- IR analysis passes (canonicalize, partition, substitute, annotate).
- Publish the corpus as the structured "how vision models fail" database.
- Self-updating conformance feed per runtime release (the automated discoverer
  becomes market signal).
