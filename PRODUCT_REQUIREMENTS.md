# PRODUCT REQUIREMENTS — cvconform

## User

**Primary persona — the CV deployment engineer** (ML infra engineer, MLOps,
edge-compute engineer, mobile/CV SDK maintainer):

- Owns a model that travels through many runtimes: trains in PyTorch, exports to
  ONNX, compiles to TensorRT/CoreML/OpenVINO/TFLite, quantizes, fuses.
- Has been bitten — silently — by a deployment that "ran" but was no longer the
  same model: wrong classes on some images, drift in boxes, FP16 artifacts.
- Needs an answer to *"is this still the same model?"* that is fast, automatic,
  and that they can put in CI/CD — before trusting the output.

**Secondary persona — the ML ops / CI owner:** wants a pass/fail gate in the
pipeline, versioned, low-noise, reproducible across machines.

**Anti-persona (we optimize away from):** the researcher who wants a training
tool. Not us.

## Core jobs-to-be-done

1. **Verify:** point at a model, get a conformance report across target runtimes
   in minutes. (CLI: `cvconform verify model.pt`)
2. **Automate:** run the same check in CI/CD and gate deploys.
3. **Diagnose:** when conformance drops, know *exactly* why — which op, which
   backend, which precision/fusion change, which images.
4. **Remember:** every failure ever found stays a permanent regression test.
5. **Discover:** proactively find where the model breaks before users do.

## Functional requirements

### F1. CLI (`cvconform verify <model>`)
- Accepts a model artifact path + a reference runtime, and a list of targets.
- Also a Python API:
  ```python
  from cvconform import verify
  result = verify(reference="pytorch", targets=["onnx","coreml","tensorrt","openvino"], dataset="production_samples")
  ```
- Returns a conformance report (human + machine-readable JSON) with a score per
  target.

### F2. Runtimes (Phase 3 slice, then expand)
- **P3:** PyTorch, ONNX / onnxruntime, CoreML. (Real, runnable locally.)
- **P5+:** TensorFlow, OpenVINO, TFLite, MLX, TensorRT (GPU, CI-only).

### F3. Differential execution
- Same seedable inputs to ref + all targets; normalized outputs; per-output-kind
  comparison (tensor / boxes / masks / classes / scores).

### F4. Conformance scoring
- Per-target score in [0,100]% with a documented, configurable policy and
  per-output-kind sub-scores.

### F5. Root-cause evidence
- Every low-scoring target yields an Evidence object: divergence node, magnitude,
  first-diverging node, mechanism hypothesis, confidence %, recommended fixes.
- No unbacked "probably a bug".

### F6. Regression memory
- Failures persist as versioned corpus entries; re-run automatically; repo ships
  with an initial corpus of known failure archetypes.

### F7. Discovery (P4/P5)
- Mutation, fuzz, adversarial, synthetic inputs; each new divergence seeds a
  regression case.

## Non-functional requirements

- **Reproducibility (P0):** same model + same versions + same seed ⇒ same
  evidence. Env + exact versions recorded in every output.
- **Framework-agnostic (P0):** core has no dependency on any one ML framework.
- **Evidence-driven (P0):** every claim traceable to a recorded run.
- **Performance:** a verify run on a small/medium model + 2-3 runtimes should
  finish in minutes on a laptop; designed to scale to GPUs.
- **Memoryful (P0):** corpus is product, not a side table.
- **Packaging:** `pip install cvconform`; minimal hard deps (numpy, onnx, onnxruntime);
  optional extras (`[coreml]`, `[pytorch]`, `[openvino]`, etc.).
- **Exact versions everywhere:** runtime wrappers pin and report their backend
  versions; conformance scores are meaningless without them.

## Metrics of success (product is not done until…)

1. A random ML engineer can `pip install cvconform`, point it at a model, and
   receive useful, *explained* conformance info within minutes.
2. They understand exactly why a deployment differs (root cause, not just a red
   score).
3. They can add it to CI/CD and gate deployments on it.
4. The repo ships a real corpus capturing real failure archetypes.

## Out of scope (v1)

- Training, fine-tuning, dataset tooling.
- Model zoo / curation.
- Deploying or hosting models.
- GPU-native support on non-NVIDIA machines (TensorRT runs in CI only).
