# ARCHITECTURE — cvconform

## High-level layout

```
                    ┌──────────────────────────────┐
                    │        cvconform lib          │
                    └──────────────┬───────────────┘
                                   │
     ┌──────────────┬──────────────┼──────────────┬───────────────┐
     │              │              │              │               │
  loaders/      ir/            runtimes/       engines/         report/
  load_pt      graph.py        run_onnx       differential     render
  load_onnx    ir.py           run_coreml     discover         store
  load_mlx     passes.py       run_mlx        reduce
  load_tf      precision.py    run_mlx        analyze/root_cause
  load_tf                       run_tensorrt
```

- **loaders/** — parse any framework artifact into a `VisionGraph` (the IR).
- **ir/** — `VisionGraph`, `Operator`, `TensorSpec`, precision model, passes.
- **runtimes/** — thin, versioned wrappers that execute a graph on a specific
  runtime and emit normalized outputs. Every runtime reports its exact version
  and build — this is what makes reproduction possible.
- **engines/** — the pipeline: `differential` (compare runtimes), `discover`
  (find failures), `reduce` (minimize a failing input), `analyze` (root cause).
- **report/** — human + machine-readable conformance reports (the "product").
  Machine-readable format doubles as the Regression Memory index.

## Vision IR (VisionGraph)

Analogous to LLVM IR but for vision models.

A `VisionGraph` is a DAG:

```
Input(1,3,640,640, fp32)
  └─ Conv2D(3→64,k3,s1,p1) [fp32]
      └─ BatchNorm [fp32]
          └─ ReLU [fp32]
              └─ MaxPool(k2,s2)
                  └─ ... backbone ...
                      └─ DetectionHead
                          └─ NMS
                              └─ Output(boxes, classes, scores)
```

Every node carries:
- **op** — canonical operator name (namespaced, e.g. `cv.Conv2D`)
- **inputs / outputs** — `TensorSpec{shape, dtype/precision, layout, quant info}`
- **attrs** — strides, pads, groups, eps, scales, auto_pad, etc.
- **provenance** — which loader created it + the original op string

Canonical operator registry (initial set):
- **tensor**: Identity, Reshape, Transpose, Flatten, Concat, Split, Slice, Gather,
  Squeeze, Unsqueeze, Pad
- **conv/nn**: Conv2D, ConvTranspose2D, DepthwiseConv, BatchNorm, GroupNorm,
  LayerNorm, InstanceNorm, Gemm/MatMul, Add, Mul, Pool (Max/Avg/Global), Relu,
  LeakyRelu(alpha), Sigmoid, Softmax, HardSwish/SiLU, Resize/Bilinear/Upsample
- **vision head**: DetectionOutput/NMS(center vs corner, IoU method, class-agnostic),
  GridAnchor (anchor parsing), ClassifierHead, SegmentationHead
- **quant**: QuantizeLinear, DequantizeLinear (per-tensor / per-channel)
- **meta**: Input, Output, Constant

**Precision model.** Every tensor is `fp32 | fp16 | bf16 | int8 | uint8 | float32-quant`. The IR records declared precision and, for int8, the scale/zero-point — this is exactly what makes "FP16 accumulation drift" and "quantization scale mismatch" diagnosable instead of mysterious.

**Passes** (the LLVM-style core, Phase 5+):
- `canonicalize` — normalize equivalent graphs (e.g. padding conventions, op
  decomposition) so LHS/RHS are comparable.
- `partition` — split a graph into independently-conformable regions.
- `substitute` — functional-equivalence checking of sub-graphs (does fused
  Conv+ReLU equal unfused?).
- `annotate` — attach expected tolerances per op/precision from a policy.

## Differential Execution Engine

The heart. For each `target` runtime:

1. Lower the source model to that runtime (export + compile).
2. Generate (or load) a batch of test inputs — seedable, reproducible.
3. Execute reference runtime and all targets on the *same* inputs.
4. Normalize outputs into a common schema:

```json
{
  "kind": "tensor|boxes|masks|classes|scores",
  "tensor":   {"shape": [...], "dtype": "fp32", "nan_frac": 0.0, "inf_frac": 0.0, "stats": {...}},
  "boxes":    [{"coords":[x1,y1,x2,y2], "class":3, "score":0.92}],
  "masks":    {"shape":[...], "stats": {...}},
  "classes":  [...],
  "scores":   [...]
}
```

5. Compare with tolerance policies (max abs err, mean abs err, IoU for boxes,
   class agreement, KL-divergence for distributions).

**Comparison semantics** differ by output kind:
- **tensors**: elementwise max-abs and RMSE against ref; NaN/Inf detection.
- **boxes**: per-class IoU, coordinate error, count mismatch.
- **classes**: class agreement rate.
- **scores**: signed error, ordering agreement.

Outputs are matched by **semantic output name** (the IR output label), not by
position — so runtimes that reorder outputs still compare correctly.

## Failure Discovery

Beyond fixed datasets. Three generators:
- **mutate** — perturb existing inputs (noise, compression, crop, brightness,
  contrast, blur, rotation, resizing).
- **fuzz** — random tensors within valid input ranges, plus adversarial extremes
  (NaN, Inf, extreme brightness, solid fills, single-pixel).
- **synthesize** — programmatic scenes (via optional dependency on a graphics
  toolchain like Taichi/PIL) to build controlled, attribute-tagged inputs.

Each discovered divergence seeds a candidate regression case.

## Automatic Failure Reduction

If a target diverges on input X, minimize X while the divergence persists:

1. **spatial**: crop / downscale and re-check.
2. **value**: reduce bit depth, quantize, drop channels — re-check.
3. **search**: greedy + bisection over which sub-region / attribute matters.

Output: a minimal reproducing input + the pass/fail matrix. (Bound by time
budget; deterministic seed and step budget recorded.)

## Root-Cause Analysis

Given a divergence, form and test hypotheses:
- precision drift (lower ref to fp16, or raise target to fp32 — does it vanish?)
- fusion order (disable fusion — does it vanish?)
- padding / auto_pad mismatch (IR attr diff)
- NMS implementation difference (compare on non-NMS head output)
- op fallback (compare against runtime-declared supported-ops set)
- quant scale mismatch (inspect int8 scale/zero-point across runtimes)

Each hypothesis produces an **Evidence** object: observed divergence location,
magnitude, first-diverging node, mechanism, confidence (what fraction of
hypothesis-rules fired). Never "probably a bug" — always "Observed X, magnitude
Y, first divergence at Z, likely mechanism W, confidence C%."

## Regression Memory

Structured, versioned storage:

```
corpus/
  failures/
    coreml/fp16_conv_bug_001/
    onnx/resize_difference_004/
    tensorrt/nms_issue_012/
  models/<hash>/          # reference model + target artifacts + hashes
  inputs/<hash>/          # repro inputs + generator recipe
  reports/<id>.json       # full evidence
```

Every failure is a permanent test that re-runs on every new version of any
runtime or model. The corpus is versioned (each entry records env + exact
versions), so `cvconform` "gets smarter forever."

## AI Research Agent

On failure, orchestrate research: search docs / GitHub issues / papers / release
notes for the mechanism; produce a research summary with matching issues,
affected/fixed versions, and a recommendation. This is a thin orchestration
layer — the real signal comes from the structured evidence (root-cause) feeding
an LLM-backed researcher that cites sources.

## Conformance report (the product)

Both human and machine readable:

```
VISION CONFORM REPORT
Model:   ...
Reference: PyTorch FP32
Targets: ✓ ONNX  ✓ TensorRT  ✓ CoreML

CONFORMANCE SCORE        (per target, per output kind, averaged per policy)
ONNX:     99.99%
TensorRT: 99.81%
CoreML:   87.22%

CRITICAL FINDING
Backend: CoreML     Affected: Conv+Activation fusion
Cause: FP16 accumulation drift
Impact: 37 production images produce different classes
Confidence: 96%
Fixes: 1. Disable operator fusion  2. Force FP32 accumulation
       3. Report upstream issue
```

Machine-readable JSON is the contract for CI/CD.

## Tenets

1. **Framework-agnostic.** Everything below the loaders is runtime-neutral.
2. **Scientific & reproducible.** Seedable inputs, recorded env, exact versions,
   deterministic reproduction — always.
3. **Evidence-driven.** Every claim carries an Evidence object. No vibes.
4. **Actively memoryful.** The corpus is a product feature, not a side table.
