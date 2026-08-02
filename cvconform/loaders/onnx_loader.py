"""Loader: ONNX -> VisionGraph.

Reads an ONNX protobuf and lowers it into the canonical :class:`VisionGraph`
IR. This is the reference lowering and the one every other loader is expected
to mirror.

The goal is *not* to perfectly represent every ONNX op (that is a large,
versioned surface); it is to capture the *semantics that matter* for
conformance: tensor shapes, dtypes, precision, quantization, and the
operation/attribute structure — the things that change when a model is
exported, fused, or lowered.
"""

from __future__ import annotations

import onnx
from onnx import numpy_helper

from cvconform.ir import Operator, OperatorRegistry, Quantization, TensorSpec, VisionGraph, Precision

# ONNX op-type -> canonical op (with a fallback to cv.Unsupported when absent).
_OP_MAP = {
    "Conv": OperatorRegistry.CONV2D,
    "ConvTranspose": OperatorRegistry.CONVTRANSPOSE2D,
    "BatchNormalization": OperatorRegistry.BATCHNORM,
    "GroupNormalization": OperatorRegistry.GROUPNORM,
    "LayerNormalization": OperatorRegistry.LAYERNORM,
    "InstanceNormalization": OperatorRegistry.INSTANCENORM,
    "Gemm": OperatorRegistry.GEMM,
    "MatMul": OperatorRegistry.MATMUL,
    "Add": OperatorRegistry.ADD,
    "Mul": OperatorRegistry.MUL,
    "Sub": OperatorRegistry.SUB,
    "Div": OperatorRegistry.DIV,
    "MaxPool": OperatorRegistry.MAXPOOL,
    "AveragePool": OperatorRegistry.AVGPOOL,
    "GlobalAveragePool": OperatorRegistry.GLOBALAVGPOOL,
    "Relu": OperatorRegistry.RELU,
    "LeakyRelu": OperatorRegistry.LEAKY_RELU,
    "Sigmoid": OperatorRegistry.SIGMOID,
    "Softmax": OperatorRegistry.SOFTMAX,
    "Resize": OperatorRegistry.RESIZE,
    "Upsample": OperatorRegistry.RESIZE,
    "Transpose": OperatorRegistry.TRANSPOSE,
    "Reshape": OperatorRegistry.RESHAPE,
    "Flatten": OperatorRegistry.FLATTEN,
    "Concat": OperatorRegistry.CONCAT,
    "Split": OperatorRegistry.SPLIT,
    "Slice": OperatorRegistry.SLICE,
    "Gather": OperatorRegistry.GATHER,
    "Squeeze": OperatorRegistry.SQUEEZE,
    "Unsqueeze": OperatorRegistry.UNSQUEEZE,
    "Pad": OperatorRegistry.PAD,
    "Identity": OperatorRegistry.IDENTITY,
    "Clip": OperatorRegistry.CLIP,
    "ReduceMean": OperatorRegistry.REDUCEMEAN,
    "Constant": OperatorRegistry.CONSTANT,
    "ConstantOfShape": OperatorRegistry.CONSTANT,
}


def _shape(value_info) -> Tuple:
    t = value_info.type.tensor_type
    if not t.HasField("shape"):
        return None
    dims = []
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:
            dims.append(None)  # dynamic dim
    return tuple(dims)


def _dtype(value_info) -> str:
    t = value_info.type.tensor_type
    return onnx.TensorProto.DataType.Name(t.elem_type).lower()


def _attrs(node) -> dict:
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode("utf-8", "replace")
        elif a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOATS:
            out[a.name] = list(a.floats)
        elif a.type == onnx.AttributeProto.STRINGS:
            out[a.name] = [s.decode("utf-8", "replace") for s in a.strings]
        elif a.type == onnx.AttributeProto.TENSOR:
            try:
                t = numpy_helper.to_array(a.t)
                out[a.name] = t.tolist() if t.size <= 64 else f"tensor{list(t.shape)}"
            except Exception:
                out[a.name] = "<tensor>"
        else:
            out[a.name] = f"<attr:{a.type}>"
    return out


def _quant_from_initializer(value_info) -> Quantization:
    """Best-effort: read scale/zero_point from an initializer value_info."""
    t = value_info.type.tensor_type
    if t.elem_type == onnx.TensorProto.INT8 or t.elem_type == onnx.TensorProto.UINT8:
        return Quantization(per_channel=True)  # presence signals quantized
    return None


def _load_value_info(m: onnx.ModelProto) -> dict:
    """Map value_info name -> TensorSpec (from graph inputs/outputs + value_info)."""
    specs: dict = {}

    def visit(vi, is_graph_input: bool):
        if not vi.name:
            return
        sp = TensorSpec(
            shape=_shape(vi),
            dtype=_dtype(vi),
        )
        specs[vi.name] = sp

    for vi in m.graph.input:
        visit(vi, True)
    for vi in m.graph.output:
        visit(vi, False)
    for vi in m.graph.value_info:
        visit(vi, False)
    return specs


def load_onnx(path: str) -> VisionGraph:
    """Parse an ONNX file and lower it to a VisionGraph."""
    model = onnx.load(path)
    return from_onnx_model(model)


def from_onnx_model(model: onnx.ModelProto) -> VisionGraph:
    m = model.graph
    g = VisionGraph(name=model.graph.name or "onnx_model", loader="onnx",
                    loader_version=onnx.__version__)
    specs = _load_value_info(model)

    # Constants from initializers / Constant nodes become tensor data available
    # to consumers; represent them as root producer edges.
    init_by_name = {}  # tensor name -> numpy array (if scalar-y)
    for t in m.initializer:
        try:
            arr = numpy_helper.to_array(t)
            if arr.size <= 64:
                init_by_name[t.name] = arr
        except Exception:
            pass

    for node in m.node:
        canonical = _OP_MAP.get(node.op_type, OperatorRegistry.UNSUPPORTED)
        provenance = f"onnx: {node.op_type}"
        name = node.name or f"{node.op_type}_{len(g.ops)}"

        inputs: list = []
        for in_name in node.input:
            if not in_name:
                continue
            spec = specs.get(in_name) or TensorSpec(dtype="unknown")
            # enrich spec with quant presence from initializers
            if in_name in init_by_name and spec.dtype not in ("unknown",):
                try:
                    if spec.dtype in ("int8", "uint8"):
                        spec.quant = Quantization(per_channel=True)
                except Exception:
                    pass
            inputs.append((in_name, spec))

        outputs: list = []
        for out_name in node.output:
            if not out_name:
                continue
            spec = specs.get(out_name) or TensorSpec(dtype="unknown")
            outputs.append((out_name, spec))

        attrs = _attrs(node)

        op = Operator(op=canonical, name=name, inputs=inputs, outputs=outputs,
                      attrs=attrs, provenance=provenance)
        g.add_op(op)

    # Turn graph I/O into boundary ops.
    return _add_boundaries(g, m, specs)


def _add_boundaries(g: VisionGraph, m, specs: dict) -> VisionGraph:
    """Add cv.Input / cv.Output boundary nodes and wire graph I/O."""
    input_ops = {name: op for op in g.ops.values() for n, _ in op.outputs for name in [n]}
    output_ops = {name: op for op in g.ops.values() for n, _ in op.inputs for name in [n]}

    # Graph inputs
    for vi in m.input:
        name = vi.name
        spec = specs.get(name) or TensorSpec(shape=_shape(vi), dtype=_dtype(vi))
        if name not in input_ops:
            op = Operator(op=OperatorRegistry.INPUT, name=f"_in_{name}",
                          inputs=[], outputs=[(name, spec)],
                          attrs={}, provenance="onnx: graph-input")
            g.add_op(op)
        else:
            # A graph input that is also produced elsewhere shouldn't happen; keep simple.
            pass

    # Graph outputs
    for vi in m.output:
        name = vi.name
        spec = specs.get(name) or TensorSpec(shape=_shape(vi), dtype=_dtype(vi))
        if name not in output_ops:
            op = Operator(op=OperatorRegistry.OUTPUT, name=f"_out_{name}",
                          inputs=[(name, spec)], outputs=[],
                          attrs={}, provenance="onnx: graph-output")
            g.add_op(op)

    g.refresh_boundaries()
    return g


def collect_runtime_tensors(g: VisionGraph):
    """Return list of (name, TensorSpec) for every edge in the graph."""
    out = {}
    for op in g.ops.values():
        for n, s in op.inputs:
            out.setdefault(n, s)
        for n, s in op.outputs:
            out.setdefault(n, s)
    return out
