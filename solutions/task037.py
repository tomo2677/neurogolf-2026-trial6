from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 10
NONZERO = 9
KERNEL = 19
MID = 9


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _diag_kernel(offsets: list[tuple[int, int]]) -> list[float]:
    values = [0.0] * (NONZERO * KERNEL * KERNEL)
    for channel in range(NONZERO):
        base = channel * KERNEL * KERNEL
        for dr, dc in offsets:
            values[base + (MID + dr) * KERNEL + (MID + dc)] = 1.0
    return values


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("nonzero_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("nonzero_ends", [1, 10, SIZE, SIZE], [4]),
        _int64_tensor("output_pads", [0, 0, 0, 0, 0, 0, 20, 20], [8]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("one_f32", [1.0], [1]),
        _f32_tensor("ul_w", _diag_kernel([(-k, -k) for k in range(SIZE)]), [NONZERO, 1, KERNEL, KERNEL]),
        _f32_tensor("dr_w", _diag_kernel([(k, k) for k in range(SIZE)]), [NONZERO, 1, KERNEL, KERNEL]),
        _f32_tensor("ur_w", _diag_kernel([(-k, k) for k in range(SIZE)]), [NONZERO, 1, KERNEL, KERNEL]),
        _f32_tensor("dl_w", _diag_kernel([(k, -k) for k in range(SIZE)]), [NONZERO, 1, KERNEL, KERNEL]),
    ]

    conv_attrs = {
        "kernel_shape": [KERNEL, KERNEL],
        "pads": [MID, MID, MID, MID],
        "group": NONZERO,
    }

    nodes = [
        helper.make_node("Slice", ["input", "nonzero_starts", "nonzero_ends"], ["input9"]),
        helper.make_node("Conv", ["input9", "ul_w"], ["ul_score"], **conv_attrs),
        helper.make_node("Conv", ["input9", "dr_w"], ["dr_score"], **conv_attrs),
        helper.make_node("Conv", ["input9", "ur_w"], ["ur_score"], **conv_attrs),
        helper.make_node("Conv", ["input9", "dl_w"], ["dl_score"], **conv_attrs),
        helper.make_node("Greater", ["ul_score", "zero_f32"], ["ul"]),
        helper.make_node("Greater", ["dr_score", "zero_f32"], ["dr"]),
        helper.make_node("Greater", ["ur_score", "zero_f32"], ["ur"]),
        helper.make_node("Greater", ["dl_score", "zero_f32"], ["dl"]),
        helper.make_node("And", ["ul", "dr"], ["main_diag"]),
        helper.make_node("And", ["ur", "dl"], ["anti_diag"]),
        helper.make_node("Or", ["main_diag", "anti_diag"], ["line_bool"]),
        helper.make_node("Cast", ["line_bool"], ["line9"], to=onnx.TensorProto.FLOAT),
        helper.make_node("ReduceMax", ["line9"], ["line_any"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["one_f32", "line_any"], ["zero10"]),
        helper.make_node("Concat", ["zero10", "line9"], ["output10"], axis=1),
        helper.make_node("Pad", ["output10", "output_pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task037_diagonal_pair_lines_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
