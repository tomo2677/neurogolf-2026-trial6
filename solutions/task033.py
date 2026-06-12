from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _bool_tensor(name: str, values: list[bool], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.BOOL, dims, values)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("template_starts", [0, 1, 1, 1], [4]),
        _int64_tensor("template_ends", [1, 10, 4, 4], [4]),
        _int64_tensor("grid_color_starts", [0, 0, 5, 0], [4]),
        _int64_tensor("grid_color_ends", [1, 10, 6, 1], [4]),
        _int64_tensor("input0_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("input0_ends", [1, 1, 17, 17], [4]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _bool_tensor("false_col1", [False] * 3, [1, 1, 3, 1]),
        _bool_tensor("false_gap3", [False] * 9, [1, 1, 3, 3]),
        _bool_tensor("false_row1", [False] * 17, [1, 1, 1, 17]),
        _bool_tensor("false_rows3", [False] * 51, [1, 1, 3, 17]),
        _bool_tensor("false_right13", [False] * (17 * 13), [1, 1, 17, 13]),
        _bool_tensor("false_bottom13", [False] * (13 * 30), [1, 1, 13, 30]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "template_starts", "template_ends"], ["template_colors"]),
        helper.make_node("ReduceMax", ["template_colors"], ["template_score"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["template_score", "zero_f32"], ["template3"]),
        helper.make_node(
            "Concat",
            ["false_col1", "template3", "false_gap3", "template3", "false_gap3", "template3", "false_col1"],
            ["template_row"],
            axis=3,
        ),
        helper.make_node(
            "Concat",
            ["false_row1", "template_row", "false_rows3", "template_row", "false_rows3", "template_row", "false_row1"],
            ["template17"],
            axis=2,
        ),
        helper.make_node("Slice", ["input", "input0_starts", "input0_ends"], ["input0_17"]),
        helper.make_node("Greater", ["input0_17", "zero_f32"], ["background17"]),
        helper.make_node("And", ["template17", "background17"], ["fill17"]),
        helper.make_node("Concat", ["fill17", "false_right13"], ["fill17_wide"], axis=3),
        helper.make_node("Concat", ["fill17_wide", "false_bottom13"], ["fill30"], axis=2),
        helper.make_node("Slice", ["input", "grid_color_starts", "grid_color_ends"], ["grid_color"]),
        helper.make_node("Where", ["fill30", "grid_color", "input"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task033_template_fill_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
