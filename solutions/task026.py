from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("left_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("left_ends", [1, 1, 5, 3], [4]),
        _int64_tensor("right_starts", [0, 0, 0, 4], [4]),
        _int64_tensor("right_ends", [1, 1, 5, 7], [4]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 25, 27], [8]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "left_starts", "left_ends"], ["left_black"]),
        helper.make_node("Slice", ["input", "right_starts", "right_ends"], ["right_black"]),
        helper.make_node("Greater", ["left_black", "zero_f32"], ["left_is_black"]),
        helper.make_node("Greater", ["right_black", "zero_f32"], ["right_is_black"]),
        helper.make_node("And", ["left_is_black", "right_is_black"], ["eight_ch"]),
        helper.make_node("Where", ["eight_ch", "eight_u8", "zero_u8"], ["color5x3"]),
        helper.make_node("Equal", ["colors10", "color5x3"], ["output5x3"]),
        helper.make_node("Pad", ["output5x3", "pads_output"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task026_color_grid_equal_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
