from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _bool_tensor(name: str, values: list[bool], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.BOOL, dims, values)


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
        _bool_tensor("false7", [False] * (7 * 5 * 3), [1, 7, 5, 3]),
        _bool_tensor("false9", [False] * (5 * 3), [1, 1, 5, 3]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "left_starts", "left_ends"], ["left_black"]),
        helper.make_node("Slice", ["input", "right_starts", "right_ends"], ["right_black"]),
        helper.make_node("Greater", ["left_black", "zero_f32"], ["left_is_black"]),
        helper.make_node("Greater", ["right_black", "zero_f32"], ["right_is_black"]),
        helper.make_node("And", ["left_is_black", "right_is_black"], ["eight_ch"]),
        helper.make_node("Not", ["eight_ch"], ["zero_ch"]),
        helper.make_node("Concat", ["zero_ch", "false7", "eight_ch", "false9"], ["output5x3"], axis=1),
        helper.make_node("Pad", ["output5x3", "pads_output"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task026_bool_output_onehot_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
