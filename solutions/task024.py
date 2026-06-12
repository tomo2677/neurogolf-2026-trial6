from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30


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
        _int64_tensor("ch1_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("ch1_ends", [1, 2, SIZE, SIZE], [4]),
        _int64_tensor("ch2_starts", [0, 2, 0, 0], [4]),
        _int64_tensor("ch2_ends", [1, 3, SIZE, SIZE], [4]),
        _int64_tensor("ch3_starts", [0, 3, 0, 0], [4]),
        _int64_tensor("ch3_ends", [1, 4, SIZE, SIZE], [4]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("three_u8", [3], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["cell_present_f32"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["cell_present_f32"], ["row_present_f32"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["cell_present_f32"], ["col_present_f32"], axes=[2], keepdims=1),
        helper.make_node("Greater", ["row_present_f32", "zero_f32"], ["row_valid"]),
        helper.make_node("Greater", ["col_present_f32", "zero_f32"], ["col_valid"]),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_area"]),
        helper.make_node("Slice", ["input", "ch1_starts", "ch1_ends"], ["ch1"]),
        helper.make_node("Slice", ["input", "ch2_starts", "ch2_ends"], ["ch2"]),
        helper.make_node("Slice", ["input", "ch3_starts", "ch3_ends"], ["ch3"]),
        helper.make_node("ReduceMax", ["ch1"], ["row_has_1_f32"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["ch3"], ["row_has_3_f32"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["ch2"], ["col_has_2_f32"], axes=[2], keepdims=1),
        helper.make_node("Greater", ["row_has_1_f32", "zero_f32"], ["row_has_1"]),
        helper.make_node("Greater", ["row_has_3_f32", "zero_f32"], ["row_has_3"]),
        helper.make_node("Greater", ["col_has_2_f32", "zero_f32"], ["col_has_2"]),
        helper.make_node("Where", ["col_has_2", "two_u8", "zero_u8"], ["col_color"]),
        helper.make_node("Where", ["row_has_3", "three_u8", "col_color"], ["row3_color"]),
        helper.make_node("Where", ["row_has_1", "one_u8", "row3_color"], ["raw_color"]),
        helper.make_node("Where", ["valid_area", "raw_color", "invalid_u8"], ["color30"]),
        helper.make_node("Equal", ["colors10", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task024_row_col_extension_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
