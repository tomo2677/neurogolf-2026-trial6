from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 15


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
        _int64_tensor("starts15", [0, 0, 0, 0], [4]),
        _int64_tensor("row15_ends", [1, 1, SIZE, 1], [4]),
        _int64_tensor("col15_ends", [1, 1, 1, SIZE], [4]),
        _int64_tensor("row_ch1_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("row_ch1_ends", [1, 2, SIZE, 1], [4]),
        _int64_tensor("row_ch3_starts", [0, 3, 0, 0], [4]),
        _int64_tensor("row_ch3_ends", [1, 4, SIZE, 1], [4]),
        _int64_tensor("col_ch2_starts", [0, 2, 0, 0], [4]),
        _int64_tensor("col_ch2_area_ends", [1, 3, SIZE, SIZE], [4]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 6, 30 - SIZE, 30 - SIZE], [8]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("three_u8", [3], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors4", list(range(4)), [1, 4, 1, 1]),
    ]

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["row_present30"], axes=[1, 3], keepdims=1),
        helper.make_node("ReduceMax", ["input"], ["col_present30"], axes=[1, 2], keepdims=1),
        helper.make_node("ReduceMax", ["input"], ["row_by_color"], axes=[3], keepdims=1),
        helper.make_node("Slice", ["row_present30", "starts15", "row15_ends"], ["row_present15"]),
        helper.make_node("Slice", ["col_present30", "starts15", "col15_ends"], ["col_present15"]),
        helper.make_node("Cast", ["row_present15"], ["row_valid"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["col_present15"], ["col_valid"], to=onnx.TensorProto.BOOL),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_area"]),
        helper.make_node("Slice", ["row_by_color", "row_ch1_starts", "row_ch1_ends"], ["row_has_1_f32"]),
        helper.make_node("Slice", ["row_by_color", "row_ch3_starts", "row_ch3_ends"], ["row_has_3_f32"]),
        helper.make_node("Slice", ["input", "col_ch2_starts", "col_ch2_area_ends"], ["col2_area"]),
        helper.make_node("ReduceMax", ["col2_area"], ["col_has_2_f32"], axes=[2], keepdims=1),
        helper.make_node("Cast", ["row_has_1_f32"], ["row_has_1"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["row_has_3_f32"], ["row_has_3"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["col_has_2_f32"], ["col_has_2"], to=onnx.TensorProto.BOOL),
        helper.make_node("Where", ["col_has_2", "two_u8", "zero_u8"], ["col_color"]),
        helper.make_node("Where", ["row_has_3", "three_u8", "col_color"], ["row3_color"]),
        helper.make_node("Where", ["row_has_1", "one_u8", "row3_color"], ["raw_color"]),
        helper.make_node("Where", ["valid_area", "raw_color", "invalid_u8"], ["color15"]),
        helper.make_node("Equal", ["colors4", "color15"], ["output4"]),
        helper.make_node("Pad", ["output4", "pads_output"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task024_colors4_bool_pad_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
