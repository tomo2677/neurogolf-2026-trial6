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


def _conv_weight(name: str, color: int, kernel_h: int, kernel_w: int) -> onnx.TensorProto:
    values = [0.0] * (10 * kernel_h * kernel_w)
    for kr in range(kernel_h):
        for kc in range(kernel_w):
            values[((color * kernel_h) + kr) * kernel_w + kc] = 1.0
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, [1, 10, kernel_h, kernel_w], values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("starts15", [0, 0], [2]),
        _int64_tensor("row15_ends", [SIZE, 1], [2]),
        _int64_tensor("col15_ends", [1, SIZE], [2]),
        _int64_tensor("axes_hw", [2, 3], [2]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 6, 30 - SIZE, 30 - SIZE], [8]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("three_u8", [3], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors4", list(range(4)), [1, 4, 1, 1]),
        _conv_weight("row1_w", 1, 1, 30),
        _conv_weight("row3_w", 3, 1, 30),
        _conv_weight("col2_w", 2, 30, 1),
    ]

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["row_present30"], axes=[1, 3], keepdims=1),
        helper.make_node("ReduceMax", ["input"], ["col_present30"], axes=[1, 2], keepdims=1),
        helper.make_node("Slice", ["row_present30", "starts15", "row15_ends", "axes_hw"], ["row_present15"]),
        helper.make_node("Slice", ["col_present30", "starts15", "col15_ends", "axes_hw"], ["col_present15"]),
        helper.make_node("Cast", ["row_present15"], ["row_valid"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["col_present15"], ["col_valid"], to=onnx.TensorProto.BOOL),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_area"]),
        helper.make_node("Conv", ["input", "row1_w"], ["row1_full"]),
        helper.make_node("Conv", ["input", "row3_w"], ["row3_full"]),
        helper.make_node("Conv", ["input", "col2_w"], ["col2_full"]),
        helper.make_node("Slice", ["row1_full", "starts15", "row15_ends", "axes_hw"], ["row_has_1_f32"]),
        helper.make_node("Slice", ["row3_full", "starts15", "row15_ends", "axes_hw"], ["row_has_3_f32"]),
        helper.make_node("Slice", ["col2_full", "starts15", "col15_ends", "axes_hw"], ["col_has_2_f32"]),
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

    graph = helper.make_graph(nodes, "task024_conv_dense_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
