from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 10


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _bool_tensor(name: str, values: list[bool], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.BOOL, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("input10_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("input10_ends", [1, 10, SIZE, SIZE], [4]),
        _int64_tensor("eight_starts", [0, 8, 0, 0], [4]),
        _int64_tensor("eight_ends", [1, 9, SIZE, SIZE], [4]),
        _int64_tensor("top_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("top_ends", [1, 1, 1, SIZE], [4]),
        _int64_tensor("bottom_starts", [0, 0, SIZE - 1, 0], [4]),
        _int64_tensor("bottom_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("left_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("left_ends", [1, 1, SIZE, 1], [4]),
        _int64_tensor("right_starts", [0, 0, 0, SIZE - 1], [4]),
        _int64_tensor("right_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("row1_starts", [1], [1]),
        _int64_tensor("row10_ends", [SIZE], [1]),
        _int64_tensor("col1_starts", [1], [1]),
        _int64_tensor("col10_ends", [SIZE], [1]),
        _int64_tensor("col0_starts", [0], [1]),
        _int64_tensor("col9_ends", [SIZE - 1], [1]),
        _int64_tensor("axis_row", [2], [1]),
        _int64_tensor("axis_col", [3], [1]),
        _int64_tensor("shape_10x10", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("top_pads", [0, 0, 3, 0, 0, 0, 6, 0], [8]),
        _int64_tensor("output_pads", [0, 0, 0, 0, 0, 0, 20, 20], [8]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _bool_tensor("false_row", [False] * SIZE, [1, 1, 1, SIZE]),
        _bool_tensor("false_col", [False] * SIZE, [1, 1, SIZE, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "input10_starts", "input10_ends"], ["input10"]),
        helper.make_node("ArgMax", ["input10"], ["color_i64"], axis=1, keepdims=1),
        helper.make_node("Cast", ["color_i64"], ["color"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["input", "eight_starts", "eight_ends"], ["is_eight_f32"]),
        helper.make_node("Cast", ["is_eight_f32"], ["is_eight"], to=onnx.TensorProto.BOOL),
        helper.make_node("Slice", ["is_eight", "row1_starts", "row10_ends", "axis_row"], ["below_core"]),
        helper.make_node("Concat", ["below_core", "false_row"], ["below"], axis=2),
        helper.make_node("Not", ["below"], ["not_below"]),
        helper.make_node("And", ["is_eight", "not_below"], ["bottom_edge"]),
        helper.make_node("Slice", ["is_eight", "col0_starts", "col9_ends", "axis_col"], ["left_core"]),
        helper.make_node("Concat", ["false_col", "left_core"], ["left_neighbor"], axis=3),
        helper.make_node("Not", ["left_neighbor"], ["not_left"]),
        helper.make_node("And", ["is_eight", "not_left"], ["left_edge"]),
        helper.make_node("Slice", ["is_eight", "col1_starts", "col10_ends", "axis_col"], ["right_core"]),
        helper.make_node("Concat", ["right_core", "false_col"], ["right_neighbor"], axis=3),
        helper.make_node("Not", ["right_neighbor"], ["not_right"]),
        helper.make_node("And", ["is_eight", "not_right"], ["right_edge"]),
        helper.make_node("Slice", ["color", "top_starts", "top_ends"], ["top_row"]),
        helper.make_node("Pad", ["top_row", "top_pads", "zero_u8"], ["top_color"], mode="constant"),
        helper.make_node("Greater", ["top_color", "zero_u8"], ["top_present"]),
        helper.make_node("And", ["is_eight", "top_present"], ["top_mask"]),
        helper.make_node("Where", ["top_mask", "top_color", "color"], ["after_top"]),
        helper.make_node("Slice", ["color", "bottom_starts", "bottom_ends"], ["bottom_row"]),
        helper.make_node("Expand", ["bottom_row", "shape_10x10"], ["bottom_color"]),
        helper.make_node("Greater", ["bottom_color", "zero_u8"], ["bottom_present"]),
        helper.make_node("And", ["bottom_edge", "bottom_present"], ["bottom_mask"]),
        helper.make_node("Where", ["bottom_mask", "bottom_color", "after_top"], ["after_bottom"]),
        helper.make_node("Slice", ["color", "left_starts", "left_ends"], ["left_col"]),
        helper.make_node("Expand", ["left_col", "shape_10x10"], ["left_color"]),
        helper.make_node("Greater", ["left_color", "zero_u8"], ["left_present"]),
        helper.make_node("And", ["left_edge", "left_present"], ["left_mask"]),
        helper.make_node("Where", ["left_mask", "left_color", "after_bottom"], ["after_left"]),
        helper.make_node("Slice", ["color", "right_starts", "right_ends"], ["right_col"]),
        helper.make_node("Expand", ["right_col", "shape_10x10"], ["right_color"]),
        helper.make_node("Greater", ["right_color", "zero_u8"], ["right_present"]),
        helper.make_node("And", ["right_edge", "right_present"], ["right_mask"]),
        helper.make_node("Where", ["right_mask", "right_color", "after_left"], ["color10"]),
        helper.make_node("Pad", ["color10", "output_pads", "invalid_u8"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task035_edge_marker_projection_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
