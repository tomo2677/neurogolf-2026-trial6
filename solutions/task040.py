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

    row_top_values = [r < 5 for r in range(SIZE) for _ in range(SIZE)]
    col_left_values = [c < 5 for _ in range(SIZE) for c in range(SIZE)]

    initializers = [
        _int64_tensor("input10_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("input10_ends", [1, 10, SIZE, SIZE], [4]),
        _int64_tensor("top_zero_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("top_zero_ends", [1, 1, 1, SIZE], [4]),
        _int64_tensor("top_left_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("top_left_ends", [1, 1, 1, 1], [4]),
        _int64_tensor("top_right_starts", [0, 0, 0, SIZE - 1], [4]),
        _int64_tensor("top_right_ends", [1, 1, 1, SIZE], [4]),
        _int64_tensor("bottom_left_starts", [0, 0, SIZE - 1, 0], [4]),
        _int64_tensor("bottom_left_ends", [1, 1, SIZE, 1], [4]),
        _int64_tensor("shape_10x10", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("output_pads", [0, 0, 0, 0, 0, 0, 20, 20], [8]),
        _u8_tensor("three_u8", [3], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _bool_tensor("row_top", row_top_values, [1, 1, SIZE, SIZE]),
        _bool_tensor("col_left", col_left_values, [1, 1, SIZE, SIZE]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "input10_starts", "input10_ends"], ["input10"]),
        helper.make_node("ArgMax", ["input10"], ["color_i64"], axis=1, keepdims=1),
        helper.make_node("Cast", ["color_i64"], ["color"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["input", "top_zero_starts", "top_zero_ends"], ["top_zero"]),
        helper.make_node("ReduceMax", ["top_zero"], ["top_has_zero_f32"], axes=[0, 1, 2, 3], keepdims=1),
        helper.make_node("Cast", ["top_has_zero_f32"], ["top_has_zero"], to=onnx.TensorProto.BOOL),
        helper.make_node("Not", ["top_has_zero"], ["use_rows"]),
        helper.make_node("Slice", ["color", "top_left_starts", "top_left_ends"], ["top_left_color"]),
        helper.make_node("Slice", ["color", "top_right_starts", "top_right_ends"], ["top_right_color"]),
        helper.make_node("Slice", ["color", "bottom_left_starts", "bottom_left_ends"], ["bottom_left_color"]),
        helper.make_node("Expand", ["top_left_color", "shape_10x10"], ["top_left_grid"]),
        helper.make_node("Expand", ["top_right_color", "shape_10x10"], ["top_right_grid"]),
        helper.make_node("Expand", ["bottom_left_color", "shape_10x10"], ["bottom_left_grid"]),
        helper.make_node("Where", ["row_top", "top_left_grid", "bottom_left_grid"], ["row_replacement"]),
        helper.make_node("Where", ["col_left", "top_left_grid", "top_right_grid"], ["col_replacement"]),
        helper.make_node("Where", ["use_rows", "row_replacement", "col_replacement"], ["replacement"]),
        helper.make_node("Equal", ["color", "three_u8"], ["marker"]),
        helper.make_node("Where", ["marker", "replacement", "color"], ["color10"]),
        helper.make_node("Pad", ["color10", "output_pads", "invalid_u8"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task040_nearest_guide_recolor_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
