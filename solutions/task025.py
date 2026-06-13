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
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("ten_u8", [10], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _int64_tensor("grid_shape_i64", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("line_shape_i64", [1, 1, SIZE, 1], [4]),
        _int64_tensor("shift_above_pads", [0, 0, -1, 0, 0, 0, 1, 0], [8]),
        _int64_tensor("shift_below_pads", [0, 0, 1, 0, 0, 0, -1, 0], [8]),
        _f32_tensor("color_conv_w", [float(i) for i in range(10)], [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Conv", ["input", "color_conv_w"], ["input_color_f32"]),
        helper.make_node("Cast", ["input_color_f32"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["input"], ["cell_present"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["cell_present"], ["row_present_f32"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["cell_present"], ["col_present_f32"], axes=[2], keepdims=1),
        helper.make_node("Cast", ["row_present_f32"], ["row_valid"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["col_present_f32"], ["col_valid"], to=onnx.TensorProto.BOOL),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_area"]),
        helper.make_node("Where", ["valid_area", "input_color_u8", "invalid_u8"], ["input_color_min_base"]),
        helper.make_node("ReduceMin", ["input_color_min_base"], ["row_min_color"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["input_color_u8"], ["row_max_color"], axes=[3], keepdims=1),
        helper.make_node("ReduceMin", ["input_color_min_base"], ["col_min_color"], axes=[2], keepdims=1),
        helper.make_node("ReduceMax", ["input_color_u8"], ["col_max_color"], axes=[2], keepdims=1),
        helper.make_node("Equal", ["row_min_color", "row_max_color"], ["row_uniform"]),
        helper.make_node("Equal", ["col_min_color", "col_max_color"], ["col_uniform"]),
        helper.make_node("Where", ["row_uniform", "row_max_color", "zero_u8"], ["row_line_color"]),
        helper.make_node("Where", ["col_uniform", "col_max_color", "zero_u8"], ["col_line_color"]),
        helper.make_node("ReduceMax", ["col_line_color"], ["has_col_u8"], axes=[0, 1, 2, 3], keepdims=1),
        helper.make_node("Greater", ["has_col_u8", "zero_u8"], ["has_col"]),
        helper.make_node("Not", ["has_col"], ["not_has_col"]),
        helper.make_node("Expand", ["has_col", "grid_shape_i64"], ["has_col_grid"]),
        helper.make_node("Expand", ["not_has_col", "grid_shape_i64"], ["not_has_col_grid"]),
        helper.make_node("Expand", ["has_col", "line_shape_i64"], ["has_col_line"]),
        helper.make_node("Transpose", ["input_color_u8"], ["input_color_t"], perm=[0, 1, 3, 2]),
        helper.make_node("Transpose", ["valid_area"], ["valid_area_t"], perm=[0, 1, 3, 2]),
        helper.make_node("Transpose", ["col_line_color"], ["col_line_color_t"], perm=[0, 1, 3, 2]),
        helper.make_node("Where", ["has_col_grid", "input_color_t", "input_color_u8"], ["canon_input_color"]),
        helper.make_node("And", ["has_col_grid", "valid_area_t"], ["valid_area_t_selected"]),
        helper.make_node("And", ["not_has_col_grid", "valid_area"], ["valid_area_orig_selected"]),
        helper.make_node("Or", ["valid_area_t_selected", "valid_area_orig_selected"], ["canon_valid_area"]),
        helper.make_node("Where", ["has_col_line", "col_line_color_t", "row_line_color"], ["canon_line_color"]),
        helper.make_node("Greater", ["canon_line_color", "zero_u8"], ["line_row"]),
        helper.make_node("And", ["line_row", "canon_valid_area"], ["line_cover_bool"]),
        helper.make_node("Where", ["line_cover_bool", "zero_u8", "canon_input_color"], ["scatter_color"]),
        helper.make_node("Greater", ["scatter_color", "zero_u8"], ["scatter_present"]),
        helper.make_node("Sub", ["ten_u8", "scatter_color"], ["scatter_inv_raw"]),
        helper.make_node("Where", ["scatter_present", "scatter_inv_raw", "zero_u8"], ["scatter_inv_color"]),
        helper.make_node("Sub", ["ten_u8", "canon_line_color"], ["canon_line_inv_color"]),
        helper.make_node("Equal", ["scatter_color", "eight_u8"], ["scatter_color8_bool"]),
        helper.make_node("Cast", ["scatter_color8_bool"], ["scatter_color8_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node(
            "MaxPool",
            ["scatter_color"],
            ["up_color"],
            kernel_shape=[SIZE, 1],
            pads=[SIZE - 1, 0, 0, 0],
        ),
        helper.make_node(
            "MaxPool",
            ["scatter_color"],
            ["down_color"],
            kernel_shape=[SIZE, 1],
            pads=[0, 0, SIZE - 1, 0],
        ),
        helper.make_node(
            "MaxPool",
            ["scatter_inv_color"],
            ["up_inv_color"],
            kernel_shape=[SIZE, 1],
            pads=[SIZE - 1, 0, 0, 0],
        ),
        helper.make_node(
            "MaxPool",
            ["scatter_inv_color"],
            ["down_inv_color"],
            kernel_shape=[SIZE, 1],
            pads=[0, 0, SIZE - 1, 0],
        ),
        helper.make_node(
            "MaxPool",
            ["scatter_color8_u8"],
            ["up_color8_u8"],
            kernel_shape=[SIZE, 1],
            pads=[SIZE - 1, 0, 0, 0],
        ),
        helper.make_node(
            "MaxPool",
            ["scatter_color8_u8"],
            ["down_color8_u8"],
            kernel_shape=[SIZE, 1],
            pads=[0, 0, SIZE - 1, 0],
        ),
        helper.make_node("Equal", ["up_color", "canon_line_color"], ["up_color_match"]),
        helper.make_node("Equal", ["down_color", "canon_line_color"], ["down_color_match"]),
        helper.make_node("Equal", ["up_inv_color", "canon_line_inv_color"], ["up_inv_match"]),
        helper.make_node("Equal", ["down_inv_color", "canon_line_inv_color"], ["down_inv_match"]),
        helper.make_node("Or", ["up_color_match", "up_inv_match"], ["up_extreme_match"]),
        helper.make_node("Or", ["down_color_match", "down_inv_match"], ["down_extreme_match"]),
        helper.make_node("Greater", ["up_color8_u8", "zero_u8"], ["up_color8_seen"]),
        helper.make_node("Greater", ["down_color8_u8", "zero_u8"], ["down_color8_seen"]),
        helper.make_node("Equal", ["canon_line_color", "eight_u8"], ["line_color8"]),
        helper.make_node("And", ["up_color8_seen", "line_color8"], ["up_color8_match"]),
        helper.make_node("And", ["down_color8_seen", "line_color8"], ["down_color8_match"]),
        helper.make_node("Or", ["up_extreme_match", "up_color8_match"], ["up_match"]),
        helper.make_node("Or", ["down_extreme_match", "down_color8_match"], ["down_match"]),
        helper.make_node("And", ["up_match", "line_cover_bool"], ["above_line_bool"]),
        helper.make_node("And", ["down_match", "line_cover_bool"], ["below_line_bool"]),
        helper.make_node("Where", ["line_cover_bool", "canon_line_color", "zero_u8"], ["line_color_full"]),
        helper.make_node("Where", ["above_line_bool", "canon_line_color", "zero_u8"], ["above_line_color"]),
        helper.make_node("Where", ["below_line_bool", "canon_line_color", "zero_u8"], ["below_line_color"]),
        helper.make_node("Pad", ["above_line_color", "shift_above_pads", "zero_u8"], ["above_proj_color"], mode="constant"),
        helper.make_node("Pad", ["below_line_color", "shift_below_pads", "zero_u8"], ["below_proj_color"], mode="constant"),
        helper.make_node("Max", ["line_color_full", "above_proj_color", "below_proj_color"], ["canon_color_grid_raw"]),
        helper.make_node("Where", ["canon_valid_area", "canon_color_grid_raw", "invalid_u8"], ["canon_color_grid"]),
        helper.make_node("Transpose", ["canon_color_grid"], ["color_grid_t"], perm=[0, 1, 3, 2]),
        helper.make_node("Where", ["has_col_grid", "color_grid_t", "canon_color_grid"], ["color_grid_final"]),
        helper.make_node("Equal", ["colors10", "color_grid_final"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task025_color_pool_projection_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 14)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
