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
        _int64_tensor("eight_starts", [0, 8, 0, 0], [4]),
        _int64_tensor("eight_ends", [1, 9, SIZE, SIZE], [4]),
        _int64_tensor("top_left_color_starts", [0, 0], [2]),
        _int64_tensor("top_color_ends", [1, SIZE], [2]),
        _int64_tensor("bottom_color_starts", [SIZE - 1, 0], [2]),
        _int64_tensor("bottom_right_color_ends", [SIZE, SIZE], [2]),
        _int64_tensor("left_color_ends", [SIZE, 1], [2]),
        _int64_tensor("right_color_starts", [0, SIZE - 1], [2]),
        _int64_tensor("row1_starts", [1], [1]),
        _int64_tensor("row4_i64", [4], [1]),
        _int64_tensor("row10_ends", [SIZE], [1]),
        _int64_tensor("col0_starts", [0], [1]),
        _int64_tensor("col9_ends", [SIZE - 1], [1]),
        _int64_tensor("axis_row", [2], [1]),
        _int64_tensor("axis_col", [3], [1]),
        _int64_tensor("pad_axes_hw", [2, 3], [2]),
        _int64_tensor("output_pads", [0, 0, 20, 20], [4]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("zero_middle", [0] * 64, [1, 1, 8, 8]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _bool_tensor("false_cell", [False], [1, 1, 1, 1]),
        _bool_tensor("right_col5", [c == 5 for c in range(SIZE)], [1, 1, 1, SIZE]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "eight_starts", "eight_ends"], ["is_eight_f32"]),
        helper.make_node("Cast", ["is_eight_f32"], ["is_eight"], to=onnx.TensorProto.BOOL),
        helper.make_node("ReduceMax", ["is_eight_f32", "axis_col"], ["row_any_f32"], keepdims=1),
        helper.make_node("Cast", ["row_any_f32"], ["row_any"], to=onnx.TensorProto.BOOL),
        helper.make_node("Slice", ["row_any", "row1_starts", "row10_ends", "axis_row"], ["below_core"]),
        helper.make_node("Concat", ["below_core", "false_cell"], ["below"], axis=2),
        helper.make_node("Not", ["below"], ["not_below"]),
        helper.make_node("And", ["row_any", "not_below"], ["bottom_edge"]),
        helper.make_node("ReduceMax", ["is_eight_f32", "axis_row"], ["col_any_f32"], keepdims=1),
        helper.make_node("Cast", ["col_any_f32"], ["col_any"], to=onnx.TensorProto.BOOL),
        helper.make_node("Slice", ["col_any", "col0_starts", "col9_ends", "axis_col"], ["left_core"]),
        helper.make_node("Concat", ["false_cell", "left_core"], ["left_neighbor"], axis=3),
        helper.make_node("Not", ["left_neighbor"], ["not_left"]),
        helper.make_node("And", ["col_any", "not_left"], ["left_edge"]),
        helper.make_node("Where", ["is_eight", "eight_u8", "zero_u8"], ["base0"]),
    ]

    color_slices = (
        ("top", "top_left_color_starts", "top_color_ends"),
        ("bottom", "bottom_color_starts", "bottom_right_color_ends"),
        ("left", "top_left_color_starts", "left_color_ends"),
        ("right", "right_color_starts", "bottom_right_color_ends"),
    )
    for name, starts, ends in color_slices:
        nodes.extend(
            [
                helper.make_node("Slice", ["input", starts, ends, "pad_axes_hw"], [f"{name}_onehot"]),
                helper.make_node("ArgMax", [f"{name}_onehot"], [f"{name}_i64"], axis=1, keepdims=1),
                helper.make_node("Cast", [f"{name}_i64"], [f"{name}_u8"], to=onnx.TensorProto.UINT8),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Slice", ["left_u8", "row1_starts", "col9_ends", "axis_row"], ["left_mid"]),
            helper.make_node("Slice", ["right_u8", "row1_starts", "col9_ends", "axis_row"], ["right_mid"]),
            helper.make_node("Concat", ["left_mid", "zero_middle", "right_mid"], ["orig_middle"], axis=3),
            helper.make_node("Concat", ["top_u8", "orig_middle", "bottom_u8"], ["orig_grid"], axis=2),
            helper.make_node("Max", ["base0", "orig_grid"], ["color"]),
            helper.make_node("Slice", ["color", "col0_starts", "axis_col", "axis_row"], ["color_rows0_2"]),
            helper.make_node("Slice", ["color", "axis_col", "row4_i64", "axis_row"], ["color_row3"]),
            helper.make_node("Slice", ["color", "row4_i64", "row10_ends", "axis_row"], ["color_rows4_9"]),
            helper.make_node("Greater", ["top_u8", "zero_u8"], ["top_present"]),
            helper.make_node("Where", ["top_present", "top_u8", "color_row3"], ["top_row"]),
            helper.make_node("Concat", ["color_rows0_2", "top_row", "color_rows4_9"], ["after_top"], axis=2),
            helper.make_node("Greater", ["bottom_u8", "zero_u8"], ["bottom_present"]),
            helper.make_node("And", ["bottom_edge", "bottom_present"], ["bottom_mask"]),
            helper.make_node("Where", ["bottom_mask", "bottom_u8", "after_top"], ["after_bottom"]),
            helper.make_node("Greater", ["left_u8", "zero_u8"], ["left_present"]),
            helper.make_node("And", ["left_edge", "left_present"], ["left_mask"]),
            helper.make_node("Where", ["left_mask", "left_u8", "after_bottom"], ["after_left"]),
            helper.make_node("Greater", ["right_u8", "zero_u8"], ["right_present"]),
            helper.make_node("And", ["right_col5", "right_present"], ["right_mask"]),
            helper.make_node("Where", ["right_mask", "right_u8", "after_left"], ["color10"]),
            helper.make_node("Pad", ["color10", "output_pads", "invalid_u8", "pad_axes_hw"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task035_rowcol_edge_masks", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
