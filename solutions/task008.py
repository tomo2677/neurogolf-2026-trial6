from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 16
CROP = 5
ANCHOR_MAX = SIZE - CROP


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _int32_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT32, dims, values)


def _uint8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _gt(nodes: list[onnx.NodeProto], left: str, right: str, output: str) -> None:
    nodes.append(helper.make_node("Greater", [left, right], [output]))


def _bbox_axis(nodes: list[onnx.NodeProto], mask: str, prefix: str, axis: str) -> tuple[str, str]:
    coord_axis = 2 if axis == "row" else 3
    reduce_axes = "pad_axis_col" if axis == "row" else "pad_axis_row"
    nodes.extend(
        [
            helper.make_node("ReduceMax", [mask, reduce_axes], [f"{prefix}_{axis}_present"], keepdims=1),
            helper.make_node(
                "ArgMax",
                [f"{prefix}_{axis}_present"],
                [f"{prefix}_{axis}_first_i64"],
                axis=coord_axis,
                keepdims=1,
                select_last_index=0,
            ),
            helper.make_node(
                "ArgMax",
                [f"{prefix}_{axis}_present"],
                [f"{prefix}_{axis}_last_i64"],
                axis=coord_axis,
                keepdims=1,
                select_last_index=1,
            ),
            helper.make_node("Cast", [f"{prefix}_{axis}_first_i64"], [f"{prefix}_{axis}_first"], to=onnx.TensorProto.INT32),
            helper.make_node("Cast", [f"{prefix}_{axis}_last_i64"], [f"{prefix}_{axis}_last"], to=onnx.TensorProto.INT32),
        ]
    )
    return f"{prefix}_{axis}_first", f"{prefix}_{axis}_last"


def _bbox_axis_last(nodes: list[onnx.NodeProto], mask: str, prefix: str, axis: str) -> str:
    coord_axis = 2 if axis == "row" else 3
    reduce_axes = "pad_axis_col" if axis == "row" else "pad_axis_row"
    nodes.extend(
        [
            helper.make_node("ReduceMax", [mask, reduce_axes], [f"{prefix}_{axis}_present"], keepdims=1),
            helper.make_node(
                "ArgMax",
                [f"{prefix}_{axis}_present"],
                [f"{prefix}_{axis}_last_i64"],
                axis=coord_axis,
                keepdims=1,
                select_last_index=1,
            ),
            helper.make_node("Cast", [f"{prefix}_{axis}_last_i64"], [f"{prefix}_{axis}_last"], to=onnx.TensorProto.INT32),
        ]
    )
    return f"{prefix}_{axis}_last"


def _bbox_axis_first_blue(nodes: list[onnx.NodeProto], mask: str, axis: str) -> tuple[str, str]:
    coord_axis = 2 if axis == "row" else 3
    reduce_axes = "pad_axis_col" if axis == "row" else "pad_axis_row"
    nodes.extend(
        [
            helper.make_node("ReduceMax", [mask, reduce_axes], [f"blue_{axis}_present"], keepdims=1),
            helper.make_node(
                "ArgMax",
                [f"blue_{axis}_present"],
                [f"blue_{axis}_first_i64"],
                axis=coord_axis,
                keepdims=1,
                select_last_index=0,
            ),
            helper.make_node("Cast", [f"blue_{axis}_first_i64"], [f"blue_{axis}_first"], to=onnx.TensorProto.INT32),
            helper.make_node("Add", [f"blue_{axis}_first", "one1111"], [f"blue_{axis}_last"]),
        ]
    )
    return f"blue_{axis}_first", f"blue_{axis}_last"


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("starts_black_bottom", [0, 0, 2]),
        _int64_tensor("ends_black_bottom", [1, SIZE, 5]),
        _int64_tensor("starts_black_right", [0, 0, 0]),
        _int64_tensor("ends_black_right", [1, 2, SIZE]),
        _int64_tensor("starts_red", [2, 0, 0]),
        _int64_tensor("ends_red", [3, SIZE, SIZE]),
        _int64_tensor("starts_blue", [8, 0, 0]),
        _int64_tensor("ends_blue", [9, SIZE - 2, SIZE - 1]),
        _int64_tensor("axes3", [1, 2, 3]),
        _int64_tensor("index_shape", [1]),
        _int64_tensor("pad_axis_row", [2]),
        _int64_tensor("pad_axis_col", [3]),
        _int64_tensor("pad_axes_spatial", [2, 3]),
        _int32_tensor("slice_axes_spatial_i32", [2, 3], [2]),
        _int64_tensor("pads_color_grid", [0, 0, 30 - SIZE, 30 - SIZE]),
        _int32_tensor("coord_rows_i32", list(range(SIZE)), [1, 1, SIZE, 1]),
        _int32_tensor("coord_cols_i32", list(range(SIZE)), [1, 1, 1, SIZE]),
        _uint8_tensor("channel_ids_u8", list(range(10)), [1, 10, 1, 1]),
        _uint8_tensor("zero_u8", [0], [1, 1, 1, 1]),
        _uint8_tensor("two_u8", [2], [1, 1, 1, 1]),
        _uint8_tensor("outside_u8", [255], [1]),
        _uint8_tensor("blue_square_u8", [8, 8, 8, 8], [1, 1, 2, 2]),
        _int32_tensor("zero1111", [0], [1, 1, 1, 1]),
        _int32_tensor("anchor_max1111", [ANCHOR_MAX], [1, 1, 1, 1]),
        _int32_tensor("blue_anchor_max1111", [SIZE - 2], [1, 1, 1, 1]),
        _int32_tensor("crop_size2", [CROP, CROP], [2]),
        _int32_tensor("one1111", [1], [1, 1, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "starts_black_bottom", "ends_black_bottom", "axes3"], ["black_bottom_probe"]),
        helper.make_node("Slice", ["input", "starts_black_right", "ends_black_right", "axes3"], ["black_right_probe"]),
        helper.make_node("Slice", ["input", "starts_red", "ends_red", "axes3"], ["red"]),
        helper.make_node("Slice", ["input", "starts_blue", "ends_blue", "axes3"], ["blue"]),
    ]

    red_top, red_bottom = _bbox_axis(nodes, "red", "red", "row")
    red_left, red_right = _bbox_axis(nodes, "red", "red", "col")
    blue_top, blue_bottom = _bbox_axis_first_blue(nodes, "blue", "row")
    blue_left, blue_right = _bbox_axis_first_blue(nodes, "blue", "col")
    black_bottom = _bbox_axis_last(nodes, "black_bottom_probe", "black", "row")
    black_right = _bbox_axis_last(nodes, "black_right_probe", "black", "col")

    _gt(nodes, blue_top, red_bottom, "move_down_bool")
    _gt(nodes, red_top, blue_bottom, "move_up_bool")
    _gt(nodes, blue_left, red_right, "move_right_bool")
    _gt(nodes, red_left, blue_right, "move_left_bool")

    nodes.extend(
        [
            helper.make_node("Sub", [blue_top, red_bottom], ["down_gap_plus_one"]),
            helper.make_node("Sub", ["down_gap_plus_one", "one1111"], ["down_offset"]),
            helper.make_node("Sub", [blue_bottom, red_top], ["up_gap_minus_one"]),
            helper.make_node("Add", ["up_gap_minus_one", "one1111"], ["up_offset"]),
            helper.make_node("Sub", [blue_left, red_right], ["right_gap_plus_one"]),
            helper.make_node("Sub", ["right_gap_plus_one", "one1111"], ["right_offset"]),
            helper.make_node("Sub", [blue_right, red_left], ["left_gap_minus_one"]),
            helper.make_node("Add", ["left_gap_minus_one", "one1111"], ["left_offset"]),
            helper.make_node("LessOrEqual", ["coord_rows_i32", black_bottom], ["valid_rows"]),
            helper.make_node("LessOrEqual", ["coord_cols_i32", black_right], ["valid_cols"]),
            helper.make_node("And", ["valid_rows", "valid_cols"], ["valid_bool"]),
            helper.make_node("Where", ["move_down_bool", "down_offset", "up_offset"], ["dy_selected"]),
            helper.make_node("Or", ["move_down_bool", "move_up_bool"], ["vertical_move_bool"]),
            helper.make_node("Where", ["vertical_move_bool", "dy_selected", "zero1111"], ["dy_offset"]),
            helper.make_node("Where", ["move_right_bool", "right_offset", "left_offset"], ["dx_selected"]),
            helper.make_node("Or", ["move_right_bool", "move_left_bool"], ["horizontal_move_bool"]),
            helper.make_node("Where", ["horizontal_move_bool", "dx_selected", "zero1111"], ["dx_offset"]),
            helper.make_node("Min", [red_top, "anchor_max1111"], ["source_top"]),
            helper.make_node("Min", [red_left, "anchor_max1111"], ["source_left"]),
            helper.make_node("Sub", [red_top, "source_top"], ["source_row_delta"]),
            helper.make_node("Sub", [red_left, "source_left"], ["source_col_delta"]),
            helper.make_node("Add", [red_top, "dy_offset"], ["target_top_raw"]),
            helper.make_node("Add", [red_left, "dx_offset"], ["target_left_raw"]),
            helper.make_node("Sub", ["target_top_raw", "source_row_delta"], ["target_top"]),
            helper.make_node("Sub", ["target_left_raw", "source_col_delta"], ["target_left"]),
            helper.make_node("Sub", ["anchor_max1111", "target_top"], ["target_bottom_pad"]),
            helper.make_node("Sub", ["anchor_max1111", "target_left"], ["target_right_pad"]),
            helper.make_node("Reshape", ["source_top", "index_shape"], ["source_top_1"]),
            helper.make_node("Reshape", ["source_left", "index_shape"], ["source_left_1"]),
            helper.make_node("Concat", ["source_top_1", "source_left_1"], ["crop_start"], axis=0),
            helper.make_node("Add", ["crop_start", "crop_size2"], ["crop_end"]),
            helper.make_node("Slice", ["red", "crop_start", "crop_end", "slice_axes_spatial_i32"], ["red_crop_float"]),
            helper.make_node("Cast", ["red_crop_float"], ["red_crop"], to=onnx.TensorProto.BOOL),
            helper.make_node("Reshape", ["target_top", "index_shape"], ["target_top_1"]),
            helper.make_node("Reshape", ["target_left", "index_shape"], ["target_left_1"]),
            helper.make_node("Reshape", ["target_bottom_pad", "index_shape"], ["target_bottom_pad_1"]),
            helper.make_node("Reshape", ["target_right_pad", "index_shape"], ["target_right_pad_1"]),
            helper.make_node(
                "Concat",
                ["target_top_1", "target_left_1", "target_bottom_pad_1", "target_right_pad_1"],
                ["target_pads_i32"],
                axis=0,
            ),
            helper.make_node("Cast", ["target_pads_i32"], ["target_pads_i64"], to=onnx.TensorProto.INT64),
            helper.make_node("Where", ["red_crop", "two_u8", "zero_u8"], ["red_crop_color_u8"]),
            helper.make_node("Pad", ["red_crop_color_u8", "target_pads_i64", "zero_u8", "pad_axes_spatial"], ["red_color_u8"], mode="constant"),
            helper.make_node("Sub", ["blue_anchor_max1111", blue_top], ["blue_bottom_pad"]),
            helper.make_node("Sub", ["blue_anchor_max1111", blue_left], ["blue_right_pad"]),
            helper.make_node("Reshape", [blue_top, "index_shape"], ["blue_top_1"]),
            helper.make_node("Reshape", [blue_left, "index_shape"], ["blue_left_1"]),
            helper.make_node("Reshape", ["blue_bottom_pad", "index_shape"], ["blue_bottom_pad_1"]),
            helper.make_node("Reshape", ["blue_right_pad", "index_shape"], ["blue_right_pad_1"]),
            helper.make_node(
                "Concat",
                ["blue_top_1", "blue_left_1", "blue_bottom_pad_1", "blue_right_pad_1"],
                ["blue_pads_i32"],
                axis=0,
            ),
            helper.make_node("Cast", ["blue_pads_i32"], ["blue_pads_i64"], to=onnx.TensorProto.INT64),
            helper.make_node("Pad", ["blue_square_u8", "blue_pads_i64", "zero_u8", "pad_axes_spatial"], ["blue_color_u8"], mode="constant"),
            helper.make_node("Max", ["red_color_u8", "blue_color_u8"], ["color_nonblack_u8"]),
            helper.make_node("Where", ["valid_bool", "color_nonblack_u8", "outside_u8"], ["color16_u8"]),
            helper.make_node("Pad", ["color16_u8", "pads_color_grid", "outside_u8", "pad_axes_spatial"], ["color_grid_u8"], mode="constant"),
            helper.make_node("Equal", ["channel_ids_u8", "color_grid_u8"], ["output"]),
        ]
    )

    value_infos = [
        helper.make_tensor_value_info("red_crop_float", onnx.TensorProto.FLOAT, [1, 1, CROP, CROP]),
        helper.make_tensor_value_info("red_crop", onnx.TensorProto.BOOL, [1, 1, CROP, CROP]),
        helper.make_tensor_value_info("red_crop_color_u8", onnx.TensorProto.UINT8, [1, 1, CROP, CROP]),
        helper.make_tensor_value_info("red_color_u8", onnx.TensorProto.UINT8, [1, 1, SIZE, SIZE]),
        helper.make_tensor_value_info("blue_color_u8", onnx.TensorProto.UINT8, [1, 1, SIZE, SIZE]),
    ]
    graph = helper.make_graph(nodes, "task008_black_probe_graph", [x], [y], initializers, value_info=value_infos)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
