from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
SELECTED_LINES = 4


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _shift_bool(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    source: str,
    output: str,
    dr: int,
    dc: int,
) -> str:
    pad_name = {
        (-1, 0): "shift_above_pads",
        (1, 0): "shift_below_pads",
        (0, -1): "shift_left_pads",
        (0, 1): "shift_right_pads",
    }[(dr, dc)]
    nodes.append(helper.make_node("Pad", [source, pad_name], [output], mode="constant"))
    return output


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _u8_tensor("colors9", list(range(1, 10)), [1, 9, 1, 1]),
        _int64_tensor("k4", [SELECTED_LINES], [1]),
        _int64_tensor("split4", [1] * SELECTED_LINES, [SELECTED_LINES]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("grid_shape_i64", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("line_shape_i64", [1, 1, SIZE, 1], [4]),
        _int64_tensor("shift_above_pads", [0, 0, -1, 0, 0, 0, 1, 0], [8]),
        _int64_tensor("shift_below_pads", [0, 0, 1, 0, 0, 0, -1, 0], [8]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ArgMax", ["input"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Cast", ["input_color_i64"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
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
        helper.make_node("Equal", ["colors9", "row_line_color"], ["row_line_color_eq"]),
        helper.make_node("Equal", ["colors9", "col_line_color"], ["col_line_color_eq"]),
        helper.make_node("Cast", ["row_line_color_eq"], ["row_line_color_eq_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["col_line_color_eq"], ["col_line_color_eq_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["row_line_color_eq_u8"], ["row_line_present9"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("ReduceMax", ["col_line_color_eq_u8"], ["col_line_present9"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("Max", ["row_line_present9", "col_line_present9"], ["line_present9_u8"]),
        helper.make_node("Cast", ["line_present9_u8"], ["line_present9_f32"], to=onnx.TensorProto.FLOAT),
        helper.make_node("TopK", ["line_present9_f32", "k4"], ["line_top_values", "line_top_idx0"], axis=0, largest=1, sorted=0),
        helper.make_node("Split", ["line_top_idx0", "split4"], [f"line_top_idx0_{slot}" for slot in range(SELECTED_LINES)], axis=0),
        helper.make_node("ReduceMax", ["col_line_present9"], ["has_col_u8"], axes=[0], keepdims=1),
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
        helper.make_node("Where", ["canon_valid_area", "zero_u8", "invalid_u8"], ["color_grid_0"]),
    ]

    current = "color_grid_0"
    for slot in range(SELECTED_LINES):
        prefix = f"s{slot}"
        nodes.extend(
            [
                helper.make_node("Add", [f"line_top_idx0_{slot}", "one_i64"], [f"{prefix}_color_i64"]),
                helper.make_node("Cast", [f"{prefix}_color_i64"], [f"{prefix}_color_u8"], to=onnx.TensorProto.UINT8),
                helper.make_node("Equal", ["canon_input_color", f"{prefix}_color_u8"], [f"{prefix}_mask_bool"]),
                helper.make_node("Equal", ["canon_line_color", f"{prefix}_color_u8"], [f"{prefix}_row_line"]),
                helper.make_node("And", [f"{prefix}_row_line", "canon_valid_area"], [f"{prefix}_line_cover_bool"]),
                helper.make_node("Not", [f"{prefix}_line_cover_bool"], [f"{prefix}_not_line_bool"]),
                helper.make_node("And", [f"{prefix}_mask_bool", f"{prefix}_not_line_bool"], [f"{prefix}_scatter_bool"]),
                helper.make_node("Cast", [f"{prefix}_scatter_bool"], [f"{prefix}_scatter_u8"], to=onnx.TensorProto.UINT8),
                helper.make_node(
                    "MaxPool",
                    [f"{prefix}_scatter_u8"],
                    [f"{prefix}_up_seen_u8"],
                    kernel_shape=[SIZE, 1],
                    pads=[SIZE - 1, 0, 0, 0],
                ),
                helper.make_node(
                    "MaxPool",
                    [f"{prefix}_scatter_u8"],
                    [f"{prefix}_down_seen_u8"],
                    kernel_shape=[SIZE, 1],
                    pads=[0, 0, SIZE - 1, 0],
                ),
                helper.make_node("Greater", [f"{prefix}_up_seen_u8", "zero_u8"], [f"{prefix}_up_seen_bool"]),
                helper.make_node("Greater", [f"{prefix}_down_seen_u8", "zero_u8"], [f"{prefix}_down_seen_bool"]),
                helper.make_node("And", [f"{prefix}_up_seen_bool", f"{prefix}_line_cover_bool"], [f"{prefix}_above_line_bool"]),
                helper.make_node("And", [f"{prefix}_down_seen_bool", f"{prefix}_line_cover_bool"], [f"{prefix}_below_line_bool"]),
            ]
        )
        above_proj = _shift_bool(nodes, initializers, f"{prefix}_above_line_bool", f"{prefix}_above_proj", -1, 0)
        below_proj = _shift_bool(nodes, initializers, f"{prefix}_below_line_bool", f"{prefix}_below_proj", 1, 0)
        nodes.extend(
            [
                helper.make_node("Or", [f"{prefix}_line_cover_bool", above_proj], [f"{prefix}_cover_or0"]),
                helper.make_node("Or", [f"{prefix}_cover_or0", below_proj], [f"{prefix}_cover_raw_bool"]),
                helper.make_node("And", [f"{prefix}_cover_raw_bool", "canon_valid_area"], [f"{prefix}_cover_bool"]),
                helper.make_node("Where", [f"{prefix}_cover_bool", f"{prefix}_color_u8", current], [f"color_grid_{slot + 1}"]),
            ]
        )
        current = f"color_grid_{slot + 1}"

    nodes.extend(
        [
            helper.make_node("Transpose", [current], ["color_grid_t"], perm=[0, 1, 3, 2]),
            helper.make_node("Where", ["has_col_grid", "color_grid_t", current], ["color_grid_final"]),
            helper.make_node("Equal", ["colors10", "color_grid_final"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task025_top4_line_colors_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
