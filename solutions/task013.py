from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _color_at_coord(nodes: list[onnx.NodeProto], row: str, col: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Reshape", [row, "shape1"], [f"{output}_row1"]),
            helper.make_node("Reshape", [col, "shape1"], [f"{output}_col1"]),
            helper.make_node("Unsqueeze", [f"{output}_row1", "unsq_axis1"], [f"{output}_row11"]),
            helper.make_node("Unsqueeze", [f"{output}_col1", "unsq_axis1"], [f"{output}_col11"]),
            helper.make_node("Concat", [f"{output}_row11", f"{output}_col11"], [f"{output}_indices12"], axis=1),
            helper.make_node("Reshape", [f"{output}_indices12", "shape112"], [f"{output}_spatial_indices"]),
            helper.make_node("Unsqueeze", [f"{output}_spatial_indices", "unsq_batch_axes"], [f"{output}_indices_batched"]),
            helper.make_node("Expand", [f"{output}_indices_batched", "gathernd_index_shape"], [f"{output}_indices"]),
            helper.make_node("GatherND", ["input", f"{output}_indices"], [f"{output}_onehot"], batch_dims=2),
            helper.make_node("ArgMax", [f"{output}_onehot"], [f"{output}_i64"], axis=1, keepdims=1),
            helper.make_node("Cast", [f"{output}_i64"], [output], to=onnx.TensorProto.UINT8),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("input0_start", [0, 0, 0], [3]),
        _int64_tensor("input0_end", [1, 30, 30], [3]),
        _int64_tensor("axes3", [1, 2, 3], [3]),
        _int64_tensor("row_idx", list(range(30)), [1, 1, 30, 1]),
        _int64_tensor("col_idx", list(range(30)), [1, 1, 1, 30]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("two_i64", [2], [1]),
        _int64_tensor("zero_i64", [0], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("shape112", [1, 1, 2], [3]),
        _int64_tensor("unsq_axis1", [1], [1]),
        _int64_tensor("unsq_batch_axes", [0, 1], [2]),
        _int64_tensor("gathernd_index_shape", [1, 10, 1, 1, 2], [5]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("outside_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["row_present"], axes=[1, 3], keepdims=1),
        helper.make_node("ReduceMax", ["input"], ["col_present"], axes=[1, 2], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["last_row"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present"], ["last_col"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("LessOrEqual", ["row_idx", "last_row"], ["row_valid"]),
        helper.make_node("LessOrEqual", ["col_idx", "last_col"], ["col_valid"]),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_area"]),
        helper.make_node("LessOrEqual", ["last_row", "last_col"], ["wide_bool"]),
        helper.make_node("Slice", ["input", "input0_start", "input0_end", "axes3"], ["input0"]),
        helper.make_node("Equal", ["input0", "zero_f32"], ["nonzero_raw"]),
        helper.make_node("And", ["valid_area", "nonzero_raw"], ["nonblack_bool"]),
        helper.make_node("Cast", ["nonblack_bool"], ["nonblack_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["nonblack_u8"], ["point_cols"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["point_cols"], ["c0"], axis=3, keepdims=1, select_last_index=0),
        helper.make_node("ArgMax", ["point_cols"], ["c1"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("ReduceMax", ["nonblack_u8"], ["point_rows"], axes=[3], keepdims=1),
        helper.make_node("ArgMax", ["point_rows"], ["r0"], axis=2, keepdims=1, select_last_index=0),
        helper.make_node("ArgMax", ["point_rows"], ["r1"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("Equal", ["col_idx", "c0"], ["c0_mask_line"]),
        helper.make_node("And", ["nonblack_bool", "c0_mask_line"], ["c0_point_mask"]),
        helper.make_node("Equal", ["col_idx", "c1"], ["c1_mask_line"]),
        helper.make_node("And", ["nonblack_bool", "c1_mask_line"], ["c1_point_mask"]),
        helper.make_node("Equal", ["row_idx", "r0"], ["r0_mask_line"]),
        helper.make_node("And", ["nonblack_bool", "r0_mask_line"], ["r0_point_mask"]),
        helper.make_node("Equal", ["row_idx", "r1"], ["r1_mask_line"]),
        helper.make_node("And", ["nonblack_bool", "r1_mask_line"], ["r1_point_mask"]),
        helper.make_node("Cast", ["c0_point_mask"], ["c0_point_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["c1_point_mask"], ["c1_point_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["r0_point_mask"], ["r0_point_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["r1_point_mask"], ["r1_point_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["c0_point_u8"], ["c0_point_rows"], axes=[3], keepdims=1),
        helper.make_node("ArgMax", ["c0_point_rows"], ["c0_row"], axis=2, keepdims=1, select_last_index=0),
        helper.make_node("ReduceMax", ["c1_point_u8"], ["c1_point_rows"], axes=[3], keepdims=1),
        helper.make_node("ArgMax", ["c1_point_rows"], ["c1_row"], axis=2, keepdims=1, select_last_index=0),
        helper.make_node("ReduceMax", ["r0_point_u8"], ["r0_point_cols"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["r0_point_cols"], ["r0_col"], axis=3, keepdims=1, select_last_index=0),
        helper.make_node("ReduceMax", ["r1_point_u8"], ["r1_point_cols"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["r1_point_cols"], ["r1_col"], axis=3, keepdims=1, select_last_index=0),
    ]
    _color_at_coord(nodes, "c0_row", "c0", "c0_color")
    _color_at_coord(nodes, "c1_row", "c1", "c1_color")
    _color_at_coord(nodes, "r0", "r0_col", "r0_color")
    _color_at_coord(nodes, "r1", "r1_col", "r1_color")

    nodes.extend(
        [
            helper.make_node("Sub", ["c1", "c0"], ["dc"]),
            helper.make_node("Equal", ["dc", "zero_i64"], ["dc_is_zero"]),
            helper.make_node("Where", ["dc_is_zero", "one_i64", "dc"], ["dc_safe"]),
            helper.make_node("Sub", ["col_idx", "c0"], ["col_from_start"]),
            helper.make_node("LessOrEqual", ["c0", "col_idx"], ["col_after_start"]),
            helper.make_node("LessOrEqual", ["col_idx", "last_col"], ["col_in_bounds"]),
            helper.make_node("Mod", ["col_from_start", "dc_safe"], ["col_mod"]),
            helper.make_node("Equal", ["col_mod", "zero_i64"], ["col_on_period"]),
            helper.make_node("Div", ["col_from_start", "dc_safe"], ["col_step"]),
            helper.make_node("Mod", ["col_step", "two_i64"], ["col_parity"]),
            helper.make_node("Equal", ["col_parity", "zero_i64"], ["col_even"]),
            helper.make_node("And", ["col_after_start", "col_in_bounds"], ["col_valid_period_area"]),
            helper.make_node("And", ["col_valid_period_area", "col_on_period"], ["target_cols"]),
            helper.make_node("Where", ["col_even", "c0_color", "c1_color"], ["h_color_line"]),
            helper.make_node("Where", ["target_cols", "h_color_line", "zero_u8"], ["h_color_grid"]),
            helper.make_node("Sub", ["r1", "r0"], ["dr"]),
            helper.make_node("Equal", ["dr", "zero_i64"], ["dr_is_zero"]),
            helper.make_node("Where", ["dr_is_zero", "one_i64", "dr"], ["dr_safe"]),
            helper.make_node("Sub", ["row_idx", "r0"], ["row_from_start"]),
            helper.make_node("LessOrEqual", ["r0", "row_idx"], ["row_after_start"]),
            helper.make_node("LessOrEqual", ["row_idx", "last_row"], ["row_in_bounds"]),
            helper.make_node("Mod", ["row_from_start", "dr_safe"], ["row_mod"]),
            helper.make_node("Equal", ["row_mod", "zero_i64"], ["row_on_period"]),
            helper.make_node("Div", ["row_from_start", "dr_safe"], ["row_step"]),
            helper.make_node("Mod", ["row_step", "two_i64"], ["row_parity"]),
            helper.make_node("Equal", ["row_parity", "zero_i64"], ["row_even"]),
            helper.make_node("And", ["row_after_start", "row_in_bounds"], ["row_valid_period_area"]),
            helper.make_node("And", ["row_valid_period_area", "row_on_period"], ["target_rows"]),
            helper.make_node("Where", ["row_even", "r0_color", "r1_color"], ["v_color_line"]),
            helper.make_node("Where", ["target_rows", "v_color_line", "zero_u8"], ["v_color_grid"]),
            helper.make_node("Where", ["wide_bool", "h_color_grid", "v_color_grid"], ["color_grid"]),
            helper.make_node("Where", ["valid_area", "color_grid", "outside_u8"], ["color30"]),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task013_periodic_lines_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
