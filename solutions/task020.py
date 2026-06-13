from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _gather_coords_bounded(nodes: list[onnx.NodeProto], src_r: str, src_c: str, output: str, *, swap_axes: bool) -> None:
    aligned = f"{output}_aligned" if swap_axes else f"{output}_raw"
    nodes.extend(
        [
            helper.make_node("GreaterOrEqual", [src_r, "zero_i32"], [f"{output}_r_ge_zero"]),
            helper.make_node("Less", [src_r, "size_i32"], [f"{output}_r_lt_size"]),
            helper.make_node("And", [f"{output}_r_ge_zero", f"{output}_r_lt_size"], [f"{output}_r_in"]),
            helper.make_node("GreaterOrEqual", [src_c, "zero_i32"], [f"{output}_c_ge_zero"]),
            helper.make_node("Less", [src_c, "size_i32"], [f"{output}_c_lt_size"]),
            helper.make_node("And", [f"{output}_c_ge_zero", f"{output}_c_lt_size"], [f"{output}_c_in"]),
            helper.make_node("Where", [f"{output}_r_in", src_r, "zero_i32"], [f"{output}_safe_r"]),
            helper.make_node("Where", [f"{output}_c_in", src_c, "zero_i32"], [f"{output}_safe_c"]),
            helper.make_node("Reshape", [f"{output}_safe_r", "shape_vec10"], [f"{output}_safe_r_vec"]),
            helper.make_node("Reshape", [f"{output}_safe_c", "shape_vec10"], [f"{output}_safe_c_vec"]),
            helper.make_node("Gather", ["input_color_u8", f"{output}_safe_r_vec"], [f"{output}_rows"], axis=2),
            helper.make_node("Gather", [f"{output}_rows", f"{output}_safe_c_vec"], [f"{output}_raw"], axis=3),
        ]
    )
    if swap_axes:
        nodes.append(helper.make_node("Transpose", [f"{output}_raw"], [aligned], perm=[0, 1, 3, 2]))
    nodes.extend(
        [
            helper.make_node("And", [f"{output}_r_in", f"{output}_c_in"], [f"{output}_in_bounds"]),
            helper.make_node("Where", [f"{output}_in_bounds", aligned, "zero_u8"], [output]),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("c1_starts", [1, 0, 0], [3]),
        _int64_tensor("c1_ends", [2, 10, 10], [3]),
        _int64_tensor("c2_starts", [2, 0, 0], [3]),
        _int64_tensor("c2_ends", [3, 10, 10], [3]),
        _int64_tensor("c3_starts", [3, 0, 0], [3]),
        _int64_tensor("c3_ends", [4, 10, 10], [3]),
        _int64_tensor("c4_starts", [4, 0, 0], [3]),
        _int64_tensor("c4_ends", [5, 10, 10], [3]),
        _int64_tensor("slice8_starts", [8, 0, 0], [3]),
        _int64_tensor("slice8_ends", [9, 10, 10], [3]),
        _int64_tensor("axes_chw", [1, 2, 3], [3]),
        _int64_tensor("two_i64", [2], [1]),
        _int32_tensor("zero_i32", [0], [1]),
        _int32_tensor("size_i32", [10], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("shape_vec10", [10], [1]),
        _int32_tensor("row_grid_i32", list(range(10)), [1, 1, 10, 1]),
        _int32_tensor("col_grid_i32", list(range(10)), [1, 1, 1, 10]),
        _int64_tensor("pad_axes_hw", [2, 3], [2]),
        _int64_tensor("reduce_axis_w", [3], [1]),
        _int64_tensor("pads_output_hw", [0, 0, 20, 20], [4]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("three_u8", [3], [1]),
        _u8_tensor("four_u8", [4], [1]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "c1_starts", "c1_ends", "axes_chw"], ["c1_f32"]),
        helper.make_node("Slice", ["input", "c2_starts", "c2_ends", "axes_chw"], ["c2_f32"]),
        helper.make_node("Slice", ["input", "c3_starts", "c3_ends", "axes_chw"], ["c3_f32"]),
        helper.make_node("Slice", ["input", "c4_starts", "c4_ends", "axes_chw"], ["c4_f32"]),
        helper.make_node("Slice", ["input", "slice8_starts", "slice8_ends", "axes_chw"], ["input8_f32"]),
        helper.make_node("Cast", ["c1_f32"], ["c1_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["c2_f32"], ["c2_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["c3_f32"], ["c3_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["c4_f32"], ["c4_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["input8_f32"], ["input8_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Where", ["c2_bool", "two_u8", "c1_u8"], ["color12_u8"]),
        helper.make_node("Where", ["c3_bool", "three_u8", "color12_u8"], ["color123_u8"]),
        helper.make_node("Where", ["c4_bool", "four_u8", "color123_u8"], ["color1234_u8"]),
        helper.make_node("Where", ["input8_bool", "eight_u8", "color1234_u8"], ["input_color_u8"]),
        helper.make_node("Min", ["input_color_u8", "one_u8"], ["nonzero_u8"]),
        helper.make_node("ReduceMax", ["nonzero_u8", "reduce_axis_w"], ["row_present"], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero_u8", "two_i64"], ["col_present"], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min_raw"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_max_raw"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min_raw"], axis=3, keepdims=1),
        helper.make_node("ArgMax", ["col_present"], ["c_max_raw"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Add", ["r_min_raw", "r_max_raw"], ["r_sum"]),
        helper.make_node("Add", ["c_min_raw", "c_max_raw"], ["c_sum"]),
        helper.make_node("Div", ["r_sum", "two_i64"], ["cr_raw"]),
        helper.make_node("Div", ["c_sum", "two_i64"], ["cc_raw"]),
        helper.make_node("Reshape", ["cr_raw", "shape1"], ["cr"]),
        helper.make_node("Reshape", ["cc_raw", "shape1"], ["cc"]),
        helper.make_node("Cast", ["cr"], ["cr_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["cc"], ["cc_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Sub", ["cr_i32", "cc_i32"], ["cr_minus_cc"]),
        helper.make_node("Sub", ["cc_i32", "cr_i32"], ["cc_minus_cr"]),
        helper.make_node("Add", ["cr_i32", "cc_i32"], ["center_sum"]),
        helper.make_node("Add", ["cr_i32", "cr_i32"], ["cr2"]),
        helper.make_node("Add", ["cc_i32", "cc_i32"], ["cc2"]),
        helper.make_node("Sub", ["center_sum", "col_grid_i32"], ["rot90_src_r"]),
        helper.make_node("Add", ["row_grid_i32", "cc_minus_cr"], ["rot90_src_c"]),
        helper.make_node("Sub", ["cr2", "row_grid_i32"], ["rot180_src_r"]),
        helper.make_node("Sub", ["cc2", "col_grid_i32"], ["rot180_src_c"]),
        helper.make_node("Add", ["col_grid_i32", "cr_minus_cc"], ["rot270_src_r"]),
        helper.make_node("Sub", ["center_sum", "row_grid_i32"], ["rot270_src_c"]),
    ]

    _gather_coords_bounded(nodes, "rot90_src_r", "rot90_src_c", "rot90", swap_axes=True)
    _gather_coords_bounded(nodes, "rot180_src_r", "rot180_src_c", "rot180", swap_axes=False)
    _gather_coords_bounded(nodes, "rot270_src_r", "rot270_src_c", "rot270", swap_axes=True)

    nodes.extend(
        [
            helper.make_node("Max", ["input_color_u8", "rot90", "rot180", "rot270"], ["placed_color"]),
            helper.make_node("Pad", ["placed_color", "pads_output_hw", "invalid_u8", "pad_axes_hw"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task020_flat_padded_source_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
