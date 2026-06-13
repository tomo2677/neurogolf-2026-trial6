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


def _gather_coords_padded(nodes: list[onnx.NodeProto], src_r: str, src_c: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Add", [src_r, "pad_offset_i32"], [f"{output}_pad_r"]),
            helper.make_node("Add", [src_c, "pad_offset_i32"], [f"{output}_pad_c"]),
            helper.make_node("Mul", [f"{output}_pad_r", "padded_width_i32"], [f"{output}_r_offset"]),
            helper.make_node("Add", [f"{output}_r_offset", f"{output}_pad_c"], [f"{output}_spatial"]),
            helper.make_node("Gather", ["input_color30_flat", f"{output}_spatial"], [output], axis=0),
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
        _int32_tensor("pad_offset_i32", [10], [1]),
        _int32_tensor("padded_width_i32", [30], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("shape_flat300", [300], [1]),
        _int32_tensor("row_grid_i32", list(range(10)), [1, 1, 10, 1]),
        _int32_tensor("col_grid_i32", list(range(10)), [1, 1, 1, 10]),
        _int64_tensor("pad_axes_hw", [2, 3], [2]),
        _int64_tensor("pad_axis_w", [3], [1]),
        _int64_tensor("reduce_axis_w", [3], [1]),
        _int64_tensor("reduce_axis_h", [2], [1]),
        _int64_tensor("pads_source_w", [10, 10], [2]),
        _int64_tensor("pads_flat900", [300, 300], [2]),
        _int64_tensor("pads_output_hw", [0, 0, 20, 20], [4]),
        _u8_tensor("zero_u8", [0], [1]),
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
        helper.make_node("Pad", ["input_color_u8", "pads_source_w", "zero_u8", "pad_axis_w"], ["input_color10x30_u8"], mode="constant"),
        helper.make_node("Reshape", ["input_color10x30_u8", "shape_flat300"], ["input_color10x30_flat"]),
        helper.make_node("Pad", ["input_color10x30_flat", "pads_flat900", "zero_u8"], ["input_color30_flat"], mode="constant"),
        helper.make_node("Greater", ["input_color_u8", "zero_u8"], ["nonzero_bool"]),
        helper.make_node("Cast", ["nonzero_bool"], ["nonzero_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["nonzero_u8", "reduce_axis_w"], ["row_present"], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero_u8", "reduce_axis_h"], ["col_present"], keepdims=1),
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

    _gather_coords_padded(nodes, "rot90_src_r", "rot90_src_c", "rot90")
    _gather_coords_padded(nodes, "rot180_src_r", "rot180_src_c", "rot180")
    _gather_coords_padded(nodes, "rot270_src_r", "rot270_src_c", "rot270")

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
