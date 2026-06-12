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
            helper.make_node("Cast", [f"{output}_spatial"], [f"{output}_indices"], to=onnx.TensorProto.INT64),
            helper.make_node("Gather", ["input_color30_flat", f"{output}_indices"], [output], axis=0),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("slice_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("slice_ends", [1, 10, 10, 10], [4]),
        _int64_tensor("two_i64", [2], [1]),
        _int32_tensor("pad_offset_i32", [10], [1]),
        _int32_tensor("padded_width_i32", [30], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("shape_flat900", [900], [1]),
        _int32_tensor("row_grid_i32", [r for r in range(10) for _ in range(10)], [1, 1, 10, 10]),
        _int32_tensor("col_grid_i32", [c for _ in range(10) for c in range(10)], [1, 1, 10, 10]),
        _int64_tensor("pads_source", [0, 0, 10, 10, 0, 0, 10, 10], [8]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 20, 20], [8]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "slice_starts", "slice_ends"], ["input10"]),
        helper.make_node("ArgMax", ["input10"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Cast", ["input_color_i64"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Pad", ["input_color_u8", "pads_source", "zero_u8"], ["input_color30_u8"], mode="constant"),
        helper.make_node("Reshape", ["input_color30_u8", "shape_flat900"], ["input_color30_flat"]),
        helper.make_node("Greater", ["input_color_u8", "zero_u8"], ["nonzero_bool"]),
        helper.make_node("Cast", ["nonzero_bool"], ["nonzero_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["nonzero_u8"], ["row_present"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero_u8"], ["col_present"], axes=[2], keepdims=1),
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
            helper.make_node("Pad", ["placed_color", "pads_output", "invalid_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task020_quarter_turn_completion_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
