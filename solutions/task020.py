from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _gather_coords(nodes: list[onnx.NodeProto], source: str, src_r: str, src_c: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Greater", [src_r, "neg_one_i64"], [f"{output}_r_nonneg"]),
            helper.make_node("Less", [src_r, "size_i64"], [f"{output}_r_lt_size"]),
            helper.make_node("Greater", [src_c, "neg_one_i64"], [f"{output}_c_nonneg"]),
            helper.make_node("Less", [src_c, "size_i64"], [f"{output}_c_lt_size"]),
            helper.make_node("And", [f"{output}_r_nonneg", f"{output}_r_lt_size"], [f"{output}_r_ok"]),
            helper.make_node("And", [f"{output}_c_nonneg", f"{output}_c_lt_size"], [f"{output}_c_ok"]),
            helper.make_node("And", [f"{output}_r_ok", f"{output}_c_ok"], [f"{output}_in_bounds"]),
            helper.make_node("Where", [f"{output}_in_bounds", src_r, "zero_i64"], [f"{output}_safe_r"]),
            helper.make_node("Where", [f"{output}_in_bounds", src_c, "zero_i64"], [f"{output}_safe_c"]),
            helper.make_node("Mul", [f"{output}_safe_r", "width_i64"], [f"{output}_safe_r_offset"]),
            helper.make_node("Add", [f"{output}_safe_r_offset", f"{output}_safe_c"], [f"{output}_safe_spatial"]),
            helper.make_node("Reshape", [f"{output}_safe_spatial", "shape_index_1x900"], [f"{output}_indices"]),
            helper.make_node("Reshape", [source, "shape_flat_1x900"], [f"{output}_source_flat"]),
            helper.make_node("GatherElements", [f"{output}_source_flat", f"{output}_indices"], [f"{output}_flat"], axis=2),
            helper.make_node("Reshape", [f"{output}_flat", "shape_1x1x30x30"], [f"{output}_raw"]),
            helper.make_node("Where", [f"{output}_in_bounds", f"{output}_raw", "zero_u8"], [output]),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("two_i64", [2], [1]),
        _int64_tensor("zero_i64", [0], [1]),
        _int64_tensor("neg_one_i64", [-1], [1]),
        _int64_tensor("ten_i64", [10], [1]),
        _int64_tensor("size_i64", [30], [1]),
        _int64_tensor("width_i64", [30], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("shape_index_1x900", [1, 1, 900], [3]),
        _int64_tensor("shape_flat_1x900", [1, 1, 900], [3]),
        _int64_tensor("shape_1x1x30x30", [1, 1, 30, 30], [4]),
        _int64_tensor("row_grid_i64", [r for r in range(30) for _ in range(30)], [1, 1, 30, 30]),
        _int64_tensor("col_grid_i64", [c for _ in range(30) for c in range(30)], [1, 1, 30, 30]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ArgMax", ["input"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Cast", ["input_color_i64"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Greater", ["input_color_i64", "zero_i64"], ["nonzero_bool"]),
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
        helper.make_node("Sub", ["cr", "cc"], ["cr_minus_cc"]),
        helper.make_node("Sub", ["cc", "cr"], ["cc_minus_cr"]),
        helper.make_node("Add", ["cr", "cc"], ["center_sum"]),
        helper.make_node("Add", ["cr", "cr"], ["cr2"]),
        helper.make_node("Add", ["cc", "cc"], ["cc2"]),
        helper.make_node("Sub", ["center_sum", "col_grid_i64"], ["rot90_src_r"]),
        helper.make_node("Add", ["row_grid_i64", "cc_minus_cr"], ["rot90_src_c"]),
        helper.make_node("Sub", ["cr2", "row_grid_i64"], ["rot180_src_r"]),
        helper.make_node("Sub", ["cc2", "col_grid_i64"], ["rot180_src_c"]),
        helper.make_node("Add", ["col_grid_i64", "cr_minus_cc"], ["rot270_src_r"]),
        helper.make_node("Sub", ["center_sum", "row_grid_i64"], ["rot270_src_c"]),
    ]

    _gather_coords(nodes, "input_color_u8", "rot90_src_r", "rot90_src_c", "rot90")
    _gather_coords(nodes, "input_color_u8", "rot180_src_r", "rot180_src_c", "rot180")
    _gather_coords(nodes, "input_color_u8", "rot270_src_r", "rot270_src_c", "rot270")

    nodes.extend(
        [
            helper.make_node("Max", ["input_color_u8", "rot90", "rot180", "rot270"], ["placed_color"]),
            helper.make_node("Less", ["row_grid_i64", "ten_i64"], ["row_valid10"]),
            helper.make_node("Less", ["col_grid_i64", "ten_i64"], ["col_valid10"]),
            helper.make_node("And", ["row_valid10", "col_valid10"], ["valid10_bool"]),
            helper.make_node("Where", ["valid10_bool", "placed_color", "invalid_u8"], ["color30"]),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task020_quarter_turn_completion_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
