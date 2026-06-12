from __future__ import annotations

import numpy as np
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


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(
        name,
        onnx.TensorProto.FLOAT16,
        dims,
        np.asarray(values, dtype=np.float16).view(np.uint16).tolist(),
        raw=False,
    )


def _shift_color(nodes: list[onnx.NodeProto], source: str, dr: str, dc: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Sub", ["row_grid_i32", dr], [f"{output}_src_r"]),
            helper.make_node("Sub", ["col_grid_i32", dc], [f"{output}_src_c"]),
            helper.make_node("Greater", [f"{output}_src_r", "neg_one_i32"], [f"{output}_r_nonneg"]),
            helper.make_node("Less", [f"{output}_src_r", "size_i32"], [f"{output}_r_lt_size"]),
            helper.make_node("Greater", [f"{output}_src_c", "neg_one_i32"], [f"{output}_c_nonneg"]),
            helper.make_node("Less", [f"{output}_src_c", "size_i32"], [f"{output}_c_lt_size"]),
            helper.make_node("And", [f"{output}_r_nonneg", f"{output}_r_lt_size"], [f"{output}_r_ok"]),
            helper.make_node("And", [f"{output}_c_nonneg", f"{output}_c_lt_size"], [f"{output}_c_ok"]),
            helper.make_node("And", [f"{output}_r_ok", f"{output}_c_ok"], [f"{output}_in_bounds"]),
            helper.make_node("Where", [f"{output}_in_bounds", f"{output}_src_r", "zero_i32"], [f"{output}_safe_r"]),
            helper.make_node("Where", [f"{output}_in_bounds", f"{output}_src_c", "zero_i32"], [f"{output}_safe_c"]),
            helper.make_node("Mul", [f"{output}_safe_r", "width_30_i32"], [f"{output}_safe_r_offset"]),
            helper.make_node("Add", [f"{output}_safe_r_offset", f"{output}_safe_c"], [f"{output}_safe_spatial"]),
            helper.make_node("Reshape", [f"{output}_safe_spatial", "shape_index_1x900"], [f"{output}_indices_i32"]),
            helper.make_node("Cast", [f"{output}_indices_i32"], [f"{output}_indices"], to=onnx.TensorProto.INT64),
            helper.make_node("Reshape", [source, "shape_flat_1x900"], [f"{output}_source_flat"]),
            helper.make_node("GatherElements", [f"{output}_source_flat", f"{output}_indices"], [f"{output}_shifted_flat"], axis=2),
            helper.make_node("Reshape", [f"{output}_shifted_flat", "shape_1x1x30x30"], [f"{output}_shifted"]),
            helper.make_node("Where", [f"{output}_in_bounds", f"{output}_shifted", "zero_u8"], [output]),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int32_tensor("row_idx", list(range(30)), [1, 1, 30, 1]),
        _int32_tensor("col_idx", list(range(30)), [1, 1, 1, 30]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("two_i64", [2], [1]),
        _int32_tensor("zero_i32", [0], [1]),
        _int32_tensor("neg_one_i32", [-1], [1]),
        _int32_tensor("size_i32", [30], [1]),
        _int32_tensor("width_30_i32", [30], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("shape_index_1x900", [1, 1, 900], [3]),
        _int64_tensor("shape_flat_1x900", [1, 1, 900], [3]),
        _int64_tensor("shape_1x1x30x30", [1, 1, 30, 30], [4]),
        _int32_tensor("row_grid_i32", [r for r in range(30) for _ in range(30)], [1, 1, 30, 30]),
        _int32_tensor("col_grid_i32", [c for _ in range(30) for c in range(30)], [1, 1, 30, 30]),
        _f16_tensor("zero_f16", [0.0], [1]),
        _f16_tensor("diag_kernel", [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], [1, 1, 3, 3]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["valid_cell_score"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["valid_cell_score"], ["row_present_score"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["valid_cell_score"], ["col_present_score"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_present_score"], ["last_row"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present_score"], ["last_col"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Add", ["last_row", "one_i64"], ["height_raw"]),
        helper.make_node("Add", ["last_col", "one_i64"], ["width_raw"]),
        helper.make_node("Reshape", ["height_raw", "shape1"], ["height"]),
        helper.make_node("Reshape", ["width_raw", "shape1"], ["width"]),
        helper.make_node("Mul", ["height", "two_i64"], ["out_height"]),
        helper.make_node("Mul", ["width", "two_i64"], ["out_width"]),
        helper.make_node("Cast", ["height"], ["height_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["width"], ["width_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["out_height"], ["out_height_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["out_width"], ["out_width_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Less", ["row_idx", "out_height_i32"], ["row_valid"]),
        helper.make_node("Less", ["col_idx", "out_width_i32"], ["col_valid"]),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_out"]),
        helper.make_node("ArgMax", ["input"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Cast", ["input_color_i64"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
    ]

    _shift_color(nodes, "input_color_u8", "zero_i32", "width_i32", "tile_right")
    _shift_color(nodes, "input_color_u8", "height_i32", "zero_i32", "tile_down")
    _shift_color(nodes, "input_color_u8", "height_i32", "width_i32", "tile_down_right")

    nodes.extend(
        [
            helper.make_node("Max", ["input_color_u8", "tile_right", "tile_down", "tile_down_right"], ["tiled_color"]),
            helper.make_node("Greater", ["tiled_color", "zero_u8"], ["colored"]),
            helper.make_node("Cast", ["colored"], ["colored_f16"], to=onnx.TensorProto.FLOAT16),
            helper.make_node(
                "Conv",
                ["colored_f16", "diag_kernel"],
                ["diag_score"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Greater", ["diag_score", "zero_f16"], ["diag_bool"]),
            helper.make_node("And", ["diag_bool", "valid_out"], ["diag_in_output"]),
            helper.make_node("Not", ["colored"], ["not_colored"]),
            helper.make_node("And", ["diag_in_output", "not_colored"], ["eight_mask"]),
            helper.make_node("Where", ["eight_mask", "eight_u8", "tiled_color"], ["filled_color"]),
            helper.make_node("Where", ["valid_out", "filled_color", "invalid_u8"], ["color30"]),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task019_tile_diag_eights_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
