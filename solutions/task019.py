from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


OUT_SIZE = 12
INPUT_SIZE = 6


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(
        name,
        onnx.TensorProto.FLOAT16,
        dims,
        np.asarray(values, dtype=np.float16).view(np.uint16).tolist(),
        raw=False,
    )


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("input0_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("input0_ends", [1, 1, INPUT_SIZE, INPUT_SIZE], [4]),
        _int64_tensor("pads_color12_to30", [0, 0, 0, 0, 0, 0, 30 - OUT_SIZE, 30 - OUT_SIZE], [8]),
        _int32_tensor("row_idx", list(range(OUT_SIZE)), [1, 1, OUT_SIZE, 1]),
        _int32_tensor("col_idx", list(range(OUT_SIZE)), [1, 1, 1, OUT_SIZE]),
        _int32_tensor("row_idx6", list(range(INPUT_SIZE)), [1, 1, INPUT_SIZE, 1]),
        _int32_tensor("col_idx6", list(range(INPUT_SIZE)), [1, 1, 1, INPUT_SIZE]),
        _int32_tensor("one_i32", [1], [1]),
        _int32_tensor("two_i32", [2], [1]),
        _int32_tensor("input_width_i32", [INPUT_SIZE], [1]),
        _int64_tensor("shape_flat36", [INPUT_SIZE * INPUT_SIZE], [1]),
        _int32_tensor("row_grid_i32", list(range(OUT_SIZE)), [1, 1, OUT_SIZE, 1]),
        _int32_tensor("col_grid_i32", list(range(OUT_SIZE)), [1, 1, 1, OUT_SIZE]),
        _f16_tensor("zero_f16", [0.0], [1]),
        _f16_tensor("diag_kernel", [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], [1, 1, 3, 3]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["row_present30"], axes=[1, 3], keepdims=1),
        helper.make_node("ReduceMax", ["input"], ["col_present30"], axes=[1, 2], keepdims=1),
        helper.make_node("ArgMax", ["row_present30"], ["last_row"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present30"], ["last_col"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["last_row"], ["last_row_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["last_col"], ["last_col_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Add", ["last_row_i32", "one_i32"], ["height_i32"]),
        helper.make_node("Add", ["last_col_i32", "one_i32"], ["width_dyn_i32"]),
        helper.make_node("Mul", ["height_i32", "two_i32"], ["out_height_i32"]),
        helper.make_node("Mul", ["width_dyn_i32", "two_i32"], ["out_width_i32"]),
        helper.make_node("Less", ["row_idx", "out_height_i32"], ["row_valid"]),
        helper.make_node("Less", ["col_idx", "out_width_i32"], ["col_valid"]),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_out"]),
        helper.make_node("MaxPool", ["input"], ["present10"], kernel_shape=[30, 30]),
        helper.make_node("ArgMax", ["present10"], ["fg_color_i64"], axis=1, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["fg_color_i64"], ["fg_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Less", ["row_idx6", "height_i32"], ["row_valid6"]),
        helper.make_node("Less", ["col_idx6", "width_dyn_i32"], ["col_valid6"]),
        helper.make_node("And", ["row_valid6", "col_valid6"], ["valid_in6"]),
        helper.make_node("Slice", ["input", "input0_starts", "input0_ends"], ["input0_6"]),
        helper.make_node("Cast", ["input0_6"], ["bg_bool6"], to=onnx.TensorProto.BOOL),
        helper.make_node("Not", ["bg_bool6"], ["not_bg6"]),
        helper.make_node("And", ["valid_in6", "not_bg6"], ["nonzero6"]),
        helper.make_node("Where", ["nonzero6", "fg_color_u8", "zero_u8"], ["input_color_u8"]),
        helper.make_node("Mod", ["row_grid_i32", "height_i32"], ["tile_src_r"], fmod=0),
        helper.make_node("Mod", ["col_grid_i32", "width_dyn_i32"], ["tile_src_c"], fmod=0),
        helper.make_node("Mul", ["tile_src_r", "input_width_i32"], ["tile_src_r_offset"]),
        helper.make_node("Add", ["tile_src_r_offset", "tile_src_c"], ["tile_src_spatial"]),
        helper.make_node("Reshape", ["input_color_u8", "shape_flat36"], ["input_color_flat"]),
        helper.make_node("Gather", ["input_color_flat", "tile_src_spatial"], ["tiled_raw"], axis=0),
        helper.make_node("Where", ["valid_out", "tiled_raw", "zero_u8"], ["tiled_color"]),
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
        helper.make_node("Where", ["valid_out", "filled_color", "invalid_u8"], ["color12"]),
        helper.make_node("Pad", ["color12", "pads_color12_to30", "invalid_u8"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task019_single_color_zero_mask_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
