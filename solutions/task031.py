from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


INPUT_SIZE = 12
OUT_H = 7
OUT_W = 8


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


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("slice_ch0_starts", [0, 0, 0], [3]),
        _int64_tensor("slice_ch0_ends", [1, INPUT_SIZE, INPUT_SIZE], [3]),
        _int64_tensor("slice_ch0_axes", [1, 2, 3], [3]),
        _int32_tensor("zero_i32", [0], [1]),
        _int32_tensor("width_i32", [INPUT_SIZE], [1]),
        _int64_tensor("shape_index_1x56", [1, 1, OUT_H * OUT_W], [3]),
        _int64_tensor("shape_flat_1x144", [1, 1, INPUT_SIZE * INPUT_SIZE], [3]),
        _int64_tensor("shape_1x1x7x8", [1, 1, OUT_H, OUT_W], [4]),
        _int64_tensor("pads_color_to30", [0, 0, 0, 0, 0, 0, 30 - OUT_H, 30 - OUT_W], [8]),
        _int32_tensor("input_row_grid_i32", [r for r in range(INPUT_SIZE) for _ in range(INPUT_SIZE)], [1, 1, INPUT_SIZE, INPUT_SIZE]),
        _int32_tensor("input_col_grid_i32", [c for _ in range(INPUT_SIZE) for c in range(INPUT_SIZE)], [1, 1, INPUT_SIZE, INPUT_SIZE]),
        _int32_tensor("crop_row_grid_i32", [r for r in range(OUT_H) for _ in range(OUT_W)], [1, 1, OUT_H, OUT_W]),
        _int32_tensor("crop_col_grid_i32", [c for _ in range(OUT_H) for c in range(OUT_W)], [1, 1, OUT_H, OUT_W]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["row_present30"], axes=[1, 3], keepdims=1),
        helper.make_node("ReduceMax", ["input"], ["col_present30"], axes=[1, 2], keepdims=1),
        helper.make_node("ArgMax", ["row_present30"], ["last_row"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present30"], ["last_col"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["last_row"], ["last_row_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["last_col"], ["last_col_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("LessOrEqual", ["input_row_grid_i32", "last_row_i32"], ["input_row_valid"]),
        helper.make_node("LessOrEqual", ["input_col_grid_i32", "last_col_i32"], ["input_col_valid"]),
        helper.make_node("And", ["input_row_valid", "input_col_valid"], ["input_valid"]),
        helper.make_node("Slice", ["input", "slice_ch0_starts", "slice_ch0_ends", "slice_ch0_axes"], ["input0_12"]),
        helper.make_node("Equal", ["input0_12", "zero_f32"], ["nonzero_raw"]),
        helper.make_node("And", ["input_valid", "nonzero_raw"], ["nonzero_bool"]),
        helper.make_node("Cast", ["nonzero_bool"], ["nonzero_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["nonzero_u8"], ["row_present"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero_u8"], ["col_present"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_max"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min"], axis=3, keepdims=1),
        helper.make_node("ArgMax", ["col_present"], ["c_max"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["r_min"], ["r_min_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["r_max"], ["r_max_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_min"], ["c_min_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_max"], ["c_max_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Add", ["crop_row_grid_i32", "r_min_i32"], ["src_r"]),
        helper.make_node("Add", ["crop_col_grid_i32", "c_min_i32"], ["src_c"]),
        helper.make_node("LessOrEqual", ["src_r", "r_max_i32"], ["row_in_crop"]),
        helper.make_node("LessOrEqual", ["src_c", "c_max_i32"], ["col_in_crop"]),
        helper.make_node("And", ["row_in_crop", "col_in_crop"], ["crop_valid"]),
        helper.make_node("Where", ["crop_valid", "src_r", "zero_i32"], ["safe_r"]),
        helper.make_node("Where", ["crop_valid", "src_c", "zero_i32"], ["safe_c"]),
        helper.make_node("Mul", ["safe_r", "width_i32"], ["safe_r_offset"]),
        helper.make_node("Add", ["safe_r_offset", "safe_c"], ["safe_spatial"]),
        helper.make_node("Reshape", ["safe_spatial", "shape_index_1x56"], ["safe_spatial_flat_i32"]),
        helper.make_node("Cast", ["safe_spatial_flat_i32"], ["safe_spatial_flat"], to=onnx.TensorProto.INT64),
        helper.make_node("ReduceMax", ["input"], ["present_colors"], axes=[0, 2, 3], keepdims=1),
        helper.make_node("ArgMax", ["present_colors"], ["fg_color_i64"], axis=1, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["fg_color_i64"], ["fg_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Reshape", ["nonzero_u8", "shape_flat_1x144"], ["nonzero_flat_u8"]),
        helper.make_node("GatherElements", ["nonzero_flat_u8", "safe_spatial_flat"], ["gathered_flat"], axis=2),
        helper.make_node("Reshape", ["gathered_flat", "shape_1x1x7x8"], ["gathered_nonzero_u8"]),
        helper.make_node("Greater", ["gathered_nonzero_u8", "zero_u8"], ["gathered_nonzero"]),
        helper.make_node("Where", ["gathered_nonzero", "fg_color_u8", "zero_u8"], ["cropped_color"]),
        helper.make_node("Where", ["crop_valid", "cropped_color", "invalid_u8"], ["color_crop"]),
        helper.make_node("Pad", ["color_crop", "pads_color_to30", "invalid_u8"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task031_output7x8_crop_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
