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
        _int64_tensor("slice_ch0_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("slice_ch0_ends", [1, 1, INPUT_SIZE, INPUT_SIZE], [4]),
        _int32_tensor("zero_i32", [0], [1]),
        _int64_tensor("pads_color_to30", [0, 0, 0, 0, 0, 0, 30 - OUT_H, 30 - OUT_W], [8]),
        _int32_tensor("input_row_grid_i32", list(range(INPUT_SIZE)), [1, 1, INPUT_SIZE, 1]),
        _int32_tensor("crop_row_grid_i32", list(range(OUT_H)), [OUT_H]),
        _int32_tensor("crop_col_grid_i32", list(range(OUT_W)), [OUT_W]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["row_present30"], axes=[1, 3], keepdims=1),
        helper.make_node("ArgMax", ["row_present30"], ["last_row"], axis=2, keepdims=0, select_last_index=1),
        helper.make_node("Cast", ["last_row"], ["last_row_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("LessOrEqual", ["input_row_grid_i32", "last_row_i32"], ["input_row_valid"]),
        helper.make_node("Slice", ["input", "slice_ch0_starts", "slice_ch0_ends"], ["input0_12"]),
        helper.make_node("Equal", ["input0_12", "zero_f32"], ["nonzero_raw"]),
        helper.make_node("And", ["input_row_valid", "nonzero_raw"], ["nonzero_bool"]),
        helper.make_node("Cast", ["nonzero_bool"], ["nonzero_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["nonzero_u8"], ["row_present"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero_u8"], ["col_present"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min"], axis=2, keepdims=0),
        helper.make_node("ArgMax", ["row_present"], ["r_max"], axis=2, keepdims=0, select_last_index=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min"], axis=3, keepdims=0),
        helper.make_node("ArgMax", ["col_present"], ["c_max"], axis=3, keepdims=0, select_last_index=1),
        helper.make_node("Squeeze", ["r_min"], ["r_min_scalar"], axes=[0, 1, 2]),
        helper.make_node("Squeeze", ["r_max"], ["r_max_scalar"], axes=[0, 1, 2]),
        helper.make_node("Squeeze", ["c_min"], ["c_min_scalar"], axes=[0, 1, 2]),
        helper.make_node("Squeeze", ["c_max"], ["c_max_scalar"], axes=[0, 1, 2]),
        helper.make_node("Cast", ["r_min_scalar"], ["r_min_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["r_max_scalar"], ["r_max_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_min_scalar"], ["c_min_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_max_scalar"], ["c_max_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Add", ["crop_row_grid_i32", "r_min_i32"], ["src_r"]),
        helper.make_node("Add", ["crop_col_grid_i32", "c_min_i32"], ["src_c"]),
        helper.make_node("LessOrEqual", ["src_r", "r_max_i32"], ["row_in_crop"]),
        helper.make_node("LessOrEqual", ["src_c", "c_max_i32"], ["col_in_crop"]),
        helper.make_node("Where", ["row_in_crop", "src_r", "zero_i32"], ["safe_r"]),
        helper.make_node("Where", ["col_in_crop", "src_c", "zero_i32"], ["safe_c"]),
        helper.make_node("Unsqueeze", ["row_in_crop"], ["row_in_crop_2d"], axes=[1]),
        helper.make_node("Unsqueeze", ["col_in_crop"], ["col_in_crop_2d"], axes=[0]),
        helper.make_node("And", ["row_in_crop_2d", "col_in_crop_2d"], ["crop_valid"]),
        helper.make_node("Gather", ["nonzero_bool", "safe_r"], ["gathered_rows"], axis=2),
        helper.make_node("Gather", ["gathered_rows", "safe_c"], ["gathered_nonzero"], axis=3),
        helper.make_node("ReduceMax", ["input"], ["present_colors"], axes=[0, 2, 3], keepdims=1),
        helper.make_node("ArgMax", ["present_colors"], ["fg_color_i64"], axis=1, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["fg_color_i64"], ["fg_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Where", ["gathered_nonzero", "fg_color_u8", "zero_u8"], ["cropped_color"]),
        helper.make_node("Where", ["crop_valid", "cropped_color", "invalid_u8"], ["color_crop"]),
        helper.make_node("Pad", ["color_crop", "pads_color_to30", "invalid_u8"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task031_bool_axis_gather_crop_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
