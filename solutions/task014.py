from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 25
OUT_H = 17
OUT_W = 18


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("count_starts", [1], [1]),
        _int64_tensor("count_ends", [10], [1]),
        _int64_tensor("k1", [1], [1]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("axes_counts", [0, 2, 3], [3]),
        _int64_tensor("crop_hw_start", [0, 0], [2]),
        _int64_tensor("crop_hw_end", [SIZE, SIZE], [2]),
        _int64_tensor("crop_hw_axes", [2, 3], [2]),
        _int64_tensor("axis_h", [2], [1]),
        _int64_tensor("axis_w", [3], [1]),
        _int64_tensor("shape_vec_h", [OUT_H], [1]),
        _int64_tensor("shape_vec_w", [OUT_W], [1]),
        _int32_tensor("zero_i32", [0], [1]),
        _int32_tensor("crop_row_grid_i32", list(range(OUT_H)), [1, 1, OUT_H, 1]),
        _int32_tensor("crop_col_grid_i32", list(range(OUT_W)), [1, 1, 1, OUT_W]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 30 - OUT_H, 30 - OUT_W], [8]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("neg_large_f32", [-10000.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("ReduceSum", ["input", "axes_counts"], ["counts10"], keepdims=0),
        helper.make_node("Slice", ["counts10", "count_starts", "count_ends"], ["counts9"]),
        helper.make_node("Greater", ["counts9", "zero_f32"], ["present9"]),
        helper.make_node("Sub", ["zero_f32", "counts9"], ["neg_counts9"]),
        helper.make_node("Where", ["present9", "neg_counts9", "neg_large_f32"], ["target_scores9"]),
        helper.make_node("TopK", ["target_scores9", "k1"], ["target_score", "target_idx0"], axis=0, largest=1, sorted=1),
        helper.make_node("Add", ["target_idx0", "one_i64"], ["target_color_i64"]),
        helper.make_node("Cast", ["target_color_i64"], ["target_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Gather", ["input", "target_color_i64"], ["target_channel30_f32"], axis=1),
        helper.make_node("Slice", ["target_channel30_f32", "crop_hw_start", "crop_hw_end", "crop_hw_axes"], ["target_channel_f32"]),
        helper.make_node("Cast", ["target_channel_f32"], ["target_mask_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["target_mask_u8", "axis_w"], ["row_present"], keepdims=1),
        helper.make_node("ReduceMax", ["target_mask_u8", "axis_h"], ["col_present"], keepdims=1),
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
        helper.make_node("Where", ["row_in_crop", "src_r", "zero_i32"], ["safe_r"]),
        helper.make_node("Where", ["col_in_crop", "src_c", "zero_i32"], ["safe_c"]),
        helper.make_node("Reshape", ["safe_r", "shape_vec_h"], ["safe_r_vec"]),
        helper.make_node("Reshape", ["safe_c", "shape_vec_w"], ["safe_c_vec"]),
        helper.make_node("Gather", ["target_mask_u8", "safe_r_vec"], ["gathered_rows"], axis=2),
        helper.make_node("Gather", ["gathered_rows", "safe_c_vec"], ["gathered_target_u8"], axis=3),
        helper.make_node("Equal", ["gathered_target_u8", "one_u8"], ["target_crop"]),
        helper.make_node("Where", ["target_crop", "target_color_u8", "zero_u8"], ["crop_color_or_zero"]),
        helper.make_node("Where", ["crop_valid", "crop_color_or_zero", "invalid_u8"], ["color_crop"]),
        helper.make_node("Pad", ["color_crop", "pads_output", "invalid_u8"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task014_direct_target_channel_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
