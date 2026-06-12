from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
OUT = 5


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT16, dims, np.asarray(values, dtype=np.float16).ravel())


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("shape_color1", [1, 1, 1, 1], [4]),
        _int64_tensor("shape_index_1x25", [1, 1, OUT * OUT], [3]),
        _int64_tensor("shape_flat_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_1x1x5x5", [1, 1, OUT, OUT], [4]),
        _int64_tensor("pair_count_ends", [10], [1]),
        _int64_tensor("pair_sum_axes", [2, 3], [2]),
        _int32_tensor("row_grid_i32", [r for r in range(OUT) for _ in range(OUT)], [1, 1, OUT, OUT]),
        _int32_tensor("col_grid_i32", [c for _ in range(OUT) for c in range(OUT)], [1, 1, OUT, OUT]),
        _int32_tensor("zero_i32", [0], [1]),
        _int64_tensor("one_i64", [1], [1]),
        _int32_tensor("width_i32", [SIZE], [1]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, SIZE - OUT, SIZE - OUT], [8]),
        _int64_tensor("pads_shift_right", [0, 0, 0, -1, 0, 0, 0, 1], [8]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("Cast", ["input"], ["input_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Pad", ["input_bool", "pads_shift_right"], ["right_bool"], mode="constant"),
        helper.make_node("And", ["input_bool", "right_bool"], ["right_pair_bool"]),
        helper.make_node("Cast", ["right_pair_bool"], ["right_pair_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("ReduceSum", ["right_pair_f16", "pair_sum_axes"], ["pair_counts10"], keepdims=0),
        helper.make_node("Slice", ["pair_counts10", "one_i64", "pair_count_ends", "one_i64"], ["pair_counts"]),
        helper.make_node("ArgMax", ["pair_counts"], ["target_idx0"], axis=1, keepdims=1),
        helper.make_node("Add", ["target_idx0", "one_i64"], ["target_idx"]),
        helper.make_node("Reshape", ["target_idx", "shape_color1"], ["target_color_i64"]),
        helper.make_node("Reshape", ["target_idx", "one_i64"], ["target_channel_start"]),
        helper.make_node("Add", ["target_channel_start", "one_i64"], ["target_channel_end"]),
        helper.make_node("Slice", ["input_bool", "target_channel_start", "target_channel_end", "one_i64"], ["target_mask_bool"]),
        helper.make_node("Cast", ["target_mask_bool"], ["target_mask_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["target_mask_u8"], ["row_present"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["target_mask_u8"], ["col_present"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_max"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min"], axis=3, keepdims=1),
        helper.make_node("ArgMax", ["col_present"], ["c_max"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["r_min"], ["r_min_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["r_max"], ["r_max_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_min"], ["c_min_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_max"], ["c_max_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Add", ["row_grid_i32", "r_min_i32"], ["src_r"]),
        helper.make_node("Add", ["col_grid_i32", "c_min_i32"], ["src_c"]),
        helper.make_node("LessOrEqual", ["src_r", "r_max_i32"], ["row_in_crop"]),
        helper.make_node("LessOrEqual", ["src_c", "c_max_i32"], ["col_in_crop"]),
        helper.make_node("And", ["row_in_crop", "col_in_crop"], ["crop_valid"]),
        helper.make_node("Where", ["crop_valid", "src_r", "zero_i32"], ["safe_r"]),
        helper.make_node("Where", ["crop_valid", "src_c", "zero_i32"], ["safe_c"]),
        helper.make_node("Mul", ["safe_r", "width_i32"], ["safe_r_offset"]),
        helper.make_node("Add", ["safe_r_offset", "safe_c"], ["safe_spatial"]),
        helper.make_node("Reshape", ["safe_spatial", "shape_index_1x25"], ["safe_spatial_flat_i32"]),
        helper.make_node("Cast", ["safe_spatial_flat_i32"], ["safe_spatial_flat"], to=onnx.TensorProto.INT64),
        helper.make_node("Reshape", ["target_mask_u8", "shape_flat_1x900"], ["target_mask_flat"]),
        helper.make_node("GatherElements", ["target_mask_flat", "safe_spatial_flat"], ["crop_mask_flat"], axis=2),
        helper.make_node("Reshape", ["crop_mask_flat", "shape_1x1x5x5"], ["crop_mask"]),
        helper.make_node("Greater", ["crop_mask", "zero_u8"], ["crop_target_bool"]),
        helper.make_node("Cast", ["target_color_i64"], ["target_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Where", ["crop_target_bool", "target_color_u8", "zero_u8"], ["crop_color"]),
        helper.make_node("Where", ["crop_valid", "crop_color", "invalid_u8"], ["color5"]),
        helper.make_node("Pad", ["color5", "pads_output", "invalid_u8"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
    ]

    value_infos = [
        helper.make_tensor_value_info("target_mask_bool", onnx.TensorProto.BOOL, [1, 1, SIZE, SIZE]),
        helper.make_tensor_value_info("target_mask_u8", onnx.TensorProto.UINT8, [1, 1, SIZE, SIZE]),
    ]

    graph = helper.make_graph(
        nodes,
        "task036_dense_color_bbox_crop_color_grid_graph",
        [x],
        [y],
        initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
