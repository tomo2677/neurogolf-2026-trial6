from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
NONZERO = 9


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


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
        _int64_tensor("nonzero_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("nonzero_ends", [1, 10, SIZE, SIZE], [4]),
        _int64_tensor("shape_color1", [1, 1, 1, 1], [4]),
        _int64_tensor("shape_index_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_flat_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_1x1x30x30", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("row_grid_i64", [r for r in range(SIZE) for _ in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int64_tensor("col_grid_i64", [c for _ in range(SIZE) for c in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int64_tensor("zero_i64", [0], [1]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("width_i64", [SIZE], [1]),
        _f16_tensor("density_w", [1.0] * (NONZERO * 3 * 3), [NONZERO, 1, 3, 3]),
        _f16_tensor("two_f16", [2.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "nonzero_starts", "nonzero_ends"], ["input_nonzero"]),
        helper.make_node("Cast", ["input_nonzero"], ["input_nonzero_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node(
            "Conv",
            ["input_nonzero_f16", "density_w"],
            ["same_color_3x3"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            group=NONZERO,
        ),
        helper.make_node("Greater", ["same_color_3x3", "two_f16"], ["dense_bool"]),
        helper.make_node("Cast", ["dense_bool"], ["dense_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("ReduceSum", ["dense_f16"], ["dense_counts"], axes=[2, 3], keepdims=0),
        helper.make_node("ArgMax", ["dense_counts"], ["target_idx"], axis=1, keepdims=1),
        helper.make_node("Add", ["target_idx", "one_i64"], ["target_color_i64_raw"]),
        helper.make_node("Reshape", ["target_color_i64_raw", "shape_color1"], ["target_color_i64"]),
        helper.make_node("ArgMax", ["input"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Equal", ["input_color_i64", "target_color_i64"], ["target_mask_bool"]),
        helper.make_node("Cast", ["target_mask_bool"], ["target_mask_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["target_mask_u8"], ["row_present"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["target_mask_u8"], ["col_present"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_max"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min"], axis=3, keepdims=1),
        helper.make_node("ArgMax", ["col_present"], ["c_max"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Add", ["row_grid_i64", "r_min"], ["src_r"]),
        helper.make_node("Add", ["col_grid_i64", "c_min"], ["src_c"]),
        helper.make_node("LessOrEqual", ["src_r", "r_max"], ["row_in_crop"]),
        helper.make_node("LessOrEqual", ["src_c", "c_max"], ["col_in_crop"]),
        helper.make_node("And", ["row_in_crop", "col_in_crop"], ["crop_valid"]),
        helper.make_node("Where", ["crop_valid", "src_r", "zero_i64"], ["safe_r"]),
        helper.make_node("Where", ["crop_valid", "src_c", "zero_i64"], ["safe_c"]),
        helper.make_node("Mul", ["safe_r", "width_i64"], ["safe_r_offset"]),
        helper.make_node("Add", ["safe_r_offset", "safe_c"], ["safe_spatial"]),
        helper.make_node("Reshape", ["safe_spatial", "shape_index_1x900"], ["safe_spatial_flat"]),
        helper.make_node("Reshape", ["target_mask_u8", "shape_flat_1x900"], ["target_mask_flat"]),
        helper.make_node("GatherElements", ["target_mask_flat", "safe_spatial_flat"], ["crop_mask_flat"], axis=2),
        helper.make_node("Reshape", ["crop_mask_flat", "shape_1x1x30x30"], ["crop_mask"]),
        helper.make_node("Greater", ["crop_mask", "zero_u8"], ["crop_target_bool"]),
        helper.make_node("Cast", ["target_color_i64"], ["target_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Where", ["crop_target_bool", "target_color_u8", "zero_u8"], ["crop_color"]),
        helper.make_node("Where", ["crop_valid", "crop_color", "invalid_u8"], ["color30"]),
        helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task036_dense_color_bbox_crop_color_grid_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
