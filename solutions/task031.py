from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("zero_i64", [0], [1]),
        _int32_tensor("zero_i32", [0], [1]),
        _int32_tensor("width_i32", [SIZE], [1]),
        _int64_tensor("shape_index_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_flat_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_1x1x30x30", [1, 1, SIZE, SIZE], [4]),
        _int32_tensor("row_grid_i32", [r for r in range(SIZE) for _ in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int32_tensor("col_grid_i32", [c for _ in range(SIZE) for c in range(SIZE)], [1, 1, SIZE, SIZE]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("ArgMax", ["input"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Greater", ["input_color_i64", "zero_i64"], ["nonzero_bool"]),
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
        helper.make_node("Add", ["row_grid_i32", "r_min_i32"], ["src_r"]),
        helper.make_node("Add", ["col_grid_i32", "c_min_i32"], ["src_c"]),
        helper.make_node("LessOrEqual", ["src_r", "r_max_i32"], ["row_in_crop"]),
        helper.make_node("LessOrEqual", ["src_c", "c_max_i32"], ["col_in_crop"]),
        helper.make_node("And", ["row_in_crop", "col_in_crop"], ["crop_valid"]),
        helper.make_node("Where", ["crop_valid", "src_r", "zero_i32"], ["safe_r"]),
        helper.make_node("Where", ["crop_valid", "src_c", "zero_i32"], ["safe_c"]),
        helper.make_node("Mul", ["safe_r", "width_i32"], ["safe_r_offset"]),
        helper.make_node("Add", ["safe_r_offset", "safe_c"], ["safe_spatial"]),
        helper.make_node("Reshape", ["safe_spatial", "shape_index_1x900"], ["safe_spatial_flat_i32"]),
        helper.make_node("Cast", ["safe_spatial_flat_i32"], ["safe_spatial_flat"], to=onnx.TensorProto.INT64),
        helper.make_node("Cast", ["input_color_i64"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Reshape", ["input_color_u8", "shape_flat_1x900"], ["color_flat_u8"]),
        helper.make_node("GatherElements", ["color_flat_u8", "safe_spatial_flat"], ["gathered_flat"], axis=2),
        helper.make_node("Reshape", ["gathered_flat", "shape_1x1x30x30"], ["gathered_color"]),
        helper.make_node("Where", ["crop_valid", "gathered_color", "invalid_u8"], ["color30"]),
        helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task031_nonzero_bbox_crop_color_grid_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
