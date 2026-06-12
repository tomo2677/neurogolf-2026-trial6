from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("nonzero_start", [1], [1]),
        _int64_tensor("nonzero_end", [10], [1]),
        _int64_tensor("axis_channel", [1], [1]),
        _int64_tensor("zero_i64", [0], [1]),
        _int64_tensor("width_i64", [SIZE], [1]),
        _int64_tensor("shape_index_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_index_10x900", [1, 10, SIZE * SIZE], [3]),
        _int64_tensor("shape_flat_10x900", [1, 10, SIZE * SIZE], [3]),
        _int64_tensor("shape_1x10x30x30", GRID_SHAPE, [4]),
        _int64_tensor("row_grid_i64", [r for r in range(SIZE) for _ in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int64_tensor("col_grid_i64", [c for _ in range(SIZE) for c in range(SIZE)], [1, 1, SIZE, SIZE]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "nonzero_start", "nonzero_end", "axis_channel"], ["input_nonzero"]),
        helper.make_node("ReduceMax", ["input_nonzero"], ["nonzero_score"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero_score"], ["row_present"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero_score"], ["col_present"], axes=[2], keepdims=1),
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
        helper.make_node("Expand", ["safe_spatial_flat", "shape_index_10x900"], ["gather_indices"]),
        helper.make_node("Reshape", ["input", "shape_flat_10x900"], ["input_flat"]),
        helper.make_node("GatherElements", ["input_flat", "gather_indices"], ["gathered_flat"], axis=2),
        helper.make_node("Reshape", ["gathered_flat", "shape_1x10x30x30"], ["gathered"]),
        helper.make_node("Cast", ["crop_valid"], ["crop_valid_f32"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Mul", ["gathered", "crop_valid_f32"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task031_nonzero_bbox_crop_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
