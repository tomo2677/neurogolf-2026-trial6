from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 10
CROP = 3


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("input10_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("input10_ends", [1, 10, SIZE, SIZE], [4]),
        _int64_tensor("zero_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("zero_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("shape_index_1x9", [1, 1, CROP * CROP], [3]),
        _int64_tensor("shape_index_10x9", [1, 10, CROP * CROP], [3]),
        _int64_tensor("shape_flat_10x100", [1, 10, SIZE * SIZE], [3]),
        _int64_tensor("shape_1x10x3x3", [1, 10, CROP, CROP], [4]),
        _int64_tensor("row_grid", [r for r in range(CROP) for _ in range(CROP)], [1, 1, CROP, CROP]),
        _int64_tensor("col_grid", [c for _ in range(CROP) for c in range(CROP)], [1, 1, CROP, CROP]),
        _int64_tensor("width_i64", [SIZE], [1]),
        _int64_tensor("output_pads", [0, 0, 0, 0, 0, 0, 27, 27], [8]),
        _f32_tensor("one_f32", [1.0], [1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "input10_starts", "input10_ends"], ["input10"]),
        helper.make_node("Slice", ["input", "zero_starts", "zero_ends"], ["zero_ch"]),
        helper.make_node("Sub", ["one_f32", "zero_ch"], ["nonzero"]),
        helper.make_node("ReduceMax", ["nonzero"], ["row_present"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero"], ["col_present"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min"], axis=3, keepdims=1),
        helper.make_node("Add", ["row_grid", "r_min"], ["src_r"]),
        helper.make_node("Add", ["col_grid", "c_min"], ["src_c"]),
        helper.make_node("Mul", ["src_r", "width_i64"], ["src_r_offset"]),
        helper.make_node("Add", ["src_r_offset", "src_c"], ["src_spatial"]),
        helper.make_node("Reshape", ["src_spatial", "shape_index_1x9"], ["src_spatial_flat"]),
        helper.make_node("Expand", ["src_spatial_flat", "shape_index_10x9"], ["gather_indices"]),
        helper.make_node("Reshape", ["input10", "shape_flat_10x100"], ["input10_flat"]),
        helper.make_node("GatherElements", ["input10_flat", "gather_indices"], ["crop_flat"], axis=2),
        helper.make_node("Reshape", ["crop_flat", "shape_1x10x3x3"], ["crop"]),
        helper.make_node("Pad", ["crop", "output_pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task039_top_left_bbox_crop3_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
