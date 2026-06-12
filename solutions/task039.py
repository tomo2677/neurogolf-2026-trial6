from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 10


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("zero_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("zero_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("axis_row", [2], [1]),
        _int64_tensor("axis_col", [3], [1]),
        _int64_tensor("shape_scalar", [1], [1]),
        _int64_tensor("offset3", [0, 1, 2], [3]),
        _int64_tensor("unsq_row_axis", [1], [1]),
        _int64_tensor("unsq_col_axis", [0], [1]),
        _int64_tensor("unsq_last_axis", [2], [1]),
        _int64_tensor("unsq_batch_axes", [0, 1], [2]),
        _int64_tensor("shape33", [3, 3], [2]),
        _int64_tensor("gathernd_index_shape", [1, 10, 3, 3, 2], [5]),
        _int64_tensor("output_pads", [0, 0, 0, 0, 0, 0, 27, 27], [8]),
        _f32_tensor("one_f32", [1.0], [1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "zero_starts", "zero_ends"], ["zero_ch"]),
        helper.make_node("Sub", ["one_f32", "zero_ch"], ["nonzero"]),
        helper.make_node("ReduceMax", ["nonzero", "axis_col"], ["row_present"], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero", "axis_row"], ["col_present"], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min_1111"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min_1111"], axis=3, keepdims=1),
        helper.make_node("Reshape", ["r_min_1111", "shape_scalar"], ["r_min"]),
        helper.make_node("Reshape", ["c_min_1111", "shape_scalar"], ["c_min"]),
        helper.make_node("Add", ["r_min", "offset3"], ["row_idx3"]),
        helper.make_node("Add", ["c_min", "offset3"], ["col_idx3"]),
        helper.make_node("Unsqueeze", ["row_idx3", "unsq_row_axis"], ["row_col"]),
        helper.make_node("Unsqueeze", ["col_idx3", "unsq_col_axis"], ["col_row"]),
        helper.make_node("Expand", ["row_col", "shape33"], ["row_mat"]),
        helper.make_node("Expand", ["col_row", "shape33"], ["col_mat"]),
        helper.make_node("Unsqueeze", ["row_mat", "unsq_last_axis"], ["row_mat1"]),
        helper.make_node("Unsqueeze", ["col_mat", "unsq_last_axis"], ["col_mat1"]),
        helper.make_node("Concat", ["row_mat1", "col_mat1"], ["spatial_indices"], axis=2),
        helper.make_node("Unsqueeze", ["spatial_indices", "unsq_batch_axes"], ["spatial_indices_batched"]),
        helper.make_node("Expand", ["spatial_indices_batched", "gathernd_index_shape"], ["gathernd_indices"]),
        helper.make_node("GatherND", ["input", "gathernd_indices"], ["crop"], batch_dims=2),
        helper.make_node("Pad", ["crop", "output_pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task039_dynamic_gathernd_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
