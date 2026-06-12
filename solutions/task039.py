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
        _int64_tensor("input10_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("input10_ends", [1, 10, SIZE, SIZE], [4]),
        _int64_tensor("zero_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("zero_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("axis_row", [2], [1]),
        _int64_tensor("axis_col", [3], [1]),
        _int64_tensor("shape_scalar", [1], [1]),
        _int64_tensor("offset3", [0, 1, 2], [3]),
        _int64_tensor("output_pads", [0, 0, 0, 0, 0, 0, 27, 27], [8]),
        _f32_tensor("one_f32", [1.0], [1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "input10_starts", "input10_ends"], ["input10"]),
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
        helper.make_node("Gather", ["input10", "row_idx3"], ["crop_rows"], axis=2),
        helper.make_node("Gather", ["crop_rows", "col_idx3"], ["crop"], axis=3),
        helper.make_node("Pad", ["crop", "output_pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task039_dynamic_gather_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
