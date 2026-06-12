from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 10


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()
    crop_info = helper.make_tensor_value_info("crop", onnx.TensorProto.FLOAT, [1, 10, 3, 3])

    initializers = [
        _int64_tensor("zero_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("zero_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("axis_row", [2], [1]),
        _int64_tensor("axis_col", [3], [1]),
        _int64_tensor("axes4", [0, 1, 2, 3], [4]),
        _int64_tensor("shape_scalar", [1], [1]),
        _int64_tensor("zero_i64", [0], [1]),
        _int64_tensor("ten_i64", [10], [1]),
        _int64_tensor("slice_sizes", [1, 10, 3, 3], [4]),
        _int64_tensor("output_pads", [0, 0, 0, 0, 0, 0, 27, 27], [8]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "zero_starts", "zero_ends"], ["zero_ch"]),
        helper.make_node("ReduceMin", ["zero_ch", "axis_col"], ["row_zero_min"], keepdims=1),
        helper.make_node("ReduceMin", ["zero_ch", "axis_row"], ["col_zero_min"], keepdims=1),
        helper.make_node("ArgMin", ["row_zero_min"], ["r_min_1111"], axis=2, keepdims=1),
        helper.make_node("ArgMin", ["col_zero_min"], ["c_min_1111"], axis=3, keepdims=1),
        helper.make_node("Reshape", ["r_min_1111", "shape_scalar"], ["r_min"]),
        helper.make_node("Reshape", ["c_min_1111", "shape_scalar"], ["c_min"]),
        helper.make_node("Concat", ["zero_i64", "zero_i64", "r_min", "c_min"], ["crop_starts"], axis=0),
        helper.make_node("Add", ["crop_starts", "slice_sizes"], ["crop_ends"]),
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "axes4"], ["crop"]),
        helper.make_node("Pad", ["crop", "output_pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task039_dynamic_slice_graph", [x], [y], initializers, value_info=[crop_info])
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
