from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("starts3", [0, 0, 0]),
        _int64_tensor("ends3", [10, 3, 3]),
        _int64_tensor("axes3", [1, 2, 3]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 27, 27]),
        _u8_tensor("lut", [0, 5, 6, 4, 3, 1, 2, 7, 9, 8], [10]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _u8_tensor("outside_u8", [255], [1]),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "starts3", "ends3", "axes3"], ["input3"]),
        helper.make_node("ArgMax", ["input3"], ["color3_i64"], axis=1, keepdims=1),
        helper.make_node("Gather", ["lut", "color3_i64"], ["mapped3"], axis=0),
        helper.make_node("Pad", ["mapped3", "pads_output", "outside_u8"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task016_color_lut_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
