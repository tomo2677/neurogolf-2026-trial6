from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("starts3", [0, 0, 0]),
        _int64_tensor("ends3", [10, 3, 3]),
        _int64_tensor("axes3", [1, 2, 3]),
        _int64_tensor("channel_perm", [0, 5, 6, 4, 3, 1, 2, 7, 9, 8], [10]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 27, 27]),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "starts3", "ends3", "axes3"], ["input3"]),
        helper.make_node("Cast", ["input3"], ["input3_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Gather", ["input3_bool", "channel_perm"], ["mapped3"], axis=1),
        helper.make_node("Pad", ["mapped3", "pads_output"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task016_bool_channel_permute", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
