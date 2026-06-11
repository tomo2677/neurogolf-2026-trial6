from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("left_start", [1, 0, 0]),
        _int64_tensor("left_end", [2, 3, 3]),
        _int64_tensor("right_start", [1, 0, 4]),
        _int64_tensor("right_end", [2, 3, 7]),
        _int64_tensor("axes3", [1, 2, 3]),
        _int64_tensor("pads_output", [0, 0, 0, 7, 27, 27]),
        helper.make_tensor("colors3", onnx.TensorProto.UINT8, [1, 3, 1, 1], [0, 2, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "left_start", "left_end", "axes3"], ["left"]),
        helper.make_node("Slice", ["input", "right_start", "right_end", "axes3"], ["right"]),
        helper.make_node("Cast", ["left"], ["left_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["right"], ["right_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Min", ["left_u8", "right_u8"], ["color2_top_u8"]),
        helper.make_node("Equal", ["colors3", "color2_top_u8"], ["output3"]),
        helper.make_node("Pad", ["output3", "pads_output", "", "axes3"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task006_direct_u8_templates", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
