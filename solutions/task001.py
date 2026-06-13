from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, make_io_value_infos


INTERNAL_TYPE = onnx.TensorProto.FLOAT16
NP_DTYPE = np.float16


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, INTERNAL_TYPE, dims, np.array(values, dtype=NP_DTYPE))


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.UINT8, GRID_SHAPE)

    initializers = [
        _int64_tensor("zero_starts", [0, 0, 0, 0]),
        _int64_tensor("zero_ends", [1, 1, 3, 3]),
        _int64_tensor("pads_hw", [0, 0, 21, 21]),
        _int64_tensor("pad_axes_hw", [2, 3]),
        _int64_tensor("block_axes", [3, 5]),
        _int64_tensor("inner_axes", [2, 4]),
        _u8_tensor("black10", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "zero_starts", "zero_ends"], ["zero_pattern3"]),
        helper.make_node("Cast", ["zero_pattern3"], ["zero_bool3"], to=onnx.TensorProto.BOOL),
        helper.make_node("Unsqueeze", ["zero_bool3", "block_axes"], ["block_zero6"]),
        helper.make_node("Unsqueeze", ["zero_bool3", "inner_axes"], ["inner_zero6"]),
        helper.make_node("Or", ["block_zero6", "inner_zero6"], ["zero_spatial6"]),
        helper.make_node("Flatten", ["zero_spatial6"], ["zero_spatial9"], axis=4),
        helper.make_node("MaxPool", ["input"], ["color10"], kernel_shape=[30, 30]),
        helper.make_node("Cast", ["color10"], ["color10_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Sub", ["color10_u8", "black10"], ["color_onehot_u8"]),
        helper.make_node("Where", ["zero_spatial9", "black10", "color_onehot_u8"], ["output9_u8"]),
        helper.make_node("Pad", ["output9_u8", "pads_hw", "", "pad_axes_hw"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task001_cast_zero_mask_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
