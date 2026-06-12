from __future__ import annotations

import onnx
import numpy as np
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT16, dims, np.array(values, dtype=np.float16))


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, GRID_SHAPE)

    initializers = [
        _int64_tensor("one_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("one_ends", [1, 2, 9, 9], [4]),
        _int64_tensor("output_pads", [0, 0, 0, 0, 0, 0, 29, 25], [8]),
        _f32_tensor("square_w", [1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]),
        _f32_tensor("one_f32", [1.0], [1]),
        _f32_tensor("three_half", [3.5], [1]),
        _f16_tensor("thresholds", [0.5, 1.5, 2.5, 3.5, 4.5], [1, 1, 1, 5]),
        _f32_tensor("zero8", [0.0] * 40, [1, 8, 1, 5]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "one_starts", "one_ends"], ["ones9"]),
        helper.make_node("Conv", ["ones9", "square_w"], ["square_score"], kernel_shape=[2, 2]),
        helper.make_node("Greater", ["square_score", "three_half"], ["square_bool"]),
        helper.make_node("Cast", ["square_bool"], ["square_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("ReduceSum", ["square_f16"], ["count"], axes=[0, 1, 2, 3], keepdims=1),
        helper.make_node("Greater", ["count", "thresholds"], ["unary_bool"]),
        helper.make_node("Cast", ["unary_bool"], ["unary"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Sub", ["one_f32", "unary"], ["zero_row"]),
        helper.make_node("Concat", ["zero_row", "unary", "zero8"], ["output10"], axis=1),
        helper.make_node("Pad", ["output10", "output_pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task038_count_2x2_ones_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
