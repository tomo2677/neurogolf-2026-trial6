from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 9
INTERNAL_TYPE = onnx.TensorProto.FLOAT16


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, INTERNAL_TYPE, dims, np.asarray(values, dtype=np.float16).ravel())


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    cardinal_kernel = [
        0.0, 1.0, 0.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
    ]
    diagonal_kernel = [
        1.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 1.0,
    ]

    initializers = [
        _int64_tensor("starts1", [1, 0, 0]),
        _int64_tensor("ends1", [2, SIZE, SIZE]),
        _int64_tensor("starts2", [2, 0, 0]),
        _int64_tensor("ends2", [3, SIZE, SIZE]),
        _int64_tensor("starts6", [6, 0, 0]),
        _int64_tensor("ends6", [7, SIZE, SIZE]),
        _int64_tensor("starts8", [8, 0, 0]),
        _int64_tensor("ends8", [9, SIZE, SIZE]),
        _int64_tensor("axes3", [1, 2, 3]),
        _int64_tensor("pads_output", [0, 0, 0, 1, 21, 21]),
        _f16_tensor("cardinal_kernel", cardinal_kernel, [1, 1, 3, 3]),
        _f16_tensor("diagonal_kernel", diagonal_kernel, [1, 1, 3, 3]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("four_u8", [4], [1]),
        _u8_tensor("six_u8", [6], [1]),
        _u8_tensor("seven_u8", [7], [1]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("colors9", list(range(9)), [1, 9, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "starts1", "ends1", "axes3"], ["mask1_f32"]),
        helper.make_node("Slice", ["input", "starts2", "ends2", "axes3"], ["mask2_f32"]),
        helper.make_node("Slice", ["input", "starts6", "ends6", "axes3"], ["mask6_f32"]),
        helper.make_node("Slice", ["input", "starts8", "ends8", "axes3"], ["mask8_f32"]),
        helper.make_node("Cast", ["mask1_f32"], ["mask1_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["mask2_f32"], ["mask2_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["mask6_f32"], ["mask6_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["mask8_f32"], ["mask8_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Where", ["mask8_bool", "eight_u8", "zero_u8"], ["color8"]),
        helper.make_node("Where", ["mask6_bool", "six_u8", "color8"], ["color68"]),
        helper.make_node("Where", ["mask2_bool", "two_u8", "color68"], ["color268"]),
        helper.make_node("Where", ["mask1_bool", "one_u8", "color268"], ["color9"]),
        helper.make_node("Cast", ["color9"], ["nonblack_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["mask1_f32"], ["mask1_f16"], to=INTERNAL_TYPE),
        helper.make_node("Cast", ["mask2_f32"], ["mask2_f16"], to=INTERNAL_TYPE),
        helper.make_node("Conv", ["mask1_f16", "cardinal_kernel"], ["add7_score"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Conv", ["mask2_f16", "diagonal_kernel"], ["add4_score"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Cast", ["add7_score"], ["add7"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["add4_score"], ["add4"], to=onnx.TensorProto.BOOL),
        helper.make_node("Where", ["add4", "four_u8", "zero_u8"], ["add4_color"]),
        helper.make_node("Where", ["add7", "seven_u8", "add4_color"], ["added_color"]),
        helper.make_node("Where", ["nonblack_bool", "color9", "added_color"], ["out9"]),
        helper.make_node("Equal", ["colors9", "out9"], ["output9"]),
        helper.make_node("Pad", ["output9", "pads_output", "", "axes3"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task015_full_pads_colors9_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
