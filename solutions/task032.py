from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


CHANNELS = 10
KERNEL_H = 10
MID = 5
GATE = 10.0
EPS = 0.25


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _conv_weight_values() -> list[float]:
    weights = [0.0] * (CHANNELS * CHANNELS * KERNEL_H)

    def add(out_ch: int, in_ch: int, kernel_row: int, value: float) -> None:
        offset = ((out_ch * CHANNELS + in_ch) * KERNEL_H) + kernel_row
        weights[offset] += value

    for color in range(1, CHANNELS):
        for kernel_row in range(MID + 1):
            add(color, color, kernel_row, 1.0)
        for kernel_row in range(MID + 1, KERNEL_H):
            add(color, 0, kernel_row, -1.0)
        for in_ch in range(CHANNELS):
            add(color, in_ch, MID, GATE)

    for kernel_row in range(MID + 1, KERNEL_H):
        add(0, 0, kernel_row, 1.0)
    for in_ch in range(1, CHANNELS):
        for kernel_row in range(MID + 1):
            add(0, in_ch, kernel_row, -1.0)
    for in_ch in range(CHANNELS):
        add(0, in_ch, MID, GATE + EPS)

    return weights


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _f32_tensor("W", _conv_weight_values(), [CHANNELS, CHANNELS, KERNEL_H, 1]),
        _f32_tensor("B", [-GATE] * CHANNELS, [CHANNELS]),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "W", "B"],
            ["output"],
            kernel_shape=[KERNEL_H, 1],
            pads=[MID, 0, KERNEL_H - MID - 1, 0],
        ),
    ]

    graph = helper.make_graph(nodes, "task032_compact_conv_column_gravity_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
