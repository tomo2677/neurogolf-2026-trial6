from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 12
CENTER_OFFSETS = [(-2, -2), (-2, 2), (-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1), (2, -2), (2, 2)]
ARM_OFFSETS = [(-2, 0), (-1, 0), (0, -2), (0, -1), (0, 1), (0, 2), (1, 0), (2, 0)]


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, np.array(values, dtype=np.float32))


def _shift(nodes: list[onnx.NodeProto], source: str, dy: int, dx: int, output: str) -> None:
    pad_name = f"pads_shift_{dy}_{dx}"
    nodes.append(helper.make_node("Pad", [source, pad_name, "zero_u8"], [output], mode="constant"))


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    cross_kernel = np.array(
        [[[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=np.float32,
    )
    initializers = [
        _int64_tensor("starts12", [0, 0, 0]),
        _int64_tensor("ends12", [10, SIZE, SIZE]),
        _int64_tensor("nonblack_start", [1, 0, 0]),
        _int64_tensor("nonblack_end", [10, SIZE, SIZE]),
        _int64_tensor("axes3", [1, 2, 3]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 18, 18]),
        _f32_tensor("cross_kernel", cross_kernel.ravel().tolist(), [1, 1, 3, 3]),
        _f32_tensor("four_f32", [4.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _u8_tensor("outside_u8", [255], [1]),
    ]
    for dy, dx in {(1, 0), *CENTER_OFFSETS, *ARM_OFFSETS}:
        top = max(dy, 0)
        bottom = min(-dy, 0)
        left = max(dx, 0)
        right = min(-dx, 0)
        if dy < 0:
            top = dy
            bottom = -dy
        if dx < 0:
            left = dx
            right = -dx
        initializers.append(_int64_tensor(f"pads_shift_{dy}_{dx}", [0, 0, top, left, 0, 0, bottom, right], [8]))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "starts12", "ends12", "axes3"], ["input12"]),
        helper.make_node("Slice", ["input", "nonblack_start", "nonblack_end", "axes3"], ["nonblack12"]),
        helper.make_node("ReduceMax", ["nonblack12"], ["nonzero_f32"], axes=[1], keepdims=1),
        helper.make_node("Cast", ["nonzero_f32"], ["nonzero_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Conv", ["nonzero_f32", "cross_kernel"], ["neighbor_count"], pads=[1, 1, 1, 1]),
        helper.make_node("Equal", ["neighbor_count", "four_f32"], ["has_four_neighbors"]),
        helper.make_node("And", ["nonzero_bool", "has_four_neighbors"], ["center_mask"]),
        helper.make_node("ArgMax", ["input12"], ["color12_i64"], axis=1, keepdims=1),
        helper.make_node("Cast", ["color12_i64"], ["color12"], to=onnx.TensorProto.UINT8),
    ]

    _shift(nodes, "color12", 1, 0, "up_color")
    nodes.extend(
        [
            helper.make_node("Where", ["center_mask", "color12", "zero_u8"], ["center_value"]),
            helper.make_node("Where", ["center_mask", "up_color", "zero_u8"], ["arm_value"]),
        ]
    )

    shifted: list[str] = []
    for index, (dy, dx) in enumerate(CENTER_OFFSETS):
        name = f"center_shift_{index}"
        _shift(nodes, "center_value", dy, dx, name)
        shifted.append(name)
    for index, (dy, dx) in enumerate(ARM_OFFSETS):
        name = f"arm_shift_{index}"
        _shift(nodes, "arm_value", dy, dx, name)
        shifted.append(name)

    nodes.extend(
        [
            helper.make_node("Max", shifted, ["color12_out"]),
            helper.make_node("Pad", ["color12_out", "pads_output", "outside_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task012_cross_expand_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
