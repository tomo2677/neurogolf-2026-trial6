from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 9
KERNEL = 13
MID = 6


def _ray_offsets(corner: str) -> set[tuple[int, int]]:
    offsets: set[tuple[int, int]] = set()
    for step in range(SIZE):
        for a in (0, 1):
            for b in (0, 1):
                if corner == "tl":
                    dr, dc = -step + a, -step + b
                elif corner == "tr":
                    dr, dc = -step + a, -1 + step + b
                elif corner == "bl":
                    dr, dc = -1 + step + a, -step + b
                else:
                    dr, dc = -1 + step + a, -1 + step + b
                if -MID <= dr <= MID and -MID <= dc <= MID:
                    offsets.add((dr, dc))
    return offsets


def _ray_kernel() -> list[float]:
    weights = [0.0] * (4 * KERNEL * KERNEL)
    for channel, corner in enumerate(("tl", "tr", "bl", "br")):
        for dr, dc in _ray_offsets(corner):
            kr = MID - dr
            kc = MID - dc
            weights[(channel * KERNEL + kr) * KERNEL + kc] = 1.0
    return weights


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT16, dims, np.array(values, dtype=np.float16))


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _bool_tensor(name: str, values: list[bool], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.BOOL, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.UINT8, GRID_SHAPE)

    initializers = [
        _int64_tensor("marker_starts", [0, 2, 0, 0], [4]),
        _int64_tensor("marker_ends", [1, 3, SIZE, SIZE], [4]),
        _int64_tensor("input0_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("input0_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("start1", [1], [1]),
        _int64_tensor("end9", [SIZE], [1]),
        _int64_tensor("start0", [0], [1]),
        _int64_tensor("end8", [SIZE - 1], [1]),
        _int64_tensor("axis_row", [2], [1]),
        _int64_tensor("axis_col", [3], [1]),
        _int64_tensor("pads_output_hw", [0, 0, 21, 21], [4]),
        _int64_tensor("pad_axes_hw", [2, 3], [2]),
        _f16_tensor("ray_w", _ray_kernel(), [1, 4, KERNEL, KERNEL]),
        _u8_tensor("black_marker10", [1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 10, 1, 1]),
        _bool_tensor("false_col", [False] * SIZE, [1, 1, SIZE, 1]),
        _bool_tensor("false_row", [False] * SIZE, [1, 1, 1, SIZE]),
        _u8_tensor("black10", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "marker_starts", "marker_ends"], ["marker2_f32"]),
        helper.make_node("Cast", ["marker2_f32"], ["marker2"], to=onnx.TensorProto.BOOL),
        helper.make_node("Slice", ["input", "input0_starts", "input0_ends"], ["input0_9"]),
        helper.make_node("Cast", ["input0_9"], ["black9"], to=onnx.TensorProto.BOOL),
        helper.make_node("Not", ["black9"], ["nonzero"]),
        helper.make_node("Slice", ["nonzero", "start1", "end9", "axis_col"], ["right_core"]),
        helper.make_node("Concat", ["right_core", "false_col"], ["right"], axis=3),
        helper.make_node("Slice", ["nonzero", "start0", "end8", "axis_col"], ["left_core"]),
        helper.make_node("Concat", ["false_col", "left_core"], ["left"], axis=3),
        helper.make_node("Slice", ["nonzero", "start1", "end9", "axis_row"], ["down_core"]),
        helper.make_node("Concat", ["down_core", "false_row"], ["down"], axis=2),
        helper.make_node("Slice", ["nonzero", "start0", "end8", "axis_row"], ["up_core"]),
        helper.make_node("Concat", ["false_row", "up_core"], ["up"], axis=2),
        helper.make_node("And", ["marker2", "right"], ["marker_right"]),
        helper.make_node("And", ["marker2", "left"], ["marker_left"]),
        helper.make_node("And", ["marker_right", "down"], ["tl"]),
        helper.make_node("And", ["marker_left", "down"], ["tr"]),
        helper.make_node("And", ["marker_right", "up"], ["bl"]),
        helper.make_node("And", ["marker_left", "up"], ["br"]),
        helper.make_node("Concat", ["tl", "tr", "bl", "br"], ["corner_bool"], axis=1),
        helper.make_node("Cast", ["corner_bool"], ["corner_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("Conv", ["corner_f16", "ray_w"], ["mask_score"], kernel_shape=[KERNEL, KERNEL], pads=[MID, MID, MID, MID]),
        helper.make_node("Cast", ["mask_score"], ["mask9"], to=onnx.TensorProto.BOOL),
        helper.make_node("MaxPool", ["input"], ["present10"], kernel_shape=[30, 30]),
        helper.make_node("Cast", ["present10"], ["present10_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Sub", ["present10_u8", "black_marker10"], ["color_onehot_u8"]),
        helper.make_node("Where", ["mask9", "color_onehot_u8", "black10"], ["output9_u8"]),
        helper.make_node("Pad", ["output9_u8", "pads_output_hw", "", "pad_axes_hw"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task034_diagonal_marker_direct_onehot", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
