from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 9
KERNEL = 17
MID = 8


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _bool_tensor(name: str, values: list[bool], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.BOOL, dims, values)


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


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("marker_starts", [0, 2, 0, 0], [4]),
        _int64_tensor("marker_ends", [1, 3, SIZE, SIZE], [4]),
        _int64_tensor("input0_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("input0_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("row_start1", [1], [1]),
        _int64_tensor("row_end9", [SIZE], [1]),
        _int64_tensor("row_start0", [0], [1]),
        _int64_tensor("row_end8", [SIZE - 1], [1]),
        _int64_tensor("col_start1", [1], [1]),
        _int64_tensor("col_end9", [SIZE], [1]),
        _int64_tensor("col_start0", [0], [1]),
        _int64_tensor("col_end8", [SIZE - 1], [1]),
        _int64_tensor("axis_row", [2], [1]),
        _int64_tensor("axis_col", [3], [1]),
        _int64_tensor("pads_color_grid", [0, 0, 0, 0, 0, 0, 21, 21], [8]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("one_f32", [1.0], [1]),
        _f32_tensor("ray_w", _ray_kernel(), [1, 4, KERNEL, KERNEL]),
        _f32_tensor("color_keep", [0.0, 1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 10, 1, 1]),
        _bool_tensor("false_col", [False] * SIZE, [1, 1, SIZE, 1]),
        _bool_tensor("false_row", [False] * SIZE, [1, 1, 1, SIZE]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "marker_starts", "marker_ends"], ["marker2_f32"]),
        helper.make_node("Greater", ["marker2_f32", "zero_f32"], ["marker2"]),
        helper.make_node("Slice", ["input", "input0_starts", "input0_ends"], ["input0_9"]),
        helper.make_node("Sub", ["one_f32", "input0_9"], ["nonzero_f32"]),
        helper.make_node("Greater", ["nonzero_f32", "zero_f32"], ["nonzero"]),
        helper.make_node("Slice", ["nonzero", "col_start1", "col_end9", "axis_col"], ["right_core"]),
        helper.make_node("Concat", ["right_core", "false_col"], ["right"], axis=3),
        helper.make_node("Slice", ["nonzero", "col_start0", "col_end8", "axis_col"], ["left_core"]),
        helper.make_node("Concat", ["false_col", "left_core"], ["left"], axis=3),
        helper.make_node("Slice", ["nonzero", "row_start1", "row_end9", "axis_row"], ["down_core"]),
        helper.make_node("Concat", ["down_core", "false_row"], ["down"], axis=2),
        helper.make_node("Slice", ["nonzero", "row_start0", "row_end8", "axis_row"], ["up_core"]),
        helper.make_node("Concat", ["false_row", "up_core"], ["up"], axis=2),
        helper.make_node("And", ["marker2", "right"], ["marker_right"]),
        helper.make_node("And", ["marker2", "left"], ["marker_left"]),
        helper.make_node("And", ["marker_right", "down"], ["tl"]),
        helper.make_node("And", ["marker_left", "down"], ["tr"]),
        helper.make_node("And", ["marker_right", "up"], ["bl"]),
        helper.make_node("And", ["marker_left", "up"], ["br"]),
        helper.make_node("Concat", ["tl", "tr", "bl", "br"], ["corner_bool"], axis=1),
        helper.make_node("Cast", ["corner_bool"], ["corner_f32"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Conv", ["corner_f32", "ray_w"], ["mask_score"], kernel_shape=[KERNEL, KERNEL], pads=[MID, MID, MID, MID]),
        helper.make_node("Greater", ["mask_score", "zero_f32"], ["mask9"]),
        helper.make_node("ReduceMax", ["input"], ["present10"], axes=[2, 3], keepdims=1),
        helper.make_node("Mul", ["present10", "color_keep"], ["color_scores"]),
        helper.make_node("ArgMax", ["color_scores"], ["color_i64"], axis=1, keepdims=1),
        helper.make_node("Cast", ["color_i64"], ["color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Where", ["mask9", "color_u8", "zero_u8"], ["color_grid9"]),
        helper.make_node("Pad", ["color_grid9", "pads_color_grid", "invalid_u8"], ["color_grid30"], mode="constant"),
        helper.make_node("Equal", ["colors10", "color_grid30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task034_diagonal_marker_rays_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
