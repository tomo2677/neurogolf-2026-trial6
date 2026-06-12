from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 9
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _pads_for_shift(dy: int, dx: int) -> list[int]:
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
    return [0, 0, top, left, 0, 0, bottom, right]


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("starts9", [0, 0, 0]),
        _int64_tensor("ends9", [10, SIZE, SIZE]),
        _int64_tensor("starts1", [1, 0, 0]),
        _int64_tensor("ends1", [2, SIZE, SIZE]),
        _int64_tensor("starts2", [2, 0, 0]),
        _int64_tensor("ends2", [3, SIZE, SIZE]),
        _int64_tensor("nonblack_start", [1, 0, 0]),
        _int64_tensor("nonblack_end", [10, SIZE, SIZE]),
        _int64_tensor("axes3", [1, 2, 3]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 21, 21]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("four_u8", [4], [1]),
        _u8_tensor("seven_u8", [7], [1]),
        _u8_tensor("outside_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]
    for dy, dx in set(CARDINALS + DIAGONALS):
        initializers.append(_int64_tensor(f"pads_{dy}_{dx}", _pads_for_shift(dy, dx), [8]))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "starts9", "ends9", "axes3"], ["input9"]),
        helper.make_node("ArgMax", ["input9"], ["color9_i64"], axis=1, keepdims=1),
        helper.make_node("Cast", ["color9_i64"], ["color9"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["input", "starts1", "ends1", "axes3"], ["mask1_f32"]),
        helper.make_node("Slice", ["input", "starts2", "ends2", "axes3"], ["mask2_f32"]),
        helper.make_node("Cast", ["mask1_f32"], ["mask1"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["mask2_f32"], ["mask2"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["input", "nonblack_start", "nonblack_end", "axes3"], ["nonblack_input"]),
        helper.make_node("ReduceMax", ["nonblack_input"], ["nonblack_f32"], axes=[1], keepdims=1),
        helper.make_node("Cast", ["nonblack_f32"], ["nonblack_bool"], to=onnx.TensorProto.BOOL),
    ]

    card_outputs: list[str] = []
    for index, (dy, dx) in enumerate(CARDINALS):
        out = f"card_{index}"
        nodes.append(helper.make_node("Pad", ["mask1", f"pads_{dy}_{dx}", "zero_u8"], [out], mode="constant"))
        card_outputs.append(out)
    diag_outputs: list[str] = []
    for index, (dy, dx) in enumerate(DIAGONALS):
        out = f"diag_{index}"
        nodes.append(helper.make_node("Pad", ["mask2", f"pads_{dy}_{dx}", "zero_u8"], [out], mode="constant"))
        diag_outputs.append(out)

    nodes.extend(
        [
            helper.make_node("Max", card_outputs, ["add7_u8"]),
            helper.make_node("Max", diag_outputs, ["add4_u8"]),
            helper.make_node("Cast", ["add7_u8"], ["add7"], to=onnx.TensorProto.BOOL),
            helper.make_node("Cast", ["add4_u8"], ["add4"], to=onnx.TensorProto.BOOL),
            helper.make_node("Where", ["add4", "four_u8", "zero_u8"], ["add4_color"]),
            helper.make_node("Where", ["add7", "seven_u8", "add4_color"], ["added_color"]),
            helper.make_node("Where", ["nonblack_bool", "color9", "added_color"], ["out9"]),
            helper.make_node("Pad", ["out9", "pads_output", "outside_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task015_single_cell_expansion_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
