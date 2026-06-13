from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 10


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _bool_tensor(name: str, values: list[bool], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.BOOL, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    row_top_values = [r < 5 for r in range(SIZE)]
    col_left_values = [c < 5 for c in range(SIZE)]
    top_edge_values = [r == 0 for r in range(SIZE)]
    bottom_edge_values = [r == SIZE - 1 for r in range(SIZE)]
    left_edge_values = [c == 0 for c in range(SIZE)]
    right_edge_values = [c == SIZE - 1 for c in range(SIZE)]

    initializers = [
        _int64_tensor("top_zero_starts", [0, 0, 0, 2], [4]),
        _int64_tensor("top_zero_ends", [1, 1, 1, 4], [4]),
        _int64_tensor("marker_starts", [0, 3, 0, 0], [4]),
        _int64_tensor("marker_ends", [1, 4, SIZE, SIZE], [4]),
        _int64_tensor("top_left_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("top_left_ends", [1, 10, 1, 1], [4]),
        _int64_tensor("top_right_starts", [0, 1, 0, SIZE - 1], [4]),
        _int64_tensor("top_right_ends", [1, 10, 1, SIZE], [4]),
        _int64_tensor("bottom_left_starts", [0, 1, SIZE - 1, 0], [4]),
        _int64_tensor("bottom_left_ends", [1, 10, SIZE, 1], [4]),
        _int64_tensor("output_pads_hw", [0, 0, 20, 20], [4]),
        _int64_tensor("output_pad_axes", [2, 3], [2]),
        _u8_tensor("bg_u8", [9], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", [9, 0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 10, 1, 1]),
        _bool_tensor("row_top", row_top_values, [1, 1, SIZE, 1]),
        _bool_tensor("col_left", col_left_values, [1, 1, 1, SIZE]),
        _bool_tensor("top_edge", top_edge_values, [1, 1, SIZE, 1]),
        _bool_tensor("bottom_edge", bottom_edge_values, [1, 1, SIZE, 1]),
        _bool_tensor("left_edge", left_edge_values, [1, 1, 1, SIZE]),
        _bool_tensor("right_edge", right_edge_values, [1, 1, 1, SIZE]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "top_zero_starts", "top_zero_ends"], ["top_zero"]),
        helper.make_node("MaxPool", ["top_zero"], ["top_has_zero_f32"], kernel_shape=[1, 2]),
        helper.make_node("Cast", ["top_has_zero_f32"], ["top_has_zero"], to=onnx.TensorProto.BOOL),
        helper.make_node("Not", ["top_has_zero"], ["use_rows"]),
        helper.make_node("Slice", ["input", "marker_starts", "marker_ends"], ["marker_f32"]),
        helper.make_node("Cast", ["marker_f32"], ["marker"], to=onnx.TensorProto.BOOL),
    ]

    for name in ("top_left", "top_right", "bottom_left"):
        nodes.extend(
            [
                helper.make_node("Slice", ["input", f"{name}_starts", f"{name}_ends"], [f"{name}_onehot"]),
                helper.make_node("ArgMax", [f"{name}_onehot"], [f"{name}_i64"], axis=1, keepdims=1),
                helper.make_node("Cast", [f"{name}_i64"], [f"{name}_color"], to=onnx.TensorProto.UINT8),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Where", ["row_top", "top_left_color", "bottom_left_color"], ["row_replacement"]),
            helper.make_node("Where", ["col_left", "top_left_color", "top_right_color"], ["col_replacement"]),
            helper.make_node("Where", ["use_rows", "row_replacement", "col_replacement"], ["replacement"]),
            helper.make_node("Where", ["top_edge", "top_left_color", "bg_u8"], ["top_line"]),
            helper.make_node("Where", ["bottom_edge", "bottom_left_color", "top_line"], ["row_guides"]),
            helper.make_node("Where", ["left_edge", "top_left_color", "bg_u8"], ["left_line"]),
            helper.make_node("Where", ["right_edge", "top_right_color", "left_line"], ["col_guides"]),
            helper.make_node("Where", ["use_rows", "row_guides", "col_guides"], ["guides"]),
            helper.make_node("Where", ["marker", "replacement", "guides"], ["color10"]),
            helper.make_node("Pad", ["color10", "output_pads_hw", "invalid_u8", "output_pad_axes"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task040_top_zero_width2_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
