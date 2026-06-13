from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


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

    initializers = [
        _int64_tensor("window_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("window_ends", [1, 10, 2, 1], [4]),
        _int64_tensor("shape_row10", [1, 1, 1, 10], [4]),
        _int64_tensor("pads_hw", [0, 0, 20, 20], [4]),
        _int64_tensor("pad_axes_hw", [2, 3], [2]),
        _bool_tensor("edge_cols", [True, False, False, False, False, False, False, False, False, True], [1, 1, 1, 10]),
        _u8_tensor("bg_u8", [9], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", [9, 0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("MaxPool", ["input"], ["row_windows"], kernel_shape=[5, 30], strides=[5, 30]),
        helper.make_node("Slice", ["row_windows", "window_starts", "window_ends"], ["nonblack_windows"]),
        helper.make_node("ArgMax", ["nonblack_windows"], ["color_idx0"], axis=1, keepdims=1),
        helper.make_node("Cast", ["color_idx0"], ["color_idx_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Split", ["color_idx_u8"], ["top_color_u8", "bottom_color_u8"], axis=2, num_outputs=2),
        helper.make_node("Expand", ["top_color_u8", "shape_row10"], ["top_full"]),
        helper.make_node("Where", ["edge_cols", "top_color_u8", "bg_u8"], ["top_edge"]),
        helper.make_node("Expand", ["bottom_color_u8", "shape_row10"], ["bottom_full"]),
        helper.make_node("Where", ["edge_cols", "bottom_color_u8", "bg_u8"], ["bottom_edge"]),
        helper.make_node(
            "Concat",
            [
                "top_full",
                "top_edge",
                "top_full",
                "top_edge",
                "top_edge",
                "bottom_edge",
                "bottom_edge",
                "bottom_full",
                "bottom_edge",
                "bottom_full",
            ],
            ["color10"],
            axis=2,
        ),
        helper.make_node("Pad", ["color10", "pads_hw", "invalid_u8", "pad_axes_hw"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task028_window_color_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
