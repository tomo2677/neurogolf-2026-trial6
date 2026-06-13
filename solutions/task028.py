from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _edge_col(top_name: str, bottom_name: str) -> list[str]:
    return [top_name] * 5 + [bottom_name] * 5


def _inner_col(top_name: str, bottom_name: str) -> list[str]:
    return [
        top_name,
        "zero_u8",
        top_name,
        "zero_u8",
        "zero_u8",
        "zero_u8",
        "zero_u8",
        bottom_name,
        "zero_u8",
        bottom_name,
    ]


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("pads_hw", [0, 0, 20, 20], [4]),
        _int64_tensor("pad_axes_hw", [2, 3], [2]),
        _u8_tensor("zero_u8", [0], [1, 1, 1, 1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("MaxPool", ["input"], ["top_present"], kernel_shape=[5, 30], strides=[30, 30]),
        helper.make_node("MaxPool", ["input"], ["both_present"], kernel_shape=[10, 30], strides=[30, 30]),
        helper.make_node("ArgMax", ["top_present"], ["top_idx"], axis=1, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["top_idx"], ["top_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["top_present"], ["top_present_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["both_present"], ["both_present_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Sub", ["both_present_u8", "top_present_u8"], ["bottom_diff"]),
        helper.make_node("ArgMax", ["bottom_diff"], ["bottom_idx"], axis=1, keepdims=1),
        helper.make_node("Cast", ["bottom_idx"], ["bottom_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Concat", _edge_col("top_color_u8", "bottom_color_u8"), ["edge_col"], axis=2),
        helper.make_node("Concat", _inner_col("top_color_u8", "bottom_color_u8"), ["inner_col"], axis=2),
        helper.make_node(
            "Concat",
            [
                "edge_col",
                "inner_col",
                "inner_col",
                "inner_col",
                "inner_col",
                "inner_col",
                "inner_col",
                "inner_col",
                "inner_col",
                "edge_col",
            ],
            ["color10"],
            axis=3,
        ),
        helper.make_node("Pad", ["color10", "pads_hw", "invalid_u8", "pad_axes_hw"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task028_zero_scalar_edge_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
