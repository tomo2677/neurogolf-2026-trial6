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

    top_pattern = [
        (r in {0, 2}) or (r in {1, 3, 4} and c in {0, 9})
        for r in range(10)
        for c in range(10)
    ]
    bottom_pattern = [
        (r in {7, 9}) or (r in {5, 6, 8} and c in {0, 9})
        for r in range(10)
        for c in range(10)
    ]

    initializers = [
        _int64_tensor("top_starts", [0, 1, 2, 0], [4]),
        _int64_tensor("top_ends", [1, 10, 3, 10], [4]),
        _int64_tensor("bottom_starts", [0, 1, 7, 0], [4]),
        _int64_tensor("bottom_ends", [1, 10, 8, 10], [4]),
        _int64_tensor("width_axis", [3], [1]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 20, 20], [8]),
        _bool_tensor("top_pattern", top_pattern, [1, 1, 10, 10]),
        _bool_tensor("bottom_pattern", bottom_pattern, [1, 1, 10, 10]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "top_starts", "top_ends"], ["top_row"]),
        helper.make_node("ReduceMax", ["top_row", "width_axis"], ["top_present"], keepdims=1),
        helper.make_node("ArgMax", ["top_present"], ["top_idx0"], axis=1, keepdims=1),
        helper.make_node("Add", ["top_idx0", "one_i64"], ["top_color_i64"]),
        helper.make_node("Cast", ["top_color_i64"], ["top_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["input", "bottom_starts", "bottom_ends"], ["bottom_row"]),
        helper.make_node("ReduceMax", ["bottom_row", "width_axis"], ["bottom_present"], keepdims=1),
        helper.make_node("ArgMax", ["bottom_present"], ["bottom_idx0"], axis=1, keepdims=1),
        helper.make_node("Add", ["bottom_idx0", "one_i64"], ["bottom_color_i64"]),
        helper.make_node("Cast", ["bottom_color_i64"], ["bottom_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Where", ["top_pattern", "top_color_u8", "zero_u8"], ["top_color10"]),
        helper.make_node("Where", ["bottom_pattern", "bottom_color_u8", "top_color10"], ["color10"]),
        helper.make_node("Pad", ["color10", "pads_output", "invalid_u8"], ["color30"], mode="constant"),
        helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task028_fixed_rows_template_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
