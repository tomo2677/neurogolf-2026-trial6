from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.UINT8, GRID_SHAPE)

    initializers = [
        _int64_tensor("blue_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("blue_ends", [1, 2, 10, 10], [4]),
        _int64_tensor("col7_tail_starts", [0, 0, 5, 7], [4]),
        _int64_tensor("col7_tail_ends", [1, 1, 7, 8], [4]),
        _int64_tensor("rev10_starts", [9, 9], [2]),
        _int64_tensor("rev10_ends", [-11, -11], [2]),
        _int64_tensor("rot10_starts", [9, 9], [2]),
        _int64_tensor("rot10_ends", [0, 0], [2]),
        _int64_tensor("rev_axes", [2, 3], [2]),
        _int64_tensor("rev_steps", [-1, -1], [2]),
        _int64_tensor("pads_rot10_hw", [1, 1, 0, 0], [4]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 7, 20, 20], [8]),
        _int64_tensor("axis_h", [2], [1]),
        _int64_tensor("axis_w", [3], [1]),
        _int64_tensor("nine_i64", [9], [1]),
        _int64_tensor("three_i64", [3], [1]),
        helper.make_tensor("one_u8", onnx.TensorProto.UINT8, [1], [1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "blue_starts", "blue_ends"], ["blue10"]),
        helper.make_node("Cast", ["blue10"], ["blue_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["blue_u8", "rev10_starts", "rev10_ends", "rev_axes", "rev_steps"], ["rot9"]),
        helper.make_node("Slice", ["blue_u8", "rot10_starts", "rot10_ends", "rev_axes", "rev_steps"], ["rot10_inner"]),
        helper.make_node("Pad", ["rot10_inner", "pads_rot10_hw", "", "rev_axes"], ["rot10"], mode="constant"),
        helper.make_node("ReduceMax", ["blue_u8", "axis_w"], ["row_present"], keepdims=1),
        helper.make_node("Slice", ["blue_u8", "col7_tail_starts", "col7_tail_ends"], ["col7_tail"]),
        helper.make_node("ReduceMax", ["col7_tail", "axis_h"], ["col7_any_u8"], keepdims=1),
        helper.make_node("Cast", ["col7_any_u8"], ["col7_hit"], to=onnx.TensorProto.BOOL),
        helper.make_node("ArgMax", ["row_present"], ["r_min"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_max"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("Add", ["r_min", "r_max"], ["r_sum"]),
        helper.make_node("Greater", ["r_sum", "nine_i64"], ["r_sum_gt9"]),
        helper.make_node("Equal", ["r_min", "three_i64"], ["r_min_is3"]),
        helper.make_node("And", ["r_min_is3", "col7_hit"], ["exception_use10"]),
        helper.make_node("Or", ["r_sum_gt9", "exception_use10"], ["use10"]),
        helper.make_node("Where", ["use10", "rot10", "rot9"], ["selected_rot"]),
        helper.make_node("Sub", ["one_u8", "blue_u8"], ["not_blue"]),
        helper.make_node("BitwiseAnd", ["selected_rot", "not_blue"], ["red_u8"]),
        helper.make_node("BitwiseXor", ["not_blue", "red_u8"], ["black_u8"]),
        helper.make_node("Concat", ["black_u8", "blue_u8", "red_u8"], ["output3"], axis=1),
        helper.make_node("Pad", ["output3", "pads_output"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task027_u8_where_select_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
