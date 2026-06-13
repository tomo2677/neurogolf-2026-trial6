from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("row_idx", [float(v) for v in range(SIZE)], [1, 1, SIZE, 1]),
        _f32_tensor("col_idx", [float(v) for v in range(SIZE)], [1, 1, 1, SIZE]),
        _int64_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _int64_tensor("axes_counts", [0, 2, 3], [3]),
        _int64_tensor("axis_col", [3], [1]),
        _int64_tensor("axis_row", [2], [1]),
        _int64_tensor("axes_all", [0, 1, 2, 3], [4]),
        _int64_tensor("row_prev_pads", [1, -1], [2]),
        _int64_tensor("col_prev_pads", [1, -1], [2]),
    ]

    nodes = [
        helper.make_node("ReduceSum", ["input", "axes_counts"], ["counts10"], keepdims=0),
        helper.make_node("ArgMax", ["counts10"], ["bg_idx"], axis=0, keepdims=1),
        helper.make_node("Equal", ["colors10", "bg_idx"], ["bg_channel_bool"]),
        helper.make_node("ReduceMax", ["input", "axis_col"], ["row_has_bg10"], keepdims=1),
        helper.make_node("ReduceMax", ["input", "axis_row"], ["col_has_bg10"], keepdims=1),
        helper.make_node("Gather", ["row_has_bg10", "bg_idx"], ["row_has_bg"], axis=1),
        helper.make_node("Gather", ["col_has_bg10", "bg_idx"], ["col_has_bg"], axis=1),
        helper.make_node("Greater", ["row_has_bg", "zero_f32"], ["row_non_sep_bool"]),
        helper.make_node("Greater", ["col_has_bg", "zero_f32"], ["col_non_sep_bool"]),
        helper.make_node("Pad", ["row_non_sep_bool", "row_prev_pads", "", "axis_row"], ["row_prev_bool"], mode="constant"),
        helper.make_node("Pad", ["col_non_sep_bool", "col_prev_pads", "", "axis_col"], ["col_prev_bool"], mode="constant"),
        helper.make_node("Not", ["row_prev_bool"], ["row_prev_not"]),
        helper.make_node("Not", ["col_prev_bool"], ["col_prev_not"]),
        helper.make_node("And", ["row_non_sep_bool", "row_prev_not"], ["row_start_bool"]),
        helper.make_node("And", ["col_non_sep_bool", "col_prev_not"], ["col_start_bool"]),
        helper.make_node("Cast", ["row_start_bool"], ["row_start_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("Cast", ["col_start_bool"], ["col_start_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("ReduceSum", ["row_start_f16", "axes_all"], ["out_h_f16"], keepdims=1),
        helper.make_node("ReduceSum", ["col_start_f16", "axes_all"], ["out_w_f16"], keepdims=1),
        helper.make_node("Cast", ["out_h_f16"], ["out_h"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Cast", ["out_w_f16"], ["out_w"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Less", ["row_idx", "out_h"], ["row_in"]),
        helper.make_node("Less", ["col_idx", "out_w"], ["col_in"]),
        helper.make_node("And", ["bg_channel_bool", "row_in"], ["bg_rows"]),
        helper.make_node("And", ["bg_rows", "col_in"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task021_f16_start_sum_cast_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
