from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 10
NONZERO = 9


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT16, dims, np.asarray(values, dtype=np.float16).ravel())


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.UINT8, GRID_SHAPE)

    initializers = [
        _int64_tensor("nonzero_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("nonzero_ends", [1, 10, SIZE, SIZE], [4]),
        _int64_tensor("output_pads", [0, 0, 0, 0, 0, 0, 20, 20], [8]),
        _f16_tensor("row_grid_i32", [float(v) for v in range(SIZE)], [1, 1, SIZE, 1]),
        _f16_tensor("col_grid_i32", [float(v) for v in range(SIZE)], [1, 1, 1, SIZE]),
        _f16_tensor("diag_offset_i32", [float(SIZE - 1)], [1]),
        _f16_tensor("zero_i32", [0.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "nonzero_starts", "nonzero_ends"], ["input9"]),
        helper.make_node("Cast", ["input9"], ["input9_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("ReduceMax", ["input9"], ["row_present"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["input9"], ["col_present"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min_i64"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_max_i64"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min_i64"], axis=3, keepdims=1),
        helper.make_node("ArgMax", ["col_present"], ["c_max_i64"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["r_min_i64"], ["r_min"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("Cast", ["r_max_i64"], ["r_max"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("Cast", ["c_min_i64"], ["c_min"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("Cast", ["c_max_i64"], ["c_max"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("LessOrEqual", ["r_min", "row_grid_i32"], ["row_after_min"]),
        helper.make_node("LessOrEqual", ["row_grid_i32", "r_max"], ["row_before_max"]),
        helper.make_node("LessOrEqual", ["c_min", "col_grid_i32"], ["col_after_min"]),
        helper.make_node("LessOrEqual", ["col_grid_i32", "c_max"], ["col_before_max"]),
        helper.make_node("And", ["row_after_min", "row_before_max"], ["row_between"]),
        helper.make_node("And", ["col_after_min", "col_before_max"], ["col_between"]),
        helper.make_node("And", ["row_between", "col_between"], ["bbox"]),
        helper.make_node("Sub", ["row_grid_i32", "col_grid_i32"], ["main_raw"]),
        helper.make_node("Add", ["main_raw", "diag_offset_i32"], ["main_grid"]),
        helper.make_node("Add", ["row_grid_i32", "col_grid_i32"], ["anti_grid"]),
        helper.make_node("Where", ["input9_bool", "main_grid", "zero_i32"], ["main_points"]),
        helper.make_node("Where", ["input9_bool", "anti_grid", "zero_i32"], ["anti_points"]),
        helper.make_node("ReduceMax", ["main_points"], ["main_const"], axes=[2, 3], keepdims=1),
        helper.make_node("ReduceMax", ["anti_points"], ["anti_const"], axes=[2, 3], keepdims=1),
        helper.make_node("Equal", ["main_grid", "main_const"], ["main_diag"]),
        helper.make_node("Equal", ["anti_grid", "anti_const"], ["anti_diag"]),
        helper.make_node("Or", ["main_diag", "anti_diag"], ["diag_any"]),
        helper.make_node("ReduceMax", ["input9"], ["present_f32"], axes=[2, 3], keepdims=1),
        helper.make_node("Cast", ["present_f32"], ["present"], to=onnx.TensorProto.BOOL),
        helper.make_node("And", ["diag_any", "bbox"], ["line_box"]),
        helper.make_node("And", ["line_box", "present"], ["line_bool"]),
        helper.make_node("Cast", ["line_bool"], ["line9_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["line9_u8"], ["line_any_u8"], axes=[1], keepdims=1),
        helper.make_node("Equal", ["line_any_u8", "zero_u8"], ["zero10_bool"]),
        helper.make_node("Cast", ["zero10_bool"], ["zero10_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Concat", ["zero10_u8", "line9_u8"], ["output10_u8"], axis=1),
        helper.make_node("Pad", ["output10_u8", "output_pads"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task037_f16_diag_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
