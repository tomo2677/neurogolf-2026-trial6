from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
OUT = 5


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT16, dims, np.asarray(values, dtype=np.float16).ravel())


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("shape_vec5", [OUT], [1]),
        _int64_tensor("pair_sum_axes", [2, 3], [2]),
        _int64_tensor("axis_h", [2], [1]),
        _int64_tensor("axis_w", [3], [1]),
        _int32_tensor("row_grid_i32", list(range(OUT)), [1, 1, OUT, 1]),
        _int32_tensor("col_grid_i32", list(range(OUT)), [1, 1, 1, OUT]),
        _int32_tensor("zero_i32", [0], [1]),
        _int64_tensor("pads_output", [0, 0, SIZE - OUT, SIZE - OUT], [4]),
        _int64_tensor("pads_shift_right", [0, -1, 0, 1], [4]),
        _int64_tensor("pads_shift_down", [-1, 0, 1, 0], [4]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
        _f32_tensor("color_conv_w", [float(i) for i in range(10)], [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("Conv", ["input", "color_conv_w"], ["input_color_f32"]),
        helper.make_node("Cast", ["input_color_f32"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Pad", ["input_color_u8", "pads_shift_right", "", "pair_sum_axes"], ["right_color"], mode="constant"),
        helper.make_node("Equal", ["input_color_u8", "right_color"], ["same_right"]),
        helper.make_node("Where", ["same_right", "input_color_u8", "zero_u8"], ["h_pair_color"]),
        helper.make_node("Sub", ["h_pair_color", "one_u8"], ["h_pair_shifted"]),
        helper.make_node("Pad", ["input_color_u8", "pads_shift_down", "", "pair_sum_axes"], ["below_color"], mode="constant"),
        helper.make_node("Equal", ["input_color_u8", "below_color"], ["same_down"]),
        helper.make_node("Where", ["same_down", "h_pair_shifted", "invalid_u8"], ["lshape_min_base"]),
        helper.make_node("ReduceMin", ["h_pair_shifted", "pair_sum_axes"], ["h_pair_min_shifted"], keepdims=1),
        helper.make_node("ReduceMin", ["lshape_min_base", "pair_sum_axes"], ["lshape_min_shifted"], keepdims=1),
        helper.make_node("Equal", ["lshape_min_shifted", "invalid_u8"], ["lshape_empty"]),
        helper.make_node("Where", ["lshape_empty", "h_pair_min_shifted", "lshape_min_shifted"], ["target_color_shifted"]),
        helper.make_node("Add", ["target_color_shifted", "one_u8"], ["target_color_u8"]),
        helper.make_node("Equal", ["input_color_u8", "target_color_u8"], ["target_mask_bool"]),
        helper.make_node("Cast", ["target_mask_bool"], ["target_mask_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["target_mask_u8", "axis_w"], ["row_present"], keepdims=1),
        helper.make_node("ReduceMax", ["target_mask_u8", "axis_h"], ["col_present"], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_max"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min"], axis=3, keepdims=1),
        helper.make_node("ArgMax", ["col_present"], ["c_max"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["r_min"], ["r_min_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["r_max"], ["r_max_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_min"], ["c_min_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_max"], ["c_max_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Add", ["row_grid_i32", "r_min_i32"], ["src_r"]),
        helper.make_node("Add", ["col_grid_i32", "c_min_i32"], ["src_c"]),
        helper.make_node("LessOrEqual", ["src_r", "r_max_i32"], ["row_in_crop"]),
        helper.make_node("LessOrEqual", ["src_c", "c_max_i32"], ["col_in_crop"]),
        helper.make_node("And", ["row_in_crop", "col_in_crop"], ["crop_valid"]),
        helper.make_node("Where", ["row_in_crop", "src_r", "zero_i32"], ["safe_r"]),
        helper.make_node("Where", ["col_in_crop", "src_c", "zero_i32"], ["safe_c"]),
        helper.make_node("Reshape", ["safe_r", "shape_vec5"], ["safe_r_vec"]),
        helper.make_node("Reshape", ["safe_c", "shape_vec5"], ["safe_c_vec"]),
        helper.make_node("Gather", ["target_mask_bool", "safe_r_vec"], ["crop_rows_bool"], axis=2),
        helper.make_node("Gather", ["crop_rows_bool", "safe_c_vec"], ["crop_target_bool"], axis=3),
        helper.make_node("Where", ["crop_target_bool", "target_color_u8", "zero_u8"], ["crop_color"]),
        helper.make_node("Where", ["crop_valid", "crop_color", "invalid_u8"], ["color5"]),
        helper.make_node("Equal", ["colors10_u8", "color5"], ["output5"]),
        helper.make_node("Pad", ["output5", "pads_output", "", "pair_sum_axes"], ["output"], mode="constant"),
    ]

    value_infos = [
        helper.make_tensor_value_info("target_mask_bool", onnx.TensorProto.BOOL, [1, 1, SIZE, SIZE]),
        helper.make_tensor_value_info("target_mask_u8", onnx.TensorProto.UINT8, [1, 1, SIZE, SIZE]),
    ]

    graph = helper.make_graph(
        nodes,
        "task036_conv_color_map_graph",
        [x],
        [y],
        initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
