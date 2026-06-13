from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 25
GRID_SIZE = 30
OUT = 23


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT16, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int32_tensor("one_i32", [1], [1]),
        _int32_tensor("zero_i32", [0], [1]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("shape9", [9], [1]),
        _int64_tensor("shape_1x9x1x1", [1, 9, 1, 1], [4]),
        _int64_tensor("shape1111", [1, 1, 1, 1], [4]),
        _int64_tensor("shape_vec23", [OUT], [1]),
        _int64_tensor("count_starts", [1], [1]),
        _int64_tensor("count_ends", [10], [1]),
        _int32_tensor("crop_row_grid_i32", list(range(OUT)), [1, 1, OUT, 1]),
        _int32_tensor("crop_col_grid_i32", list(range(OUT)), [1, 1, 1, OUT]),
        _int64_tensor("crop_hw_start", [0, 0], [2]),
        _int64_tensor("crop_hw_end", [SIZE, SIZE], [2]),
        _int64_tensor("crop_hw_axes", [2, 3], [2]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, GRID_SIZE - OUT, GRID_SIZE - OUT], [8]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("score_base", [1000.0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors9_u8", list(range(1, 10)), [1, 9, 1, 1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
        _f32_tensor("color_conv_w", [float(i) for i in range(10)], [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input"], ["counts10"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("Slice", ["counts10", "count_starts", "count_ends"], ["color_counts9"]),
        helper.make_node("Conv", ["input", "color_conv_w"], ["input_color30_f32"]),
        helper.make_node("Cast", ["input_color30_f32"], ["input_color30_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["input_color30_u8", "crop_hw_start", "crop_hw_end", "crop_hw_axes"], ["input_color_u8"]),
        helper.make_node("Equal", ["input_color_u8", "colors9_u8"], ["color_masks9"]),
        helper.make_node("Cast", ["color_masks9"], ["color_masks9_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Reshape", ["color_counts9", "shape_1x9x1x1"], ["color_counts9_4d"]),
        helper.make_node("ReduceMax", ["color_masks9_u8"], ["row_score9"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["color_masks9_u8"], ["col_score9"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_score9"], ["r_min9"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["row_score9"], ["r_max9"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_score9"], ["c_min9"], axis=3, keepdims=1),
        helper.make_node("ArgMax", ["col_score9"], ["c_max9"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["r_min9"], ["r_min9_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["r_max9"], ["r_max9_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_min9"], ["c_min9_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Cast", ["c_max9"], ["c_max9_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Sub", ["r_max9_i32", "r_min9_i32"], ["height_delta9"]),
        helper.make_node("Sub", ["c_max9_i32", "c_min9_i32"], ["width_delta9"]),
        helper.make_node("Greater", ["height_delta9", "one_i32"], ["height_ok9"]),
        helper.make_node("Greater", ["width_delta9", "one_i32"], ["width_ok9"]),
        helper.make_node("Add", ["height_delta9", "width_delta9"], ["perimeter_half9_i32"]),
        helper.make_node("Add", ["perimeter_half9_i32", "perimeter_half9_i32"], ["perimeter9_i32"]),
        helper.make_node("Cast", ["perimeter9_i32"], ["perimeter9_f32"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Equal", ["color_counts9_4d", "perimeter9_f32"], ["perimeter_ok9"]),
        helper.make_node("And", ["height_ok9", "width_ok9"], ["size_ok9"]),
        helper.make_node("And", ["size_ok9", "perimeter_ok9"], ["valid9"]),
        helper.make_node("Sub", ["score_base", "color_counts9_4d"], ["valid_score_raw9"]),
        helper.make_node("Where", ["valid9", "valid_score_raw9", "zero_f32"], ["frame_scores4d"]),
        helper.make_node("Reshape", ["frame_scores4d", "shape9"], ["frame_scores"]),
        helper.make_node("Reshape", ["r_min9_i32", "shape9"], ["r_min_values"]),
        helper.make_node("Reshape", ["r_max9_i32", "shape9"], ["r_max_values"]),
        helper.make_node("Reshape", ["c_min9_i32", "shape9"], ["c_min_values"]),
        helper.make_node("Reshape", ["c_max9_i32", "shape9"], ["c_max_values"]),
            helper.make_node("TopK", ["frame_scores", "one_i64"], ["top_score", "frame_idx"], axis=0, largest=1, sorted=1),
            helper.make_node("Gather", ["r_min_values", "frame_idx"], ["r_min_1"], axis=0),
            helper.make_node("Gather", ["r_max_values", "frame_idx"], ["r_max_1"], axis=0),
            helper.make_node("Gather", ["c_min_values", "frame_idx"], ["c_min_1"], axis=0),
            helper.make_node("Gather", ["c_max_values", "frame_idx"], ["c_max_1"], axis=0),
            helper.make_node("Add", ["r_min_1", "one_i32"], ["inner_r0_1"]),
            helper.make_node("Add", ["c_min_1", "one_i32"], ["inner_c0_1"]),
            helper.make_node("Reshape", ["inner_r0_1", "shape1111"], ["inner_r0"]),
            helper.make_node("Reshape", ["inner_c0_1", "shape1111"], ["inner_c0"]),
            helper.make_node("Reshape", ["r_max_1", "shape1111"], ["inner_r1"]),
            helper.make_node("Reshape", ["c_max_1", "shape1111"], ["inner_c1"]),
            helper.make_node("Add", ["crop_row_grid_i32", "inner_r0"], ["src_r"]),
            helper.make_node("Add", ["crop_col_grid_i32", "inner_c0"], ["src_c"]),
            helper.make_node("Less", ["src_r", "inner_r1"], ["row_in_crop"]),
            helper.make_node("Less", ["src_c", "inner_c1"], ["col_in_crop"]),
            helper.make_node("And", ["row_in_crop", "col_in_crop"], ["crop_valid"]),
            helper.make_node("Where", ["row_in_crop", "src_r", "zero_i32"], ["safe_r"]),
            helper.make_node("Where", ["col_in_crop", "src_c", "zero_i32"], ["safe_c"]),
            helper.make_node("Reshape", ["safe_r", "shape_vec23"], ["safe_r_vec"]),
            helper.make_node("Reshape", ["safe_c", "shape_vec23"], ["safe_c_vec"]),
            helper.make_node("Gather", ["input_color_u8", "safe_r_vec"], ["gathered_rows"], axis=2),
            helper.make_node("Gather", ["gathered_rows", "safe_c_vec"], ["gathered_color"], axis=3),
            helper.make_node("Where", ["crop_valid", "gathered_color", "invalid_u8"], ["color23"]),
            helper.make_node("Pad", ["color23", "pads_output", "invalid_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]

    graph = helper.make_graph(nodes, "task029_top5_frame_inner_crop_color_grid_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
