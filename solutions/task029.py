from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
OUT = 23
COLORS = list(range(1, 10))


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _add_frame_candidate(nodes: list[onnx.NodeProto], color: int) -> tuple[str, str, str, str, str]:
    prefix = f"c{color}"
    nodes.extend(
        [
            helper.make_node("Equal", ["input_color_u8", f"{prefix}_u8"], [f"{prefix}_mask"]),
            helper.make_node("Cast", [f"{prefix}_mask"], [f"{prefix}_mask_f16"], to=onnx.TensorProto.FLOAT16),
            helper.make_node("ReduceMax", [f"{prefix}_mask_f16"], [f"{prefix}_row_score"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", [f"{prefix}_mask_f16"], [f"{prefix}_col_score"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", [f"{prefix}_row_score"], [f"{prefix}_r_min"], axis=2, keepdims=1),
            helper.make_node("ArgMax", [f"{prefix}_row_score"], [f"{prefix}_r_max"], axis=2, keepdims=1, select_last_index=1),
            helper.make_node("ArgMax", [f"{prefix}_col_score"], [f"{prefix}_c_min"], axis=3, keepdims=1),
            helper.make_node("ArgMax", [f"{prefix}_col_score"], [f"{prefix}_c_max"], axis=3, keepdims=1, select_last_index=1),
            helper.make_node("Cast", [f"{prefix}_r_min"], [f"{prefix}_r_min_i32"], to=onnx.TensorProto.INT32),
            helper.make_node("Cast", [f"{prefix}_r_max"], [f"{prefix}_r_max_i32"], to=onnx.TensorProto.INT32),
            helper.make_node("Cast", [f"{prefix}_c_min"], [f"{prefix}_c_min_i32"], to=onnx.TensorProto.INT32),
            helper.make_node("Cast", [f"{prefix}_c_max"], [f"{prefix}_c_max_i32"], to=onnx.TensorProto.INT32),
            helper.make_node("Sub", [f"{prefix}_r_max_i32", f"{prefix}_r_min_i32"], [f"{prefix}_height_delta"]),
            helper.make_node("Sub", [f"{prefix}_c_max_i32", f"{prefix}_c_min_i32"], [f"{prefix}_width_delta"]),
            helper.make_node("Greater", [f"{prefix}_height_delta", "one_i32"], [f"{prefix}_height_ok"]),
            helper.make_node("Greater", [f"{prefix}_width_delta", "one_i32"], [f"{prefix}_width_ok"]),
            helper.make_node("GreaterOrEqual", ["row_grid_i32", f"{prefix}_r_min_i32"], [f"{prefix}_row_ge_min"]),
            helper.make_node("LessOrEqual", ["row_grid_i32", f"{prefix}_r_max_i32"], [f"{prefix}_row_le_max"]),
            helper.make_node("GreaterOrEqual", ["col_grid_i32", f"{prefix}_c_min_i32"], [f"{prefix}_col_ge_min"]),
            helper.make_node("LessOrEqual", ["col_grid_i32", f"{prefix}_c_max_i32"], [f"{prefix}_col_le_max"]),
            helper.make_node("And", [f"{prefix}_row_ge_min", f"{prefix}_row_le_max"], [f"{prefix}_row_in"]),
            helper.make_node("And", [f"{prefix}_col_ge_min", f"{prefix}_col_le_max"], [f"{prefix}_col_in"]),
            helper.make_node("And", [f"{prefix}_row_in", f"{prefix}_col_in"], [f"{prefix}_bbox"]),
            helper.make_node("Equal", ["row_grid_i32", f"{prefix}_r_min_i32"], [f"{prefix}_row_top"]),
            helper.make_node("Equal", ["row_grid_i32", f"{prefix}_r_max_i32"], [f"{prefix}_row_bottom"]),
            helper.make_node("Equal", ["col_grid_i32", f"{prefix}_c_min_i32"], [f"{prefix}_col_left"]),
            helper.make_node("Equal", ["col_grid_i32", f"{prefix}_c_max_i32"], [f"{prefix}_col_right"]),
            helper.make_node("Or", [f"{prefix}_row_top", f"{prefix}_row_bottom"], [f"{prefix}_border_row"]),
            helper.make_node("Or", [f"{prefix}_col_left", f"{prefix}_col_right"], [f"{prefix}_border_col"]),
            helper.make_node("Or", [f"{prefix}_border_row", f"{prefix}_border_col"], [f"{prefix}_border_line"]),
            helper.make_node("And", [f"{prefix}_bbox", f"{prefix}_border_line"], [f"{prefix}_border"]),
            helper.make_node("Not", [f"{prefix}_mask"], [f"{prefix}_not_mask"]),
            helper.make_node("And", [f"{prefix}_border", f"{prefix}_not_mask"], [f"{prefix}_missing_border"]),
            helper.make_node("Greater", ["row_grid_i32", f"{prefix}_r_min_i32"], [f"{prefix}_row_gt_min"]),
            helper.make_node("Less", ["row_grid_i32", f"{prefix}_r_max_i32"], [f"{prefix}_row_lt_max"]),
            helper.make_node("Greater", ["col_grid_i32", f"{prefix}_c_min_i32"], [f"{prefix}_col_gt_min"]),
            helper.make_node("Less", ["col_grid_i32", f"{prefix}_c_max_i32"], [f"{prefix}_col_lt_max"]),
            helper.make_node("And", [f"{prefix}_row_gt_min", f"{prefix}_row_lt_max"], [f"{prefix}_inner_row"]),
            helper.make_node("And", [f"{prefix}_col_gt_min", f"{prefix}_col_lt_max"], [f"{prefix}_inner_col"]),
            helper.make_node("And", [f"{prefix}_inner_row", f"{prefix}_inner_col"], [f"{prefix}_inner"]),
            helper.make_node("And", [f"{prefix}_mask", f"{prefix}_inner"], [f"{prefix}_interior_hit"]),
            helper.make_node("Cast", [f"{prefix}_missing_border"], [f"{prefix}_missing_f16"], to=onnx.TensorProto.FLOAT16),
            helper.make_node("Cast", [f"{prefix}_interior_hit"], [f"{prefix}_interior_f16"], to=onnx.TensorProto.FLOAT16),
            helper.make_node("ReduceSum", [f"{prefix}_missing_f16"], [f"{prefix}_missing_count"], axes=[0, 1, 2, 3], keepdims=1),
            helper.make_node("ReduceSum", [f"{prefix}_interior_f16"], [f"{prefix}_interior_count"], axes=[0, 1, 2, 3], keepdims=1),
            helper.make_node("Equal", [f"{prefix}_missing_count", "zero_f16"], [f"{prefix}_border_ok"]),
            helper.make_node("Equal", [f"{prefix}_interior_count", "zero_f16"], [f"{prefix}_interior_ok"]),
            helper.make_node("And", [f"{prefix}_height_ok", f"{prefix}_width_ok"], [f"{prefix}_size_ok"]),
            helper.make_node("And", [f"{prefix}_border_ok", f"{prefix}_interior_ok"], [f"{prefix}_shape_ok"]),
            helper.make_node("And", [f"{prefix}_size_ok", f"{prefix}_shape_ok"], [f"{prefix}_valid"]),
            helper.make_node("Cast", [f"{prefix}_valid"], [f"{prefix}_valid_f32_4d"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Reshape", [f"{prefix}_valid_f32_4d", "shape1"], [f"{prefix}_score"]),
            helper.make_node("Reshape", [f"{prefix}_r_min_i32", "shape1"], [f"{prefix}_r_min_1"]),
            helper.make_node("Reshape", [f"{prefix}_r_max_i32", "shape1"], [f"{prefix}_r_max_1"]),
            helper.make_node("Reshape", [f"{prefix}_c_min_i32", "shape1"], [f"{prefix}_c_min_1"]),
            helper.make_node("Reshape", [f"{prefix}_c_max_i32", "shape1"], [f"{prefix}_c_max_1"]),
        ]
    )
    return (
        f"{prefix}_score",
        f"{prefix}_r_min_1",
        f"{prefix}_r_max_1",
        f"{prefix}_c_min_1",
        f"{prefix}_c_max_1",
    )


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int32_tensor("one_i32", [1], [1]),
        _int32_tensor("zero_i32", [0], [1]),
        _int32_tensor("width_i32", [SIZE], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("shape1111", [1, 1, 1, 1], [4]),
        _int64_tensor("shape_index_1x529", [1, 1, OUT * OUT], [3]),
        _int64_tensor("shape_flat_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_1x1x23x23", [1, 1, OUT, OUT], [4]),
        _int32_tensor("row_grid_i32", [r for r in range(SIZE) for _ in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int32_tensor("col_grid_i32", [c for _ in range(SIZE) for c in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int32_tensor("crop_row_grid_i32", [r for r in range(OUT) for _ in range(OUT)], [1, 1, OUT, OUT]),
        _int32_tensor("crop_col_grid_i32", [c for _ in range(OUT) for c in range(OUT)], [1, 1, OUT, OUT]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, SIZE - OUT, SIZE - OUT], [8]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]
    for color in COLORS:
        initializers.append(_u8_tensor(f"c{color}_u8", [color], [1]))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ArgMax", ["input"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Cast", ["input_color_i64"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
    ]
    nodes.append(helper.make_node("Cast", ["zero_f32"], ["zero_f16"], to=onnx.TensorProto.FLOAT16))
    scores: list[str] = []
    r_mins: list[str] = []
    r_maxes: list[str] = []
    c_mins: list[str] = []
    c_maxes: list[str] = []
    for color in COLORS:
        score, r_min, r_max, c_min, c_max = _add_frame_candidate(nodes, color)
        scores.append(score)
        r_mins.append(r_min)
        r_maxes.append(r_max)
        c_mins.append(c_min)
        c_maxes.append(c_max)

    nodes.extend(
        [
            helper.make_node("Concat", scores, ["frame_scores"], axis=0),
            helper.make_node("Concat", r_mins, ["r_min_values"], axis=0),
            helper.make_node("Concat", r_maxes, ["r_max_values"], axis=0),
            helper.make_node("Concat", c_mins, ["c_min_values"], axis=0),
            helper.make_node("Concat", c_maxes, ["c_max_values"], axis=0),
            helper.make_node("TopK", ["frame_scores", "shape1"], ["top_score", "frame_idx"], axis=0, largest=1, sorted=1),
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
            helper.make_node("Where", ["crop_valid", "src_r", "zero_i32"], ["safe_r"]),
            helper.make_node("Where", ["crop_valid", "src_c", "zero_i32"], ["safe_c"]),
            helper.make_node("Mul", ["safe_r", "width_i32"], ["safe_r_offset"]),
            helper.make_node("Add", ["safe_r_offset", "safe_c"], ["safe_spatial"]),
            helper.make_node("Reshape", ["safe_spatial", "shape_index_1x529"], ["safe_spatial_flat_i32"]),
            helper.make_node("Cast", ["safe_spatial_flat_i32"], ["safe_spatial_flat"], to=onnx.TensorProto.INT64),
            helper.make_node("Reshape", ["input_color_u8", "shape_flat_1x900"], ["color_flat_u8"]),
            helper.make_node("GatherElements", ["color_flat_u8", "safe_spatial_flat"], ["gathered_flat"], axis=2),
            helper.make_node("Reshape", ["gathered_flat", "shape_1x1x23x23"], ["gathered_color"]),
            helper.make_node("Where", ["crop_valid", "gathered_color", "invalid_u8"], ["color23"]),
            helper.make_node("Pad", ["color23", "pads_output", "invalid_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task029_frame_inner_crop_color_grid_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
