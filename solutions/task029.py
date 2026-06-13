from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 25
GRID_SIZE = 30
OUT = 23
TOP_COLORS = 5


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


def _add_frame_candidate(nodes: list[onnx.NodeProto], prefix: str, color_u8: str, color_count_f32: str) -> tuple[str, str, str, str, str]:
    nodes.extend(
        [
            helper.make_node("Equal", ["input_color_u8", color_u8], [f"{prefix}_mask"]),
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
            helper.make_node("Add", [f"{prefix}_height_delta", f"{prefix}_width_delta"], [f"{prefix}_perimeter_half_i32"]),
            helper.make_node("Add", [f"{prefix}_perimeter_half_i32", f"{prefix}_perimeter_half_i32"], [f"{prefix}_perimeter_i32"]),
            helper.make_node("Cast", [f"{prefix}_perimeter_i32"], [f"{prefix}_perimeter_f32"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Equal", [color_count_f32, f"{prefix}_perimeter_f32"], [f"{prefix}_perimeter_ok"]),
            helper.make_node("And", [f"{prefix}_height_ok", f"{prefix}_width_ok"], [f"{prefix}_size_ok"]),
            helper.make_node("And", [f"{prefix}_size_ok", f"{prefix}_perimeter_ok"], [f"{prefix}_valid"]),
            helper.make_node("Sub", ["score_base", color_count_f32], [f"{prefix}_valid_score_raw"]),
            helper.make_node("Where", [f"{prefix}_valid", f"{prefix}_valid_score_raw", "zero_f32"], [f"{prefix}_score_4d"]),
            helper.make_node("Reshape", [f"{prefix}_score_4d", "one_i64"], [f"{prefix}_score"]),
            helper.make_node("Reshape", [f"{prefix}_r_min_i32", "one_i64"], [f"{prefix}_r_min_1"]),
            helper.make_node("Reshape", [f"{prefix}_r_max_i32", "one_i64"], [f"{prefix}_r_max_1"]),
            helper.make_node("Reshape", [f"{prefix}_c_min_i32", "one_i64"], [f"{prefix}_c_min_1"]),
            helper.make_node("Reshape", [f"{prefix}_c_max_i32", "one_i64"], [f"{prefix}_c_max_1"]),
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
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("k_top_colors", [TOP_COLORS], [1]),
        _int64_tensor("shape1111", [1, 1, 1, 1], [4]),
        _int64_tensor("shape_vec23", [OUT], [1]),
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
        helper.make_node("Conv", ["input", "color_conv_w"], ["input_color30_f32"]),
        helper.make_node("Cast", ["input_color30_f32"], ["input_color30_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["input_color30_u8", "crop_hw_start", "crop_hw_end", "crop_hw_axes"], ["input_color_u8"]),
        helper.make_node("Equal", ["input_color_u8", "colors9_u8"], ["color_masks9"]),
        helper.make_node("Cast", ["color_masks9"], ["color_masks9_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("ReduceSum", ["color_masks9_f16"], ["color_counts9_f16"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("Cast", ["color_counts9_f16"], ["color_counts9"], to=onnx.TensorProto.FLOAT),
        helper.make_node("TopK", ["color_counts9", "k_top_colors"], ["top_color_counts", "top_color_idx0"], axis=0, largest=1, sorted=0),
        helper.make_node("Add", ["top_color_idx0", "one_i64"], ["top_color_i64"]),
        helper.make_node(
            "Split",
            ["top_color_counts"],
            [f"top_color_count_{slot}" for slot in range(TOP_COLORS)],
            axis=0,
            split=[1] * TOP_COLORS,
        ),
        helper.make_node(
            "Split",
            ["top_color_i64"],
            [f"top_color_i64_{slot}" for slot in range(TOP_COLORS)],
            axis=0,
            split=[1] * TOP_COLORS,
        ),
    ]
    scores: list[str] = []
    r_mins: list[str] = []
    r_maxes: list[str] = []
    c_mins: list[str] = []
    c_maxes: list[str] = []
    for slot in range(TOP_COLORS):
        nodes.append(
            helper.make_node(
                "Cast",
                [f"top_color_i64_{slot}"],
                [f"top_color_u8_{slot}"],
                to=onnx.TensorProto.UINT8,
            )
        )
        score, r_min, r_max, c_min, c_max = _add_frame_candidate(
            nodes, f"s{slot}", f"top_color_u8_{slot}", f"top_color_count_{slot}"
        )
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
    )

    graph = helper.make_graph(nodes, "task029_top5_frame_inner_crop_color_grid_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
