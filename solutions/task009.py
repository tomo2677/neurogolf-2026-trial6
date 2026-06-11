from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


PIX = 29
FULL = 30
CELLS = 10
OBJECT_COLORS = 4
INTERNAL_TYPE = onnx.TensorProto.UINT8
NP_DTYPE = np.uint8


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT32, dims, values)


def _uint8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _gt_one(nodes: list[onnx.NodeProto], value: str, output: str) -> None:
    nodes.append(helper.make_node("Greater", [value, "one1111"], [output]))


def _gt_zero(nodes: list[onnx.NodeProto], value: str, output: str) -> None:
    nodes.append(helper.make_node("Greater", [value, "zero_score_u8"], [output]))


def _line_fill(nodes: list[onnx.NodeProto], occ: str, prefix: str, axis: int, indices: str) -> str:
    nodes.extend(
        [
            helper.make_node("ReduceMax", [occ], [f"{prefix}_any_u8"], axes=[axis], keepdims=1),
            helper.make_node("Greater", [f"{prefix}_any_u8", "zero_f32"], [f"{prefix}_any"]),
            helper.make_node("ArgMax", [occ], [f"{prefix}_left_idx"], axis=axis, keepdims=1, select_last_index=0),
            helper.make_node("ArgMax", [occ], [f"{prefix}_right_idx"], axis=axis, keepdims=1, select_last_index=1),
            helper.make_node("LessOrEqual", [f"{prefix}_left_idx", indices], [f"{prefix}_after_left"]),
            helper.make_node("LessOrEqual", [indices, f"{prefix}_right_idx"], [f"{prefix}_before_right"]),
            helper.make_node("And", [f"{prefix}_after_left", f"{prefix}_before_right"], [f"{prefix}_between"]),
            helper.make_node("And", [f"{prefix}_any", f"{prefix}_between"], [f"{prefix}_fill"]),
        ]
    )
    return f"{prefix}_fill"


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    body_mask = np.zeros((1, 1, FULL, FULL), dtype=np.bool_)
    for row in range(FULL):
        for col in range(FULL):
            if row % 3 != 2 and col % 3 != 2:
                body_mask[0, 0, row, col] = True

    initializers = [
        _int32_tensor("grid_pixel_start", [0, 2, 2]),
        _int32_tensor("grid_pixel_end", [10, 3, 3]),
        _int32_tensor("present_start", [1]),
        _int32_tensor("present_end", [10]),
        _int64_tensor("k4", [OBJECT_COLORS]),
        _int64_tensor("one_i64", [1]),
        _int64_tensor("cell_indices_i64", list(range(CELLS)), [1, 1, 1, CELLS]),
        _int64_tensor("cell_row_indices_i64", list(range(CELLS)), [1, 1, CELLS, 1]),
        _int32_tensor("input_channel_ids", list(range(1, 10)), [9]),
        _int32_tensor("slice_axes3", [1, 2, 3], [3]),
        _int32_tensor("slice_zero", [0], [1]),
        _int32_tensor("selected_delta", [1, FULL, FULL], [3]),
        _int32_tensor("cell_slice_steps", [1, 3, 3], [3]),
        _int64_tensor("resize_sizes", [1, 1, FULL, FULL], [4]),
        helper.make_tensor("zero_f32", onnx.TensorProto.FLOAT, [1, 1, 1, 1], np.array([0.0], dtype=np.float32)),
        _uint8_tensor("channel_ids_u8", list(range(10)), [1, 10, 1, 1]),
        _uint8_tensor("zero_score_u8", [0], [1]),
        _uint8_tensor("outside_u8", [255], [1]),
        helper.make_tensor("body_mask", onnx.TensorProto.BOOL, [1, 1, FULL, FULL], body_mask.ravel()),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["row_present"], axes=[1, 3], keepdims=1),
        helper.make_node("Greater", ["row_present", "zero_f32"], ["row_valid"]),
        helper.make_node("Transpose", ["row_valid"], ["col_valid"], perm=[0, 1, 3, 2]),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_area"]),
        helper.make_node("Slice", ["input", "grid_pixel_start", "grid_pixel_end", "slice_axes3"], ["grid_pixel"]),
        helper.make_node("ArgMax", ["grid_pixel"], ["grid_color_i64"], axis=1, keepdims=1),
        helper.make_node("Cast", ["grid_color_i64"], ["grid_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["grid_color_i64"], ["grid_color_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Reshape", ["grid_color_i32", "one_i64"], ["grid_shifted_i32"]),
        helper.make_node("Equal", ["input_channel_ids", "grid_shifted_i32"], ["grid_score_mask"]),
        helper.make_node("ReduceMax", ["input"], ["present_scores10"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("Cast", ["present_scores10"], ["present_scores10_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["present_scores10_u8", "present_start", "present_end"], ["present_scores"]),
        helper.make_node("Where", ["grid_score_mask", "zero_score_u8", "present_scores"], ["object_scores"]),
        helper.make_node("TopK", ["object_scores", "k4"], ["top_scores", "top_indices"], axis=0, largest=1, sorted=1),
        helper.make_node(
            "Split",
            ["top_indices"],
            [f"top_idx_{slot}" for slot in range(OBJECT_COLORS)],
            axis=0,
        ),
    ]

    cell_color_candidates: list[str] = []
    for slot in range(OBJECT_COLORS):
        nodes.extend(
            [
                helper.make_node("Gather", ["input_channel_ids", f"top_idx_{slot}"], [f"input_channel_{slot}"], axis=0),
                helper.make_node("Concat", [f"input_channel_{slot}", "slice_zero", "slice_zero"], [f"selected_start_{slot}"], axis=0),
                helper.make_node("Add", [f"selected_start_{slot}", "selected_delta"], [f"selected_end_{slot}"]),
                helper.make_node(
                    "Slice",
                    ["input", f"selected_start_{slot}", f"selected_end_{slot}", "slice_axes3", "cell_slice_steps"],
                    [f"occ_{slot}_f32"],
                ),
            ]
        )
        hfill = _line_fill(nodes, f"occ_{slot}_f32", f"h_{slot}", 3, "cell_indices_i64")
        vfill = _line_fill(nodes, f"occ_{slot}_f32", f"v_{slot}", 2, "cell_row_indices_i64")
        cell_color_candidates.append(f"cell_color_candidate_{slot}")
        nodes.extend(
            [
                helper.make_node("Or", [hfill, vfill], [f"cell_bool_{slot}"]),
                helper.make_node("Cast", [f"input_channel_{slot}"], [f"input_channel_u8_{slot}"], to=onnx.TensorProto.UINT8),
                helper.make_node("Where", [f"cell_bool_{slot}", f"input_channel_u8_{slot}", "zero_score_u8"], [f"cell_color_candidate_{slot}"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Max", cell_color_candidates, ["cell_or_black_u8"]),
            helper.make_node(
                "Resize",
                ["cell_or_black_u8", "", "", "resize_sizes"],
                ["cell_or_black30_u8"],
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            ),
            helper.make_node("Where", ["body_mask", "cell_or_black30_u8", "grid_color_u8"], ["color30_raw_u8"]),
            helper.make_node("Where", ["valid_area", "color30_raw_u8", "outside_u8"], ["color_grid_u8"]),
            helper.make_node("Equal", ["channel_ids_u8", "color_grid_u8"], ["output"]),
        ]
    )

    value_infos = [
        helper.make_tensor_value_info("row_present", onnx.TensorProto.FLOAT, [1, 1, FULL, 1]),
        helper.make_tensor_value_info("row_valid", onnx.TensorProto.BOOL, [1, 1, FULL, 1]),
        helper.make_tensor_value_info("col_valid", onnx.TensorProto.BOOL, [1, 1, 1, FULL]),
        helper.make_tensor_value_info("valid_area", onnx.TensorProto.BOOL, [1, 1, FULL, FULL]),
        helper.make_tensor_value_info("grid_pixel", onnx.TensorProto.FLOAT, [1, 10, 1, 1]),
        helper.make_tensor_value_info("grid_color_i64", onnx.TensorProto.INT64, [1, 1, 1, 1]),
        helper.make_tensor_value_info("grid_color_u8", onnx.TensorProto.UINT8, [1, 1, 1, 1]),
        helper.make_tensor_value_info("grid_color_i32", onnx.TensorProto.INT32, [1, 1, 1, 1]),
        helper.make_tensor_value_info("grid_shifted_i32", onnx.TensorProto.INT32, [1]),
        helper.make_tensor_value_info("grid_score_mask", onnx.TensorProto.BOOL, [9]),
        helper.make_tensor_value_info("object_scores", onnx.TensorProto.UINT8, [9]),
        helper.make_tensor_value_info("cell_or_black_u8", onnx.TensorProto.UINT8, [1, 1, CELLS, CELLS]),
        helper.make_tensor_value_info("cell_or_black30_u8", onnx.TensorProto.UINT8, [1, 1, FULL, FULL]),
    ]
    for slot in range(OBJECT_COLORS):
        value_infos.extend(
            [
                helper.make_tensor_value_info(f"occ_{slot}_f32", onnx.TensorProto.FLOAT, [1, 1, CELLS, CELLS]),
            ]
        )

    graph = helper.make_graph(nodes, "task009_u8_cell_select", [x], [y], initializers, value_info=value_infos)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
