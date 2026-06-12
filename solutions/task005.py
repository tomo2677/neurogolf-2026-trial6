from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
INTERNAL_TYPE = onnx.TensorProto.FLOAT16
SELECTED_COLORS = 4


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _uint8_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.UINT8, shape, values)


def _float_tensor(name: str, values: np.ndarray) -> onnx.TensorProto:
    return helper.make_tensor(name, INTERNAL_TYPE, list(values.shape), values.astype(np.float16).ravel())


def _axis_index(delta: int, step: int) -> int:
    if delta > 0:
        return 3 - step
    if delta < 0:
        return step
    return 0


def _repeat_kernel_and_pads(dy: int, dx: int) -> tuple[np.ndarray, list[int], list[int]]:
    height = 4 if dy else 1
    width = 4 if dx else 1
    top = 12 if dy > 0 else 0
    bottom = 12 if dy < 0 else 0
    left = 12 if dx > 0 else 0
    right = 12 if dx < 0 else 0
    kernel = np.zeros((1, 1, height, width), dtype=np.float16)
    for step in range(1, 4):
        row = _axis_index(dy, step)
        col = _axis_index(dx, step)
        kernel[0, 0, row, col] = 1
    return kernel, [top, left, bottom, right], [4 if dy else 1, 4 if dx else 1]


def _shift_2d(nodes: list[onnx.NodeProto], source: str, dy: int, dx: int, output: str) -> None:
    if dy == 0 and dx == 0:
        nodes.append(helper.make_node("Identity", [source], [output]))
        return
    nodes.append(helper.make_node("Pad", [source, f"pads_shift_{dy}_{dx}"], [output], mode="constant"))


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        helper.make_tensor("zero11", INTERNAL_TYPE, [1, 1], [0.0]),
        helper.make_tensor("one11", INTERNAL_TYPE, [1, 1], [1.0]),
        helper.make_tensor("compact_eps", INTERNAL_TYPE, [1], [0.1]),
        helper.make_tensor("tie_eps", INTERNAL_TYPE, [1], [0.01]),
        _int32_tensor("score_start", [1]),
        _int32_tensor("score_end", [10]),
        _int64_tensor("k4", [SELECTED_COLORS]),
        _int64_tensor("color21_shape", [1, 1, 21, 21]),
        _int32_tensor("channel_ids_i32", list(range(1, 10)), [9]),
        _int32_tensor("slice_zero", [0], [1]),
        _int32_tensor("slice_one", [1], [1]),
        _int32_tensor("slice_21", [21], [1]),
        _int32_tensor("slice_axes3", [1, 2, 3], [3]),
        _uint8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
        _uint8_tensor("pad_sentinel_u8", [255], [1]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 9, 9]),
    ]
    repeat_pads: dict[tuple[int, int], list[int]] = {}
    repeat_shapes: dict[tuple[int, int], list[int]] = {}
    repeat_dilations: dict[tuple[int, int], list[int]] = {}
    for dy, dx in DIRECTIONS:
        kernel, pads, dilations = _repeat_kernel_and_pads(dy, dx)
        initializers.append(_float_tensor(f"repeat_kernel_{dy}_{dx}", kernel))
        repeat_pads[(dy, dx)] = pads
        repeat_shapes[(dy, dx)] = list(kernel.shape[2:])
        repeat_dilations[(dy, dx)] = dilations
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["present_scores10"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("Slice", ["present_scores10", "score_start", "score_end"], ["present_scores"]),
        helper.make_node("TopK", ["present_scores", "k4"], ["top_scores", "top_indices"], axis=0, largest=1, sorted=1),
        helper.make_node("Split", ["top_indices"], [f"top_idx_{slot}" for slot in range(SELECTED_COLORS)], axis=0, split=[1] * SELECTED_COLORS),
    ]
    selected_stamp_scores: list[str] = []
    for slot in range(SELECTED_COLORS):
        nodes.extend(
            [
                helper.make_node("Gather", ["channel_ids_i32", f"top_idx_{slot}"], [f"channel_id_{slot}"], axis=0),
                helper.make_node("Add", [f"channel_id_{slot}", "slice_one"], [f"channel_end_{slot}"]),
                helper.make_node("Concat", [f"channel_id_{slot}", "slice_zero", "slice_zero"], [f"selected_start_{slot}"], axis=0),
                helper.make_node("Concat", [f"channel_end_{slot}", "slice_21", "slice_21"], [f"selected_end_{slot}"], axis=0),
                helper.make_node("Slice", ["input", f"selected_start_{slot}", f"selected_end_{slot}", "slice_axes3"], [f"selected_{slot}_f32"]),
                helper.make_node("Cast", [f"selected_{slot}_f32"], [f"selected_{slot}"], to=INTERNAL_TYPE),
                helper.make_node("Cast", [f"channel_id_{slot}"], [f"color_id_{slot}"], to=INTERNAL_TYPE),
                helper.make_node("ReduceSum", [f"selected_{slot}"], [f"stamp_count_{slot}"], axes=[2, 3], keepdims=0),
                helper.make_node("ReduceSum", [f"selected_{slot}"], [f"row_counts_{slot}"], axes=[3], keepdims=0),
                helper.make_node("ReduceMax", [f"row_counts_{slot}"], [f"max_row_count_{slot}"], axes=[2], keepdims=0),
                helper.make_node("Mul", [f"max_row_count_{slot}", "compact_eps"], [f"compact_bonus_{slot}"]),
                helper.make_node("Mul", [f"color_id_{slot}", "tie_eps"], [f"tie_bonus_{slot}"]),
                helper.make_node("Add", [f"stamp_count_{slot}", f"compact_bonus_{slot}"], [f"stamp_base_{slot}"]),
                helper.make_node("Add", [f"stamp_base_{slot}", f"tie_bonus_{slot}"], [f"stamp_score_{slot}"]),
            ]
        )
        selected_stamp_scores.append(f"stamp_score_{slot}")
    nodes.extend(
        [
            helper.make_node("Concat", selected_stamp_scores, ["selected_stamp_scores"], axis=1),
            helper.make_node("ArgMax", ["selected_stamp_scores"], ["seed_slot_idx"], axis=1, keepdims=0),
            helper.make_node("Gather", ["top_indices", "seed_slot_idx"], ["seed_top_idx"], axis=0),
            helper.make_node("Gather", ["channel_ids_i32", "seed_top_idx"], ["seed_channel_id"], axis=0),
            helper.make_node("Add", ["seed_channel_id", "slice_one"], ["seed_channel_end"]),
            helper.make_node("Concat", ["seed_channel_id", "slice_zero", "slice_zero"], ["seed_start"], axis=0),
            helper.make_node("Concat", ["seed_channel_end", "slice_21", "slice_21"], ["seed_end"], axis=0),
            helper.make_node("Slice", ["input", "seed_start", "seed_end", "slice_axes3"], ["seed_mask_f32"]),
            helper.make_node("Cast", ["seed_mask_f32"], ["seed_mask"], to=INTERNAL_TYPE),
            helper.make_node("Cast", ["seed_channel_id"], ["seed_color_id"], to=INTERNAL_TYPE),
            helper.make_node("Flatten", ["seed_mask"], ["seed_flat"], axis=1),
            helper.make_node("Mul", ["seed_flat", "seed_color_id"], ["colored_seed_out_flat"]),
        ]
    )

    seed_mask = "seed_mask"
    direction_repeats: dict[tuple[int, int], str] = {}
    for dy, dx in DIRECTIONS:
        raw_repeat = f"repeat_{dy}_{dx}_raw"
        nodes.append(
            helper.make_node(
                "Conv",
                [seed_mask, f"repeat_kernel_{dy}_{dx}"],
                [raw_repeat],
                kernel_shape=repeat_shapes[(dy, dx)],
                pads=repeat_pads[(dy, dx)],
                dilations=repeat_dilations[(dy, dx)],
            )
        )
        direction_repeats[(dy, dx)] = raw_repeat

    repeat_flats: list[str] = []
    for dy, dx in DIRECTIONS:
        repeat_flats.append(f"repeat_flat_{dy}_{dx}")
        nodes.append(helper.make_node("Flatten", [direction_repeats[(dy, dx)]], [f"repeat_flat_{dy}_{dx}"], axis=1))
    nodes.append(helper.make_node("Concat", repeat_flats, ["repeat_matrix"], axis=0))

    colored_outputs: list[str] = ["colored_seed_out_flat"]
    for slot in range(SELECTED_COLORS):
        nodes.append(helper.make_node("Flatten", [f"selected_{slot}"], [f"selected_{slot}_flat"], axis=1))
        selectors: list[str] = []
        for dy, dx in DIRECTIONS:
            nodes.extend(
                [
                    helper.make_node("Gemm", [f"selected_{slot}_flat", f"repeat_flat_{dy}_{dx}", "zero11"], [f"frag_count_{slot}_{dy}_{dx}"], transB=1),
                    helper.make_node("Min", [f"frag_count_{slot}_{dy}_{dx}", "one11"], [f"frag_selector_{slot}_{dy}_{dx}"]),
                ]
            )
            selectors.append(f"frag_selector_{slot}_{dy}_{dx}")
        nodes.extend(
            [
                helper.make_node("Concat", selectors, [f"direction_selector_{slot}"], axis=1),
                helper.make_node("Mul", [f"direction_selector_{slot}", f"color_id_{slot}"], [f"colored_direction_selector_{slot}"]),
                helper.make_node("MatMul", [f"colored_direction_selector_{slot}", "repeat_matrix"], [f"colored_{slot}_out_flat"]),
            ]
        )
        colored_outputs.append(f"colored_{slot}_out_flat")

    nodes.extend(
        [
            helper.make_node("Max", colored_outputs, ["color21_flat"]),
            helper.make_node("Cast", ["color21_flat"], ["color21_u8_flat"], to=onnx.TensorProto.UINT8),
            helper.make_node("Reshape", ["color21_u8_flat", "color21_shape"], ["color21_u8"]),
            helper.make_node("Pad", ["color21_u8", "pads_output", "pad_sentinel_u8"], ["color30_u8"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30_u8"], ["output"]),
        ]
    )

    value_infos = []
    for slot in range(SELECTED_COLORS):
        value_infos.extend(
            [
                helper.make_tensor_value_info(f"selected_{slot}_f32", onnx.TensorProto.FLOAT, [1, 1, 21, 21]),
                helper.make_tensor_value_info(f"selected_{slot}", INTERNAL_TYPE, [1, 1, 21, 21]),
            ]
        )
    value_infos.extend(
        [
            helper.make_tensor_value_info("seed_mask_f32", onnx.TensorProto.FLOAT, [1, 1, 21, 21]),
            helper.make_tensor_value_info("seed_mask", INTERNAL_TYPE, [1, 1, 21, 21]),
        ]
    )

    graph = helper.make_graph(nodes, "task005_seed_reducesum_score_graph", [x], [y], initializers, value_info=value_infos)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
