from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
PROPAGATION_STEPS = 9
SQUARE = [(0, 0), (0, 1), (1, 0), (1, 1)]
H_BAR = [(0, 0), (0, 1), (0, 2)]
V_BAR = [(0, 0), (1, 0), (2, 0)]
INTERNAL_TYPE = onnx.TensorProto.FLOAT16


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, INTERNAL_TYPE, dims, np.asarray(values, dtype=np.float16).ravel())


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _shift(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    source: str,
    output: str,
    dr: int,
    dc: int,
) -> str:
    row_start = max(0, -dr)
    row_end = SIZE - max(0, dr)
    col_start = max(0, -dc)
    col_end = SIZE - max(0, dc)
    pad_top = max(0, dr)
    pad_bottom = max(0, -dr)
    pad_left = max(0, dc)
    pad_right = max(0, -dc)
    initializers.extend(
        [
            _int64_tensor(f"{output}_starts", [0, 0, row_start, col_start], [4]),
            _int64_tensor(f"{output}_ends", [1, 1, row_end, col_end], [4]),
            _int64_tensor(f"{output}_pads", [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right], [8]),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Slice", [source, f"{output}_starts", f"{output}_ends"], [f"{output}_crop"]),
            helper.make_node("Pad", [f"{output}_crop", f"{output}_pads"], [output], mode="constant"),
        ]
    )
    return output


def _sum_many(nodes: list[onnx.NodeProto], sources: list[str], output: str) -> str:
    if len(sources) == 1:
        nodes.append(helper.make_node("Identity", [sources[0]], [output]))
        return output
    current = sources[0]
    for index, source in enumerate(sources[1:], start=1):
        target = output if index == len(sources) - 1 else f"{output}_partial_{index}"
        nodes.append(helper.make_node("Add", [current, source], [target]))
        current = target
    return output


def _mul_many(nodes: list[onnx.NodeProto], sources: list[str], output: str) -> str:
    if len(sources) == 1:
        nodes.append(helper.make_node("Identity", [sources[0]], [output]))
        return output
    current = sources[0]
    for index, source in enumerate(sources[1:], start=1):
        target = output if index == len(sources) - 1 else f"{output}_partial_{index}"
        nodes.append(helper.make_node("Mul", [current, source], [target]))
        current = target
    return output


def _active_tiles(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    remaining: str,
    offsets: list[tuple[int, int]],
    prefix: str,
) -> str:
    aligned = [
        _shift(nodes, initializers, remaining, f"{prefix}_cell_{index}", -dr, -dc)
        for index, (dr, dc) in enumerate(offsets)
    ]
    return _mul_many(nodes, aligned, f"{prefix}_active")


def _cover_from_tiles(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    active: str,
    offsets: list[tuple[int, int]],
    prefix: str,
) -> str:
    cover_parts = [
        _shift(nodes, initializers, active, f"{prefix}_cover_{index}", dr, dc)
        for index, (dr, dc) in enumerate(offsets)
    ]
    cover_sum = _sum_many(nodes, cover_parts, f"{prefix}_cover_sum")
    nodes.extend(
        [
            helper.make_node("Greater", [cover_sum, "zero_f32"], [f"{prefix}_cover_bool"]),
            helper.make_node("Cast", [f"{prefix}_cover_bool"], [f"{prefix}_cover_f32"], to=INTERNAL_TYPE),
        ]
    )
    return f"{prefix}_cover_f32"


def _selected_tiles(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    active: str,
    forced: str,
    offsets: list[tuple[int, int]],
    prefix: str,
) -> str:
    forced_at_anchor = [
        _shift(nodes, initializers, forced, f"{prefix}_forced_{index}", -dr, -dc)
        for index, (dr, dc) in enumerate(offsets)
    ]
    forced_score = _sum_many(nodes, forced_at_anchor, f"{prefix}_forced_score")
    nodes.extend(
        [
            helper.make_node("Greater", [forced_score, "zero_f32"], [f"{prefix}_forced_bool"]),
            helper.make_node("Cast", [f"{prefix}_forced_bool"], [f"{prefix}_forced_f32"], to=INTERNAL_TYPE),
            helper.make_node("Mul", [active, f"{prefix}_forced_f32"], [f"{prefix}_selected"]),
        ]
    )
    return f"{prefix}_selected"


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _f16_tensor("zero_f32", [0.0], [1]),
        _f16_tensor("one_f32", [1.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
        _int64_tensor("black_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("black_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("gray_starts", [0, 5, 0, 0], [4]),
        _int64_tensor("gray_ends", [1, 6, SIZE, SIZE], [4]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "black_starts", "black_ends"], ["input0_f32"]),
        helper.make_node("Slice", ["input", "gray_starts", "gray_ends"], ["gray_f32"]),
        helper.make_node("Cast", ["input0_f32"], ["input0"], to=INTERNAL_TYPE),
        helper.make_node("Cast", ["gray_f32"], ["gray"], to=INTERNAL_TYPE),
        helper.make_node("Greater", ["input0", "zero_f32"], ["input0_bool"]),
        helper.make_node("Mul", ["gray", "zero_f32"], ["zero_grid"]),
        helper.make_node("Identity", ["zero_grid"], ["square_acc_0"]),
        helper.make_node("Identity", ["zero_grid"], ["bar_acc_0"]),
        helper.make_node("Identity", ["zero_grid"], ["covered_0"]),
        helper.make_node("Identity", ["gray"], ["remaining_0"]),
    ]

    square_acc = "square_acc_0"
    bar_acc = "bar_acc_0"
    covered = "covered_0"
    remaining = "remaining_0"
    for step in range(PROPAGATION_STEPS):
        s_active = _active_tiles(nodes, initializers, remaining, SQUARE, f"step{step}_s")
        h_active = _active_tiles(nodes, initializers, remaining, H_BAR, f"step{step}_h")
        v_active = _active_tiles(nodes, initializers, remaining, V_BAR, f"step{step}_v")

        cover_terms = []
        for name, active, offsets in (("s", s_active, SQUARE), ("h", h_active, H_BAR), ("v", v_active, V_BAR)):
            for index, (dr, dc) in enumerate(offsets):
                cover_terms.append(_shift(nodes, initializers, active, f"step{step}_{name}_count_{index}", dr, dc))
        count = _sum_many(nodes, cover_terms, f"step{step}_candidate_count")

        nodes.extend(
            [
                helper.make_node("Equal", [count, "one_f32"], [f"step{step}_count_one_bool"]),
                helper.make_node("Cast", [f"step{step}_count_one_bool"], [f"step{step}_count_one_f32"], to=INTERNAL_TYPE),
                helper.make_node("Mul", [remaining, f"step{step}_count_one_f32"], [f"step{step}_forced"]),
            ]
        )
        forced = f"step{step}_forced"

        s_selected = _selected_tiles(nodes, initializers, s_active, forced, SQUARE, f"step{step}_s")
        h_selected = _selected_tiles(nodes, initializers, h_active, forced, H_BAR, f"step{step}_h")
        v_selected = _selected_tiles(nodes, initializers, v_active, forced, V_BAR, f"step{step}_v")

        s_cover = _cover_from_tiles(nodes, initializers, s_selected, SQUARE, f"step{step}_s_selected")
        h_cover = _cover_from_tiles(nodes, initializers, h_selected, H_BAR, f"step{step}_h_selected")
        v_cover = _cover_from_tiles(nodes, initializers, v_selected, V_BAR, f"step{step}_v_selected")
        bar_cover = _sum_many(nodes, [h_cover, v_cover], f"step{step}_bar_cover_sum")
        nodes.extend(
            [
                helper.make_node("Greater", [bar_cover, "zero_f32"], [f"step{step}_bar_cover_bool"]),
                helper.make_node("Cast", [f"step{step}_bar_cover_bool"], [f"step{step}_bar_cover"], to=INTERNAL_TYPE),
                helper.make_node("Add", [square_acc, s_cover], [f"step{step}_square_acc_sum"]),
                helper.make_node("Greater", [f"step{step}_square_acc_sum", "zero_f32"], [f"step{step}_square_acc_bool"]),
                helper.make_node("Cast", [f"step{step}_square_acc_bool"], [f"step{step}_square_acc"], to=INTERNAL_TYPE),
                helper.make_node("Add", [bar_acc, f"step{step}_bar_cover"], [f"step{step}_bar_acc_sum"]),
                helper.make_node("Greater", [f"step{step}_bar_acc_sum", "zero_f32"], [f"step{step}_bar_acc_bool"]),
                helper.make_node("Cast", [f"step{step}_bar_acc_bool"], [f"step{step}_bar_acc"], to=INTERNAL_TYPE),
            ]
        )
        new_cover_sum = _sum_many(nodes, [s_cover, h_cover, v_cover], f"step{step}_new_cover_sum")
        nodes.extend(
            [
                helper.make_node("Add", [covered, new_cover_sum], [f"step{step}_covered_sum"]),
                helper.make_node("Greater", [f"step{step}_covered_sum", "zero_f32"], [f"step{step}_covered_bool"]),
                helper.make_node("Cast", [f"step{step}_covered_bool"], [f"step{step}_covered"], to=INTERNAL_TYPE),
                helper.make_node("Sub", ["one_f32", f"step{step}_covered"], [f"step{step}_not_covered"]),
                helper.make_node("Mul", ["gray", f"step{step}_not_covered"], [f"step{step}_remaining"]),
            ]
        )
        square_acc = f"step{step}_square_acc"
        bar_acc = f"step{step}_bar_acc"
        covered = f"step{step}_covered"
        remaining = f"step{step}_remaining"

    nodes.extend(
        [
            helper.make_node("Where", ["input0_bool", "zero_u8", "invalid_u8"], ["color_base"]),
            helper.make_node("Greater", [bar_acc, "zero_f32"], ["bar_bool"]),
            helper.make_node("Where", ["bar_bool", "two_u8", "color_base"], ["color_bar"]),
            helper.make_node("Greater", [square_acc, "zero_f32"], ["square_bool"]),
            helper.make_node("Where", ["square_bool", "eight_u8", "color_bar"], ["color30"]),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task023_exact_cover_propagation_f16_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
