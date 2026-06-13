from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


HEIGHT = 9
WIDTH = 11
PROPAGATION_STEPS = 3
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


def _initializer_key(tensor: onnx.TensorProto) -> tuple:
    return (
        tensor.data_type,
        tuple(tensor.dims),
        bytes(tensor.raw_data),
        tuple(tensor.int32_data),
        tuple(tensor.int64_data),
        tuple(tensor.float_data),
        tuple(tensor.double_data),
        tuple(tensor.string_data),
    )


def _dedupe_initializers(graph: onnx.GraphProto) -> None:
    canonical: dict[tuple, str] = {}
    rename: dict[str, str] = {}
    unique: list[onnx.TensorProto] = []
    for initializer in graph.initializer:
        key = _initializer_key(initializer)
        existing = canonical.get(key)
        if existing is None:
            canonical[key] = initializer.name
            unique.append(initializer)
        else:
            rename[initializer.name] = existing
    if not rename:
        return
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in rename:
                node.input[index] = rename[name]
    del graph.initializer[:]
    graph.initializer.extend(unique)


def _shift(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    source: str,
    output: str,
    dr: int,
    dc: int,
) -> str:
    initializers.append(_int64_tensor(f"{output}_pads_hw", [dr, dc, -dr, -dc], [4]))
    nodes.append(helper.make_node("Pad", [source, f"{output}_pads_hw", "", "axes_hw"], [output], mode="constant"))
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


def _and_many(nodes: list[onnx.NodeProto], sources: list[str], output: str) -> str:
    if len(sources) == 1:
        nodes.append(helper.make_node("Identity", [sources[0]], [output]))
        return output
    current = sources[0]
    for index, source in enumerate(sources[1:], start=1):
        target = output if index == len(sources) - 1 else f"{output}_partial_{index}"
        nodes.append(helper.make_node("And", [current, source], [target]))
        current = target
    return output


def _or_many(nodes: list[onnx.NodeProto], sources: list[str], output: str) -> str:
    if len(sources) == 1:
        nodes.append(helper.make_node("Identity", [sources[0]], [output]))
        return output
    current = sources[0]
    for index, source in enumerate(sources[1:], start=1):
        target = output if index == len(sources) - 1 else f"{output}_partial_{index}"
        nodes.append(helper.make_node("Or", [current, source], [target]))
        current = target
    return output


def _sum_bool_as_u8(nodes: list[onnx.NodeProto], sources: list[str], output: str) -> str:
    casted = []
    for index, source in enumerate(sources):
        casted_name = f"{output}_cast_{index}"
        nodes.append(helper.make_node("Cast", [source], [casted_name], to=onnx.TensorProto.UINT8))
        casted.append(casted_name)
    return _sum_many(nodes, casted, output)


def _conv_sum(
    nodes: list[onnx.NodeProto],
    source: str,
    output: str,
    weight: str,
    pads: list[int],
) -> str:
    nodes.extend(
        [
            helper.make_node("Cast", [source], [f"{output}_f16"], to=INTERNAL_TYPE),
            helper.make_node("Conv", [f"{output}_f16", weight], [output], pads=pads),
        ]
    )
    return output


def _conv_sum_f16(
    nodes: list[onnx.NodeProto],
    source_f16: str,
    output: str,
    weight: str,
    pads: list[int],
) -> str:
    nodes.append(helper.make_node("Conv", [source_f16, weight], [output], pads=pads))
    return output


def _active_tiles_conv(nodes: list[onnx.NodeProto], remaining: str, prefix: str, kind: str) -> str:
    if kind == "s":
        weight, pads, target = "w_square", [0, 0, 1, 1], "four_f16"
    elif kind == "h":
        weight, pads, target = "w_hbar", [0, 0, 0, 2], "three_f16"
    elif kind == "v":
        weight, pads, target = "w_vbar", [0, 0, 2, 0], "three_f16"
    else:
        raise ValueError(kind)
    score = _conv_sum(nodes, remaining, f"{prefix}_score", weight, pads)
    nodes.append(helper.make_node("Equal", [score, target], [f"{prefix}_active"]))
    return f"{prefix}_active"


def _active_tiles_conv_f16(nodes: list[onnx.NodeProto], remaining_f16: str, prefix: str, kind: str) -> str:
    if kind == "s":
        weight, pads, target = "w_square", [0, 0, 1, 1], "four_f16"
    elif kind == "h":
        weight, pads, target = "w_hbar", [0, 0, 0, 2], "three_f16"
    elif kind == "v":
        weight, pads, target = "w_vbar", [0, 0, 2, 0], "three_f16"
    else:
        raise ValueError(kind)
    score = _conv_sum_f16(nodes, remaining_f16, f"{prefix}_score", weight, pads)
    nodes.append(helper.make_node("Equal", [score, target], [f"{prefix}_active"]))
    return f"{prefix}_active"


def _cover_score_conv(nodes: list[onnx.NodeProto], active: str, prefix: str, kind: str) -> str:
    if kind == "s":
        weight, pads = "w_square", [1, 1, 0, 0]
    elif kind == "h":
        weight, pads = "w_hbar", [0, 2, 0, 0]
    elif kind == "v":
        weight, pads = "w_vbar", [2, 0, 0, 0]
    else:
        raise ValueError(kind)
    return _conv_sum(nodes, active, f"{prefix}_cover_score", weight, pads)


def _cover_from_tiles_direct(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    active: str,
    offsets: list[tuple[int, int]],
    prefix: str,
) -> str:
    cover_parts = [
        active if (dr, dc) == (0, 0) else _shift(nodes, initializers, active, f"{prefix}_cover_{index}", dr, dc)
        for index, (dr, dc) in enumerate(offsets)
    ]
    return _or_many(nodes, cover_parts, f"{prefix}_cover")


def _cover_bool_conv(nodes: list[onnx.NodeProto], active: str, prefix: str, kind: str) -> str:
    score = _cover_score_conv(nodes, active, prefix, kind)
    nodes.append(helper.make_node("Greater", [score, "zero_f16"], [f"{prefix}_cover"]))
    return f"{prefix}_cover"


def _selected_tiles_conv(nodes: list[onnx.NodeProto], active: str, forced: str, prefix: str, kind: str) -> str:
    if kind == "s":
        weight, pads = "w_square", [0, 0, 1, 1]
    elif kind == "h":
        weight, pads = "w_hbar", [0, 0, 0, 2]
    elif kind == "v":
        weight, pads = "w_vbar", [0, 0, 2, 0]
    else:
        raise ValueError(kind)
    forced_score = _conv_sum(nodes, forced, f"{prefix}_forced_score", weight, pads)
    nodes.extend(
        [
            helper.make_node("Greater", [forced_score, "zero_f16"], [f"{prefix}_forced_bool"]),
            helper.make_node("And", [active, f"{prefix}_forced_bool"], [f"{prefix}_selected"]),
        ]
    )
    return f"{prefix}_selected"


def _selected_tiles_conv_f16(nodes: list[onnx.NodeProto], active: str, forced_f16: str, prefix: str, kind: str) -> str:
    if kind == "s":
        weight, pads = "w_square", [0, 0, 1, 1]
    elif kind == "h":
        weight, pads = "w_hbar", [0, 0, 0, 2]
    elif kind == "v":
        weight, pads = "w_vbar", [0, 0, 2, 0]
    else:
        raise ValueError(kind)
    forced_score = _conv_sum_f16(nodes, forced_f16, f"{prefix}_forced_score", weight, pads)
    nodes.extend(
        [
            helper.make_node("Greater", [forced_score, "zero_f16"], [f"{prefix}_forced_bool"]),
            helper.make_node("And", [active, f"{prefix}_forced_bool"], [f"{prefix}_selected"]),
        ]
    )
    return f"{prefix}_selected"


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
    return _and_many(nodes, aligned, f"{prefix}_active")


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
    return _or_many(nodes, cover_parts, f"{prefix}_cover")


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
    forced_bool = _or_many(nodes, forced_at_anchor, f"{prefix}_forced_bool")
    nodes.extend(
        [
            helper.make_node("And", [active, forced_bool], [f"{prefix}_selected"]),
        ]
    )
    return f"{prefix}_selected"


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
        _f16_tensor("zero_f16", [0.0], [1]),
        _f16_tensor("one_f16", [1.0], [1]),
        _f16_tensor("three_f16", [3.0], [1]),
        _f16_tensor("four_f16", [4.0], [1]),
        _f16_tensor("w_square", [1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]),
        _f16_tensor("w_hbar", [1.0, 1.0, 1.0], [1, 1, 1, 3]),
        _f16_tensor("w_vbar", [1.0, 1.0, 1.0], [1, 1, 3, 1]),
        _f16_tensor(
            "w_cross5",
            [
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
            ],
            [1, 1, 5, 5],
        ),
        _int64_tensor("pads_output", [0, 0, 30 - HEIGHT, 30 - WIDTH], [4]),
        _int64_tensor("black_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("black_ends", [1, 1, HEIGHT, WIDTH], [4]),
        _int64_tensor("gray_starts", [0, 5, 0, 0], [4]),
        _int64_tensor("gray_ends", [1, 6, HEIGHT, WIDTH], [4]),
        _int64_tensor("axes_hw", [2, 3], [2]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "black_starts", "black_ends"], ["input0_f32"]),
        helper.make_node("Slice", ["input", "gray_starts", "gray_ends"], ["gray_f32"]),
        helper.make_node("Cast", ["input0_f32"], ["input0_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["gray_f32"], ["gray_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("And", ["input0_bool", "gray_bool"], ["zero_grid"]),
    ]

    square_acc = "zero_grid"
    bar_acc = "zero_grid"
    remaining = "gray_bool"
    for step in range(PROPAGATION_STEPS):
        nodes.append(helper.make_node("Cast", [remaining], [f"step{step}_remaining_f16"], to=INTERNAL_TYPE))
        remaining_f16 = f"step{step}_remaining_f16"
        s_active = _active_tiles_conv_f16(nodes, remaining_f16, f"step{step}_s", "s")
        h_active = _active_tiles_conv_f16(nodes, remaining_f16, f"step{step}_h", "h")
        v_active = _active_tiles_conv_f16(nodes, remaining_f16, f"step{step}_v", "v")
        s_count = _cover_score_conv(nodes, s_active, f"step{step}_s_count", "s")
        h_count = _cover_score_conv(nodes, h_active, f"step{step}_h_count", "h")
        v_count = _cover_score_conv(nodes, v_active, f"step{step}_v_count", "v")

        nodes.extend(
            [
                helper.make_node("Add", [s_count, h_count], [f"step{step}_candidate_count_partial"]),
                helper.make_node("Add", [f"step{step}_candidate_count_partial", v_count], [f"step{step}_candidate_count"]),
                helper.make_node("Equal", [f"step{step}_candidate_count", "one_f16"], [f"step{step}_count_one_bool"]),
                helper.make_node("And", [remaining, f"step{step}_count_one_bool"], [f"step{step}_forced"]),
            ]
        )
        forced = f"step{step}_forced"
        nodes.append(helper.make_node("Cast", [forced], [f"step{step}_forced_f16"], to=INTERNAL_TYPE))
        forced_f16 = f"step{step}_forced_f16"

        s_selected = _selected_tiles_conv_f16(nodes, s_active, forced_f16, f"step{step}_s", "s")
        h_selected = _selected_tiles_conv_f16(nodes, h_active, forced_f16, f"step{step}_h", "h")
        v_selected = _selected_tiles_conv_f16(nodes, v_active, forced_f16, f"step{step}_v", "v")

        s_cover = _cover_bool_conv(nodes, s_selected, f"step{step}_s_selected", "s")
        h_cover = _cover_from_tiles_direct(nodes, initializers, h_selected, H_BAR, f"step{step}_h_selected")
        v_cover = _cover_from_tiles_direct(nodes, initializers, v_selected, V_BAR, f"step{step}_v_selected")
        bar_cover = _or_many(nodes, [h_cover, v_cover], f"step{step}_bar_cover")
        nodes.extend(
            [
                helper.make_node("Or", [square_acc, s_cover], [f"step{step}_square_acc"]),
                helper.make_node("Or", [bar_acc, bar_cover], [f"step{step}_bar_acc"]),
            ]
        )
        nodes.append(helper.make_node("Or", [s_cover, bar_cover], [f"step{step}_new_cover"]))
        new_cover = f"step{step}_new_cover"
        nodes.extend(
            [
                helper.make_node("Not", [new_cover], [f"step{step}_not_new_cover"]),
                helper.make_node("And", [remaining, f"step{step}_not_new_cover"], [f"step{step}_remaining"]),
            ]
        )
        square_acc = f"step{step}_square_acc"
        bar_acc = f"step{step}_bar_acc"
        remaining = f"step{step}_remaining"

    nodes.append(helper.make_node("Cast", [remaining], ["remaining_final_f16"], to=INTERNAL_TYPE))
    remaining_s_active = _active_tiles_conv_f16(nodes, "remaining_final_f16", "remaining_s", "s")
    remaining_s_cover = _cover_from_tiles_direct(nodes, initializers, remaining_s_active, SQUARE, "remaining_s")
    nodes.extend(
        [
            helper.make_node("Not", [remaining_s_cover], ["not_remaining_square_cover"]),
            helper.make_node("And", [remaining, "not_remaining_square_cover"], ["remaining_bar_seed"]),
            helper.make_node("Cast", ["remaining_bar_seed"], ["remaining_bar_seed_f16"], to=INTERNAL_TYPE),
            helper.make_node("Conv", ["remaining_bar_seed_f16", "w_cross5"], ["remaining_bar_seed_cross"], pads=[2, 2, 2, 2]),
            helper.make_node("Greater", ["remaining_bar_seed_cross", "zero_f16"], ["remaining_bar_seed_grown"]),
            helper.make_node("And", [remaining, "remaining_bar_seed_grown"], ["remaining_bar"]),
            helper.make_node("Or", [bar_acc, "remaining_bar"], ["bar_acc_with_remaining"]),
            helper.make_node("Or", [square_acc, remaining], ["square_acc_with_remaining"]),
        ]
    )
    bar_acc = "bar_acc_with_remaining"
    square_acc = "square_acc_with_remaining"

    nodes.extend(
        [
            helper.make_node("Where", ["input0_bool", "zero_u8", "invalid_u8"], ["color_base"]),
            helper.make_node("Where", [square_acc, "eight_u8", "color_base"], ["color_square"]),
            helper.make_node("Where", [bar_acc, "two_u8", "color_square"], ["color11"]),
            helper.make_node("Pad", ["color11", "pads_output", "invalid_u8", "axes_hw"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task023_window11_cover_propagation_graph", [x], [y], initializers)
    _dedupe_initializers(graph)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
