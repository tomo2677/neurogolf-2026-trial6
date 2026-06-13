from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


HEIGHT = 9
WIDTH = 11
PROPAGATION_STEPS = 4
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
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("eight_u8", [8], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
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
        helper.make_node("Cast", ["input0_f32"], ["input0"], to=INTERNAL_TYPE),
        helper.make_node("Cast", ["gray_f32"], ["gray"], to=INTERNAL_TYPE),
        helper.make_node("Cast", ["input0"], ["input0_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["gray"], ["gray_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("And", ["input0_bool", "gray_bool"], ["zero_grid"]),
    ]

    square_acc = "zero_grid"
    bar_acc = "zero_grid"
    covered = "zero_grid"
    remaining = "gray_bool"
    for step in range(PROPAGATION_STEPS):
        s_active = _active_tiles(nodes, initializers, remaining, SQUARE, f"step{step}_s")
        h_active = _active_tiles(nodes, initializers, remaining, H_BAR, f"step{step}_h")
        v_active = _active_tiles(nodes, initializers, remaining, V_BAR, f"step{step}_v")

        cover_terms = []
        for name, active, offsets in (("s", s_active, SQUARE), ("h", h_active, H_BAR), ("v", v_active, V_BAR)):
            for index, (dr, dc) in enumerate(offsets):
                cover_terms.append(_shift(nodes, initializers, active, f"step{step}_{name}_count_{index}", dr, dc))
        count = _sum_bool_as_u8(nodes, cover_terms, f"step{step}_candidate_count")

        nodes.extend(
            [
                helper.make_node("Equal", [count, "one_u8"], [f"step{step}_count_one_bool"]),
                helper.make_node("And", [remaining, f"step{step}_count_one_bool"], [f"step{step}_forced"]),
            ]
        )
        forced = f"step{step}_forced"

        s_selected = _selected_tiles(nodes, initializers, s_active, forced, SQUARE, f"step{step}_s")
        h_selected = _selected_tiles(nodes, initializers, h_active, forced, H_BAR, f"step{step}_h")
        v_selected = _selected_tiles(nodes, initializers, v_active, forced, V_BAR, f"step{step}_v")

        s_cover = _cover_from_tiles(nodes, initializers, s_selected, SQUARE, f"step{step}_s_selected")
        h_cover = _cover_from_tiles(nodes, initializers, h_selected, H_BAR, f"step{step}_h_selected")
        v_cover = _cover_from_tiles(nodes, initializers, v_selected, V_BAR, f"step{step}_v_selected")
        bar_cover = _or_many(nodes, [h_cover, v_cover], f"step{step}_bar_cover")
        nodes.extend(
            [
                helper.make_node("Or", [square_acc, s_cover], [f"step{step}_square_acc"]),
                helper.make_node("Or", [bar_acc, bar_cover], [f"step{step}_bar_acc"]),
            ]
        )
        new_cover = _or_many(nodes, [s_cover, h_cover, v_cover], f"step{step}_new_cover")
        nodes.extend(
            [
                helper.make_node("Or", [covered, new_cover], [f"step{step}_covered"]),
                helper.make_node("Not", [f"step{step}_covered"], [f"step{step}_not_covered"]),
                helper.make_node("And", ["gray_bool", f"step{step}_not_covered"], [f"step{step}_remaining"]),
            ]
        )
        square_acc = f"step{step}_square_acc"
        bar_acc = f"step{step}_bar_acc"
        covered = f"step{step}_covered"
        remaining = f"step{step}_remaining"

    nodes.extend(
        [
            helper.make_node("Or", [square_acc, remaining], ["square_acc_with_remaining"]),
        ]
    )
    square_acc = "square_acc_with_remaining"

    nodes.extend(
        [
            helper.make_node("Where", ["input0_bool", "zero_u8", "invalid_u8"], ["color_base"]),
            helper.make_node("Where", [bar_acc, "two_u8", "color_base"], ["color_bar"]),
            helper.make_node("Where", [square_acc, "eight_u8", "color_bar"], ["color11"]),
            helper.make_node("Pad", ["color11", "pads_output", "invalid_u8", "axes_hw"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task023_window11_cover_propagation_graph", [x], [y], initializers)
    _dedupe_initializers(graph)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
