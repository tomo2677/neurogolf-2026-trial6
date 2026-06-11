from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def _add_chain(nodes: list[onnx.NodeProto], terms: list[str], output: str) -> None:
    if not terms:
        nodes.append(helper.make_node("Identity", ["zero_channel"], [output]))
        return
    current = terms[0]
    for index, term in enumerate(terms[1:], start=1):
        name = f"{output}_add_{index}"
        nodes.append(helper.make_node("Add", [current, term], [name]))
        current = name
    nodes.append(helper.make_node("Identity", [current], [output]))


def _add_scalar_chain(nodes: list[onnx.NodeProto], terms: list[str], output: str) -> None:
    if not terms:
        nodes.append(helper.make_node("Identity", ["zero1111"], [output]))
        return
    current = terms[0]
    for index, term in enumerate(terms[1:], start=1):
        name = f"{output}_add_{index}"
        nodes.append(helper.make_node("Add", [current, term], [name]))
        current = name
    nodes.append(helper.make_node("Identity", [current], [output]))


def _gt(nodes: list[onnx.NodeProto], left: str, right: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Sub", [left, right], [f"{output}_diff"]),
            helper.make_node("Max", [f"{output}_diff", "zero1111"], [f"{output}_positive"]),
            helper.make_node("Min", [f"{output}_positive", "one1111"], [output]),
        ]
    )


def _shift_axis(nodes: list[onnx.NodeProto], source: str, amount: int, axis: int, output: str) -> None:
    if amount == 0:
        nodes.append(helper.make_node("Identity", [source], [output]))
        return

    abs_amount = abs(amount)
    if axis == 2:
        if amount > 0:
            nodes.extend(
                [
                    helper.make_node("Slice", [source, f"shift_down_start_{abs_amount}", f"shift_down_end_{abs_amount}", "axes4", "steps4"], [f"{output}_body"]),
                    helper.make_node("ConstantOfShape", [f"shape_shift_rows_{abs_amount}"], [f"{output}_pad"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
                    helper.make_node("Concat", [f"{output}_pad", f"{output}_body"], [output], axis=2),
                ]
            )
        else:
            nodes.extend(
                [
                    helper.make_node("Slice", [source, f"shift_up_start_{abs_amount}", f"shift_up_end_{abs_amount}", "axes4", "steps4"], [f"{output}_body"]),
                    helper.make_node("ConstantOfShape", [f"shape_shift_rows_{abs_amount}"], [f"{output}_pad"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
                    helper.make_node("Concat", [f"{output}_body", f"{output}_pad"], [output], axis=2),
                ]
            )
        return

    if amount > 0:
        nodes.extend(
            [
                helper.make_node("Slice", [source, f"shift_right_start_{abs_amount}", f"shift_right_end_{abs_amount}", "axes4", "steps4"], [f"{output}_body"]),
                helper.make_node("ConstantOfShape", [f"shape_shift_cols_{abs_amount}"], [f"{output}_pad"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
                helper.make_node("Concat", [f"{output}_pad", f"{output}_body"], [output], axis=3),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Slice", [source, f"shift_left_start_{abs_amount}", f"shift_left_end_{abs_amount}", "axes4", "steps4"], [f"{output}_body"]),
                helper.make_node("ConstantOfShape", [f"shape_shift_cols_{abs_amount}"], [f"{output}_pad"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
                helper.make_node("Concat", [f"{output}_body", f"{output}_pad"], [output], axis=3),
            ]
        )


def _shift_2d(nodes: list[onnx.NodeProto], source: str, dy: int, dx: int, output: str) -> None:
    if dy == 0:
        vertical = source
    else:
        vertical = f"{output}_v"
        _shift_axis(nodes, source, dy, 2, vertical)
    _shift_axis(nodes, vertical, dx, 3, output)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("axes4", [0, 1, 2, 3]),
        _int64_tensor("steps4", [1, 1, 1, 1]),
        _int64_tensor("shape_channel", [1, 1, 30, 30]),
        helper.make_tensor("zero1111", DATA_TYPE, [1, 1, 1, 1], [0.0]),
        helper.make_tensor("one1111", DATA_TYPE, [1, 1, 1, 1], [1.0]),
        helper.make_tensor("stamp_kernel", DATA_TYPE, [1, 1, 3, 3], np.ones((1, 1, 3, 3), dtype=np.float32).ravel()),
    ]
    for color in range(10):
        initializers.append(_int64_tensor(f"color_start_{color}", [0, color, 0, 0]))
        initializers.append(_int64_tensor(f"color_end_{color}", [1, color + 1, 30, 30]))
    for amount in range(4, 25, 4):
        initializers.append(_int64_tensor(f"shape_shift_rows_{amount}", [1, 1, amount, 30]))
        initializers.append(_int64_tensor(f"shape_shift_cols_{amount}", [1, 1, 30, amount]))
        initializers.append(_int64_tensor(f"shift_down_start_{amount}", [0, 0, 0, 0]))
        initializers.append(_int64_tensor(f"shift_down_end_{amount}", [1, 1, 30 - amount, 30]))
        initializers.append(_int64_tensor(f"shift_up_start_{amount}", [0, 0, amount, 0]))
        initializers.append(_int64_tensor(f"shift_up_end_{amount}", [1, 1, 30, 30]))
        initializers.append(_int64_tensor(f"shift_right_start_{amount}", [0, 0, 0, 0]))
        initializers.append(_int64_tensor(f"shift_right_end_{amount}", [1, 1, 30, 30 - amount]))
        initializers.append(_int64_tensor(f"shift_left_start_{amount}", [0, 0, 0, amount]))
        initializers.append(_int64_tensor(f"shift_left_end_{amount}", [1, 1, 30, 30]))

    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input"], ["valid_mask"], axes=[1], keepdims=1),
        helper.make_node("ConstantOfShape", ["shape_channel"], ["zero_channel"], value=zero_value),
    ]

    color_masks: dict[int, str] = {}
    for color in range(1, 10):
        color_masks[color] = f"color_{color}"
        nodes.extend(
            [
                helper.make_node("Slice", ["input", f"color_start_{color}", f"color_end_{color}", "axes4", "steps4"], [f"color_{color}"]),
                helper.make_node("Conv", [f"color_{color}", "stamp_kernel"], [f"color_{color}_stamp_counts"], kernel_shape=[3, 3]),
                helper.make_node("ReduceMax", [f"color_{color}_stamp_counts"], [f"color_{color}_score"], axes=[2, 3], keepdims=1),
            ]
        )

    for color in range(1, 10):
        greater_terms: list[str] = []
        for other in range(1, 10):
            if other == color:
                continue
            _gt(nodes, f"color_{other}_score", f"color_{color}_score", f"color_{other}_gt_{color}")
            greater_terms.append(f"color_{other}_gt_{color}")
        _add_scalar_chain(nodes, greater_terms, f"color_{color}_greater_count")
        nodes.extend(
            [
                helper.make_node("Min", [f"color_{color}_greater_count", "one1111"], [f"color_{color}_has_greater"]),
                helper.make_node("Sub", ["one1111", f"color_{color}_has_greater"], [f"seed_selector_{color}"]),
                helper.make_node("Mul", [f"color_{color}", f"seed_selector_{color}"], [f"seed_part_{color}"]),
            ]
        )

    _add_chain(nodes, [f"seed_part_{color}" for color in range(1, 10)], "seed_mask")

    shifted_seed: dict[tuple[int, int], str] = {(0, 0): "seed_mask"}
    direction_repeats: dict[tuple[int, int], str] = {}
    for dy, dx in DIRECTIONS:
        repeat_terms: list[str] = []
        for step in range(1, 7):
            shift = (4 * step * dy, 4 * step * dx)
            if shift not in shifted_seed:
                shifted_seed[shift] = f"seed_shift_{shift[0]}_{shift[1]}"
                _shift_2d(nodes, "seed_mask", shift[0], shift[1], shifted_seed[shift])
            repeat_terms.append(shifted_seed[shift])
        raw_repeat = f"repeat_{dy}_{dx}_raw"
        _add_chain(nodes, repeat_terms, raw_repeat)
        nodes.append(helper.make_node("Mul", [raw_repeat, "valid_mask"], [f"repeat_{dy}_{dx}"]))
        direction_repeats[(dy, dx)] = f"repeat_{dy}_{dx}"

    color_outputs: list[str] = []
    for color in range(1, 10):
        terms = [f"seed_part_{color}"]
        for dy, dx in DIRECTIONS:
            neighbor = shifted_seed[(4 * dy, 4 * dx)]
            nodes.extend(
                [
                    helper.make_node("Mul", [color_masks[color], neighbor], [f"frag_overlap_{color}_{dy}_{dx}"]),
                    helper.make_node("ReduceSum", [f"frag_overlap_{color}_{dy}_{dx}"], [f"frag_count_{color}_{dy}_{dx}"], axes=[2, 3], keepdims=1),
                    helper.make_node("Min", [f"frag_count_{color}_{dy}_{dx}", "one1111"], [f"frag_selector_{color}_{dy}_{dx}"]),
                    helper.make_node("Mul", [direction_repeats[(dy, dx)], f"frag_selector_{color}_{dy}_{dx}"], [f"frag_out_{color}_{dy}_{dx}"]),
                ]
            )
            terms.append(f"frag_out_{color}_{dy}_{dx}")
        _add_chain(nodes, terms, f"color_{color}_out")
        color_outputs.append(f"color_{color}_out")

    _add_chain(nodes, color_outputs, "nonblack_out")
    nodes.append(helper.make_node("Sub", ["valid_mask", "nonblack_out"], ["black_out"]))
    nodes.append(
        helper.make_node(
            "Concat",
            [
                "black_out",
                "color_1_out",
                "color_2_out",
                "color_3_out",
                "color_4_out",
                "color_5_out",
                "color_6_out",
                "color_7_out",
                "color_8_out",
                "color_9_out",
            ],
            ["output"],
            axis=1,
        )
    )

    graph = helper.make_graph(nodes, "task005_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
