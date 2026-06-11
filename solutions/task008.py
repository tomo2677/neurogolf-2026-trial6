from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def _float_tensor(name: str, value: float) -> onnx.TensorProto:
    return helper.make_tensor(name, DATA_TYPE, [1, 1, 1, 1], [value])


def _add_chain(nodes: list[onnx.NodeProto], terms: list[str], output: str) -> None:
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


def _eq_const(nodes: list[onnx.NodeProto], value: str, const: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Sub", [value, const], [f"{output}_diff"]),
            helper.make_node("Abs", [f"{output}_diff"], [f"{output}_abs"]),
            helper.make_node("Min", [f"{output}_abs", "one1111"], [f"{output}_nonmatch"]),
            helper.make_node("Sub", ["one1111", f"{output}_nonmatch"], [output]),
        ]
    )


def _bbox_axis(nodes: list[onnx.NodeProto], mask: str, prefix: str, axis: str) -> tuple[str, str]:
    present: list[str] = []
    for index in range(30):
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [mask, f"{axis}_start_{index}", f"{axis}_end_{index}", "axes4", "steps4"],
                    [f"{prefix}_{axis}_slice_{index}"],
                ),
                helper.make_node(
                    "ReduceSum",
                    [f"{prefix}_{axis}_slice_{index}"],
                    [f"{prefix}_{axis}_sum_{index}"],
                    axes=[2, 3],
                    keepdims=1,
                ),
                helper.make_node("Min", [f"{prefix}_{axis}_sum_{index}", "one1111"], [f"{prefix}_{axis}_present_{index}"]),
            ]
        )
        present.append(f"{prefix}_{axis}_present_{index}")

    first_terms: list[str] = []
    last_terms: list[str] = []
    for index, item in enumerate(present):
        before = f"{prefix}_{axis}_before_{index}"
        after = f"{prefix}_{axis}_after_{index}"
        _add_chain(nodes, present[:index], before)
        _add_chain(nodes, present[index + 1 :], after)
        nodes.extend(
            [
                helper.make_node("Min", [before, "one1111"], [f"{before}_any"]),
                helper.make_node("Sub", ["one1111", f"{before}_any"], [f"{prefix}_{axis}_no_before_{index}"]),
                helper.make_node("Mul", [item, f"{prefix}_{axis}_no_before_{index}"], [f"{prefix}_{axis}_first_hot_{index}"]),
                helper.make_node("Mul", [f"{prefix}_{axis}_first_hot_{index}", f"coord_{index}"], [f"{prefix}_{axis}_first_val_{index}"]),
                helper.make_node("Min", [after, "one1111"], [f"{after}_any"]),
                helper.make_node("Sub", ["one1111", f"{after}_any"], [f"{prefix}_{axis}_no_after_{index}"]),
                helper.make_node("Mul", [item, f"{prefix}_{axis}_no_after_{index}"], [f"{prefix}_{axis}_last_hot_{index}"]),
                helper.make_node("Mul", [f"{prefix}_{axis}_last_hot_{index}", f"coord_{index}"], [f"{prefix}_{axis}_last_val_{index}"]),
            ]
        )
        first_terms.append(f"{prefix}_{axis}_first_val_{index}")
        last_terms.append(f"{prefix}_{axis}_last_val_{index}")

    first = f"{prefix}_{axis}_first"
    last = f"{prefix}_{axis}_last"
    _add_chain(nodes, first_terms, first)
    _add_chain(nodes, last_terms, last)
    return first, last


def _fixed_shift(nodes: list[onnx.NodeProto], source: str, dy: int, dx: int, output: str) -> None:
    if dy != 0:
        amount = abs(dy)
        if dy > 0:
            nodes.extend(
                [
                    helper.make_node("Slice", [source, f"shift_down_start_{dy}", f"shift_down_end_{dy}", "axes4", "steps4"], [f"{output}_body"]),
                    helper.make_node("ConstantOfShape", [f"shape_shift_rows_{amount}"], [f"{output}_pad"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
                    helper.make_node("Concat", [f"{output}_pad", f"{output}_body"], [output], axis=2),
                ]
            )
        else:
            nodes.extend(
                [
                    helper.make_node("Slice", [source, f"shift_up_start_{amount}", f"shift_up_end_{amount}", "axes4", "steps4"], [f"{output}_body"]),
                    helper.make_node("ConstantOfShape", [f"shape_shift_rows_{amount}"], [f"{output}_pad"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
                    helper.make_node("Concat", [f"{output}_body", f"{output}_pad"], [output], axis=2),
                ]
            )
        return

    if dx != 0:
        amount = abs(dx)
        if dx > 0:
            nodes.extend(
                [
                    helper.make_node("Slice", [source, f"shift_right_start_{dx}", f"shift_right_end_{dx}", "axes4", "steps4"], [f"{output}_body"]),
                    helper.make_node("ConstantOfShape", [f"shape_shift_cols_{amount}"], [f"{output}_pad"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
                    helper.make_node("Concat", [f"{output}_pad", f"{output}_body"], [output], axis=3),
                ]
            )
        else:
            nodes.extend(
                [
                    helper.make_node("Slice", [source, f"shift_left_start_{amount}", f"shift_left_end_{amount}", "axes4", "steps4"], [f"{output}_body"]),
                    helper.make_node("ConstantOfShape", [f"shape_shift_cols_{amount}"], [f"{output}_pad"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
                    helper.make_node("Concat", [f"{output}_body", f"{output}_pad"], [output], axis=3),
                ]
            )
        return

    nodes.append(helper.make_node("Identity", [source], [output]))


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("starts_red", [0, 2, 0, 0]),
        _int64_tensor("ends_red", [1, 3, 30, 30]),
        _int64_tensor("starts_blue", [0, 8, 0, 0]),
        _int64_tensor("ends_blue", [1, 9, 30, 30]),
        _int64_tensor("axes4", [0, 1, 2, 3]),
        _int64_tensor("steps4", [1, 1, 1, 1]),
        _int64_tensor("shape_channel", [1, 1, 30, 30]),
        _float_tensor("zero1111", 0.0),
        _float_tensor("one1111", 1.0),
        _float_tensor("minus_one1111", -1.0),
    ]
    for index in range(30):
        initializers.append(_float_tensor(f"coord_{index}", float(index)))
        initializers.append(_int64_tensor(f"row_start_{index}", [0, 0, index, 0]))
        initializers.append(_int64_tensor(f"row_end_{index}", [1, 1, index + 1, 30]))
        initializers.append(_int64_tensor(f"col_start_{index}", [0, 0, 0, index]))
        initializers.append(_int64_tensor(f"col_end_{index}", [1, 1, 30, index + 1]))
        initializers.append(_float_tensor(f"offset_{index}", float(index)))
        if index > 0:
            initializers.append(_float_tensor(f"offset_neg_{index}", float(-index)))
            initializers.append(_int64_tensor(f"shape_shift_rows_{index}", [1, 1, index, 30]))
            initializers.append(_int64_tensor(f"shape_shift_cols_{index}", [1, 1, 30, index]))
            initializers.append(_int64_tensor(f"shift_down_start_{index}", [0, 0, 0, 0]))
            initializers.append(_int64_tensor(f"shift_down_end_{index}", [1, 1, 30 - index, 30]))
            initializers.append(_int64_tensor(f"shift_up_start_{index}", [0, 0, index, 0]))
            initializers.append(_int64_tensor(f"shift_up_end_{index}", [1, 1, 30, 30]))
            initializers.append(_int64_tensor(f"shift_right_start_{index}", [0, 0, 0, 0]))
            initializers.append(_int64_tensor(f"shift_right_end_{index}", [1, 1, 30, 30 - index]))
            initializers.append(_int64_tensor(f"shift_left_start_{index}", [0, 0, 0, index]))
            initializers.append(_int64_tensor(f"shift_left_end_{index}", [1, 1, 30, 30]))

    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "starts_red", "ends_red", "axes4", "steps4"], ["red"]),
        helper.make_node("Slice", ["input", "starts_blue", "ends_blue", "axes4", "steps4"], ["blue"]),
        helper.make_node("ReduceSum", ["input"], ["valid_mask"], axes=[1], keepdims=1),
        helper.make_node("ConstantOfShape", ["shape_channel"], ["zero_channel"], value=zero_value),
    ]

    red_top, red_bottom = _bbox_axis(nodes, "red", "red", "row")
    red_left, red_right = _bbox_axis(nodes, "red", "red", "col")
    blue_top, blue_bottom = _bbox_axis(nodes, "blue", "blue", "row")
    blue_left, blue_right = _bbox_axis(nodes, "blue", "blue", "col")

    _gt(nodes, blue_top, red_bottom, "move_down")
    _gt(nodes, red_top, blue_bottom, "move_up")
    _gt(nodes, blue_left, red_right, "move_right")
    _gt(nodes, red_left, blue_right, "move_left")

    nodes.extend(
        [
            helper.make_node("Sub", [blue_top, red_bottom], ["down_gap_plus_one"]),
            helper.make_node("Sub", ["down_gap_plus_one", "one1111"], ["down_offset"]),
            helper.make_node("Sub", [blue_bottom, red_top], ["up_gap_minus_one"]),
            helper.make_node("Add", ["up_gap_minus_one", "one1111"], ["up_offset"]),
            helper.make_node("Sub", [blue_left, red_right], ["right_gap_plus_one"]),
            helper.make_node("Sub", ["right_gap_plus_one", "one1111"], ["right_offset"]),
            helper.make_node("Sub", [blue_right, red_left], ["left_gap_minus_one"]),
            helper.make_node("Add", ["left_gap_minus_one", "one1111"], ["left_offset"]),
            helper.make_node("Mul", ["down_offset", "move_down"], ["dy_down"]),
            helper.make_node("Mul", ["up_offset", "move_up"], ["dy_up"]),
            helper.make_node("Add", ["dy_down", "dy_up"], ["dy_offset"]),
            helper.make_node("Add", ["move_down", "move_up"], ["vertical_move"]),
            helper.make_node("Mul", ["right_offset", "move_right"], ["dx_right"]),
            helper.make_node("Mul", ["left_offset", "move_left"], ["dx_left"]),
            helper.make_node("Add", ["dx_right", "dx_left"], ["dx_offset"]),
            helper.make_node("Add", ["move_right", "move_left"], ["horizontal_move"]),
        ]
    )

    vertical_terms: list[str] = []
    horizontal_terms: list[str] = []
    for dy in range(-29, 30):
        const_name = f"offset_neg_{abs(dy)}" if dy < 0 else f"offset_{dy}"
        _eq_const(nodes, "dy_offset", const_name, f"dy_eq_{dy}")
        nodes.append(helper.make_node("Mul", [f"dy_eq_{dy}", "vertical_move"], [f"dy_selector_{dy}"]))
        _fixed_shift(nodes, "red", dy, 0, f"red_shift_dy_{dy}")
        nodes.append(helper.make_node("Mul", [f"red_shift_dy_{dy}", f"dy_selector_{dy}"], [f"red_selected_dy_{dy}"]))
        vertical_terms.append(f"red_selected_dy_{dy}")

    for dx in range(-29, 30):
        const_name = f"offset_neg_{abs(dx)}" if dx < 0 else f"offset_{dx}"
        _eq_const(nodes, "dx_offset", const_name, f"dx_eq_{dx}")
        nodes.append(helper.make_node("Mul", [f"dx_eq_{dx}", "horizontal_move"], [f"dx_selector_{dx}"]))
        _fixed_shift(nodes, "red", 0, dx, f"red_shift_dx_{dx}")
        nodes.append(helper.make_node("Mul", [f"red_shift_dx_{dx}", f"dx_selector_{dx}"], [f"red_selected_dx_{dx}"]))
        horizontal_terms.append(f"red_selected_dx_{dx}")

    _add_chain(nodes, vertical_terms, "red_vertical")
    _add_chain(nodes, horizontal_terms, "red_horizontal")
    nodes.extend(
        [
            helper.make_node("Add", ["red_vertical", "red_horizontal"], ["red_out"]),
            helper.make_node("Add", ["red_out", "blue"], ["nonblack_out"]),
            helper.make_node("Sub", ["valid_mask", "nonblack_out"], ["black_out"]),
            helper.make_node(
                "Concat",
                [
                    "black_out",
                    "zero_channel",
                    "red_out",
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                    "blue",
                    "zero_channel",
                ],
                ["output"],
                axis=1,
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task008_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
