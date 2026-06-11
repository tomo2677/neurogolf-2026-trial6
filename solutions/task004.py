from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


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


def _last_hot_axis(nodes: list[onnx.NodeProto], mask: str, prefix: str, axis: str) -> list[str]:
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

    hot: list[str] = []
    for index, item in enumerate(present):
        after = f"{prefix}_{axis}_after_{index}"
        _add_chain(nodes, present[index + 1 :], after)
        nodes.extend(
            [
                helper.make_node("Min", [after, "one1111"], [f"{after}_any"]),
                helper.make_node("Sub", ["one1111", f"{after}_any"], [f"{prefix}_{axis}_no_after_{index}"]),
                helper.make_node("Mul", [item, f"{prefix}_{axis}_no_after_{index}"], [f"{prefix}_{axis}_last_hot_{index}"]),
            ]
        )
        hot.append(f"{prefix}_{axis}_last_hot_{index}")
    return hot


def _shift_right_one(nodes: list[onnx.NodeProto], source: str, output: str) -> None:
    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])
    nodes.extend(
        [
            helper.make_node("Slice", [source, "shift_right_start", "shift_right_end", "axes4", "steps4"], [f"{output}_body"]),
            helper.make_node("ConstantOfShape", ["shape_left_zero"], [f"{output}_left"], value=zero_value),
            helper.make_node("Concat", [f"{output}_left", f"{output}_body"], [output], axis=3),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("axes4", [0, 1, 2, 3]),
        _int64_tensor("steps4", [1, 1, 1, 1]),
        _int64_tensor("shape_channel", [1, 1, 30, 30]),
        _int64_tensor("shape_left_zero", [1, 1, 30, 1]),
        _int64_tensor("tile_row_repeats", [1, 1, 1, 30]),
        _int64_tensor("tile_col_repeats", [1, 1, 30, 1]),
        _int64_tensor("shift_right_start", [0, 0, 0, 0]),
        _int64_tensor("shift_right_end", [1, 1, 30, 29]),
        helper.make_tensor("zero1111", DATA_TYPE, [1, 1, 1, 1], [0.0]),
        helper.make_tensor("one1111", DATA_TYPE, [1, 1, 1, 1], [1.0]),
    ]
    for color in range(10):
        initializers.append(_int64_tensor(f"color_start_{color}", [0, color, 0, 0]))
        initializers.append(_int64_tensor(f"color_end_{color}", [1, color + 1, 30, 30]))
    for index in range(30):
        initializers.append(_int64_tensor(f"row_start_{index}", [0, 0, index, 0]))
        initializers.append(_int64_tensor(f"row_end_{index}", [1, 1, index + 1, 30]))
        initializers.append(_int64_tensor(f"col_start_{index}", [0, 0, 0, index]))
        initializers.append(_int64_tensor(f"col_end_{index}", [1, 1, 30, index + 1]))

    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input"], ["valid_mask"], axes=[1], keepdims=1),
        helper.make_node("ConstantOfShape", ["shape_channel"], ["zero_channel"], value=zero_value),
    ]

    color_outputs: list[str] = []
    for color in range(1, 10):
        nodes.append(helper.make_node("Slice", ["input", f"color_start_{color}", f"color_end_{color}", "axes4", "steps4"], [f"color_{color}"]))

        bottom_hot = _last_hot_axis(nodes, f"color_{color}", f"color_{color}", "row")
        right_hot = _last_hot_axis(nodes, f"color_{color}", f"color_{color}", "col")

        nodes.append(helper.make_node("Concat", bottom_hot, [f"color_{color}_bottom_vec"], axis=2))
        nodes.append(helper.make_node("Tile", [f"color_{color}_bottom_vec", "tile_row_repeats"], [f"color_{color}_bottom_mask"]))
        nodes.append(helper.make_node("Concat", right_hot, [f"color_{color}_right_vec"], axis=3))
        nodes.append(helper.make_node("Tile", [f"color_{color}_right_vec", "tile_col_repeats"], [f"color_{color}_right_mask"]))
        nodes.extend(
            [
                helper.make_node("Add", [f"color_{color}_bottom_mask", f"color_{color}_right_mask"], [f"color_{color}_keep_union_raw"]),
                helper.make_node("Min", [f"color_{color}_keep_union_raw", "one1111"], [f"color_{color}_keep_union"]),
                helper.make_node("Sub", ["one1111", f"color_{color}_bottom_mask"], [f"color_{color}_not_bottom"]),
                helper.make_node("Sub", ["one1111", f"color_{color}_right_mask"], [f"color_{color}_not_right"]),
                helper.make_node("Mul", [f"color_{color}", f"color_{color}_keep_union"], [f"color_{color}_kept"]),
                helper.make_node("Mul", [f"color_{color}", f"color_{color}_not_bottom"], [f"color_{color}_shift_candidate_a"]),
                helper.make_node("Mul", [f"color_{color}_shift_candidate_a", f"color_{color}_not_right"], [f"color_{color}_shift_source"]),
            ]
        )
        _shift_right_one(nodes, f"color_{color}_shift_source", f"color_{color}_shifted")
        nodes.append(helper.make_node("Add", [f"color_{color}_kept", f"color_{color}_shifted"], [f"color_{color}_out"]))
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

    graph = helper.make_graph(nodes, "task004_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
