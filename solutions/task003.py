from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def _float_tensor(name: str, shape: list[int], values: list[float]) -> onnx.TensorProto:
    return helper.make_tensor(name, DATA_TYPE, shape, values)


def _slice_row(nodes: list[onnx.NodeProto], source: str, row: int, output: str) -> None:
    nodes.append(
        helper.make_node(
            "Slice",
            [source, f"row_start_{row}", f"row_end_{row}", "axes4", "steps4"],
            [output],
        )
    )


def _row_equal(nodes: list[onnx.NodeProto], left: str, right: str, name: str) -> None:
    nodes.extend(
        [
            helper.make_node("Sub", [left, right], [f"{name}_diff"]),
            helper.make_node("Abs", [f"{name}_diff"], [f"{name}_abs"]),
            helper.make_node("ReduceSum", [f"{name}_abs"], [f"{name}_sum"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("Min", [f"{name}_sum", "one1111"], [f"{name}_any"]),
            helper.make_node("Sub", ["one1111", f"{name}_any"], [name]),
        ]
    )


def _period_pattern(nodes: list[onnx.NodeProto], p: int) -> str:
    nodes.extend(
        [
            helper.make_node(
                "Slice",
                ["x1_6x3", f"pattern_start_{p}", f"pattern_end_{p}", "axes4", "steps4"],
                [f"period_{p}"],
            ),
            helper.make_node("Tile", [f"period_{p}", f"tile_repeats_{p}"], [f"period_{p}_tiled"]),
            helper.make_node(
                "Slice",
                [f"period_{p}_tiled", "out_start", "out_end", "axes4", "steps4"],
                [f"pattern_{p}"],
            ),
            helper.make_node("Mul", [f"pattern_{p}", f"selector_{p}"], [f"selected_{p}"]),
        ]
    )
    return f"selected_{p}"


def _pad_channel(nodes: list[onnx.NodeProto], top: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("ConstantOfShape", ["shape_right_zero"], [f"{output}_right"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
            helper.make_node("Concat", [top, f"{output}_right"], [f"{output}_top"], axis=3),
            helper.make_node("ConstantOfShape", ["shape_bottom_zero"], [f"{output}_bottom"], value=helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])),
            helper.make_node("Concat", [f"{output}_top", f"{output}_bottom"], [output], axis=2),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("starts_x1", [0, 1, 0, 0]),
        _int64_tensor("ends_x1", [1, 2, 6, 3]),
        _int64_tensor("axes4", [0, 1, 2, 3]),
        _int64_tensor("steps4", [1, 1, 1, 1]),
        _int64_tensor("out_start", [0, 0, 0, 0]),
        _int64_tensor("out_end", [1, 1, 9, 3]),
        _int64_tensor("shape_top", [1, 1, 9, 3]),
        _int64_tensor("shape_channel", [1, 1, 30, 30]),
        _int64_tensor("shape_right_zero", [1, 1, 9, 27]),
        _int64_tensor("shape_bottom_zero", [1, 1, 21, 30]),
        _float_tensor("one1111", [1, 1, 1, 1], [1.0]),
    ]

    for row in range(6):
        initializers.append(_int64_tensor(f"row_start_{row}", [0, 0, row, 0]))
        initializers.append(_int64_tensor(f"row_end_{row}", [1, 1, row + 1, 3]))
    for p in range(1, 7):
        initializers.append(_int64_tensor(f"pattern_start_{p}", [0, 0, 0, 0]))
        initializers.append(_int64_tensor(f"pattern_end_{p}", [1, 1, p, 3]))
        initializers.append(_int64_tensor(f"tile_repeats_{p}", [1, 1, (9 + p - 1) // p, 1]))

    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])
    one_value = helper.make_tensor("one_value", DATA_TYPE, [1], [1.0])

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "starts_x1", "ends_x1", "axes4", "steps4"], ["x1_6x3"]),
        helper.make_node("ConstantOfShape", ["shape_top"], ["ones_top"], value=one_value),
        helper.make_node("ConstantOfShape", ["shape_channel"], ["zero_channel"], value=zero_value),
    ]

    for row in range(6):
        _slice_row(nodes, "x1_6x3", row, f"row_{row}")

    equalities: dict[tuple[int, int], str] = {}
    for left in range(6):
        for right in range(left):
            name = f"eq_{left}_{right}"
            _row_equal(nodes, f"row_{left}", f"row_{right}", name)
            equalities[(left, right)] = name

    period_checks = {
        1: [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)],
        2: [(2, 0), (3, 1), (4, 2), (5, 3)],
        3: [(3, 0), (4, 1), (5, 2)],
        4: [(4, 0), (5, 1)],
        5: [(5, 0)],
    }

    for p in range(1, 6):
        terms = [equalities[pair] for pair in period_checks[p]]
        current = terms[0]
        for index, term in enumerate(terms[1:], start=1):
            out = f"ok_{p}_mul_{index}"
            nodes.append(helper.make_node("Mul", [current, term], [out]))
            current = out
        nodes.append(helper.make_node("Identity", [current], [f"ok_{p}"]))

    remaining = "one1111"
    for p in range(1, 6):
        nodes.append(helper.make_node("Mul", [f"ok_{p}", remaining], [f"selector_{p}"]))
        nodes.append(helper.make_node("Sub", [remaining, f"selector_{p}"], [f"remaining_after_{p}"]))
        remaining = f"remaining_after_{p}"
    nodes.append(helper.make_node("Identity", [remaining], ["selector_6"]))

    selected = [_period_pattern(nodes, p) for p in range(1, 7)]
    current = selected[0]
    for index, term in enumerate(selected[1:], start=2):
        out = f"color2_top_acc_{index}"
        nodes.append(helper.make_node("Add", [current, term], [out]))
        current = out
    nodes.append(helper.make_node("Identity", [current], ["color2_top"]))
    nodes.append(helper.make_node("Sub", ["ones_top", "color2_top"], ["black_top"]))

    _pad_channel(nodes, "black_top", "black_channel")
    _pad_channel(nodes, "color2_top", "color2_channel")

    nodes.append(
        helper.make_node(
            "Concat",
            [
                "black_channel",
                "zero_channel",
                "color2_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
            ],
            ["output"],
            axis=1,
        )
    )

    graph = helper.make_graph(nodes, "task003_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
