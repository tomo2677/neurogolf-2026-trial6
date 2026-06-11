from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def _float_tensor(name: str, value: float) -> onnx.TensorProto:
    return helper.make_tensor(name, DATA_TYPE, [1, 1, 1, 1], [value])


def _slice_channel(nodes: list[onnx.NodeProto], channel: int, output: str) -> None:
    nodes.append(
        helper.make_node(
            "Slice",
            ["input", f"starts_ch_{channel}", f"ends_ch_{channel}", "axes4", "steps4"],
            [output],
        )
    )


def _pad_channel(nodes: list[onnx.NodeProto], top: str, output: str) -> None:
    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])
    nodes.extend(
        [
            helper.make_node("ConstantOfShape", ["shape_right_zero"], [f"{output}_right"], value=zero_value),
            helper.make_node("Concat", [top, f"{output}_right"], [f"{output}_top"], axis=3),
            helper.make_node("ConstantOfShape", ["shape_bottom_zero"], [f"{output}_bottom"], value=zero_value),
            helper.make_node("Concat", [f"{output}_top", f"{output}_bottom"], [output], axis=2),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("axes4", [0, 1, 2, 3]),
        _int64_tensor("steps4", [1, 1, 1, 1]),
        _int64_tensor("shape_channel", [1, 1, 30, 30]),
        _int64_tensor("shape_right_zero", [1, 1, 9, 21]),
        _int64_tensor("shape_bottom_zero", [1, 1, 21, 30]),
        _float_tensor("zero1111", 0.0),
        _float_tensor("one1111", 1.0),
    ]
    for channel in (0, 5):
        initializers.append(_int64_tensor(f"starts_ch_{channel}", [0, channel, 0, 0]))
        initializers.append(_int64_tensor(f"ends_ch_{channel}", [1, channel + 1, 30, 30]))
    for col in range(9):
        initializers.append(_int64_tensor(f"col_start_{col}", [0, 0, 0, col]))
        initializers.append(_int64_tensor(f"col_end_{col}", [1, 1, 9, col + 1]))
    for rank in range(4):
        initializers.append(_float_tensor(f"rank_value_{rank}", float(rank)))

    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ConstantOfShape", ["shape_channel"], ["zero_channel"], value=zero_value),
    ]

    _slice_channel(nodes, 0, "black_channel")
    _slice_channel(nodes, 5, "gray")

    for col in range(9):
        nodes.extend(
            [
                helper.make_node("Slice", ["gray", f"col_start_{col}", f"col_end_{col}", "axes4", "steps4"], [f"bar_col_{col}"]),
                helper.make_node("ReduceSum", [f"bar_col_{col}"], [f"height_{col}"], axes=[2, 3], keepdims=1),
                helper.make_node("Min", [f"height_{col}", "one1111"], [f"bar_exists_{col}"]),
            ]
        )

    for col in range(9):
        greater_terms: list[str] = []
        for other in range(9):
            if other == col:
                continue
            nodes.extend(
                [
                    helper.make_node("Sub", [f"height_{other}", f"height_{col}"], [f"diff_{other}_gt_{col}"]),
                    helper.make_node("Max", [f"diff_{other}_gt_{col}", "zero1111"], [f"positive_{other}_gt_{col}"]),
                    helper.make_node("Min", [f"positive_{other}_gt_{col}", "one1111"], [f"greater_{other}_gt_{col}"]),
                ]
            )
            greater_terms.append(f"greater_{other}_gt_{col}")

        current = greater_terms[0]
        for index, term in enumerate(greater_terms[1:], start=1):
            out = f"greater_count_{col}_{index}"
            nodes.append(helper.make_node("Add", [current, term], [out]))
            current = out
        nodes.append(helper.make_node("Identity", [current], [f"rank_count_{col}"]))

        for rank in range(4):
            nodes.extend(
                [
                    helper.make_node("Sub", [f"rank_count_{col}", f"rank_value_{rank}"], [f"rank_diff_{rank}_{col}"]),
                    helper.make_node("Abs", [f"rank_diff_{rank}_{col}"], [f"rank_abs_{rank}_{col}"]),
                    helper.make_node("Min", [f"rank_abs_{rank}_{col}", "one1111"], [f"rank_nonmatch_{rank}_{col}"]),
                    helper.make_node("Sub", ["one1111", f"rank_nonmatch_{rank}_{col}"], [f"rank_match_raw_{rank}_{col}"]),
                    helper.make_node("Mul", [f"rank_match_raw_{rank}_{col}", f"bar_exists_{col}"], [f"rank_match_{rank}_{col}"]),
                    helper.make_node("Mul", [f"bar_col_{col}", f"rank_match_{rank}_{col}"], [f"rank_{rank}_col_{col}"]),
                ]
            )

    rank_channels: list[str] = []
    for rank in range(4):
        top = f"rank_{rank}_top"
        nodes.append(
            helper.make_node(
                "Concat",
                [f"rank_{rank}_col_{col}" for col in range(9)],
                [top],
                axis=3,
            )
        )
        channel = f"rank_{rank}_channel"
        _pad_channel(nodes, top, channel)
        rank_channels.append(channel)

    nodes.append(
        helper.make_node(
            "Concat",
            [
                "black_channel",
                rank_channels[0],
                rank_channels[1],
                rank_channels[2],
                rank_channels[3],
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

    graph = helper.make_graph(nodes, "task010_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
