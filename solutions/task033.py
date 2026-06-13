from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _bool_tensor(name: str, values: list[bool], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.BOOL, dims, values)


def _gap3() -> list[str]:
    return ["false_col1"] * 3


def _tail13() -> list[str]:
    return ["false_col1"] * 13


def _rows3() -> list[str]:
    return ["false_row30"] * 3


def _rows13() -> list[str]:
    return ["false_row30"] * 13


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _bool_tensor("false_col1", [False] * 3, [1, 1, 3, 1]),
        _bool_tensor("false_row30", [False] * 30, [1, 1, 1, 30]),
    ]

    nodes: list[onnx.NodeProto] = []
    for br in range(3):
        for bc in range(3):
            name = f"bg{br}{bc}"
            row = (1, 7, 13)[br]
            col = (1, 7, 13)[bc]
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        ["input"],
                        [f"{name}_f32"],
                        starts=[0, row, col],
                        ends=[1, row + 3, col + 3],
                        axes=[1, 2, 3],
                    ),
                    helper.make_node("Cast", [f"{name}_f32"], [f"{name}_bool"], to=onnx.TensorProto.BOOL),
                ]
            )

    nodes.append(helper.make_node("Not", ["bg00_bool"], ["template3"]))
    for br in range(3):
        for bc in range(3):
            if br == 0 and bc == 0:
                continue
            nodes.append(helper.make_node("And", ["template3", f"bg{br}{bc}_bool"], [f"fill{br}{bc}"]))

    nodes.extend(
        [
        helper.make_node(
            "Concat",
            ["false_col1"] + _gap3() + _gap3() + ["fill01"] + _gap3() + ["fill02", "false_col1"] + _tail13(),
            ["fill_row0"],
            axis=3,
        ),
        helper.make_node(
            "Concat",
            ["false_col1", "fill10"] + _gap3() + ["fill11"] + _gap3() + ["fill12", "false_col1"] + _tail13(),
            ["fill_row1"],
            axis=3,
        ),
        helper.make_node(
            "Concat",
            ["false_col1", "fill20"] + _gap3() + ["fill21"] + _gap3() + ["fill22", "false_col1"] + _tail13(),
            ["fill_row2"],
            axis=3,
        ),
        helper.make_node(
            "Concat",
            ["false_row30", "fill_row0"] + _rows3() + ["fill_row1"] + _rows3() + ["fill_row2", "false_row30"] + _rows13(),
            ["fill30"],
            axis=2,
        ),
        helper.make_node("Slice", ["input"], ["grid_color"], starts=[5, 0], ends=[6, 1], axes=[2, 3]),
        helper.make_node("Where", ["fill30", "grid_color", "input"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task033_direct_fill30_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 9)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
