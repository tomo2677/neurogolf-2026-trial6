from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _bool_tensor(name: str, values: list[bool], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.BOOL, dims, values)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("grid_color_starts", [0, 0, 5, 0], [4]),
        _int64_tensor("grid_color_ends", [1, 10, 6, 1], [4]),
        _int64_tensor("pads_fill30_hw", [0, 0, 13, 13], [4]),
        _int64_tensor("pad_axes_hw", [2, 3], [2]),
        _bool_tensor("false_col1", [False] * 3, [1, 1, 3, 1]),
        _bool_tensor("false_gap3", [False] * 9, [1, 1, 3, 3]),
        _bool_tensor("false_row1", [False] * 17, [1, 1, 1, 17]),
        _bool_tensor("false_rows3", [False] * 51, [1, 1, 3, 17]),
    ]
    for br, row in enumerate((1, 7, 13)):
        for bc, col in enumerate((1, 7, 13)):
            name = f"bg{br}{bc}"
            initializers.extend(
                [
                    _int64_tensor(f"{name}_starts", [0, 0, row, col], [4]),
                    _int64_tensor(f"{name}_ends", [1, 1, row + 3, col + 3], [4]),
                ]
            )

    nodes: list[onnx.NodeProto] = []
    for br in range(3):
        for bc in range(3):
            name = f"bg{br}{bc}"
            nodes.extend(
                [
                    helper.make_node("Slice", ["input", f"{name}_starts", f"{name}_ends"], [f"{name}_f32"]),
                    helper.make_node("Cast", [f"{name}_f32"], [f"{name}_bool"], to=onnx.TensorProto.BOOL),
                ]
            )

    nodes.append(helper.make_node("Not", ["bg00_bool"], ["template3"]))
    for br in range(3):
        for bc in range(3):
            nodes.append(helper.make_node("And", ["template3", f"bg{br}{bc}_bool"], [f"fill{br}{bc}"]))

    nodes.extend(
        [
        helper.make_node(
            "Concat",
            ["false_col1", "fill00", "false_gap3", "fill01", "false_gap3", "fill02", "false_col1"],
            ["fill_row0"],
            axis=3,
        ),
        helper.make_node(
            "Concat",
            ["false_col1", "fill10", "false_gap3", "fill11", "false_gap3", "fill12", "false_col1"],
            ["fill_row1"],
            axis=3,
        ),
        helper.make_node(
            "Concat",
            ["false_col1", "fill20", "false_gap3", "fill21", "false_gap3", "fill22", "false_col1"],
            ["fill_row2"],
            axis=3,
        ),
        helper.make_node(
            "Concat",
            ["false_row1", "fill_row0", "false_rows3", "fill_row1", "false_rows3", "fill_row2", "false_row1"],
            ["fill17"],
            axis=2,
        ),
        helper.make_node("Pad", ["fill17", "pads_fill30_hw", "", "pad_axes_hw"], ["fill30"], mode="constant"),
        helper.make_node("Slice", ["input", "grid_color_starts", "grid_color_ends"], ["grid_color"]),
        helper.make_node("Where", ["fill30", "grid_color", "input"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task033_template_fill_pad_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
