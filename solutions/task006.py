from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


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
        _int64_tensor("left_start", [0, 1, 0, 0]),
        _int64_tensor("left_end", [1, 2, 3, 3]),
        _int64_tensor("right_start", [0, 1, 0, 4]),
        _int64_tensor("right_end", [1, 2, 3, 7]),
        _int64_tensor("axes4", [0, 1, 2, 3]),
        _int64_tensor("steps4", [1, 1, 1, 1]),
        _int64_tensor("shape_top", [1, 1, 3, 3]),
        _int64_tensor("shape_channel", [1, 1, 30, 30]),
        _int64_tensor("shape_right_zero", [1, 1, 3, 27]),
        _int64_tensor("shape_bottom_zero", [1, 1, 27, 30]),
    ]

    one_value = helper.make_tensor("one_value", DATA_TYPE, [1], [1.0])
    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "left_start", "left_end", "axes4", "steps4"], ["left"]),
        helper.make_node("Slice", ["input", "right_start", "right_end", "axes4", "steps4"], ["right"]),
        helper.make_node("Mul", ["left", "right"], ["color2_top"]),
        helper.make_node("ConstantOfShape", ["shape_top"], ["ones_top"], value=one_value),
        helper.make_node("Sub", ["ones_top", "color2_top"], ["black_top"]),
        helper.make_node("ConstantOfShape", ["shape_channel"], ["zero_channel"], value=zero_value),
    ]

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

    graph = helper.make_graph(nodes, "task006_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
