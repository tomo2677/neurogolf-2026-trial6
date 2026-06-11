from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    border = np.zeros((1, 1, 30, 30), dtype=np.float32)
    border[:, :, 0, :] = 1.0
    border[:, :, -1, :] = 1.0
    border[:, :, :, 0] = 1.0
    border[:, :, :, -1] = 1.0

    cross = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cross[0, 0, 1, 1] = 1.0
    cross[0, 0, 0, 1] = 1.0
    cross[0, 0, 2, 1] = 1.0
    cross[0, 0, 1, 0] = 1.0
    cross[0, 0, 1, 2] = 1.0

    initializers = [
        _int64_tensor("starts_black", [0, 0, 0, 0]),
        _int64_tensor("ends_black", [1, 1, 30, 30]),
        _int64_tensor("starts_green", [0, 3, 0, 0]),
        _int64_tensor("ends_green", [1, 4, 30, 30]),
        _int64_tensor("axes4", [0, 1, 2, 3]),
        _int64_tensor("steps4", [1, 1, 1, 1]),
        _int64_tensor("shape_cell", [1, 1, 30, 30]),
        helper.make_tensor("border_mask", DATA_TYPE, [1, 1, 30, 30], border.ravel()),
        helper.make_tensor("cross_kernel", DATA_TYPE, [1, 1, 3, 3], cross.ravel()),
    ]

    one_value = helper.make_tensor("one_value", DATA_TYPE, [1], [1.0])
    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])

    nodes = [
        helper.make_node(
            "Slice",
            ["input", "starts_black", "ends_black", "axes4", "steps4"],
            ["black"],
        ),
        helper.make_node(
            "Slice",
            ["input", "starts_green", "ends_green", "axes4", "steps4"],
            ["green"],
        ),
        helper.make_node("ConstantOfShape", ["shape_cell"], ["ones"], value=one_value),
        helper.make_node("ConstantOfShape", ["shape_cell"], ["zeros"], value=zero_value),
        helper.make_node("Sub", ["ones", "green"], ["passable"]),
        helper.make_node("Mul", ["border_mask", "passable"], ["external_0"]),
    ]

    external = "external_0"
    for index in range(64):
        dilated = f"external_dilated_{index}"
        capped = f"external_{index + 1}"
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    [external, "cross_kernel"],
                    [dilated],
                    pads=[1, 1, 1, 1],
                    kernel_shape=[3, 3],
                ),
                helper.make_node("Min", [dilated, "passable"], [capped]),
            ]
        )
        external = capped

    nodes.extend(
        [
            helper.make_node("Sub", ["ones", external], ["not_external"]),
            helper.make_node("Mul", ["black", external], ["black_out"]),
            helper.make_node("Mul", ["black", "not_external"], ["yellow"]),
            helper.make_node(
                "Concat",
                [
                    "black_out",
                    "zeros",
                    "zeros",
                    "green",
                    "yellow",
                    "zeros",
                    "zeros",
                    "zeros",
                    "zeros",
                    "zeros",
                ],
                ["output"],
                axis=1,
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task002_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
