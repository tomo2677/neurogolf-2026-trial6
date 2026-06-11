from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    masks = []
    for remainder in range(3):
        mask = np.zeros((1, 1, 30, 30), dtype=np.float32)
        for row in range(7):
            for col in range(7):
                if (row + col) % 3 == remainder:
                    mask[0, 0, row, col] = 1.0
        masks.append(mask)

    nonzero_channels = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_channels[:, 0, :, :] = 0.0

    initializers = [
        _int64_tensor("shape_vec", [1, 10, 1, 1]),
        helper.make_tensor("nonzero_channels", DATA_TYPE, [1, 10, 1, 1], nonzero_channels.ravel()),
        helper.make_tensor("one_vec", DATA_TYPE, [1, 10, 1, 1], np.ones((1, 10, 1, 1), dtype=np.float32).ravel()),
    ]
    for remainder, mask in enumerate(masks):
        initializers.append(helper.make_tensor(f"mask_{remainder}", DATA_TYPE, [1, 1, 30, 30], mask.ravel()))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Mul", ["input", "nonzero_channels"], ["colored_input"]),
    ]

    pieces: list[str] = []
    for remainder in range(3):
        nodes.extend(
            [
                helper.make_node("Mul", ["colored_input", f"mask_{remainder}"], [f"hint_cells_{remainder}"]),
                helper.make_node("ReduceSum", [f"hint_cells_{remainder}"], [f"hint_sum_{remainder}"], axes=[2, 3], keepdims=1),
                helper.make_node("Min", [f"hint_sum_{remainder}", "one_vec"], [f"color_vec_{remainder}"]),
                helper.make_node("Mul", [f"color_vec_{remainder}", f"mask_{remainder}"], [f"piece_{remainder}"]),
            ]
        )
        pieces.append(f"piece_{remainder}")

    nodes.extend(
        [
            helper.make_node("Add", [pieces[0], pieces[1]], ["sum_01"]),
            helper.make_node("Add", ["sum_01", pieces[2]], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task007_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
