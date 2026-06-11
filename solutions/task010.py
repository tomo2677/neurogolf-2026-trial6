from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


INTERNAL_TYPE = onnx.TensorProto.FLOAT16
NP_DTYPE = np.float16


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, INTERNAL_TYPE, dims, np.array(values, dtype=NP_DTYPE))


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("axes_slice", [1, 2, 3]),
        _int64_tensor("axis_height", [2]),
        _int64_tensor("pads_output", [0, 0, 0, 5, 21, 21]),
        _u8_tensor("one_scalar", [1], [1]),
        _u8_tensor("zero_col", [0] * 9, [1, 1, 9, 1]),
        _u8_tensor("colors5", [0, 1, 2, 3, 4], [1, 5, 1, 1]),
    ]
    for index, col in enumerate((1, 3, 5, 7)):
        initializers.append(_int64_tensor(f"bar{index}_starts", [5, 0, col]))
        initializers.append(_int64_tensor(f"bar{index}_ends", [6, 9, col + 1]))

    nodes: list[onnx.NodeProto] = []
    for index in range(4):
        nodes.extend(
            [
                helper.make_node("Slice", ["input", f"bar{index}_starts", f"bar{index}_ends", "axes_slice"], [f"bar{index}_f32"]),
                helper.make_node("ReduceSum", [f"bar{index}_f32", "axis_height"], [f"height{index}_1111"], keepdims=1),
                helper.make_node("Cast", [f"bar{index}_f32"], [f"bar{index}_bool"], to=onnx.TensorProto.BOOL),
            ]
        )

    rank_terms = {index: [] for index in range(4)}
    for left in range(4):
        for right in range(left + 1, 4):
            greater_name = f"height{left}_gt_height{right}"
            left_taller = f"{greater_name}_u8"
            right_taller = f"height{right}_gt_height{left}_u8"
            nodes.extend(
                [
                    helper.make_node("Greater", [f"height{left}_1111", f"height{right}_1111"], [greater_name]),
                    helper.make_node("Cast", [greater_name], [left_taller], to=onnx.TensorProto.UINT8),
                    helper.make_node("Sub", ["one_scalar", left_taller], [right_taller]),
                ]
            )
            rank_terms[right].append(left_taller)
            rank_terms[left].append(right_taller)

    for index in range(4):
        taller_terms = rank_terms[index]
        nodes.extend(
            [
                helper.make_node("Add", [taller_terms[0], taller_terms[1]], [f"c{index}_a"]),
                helper.make_node("Add", [f"c{index}_a", taller_terms[2]], [f"c{index}_b"]),
                helper.make_node("Add", [f"c{index}_b", "one_scalar"], [f"c{index}"]),
            ]
        )

    for index in range(4):
        nodes.append(helper.make_node("Where", [f"bar{index}_bool", f"c{index}", "zero_col"], [f"v{index}"]))

    nodes.extend(
        [
            helper.make_node(
                "Concat",
                ["zero_col", "v0", "zero_col", "v1", "zero_col", "v2", "zero_col", "v3", "zero_col"],
                ["color9"],
                axis=3,
            ),
            helper.make_node("Equal", ["colors5", "color9"], ["output5"]),
            helper.make_node("Pad", ["output5", "pads_output", "", "axes_slice"], ["output"], mode="constant"),
        ]
    )

    graph = helper.make_graph(nodes, "task010_toprow_argmax_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
