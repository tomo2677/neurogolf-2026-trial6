from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def _add_chain(nodes: list[onnx.NodeProto], terms: list[str], output: str) -> None:
    current = terms[0]
    for index, term in enumerate(terms[1:], start=1):
        name = f"{output}_add_{index}"
        nodes.append(helper.make_node("Add", [current, term], [name]))
        current = name
    nodes.append(helper.make_node("Identity", [current], [output]))


def _gt_one(nodes: list[onnx.NodeProto], value: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Sub", [value, "one1111"], [f"{output}_diff"]),
            helper.make_node("Max", [f"{output}_diff", "zero1111"], [f"{output}_positive"]),
            helper.make_node("Min", [f"{output}_positive", "one1111"], [output]),
        ]
    )


def _line_fill(nodes: list[onnx.NodeProto], occ: str, prefix: str) -> str:
    nodes.extend(
        [
            helper.make_node("ReduceSum", [occ], [f"{prefix}_count"], axes=[3], keepdims=1),
        ]
    )
    _gt_one(nodes, f"{prefix}_count", f"{prefix}_multi")
    nodes.extend(
        [
            helper.make_node("MatMul", [occ, "cum_left"], [f"{prefix}_left_count"]),
            helper.make_node("MatMul", [occ, "cum_right"], [f"{prefix}_right_count"]),
            helper.make_node("Min", [f"{prefix}_left_count", "one_grid10"], [f"{prefix}_left_any"]),
            helper.make_node("Min", [f"{prefix}_right_count", "one_grid10"], [f"{prefix}_right_any"]),
            helper.make_node("Mul", [f"{prefix}_left_any", f"{prefix}_right_any"], [f"{prefix}_between"]),
            helper.make_node("Mul", [f"{prefix}_between", f"{prefix}_multi"], [f"{prefix}_fill"]),
        ]
    )
    return f"{prefix}_fill"


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    left = np.zeros((10, 10), dtype=np.float32)
    right = np.zeros((10, 10), dtype=np.float32)
    for src in range(10):
        for dst in range(10):
            if src <= dst:
                left[src, dst] = 1.0
            if src >= dst:
                right[src, dst] = 1.0

    initializers = [
        _int64_tensor("axes4", [0, 1, 2, 3]),
        _int64_tensor("steps4", [1, 1, 1, 1]),
        _int64_tensor("shape_right_zero", [1, 1, 29, 1]),
        _int64_tensor("shape_bottom_zero", [1, 1, 1, 30]),
        helper.make_tensor("zero1111", DATA_TYPE, [1, 1, 1, 1], [0.0]),
        helper.make_tensor("one1111", DATA_TYPE, [1, 1, 1, 1], [1.0]),
        helper.make_tensor("one_grid10", DATA_TYPE, [1, 1, 10, 10], np.ones((1, 1, 10, 10), dtype=np.float32).ravel()),
        helper.make_tensor("one_channel", DATA_TYPE, [1, 1, 30, 30], np.ones((1, 1, 30, 30), dtype=np.float32).ravel()),
        helper.make_tensor("body_kernel", DATA_TYPE, [1, 1, 2, 2], np.ones((1, 1, 2, 2), dtype=np.float32).ravel()),
        helper.make_tensor("expand_kernel", DATA_TYPE, [1, 1, 2, 2], np.ones((1, 1, 2, 2), dtype=np.float32).ravel()),
        helper.make_tensor("cum_left", DATA_TYPE, [10, 10], left.ravel()),
        helper.make_tensor("cum_right", DATA_TYPE, [10, 10], right.ravel()),
    ]
    for color in range(10):
        initializers.append(_int64_tensor(f"color_start_{color}", [0, color, 0, 0]))
        initializers.append(_int64_tensor(f"color_end_{color}", [1, color + 1, 30, 30]))

    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])
    nodes: list[onnx.NodeProto] = []
    nodes.append(helper.make_node("Slice", ["input", "color_start_0", "color_end_0", "axes4", "steps4"], ["input_black"]))

    color_outputs: list[str] = []
    fill_outputs: list[str] = []
    for color in range(1, 10):
        nodes.extend(
            [
                helper.make_node("Slice", ["input", f"color_start_{color}", f"color_end_{color}", "axes4", "steps4"], [f"input_color_{color}"]),
                helper.make_node(
                    "Conv",
                    [f"input_color_{color}", "body_kernel"],
                    [f"body_count_{color}"],
                    strides=[3, 3],
                    kernel_shape=[2, 2],
                ),
                helper.make_node("Min", [f"body_count_{color}", "one_grid10"], [f"occ_{color}"]),
            ]
        )

        hfill = _line_fill(nodes, f"occ_{color}", f"h_{color}")
        nodes.append(helper.make_node("Transpose", [f"occ_{color}"], [f"occ_t_{color}"], perm=[0, 1, 3, 2]))
        vfill_t = _line_fill(nodes, f"occ_t_{color}", f"v_t_{color}")
        nodes.append(helper.make_node("Transpose", [vfill_t], [f"v_{color}_fill"], perm=[0, 1, 3, 2]))
        nodes.extend(
            [
                helper.make_node("Add", [hfill, f"v_{color}_fill"], [f"cell_fill_raw_{color}"]),
                helper.make_node("Min", [f"cell_fill_raw_{color}", "one_grid10"], [f"cell_fill_{color}"]),
                helper.make_node(
                    "ConvTranspose",
                    [f"cell_fill_{color}", "expand_kernel"],
                    [f"fill_29_{color}"],
                    strides=[3, 3],
                    kernel_shape=[2, 2],
                ),
                helper.make_node("ConstantOfShape", ["shape_right_zero"], [f"fill_right_zero_{color}"], value=zero_value),
                helper.make_node("Concat", [f"fill_29_{color}", f"fill_right_zero_{color}"], [f"fill_top_{color}"], axis=3),
                helper.make_node("ConstantOfShape", ["shape_bottom_zero"], [f"fill_bottom_zero_{color}"], value=zero_value),
                helper.make_node("Concat", [f"fill_top_{color}", f"fill_bottom_zero_{color}"], [f"fill_{color}"], axis=2),
                helper.make_node("Add", [f"input_color_{color}", f"fill_{color}"], [f"color_{color}_out"]),
            ]
        )
        color_outputs.append(f"color_{color}_out")
        fill_outputs.append(f"fill_{color}")

    _add_chain(nodes, fill_outputs, "fill_sum")
    nodes.extend(
        [
            helper.make_node("Min", ["fill_sum", "one_channel"], ["any_fill"]),
            helper.make_node("Sub", ["one_channel", "any_fill"], ["not_fill"]),
            helper.make_node("Mul", ["input_black", "not_fill"], ["black_out"]),
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
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task009_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
