from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("starts_pattern", [0, 0, 0, 0]),
        _int64_tensor("ends_pattern", [1, 10, 3, 3]),
        _int64_tensor("starts_nonzero", [0, 1, 0, 0]),
        _int64_tensor("ends_nonzero", [1, 10, 3, 3]),
        _int64_tensor("axes4", [0, 1, 2, 3]),
        _int64_tensor("steps4", [1, 1, 1, 1]),
        _int64_tensor("tile_repeats", [1, 1, 3, 3]),
        _int64_tensor("shape_ones9", [1, 1, 9, 9]),
        _int64_tensor("shape_right_zero", [1, 10, 9, 21]),
        _int64_tensor("shape_bottom_zero", [1, 10, 21, 30]),
        helper.make_tensor("mask_kernel", DATA_TYPE, [1, 1, 3, 3], np.ones((1, 1, 3, 3), dtype=np.float32).ravel()),
        helper.make_tensor("color0_selector", DATA_TYPE, [1, 10, 1, 1], [1.0] + [0.0] * 9),
    ]

    one_value = helper.make_tensor("one_value", DATA_TYPE, [1], [1.0])
    zero_value = helper.make_tensor("zero_value", DATA_TYPE, [1], [0.0])

    nodes = [
        helper.make_node(
            "Slice",
            ["input", "starts_pattern", "ends_pattern", "axes4", "steps4"],
            ["pattern3"],
        ),
        helper.make_node(
            "Slice",
            ["input", "starts_nonzero", "ends_nonzero", "axes4", "steps4"],
            ["nonzero_channels3"],
        ),
        helper.make_node("ReduceSum", ["nonzero_channels3"], ["mask3"], axes=[1], keepdims=1),
        helper.make_node("Tile", ["pattern3", "tile_repeats"], ["pattern9"]),
        helper.make_node(
            "ConvTranspose",
            ["mask3", "mask_kernel"],
            ["mask9"],
            strides=[3, 3],
            kernel_shape=[3, 3],
        ),
        helper.make_node("ConstantOfShape", ["shape_ones9"], ["ones9"], value=one_value),
        helper.make_node("Sub", ["ones9", "mask9"], ["inverse_mask9"]),
        helper.make_node("Mul", ["pattern9", "mask9"], ["masked_pattern9"]),
        helper.make_node("Mul", ["inverse_mask9", "color0_selector"], ["zero_block_fill9"]),
        helper.make_node("Add", ["masked_pattern9", "zero_block_fill9"], ["output9"]),
        helper.make_node("ConstantOfShape", ["shape_right_zero"], ["right_zero"], value=zero_value),
        helper.make_node("Concat", ["output9", "right_zero"], ["output_top"], axis=3),
        helper.make_node("ConstantOfShape", ["shape_bottom_zero"], ["bottom_zero"], value=zero_value),
        helper.make_node("Concat", ["output_top", "bottom_zero"], ["output"], axis=2),
    ]

    graph = helper.make_graph(nodes, "task001_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
