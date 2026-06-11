from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import DATA_TYPE, GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("zero_starts", [0, 0, 0, 0]),
        _int64_tensor("zero_ends", [1, 1, 3, 3]),
        _int64_tensor("split_color", [1, 9]),
        _int64_tensor("axes_block", [3, 5]),
        _int64_tensor("axes_inner", [2, 4]),
        _int64_tensor("shape_spatial9", [1, 1, 9, 9]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 21, 21]),
        helper.make_tensor("zero_scalar", DATA_TYPE, [1], [0.0]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "zero_starts", "zero_ends"], ["zero_pattern3"]),
        helper.make_node("Equal", ["zero_pattern3", "zero_scalar"], ["mask_bool3"]),
        helper.make_node("ReduceMax", ["input"], ["color10"], axes=[2, 3], keepdims=1),
        helper.make_node("Cast", ["color10"], ["color10_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Split", ["color10_bool", "split_color"], ["color0_bool", "color_selector_bool"], axis=1),
        helper.make_node("Unsqueeze", ["mask_bool3", "axes_block"], ["block_mask6"]),
        helper.make_node("Unsqueeze", ["mask_bool3", "axes_inner"], ["inner_mask6"]),
        helper.make_node("And", ["block_mask6", "inner_mask6"], ["spatial_bool6"]),
        helper.make_node("Reshape", ["spatial_bool6", "shape_spatial9"], ["spatial_bool9"]),
        helper.make_node("And", ["color_selector_bool", "spatial_bool9"], ["nonzero_bool9"]),
        helper.make_node("Not", ["spatial_bool9"], ["zero_bool9"]),
        helper.make_node("Concat", ["zero_bool9", "nonzero_bool9"], ["output_bool9"], axis=1),
        helper.make_node("Pad", ["output_bool9", "pads_output"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task001_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
