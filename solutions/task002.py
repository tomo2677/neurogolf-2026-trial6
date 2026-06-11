from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


INTERNAL_TYPE = onnx.TensorProto.FLOAT16
NP_DTYPE = np.float16


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, INTERNAL_TYPE, dims, np.array(values, dtype=NP_DTYPE))


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, np.array(values, dtype=np.float32))


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("green_starts", [0, 3, 0, 0]),
        _int64_tensor("green_ends", [1, 4, 20, 20]),
        _f32_tensor("row_idx20", [float(v) for v in range(20)], [1, 1, 20, 1]),
        _int64_tensor("pads_color", [0, 0, 0, 0, 0, 0, 10, 10]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("green_u8", [3], [1]),
        _u8_tensor("yellow_u8", [4], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["row_present"], axes=[1, 3], keepdims=1),
        helper.make_node("ReduceSum", ["row_present"], ["row_count"], keepdims=1),
        helper.make_node("Less", ["row_idx20", "row_count"], ["row_valid"]),
        helper.make_node("Transpose", ["row_valid"], ["col_valid"], perm=[0, 1, 3, 2]),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_area"]),
        helper.make_node("Slice", ["input", "green_starts", "green_ends"], ["green_f32"]),
        helper.make_node("Cast", ["green_f32"], ["green_mask_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["green_mask_u8"], ["green_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("MaxPool", ["green_mask_u8"], ["left_seen"], kernel_shape=[1, 20], pads=[0, 19, 0, 0]),
        helper.make_node("MaxPool", ["green_mask_u8"], ["right_seen"], kernel_shape=[1, 20], pads=[0, 0, 0, 19]),
        helper.make_node("MaxPool", ["green_mask_u8"], ["up_seen"], kernel_shape=[20, 1], pads=[19, 0, 0, 0]),
        helper.make_node("MaxPool", ["green_mask_u8"], ["down_seen"], kernel_shape=[20, 1], pads=[0, 0, 19, 0]),
        helper.make_node("Min", ["left_seen", "right_seen", "up_seen", "down_seen"], ["all_seen"]),
        helper.make_node("Sub", ["one_u8", "all_seen"], ["open_any_u8"]),
    ]

    external = "open_any_u8"
    flood_sequence = "HVHVVVHHVHVHV"
    for index, direction in enumerate(flood_sequence):
        pooled = f"external_pooled_{index}"
        capped = f"external_{index + 1}"
        if direction == "V":
            kernel_shape = [3, 1]
            pads = [1, 0, 1, 0]
        else:
            kernel_shape = [1, 3]
            pads = [0, 1, 0, 1]
        nodes.append(helper.make_node("MaxPool", [external], [pooled], kernel_shape=kernel_shape, pads=pads))
        if index == len(flood_sequence) - 1:
            external = pooled
        else:
            nodes.append(helper.make_node("Where", ["green_bool", "zero_u8", pooled], [capped]))
            external = capped

    nodes.extend(
        [
            helper.make_node("Cast", [external], ["external_bool"], to=onnx.TensorProto.BOOL),
            helper.make_node("Where", ["external_bool", "zero_u8", "yellow_u8"], ["fill_color"]),
            helper.make_node("Where", ["green_bool", "green_u8", "fill_color"], ["valid_color20"]),
            helper.make_node("Where", ["valid_area", "valid_color20", "invalid_u8"], ["color20"]),
            helper.make_node("Pad", ["color20", "pads_color", "invalid_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task002_pad_color_grid_output_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 14)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
