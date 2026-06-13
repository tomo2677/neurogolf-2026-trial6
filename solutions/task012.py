from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 12
CENTER_OFFSETS = [(-2, -2), (-2, 2), (-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1), (2, -2), (2, 2)]
ARM_OFFSETS = [(-2, 0), (-1, 0), (0, -2), (0, -1), (0, 1), (0, 2), (1, 0), (2, 0)]
INTERNAL_TYPE = onnx.TensorProto.FLOAT16


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT32, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, INTERNAL_TYPE, dims, np.asarray(values, dtype=np.float16).ravel())


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, np.array(values, dtype=np.float32))


def _kernel(offsets: list[tuple[int, int]]) -> list[float]:
    values = [0.0] * 25
    for dr, dc in offsets:
        values[(2 - dr) * 5 + (2 - dc)] = 1.0
    return values


def _weighted_kernel() -> list[float]:
    values = [0.0] * 25
    for dr, dc in ARM_OFFSETS:
        values[(2 - dr) * 5 + (2 - dc)] = 1.0
    for dr, dc in CENTER_OFFSETS:
        values[(2 - dr) * 5 + (2 - dc)] = 2.0
    return values


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("present_start", [1]),
        _int64_tensor("present_end", [10]),
        _int64_tensor("k2", [2]),
        _int32_tensor("input_channel_ids", list(range(1, 10)), [9]),
        _int32_tensor("slice_one", [1], [1]),
        _int32_tensor("slice_two", [2], [1]),
        _int32_tensor("slice_ten", [10], [1]),
        _int32_tensor("axes3", [1, 2, 3], [3]),
        _int64_tensor("reduce_present_axes", [0, 2, 3], [3]),
        _int64_tensor("pad_axes_hw", [2, 3], [2]),
        _int64_tensor("pads_output_hw", [0, 0, 18, 18], [4]),
        _f16_tensor("one_f16", [1.0], [1]),
        _f16_tensor("fill_kernel", _weighted_kernel(), [1, 1, 5, 5]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _u8_tensor("outside_u8", [255], [1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input", "reduce_present_axes"], ["counts10"], keepdims=0),
        helper.make_node("Slice", ["counts10", "present_start", "present_end"], ["present_scores"]),
        helper.make_node("TopK", ["present_scores", "k2"], ["top_scores", "top_indices"], axis=0, largest=1, sorted=1),
        helper.make_node("Split", ["top_indices"], ["top_idx0", "top_idx1"], axis=0, num_outputs=2),
        helper.make_node("Gather", ["input_channel_ids", "top_idx0"], ["arm_channel_id"], axis=0),
        helper.make_node("Gather", ["input_channel_ids", "top_idx1"], ["center_channel_id"], axis=0),
        helper.make_node("Add", ["center_channel_id", "slice_one"], ["center_channel_end"]),
        helper.make_node("Concat", ["center_channel_id", "slice_two", "slice_two"], ["center_start"], axis=0),
        helper.make_node("Concat", ["center_channel_end", "slice_ten", "slice_ten"], ["center_end"], axis=0),
        helper.make_node("Slice", ["input", "center_start", "center_end", "axes3"], ["center_f32"]),
        helper.make_node("Cast", ["arm_channel_id"], ["arm_color"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["center_channel_id"], ["center_color"], to=onnx.TensorProto.UINT8),
    ]

    nodes.extend(
        [
            helper.make_node("Cast", ["center_f32"], ["center_mask_f16"], to=INTERNAL_TYPE),
            helper.make_node("Conv", ["center_mask_f16", "fill_kernel"], ["fill_score"], kernel_shape=[5, 5], pads=[4, 4, 4, 4]),
            helper.make_node("Greater", ["fill_score", "one_f16"], ["center_fill"]),
            helper.make_node("Cast", ["fill_score"], ["arm_fill"], to=onnx.TensorProto.BOOL),
            helper.make_node("Where", ["arm_fill", "arm_color", "zero_u8"], ["arm_out"]),
            helper.make_node("Where", ["center_fill", "center_color", "arm_out"], ["color12_out"]),
            helper.make_node("Pad", ["color12_out", "pads_output_hw", "outside_u8", "pad_axes_hw"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )
    value_infos = [
        helper.make_tensor_value_info("center_f32", onnx.TensorProto.FLOAT, [1, 1, 8, 8]),
    ]

    graph = helper.make_graph(nodes, "task012_direct_f16_center_graph", [x], [y], initializers, value_info=value_infos)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
