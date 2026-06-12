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


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    cross_kernel = np.array(
        [[[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=np.float32,
    )
    initializers = [
        _int64_tensor("present_start", [1]),
        _int64_tensor("present_end", [10]),
        _int64_tensor("k2", [2]),
        _int32_tensor("input_channel_ids", list(range(1, 10)), [9]),
        _int32_tensor("slice_zero", [0], [1]),
        _int32_tensor("slice_one", [1], [1]),
        _int32_tensor("slice_12", [SIZE], [1]),
        _int32_tensor("axes3", [1, 2, 3], [3]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 18, 18]),
        _f32_tensor("four_f32", [4.0], [1]),
        _f32_tensor("cross_kernel", cross_kernel.ravel().tolist(), [1, 1, 3, 3]),
        _f16_tensor("zero_f16", [0.0], [1]),
        _f16_tensor("center_kernel", _kernel(CENTER_OFFSETS), [1, 1, 5, 5]),
        _f16_tensor("arm_kernel", _kernel(ARM_OFFSETS), [1, 1, 5, 5]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _u8_tensor("outside_u8", [255], [1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["present_scores10"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("Slice", ["present_scores10", "present_start", "present_end"], ["present_scores"]),
        helper.make_node("TopK", ["present_scores", "k2"], ["top_scores", "top_indices"], axis=0, largest=1, sorted=0),
        helper.make_node("Split", ["top_indices"], ["top_idx0", "top_idx1"], axis=0),
    ]

    for slot in range(2):
        nodes.extend(
            [
                helper.make_node("Gather", ["input_channel_ids", f"top_idx{slot}"], [f"channel_id_{slot}"], axis=0),
                helper.make_node("Add", [f"channel_id_{slot}", "slice_one"], [f"channel_end_{slot}"]),
                helper.make_node("Concat", [f"channel_id_{slot}", "slice_zero", "slice_zero"], [f"selected_start_{slot}"], axis=0),
                helper.make_node("Concat", [f"channel_end_{slot}", "slice_12", "slice_12"], [f"selected_end_{slot}"], axis=0),
                helper.make_node("Slice", ["input", f"selected_start_{slot}", f"selected_end_{slot}", "axes3"], [f"selected_{slot}_f32"]),
                helper.make_node("Cast", [f"selected_{slot}_f32"], [f"selected_{slot}_bool"], to=onnx.TensorProto.BOOL),
                helper.make_node("Cast", [f"channel_id_{slot}"], [f"color_id_{slot}"], to=onnx.TensorProto.UINT8),
                helper.make_node("Where", [f"selected_{slot}_bool", f"color_id_{slot}", "zero_u8"], [f"selected_color_{slot}"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Max", ["selected_0_f32", "selected_1_f32"], ["nonzero_f32"]),
            helper.make_node("Cast", ["nonzero_f32"], ["nonzero_bool"], to=onnx.TensorProto.BOOL),
            helper.make_node("Max", ["selected_color_0", "selected_color_1"], ["color12"]),
            helper.make_node("Conv", ["nonzero_f32", "cross_kernel"], ["neighbor_count"], pads=[1, 1, 1, 1]),
            helper.make_node("Equal", ["neighbor_count", "four_f32"], ["has_four_neighbors"]),
            helper.make_node("And", ["nonzero_bool", "has_four_neighbors"], ["center_mask"]),
            helper.make_node("Cast", ["center_mask"], ["center_mask_f16"], to=INTERNAL_TYPE),
            helper.make_node("Conv", ["center_mask_f16", "center_kernel"], ["center_fill_score"], kernel_shape=[5, 5], pads=[2, 2, 2, 2]),
            helper.make_node("Conv", ["center_mask_f16", "arm_kernel"], ["arm_fill_score"], kernel_shape=[5, 5], pads=[2, 2, 2, 2]),
            helper.make_node("Greater", ["center_fill_score", "zero_f16"], ["center_fill"]),
            helper.make_node("Greater", ["arm_fill_score", "zero_f16"], ["arm_fill"]),
            helper.make_node("Where", ["center_mask", "color12", "zero_u8"], ["center_value"]),
            helper.make_node("Pad", ["color12", "pads_shift_1_0", "zero_u8"], ["up_color"], mode="constant"),
            helper.make_node("Where", ["center_mask", "up_color", "zero_u8"], ["arm_value"]),
            helper.make_node("ReduceMax", ["center_value"], ["center_color"], axes=[0, 1, 2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["arm_value"], ["arm_color"], axes=[0, 1, 2, 3], keepdims=1),
            helper.make_node("Where", ["arm_fill", "arm_color", "zero_u8"], ["arm_out"]),
            helper.make_node("Where", ["center_fill", "center_color", "arm_out"], ["color12_out"]),
            helper.make_node("Pad", ["color12_out", "pads_output", "outside_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )
    initializers.append(_int64_tensor("pads_shift_1_0", [0, 0, 1, 0, 0, 0, -1, 0], [8]))

    value_infos = []
    for slot in range(2):
        value_infos.extend(
            [
                helper.make_tensor_value_info(f"selected_{slot}_f32", onnx.TensorProto.FLOAT, [1, 1, SIZE, SIZE]),
                helper.make_tensor_value_info(f"selected_{slot}_bool", onnx.TensorProto.BOOL, [1, 1, SIZE, SIZE]),
                helper.make_tensor_value_info(f"selected_color_{slot}", onnx.TensorProto.UINT8, [1, 1, SIZE, SIZE]),
            ]
        )
    value_infos.extend(
        [
            helper.make_tensor_value_info("nonzero_f32", onnx.TensorProto.FLOAT, [1, 1, SIZE, SIZE]),
            helper.make_tensor_value_info("nonzero_bool", onnx.TensorProto.BOOL, [1, 1, SIZE, SIZE]),
            helper.make_tensor_value_info("color12", onnx.TensorProto.UINT8, [1, 1, SIZE, SIZE]),
        ]
    )

    graph = helper.make_graph(nodes, "task012_top2_cross_expand_conv_graph", [x], [y], initializers, value_info=value_infos)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
