from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("present_start", [1]),
        _int64_tensor("present_end", [10]),
        _int64_tensor("row_idx", list(range(16)), [1, 1, 16, 1]),
        _int64_tensor("col_idx", list(range(16)), [1, 1, 1, 16]),
        _int64_tensor("k3", [3], [1]),
        _int32_tensor("slice_zero", [0], [1]),
        _int32_tensor("one_i32", [1], [1]),
        _int32_tensor("slice_end_delta", [1, 16, 16], [3]),
        _int32_tensor("slice_axes3", [1, 2, 3], [3]),
        _int64_tensor("pads_shift", [0, 0, 0, 1, 0, 0, 0, -1]),
        _int64_tensor("pads_color", [0, 0, 0, 0, 0, 0, 14, 14]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["row_present_full"], axes=[1, 3], keepdims=1),
        helper.make_node("ReduceMax", ["input"], ["col_present_full"], axes=[1, 2], keepdims=1),
        helper.make_node("ArgMax", ["row_present_full"], ["last_row"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present_full"], ["last_col"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("LessOrEqual", ["row_idx", "last_row"], ["row_valid"]),
        helper.make_node("LessOrEqual", ["col_idx", "last_col"], ["col_valid"]),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_bool"]),
        helper.make_node("ReduceMax", ["input"], ["present10"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("Cast", ["present10"], ["present10_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["present10_u8", "present_start", "present_end"], ["present9"]),
        helper.make_node("TopK", ["present9", "k3"], ["top_values", "top_idx3"], axis=0, largest=1, sorted=0),
        helper.make_node("Cast", ["top_idx3"], ["top_idx3_i32"], to=onnx.TensorProto.INT32),
        helper.make_node("Add", ["top_idx3_i32", "one_i32"], ["channel_ids3"]),
        helper.make_node("Split", ["channel_ids3"], ["channel_id_0", "channel_id_1", "channel_id_2"], axis=0, split=[1, 1, 1]),
    ]

    color_slots: list[str] = []
    for slot in range(3):
        selected = f"selected_{slot}"
        color_slots.append(f"slot_color_{slot}")
        nodes.extend(
            [
                helper.make_node("Concat", [f"channel_id_{slot}", "slice_zero", "slice_zero"], [f"selected_start_{slot}"], axis=0),
                helper.make_node("Add", [f"selected_start_{slot}", "slice_end_delta"], [f"selected_end_{slot}"]),
                helper.make_node("Slice", ["input", f"selected_start_{slot}", f"selected_end_{slot}", "slice_axes3"], [f"{selected}_f32"]),
                helper.make_node("Cast", [f"{selected}_f32"], [f"{selected}_bool"], to=onnx.TensorProto.BOOL),
                helper.make_node("Cast", [f"channel_id_{slot}"], [f"slot_color_id_{slot}"], to=onnx.TensorProto.UINT8),
                helper.make_node("Where", [f"{selected}_bool", f"slot_color_id_{slot}", "zero_u8"], [f"selected_color_{slot}"]),
                helper.make_node("ReduceMax", [f"selected_color_{slot}"], [f"row_present_{slot}"], axes=[3], keepdims=1),
                helper.make_node("ReduceMax", [f"selected_color_{slot}"], [f"col_present_{slot}"], axes=[2], keepdims=1),
                helper.make_node("ArgMax", [f"row_present_{slot}"], [f"bottom_idx_{slot}"], axis=2, keepdims=1, select_last_index=1),
                helper.make_node("ArgMax", [f"col_present_{slot}"], [f"right_idx_{slot}"], axis=3, keepdims=1, select_last_index=1),
                helper.make_node("Equal", ["row_idx", f"bottom_idx_{slot}"], [f"bottom_vec_{slot}"]),
                helper.make_node("Equal", ["col_idx", f"right_idx_{slot}"], [f"right_vec_{slot}"]),
                helper.make_node("Or", [f"bottom_vec_{slot}", f"right_vec_{slot}"], [f"keep_mask_{slot}"]),
                helper.make_node("Where", [f"keep_mask_{slot}", f"selected_color_{slot}", "zero_u8"], [f"kept_{slot}"]),
                helper.make_node("Where", [f"keep_mask_{slot}", "zero_u8", f"selected_color_{slot}"], [f"shift_source_{slot}"]),
                helper.make_node("Pad", [f"shift_source_{slot}", "pads_shift"], [f"shifted_{slot}"], mode="constant"),
                helper.make_node("Max", [f"kept_{slot}", f"shifted_{slot}"], [f"slot_color_{slot}"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Max", color_slots, ["nonblack_color_u8"]),
            helper.make_node("Where", ["valid_bool", "nonblack_color_u8", "invalid_u8"], ["color16"]),
            helper.make_node("Pad", ["color16", "pads_color", "invalid_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    value_infos = [
        helper.make_tensor_value_info("row_present_full", onnx.TensorProto.FLOAT, [1, 1, 30, 1]),
        helper.make_tensor_value_info("col_present_full", onnx.TensorProto.FLOAT, [1, 1, 1, 30]),
    ]
    for slot in range(3):
        value_infos.append(helper.make_tensor_value_info(f"selected_{slot}_f32", onnx.TensorProto.FLOAT, [1, 1, 16, 16]))

    graph = helper.make_graph(nodes, "task004_rect_valid_float_rowcol_graph", [x], [y], initializers, value_info=value_infos)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
