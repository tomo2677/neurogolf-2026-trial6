from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


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


BLOCK_STARTS = (0, 4, 8)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int32_tensor("axes3", [1, 2, 3]),
        _int64_tensor("pads_output_chw", [0, 0, 0, 3, 19, 19]),
        _int64_tensor("pad_axes_chw", [1, 2, 3]),
        _int64_tensor("pads_pattern4_hw", [0, 0, 1, 1], [4]),
        _int64_tensor("pad_axes_hw", [2, 3], [2]),
        _int32_tensor("slice_channel_start", [0], [1]),
        _int32_tensor("slice_channel_end", [7], [1]),
        _int32_tensor("three_i32", [3], [1]),
        _int32_tensor("four_i32", [4], [1]),
        _int64_tensor("expand_index11", [0, 0, 0, 3, 1, 1, 1, 3, 2, 2, 2], [11]),
        _u8_tensor("colors7", list(range(7)), [1, 7, 1, 1]),
        _u8_tensor("five_u8", [5], [1]),
    ]
    for br, row in enumerate(BLOCK_STARTS):
        for bc, col in enumerate(BLOCK_STARTS):
            initializers.extend(
                [
                    _int64_tensor(f"block{br}{bc}_starts", [8, row, col], [3]),
                    _int64_tensor(f"block{br}{bc}_ends", [9, row + 3, col + 3], [3]),
                ]
            )
    nodes: list[onnx.NodeProto] = [
    ]
    has8_scalars: list[str] = []
    for br in range(3):
        for bc in range(3):
            key = f"block{br}{bc}"
            nodes.extend(
                [
                    helper.make_node("Slice", ["input", f"{key}_starts", f"{key}_ends", "pad_axes_chw"], [f"{key}_8"]),
                    helper.make_node("MaxPool", [f"{key}_8"], [f"{key}_has8_f32"], kernel_shape=[3, 3]),
                    helper.make_node("Cast", [f"{key}_has8_f32"], [f"{key}_has8_u8"], to=onnx.TensorProto.UINT8),
                ]
            )
            has8_scalars.append(f"{key}_has8_u8")
    nodes.extend(
        [
            helper.make_node("Concat", has8_scalars, ["has8_grid"], axis=2),
            helper.make_node("Flatten", ["has8_grid"], ["has8_flat"], axis=2),
            helper.make_node("ArgMin", ["has8_flat"], ["selected_index"], axis=1, keepdims=0),
            helper.make_node("Cast", ["selected_index"], ["selected_index_i32"], to=onnx.TensorProto.INT32),
            helper.make_node("Div", ["selected_index_i32", "three_i32"], ["row_block"]),
            helper.make_node("Mod", ["selected_index_i32", "three_i32"], ["col_block"]),
            helper.make_node("Mul", ["row_block", "four_i32"], ["row_start"]),
            helper.make_node("Mul", ["col_block", "four_i32"], ["col_start"]),
            helper.make_node("Add", ["row_start", "three_i32"], ["row_end"]),
            helper.make_node("Add", ["col_start", "three_i32"], ["col_end"]),
            helper.make_node("Concat", ["slice_channel_start", "row_start", "col_start"], ["selected_start"], axis=0),
            helper.make_node("Concat", ["slice_channel_end", "row_end", "col_end"], ["selected_end"], axis=0),
            helper.make_node("Slice", ["input", "selected_start", "selected_end", "axes3"], ["selected_onehot"]),
            helper.make_node("ArgMax", ["selected_onehot"], ["pattern_i64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["pattern_i64"], ["pattern"], to=onnx.TensorProto.UINT8),
            helper.make_node("Pad", ["pattern", "pads_pattern4_hw", "five_u8", "pad_axes_hw"], ["pattern4"], mode="constant"),
            helper.make_node("Gather", ["pattern4", "expand_index11"], ["expanded11x4"], axis=2),
            helper.make_node("Gather", ["expanded11x4", "expand_index11"], ["color11"], axis=3),
            helper.make_node("Equal", ["colors7", "color11"], ["output7"]),
            helper.make_node("Pad", ["output7", "pads_output_chw", "", "pad_axes_chw"], ["output"], mode="constant"),
        ]
    )

    value_infos = [
        helper.make_tensor_value_info("selected_onehot", onnx.TensorProto.FLOAT, [1, 7, 3, 3]),
        helper.make_tensor_value_info("pattern_i64", onnx.TensorProto.INT64, [1, 1, 3, 3]),
        helper.make_tensor_value_info("pattern", onnx.TensorProto.UINT8, [1, 1, 3, 3]),
        helper.make_tensor_value_info("pattern4", onnx.TensorProto.UINT8, [1, 1, 4, 4]),
        helper.make_tensor_value_info("expanded11x4", onnx.TensorProto.UINT8, [1, 1, 11, 4]),
        helper.make_tensor_value_info("color11", onnx.TensorProto.UINT8, [1, 1, 11, 11]),
        helper.make_tensor_value_info("output7", onnx.TensorProto.BOOL, [1, 7, 11, 11]),
    ]
    graph = helper.make_graph(nodes, "task011_bool7_pad_graph", [x], [y], initializers, value_info=value_infos)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
