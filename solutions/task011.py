from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


STARTS = [0, 4, 8]


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("axes3", [1, 2, 3]),
        _int64_tensor("resize_sizes", [1, 1, 9, 9]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 19, 19]),
        _int64_tensor("channel8_starts", [0, 8, 0, 0], [4]),
        _int64_tensor("channel8_ends", [1, 9, 11, 11], [4]),
        _int64_tensor("slice_channel_start", [0], [1]),
        _int64_tensor("slice_channel_end", [7], [1]),
        _int64_tensor("shape_1x9", [1, 9], [2]),
        _int64_tensor("row_start_table", [0, 0, 0, 4, 4, 4, 8, 8, 8], [9]),
        _int64_tensor("col_start_table", [0, 4, 8, 0, 4, 8, 0, 4, 8], [9]),
        _int64_tensor("row_end_table", [3, 3, 3, 7, 7, 7, 11, 11, 11], [9]),
        _int64_tensor("col_end_table", [3, 7, 11, 3, 7, 11, 3, 7, 11], [9]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _u8_tensor("outside_u8", [255], [1]),
        _u8_tensor("sep_row", [5] * 9, [1, 1, 1, 9]),
        _u8_tensor("sep_col", [5] * 11, [1, 1, 11, 1]),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "channel8_starts", "channel8_ends"], ["blue11"]),
        helper.make_node("MaxPool", ["blue11"], ["has8_grid"], kernel_shape=[3, 3], strides=[4, 4]),
        helper.make_node("Cast", ["has8_grid"], ["has8_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Not", ["has8_bool"], ["no8_bool"]),
        helper.make_node("Cast", ["no8_bool"], ["no8_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Reshape", ["no8_u8", "shape_1x9"], ["no8_flat"]),
        helper.make_node("ArgMax", ["no8_flat"], ["selected_index"], axis=1, keepdims=0),
        helper.make_node("Gather", ["row_start_table", "selected_index"], ["row_start"], axis=0),
        helper.make_node("Gather", ["col_start_table", "selected_index"], ["col_start"], axis=0),
        helper.make_node("Gather", ["row_end_table", "selected_index"], ["row_end"], axis=0),
        helper.make_node("Gather", ["col_end_table", "selected_index"], ["col_end"], axis=0),
    ]
    row_start = "row_start"
    col_start = "col_start"
    row_end = "row_end"
    col_end = "col_end"

    nodes.extend(
        [
            helper.make_node("Concat", ["slice_channel_start", row_start, col_start], ["selected_start"], axis=0),
            helper.make_node("Concat", ["slice_channel_end", row_end, col_end], ["selected_end"], axis=0),
            helper.make_node("Slice", ["input", "selected_start", "selected_end", "axes3"], ["selected_onehot"]),
            helper.make_node("ArgMax", ["selected_onehot"], ["pattern_i64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["pattern_i64"], ["pattern"], to=onnx.TensorProto.UINT8),
            helper.make_node(
                "Resize",
                ["pattern", "", "", "resize_sizes"],
                ["expanded9"],
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            ),
            helper.make_node(
                "Split",
                ["expanded9"],
                ["row_block0", "row_block1", "row_block2"],
                axis=2,
            ),
            helper.make_node(
                "Concat",
                ["row_block0", "sep_row", "row_block1", "sep_row", "row_block2"],
                ["expanded11x9"],
                axis=2,
            ),
            helper.make_node(
                "Split",
                ["expanded11x9"],
                ["col_block0", "col_block1", "col_block2"],
                axis=3,
            ),
            helper.make_node(
                "Concat",
                ["col_block0", "sep_col", "col_block1", "sep_col", "col_block2"],
                ["color11"],
                axis=3,
            ),
            helper.make_node("Pad", ["color11", "pads_output", "outside_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    value_infos = [
        helper.make_tensor_value_info("selected_onehot", onnx.TensorProto.FLOAT, [1, 7, 3, 3]),
        helper.make_tensor_value_info("pattern_i64", onnx.TensorProto.INT64, [1, 1, 3, 3]),
        helper.make_tensor_value_info("pattern", onnx.TensorProto.UINT8, [1, 1, 3, 3]),
    ]
    graph = helper.make_graph(nodes, "task011_dynamic_slice_block_graph", [x], [y], initializers, value_info=value_infos)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
