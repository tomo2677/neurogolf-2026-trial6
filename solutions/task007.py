from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


POINTS_BY_REMAINDER = {
    1: [(0, 1), (0, 4), (1, 6), (4, 6)],
    2: [(0, 2), (0, 5), (2, 6), (5, 6)],
}


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT32, dims, values)


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    if dims is None:
        dims = [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _uint8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    remainder_index = np.fromfunction(lambda r, c: (r + c) % 3, (7, 7), dtype=int).astype(np.int64)

    initializers = [
        _int64_tensor("slice_axes3", [1, 2, 3], [3]),
        _int64_tensor("count_axes", [0, 2, 3], [3]),
        _int64_tensor("present_start", [1], [1]),
        _int64_tensor("present_end", [10], [1]),
        _int64_tensor("k3", [3], [1]),
        _int64_tensor("vector_shape", [1], [1]),
        _int64_tensor("axis0", [0], [1]),
        _int64_tensor("remainder_index", remainder_index.ravel().tolist(), [7, 7]),
        _uint8_tensor("channel_ids_u8", list(range(9)), [1, 9, 1, 1]),
        _uint8_tensor("zero_u8", [0], [1]),
        _int64_tensor("pads_output", [1, 0, 0, 0, 23, 23], [6]),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input", "count_axes"], ["color_counts10"], keepdims=0),
        helper.make_node("Slice", ["color_counts10", "present_start", "present_end"], ["present_counts"]),
        helper.make_node("TopK", ["present_counts", "k3"], ["top_counts", "top_indices"], axis=0, largest=1, sorted=1),
    ]

    known_u8_vector: dict[int, str] = {}
    for rem, points in POINTS_BY_REMAINDER.items():
        point_outputs: list[str] = []
        for idx, (row, col) in enumerate(points):
            start_name = f"p{rem}_{idx}_start"
            end_name = f"p{rem}_{idx}_end"
            point_name = f"p{rem}_{idx}"
            initializers.extend(
                [
                    _int64_tensor(start_name, [1, row, col], [3]),
                    _int64_tensor(end_name, [10, row + 1, col + 1], [3]),
                ]
            )
            nodes.append(helper.make_node("Slice", ["input", start_name, end_name, "slice_axes3"], [point_name]))
            point_outputs.append(point_name)
        group_name = f"rem{rem}_onehot"
        nodes.append(helper.make_node("Max", point_outputs, [group_name]))
        argmax_name = f"rem{rem}_color_i64_111"
        u8_111_name = f"rem{rem}_color_u8_111"
        u8_vector_name = f"rem{rem}_color_u8"
        nodes.extend(
            [
                helper.make_node("ArgMax", [group_name], [argmax_name], axis=1, keepdims=0, select_last_index=0),
                helper.make_node("Cast", [argmax_name], [u8_111_name], to=onnx.TensorProto.UINT8),
                helper.make_node("Reshape", [u8_111_name, "vector_shape"], [u8_vector_name]),
            ]
        )
        known_u8_vector[rem] = u8_vector_name

    nodes.extend(
        [
            helper.make_node("Cast", ["top_indices"], ["top_indices_u8"], to=onnx.TensorProto.UINT8),
            helper.make_node("Equal", ["top_indices_u8", known_u8_vector[1]], ["top_is_rem1"]),
            helper.make_node("Equal", ["top_indices_u8", known_u8_vector[2]], ["top_is_rem2"]),
            helper.make_node("Or", ["top_is_rem1", "top_is_rem2"], ["top_is_known"]),
            helper.make_node("Not", ["top_is_known"], ["top_is_rem0"]),
            helper.make_node("Where", ["top_is_rem0", "top_indices_u8", "zero_u8"], ["rem0_terms"]),
            helper.make_node("ReduceMax", ["rem0_terms", "axis0"], ["rem0_color_u8"], keepdims=1),
            helper.make_node(
                "Concat",
                ["rem0_color_u8", known_u8_vector[1], known_u8_vector[2]],
                ["rem_color_u8"],
                axis=0,
            ),
            helper.make_node("Gather", ["rem_color_u8", "remainder_index"], ["selected_grid_u8"], axis=0),
            helper.make_node("Equal", ["channel_ids_u8", "selected_grid_u8"], ["output9"]),
            helper.make_node("Pad", ["output9", "pads_output", "", "slice_axes3"], ["output"], mode="constant"),
        ]
    )

    graph = helper.make_graph(nodes, "task007_two_residue_remaining", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
