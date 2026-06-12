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
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _u8_tensor("outside_u8", [255], [1]),
        _u8_tensor("sep_row", [5] * 9, [1, 1, 1, 9]),
        _u8_tensor("sep_col", [5] * 11, [1, 1, 11, 1]),
    ]

    nodes: list[onnx.NodeProto] = []
    pattern = "zero_u8"
    for br, row in enumerate(STARTS):
        for bc, col in enumerate(STARTS):
            name = f"b{br}{bc}"
            initializers.extend(
                [
                    _int64_tensor(f"{name}_start", [0, row, col], [3]),
                    _int64_tensor(f"{name}_end", [10, row + 3, col + 3], [3]),
                    _int64_tensor(f"{name}_blue_start", [8, row, col], [3]),
                    _int64_tensor(f"{name}_blue_end", [9, row + 3, col + 3], [3]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Slice", ["input", f"{name}_blue_start", f"{name}_blue_end", "axes3"], [f"{name}_blue"]),
                    helper.make_node("ReduceMax", [f"{name}_blue"], [f"{name}_has8_f32"], axes=[0, 1, 2, 3], keepdims=0),
                    helper.make_node("Cast", [f"{name}_has8_f32"], [f"{name}_has8_u8"], to=onnx.TensorProto.UINT8),
                    helper.make_node("Equal", [f"{name}_has8_u8", "zero_u8"], [f"{name}_no8"]),
                    helper.make_node("Slice", ["input", f"{name}_start", f"{name}_end", "axes3"], [f"{name}_onehot"]),
                    helper.make_node("ArgMax", [f"{name}_onehot"], [f"{name}_color_i64"], axis=1, keepdims=1),
                    helper.make_node("Cast", [f"{name}_color_i64"], [f"{name}_color_u8"], to=onnx.TensorProto.UINT8),
                    helper.make_node("Where", [f"{name}_no8", f"{name}_color_u8", pattern], [f"pattern_{name}"]),
                ]
            )
            pattern = f"pattern_{name}"

    nodes.extend(
        [
            helper.make_node(
                "Resize",
                [pattern, "", "", "resize_sizes"],
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

    graph = helper.make_graph(nodes, "task011_missing_block_expand_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
