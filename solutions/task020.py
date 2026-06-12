from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _shift(nodes: list[onnx.NodeProto], source: str, dr: str, dc: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Sub", ["zero_i64", dr], [f"{output}_neg_dr"]),
            helper.make_node("Sub", ["zero_i64", dc], [f"{output}_neg_dc"]),
            helper.make_node(
                "Concat",
                ["zero_pad", "zero_pad", dr, dc, "zero_pad", "zero_pad", f"{output}_neg_dr", f"{output}_neg_dc"],
                [f"{output}_pads"],
                axis=0,
            ),
            helper.make_node("Pad", [source, f"{output}_pads", "zero_f32"], [output], mode="constant"),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    valid10 = [1.0 if r < 10 and c < 10 else 0.0 for r in range(30) for c in range(30)]
    initializers = [
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("two_i64", [2], [1]),
        _int64_tensor("zero_i64", [0], [1]),
        _int64_tensor("last_i64", [29], [1]),
        _int64_tensor("zero_pad", [0], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("nonzero_start", [1], [1]),
        _int64_tensor("nonzero_end", [10], [1]),
        _int64_tensor("axis_channel", [1], [1]),
        _int64_tensor("reverse_idx", list(reversed(range(30))), [30]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("valid10", valid10, [1, 1, 30, 30]),
        _f32_tensor("black_pixel", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "nonzero_start", "nonzero_end", "axis_channel"], ["input_nonzero"]),
        helper.make_node("ReduceMax", ["input_nonzero"], ["nonzero_score"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["nonzero_score", "zero_f32"], ["nonzero_bool"]),
        helper.make_node("Where", ["nonzero_bool", "input", "zero_f32"], ["colored0"]),
        helper.make_node("ReduceMax", ["nonzero_score"], ["row_present"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["nonzero_score"], ["col_present"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_min_raw"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["row_present"], ["r_max_raw"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present"], ["c_min_raw"], axis=3, keepdims=1),
        helper.make_node("ArgMax", ["col_present"], ["c_max_raw"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Add", ["r_min_raw", "r_max_raw"], ["r_sum"]),
        helper.make_node("Add", ["c_min_raw", "c_max_raw"], ["c_sum"]),
        helper.make_node("Div", ["r_sum", "two_i64"], ["cr_raw"]),
        helper.make_node("Div", ["c_sum", "two_i64"], ["cc_raw"]),
        helper.make_node("Reshape", ["cr_raw", "shape1"], ["cr"]),
        helper.make_node("Reshape", ["cc_raw", "shape1"], ["cc"]),
        helper.make_node("Transpose", ["colored0"], ["transposed"], perm=[0, 1, 3, 2]),
        helper.make_node("Gather", ["transposed", "reverse_idx"], ["trans_hflip"], axis=3),
        helper.make_node("Gather", ["colored0", "reverse_idx"], ["hflip"], axis=3),
        helper.make_node("Gather", ["hflip", "reverse_idx"], ["hvflip"], axis=2),
        helper.make_node("Gather", ["transposed", "reverse_idx"], ["trans_vflip"], axis=2),
        helper.make_node("Sub", ["cr", "cc"], ["rot90_dr"]),
        helper.make_node("Add", ["cr", "cc"], ["center_sum"]),
        helper.make_node("Sub", ["center_sum", "last_i64"], ["rot90_dc"]),
        helper.make_node("Add", ["cr", "cr"], ["cr2"]),
        helper.make_node("Add", ["cc", "cc"], ["cc2"]),
        helper.make_node("Sub", ["cr2", "last_i64"], ["rot180_dr"]),
        helper.make_node("Sub", ["cc2", "last_i64"], ["rot180_dc"]),
        helper.make_node("Identity", ["rot90_dc"], ["rot270_dr"]),
        helper.make_node("Sub", ["cc", "cr"], ["rot270_dc"]),
    ]

    _shift(nodes, "trans_hflip", "rot90_dr", "rot90_dc", "rot90")
    _shift(nodes, "hvflip", "rot180_dr", "rot180_dc", "rot180")
    _shift(nodes, "trans_vflip", "rot270_dr", "rot270_dc", "rot270")

    nodes.extend(
        [
            helper.make_node("Max", ["colored0", "rot90", "rot180", "rot270"], ["placed_nonzero"]),
            helper.make_node("ReduceMax", ["placed_nonzero"], ["placed_mask"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["placed_mask", "zero_f32"], ["placed_bool"]),
            helper.make_node("Greater", ["valid10", "zero_f32"], ["valid10_bool"]),
            helper.make_node("Not", ["placed_bool"], ["not_placed"]),
            helper.make_node("And", ["valid10_bool", "not_placed"], ["blank_bool"]),
            helper.make_node("Where", ["blank_bool", "black_pixel", "zero_f32"], ["blank_output"]),
            helper.make_node("Max", ["placed_nonzero", "blank_output"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task020_quarter_turn_completion_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
