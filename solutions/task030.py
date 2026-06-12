from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _shift_color(nodes: list[onnx.NodeProto], color: int) -> str:
    prefix = f"c{color}"
    nodes.extend(
        [
            helper.make_node("Slice", ["input", f"{prefix}_starts", f"{prefix}_ends"], [f"{prefix}_ch"]),
            helper.make_node("ReduceMax", [f"{prefix}_ch"], [f"{prefix}_row_score"], axes=[3], keepdims=1),
            helper.make_node("ArgMax", [f"{prefix}_row_score"], [f"{prefix}_top"], axis=2, keepdims=1),
            helper.make_node("Sub", [f"{prefix}_top", "ref_top"], [f"{prefix}_delta"]),
            helper.make_node("Add", ["row_grid_i64", f"{prefix}_delta"], [f"{prefix}_src_r"]),
            helper.make_node("GreaterOrEqual", [f"{prefix}_src_r", "zero_i64"], [f"{prefix}_r_ge_zero"]),
            helper.make_node("Less", [f"{prefix}_src_r", "size_i64"], [f"{prefix}_r_lt_size"]),
            helper.make_node("And", [f"{prefix}_r_ge_zero", f"{prefix}_r_lt_size"], [f"{prefix}_in_bounds"]),
            helper.make_node("Where", [f"{prefix}_in_bounds", f"{prefix}_src_r", "zero_i64"], [f"{prefix}_safe_r"]),
            helper.make_node("Mul", [f"{prefix}_safe_r", "width_i64"], [f"{prefix}_safe_r_offset"]),
            helper.make_node("Add", [f"{prefix}_safe_r_offset", "col_grid_i64"], [f"{prefix}_safe_spatial"]),
            helper.make_node("Reshape", [f"{prefix}_safe_spatial", "shape_index_1x900"], [f"{prefix}_safe_spatial_flat"]),
            helper.make_node("Reshape", [f"{prefix}_ch", "shape_flat_1x900"], [f"{prefix}_source_flat"]),
            helper.make_node("GatherElements", [f"{prefix}_source_flat", f"{prefix}_safe_spatial_flat"], [f"{prefix}_shifted_flat"], axis=2),
            helper.make_node("Reshape", [f"{prefix}_shifted_flat", "shape_1x1x30x30"], [f"{prefix}_shifted_raw"]),
            helper.make_node("Cast", [f"{prefix}_in_bounds"], [f"{prefix}_in_bounds_f32"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Mul", [f"{prefix}_shifted_raw", f"{prefix}_in_bounds_f32"], [f"{prefix}_shifted"]),
        ]
    )
    return f"{prefix}_shifted"


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("zero_i64", [0], [1]),
        _int64_tensor("size_i64", [SIZE], [1]),
        _int64_tensor("width_i64", [SIZE], [1]),
        _int64_tensor("row_grid_i64", [r for r in range(SIZE) for _ in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int64_tensor("col_grid_i64", [c for _ in range(SIZE) for c in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int64_tensor("shape_index_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_flat_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_1x1x30x30", [1, 1, SIZE, SIZE], [4]),
        _f32_tensor("zero_f32", [0.0], [1]),
    ]
    for color in (1, 2, 4):
        initializers.extend(
            [
                _int64_tensor(f"c{color}_starts", [0, color, 0, 0], [4]),
                _int64_tensor(f"c{color}_ends", [1, color + 1, SIZE, SIZE], [4]),
            ]
        )

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "c1_starts", "c1_ends"], ["c1_ref"]),
        helper.make_node("ReduceMax", ["c1_ref"], ["c1_ref_row_score"], axes=[3], keepdims=1),
        helper.make_node("ArgMax", ["c1_ref_row_score"], ["ref_top"], axis=2, keepdims=1),
    ]
    shifted1 = _shift_color(nodes, 1)
    shifted2 = _shift_color(nodes, 2)
    shifted4 = _shift_color(nodes, 4)
    nodes.extend(
        [
            helper.make_node("Greater", [shifted1, "zero_f32"], ["c1_bool"]),
            helper.make_node("Greater", [shifted2, "zero_f32"], ["c2_bool"]),
            helper.make_node("Greater", [shifted4, "zero_f32"], ["c4_bool"]),
            helper.make_node("Or", ["c1_bool", "c2_bool"], ["placed12"]),
            helper.make_node("Or", ["placed12", "c4_bool"], ["placed_any"]),
            helper.make_node("ReduceMax", ["input"], ["cell_present_f32"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["cell_present_f32", "zero_f32"], ["valid_area"]),
            helper.make_node("Not", ["placed_any"], ["not_placed"]),
            helper.make_node("And", ["valid_area", "not_placed"], ["black_bool"]),
            helper.make_node("Cast", ["black_bool"], ["black"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Mul", ["black", "zero_f32"], ["zero_channel"]),
            helper.make_node(
                "Concat",
                [
                    "black",
                    shifted1,
                    shifted2,
                    "zero_channel",
                    shifted4,
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                ],
                ["output"],
                axis=1,
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task030_vertical_align_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
