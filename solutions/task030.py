from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 10


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _shift_mask(nodes: list[onnx.NodeProto], source: str, delta: str, prefix: str) -> str:
    nodes.extend(
        [
            helper.make_node("Cast", [delta], [f"{prefix}_delta_i32"], to=onnx.TensorProto.INT32),
            helper.make_node("Add", ["row_grid_i32", f"{prefix}_delta_i32"], [f"{prefix}_src_r"]),
            helper.make_node("GreaterOrEqual", [f"{prefix}_src_r", "zero_i32"], [f"{prefix}_r_ge_zero"]),
            helper.make_node("Less", [f"{prefix}_src_r", "size_i32"], [f"{prefix}_r_lt_size"]),
            helper.make_node("And", [f"{prefix}_r_ge_zero", f"{prefix}_r_lt_size"], [f"{prefix}_in_bounds"]),
            helper.make_node("Where", [f"{prefix}_in_bounds", f"{prefix}_src_r", "zero_i32"], [f"{prefix}_safe_r"]),
            helper.make_node("Mul", [f"{prefix}_safe_r", "size_i32"], [f"{prefix}_safe_r_offset"]),
            helper.make_node("Add", [f"{prefix}_safe_r_offset", "col_grid_i32"], [f"{prefix}_safe_spatial"]),
            helper.make_node("Cast", [f"{prefix}_safe_spatial"], [f"{prefix}_indices"], to=onnx.TensorProto.INT64),
            helper.make_node("Reshape", [source, "shape_flat100"], [f"{prefix}_source_flat"]),
            helper.make_node("Gather", [f"{prefix}_source_flat", f"{prefix}_indices"], [f"{prefix}_shifted_raw"], axis=0),
            helper.make_node("Where", [f"{prefix}_in_bounds", f"{prefix}_shifted_raw", "zero_u8"], [f"{prefix}_shifted"]),
        ]
    )
    return f"{prefix}_shifted"


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("slice_hw_starts", [0, 0], [2]),
        _int64_tensor("slice_hw_ends", [SIZE, SIZE], [2]),
        _int64_tensor("slice_hw_axes", [2, 3], [2]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("two_i64", [2], [1]),
        _int64_tensor("four_i64", [4], [1]),
        _int32_tensor("zero_i32", [0], [1]),
        _int32_tensor("size_i32", [SIZE], [1]),
        _int32_tensor("row_grid_i32", [r for r in range(SIZE) for _ in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int32_tensor("col_grid_i32", [c for _ in range(SIZE) for c in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int64_tensor("shape_flat100", [SIZE * SIZE], [1]),
        _int64_tensor("pads_color10_to30", [0, 0, 0, 0, 0, 0, 30 - SIZE, 30 - SIZE], [8]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("four_u8", [4], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "slice_hw_starts", "slice_hw_ends", "slice_hw_axes"], ["input10"]),
        helper.make_node("ArgMax", ["input10"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Equal", ["input_color_i64", "one_i64"], ["c1_bool"]),
        helper.make_node("Equal", ["input_color_i64", "two_i64"], ["c2_bool"]),
        helper.make_node("Equal", ["input_color_i64", "four_i64"], ["c4_bool"]),
        helper.make_node("Cast", ["c1_bool"], ["c1_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["c2_bool"], ["c2_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["c4_bool"], ["c4_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["c1_u8"], ["c1_ref_row_score"], axes=[3], keepdims=1),
        helper.make_node("ArgMax", ["c1_ref_row_score"], ["ref_top"], axis=2, keepdims=1),
    ]

    shifted_masks: list[tuple[str, str]] = [("c1_u8", "one_u8")]
    for color_name, mask_name, color_value in (("c2", "c2_u8", "two_u8"), ("c4", "c4_u8", "four_u8")):
        nodes.extend(
            [
                helper.make_node("ReduceMax", [mask_name], [f"{color_name}_row_score"], axes=[3], keepdims=1),
                helper.make_node("ArgMax", [f"{color_name}_row_score"], [f"{color_name}_top"], axis=2, keepdims=1),
                helper.make_node("Sub", [f"{color_name}_top", "ref_top"], [f"{color_name}_delta"]),
            ]
        )
        shifted = _shift_mask(nodes, mask_name, f"{color_name}_delta", color_name)
        shifted_masks.append((shifted, color_value))

    color_terms: list[str] = ["c1_u8"]
    for shifted, color_value in shifted_masks:
        if shifted == "c1_u8":
            continue
        prefix = shifted.removesuffix("_shifted")
        nodes.extend(
            [
                helper.make_node("Greater", [shifted, "zero_u8"], [f"{prefix}_placed_bool"]),
                helper.make_node("Where", [f"{prefix}_placed_bool", color_value, "zero_u8"], [f"{prefix}_color"]),
            ]
        )
        color_terms.append(f"{prefix}_color")

    nodes.extend(
        [
            helper.make_node("Max", color_terms, ["placed_color"]),
            helper.make_node("ReduceMax", ["input10"], ["cell_present_f32"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["cell_present_f32", "zero_f32"], ["valid_area"]),
            helper.make_node("Where", ["valid_area", "placed_color", "invalid_u8"], ["color10"]),
            helper.make_node("Pad", ["color10", "pads_color10_to30", "invalid_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task030_window10_vertical_align_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
