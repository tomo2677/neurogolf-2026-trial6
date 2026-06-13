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
            helper.make_node("Reshape", [f"{prefix}_safe_r", "shape_vec10"], [f"{prefix}_safe_r_vec"]),
            helper.make_node("Gather", [source, f"{prefix}_safe_r_vec"], [f"{prefix}_shifted_raw"], axis=2),
            helper.make_node("Where", [f"{prefix}_in_bounds", f"{prefix}_shifted_raw", "zero_u8"], [f"{prefix}_shifted"]),
        ]
    )
    return f"{prefix}_shifted"


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("h10_starts", [0, 9, 0], [3]),
        _int64_tensor("h10_ends", [1, 10, 1], [3]),
        _int64_tensor("c1_starts", [1, 0, 0], [3]),
        _int64_tensor("c1_ends", [2, SIZE, SIZE], [3]),
        _int64_tensor("c2_starts", [2, 0, 0], [3]),
        _int64_tensor("c2_ends", [3, SIZE, SIZE], [3]),
        _int64_tensor("c4_starts", [4, 0, 0], [3]),
        _int64_tensor("c4_ends", [5, SIZE, SIZE], [3]),
        _int64_tensor("axes_chw", [1, 2, 3], [3]),
        _int64_tensor("axis_col", [3], [1]),
        _int32_tensor("zero_i32", [0], [1]),
        _int32_tensor("size_i32", [SIZE], [1]),
        _int32_tensor("row_grid_i32", list(range(SIZE)), [1, 1, SIZE, 1]),
        _int64_tensor("shape_vec10", [SIZE], [1]),
        _int64_tensor("pads_output", [0, 0, 0, 5, 30 - SIZE, 30 - SIZE], [6]),
        _u8_tensor("zero_u8", [0], [1]),
        helper.make_tensor("true_bool", onnx.TensorProto.BOOL, [1, 1, 1, 1], [1]),
        _u8_tensor("two_u8", [2], [1]),
        _u8_tensor("four_u8", [4], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors5_u8", list(range(5)), [1, 5, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "h10_starts", "h10_ends", "axes_chw"], ["h10_f32"]),
        helper.make_node("Slice", ["input", "c1_starts", "c1_ends", "axes_chw"], ["c1_f32"]),
        helper.make_node("Slice", ["input", "c2_starts", "c2_ends", "axes_chw"], ["c2_f32"]),
        helper.make_node("Slice", ["input", "c4_starts", "c4_ends", "axes_chw"], ["c4_f32"]),
        helper.make_node("Cast", ["h10_f32"], ["h10_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["c1_f32"], ["c1_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["c2_f32"], ["c2_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["c4_f32"], ["c4_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["c1_u8", "axis_col"], ["c1_ref_row_score"], keepdims=1),
        helper.make_node("ArgMax", ["c1_ref_row_score"], ["ref_top"], axis=2, keepdims=1),
    ]

    shifted_masks: list[tuple[str, str]] = []
    for color_name, mask_name, color_value in (("c2", "c2_u8", "two_u8"), ("c4", "c4_u8", "four_u8")):
        nodes.extend(
            [
                helper.make_node("ReduceMax", [mask_name, "axis_col"], [f"{color_name}_row_score"], keepdims=1),
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
                helper.make_node("Mul", [shifted, color_value], [f"{prefix}_color"]),
            ]
        )
        color_terms.append(f"{prefix}_color")

    nodes.extend(
        [
            helper.make_node("Max", color_terms, ["placed_color"]),
            helper.make_node(
                "Concat",
                [
                    "true_bool",
                    "true_bool",
                    "true_bool",
                    "true_bool",
                    "true_bool",
                    "h10_bool",
                    "h10_bool",
                    "h10_bool",
                    "h10_bool",
                    "h10_bool",
                ],
                ["valid_rows"],
                axis=2,
            ),
            helper.make_node("Where", ["valid_rows", "placed_color", "invalid_u8"], ["color10"]),
            helper.make_node("Equal", ["colors5_u8", "color10"], ["output5"]),
            helper.make_node("Pad", ["output5", "pads_output", "", "axes_chw"], ["output"], mode="constant"),
        ]
    )

    graph = helper.make_graph(nodes, "task030_colors5_bool_pad_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
