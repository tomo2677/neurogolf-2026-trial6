from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
FLOOD_STEPS = SIZE * SIZE


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("black_starts", [0, 0, 0, 0], [4]),
        _int64_tensor("black_ends", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("green_starts", [0, 3, 0, 0], [4]),
        _int64_tensor("green_ends", [1, 4, SIZE, SIZE], [4]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("one_f32", [1.0], [1]),
        _f32_tensor("row_idx", [float(v) for v in range(SIZE)], [1, 1, SIZE, 1]),
        _f32_tensor("col_idx", [float(v) for v in range(SIZE)], [1, 1, 1, SIZE]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("green_u8", [3], [1]),
        _u8_tensor("yellow_u8", [4], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["cell_present"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["cell_present"], ["row_present_f32"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["cell_present"], ["col_present_f32"], axes=[2], keepdims=1),
        helper.make_node("Greater", ["row_present_f32", "zero_f32"], ["row_present"]),
        helper.make_node("Greater", ["col_present_f32", "zero_f32"], ["col_present"]),
        helper.make_node("And", ["row_present", "col_present"], ["valid_area"]),
        helper.make_node("ReduceSum", ["row_present_f32"], ["row_count"], axes=[0, 1, 2, 3], keepdims=1),
        helper.make_node("ReduceSum", ["col_present_f32"], ["col_count"], axes=[0, 1, 2, 3], keepdims=1),
        helper.make_node("Sub", ["row_count", "one_f32"], ["last_row"]),
        helper.make_node("Sub", ["col_count", "one_f32"], ["last_col"]),
        helper.make_node("Equal", ["row_idx", "zero_f32"], ["first_row"]),
        helper.make_node("Equal", ["col_idx", "zero_f32"], ["first_col"]),
        helper.make_node("Equal", ["row_idx", "last_row"], ["last_row_mask"]),
        helper.make_node("Equal", ["col_idx", "last_col"], ["last_col_mask"]),
        helper.make_node("Or", ["first_row", "last_row_mask"], ["row_border"]),
        helper.make_node("Or", ["first_col", "last_col_mask"], ["col_border"]),
        helper.make_node("Or", ["row_border", "col_border"], ["border_raw"]),
        helper.make_node("And", ["border_raw", "valid_area"], ["border"]),
        helper.make_node("Slice", ["input", "black_starts", "black_ends"], ["black_f32"]),
        helper.make_node("Slice", ["input", "green_starts", "green_ends"], ["green_f32"]),
        helper.make_node("Greater", ["black_f32", "zero_f32"], ["black_bool"]),
        helper.make_node("Greater", ["green_f32", "zero_f32"], ["green_bool"]),
        helper.make_node("And", ["black_bool", "border"], ["seed_bool"]),
        helper.make_node("Cast", ["seed_bool"], ["external_0"], to=onnx.TensorProto.UINT8),
    ]

    external = "external_0"
    for step in range(FLOOD_STEPS):
        nodes.extend(
            [
                helper.make_node(
                    "MaxPool",
                    [external],
                    [f"external_h_{step}"],
                    kernel_shape=[1, 3],
                    pads=[0, 1, 0, 1],
                ),
                helper.make_node(
                    "MaxPool",
                    [external],
                    [f"external_v_{step}"],
                    kernel_shape=[3, 1],
                    pads=[1, 0, 1, 0],
                ),
                helper.make_node("Max", [f"external_h_{step}", f"external_v_{step}"], [f"external_dilated_{step}"]),
                helper.make_node("Where", ["black_bool", f"external_dilated_{step}", "zero_u8"], [f"external_{step + 1}"]),
            ]
        )
        external = f"external_{step + 1}"

    nodes.extend(
        [
            helper.make_node("Cast", [external], ["external_bool"], to=onnx.TensorProto.BOOL),
            helper.make_node("Where", ["external_bool", "zero_u8", "yellow_u8"], ["fill_color"]),
            helper.make_node("Where", ["green_bool", "green_u8", "fill_color"], ["valid_color"]),
            helper.make_node("Where", ["valid_area", "valid_color", "invalid_u8"], ["color30"]),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task002_exact_30x30_flood_fill_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
