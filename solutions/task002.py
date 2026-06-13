from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
LINE_CLOSURE_STEPS = 10


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _horizontal_closure(nodes: list[onnx.NodeProto], seed: str, prefix: str) -> str:
    nodes.extend(
        [
            helper.make_node("CumSum", [seed, "axis_w"], [f"{prefix}_cum"]),
            helper.make_node("Where", ["green_bool", f"{prefix}_cum", "zero_f32"], [f"{prefix}_green_cum"]),
            helper.make_node(
                "MaxPool",
                [f"{prefix}_green_cum"],
                [f"{prefix}_last_green_cum"],
                kernel_shape=[1, SIZE],
                pads=[0, SIZE - 1, 0, 0],
            ),
            helper.make_node("Greater", [f"{prefix}_cum", f"{prefix}_last_green_cum"], [f"{prefix}_left"]),
            helper.make_node("Gather", [seed, "reverse_idx"], [f"{prefix}_rev"], axis=3),
            helper.make_node("CumSum", [f"{prefix}_rev", "axis_w"], [f"{prefix}_rev_cum"]),
            helper.make_node("Gather", ["green_bool", "reverse_idx"], [f"{prefix}_green_rev"], axis=3),
            helper.make_node(
                "Where",
                [f"{prefix}_green_rev", f"{prefix}_rev_cum", "zero_f32"],
                [f"{prefix}_rev_green_cum"],
            ),
            helper.make_node(
                "MaxPool",
                [f"{prefix}_rev_green_cum"],
                [f"{prefix}_rev_last_green_cum"],
                kernel_shape=[1, SIZE],
                pads=[0, SIZE - 1, 0, 0],
            ),
            helper.make_node(
                "Greater",
                [f"{prefix}_rev_cum", f"{prefix}_rev_last_green_cum"],
                [f"{prefix}_right_rev"],
            ),
            helper.make_node("Gather", [f"{prefix}_right_rev", "reverse_idx"], [f"{prefix}_right"], axis=3),
            helper.make_node("Or", [f"{prefix}_left", f"{prefix}_right"], [f"{prefix}_connected"]),
            helper.make_node("And", [f"{prefix}_connected", "black_bool"], [f"{prefix}_black_connected"]),
            helper.make_node("Cast", [f"{prefix}_black_connected"], [f"{prefix}_closed"], to=onnx.TensorProto.FLOAT),
        ]
    )
    return f"{prefix}_closed"


def _vertical_closure(nodes: list[onnx.NodeProto], seed: str, prefix: str) -> str:
    nodes.extend(
        [
            helper.make_node("CumSum", [seed, "axis_h"], [f"{prefix}_cum"]),
            helper.make_node("Where", ["green_bool", f"{prefix}_cum", "zero_f32"], [f"{prefix}_green_cum"]),
            helper.make_node(
                "MaxPool",
                [f"{prefix}_green_cum"],
                [f"{prefix}_last_green_cum"],
                kernel_shape=[SIZE, 1],
                pads=[SIZE - 1, 0, 0, 0],
            ),
            helper.make_node("Greater", [f"{prefix}_cum", f"{prefix}_last_green_cum"], [f"{prefix}_up"]),
            helper.make_node("Gather", [seed, "reverse_idx"], [f"{prefix}_rev"], axis=2),
            helper.make_node("CumSum", [f"{prefix}_rev", "axis_h"], [f"{prefix}_rev_cum"]),
            helper.make_node("Gather", ["green_bool", "reverse_idx"], [f"{prefix}_green_rev"], axis=2),
            helper.make_node(
                "Where",
                [f"{prefix}_green_rev", f"{prefix}_rev_cum", "zero_f32"],
                [f"{prefix}_rev_green_cum"],
            ),
            helper.make_node(
                "MaxPool",
                [f"{prefix}_rev_green_cum"],
                [f"{prefix}_rev_last_green_cum"],
                kernel_shape=[SIZE, 1],
                pads=[SIZE - 1, 0, 0, 0],
            ),
            helper.make_node(
                "Greater",
                [f"{prefix}_rev_cum", f"{prefix}_rev_last_green_cum"],
                [f"{prefix}_down_rev"],
            ),
            helper.make_node("Gather", [f"{prefix}_down_rev", "reverse_idx"], [f"{prefix}_down"], axis=2),
            helper.make_node("Or", [f"{prefix}_up", f"{prefix}_down"], [f"{prefix}_connected"]),
            helper.make_node("And", [f"{prefix}_connected", "black_bool"], [f"{prefix}_black_connected"]),
            helper.make_node("Cast", [f"{prefix}_black_connected"], [f"{prefix}_closed"], to=onnx.TensorProto.FLOAT),
        ]
    )
    return f"{prefix}_closed"


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
        _int64_tensor("axis_h", [2], [1]),
        _int64_tensor("axis_w", [3], [1]),
        _int64_tensor("reverse_idx", list(reversed(range(SIZE))), [SIZE]),
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
        helper.make_node("Cast", ["seed_bool"], ["external_0"], to=onnx.TensorProto.FLOAT),
    ]

    external = "external_0"
    for step in range(LINE_CLOSURE_STEPS):
        horizontal = _horizontal_closure(nodes, external, f"step{step}_h")
        external = _vertical_closure(nodes, horizontal, f"step{step}_v")

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
