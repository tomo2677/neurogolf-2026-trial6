from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
COLORS = range(1, 10)


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _initializer_key(tensor: onnx.TensorProto) -> tuple:
    return (
        tensor.data_type,
        tuple(tensor.dims),
        bytes(tensor.raw_data),
        tuple(tensor.int32_data),
        tuple(tensor.int64_data),
        tuple(tensor.float_data),
        tuple(tensor.double_data),
        tuple(tensor.string_data),
    )


def _dedupe_initializers(graph: onnx.GraphProto) -> None:
    canonical: dict[tuple, str] = {}
    rename: dict[str, str] = {}
    unique: list[onnx.TensorProto] = []
    for initializer in graph.initializer:
        key = _initializer_key(initializer)
        existing = canonical.get(key)
        if existing is None:
            canonical[key] = initializer.name
            unique.append(initializer)
        else:
            rename[initializer.name] = existing
    if not rename:
        return
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in rename:
                node.input[index] = rename[name]
    del graph.initializer[:]
    graph.initializer.extend(unique)


def _shift_bool(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    source: str,
    output: str,
    dr: int,
    dc: int,
) -> str:
    initializers.append(_int64_tensor(f"{output}_pads", [0, 0, dr, dc, 0, 0, -dr, -dc], [8]))
    nodes.append(helper.make_node("Pad", [source, f"{output}_pads"], [output], mode="constant"))
    return output


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _f32_tensor("zero_f32", [0.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _int64_tensor("sum_axes_all", [0, 1, 2, 3], [4]),
        _int64_tensor("sum_axis_h", [2], [1]),
        _int64_tensor("sum_axis_w", [3], [1]),
    ]
    for color in COLORS:
        initializers.append(_u8_tensor(f"color{color}_u8", [color], [1]))

    nodes = [
        helper.make_node("ArgMax", ["input"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Cast", ["input_color_i64"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["input"], ["cell_present"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["cell_present"], ["row_present_f32"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["cell_present"], ["col_present_f32"], axes=[2], keepdims=1),
        helper.make_node("Greater", ["row_present_f32", "zero_f32"], ["row_valid"]),
        helper.make_node("Greater", ["col_present_f32", "zero_f32"], ["col_valid"]),
        helper.make_node("And", ["row_valid", "col_valid"], ["valid_area"]),
        helper.make_node("ReduceSum", ["row_present_f32", "sum_axes_all"], ["height_count"], keepdims=1),
        helper.make_node("ReduceSum", ["col_present_f32", "sum_axes_all"], ["width_count"], keepdims=1),
        helper.make_node("Cast", ["height_count"], ["height_count_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("Cast", ["width_count"], ["width_count_f16"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("Where", ["valid_area", "zero_u8", "invalid_u8"], ["color_grid_0"]),
    ]

    current = "color_grid_0"
    for color in COLORS:
        prefix = f"c{color}"
        nodes.extend(
            [
                helper.make_node("Equal", ["input_color_u8", f"color{color}_u8"], [f"{prefix}_mask_bool"]),
                helper.make_node("Cast", [f"{prefix}_mask_bool"], [f"{prefix}_mask_f16"], to=onnx.TensorProto.FLOAT16),
                helper.make_node("ReduceSum", [f"{prefix}_mask_f16", "sum_axis_w"], [f"{prefix}_row_count"], keepdims=1),
                helper.make_node("ReduceSum", [f"{prefix}_mask_f16", "sum_axis_h"], [f"{prefix}_col_count"], keepdims=1),
                helper.make_node("Equal", [f"{prefix}_row_count", "width_count_f16"], [f"{prefix}_row_line_raw"]),
                helper.make_node("Equal", [f"{prefix}_col_count", "height_count_f16"], [f"{prefix}_col_line_raw"]),
                helper.make_node("And", [f"{prefix}_row_line_raw", "row_valid"], [f"{prefix}_row_line"]),
                helper.make_node("And", [f"{prefix}_col_line_raw", "col_valid"], [f"{prefix}_col_line"]),
                helper.make_node("And", [f"{prefix}_row_line", "valid_area"], [f"{prefix}_row_line_area_bool"]),
                helper.make_node("And", [f"{prefix}_col_line", "valid_area"], [f"{prefix}_col_line_area_bool"]),
                helper.make_node("Or", [f"{prefix}_row_line_area_bool", f"{prefix}_col_line_area_bool"], [f"{prefix}_line_cover_bool"]),
                helper.make_node("Not", [f"{prefix}_line_cover_bool"], [f"{prefix}_not_line_bool"]),
                helper.make_node("And", [f"{prefix}_mask_bool", f"{prefix}_not_line_bool"], [f"{prefix}_scatter_bool"]),
                helper.make_node("Cast", [f"{prefix}_scatter_bool"], [f"{prefix}_scatter_u8"], to=onnx.TensorProto.UINT8),
                helper.make_node(
                    "MaxPool",
                    [f"{prefix}_scatter_u8"],
                    [f"{prefix}_up_seen_u8"],
                    kernel_shape=[SIZE, 1],
                    pads=[SIZE - 1, 0, 0, 0],
                ),
                helper.make_node(
                    "MaxPool",
                    [f"{prefix}_scatter_u8"],
                    [f"{prefix}_down_seen_u8"],
                    kernel_shape=[SIZE, 1],
                    pads=[0, 0, SIZE - 1, 0],
                ),
                helper.make_node(
                    "MaxPool",
                    [f"{prefix}_scatter_u8"],
                    [f"{prefix}_left_seen_u8"],
                    kernel_shape=[1, SIZE],
                    pads=[0, SIZE - 1, 0, 0],
                ),
                helper.make_node(
                    "MaxPool",
                    [f"{prefix}_scatter_u8"],
                    [f"{prefix}_right_seen_u8"],
                    kernel_shape=[1, SIZE],
                    pads=[0, 0, 0, SIZE - 1],
                ),
                helper.make_node("Greater", [f"{prefix}_up_seen_u8", "zero_u8"], [f"{prefix}_up_seen_bool"]),
                helper.make_node("Greater", [f"{prefix}_down_seen_u8", "zero_u8"], [f"{prefix}_down_seen_bool"]),
                helper.make_node("Greater", [f"{prefix}_left_seen_u8", "zero_u8"], [f"{prefix}_left_seen_bool"]),
                helper.make_node("Greater", [f"{prefix}_right_seen_u8", "zero_u8"], [f"{prefix}_right_seen_bool"]),
                helper.make_node("And", [f"{prefix}_up_seen_bool", f"{prefix}_row_line_area_bool"], [f"{prefix}_above_line_bool"]),
                helper.make_node("And", [f"{prefix}_down_seen_bool", f"{prefix}_row_line_area_bool"], [f"{prefix}_below_line_bool"]),
                helper.make_node("And", [f"{prefix}_left_seen_bool", f"{prefix}_col_line_area_bool"], [f"{prefix}_left_line_bool"]),
                helper.make_node("And", [f"{prefix}_right_seen_bool", f"{prefix}_col_line_area_bool"], [f"{prefix}_right_line_bool"]),
            ]
        )
        above_proj = _shift_bool(nodes, initializers, f"{prefix}_above_line_bool", f"{prefix}_above_proj", -1, 0)
        below_proj = _shift_bool(nodes, initializers, f"{prefix}_below_line_bool", f"{prefix}_below_proj", 1, 0)
        left_proj = _shift_bool(nodes, initializers, f"{prefix}_left_line_bool", f"{prefix}_left_proj", 0, -1)
        right_proj = _shift_bool(nodes, initializers, f"{prefix}_right_line_bool", f"{prefix}_right_proj", 0, 1)
        nodes.extend(
            [
                helper.make_node("Or", [f"{prefix}_line_cover_bool", above_proj], [f"{prefix}_cover_or0"]),
                helper.make_node("Or", [f"{prefix}_cover_or0", below_proj], [f"{prefix}_cover_or1"]),
                helper.make_node("Or", [f"{prefix}_cover_or1", left_proj], [f"{prefix}_cover_or2"]),
                helper.make_node("Or", [f"{prefix}_cover_or2", right_proj], [f"{prefix}_cover_raw_bool"]),
                helper.make_node("And", [f"{prefix}_cover_raw_bool", "valid_area"], [f"{prefix}_cover_bool"]),
                helper.make_node("Where", [f"{prefix}_cover_bool", f"color{color}_u8", current], [f"color_grid_{color}"]),
            ]
        )
        current = f"color_grid_{color}"

    nodes.extend(
        [
            helper.make_node("Equal", ["colors10", current], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task025_line_projection_u8_graph", [x], [y], initializers)
    _dedupe_initializers(graph)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
