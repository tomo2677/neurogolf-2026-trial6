from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


WINDOW = 11


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


def _shift(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    source: str,
    output: str,
    dr: int,
    dc: int,
) -> None:
    key = f"{output}_shift"
    initializers.append(_int64_tensor(f"{key}_pads", [dr, dc, -dr, -dc], [4]))
    nodes.append(helper.make_node("Pad", [source, f"{key}_pads", "", "slice_hw_axes"], [output], mode="constant"))


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("slice_hw_starts", [0, 0], [2]),
        _int64_tensor("slice_hw_ends", [WINDOW, WINDOW], [2]),
        _int64_tensor("slice_hw_axes", [2, 3], [2]),
        _int64_tensor("pads_output", [0, 0, 27, 27], [4]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("five_u8", [5], [1, 1, 1, 1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
        _f32_tensor("five_f32", [5.0], [1, 1, 1, 1]),
        _f32_tensor("color_conv_w", [float(i) for i in range(10)], [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "slice_hw_starts", "slice_hw_ends", "slice_hw_axes"], ["input11"]),
        helper.make_node("Conv", ["input11", "color_conv_w"], ["input_color_f32"]),
        helper.make_node("Cast", ["input_color_f32"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Equal", ["input_color_f32", "five_f32"], ["gray_bool"]),
    ]

    slots: list[str] = []
    for row, dr in enumerate((-1, 0, 1)):
        for col, dc in enumerate((-1, 0, 1)):
            prefix = f"slot_{row}_{col}"
            if row == 1 and col == 1:
                slots.append("five_u8")
                continue
            _shift(nodes, initializers, "gray_bool", f"{prefix}_neighbor_bool", dr, dc)
            nodes.extend(
                [
                    helper.make_node("Where", [f"{prefix}_neighbor_bool", "input_color_u8", "zero_u8"], [f"{prefix}_color_grid"]),
                    helper.make_node("ReduceMax", [f"{prefix}_color_grid", "slice_hw_axes"], [f"{prefix}_color"], keepdims=1),
                ]
            )
            slots.append(f"{prefix}_color")

    nodes.extend(
        [
            helper.make_node("Concat", slots[0:3], ["row0"], axis=3),
            helper.make_node("Concat", slots[3:6], ["row1"], axis=3),
            helper.make_node("Concat", slots[6:9], ["row2"], axis=3),
            helper.make_node("Concat", ["row0", "row1", "row2"], ["color3"], axis=2),
            helper.make_node("Equal", ["colors10_u8", "color3"], ["output3"]),
            helper.make_node("Pad", ["output3", "pads_output", "", "slice_hw_axes"], ["output"], mode="constant"),
        ]
    )

    graph = helper.make_graph(nodes, "task022_conv_color_map_graph", [x], [y], initializers)
    _dedupe_initializers(graph)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
