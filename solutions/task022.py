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


def _shift(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    source: str,
    output: str,
    dr: int,
    dc: int,
) -> None:
    row_start = max(0, -dr)
    row_end = WINDOW - max(0, dr)
    col_start = max(0, -dc)
    col_end = WINDOW - max(0, dc)
    pad_top = max(0, dr)
    pad_bottom = max(0, -dr)
    pad_left = max(0, dc)
    pad_right = max(0, -dc)
    key = f"{output}_shift"
    initializers.extend(
        [
            _int64_tensor(f"{key}_starts", [0, 0, row_start, col_start], [4]),
            _int64_tensor(f"{key}_ends", [1, 1, row_end, col_end], [4]),
            _int64_tensor(f"{key}_pads", [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right], [8]),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Slice", [source, f"{key}_starts", f"{key}_ends"], [f"{output}_crop"]),
            helper.make_node("Pad", [f"{output}_crop", f"{key}_pads"], [output], mode="constant"),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("slice_hw_starts", [0, 0], [2]),
        _int64_tensor("slice_hw_ends", [WINDOW, WINDOW], [2]),
        _int64_tensor("slice_hw_axes", [2, 3], [2]),
        _int64_tensor("five_i64", [5], [1]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 27, 27], [8]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("five_u8", [5], [1, 1, 1, 1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "slice_hw_starts", "slice_hw_ends", "slice_hw_axes"], ["input11"]),
        helper.make_node("ArgMax", ["input11"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Cast", ["input_color_i64"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Equal", ["input_color_i64", "five_i64"], ["gray_bool"]),
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
                    helper.make_node("ReduceMax", [f"{prefix}_color_grid"], [f"{prefix}_color"], axes=[2, 3], keepdims=1),
                ]
            )
            slots.append(f"{prefix}_color")

    nodes.extend(
        [
            helper.make_node("Concat", slots[0:3], ["row0"], axis=3),
            helper.make_node("Concat", slots[3:6], ["row1"], axis=3),
            helper.make_node("Concat", slots[6:9], ["row2"], axis=3),
            helper.make_node("Concat", ["row0", "row1", "row2"], ["color3"], axis=2),
            helper.make_node("Pad", ["color3", "pads_output", "invalid_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task022_window11_gray_overlay_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
