from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 10


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _guide_row(left: str, right: str) -> list[str]:
    return [left] + ["bg_u8"] * 8 + [right]


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("marker_starts", [3, 0, 0], [3]),
        _int64_tensor("marker_ends", [4, SIZE, SIZE], [3]),
        _int64_tensor("top_left_starts", [1, 0, 0], [3]),
        _int64_tensor("top_left_ends", [10, 1, 1], [3]),
        _int64_tensor("top_right_starts", [1, 0, SIZE - 1], [3]),
        _int64_tensor("top_right_ends", [10, 1, SIZE], [3]),
        _int64_tensor("bottom_left_starts", [1, SIZE - 1, 0], [3]),
        _int64_tensor("bottom_left_ends", [10, SIZE, 1], [3]),
        _int64_tensor("axes_chw", [1, 2, 3], [3]),
        _int64_tensor("output_pads_hw", [0, 0, 20, 20], [4]),
        _int64_tensor("output_pad_axes", [2, 3], [2]),
        _u8_tensor("bg_u8", [9], [1, 1, 1, 1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10", [9, 0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 10, 1, 1]),
        _f32_tensor("shift_color_w", [float(c) for c in range(9)], [1, 9, 1, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "marker_starts", "marker_ends", "axes_chw"], ["marker_f32"]),
        helper.make_node("Cast", ["marker_f32"], ["marker"], to=onnx.TensorProto.BOOL),
    ]

    for name in ("top_left", "top_right", "bottom_left"):
        nodes.extend(
            [
                helper.make_node("Slice", ["input", f"{name}_starts", f"{name}_ends", "axes_chw"], [f"{name}_onehot"]),
                helper.make_node("Conv", [f"{name}_onehot", "shift_color_w"], [f"{name}_f32"], kernel_shape=[1, 1]),
                helper.make_node("Cast", [f"{name}_f32"], [f"{name}_color"], to=onnx.TensorProto.UINT8),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Equal", ["top_left_color", "top_right_color"], ["use_rows"]),
            helper.make_node(
                "Concat",
                [
                    "top_left_color",
                    "top_left_color",
                    "top_left_color",
                    "top_left_color",
                    "top_left_color",
                    "bottom_left_color",
                    "bottom_left_color",
                    "bottom_left_color",
                    "bottom_left_color",
                    "bottom_left_color",
                ],
                ["row_replacement"],
                axis=2,
            ),
            helper.make_node(
                "Concat",
                [
                    "top_left_color",
                    "top_left_color",
                    "top_left_color",
                    "top_left_color",
                    "top_left_color",
                    "top_right_color",
                    "top_right_color",
                    "top_right_color",
                    "top_right_color",
                    "top_right_color",
                ],
                ["col_replacement"],
                axis=3,
            ),
            helper.make_node("Where", ["use_rows", "row_replacement", "col_replacement"], ["replacement"]),
            helper.make_node("Concat", _guide_row("top_left_color", "bottom_left_color"), ["row_guides"], axis=2),
            helper.make_node("Concat", _guide_row("top_left_color", "top_right_color"), ["col_guides"], axis=3),
            helper.make_node("Where", ["use_rows", "row_guides", "col_guides"], ["guides"]),
            helper.make_node("Where", ["marker", "replacement", "guides"], ["color10"]),
            helper.make_node("Pad", ["color10", "output_pads_hw", "invalid_u8", "output_pad_axes"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task040_corner_equal_orient_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
