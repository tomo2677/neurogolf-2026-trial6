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


def _shift(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    source: str,
    output: str,
    dr: int,
    dc: int,
) -> None:
    row_start = max(0, -dr)
    row_end = SIZE - max(0, dr)
    col_start = max(0, -dc)
    col_end = SIZE - max(0, dc)
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
    x, y = make_io_value_infos()

    initializers = [
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("channel0", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 10, 1, 1]),
        _f32_tensor("nonzero_channels", [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 10, 1, 1]),
        _int64_tensor("gray_starts", [0, 5, 0, 0], [4]),
        _int64_tensor("gray_ends", [1, 6, SIZE, SIZE], [4]),
        _int64_tensor("nonzero_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("nonzero_ends", [1, 10, 1, 1], [4]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, SIZE - 3, SIZE - 3], [8]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "gray_starts", "gray_ends"], ["gray_mask"]),
    ]

    slots: list[str] = []
    for row, dr in enumerate((-1, 0, 1)):
        for col, dc in enumerate((-1, 0, 1)):
            prefix = f"slot_{row}_{col}"
            _shift(nodes, initializers, "gray_mask", f"{prefix}_neighbor_mask", dr, dc)
            nodes.extend(
                [
                    helper.make_node("Mul", ["input", f"{prefix}_neighbor_mask"], [f"{prefix}_masked_input"]),
                    helper.make_node("ReduceMax", [f"{prefix}_masked_input"], [f"{prefix}_raw"], axes=[2, 3], keepdims=1),
                    helper.make_node("Slice", [f"{prefix}_raw", "nonzero_starts", "nonzero_ends"], [f"{prefix}_raw_nonzero"]),
                    helper.make_node("ReduceMax", [f"{prefix}_raw_nonzero"], [f"{prefix}_nonzero_any"], axes=[1], keepdims=1),
                    helper.make_node("Equal", [f"{prefix}_nonzero_any", "zero_f32"], [f"{prefix}_zero_bool"]),
                    helper.make_node("Cast", [f"{prefix}_zero_bool"], [f"{prefix}_zero_f32"], to=onnx.TensorProto.FLOAT),
                    helper.make_node("Mul", [f"{prefix}_raw", "nonzero_channels"], [f"{prefix}_nonzero_part"]),
                    helper.make_node("Mul", [f"{prefix}_zero_f32", "channel0"], [f"{prefix}_zero_part"]),
                    helper.make_node("Max", [f"{prefix}_nonzero_part", f"{prefix}_zero_part"], [f"{prefix}_onehot"]),
                ]
            )
            slots.append(f"{prefix}_onehot")

    nodes.extend(
        [
            helper.make_node("Concat", slots[0:3], ["row0"], axis=3),
            helper.make_node("Concat", slots[3:6], ["row1"], axis=3),
            helper.make_node("Concat", slots[6:9], ["row2"], axis=3),
            helper.make_node("Concat", ["row0", "row1", "row2"], ["output3"], axis=2),
            helper.make_node("Pad", ["output3", "pads_output"], ["output"], mode="constant"),
        ]
    )

    graph = helper.make_graph(nodes, "task022_gray_center_overlay_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
