from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


H = 21
W = 21
MAX_PERIOD = 21


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _period_mask(p: int) -> np.ndarray:
    mask = np.zeros((H * W, p * p), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            mask[r * W + c, (r % p) * p + (c % p)] = 1.0
    return mask


def _period_index_map(p: int) -> list[int]:
    return [(r % p) * p + (c % p) for r in range(H) for c in range(W)]


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("input21_start", [1, 0, 0], [3]),
        _int64_tensor("input21_end", [10, H, W], [3]),
        _int64_tensor("axes_chw", [1, 2, 3], [3]),
        _int64_tensor("shape_flat", [1, 9, H * W], [3]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 9, 9], [8]),
        _int64_tensor("one_i64", [1], [1]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("one_f32", [1.0], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _u8_tensor("outside_u8", [255], [1]),
    ]

    for p in range(1, MAX_PERIOD + 1):
        p2 = p * p
        initializers.extend(
            [
                helper.make_tensor(
                    f"period_mask_{p}",
                    onnx.TensorProto.FLOAT,
                    [H * W, p2],
                    _period_mask(p).ravel(),
                ),
                _int64_tensor(f"shape_tile_{p}", [p2], [1]),
                _int64_tensor(f"period_index_{p}", _period_index_map(p), [1, 1, H, W]),
                _f32_tensor(f"period_size_{p}", [float(p2)], [1]),
            ]
        )

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "input21_start", "input21_end", "axes_chw"], ["input_nonzero"]),
        helper.make_node("Reshape", ["input_nonzero", "shape_flat"], ["input_flat"]),
    ]

    color_candidates: list[str] = []
    for p in range(1, MAX_PERIOD + 1):
        nodes.extend(
            [
                helper.make_node("MatMul", ["input_flat", f"period_mask_{p}"], [f"counts_{p}"]),
                helper.make_node("Greater", [f"counts_{p}", "zero_f32"], [f"seen_{p}"]),
                helper.make_node("Cast", [f"seen_{p}"], [f"seen_f32_{p}"], to=onnx.TensorProto.FLOAT),
                helper.make_node("ReduceSum", [f"seen_f32_{p}"], [f"color_count_{p}"], axes=[1], keepdims=0),
                helper.make_node("Equal", [f"color_count_{p}", "one_f32"], [f"residue_ok_{p}"]),
                helper.make_node("Cast", [f"residue_ok_{p}"], [f"residue_ok_f32_{p}"], to=onnx.TensorProto.FLOAT),
                helper.make_node("ReduceSum", [f"residue_ok_f32_{p}"], [f"ok_count_{p}"], axes=[0, 1], keepdims=0),
                helper.make_node("Equal", [f"ok_count_{p}", f"period_size_{p}"], [f"period_ok_{p}"]),
                helper.make_node("ArgMax", [f"counts_{p}"], [f"tile_zero_based_{p}"], axis=1, keepdims=0),
                helper.make_node("Reshape", [f"tile_zero_based_{p}", f"shape_tile_{p}"], [f"tile_zero_flat_{p}"]),
                helper.make_node("Add", [f"tile_zero_flat_{p}", "one_i64"], [f"tile_color_i64_{p}"]),
                helper.make_node("Cast", [f"tile_color_i64_{p}"], [f"tile_color_u8_{p}"], to=onnx.TensorProto.UINT8),
                helper.make_node("Gather", [f"tile_color_u8_{p}", f"period_index_{p}"], [f"color21_{p}"], axis=0),
                helper.make_node("Pad", [f"color21_{p}", "pads_output", "outside_u8"], [f"color30_{p}"], mode="constant"),
            ]
        )
        color_candidates.append(f"color30_{p}")

    current = color_candidates[-1]
    for p in range(MAX_PERIOD - 1, 0, -1):
        selected = f"selected_color30_{p}"
        nodes.append(helper.make_node("Where", [f"period_ok_{p}", color_candidates[p - 1], current], [selected]))
        current = selected
    nodes.append(helper.make_node("Equal", ["colors10", current], ["output"]))

    graph = helper.make_graph(nodes, "task017_periodic_fill_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
