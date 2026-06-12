from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


H = 21
W = 21
PERIODS = [4, 5, 6, 7, 8, 9]
INTERNAL_TYPE = onnx.TensorProto.FLOAT16


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f16_tensor(name: str, values: list[float] | np.ndarray, dims: list[int] | None = None) -> onnx.TensorProto:
    arr = np.asarray(values, dtype=np.float16)
    shape = list(arr.shape) if dims is None else dims
    return helper.make_tensor(name, INTERNAL_TYPE, shape, arr.ravel())


def _period_mask(p: int) -> np.ndarray:
    mask = np.zeros((H * W, p * p), dtype=np.float16)
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
        _int64_tensor("axes_nchw", [0, 1, 2, 3], [4]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 9, 9], [8]),
        _int64_tensor("one_i64", [1], [1]),
        _f16_tensor("zero_f32", [0.0], [1]),
        _f16_tensor("one_f32", [1.0], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
        _u8_tensor("outside_u8", [255], [1]),
    ]

    for p in PERIODS:
        p2 = p * p
        initializers.extend(
            [
                _int64_tensor(f"period_slice_end_{p}", [1, 9, H, W], [4]),
                _int64_tensor(f"period_slice_steps_{p}", [1, 1, p, p], [4]),
                _int64_tensor(f"shape_tile_{p}", [p2], [1]),
                _int64_tensor(f"period_index_{p}", _period_index_map(p), [1, 1, H, W]),
                _f16_tensor(f"period_size_{p}", [float(p2)], [1]),
            ]
        )
        for residue in range(p2):
            rr, cc = divmod(residue, p)
            initializers.append(_int64_tensor(f"period_slice_start_{p}_{residue}", [0, 0, rr, cc], [4]))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "input21_start", "input21_end", "axes_chw"], ["input_nonzero_f32"]),
        helper.make_node("Cast", ["input_nonzero_f32"], ["input_nonzero"], to=INTERNAL_TYPE),
    ]

    color_candidates: list[tuple[int, str]] = []
    for p in PERIODS:
        residue_counts: list[str] = []
        for residue in range(p * p):
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        [
                            "input_nonzero",
                            f"period_slice_start_{p}_{residue}",
                            f"period_slice_end_{p}",
                            "axes_nchw",
                            f"period_slice_steps_{p}",
                        ],
                        [f"residue_cells_{p}_{residue}"],
                    ),
                    helper.make_node("ReduceSum", [f"residue_cells_{p}_{residue}"], [f"residue_sum_{p}_{residue}"], axes=[2, 3], keepdims=0),
                    helper.make_node("Unsqueeze", [f"residue_sum_{p}_{residue}"], [f"residue_count_{p}_{residue}"], axes=[2]),
                ]
            )
            residue_counts.append(f"residue_count_{p}_{residue}")
        nodes.extend(
            [
                helper.make_node("Concat", residue_counts, [f"counts_{p}"], axis=2),
                helper.make_node("Greater", [f"counts_{p}", "zero_f32"], [f"seen_{p}"]),
                helper.make_node("Cast", [f"seen_{p}"], [f"seen_f32_{p}"], to=INTERNAL_TYPE),
                helper.make_node("ReduceSum", [f"seen_f32_{p}"], [f"color_count_{p}"], axes=[1], keepdims=0),
                helper.make_node("Equal", [f"color_count_{p}", "one_f32"], [f"residue_ok_{p}"]),
                helper.make_node("Cast", [f"residue_ok_{p}"], [f"residue_ok_f32_{p}"], to=INTERNAL_TYPE),
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
        color_candidates.append((p, f"color30_{p}"))

    current = color_candidates[-1][1]
    for p, candidate in reversed(color_candidates[:-1]):
        selected = f"selected_color30_{p}"
        nodes.append(helper.make_node("Where", [f"period_ok_{p}", candidate, current], [selected]))
        current = selected
    nodes.append(helper.make_node("Equal", ["colors10", current], ["output"]))

    graph = helper.make_graph(nodes, "task017_periodic_fill_subset_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
