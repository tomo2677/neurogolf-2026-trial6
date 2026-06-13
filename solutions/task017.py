from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


H = 21
W = 21
PERIODS = [4, 5, 6, 7, 8, 9]


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _period_index_map(p: int) -> list[int]:
    return [(r % p) * p + (c % p) for r in range(H) for c in range(W)]


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("axes_nchw", [0, 1, 2, 3], [4]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 9, 9], [8]),
        _int64_tensor("shape_111", [1, 1, 1], [3]),
        _int64_tensor("period_slice_end", [1, 1, H, W], [4]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("one_u8", [1], [1]),
        _u8_tensor("outside_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    for p in PERIODS:
        p2 = p * p
        initializers.append(_int64_tensor(f"period_slice_steps_{p}", [1, 1, p, p], [4]))
        if p == 7:
            initializers.extend(
                [
                    _int64_tensor(f"shape_tile_{p}", [1, 1, p, p], [4]),
                    _int64_tensor("tile_repeats_7", [1, 1, 3, 3], [4]),
                ]
            )
        else:
            initializers.extend(
                [
                    _int64_tensor(f"shape_tile_{p}", [p2], [1]),
                    _int64_tensor(f"period_index_{p}", _period_index_map(p), [1, 1, H, W]),
                ]
            )
        for residue in range(p2):
            rr, cc = divmod(residue, p)
            initializers.append(_int64_tensor(f"period_slice_start_{p}_{residue}", [0, 0, rr, cc], [4]))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ArgMax", ["input"], ["input_color_i64"], axis=1, keepdims=1),
        helper.make_node("Cast", ["input_color_i64"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
    ]

    color_candidates: list[tuple[int, str]] = []
    for p in PERIODS:
        residue_colors: list[str] = []
        residue_ok: list[str] = []
        for residue in range(p * p):
            prefix = f"r{p}_{residue}"
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        [
                            "input_color_u8",
                            f"period_slice_start_{p}_{residue}",
                            "period_slice_end",
                            "axes_nchw",
                            f"period_slice_steps_{p}",
                        ],
                        [f"{prefix}_colors"],
                    ),
                    helper.make_node("Greater", [f"{prefix}_colors", "zero_u8"], [f"{prefix}_seen"]),
                    helper.make_node("Where", [f"{prefix}_seen", f"{prefix}_colors", "outside_u8"], [f"{prefix}_min_base"]),
                    helper.make_node("ReduceMin", [f"{prefix}_min_base"], [f"{prefix}_min"], axes=[2, 3], keepdims=0),
                    helper.make_node("ReduceMax", [f"{prefix}_colors"], [f"{prefix}_max"], axes=[2, 3], keepdims=0),
                    helper.make_node("Equal", [f"{prefix}_min", f"{prefix}_max"], [f"{prefix}_ok"]),
                    helper.make_node("Reshape", [f"{prefix}_max", "shape_111"], [f"{prefix}_color"]),
                    helper.make_node("Reshape", [f"{prefix}_ok", "shape_111"], [f"{prefix}_ok_u"]),
                ]
            )
            residue_colors.append(f"{prefix}_color")
            residue_ok.append(f"{prefix}_ok_u")

        nodes.extend(
            [
                helper.make_node("Concat", residue_colors, [f"tile_color_rank3_{p}"], axis=2),
                helper.make_node("Reshape", [f"tile_color_rank3_{p}", f"shape_tile_{p}"], [f"tile_color_u8_{p}"]),
            ]
        )
        if p == 7:
            nodes.append(helper.make_node("Tile", [f"tile_color_u8_{p}", "tile_repeats_7"], [f"color21_{p}"]))
        else:
            nodes.append(helper.make_node("Gather", [f"tile_color_u8_{p}", f"period_index_{p}"], [f"color21_{p}"], axis=0))
        if p != PERIODS[-1]:
            nodes.extend(
                [
                    helper.make_node("Concat", residue_ok, [f"residue_ok_{p}"], axis=2),
                    helper.make_node("Cast", [f"residue_ok_{p}"], [f"residue_ok_u8_{p}"], to=onnx.TensorProto.UINT8),
                    helper.make_node("ReduceMin", [f"residue_ok_u8_{p}"], [f"all_ok_u8_{p}"], axes=[0, 1, 2], keepdims=0),
                    helper.make_node("Equal", [f"all_ok_u8_{p}", "one_u8"], [f"period_ok_{p}"]),
                ]
            )
        color_candidates.append((p, f"color21_{p}"))

    current = color_candidates[-1][1]
    for p, candidate in reversed(color_candidates[:-1]):
        selected = f"selected_color21_{p}"
        nodes.append(helper.make_node("Where", [f"period_ok_{p}", candidate, current], [selected]))
        current = selected

    nodes.extend(
        [
            helper.make_node("Pad", [current, "pads_output", "outside_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task017_period_minmax_color_grid_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
