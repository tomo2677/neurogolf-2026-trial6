from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _shift(nodes: list[onnx.NodeProto], source: str, dr: str, dc: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Sub", ["row_grid_i64", dr], [f"{output}_src_r"]),
            helper.make_node("Sub", ["col_grid_i64", dc], [f"{output}_src_c"]),
            helper.make_node("Greater", [f"{output}_src_r", "neg_one_i64"], [f"{output}_r_nonneg"]),
            helper.make_node("Less", [f"{output}_src_r", "size_i64"], [f"{output}_r_lt_size"]),
            helper.make_node("Greater", [f"{output}_src_c", "neg_one_i64"], [f"{output}_c_nonneg"]),
            helper.make_node("Less", [f"{output}_src_c", "size_i64"], [f"{output}_c_lt_size"]),
            helper.make_node("And", [f"{output}_r_nonneg", f"{output}_r_lt_size"], [f"{output}_r_ok"]),
            helper.make_node("And", [f"{output}_c_nonneg", f"{output}_c_lt_size"], [f"{output}_c_ok"]),
            helper.make_node("And", [f"{output}_r_ok", f"{output}_c_ok"], [f"{output}_in_bounds"]),
            helper.make_node("Where", [f"{output}_in_bounds", f"{output}_src_r", "zero_i64"], [f"{output}_safe_r"]),
            helper.make_node("Where", [f"{output}_in_bounds", f"{output}_src_c", "zero_i64"], [f"{output}_safe_c"]),
            helper.make_node("Mul", [f"{output}_safe_r", "width_30_i64"], [f"{output}_safe_r_offset"]),
            helper.make_node("Add", [f"{output}_safe_r_offset", f"{output}_safe_c"], [f"{output}_safe_spatial"]),
            helper.make_node("Reshape", [f"{output}_safe_spatial", "shape_index_1x900"], [f"{output}_safe_spatial_flat"]),
            helper.make_node("Expand", [f"{output}_safe_spatial_flat", "shape_index_10x900"], [f"{output}_indices"]),
            helper.make_node("Reshape", [source, "shape_flat_10x900"], [f"{output}_source_flat"]),
            helper.make_node("GatherElements", [f"{output}_source_flat", f"{output}_indices"], [f"{output}_shifted_flat"], axis=2),
            helper.make_node("Reshape", [f"{output}_shifted_flat", "shape_1x10x30x30"], [f"{output}_shifted"]),
            helper.make_node("Cast", [f"{output}_in_bounds"], [f"{output}_in_bounds_f32"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Mul", [f"{output}_shifted", f"{output}_in_bounds_f32"], [output]),
        ]
    )


def _shift_static(nodes: list[onnx.NodeProto], source: str, dr: int, dc: int, output: str) -> None:
    nodes.append(
        helper.make_node(
            "Pad",
            [source, _pad_name(dr, dc), "zero_f32"],
            [output],
            mode="constant",
        )
    )


def _pad_name(dr: int, dc: int) -> str:
    row = f"m{-dr}" if dr < 0 else f"p{dr}"
    col = f"m{-dc}" if dc < 0 else f"p{dc}"
    return f"pads_{row}_{col}"


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("row_idx", list(range(30)), [1, 1, 30, 1]),
        _int64_tensor("col_idx", list(range(30)), [1, 1, 1, 30]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("zero_i64", [0], [1]),
        _int64_tensor("neg_one_i64", [-1], [1]),
        _int64_tensor("size_i64", [30], [1]),
        _int64_tensor("width_30_i64", [30], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("shape_index_1x900", [1, 1, 900], [3]),
        _int64_tensor("shape_index_10x900", [1, 10, 900], [3]),
        _int64_tensor("shape_flat_10x900", [1, 10, 900], [3]),
        _int64_tensor("shape_1x10x30x30", [1, 10, 30, 30], [4]),
        _int64_tensor("nonzero_start", [1], [1]),
        _int64_tensor("nonzero_end", [10], [1]),
        _int64_tensor("axis_channel", [1], [1]),
        _int64_tensor("row_grid_i64", [r for r in range(30) for _ in range(30)], [1, 1, 30, 30]),
        _int64_tensor("col_grid_i64", [c for _ in range(30) for c in range(30)], [1, 1, 30, 30]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("color8_pixel", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1, 10, 1, 1]),
    ]
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        initializers.append(_int64_tensor(_pad_name(dr, dc), [0, 0, dr, dc, 0, 0, -dr, -dc], [8]))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["valid_cell_score"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["valid_cell_score"], ["row_present_score"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["valid_cell_score"], ["col_present_score"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_present_score"], ["last_row"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("ArgMax", ["col_present_score"], ["last_col"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Add", ["last_row", "one_i64"], ["height_raw"]),
        helper.make_node("Add", ["last_col", "one_i64"], ["width_raw"]),
        helper.make_node("Reshape", ["height_raw", "shape1"], ["height"]),
        helper.make_node("Reshape", ["width_raw", "shape1"], ["width"]),
    ]

    _shift(nodes, "input", "zero_i64", "width", "tile_right")
    _shift(nodes, "input", "height", "zero_i64", "tile_down")
    _shift(nodes, "input", "height", "width", "tile_down_right")
    nodes.extend(
        [
            helper.make_node("Max", ["input", "tile_right", "tile_down", "tile_down_right"], ["tiled"]),
            helper.make_node("Slice", ["tiled", "nonzero_start", "nonzero_end", "axis_channel"], ["tiled_nonzero"]),
            helper.make_node("ReduceMax", ["tiled_nonzero"], ["colored_score"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["colored_score", "zero_f32"], ["colored"]),
            helper.make_node("Cast", ["colored"], ["colored_f32"], to=onnx.TensorProto.FLOAT),
        ]
    )

    shifted = []
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        name = f"diag_{dr}_{dc}"
        _shift_static(nodes, "colored_f32", dr, dc, name)
        shifted.append(name)

    nodes.extend(
        [
            helper.make_node("Max", shifted, ["diag_score"]),
            helper.make_node("Greater", ["diag_score", "zero_f32"], ["diag_bool"]),
            helper.make_node("ReduceMax", ["tiled"], ["valid_out_score"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["valid_out_score", "zero_f32"], ["valid_out"]),
            helper.make_node("And", ["diag_bool", "valid_out"], ["diag_in_output"]),
            helper.make_node("Not", ["colored"], ["not_colored"]),
            helper.make_node("And", ["diag_in_output", "not_colored"], ["eight_mask"]),
            helper.make_node("Where", ["eight_mask", "color8_pixel", "tiled"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task019_tile_diag_eights_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
