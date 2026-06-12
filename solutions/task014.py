from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _color_at_coord(nodes: list[onnx.NodeProto], row: str, col: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("Reshape", [row, "shape1"], [f"{output}_row1"]),
            helper.make_node("Reshape", [col, "shape1"], [f"{output}_col1"]),
            helper.make_node("Unsqueeze", [f"{output}_row1", "unsq_axis1"], [f"{output}_row11"]),
            helper.make_node("Unsqueeze", [f"{output}_col1", "unsq_axis1"], [f"{output}_col11"]),
            helper.make_node("Concat", [f"{output}_row11", f"{output}_col11"], [f"{output}_indices12"], axis=1),
            helper.make_node("Reshape", [f"{output}_indices12", "shape112"], [f"{output}_spatial_indices"]),
            helper.make_node("Unsqueeze", [f"{output}_spatial_indices", "unsq_batch_axes"], [f"{output}_indices_batched"]),
            helper.make_node("Expand", [f"{output}_indices_batched", "gathernd_index_shape"], [f"{output}_indices"]),
            helper.make_node("GatherND", ["input", f"{output}_indices"], [f"{output}_onehot"], batch_dims=2),
            helper.make_node("ArgMax", [f"{output}_onehot"], [output], axis=1, keepdims=1, select_last_index=1),
        ]
    )


def _color_id(nodes: list[onnx.NodeProto], mask: str, output: str) -> None:
    nodes.extend(
        [
            helper.make_node("And", [mask, "nonblack_bool"], [f"{output}_nonblack_bool"]),
            helper.make_node("Where", [f"{output}_nonblack_bool", "input_color_u8", "zero_u8"], [f"{output}_color_grid"]),
            helper.make_node("ReduceMax", [f"{output}_color_grid"], [output], axes=[2, 3], keepdims=1),
        ]
    )


def _unique(nodes: list[onnx.NodeProto], name: str, color: str, others: list[str]) -> None:
    eqs: list[str] = []
    for index, other in enumerate(others):
        eq = f"{name}_eq_{index}"
        nodes.append(helper.make_node("Equal", [color, other], [eq]))
        eqs.append(eq)
    nodes.extend(
        [
            helper.make_node("Or", [eqs[0], eqs[1]], [f"{name}_eq_any01"]),
            helper.make_node("Or", [f"{name}_eq_any01", eqs[2]], [f"{name}_eq_any"]),
            helper.make_node("Not", [f"{name}_eq_any"], [name]),
        ]
    )


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("row_idx", list(range(25)), [1, 1, 25, 1]),
        _int64_tensor("col_idx", list(range(25)), [1, 1, 1, 25]),
        _int64_tensor("one_i64", [1], [1, 1, 1, 1]),
        _int64_tensor("zero_i64", [0], [1, 1, 1, 1]),
        _int64_tensor("zero_pad", [0], [1]),
        _int64_tensor("shape1", [1], [1]),
        _int64_tensor("slice_hw_starts", [0, 0], [2]),
        _int64_tensor("slice_hw_ends", [25, 25], [2]),
        _int64_tensor("slice_hw_axes", [2, 3], [2]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 0, 5, 5], [8]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("outside_u8", [255], [1]),
        _u8_tensor("colors10", list(range(10)), [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ArgMax", ["input"], ["input_color_i64"], axis=1, keepdims=1, select_last_index=0),
        helper.make_node("Cast", ["input_color_i64"], ["input_color30_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Slice", ["input_color30_u8", "slice_hw_starts", "slice_hw_ends", "slice_hw_axes"], ["input_color_u8"]),
        helper.make_node("Cast", ["input_color_u8"], ["nonblack_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["nonblack_bool"], ["nonblack_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMax", ["nonblack_u8"], ["row_has"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["nonblack_u8"], ["col_has"], axes=[2], keepdims=1),
        helper.make_node("Equal", ["row_has", "zero_u8"], ["row_zero"]),
        helper.make_node("Equal", ["col_has", "zero_u8"], ["col_zero"]),
        helper.make_node("Cast", ["row_zero"], ["row_zero_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["col_zero"], ["col_zero_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ArgMax", ["row_zero_u8"], ["top_end"], axis=2, keepdims=1, select_last_index=0),
        helper.make_node("ArgMax", ["col_zero_u8"], ["left_end"], axis=3, keepdims=1, select_last_index=0),
        helper.make_node("Greater", ["row_idx", "top_end"], ["row_after_gap"]),
        helper.make_node("Cast", ["row_has"], ["row_has_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("And", ["row_after_gap", "row_has_bool"], ["bottom_start_candidates"]),
        helper.make_node("Cast", ["bottom_start_candidates"], ["bottom_start_scores"], to=onnx.TensorProto.UINT8),
        helper.make_node("ArgMax", ["bottom_start_scores"], ["bottom_start"], axis=2, keepdims=1, select_last_index=0),
        helper.make_node("ArgMax", ["row_has"], ["last_row"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("Add", ["last_row", "one_i64"], ["bottom_end"]),
        helper.make_node("Greater", ["col_idx", "left_end"], ["col_after_gap"]),
        helper.make_node("Cast", ["col_has"], ["col_has_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("And", ["col_after_gap", "col_has_bool"], ["right_start_candidates"]),
        helper.make_node("Cast", ["right_start_candidates"], ["right_start_scores"], to=onnx.TensorProto.UINT8),
        helper.make_node("ArgMax", ["right_start_scores"], ["right_start"], axis=3, keepdims=1, select_last_index=0),
        helper.make_node("ArgMax", ["col_has"], ["last_col"], axis=3, keepdims=1, select_last_index=1),
        helper.make_node("Add", ["last_col", "one_i64"], ["right_end"]),
        helper.make_node("Less", ["row_idx", "top_end"], ["top_rows"]),
        helper.make_node("LessOrEqual", ["bottom_start", "row_idx"], ["bottom_after_start"]),
        helper.make_node("Less", ["row_idx", "bottom_end"], ["bottom_before_end"]),
        helper.make_node("And", ["bottom_after_start", "bottom_before_end"], ["bottom_rows"]),
        helper.make_node("Less", ["col_idx", "left_end"], ["left_cols"]),
        helper.make_node("LessOrEqual", ["right_start", "col_idx"], ["right_after_start"]),
        helper.make_node("Less", ["col_idx", "right_end"], ["right_before_end"]),
        helper.make_node("And", ["right_after_start", "right_before_end"], ["right_cols"]),
        helper.make_node("And", ["top_rows", "left_cols"], ["mask_tl"]),
        helper.make_node("And", ["top_rows", "right_cols"], ["mask_tr"]),
        helper.make_node("And", ["bottom_rows", "left_cols"], ["mask_bl"]),
        helper.make_node("And", ["bottom_rows", "right_cols"], ["mask_br"]),
    ]

    _color_id(nodes, "mask_tl", "color_tl")
    _color_id(nodes, "mask_tr", "color_tr")
    _color_id(nodes, "mask_bl", "color_bl")
    _color_id(nodes, "mask_br", "color_br")
    _unique(nodes, "unique_tl", "color_tl", ["color_tr", "color_bl", "color_br"])
    _unique(nodes, "unique_tr", "color_tr", ["color_tl", "color_bl", "color_br"])
    _unique(nodes, "unique_bl", "color_bl", ["color_tl", "color_tr", "color_br"])
    _unique(nodes, "unique_br", "color_br", ["color_tl", "color_tr", "color_bl"])

    nodes.extend(
        [
            helper.make_node("And", ["mask_tl", "unique_tl"], ["target_tl"]),
            helper.make_node("And", ["mask_tr", "unique_tr"], ["target_tr"]),
            helper.make_node("And", ["mask_bl", "unique_bl"], ["target_bl"]),
            helper.make_node("And", ["mask_br", "unique_br"], ["target_br"]),
            helper.make_node("Or", ["target_tl", "target_tr"], ["target_top"]),
            helper.make_node("Or", ["target_bl", "target_br"], ["target_bottom"]),
            helper.make_node("Or", ["target_top", "target_bottom"], ["target_mask"]),
            helper.make_node("Or", ["unique_bl", "unique_br"], ["target_is_bottom"]),
            helper.make_node("Or", ["unique_tr", "unique_br"], ["target_is_right"]),
            helper.make_node("Where", ["target_is_bottom", "bottom_start", "zero_i64"], ["target_row_start"]),
            helper.make_node("Where", ["target_is_right", "right_start", "zero_i64"], ["target_col_start"]),
            helper.make_node("Sub", ["zero_i64", "target_row_start"], ["neg_row_start"]),
            helper.make_node("Sub", ["zero_i64", "target_col_start"], ["neg_col_start"]),
            helper.make_node("Reshape", ["neg_row_start", "shape1"], ["neg_row_start_1"]),
            helper.make_node("Reshape", ["neg_col_start", "shape1"], ["neg_col_start_1"]),
            helper.make_node("Reshape", ["target_row_start", "shape1"], ["target_row_start_1"]),
            helper.make_node("Reshape", ["target_col_start", "shape1"], ["target_col_start_1"]),
            helper.make_node(
                "Concat",
                [
                    "zero_pad",
                    "zero_pad",
                    "neg_row_start_1",
                    "neg_col_start_1",
                    "zero_pad",
                    "zero_pad",
                    "target_row_start_1",
                    "target_col_start_1",
                ],
                ["target_pads"],
                axis=0,
            ),
            helper.make_node("Where", ["target_mask", "input_color_u8", "outside_u8"], ["target_color_u8"]),
            helper.make_node("Pad", ["target_color_u8", "target_pads", "outside_u8"], ["shifted_color_u8"], mode="constant"),
            helper.make_node("Pad", ["shifted_color_u8", "pads_output", "outside_u8"], ["shifted_color30_u8"], mode="constant"),
            helper.make_node("Equal", ["colors10", "shifted_color30_u8"], ["output"]),
        ]
    )

    value_infos = [
        helper.make_tensor_value_info("input_color_u8", onnx.TensorProto.UINT8, [1, 1, 25, 25]),
        helper.make_tensor_value_info("shifted_color_u8", onnx.TensorProto.UINT8, [1, 1, 25, 25]),
        helper.make_tensor_value_info("shifted_color30_u8", onnx.TensorProto.UINT8, [1, 1, 30, 30]),
    ]
    graph = helper.make_graph(nodes, "task014_unique_block_crop_graph", [x], [y], initializers, value_info=value_infos)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
