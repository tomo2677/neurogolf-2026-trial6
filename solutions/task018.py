from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
GROW_STEPS = 15
TRANSFORMS = ("vflip", "transpose", "trans_hflip", "trans_vflip")


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _cast_f32(nodes: list[onnx.NodeProto], source: str, output: str) -> None:
    nodes.append(helper.make_node("Cast", [source], [output], to=onnx.TensorProto.FLOAT))


def _sum(nodes: list[onnx.NodeProto], source: str, output: str) -> None:
    nodes.append(helper.make_node("ReduceSum", [source], [output], axes=[0, 1, 2, 3], keepdims=0))


def _grow_component(nodes: list[onnx.NodeProto], seed: str, nonzero: str, prefix: str) -> str:
    current = seed
    for step in range(GROW_STEPS):
        _cast_f32(nodes, current, f"{prefix}_grow_f32_{step}")
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    [f"{prefix}_grow_f32_{step}", "cross_kernel"],
                    [f"{prefix}_dilated_score_{step}"],
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                ),
                helper.make_node("Greater", [f"{prefix}_dilated_score_{step}", "zero_f32"], [f"{prefix}_dilated_{step}"]),
                helper.make_node("And", [f"{prefix}_dilated_{step}", nonzero], [f"{prefix}_grown_{step}"]),
            ]
        )
        current = f"{prefix}_grown_{step}"
    return current


def _transform_tensor(nodes: list[onnx.NodeProto], source: str, name: str, transform: str) -> str:
    if transform == "id":
        nodes.append(helper.make_node("Identity", [source], [name]))
        return name

    current = source
    if transform.startswith("trans"):
        nodes.append(helper.make_node("Transpose", [current], [f"{name}_t"], perm=[0, 1, 3, 2]))
        current = f"{name}_t"

    if transform in {"hflip", "hvflip", "trans_hflip", "trans_hvflip"}:
        nodes.append(helper.make_node("Gather", [current, "reverse_idx"], [f"{name}_h"], axis=3))
        current = f"{name}_h"

    if transform in {"vflip", "hvflip", "trans_vflip", "trans_hvflip"}:
        nodes.append(helper.make_node("Gather", [current, "reverse_idx"], [f"{name}_v"], axis=2))
        current = f"{name}_v"

    if current != name:
        nodes.append(helper.make_node("Identity", [current], [name]))
    return name


def _transform_point(nodes: list[onnx.NodeProto], row: str, col: str, prefix: str, transform: str) -> tuple[str, str]:
    if transform == "id":
        nodes.extend(
            [
                helper.make_node("Identity", [row], [f"{prefix}_tr"]),
                helper.make_node("Identity", [col], [f"{prefix}_tc"]),
            ]
        )
    elif transform == "hflip":
        nodes.extend(
            [
                helper.make_node("Identity", [row], [f"{prefix}_tr"]),
                helper.make_node("Sub", ["last_i64", col], [f"{prefix}_tc"]),
            ]
        )
    elif transform == "vflip":
        nodes.extend(
            [
                helper.make_node("Sub", ["last_i64", row], [f"{prefix}_tr"]),
                helper.make_node("Identity", [col], [f"{prefix}_tc"]),
            ]
        )
    elif transform == "hvflip":
        nodes.extend(
            [
                helper.make_node("Sub", ["last_i64", row], [f"{prefix}_tr"]),
                helper.make_node("Sub", ["last_i64", col], [f"{prefix}_tc"]),
            ]
        )
    elif transform == "transpose":
        nodes.extend(
            [
                helper.make_node("Identity", [col], [f"{prefix}_tr"]),
                helper.make_node("Identity", [row], [f"{prefix}_tc"]),
            ]
        )
    elif transform == "trans_hflip":
        nodes.extend(
            [
                helper.make_node("Identity", [col], [f"{prefix}_tr"]),
                helper.make_node("Sub", ["last_i64", row], [f"{prefix}_tc"]),
            ]
        )
    elif transform == "trans_vflip":
        nodes.extend(
            [
                helper.make_node("Sub", ["last_i64", col], [f"{prefix}_tr"]),
                helper.make_node("Identity", [row], [f"{prefix}_tc"]),
            ]
        )
    elif transform == "trans_hvflip":
        nodes.extend(
            [
                helper.make_node("Sub", ["last_i64", col], [f"{prefix}_tr"]),
                helper.make_node("Sub", ["last_i64", row], [f"{prefix}_tc"]),
            ]
        )
    else:
        raise ValueError(transform)
    return f"{prefix}_tr", f"{prefix}_tc"


def _dynamic_shift(nodes: list[onnx.NodeProto], source: str, dr: str, dc: str, output: str) -> None:
    raise NotImplementedError("Use _static_shift for public-rule compliant fixed-shape shifts.")


def _static_shift(nodes: list[onnx.NodeProto], source: str, dr: str, dc: str, output: str, channels: int) -> None:
    if channels == 10:
        flat_shape = "shape_flat_10x900"
        index_shape = "shape_index_10x900"
        output_shape = "shape_1x10x30x30"
    elif channels == 1:
        flat_shape = "shape_flat_1x900"
        index_shape = "shape_index_1x900"
        output_shape = "shape_1x1x30x30"
    else:
        raise ValueError(channels)

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
            helper.make_node("Mul", [f"{output}_safe_r", "width_i64"], [f"{output}_safe_r_offset"]),
            helper.make_node("Add", [f"{output}_safe_r_offset", f"{output}_safe_c"], [f"{output}_safe_spatial"]),
            helper.make_node("Reshape", [f"{output}_safe_spatial", "shape_index_1x900"], [f"{output}_safe_spatial_flat"]),
            helper.make_node("Expand", [f"{output}_safe_spatial_flat", index_shape], [f"{output}_indices"]),
            helper.make_node("Reshape", [source, flat_shape], [f"{output}_source_flat"]),
            helper.make_node("GatherElements", [f"{output}_source_flat", f"{output}_indices"], [f"{output}_shifted_flat"], axis=2),
            helper.make_node("Reshape", [f"{output}_shifted_flat", output_shape], [f"{output}_shifted"]),
            helper.make_node("Cast", [f"{output}_in_bounds"], [f"{output}_in_bounds_f32"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Mul", [f"{output}_shifted", f"{output}_in_bounds_f32"], [output]),
        ]
    )


def _flat_position(nodes: list[onnx.NodeProto], flat_index: str, prefix: str) -> tuple[str, str]:
    nodes.extend(
        [
            helper.make_node("Div", [flat_index, "width_i64"], [f"{prefix}_row"]),
            helper.make_node("Mod", [flat_index, "width_i64"], [f"{prefix}_col"]),
        ]
    )
    return f"{prefix}_row", f"{prefix}_col"


def _component_outputs(
    nodes: list[onnx.NodeProto],
    comp_mask: str,
    target_rows: list[str],
    target_cols: list[str],
    prefix: str,
) -> list[str]:
    outputs: list[str] = []
    nodes.extend(
        [
            helper.make_node("And", [comp_mask, "base_mask"], [f"{prefix}_base_bool"]),
            helper.make_node("Not", ["base_mask"], [f"{prefix}_not_base"]),
            helper.make_node("And", [comp_mask, f"{prefix}_not_base"], [f"{prefix}_marker_bool"]),
            helper.make_node("And", [comp_mask, "anchor_color_mask"], [f"{prefix}_anchor_bool"]),
            helper.make_node("Where", [comp_mask, "input", "zero_f32"], [f"{prefix}_onehot"]),
        ]
    )
    _cast_f32(nodes, comp_mask, f"{prefix}_mask_f32")
    _cast_f32(nodes, f"{prefix}_base_bool", f"{prefix}_base_f32")
    _cast_f32(nodes, f"{prefix}_marker_bool", f"{prefix}_marker_f32")
    _sum(nodes, f"{prefix}_mask_f32", f"{prefix}_count")
    _sum(nodes, f"{prefix}_base_f32", f"{prefix}_base_count")
    _sum(nodes, f"{prefix}_marker_f32", f"{prefix}_marker_count")

    nodes.extend(
        [
            helper.make_node("Cast", [f"{prefix}_anchor_bool"], [f"{prefix}_anchor_f32"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Reshape", [f"{prefix}_anchor_f32", "shape_flat900"], [f"{prefix}_anchor_flat"]),
            helper.make_node("ArgMax", [f"{prefix}_anchor_flat"], [f"{prefix}_anchor_index"], axis=0, keepdims=1),
        ]
    )
    source_row, source_col = _flat_position(nodes, f"{prefix}_anchor_index", f"{prefix}_source_anchor")

    for transform in TRANSFORMS:
        transformed_onehot = _transform_tensor(nodes, f"{prefix}_onehot", f"{prefix}_{transform}_onehot", transform)
        transformed_base = _transform_tensor(nodes, f"{prefix}_base_f32", f"{prefix}_{transform}_base", transform)
        transformed_row, transformed_col = _transform_point(
            nodes, source_row, source_col, f"{prefix}_{transform}_source_anchor", transform
        )

        for target_index, (target_row, target_col) in enumerate(zip(target_rows, target_cols)):
            place = f"{prefix}_{transform}_target{target_index}"
            nodes.extend(
                [
                    helper.make_node("Sub", [target_row, transformed_row], [f"{place}_dr"]),
                    helper.make_node("Sub", [target_col, transformed_col], [f"{place}_dc"]),
                ]
            )
            _static_shift(nodes, transformed_onehot, f"{place}_dr", f"{place}_dc", f"{place}_shifted_onehot", 10)
            _static_shift(nodes, transformed_base, f"{place}_dr", f"{place}_dc", f"{place}_shifted_base", 1)

            nodes.extend(
                [
                    helper.make_node(
                        "ReduceMax", [f"{place}_shifted_onehot"], [f"{place}_shifted_mask"], axes=[1], keepdims=1
                    ),
                    helper.make_node("Mul", [f"{place}_shifted_mask", "valid_cell_f32"], [f"{place}_inside_grid"]),
                    helper.make_node("Mul", [f"{place}_shifted_base", "input0"], [f"{place}_base_on_zero"]),
                    helper.make_node("Greater", [f"{place}_shifted_base", "zero_f32"], [f"{place}_base_bool"]),
                    helper.make_node(
                        "Where", [f"{place}_base_bool", "zero_f32", f"{place}_shifted_onehot"], [f"{place}_markers"]
                    ),
                    helper.make_node("Mul", [f"{place}_markers", "input"], [f"{place}_marker_matches"]),
                    helper.make_node(
                        "ReduceMax", [f"{place}_markers"], [f"{place}_marker_mask"], axes=[1], keepdims=1
                    ),
                    helper.make_node("Mul", [f"{place}_marker_mask", "source_mask_all_f32"], [f"{place}_marker_source_overlap"]),
                ]
            )
            _sum(nodes, f"{place}_shifted_mask", f"{place}_shifted_count")
            _sum(nodes, f"{place}_inside_grid", f"{place}_inside_count")
            _sum(nodes, f"{place}_base_on_zero", f"{place}_base_zero_count")
            _sum(nodes, f"{place}_marker_matches", f"{place}_marker_match_count")
            _sum(nodes, f"{place}_marker_source_overlap", f"{place}_marker_source_overlap_count")

            nodes.extend(
                [
                    helper.make_node("Equal", [f"{place}_shifted_count", f"{prefix}_count"], [f"{place}_same_count"]),
                    helper.make_node("Equal", [f"{place}_inside_count", f"{prefix}_count"], [f"{place}_inside_ok"]),
                    helper.make_node(
                        "Equal", [f"{place}_base_zero_count", f"{prefix}_base_count"], [f"{place}_base_ok"]
                    ),
                    helper.make_node(
                        "Equal", [f"{place}_marker_match_count", f"{prefix}_marker_count"], [f"{place}_marker_ok"]
                    ),
                    helper.make_node(
                        "Equal", [f"{place}_marker_source_overlap_count", "zero_f32"], [f"{place}_outside_source_ok"]
                    ),
                    helper.make_node("And", [f"{place}_same_count", f"{place}_inside_ok"], [f"{place}_valid_a"]),
                    helper.make_node("And", [f"{place}_base_ok", f"{place}_marker_ok"], [f"{place}_valid_b"]),
                    helper.make_node("And", [f"{place}_valid_a", f"{place}_valid_b"], [f"{place}_valid_c"]),
                    helper.make_node(
                        "And", [f"{place}_valid_c", f"{place}_outside_source_ok"], [f"{place}_valid"]
                    ),
                    helper.make_node("Where", [f"{place}_valid", f"{place}_shifted_onehot", "zero_f32"], [f"{place}_output"]),
                ]
            )
            outputs.append(f"{place}_output")

    return outputs


def build_model() -> onnx.ModelProto:
    x, y = make_io_value_infos()

    initializers = [
        _int64_tensor("slice_nonzero_start", [1], [1]),
        _int64_tensor("slice_nonzero_end", [10], [1]),
        _int64_tensor("slice_zero_start", [0], [1]),
        _int64_tensor("slice_one_end", [1], [1]),
        _int64_tensor("axis_channel", [1], [1]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("zero_i64", [0], [1]),
        _int64_tensor("neg_one_i64", [-1], [1]),
        _int64_tensor("size_i64", [SIZE], [1]),
        _int64_tensor("last_i64", [SIZE - 1], [1]),
        _int64_tensor("width_i64", [SIZE], [1]),
        _int64_tensor("k2", [2], [1]),
        _int64_tensor("shape1111", [1, 1, 1, 1], [4]),
        _int64_tensor("shape_flat900", [SIZE * SIZE], [1]),
        _int64_tensor("shape_index_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_index_10x900", [1, 10, SIZE * SIZE], [3]),
        _int64_tensor("shape_flat_1x900", [1, 1, SIZE * SIZE], [3]),
        _int64_tensor("shape_flat_10x900", [1, 10, SIZE * SIZE], [3]),
        _int64_tensor("shape_1x1x30x30", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("shape_1x10x30x30", [1, 10, SIZE, SIZE], [4]),
        _int64_tensor("reverse_idx", list(reversed(range(SIZE))), [SIZE]),
        _int64_tensor("flat_index", list(range(SIZE * SIZE)), [SIZE * SIZE]),
        _int64_tensor("row_grid_i64", [r for r in range(SIZE) for _ in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int64_tensor("col_grid_i64", [c for _ in range(SIZE) for c in range(SIZE)], [1, 1, SIZE, SIZE]),
        _int64_tensor("color_ids", list(range(1, 10)), [9]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _f32_tensor("black_pixel", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 10, 1, 1]),
        _f32_tensor("cross_kernel", [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0], [1, 1, 3, 3]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["valid_cell_score"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["valid_cell_score", "zero_f32"], ["valid_cell"]),
        helper.make_node("Cast", ["valid_cell"], ["valid_cell_f32"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Slice", ["input", "slice_nonzero_start", "slice_nonzero_end", "axis_channel"], ["input_nonzero"]),
        helper.make_node("Slice", ["input", "slice_zero_start", "slice_one_end", "axis_channel"], ["input0"]),
        helper.make_node("ReduceMax", ["input_nonzero"], ["nonzero_score"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["nonzero_score", "zero_f32"], ["nonzero_mask"]),
        helper.make_node("ReduceSum", ["input_nonzero"], ["color_counts"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("ArgMax", ["color_counts"], ["base_idx"], axis=0, keepdims=1),
        helper.make_node("Add", ["base_idx", "one_i64"], ["base_color"]),
        helper.make_node("ArgMax", ["input"], ["color_grid"], axis=1, keepdims=1),
        helper.make_node("Reshape", ["base_color", "shape1111"], ["base_color1111"]),
        helper.make_node("Equal", ["color_grid", "base_color1111"], ["base_mask"]),
        helper.make_node("Greater", ["color_counts", "zero_f32"], ["color_present"]),
        helper.make_node("Equal", ["color_ids", "base_color"], ["base_color_vector"]),
        helper.make_node("Not", ["base_color_vector"], ["not_base_color_vector"]),
        helper.make_node("And", ["color_present", "not_base_color_vector"], ["marker_color_vector"]),
        helper.make_node("Cast", ["marker_color_vector"], ["marker_color_score"], to=onnx.TensorProto.FLOAT),
        helper.make_node("ArgMax", ["marker_color_score"], ["anchor_idx"], axis=0, keepdims=1),
        helper.make_node("Add", ["anchor_idx", "one_i64"], ["anchor_color"]),
        helper.make_node("Reshape", ["anchor_color", "shape1111"], ["anchor_color1111"]),
        helper.make_node("Equal", ["color_grid", "anchor_color1111"], ["anchor_color_mask"]),
    ]

    source_mask_all = _grow_component(nodes, "base_mask", "nonzero_mask", "source_all")
    nodes.append(helper.make_node("Identity", [source_mask_all], ["source_mask_all"]))
    _cast_f32(nodes, "source_mask_all", "source_mask_all_f32")

    nodes.extend(
        [
            helper.make_node("Cast", ["base_mask"], ["base_mask_f32"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Reshape", ["base_mask_f32", "shape_flat900"], ["base_mask_flat"]),
            helper.make_node("ArgMax", ["base_mask_flat"], ["first_base_index"], axis=0, keepdims=1),
            helper.make_node("Equal", ["flat_index", "first_base_index"], ["first_base_flat"]),
            helper.make_node("Reshape", ["first_base_flat", "shape1111"], ["first_base_bad_shape"]),
        ]
    )
    # Reshape above makes [1,1,1,1]; rebuild the seed with a 30x30 shape.
    nodes.pop()
    nodes.append(helper.make_node("Reshape", ["first_base_flat", "seed_shape"], ["first_base_seed"]))
    initializers.append(_int64_tensor("seed_shape", [1, 1, SIZE, SIZE], [4]))

    comp1 = _grow_component(nodes, "first_base_seed", "nonzero_mask", "comp1")
    nodes.append(helper.make_node("Identity", [comp1], ["comp1_mask"]))
    nodes.extend(
        [
            helper.make_node("Not", ["comp1_mask"], ["not_comp1_mask"]),
            helper.make_node("And", ["source_mask_all", "not_comp1_mask"], ["comp2_mask"]),
            helper.make_node("Not", ["source_mask_all"], ["not_source_mask"]),
            helper.make_node("And", ["anchor_color_mask", "not_source_mask"], ["target_anchor_mask"]),
            helper.make_node("Cast", ["target_anchor_mask"], ["target_anchor_f32"], to=onnx.TensorProto.FLOAT),
            helper.make_node("Reshape", ["target_anchor_f32", "shape_flat900"], ["target_anchor_flat"]),
            helper.make_node("TopK", ["target_anchor_flat", "k2"], ["target_anchor_values", "target_anchor_indices"], axis=0),
            helper.make_node("Split", ["target_anchor_indices"], ["target_anchor_index0", "target_anchor_index1"], axis=0, split=[1, 1]),
        ]
    )

    target0_row, target0_col = _flat_position(nodes, "target_anchor_index0", "target0")
    target1_row, target1_col = _flat_position(nodes, "target_anchor_index1", "target1")
    candidate_outputs = []
    candidate_outputs.extend(
        _component_outputs(nodes, "comp1_mask", [target0_row, target1_row], [target0_col, target1_col], "comp1")
    )
    candidate_outputs.extend(
        _component_outputs(nodes, "comp2_mask", [target0_row, target1_row], [target0_col, target1_col], "comp2")
    )
    nodes.extend(
        [
            helper.make_node("Max", candidate_outputs, ["placed_nonzero"]),
            helper.make_node("ReduceMax", ["placed_nonzero"], ["placed_mask"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["placed_mask", "zero_f32"], ["placed_bool"]),
            helper.make_node("Not", ["placed_bool"], ["not_placed_bool"]),
            helper.make_node("And", ["valid_cell", "not_placed_bool"], ["blank_bool"]),
            helper.make_node("Where", ["blank_bool", "black_pixel", "zero_f32"], ["blank_output"]),
            helper.make_node("Max", ["placed_nonzero", "blank_output"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task018_template_copy_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
