from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


SIZE = 30
GRID_SIZE = 30
GROW_STEPS = 8
TRANSFORMS = ("vflip", "transpose", "trans_hflip", "trans_vflip")


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _int32_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT32, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _f16_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT16, dims, values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


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


def _cast_f32(nodes: list[onnx.NodeProto], source: str, output: str) -> None:
    nodes.append(helper.make_node("Cast", [source], [output], to=onnx.TensorProto.FLOAT16))


def _sum(nodes: list[onnx.NodeProto], source: str, output: str) -> None:
    nodes.append(helper.make_node("ReduceSum", [source], [output], axes=[0, 1, 2, 3], keepdims=0))


def _grow_component(nodes: list[onnx.NodeProto], seed: str, nonzero: str, prefix: str) -> str:
    nodes.append(helper.make_node("Cast", [seed], [f"{prefix}_seed_u8"], to=onnx.TensorProto.UINT8))
    current = f"{prefix}_seed_u8"
    for step in range(GROW_STEPS):
        nodes.extend(
            [
                helper.make_node(
                    "MaxPool",
                    [current],
                    [f"{prefix}_dilated_u8_{step}"],
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                ),
                helper.make_node("Min", [f"{prefix}_dilated_u8_{step}", nonzero], [f"{prefix}_grown_u8_{step}"]),
            ]
        )
        current = f"{prefix}_grown_u8_{step}"
    nodes.append(helper.make_node("Greater", [current, "zero_u8"], [f"{prefix}_grown_bool"]))
    return f"{prefix}_grown_bool"


def _transform_tensor(nodes: list[onnx.NodeProto], source: str, name: str, transform: str) -> str:
    if transform == "id":
        return source

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

    return current


def _transform_point(nodes: list[onnx.NodeProto], row: str, col: str, prefix: str, transform: str) -> tuple[str, str]:
    if transform == "id":
        return row, col
    elif transform == "hflip":
        nodes.append(helper.make_node("Sub", ["last_i64", col], [f"{prefix}_tc"]))
        return row, f"{prefix}_tc"
    elif transform == "vflip":
        nodes.append(helper.make_node("Sub", ["last_i64", row], [f"{prefix}_tr"]))
        return f"{prefix}_tr", col
    elif transform == "hvflip":
        nodes.extend(
            [
                helper.make_node("Sub", ["last_i64", row], [f"{prefix}_tr"]),
                helper.make_node("Sub", ["last_i64", col], [f"{prefix}_tc"]),
            ]
        )
        return f"{prefix}_tr", f"{prefix}_tc"
    elif transform == "transpose":
        return col, row
    elif transform == "trans_hflip":
        nodes.append(helper.make_node("Sub", ["last_i64", row], [f"{prefix}_tc"]))
        return col, f"{prefix}_tc"
    elif transform == "trans_vflip":
        nodes.append(helper.make_node("Sub", ["last_i64", col], [f"{prefix}_tr"]))
        return f"{prefix}_tr", row
    elif transform == "trans_hvflip":
        nodes.extend(
            [
                helper.make_node("Sub", ["last_i64", col], [f"{prefix}_tr"]),
                helper.make_node("Sub", ["last_i64", row], [f"{prefix}_tc"]),
            ]
        )
        return f"{prefix}_tr", f"{prefix}_tc"
    else:
        raise ValueError(transform)


def _static_shift_color_base(
    nodes: list[onnx.NodeProto],
    color_flat_source: str,
    dr: str,
    dc: str,
    color_output: str,
) -> None:
    prefix = f"{color_output}_shared"
    nodes.extend(
        [
            helper.make_node("Sub", ["row_grid_i32", dr], [f"{prefix}_src_r"]),
            helper.make_node("Sub", ["col_grid_i32", dc], [f"{prefix}_src_c"]),
            helper.make_node("Mul", [dr, "width_i32"], [f"{prefix}_dr_offset"]),
            helper.make_node("Add", [f"{prefix}_dr_offset", dc], [f"{prefix}_offset"]),
            helper.make_node("Sub", ["flat_grid_i32", f"{prefix}_offset"], [f"{prefix}_src_spatial"]),
            helper.make_node("Greater", [f"{prefix}_src_r", "neg_one_i32"], [f"{prefix}_r_nonneg"]),
            helper.make_node("Less", [f"{prefix}_src_r", "size_i32"], [f"{prefix}_r_lt_size"]),
            helper.make_node("Greater", [f"{prefix}_src_c", "neg_one_i32"], [f"{prefix}_c_nonneg"]),
            helper.make_node("Less", [f"{prefix}_src_c", "size_i32"], [f"{prefix}_c_lt_size"]),
            helper.make_node("And", [f"{prefix}_r_nonneg", f"{prefix}_r_lt_size"], [f"{prefix}_r_ok"]),
            helper.make_node("And", [f"{prefix}_c_nonneg", f"{prefix}_c_lt_size"], [f"{prefix}_c_ok"]),
            helper.make_node("And", [f"{prefix}_r_ok", f"{prefix}_c_ok"], [f"{prefix}_in_bounds"]),
            helper.make_node("Where", [f"{prefix}_in_bounds", f"{prefix}_src_spatial", "zero_i32"], [f"{prefix}_safe_spatial"]),
            helper.make_node("Gather", [color_flat_source, f"{prefix}_safe_spatial"], [f"{prefix}_color_shifted"], axis=0),
            helper.make_node("Where", [f"{prefix}_in_bounds", f"{prefix}_color_shifted", "zero_u8"], [color_output]),
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
    source_mask: str,
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
            helper.make_node("Where", [comp_mask, "input_color_u8", "zero_u8"], [f"{prefix}_color"]),
        ]
    )
    _cast_f32(nodes, comp_mask, f"{prefix}_mask_f32")
    _cast_f32(nodes, f"{prefix}_marker_bool", f"{prefix}_marker_f32")
    _sum(nodes, f"{prefix}_mask_f32", f"{prefix}_count")
    _sum(nodes, f"{prefix}_marker_f32", f"{prefix}_marker_count")

    nodes.extend(
        [
            helper.make_node("Cast", [f"{prefix}_anchor_bool"], [f"{prefix}_anchor_f16"], to=onnx.TensorProto.FLOAT16),
            helper.make_node("Reshape", [f"{prefix}_anchor_f16", "shape_flat900"], [f"{prefix}_anchor_flat"]),
            helper.make_node("ArgMax", [f"{prefix}_anchor_flat"], [f"{prefix}_anchor_index"], axis=0, keepdims=1),
        ]
    )
    source_row, source_col = _flat_position(nodes, f"{prefix}_anchor_index", f"{prefix}_source_anchor")

    for transform in TRANSFORMS:
        transformed_color = _transform_tensor(nodes, f"{prefix}_color", f"{prefix}_{transform}_color", transform)
        nodes.extend(
            [
                helper.make_node("Reshape", [transformed_color, "shape_flat900"], [f"{prefix}_{transform}_color_flat"]),
            ]
        )
        transformed_row, transformed_col = _transform_point(
            nodes, source_row, source_col, f"{prefix}_{transform}_source_anchor", transform
        )

        for target_index, (target_row, target_col) in enumerate(zip(target_rows, target_cols)):
            place = f"{prefix}_{transform}_target{target_index}"
            nodes.extend(
                [
                    helper.make_node("Sub", [target_row, transformed_row], [f"{place}_dr"]),
                    helper.make_node("Sub", [target_col, transformed_col], [f"{place}_dc"]),
                    helper.make_node("Cast", [f"{place}_dr"], [f"{place}_dr_i32"], to=onnx.TensorProto.INT32),
                    helper.make_node("Cast", [f"{place}_dc"], [f"{place}_dc_i32"], to=onnx.TensorProto.INT32),
                ]
            )
            _static_shift_color_base(
                nodes,
                f"{prefix}_{transform}_color_flat",
                f"{place}_dr_i32",
                f"{place}_dc_i32",
                f"{place}_shifted_color",
            )

            nodes.extend(
                [
                    helper.make_node("Greater", [f"{place}_shifted_color", "zero_u8"], [f"{place}_shifted_mask_bool"]),
                    helper.make_node("Equal", [f"{place}_shifted_color", "base_color1111"], [f"{place}_shifted_base_color_bool"]),
                    helper.make_node("Not", [f"{place}_shifted_base_color_bool"], [f"{place}_not_base_bool"]),
                    helper.make_node("And", [f"{place}_shifted_mask_bool", f"{place}_not_base_bool"], [f"{place}_marker_mask_bool"]),
                    helper.make_node("Equal", [f"{place}_shifted_color", "input_color_u8"], [f"{place}_marker_color_equal"]),
                    helper.make_node("And", [f"{place}_marker_mask_bool", f"{place}_marker_color_equal"], [f"{place}_marker_matches_bool"]),
                    helper.make_node("Cast", [f"{place}_marker_matches_bool"], [f"{place}_marker_matches"], to=onnx.TensorProto.FLOAT16),
                ]
            )
            _sum(nodes, f"{place}_marker_matches", f"{place}_marker_match_count")

            nodes.extend(
                [
                    helper.make_node(
                        "Equal", [f"{place}_marker_match_count", f"{prefix}_marker_count"], [f"{place}_marker_ok"]
                    ),
                    helper.make_node("Where", [f"{place}_marker_ok", f"{place}_shifted_color", "zero_u8"], [f"{place}_output"]),
                ]
            )
            outputs.append(f"{place}_output")

    return outputs


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("slice_nonzero_start", [1], [1]),
        _int64_tensor("slice_nonzero_end", [10], [1]),
        _int64_tensor("slice_zero_start", [0], [1]),
        _int64_tensor("slice_one_end", [1], [1]),
        _int64_tensor("axis0", [0], [1]),
        _int64_tensor("axis_channel", [1], [1]),
        _int64_tensor("one_i64", [1], [1]),
        _int64_tensor("last_i64", [SIZE - 1], [1]),
        _int64_tensor("width_i64", [SIZE], [1]),
        _int32_tensor("zero_i32", [0], [1]),
        _int32_tensor("neg_one_i32", [-1], [1]),
        _int32_tensor("size_i32", [SIZE], [1]),
        _int32_tensor("width_i32", [SIZE], [1]),
        _int64_tensor("k2", [2], [1]),
        _int64_tensor("crop_hw_start", [0, 0], [2]),
        _int64_tensor("crop_hw_end", [SIZE, SIZE], [2]),
        _int64_tensor("crop_hw_axes", [2, 3], [2]),
        _int64_tensor("pads_to_grid", [0, 0, 0, 0, 0, 0, GRID_SIZE - SIZE, GRID_SIZE - SIZE], [8]),
        _int64_tensor("shape1111", [1, 1, 1, 1], [4]),
        _int64_tensor("shape_flat900", [SIZE * SIZE], [1]),
        _int64_tensor("shape_1x1x30x30", [1, 1, SIZE, SIZE], [4]),
        _int64_tensor("reverse_idx", list(reversed(range(SIZE))), [SIZE]),
        _int64_tensor("flat_index", list(range(SIZE * SIZE)), [SIZE * SIZE]),
        _int32_tensor("row_grid_i32", list(range(SIZE)), [1, 1, SIZE, 1]),
        _int32_tensor("col_grid_i32", list(range(SIZE)), [1, 1, 1, SIZE]),
        _int32_tensor("flat_grid_i32", list(range(SIZE * SIZE)), [1, 1, SIZE, SIZE]),
        _int64_tensor("color_ids", list(range(1, 10)), [9]),
        _f32_tensor("zero_f32", [0.0], [1]),
        _u8_tensor("zero_u8", [0], [1]),
        _u8_tensor("invalid_u8", [255], [1]),
        _u8_tensor("colors10_u8", list(range(10)), [1, 10, 1, 1]),
        _f32_tensor("color_conv_w", [float(i) for i in range(10)], [1, 10, 1, 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["valid_cell_score"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["valid_cell_score", "zero_f32"], ["valid_cell"]),
        helper.make_node("ReduceSum", ["input"], ["color_counts10"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("Slice", ["color_counts10", "slice_nonzero_start", "slice_nonzero_end", "axis0"], ["color_counts"]),
        helper.make_node("ArgMax", ["color_counts"], ["base_idx"], axis=0, keepdims=1),
        helper.make_node("Add", ["base_idx", "one_i64"], ["base_color"]),
        helper.make_node("Cast", ["base_color"], ["base_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Conv", ["input", "color_conv_w"], ["input_color_f32"]),
        helper.make_node("Cast", ["input_color_f32"], ["input_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Greater", ["input_color_u8", "zero_u8"], ["nonzero_mask"]),
        helper.make_node("Cast", ["nonzero_mask"], ["nonzero_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Reshape", ["base_color_u8", "shape1111"], ["base_color1111"]),
        helper.make_node("Equal", ["input_color_u8", "base_color1111"], ["base_mask"]),
        helper.make_node("Greater", ["color_counts", "zero_f32"], ["color_present"]),
        helper.make_node("Equal", ["color_ids", "base_color"], ["base_color_vector"]),
        helper.make_node("Not", ["base_color_vector"], ["not_base_color_vector"]),
        helper.make_node("And", ["color_present", "not_base_color_vector"], ["marker_color_vector"]),
        helper.make_node("Cast", ["marker_color_vector"], ["marker_color_score"], to=onnx.TensorProto.FLOAT),
        helper.make_node("ArgMax", ["marker_color_score"], ["anchor_idx"], axis=0, keepdims=1),
        helper.make_node("Add", ["anchor_idx", "one_i64"], ["anchor_color"]),
        helper.make_node("Cast", ["anchor_color"], ["anchor_color_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Reshape", ["anchor_color_u8", "shape1111"], ["anchor_color1111"]),
        helper.make_node("Equal", ["input_color_u8", "anchor_color1111"], ["anchor_color_mask"]),
    ]

    source_mask_all = _grow_component(nodes, "base_mask", "nonzero_u8", "source_all")

    nodes.extend(
        [
            helper.make_node("Cast", ["base_mask"], ["base_mask_f16"], to=onnx.TensorProto.FLOAT16),
            helper.make_node("Reshape", ["base_mask_f16", "shape_flat900"], ["base_mask_flat"]),
            helper.make_node("ArgMax", ["base_mask_flat"], ["first_base_index"], axis=0, keepdims=1),
            helper.make_node("Equal", ["flat_index", "first_base_index"], ["first_base_flat"]),
            helper.make_node("Reshape", ["first_base_flat", "shape1111"], ["first_base_bad_shape"]),
        ]
    )
    # Reshape above makes [1,1,1,1]; rebuild the seed with a 30x30 shape.
    nodes.pop()
    nodes.append(helper.make_node("Reshape", ["first_base_flat", "seed_shape"], ["first_base_seed"]))
    initializers.append(_int64_tensor("seed_shape", [1, 1, SIZE, SIZE], [4]))

    comp1 = _grow_component(nodes, "first_base_seed", "nonzero_u8", "comp1")
    comp1_mask = comp1
    nodes.extend(
        [
            helper.make_node("Not", [comp1_mask], ["not_comp1_mask"]),
            helper.make_node("And", [source_mask_all, "not_comp1_mask"], ["comp2_mask"]),
            helper.make_node("Not", [source_mask_all], ["not_source_mask"]),
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
        _component_outputs(nodes, comp1_mask, source_mask_all, [target0_row, target1_row], [target0_col, target1_col], "comp1")
    )
    candidate_outputs.extend(
        _component_outputs(nodes, "comp2_mask", source_mask_all, [target0_row, target1_row], [target0_col, target1_col], "comp2")
    )
    nodes.extend(
        [
            helper.make_node("Max", candidate_outputs, ["color24"]),
        ]
    )

    for node in nodes:
        for index, input_name in enumerate(node.input):
            if input_name == "input":
                node.input[index] = "input24"
    nodes.insert(0, helper.make_node("Slice", ["input", "crop_hw_start", "crop_hw_end", "crop_hw_axes"], ["input24"]))
    nodes.extend(
        [
            helper.make_node("Where", ["valid_cell", "color24", "invalid_u8"], ["color24_valid"]),
            helper.make_node("Pad", ["color24_valid", "pads_to_grid", "invalid_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["colors10_u8", "color30"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task018_template_copy_graph", [x], [y], initializers)
    _dedupe_initializers(graph)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
