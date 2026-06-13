from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("blue_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("blue_ends", [1, 2, 10, 10], [4]),
        _int64_tensor("inner_starts", [0, 0, 1, 1], [4]),
        _int64_tensor("inner_ends", [1, 1, 10, 10], [4]),
        _int64_tensor("rev10_starts", [9, 9], [2]),
        _int64_tensor("rev10_ends", [-11, -11], [2]),
        _int64_tensor("rev9_starts", [8, 8], [2]),
        _int64_tensor("rev9_ends", [-10, -10], [2]),
        _int64_tensor("rev_axes", [2, 3], [2]),
        _int64_tensor("rev_steps", [-1, -1], [2]),
        _int64_tensor("pads_rot10", [0, 0, 1, 1, 0, 0, 0, 0], [8]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 7, 20, 20], [8]),
        _int64_tensor("sum_axes4", [0, 1, 2, 3], [4]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "blue_starts", "blue_ends"], ["blue10"]),
        helper.make_node("Cast", ["blue10"], ["blue_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Slice", ["blue_bool", "rev10_starts", "rev10_ends", "rev_axes", "rev_steps"], ["rot9"]),
        helper.make_node("Slice", ["blue_bool", "inner_starts", "inner_ends"], ["inner9"]),
        helper.make_node("Slice", ["inner9", "rev9_starts", "rev9_ends", "rev_axes", "rev_steps"], ["rot10_inner"]),
        helper.make_node("Pad", ["rot10_inner", "pads_rot10"], ["rot10"], mode="constant"),
        helper.make_node("And", ["blue_bool", "rot9"], ["overlap9_bool"]),
        helper.make_node("And", ["inner9", "rot10_inner"], ["overlap10_bool"]),
        helper.make_node("Cast", ["overlap9_bool"], ["overlap9_cells"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("Cast", ["overlap10_bool"], ["overlap10_cells"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("ReduceSum", ["overlap9_cells", "sum_axes4"], ["overlap9"], keepdims=0),
        helper.make_node("ReduceSum", ["overlap10_cells", "sum_axes4"], ["overlap10"], keepdims=0),
        helper.make_node("Greater", ["overlap10", "overlap9"], ["use10"]),
        helper.make_node("Xor", ["rot9", "rot10"], ["rot_diff"]),
        helper.make_node("And", ["use10", "rot_diff"], ["selected_delta"]),
        helper.make_node("Xor", ["rot9", "selected_delta"], ["selected_rot"]),
        helper.make_node("Not", ["blue_bool"], ["not_blue"]),
        helper.make_node("And", ["selected_rot", "not_blue"], ["red_bool"]),
        helper.make_node("Xor", ["not_blue", "red_bool"], ["black_bool"]),
        helper.make_node("Concat", ["black_bool", "blue_bool", "red_bool"], ["output3"], axis=1),
        helper.make_node("Pad", ["output3", "pads_output"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task027_u8_overlap_count_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
