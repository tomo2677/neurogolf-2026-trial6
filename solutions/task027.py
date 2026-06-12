from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int], dims: list[int] | None = None) -> onnx.TensorProto:
    shape = [len(values)] if dims is None else dims
    return helper.make_tensor(name, onnx.TensorProto.INT64, shape, values)


def _f32_tensor(name: str, values: list[float], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dims, values)


def _bool_tensor(name: str, values: list[bool], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.BOOL, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("blue_starts", [0, 1, 0, 0], [4]),
        _int64_tensor("blue_ends", [1, 2, 10, 10], [4]),
        _int64_tensor("inner_starts", [0, 0, 1, 1], [4]),
        _int64_tensor("inner_ends", [1, 1, 10, 10], [4]),
        _int64_tensor("reverse10", list(range(9, -1, -1)), [10]),
        _int64_tensor("reverse9", list(range(8, -1, -1)), [9]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 7, 20, 20], [8]),
        _int64_tensor("sum_axes4", [0, 1, 2, 3], [4]),
        _bool_tensor("false_col9", [False] * 9, [1, 1, 9, 1]),
        _bool_tensor("false_row10", [False] * 10, [1, 1, 1, 10]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "blue_starts", "blue_ends"], ["blue10"]),
        helper.make_node("Cast", ["blue10"], ["blue_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Gather", ["blue_bool", "reverse10"], ["rot9_v"], axis=2),
        helper.make_node("Gather", ["rot9_v", "reverse10"], ["rot9"], axis=3),
        helper.make_node("Slice", ["blue_bool", "inner_starts", "inner_ends"], ["inner9"]),
        helper.make_node("Gather", ["inner9", "reverse9"], ["rot10_inner_v"], axis=2),
        helper.make_node("Gather", ["rot10_inner_v", "reverse9"], ["rot10_inner"], axis=3),
        helper.make_node("Concat", ["false_col9", "rot10_inner"], ["rot10_body"], axis=3),
        helper.make_node("Concat", ["false_row10", "rot10_body"], ["rot10"], axis=2),
        helper.make_node("And", ["blue_bool", "rot9"], ["overlap9_bool"]),
        helper.make_node("And", ["blue_bool", "rot10"], ["overlap10_bool"]),
        helper.make_node("Cast", ["overlap9_bool"], ["overlap9_cells"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("Cast", ["overlap10_bool"], ["overlap10_cells"], to=onnx.TensorProto.FLOAT16),
        helper.make_node("ReduceSum", ["overlap9_cells", "sum_axes4"], ["overlap9"], keepdims=0),
        helper.make_node("ReduceSum", ["overlap10_cells", "sum_axes4"], ["overlap10"], keepdims=0),
        helper.make_node("Greater", ["overlap10", "overlap9"], ["use10"]),
        helper.make_node("Not", ["use10"], ["use9"]),
        helper.make_node("And", ["use10", "rot10"], ["selected10"]),
        helper.make_node("And", ["use9", "rot9"], ["selected9"]),
        helper.make_node("Or", ["selected10", "selected9"], ["selected_rot"]),
        helper.make_node("Not", ["blue_bool"], ["not_blue"]),
        helper.make_node("And", ["selected_rot", "not_blue"], ["red_bool"]),
        helper.make_node("Xor", ["not_blue", "red_bool"], ["black_bool"]),
        helper.make_node("Concat", ["black_bool", "blue_bool", "red_bool"], ["output3"], axis=1),
        helper.make_node("Pad", ["output3", "pads_output"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task027_direct_onehot_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
