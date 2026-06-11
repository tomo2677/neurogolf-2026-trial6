from __future__ import annotations

import onnx
from onnx import helper

from neurogolf_onnx import GRID_SHAPE, IR_VERSION, make_io_value_infos


def _int64_tensor(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)


def _u8_tensor(name: str, values: list[int], dims: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.UINT8, dims, values)


def build_model() -> onnx.ModelProto:
    x, _ = make_io_value_infos()
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)

    initializers = [
        _int64_tensor("input_x1_start", [0, 1, 0, 0]),
        _int64_tensor("input_x1_end", [1, 2, 4, 3]),
        _int64_tensor("pads_output", [0, 0, 0, 0, 0, 7, 21, 27]),
        _u8_tensor("colors3", [0, 2, 1], [1, 3, 1, 1]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "input_x1_start", "input_x1_end"], ["x1"]),
        helper.make_node("Cast", ["x1"], ["x1_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Split", ["x1_u8"], ["row0", "row1", "row2", "row3"], axis=2),
        helper.make_node("Equal", ["row0", "row2"], ["eq02"]),
        helper.make_node("Equal", ["row1", "row3"], ["eq13"]),
        helper.make_node("And", ["eq02", "eq13"], ["p2_vec"]),
        helper.make_node("Cast", ["p2_vec"], ["p2_vec_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMin", ["p2_vec_u8"], ["p2_ok_u8"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("Equal", ["row0", "row3"], ["eq03"]),
        helper.make_node("Cast", ["eq03"], ["eq03_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ReduceMin", ["eq03_u8"], ["p3_ok_u8"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("Cast", ["p2_ok_u8"], ["p2_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Cast", ["p3_ok_u8"], ["p3_bool"], to=onnx.TensorProto.BOOL),
        helper.make_node("Or", ["p2_bool", "p3_bool"], ["p23_bool"]),
        helper.make_node("Where", ["p3_bool", "row1", "row0"], ["row4"]),
        helper.make_node("Where", ["p3_bool", "row2", "row1"], ["row5"]),
        helper.make_node("Where", ["p23_bool", "row0", "row2"], ["row6"]),
        helper.make_node("Where", ["p23_bool", "row1", "row3"], ["row7"]),
        helper.make_node("Where", ["p3_bool", "row2", "row0"], ["row8"]),
        helper.make_node("Concat", ["row0", "row1", "row2", "row3", "row4", "row5", "row6", "row7", "row8"], ["red_top_u8"], axis=2),
        helper.make_node("Equal", ["colors3", "red_top_u8"], ["output3"]),
        helper.make_node("Pad", ["output3", "pads_output"], ["output"], mode="constant"),
    ]

    graph = helper.make_graph(nodes, "task003_direct_row_select_graph", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])
    assert list(model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(4)) == GRID_SHAPE
    return model
