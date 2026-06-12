from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import onnx
import onnxruntime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurogolf_onnx import GRID_SHAPE, ROOT, load_official_utils


FILE_SIZE_LIMIT_BYTES = int(1.44 * 1024 * 1024)
BANNED_OP_TYPES = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
ALLOWED_DOMAINS = {"", "ai.onnx"}
RULES_REFERENCE = "docs/neurogolf_public_facts.md"


class RuleValidationError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        issues = report.get("issues") or []
        summary = "; ".join(str(issue.get("message", issue)) for issue in issues[:3])
        if len(issues) > 3:
            summary += f"; ... {len(issues) - 3} more"
        super().__init__(summary or "Public rules validation failed")


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    issue: dict[str, Any] = {"code": code, "message": message}
    issue.update(details)
    return issue


def _tensor_shape_issues(name: str, value_info: onnx.ValueInfoProto) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if value_info.type.HasField("sequence_type"):
        return [_issue("sequence_tensor", f"{name} is a sequence tensor")]
    if not value_info.type.HasField("tensor_type"):
        return [_issue("non_tensor_value", f"{name} is not a tensor")]
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return [_issue("missing_shape", f"{name} has no static shape")]
    for index, dim in enumerate(tensor_type.shape.dim):
        if dim.HasField("dim_param"):
            issues.append(_issue("symbolic_shape", f"{name} dimension {index} is symbolic", dim_param=dim.dim_param))
        elif not dim.HasField("dim_value"):
            issues.append(_issue("missing_dim_value", f"{name} dimension {index} has no dim_value"))
        elif dim.dim_value <= 0:
            issues.append(_issue("invalid_dim_value", f"{name} dimension {index} is not positive", dim_value=dim.dim_value))
    return issues


def _shape(value_info: onnx.ValueInfoProto) -> list[int | None]:
    if not value_info.type.HasField("tensor_type"):
        return []
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims: list[int | None] = []
    for dim in tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else None)
    return dims


def _dtype_name(elem_type: int) -> str:
    try:
        return onnx.TensorProto.DataType.Name(elem_type)
    except ValueError:
        return str(elem_type)


def _check_graph(model: onnx.ModelProto, issues: list[dict[str, Any]]) -> onnx.ModelProto | None:
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        issues.append(_issue("onnx_checker_failed", "onnx.checker rejected the model", error=str(exc)))
        return None

    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        issues.append(_issue("shape_inference_failed", "Strict shape inference failed", error=str(exc)))
        return None

    graph = inferred.graph
    if len(graph.input) != 1:
        issues.append(_issue("input_count", "Model must have exactly one input", count=len(graph.input)))
    if len(graph.output) != 1:
        issues.append(_issue("output_count", "Model must have exactly one output", count=len(graph.output)))
    if graph.input and graph.input[0].name != "input":
        issues.append(_issue("input_name", "Model input must be named input", actual=graph.input[0].name))
    if graph.output and graph.output[0].name != "output":
        issues.append(_issue("output_name", "Model output must be named output", actual=graph.output[0].name))
    if graph.input and _shape(graph.input[0]) != GRID_SHAPE:
        issues.append(_issue("input_shape", "Model input shape must be [1, 10, 30, 30]", actual=_shape(graph.input[0])))
    if graph.output and _shape(graph.output[0]) != GRID_SHAPE:
        issues.append(_issue("output_shape", "Model output shape must be [1, 10, 30, 30]", actual=_shape(graph.output[0])))
    if graph.input and graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        issues.append(_issue("input_dtype", "Model input dtype must be FLOAT"))

    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {value.name for value in [*graph.input, *graph.output]}
    collisions = sorted(io_names.intersection(init_names))
    if collisions:
        issues.append(_issue("initializer_io_collision", "Initializer names must not collide with graph inputs/outputs", names=collisions))

    if inferred.functions:
        issues.append(_issue("model_functions", "Model functions are not permitted", count=len(inferred.functions)))
    for opset in inferred.opset_import:
        if opset.domain not in ALLOWED_DOMAINS:
            issues.append(_issue("opset_domain", "Only default ONNX domains are permitted", domain=opset.domain))

    seen_value_info: set[str] = set()
    for value_info in [*graph.input, *graph.value_info, *graph.output]:
        if value_info.name in seen_value_info:
            issues.append(_issue("duplicate_value_info", "Duplicate value_info names are not permitted", name=value_info.name))
        seen_value_info.add(value_info.name)
        if "kernel_time" in value_info.name:
            issues.append(_issue("profiler_name", "Tensor names must not contain kernel_time", name=value_info.name))
        issues.extend(_tensor_shape_issues(value_info.name, value_info))

    tensor_map = {value.name: value for value in [*graph.input, *graph.value_info, *graph.output]}
    for node in graph.node:
        op_upper = node.op_type.upper()
        if op_upper in BANNED_OP_TYPES or "SEQUENCE" in op_upper:
            issues.append(_issue("banned_op", "Operator is not permitted", op_type=node.op_type))
        if node.domain not in ALLOWED_DOMAINS:
            issues.append(_issue("node_domain", "Custom node domains are not permitted", op_type=node.op_type, domain=node.domain))
        if not node.output:
            issues.append(_issue("node_without_output", "Nodes must have at least one output", op_type=node.op_type))
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                issues.append(_issue("subgraph", "Subgraphs are not permitted", op_type=node.op_type, attr=attr.name))
        if node.op_type == "TopK" and node.input:
            data_input = tensor_map.get(node.input[0])
            if data_input is None or not data_input.type.HasField("tensor_type"):
                issues.append(_issue("topk_input_metadata", "TopK data input must have inferred tensor metadata", input=node.input[0]))
            else:
                elem_type = data_input.type.tensor_type.elem_type
                if elem_type != onnx.TensorProto.FLOAT:
                    issues.append(
                        _issue(
                            "topk_input_dtype",
                            "TopK data input must be FLOAT for official runtime compatibility",
                            input=node.input[0],
                            dtype=_dtype_name(elem_type),
                        )
                    )
        for output_name in node.output:
            if not output_name:
                continue
            if "kernel_time" in output_name:
                issues.append(_issue("profiler_name", "Tensor names must not contain kernel_time", name=output_name))
            if output_name != "output":
                item = tensor_map.get(output_name)
                if item is None or not item.type.HasField("tensor_type"):
                    issues.append(_issue("missing_inferred_value", "Node output must have inferred tensor metadata", name=output_name))
    return inferred


def _check_sanitizer_and_ort(model: onnx.ModelProto, issues: list[dict[str, Any]]) -> str:
    official = load_official_utils()
    source = "unavailable"
    sanitized = model
    if official is not None:
        source = official.source
        try:
            sanitized = official.module.sanitize_model(onnx.ModelProto().FromString(model.SerializeToString()))
        except Exception as exc:
            issues.append(_issue("sanitize_failed", "Official sanitizer raised an exception", error=str(exc)))
            return source
        if sanitized is None:
            issues.append(_issue("sanitize_failed", "Official sanitizer returned None"))
            return source

    try:
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as exc:
        issues.append(_issue("ort_load_failed", "ONNX Runtime could not load the sanitized model", error=str(exc)))
    return source


def validate_model_public_rules(model: onnx.ModelProto, onnx_path: Path | None = None, file_size: int | None = None) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    serialized = model.SerializeToString()
    measured_file_size = len(serialized) if file_size is None else file_size
    if measured_file_size > FILE_SIZE_LIMIT_BYTES:
        issues.append(
            _issue(
                "file_size",
                "ONNX file size exceeds the 1.44 MB public limit",
                file_size=measured_file_size,
                limit=FILE_SIZE_LIMIT_BYTES,
            )
        )

    inferred = _check_graph(model, issues)
    sanitizer_source = "not_run"
    if inferred is not None:
        sanitizer_source = _check_sanitizer_and_ort(inferred, issues)

    valid = not issues
    return {
        "status": "passes_rules" if valid else "rule_invalid",
        "valid": valid,
        "issues": issues,
        "onnx_path": _rel(onnx_path),
        "file_size": measured_file_size,
        "file_size_limit": FILE_SIZE_LIMIT_BYTES,
        "rules_reference": RULES_REFERENCE,
        "sanitizer_source": sanitizer_source,
    }


def validate_onnx_file_public_rules(path: Path) -> dict[str, Any]:
    try:
        model = onnx.load(path)
    except Exception:
        return {
            "status": "rule_invalid",
            "valid": False,
            "issues": [_issue("load_failed", "Could not load ONNX model", error=traceback.format_exc())],
            "onnx_path": _rel(path),
            "file_size": path.stat().st_size if path.exists() else None,
            "file_size_limit": FILE_SIZE_LIMIT_BYTES,
            "rules_reference": RULES_REFERENCE,
            "sanitizer_source": "not_run",
        }
    return validate_model_public_rules(model, path, file_size=path.stat().st_size)


def require_public_rules(report: dict[str, Any]) -> None:
    if not report.get("valid"):
        raise RuleValidationError(report)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = validate_onnx_file_public_rules(args.onnx)
    if args.report is not None:
        write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
