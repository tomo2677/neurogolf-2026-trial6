from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurogolf_onnx import (
    binary_tensor_to_grid,
    count_params_approx,
    example_over_30,
    grid_to_one_hot,
    iter_examples,
    load_official_utils,
    load_task,
    local_points,
    normalize_task_id,
    onnx_path,
    report_path,
    update_ledger,
    utc_timestamp,
)


def make_session(model: onnx.ModelProto, official: Any, profile_prefix: Path):
    sanitized = model
    if official is not None:
        sanitized = official.module.sanitize_model(onnx.ModelProto().FromString(model.SerializeToString()))
        if sanitized is None:
            raise RuntimeError("official sanitize_model returned None")
    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    profile_prefix.parent.mkdir(parents=True, exist_ok=True)
    options.profile_file_prefix = str(profile_prefix)
    session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    return sanitized, session


def run_network(session: onnxruntime.InferenceSession, official: Any, input_tensor: np.ndarray) -> np.ndarray:
    if official is not None:
        return official.module.run_network(session, input_tensor)
    output = session.run(["output"], {"input": input_tensor})[0]
    return (output > 0.0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    task_id = normalize_task_id(args.task)
    model_path = args.onnx if args.onnx is not None else onnx_path(task_id)
    out_report_path = args.report if args.report is not None else report_path(task_id)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)

    checked = 0
    passed = 0
    failed = 0
    ignored_over_30 = 0
    first_failure: dict[str, Any] | None = None
    memory_bytes_approx: int | None = None
    params: int | None = None
    points: float | None = None
    status = "score_failed"
    scoring_source = "fallback"
    trace_path: str | None = None

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing ONNX model: {model_path}")

        model = onnx.load(model_path)
        official = load_official_utils()
        if official is not None:
            scoring_source = official.source
        profile_prefix = Path("outputs/reports/ort_profile") if args.report is None else out_report_path.parent / "ort_profile"
        sanitized, session = make_session(model, official, profile_prefix)
        task_data = load_task(task_id)

        for split, index, example in iter_examples(task_data):
            if example_over_30(example):
                ignored_over_30 += 1
                continue
            checked += 1
            input_tensor = grid_to_one_hot(example["input"])
            expected = grid_to_one_hot(example["output"])
            actual = run_network(session, official, input_tensor)
            if np.array_equal(actual, expected):
                passed += 1
                continue

            failed += 1
            if first_failure is None:
                height = len(example["output"])
                width = len(example["output"][0]) if height else 0
                first_failure = {
                    "split": split,
                    "index": index,
                    "input": example["input"],
                    "expected": example["output"],
                    "actual": binary_tensor_to_grid(actual, height, width),
                }

        trace_path = session.end_profiling()
        if official is not None:
            memory_bytes_approx, params = official.module.score_network(sanitized, trace_path)
        if params is None:
            params = count_params_approx(sanitized)

        if checked == 0:
            status = "score_failed"
        elif failed == 0:
            status = "passes_local"
            points = local_points(memory_bytes_approx, params)
        else:
            status = "fails_local"
            points = 0.0

    except Exception:
        error = traceback.format_exc()
        report = {
            "task": task_id,
            "status": "score_failed",
            "error": error,
            "checked_examples": checked,
            "passed_examples": passed,
            "failed_examples": failed,
            "ignored_over_30": ignored_over_30,
            "first_failure": first_failure,
            "memory_bytes_approx": memory_bytes_approx,
            "params": params,
            "local_points": points,
            "scoring_source": scoring_source,
            "trace_path": trace_path,
        }
        out_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not args.no_ledger:
            update_ledger(task_id, status="score_failed", local_points=None, memory_bytes_approx=memory_bytes_approx, params=params, updated_at=utc_timestamp())
        print(error)
        return 1

    accuracy = passed / checked if checked else None
    report = {
        "task": task_id,
        "status": status,
        "local_points": points,
        "accuracy": accuracy,
        "checked_examples": checked,
        "passed_examples": passed,
        "failed_examples": failed,
        "ignored_over_30": ignored_over_30,
        "first_failure": first_failure,
        "memory_bytes_approx": memory_bytes_approx,
        "params": params,
        "onnx_path": str(model_path),
        "report_path": str(out_report_path),
        "scoring_source": scoring_source,
        "trace_path": trace_path,
    }
    out_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.no_ledger:
        update_ledger(task_id, status=status, local_points=points, memory_bytes_approx=memory_bytes_approx, params=params, updated_at=utc_timestamp())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "passes_local" else 1


if __name__ == "__main__":
    raise SystemExit(main())
