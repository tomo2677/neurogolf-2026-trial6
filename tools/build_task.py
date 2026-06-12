from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurogolf_onnx import check_and_save_model, normalize_task_id, onnx_path, solution_path, update_ledger, utc_timestamp
from tools.validate_public_rules import RuleValidationError, require_public_rules, validate_model_public_rules


def import_solution(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--solution", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    task_id = normalize_task_id(args.task)
    sol_path = args.solution if args.solution is not None else solution_path(task_id)
    out_path = args.onnx if args.onnx is not None else onnx_path(task_id)

    try:
        if not sol_path.exists():
            raise FileNotFoundError(f"Missing solution file: {sol_path}")
        module = import_solution(sol_path)
        if not hasattr(module, "build_model"):
            raise AttributeError(f"{sol_path} must define build_model()")
        model = module.build_model()
        if not isinstance(model, onnx.ModelProto):
            raise TypeError("build_model() must return onnx.ModelProto")
        rules_report = validate_model_public_rules(model, out_path)
        require_public_rules(rules_report)
        file_size = check_and_save_model(model, out_path)
    except RuleValidationError as exc:
        if not args.no_ledger:
            update_ledger(task_id, status="rule_invalid", local_points=None, memory_bytes_approx=None, params=None, updated_at=utc_timestamp())
        print("Public rules validation failed:")
        print(exc)
        print("Full rule report:")
        print(json.dumps(exc.report, indent=2, ensure_ascii=False))
        return 1
    except Exception:
        if not args.no_ledger:
            update_ledger(task_id, status="build_failed", updated_at=utc_timestamp())
        traceback.print_exc()
        return 1

    if not args.no_ledger:
        update_ledger(task_id, status="generated", updated_at=utc_timestamp())
    print(f"Built {out_path} ({file_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
