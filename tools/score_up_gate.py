from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurogolf_onnx import ROOT, load_ledger, normalize_task_id


LOW_SCORE_THRESHOLD = 2.0
HIGH_SCORE_THRESHOLD = 1.0
HIGH_SCORE_MIN = 20.0
PENDING_OFFICIAL_STATUSES = {"submitted", "pending", "not_found", "poll_failed"}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def trigger_threshold(local_points: float) -> float:
    return HIGH_SCORE_THRESHOLD if local_points >= HIGH_SCORE_MIN else LOW_SCORE_THRESHOLD


def should_submit_for_values(
    local_points: Any,
    official_public_score: Any,
    official_status: Any,
) -> dict[str, Any]:
    if official_status != "complete":
        return {
            "can_submit": False,
            "reason": "official_status_not_complete",
            "official_status": official_status,
        }
    if not is_number(local_points):
        return {"can_submit": False, "reason": "missing_local_points"}
    if not is_number(official_public_score):
        return {"can_submit": False, "reason": "missing_official_public_score"}
    local = float(local_points)
    official = float(official_public_score)
    if official == 0.0:
        return {
            "can_submit": False,
            "reason": "official_zero_requires_repair",
            "local_points": local,
            "official_public_score": official,
        }
    delta = local - official
    threshold = trigger_threshold(local)
    return {
        "can_submit": delta >= threshold,
        "reason": "threshold_met" if delta >= threshold else "threshold_not_met",
        "local_points": local,
        "official_public_score": official,
        "delta": delta,
        "threshold": threshold,
    }


def should_submit_for_task(task_id: str, ledger: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entry = ledger.get(task_id)
    if entry is None:
        return {"task": task_id, "can_submit": False, "reason": "missing_ledger_row"}
    result = should_submit_for_values(
        entry.get("local_points"),
        entry.get("official_public_score"),
        entry.get("official_status"),
    )
    result["task"] = task_id
    return result


def spec_task_ids() -> list[str]:
    task_ids: list[str] = []
    for spec_path in sorted((ROOT / "task_specs").glob("task*.md")):
        try:
            task_ids.append(normalize_task_id(spec_path.stem))
        except ValueError:
            continue
    return task_ids


def classify_status() -> dict[str, Any]:
    ledger = load_ledger()
    baseline_targets: list[dict[str, Any]] = []
    official_pending: list[dict[str, Any]] = []
    official_zero: list[dict[str, Any]] = []
    score_up_candidates: list[dict[str, Any]] = []

    for task_id in spec_task_ids():
        entry = ledger.get(task_id)
        if entry is None:
            baseline_targets.append({"task": task_id, "reason": "missing_ledger_row"})
            continue

        status = entry.get("status")
        official_status = entry.get("official_status")
        official_public_score = entry.get("official_public_score")

        if status != "passes_local":
            baseline_targets.append({"task": task_id, "reason": "not_passes_local", "status": status})
            continue

        if official_status in (None, ""):
            official_pending.append({"task": task_id, "reason": "missing_official_status"})
            continue

        if official_status == "quota_skipped":
            official_pending.append({"task": task_id, "reason": "quota_skipped"})
            continue

        if official_status in PENDING_OFFICIAL_STATUSES:
            official_pending.append({"task": task_id, "reason": official_status})
            continue

        if official_status == "complete" and official_public_score == 0.0:
            official_zero.append({"task": task_id, "reason": "official_zero"})
            continue

        gate = should_submit_for_task(task_id, ledger)
        if gate.get("can_submit"):
            score_up_candidates.append(gate)

    return {
        "baseline_targets": baseline_targets,
        "official_pending": official_pending,
        "official_zero": official_zero,
        "score_up_candidates": score_up_candidates,
    }


def command_status(_args: argparse.Namespace) -> int:
    print(json.dumps(classify_status(), indent=2, ensure_ascii=False))
    return 0


def command_should_submit(args: argparse.Namespace) -> int:
    task_id = normalize_task_id(args.task)
    result = should_submit_for_task(task_id, load_ledger())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=command_status)

    should_submit_parser = subparsers.add_parser("should-submit")
    should_submit_parser.add_argument("--task", required=True)
    should_submit_parser.set_defaults(func=command_should_submit)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
