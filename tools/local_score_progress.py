from __future__ import annotations

import argparse
import json
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurogolf_onnx import ROOT, load_ledger, normalize_task_id, task_number, utc_timestamp


TARGET_SCORE = Decimal("8000")
SCALE_TASK_COUNT = Decimal("400")
TWO_PLACES = Decimal("0.01")
SIX_PLACES = Decimal("0.000001")
HISTORY_COLUMNS = [
    "recorded_at",
    "git_commit",
    "solved_task_count",
    "local_total_score",
    "local_average_score",
    "local_scaled_score_400",
    "gap_to_8000",
    "official_synced_count",
    "official_drift_count",
    "best_task",
    "worst_task",
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def score2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP))


def score6(value: Any) -> float:
    return float(Decimal(str(value)).quantize(SIX_PLACES, rounding=ROUND_HALF_UP))


def git_commit_short() -> str | None:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def progress_json_path() -> Path:
    return ROOT / "local_score_progress.json"


def progress_md_path() -> Path:
    return ROOT / "local_score_progress.md"


def eligible_rows_for_ledger(ledger: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for raw_task_id, entry in sorted(ledger.items(), key=lambda item: task_number(item[0])):
        task_id = normalize_task_id(raw_task_id)
        status = entry.get("status")
        local_points = entry.get("local_points")
        if status != "passes_local":
            excluded.append({"task": task_id, "reason": "not_passes_local", "status": status})
            continue
        if not is_number(local_points):
            excluded.append({"task": task_id, "reason": "missing_local_points", "local_points": local_points})
            continue
        eligible.append(
            {
                "task": task_id,
                "local_points": float(local_points),
                "official_sync_status": entry.get("official_sync_status"),
            }
        )
    return eligible, excluded


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    total = sum(Decimal(str(row["local_points"])) for row in rows)
    average = None if count == 0 else total / Decimal(count)
    scaled = None if average is None else average * SCALE_TASK_COUNT
    best = max(rows, key=lambda row: (row["local_points"], -task_number(row["task"])), default=None)
    worst = min(rows, key=lambda row: (row["local_points"], task_number(row["task"])), default=None)
    synced_count = sum(1 for row in rows if row.get("official_sync_status") == "synced")
    drift_count = sum(1 for row in rows if row.get("official_sync_status") == "drift")
    return {
        "solved_task_count": count,
        "local_total_score": score2(total),
        "local_average_score": None if average is None else score6(average),
        "local_scaled_score_400": None if scaled is None else score6(scaled),
        "gap_to_8000": None if scaled is None else score2(TARGET_SCORE - scaled),
        "official_synced_count": synced_count,
        "official_drift_count": drift_count,
        "best_task": None if best is None else f"{best['task']}:{score6(best['local_points'])}",
        "worst_task": None if worst is None else f"{worst['task']}:{score6(worst['local_points'])}",
    }


def progress_record_for_ledger(ledger: dict[str, dict[str, Any]], *, recorded_at: str | None = None) -> dict[str, Any]:
    eligible, excluded = eligible_rows_for_ledger(ledger)
    summary = summarize_rows(eligible)
    return {
        "recorded_at": recorded_at or utc_timestamp(),
        "git_commit": git_commit_short(),
        **summary,
        "tasks": [row["task"] for row in eligible],
        "excluded": excluded,
    }


def load_progress_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path} records must be a list")
    return [record for record in records if isinstance(record, dict)]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_progress_markdown(records: list[dict[str, Any]], path: Path) -> None:
    header = "| " + " | ".join(HISTORY_COLUMNS) + " |"
    sep = "| " + " | ".join(["---"] * len(HISTORY_COLUMNS)) + " |"
    lines = [header, sep]
    for record in records:
        row = []
        for column in HISTORY_COLUMNS:
            value = record.get(column)
            row.append("" if value is None else str(value))
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def upsert_progress_history(record: dict[str, Any], *, json_path: Path | None = None, md_path: Path | None = None) -> list[dict[str, Any]]:
    target_json = json_path or progress_json_path()
    target_md = md_path or progress_md_path()
    existing = load_progress_history(target_json)
    records = [record, *[item for item in existing if item.get("recorded_at") != record.get("recorded_at")]]
    write_json(target_json, {"records": records})
    write_progress_markdown(records, target_md)
    return records


def command_plan(_args: argparse.Namespace) -> int:
    record = progress_record_for_ledger(load_ledger())
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


def command_record(_args: argparse.Namespace) -> int:
    record = progress_record_for_ledger(load_ledger())
    upsert_progress_history(record)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.set_defaults(func=command_plan)

    record_parser = subparsers.add_parser("record")
    record_parser.set_defaults(func=command_record)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
