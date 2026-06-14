from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurogolf_onnx import ROOT, load_ledger, normalize_task_id, onnx_path, task_number, utc_timestamp, write_ledger
from tools.official_submission import (
    COMPETITION,
    file_sha256,
    git_commit_short,
    git_status_text,
    kaggle_executable,
    kaggle_submissions_csv,
    load_manifest,
    normalize_kaggle_status,
    parse_float,
    parse_submissions_csv,
    quota_status_from_result,
    read_json,
    relative,
    run_command,
    save_manifest,
    unavailable_quota_status,
    utc_compact,
    write_json,
)


MATCH_TOLERANCE = Decimal("0.01")
TWO_PLACES = Decimal("0.01")
SIX_PLACES = Decimal("0.000001")
HISTORY_COLUMNS = [
    "completed_at",
    "submitted_at",
    "run_id",
    "task_count",
    "expected_public_score",
    "actual_public_score",
    "delta_actual_vs_expected",
    "average_public_score",
    "scaled_public_score_400",
    "sync_status",
    "submission_ref",
    "git_commit",
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def score2(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def score2_float(value: Any) -> float:
    return float(score2(value))


def score6_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(Decimal(str(value)).quantize(SIX_PLACES, rounding=ROUND_HALF_UP))


def snapshot_run_id() -> str:
    return f"snapshot-{utc_compact()}-{git_commit_short()}"


def history_json_path() -> Path:
    return ROOT / "official_score_snapshots.json"


def history_md_path() -> Path:
    return ROOT / "official_score_snapshots.md"


def eligible_tasks_for_ledger(ledger: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        eligible.append({"task": task_id, "local_points": float(local_points)})
    return eligible, excluded


def expected_public_score(details: list[dict[str, Any]]) -> float:
    total = sum(Decimal(str(detail["local_points_at_submit"])) for detail in details)
    return score2_float(total)


def snapshot_metrics(task_count: int, expected_public_score_value: Any, actual_public_score: Any) -> dict[str, Any]:
    actual = None if actual_public_score is None else score2(actual_public_score)
    expected = score2(expected_public_score_value)
    delta = None if actual is None else actual - expected
    average = None if actual is None or task_count == 0 else actual / Decimal(task_count)
    scaled = None if average is None else average * Decimal(400)
    return {
        "delta_actual_vs_expected": score2_float(delta) if delta is not None else None,
        "average_public_score": score6_float(average),
        "scaled_public_score_400": score6_float(scaled),
    }


def build_submit_message(manifest: dict[str, Any]) -> str:
    zip_sha = str(manifest["zip_sha256"])[:12]
    tasks = manifest["tasks"]
    return (
        f"{manifest['run_id']} score_snapshot tasks={len(tasks)} "
        f"first={tasks[0]} last={tasks[-1]} zip_sha={zip_sha} expected={manifest['expected_public_score']}"
    )


def resolution_for_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    official = manifest.get("official") or {}
    normalized_status = official.get("normalized_status")
    if normalized_status != "complete":
        return {
            "status": "not_complete",
            "sync_tasks": [],
            "reason": normalized_status or manifest.get("status"),
        }
    actual = parse_float(official.get("public_score"))
    if actual is None:
        return {"status": "missing_public_score", "sync_tasks": []}
    expected = float(manifest["expected_public_score"])
    if abs(score2(actual) - score2(expected)) <= MATCH_TOLERANCE:
        return {
            "status": "matched",
            "sync_tasks": list(manifest["tasks"]),
            "expected_public_score": expected,
            "actual_public_score": actual,
        }
    return {
        "status": "mismatch",
        "sync_tasks": [],
        "expected_public_score": expected,
        "actual_public_score": actual,
        "delta_actual_vs_expected": score2_float(score2(actual) - score2(expected)),
    }


def sync_ledger_rows(
    ledger: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    tasks_to_sync: list[str],
    *,
    observed_at: str,
) -> dict[str, dict[str, Any]]:
    details_by_task = {str(detail["task"]): detail for detail in manifest["task_details"]}
    official = manifest.get("official") or {}
    row = official.get("submission_row") or {}
    for task_id in tasks_to_sync:
        detail = details_by_task[task_id]
        entry = ledger[task_id]
        official_score = float(detail["local_points_at_submit"])
        entry.update(
            {
                "official_status": "complete",
                "official_public_score": official_score,
                "official_submission_ref": row.get("ref"),
                "official_run_id": manifest["run_id"],
                "official_zip_sha256": manifest["zip_sha256"],
                "official_submitted_at": manifest.get("submitted_at") or row.get("date"),
                "official_completed_at": observed_at,
                "updated_at": observed_at,
            }
        )
    return ledger


def snapshot_record_for_resolution(manifest: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
    official = manifest.get("official") or {}
    row = official.get("submission_row") or {}
    actual = parse_float(official.get("public_score"))
    task_count = len(manifest.get("tasks") or [])
    metrics = snapshot_metrics(task_count, manifest["expected_public_score"], actual)
    return {
        "completed_at": resolution.get("resolved_at") or official.get("observed_at") or utc_timestamp(),
        "submitted_at": manifest.get("submitted_at") or row.get("date"),
        "run_id": manifest["run_id"],
        "task_count": task_count,
        "expected_public_score": manifest["expected_public_score"],
        "actual_public_score": actual,
        "delta_actual_vs_expected": metrics["delta_actual_vs_expected"],
        "average_public_score": metrics["average_public_score"],
        "scaled_public_score_400": metrics["scaled_public_score_400"],
        "sync_status": resolution["status"],
        "submission_ref": row.get("ref"),
        "git_commit": (manifest.get("git") or {}).get("commit_short"),
        "official_status": official.get("normalized_status"),
        "tasks": list(manifest.get("tasks") or []),
        "zip_sha256": manifest.get("zip_sha256"),
        "run_dir": (manifest.get("paths") or {}).get("run_dir"),
    }


def load_snapshot_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_json(path)
    snapshots = data.get("snapshots", [])
    if not isinstance(snapshots, list):
        raise ValueError(f"{path} snapshots must be a list")
    return [snapshot for snapshot in snapshots if isinstance(snapshot, dict)]


def write_snapshot_markdown(records: list[dict[str, Any]], path: Path) -> None:
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


def upsert_snapshot_history(record: dict[str, Any], *, json_path: Path | None = None, md_path: Path | None = None) -> list[dict[str, Any]]:
    target_json = json_path or history_json_path()
    target_md = md_path or history_md_path()
    existing = load_snapshot_history(target_json)
    records = [record, *[item for item in existing if item.get("run_id") != record.get("run_id")]]
    write_json(target_json, {"snapshots": records})
    write_snapshot_markdown(records, target_md)
    return records


def command_plan(_args: argparse.Namespace) -> int:
    eligible, excluded = eligible_tasks_for_ledger(load_ledger())
    details = [{"task": row["task"], "local_points_at_submit": row["local_points"]} for row in eligible]
    print(
        json.dumps(
            {
                "status": "ready" if eligible else "no_eligible_tasks",
                "task_count": len(eligible),
                "tasks": [row["task"] for row in eligible],
                "expected_public_score": expected_public_score(details) if details else None,
                "excluded": excluded,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if eligible else 1


def command_prepare(args: argparse.Namespace) -> int:
    run_id = args.run_id or snapshot_run_id()
    run_dir = ROOT / "submissions" / "official_score_snapshot" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    ledger = load_ledger()
    eligible, excluded = eligible_tasks_for_ledger(ledger)
    if not eligible:
        raise RuntimeError("No eligible passes_local tasks with numeric local_points.")
    tasks = [row["task"] for row in eligible]

    task_details: list[dict[str, Any]] = []
    for task_id in tasks:
        build = run_command([sys.executable, "tools/build_task.py", "--task", task_id, "--no-ledger"])
        write_json(run_dir / f"{task_id}_build_command.json", build)
        if build["returncode"] != 0:
            raise RuntimeError(f"{task_id} build failed")

        score_report_path = run_dir / f"{task_id}_score.json"
        score = run_command(
            [
                sys.executable,
                "tools/score_task.py",
                "--task",
                task_id,
                "--report",
                str(score_report_path),
                "--no-ledger",
            ]
        )
        write_json(run_dir / f"{task_id}_score_command.json", score)
        if score["returncode"] != 0:
            raise RuntimeError(f"{task_id} score failed")

        local_report = read_json(score_report_path)
        if local_report.get("status") != "passes_local":
            raise RuntimeError(f"{task_id} is not passes_local: {local_report.get('status')}")

        source_onnx = onnx_path(task_id)
        staged_onnx = run_dir / f"{task_id}.onnx"
        shutil.copy2(source_onnx, staged_onnx)
        entry = ledger.get(task_id, {})
        task_details.append(
            {
                "task": task_id,
                "local_points_at_submit": local_report.get("local_points"),
                "memory_bytes_approx_at_submit": local_report.get("memory_bytes_approx"),
                "params_at_submit": local_report.get("params"),
                "previous_official_public_score": entry.get("official_public_score"),
                "previous_official_status": entry.get("official_status"),
                "previous_official_sync_status": entry.get("official_sync_status"),
                "onnx_sha256": file_sha256(staged_onnx),
                "local_report": relative(score_report_path),
            }
        )

    zip_path = run_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id in tasks:
            zf.write(run_dir / f"{task_id}.onnx", arcname=f"{task_id}.onnx")
    with zipfile.ZipFile(zip_path) as zf:
        zip_contents = zf.namelist()
    expected_contents = [f"{task_id}.onnx" for task_id in tasks]
    if zip_contents != expected_contents:
        raise RuntimeError(f"Unexpected zip contents: {zip_contents}")

    manifest: dict[str, Any] = {
        "competition": COMPETITION,
        "run_id": run_id,
        "status": "prepared",
        "created_at": utc_timestamp(),
        "updated_at": utc_timestamp(),
        "tasks": tasks,
        "excluded": excluded,
        "task_details": task_details,
        "expected_public_score": expected_public_score(task_details),
        "match_tolerance": float(MATCH_TOLERANCE),
        "git": {"commit_short": git_commit_short(), "status_porcelain": git_status_text()},
        "paths": {"run_dir": relative(run_dir), "zip": relative(zip_path)},
        "zip_sha256": file_sha256(zip_path),
        "zip_contents": zip_contents,
        "submitted_at": None,
        "official": None,
        "resolution": None,
    }
    manifest["submit_message"] = build_submit_message(manifest)
    write_json(run_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "prepared",
                "run_dir": relative(run_dir),
                "run_id": run_id,
                "task_count": len(tasks),
                "expected_public_score": manifest["expected_public_score"],
                "zip": relative(zip_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def command_submit(args: argparse.Namespace) -> int:
    if not args.confirm_submit:
        print("Refusing to submit without --confirm-submit.")
        return 2
    run_dir = args.run_dir.resolve()
    manifest_path, manifest = load_manifest(run_dir)
    zip_path = (ROOT / manifest["paths"]["zip"]).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip: {zip_path}")

    before = kaggle_submissions_csv()
    write_json(run_dir / "submissions_before_command.json", before)
    try:
        quota_status = quota_status_from_result(before)
    except Exception as exc:
        quota_status = unavailable_quota_status(exc)
    write_json(run_dir / "quota_check.json", quota_status)
    if not quota_status["can_submit"]:
        manifest["status"] = "quota_skipped"
        manifest["quota"] = quota_status
        manifest["official"] = {"normalized_status": "quota_skipped", "quota": quota_status, "observed_at": utc_timestamp()}
        save_manifest(manifest_path, manifest)
        print(json.dumps({"status": "quota_skipped", "quota": quota_status}, indent=2, ensure_ascii=False))
        return 3

    submitted_at = utc_timestamp()
    result = run_command(
        [
            kaggle_executable(),
            "competitions",
            "submit",
            COMPETITION,
            "-f",
            str(zip_path),
            "-m",
            manifest["submit_message"],
        ]
    )
    write_json(run_dir / "submit_command.json", result)
    manifest["submitted_at"] = submitted_at
    if result["returncode"] != 0:
        manifest["status"] = "submit_failed"
        manifest["official"] = {"normalized_status": "submit_failed", "observed_at": utc_timestamp()}
        save_manifest(manifest_path, manifest)
        print(result["stderr"] or result["stdout"])
        return result["returncode"] or 1
    manifest["status"] = "submitted"
    save_manifest(manifest_path, manifest)
    return command_poll(args)


def poll_once(run_dir: Path, attempt: int) -> tuple[str, dict[str, Any] | None]:
    manifest_path, manifest = load_manifest(run_dir)
    result = kaggle_submissions_csv()
    write_json(run_dir / f"submissions_poll_{attempt:03d}_command.json", result)
    if result["returncode"] != 0:
        manifest["status"] = "poll_failed"
        manifest["poll_error"] = result
        manifest["official"] = {"normalized_status": "poll_failed", "observed_at": utc_timestamp()}
        save_manifest(manifest_path, manifest)
        return "poll_failed", None

    csv_path = run_dir / f"submissions_poll_{attempt:03d}.csv"
    csv_path.write_text(result["stdout"], encoding="utf-8")
    rows = parse_submissions_csv(result["stdout"])
    matches = [row for row in rows if manifest["run_id"] in row.get("description", "")]

    if len(matches) > 1:
        manifest["status"] = "ambiguous"
        manifest["official"] = {"normalized_status": "ambiguous", "matches": matches, "observed_at": utc_timestamp()}
        save_manifest(manifest_path, manifest)
        return "ambiguous", manifest["official"]

    if not matches:
        manifest["status"] = "not_found"
        manifest["official"] = {"normalized_status": "not_found", "matches": [], "observed_at": utc_timestamp()}
        save_manifest(manifest_path, manifest)
        return "not_found", None

    row = matches[0]
    normalized_status = normalize_kaggle_status(row.get("status"))
    manifest["status"] = normalized_status
    manifest["official"] = {
        "submission_row": row,
        "normalized_status": normalized_status,
        "public_score": parse_float(row.get("publicScore")),
        "observed_at": utc_timestamp(),
    }
    save_manifest(manifest_path, manifest)
    return normalized_status, manifest["official"]


def command_poll(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    deadline = time.monotonic() + args.timeout_seconds
    attempt = 1
    while True:
        status, official = poll_once(run_dir, attempt)
        if status == "complete":
            print(json.dumps({"status": status, "official": official}, indent=2, ensure_ascii=False))
            return 0
        if status in {"failed", "ambiguous", "poll_failed"}:
            print(json.dumps({"status": status, "official": official}, indent=2, ensure_ascii=False))
            return 1
        if time.monotonic() >= deadline:
            print(json.dumps({"status": status, "message": "Timed out before official score completed."}, indent=2))
            return 1
        attempt += 1
        time.sleep(args.poll_interval_seconds)


def command_resolve(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest_path, manifest = load_manifest(run_dir)
    resolution = resolution_for_manifest(manifest)
    resolution["resolved_at"] = utc_timestamp()
    record = snapshot_record_for_resolution(manifest, resolution)
    if args.dry_run:
        print(json.dumps({"resolution": resolution, "snapshot_record": record}, indent=2, ensure_ascii=False))
        return 0 if resolution["status"] == "matched" else 1

    manifest["resolution"] = resolution
    if resolution["sync_tasks"]:
        ledger = load_ledger()
        synced = sync_ledger_rows(ledger, manifest, list(resolution["sync_tasks"]), observed_at=resolution["resolved_at"])
        write_ledger(synced)
        manifest["status"] = "resolved"
    else:
        manifest["status"] = "unresolved"
    save_manifest(manifest_path, manifest)
    upsert_snapshot_history(record)
    print(json.dumps({"resolution": resolution, "snapshot_record": record}, indent=2, ensure_ascii=False))
    return 0 if resolution["status"] == "matched" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.set_defaults(func=command_plan)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--run-id", default=None)
    prepare_parser.set_defaults(func=command_prepare)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--run-dir", type=Path, required=True)
    submit_parser.add_argument("--confirm-submit", action="store_true")
    submit_parser.add_argument("--poll-interval-seconds", type=int, default=20)
    submit_parser.add_argument("--timeout-seconds", type=int, default=900)
    submit_parser.set_defaults(func=command_submit)

    poll_parser = subparsers.add_parser("poll")
    poll_parser.add_argument("--run-dir", type=Path, required=True)
    poll_parser.add_argument("--poll-interval-seconds", type=int, default=20)
    poll_parser.add_argument("--timeout-seconds", type=int, default=900)
    poll_parser.set_defaults(func=command_poll)

    resolve_parser = subparsers.add_parser("resolve")
    resolve_parser.add_argument("--run-dir", type=Path, required=True)
    resolve_parser.add_argument("--dry-run", action="store_true")
    resolve_parser.set_defaults(func=command_resolve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
