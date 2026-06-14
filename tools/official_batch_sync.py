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

from neurogolf_onnx import ROOT, load_ledger, normalize_task_id, onnx_path, report_path, task_number, utc_timestamp, write_ledger
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


BATCH_SIZE = 10
MATCH_TOLERANCE = Decimal("0.01")
BATCH_SYNC_DELTA_MIN = 0.01
TWO_PLACES = Decimal("0.01")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def score2(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def score2_float(value: Any) -> float:
    return float(score2(value))


def batch_run_id() -> str:
    return f"batch-{utc_compact()}-{git_commit_short()}"


def build_submit_message(manifest: dict[str, Any]) -> str:
    tasks = ",".join(manifest["tasks"])
    zip_sha = str(manifest["zip_sha256"])[:12]
    expected = manifest["expected_public_score"]
    return f"{manifest['run_id']} batch_tasks={tasks} zip_sha={zip_sha} expected={expected}"


def batch_candidates_for_ledger(ledger: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for task_id, entry in ledger.items():
        if entry.get("status") != "passes_local":
            continue
        if entry.get("official_status") != "complete":
            continue
        local_points = entry.get("local_points")
        official_public_score = entry.get("official_public_score")
        if not is_number(local_points) or not is_number(official_public_score):
            continue
        local = float(local_points)
        official = float(official_public_score)
        if official == 0.0:
            continue
        delta = local - official
        if delta <= BATCH_SYNC_DELTA_MIN:
            continue
        candidates.append(
            {
                "task": normalize_task_id(task_id),
                "local_points": local,
                "official_public_score": official,
                "delta": delta,
            }
        )
    candidates.sort(key=lambda row: (-float(row["delta"]), task_number(str(row["task"]))))
    return candidates


def select_batch(candidates: list[dict[str, Any]], limit: int = BATCH_SIZE) -> list[dict[str, Any]]:
    if len(candidates) < limit:
        return []
    return candidates[:limit]


def expected_public_score(details: list[dict[str, Any]]) -> float:
    total = sum(Decimal(str(detail["local_points_at_submit"])) for detail in details)
    return score2_float(total)


def one_zero_candidates(manifest: dict[str, Any], actual_public_score: float) -> list[str]:
    expected = score2(manifest["expected_public_score"])
    actual = score2(actual_public_score)
    deficit = expected - actual
    if deficit <= Decimal("0"):
        return []
    candidates: list[str] = []
    for detail in manifest["task_details"]:
        contribution = score2(detail["local_points_at_submit"])
        if abs(deficit - contribution) <= MATCH_TOLERANCE:
            candidates.append(str(detail["task"]))
    return candidates


def resolution_for_manifest(manifest: dict[str, Any], zero_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    official = manifest.get("official") or {}
    if official.get("normalized_status") != "complete":
        return {"status": "not_complete", "sync_tasks": [], "reason": official.get("normalized_status")}
    actual = parse_float(official.get("public_score"))
    if actual is None:
        return {"status": "missing_public_score", "sync_tasks": []}
    expected = float(manifest["expected_public_score"])
    if abs(score2(actual) - score2(expected)) <= MATCH_TOLERANCE:
        return {"status": "matched", "sync_tasks": list(manifest["tasks"])}

    candidates = one_zero_candidates(manifest, actual)
    if len(candidates) != 1:
        return {
            "status": "unresolved_mismatch",
            "sync_tasks": [],
            "expected_public_score": expected,
            "actual_public_score": actual,
            "one_zero_candidates": candidates,
        }
    suspected = candidates[0]
    if zero_manifest is None:
        return {
            "status": "one_zero_probe_required",
            "sync_tasks": [],
            "suspected_task": suspected,
            "expected_public_score": expected,
            "actual_public_score": actual,
        }

    zero_official = zero_manifest.get("official") or {}
    zero_task = normalize_task_id(zero_manifest.get("task", ""))
    zero_score = parse_float(zero_official.get("public_score"))
    if zero_task != suspected:
        return {"status": "zero_probe_task_mismatch", "sync_tasks": [], "suspected_task": suspected, "zero_task": zero_task}
    if zero_official.get("normalized_status") != "complete":
        return {"status": "zero_probe_not_complete", "sync_tasks": [], "suspected_task": suspected}
    if zero_score != 0.0:
        return {
            "status": "zero_probe_disproved",
            "sync_tasks": [],
            "suspected_task": suspected,
            "zero_public_score": zero_score,
        }
    return {
        "status": "one_zero_confirmed",
        "sync_tasks": [task for task in manifest["tasks"] if task != suspected],
        "suspected_task": suspected,
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
        local_points = entry.get("local_points")
        delta = official_score - float(local_points) if is_number(local_points) else None
        entry.update(
            {
                "official_status": "complete",
                "official_public_score": official_score,
                "official_delta_public_vs_local": delta,
                "official_submission_ref": row.get("ref"),
                "official_run_id": manifest["run_id"],
                "official_zip_sha256": manifest["zip_sha256"],
                "official_submitted_at": manifest.get("submitted_at") or row.get("date"),
                "official_completed_at": observed_at,
                "updated_at": observed_at,
            }
        )
    return ledger


def command_plan(args: argparse.Namespace) -> int:
    candidates = batch_candidates_for_ledger(load_ledger())
    selected = select_batch(candidates, args.limit)
    print(
        json.dumps(
            {
                "status": "ready" if selected else "insufficient_candidates",
                "limit": args.limit,
                "eligible_count": len(candidates),
                "selected": selected,
                "remaining_after_selected": max(0, len(candidates) - len(selected)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if selected else 1


def command_prepare(args: argparse.Namespace) -> int:
    tasks = sorted({normalize_task_id(task) for task in args.tasks}, key=task_number)
    if len(tasks) != args.limit:
        raise ValueError(f"Batch prepare requires exactly {args.limit} unique tasks; got {len(tasks)}")
    run_id = args.run_id or batch_run_id()
    run_dir = ROOT / "submissions" / "official_batch" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    task_details: list[dict[str, Any]] = []
    ledger = load_ledger()
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
                "tasks": tasks,
                "expected_public_score": manifest["expected_public_score"],
                "zip": relative(zip_path),
            },
            indent=2,
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
        save_manifest(manifest_path, manifest)
        return "poll_failed", None

    csv_path = run_dir / f"submissions_poll_{attempt:03d}.csv"
    csv_path.write_text(result["stdout"], encoding="utf-8")
    rows = parse_submissions_csv(result["stdout"])
    matches = [row for row in rows if manifest["run_id"] in row.get("description", "")]

    if len(matches) > 1:
        manifest["status"] = "ambiguous"
        manifest["official"] = {"matches": matches}
        save_manifest(manifest_path, manifest)
        return "ambiguous", {"matches": matches}

    if not matches:
        manifest["status"] = "not_found"
        manifest["official"] = {"matches": []}
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
    zero_manifest = None
    if args.zero_run_dir is not None:
        _, zero_manifest = load_manifest(args.zero_run_dir.resolve())
    resolution = resolution_for_manifest(manifest, zero_manifest)
    resolution["resolved_at"] = utc_timestamp()
    if args.dry_run:
        resolution["dry_run"] = True
        print(json.dumps(resolution, indent=2, ensure_ascii=False))
        return 0 if resolution["sync_tasks"] else 2 if resolution["status"] == "one_zero_probe_required" else 1
    manifest["resolution"] = resolution

    if resolution["sync_tasks"]:
        ledger = load_ledger()
        synced = sync_ledger_rows(ledger, manifest, list(resolution["sync_tasks"]), observed_at=resolution["resolved_at"])
        write_ledger(synced)
        manifest["status"] = "resolved"
        save_manifest(manifest_path, manifest)
        print(json.dumps(resolution, indent=2, ensure_ascii=False))
        return 0

    manifest["status"] = "unresolved"
    save_manifest(manifest_path, manifest)
    print(json.dumps(resolution, indent=2, ensure_ascii=False))
    return 2 if resolution["status"] == "one_zero_probe_required" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--limit", type=int, default=BATCH_SIZE)
    plan_parser.set_defaults(func=command_plan)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--tasks", nargs="+", required=True)
    prepare_parser.add_argument("--limit", type=int, default=BATCH_SIZE)
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
    resolve_parser.add_argument("--zero-run-dir", type=Path, default=None)
    resolve_parser.add_argument("--dry-run", action="store_true")
    resolve_parser.set_defaults(func=command_resolve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
