from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurogolf_onnx import ROOT, normalize_task_id, onnx_path, report_path, update_ledger, utc_timestamp


COMPETITION = "neurogolf-2026"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(args: list[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def require_success(result: dict[str, Any], label: str) -> None:
    if result["returncode"] != 0:
        raise RuntimeError(f"{label} failed with return code {result['returncode']}")


def git_commit_short() -> str:
    result = run_command(["git", "rev-parse", "--short", "HEAD"])
    if result["returncode"] != 0:
        return "nogit"
    return result["stdout"].strip() or "nogit"


def git_status_text() -> str:
    result = run_command(["git", "status", "--porcelain"])
    if result["returncode"] != 0:
        return ""
    return result["stdout"]


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(task_id: str) -> str:
    return f"{task_id}-{utc_compact()}-{git_commit_short()}"


def build_submit_message(manifest: dict[str, Any]) -> str:
    zip_sha = str(manifest["zip_sha256"])[:12]
    local_points = manifest.get("local_points_at_submit")
    return f"{manifest['run_id']} task={manifest['task']} zip_sha={zip_sha} local={local_points}"


def kaggle_executable() -> str:
    exe = shutil.which("kaggle")
    if exe is None:
        raise RuntimeError("Missing `kaggle` executable. Run `uv sync`, then use `uv run python tools/official_submission.py ...`.")
    return exe


def kaggle_submissions_csv() -> dict[str, Any]:
    return run_command(
        [
            kaggle_executable(),
            "competitions",
            "submissions",
            COMPETITION,
            "-v",
            "-q",
            "--page-size",
            "200",
        ]
    )


def clean_csv_text(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("ref,"):
            return "\n".join(lines[index:]) + ("\n" if lines[index:] else "")
    return text


def parse_submissions_csv(text: str) -> list[dict[str, str]]:
    cleaned = clean_csv_text(text)
    if not cleaned.strip():
        return []
    return list(csv.DictReader(cleaned.splitlines()))


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_kaggle_status(raw: str | None) -> str:
    text = (raw or "").upper()
    if "COMPLETE" in text:
        return "complete"
    if "ERROR" in text or "FAILED" in text or "CANCEL" in text:
        return "failed"
    return "pending"


def load_manifest(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return manifest_path, read_json(manifest_path)


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_timestamp()
    write_json(manifest_path, manifest)


def prepare(args: argparse.Namespace) -> int:
    task_id = normalize_task_id(args.task)
    run_id = args.run_id or make_run_id(task_id)
    run_dir = ROOT / "submissions" / "official" / task_id / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    build = run_command([sys.executable, "tools/build_task.py", "--task", task_id])
    write_json(run_dir / "build_command.json", build)
    require_success(build, "build")

    score = run_command([sys.executable, "tools/score_task.py", "--task", task_id])
    write_json(run_dir / "score_command.json", score)
    require_success(score, "score")

    score_report_path = report_path(task_id)
    local_report = read_json(score_report_path)
    if local_report.get("status") != "passes_local":
        raise RuntimeError(f"{task_id} is not passes_local: {local_report.get('status')}")

    source_onnx = onnx_path(task_id)
    if not source_onnx.exists():
        raise FileNotFoundError(f"Missing ONNX model: {source_onnx}")

    staged_onnx = run_dir / f"{task_id}.onnx"
    shutil.copy2(source_onnx, staged_onnx)

    zip_path = run_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(staged_onnx, arcname=f"{task_id}.onnx")

    with zipfile.ZipFile(zip_path) as zf:
        zip_contents = zf.namelist()
    expected_contents = [f"{task_id}.onnx"]
    if zip_contents != expected_contents:
        raise RuntimeError(f"Unexpected zip contents: {zip_contents}")

    manifest: dict[str, Any] = {
        "competition": COMPETITION,
        "task": task_id,
        "run_id": run_id,
        "status": "prepared",
        "created_at": utc_timestamp(),
        "updated_at": utc_timestamp(),
        "git": {
            "commit_short": git_commit_short(),
            "status_porcelain": git_status_text(),
        },
        "paths": {
            "run_dir": relative(run_dir),
            "staged_onnx": relative(staged_onnx),
            "zip": relative(zip_path),
            "local_report": relative(score_report_path),
        },
        "local_report": local_report,
        "local_points_at_submit": local_report.get("local_points"),
        "memory_bytes_approx_at_submit": local_report.get("memory_bytes_approx"),
        "params_at_submit": local_report.get("params"),
        "onnx_sha256": file_sha256(staged_onnx),
        "zip_sha256": file_sha256(zip_path),
        "zip_contents": zip_contents,
        "submitted_at": None,
        "official": None,
    }
    manifest["submit_message"] = build_submit_message(manifest)
    write_json(run_dir / "manifest.json", manifest)

    print(json.dumps({"status": "prepared", "run_dir": relative(run_dir), "run_id": run_id, "zip": relative(zip_path)}, indent=2))
    return 0


def submit(args: argparse.Namespace) -> int:
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

    if result["returncode"] != 0:
        manifest["status"] = "submit_failed"
        manifest["submitted_at"] = submitted_at
        save_manifest(manifest_path, manifest)
        update_ledger(
            manifest["task"],
            official_status="submit_failed",
            official_run_id=manifest["run_id"],
            official_zip_sha256=manifest["zip_sha256"],
            official_submitted_at=submitted_at,
            official_completed_at=None,
            updated_at=utc_timestamp(),
        )
        print(result["stderr"] or result["stdout"])
        return result["returncode"] or 1

    manifest["status"] = "submitted"
    manifest["submitted_at"] = submitted_at
    save_manifest(manifest_path, manifest)
    update_ledger(
        manifest["task"],
        official_status="submitted",
        official_public_score=None,
        official_private_score=None,
        official_delta_public_vs_local=None,
        official_submission_ref=None,
        official_run_id=manifest["run_id"],
        official_zip_sha256=manifest["zip_sha256"],
        official_submitted_at=submitted_at,
        official_completed_at=None,
        updated_at=utc_timestamp(),
    )

    return poll(args)


def update_ledger_from_match(manifest: dict[str, Any], row: dict[str, str], normalized_status: str) -> None:
    public_score = parse_float(row.get("publicScore"))
    private_score = parse_float(row.get("privateScore"))
    local_points = parse_float(manifest.get("local_points_at_submit"))
    delta = None
    if public_score is not None and local_points is not None:
        delta = public_score - local_points

    update_ledger(
        manifest["task"],
        official_status=normalized_status,
        official_public_score=public_score if normalized_status == "complete" else None,
        official_private_score=private_score if normalized_status == "complete" else None,
        official_delta_public_vs_local=delta if normalized_status == "complete" else None,
        official_submission_ref=row.get("ref"),
        official_run_id=manifest["run_id"],
        official_zip_sha256=manifest["zip_sha256"],
        official_submitted_at=manifest.get("submitted_at") or row.get("date"),
        official_completed_at=utc_timestamp() if normalized_status == "complete" else None,
        updated_at=utc_timestamp(),
    )


def poll_once(run_dir: Path, attempt: int) -> tuple[str, dict[str, Any] | None]:
    manifest_path, manifest = load_manifest(run_dir)
    result = kaggle_submissions_csv()
    write_json(run_dir / f"submissions_poll_{attempt:03d}_command.json", result)
    if result["returncode"] != 0:
        manifest["status"] = "poll_failed"
        manifest["poll_error"] = result
        save_manifest(manifest_path, manifest)
        update_ledger(
            manifest["task"],
            official_status="poll_failed",
            official_run_id=manifest["run_id"],
            official_zip_sha256=manifest["zip_sha256"],
            updated_at=utc_timestamp(),
        )
        return "poll_failed", None

    csv_path = run_dir / f"submissions_poll_{attempt:03d}.csv"
    csv_path.write_text(result["stdout"], encoding="utf-8")
    rows = parse_submissions_csv(result["stdout"])
    matches = [row for row in rows if manifest["run_id"] in row.get("description", "")]

    if len(matches) > 1:
        manifest["status"] = "ambiguous"
        manifest["official"] = {"matches": matches}
        save_manifest(manifest_path, manifest)
        update_ledger(
            manifest["task"],
            official_status="ambiguous",
            official_run_id=manifest["run_id"],
            official_zip_sha256=manifest["zip_sha256"],
            updated_at=utc_timestamp(),
        )
        return "ambiguous", {"matches": matches}

    if not matches:
        manifest["status"] = "not_found"
        manifest["official"] = {"matches": []}
        save_manifest(manifest_path, manifest)
        update_ledger(
            manifest["task"],
            official_status="not_found",
            official_run_id=manifest["run_id"],
            official_zip_sha256=manifest["zip_sha256"],
            updated_at=utc_timestamp(),
        )
        return "not_found", None

    row = matches[0]
    normalized_status = normalize_kaggle_status(row.get("status"))
    manifest["status"] = normalized_status
    manifest["official"] = {
        "submission_row": row,
        "normalized_status": normalized_status,
        "public_score": parse_float(row.get("publicScore")),
        "private_score": parse_float(row.get("privateScore")),
        "observed_at": utc_timestamp(),
    }
    save_manifest(manifest_path, manifest)
    update_ledger_from_match(manifest, row, normalized_status)
    return normalized_status, manifest["official"]


def poll(args: argparse.Namespace) -> int:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--task", required=True)
    prepare_parser.add_argument("--run-id", default=None)
    prepare_parser.set_defaults(func=prepare)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--run-dir", type=Path, required=True)
    submit_parser.add_argument("--confirm-submit", action="store_true")
    submit_parser.add_argument("--poll-interval-seconds", type=int, default=20)
    submit_parser.add_argument("--timeout-seconds", type=int, default=900)
    submit_parser.set_defaults(func=submit)

    poll_parser = subparsers.add_parser("poll")
    poll_parser.add_argument("--run-dir", type=Path, required=True)
    poll_parser.add_argument("--poll-interval-seconds", type=int, default=20)
    poll_parser.add_argument("--timeout-seconds", type=int, default=900)
    poll_parser.set_defaults(func=poll)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
