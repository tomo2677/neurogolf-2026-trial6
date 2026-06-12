from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurogolf_onnx import ROOT, load_ledger, normalize_task_id, report_path, solution_path, task_spec_path


MODES = {"impl_opt", "rule_redesign"}
LOG_LIMIT = 25


def resolve_cli_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw.resolve()
    return (Path.cwd() / raw).resolve()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_status_for(path: Path) -> dict[str, Any]:
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        return {"dirty": None, "status": "outside_repo"}
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", str(rel)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    text = proc.stdout.strip()
    return {"dirty": bool(text), "status": text}


def next_experiment_dir(task_id: str) -> tuple[str, Path]:
    base = ROOT / "outputs" / "experiments" / task_id
    base.mkdir(parents=True, exist_ok=True)
    max_seen = 0
    for path in base.glob("exp[0-9][0-9][0-9]"):
        if path.is_dir():
            max_seen = max(max_seen, int(path.name.removeprefix("exp")))
    exp_id = f"exp{max_seen + 1:03d}"
    exp_dir = base / exp_id
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_id, exp_dir


def run_tool(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "args": [sys.executable, *args],
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def score_value(report: dict[str, Any] | None, key: str) -> Any:
    if report is None:
        return None
    return report.get(key)


def points_delta(candidate_points: Any, baseline_points: Any) -> float | None:
    if not isinstance(candidate_points, (int, float)) or not isinstance(baseline_points, (int, float)):
        return None
    return float(candidate_points) - float(baseline_points)


def markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|")


def replace_section(text: str, header: str, body: str) -> str:
    body = body.rstrip() + "\n"
    pattern = rf"{re.escape(header)}\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, lambda _: f"{header}\n{body}", text, count=1, flags=re.S)
    suffix = "" if text.endswith("\n") else "\n"
    return f"{text}{suffix}\n{header}\n{body}"


def ensure_section(text: str, header: str, body: str) -> str:
    if header in text:
        return text
    suffix = "" if text.endswith("\n") else "\n"
    return f"{text}{suffix}\n{header}\n{body.rstrip()}\n"


def extract_log_rows(text: str) -> list[str]:
    match = re.search(r"## Experiment Log\n(.*?)(?=\n## |\Z)", text, flags=re.S)
    if match is None:
        return []
    rows: list[str] = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("| exp_id ") or stripped.startswith("| ---"):
            continue
        rows.append(stripped)
    return rows


def current_best_body(task_id: str, ledger_entry: dict[str, Any] | None, accepted_exp: str | None) -> str:
    entry = ledger_entry or {}
    source = accepted_exp if accepted_exp is not None else "ledger"
    return "\n".join(
        [
            "| status | local_points | memory_bytes_approx | params | updated_at | source |",
            "| --- | --- | --- | --- | --- | --- |",
            "| "
            + " | ".join(
                markdown_cell(value)
                for value in (
                    entry.get("status", ""),
                    entry.get("local_points", ""),
                    entry.get("memory_bytes_approx", ""),
                    entry.get("params", ""),
                    entry.get("updated_at", ""),
                    source,
                )
            )
            + " |",
        ]
    )


def default_note(task_id: str) -> str:
    return "\n".join(
        [
            f"# {task_id} Cost Experiments",
            "",
            "## Current Best",
            current_best_body(task_id, load_ledger().get(task_id), None),
            "",
            "## Active Hypotheses",
            "Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.",
            "",
            "| id | mode | hypothesis | status |",
            "| --- | --- | --- | --- |",
            "",
            "## Experiment Log",
            "| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "",
            "## Archived Summary",
            "- None yet.",
            "",
        ]
    )


def update_note(task_id: str, row: dict[str, Any], ledger_entry: dict[str, Any] | None, accepted_exp: str | None) -> None:
    note_path = ROOT / "experiments" / f"{task_id}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    if note_path.exists():
        text = note_path.read_text(encoding="utf-8")
    else:
        text = default_note(task_id)

    text = replace_section(text, "## Current Best", current_best_body(task_id, ledger_entry, accepted_exp))
    text = ensure_section(
        text,
        "## Active Hypotheses",
        "\n".join(
            [
                "Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.",
                "",
                "| id | mode | hypothesis | status |",
                "| --- | --- | --- | --- |",
            ]
        ),
    )
    text = ensure_section(
        text,
        "## Archived Summary",
        "- None yet.",
    )

    new_row = "| " + " | ".join(
        markdown_cell(row.get(key))
        for key in (
            "exp_id",
            "mode",
            "hypothesis_id",
            "status",
            "local_points",
            "memory_bytes_approx",
            "params",
            "delta",
            "decision",
            "takeaway",
        )
    ) + " |"
    rows = [*extract_log_rows(text), new_row][-LOG_LIMIT:]
    log_body = "\n".join(
        [
            "| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *rows,
        ]
    )
    text = replace_section(text, "## Experiment Log", log_body)
    note_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def promote_candidate(task_id: str, candidate_snapshot: Path, baseline_points: float | None) -> dict[str, Any]:
    canonical_path = solution_path(task_id)
    original_bytes = canonical_path.read_bytes()

    result: dict[str, Any] = {
        "attempted": True,
        "decision": "promotion_failed",
        "build": None,
        "score": None,
        "canonical_report": None,
        "restore_build": None,
        "restore_score": None,
    }

    try:
        shutil.copy2(candidate_snapshot, canonical_path)
        result["build"] = run_tool(["tools/build_task.py", "--task", task_id])
        result["score"] = run_tool(["tools/score_task.py", "--task", task_id])
        canonical_report = read_json(report_path(task_id))
        result["canonical_report"] = canonical_report
        canonical_points = score_value(canonical_report, "local_points")
        ok = (
            result["build"]["returncode"] == 0
            and result["score"]["returncode"] == 0
            and score_value(canonical_report, "status") == "passes_local"
            and isinstance(canonical_points, (int, float))
            and (baseline_points is None or float(canonical_points) > baseline_points)
        )
        if ok:
            result["decision"] = "promoted"
            return result
    finally:
        if result["decision"] != "promoted":
            canonical_path.write_bytes(original_bytes)
            result["restore_build"] = run_tool(["tools/build_task.py", "--task", task_id])
            result["restore_score"] = run_tool(["tools/score_task.py", "--task", task_id])

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    args = parser.parse_args()

    task_id = normalize_task_id(args.task)
    candidate_path = resolve_cli_path(args.candidate)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing candidate file: {candidate_path}")

    ledger = load_ledger()
    baseline = ledger.get(task_id)
    baseline_status = None if baseline is None else baseline.get("status")
    if baseline is None or baseline_status not in {"passes_local", "rule_invalid"}:
        print(f"{task_id} is not passes_local or rule_invalid in task_ledger.json; cost experiments require a passing or quarantined baseline.")
        return 2
    baseline_points = baseline.get("local_points")
    if baseline_status == "passes_local" and not isinstance(baseline_points, (int, float)):
        print(f"{task_id} has no numeric local_points baseline.")
        return 2
    comparison_baseline = float(baseline_points) if isinstance(baseline_points, (int, float)) else None

    exp_id, exp_dir = next_experiment_dir(task_id)
    candidate_snapshot = exp_dir / "candidate.py"
    if candidate_path.resolve() != candidate_snapshot.resolve():
        shutil.copy2(candidate_path, candidate_snapshot)

    model_path = exp_dir / f"{task_id}_{exp_id}.onnx"
    score_report_path = exp_dir / "score.json"
    experiment_report_path = exp_dir / "experiment.json"
    spec_path = task_spec_path(task_id)

    report: dict[str, Any] = {
        "task": task_id,
        "exp_id": exp_id,
        "mode": args.mode,
        "hypothesis_id": args.hypothesis_id,
        "candidate_input": relative(candidate_path),
        "candidate_snapshot": relative(candidate_snapshot),
        "onnx_path": relative(model_path),
        "score_report_path": relative(score_report_path),
        "baseline": baseline,
        "spec": {
            "path": relative(spec_path),
            "sha256": file_sha256(spec_path),
            "git": git_status_for(spec_path),
        },
        "build": None,
        "score": None,
        "candidate_report": None,
        "promotion": {"attempted": False, "decision": "not_attempted"},
        "decision": None,
    }

    build = run_tool(
        [
            "tools/build_task.py",
            "--task",
            task_id,
            "--solution",
            str(candidate_snapshot),
            "--onnx",
            str(model_path),
            "--no-ledger",
        ]
    )
    report["build"] = build

    if build["returncode"] != 0:
        report["decision"] = "build_failed"
        write_json(experiment_report_path, report)
        update_note(
            task_id,
            {
                "exp_id": exp_id,
                "mode": args.mode,
                "hypothesis_id": args.hypothesis_id,
                "status": "build_failed",
                "decision": "build_failed",
                "takeaway": "Candidate did not build.",
            },
            baseline,
            None,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    score = run_tool(
        [
            "tools/score_task.py",
            "--task",
            task_id,
            "--onnx",
            str(model_path),
            "--report",
            str(score_report_path),
            "--no-ledger",
        ]
    )
    report["score"] = score
    candidate_report = read_json(score_report_path)
    report["candidate_report"] = candidate_report

    status = score_value(candidate_report, "status") or "score_failed"
    candidate_points = score_value(candidate_report, "local_points")
    delta = points_delta(candidate_points, baseline_points)

    should_promote = score["returncode"] == 0 and status == "passes_local" and isinstance(candidate_points, (int, float)) and (
        (baseline_status == "rule_invalid") or (isinstance(delta, float) and delta > 0.0)
    )

    accepted_exp: str | None = None
    note_entry = baseline
    if should_promote:
        promotion = promote_candidate(task_id, candidate_snapshot, comparison_baseline)
        report["promotion"] = promotion
        report["decision"] = promotion["decision"]
        if promotion["decision"] == "promoted":
            accepted_exp = exp_id
            ledger = load_ledger()
            note_entry = ledger.get(task_id)
        else:
            note_entry = load_ledger().get(task_id, baseline)
    else:
        report["decision"] = "not_better" if status == "passes_local" else status

    write_json(experiment_report_path, report)

    decision = str(report["decision"])
    if decision == "promoted":
        takeaway = "Auto promoted after canonical re-score."
    elif status == "passes_local":
        takeaway = "Passed but did not improve local_points."
    else:
        takeaway = "Candidate did not pass local validation."

    update_note(
        task_id,
        {
            "exp_id": exp_id,
            "mode": args.mode,
            "hypothesis_id": args.hypothesis_id,
            "status": status,
            "local_points": candidate_points,
            "memory_bytes_approx": score_value(candidate_report, "memory_bytes_approx"),
            "params": score_value(candidate_report, "params"),
            "delta": None if delta is None else f"{delta:.12g}",
            "decision": decision,
            "takeaway": takeaway,
        },
        note_entry,
        accepted_exp,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if decision == "promotion_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
