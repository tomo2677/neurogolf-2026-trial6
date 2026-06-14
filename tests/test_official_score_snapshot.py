from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import official_score_snapshot


def detail(task: str, points: float) -> dict:
    return {"task": task, "local_points_at_submit": points}


def manifest(points: list[tuple[str, float]], public_score: float | None, status: str = "complete") -> dict:
    details = [detail(task, value) for task, value in points]
    official = {"normalized_status": status, "observed_at": "2026-06-14T00:00:30+00:00"}
    if status == "complete":
        official.update(
            {
                "public_score": public_score,
                "submission_row": {"ref": "123", "date": "2026-06-14T00:00:10+00:00"},
            }
        )
    return {
        "run_id": "snapshot-test",
        "tasks": [task for task, _ in points],
        "task_details": details,
        "expected_public_score": official_score_snapshot.expected_public_score(details),
        "zip_sha256": "abc123",
        "submitted_at": "2026-06-14T00:00:00+00:00",
        "git": {"commit_short": "abc1234"},
        "paths": {"run_dir": "submissions/official_score_snapshot/snapshot-test"},
        "official": official,
    }


class OfficialScoreSnapshotTests(unittest.TestCase):
    def test_eligible_selection_excludes_nonpassing_or_missing_local_points(self) -> None:
        ledger = {
            "task001": {"status": "passes_local", "local_points": 11.0},
            "task002": {"status": "build_failed", "local_points": None},
            "task003": {"status": "passes_local", "local_points": None},
        }
        eligible, excluded = official_score_snapshot.eligible_tasks_for_ledger(ledger)
        self.assertEqual(eligible, [{"task": "task001", "local_points": 11.0}])
        self.assertEqual([row["task"] for row in excluded], ["task002", "task003"])

    def test_snapshot_metrics_use_submitted_scores(self) -> None:
        details = [detail("task001", 10.004), detail("task002", 10.005)]
        self.assertEqual(official_score_snapshot.expected_public_score(details), 20.01)
        metrics = official_score_snapshot.snapshot_metrics(2, 20.01, 20.0)
        self.assertEqual(metrics["delta_actual_vs_expected"], -0.01)
        self.assertEqual(metrics["average_public_score"], 10.0)
        self.assertEqual(metrics["scaled_public_score_400"], 4000.0)

    def test_resolve_matched_accepts_one_cent_and_syncs_ledger(self) -> None:
        m = manifest([("task001", 80.0), ("task002", 83.69)], 163.68)
        resolution = official_score_snapshot.resolution_for_manifest(m)
        self.assertEqual(resolution["status"], "matched")
        self.assertEqual(resolution["sync_tasks"], ["task001", "task002"])

        ledger = {
            "task001": {"task": "task001", "local_points": 80.0},
            "task002": {"task": "task002", "local_points": 83.69},
        }
        synced = official_score_snapshot.sync_ledger_rows(ledger, m, resolution["sync_tasks"], observed_at="now")
        self.assertEqual(synced["task001"]["official_status"], "complete")
        self.assertEqual(synced["task001"]["official_public_score"], 80.0)
        self.assertEqual(synced["task001"]["official_run_id"], "snapshot-test")

    def test_mismatch_and_not_complete_do_not_sync(self) -> None:
        mismatch = official_score_snapshot.resolution_for_manifest(manifest([("task001", 10.0), ("task002", 20.0)], 25.0))
        self.assertEqual(mismatch["status"], "mismatch")
        self.assertEqual(mismatch["sync_tasks"], [])

        not_complete = official_score_snapshot.resolution_for_manifest(
            manifest([("task001", 10.0)], None, status="quota_skipped")
        )
        self.assertEqual(not_complete["status"], "not_complete")
        self.assertEqual(not_complete["sync_tasks"], [])

    def test_history_is_newest_first_and_markdown_columns_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "official_score_snapshots.json"
            md_path = Path(tmpdir) / "official_score_snapshots.md"
            older = official_score_snapshot.snapshot_record_for_resolution(
                manifest([("task001", 10.0)], 10.0),
                {"status": "matched", "sync_tasks": ["task001"], "resolved_at": "older"},
            )
            newer_manifest = manifest([("task001", 12.0)], 12.0)
            newer_manifest["run_id"] = "snapshot-newer"
            newer = official_score_snapshot.snapshot_record_for_resolution(
                newer_manifest,
                {"status": "matched", "sync_tasks": ["task001"], "resolved_at": "newer"},
            )
            official_score_snapshot.upsert_snapshot_history(older, json_path=json_path, md_path=md_path)
            official_score_snapshot.upsert_snapshot_history(newer, json_path=json_path, md_path=md_path)

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual([row["run_id"] for row in data["snapshots"]], ["snapshot-newer", "snapshot-test"])

            header = md_path.read_text(encoding="utf-8").splitlines()[0]
            columns = [column.strip() for column in header.strip("|").split("|")]
            self.assertEqual(columns, official_score_snapshot.HISTORY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
