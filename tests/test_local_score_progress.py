from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import local_score_progress


class LocalScoreProgressTests(unittest.TestCase):
    def test_eligible_selection_excludes_nonpassing_or_missing_local_points(self) -> None:
        ledger = {
            "task001": {"status": "passes_local", "local_points": 11.0, "official_sync_status": "synced"},
            "task002": {"status": "build_failed", "local_points": None},
            "task003": {"status": "passes_local", "local_points": None},
            "task004": {"status": "passes_local", "local_points": False},
        }
        eligible, excluded = local_score_progress.eligible_rows_for_ledger(ledger)
        self.assertEqual(
            eligible,
            [{"task": "task001", "local_points": 11.0, "official_sync_status": "synced"}],
        )
        self.assertEqual([row["task"] for row in excluded], ["task002", "task003", "task004"])

    def test_summary_metrics_and_sync_counts(self) -> None:
        rows = [
            {"task": "task001", "local_points": 10.004, "official_sync_status": "synced"},
            {"task": "task002", "local_points": 20.006, "official_sync_status": "drift"},
            {"task": "task003", "local_points": 30.0, "official_sync_status": "unknown"},
        ]
        summary = local_score_progress.summarize_rows(rows)
        self.assertEqual(summary["solved_task_count"], 3)
        self.assertEqual(summary["local_total_score"], 60.01)
        self.assertEqual(summary["local_average_score"], 20.003333)
        self.assertEqual(summary["local_scaled_score_400"], 8001.333333)
        self.assertEqual(summary["gap_to_8000"], -1.33)
        self.assertEqual(summary["official_synced_count"], 1)
        self.assertEqual(summary["official_drift_count"], 1)
        self.assertEqual(summary["best_task"], "task003:30.0")
        self.assertEqual(summary["worst_task"], "task001:10.004")

    def test_record_for_ledger_contains_task_list_and_excluded_rows(self) -> None:
        ledger = {
            "task001": {"status": "passes_local", "local_points": 10.0, "official_sync_status": "synced"},
            "task002": {"status": "fails_local", "local_points": 0.0},
        }
        record = local_score_progress.progress_record_for_ledger(ledger, recorded_at="now")
        self.assertEqual(record["recorded_at"], "now")
        self.assertEqual(record["solved_task_count"], 1)
        self.assertEqual(record["tasks"], ["task001"])
        self.assertEqual([row["task"] for row in record["excluded"]], ["task002"])

    def test_history_is_newest_first_and_markdown_columns_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "local_score_progress.json"
            md_path = Path(tmpdir) / "local_score_progress.md"
            older = {
                "recorded_at": "older",
                "git_commit": "abc",
                "solved_task_count": 1,
                "local_total_score": 10.0,
            }
            newer = {
                "recorded_at": "newer",
                "git_commit": "def",
                "solved_task_count": 2,
                "local_total_score": 20.0,
            }
            local_score_progress.upsert_progress_history(older, json_path=json_path, md_path=md_path)
            local_score_progress.upsert_progress_history(newer, json_path=json_path, md_path=md_path)

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual([row["recorded_at"] for row in data["records"]], ["newer", "older"])

            header = md_path.read_text(encoding="utf-8").splitlines()[0]
            columns = [column.strip() for column in header.strip("|").split("|")]
            self.assertEqual(columns, local_score_progress.HISTORY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
