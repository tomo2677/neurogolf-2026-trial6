from __future__ import annotations

import unittest

from tools import official_batch_sync


def detail(task: str, points: float) -> dict:
    return {"task": task, "local_points_at_submit": points}


def manifest(points: list[tuple[str, float]], public_score: float) -> dict:
    details = [detail(task, value) for task, value in points]
    return {
        "run_id": "batch-test",
        "tasks": [task for task, _ in points],
        "task_details": details,
        "expected_public_score": official_batch_sync.expected_public_score(details),
        "zip_sha256": "abc123",
        "submitted_at": "2026-06-14T00:00:00+00:00",
        "official": {
            "normalized_status": "complete",
            "public_score": public_score,
            "submission_row": {"ref": "123", "date": "2026-06-14T00:00:10+00:00"},
        },
    }


class OfficialBatchSyncTests(unittest.TestCase):
    def test_expected_score_rounds_to_two_decimals(self) -> None:
        details = [detail("task001", 10.004), detail("task002", 10.005)]
        self.assertEqual(official_batch_sync.expected_public_score(details), 20.01)

    def test_plan_selects_top_ten_by_delta(self) -> None:
        ledger = {
            f"task{i:03d}": {
                "status": "passes_local",
                "official_status": "complete",
                "local_points": float(i),
                "official_public_score": 0.5,
            }
            for i in range(1, 13)
        }
        candidates = official_batch_sync.batch_candidates_for_ledger(ledger)
        selected = official_batch_sync.select_batch(candidates, 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual(selected[0]["task"], "task012")
        self.assertEqual(selected[-1]["task"], "task003")

    def test_resolve_exact_match_syncs_all_tasks(self) -> None:
        m = manifest([("task001", 11.111), ("task002", 12.222)], 23.33)
        resolution = official_batch_sync.resolution_for_manifest(m)
        self.assertEqual(resolution["status"], "matched")
        self.assertEqual(resolution["sync_tasks"], ["task001", "task002"])

    def test_one_zero_probe_required_and_confirmed_syncs_remaining_tasks(self) -> None:
        m = manifest([("task001", 11.11), ("task002", 12.22), ("task003", 13.33)], 23.33)
        required = official_batch_sync.resolution_for_manifest(m)
        self.assertEqual(required["status"], "one_zero_probe_required")
        self.assertEqual(required["suspected_task"], "task003")

        zero_manifest = {
            "task": "task003",
            "official": {"normalized_status": "complete", "public_score": 0.0},
        }
        confirmed = official_batch_sync.resolution_for_manifest(m, zero_manifest)
        self.assertEqual(confirmed["status"], "one_zero_confirmed")
        self.assertEqual(confirmed["sync_tasks"], ["task001", "task002"])

    def test_ambiguous_mismatch_does_not_sync(self) -> None:
        m = manifest([("task001", 10.0), ("task002", 20.0)], 25.0)
        resolution = official_batch_sync.resolution_for_manifest(m)
        self.assertEqual(resolution["status"], "unresolved_mismatch")
        self.assertEqual(resolution["sync_tasks"], [])

    def test_sync_ledger_rows_updates_only_requested_tasks(self) -> None:
        m = manifest([("task001", 11.11), ("task002", 12.22)], 23.33)
        ledger = {
            "task001": {"task": "task001", "local_points": 11.11},
            "task002": {"task": "task002", "local_points": 12.22},
        }
        synced = official_batch_sync.sync_ledger_rows(ledger, m, ["task001"], observed_at="now")
        self.assertEqual(synced["task001"]["official_status"], "complete")
        self.assertEqual(synced["task001"]["official_public_score"], 11.11)
        self.assertNotIn("official_status", synced["task002"])


if __name__ == "__main__":
    unittest.main()
