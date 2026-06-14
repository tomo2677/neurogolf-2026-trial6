from __future__ import annotations

import unittest

from tools import score_up_gate


class ScoreUpGateTests(unittest.TestCase):
    def test_single_submit_threshold_is_three_points(self) -> None:
        below = score_up_gate.should_submit_for_values(12.99, 10.0, "complete")
        self.assertFalse(below["can_submit"])
        self.assertEqual(below["threshold"], 3.0)

        at_threshold = score_up_gate.should_submit_for_values(13.0, 10.0, "complete")
        self.assertTrue(at_threshold["can_submit"])
        self.assertEqual(at_threshold["reason"], "threshold_met")

    def test_missing_and_zero_official_scores_are_not_single_candidates(self) -> None:
        missing = score_up_gate.should_submit_for_values(13.0, None, "complete")
        self.assertEqual(missing["reason"], "missing_official_public_score")

        zero = score_up_gate.should_submit_for_values(13.0, 0.0, "complete")
        self.assertFalse(zero["can_submit"])
        self.assertEqual(zero["reason"], "official_zero_requires_repair")

    def test_batch_sync_candidates_are_sorted_by_delta_then_task(self) -> None:
        ledger = {
            "task002": {
                "status": "passes_local",
                "official_status": "complete",
                "local_points": 12.0,
                "official_public_score": 10.0,
            },
            "task001": {
                "status": "passes_local",
                "official_status": "complete",
                "local_points": 11.0,
                "official_public_score": 9.0,
            },
            "task003": {
                "status": "passes_local",
                "official_status": "complete",
                "local_points": 10.005,
                "official_public_score": 10.0,
            },
            "task004": {
                "status": "passes_local",
                "official_status": "complete",
                "local_points": 10.5,
                "official_public_score": 0.0,
            },
        }
        candidates = score_up_gate.batch_sync_candidates_for_ledger(ledger)
        self.assertEqual([row["task"] for row in candidates], ["task001", "task002"])


if __name__ == "__main__":
    unittest.main()
