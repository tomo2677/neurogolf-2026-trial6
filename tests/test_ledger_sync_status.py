from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import neurogolf_onnx


class LedgerSyncStatusTests(unittest.TestCase):
    def test_compute_official_sync_status_buckets(self) -> None:
        self.assertEqual(
            neurogolf_onnx.compute_official_sync_status(
                {"official_status": "complete", "local_points": 10.0, "official_public_score": 10.01}
            ),
            "synced",
        )
        self.assertEqual(
            neurogolf_onnx.compute_official_sync_status(
                {"official_status": "complete", "local_points": 10.0, "official_public_score": 10.02}
            ),
            "drift",
        )
        self.assertEqual(
            neurogolf_onnx.compute_official_sync_status(
                {"official_status": "complete", "local_points": 10.0, "official_public_score": 0.0}
            ),
            "official_zero",
        )
        self.assertEqual(
            neurogolf_onnx.compute_official_sync_status(
                {"official_status": "poll_failed", "local_points": 10.0, "official_public_score": None}
            ),
            "pending",
        )
        self.assertEqual(
            neurogolf_onnx.compute_official_sync_status(
                {"official_status": "complete", "local_points": None, "official_public_score": 10.0}
            ),
            "unknown",
        )

    def test_write_ledger_backfills_status_and_places_column_after_delta(self) -> None:
        old_root = neurogolf_onnx.ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            neurogolf_onnx.ROOT = Path(tmpdir)
            try:
                neurogolf_onnx.write_ledger(
                    {
                        "task001": {
                            "task": "task001",
                            "status": "passes_local",
                            "local_points": 10.0,
                            "official_public_score": 10.0,
                            "official_status": "complete",
                        }
                    }
                )
                header = (Path(tmpdir) / "task_ledger.md").read_text(encoding="utf-8").splitlines()[0]
                columns = [column.strip() for column in header.strip("|").split("|")]
                delta_index = columns.index("official_delta_public_vs_local")
                self.assertEqual(columns[delta_index + 1], "official_sync_status")

                ledger = json.loads((Path(tmpdir) / "task_ledger.json").read_text(encoding="utf-8"))
                self.assertEqual(ledger["task001"]["official_sync_status"], "synced")
                self.assertEqual(ledger["task001"]["official_delta_public_vs_local"], 0.0)
            finally:
                neurogolf_onnx.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
