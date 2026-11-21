"""Runnerを起動せず、保存Runの公開形式から受入判定を検証する。"""

import json
from pathlib import Path
import tempfile
import unittest

from run_acceptance import assess_case


class AcceptanceFallbackTest(unittest.TestCase):
    def assess(self, logs: str, records: list[dict], definitions: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="prd117-acceptance-") as temporary:
            directory = Path(temporary)
            (directory / (directory.name + ".log")).write_text(
                logs + "\n00:00:01.000: [I] Draining background observers: mode=wait timeout=3600s\n",
                encoding="utf-8")
            (directory / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            metadata = {"config": {"app.save_agent_on_close": "false"}, "scalar_defs": definitions}
            return assess_case({"id": 1, "name": "acceptance-fixture"}, metadata, directory,
                               ["fixture"], directory / "input.txt", 1, 0, 0.0)

    def test_no_learn_definitions_accepts_a_runner_learn_counter(self):
        # trainの公開ログとscalarのどちらからでも、実測したlearn stepを確認できる。
        for source in ("log", "metric"):
            with self.subTest(source=source):
                definitions = {"counter": {"event": "train", "scope": "train",
                                           "target": "runner", "source_key": "learn_step"}}
                logs = "00:00:00.000: [I] train_step=2 exp_step=8 learn_step=1" if source == "log" else ""
                records = [{"type": "scalar", "tag": "counter", "step": 2, "value": 1}] if source == "metric" else []
                self.assertTrue(self.assess(logs, records, definitions)["passed"])

    def test_no_counter_evidence_does_not_claim_learning(self):
        # exp stepやcheckpointの説明を、今回のRunの学習到達へ読み替えない。
        logs = "00:00:00.000: [I] train_step=2 exp_step=8 learn_step=0\nLoaded checkpoint: learn_step=100"
        self.assertFalse(self.assess(logs, [], {})["passed"])

    def test_declared_learn_scalar_still_requires_an_actual_record(self):
        definitions = {"loss": {"event": "learn", "scope": "train"}}
        logs = "00:00:00.000: [I] train_step=2 exp_step=8 learn_step=1"
        self.assertFalse(self.assess(logs, [], definitions)["passed"])


if __name__ == "__main__":
    unittest.main()
