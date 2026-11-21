"""PRD130: 比較の道具が欠測や学習値の差を見逃さないことを確認する。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("prd130_validation", Path(__file__).with_name("run_validation.py"))
VALIDATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATION)


class ScalarComparisonTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="anet-prd130-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.previous_root = VALIDATION.ROOT
        VALIDATION.ROOT = self.root
        self.addCleanup(setattr, VALIDATION, "ROOT", self.previous_root)

    def make_run(self, label, case, reward=1.0, variance=2.0, empty=False, name=None):
        run = self.root / label / "runs" / (name or "p130_" + label + "_" + case)
        run.mkdir(parents=True)
        destination = self.root / label / case
        destination.mkdir()
        (destination / "inputs.json").write_text(json.dumps({"run": str(run)}), encoding="utf-8")
        records = [] if empty else [
            {"type": "scalar", "tag": "reward", "step": 100000, "value": reward},
            {"type": "scalar", "tag": VALIDATION.FEATURE_VARIANCE, "step": 100000, "value": variance}]
        records.append({"type": "scalar", "tag": "90_perf/time", "step": 100000, "value": 3.0})
        (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    def test_empty_learning_metrics_are_rejected(self):
        for label in ("before", "stage1"):
            self.make_run(label, "off", empty=True)
        with self.assertRaisesRegex(RuntimeError, "Empty"):
            VALIDATION.compare("before", "stage1", "off")

    def test_child_environment_excludes_test_switches(self):
        # 保存した古い実行体にも、呼び出し元の試作・検証スイッチを持ち込まない。
        key = "ANET_PRD130_TEST_SWITCH"
        with patch.dict(VALIDATION.os.environ, {key: "1", "OMP_NUM_THREADS": "8"}):
            environment = VALIDATION.make_environment()
        self.assertNotIn(key, environment)
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")

    def test_learning_scalar_difference_is_rejected(self):
        self.make_run("before", "off", reward=1.0)
        self.make_run("stage1", "off", reward=2.0)
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            VALIDATION.compare("before", "stage1", "off")

    def test_comparison_reads_the_recorded_run_path(self):
        # Run 名を推測せず、実行時に記録した出力先を比較する。
        for label in ("before", "stage1"):
            self.make_run(label, "off", name="run_from_manifest")
        VALIDATION.compare("before", "stage1", "off")

    def test_stage1_feedback_allows_only_feature_variance_to_change(self):
        self.make_run("before", "feedback", variance=2.0)
        self.make_run("stage1", "feedback", variance=3.0)
        VALIDATION.compare("before", "stage1", "feedback")

    def test_refactor_keeps_feature_variance_in_comparison(self):
        self.make_run("stage1", "feedback", variance=2.0)
        self.make_run("stage2", "feedback", variance=3.0)
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            VALIDATION.compare("stage1", "stage2", "feedback")

    def test_checkpoint_continuation_compares_recorded_load_runs_without_variance_exception(self):
        # 同じ checkpoint の続きも、時間・資源以外の全 scalar を比較する。
        self.make_run("stage1-reference", "load_feedback", variance=2.0, name="reference_continuation")
        self.make_run("stage2", "load_feedback", variance=3.0, name="refactored_continuation")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            VALIDATION.compare("stage1-reference", "stage2", "feedback", checkpoint=True)


if __name__ == "__main__":
    unittest.main()
