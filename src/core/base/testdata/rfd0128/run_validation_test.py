"""RFD0128: 成果物の差を、比較の道具が見逃さないことを確認する。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("rfd0128_validation", Path(__file__).with_name("run_validation.py"))
VALIDATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATION)


class ArtifactComparisonTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="anet-rfd0128-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def make_run(self, name, warning="Warning message", config="alpha = 1\nbeta = 2\n", json_data=None):
        run = self.root / name
        (run / "config").mkdir(parents=True)
        (run / "config/DefaultDQNAgent.txt").write_text(config, encoding="utf-8")
        (run / "metrics.jsonl").write_text(json.dumps({
            "type": "json", "tag": "DefaultDQNAgent", "data": json_data or {"alpha": 1}}) + "\n", encoding="utf-8")
        (run / (name + ".log")).write_text(
            "12:00:00.000: [I] Ignored information\n"
            + "12:00:01.000: [W] " + warning + " path=" + str(run) + " run=" + name + "\n"
            + "12:00:02.000: [E] Error message\n", encoding="utf-8")
        normalized = self.root / (name + "-result/normalized.jsonl")
        normalized.parent.mkdir()
        normalized.write_text("", encoding="utf-8")
        normalized.with_name("inputs.json").write_text(json.dumps({"run": str(run)}), encoding="utf-8")
        return normalized

    def test_run_name_path_and_timestamp_do_not_change_warning_comparison(self):
        first, second = self.make_run("before"), self.make_run("after")
        records = VALIDATION.artifact_records(first)
        self.assertEqual(len(records["warnings"]), 2)
        self.assertTrue(all(VALIDATION.compare_artifacts(first, second).values()))

    def test_changed_warning_and_config_order_are_detected(self):
        first = self.make_run("before")
        second = self.make_run("after", warning="Changed warning", config="beta = 2\nalpha = 1\n")
        result = VALIDATION.compare_artifacts(first, second)
        self.assertFalse(result["warnings"])
        self.assertFalse(result["config"])
        self.assertTrue(result["config_json"])

    def test_changed_json_record_is_detected(self):
        first = self.make_run("before")
        second = self.make_run("after", json_data={"alpha": 2})
        self.assertFalse(VALIDATION.compare_artifacts(first, second)["config_json"])


if __name__ == "__main__":
    unittest.main()
