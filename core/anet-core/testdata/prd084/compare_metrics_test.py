import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("compare_metrics", Path(__file__).with_name("compare_metrics.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class NormalizeMetricsTest(unittest.TestCase):
    def test_on_comparison_retains_noisy_metrics_but_excludes_performance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            rows = [
                {"type": "scalar", "tag": "47_agent_noisy/q_diff", "step": 1, "value": 0.25},
                {"type": "scalar", "tag": "40_agent/loss", "step": 1, "value": 0.5},
                {"type": "scalar", "tag": "90_perf/time", "step": 1, "value": 3.0},
            ]
            (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            default = root / "off.jsonl"
            complete = root / "on.jsonl"
            metrics.normalize(str(run), str(default))
            metrics.normalize(str(run), str(complete), include_noisy=True)
            self.assertEqual(len(default.read_text().splitlines()), 1)
            retained = [json.loads(row)[0] for row in complete.read_text().splitlines()]
            self.assertEqual(retained, ["40_agent/loss", "47_agent_noisy/q_diff"])


if __name__ == "__main__":
    unittest.main()
