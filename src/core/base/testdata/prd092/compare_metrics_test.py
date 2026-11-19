"""既存指標を残したまま、新診断のprefixだけを除外できる。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "prd084/compare_metrics.py"

class CompareMetricsTest(unittest.TestCase):
    def test_additional_prefix_keeps_existing_noisy(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            rows = [{"type":"scalar", "tag":tag, "step":1, "value":2.0}
                    for tag in ["loss", "47_agent_noisy/test", "49_agent_gradnoise/test", "90_perf/test"]]
            (run / "metrics.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows), encoding="utf-8")
            result = subprocess.run([sys.executable, str(TOOL), "normalize", str(run), str(run / "points"),
                                     "--include-noisy", "--exclude-prefix", "49_agent_gradnoise/"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            tags = [json.loads(line)[0] for line in (run / "points").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(tags, ["47_agent_noisy/test", "loss"])

if __name__ == "__main__":
    unittest.main()
