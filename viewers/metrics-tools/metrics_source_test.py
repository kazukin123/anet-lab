#!/usr/bin/env python3

import gzip
from pathlib import Path
import tempfile
import unittest

from metrics_source import open_metrics_text, resolve_metrics_path, resolve_run_metrics


class MetricsSourceTest(unittest.TestCase):
    def test_prefers_raw_and_falls_back_to_gzip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            raw = run_dir / "metrics.jsonl"
            compressed = run_dir / "metrics.jsonl.gz"
            raw.write_text("raw\n", encoding="utf-8")
            with gzip.open(compressed, "wt", encoding="utf-8") as stream:
                stream.write("gzip\n")

            self.assertEqual(resolve_run_metrics(run_dir), raw)
            raw.unlink()
            self.assertEqual(resolve_run_metrics(run_dir), compressed)
            self.assertEqual(resolve_metrics_path(run_dir / "metrics.jsonl"), compressed)
            with open_metrics_text(compressed) as stream:
                self.assertEqual(stream.read(), "gzip\n")


if __name__ == "__main__":
    unittest.main()
