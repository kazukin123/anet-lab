#!/usr/bin/env python3

import gzip
from pathlib import Path
import tempfile
import unittest

from optuna_common import resolve_metrics_path, scalar_records


class OptunaMetricsGzipTest(unittest.TestCase):
    def test_prefers_raw_and_falls_back_to_gzip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            raw = run_dir / "metrics.jsonl"
            compressed = run_dir / "metrics.jsonl.gz"
            raw.write_text('{"type":"scalar","tag":"raw","step":1,"value":1}\n', encoding="utf-8")
            with gzip.open(compressed, "wt", encoding="utf-8") as stream:
                stream.write('{"type":"scalar","tag":"gzip","step":2,"value":2}\n')

            self.assertEqual(resolve_metrics_path(run_dir), raw)
            self.assertEqual([record["tag"] for record in scalar_records(run_dir)], ["raw"])
            raw.unlink()
            self.assertEqual(resolve_metrics_path(run_dir), compressed)
            self.assertEqual([record["tag"] for record in scalar_records(run_dir)], ["gzip"])


if __name__ == "__main__":
    unittest.main()
