#!/usr/bin/env python3

import gzip
from pathlib import Path
import tempfile
import unittest

from tb_bridge import RunState, tail_run_state


class TensorBoardGzipReaderTest(unittest.TestCase):
    def test_raw_incomplete_tail_is_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.jsonl"
            complete_line = b'{"value":1}\n'
            metrics_path.write_bytes(complete_line + b'{"value":')
            state = RunState(object(), metrics_path)

            lines, offset = tail_run_state(state)
            state.last_pos = offset
            with metrics_path.open("ab") as stream:
                stream.write(b'2}\n')
            next_lines, next_offset = tail_run_state(state)

            self.assertEqual(lines, ['{"value":1}'])
            self.assertEqual(offset, len(complete_line))
            self.assertEqual(next_lines, ['{"value":2}'])
            self.assertEqual(next_offset, metrics_path.stat().st_size)

    def test_keeps_gzip_stream_at_eof(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.jsonl.gz"
            with gzip.open(metrics_path, "wb") as stream:
                stream.write(b'{"value":1}\n')
            state = RunState(object(), metrics_path)

            lines, offset = tail_run_state(state)
            opened_stream = state.gzip_stream
            state.last_pos = offset
            next_lines, next_offset = tail_run_state(state)

            self.assertEqual(lines, ['{"value":1}'])
            self.assertEqual(next_lines, [])
            self.assertEqual(next_offset, offset)
            self.assertIs(state.gzip_stream, opened_stream)
            state.gzip_stream.close()


if __name__ == "__main__":
    unittest.main()
