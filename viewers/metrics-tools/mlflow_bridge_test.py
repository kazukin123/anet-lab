#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import pathlib
import re
import tempfile
import unittest
from unittest import mock

from mlflow_bridge import (
    LATEST_BATCHES_PER_HISTORY_BATCH,
    MonitoredRun,
    PollScheduleState,
    entry_to_metric,
    find_run_metrics,
    format_bridge_status,
    format_historical_status,
    load_config_params,
    poll_monitored_runs,
    poll_run,
    read_jsonl_batch,
    source_metrics_key,
    tracking_uri_from_path,
)


class LoadConfigParamsTest(unittest.TestCase):
    def test_loads_key_value_lines_and_preserves_value_delimiters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "config_data.txt"
            config_path.write_text(
                "alpha = 1\n"
                "group.[name].expression = left = right\n"
                "symbol@key = normalized\n"
                "\n"
                "empty = \n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_config_params(config_path),
                {
                    "alpha": "1",
                    "group.name.expression": "left = right",
                    "symbol_key": "normalized",
                    "empty": "",
                },
            )

    def test_rejects_invalid_or_duplicate_keys(self):
        cases = {
            "missing separator": "alpha: 1\n",
            "empty key": " = value\n",
            "whitespace key": "   = value\n",
            "duplicate key": "alpha = 1\nalpha = 2\n",
            "normalized key collision": "alpha[beta] = 1\nalphabeta = 2\n",
        }

        for case_name, contents in cases.items():
            with self.subTest(case_name=case_name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config_path = pathlib.Path(temp_dir) / "config_data.txt"
                    config_path.write_text(contents, encoding="utf-8")

                    with self.assertRaisesRegex(
                        ValueError,
                        rf"{re.escape(str(config_path))}:\d+:",
                    ):
                        load_config_params(config_path)

    def test_rejects_missing_file(self):
        config_path = pathlib.Path("missing") / "config_data.txt"

        with self.assertRaisesRegex(
            FileNotFoundError,
            rf"Config data file not found: {re.escape(str(config_path))}",
        ):
            load_config_params(config_path)


class RunDiscoveryTest(unittest.TestCase):
    def test_finds_only_direct_run_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_path = pathlib.Path(temp_dir)
            direct_metrics = runs_path / "run_direct" / "metrics.jsonl"
            direct_metrics.parent.mkdir()
            direct_metrics.write_text("", encoding="utf-8")

            nested_metrics = runs_path / "group" / "run_nested" / "metrics.jsonl"
            nested_metrics.parent.mkdir(parents=True)
            nested_metrics.write_text("", encoding="utf-8")

            deeper_metrics = runs_path / "run_parent" / "run_child" / "metrics.jsonl"
            deeper_metrics.parent.mkdir(parents=True)
            deeper_metrics.write_text("", encoding="utf-8")

            self.assertEqual(find_run_metrics(runs_path), [direct_metrics])

    def test_uses_raw_first_and_falls_back_to_gzip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_path = pathlib.Path(temp_dir)
            raw_run = runs_path / "run_raw"
            gzip_run = runs_path / "run_gzip"
            raw_run.mkdir()
            gzip_run.mkdir()
            raw = raw_run / "metrics.jsonl"
            raw.write_text("", encoding="utf-8")
            with gzip.open(raw_run / "metrics.jsonl.gz", "wb") as stream:
                stream.write(b"ignored\n")
            with gzip.open(gzip_run / "metrics.jsonl.gz", "wb") as stream:
                stream.write(b"{}\n")

            self.assertEqual(
                find_run_metrics(runs_path),
                [gzip_run / "metrics.jsonl.gz", raw],
            )
            self.assertEqual(
                source_metrics_key(raw),
                source_metrics_key(raw_run / "metrics.jsonl.gz"),
            )


class TrackingDatabaseTest(unittest.TestCase):
    def test_builds_absolute_sqlite_uri_for_selected_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_db = pathlib.Path(temp_dir) / "workspace" / "runs" / "mlflow.db"

            resolved_path, uri = tracking_uri_from_path(tracking_db)

            self.assertEqual(resolved_path, tracking_db.absolute())
            self.assertEqual(uri, f"sqlite:///{tracking_db.absolute().as_posix()}")


class JsonlReaderTest(unittest.TestCase):
    def test_keeps_partial_last_line_for_next_poll(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = pathlib.Path(temp_dir) / "metrics.jsonl"
            complete_line = b'{"step":0,"tag":"zero","value":0}\n'
            metrics_path.write_bytes(complete_line + b'{"step":1')

            entries, offset = read_jsonl_batch(metrics_path)

            self.assertEqual(entries, [{"step": 0, "tag": "zero", "value": 0}])
            self.assertEqual(offset, len(complete_line))

    def test_skips_complete_malformed_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = pathlib.Path(temp_dir) / "metrics.jsonl"
            invalid_line = b"not-json\n"
            valid_line = b'{"step":2,"tag":"metric","value":3}\n'
            metrics_path.write_bytes(invalid_line + valid_line)

            entries, offset = read_jsonl_batch(metrics_path)

            self.assertEqual(entries, [{"step": 2, "tag": "metric", "value": 3}])
            self.assertEqual(offset, len(invalid_line + valid_line))

    def test_converts_zero_value_and_step(self):
        metric = entry_to_metric(
            {"step": 0, "tag": "zero", "type": "scalar", "value": 0},
            timestamp_ms=123,
        )

        self.assertEqual(metric.key, "zero")
        self.assertEqual(metric.value, 0)
        self.assertEqual(metric.step, 0)
        self.assertEqual(metric.timestamp, 123)

    def test_reads_gzip_with_decompressed_byte_offset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = pathlib.Path(temp_dir) / "metrics.jsonl.gz"
            first = b'{"step":0,"tag":"zero","value":0}\n'
            second = b'{"step":1,"tag":"one","value":1}\n'
            with gzip.open(metrics_path, "wb") as stream:
                stream.write(first + second)

            entries, offset = read_jsonl_batch(metrics_path, len(first), max_lines=1)

            self.assertEqual(entries, [{"step": 1, "tag": "one", "value": 1}])
            self.assertEqual(offset, len(first + second))

    def test_poll_reuses_gzip_stream_between_batches(self):
        class FakeClient:
            def log_batch(self, *args, **kwargs):
                pass

            def set_tag(self, *args, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = pathlib.Path(temp_dir) / "metrics.jsonl.gz"
            with gzip.open(metrics_path, "wb") as stream:
                stream.write(
                    b'{"step":0,"tag":"zero","value":0}\n'
                    b'{"step":1,"tag":"one","value":1}\n'
                )
            monitored = MonitoredRun(metrics_path, "run-id", 0)

            with mock.patch("mlflow_bridge.READ_BATCH_SIZE", 1):
                self.assertTrue(poll_run(FakeClient(), monitored))
                opened_stream = monitored.gzip_stream
                self.assertTrue(poll_run(FakeClient(), monitored))

            self.assertIs(monitored.gzip_stream, opened_stream)
            monitored.gzip_stream.close()

    def test_poll_continues_same_decompressed_offset_after_raw_to_gzip_switch(self):
        class FakeClient:
            def __init__(self):
                self.metrics = []

            def log_batch(self, *args, **kwargs):
                self.metrics.extend(kwargs.get("metrics", []))

            def set_tag(self, *args, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "run_switch"
            run_dir.mkdir()
            raw = run_dir / "metrics.jsonl"
            compressed = run_dir / "metrics.jsonl.gz"
            first = b'{"step":0,"tag":"zero","value":0}\n'
            second = b'{"step":1,"tag":"one","value":1}\n'
            raw.write_bytes(first + second)
            monitored = MonitoredRun(raw, "run-id", len(first))
            with gzip.open(compressed, "wb") as stream:
                stream.write(first + second)
            raw.unlink()
            client = FakeClient()

            self.assertTrue(poll_run(client, monitored))

            self.assertEqual(monitored.metrics_path, compressed)
            self.assertEqual(monitored.offset, len(first + second))
            self.assertEqual([metric.key for metric in client.metrics], ["one"])
            monitored.gzip_stream.close()


class PollMonitoredRunsTest(unittest.TestCase):
    def test_prioritizes_latest_updated_run_before_polling_historical_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_path = pathlib.Path(temp_dir)
            old_metrics = runs_path / "run_old" / "metrics.jsonl"
            new_metrics = runs_path / "run_new" / "metrics.jsonl"
            old_metrics.parent.mkdir()
            new_metrics.parent.mkdir()
            old_metrics.write_text("", encoding="utf-8")
            new_metrics.write_text("", encoding="utf-8")
            os.utime(old_metrics, (100, 100))
            os.utime(new_metrics, (200, 200))

            monitored_runs = {
                "old": MonitoredRun(old_metrics, "old-id", 0),
                "new": MonitoredRun(new_metrics, "new-id", 0),
            }
            progress = {
                "run_new": iter([True, True, False]),
                "run_old": iter([True]),
            }
            calls = []
            schedule_state = PollScheduleState()

            def poll_side_effect(client, monitored_run):
                run_name = monitored_run.metrics_path.parent.name
                calls.append(run_name)
                return next(progress[run_name])

            with mock.patch(
                "mlflow_bridge.poll_run",
                side_effect=poll_side_effect,
            ):
                poll_results = [
                    poll_monitored_runs(
                        object(),
                        monitored_runs,
                        schedule_state,
                    )
                    for _ in range(3)
                ]

            self.assertEqual(
                [latest_progress for latest_progress, _ in poll_results],
                [True, True, False],
            )
            self.assertEqual(
                [
                    [monitored_run.run_id for monitored_run in historical_runs]
                    for _, historical_runs in poll_results
                ],
                [[], [], ["old-id"]],
            )
            self.assertEqual(
                calls,
                ["run_new", "run_new", "run_new", "run_old"],
            )

    def test_polls_historical_runs_during_continuous_latest_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_path = pathlib.Path(temp_dir)
            monitored_runs = {}
            for run_name, modified_time in (
                ("run_old_b", 50),
                ("run_old_a", 100),
                ("run_new", 200),
            ):
                metrics_path = runs_path / run_name / "metrics.jsonl"
                metrics_path.parent.mkdir()
                metrics_path.write_text("", encoding="utf-8")
                os.utime(metrics_path, (modified_time, modified_time))
                monitored_runs[run_name] = MonitoredRun(
                    metrics_path,
                    f"{run_name}-id",
                    0,
                )

            calls = []
            schedule_state = PollScheduleState()

            def poll_side_effect(client, monitored_run):
                calls.append(monitored_run.metrics_path.parent.name)
                return True

            with mock.patch(
                "mlflow_bridge.poll_run",
                side_effect=poll_side_effect,
            ):
                historical_run_ids = []
                for _ in range(2 * LATEST_BATCHES_PER_HISTORY_BATCH):
                    latest_progress, historical_runs = poll_monitored_runs(
                        object(),
                        monitored_runs,
                        schedule_state,
                    )
                    self.assertTrue(latest_progress)
                    historical_run_ids.extend(
                        monitored_run.run_id
                        for monitored_run in historical_runs
                    )

            self.assertEqual(
                calls,
                ["run_new"] * LATEST_BATCHES_PER_HISTORY_BATCH
                + ["run_old_a"]
                + ["run_new"] * LATEST_BATCHES_PER_HISTORY_BATCH
                + ["run_old_b"],
            )
            self.assertEqual(
                historical_run_ids,
                ["run_old_a-id", "run_old_b-id"],
            )

    def test_formats_latest_run_progress_for_console(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_path = pathlib.Path(temp_dir)
            old_metrics = runs_path / "run_old" / "metrics.jsonl"
            new_metrics = runs_path / "run_new" / "metrics.jsonl"
            old_metrics.parent.mkdir()
            new_metrics.parent.mkdir()
            old_metrics.write_text("", encoding="utf-8")
            with new_metrics.open("wb") as metrics_file:
                metrics_file.truncate(3 * 1024 * 1024)
            os.utime(old_metrics, (100, 100))
            os.utime(new_metrics, (200, 200))

            monitored_runs = {
                "old": MonitoredRun(old_metrics, "old-id", 0),
                "new": MonitoredRun(new_metrics, "new-id", 1024 * 1024),
            }

            self.assertEqual(
                format_bridge_status(monitored_runs),
                "[INFO] Latest progress: runs=2, run=run_new, "
                "offset=1.0/3.0 MiB, lag=2.0 MiB",
            )
            self.assertEqual(
                format_historical_status(monitored_runs["old"]),
                "[INFO] Historical progress: run=run_old, "
                "offset=0.0/0.0 MiB, lag=0.0 MiB",
            )


if __name__ == "__main__":
    unittest.main()
