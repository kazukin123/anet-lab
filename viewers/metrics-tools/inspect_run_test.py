#!/usr/bin/env python3

import contextlib
import gzip
import hashlib
import io
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import inspect_run as subject


class InspectRunTestBase(unittest.TestCase):
    def make_run(self, repo: Path, workspace: str, run_name: str) -> Path:
        run_dir = repo / "apps" / "runner" / "workspaces" / workspace / "runs" / run_name
        run_dir.mkdir(parents=True)
        return run_dir

    def workspaces_root(self, repo: Path) -> Path:
        return repo / "apps" / "runner" / "workspaces"

    def write_config(self, run_dir: Path, entries: dict) -> Path:
        config_dir = run_dir / "config"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "config_data.txt"
        text = "".join(f"{key} = {value}\n" for key, value in entries.items())
        config_path.write_text(text, encoding="utf-8")
        return config_path

    def scalar_lines(self, points: list) -> str:
        # points は (tag, step, value) の並び。MetricsLogger と同じ4 field だけを書く。
        lines = []
        for tag, step, value in points:
            lines.append(json.dumps({"step": step, "tag": tag, "type": "scalar", "value": value}))
        return "\n".join(lines) + "\n"

    def write_raw_master(self, run_dir: Path, points: list) -> Path:
        master_path = run_dir / "metrics.jsonl"
        master_path.write_text(self.scalar_lines(points), encoding="utf-8")
        return master_path

    def run_cli(self, repo: Path, argv: list):
        # public な CLI 契約だけを叩く。workspace 探索 root だけを一時 repo へ差し替える。
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(subject, "WORKSPACES_ROOT", self.workspaces_root(repo)):
            code = subject.main(argv, stdout=stdout, stderr=stderr)
        return code, stdout.getvalue(), stderr.getvalue()

    def run_cli_json(self, repo: Path, argv: list):
        code, out, err = self.run_cli(repo, argv)
        return code, json.loads(out), err


class TracerBulletTest(InspectRunTestBase):
    def test_extracts_metric_from_raw_master_for_workspace_run_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
            self.write_raw_master(run_dir, [("t/a", 0, 1.0), ("t/a", 10, 2.0), ("t/a", 20, 3.0)])

            code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            self.assertEqual(result["schema_version"], 1)
            run = result["runs"][0]
            self.assertEqual(run["input"], "run_a")
            self.assertEqual(run["run_name"], "run_a")
            self.assertEqual(run["workspace"], "ws1")
            self.assertEqual(run["run_dir"], str(run_dir.resolve()))
            self.assertEqual(run["metrics_source"]["selected"], "master")
            metric = run["metrics"][0]
            self.assertEqual(metric["tag"], "t/a")
            self.assertEqual(metric["step_axis"], "exp_step")
            self.assertEqual(metric["status"], "ok")
            window = metric["windows"][0]
            self.assertEqual(window["label"], "all")
            self.assertEqual(window["status"], "ok")
            self.assertEqual(window["count"], 3)
            self.assertEqual(window["mean"], 2.0)
            self.assertEqual(window["first"], 1.0)
            self.assertEqual(window["last"], 3.0)
            self.assertEqual(window["min_step"], 0)
            self.assertEqual(window["max_step"], 20)
            self.assertEqual(window["series"], [[0, 1.0], [10, 2.0], [20, 3.0]])


class RunResolutionTest(InspectRunTestBase):
    def test_resolves_absolute_and_relative_directory_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")

            code, result, _ = self.run_cli_json(repo, [str(run_dir)])
            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["run_name"], "run_a")
            self.assertEqual(result["runs"][0]["workspace"], "ws1")

            with contextlib.chdir(run_dir.parent):
                code, result, _ = self.run_cli_json(repo, ["run_a"])
            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["input"], "run_a")
            self.assertEqual(result["runs"][0]["run_dir"], str(run_dir.resolve()))

    def test_legacy_directory_resolves_only_by_explicit_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            legacy_dir = repo / "apps" / "runner" / "runs_lunarlander" / "run_legacy"
            legacy_dir.mkdir(parents=True)
            self.workspaces_root(repo).mkdir(parents=True)

            code, result, _ = self.run_cli_json(repo, [str(legacy_dir)])
            self.assertEqual(code, 0)
            self.assertIsNone(result["runs"][0]["workspace"])

            code, _, err = self.run_cli(repo, ["run_legacy"])
            self.assertEqual(code, 2)
            self.assertIn("run_legacy", err)

    def test_duplicate_run_name_across_workspaces_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            first = self.make_run(repo, "ws1", "run_a")
            second = self.make_run(repo, "ws2", "run_a")

            code, _, err = self.run_cli(repo, ["run_a"])

            self.assertEqual(code, 2)
            self.assertIn(str(first.resolve()), err)
            self.assertIn(str(second.resolve()), err)


class LightInspectionTest(InspectRunTestBase):
    def test_without_options_reports_artifacts_without_opening_master(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            config_path = self.write_config(run_dir, {"agent.class_id": "DefaultDQNAgent"})
            master_path = self.write_raw_master(run_dir, [("t/a", 0, 1.0)])
            (run_dir / "run_a.log").write_text("log", encoding="utf-8")
            (run_dir / "stdout.log").write_text("", encoding="utf-8")
            (run_dir / "stderr.log").write_text("", encoding="utf-8")
            (run_dir / "agent_close.anet").write_bytes(b"\x00\x01")
            nested = run_dir / "json"
            nested.mkdir()
            (nested / "buried.log").write_text("should not be listed", encoding="utf-8")

            expected_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

            with mock.patch.object(subject, "open_metrics_binary") as opener:
                code, result, _ = self.run_cli_json(repo, ["run_a"])

            self.assertEqual(code, 0)
            opener.assert_not_called()
            artifacts = result["runs"][0]["artifacts"]
            self.assertTrue(artifacts["config"]["exists"])
            self.assertEqual(artifacts["config"]["path"], str(config_path.resolve()))
            self.assertEqual(artifacts["config"]["sha256"], expected_sha)
            self.assertEqual(artifacts["config"]["size"], config_path.stat().st_size)
            self.assertTrue(artifacts["master"]["raw_exists"])
            self.assertFalse(artifacts["master"]["gzip_exists"])
            self.assertEqual(artifacts["master"]["path"], str(master_path.resolve()))
            self.assertEqual(artifacts["master"]["kind"], "jsonl")
            self.assertFalse(artifacts["cache"]["exists"])
            listed = sorted(item["name"] for item in artifacts["files"])
            self.assertEqual(listed, ["agent_close.anet", "run_a.log", "stderr.log", "stdout.log"])

    def test_missing_config_and_master_are_reported_as_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_run(repo, "ws1", "run_a")

            code, result, _ = self.run_cli_json(repo, ["run_a"])

            self.assertEqual(code, 0)
            artifacts = result["runs"][0]["artifacts"]
            self.assertFalse(artifacts["config"]["exists"])
            self.assertIsNone(artifacts["config"]["sha256"])
            self.assertIsNone(artifacts["master"]["path"])
            self.assertEqual(artifacts["files"], [])


class ConfigSelectorTest(InspectRunTestBase):
    def test_exact_glob_and_bracket_literal_selectors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(
                run_dir,
                {
                    "agent.class_id": "DefaultDQNAgent",
                    "agent.device_type": "1",
                    "train.eval.[eval1].run_mode": "eval1",
                    "train.eval.[eval2].run_mode": "eval2",
                },
            )

            code, result, _ = self.run_cli_json(
                repo,
                [
                    "run_a",
                    "--config-key",
                    "agent.class_id",
                    "--config-key",
                    "agent.*",
                    "--config-key",
                    "train.eval.[eval1].run_mode",
                    "--config-key",
                    "train.eval.[*].run_mode",
                    "--config-key",
                    "nope.*",
                ],
            )

            self.assertEqual(code, 0)
            config = result["runs"][0]["config"]
            pairs = [(item["key"], item["value"]) for item in config["values"]]
            self.assertEqual(
                pairs,
                [
                    ("agent.class_id", "DefaultDQNAgent"),
                    ("agent.device_type", "1"),
                    ("train.eval.[eval1].run_mode", "eval1"),
                    ("train.eval.[eval2].run_mode", "eval2"),
                ],
            )
            statuses = {item["selector"]: item["status"] for item in config["selectors"]}
            self.assertEqual(statuses["agent.class_id"], "ok")
            self.assertEqual(statuses["train.eval.[eval1].run_mode"], "ok")
            self.assertEqual(statuses["train.eval.[*].run_mode"], "ok")
            self.assertEqual(statuses["nope.*"], "missing")

    def test_diff_reports_differing_keys_and_absence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            first = self.make_run(repo, "ws1", "run_a")
            second = self.make_run(repo, "ws1", "run_b")
            self.write_config(first, {"agent.lr": "1e-3", "agent.gamma": "0.99", "only_a": "x"})
            self.write_config(second, {"agent.lr": "5e-4", "agent.gamma": "0.99"})

            code, result, _ = self.run_cli_json(repo, ["run_a", "run_b", "--diff-config"])

            self.assertEqual(code, 0)
            diff = {item["key"]: item["runs"] for item in result["config_diff"]}
            self.assertEqual(sorted(diff), ["agent.lr", "only_a"])
            self.assertEqual([entry["value"] for entry in diff["agent.lr"]], ["1e-3", "5e-4"])
            self.assertEqual([entry["present"] for entry in diff["only_a"]], [True, False])
            self.assertIsNone(diff["only_a"][1]["value"])

    def test_diff_is_limited_by_config_key_and_empty_for_single_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            first = self.make_run(repo, "ws1", "run_a")
            second = self.make_run(repo, "ws1", "run_b")
            self.write_config(first, {"agent.lr": "1e-3", "env.seed": "1"})
            self.write_config(second, {"agent.lr": "5e-4", "env.seed": "2"})

            code, result, _ = self.run_cli_json(
                repo, ["run_a", "run_b", "--diff-config", "--config-key", "agent.*"]
            )
            self.assertEqual(code, 0)
            self.assertEqual([item["key"] for item in result["config_diff"]], ["agent.lr"])

            code, result, _ = self.run_cli_json(repo, ["run_a", "--diff-config"])
            self.assertEqual(code, 0)
            self.assertEqual(result["config_diff"], [])

    def test_config_selector_missing_in_all_runs_exits_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {"agent.lr": "1e-3"})

            code, result, _ = self.run_cli(repo, ["run_a", "--config-key", "absent.*"])

            self.assertEqual(code, 1)
            self.assertEqual(json.loads(result)["runs"][0]["config"]["values"], [])


class WindowTest(InspectRunTestBase):
    def make_axis_run(self, repo: Path, run_name: str, steps: list, axis_token: str = "$exp_step"):
        run_dir = self.make_run(repo, "ws1", run_name)
        self.write_config(run_dir, {"metrics.scalar.[t/a]": f"a_key @train {axis_token}"})
        self.write_raw_master(run_dir, [("t/a", step, float(step)) for step in steps])
        return run_dir

    def test_absolute_window_suffixes_and_inclusive_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_axis_run(repo, "run_a", [0, 1000, 2000, 1_000_000, 2_000_000])

            code, result, _ = self.run_cli_json(
                repo,
                ["run_a", "--metric", "t/a", "--window", "1k:2K", "--window", "1M:2m",
                 "--window", "2000:2000"],
            )

            self.assertEqual(code, 0)
            windows = result["runs"][0]["metrics"][0]["windows"]
            self.assertEqual([w["label"] for w in windows], ["1k:2K", "1M:2m", "2000:2000"])
            self.assertEqual([w["kind"] for w in windows], ["absolute"] * 3)
            self.assertEqual([w["start"] for w in windows], [1000, 1_000_000, 2000])
            self.assertEqual([w["end"] for w in windows], [2000, 2_000_000, 2000])
            self.assertEqual([w["count"] for w in windows], [2, 2, 1])
            self.assertEqual(windows[2]["first"], 2000.0)

    def test_percentage_window_resolves_per_run_max_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_axis_run(repo, "run_a", list(range(0, 101, 10)))
            self.make_axis_run(repo, "run_b", list(range(0, 51, 10)))

            code, result, _ = self.run_cli_json(
                repo, ["run_a", "run_b", "--metric", "t/a", "--window", "80%:100%"]
            )

            self.assertEqual(code, 0)
            first = result["runs"][0]["metrics"][0]["windows"][0]
            second = result["runs"][1]["metrics"][0]["windows"][0]
            self.assertEqual(first["kind"], "percentage")
            self.assertEqual(first["label"], "80%:100%")
            self.assertEqual((first["start"], first["end"]), (80, 100))
            self.assertEqual((second["start"], second["end"]), (40, 50))
            self.assertEqual(first["count"], 3)
            self.assertEqual(second["count"], 2)

    def test_absolute_and_percentage_windows_combine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_axis_run(repo, "run_a", list(range(0, 101, 10)))

            code, result, _ = self.run_cli_json(
                repo, ["run_a", "--metric", "t/a", "--window", "0:20", "--window", "50%:100%"]
            )

            self.assertEqual(code, 0)
            self.assertEqual(
                [w["label"] for w in result["windows"]], ["0:20", "50%:100%"]
            )
            windows = result["runs"][0]["metrics"][0]["windows"]
            self.assertEqual([w["count"] for w in windows], [3, 6])

    def test_invalid_window_syntax_exits_two(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_axis_run(repo, "run_a", [0, 10])

            for bad in ["10:0", "0:1.5", "0%:120%", "0:50%", "0:", ":10", "-1:5", "abc"]:
                code, _, err = self.run_cli(repo, ["run_a", "--metric", "t/a", "--window", bad])
                self.assertEqual(code, 2, msg=f"window={bad}")
                self.assertIn(bad, err)

    def test_percentage_on_unknown_axis_exits_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {"agent.class_id": "DefaultDQNAgent"})
            self.write_raw_master(run_dir, [("t/a", 0, 1.0), ("t/a", 10, 2.0)])

            code, _, err = self.run_cli(repo, ["run_a", "--metric", "t/a", "--window", "0%:100%"])

            self.assertEqual(code, 1)
            self.assertIn("t/a", err)
            self.assertIn("run_a", err)


class ProfileTest(InspectRunTestBase):
    def write_profile(self, repo: Path, payload) -> Path:
        profile_path = repo / "profile.json"
        profile_path.write_text(json.dumps(payload), encoding="utf-8")
        return profile_path

    def valid_payload(self, **overrides):
        payload = {
            "version": 1,
            "name": "dropmerge-iqn-k-search",
            "metrics": ["t/a"],
            "config_keys": ["agent.*"],
            "windows": ["0:20"],
        }
        payload.update(overrides)
        return payload

    def make_profile_run(self, repo: Path, run_name: str = "run_a") -> Path:
        run_dir = self.make_run(repo, "ws1", run_name)
        self.write_config(
            run_dir,
            {"metrics.scalar.[t/a]": "a_key @train $exp_step", "agent.lr": "1e-3"},
        )
        self.write_raw_master(run_dir, [("t/a", step, float(step)) for step in range(0, 41, 10)])
        return run_dir

    def test_profile_supplies_metrics_config_keys_and_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_profile_run(repo)
            profile_path = self.write_profile(repo, self.valid_payload())

            code, result, _ = self.run_cli_json(repo, ["run_a", "--profile", str(profile_path)])

            self.assertEqual(code, 0)
            self.assertEqual(result["profile"]["name"], "dropmerge-iqn-k-search")
            self.assertEqual(result["profile"]["path"], str(profile_path.resolve()))
            self.assertEqual([w["label"] for w in result["windows"]], ["0:20"])
            run = result["runs"][0]
            self.assertEqual([m["tag"] for m in run["metrics"]], ["t/a"])
            self.assertEqual(run["metrics"][0]["windows"][0]["count"], 3)
            self.assertEqual([item["key"] for item in run["config"]["values"]], ["agent.lr"])

    def test_cli_appends_to_profile_arrays_and_replaces_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_profile_run(repo)
            self.write_config(
                run_dir,
                {
                    "metrics.scalar.[t/a]": "a_key @train $exp_step",
                    "metrics.scalar.[t/b]": "b_key @learn",
                    "agent.lr": "1e-3",
                },
            )
            self.write_raw_master(
                run_dir,
                [("t/a", step, float(step)) for step in range(0, 41, 10)]
                + [("t/b", step, 1.0) for step in range(0, 41, 10)],
            )
            profile_path = self.write_profile(repo, self.valid_payload(metrics=["t/a", "t/b"]))

            code, result, _ = self.run_cli_json(
                repo,
                [
                    "run_a",
                    "--profile",
                    str(profile_path),
                    "--metric",
                    "t/b",
                    "--metric",
                    "t/a",
                    "--window",
                    "0:10",
                    "--window",
                    "30:40",
                ],
            )

            self.assertEqual(code, 0)
            run = result["runs"][0]
            self.assertEqual([m["tag"] for m in run["metrics"]], ["t/a", "t/b"])
            self.assertEqual([w["label"] for w in result["windows"]], ["0:10", "30:40"])
            self.assertEqual([w["count"] for w in run["metrics"][0]["windows"]], [2, 2])

    def test_empty_profile_windows_mean_full_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_profile_run(repo)
            profile_path = self.write_profile(repo, self.valid_payload(windows=[]))

            code, result, _ = self.run_cli_json(repo, ["run_a", "--profile", str(profile_path)])

            self.assertEqual(code, 0)
            self.assertEqual([w["label"] for w in result["windows"]], ["all"])
            self.assertEqual(result["runs"][0]["metrics"][0]["windows"][0]["count"], 5)

    def test_profile_contract_violations_exit_two(self):
        cases = {
            "unknown_field": {"extra": 1},
            "unknown_version": {"version": 2},
            "bool_version": {"version": True},
            "blank_name": {"name": "   "},
            "non_string_metric": {"metrics": [1]},
            "empty_string_selector": {"config_keys": [""]},
            "both_selector_arrays_empty": {"metrics": [], "config_keys": []},
            "windows_not_array": {"windows": "0:20"},
            "bad_window": {"windows": ["10:0"]},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_profile_run(repo)
            for label, overrides in cases.items():
                payload = self.valid_payload(**overrides)
                profile_path = self.write_profile(repo, payload)
                code, _, err = self.run_cli(repo, ["run_a", "--profile", str(profile_path)])
                self.assertEqual(code, 2, msg=label)
                self.assertTrue(err.strip(), msg=label)

    def test_missing_profile_field_and_unreadable_profile_exit_two(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_profile_run(repo)

            payload = self.valid_payload()
            del payload["windows"]
            profile_path = self.write_profile(repo, payload)
            code, _, err = self.run_cli(repo, ["run_a", "--profile", str(profile_path)])
            self.assertEqual(code, 2)
            self.assertIn("windows", err)

            broken = repo / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            code, _, _ = self.run_cli(repo, ["run_a", "--profile", str(broken)])
            self.assertEqual(code, 2)

            code, _, _ = self.run_cli(repo, ["run_a", "--profile", str(repo / "nope.json")])
            self.assertEqual(code, 2)


class MasterSourceTest(InspectRunTestBase):
    def make_source_run(self, repo: Path, run_name: str = "run_a") -> Path:
        run_dir = self.make_run(repo, "ws1", run_name)
        self.write_config(run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
        return run_dir

    def write_gzip_master(self, run_dir: Path, points: list, terminated: bool = True) -> Path:
        gzip_path = run_dir / "metrics.jsonl.gz"
        text = self.scalar_lines(points)
        if not terminated:
            text = text.rstrip("\n")
        with gzip.open(gzip_path, "wb") as handle:
            handle.write(text.encode("utf-8"))
        return gzip_path

    def test_prefers_raw_and_falls_back_to_gzip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_source_run(repo)
            raw_path = self.write_raw_master(run_dir, [("t/a", 0, 1.0), ("t/a", 10, 2.0)])
            gzip_path = self.write_gzip_master(run_dir, [("t/a", 0, 9.0)])

            code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])
            self.assertEqual(code, 0)
            source = result["runs"][0]["metrics_source"]
            self.assertEqual(source["master_path"], str(raw_path.resolve()))
            self.assertEqual(result["runs"][0]["artifacts"]["master"]["kind"], "jsonl")
            self.assertEqual(result["runs"][0]["metrics"][0]["windows"][0]["count"], 2)

            raw_path.unlink()
            code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])
            self.assertEqual(code, 0)
            source = result["runs"][0]["metrics_source"]
            self.assertEqual(source["master_path"], str(gzip_path.resolve()))
            self.assertEqual(result["runs"][0]["artifacts"]["master"]["kind"], "jsonl.gz")
            self.assertEqual(result["runs"][0]["metrics"][0]["windows"][0]["count"], 1)

    def test_run_without_master_reports_source_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_source_run(repo)

            code, result, _ = self.run_cli(repo, ["run_a", "--metric", "t/a"])
            parsed = json.loads(result)

            self.assertEqual(code, 1)
            self.assertEqual(parsed["runs"][0]["metrics"][0]["status"], "source_missing")

    def test_raw_snapshot_drops_unterminated_tail_and_flags_provisional(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_source_run(repo)
            master = run_dir / "metrics.jsonl"
            text = self.scalar_lines([("t/a", 0, 1.0), ("t/a", 10, 2.0)])
            master.write_text(text + '{"step":20,"tag":"t/a","type":"scal', encoding="utf-8")

            code, result, err = self.run_cli(repo, ["run_a", "--metric", "t/a"])
            parsed = json.loads(result)

            self.assertEqual(code, 0)
            self.assertEqual(parsed["runs"][0]["metrics"][0]["windows"][0]["count"], 2)
            self.assertTrue(parsed["runs"][0]["metrics_source"]["provisional"])
            self.assertIn("unterminated", err)

    def test_raw_append_during_read_is_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_source_run(repo)
            master = self.write_raw_master(
                run_dir, [("t/a", 0, 1.0), ("t/a", 10, 2.0), ("t/a", 20, 3.0)]
            )
            real_open = subject.open_metrics_binary

            def opener(path):
                handle = real_open(path)
                # 読み取り開始後の追記は同じ結果へ混ぜない。
                with master.open("a", encoding="utf-8") as appended:
                    appended.write(self.scalar_lines([("t/a", 30, 4.0)]))
                return handle

            with mock.patch.object(subject, "open_metrics_binary", side_effect=opener):
                code, result, err = self.run_cli(repo, ["run_a", "--metric", "t/a"])
            parsed = json.loads(result)

            self.assertEqual(code, 0)
            self.assertEqual(parsed["runs"][0]["metrics"][0]["windows"][0]["count"], 3)
            self.assertTrue(parsed["runs"][0]["metrics_source"]["source_changed_during_read"])
            self.assertIn("changed", err)

    def test_gzip_unterminated_tail_is_source_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_source_run(repo)
            self.write_gzip_master(run_dir, [("t/a", 0, 1.0)], terminated=False)

            code, result, _ = self.run_cli(repo, ["run_a", "--metric", "t/a"])
            parsed = json.loads(result)

            self.assertEqual(code, 1)
            self.assertEqual(parsed["runs"][0]["metrics"][0]["status"], "source_error")

    def test_malformed_json_line_is_source_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_source_run(repo)
            (run_dir / "metrics.jsonl").write_text(
                self.scalar_lines([("t/a", 0, 1.0)]) + "{oops\n", encoding="utf-8"
            )

            code, result, _ = self.run_cli(repo, ["run_a", "--metric", "t/a"])
            parsed = json.loads(result)

            self.assertEqual(code, 1)
            self.assertEqual(parsed["runs"][0]["metrics"][0]["status"], "source_error")

    def test_single_pass_opens_each_master_once_for_multiple_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            for name in ("run_a", "run_b"):
                run_dir = self.make_run(repo, "ws1", name)
                self.write_config(
                    run_dir,
                    {
                        "metrics.scalar.[t/a]": "a_key @train $exp_step",
                        "metrics.scalar.[t/b]": "b_key @train $exp_step",
                    },
                )
                self.write_raw_master(
                    run_dir, [("t/a", 0, 1.0), ("t/b", 0, 2.0), ("t/a", 10, 3.0)]
                )
            real_open = subject.open_metrics_binary

            with mock.patch.object(subject, "open_metrics_binary", side_effect=real_open) as opener:
                code, result, _ = self.run_cli_json(
                    repo,
                    ["run_a", "run_b", "--metric", "t/a", "--metric", "t/b",
                     "--window", "0:100", "--window", "0:0"],
                )

            self.assertEqual(code, 0)
            self.assertEqual(opener.call_count, 2)
            self.assertEqual(len(result["runs"]), 2)
            self.assertEqual([m["tag"] for m in result["runs"][0]["metrics"]], ["t/a", "t/b"])


class StatsAndSeriesTest(InspectRunTestBase):
    def make_stats_run(self, repo: Path, points: list, run_name: str = "run_a") -> Path:
        run_dir = self.make_run(repo, "ws1", run_name)
        self.write_config(run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
        self.write_raw_master(run_dir, points)
        return run_dir

    def test_statistics_match_hand_computed_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_stats_run(
                repo,
                [("t/a", 0, 1.0), ("t/a", 10, 2.0), ("t/a", 20, 3.0), ("t/a", 30, 4.0)],
            )

            code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            window = result["runs"][0]["metrics"][0]["windows"][0]
            self.assertEqual(window["count"], 4)
            self.assertAlmostEqual(window["mean"], 2.5)
            self.assertAlmostEqual(window["population_std"], math.sqrt(1.25))
            self.assertEqual(window["min"], 1.0)
            self.assertEqual(window["max"], 4.0)
            self.assertEqual((window["first"], window["first_step"]), (1.0, 0))
            self.assertEqual((window["last"], window["last_step"]), (4.0, 30))
            self.assertEqual((window["min_step"], window["max_step"]), (0, 30))

    def test_empty_window_reports_empty_status_with_null_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_stats_run(repo, [("t/a", 0, 1.0), ("t/a", 10, 2.0)])

            code, result, _ = self.run_cli_json(
                repo, ["run_a", "--metric", "t/a", "--window", "1000:2000"]
            )

            self.assertEqual(code, 0)
            window = result["runs"][0]["metrics"][0]["windows"][0]
            self.assertEqual(window["status"], "empty")
            self.assertEqual(window["count"], 0)
            for name in ("mean", "population_std", "min", "max", "first", "last",
                         "first_step", "last_step", "min_step", "max_step"):
                self.assertIsNone(window[name], msg=name)
            self.assertEqual(window["series"], [])

    def test_invalid_values_are_excluded_and_counted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
            lines = [
                '{"step":0,"tag":"t/a","type":"scalar","value":1.0}',
                '{"step":10,"tag":"t/a","type":"scalar","value":null}',
                '{"step":20,"tag":"t/a","type":"scalar","value":"nope"}',
                '{"step":30,"tag":"t/a","type":"scalar","value":NaN}',
                '{"step":40,"tag":"t/a","type":"scalar","value":Infinity}',
                '{"step":50,"tag":"t/a","type":"scalar","value":1e39}',
                '{"step":60,"tag":"t/a","type":"scalar","value":true}',
                '{"step":70,"tag":"t/a","type":"scalar","value":5.0}',
                '{"type":"meta","event":"start","timestamp":"2026-08-15T00:00:00"}',
                '{"type":"json","tag":"backend","data":{"x":1},"timestamp":"2026-08-15T00:00:00"}',
            ]
            (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

            code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            metric = result["runs"][0]["metrics"][0]
            self.assertEqual(metric["status"], "ok")
            self.assertEqual(metric["excluded"], 6)
            self.assertEqual(metric["windows"][0]["count"], 2)
            self.assertEqual(metric["windows"][0]["series"], [[0, 1.0], [70, 5.0]])

    def test_step_regression_quarantines_tag_and_keeps_valid_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(
                run_dir,
                {
                    "metrics.scalar.[t/a]": "a_key @train $exp_step",
                    "metrics.scalar.[t/b]": "b_key @train $exp_step",
                },
            )
            self.write_raw_master(
                run_dir,
                [
                    ("t/a", 0, 1.0),
                    ("t/b", 0, 7.0),
                    ("t/a", 10, 2.0),
                    ("t/a", 5, 3.0),
                    ("t/a", 20, 4.0),
                    ("t/b", 10, 8.0),
                ],
            )

            code, result, _ = self.run_cli_json(
                repo, ["run_a", "--metric", "t/a", "--metric", "t/b"]
            )

            self.assertEqual(code, 0)
            first, second = result["runs"][0]["metrics"]
            self.assertEqual(first["status"], "quarantined")
            self.assertEqual(first["windows"][0]["series"], [[0, 1.0], [10, 2.0]])
            self.assertEqual(second["status"], "ok")
            self.assertEqual(second["windows"][0]["count"], 2)

    def test_series_keeps_all_points_up_to_the_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            points = [("t/a", step, float(step)) for step in range(128)]
            self.make_stats_run(repo, points)

            code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            series = result["runs"][0]["metrics"][0]["windows"][0]["series"]
            self.assertEqual(len(series), 128)
            self.assertEqual(series, [[step, float(step)] for step in range(128)])

    def test_series_downsamples_deterministically_and_keeps_endpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            points = [
                ("t/a", step, math.sin(step) * 100.0 + step)
                for step in range(1000)
            ]
            self.make_stats_run(repo, points)

            code, first, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])
            _, second, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            series = first["runs"][0]["metrics"][0]["windows"][0]["series"]
            self.assertLessEqual(len(series), 128)
            self.assertGreater(len(series), 64)
            self.assertEqual(series[0][0], 0)
            self.assertEqual(series[-1][0], 999)
            self.assertEqual([point[0] for point in series],
                             sorted(point[0] for point in series))
            self.assertEqual(series, second["runs"][0]["metrics"][0]["windows"][0]["series"])

            values = [point[2] for point in points]
            series_values = [point[1] for point in series]
            self.assertIn(max(values), series_values)
            self.assertIn(min(values), series_values)

    def test_series_never_exceeds_limit_for_monotonic_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            # 単調増加では bucket ごとの min/max/last が重複しにくく、点数上限の最悪ケースになる。
            for size in (129, 500, 5000):
                run_name = f"run_{size}"
                points = [("t/a", step, float(step)) for step in range(size)]
                self.make_stats_run(repo, points, run_name=run_name)
                code, result, _ = self.run_cli_json(repo, [run_name, "--metric", "t/a"])
                series = result["runs"][0]["metrics"][0]["windows"][0]["series"]
                self.assertEqual(code, 0)
                self.assertLessEqual(len(series), 128, msg=f"size={size}")
                self.assertEqual(series[0], [0, 0.0], msg=f"size={size}")
                self.assertEqual(series[-1], [size - 1, float(size - 1)], msg=f"size={size}")


CACHE_TABLE_DDL = {
    "tags": """CREATE TABLE tags(
        id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL,
        type TEXT NOT NULL CHECK(type = 'scalar'),
        status TEXT NOT NULL CHECK(status IN ('ok', 'error')),
        error_code TEXT, error_message TEXT, error_source_offset INTEGER,
        error_previous_step INTEGER, error_step INTEGER)""",
    "scalars": """CREATE TABLE scalars(
        tag_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
        step INTEGER NOT NULL, value REAL NOT NULL,
        PRIMARY KEY(tag_id, ordinal)) WITHOUT ROWID""",
    "scalars_lod": """CREATE TABLE scalars_lod(
        tag_id INTEGER NOT NULL, level INTEGER NOT NULL, bucket INTEGER NOT NULL,
        cnt INTEGER NOT NULL, step_first INTEGER NOT NULL, step_last INTEGER NOT NULL,
        min_ordinal INTEGER NOT NULL, min_step INTEGER NOT NULL, vmin REAL NOT NULL,
        max_ordinal INTEGER NOT NULL, max_step INTEGER NOT NULL, vmax REAL NOT NULL,
        vmean REAL NOT NULL, vlast REAL NOT NULL,
        PRIMARY KEY(tag_id, level, bucket)) WITHOUT ROWID""",
    "tag_stats": """CREATE TABLE tag_stats(
        tag_id INTEGER PRIMARY KEY, count INTEGER NOT NULL, mean REAL NOT NULL,
        m2 REAL NOT NULL, min_value REAL NOT NULL, max_value REAL NOT NULL,
        min_step INTEGER NOT NULL, max_step INTEGER NOT NULL,
        last_value REAL NOT NULL) WITHOUT ROWID""",
    "json_lines": """CREATE TABLE json_lines(
        ordinal INTEGER PRIMARY KEY, type TEXT NOT NULL, tag TEXT, step INTEGER,
        timestamp TEXT, json TEXT NOT NULL)""",
    "source_meta": "CREATE TABLE source_meta(k TEXT PRIMARY KEY, v TEXT NOT NULL) WITHOUT ROWID",
}


class CacheTestBase(InspectRunTestBase):
    def sha256_range(self, path: Path, start: int, length: int) -> str:
        with path.open("rb") as handle:
            handle.seek(start)
            return hashlib.sha256(handle.read(length)).hexdigest()

    def make_cache(self, run_dir: Path, master_path: Path, tag_points: dict, **overrides):
        """Metrics Viewer schema v1 の cache を fixture として作る。"""
        size = master_path.stat().st_size
        mtime_ms = master_path.stat().st_mtime_ns // 1_000_000
        committed_offset = overrides.get("committed_offset", size)
        head_length = min(size, overrides.get("source_size", size), 65536)
        tail_end = max(0, min(committed_offset, size))
        tail_start = max(0, tail_end - 65536)

        meta = {
            "generation": "0f7c1b2e-0000-4000-8000-000000000001",
            "source_kind": overrides.get(
                "source_kind", "jsonl.gz" if master_path.name.endswith(".gz") else "jsonl"
            ),
            "source_size": str(overrides.get("source_size", size)),
            "source_mtime": str(overrides.get("source_mtime", mtime_ms)),
            "source_head_sha256": overrides.get(
                "source_head_sha256", self.sha256_range(master_path, 0, head_length)
            ),
            "source_commit_tail_sha256": overrides.get(
                "source_commit_tail_sha256",
                self.sha256_range(master_path, tail_start, tail_end - tail_start),
            ),
            "committed_offset": str(committed_offset),
            "state": overrides.get("state", "ready"),
        }
        meta.update(overrides.get("extra_meta", {}))
        for dropped in overrides.get("drop_meta", ()):
            meta.pop(dropped, None)

        db_path = run_dir / "metrics_cache.db"
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                "PRAGMA application_id = %d" % overrides.get("application_id", 0x414E4554)
            )
            for name, ddl in CACHE_TABLE_DDL.items():
                if name in overrides.get("drop_tables", ()):
                    continue
                connection.execute(ddl)
            for tag_id, (tag, points) in enumerate(tag_points.items(), start=1):
                status = overrides.get("tag_status", {}).get(tag, "ok")
                error_code = "tag_step_regression" if status == "error" else None
                connection.execute(
                    "INSERT INTO tags(id, key, type, status, error_code) VALUES(?,?,?,?,?)",
                    (tag_id, tag, "scalar", status, error_code),
                )
                for ordinal, (step, value) in enumerate(points):
                    connection.execute(
                        "INSERT INTO scalars(tag_id, ordinal, step, value) VALUES(?,?,?,?)",
                        (tag_id, ordinal, step, value),
                    )
            if "source_meta" not in overrides.get("drop_tables", ()):
                connection.executemany(
                    "INSERT INTO source_meta(k, v) VALUES(?,?)", sorted(meta.items())
                )
            connection.execute("PRAGMA user_version = %d" % overrides.get("user_version", 1))
            connection.commit()
        finally:
            connection.close()
        return db_path

    def make_cached_run(self, repo: Path, run_name: str = "run_a", **overrides):
        run_dir = self.make_run(repo, "ws1", run_name)
        self.write_config(run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
        points = overrides.pop("points", [(0, 1.0), (10, 2.0), (20, 3.0), (30, 4.0)])
        master = self.write_raw_master(run_dir, [("t/a", step, value) for step, value in points])
        cache = self.make_cache(run_dir, master, {"t/a": points}, **overrides)
        return run_dir, master, cache


class CacheEligibilityTest(CacheTestBase):
    def test_current_cache_is_used_without_opening_master(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            _, _, cache = self.make_cached_run(repo)

            with mock.patch.object(subject, "open_metrics_binary") as opener:
                code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            opener.assert_not_called()
            source = result["runs"][0]["metrics_source"]
            self.assertEqual(source["selected"], "cache")
            self.assertEqual(source["cache_status"], "current")
            self.assertEqual(source["cache_path"], str(cache.resolve()))
            window = result["runs"][0]["metrics"][0]["windows"][0]
            self.assertEqual(window["count"], 4)
            self.assertEqual(window["mean"], 2.5)

    def test_cache_and_master_paths_agree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            points = [(step, float(step % 7)) for step in range(0, 900, 3)]
            _, _, cache = self.make_cached_run(repo, points=points)

            _, cached, _ = self.run_cli_json(
                repo, ["run_a", "--metric", "t/a", "--window", "0:500", "--window", "50%:100%"]
            )
            cache.unlink()
            _, from_master, _ = self.run_cli_json(
                repo, ["run_a", "--metric", "t/a", "--window", "0:500", "--window", "50%:100%"]
            )

            self.assertEqual(cached["runs"][0]["metrics_source"]["selected"], "cache")
            self.assertEqual(from_master["runs"][0]["metrics_source"]["selected"], "master")
            self.assertEqual(cached["runs"][0]["metrics"], from_master["runs"][0]["metrics"])

    def test_cache_is_not_modified_by_inspection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir, master, cache = self.make_cached_run(repo)
            before = {
                path: (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
                for path in (master, cache)
            }

            code, _, _ = self.run_cli(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            for path, snapshot in before.items():
                self.assertEqual(
                    (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes()),
                    snapshot,
                    msg=str(path),
                )
            self.assertFalse((run_dir / "metrics_cache.db-wal").exists())
            self.assertFalse((run_dir / "metrics_cache.db-shm").exists())

    def test_ineligible_caches_fall_back_to_master(self):
        cases = {
            "partial": ({"committed_offset": 10}, "partial"),
            "stale_size": ({"source_size": 999999}, "stale"),
            "stale_head": ({"source_head_sha256": "0" * 64}, "stale"),
            "stale_tail": ({"source_commit_tail_sha256": "0" * 64}, "stale"),
            "stale_kind": ({"source_kind": "jsonl.gz"}, "stale"),
            "error_state": ({"state": "error"}, "error"),
            "converting_state": ({"state": "converting"}, "partial"),
            "bad_application_id": ({"application_id": 0x11111111}, "invalid"),
            "bad_user_version": ({"user_version": 7}, "invalid"),
            "missing_table": ({"drop_tables": ("scalars_lod",)}, "invalid"),
            "missing_meta_key": ({"drop_meta": ("committed_offset",)}, "invalid"),
            "unparsable_meta": ({"extra_meta": {"source_size": "abc"}}, "invalid"),
        }
        for label, (overrides, expected_status) in cases.items():
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as temp_dir:
                    repo = Path(temp_dir)
                    self.make_cached_run(repo, **overrides)

                    code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])

                    self.assertEqual(code, 0)
                    source = result["runs"][0]["metrics_source"]
                    self.assertEqual(source["selected"], "master")
                    self.assertEqual(source["cache_status"], expected_status)
                    self.assertTrue(source["cache_reason"])
                    self.assertEqual(result["runs"][0]["metrics"][0]["windows"][0]["count"], 4)

    def test_absent_cache_is_reported_and_master_is_used(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
            self.write_raw_master(run_dir, [("t/a", 0, 1.0)])

            code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            source = result["runs"][0]["metrics_source"]
            self.assertEqual(source["selected"], "master")
            self.assertEqual(source["cache_status"], "absent")
            self.assertIsNone(source["cache_path"])

    def test_light_inspection_reports_cache_status_and_source_meta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_cached_run(repo)

            code, result, _ = self.run_cli_json(repo, ["run_a"])

            self.assertEqual(code, 0)
            cache_info = result["runs"][0]["artifacts"]["cache"]
            self.assertTrue(cache_info["exists"])
            self.assertEqual(cache_info["status"], "current")
            self.assertEqual(cache_info["source_meta"]["state"], "ready")
            self.assertEqual(cache_info["source_meta"]["source_kind"], "jsonl")

    def test_cache_tag_error_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_cached_run(repo, tag_status={"t/a": "error"})

            code, result, _ = self.run_cli_json(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            metric = result["runs"][0]["metrics"][0]
            self.assertEqual(result["runs"][0]["metrics_source"]["selected"], "cache")
            self.assertEqual(metric["status"], "quarantined")
            self.assertEqual(metric["windows"][0]["count"], 4)

    def test_missing_tag_in_cache_is_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_cached_run(repo)

            code, result, _ = self.run_cli(repo, ["run_a", "--metric", "t/a", "--metric", "t/zz"])
            parsed = json.loads(result)

            self.assertEqual(code, 0)
            self.assertEqual(parsed["runs"][0]["metrics"][1]["status"], "missing")


class OutputTest(InspectRunTestBase):
    def make_output_run(self, repo: Path, run_name: str = "run_a") -> Path:
        run_dir = self.make_run(repo, "ws1", run_name)
        self.write_config(
            run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step", "agent.lr": "1e-3"}
        )
        self.write_raw_master(run_dir, [("t/a", 0, 1.0), ("t/a", 10, 2.5), ("t/a", 20, 3.0)])
        return run_dir

    def test_json_is_strict_and_free_of_non_finite_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
            (run_dir / "metrics.jsonl").write_text(
                '{"step":0,"tag":"t/a","type":"scalar","value":NaN}\n'
                '{"step":10,"tag":"t/a","type":"scalar","value":1.0}\n',
                encoding="utf-8",
            )

            code, out, _ = self.run_cli(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            self.assertNotIn("NaN", out)
            self.assertNotIn("Infinity", out)
            json.loads(out, parse_constant=self.fail)

    def test_markdown_renders_the_same_result_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_output_run(repo)

            code, text, _ = self.run_cli(
                repo,
                ["run_a", "--metric", "t/a", "--config-key", "agent.lr", "--format", "md"],
            )

            self.assertEqual(code, 0)
            self.assertIn("run_a", text)
            self.assertIn("t/a", text)
            self.assertIn("exp_step", text)
            self.assertIn("agent.lr", text)
            self.assertIn("1e-3", text)
            self.assertIn("0:1", text)
            self.assertIn("20:3", text)
            for heading in ("## ", "|"):
                self.assertIn(heading, text)

    def test_output_file_is_written_and_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_output_run(repo)
            target = repo / "out.json"
            target.write_text("stale content", encoding="utf-8")

            code, out, _ = self.run_cli(
                repo, ["run_a", "--metric", "t/a", "--output", str(target)]
            )

            self.assertEqual(code, 0)
            self.assertEqual(out, "")
            written = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(written["runs"][0]["run_name"], "run_a")
            self.assertEqual(sorted(item.name for item in repo.glob("out.json*")), ["out.json"])

    def test_output_write_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_output_run(repo)
            target = repo / "out.json"
            target.write_text("keep me", encoding="utf-8")

            with mock.patch.object(subject.os, "replace", side_effect=OSError("boom")):
                code, _, err = self.run_cli(
                    repo, ["run_a", "--metric", "t/a", "--output", str(target)]
                )

            self.assertEqual(code, 1)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep me")
            self.assertIn("boom", err)
            self.assertEqual(sorted(item.name for item in repo.glob("out.json*")), ["out.json"])

    def test_output_parent_directory_must_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_output_run(repo)

            code, _, err = self.run_cli(
                repo,
                ["run_a", "--metric", "t/a", "--output", str(repo / "missing" / "out.json")],
            )

            self.assertEqual(code, 2)
            self.assertIn("missing", err)

    def test_warnings_go_to_stderr_and_keep_stdout_parsable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_output_run(repo)
            master = run_dir / "metrics.jsonl"
            master.write_text(
                master.read_text(encoding="utf-8") + '{"step":30,"tag":"t/a"',
                encoding="utf-8",
            )

            code, out, err = self.run_cli(repo, ["run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            self.assertTrue(err.strip())
            parsed = json.loads(out)
            self.assertTrue(parsed["warnings"])

    def test_partial_tag_absence_across_runs_exits_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_output_run(repo, "run_a")
            other = self.make_run(repo, "ws1", "run_b")
            self.write_config(other, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
            self.write_raw_master(other, [("t/other", 0, 1.0)])

            code, result, _ = self.run_cli_json(repo, ["run_a", "run_b", "--metric", "t/a"])

            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["metrics"][0]["status"], "ok")
            self.assertEqual(result["runs"][1]["metrics"][0]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
