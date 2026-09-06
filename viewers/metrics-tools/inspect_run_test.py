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


class InspectRunTestBase(unittest.TestCase):
    # ------------------------------------------------------------------
    # fixture 生成
    # ------------------------------------------------------------------

    def workspaces_root(self, repo: Path) -> Path:
        return repo / "apps" / "runner" / "workspaces"

    def make_run(self, repo: Path, workspace: str, run_name: str) -> Path:
        run_dir = self.workspaces_root(repo) / workspace / "runs" / run_name
        run_dir.mkdir(parents=True)
        return run_dir

    def write_config(self, run_dir: Path, entries: dict, name: str = "config_data.txt") -> Path:
        config_dir = run_dir / "config"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / name
        text = "".join(f"{key} = {value}\n" for key, value in entries.items())
        config_path.write_text(text, encoding="utf-8")
        return config_path

    def write_resolution(self, run_dir: Path, payload: dict, legacy: bool = False) -> Path:
        relative_path = Path("config/config_resolution.json") if legacy else Path(
            "json/config_resolution.json"
        )
        path = run_dir / relative_path
        path.parent.mkdir(exist_ok=True)
        document = payload if legacy else {
            "type": "json",
            "tag": "config_resolution",
            "data": payload,
        }
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def scalar_lines(self, points: list) -> str:
        # points は (tag, step, value)。MetricsLogger と同じ4 field だけを書く。
        return "".join(
            json.dumps({"step": step, "tag": tag, "type": "scalar", "value": value}) + "\n"
            for tag, step, value in points
        )

    def defs_line(self, defs: dict, tag: str = "metrics.scalar.defs") -> str:
        record = {
            "type": "json",
            "tag": tag,
            "timestamp": "2026-08-15T00:00:00",
            "data": defs,
        }
        return json.dumps(record) + "\n"

    def write_raw_master(self, run_dir: Path, points: list, defs: dict | None = None) -> Path:
        master_path = run_dir / "metrics.jsonl"
        text = self.defs_line(defs) if defs else ""
        master_path.write_text(text + self.scalar_lines(points), encoding="utf-8")
        return master_path

    def write_gzip_master(self, run_dir: Path, points: list, terminated: bool = True) -> Path:
        gzip_path = run_dir / "metrics.jsonl.gz"
        text = self.scalar_lines(points)
        if not terminated:
            text = text.rstrip("\n")
        with gzip.open(gzip_path, "wb") as handle:
            handle.write(text.encode("utf-8"))
        return gzip_path

    def metric_def(self, step_axis="exp_step", runner="train", event="train",
                   target="env", source_key="src", ema_alpha=None, interval=None) -> dict:
        return {
            "step_axis": step_axis,
            "runner": runner,
            "event": event,
            "target": target,
            "source_key": source_key,
            "ema_alpha": ema_alpha,
            "interval": interval,
        }

    # ------------------------------------------------------------------
    # CLI 起動
    # ------------------------------------------------------------------

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


class RunsSubcommandTest(InspectRunTestBase):
    def test_lists_every_run_across_workspaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_run(repo, "ws1", "run_a")
            self.make_run(repo, "ws1", "run_b")
            self.make_run(repo, "ws2", "run_c")

            code, result, _ = self.run_cli_json(repo, ["runs"])

            self.assertEqual(code, 0)
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["subcommand"], "runs")
            listed = [(run["workspace"], run["run_name"]) for run in result["runs"]]
            self.assertEqual(listed, [("ws1", "run_a"), ("ws1", "run_b"), ("ws2", "run_c")])
            self.assertIsNone(result["runs"][0]["input"])

    def test_workspace_option_limits_the_listing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_run(repo, "ws1", "run_a")
            self.make_run(repo, "ws2", "run_c")

            code, result, _ = self.run_cli_json(repo, ["runs", "--workspace", "ws2"])

            self.assertEqual(code, 0)
            self.assertEqual([run["run_name"] for run in result["runs"]], ["run_c"])

            code, _, err = self.run_cli(repo, ["runs", "--workspace", "nope"])
            self.assertEqual(code, 2)
            self.assertIn("nope", err)


class CacheFixtureMixin(InspectRunTestBase):
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
                if points:
                    steps = [step for step, _ in points]
                    values = [value for _, value in points]
                    connection.execute(
                        "INSERT INTO tag_stats(tag_id, count, mean, m2, min_value, max_value,"
                        " min_step, max_step, last_value) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            tag_id, len(points), sum(values) / len(values), 0.0,
                            min(values), max(values), min(steps), max(steps), values[-1],
                        ),
                    )
            defs = overrides.get("defs")
            if defs is not None and "json_lines" not in overrides.get("drop_tables", ()):
                connection.execute(
                    "INSERT INTO json_lines(ordinal, type, tag, json) VALUES(?,?,?,?)",
                    (1, "json", "metrics.scalar.defs", self.defs_line(defs).strip()),
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
        defs = overrides.pop("run_defs", {"t/a": self.metric_def(source_key="a_key")})
        self.write_config(run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
        points = overrides.pop("points", [(0, 1.0), (10, 2.0), (20, 3.0), (30, 4.0)])
        master = self.write_raw_master(
            run_dir, [("t/a", step, value) for step, value in points], defs=defs
        )
        overrides.setdefault("defs", defs)
        cache = self.make_cache(run_dir, master, {"t/a": points}, **overrides)
        return run_dir, master, cache


class RunResolutionTest(InspectRunTestBase):
    def test_resolves_absolute_and_relative_directory_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")

            code, result, _ = self.run_cli_json(repo, ["runs", str(run_dir)])
            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["run_name"], "run_a")
            self.assertEqual(result["runs"][0]["workspace"], "ws1")

            with contextlib.chdir(run_dir.parent):
                code, result, _ = self.run_cli_json(repo, ["runs", "run_a"])
            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["input"], "run_a")
            self.assertEqual(result["runs"][0]["run_dir"], str(run_dir.resolve()))

    def test_legacy_directory_resolves_only_by_explicit_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            legacy_dir = repo / "apps" / "runner" / "runs_lunarlander" / "run_legacy"
            legacy_dir.mkdir(parents=True)
            self.workspaces_root(repo).mkdir(parents=True)

            code, result, _ = self.run_cli_json(repo, ["runs", str(legacy_dir)])
            self.assertEqual(code, 0)
            self.assertIsNone(result["runs"][0]["workspace"])

            code, _, err = self.run_cli(repo, ["runs", "run_legacy"])
            self.assertEqual(code, 2)
            self.assertIn("run_legacy", err)

    def test_duplicate_run_name_across_workspaces_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            first = self.make_run(repo, "ws1", "run_a")
            second = self.make_run(repo, "ws2", "run_a")

            code, _, err = self.run_cli(repo, ["runs", "run_a"])

            self.assertEqual(code, 2)
            self.assertIn(str(first.resolve()), err)
            self.assertIn(str(second.resolve()), err)


class ArtifactInspectionTest(CacheFixtureMixin):
    def test_reports_artifacts_without_opening_the_master(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            config_path = self.write_config(run_dir, {"agent.class_id": "DefaultDQNAgent"})
            master_path = self.write_raw_master(run_dir, [("t/a", 0, 1.0)])
            (run_dir / "run_a.log").write_text("log", encoding="utf-8")
            (run_dir / "agent_close.anet").write_bytes(b"\x00\x01")
            nested = run_dir / "json"
            nested.mkdir()
            (nested / "buried.log").write_text("not listed", encoding="utf-8")

            expected_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

            with mock.patch.object(subject, "open_metrics_binary") as opener:
                code, result, _ = self.run_cli_json(repo, ["runs", "run_a"])

            self.assertEqual(code, 0)
            opener.assert_not_called()
            artifacts = result["runs"][0]["artifacts"]
            self.assertTrue(artifacts["config"]["exists"])
            self.assertEqual(artifacts["config"]["sha256"], expected_sha)
            self.assertTrue(artifacts["master"]["raw_exists"])
            self.assertFalse(artifacts["master"]["gzip_exists"])
            self.assertEqual(artifacts["master"]["path"], str(master_path.resolve()))
            self.assertEqual(artifacts["master"]["kind"], "jsonl")
            listed = sorted(item["name"] for item in artifacts["files"])
            self.assertEqual(listed, ["agent_close.anet", "run_a.log"])

    def test_missing_config_and_master_are_reported_as_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_run(repo, "ws1", "run_a")

            code, result, _ = self.run_cli_json(repo, ["runs", "run_a"])

            self.assertEqual(code, 0)
            artifacts = result["runs"][0]["artifacts"]
            self.assertFalse(artifacts["config"]["exists"])
            self.assertIsNone(artifacts["master"]["path"])
            self.assertEqual(artifacts["cache"]["status"], "absent")
            self.assertEqual(artifacts["files"], [])

    def test_current_cache_is_reported_with_source_meta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_cached_run(repo)

            code, result, _ = self.run_cli_json(repo, ["runs", "run_a"])

            self.assertEqual(code, 0)
            cache_info = result["runs"][0]["artifacts"]["cache"]
            self.assertEqual(cache_info["status"], "current")
            self.assertIsNone(cache_info["reason"])
            self.assertEqual(cache_info["source_meta"]["state"], "ready")

    def test_cache_states_are_classified_with_reasons(self):
        cases = {
            "partial": ({"committed_offset": 10}, "partial"),
            "stale_size": ({"source_size": 999999}, "stale"),
            "stale_head": ({"source_head_sha256": "0" * 64}, "stale"),
            "stale_kind": ({"source_kind": "jsonl.gz"}, "stale"),
            "error_state": ({"state": "error"}, "error"),
            "converting_state": ({"state": "converting"}, "partial"),
            "bad_application_id": ({"application_id": 0x11111111}, "invalid"),
            "bad_user_version": ({"user_version": 7}, "invalid"),
            "missing_table": ({"drop_tables": ("scalars_lod",)}, "invalid"),
            "missing_meta_key": ({"drop_meta": ("committed_offset",)}, "invalid"),
            "unparsable_meta": ({"extra_meta": {"source_size": "abc"}}, "invalid"),
        }
        for label, (overrides, expected) in cases.items():
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as temp_dir:
                    repo = Path(temp_dir)
                    self.make_cached_run(repo, **overrides)

                    code, result, _ = self.run_cli_json(repo, ["runs", "run_a"])

                    self.assertEqual(code, 0)
                    cache_info = result["runs"][0]["artifacts"]["cache"]
                    self.assertEqual(cache_info["status"], expected)
                    self.assertTrue(cache_info["reason"])

    def test_runs_markdown_lists_files_and_cache_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir, _, _ = self.make_cached_run(repo)
            (run_dir / "agent_close.anet").write_bytes(b"\x00")

            code, text, _ = self.run_cli(repo, ["runs", "run_a", "--format", "md"])

            self.assertEqual(code, 0)
            self.assertIn("run_a", text)
            self.assertIn("current", text)
            self.assertIn("agent_close.anet", text)


EVAL_DEFS = {
    "51_eval1/13_double_suika_created_mean": {
        "step_axis": "exp_step", "runner": "train", "event": "session_end",
        "target": "env", "source_key": "mean.ep_double_suika_created",
        "ema_alpha": None, "interval": None,
    },
    "51_eval1/41_noop_uqe_win_rate": {
        "step_axis": "exp_step", "runner": "eval1", "event": "train",
        "target": "action_info", "source_key": "action_uqe_win_rate.[0]",
        "ema_alpha": None, "interval": None,
    },
}

EVAL_CONFIG = {
    "metrics.scalar.[51_eval1/13_double_suika_created_mean]":
        "$eval.[eval1] @session_end $env $exp_step mean.ep_double_suika_created",
    "metrics.scalar.[51_eval1/41_noop_uqe_win_rate]":
        "$eval.[eval1] @train $exp_step action_uqe_win_rate.[0] $action_info",
}


class TagsSubcommandTest(CacheFixtureMixin):
    def make_eval_run(self, repo: Path, run_name: str = "run_a", with_defs: bool = True,
                      train_steps=(1000, 2000, 3000), eval_steps=(10, 20, 30)):
        run_dir = self.make_run(repo, "ws1", run_name)
        self.write_config(run_dir, EVAL_CONFIG)
        points = [
            ("51_eval1/13_double_suika_created_mean", step, float(step))
            for step in train_steps
        ] + [
            ("51_eval1/41_noop_uqe_win_rate", step, float(step)) for step in eval_steps
        ]
        self.write_raw_master(run_dir, points, defs=EVAL_DEFS if with_defs else None)
        return run_dir

    def test_reports_declared_definition_and_observed_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_eval_run(repo)

            code, result, _ = self.run_cli_json(repo, ["tags", "run_a"])

            self.assertEqual(code, 0)
            self.assertEqual(result["subcommand"], "tags")
            run = result["runs"][0]
            self.assertEqual(run["def_source"], "metrics_defs")
            by_tag = {item["tag"]: item for item in run["tags"]}
            train_side = by_tag["51_eval1/13_double_suika_created_mean"]
            eval_side = by_tag["51_eval1/41_noop_uqe_win_rate"]

            # 同じ軸名でも Runner が違えば別座標系。
            self.assertEqual(train_side["step_axis"], "exp_step")
            self.assertEqual(eval_side["step_axis"], "exp_step")
            self.assertEqual(train_side["runner"], "train")
            self.assertEqual(eval_side["runner"], "eval1")
            self.assertEqual(eval_side["source_key"], "action_uqe_win_rate.[0]")
            self.assertEqual(train_side["observed"]["max_step"], 3000)
            self.assertEqual(eval_side["observed"]["max_step"], 30)
            self.assertEqual(eval_side["observed"]["count"], 3)

    def test_no_observed_skips_the_master_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_eval_run(repo)

            with mock.patch.object(subject, "open_metrics_binary") as opener:
                code, result, _ = self.run_cli_json(repo, ["tags", "run_a", "--no-observed"])

            self.assertEqual(code, 0)
            opener.assert_not_called()
            run = result["runs"][0]
            self.assertEqual(run["def_source"], "config_derived")
            self.assertTrue(all(item["observed"] is None for item in run["tags"]))
            by_tag = {item["tag"]: item for item in run["tags"]}
            self.assertEqual(by_tag["51_eval1/41_noop_uqe_win_rate"]["runner"], "eval1")

    def test_config_fallback_marks_def_source_and_warns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_eval_run(repo, with_defs=False)

            code, result, err = self.run_cli_json(repo, ["tags", "run_a"])

            self.assertEqual(code, 0)
            run = result["runs"][0]
            self.assertEqual(run["def_source"], "config_derived")
            self.assertIn("config", err)
            by_tag = {item["tag"]: item for item in run["tags"]}
            # fallback でも Runner の導出規則は同じ結果になる。
            self.assertEqual(by_tag["51_eval1/41_noop_uqe_win_rate"]["runner"], "eval1")
            self.assertEqual(by_tag["51_eval1/13_double_suika_created_mean"]["runner"], "train")
            self.assertEqual(
                by_tag["51_eval1/13_double_suika_created_mean"]["source_key"],
                "mean.ep_double_suika_created",
            )

    def test_current_cache_supplies_tags_without_opening_master(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_cached_run(repo)

            with mock.patch.object(subject, "open_metrics_binary") as opener:
                code, result, _ = self.run_cli_json(repo, ["tags", "run_a"])

            self.assertEqual(code, 0)
            opener.assert_not_called()
            run = result["runs"][0]
            self.assertEqual(run["metrics_source"]["selected"], "cache")
            entry = run["tags"][0]
            self.assertEqual(entry["tag"], "t/a")
            self.assertEqual(entry["observed"]["count"], 4)
            self.assertEqual(entry["observed"]["max_step"], 30)

    def test_reads_the_payload_shape_the_runner_emits(self):
        # C++ 側 ScalarMetricDefsToJson の出力そのままの形。
        # target と ema_alpha は未設定時 null、interval は常に整数。
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {})
            defs = {
                "51_eval1/41_noop_uqe_win_rate": {
                    "step_axis": "exp_step", "runner": "eval1", "event": "train",
                    "target": "action_info", "source_key": "action_uqe_win_rate.[0]",
                    "ema_alpha": 0.01, "interval": 100,
                },
                "37_agent_qtd/01_td_mean": {
                    "step_axis": "exp_step", "runner": "train", "event": "learn",
                    "target": None, "source_key": "td_mean",
                    "ema_alpha": None, "interval": 1,
                },
            }
            self.write_raw_master(
                run_dir, [("37_agent_qtd/01_td_mean", 10, 1.0)], defs=defs
            )

            code, result, _ = self.run_cli_json(repo, ["tags", "run_a"])

            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["def_source"], "metrics_defs")
            by_tag = {item["tag"]: item for item in result["runs"][0]["tags"]}
            eval_side = by_tag["51_eval1/41_noop_uqe_win_rate"]
            self.assertEqual(eval_side["runner"], "eval1")
            self.assertEqual(eval_side["ema_alpha"], 0.01)
            self.assertEqual(eval_side["interval"], 100)
            learn_side = by_tag["37_agent_qtd/01_td_mean"]
            self.assertIsNone(learn_side["target"])
            self.assertIsNone(learn_side["ema_alpha"])
            self.assertEqual(learn_side["interval"], 1)

    def test_tags_markdown_contains_runner_and_source_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_eval_run(repo)

            code, text, _ = self.run_cli(repo, ["tags", "run_a", "--format", "md"])

            self.assertEqual(code, 0)
            self.assertIn("eval1", text)
            self.assertIn("action_uqe_win_rate.[0]", text)
            self.assertIn("exp_step", text)


class ConfigSubcommandTest(InspectRunTestBase):
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
                ["config", "run_a",
                 "--config-key", "agent.class_id",
                 "--config-key", "agent.*",
                 "--config-key", "train.eval.[eval1].run_mode",
                 "--config-key", "train.eval.[*].run_mode",
                 "--config-key", "nope.*"],
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
            self.assertEqual(statuses["train.eval.[eval1].run_mode"], "ok")
            self.assertEqual(statuses["train.eval.[*].run_mode"], "ok")
            self.assertEqual(statuses["nope.*"], "missing")

    def test_effective_marking_uses_module_dumps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(
                run_dir,
                {
                    "app.log_level": "info",
                    "app.online.fps": "15",
                    "net.block.[FC1].out": "128",
                },
            )
            # module dump は完全修飾キーで、実際に読まれた設定だけを含む。
            self.write_config(run_dir, {"app.log_level": "info"}, name="app.txt")

            code, result, _ = self.run_cli_json(
                repo, ["config", "run_a", "--config-key", "app.*", "--config-key", "net.*"]
            )

            self.assertEqual(code, 0)
            marks = {item["key"]: item["effective"] for item in result["runs"][0]["config"]["values"]}
            self.assertTrue(marks["app.log_level"])
            # 判定できないものを false と言わない。
            self.assertIsNone(marks["app.online.fps"])
            self.assertIsNone(marks["net.block.[FC1].out"])

            code, result, _ = self.run_cli_json(
                repo, ["config", "run_a", "--config-key", "app.*", "--effective-only"]
            )
            self.assertEqual(code, 0)
            keys = [item["key"] for item in result["runs"][0]["config"]["values"]]
            self.assertEqual(keys, ["app.log_level"])

    def test_diff_reports_differing_keys_and_absence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            first = self.make_run(repo, "ws1", "run_a")
            second = self.make_run(repo, "ws1", "run_b")
            self.write_config(first, {"agent.lr": "1e-3", "agent.gamma": "0.99", "only_a": "x"})
            self.write_config(second, {"agent.lr": "5e-4", "agent.gamma": "0.99"})

            code, result, _ = self.run_cli_json(repo, ["config", "run_a", "run_b", "--diff"])

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
                repo, ["config", "run_a", "run_b", "--diff", "--config-key", "agent.*"]
            )
            self.assertEqual(code, 0)
            self.assertEqual([item["key"] for item in result["config_diff"]], ["agent.lr"])

            code, result, _ = self.run_cli_json(repo, ["config", "run_a", "--diff"])
            self.assertEqual(code, 0)
            self.assertEqual(result["config_diff"], [])

    def test_selector_missing_in_all_runs_exits_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {"agent.lr": "1e-3"})

            code, out, _ = self.run_cli(repo, ["config", "run_a", "--config-key", "absent.*"])

            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["runs"][0]["config"]["values"], [])

    def test_config_markdown_contains_values_and_diff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            first = self.make_run(repo, "ws1", "run_a")
            second = self.make_run(repo, "ws1", "run_b")
            self.write_config(first, {"agent.lr": "1e-3"})
            self.write_config(second, {"agent.lr": "5e-4"})

            code, text, _ = self.run_cli(
                repo, ["config", "run_a", "run_b", "--config-key", "agent.*", "--diff",
                       "--format", "md"]
            )

            self.assertEqual(code, 0)
            self.assertIn("agent.lr", text)
            self.assertIn("1e-3", text)
            self.assertIn("5e-4", text)


class ResolutionSubcommandTest(InspectRunTestBase):
    def test_reads_envelope_and_reports_named_trunk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_resolution(run_dir, {
                "schema_version": 1,
                "selections": [
                    {
                        "key": "run.$",
                        "chain": [{"term": "@verify", "resolved": "run.@verify"}],
                    },
                    {
                        "key": "AtariEnv.$",
                        "chain": [{"term": "@v5", "resolved": "AtariEnv.@v5"}],
                    },
                ],
                "references": [
                    {"source": "app.limit", "target": "@vars.limit", "value": "100"}
                ],
            })

            code, result, _ = self.run_cli_json(repo, ["resolution", "run_a"])

            self.assertEqual(code, 0)
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["subcommand"], "resolution")
            resolution = result["runs"][0]["resolution"]
            self.assertEqual(resolution["status"], "ok")
            self.assertEqual(resolution["source"], "json/config_resolution.json")
            self.assertEqual(resolution["schema_version"], 1)
            self.assertEqual(resolution["trunk"]["key"], "run.$")
            self.assertEqual(resolution["selections"][1]["key"], "AtariEnv.$")
            self.assertEqual(resolution["references"][0]["value"], "100")

    def test_falls_back_to_legacy_raw_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_resolution(run_dir, {
                "schema_version": 1,
                "selections": [],
                "references": [],
            }, legacy=True)

            code, result, _ = self.run_cli_json(repo, ["resolution", "run_a"])

            self.assertEqual(code, 0)
            resolution = result["runs"][0]["resolution"]
            self.assertEqual(resolution["status"], "ok")
            self.assertEqual(resolution["source"], "config/config_resolution.json")
            self.assertIsNone(resolution["trunk"])

    def test_reports_missing_resolution_as_normal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_run(repo, "ws1", "run_a")

            code, result, err = self.run_cli_json(repo, ["resolution", "run_a"])

            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            resolution = result["runs"][0]["resolution"]
            self.assertEqual(resolution["status"], "missing")
            self.assertIsNone(resolution["source"])
            self.assertIsNone(resolution["schema_version"])
            self.assertIsNone(resolution["trunk"])
            self.assertEqual(resolution["selections"], [])
            self.assertEqual(resolution["references"], [])

    def test_warns_and_displays_unknown_resolution_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_resolution(run_dir, {
                "schema_version": 2,
                "selections": [
                    {"key": "Env.$", "chain": [{"term": "@a", "resolved": "Env.@a"}]}
                ],
                "references": [],
            })

            code, result, err = self.run_cli_json(repo, ["resolution", "run_a"])

            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["resolution"]["schema_version"], 2)
            self.assertEqual(result["runs"][0]["resolution"]["selections"][0]["key"], "Env.$")
            self.assertIn("unsupported resolution schema_version=2", err)
            self.assertIn("unsupported resolution schema_version=2", result["runs"][0]["warnings"])

    def test_primary_source_error_does_not_fall_back_to_legacy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_resolution(run_dir, {
                "schema_version": 1,
                "selections": [],
                "references": [],
            }, legacy=True)
            primary = run_dir / "json" / "config_resolution.json"
            primary.parent.mkdir(exist_ok=True)
            primary.write_text("{broken", encoding="utf-8")

            code, out, err = self.run_cli(repo, ["resolution", "run_a"])
            result = json.loads(out)

            self.assertEqual(code, 1)
            resolution = result["runs"][0]["resolution"]
            self.assertEqual(resolution["status"], "source_error")
            self.assertEqual(resolution["source"], "json/config_resolution.json")
            self.assertEqual(resolution["selections"], [])
            self.assertIn("Failed to read resolution artifact", err)

    def test_invalid_resolution_payload_is_a_source_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_resolution(run_dir, {
                "schema_version": 1,
                "selections": {},
                "references": [],
            })

            code, result, err = self.run_cli_json(repo, ["resolution", "run_a"])

            self.assertEqual(code, 1)
            self.assertEqual(result["runs"][0]["resolution"]["status"], "source_error")
            self.assertIn("expected selections array", err)

    def test_multiple_runs_report_resolution_independently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            broken = self.make_run(repo, "ws1", "run_a")
            broken_path = broken / "json" / "config_resolution.json"
            broken_path.parent.mkdir(exist_ok=True)
            broken_path.write_text("{broken", encoding="utf-8")
            valid = self.make_run(repo, "ws1", "run_b")
            self.write_resolution(valid, {
                "schema_version": 1,
                "selections": [],
                "references": [],
            })

            code, result, _ = self.run_cli_json(
                repo, ["resolution", "run_a", "run_b"]
            )

            self.assertEqual(code, 1)
            self.assertEqual(
                [run["resolution"]["status"] for run in result["runs"]],
                ["source_error", "ok"],
            )

    def test_markdown_reports_trunk_before_resolution_details(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_resolution(run_dir, {
                "schema_version": 1,
                "selections": [
                    {
                        "key": "run.$",
                        "chain": [{"term": "@verify", "resolved": "run.@verify"}],
                    },
                    {
                        "key": "Env.$",
                        "chain": [{"term": "@a", "resolved": "Env.@a"}],
                    },
                ],
                "references": [
                    {"source": "app.limit", "target": "@vars.limit", "value": "100"}
                ],
            })

            code, text, _ = self.run_cli(
                repo, ["resolution", "run_a", "--format", "md"]
            )

            self.assertEqual(code, 0)
            self.assertIn("# Config resolution", text)
            self.assertIn("- trunk: @verify => run.@verify", text)
            self.assertLess(text.index("- trunk:"), text.index("### Selections"))
            self.assertIn("| Env.$ | @a | Env.@a |", text)
            self.assertIn("| app.limit | @vars.limit | 100 |", text)

    def test_markdown_omits_trunk_summary_when_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_resolution(run_dir, {
                "schema_version": 1,
                "selections": [
                    {"key": "Env.$", "chain": [{"term": "@a", "resolved": "Env.@a"}]}
                ],
                "references": [],
            })

            code, text, _ = self.run_cli(
                repo, ["resolution", "run_a", "--format", "md"]
            )

            self.assertEqual(code, 0)
            self.assertNotIn("- trunk:", text)
            self.assertIn("| Env.$ | @a | Env.@a |", text)

    def test_does_not_open_metrics_sources_or_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_resolution(run_dir, {
                "schema_version": 1,
                "selections": [],
                "references": [],
            })

            with mock.patch.object(subject, "open_run", side_effect=AssertionError("open_run")), \
                    mock.patch.object(
                        subject, "resolve_run_metrics", side_effect=AssertionError("metrics")
                    ), mock.patch.object(
                        subject.sqlite3, "connect", side_effect=AssertionError("cache")
                    ):
                code, result, _ = self.run_cli_json(repo, ["resolution", "run_a"])

            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["resolution"]["status"], "ok")


class MetricsSubcommandTest(CacheFixtureMixin):
    def make_metric_run(self, repo: Path, run_name: str, points: list, defs: dict,
                        config: dict | None = None) -> Path:
        run_dir = self.make_run(repo, "ws1", run_name)
        self.write_config(run_dir, config or {})
        self.write_raw_master(run_dir, points, defs=defs)
        return run_dir

    def make_two_space_run(self, repo: Path, run_name: str = "run_a"):
        """同じ exp_step でも Runner が違う 2 tag を持つ Run。"""
        points = [("train/x", step, float(step)) for step in range(1000, 10001, 1000)]
        points += [("eval1/y", step, float(step)) for step in range(10, 101, 10)]
        defs = {
            "train/x": self.metric_def(step_axis="exp_step", runner="train", source_key="x"),
            "eval1/y": self.metric_def(
                step_axis="exp_step", runner="eval1", event="train",
                target="action_info", source_key="y",
            ),
        }
        return self.make_metric_run(repo, run_name, points, defs)

    def test_extracts_metric_over_the_full_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_metric_run(
                repo, "run_a",
                [("t/a", 0, 1.0), ("t/a", 10, 2.0), ("t/a", 20, 3.0), ("t/a", 30, 4.0)],
                {"t/a": self.metric_def(source_key="a_key")},
            )

            code, result, _ = self.run_cli_json(repo, ["metrics", "run_a", "--metric", "t/a"])

            self.assertEqual(code, 0)
            self.assertEqual(result["subcommand"], "metrics")
            self.assertEqual([item["label"] for item in result["ranges"]], ["all"])
            metric = result["runs"][0]["metrics"][0]
            self.assertEqual(metric["tag"], "t/a")
            self.assertEqual(metric["runner"], "train")
            self.assertEqual(metric["source_key"], "a_key")
            self.assertEqual(metric["status"], "ok")
            self.assertEqual(metric["observed"], {"count": 4, "min_step": 0, "max_step": 30})
            entry = metric["ranges"][0]
            self.assertEqual(entry["status"], "ok")
            self.assertEqual(entry["count"], 4)
            self.assertAlmostEqual(entry["mean"], 2.5)
            self.assertAlmostEqual(entry["population_std"], math.sqrt(1.25))
            self.assertEqual((entry["first"], entry["last"]), (1.0, 4.0))
            self.assertEqual((entry["min_step"], entry["max_step"]), (0, 30))
            self.assertNotIn("series", entry)

    def test_relative_range_resolves_per_step_coordinate_space(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_two_space_run(repo)

            code, result, _ = self.run_cli_json(
                repo,
                ["metrics", "run_a", "--metric", "train/x", "--metric", "eval1/y",
                 "--range", "-20%:"],
            )

            self.assertEqual(code, 0)
            by_tag = {item["tag"]: item for item in result["runs"][0]["metrics"]}
            train_range = by_tag["train/x"]["ranges"][0]
            eval_range = by_tag["eval1/y"]["ranges"][0]
            # 軸名が同じでも Runner が違えば 100% の基準が違う。
            self.assertEqual((train_range["start"], train_range["end"]), (8000, 10000))
            self.assertEqual((eval_range["start"], eval_range["end"]), (80, 100))
            self.assertEqual(train_range["count"], 3)
            self.assertEqual(eval_range["count"], 3)

    def test_negative_and_open_endpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_two_space_run(repo)

            code, result, _ = self.run_cli_json(
                repo,
                ["metrics", "run_a", "--metric", "train/x",
                 "--range", "-3000:", "--range", ":2000", "--range", "5000:",
                 "--range", ":-8000", "--range", "1k:2K"],
            )

            self.assertEqual(code, 0)
            entries = result["runs"][0]["metrics"][0]["ranges"]
            bounds = {item["label"]: (item["start"], item["end"]) for item in entries}
            self.assertEqual(bounds["-3000:"], (7000, 10000))
            self.assertEqual(bounds[":2000"], (0, 2000))
            self.assertEqual(bounds["5000:"], (5000, 10000))
            self.assertEqual(bounds[":-8000"], (0, 2000))
            self.assertEqual(bounds["1k:2K"], (1000, 2000))
            counts = {item["label"]: item["count"] for item in entries}
            self.assertEqual(counts["-3000:"], 4)
            self.assertEqual(counts["1k:2K"], 2)

    def test_range_mode_common_intersects_runs_per_space(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            defs = {"t/a": self.metric_def(source_key="a")}
            self.make_metric_run(
                repo, "run_a",
                [("t/a", step, float(step)) for step in range(0, 2001, 500)], defs)
            self.make_metric_run(
                repo, "run_b",
                [("t/a", step, float(step)) for step in range(500, 1501, 500)], defs)

            code, result, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "run_b", "--metric", "t/a", "--range-mode", "common"]
            )

            self.assertEqual(code, 0)
            first = result["runs"][0]["metrics"][0]["ranges"][0]
            second = result["runs"][1]["metrics"][0]["ranges"][0]
            self.assertEqual(first["label"], "common")
            self.assertEqual((first["start"], first["end"]), (500, 1500))
            self.assertEqual((second["start"], second["end"]), (500, 1500))
            self.assertEqual(first["count"], 3)
            self.assertEqual(second["count"], 3)

    def test_invalid_range_syntax_exits_two(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_two_space_run(repo)

            for bad in ["20M:10M", "0:1.5", "0%:120%", "0:50%", ":", "abc", "10"]:
                code, _, err = self.run_cli(
                    repo, ["metrics", "run_a", "--metric", "train/x", "--range", bad]
                )
                self.assertEqual(code, 2, msg=f"range={bad}")
                self.assertIn(bad, err, msg=f"range={bad}")

            code, _, err = self.run_cli(
                repo, ["metrics", "run_a", "--metric", "train/x", "--range-mode", "nope"]
            )
            self.assertEqual(code, 2)
            self.assertIn("nope", err)

    def test_relative_range_on_unknown_space_exits_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {"agent.class_id": "X"})
            self.write_raw_master(run_dir, [("t/a", 0, 1.0), ("t/a", 10, 2.0)])

            code, _, err = self.run_cli(
                repo, ["metrics", "run_a", "--metric", "t/a", "--range", "0%:100%"]
            )
            self.assertEqual(code, 1)
            self.assertIn("t/a", err)

            # 絶対指定だけなら座標系が unknown でも解決できる。
            code, result, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "--metric", "t/a", "--range", "0:5"]
            )
            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["metrics"][0]["ranges"][0]["count"], 1)

    def test_metric_glob_expands_against_known_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_two_space_run(repo)

            code, result, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "--metric", "eval1/*"]
            )

            self.assertEqual(code, 0)
            self.assertEqual(
                [item["tag"] for item in result["runs"][0]["metrics"]], ["eval1/y"]
            )

    def test_empty_and_missing_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_two_space_run(repo)

            code, out, err = self.run_cli(
                repo,
                ["metrics", "run_a", "--metric", "train/x", "--metric", "train/zz",
                 "--range", "500000:600000"],
            )
            result = json.loads(out)

            self.assertEqual(code, 0)
            by_tag = {item["tag"]: item for item in result["runs"][0]["metrics"]}
            self.assertEqual(by_tag["train/zz"]["status"], "missing")
            self.assertEqual(by_tag["train/x"]["status"], "ok")
            self.assertEqual(by_tag["train/x"]["ranges"][0]["status"], "empty")
            self.assertIsNone(by_tag["train/x"]["ranges"][0]["mean"])


class ComparisonAndOutputTest(CacheFixtureMixin):
    DEFS = {"t/a": {"step_axis": "exp_step", "runner": "train", "event": "train",
                    "target": "env", "source_key": "a_key",
                    "ema_alpha": None, "interval": None}}

    def make_simple_run(self, repo: Path, run_name: str, values: list) -> Path:
        run_dir = self.make_run(repo, "ws1", run_name)
        self.write_config(run_dir, {})
        self.write_raw_master(
            run_dir,
            [("t/a", step * 10, value) for step, value in enumerate(values)],
            defs=self.DEFS,
        )
        return run_dir

    def test_two_runs_get_delta_and_ratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_simple_run(repo, "run_a", [1.0, 3.0])
            self.make_simple_run(repo, "run_b", [2.0, 4.0])

            code, result, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "run_b", "--metric", "t/a"]
            )

            self.assertEqual(code, 0)
            block = result["comparison"][0]
            self.assertEqual(block["stat"], "mean")
            row = block["rows"][0]
            self.assertEqual([item["value"] for item in row["values"]], [2.0, 3.0])
            self.assertAlmostEqual(row["delta"], 1.0)
            self.assertAlmostEqual(row["delta_ratio"], 0.5)

    def test_three_runs_get_spread_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_simple_run(repo, "run_a", [1.0, 1.0])
            self.make_simple_run(repo, "run_b", [2.0, 2.0])
            self.make_simple_run(repo, "run_c", [3.0, 3.0])

            code, result, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "run_b", "run_c", "--metric", "t/a"]
            )

            self.assertEqual(code, 0)
            row = result["comparison"][0]["rows"][0]
            self.assertAlmostEqual(row["mean"], 2.0)
            self.assertAlmostEqual(row["population_std"], math.sqrt(2.0 / 3.0))
            self.assertAlmostEqual(row["range"], 2.0)

    def test_stat_option_switches_the_compared_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_simple_run(repo, "run_a", [1.0, 5.0])

            code, result, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "--metric", "t/a", "--stat", "last"]
            )

            self.assertEqual(code, 0)
            self.assertEqual(result["comparison"][0]["rows"][0]["values"][0]["value"], 5.0)

    def test_markdown_has_comparison_and_detail_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_simple_run(repo, "run_a", [1.0, 3.0])
            self.make_simple_run(repo, "run_b", [2.0, 4.0])

            code, text, _ = self.run_cli(
                repo, ["metrics", "run_a", "run_b", "--metric", "t/a", "--format", "md"]
            )

            self.assertEqual(code, 0)
            self.assertIn("## Comparison", text)
            self.assertIn("## Detail", text)
            self.assertIn("delta_ratio", text)
            self.assertIn("range_status", text)
            self.assertIn("source_key", text)
            self.assertNotIn("## Series", text)

    def test_series_is_opt_in(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_simple_run(repo, "run_a", [1.0, 2.5, 3.0])

            code, result, _ = self.run_cli_json(repo, ["metrics", "run_a", "--metric", "t/a"])
            self.assertEqual(code, 0)
            self.assertNotIn("series", result["runs"][0]["metrics"][0]["ranges"][0])

            code, result, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "--metric", "t/a", "--series"]
            )
            self.assertEqual(code, 0)
            self.assertEqual(
                result["runs"][0]["metrics"][0]["ranges"][0]["series"],
                [[0, 1.0], [10, 2.5], [20, 3.0]],
            )

            code, text, _ = self.run_cli(
                repo, ["metrics", "run_a", "--metric", "t/a", "--series", "--format", "md"]
            )
            self.assertIn("## Series", text)
            self.assertIn("0:1", text)

    def test_series_downsamples_deterministically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            values = [math.sin(index) * 100.0 + index for index in range(1000)]
            self.make_simple_run(repo, "run_a", values)

            code, first, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "--metric", "t/a", "--series"]
            )
            _, second, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "--metric", "t/a", "--series"]
            )

            self.assertEqual(code, 0)
            series = first["runs"][0]["metrics"][0]["ranges"][0]["series"]
            self.assertLessEqual(len(series), 128)
            self.assertEqual(series[0][0], 0)
            self.assertEqual(series[-1][0], 9990)
            self.assertIn(max(values), [point[1] for point in series])
            self.assertEqual(series, second["runs"][0]["metrics"][0]["ranges"][0]["series"])

    def test_invalid_values_and_step_regression(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {})
            lines = [self.defs_line({"t/a": self.metric_def(), "t/b": self.metric_def()})]
            lines.append(
                '{"step":0,"tag":"t/a","type":"scalar","value":1.0}\n'
                '{"step":10,"tag":"t/a","type":"scalar","value":null}\n'
                '{"step":20,"tag":"t/a","type":"scalar","value":"nope"}\n'
                '{"step":30,"tag":"t/a","type":"scalar","value":NaN}\n'
                '{"step":40,"tag":"t/a","type":"scalar","value":1e39}\n'
                '{"step":50,"tag":"t/a","type":"scalar","value":true}\n'
                '{"step":60,"tag":"t/a","type":"scalar","value":5.0}\n'
                '{"step":0,"tag":"t/b","type":"scalar","value":1.0}\n'
                '{"step":10,"tag":"t/b","type":"scalar","value":2.0}\n'
                '{"step":5,"tag":"t/b","type":"scalar","value":9.0}\n'
                '{"step":20,"tag":"t/b","type":"scalar","value":3.0}\n'
            )
            (run_dir / "metrics.jsonl").write_text("".join(lines), encoding="utf-8")

            code, result, _ = self.run_cli_json(
                repo, ["metrics", "run_a", "--metric", "t/a", "--metric", "t/b", "--series"]
            )

            self.assertEqual(code, 0)
            by_tag = {item["tag"]: item for item in result["runs"][0]["metrics"]}
            self.assertEqual(by_tag["t/a"]["excluded"], 5)
            self.assertEqual(by_tag["t/a"]["ranges"][0]["count"], 2)
            self.assertEqual(by_tag["t/b"]["status"], "quarantined")
            self.assertEqual(by_tag["t/b"]["ranges"][0]["series"], [[0, 1.0], [10, 2.0]])

    def test_cache_and_master_paths_agree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            points = [(step, float(step % 7)) for step in range(0, 900, 3)]
            _, _, cache = self.make_cached_run(repo, points=points)

            argv = ["metrics", "run_a", "--metric", "t/a", "--range", "0:500", "--range", "-50%:"]
            _, cached, _ = self.run_cli_json(repo, argv)
            cache.unlink()
            _, from_master, _ = self.run_cli_json(repo, argv)

            self.assertEqual(cached["runs"][0]["metrics_source"]["selected"], "cache")
            self.assertEqual(from_master["runs"][0]["metrics_source"]["selected"], "master")
            self.assertEqual(cached["runs"][0]["metrics"], from_master["runs"][0]["metrics"])

    def test_raw_snapshot_and_source_change_are_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_simple_run(repo, "run_a", [1.0, 2.0, 3.0])
            master = run_dir / "metrics.jsonl"
            real_open = subject.open_metrics_binary

            def opener(path):
                handle = real_open(path)
                with master.open("a", encoding="utf-8") as appended:
                    appended.write(self.scalar_lines([("t/a", 999, 9.0)]))
                return handle

            with mock.patch.object(subject, "open_metrics_binary", side_effect=opener):
                code, out, err = self.run_cli(repo, ["metrics", "run_a", "--metric", "t/a"])
            result = json.loads(out)

            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["metrics"][0]["ranges"][0]["count"], 3)
            self.assertTrue(result["runs"][0]["metrics_source"]["source_changed_during_read"])
            self.assertIn("changed", err)

    def test_gzip_source_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            run_dir = self.make_run(repo, "ws1", "run_a")
            self.write_config(run_dir, {"metrics.scalar.[t/a]": "a_key @train $exp_step"})
            self.write_gzip_master(run_dir, [("t/a", 0, 1.0), ("t/a", 10, 2.0)])

            code, result, _ = self.run_cli_json(repo, ["metrics", "run_a", "--metric", "t/a"])
            self.assertEqual(code, 0)
            self.assertTrue(result["runs"][0]["metrics_source"]["master_path"].endswith(".gz"))
            self.assertEqual(result["runs"][0]["metrics"][0]["ranges"][0]["count"], 2)

            (run_dir / "metrics.jsonl").write_text("{oops\n", encoding="utf-8")
            code, out, _ = self.run_cli(repo, ["metrics", "run_a", "--metric", "t/a"])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["runs"][0]["metrics"][0]["status"], "source_error")

    def test_json_is_strict_and_output_file_is_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_simple_run(repo, "run_a", [1.0, 2.0])
            target = repo / "out.json"
            target.write_text("stale content", encoding="utf-8")

            code, out, _ = self.run_cli(
                repo, ["metrics", "run_a", "--metric", "t/a", "--output", str(target)]
            )

            self.assertEqual(code, 0)
            self.assertEqual(out, "")
            written = json.loads(target.read_text(encoding="utf-8"), parse_constant=self.fail)
            self.assertEqual(written["runs"][0]["run_name"], "run_a")
            self.assertEqual(sorted(item.name for item in repo.glob("out.json*")), ["out.json"])

    def test_output_failures_are_reported_without_traceback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_simple_run(repo, "run_a", [1.0])
            target = repo / "out.json"
            target.write_text("keep me", encoding="utf-8")

            with mock.patch.object(subject.os, "replace", side_effect=OSError("boom")):
                code, _, err = self.run_cli(
                    repo, ["metrics", "run_a", "--metric", "t/a", "--output", str(target)]
                )
            self.assertEqual(code, 1)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep me")
            self.assertIn("boom", err)
            self.assertEqual(sorted(item.name for item in repo.glob("out.json*")), ["out.json"])

            # 一時 file 作成自体が失敗しても traceback を出さない。
            with mock.patch.object(
                subject.tempfile, "NamedTemporaryFile", side_effect=PermissionError("denied")
            ):
                code, _, err = self.run_cli(
                    repo, ["metrics", "run_a", "--metric", "t/a", "--output", str(target)]
                )
            self.assertEqual(code, 1)
            self.assertIn("denied", err)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep me")

            code, _, err = self.run_cli(
                repo,
                ["metrics", "run_a", "--metric", "t/a", "--output", str(repo / "missing" / "o.json")],
            )
            self.assertEqual(code, 2)
            self.assertIn("missing", err)

    def test_single_pass_opens_each_master_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.make_simple_run(repo, "run_a", [1.0, 2.0])
            self.make_simple_run(repo, "run_b", [3.0, 4.0])
            real_open = subject.open_metrics_binary

            with mock.patch.object(subject, "open_metrics_binary", side_effect=real_open) as opener:
                code, result, _ = self.run_cli_json(
                    repo,
                    ["metrics", "run_a", "run_b", "--metric", "t/a",
                     "--range", "0:100", "--range", "0:0"],
                )

            self.assertEqual(code, 0)
            self.assertEqual(opener.call_count, 2)
            self.assertEqual(len(result["runs"][0]["metrics"][0]["ranges"]), 2)


class TraceChannelDefinitionsTest(CacheFixtureMixin):
    def test_master_old_and_new_defs_ignore_same_tag_trace(self):
        for name in ("metrics.defs", "metrics.scalar.defs"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                repo = Path(temp)
                run = self.make_run(repo, "ws1", "run_trace")
                master = run / "metrics.jsonl"
                records = self.defs_line({"score": self.metric_def()}, tag=name)
                records += json.dumps({"type": "trace", "tag": "score", "step": 5,
                                       "lane": 0, "data": {"score": 9}}) + "\n"
                records += json.dumps({"type": "scalar", "tag": "score", "step": 5,
                                       "value": 2}) + "\n"
                master.write_text(records, encoding="utf-8")
                code, result, err = self.run_cli_json(repo, ["tags", str(run)])
                self.assertEqual(code, 0)
                self.assertEqual(err, "")
                self.assertEqual(result["runs"][0]["def_source"], "metrics_defs")
                self.assertEqual(len(result["runs"][0]["tags"]), 1)
                code, result, err = self.run_cli_json(repo, ["metrics", str(run), "--metric", "score"])
                self.assertEqual(code, 0)
                self.assertEqual(err, "")

    def test_cache_reads_old_and_new_definitions_with_new_name_priority(self):
        for names in (("metrics.defs",), ("metrics.scalar.defs",),
                      ("metrics.scalar.defs", "metrics.defs"), ("metrics.defs", "metrics.scalar.defs")):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temp:
                repo = Path(temp)
                run, master, cache = self.make_cached_run(repo)
                with sqlite3.connect(cache) as conn:
                    conn.execute("DELETE FROM json_lines")
                    for ordinal, name in enumerate(names):
                        record = {"type": "json", "tag": name, "data": {
                            "t/a": self.metric_def(source_key=name)}}
                        conn.execute("INSERT INTO json_lines(ordinal,type,tag,json) VALUES(?,?,?,?)",
                                     (ordinal, "json", name, json.dumps(record)))
                conn.close()
                code, result, err = self.run_cli_json(repo, ["tags", str(run)])
                self.assertEqual(code, 0)
                self.assertEqual(err, "")
                self.assertEqual(result["runs"][0]["def_source"], "metrics_defs")
                expected = "metrics.scalar.defs" if "metrics.scalar.defs" in names else "metrics.defs"
                self.assertEqual(result["runs"][0]["tags"][0]["source_key"], expected)

    def test_uncached_session_end_definition_uses_train_counts_and_exp_default(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            run = self.make_run(repo, "ws1", "run_session")
            self.write_config(run, {
                "metrics.scalar.[score]": "$eval.[eval1] @session_end $env mean.score",
                "metrics.scalar.[explicit]": "$eval.[eval1] event:session_end $env $learn_step mean.score",
            })
            self.write_raw_master(run, [("score", 17, 2.0), ("explicit", 3, 2.0)])
            code, result, err = self.run_cli_json(repo, ["tags", str(run), "--no-observed"])
            self.assertEqual(code, 0)
            by_tag = {r["tag"]: r for r in result["runs"][0]["tags"]}
            self.assertEqual(by_tag["score"]["event"], "session_end")
            self.assertEqual(by_tag["score"]["step_axis"], "exp_step")
            self.assertEqual(by_tag["score"]["runner"], "train")
            self.assertEqual(by_tag["explicit"]["step_axis"], "learn_step")
            code, result, err = self.run_cli_json(repo, ["metrics", str(run), "--metric", "s*"])
            self.assertEqual(code, 0)
            self.assertEqual(result["runs"][0]["metrics"][0]["step_axis"], "exp_step")

    def test_master_prefers_scalar_defs_regardless_of_record_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            old = self.defs_line({"score": self.metric_def(source_key="old")}, tag="metrics.defs")
            new = self.defs_line({"score": self.metric_def(source_key="new", event="session_end")}).replace(
                '"metrics.defs"', '"metrics.scalar.defs"')
            for records in (old + new, new + old):
                with self.subTest(records=records):
                    path.write_text(records, encoding="utf-8")
                    scan = subject.scan_master(path, None, None)
                    self.assertEqual(scan.defs["score"].source_key, "new")
                    self.assertEqual(scan.defs["score"].event, "session_end")


class MetricDefinitionMetadataTest(CacheFixtureMixin):
    def test_master_and_cache_preserve_subscription_eval_conditions_and_clip(self):
        defs = {
            "t/a": {**self.metric_def(event="session_end"), "scope": "eval", "eval_name": "eval1",
                    "eval_episodes": 3, "num_envs": 2, "clip": 7.5},
            "other": {**self.metric_def(runner="eval2", event="train"), "scope": "eval", "eval_name": "eval2",
                      "eval_episodes": 1, "num_envs": 4, "clip": None},
            "52_eval2/misleading": {**self.metric_def(), "scope": "train", "eval_name": None,
                                   "eval_episodes": None, "num_envs": None, "clip": None},
        }
        for cached in (False, True):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as temp:
                repo = Path(temp)
                if cached:
                    run, _, _ = self.make_cached_run(repo, run_defs=defs)
                else:
                    run = self.make_run(repo, "ws1", "run_a")
                    self.write_raw_master(run, [("t/a", 10, 2.0)], defs=defs)
                code, result, err = self.run_cli_json(repo, ["tags", str(run)])
                self.assertEqual((code, err), (0, ""))
                node = result["runs"][0]
                self.assertEqual(node["metrics_source"]["selected"], "cache" if cached else "master")
                by_tag = {item["tag"]: item for item in node["tags"]}
                for tag, expected in defs.items():
                    for key, value in expected.items():
                        self.assertEqual(by_tag[tag][key], value, (tag, key))
                code, output, err = self.run_cli(repo, ["tags", str(run), "--format", "md"])
                self.assertEqual((code, err), (0, ""))
                self.assertIn("| clip | scope | eval_name | eval_episodes | num_envs |", output)
                self.assertIn("| 7.5 | eval | eval1 | 3 | 2 |", output)

    def test_old_records_leave_missing_metadata_unknown_even_with_config(self):
        for cached in (False, True):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as temp:
                repo = Path(temp)
                run, _, cache = self.make_cached_run(repo)
                if not cached:
                    cache.unlink()
                code, result, err = self.run_cli_json(repo, ["tags", str(run)])
                self.assertEqual((code, err), (0, ""))
                definition = result["runs"][0]["tags"][0]
                for key in ("scope", "eval_name", "eval_episodes", "num_envs", "clip"):
                    self.assertIsNone(definition[key], key)

    def test_config_derives_subscription_and_clip_without_assuming_eval_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            run = self.make_run(repo, "ws1", "run_config")
            self.write_config(run, {
                "metrics.scalar.[a]": "$eval.[eval1] @session_end $env mean.score clip:2 clip:3",
                "metrics.scalar.[b]": "$eval.[eval2] @train $action_info score",
                "metrics.scalar.[c]": "$eval.[ignored] $train @train $env score clip:0",
                "train.eval.[eval1].eval_episodes": "10",
                "train.eval.[eval1].eval_batch_size": "5",
            })
            self.write_raw_master(run, [("a", 10, 1.0)])
            for options in ([], ["--no-observed"]):
                with self.subTest(options=options):
                    code, result, _ = self.run_cli_json(repo, ["tags", str(run)] + options)
                    self.assertEqual(code, 0)
                    node = result["runs"][0]
                    self.assertEqual(node["def_source"], "config_derived")
                    by_tag = {item["tag"]: item for item in node["tags"]}
                    for tag, scope, name, runner, clip in (
                        ("a", "eval", "eval1", "train", 3.0),
                        ("b", "eval", "eval2", "eval2", None),
                        ("c", "train", None, "train", 0.0),
                    ):
                        entry = by_tag[tag]
                        self.assertEqual((entry["scope"], entry["eval_name"], entry["runner"], entry["clip"]),
                                         (scope, name, runner, clip))
                        self.assertIsNone(entry["eval_episodes"])
                        self.assertIsNone(entry["num_envs"])


if __name__ == "__main__":
    unittest.main()
