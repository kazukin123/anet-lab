"""RFD0130: 性能受入の判定を、Runner を起動せずに確かめる。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("rfd0130_performance", Path(__file__).with_name("run_performance.py"))
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)


class PerformanceAcceptanceTest(unittest.TestCase):
    def test_cpu_history_chooses_last_provably_early_sample(self):
        # 初期化が長くても、遅い最新サンプルを使わず開始前の履歴を選ぶ。
        samples = [{"offset_seconds": value, "cpu": {"123": value / 100}}
                   for value in (0.0, 40.0, 51.0, 53.0)]
        self.assertEqual(performance.select_paused_sample(samples, 52.45), samples[2])
        with self.assertRaisesRegex(RuntimeError, "No CPU sample"):
            performance.select_paused_sample(samples[3:], 52.45)

    def test_cpu_history_rejects_missing_counters_and_time_regression(self):
        for last in ({}, {"123": float("nan")}, {"123": 0.5}):
            with self.subTest(last=last), self.assertRaisesRegex(RuntimeError, "CPU sample"):
                performance.select_paused_sample([
                    {"offset_seconds": 0.0, "cpu": {"123": 1.0}},
                    {"offset_seconds": 1.0, "cpu": last}], 10.0)

    def test_delayed_clock_output_cannot_hide_a_late_cpu_sample(self):
        # ファイルの経験数が古くても、実測時計より遅い CPU 読み取りを拒否する。
        record = {"paused_guard_cpu_seconds": {"123": 0.1},
                  "paused_guard_end_experience": 49216, "window_experiences": 16384,
                  "paused_run_state_before": {}, "paused_run_state_after": {},
                  "paused_guard_start_offset_seconds": 51.0, "measurement_start_elapsed_seconds": 50.0}
        with self.assertRaisesRegex(RuntimeError, "does not cover measurement"):
            performance.verify_paused_run({"paused_pids": [123]}, record)

    def test_paused_run_progress_is_rejected_even_with_low_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run_paused"
            run.mkdir()
            metrics = run / "metrics.jsonl"
            metrics.write_text("old\n", encoding="utf-8")
            (run / "run_paused.log").write_text("paused\n", encoding="utf-8")
            plan = {"paused_pids": [123], "paused_runs": {"123": str(run)}}
            before = performance.paused_run_state(plan)
            # 実際のファイルの進行を公開の状態読み取りから検出する。
            metrics.write_text("new progress\n", encoding="utf-8")
            record = {"paused_guard_cpu_seconds": {"123": 0.1},
                      "paused_guard_end_experience": 49216, "window_experiences": 16384,
                      "paused_guard_start_offset_seconds": 40.0, "measurement_start_elapsed_seconds": 50.0,
                      "paused_run_state_before": before, "paused_run_state_after": performance.paused_run_state(plan)}
            with self.assertRaisesRegex(RuntimeError, "metrics or log changed"):
                performance.verify_paused_run(plan, record)

    def test_late_paused_guard_start_is_rejected(self):
        record = {"paused_guard_cpu_seconds": {"123": 0.1},
                  "paused_guard_end_experience": 49216, "window_experiences": 16384,
                  "paused_guard_start_offset_seconds": 51.0, "measurement_start_elapsed_seconds": 50.0,
                  "paused_run_state_before": {}, "paused_run_state_after": {}}
        with self.assertRaisesRegex(RuntimeError, "does not cover measurement"):
            performance.verify_paused_run({"paused_pids": [123]}, record)

    def test_paused_pid_requires_its_run_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "requires one --paused-run"):
                performance.prepare(Path(folder) / "out", Path(folder), [123], stage=2)

    def test_paused_guard_counts_the_covered_interval_and_keeps_one_second_limit(self):
        plan = {"paused_pids": [123]}
        state = {"123": {"metrics": {"size": 100, "mtime_ns": 1}}}
        record = {"paused_cpu_seconds": {"123": 1.1}, "paused_guard_cpu_seconds": {"123": 0.38},
                  "paused_guard_end_experience": 49216,
                  "window_experiences": 16384, "paused_run_state_before": state, "paused_run_state_after": state,
                  "paused_guard_start_offset_seconds": 40.0, "measurement_start_elapsed_seconds": 50.0}
        performance.verify_paused_run(plan, record)
        record["paused_guard_cpu_seconds"]["123"] = 1.1
        with self.assertRaises(RuntimeError):
            performance.verify_paused_run(plan, record)

    def test_stage2_accepts_difference_inside_spread_and_rejects_wrong_order(self):
        order = ("stage1", "stage2", "stage2", "stage1") * 3
        records = [{"arm": arm, "measurement_seconds": (20 + 0.4 * (i % 2)) if arm == "stage1" else 20.3,
                    "window_experiences": 8192} for i, arm in enumerate(order)]
        self.assertTrue(performance.summarize_records("legacy", records, stage=2)["accepted"])
        records[-1]["arm"] = "stage2"
        with self.assertRaises(ValueError):
            performance.summarize_records("legacy", records, stage=2)

    def test_stage2_prepare_selects_accepted_reference_and_functional_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            # 実行体と受入記録だけを持つ保存基準を作り、Runner は起動しない。
            root = Path(folder)
            packages = root / "packages"
            config = packages / "before/package/config"
            config.mkdir(parents=True)
            (config / "Atari.txt").write_text("run.$ = run.@fixture\n", encoding="utf-8")
            hashes = {}
            for label in ("stage1-accepted", "stage2"):
                package = packages / label / "package"
                (package / "Release").mkdir(parents=True)
                binary = package / "Release/AnetRLRunner.exe"
                binary.write_bytes(label.encode())
                hashes[label] = performance.grid.digest(binary)
                (package / "manifest.json").write_text(json.dumps({"sha256": {
                    "Release/AnetRLRunner.exe": hashes[label]}}), encoding="utf-8")
            (packages / "stage2/functional-acceptance.json").write_text(json.dumps({
                "criteria_passed": [11, 12, 14, 15], "runner_sha256": {"Release": hashes["stage2"]}}), encoding="utf-8")
            out = root / "measurements"
            # OS の稼働プロセスの列挙だけを置き換え、実際の prepare の出力を読む。
            with patch.object(performance, "check_idle"):
                performance.prepare(out, packages, [], stage=2)
            plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(set(plan["packages"]), {"stage1", "stage2"})
            self.assertEqual(plan["packages"]["stage1"]["sha256"], hashes["stage1-accepted"])
            self.assertEqual(plan["packages"]["stage2"]["sha256"], hashes["stage2"])
            self.assertEqual(set(plan["sets"]), {"main", "legacy", "taiga"})

    def test_stage2_taiga_checks_resolved_mode_and_learning_cadence(self):
        with tempfile.TemporaryDirectory() as folder:
            # Taiga の解決値を受け付け、学習の刻みが変わった入力を拒否する。
            run = Path(folder) / "run_test"
            (run / "config").mkdir(parents=True)
            values = {"run.train.num_envs": "64", "backend.torch_num_threads": "1",
                "DefaultDQNAgent.learner.update_warmup_steps": "20000",
                "DefaultDQNAgent.intrinsic.enabled": "true", "agent.device": "cuda",
                "DefaultDQNAgent.intrinsic.rnd.warmup_steps": "12800",
                "DefaultDQNAgent.intrinsic.rnd.statistics_mode": "variance_feedback",
                "DefaultDQNAgent.intrinsic.observation": "state",
                "DefaultDQNAgent.intrinsic.learn_batch_size": "1",
                "DefaultDQNAgent.intrinsic.rnd.multiplier_mode": "unclipped",
                "DefaultDQNAgent.intrinsic.learn_ratio": "1",
                "DefaultDQNAgent.intrinsic.beta": "0.00005",
                "DefaultDQNAgent.intrinsic.rnd.adam_eps": "1e-5"}
            path = run / "config/config_data.txt"
            path.write_text("\n".join(key + " = " + value for key, value in values.items()), encoding="utf-8")
            (run / "run_test.log").write_text("", encoding="utf-8")
            performance.verify_config(run, "taiga", "stage2", stage=2)
            values["DefaultDQNAgent.intrinsic.learn_batch_size"] = "32"
            path.write_text("\n".join(key + " = " + value for key, value in values.items()), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                performance.verify_config(run, "taiga", "stage2", stage=2)

    def test_stage2_arms_share_config_and_taiga_selects_its_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            # 同じ入力の連鎖に対し、実行体の選択だけが違う両群の設定を比べる。
            out = Path(folder)
            (out / "config").mkdir()
            (out / "config/Atari.txt").write_text("run.$ = run.@fixture\n", encoding="utf-8")
            for name in ("main", "legacy", "taiga"):
                first = performance.build(out, "same", name, "stage1", 8192, stage=2)
                second = performance.build(out, "same", name, "stage2", 8192, stage=2)
                self.assertEqual(first, second)
                self.assertIn("intrinsic.rnd.warmup_steps = 12800", first)
                if name == "taiga":
                    self.assertIn("run.@rnd_taiga", first)
                elif name == "main":
                    self.assertIn("intrinsic.value_mode=decomposed", first)

    def test_stage2_regression_outside_stage1_spread_is_not_accepted(self):
        order = ("stage1", "stage2", "stage2", "stage1") * 3
        records = [{"arm": arm, "measurement_seconds": (20 + 0.4 * (i % 2)) if arm == "stage1" else 20.7,
                    "window_experiences": 8192} for i, arm in enumerate(order)]
        result = performance.summarize_records("main", records, stage=2)
        self.assertFalse(result["accepted"])
        self.assertAlmostEqual(result["stage1_spread_seconds"], 0.4)

    def test_main_ratio_above_limit_is_not_accepted(self):
        records = [{"arm": arm, "measurement_seconds": 20 if arm == "off" else 23.2,
                    "window_experiences": 8192} for arm in ("off", "on", "on", "off") * 3]
        result = performance.summarize_records("main", records)
        self.assertFalse(result["accepted"])
        self.assertAlmostEqual(result["ratio"], 1.16)

    def test_different_experience_windows_are_rejected(self):
        records = [{"arm": arm, "measurement_seconds": 20, "window_experiences": 8192}
                   for arm in ("off", "on", "on", "off") * 3]
        records[-1]["window_experiences"] = 16384
        with self.assertRaises(ValueError):
            performance.summarize_records("main", records)

    def test_grid_regression_outside_before_spread_is_not_accepted(self):
        records = [{"arm": arm, "measurement_seconds": 20 if arm == "before" else 23,
                    "window_experiences": 8192} for arm in ("before", "stage1", "stage1", "before") * 3]
        self.assertFalse(performance.summarize_records("grid", records)["accepted"])

    def test_grid_config_does_not_schedule_atari_only_evaluation_tags(self):
        config = performance.build(Path("."), "test", "grid", "before", 8192)
        self.assertNotIn("run.eval_schedule.[greedy]", config)
        self.assertNotIn("run.eval_schedule.[greedy_single]", config)


if __name__ == "__main__":
    unittest.main()
