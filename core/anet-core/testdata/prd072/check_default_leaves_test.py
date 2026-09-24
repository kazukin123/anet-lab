"""既定葉検査の抽出から判定までを合成した入力で検証する。"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

import check_default_leaves as checker


class DefaultLeafAuditTest(unittest.TestCase):
    def test_shared_actor_catalog_leaves_are_defaults(self):
        groups = [
            ([f"ImageClsAgent.actor.[{tag}].{leaf}"
              for tag in ("train", "eval", "eval_target")
              for leaf in ("clone_model", "bf16")], "?="),
            (["DefaultDQNAgent.actor.@target.network"], "="),
            (["DefaultDQNAgent.actor.@custom.network",
              "DefaultDQNAgent.@munchausen.enabled",
              "DefaultDQNAgent.@random.policy_type"], "="),
        ]
        for keys, expected_operator in groups:
            for operator in ("=", "?="):
                with self.subTest(keys=keys, operator=operator):
                    code, report = self.audit([
                        ("apps/runner/config/agent.txt", f"{key} {operator} value") for key in keys])
                    self.assertEqual(code, 0 if operator == expected_operator else 1)
                    self.assertEqual(len(report), len(keys))
                    self.assertTrue(all(row["expected_operator"] == expected_operator for row in report))

    def audit(self, lines):
        records = [json.dumps({"type": "match", "data": {
            "path": {"text": line[0] if isinstance(line, tuple) else "apps/runner/config/Example.txt"},
            "line_number": number, "lines": {"text": line[1] if isinstance(line, tuple) else line}}})
            for number, line in enumerate(lines, 1)]
        process = subprocess.CompletedProcess([], 0, "\n".join(records), "")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(checker, "ROOT", Path(directory)), patch.object(checker.subprocess, "run", return_value=process):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = checker.main()
            report = json.loads((Path(directory) / "default-leaf-audit.json").read_text(encoding="utf-8"))
        return code, report

    def test_default_and_selected_sources(self):
        code, report = self.audit(["Env.$ = @base > Experiment", "Env.k ?= fallback",
                                   "Env.@base : k = base", "Experiment.k = changed"])
        self.assertEqual(code, 0)
        self.assertTrue(any(row["reason"] == "default leaf" for row in report))

    def test_unexplained_leaf_fails(self):
        code, report = self.audit(["Env.$ = @base", "Env.k = accidental"])
        self.assertEqual(code, 1)
        self.assertEqual(report[0]["key"], "Env.k")
        self.assertIsNone(report[0]["reason"])

    def test_run_only_owner_is_checked(self):
        code, report = self.audit(["run.@new : Env.$ = @base", "Env.k = accidental"])
        self.assertEqual(code, 1)
        self.assertEqual(report[0]["owner"], "Env")

    def test_run_leaf_is_intentional(self):
        code, report = self.audit(["Env.$ = @base", "run.@new : Env.k = experiment"])
        self.assertEqual(code, 0)
        self.assertEqual(report[0]["reason"], "explicit Run assignment")

    def test_optuna_run_seed_is_an_explicit_trial_assignment(self):
        code, report = self.audit([
            ("apps/runner/tools/dropmerge_optuna.py", '"run.$ = run.@trial",'),
            ("apps/runner/tools/dropmerge_optuna.py", 'f"run.seed = {args.seed}",')])
        self.assertEqual(code, 0)
        self.assertEqual(report[0]["reason"], "explicit trial seed")

    def test_similarly_named_sibling_is_not_owned(self):
        code, report = self.audit(["Env.$ = @base", "Environment.k = separate"])
        self.assertEqual(code, 0)
        self.assertEqual(report, [])

    def test_convention_checks_bases_without_selection_owners(self):
        code, report = self.audit([
            ("apps/runner/config/common.txt", "run.eval_device = auto"),
            ("apps/runner/config/agent.txt", "DefaultDQNAgent.@baseline : k = base"),
            ("apps/runner/config/nn.txt", "net.block.[Linear].type = Linear")])
        self.assertEqual(code, 1)
        self.assertEqual(len(report), 3)
        self.assertTrue(all(row["expected_operator"] == "?=" for row in report))

    def test_experiment_choice_is_strong_and_base_is_weak(self):
        code, report = self.audit([
            ("apps/runner/config/DropMerge.txt", "DefaultDQNAgent.net.branch.[vector_feature].structure = Embed5846_v2"),
            ("apps/runner/config/DropMerge.txt", "app.run_name = run_example"),
            ("apps/runner/config/DropMerge.txt", "DefaultDQNAgent.net.branch.[value_stream].structure = HeadFC1024 > SiLU"),
            ("apps/runner/config/DropMerge.txt", "A2.learner.replay_batch_size = 256")])
        self.assertEqual(code, 0)
        self.assertEqual([row["expected_operator"] for row in report], ["=", "=", "=", "="])
        code, _ = self.audit([("apps/runner/config/DropMerge.txt", "app.run_name ?= run_example")])
        self.assertEqual(code, 1)

    def test_environment_defaults_do_not_require_comments(self):
        file = "apps/runner/config/DropMerge.txt"
        code, plain = self.audit([(file, "DropMergeEnv.grid_cols ?= 40")])
        self.assertEqual(code, 0)
        code, commented = self.audit([(file, "# 任意の説明"), (file, "DropMergeEnv.grid_cols ?= 40")])
        self.assertEqual(code, 0)
        self.assertEqual(plain[0]["reason"], commented[0]["reason"])

    def test_migration_preserves_comments_without_adding_markers(self):
        from migrate import migrate_text
        source = "# 好きな説明\nDropMergeEnv.grid_cols = 40\n"
        result, _ = migrate_text("DropMerge.txt", source)
        self.assertEqual(result, "# 好きな説明\nDropMergeEnv.grid_cols ?= 40\n")
        result, _ = migrate_text("DropMerge.txt", "DropMergeEnv.grid_cols = 40\n")
        self.assertEqual(result, "DropMergeEnv.grid_cols ?= 40\n")

    def test_optuna_extra_config_inherits_dropmerge_owners(self):
        code, report = self.audit([
            ("apps/runner/config/DropMerge.txt", "DropMergeEnv.$ = @base"),
            ("apps/runner/config/DropMerge_optuna.txt", "DropMergeEnv.seed_mode = global_fixed"),
            ("apps/runner/config/DropMerge_optuna.txt", "DropMergeEnv.global_seed = 12345"),
            ("apps/runner/config/DropMerge_optuna.txt", "DropMergeEnv.grid_cols = 40")])
        self.assertEqual(code, 1)
        self.assertEqual([row["key"] for row in report if row["reason"] is None], ["DropMergeEnv.grid_cols"])
        seeds = [row for row in report if row["key"] in ("DropMergeEnv.seed_mode", "DropMergeEnv.global_seed")]
        self.assertEqual(len(seeds), 2)
        self.assertTrue(all(row["reason"] == "intentional Optuna seed override" for row in seeds))


if __name__ == "__main__":
    unittest.main()
