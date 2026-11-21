"""合成指標で smoke の対象点を確かめる。Runner は起動しない。"""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / "prd116"))
spec = importlib.util.spec_from_file_location("prd116_smoke", HERE.parent / "prd116/run_validation.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class SmokeValidationTest(unittest.TestCase):
    def test_learning_during_warmup_does_not_select_empty_error_statistics(self):
        # warm-up 中にも learn_count が増えるが、誤差の統計はまだ空である。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "bin/runner.exe"
            runner.parent.mkdir()
            runner.write_bytes(b"synthetic")
            run = root / "run"
            (run / "config").mkdir(parents=True)
            (run / "config/config_data.txt").write_text(
                "DefaultDQNAgent.intrinsic.rnd.warmup_steps = 16\n", encoding="utf-8")
            tags = ("01_bonus_mean", "02_bonus_max", "03_bonus_share", "04_bonus_min", "05_bonus_pos_ratio",
                    "11_error_mean", "12_error_std", "13_error_step", "14_target_feature_var", "15_target_feature_max",
                    "21_clip_ratio", "31_predictor_loss", "32_learn_count")
            rows = []
            for step in (16, 24):
                for tag in tags:
                    value = 0.5
                    if step == 16 and tag in ("11_error_mean", "12_error_std"):
                        value = float("nan")
                    rows.append({"step": step, "tag": "56_agent_in/" + tag, "value": value})
            reports = {}
            execute = mock.Mock(return_value=run)
            driver = SimpleNamespace(OUT=root, RUNNER=runner,
                settings=lambda *a, **kw: ["run.$ = run.@base > run.@validation"],
                execute=execute, rows=lambda path: rows,
                save=lambda path, report: reports.update({path.name: report}))
            fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
            with mock.patch.object(validation.sys, "argv", ["smoke", "smoke", "--env", "Atari", "--label", "synthetic"]), \
                 mock.patch.object(validation.importlib.util, "spec_from_file_location", return_value=fake_spec), \
                 mock.patch.object(validation.importlib.util, "module_from_spec", return_value=driver):
                validation.main()
            report = reports["synthetic_smoke_Atari.json"]
            self.assertTrue(report["acceptance"]["passed"])
            self.assertEqual(report["11_error_mean"]["count"], 1)
            self.assertEqual(report["12_error_std"]["count"], 1)
            self.assertIn("DefaultDQNAgent.intrinsic.rnd.warmup_steps=12800", execute.call_args.kwargs["overrides"])


if __name__ == "__main__":
    unittest.main()
