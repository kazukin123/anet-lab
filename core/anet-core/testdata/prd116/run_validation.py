"""PRD116: isolate PRD098's exact OFF comparison from historical baselines."""
import importlib.util
import argparse
import json
import math
import os
import shutil
import sys

sys.dont_write_bytecode = True
from compare_checkpoint import compare as compare_checkpoint
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "after", "compare", "checkpoint", "smoke"))
    parser.add_argument("--env", nargs="+", choices=("GridMaze", "LunarLander", "Atari", "DropMerge", "Montezuma"),
                        default=["GridMaze", "LunarLander", "Atari", "DropMerge"])
    parser.add_argument("--label", default="after")
    args = parser.parse_args()
    # 既存の比較手順を再利用するが、過去の基準実行体と Run は変更しない。
    repo = Path(__file__).resolve().parents[4]
    source = repo / "core/anet-core/testdata/prd098/run_validation.py"
    spec = importlib.util.spec_from_file_location("prd098_validation", source)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.OUT = repo / ".scratch/prd116/baseline-20261106"
    driver.RUN_PREFIX = "prd116"
    if args.stage in ("checkpoint", "smoke"):
        # 実行中の再ビルドから切り離し、記録済み実行体の上書きを拒否する。
        source = driver.RUNNER
        folder = driver.OUT / (args.label + "-bin")
        folder.mkdir(exist_ok=True)
        snapshot = folder / source.name
        if snapshot.exists():
            if snapshot.read_bytes() != source.read_bytes():
                raise RuntimeError("Binary changed; choose a fresh label")
        else:
            shutil.copy2(source, snapshot)
            for dll in source.parent.glob("*.dll"):
                os.link(dll, folder / dll.name)
        driver.RUNNER = snapshot
        if args.stage == "checkpoint":
            checkpoints = []
            for before in (True, False):
                name = args.label + ("_checkpoint_before" if before else "_checkpoint_after")
                run = driver.execute(name, "GridMaze", before=before, overrides=(
                    "app.save_agent_on_close=true", "app.exp_exit_step=16", "app.batchrun.exp_exit_step=16"))
                checkpoint = run / "agent_close.anet"
                checkpoints.append(checkpoint)
            report = compare_checkpoint(*checkpoints)
            driver.save(driver.OUT / (args.label + "_checkpoint.json"), report)
            if not all(report[key] for key in ("header_and_config_equal", "network_bytes_equal", "optimizer_records_equal")):
                raise RuntimeError("Disabled checkpoint content differs")
            print("CHECKPOINT", json.dumps(report), flush=True)
        else:
            if any(env not in ("GridMaze", "Atari", "Montezuma") for env in args.env):
                parser.error("smoke requires --env GridMaze Atari Montezuma (or a subset)")
            original_settings = driver.settings
            def rnd_settings(*settings_args, **settings_kwargs):
                lines = original_settings(*settings_args, **settings_kwargs)
                # 短縮用の設定を最後に保ち、Q比較群とRNDの選択をその前へ挿す。
                assignment, selected = lines[-1].split("=", 1)
                chain, _, validation = selected.strip().rpartition(" > ")
                profiles = "run.@rainbow_like > run.@rnd > run.@rnd_taiga" if current_env == "Montezuma" else "run.@rnd"
                lines[-1] = assignment.strip() + " = " + " > ".join(part for part in (chain, profiles, validation) if part)
                return lines
            driver.settings = rnd_settings
            for env in args.env:
                current_env = env
                overrides = ["app.exp_exit_step=100000", "app.batchrun.exp_exit_step=100000",
                             "app.save_agent_on_close=true"]
                if env == "Atari":
                    overrides.append("E1.game=kung_fu_master")
                if env == "Montezuma":
                    overrides.append("E1.game=montezuma_revenge")
                run = driver.execute(args.label + "_tmp_smoke_" + env,
                                     "Atari" if env == "Montezuma" else env, overrides=overrides)
                summary = {}
                by_step = {}
                for row in driver.rows(run):
                    tag = row.get("tag", "")
                    if not tag.startswith("56_agent_in/"):
                        continue
                    by_step.setdefault(row["step"], {})[tag.split("/", 1)[1]] = row["value"]
                expected = {"01_bonus_mean", "02_bonus_max", "03_bonus_share", "04_bonus_min",
                            "05_bonus_pos_ratio", "11_error_mean", "12_error_std", "13_error_step",
                            "14_target_feature_var", "15_target_feature_max", "21_clip_ratio",
                            "31_predictor_loss", "32_learn_count"}
                for step, values in by_step.items():
                    if values.get("32_learn_count", 0) <= 0:
                        continue
                    for tag in expected:
                        value = values.get(tag, float("nan"))
                        if not math.isfinite(value):
                            if tag == "03_bonus_share" and values.get("01_bonus_mean") == 0:
                                continue
                            raise RuntimeError(f"Nonfinite/missing intrinsic metric: {env} step={step} {tag}")
                        summary.setdefault(tag, []).append((step, value))
                if not summary:
                    raise RuntimeError("No successful auxiliary learning: " + env)
                report = {}
                for tag, points in summary.items():
                    values = [value for _, value in points]
                    quarter = max(1, len(values) // 4)
                    report[tag] = {"count": len(values), "min": min(values), "max": max(values),
                                   "first_quarter_mean": sum(values[:quarter]) / quarter,
                                   "last_quarter_mean": sum(values[-quarter:]) / quarter}
                checks = {}
                if env == "GridMaze":
                    points = summary["01_bonus_mean"]
                    early = [value for step, value in points if step < 50000]
                    late = [value for step, value in points if step >= 50000]
                    checks["positive_ratio"] = sum(value > 0 for _, value in points) / len(points)
                    checks["decreases"] = bool(early and late and sum(late) / len(late) < sum(early) / len(early))
                    checks["passed"] = checks["positive_ratio"] >= 0.9 and checks["decreases"]
                elif env == "Atari":
                    checks["passed"] = all(value < 1 for _, value in summary["21_clip_ratio"])
                else:
                    points = summary["04_bonus_min"]
                    checks["negative_ratio"] = sum(value < 0 for _, value in points) / len(points)
                    checks["passed"] = checks["negative_ratio"] >= 0.9
                report["acceptance"] = checks
                driver.save(driver.OUT / (args.label + "_smoke_" + env + ".json"), report)
                print("SMOKE metrics", env, json.dumps(report), flush=True)
                if not checks["passed"]:
                    raise RuntimeError("Smoke acceptance failed: " + env)
        return
    driver.main()
    if args.stage == "compare":
        # scalar 点とは別に、指標定義も全体を照合する。
        for env in args.env:
            definitions = []
            for side in ("before", args.label):
                execution = json.loads((driver.OUT / f"{side}_{env}/execution.json").read_text(encoding="utf-8"))
                path = Path(execution["run"]) / "json/metrics.scalar.defs.json"
                definitions.append(json.loads(path.read_text(encoding="utf-8-sig")))
            if definitions[0] != definitions[1]:
                raise RuntimeError("Metric definitions differ: " + env)
            print("PASS definitions", env, flush=True)


if __name__ == "__main__":
    main()
