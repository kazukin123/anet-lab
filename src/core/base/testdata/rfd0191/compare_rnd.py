"""Compare RFD0191's five deterministic LunarLander configurations."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from runpy import run_path


REPO = Path(__file__).resolve().parents[5]
TESTDATA = REPO / "src/core/base/testdata"
COMMON = run_path(str(TESTDATA / "rfd0109/run_validation.py"))
METRICS = run_path(str(TESTDATA / "rfd0084/compare_metrics.py"))
RUNTIME = REPO / "apps/runner/bin/Release"
CASES = ("off", "cumulative", "ema", "feedback", "collect")
PROFILE = "run.@rfd0191_compare"
PREFIX = "DefaultDQNAgent.intrinsic."
TAIGA_KEYS = {
    "learn_ratio": "learn.intake.ratio",
    "learn_batch_size": "learn.intake.batch_size",
    "rnd.statistics_mode": "rnd.statistics.mode",
    "rnd.warmup_steps": "rnd.predictor.warmup_steps",
}


def settings(config, baseline_config, case, before):
    # 既定チェーンに比較用の条件を足し、両側で同じ seed と停止条件を使う。
    lunar = (config / "LunarLander.txt").read_text(encoding="utf-8-sig")
    common = (config / "common.txt").read_text(encoding="utf-8-sig")
    chain = re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)", lunar)[-1].strip()
    selected = chain + (" > run.@rnd" if case != "off" else "") + " > " + PROFILE
    overrides = [
        "app.$ = app.batchrun > P1", "app.show_error_dialog = false",
        "app.train_auto_start = true", "app.eval_panel.auto_start = false",
        "app.save_agent_on_close = true", "app.train_panel.fps = 0",
        "app.batchrun.train_panel.fps = 0", "metrics.image.interval = 0",
        "app.exp_exit_step = 100000", "app.batchrun.exp_exit_step = 100000",
        "backend.$ = backend.@deterministic", "backend.deterministic_warn_only = false",
        "backend.torch_num_threads = 1", "run.seed = 1", "agent.device = cuda",
    ]
    tags = sorted(set(re.findall(r"run\.eval(?:_schedule)?\.\[([^]]+)\]", lunar + common)))
    overrides += [f"run.eval_schedule.[{tag}].interval = 0" for tag in tags]
    # 旧設定の値を使い、新しい実行体へ渡す側だけキーと値の表記を移す。
    if case == "feedback":
        atari = (baseline_config / "Atari.txt").read_text(encoding="utf-8-sig")
        profile = re.findall(r"(?m)^run\.@rnd_taiga\s*:\s*DefaultDQNAgent\.intrinsic\.([^#\n]+)", atari)
        if len(profile) != 8:
            raise ValueError(f"Expected 8 Taiga settings, found {len(profile)}")
        for assignment in profile:
            key, value = (part.strip() for part in assignment.split("=", 1))
            if not before:
                if key == "rnd.multiplier_mode":
                    key, value = "rnd.statistics.variance_feedback.use_clipping", str(value == "clipped").lower()
                else:
                    key = TAIGA_KEYS.get(key, key)
            overrides.append(f"{PREFIX}{key} = {value}")
    if case == "ema":
        key = "rnd.error_norm.average_mode" if before else "rnd.error_norm.standardize.average_mode"
        overrides.append(f"{PREFIX}{key} = ema")
    if case == "collect":
        if before:
            overrides.append(f"{PREFIX}rnd.warmup_mode = collect_errors")
        else:
            overrides += [
                f"{PREFIX}rnd.error_norm.standardize.warmup_steps = 0",
                f"{PREFIX}rnd.predictor.warmup_steps = 12800",
            ]
    # 各実行体と同じ側の設定を読み込む自己記述のファイルを作る。
    lines = [f"$include <{config.as_posix()}/{name}>" for name in ("_main.txt", "LunarLander.txt")]
    lines += [PROFILE + " : " + value for value in overrides]
    return "\n".join(lines + ["run.$ = " + selected, ""])


def run_case(root, baseline, side, case, checkpoint=None):
    # 実行体・設定・出力先を対応づけ、指定時は旧 checkpoint を読み込む。
    before = side == "before"
    baseline_config = baseline / "config"
    config_dir = baseline_config if before else REPO / "apps/runner/config"
    executable = (baseline if before else RUNTIME) / "AnetRLRunner.exe"
    name = f"run_p191_{side}_{case}" + ("_load" if checkpoint else "")
    runs = root / "runs"
    folder = root / name
    config = folder.with_suffix(".txt")
    with config.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(settings(config_dir, baseline_config, case, before))
    environment = {key: value for key, value in os.environ.items() if not key.startswith("ANET_RFD")}
    environment["OMP_NUM_THREADS"] = "1"
    environment["PATH"] = str(RUNTIME) + os.pathsep + environment["PATH"]
    command = [str(executable), "--config", str(config), f"app.run_name={name}",
               f"app.runs_dir={runs.as_posix()}"]
    if checkpoint:
        command.append(f"DefaultDQNAgent.auto_load_file={checkpoint.as_posix()}")
    info = COMMON["execute"](command, folder, timeout=7200, env=environment)
    run = runs / name
    if info["exit_code"] or not (run / "agent_close.anet").is_file():
        raise RuntimeError(f"Run failed: {name}; see {folder / 'process.log'}")
    # 完走した Run を、比較可能な scalar 点の並びに正規化する。
    METRICS["normalize"](run, folder / "scalar.jsonl", include_noisy=True)
    summary = json.loads((folder / "scalar.summary.json").read_text(encoding="utf-8"))
    if not summary["point_count"]:
        raise RuntimeError(f"No scalar points: {name}")
    return {"run_dir": str(run), "config_sha256": COMMON["digest"](config), "metrics": summary}


def main():
    # 比較の片側を指定し、同じ置き場に実行結果を集める。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=("before", "after"))
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline.resolve()
    report = {}
    # 変更後は条件ごとに変更前の scalar と突き合わせる。
    for case in CASES:
        try:
            result = run_case(root, baseline, args.side, case)
            if args.side == "after":
                old_summary = root / f"run_p191_before_{case}/scalar.summary.json"
                old = json.loads(old_summary.read_text(encoding="utf-8"))
                result["exact_match"] = result["metrics"]["sha256"] == old["sha256"]
            report[case] = result
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            report[case] = {"error": str(error)}
            print(f"ERROR {case}: {error}", flush=True)
    # 既定の旧 checkpoint で、変更後の実行体が学習を再開できるか確認する。
    if args.side == "after":
        checkpoint = root / "runs/run_p191_before_cumulative/agent_close.anet"
        try:
            report["load"] = run_case(root, baseline, "after", "cumulative", checkpoint=checkpoint)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            report["load"] = {"error": str(error)}
            print(f"ERROR load: {error}", flush=True)
    COMMON["save"](root / f"{args.side}.json", report)
    return int(any("error" in value or not value.get("exact_match", True) for value in report.values()))


if __name__ == "__main__":
    raise SystemExit(main())
