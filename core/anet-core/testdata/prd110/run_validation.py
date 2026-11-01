"""PRD110: process-isolated GUI cases, suite failure sets, and invalid mode."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True


REPO = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "prd110_validation_common", REPO / "core/anet-core/testdata/prd109/run_validation.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def gui(output, tags):
    # RunnerApp は再初期化せず、case ごとに独立した process で検査する。
    output.mkdir(parents=True, exist_ok=False)
    if not tags:
        source = (REPO / "apps/runner/src/RunnerFrame_test.cpp").read_text(encoding="utf-8-sig")
        tags = re.findall(r'\[prd110_main_pane\]\[([^\]]+)\]', source)
        tags += ["eval_steps_menu", "prd105_ui_keys", "prd109_ui"]
    for tag in tags:
        result = common.execute(
            [str(common.BIN / "AnetRLRunner-gui-test.exe"), "[" + tag + "]",
             "--reporter", "junit", "--out", str(output / (tag + ".xml"))],
            output / tag, timeout=60, cwd=REPO)
        if result["exit_code"]:
            raise RuntimeError("GUI test failed: " + tag)
        if not list(common.ET.parse(output / (tag + ".xml")).iter("testcase")):
            raise RuntimeError("GUI tag matched no cases: " + tag)


def fatal(output):
    # Config の検証は Run 作成前なので、製品の非modal診断をprocess出力で読む。
    output.mkdir(parents=True, exist_ok=False)
    config = output / "invalid.txt"
    config.write_text(
        "env.class_id = CartPoleEnv\nenv.worker_type = 1\nenv.device = cpu\n"
        "agent.class_id = DefaultDQNAgent\nagent.device = cpu\nrun.eval_device = cpu\n"
        "app.main_view = center\napp.show_error_dialog = false\n"
        "app.save_agent_on_close = false\napp.runs_dir = "
        + (output / "runs").as_posix() + "\n", encoding="utf-8", newline="\n")
    result = common.execute([str(common.BIN / "AnetRLRunner.exe"), "--config", str(config)],
                            output / "process", timeout=60, cwd=REPO)
    diagnostic = "Invalid app.main_view: value=center (expected: train, eval, both)"
    log = (output / "process/process.log").read_text(encoding="utf-8-sig", errors="replace")
    if result["exit_code"] == 0 or diagnostic not in log:
        raise RuntimeError("Expected nonzero exit and diagnostic: " + diagnostic)


def config_check(output, baseline):
    # 全値の比較は移行時の検証資材に限定し、恒久テストには置かない。
    if baseline is None:
        raise ValueError("config requires --common-baseline")
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"before": baseline.resolve(), "after": REPO / "apps/runner/config/common.txt"}
    common.save(output / "inputs.json", {
        label: {"path": str(path), "sha256": common.digest(path)}
        for label, path in inputs.items()})
    for label, path in inputs.items():
        # 既存の任意入力用採取器は ConfigManager の解決結果を出力するだけ。
        result = common.execute(
            [str(REPO / "core/anet-core/bin/Debug/anet-core-test.exe"), "[prd103-resolve]",
             "--reporter", "junit", "--out", str(output / (label + ".xml"))],
            output / label, cwd=REPO,
            env={**os.environ, "ANET_PRD103_INPUT": str(path),
                 "ANET_PRD103_OUTPUT": str(output / (label + ".json"))})
        if result["exit_code"]:
            raise RuntimeError("Configuration capture failed: " + label)
    before = json.loads((output / "before.json").read_text(encoding="utf-8"))
    after = json.loads((output / "after.json").read_text(encoding="utf-8"))
    report = {
        "added": {key: after[key] for key in after.keys() - before.keys()},
        "removed": {key: before[key] for key in before.keys() - after.keys()},
        "changed": {key: {"before": before[key], "after": after[key]}
                    for key in before.keys() & after.keys() if before[key] != after[key]}}
    common.save(output / "comparison.json", report)
    if report != {"added": {"app.main_view": "eval"}, "removed": {}, "changed": {}}:
        raise RuntimeError("Common configuration delta differs from app.main_view=eval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("suites", "gui", "fatal", "config"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-gui", type=Path)
    parser.add_argument("--tag", action="append")
    parser.add_argument("--common-baseline", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == "suites":
        common.suites(output, args.baseline.resolve() if args.baseline else None,
                      args.baseline_gui.resolve() if args.baseline_gui else None)
    elif args.mode == "gui":
        gui(output, args.tag)
    elif args.mode == "fatal":
        fatal(output)
    else:
        config_check(output, args.common_baseline)


if __name__ == "__main__":
    main()
