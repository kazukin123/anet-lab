"""PRD109: isolated suites, process failures, and four-environment comparison."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[4]
BIN = REPO / "apps/runner/bin/Debug"
ENVS = ("GridMaze", "LunarLander", "Atari", "DropMerge")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(command, folder, timeout=3600, cwd=None, env=None):
    # 新しい出力先だけを受け付け、前の検証結果を上書きしない。
    folder.mkdir(parents=True, exist_ok=False)
    info = {"arguments": [str(value) for value in command], "binary_sha256": digest(Path(command[0]))}
    start = time.monotonic()
    print("START", folder.name, flush=True)
    with (folder / "process.log").open("w", encoding="utf-8") as stream:
        try:
            # UIテスト以外も別processで起動し、出力と終了値を対応づける。
            result = subprocess.run(command, cwd=cwd or REPO / "apps/runner", stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout,
                                    creationflags=subprocess.CREATE_NO_WINDOW, env=env)
            info["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            info["timed_out"] = True
            raise
        finally:
            # 失敗やtimeoutでも実時間と起動条件を保存する。
            info["seconds"] = time.monotonic() - start
            save(folder / "execution.json", info)
    print("END", folder.name, info["exit_code"], round(info["seconds"], 2), flush=True)
    return info


def snapshot(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "bin").mkdir()
    for path in BIN.iterdir():
        if path.suffix.lower() in (".dll", ".exe"):
            shutil.copy2(path, output / "bin" / path.name)
    shutil.copytree(REPO / "apps/runner/config", output / "config")
    save(output / "hashes.json", {str(path.relative_to(output)): digest(path)
                                 for path in output.rglob("*") if path.is_file()})


def failures(xml):
    return sorted({case.get("classname", "") + "::" + case.get("name", "")
                   for case in ET.parse(xml).iter("testcase")
                   if case.find("failure") is not None or case.find("error") is not None})


def suites(output, baseline=None, baseline_gui=None):
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    # coreの相対fixtureを正しく解決するため、既定suiteはrepo rootで実行する。
    for name, binary in (("core", REPO / "core/anet-core/bin/Debug/anet-core-test.exe"),
                         ("gui", BIN / "AnetRLRunner-gui-test.exe")):
        xml = output / (name + ".xml")
        info = execute([str(binary), "--reporter", "junit", "--out", str(xml)], output / name, cwd=REPO)
        reports[name] = {**info, "failures": failures(xml)}
    save(output / "suites.json", reports)
    if baseline:
        # 既知の失敗を成功へ読み替えず、失敗したcaseの集合そのものを前後比較する。
        before = json.loads((baseline / "suites.json").read_text(encoding="utf-8"))
        if baseline_gui:
            before["gui"]["failures"] = failures(baseline_gui)
        if any(before[name]["failures"] != reports[name]["failures"] for name in reports):
            raise RuntimeError("Default suite failure set changed")


def compare(baseline, output):
    # 条件生成・正規化の既存実装を使い、凍結した設定を両方の実行体へ渡す。
    output.mkdir(parents=True, exist_ok=False)
    driver = load("prd109_stage0", REPO / "core/anet-core/testdata/prd098/run_validation.py")
    comparator = load("prd109_compare", REPO / "core/anet-core/testdata/prd084/compare_metrics.py")
    driver.OUT = output
    shutil.copytree(baseline / "config", output / "baseline-config")
    # 選択済みのeval tagをConfigManagerから採る。未選択profileの定義を混ぜない。
    info = execute([str(BIN / "AnetRLRunner-gui-test.exe"), "[prd109_config]"], output / "capture",
                   cwd=REPO, env={**os.environ, "PRD109_CONFIG_OUTPUT": str(output)})
    if info["exit_code"]:
        raise RuntimeError("Frozen configuration capture failed")
    reports = {}
    for env in ENVS:
        # 環境ごとに変更前→変更後を逐次実行し、同じ入力の全scalarを比較する。
        normalized = []
        text = "\n".join(driver.settings(env, before=True)).replace("prd098", "prd109") + "\n"
        for side, binary in (("before", baseline / "bin/AnetRLRunner.exe"),
                             ("after", BIN / "AnetRLRunner.exe")):
            config = output / (side + "_" + env + ".txt")
            config.write_text(text, encoding="utf-8", newline="\n")
            suffix = "prd109_" + side + "_" + env.lower()
            info = execute([str(binary), "--config", str(config), "app.run_name=run_{t}_" + suffix],
                           output / (side + "_" + env))
            runs = list((output / "runs").glob("run_*_" + suffix))
            if info["exit_code"] or len(runs) != 1:
                raise RuntimeError("Run failed: " + side + " " + env)
            target = output / (side + "_" + env) / "normalized.jsonl"
            # 時間・資源の3prefixだけを除き、Noisyの診断も比較へ含める。
            comparator.normalize(runs[0], target, include_noisy=True)
            normalized.append(target.read_bytes())
            save(target.parent / "inputs.json", {"run": str(runs[0]), "config_sha256": digest(config)})
        reports[env] = {"exact_match": normalized[0] == normalized[1],
                        "before_points": normalized[0].count(b"\n"), "after_points": normalized[1].count(b"\n"),
                        "before_sha256": hashlib.sha256(normalized[0]).hexdigest(),
                        "after_sha256": hashlib.sha256(normalized[1]).hexdigest()}
        save(output / "comparison.json", reports)
        print("COMPARE", env, reports[env], flush=True)
    if not all(item["exact_match"] for item in reports.values()):
        raise RuntimeError("Learning scalar mismatch")


def fatal(output):
    output.mkdir(parents=True, exist_ok=False)
    config = output / "fatal.txt"
    config.write_text("env.class_id = CartPoleEnv\nenv.worker_type = 1\nenv.device = cpu\n"
                      "agent.class_id = DefaultDQNAgent\nagent.device = cpu\nrun.eval_device = cpu\n"
                      "run.eval_schedule.[nosuch].interval = 1\napp.show_error_dialog = false\n"
                      "app.save_agent_on_close = false\napp.run_name = prd109_fatal\napp.runs_dir = "
                      + (output / "runs").as_posix() + "\n", encoding="utf-8", newline="\n")
    info = execute([str(BIN / "AnetRLRunner.exe"), "--config", str(config)], output / "fatal")
    if info["exit_code"] != 1:
        raise RuntimeError("Expected fatal exit code 1")
    run = output / "runs/prd109_fatal"
    diagnostic = "Unknown run.eval_schedule tag 'nosuch'"
    run_log = run / "prd109_fatal.log"
    if diagnostic not in run_log.read_text(encoding="utf-8-sig"):
        raise RuntimeError("Missing fatal diagnostic: " + str(run_log))
    stderr_log = run / "stderr.log"
    if diagnostic in stderr_log.read_text(encoding="utf-8-sig"):
        raise RuntimeError("Unexpected fatal diagnostic: " + str(stderr_log))


def gui(output, tags):
    output.mkdir(parents=True, exist_ok=False)
    if not tags:
        source = (REPO / "apps/runner/src/RunnerApp_test.cpp").read_text(encoding="utf-8-sig")
        tags = re.findall(r'\[\.\]\[prd109_startup\]\[([^\]]+)\]', source)
        tags += ["eval_steps_menu", "prd105_ui_keys"]
    for tag in tags:
        xml = output / (tag + ".xml")
        info = execute([str(BIN / "AnetRLRunner-gui-test.exe"), "[" + tag + "]",
                        "--reporter", "junit", "--out", str(xml)], output / tag, timeout=60, cwd=REPO)
        if info["exit_code"]:
            raise RuntimeError("GUI test failed: " + tag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "suites", "compare", "fatal", "gui"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-gui", type=Path)
    parser.add_argument("--tag", action="append")
    args = parser.parse_args()
    output = args.output.resolve()
    baseline = args.baseline.resolve() if args.baseline else None
    if args.mode == "snapshot":
        snapshot(output)
    elif args.mode == "suites":
        suites(output, baseline, args.baseline_gui)
    elif args.mode == "fatal":
        fatal(output)
    elif args.mode == "gui":
        gui(output, args.tag)
    else:
        if not baseline:
            parser.error("compare requires --baseline")
        compare(baseline, output)


if __name__ == "__main__":
    main()
