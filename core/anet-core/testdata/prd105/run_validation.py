"""Run PRD105's frozen/current four-environment scalar comparison sequentially."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[4]
ENVS = ("GridMaze", "LunarLander", "Atari", "DropMerge")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def config_digest(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.txt")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-ui", action="store_true", help="Prepare paused real-env GUI inputs without starting Runs")
    args = parser.parse_args()
    baseline, output = args.baseline.resolve(), args.output.resolve()
    before_binary = baseline / "bin/AnetRLRunner.exe"
    current_binary = REPO / "apps/runner/bin/Debug/AnetRLRunner.exe"
    if not before_binary.is_file() or not (baseline / "config/_main.txt").is_file():
        parser.error("Baseline must contain bin/AnetRLRunner.exe and config/_main.txt")
    output.mkdir(parents=True, exist_ok=False)

    # 過去のドライバは変更せず、段0の条件生成だけを呼ぶ。今回の入力と結果を隔離する。
    driver = load_module("prd105_stage0", REPO / "core/anet-core/testdata/prd098/run_validation.py")
    comparator = load_module("prd105_compare", REPO / "core/anet-core/testdata/prd084/compare_metrics.py")
    driver.OUT = output
    snapshots = REPO / ".scratch/prd105/config-before-20261030-0138"
    shutil.copytree(snapshots, output / "resolved-before")
    shutil.copytree(baseline / "config", output / "baseline-config")
    after_config = output / "after-config"
    shutil.copytree(REPO / "apps/runner/config", after_config)
    if args.prepare_ui:
        # 実wxテストは環境ごとに別process。学習とtimer stepを止めて手動stepだけを通す。
        for env in ("Atari", "LunarLander"):
            lines = driver.settings(env)
            text = "\n".join(lines).replace((REPO / "apps/runner/config").as_posix(), after_config.as_posix())
            text = text.replace("prd098", "prd105").replace("app.batchrun > P1", "app.online > P1")
            overrides = ("app.train_auto_start = false", "app.eval_panel.auto_start = false",
                         "app.train_panel.fps = 0", "app.eval_panel.fps = 0", "run.train.num_envs = 1",
                         "app.eval_trace_panel.columns.[1] = $runner exp_step label:step",
                         "app.eval_trace_panel.columns.[2] = $runner forced label:forced",
                         "app.eval_trace_panel.columns.[3] = $action_info action",
                         "app.run_name = run_{t}_prd105_ui_" + env.lower())
            text += "\n" + "\n".join("run.@prd105_validation : " + value for value in overrides) + "\n"
            (output / (env + "-ui.txt")).write_text(text, encoding="utf-8", newline="\n")
        return
    reports = {}
    for env in ENVS:
        normalized = []
        for side, before in (("before", True), ("after", False)):
            folder = output / (side + "_" + env)
            folder.mkdir()
            root = baseline / "config" if before else after_config
            generated_root = output / "baseline-config" if before else REPO / "apps/runner/config"
            if not before and config_digest(generated_root) != config_digest(after_config):
                raise RuntimeError("Current config changed after snapshot; use a new output directory")
            lines = driver.settings(env, before=before)
            text = "\n".join(lines).replace(generated_root.as_posix(), root.as_posix())
            text = text.replace("prd098", "prd105") + "\n"
            config = folder / "run.txt"
            config.write_text(text, encoding="utf-8", newline="\n")
            binary = before_binary if before else current_binary
            suffix = "prd105_" + side + "_" + env.lower()
            command = [str(binary), "--config", str(config), "app.run_name=run_{t}_" + suffix]
            info = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    "config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "config_inputs_sha256": config_digest(root), "arguments": command}
            save(folder / "execution.json", info)
            print("START", side, env, flush=True)
            start = time.monotonic()
            with (folder / "launcher.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=REPO / "apps/runner", stdout=log,
                                           stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                info["pid"] = process.pid
                save(folder / "execution.json", info)
                code = process.wait()
            runs = list((output / "runs").glob("run_*_" + suffix))
            info.update(exit_code=code, seconds=time.monotonic() - start,
                        run=str(runs[0]) if len(runs) == 1 else None)
            save(folder / "execution.json", info)
            print("END", side, env, code, round(info["seconds"], 2), flush=True)
            if code or len(runs) != 1:
                raise RuntimeError("Run failed: " + str(folder))
            # Noisyを含め、時間・資源の3prefixだけを既存の正規化器で除く。
            target = folder / "normalized.jsonl"
            comparator.normalize(runs[0], target, include_noisy=True)
            normalized.append(target.read_bytes())
        report = {"exact_match": normalized[0] == normalized[1],
                  "before_points": normalized[0].count(b"\n"), "after_points": normalized[1].count(b"\n"),
                  "before_sha256": hashlib.sha256(normalized[0]).hexdigest(),
                  "after_sha256": hashlib.sha256(normalized[1]).hexdigest()}
        save(output / (env + "-comparison.json"), report)
        reports[env] = report
        save(output / "comparison.json", reports)
        print("COMPARE", env, report["exact_match"], report["after_points"], flush=True)
    if not all(report["exact_match"] for report in reports.values()):
        raise RuntimeError("Learning metric mismatch; inspect per-environment comparison files")


if __name__ == "__main__":
    main()
