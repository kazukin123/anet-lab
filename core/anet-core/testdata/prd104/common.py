"""PRD104検証の公開resolver・実行記録・指標読み取りをまとめる。"""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
OUT = REPO / ".scratch/prd104"


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT.parent / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(config, destination, name="resolved", binary=None):
    # 現用のConfigManagerテストを使い、独自resolverを持たない。
    target = destination / (name + ".json")
    environment = dict(os.environ, ANET_PRD103_INPUT=str(config), ANET_PRD103_OUTPUT=str(target))
    with (destination / (name + ".log")).open("w", encoding="utf-8") as log:
        executable = binary or REPO / "core/anet-core/bin/Debug/anet-core-test.exe"
        subprocess.run([str(executable), "[prd103-resolve]"],
                       cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    return json.loads(target.read_text(encoding="utf-8"))


def write_config(destination, env, chain, leaves, snapshot=False):
    base = OUT / "baseline-config" if snapshot else OUT / "current-config"
    if not base.exists():
        base = REPO / "apps/runner/config"
    config = destination / "run.txt"
    arms = base / "prd104-arms.txt"
    if not arms.exists():
        arms = ROOT / "arms.txt"
    lines = [f"$include <{base.as_posix()}/_main.txt>", f"$include <{base.as_posix()}/{env}.txt>",
             f"$include <{arms.as_posix()}>"]
    lines += [f"run.@prd104 : {key} = {value}" for key, value in leaves.items()]
    lines += ["run.$ = " + chain + " > run.@prd104"]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return config


def execute(destination, config, name, build="Release", binary=None):
    # コマンド、実行体、入力を固定し、失敗したRunも証拠を残す。
    binary = binary or REPO / f"apps/runner/bin/{build}/AnetRLRunner.exe"
    runs = OUT / "runs"
    arguments = [str(binary), "--config", str(config), f"app.runs_dir={runs.as_posix()}",
                 "app.run_name=run_{t}_" + name]
    record = {"arguments": arguments, "binary_sha256": digest(binary), "config_sha256": digest(config),
              "name": name, "build": build}
    save(destination / "execution.json", record)
    print("START", name, flush=True)
    started = time.monotonic()
    with (destination / "launcher.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(arguments, cwd=REPO / "apps/runner", stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    matches = list(runs.glob("run_*_" + name))
    record.update(exit_code=process.returncode, seconds=time.monotonic() - started,
                  run=str(matches[0]) if len(matches) == 1 else None)
    save(destination / "execution.json", record)
    print("END", name, process.returncode, round(record["seconds"], 1), flush=True)
    if process.returncode or len(matches) != 1:
        raise RuntimeError(f"Run failed: {name}; see {destination}")
    return Path(record["run"])


def rows(run):
    path = run / "metrics.jsonl"
    opener = open
    if not path.exists():
        path = run / "metrics.jsonl.gz"
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line in stream:
            yield json.loads(line)


def batch_leaves(budget, seed=1):
    return {"app.$": "app.batchrun > P1", "app.show_error_dialog": "false",
            "app.train_auto_start": "true", "app.eval_panel.auto_start": "false",
            "app.save_agent_on_close": "false", "app.exp_exit_step": budget,
            "app.batchrun.exp_exit_step": budget, "run.seed": seed}
