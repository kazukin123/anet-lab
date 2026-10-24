"""PRD100: isolated GridMaze baseline, exact metrics comparison and startup logs."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / ".scratch/prd100"
RUNNER = REPO / "apps/runner/bin/Debug/AnetRLRunner.exe"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def execute(stage):
    destination = OUT / stage
    destination.mkdir(parents=True, exist_ok=False)
    binary_dir = destination / "bin"
    binary_dir.mkdir()
    source = OUT / "baseline-bin/AnetRLRunner.exe" if stage == "before" else RUNNER
    # 実行体とDLLを固定し、実行中のビルドや別Runから切り離す。
    for path in source.parent.iterdir():
        if path.suffix.lower() in (".exe", ".dll"):
            if path.suffix.lower() == ".dll":
                os.link(OUT / "baseline-bin" / path.name, binary_dir / path.name)
            else:
                shutil.copy2(path, binary_dir / path.name)
    config_root = OUT / "baseline-config"
    log_run = stage.startswith("log-")
    seed = 2 if stage == "log-retry" else 6545537233705677626
    overrides = [
        "app.$ = app.batchrun > P1", "app.show_error_dialog = false",
        "app.train_auto_start = true", "app.eval_panel.auto_start = false",
        "app.save_agent_on_close = false", "run.seed = 1",
        "app.runs_dir = " + (OUT / "runs").as_posix(),
        "backend.torch_num_threads = 1", "backend.$ = backend.@deterministic",
        "backend.deterministic_warn_only = false", "agent.device = cpu",
        "DefaultDQNAgent.skip.enabled = false", "DefaultDQNAgent.learner.use_amp = false",
        "metrics.scalar.$ = metrics.scalar.@baseline > M1 > metrics.scalar.@noisy",
        "DefaultDQNAgent.learner.replay_batch_size = 16",
        "DefaultDQNAgent.learner.replay_capacity = 4096",
        "DefaultDQNAgent.learner.replay_ratio = 1",
        "DefaultDQNAgent.learner.update_warmup_steps = " + ("100000" if log_run else "256"),
        "run.train.num_envs = " + ("256" if log_run else "8"),
        "GridMazeEnv.seed = " + str(seed),
        "app.exp_exit_step = " + ("256" if log_run else "8704"),
        "app.batchrun.exp_exit_step = " + ("256" if log_run else "8704"),
    ]
    if not log_run:
        overrides += ["run.eval_schedule.[eval].interval = 0", "run.eval_schedule.[eval_target].interval = 0"]
    profile = "run.@prd100_validation"
    lines = ["$include <" + (config_root / name).as_posix() + ">" for name in ("_main.txt", "GridMaze.txt")]
    lines += [profile + " : " + line for line in overrides]
    lines += ["run.$ = " + profile]
    config = destination / "run.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    name = "prd100_" + stage.replace("-", "_") + "_" + time.strftime("%Y%m%d%H%M%S")
    args = [str(binary_dir / source.name), "--config", str(config), "app.run_name=run_{t}_" + name]
    inputs = {str(path.relative_to(config_root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(config_root.rglob("*.txt"))}
    record = {"arguments": args, "binary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "config_inputs": inputs}
    save(destination / "execution.json", record)
    print("START", stage, flush=True)
    start = time.monotonic()
    with (destination / "launcher.log").open("w", encoding="utf-8") as stream:
        code = subprocess.run(args, cwd=REPO / "apps/runner", stdout=stream, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW).returncode
    runs = list((OUT / "runs").glob("run_*_" + name))
    record.update(exit_code=code, seconds=time.monotonic() - start, run=str(runs[0]) if len(runs) == 1 else None)
    save(destination / "execution.json", record)
    if code or len(runs) != 1:
        raise RuntimeError("Run failed; inspect " + str(destination))
    run = runs[0]
    if log_run:
        logs = list(run.glob("*.log"))
        text = "\n".join(path.read_text(encoding="utf-8-sig", errors="replace") for path in logs)
        notices = [line for line in text.splitlines() if "GridMaze maze regenerated:" in line]
        expected = 1 if stage == "log-retry" else 0
        save(destination / "log-check.json", {"expected": expected, "notices": notices, "logs": [str(p) for p in logs]})
        if len(notices) != expected:
            raise RuntimeError("Unexpected regeneration notice count: " + str(len(notices)))
        if notices and not all(key in notices[0] for key in ("seed=2", "retries=", "actual_seed=")):
            raise RuntimeError("Regeneration notice lacks required values")
    else:
        spec = importlib.util.spec_from_file_location("prd084_metrics", REPO / "core/anet-core/testdata/prd084/compare_metrics.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.normalize(str(run), str(destination / "normalized.jsonl"), include_noisy=True)
        # 無効Noisyの未成立値を、比較から除外した値と混同しない。
        dump = (run / "config/config_data.txt").read_text(encoding="utf-8-sig")
        configured = sorted(set(re.findall(r"^metrics\.scalar\.\[(47_agent_noisy/[^\]]+)\]", dump, re.MULTILINE)))
        summary = json.loads((destination / "normalized.summary.json").read_text(encoding="utf-8"))
        observed = sorted(tag for tag in summary["counts"] if tag.startswith("47_agent_noisy/"))
        save(destination / "noisy-coverage.json", {"configured": configured, "observed": observed})
        if len(configured) != 11:
            raise RuntimeError("Expected all 11 Noisy metric definitions")
    print("END", stage, code, round(record["seconds"], 1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "after", "compare", "log-default", "log-retry"))
    args = parser.parse_args()
    if args.stage == "compare":
        before = OUT / "before/normalized.jsonl"
        after = OUT / "after/normalized.jsonl"
        left, right = before.read_bytes(), after.read_bytes()
        result = {"exact_match": left == right, "before_sha256": hashlib.sha256(left).hexdigest(),
                  "after_sha256": hashlib.sha256(right).hexdigest(), "before_points": left.count(b"\n"),
                  "after_points": right.count(b"\n")}
        save(OUT / "comparison.json", result)
        print(json.dumps(result))
        if not result["exact_match"] or not result["before_points"]:
            raise RuntimeError("Learning metrics mismatch or empty baseline")
    else:
        execute(args.stage)


if __name__ == "__main__":
    main()
