"""Measure the three RFD0191 learning schedules in the same Atari configuration."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from runpy import run_path

from compare_rnd import COMMON, REPO, RUNTIME, TESTDATA


REFERENCE = run_path(str(TESTDATA / "rfd0130/run_performance.py"))
CONFIG = REPO / "apps/runner/config"
RUNNER = RUNTIME / "AnetRLRunner.exe"
LANES = 128
WINDOW = LANES * 128
START = 2 * WINDOW
END = START + WINDOW
CLOCK = "90_perf/99_rfd0191_time"
PROFILE = "run.@rfd0191_measure"
MODES = ("intake", "batch", "spread")


def prepare(root, case, mode, profiling):
    config = root / f"{case}.txt"
    end = END + 2 * LANES + (WINDOW if profiling else 0)
    settings = [
        *REFERENCE["BASE_ARGS"],
        "DefaultDQNAgent.learner.update_warmup_steps = 20000",
        # 測定区間の全経験が入る大きさで確保し、比較の各群で同じ容量を使う。
        "DefaultDQNAgent.learner.replay_capacity = 131072",
        "DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps = 12800",
        "run.train.num_envs = 128", "agent.device = cuda",
        "app.$ = app.batchrun > P1", "app.show_error_dialog = false",
        "app.train_auto_start = true", "app.eval_panel.auto_start = false",
        "app.save_agent_on_close = false", "app.train_panel.fps = 0",
        "app.batchrun.train_panel.fps = 0", "metrics.image.interval = 0",
        f"app.exp_exit_step = {end}", f"app.batchrun.exp_exit_step = {end}",
        f"metrics.scalar.[{CLOCK}] = elapse_hour @train $runner $exp_step interval:{LANES}",
    ]
    settings += [f"run.eval_schedule.[{tag}].interval = 0" for tag in REFERENCE["EVAL_TAGS"]]
    if mode != "intake":
        settings.append("DefaultDQNAgent.intrinsic.learn.mode = rollout")
        settings.append("DefaultDQNAgent.intrinsic.learn.rollout.use_spread = " + str(mode == "spread").lower())
    includes = [f"$include <{(CONFIG / name).as_posix()}>" for name in ("_main.txt", "Atari.txt")]
    chain = [f"run.$ = {REFERENCE['MAIN_CHAIN']} > {PROFILE}"]
    with config.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(includes + [PROFILE + " : " + item for item in settings] + chain + [""]))
    runs = root / "runs"
    return [str(RUNNER), "--config", str(config), f"app.run_name={case}", f"app.runs_dir={runs.as_posix()}"], runs / case


def environment():
    env = {key: value for key, value in os.environ.items() if not key.startswith("ANET_RFD")}
    env.update(OMP_NUM_THREADS="1", TRACY_PORT="18091")
    return env


def measure(root, name, mode):
    # 条件ごとに独立した Run を完走させる。
    command, run = prepare(root, name, mode, False)
    info = COMMON["execute"](command, root / name, timeout=3600, env=environment())
    if info["exit_code"]:
        raise RuntimeError(f"Runner failed: {name}")
    # 起動や終了の時間を外し、経験数の同じ両端で経過時刻を引く。
    points = {}
    with (run / "metrics.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("tag") == CLOCK:
                points[int(row["step"])] = row["value"]
    if START not in points or END not in points or points[END] <= points[START]:
        raise RuntimeError(f"Missing measurement endpoints in {run}")
    return (points[END] - points[START]) * 3600


def profile(root, tools):
    # 一括学習の Run を用意し、実時間の比較とは別に採取する。
    name = "run_p191_tmp_profile"
    command, run = prepare(root, name, "batch", True)
    capture = tools / "tracy-capture.exe"
    exporter = tools / "tracy-csvexport.exe"
    if not (capture.is_file() and exporter.is_file()):
        raise OSError(f"Missing Tracy tools in {tools}")
    target = root / "rollout.tracy"
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW}
    with (root / "profile-runner.log").open("w", encoding="utf-8") as log:
        runner = subprocess.Popen(command, cwd=REPO / "apps/runner", env=environment(),
                                  stdout=log, stderr=subprocess.STDOUT, **hidden)
        try:
            # Run が起動したら、学習が始まる前に Tracy を接続する。
            deadline = time.monotonic() + 180
            while not (run / "metrics.jsonl").is_file() and runner.poll() is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Profile Run did not create a metrics file")
                time.sleep(0.25)
            with (root / "profile-capture.log").open("w", encoding="utf-8") as capture_log:
                subprocess.run([str(capture), "-a", "127.0.0.1", "-p", "18091",
                                "-o", str(target), "-s", "60", "-m", "5"],
                               stdout=capture_log, stderr=subprocess.STDOUT,
                               timeout=120, check=True, **hidden)
            if runner.wait(timeout=1800):
                raise RuntimeError("Profile Run failed")
        finally:
            if runner.poll() is None:
                runner.terminate()
                runner.wait(timeout=30)
    # 採取が終わってから、一括学習を完了した区間だけを取り出す。
    table = subprocess.check_output([str(exporter), "-f", "intrinsic_rollout_batch", str(target)],
                                    encoding="utf-8", **hidden)
    (root / "rollout-zones.csv").write_text(table, encoding="utf-8")
    zones = list(csv.DictReader(io.StringIO(table)))
    if not zones or not all(int(zone["counts"]) > 0 for zone in zones):
        raise RuntimeError("No completed rollout learning zone in the capture")
    print(table, end="")


def main():
    # 比較用の実時間と、別 Run の一括学習時間を指定に応じて採取する。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("measure", "profile"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tracy-tools", type=Path, default=REPO / ".scratch/prd116/tracy-0.13.1/tools")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.mode == "profile":
        profile(root, args.tracy_tools.resolve())
        return
    # 各方式が先頭になるよう順を回し、全試行が揃ってから中央値を保存する。
    samples = {mode: [] for mode in MODES}
    failures = {}
    for order in range(len(MODES)):
        for mode in MODES[order:] + MODES[:order]:
            name = f"run_p191_tmp_{mode}_{order + 1}"
            try:
                seconds = measure(root, name, mode)
                samples[mode].append(seconds)
                print(f"MEASURE {mode} {seconds:.3f} s", flush=True)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                failures[name] = str(error)
                print(f"ERROR {name}: {error}", flush=True)
    if failures:
        COMMON["save"](root / "incomplete.json", {"samples_seconds": samples, "errors": failures})
        raise SystemExit(1)
    report = {mode: {"samples_seconds": values, "median_seconds": statistics.median(values)}
              for mode, values in samples.items()}
    COMMON["save"](root / "medians.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
