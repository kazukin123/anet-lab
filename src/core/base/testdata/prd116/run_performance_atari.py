"""PRD116 Atari: CUDA Release timing of RND off / default / Taiga and separate Tracy captures."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

REPO = Path(__file__).resolve().parents[5]
# GridMaze版の保存・集計の部品を借り、Atariで変わる条件だけをここに置く。
_spec = importlib.util.spec_from_file_location("prd116_grid_timing", Path(__file__).with_name("run_performance.py"))
grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grid)

GAME = "montezuma_revenge"
LANES = 64
Q_WARMUP = 20000
WARM = 32768
CLOCK = grid.CLOCK
ARMS = ("off", "on", "taiga")
ORDER = ("off", "on", "taiga", "taiga", "on", "off") * 3
CYCLE = 6
PILOT_WINDOW = 8192
FASTEST_TARGET_SECONDS = 20
PROFILE_ARMS = ("on", "taiga")
PREFIX = "prd116_ap_"
PROFILE = "run.@prd116_aperf"
ARM_PROFILES = {"off": (), "on": ("run.@rnd",), "taiga": ("run.@rnd", "run.@rnd_taiga")}


def plain(value):
    return None if value is None else str(value).replace(",", "").strip()


def base_chain(root):
    source = (root / "Atari.txt").read_text(encoding="utf-8-sig")
    chains = re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)", source)
    if not chains:
        raise RuntimeError(f"Config {root / 'Atari.txt'}: run.$ actual=missing, expected one active chain")
    return chains[-1].strip()


def config(out, name, arm, window, eval_tags):
    # 既定のAtariのチェーンに群のprofileを重ね、測定用の設定キーを最後のprofileで確定する。
    root = out / "config"
    end = WARM + window + 2 * LANES
    enabled = "false" if arm == "off" else "true"
    overrides = ["app.$ = app.batchrun > P1", "app.show_error_dialog = false",
        "app.train_auto_start = true", "app.eval_panel.auto_start = false",
        "app.save_agent_on_close = false", f"app.runs_dir = {(out / 'runs').as_posix()}",
        f"app.run_name = run_{{t}}_{PREFIX}{name}", f"app.exp_exit_step = {end}",
        f"app.batchrun.exp_exit_step = {end}", "app.train_panel.fps = 0",
        "app.batchrun.train_panel.fps = 0", "app.batchrun.eval_panel.fps = 0", "run.seed = 1",
        "agent.device = cuda", f"run.train.num_envs = {LANES}", "metrics.image.interval = 0",
        f"E1.game = {GAME}", f"DefaultDQNAgent.learner.update_warmup_steps = {Q_WARMUP}",
        f"DefaultDQNAgent.intrinsic.enabled = {enabled}",
        f"metrics.scalar.[{CLOCK}] = elapse_hour @train $runner $exp_step interval:{LANES}"]
    overrides += [f"run.eval_schedule.[{tag}].interval = 0" for tag in eval_tags]
    chain = " > ".join([base_chain(root), *ARM_PROFILES[arm], PROFILE])
    return "\n".join([f"$include <{root.as_posix()}/_main.txt>", f"$include <{root.as_posix()}/Atari.txt>"]
        + [PROFILE + " : " + line for line in overrides] + ["run.$ = " + chain]) + "\n"


def resolve(out, arm, eval_tags):
    path = out / ("resolve_" + arm + ".txt")
    path.write_text(config(out, "resolve_" + arm, arm, PILOT_WINDOW, eval_tags), encoding="utf-8", newline="\n")
    target = out / (arm + ".resolved.json")
    env = os.environ.copy()
    env.update(ANET_PRD103_INPUT=str(path), ANET_PRD103_OUTPUT=str(target), OMP_NUM_THREADS="1")
    with (out / (arm + ".resolve.log")).open("w", encoding="utf-8") as log:
        subprocess.run([str(REPO / "out/test/Debug/anet-core-base-test.exe"), "[prd103-resolve]"],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, **grid.hidden())
    return json.loads(target.read_text(encoding="utf-8-sig"))


def eval_tags_of(resolved):
    tags = set()
    for key in resolved:
        match = re.match(r"^run\.eval(?:_schedule)?\.\[([^]]+)\]\.", key)
        if match:
            tags.add(match[1])
    return sorted(tags)


def check_resolved(arm, data, target):
    expected = {"agent.device": "cuda", "run.train.num_envs": str(LANES),
        "DefaultDQNAgent.learner.update_warmup_steps": str(Q_WARMUP),
        "DefaultDQNAgent.intrinsic.enabled": "false" if arm == "off" else "true"}
    if arm != "off":
        expected["DefaultDQNAgent.intrinsic.learn_batch_size"] = "1" if arm == "taiga" else "32"
        expected["DefaultDQNAgent.intrinsic.rnd.statistics_mode"] = "variance_feedback" if arm == "taiga" else "whiten"
    for key, value in expected.items():
        if plain(data.get(key)) != value:
            raise RuntimeError(f"Resolved config {target}, key={key}: actual={data.get(key)!r}, expected={value!r}")
    games = {key: value for key, value in data.items() if key.endswith(".game")}
    if not games or any(plain(value) != GAME for value in games.values()):
        raise RuntimeError(f"Resolved config {target}: games={games}, expected all {GAME}")
    active = {key: value for key, value in data.items()
              if key.startswith("run.eval_schedule.[") and key.endswith(".interval") and plain(value) != "0"}
    if active:
        raise RuntimeError(f"Resolved config {target}: active intervals={active}, expected all intervals=0")


def unexpected_differences(left, right):
    keys = sorted(set(left) | set(right))
    return {key: [left.get(key), right.get(key)] for key in keys if left.get(key) != right.get(key)
            and key != "app.run_name" and not key.startswith("DefaultDQNAgent.intrinsic.")
            and not key.startswith("metrics.scalar.[56_agent_in/")}


def prepare(out):
    # 実行体と設定を複製して測定の入力を固定する。Runnerは起動しないので、走行中のRunとは重ねられる。
    if out.exists():
        raise FileExistsError("Use a fresh output directory: " + str(out))
    out.mkdir(parents=True)
    shutil.copytree(REPO / "apps/runner/config", out / "config", ignore=shutil.ignore_patterns("*.log"))
    (out / "bin").mkdir()
    binary = REPO / "apps/runner/bin/Release/AnetRLRunner.exe"
    shutil.copy2(binary, out / "bin/AnetRLRunner.exe")
    for dll in binary.parent.glob("*.dll"):
        shutil.copy2(dll, out / "bin" / dll.name)
    first = resolve(out, "off", ())
    tags = eval_tags_of(first)
    resolved = {arm: resolve(out, arm, tags) for arm in ARMS}
    for arm, data in resolved.items():
        check_resolved(arm, data, out / (arm + ".resolved.json"))
    for left, right in (("off", "on"), ("on", "taiga")):
        unexpected = unexpected_differences(resolved[left], resolved[right])
        if unexpected:
            raise RuntimeError(f"{left}/{right} config differences in {out}: actual={unexpected}, "
                "expected only RND, group 56, and Run name differences")
    plan = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "binary_sha256": grid.digest(out / "bin/AnetRLRunner.exe"), "build": "Release", "device": "cuda",
        "game": GAME, "lanes": LANES, "q_update_warmup_steps": Q_WARMUP, "warm_experiences": WARM,
        "order": ORDER, "eval_tags_disabled": tags,
        "source_config_sha256": {p.name: grid.digest(p) for p in (out / "config").glob("*.txt")}}
    grid.save(out / "plan.json", plan)
    subprocess.run(["git", "diff", "--no-ext-diff", "--no-color", "--output=" + str(out / "source-diff.patch"),
        "HEAD", "--", "core", "src/apps/runner/src", "apps/runner/include"], cwd=REPO, check=True)
    print("PREPARED", str(out), "eval_tags=" + ",".join(tags), flush=True)


def clock_points(run):
    return {int(row["step"]): row["value"] for row in grid.rows(run) if row.get("tag") == CLOCK}


def execute(out, name, arm, window, tools=None):
    # 固定した実行体と設定で通常のRunnerを起動し、warm-up後の同じ経験区間の時計差を読む。
    grid.idle()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    folder = out / "cases" / name
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "run.txt"
    path.write_text(config(out, name, arm, window, plan["eval_tags_disabled"]), encoding="utf-8", newline="\n")
    binary = out / "bin/AnetRLRunner.exe"
    actual_sha = grid.digest(binary)
    if actual_sha != plan["binary_sha256"]:
        raise RuntimeError(f"Case {name}, binary {binary}: actual={actual_sha}, expected={plan['binary_sha256']}")
    command = [str(binary), "--config", str(path)]
    record = {"name": name, "arm": arm, "command": command, "binary_sha256": actual_sha,
        "config_sha256": grid.digest(path), "window_experiences": window, "profiled": tools is not None}
    grid.save(folder / "execution.json", record)
    print("START", name, flush=True)
    started = time.monotonic()
    capture = None
    capture_log = None
    pattern = "run_*_" + PREFIX + name
    with (folder / "launcher.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO / "apps/runner", stdout=log, stderr=subprocess.STDOUT,
                                   **grid.hidden())
        record["pid"] = process.pid
        grid.save(folder / "execution.json", record)
        if tools:
            # 別RunでTracyを接続する。通常の比較の実時間には採取の負荷を混ぜない。
            while process.poll() is None:
                found = list((out / "runs").glob(pattern))
                if found and grid.clock_progress(found[0]) >= WARM:
                    capture_log = (folder / "capture.log").open("w", encoding="utf-8")
                    capture = subprocess.Popen([str(tools / "tracy-capture.exe"), "-a", "127.0.0.1",
                        "-o", str(folder / "capture.tracy"), "-s", "8", "-m", "5"],
                        stdout=capture_log, stderr=subprocess.STDOUT, **grid.hidden())
                    record["capture_requested_after_exp_step"] = grid.clock_progress(found[0])
                    break
                time.sleep(0.5)
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print("RUNNING", name, "seconds=" + str(round(time.monotonic() - started)), flush=True)
    record.update(exit_code=process.returncode, process_seconds=time.monotonic() - started)
    found = list((out / "runs").glob(pattern))
    record["run"] = str(found[0]) if len(found) == 1 else None
    grid.save(folder / "execution.json", record)
    if process.returncode or len(found) != 1:
        raise RuntimeError(f"Case {name}, runs directory {out / 'runs'}: exit_code={process.returncode}, "
            f"expected_exit_code=0; run_count={len(found)}, expected_run_count=1; matches={found}")
    if tools:
        if capture is None:
            raise RuntimeError(f"Case {name}: Tracy capture actual=not started, expected connection after "
                f"exp_step >= {WARM}; last_clock_step={grid.clock_progress(found[0])}")
        capture.wait(timeout=120)
        capture_log.close()
        record["capture_exit_code"] = capture.returncode
        grid.save(folder / "execution.json", record)
        if capture.returncode or not (folder / "capture.tracy").is_file():
            raise RuntimeError(f"Case {name}, trace {folder / 'capture.tracy'}: capture_exit_code="
                f"{capture.returncode}, trace_exists={(folder / 'capture.tracy').is_file()}")
        for kind, options, zone_filter in (("zones", [], "RndNoveltyEstimator"),
                ("events", ["-u"], "RndNoveltyEstimator"), ("facade_events", ["-u"], "intrinsic_")):
            with (folder / (kind + ".csv")).open("w", encoding="utf-8", newline="") as stream:
                subprocess.run([str(tools / "tracy-csvexport.exe"), *options, "-f", zone_filter,
                    str(folder / "capture.tracy")], stdout=stream, stderr=subprocess.PIPE, check=True, **grid.hidden())
        print("CAPTURED", name, flush=True)
    points = clock_points(found[0])
    if WARM not in points or WARM + window not in points:
        missing = [step for step in (WARM, WARM + window) if step not in points]
        raise RuntimeError(f"Case {name}, Run {found[0]}: missing={missing}, expected boundaries="
            f"{[WARM, WARM + window]}, observed_points={len(points)}, last_steps={sorted(points)[-5:]}")
    seconds = (points[WARM + window] - points[WARM]) * 3600
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"Case {name}, Run {found[0]}: seconds={seconds}, expected finite and > 0")
    record.update(measurement_seconds=seconds, experiences_per_second=window / seconds)
    if arm != "off":
        counts = [row["value"] for row in grid.rows(found[0]) if row.get("tag") == "56_agent_in/32_learn_count"]
        if not counts or max(counts) <= 0:
            raise RuntimeError(f"Case {name}, Run {found[0]}, 56_agent_in/32_learn_count: points={len(counts)}, "
                f"actual={max(counts) if counts else 'missing'}, expected > 0")
        record["learn_count_max"] = max(counts)
    grid.save(folder / "execution.json", record)
    print("END", name, "measurement_seconds=" + str(round(seconds, 3)),
          "exp_per_second=" + str(round(window / seconds, 1)), flush=True)
    return record


def choose_window(pilots):
    # いちばん速い群でも目標秒数を測れる区間をlane単位に切り上げる。遅い群はその分だけ長くなる。
    fastest = min(item["measurement_seconds"] for item in pilots)
    return max(2 * PILOT_WINDOW, math.ceil(PILOT_WINDOW * FASTEST_TARGET_SECONDS / fastest / LANES) * LANES)


def summarize(results):
    if len(results) != len(ORDER) or any("error" in item or item.get("profiled") for item in results):
        raise RuntimeError(f"Measurements: count actual={len(results)}, expected={len(ORDER)}; invalid="
            f"{[item for item in results if 'error' in item or item.get('profiled')]}")
    if [item["arm"] for item in results] != list(ORDER):
        raise RuntimeError(f"Measurements: order actual={[item['arm'] for item in results]}, expected={list(ORDER)}")
    if len({item["window_experiences"] for item in results}) != 1 or len({item["binary_sha256"] for item in results}) != 1:
        raise RuntimeError("Measurements: expected exactly one experience budget and one executable SHA256")
    window = results[0]["window_experiences"]
    medians = {arm: statistics.median(item["measurement_seconds"] for item in results if item["arm"] == arm)
               for arm in ARMS}
    cycles = []
    for start in range(0, len(results), CYCLE):
        part = results[start:start + CYCLE]
        cycle = {arm: statistics.median(item["measurement_seconds"] for item in part if item["arm"] == arm)
                 for arm in ARMS}
        cycles.append({"on_off": cycle["on"] / cycle["off"], "taiga_off": cycle["taiga"] / cycle["off"]})
    return {"medians_seconds": medians, "ratios": {"on_off": medians["on"] / medians["off"],
        "taiga_off": medians["taiga"] / medians["off"]},
        "experiences_per_second": {arm: window / seconds for arm, seconds in medians.items()},
        "ranges_seconds": {arm: [min(item["measurement_seconds"] for item in results if item["arm"] == arm),
            max(item["measurement_seconds"] for item in results if item["arm"] == arm)] for arm in ARMS},
        "cycle_ratios": cycles, "window_experiences": window,
        "acceptance": "No speed threshold; record timing and Tracy breakdown"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "measure", "profile", "summarize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracy-tools", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.phase == "prepare":
        prepare(out)
    elif args.phase == "pilot":
        pilots = [execute(out, "pilot_" + arm, arm, PILOT_WINDOW) for arm in ARMS]
        window = choose_window(pilots)
        grid.save(out / "pilot.json", {"runs": pilots, "selected_window": window,
            "predicted_run_seconds": {item["arm"]: item["process_seconds"] + (window / PILOT_WINDOW - 1)
                * item["measurement_seconds"] for item in pilots}})
        print("PILOT selected_window=" + str(window), flush=True)
    elif args.phase == "measure":
        if (out / "measurements.json").exists() or (out / "cases" / "r01_off").exists():
            raise FileExistsError(f"Recorded measurements exist in {out}; expected a fresh output directory")
        window = json.loads((out / "pilot.json").read_text(encoding="utf-8"))["selected_window"]
        results = []
        for index, arm in enumerate(ORDER):
            name = f"r{index + 1:02d}_{arm}"
            try:
                results.append(execute(out, name, arm, window))
            except Exception as error:
                results.append({"name": name, "arm": arm, "error": str(error)})
                print("FAILED", name, str(error), flush=True)
            grid.save(out / "measurements.json", results)
        if any("error" in item for item in results):
            raise RuntimeError(f"Performance measurements {out}: failed={[item for item in results if 'error' in item]}")
    elif args.phase == "profile":
        if args.tracy_tools is None:
            parser.error("profile requires --tracy-tools")
        window = json.loads((out / "pilot.json").read_text(encoding="utf-8"))["selected_window"]
        for arm in PROFILE_ARMS:
            execute(out, "profile_" + arm, arm, window, args.tracy_tools.resolve())
    else:
        summary = summarize(json.loads((out / "measurements.json").read_text(encoding="utf-8")))
        grid.LANES = LANES
        summary["tracy"] = {}
        for arm in PROFILE_ARMS:
            folder = out / "cases" / ("profile_" + arm)
            if (folder / "events.csv").exists():
                try:
                    summary["tracy"][arm] = grid.profile_summary(folder)
                except RuntimeError as error:
                    summary["tracy"][arm] = {"error": str(error)}
        grid.save(out / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "tracy"}), flush=True)


if __name__ == "__main__":
    main()
