"""PRD116 GridMaze: isolated CPU Release timing and separate Tracy capture."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

REPO = Path(__file__).resolve().parents[4]
WARM = 16384
LANES = 256
CLOCK = "90_perf/98_prd116_time"
ORDER = ("off", "on", "on", "off") * 3


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hidden():
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    return {"startupinfo": startup, "creationflags": subprocess.CREATE_NO_WINDOW}


def idle():
    result = subprocess.run(["powershell", "-NoProfile", "-Command",
        "@(Get-CimInstance Win32_Process -Filter \"Name LIKE 'AnetRLRunner%.exe'\").Count"],
        capture_output=True, text=True, check=True, **hidden())
    count = int(result.stdout.strip())
    if count:
        raise RuntimeError(f"Runner process count: actual={count}, expected=0; measurement was not started")


def config(out, name, arm, window):
    # 共通のQ・環境設定を複製し、RNDの選択と測定用の設定キーだけを末尾で確定する。
    root = out / "config"
    end = WARM + window + 2 * LANES
    overrides = ["app.$ = app.batchrun > P1", "app.show_error_dialog = false",
        "app.train_auto_start = true", "app.eval_panel.auto_start = false",
        "app.save_agent_on_close = false", f"app.runs_dir = {(out / 'runs').as_posix()}",
        f"app.run_name = run_{{t}}_prd116_gp_{name}", f"app.exp_exit_step = {end}",
        f"app.batchrun.exp_exit_step = {end}", "app.train_panel.fps = 0",
        "app.batchrun.train_panel.fps = 0", "run.seed = 1", "agent.device = cpu",
        "run.eval_device = cpu", f"run.train.num_envs = {LANES}",
        "backend.torch_num_threads = 1", "backend.$ = backend.@deterministic",
        "backend.deterministic_warn_only = false", "metrics.image.interval = 0",
        f"DefaultDQNAgent.intrinsic.enabled = {'true' if arm == 'on' else 'false'}",
        f"metrics.scalar.[{CLOCK}] = elapse_hour @train $runner $exp_step interval:{LANES}"]
    overrides += [f"run.eval_schedule.[{tag}].interval = 0" for tag in ("eval", "eval_target", "eval_panel")]
    profile = "run.@prd116_perf"
    chain = "run.@rnd > " + profile if arm == "on" else profile
    return "\n".join([f"$include <{root.as_posix()}/_main.txt>",
        f"$include <{root.as_posix()}/GridMaze.txt>"] + [profile + " : " + line for line in overrides]
        + ["run.$ = " + chain]) + "\n"


def prepare(out):
    # 他のRunとの競合と既存記録の上書きを避け、測定の入力を固定する。
    idle()
    if out.exists():
        raise FileExistsError("Use a fresh output directory: " + str(out))
    out.mkdir(parents=True)
    shutil.copytree(REPO / "apps/runner/config", out / "config", ignore=shutil.ignore_patterns("*.log"))
    (out / "bin").mkdir()
    binary = REPO / "apps/runner/bin/Release/AnetRLRunner.exe"
    shutil.copy2(binary, out / "bin/AnetRLRunner.exe")
    # DLLは一度だけ複製し、以後のビルドから測定用の実行体を切り離す。
    for dll in binary.parent.glob("*.dll"):
        shutil.copy2(dll, out / "bin" / dll.name)
    metadata = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "binary_sha256": digest(out / "bin/AnetRLRunner.exe"), "build": "Release", "device": "cpu",
        "lanes": LANES, "torch_threads": 1, "warm_experiences": WARM, "order": ORDER,
        "source_config_sha256": {p.name: digest(p) for p in (out / "config").glob("*.txt")}}
    save(out / "plan.json", metadata)
    subprocess.run(["git", "diff", "--no-ext-diff", "--no-color", "--output=" + str(out / "source-diff.patch"),
        "HEAD", "--", "core", "apps/runner/src", "apps/runner/include"], cwd=REPO, check=True)
    # 同じ設定解決の経路で両群を比較し、Qや環境の差が入ったら採取を始めない。
    resolved = {}
    for arm in ("off", "on"):
        path = out / (arm + ".txt")
        path.write_text(config(out, "resolve_" + arm, arm, 8192), encoding="utf-8", newline="\n")
        target = out / (arm + ".resolved.json")
        env = os.environ.copy()
        env.update(ANET_PRD103_INPUT=str(path), ANET_PRD103_OUTPUT=str(target), OMP_NUM_THREADS="1")
        with (out / (arm + ".resolve.log")).open("w", encoding="utf-8") as log:
            subprocess.run([str(REPO / "core/anet-core/bin/Debug/anet-core-test.exe"), "[prd103-resolve]"],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, **hidden())
        resolved[arm] = json.loads(target.read_text(encoding="utf-8-sig"))
        data = resolved[arm]
        for key, expected in {"agent.device": "cpu", "run.train.num_envs": str(LANES),
                "DefaultDQNAgent.intrinsic.enabled": "true" if arm == "on" else "false"}.items():
            if data.get(key) != expected:
                raise RuntimeError(f"Resolved config {target}, key={key}: actual={data.get(key)!r}, expected={expected!r}")
        active = {key: value for key, value in data.items()
                  if key.startswith("run.eval_schedule.[") and key.endswith(".interval") and value != "0"}
        if active:
            raise RuntimeError(f"Resolved config {target}: active intervals={active}, expected all intervals=0")
    differences = {key: [resolved["off"].get(key), resolved["on"].get(key)]
        for key in sorted(set(resolved["off"]) | set(resolved["on"]))
        if resolved["off"].get(key) != resolved["on"].get(key)}
    save(out / "resolved-differences.json", differences)
    unexpected = [key for key in differences if key != "app.run_name"
        and not key.startswith("DefaultDQNAgent.intrinsic.")
        and not key.startswith("metrics.scalar.[56_agent_in/")]
    if unexpected:
        raise RuntimeError(f"OFF/ON config differences in {out}: actual="
            f"{ {key: differences[key] for key in unexpected} }, expected only RND, group 56, and Run name differences")
    print("PREPARED", str(out), "resolved_differences=" + str(len(differences)), flush=True)


def rows(run):
    path = run / "metrics.jsonl"
    stream = path.open(encoding="utf-8-sig") if path.exists() else gzip.open(
        str(path) + ".gz", "rt", encoding="utf-8-sig")
    with stream:
        for line in stream:
            yield json.loads(line)


def clock_progress(run):
    # 採取開始をwarm-up後へ置くための短い末尾読み。通常の時間比較では呼ばない。
    path = run / "metrics.jsonl"
    if not path.exists():
        return 0
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 131072))
        lines = stream.read().decode("utf-8", errors="ignore").splitlines()
    result = 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("tag") == CLOCK:
            result = max(result, int(row["step"]))
    return result


def execute(out, name, arm, window, tools=None):
    # caseを新設し、固定した実行体と設定で通常のRunnerを起動する。
    idle()
    folder = out / "cases" / name
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "run.txt"
    path.write_text(config(out, name, arm, window), encoding="utf-8", newline="\n")
    binary = out / "bin/AnetRLRunner.exe"
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    actual_sha = digest(binary)
    if actual_sha != plan["binary_sha256"]:
        raise RuntimeError(f"Case {name}, binary {binary}: actual={actual_sha}, expected={plan['binary_sha256']}")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
    command = [str(binary), "--config", str(path)]
    record = {"name": name, "arm": arm, "command": command, "binary_sha256": digest(binary),
        "config_sha256": digest(path), "window_experiences": window, "profiled": tools is not None}
    save(folder / "execution.json", record)
    print("START", name, flush=True)
    started = time.monotonic()
    capture = None
    with (folder / "launcher.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO / "apps/runner", env=env,
                                   stdout=log, stderr=subprocess.STDOUT, **hidden())
        record["pid"] = process.pid
        save(folder / "execution.json", record)
        if tools:
            # 別RunでTracyを接続する。通常のABBAの実時間には採取負荷を混ぜない。
            while process.poll() is None:
                found = list((out / "runs").glob("run_*_prd116_gp_" + name))
                if found and clock_progress(found[0]) >= WARM:
                    capture_log = (folder / "capture.log").open("w", encoding="utf-8")
                    capture = subprocess.Popen([str(tools / "tracy-capture.exe"), "-a", "127.0.0.1",
                        "-o", str(folder / "capture.tracy"), "-s", "8", "-m", "5"],
                        stdout=capture_log, stderr=subprocess.STDOUT, **hidden())
                    record["capture_requested_after_exp_step"] = clock_progress(found[0])
                    break
                time.sleep(0.5)
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print("RUNNING", name, "seconds=" + str(round(time.monotonic() - started)), flush=True)
    record.update(exit_code=process.returncode, process_seconds=time.monotonic() - started)
    found = list((out / "runs").glob("run_*_prd116_gp_" + name))
    record["run"] = str(found[0]) if len(found) == 1 else None
    save(folder / "execution.json", record)
    if process.returncode or len(found) != 1:
        raise RuntimeError(f"Case {name}, runs directory {out / 'runs'}: exit_code={process.returncode}, "
            f"expected_exit_code=0; run_count={len(found)}, expected_run_count=1; matches={found}")
    if tools:
        if capture is None:
            raise RuntimeError(f"Case {name}: Tracy capture actual=not started, expected connection after "
                f"exp_step >= {WARM}; Runner exit_code={process.returncode}, last_clock_step={clock_progress(found[0])}")
        capture.wait(timeout=120)
        capture_log.close()
        record["capture_exit_code"] = capture.returncode
        save(folder / "execution.json", record)
        if capture.returncode or not (folder / "capture.tracy").is_file():
            raise RuntimeError(f"Case {name}, trace {folder / 'capture.tracy'}: capture_exit_code={capture.returncode}, "
                f"trace_exists={(folder / 'capture.tracy').is_file()}, expected exit_code=0 and trace_exists=True")
        for kind, options, zone_filter in (("zones", [], "RndNoveltyEstimator"),
                ("events", ["-u"], "RndNoveltyEstimator"), ("facade_events", ["-u"], "intrinsic_")):
            with (folder / (kind + ".csv")).open("w", encoding="utf-8", newline="") as stream:
                subprocess.run([str(tools / "tracy-csvexport.exe"), *options,
                    "-f", zone_filter, str(folder / "capture.tracy")],
                    stdout=stream, stderr=subprocess.PIPE, check=True, **hidden())
        print("CAPTURED", name, flush=True)
    # 起動・warm-upを除く同一経験区間だけを比較し、ONでは補助学習の進行も確かめる。
    points = {int(row["step"]): row["value"] for row in rows(found[0]) if row.get("tag") == CLOCK}
    if WARM not in points or WARM + window not in points:
        missing = [step for step in (WARM, WARM + window) if step not in points]
        raise RuntimeError(f"Case {name}, Run {found[0]}: missing={missing}, expected boundaries="
            f"{[WARM, WARM + window]}, observed_points={len(points)}, last_steps={sorted(points)[-5:]}")
    seconds = (points[WARM + window] - points[WARM]) * 3600
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"Case {name}, Run {found[0]}: clock boundaries={WARM}/{WARM + window}, "
            f"clock_hours={points[WARM]}/{points[WARM + window]}, seconds={seconds}, expected finite and > 0")
    record.update(measurement_seconds=seconds, experiences_per_second=window / seconds)
    if arm == "on":
        counts = [row["value"] for row in rows(found[0]) if row.get("tag") == "56_agent_in/32_learn_count"]
        if not counts or max(counts) <= 0:
            raise RuntimeError(f"Case {name}, Run {found[0]}, 56_agent_in/32_learn_count: "
                f"points={len(counts)}, actual={max(counts) if counts else 'missing'}, expected > 0")
        record["learn_count_max"] = max(counts)
    save(folder / "execution.json", record)
    print("END", name, "measurement_seconds=" + str(round(seconds, 3)),
          "exp_per_second=" + str(round(window / seconds, 1)), flush=True)
    return record


def profile_summary(folder):
    # CSVのtotal_percはprocess起点を分母にするので使わず、完結したRND区間で揃える。
    def read_events(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return [{"name": row["name"], "start": int(row["ns_since_start"]),
                     "duration": int(row["exec_time_ns"]), "thread": row["thread"]}
                    for row in csv.DictReader(stream)]

    roots = [row for row in read_events(folder / "facade_events.csv")
             if row["name"].endswith(".intrinsic_reward") and row["duration"] > 0]
    if not roots:
        raise RuntimeError(f"Tracy CSV {folder / 'facade_events.csv'}: complete intrinsic_reward zones "
            "actual=0, expected > 0")
    def contained(row):
        return row["duration"] >= 0 and any(row["thread"] == root["thread"]
            and root["start"] <= row["start"]
            and row["start"] + row["duration"] <= root["start"] + root["duration"] for root in roots)

    # 採取端で途切れた区間を混ぜず、同じthreadの完結した親に含まれるイベントを集計する。
    raw = read_events(folder / "events.csv")
    events = [row for row in raw if contained(row)]
    groups = {}
    functions = {}
    for row in events:
        destination = groups if "." in row["name"] else functions
        name = row["name"].rsplit(".", 1)[-1]
        item = destination.setdefault(name, {"count": 0, "total_ns": 0})
        item["count"] += 1
        item["total_ns"] += row["duration"]
    required = {"target_forward", "predictor_forward", "backward", "optimizer_step", "statistics", "normalize",
                "observation_transfer", "input_transfer", "error_transfer", "result_transfer"}
    if not required.issubset(groups):
        raise RuntimeError(f"Tracy CSV {folder / 'events.csv'}: missing={sorted(required - set(groups))}, "
            f"actual phases={sorted(groups)}, expected phases={sorted(required)}")
    total_ns = sum(root["duration"] for root in roots)
    span_ns = max(root["start"] + root["duration"] for root in roots) - min(root["start"] for root in roots)
    for item in [*groups.values(), *functions.values()]:
        item.update(mean_us=item["total_ns"] / item["count"] / 1000,
                    total_ms=item["total_ns"] / 1e6, percent_intrinsic=item["total_ns"] / total_ns * 100)
    result = {"intrinsic_intakes": len(roots), "observation_count": len(roots) * LANES,
        "intrinsic_total_ms": total_ns / 1e6, "intrinsic_ms_per_intake": total_ns / len(roots) / 1e6,
        "complete_zone_span_seconds": span_ns / 1e9, "intrinsic_percent_span": total_ns / span_ns * 100,
        "functions": functions, "phases": groups,
        "unclassified_intrinsic_ms": (total_ns - sum(item["total_ns"] for item in groups.values())) / 1e6,
        "excluded_events_outside_complete_intakes": len(raw) - len(events),
        "trace_sha256": digest(folder / "capture.tracy"),
        "note": "CPU host inclusive ranges in a separate profiled Run; not substituted for unprofiled ON/OFF timing"}
    save(folder / "breakdown.json", result)
    return result


def main():
    # 測定と後処理を分け、集計や記録の確認でRunnerを起動しないようにする。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "measure", "profile", "summarize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracy-tools", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.phase == "prepare":
        prepare(out)
    elif args.phase == "pilot":
        # 遅い群で約30秒となる共通区間をlane単位へ切り上げ、以後は固定する。
        pilots = [execute(out, "pilot_" + arm, arm, 8192) for arm in ("off", "on")]
        slowest = max(item["measurement_seconds"] for item in pilots)
        window = max(32768, math.ceil(8192 * 30 / slowest / LANES) * LANES)
        save(out / "pilot.json", {"runs": pilots, "selected_window": window,
            "predicted_run_seconds": [item["process_seconds"] + (window / 8192 - 1)
                * item["measurement_seconds"] for item in pilots]})
        print("PILOT selected_window=" + str(window), flush=True)
    elif args.phase == "measure":
        # 記録が一部でもある保存先は再利用せず、全12本を順次実行する。
        if (out / "measurements.json").exists() or any((out / "cases" / f"r{index + 1:02d}_{arm}").exists()
                                                      for index, arm in enumerate(ORDER)):
            raise FileExistsError(f"Recorded measurements exist in {out}; expected a fresh output directory; "
                "existing measurements must be retained")
        window = json.loads((out / "pilot.json").read_text(encoding="utf-8"))["selected_window"]
        results = []
        for index, arm in enumerate(ORDER):
            try:
                results.append(execute(out, f"r{index + 1:02d}_{arm}", arm, window))
            except Exception as error:
                results.append({"name": f"r{index + 1:02d}_{arm}", "error": str(error)})
                print("FAILED", index + 1, str(error), flush=True)
            save(out / "measurements.json", results)
        if any("error" in item for item in results):
            raise RuntimeError(f"Performance measurements {out}: failed="
                f"{[item for item in results if 'error' in item]}, expected all 12 Runs successful")
    elif args.phase == "profile":
        if args.tracy_tools is None:
            parser.error("profile requires --tracy-tools")
        window = json.loads((out / "pilot.json").read_text(encoding="utf-8"))["selected_window"]
        execute(out, "profile_on", "on", 2 * window, args.tracy_tools.resolve())
    else:
        # 通常比較の中央値と、別Runで採ったTracyの内訳を区別して保存する。
        results = json.loads((out / "measurements.json").read_text(encoding="utf-8"))
        if len(results) != 12 or any("error" in item or item["profiled"] for item in results):
            raise RuntimeError(f"Measurements {out / 'measurements.json'}: count actual={len(results)}, expected=12; "
                f"invalid={[item for item in results if 'error' in item or item.get('profiled')]}, "
                "expected all unprofiled and without errors")
        if len({item["window_experiences"] for item in results}) != 1 or len({item["binary_sha256"] for item in results}) != 1:
            raise RuntimeError(f"Measurements {out / 'measurements.json'}: "
                f"budgets={sorted({item['window_experiences'] for item in results})}, "
                f"binary_sha256={sorted({item['binary_sha256'] for item in results})}; "
                "expected exactly one experience budget and exactly one executable SHA256")
        medians = {arm: statistics.median(item["measurement_seconds"] for item in results if item["arm"] == arm)
                   for arm in ("off", "on")}
        ratio = medians["on"] / medians["off"]
        summary = {"medians_seconds": medians, "on_off_ratio": ratio, "overhead_percent": (ratio - 1) * 100,
            "experiences_per_second": {arm: results[0]["window_experiences"] / seconds
                                      for arm, seconds in medians.items()}, "runs": results,
            "acceptance": "No speed threshold; record timing and Tracy breakdown"}
        summary["ranges_seconds"] = {arm: [min(item["measurement_seconds"] for item in results if item["arm"] == arm),
            max(item["measurement_seconds"] for item in results if item["arm"] == arm)] for arm in ("off", "on")}
        summary["cycle_ratios"] = [statistics.median(item["measurement_seconds"] for item in results[start:start + 4]
            if item["arm"] == "on") / statistics.median(item["measurement_seconds"] for item in results[start:start + 4]
            if item["arm"] == "off") for start in (0, 4, 8)]
        if (out / "cases/profile_on/events.csv").exists():
            summary["tracy"] = profile_summary(out / "cases/profile_on")
        save(out / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "runs"}), flush=True)


if __name__ == "__main__":
    main()
