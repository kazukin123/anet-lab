"""PRD130: 固定した Release 実行体の性能受入と、別 Run の Tracy 採取。"""
import statistics
import math
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import ctypes
from ctypes import wintypes
from contextlib import contextmanager

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
GAME = "montezuma_revenge"
Q_WARMUP = 20000
RND_WARMUP = 12800
WARM = 32768
PILOT_WINDOW = 8192
FASTEST_TARGET_SECONDS = 20
PAUSED_CPU_LIMIT = 1.0
SETS = {
    "main": {"arms": ("off", "on"), "order": ("off", "on", "on", "off") * 3, "limit": 1.15},
    "legacy": {"arms": ("off", "on"), "order": ("off", "on", "on", "off") * 3, "limit": 1.25},
    "grid": {"arms": ("before", "stage1"), "order": ("before", "stage1", "stage1", "before") * 3},
}
STAGE2_SETS = {
    name: {"arms": ("stage1", "stage2"), "order": ("stage1", "stage2", "stage2", "stage1") * 3}
    for name in ("main", "legacy", "taiga")
}


def load_module(name, relative):
    # 既存の PRD116 の時計・保存・設定の組み立てを借り、原本は変えない。
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grid = load_module("prd130_grid", "core/anet-core/testdata/prd116/run_performance.py")
atari = load_module("prd130_atari", "core/anet-core/testdata/prd116/run_performance_atari.py")
validation = load_module("prd130_validation", "core/anet-core/testdata/prd130/run_validation.py")
CLOCK = grid.CLOCK


def summarize_records(set_name, records, stage=1):
    # 同じ経験区間の各群の中央値から、PRD の上限を判定する。
    spec = (STAGE2_SETS if stage == 2 else SETS)[set_name]
    if ([r["arm"] for r in records] != list(spec["order"])
            or len({r["window_experiences"] for r in records}) != 1
            or any(not math.isfinite(r["measurement_seconds"]) or r["measurement_seconds"] <= 0
                   or r.get("profiled") or "error" in r for r in records)):
        raise ValueError("Invalid measurement order, window, duration, or profiled Run")
    seconds = {arm: [r["measurement_seconds"] for r in records if r["arm"] == arm]
               for arm in spec["arms"]}
    medians = {arm: statistics.median(values) for arm, values in seconds.items()}
    ranges = {arm: [min(values), max(values)] for arm, values in seconds.items()}
    if stage == 2:
        delta = medians["stage2"] - medians["stage1"]
        spread = ranges["stage1"][1] - ranges["stage1"][0]
        return {"medians_seconds": medians, "ranges_seconds": ranges, "delta_seconds": delta,
                "stage1_spread_seconds": spread, "accepted": abs(delta) <= spread}
    if set_name == "grid":
        delta = medians["stage1"] - medians["before"]
        spread = ranges["before"][1] - ranges["before"][0]
        return {"medians_seconds": medians, "ranges_seconds": ranges, "delta_seconds": delta,
                "before_spread_seconds": spread, "accepted": delta <= spread}
    ratio = medians["on"] / medians["off"]
    limit = 1.15 if set_name == "main" else 1.25
    return {"medians_seconds": medians, "ranges_seconds": ranges, "ratio": ratio,
            "limit": limit, "accepted": ratio <= limit}


MAIN_CHAIN = ("run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard500>run.@rr4>run.@munch>run.@rfit"
              ">run.@a5_metrics>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch>run.@to_30m>run.@rnd>run.@batch")

BASE_ARGS = [
    "E1.game=" + GAME, "run.seed=1", "run.eval.[greedy].eval_episodes=16",
    "DefaultDQNAgent.actor.[train].policy.tau_rule.num_taus=32",
    "DefaultDQNAgent.actor.[eval].policy.tau_rule.num_taus=32",
    "net.block.[AtariImpalaConv32].init.mode=default", "net.block.[AtariImpalaConv64].init.mode=default",
    "net.block.[AtariImpalaRes32SN].init1.mode=default", "net.block.[AtariImpalaRes32SN].init2.mode=default",
    "net.block.[AtariImpalaRes64SN].init1.mode=default", "net.block.[AtariImpalaRes64SN].init2.mode=default",
    "net.block.[AtariIQNTauProj2304].init.mode=default",
    "DefaultDQNAgent.learner.replay_ratio=8", "DefaultDQNAgent.model.hard_update_interval=250",
    "DefaultDQNAgent.learner.per_alpha=0.5", "DefaultDQNAgent.learner.per_beta.start=0.5",
    "DefaultDQNAgent.learner.per_beta.end=0.5",
    "DefaultDQNAgent.learner.munchausen.enabled=false", "DefaultDQNAgent.learner.use_double_dqn=true",
]

RND_ARGS = [
    "DefaultDQNAgent.intrinsic.learn_ratio=1", "DefaultDQNAgent.intrinsic.learn_batch_size=128",
    f"DefaultDQNAgent.intrinsic.rnd.warmup_steps={RND_WARMUP}",
]

VD_ARGS = [
    "DefaultDQNAgent.intrinsic.value_mode=decomposed", "DefaultDQNAgent.intrinsic.reward_scaler.constant_scale=0.01",
    "DefaultDQNAgent.intrinsic.beta=1",
    "DefaultDQNAgent.learner.intrinsic.discount_mode=separate", "DefaultDQNAgent.learner.intrinsic.gamma=0.99",
    "DefaultDQNAgent.learner.intrinsic.terminal_mode=non_episodic",
    "metrics.scalar.$=metrics.scalar.@baseline>metrics.scalar.@munchausen>metrics.scalar.@replay_fit>M1>M2"
    ">metrics.scalar.@value_decomposition",
    "metrics.scalar.[77_agent_value/41_greedy_disagree]=extrinsic_q_session_argmax_disagreement_rate $eval.[greedy] $actor @session_end",
    "metrics.scalar.[77_agent_value/42_greedy_regret]=extrinsic_q_session_regret_mean $eval.[greedy] $actor @session_end",
    "metrics.scalar.[77_agent_value/43_greedy_ext_gap]=extrinsic_q_session_gap_mean $eval.[greedy] $actor @session_end",
    "metrics.scalar.[77_agent_value/44_greedy_int_gap]=intrinsic_q_session_gap_mean $eval.[greedy] $actor @session_end",
]

TRACY_FILTERS = ("RndNoveltyEstimator", "IntrinsicRewardCoordinator", "UpdateFromBatch", "RunningMeanStd",
                 "Normalizer", "RewardScaler")

def read_dump(run):
    values = {}
    for line in (run / "config/config_data.txt").read_text(encoding="utf-8-sig", errors="replace").splitlines():
        key, sep, value = line.partition(" = ")
        if sep:
            values.setdefault(key.strip(), set()).add(value.replace(",", "").strip())
    return values

def export_tracy(folder, tools):
    # 全区間の合計の表と、RND の採点に関わる区間の個々の記録を CSV に出す。
    with (folder / "zones_all.csv").open("w", encoding="utf-8", newline="") as stream:
        subprocess.run([str(tools / "tracy-csvexport.exe"), str(folder / "capture.tracy")], stdout=stream,
                       stderr=subprocess.PIPE, check=True, **grid.hidden())
    for zone_filter in TRACY_FILTERS:
        with (folder / f"events_{zone_filter}.csv").open("w", encoding="utf-8", newline="") as stream:
            subprocess.run([str(tools / "tracy-csvexport.exe"), "-u", "-f", zone_filter, str(folder / "capture.tracy")],
                           stdout=stream, stderr=subprocess.PIPE, check=True, **grid.hidden())
    print("CAPTURED", folder.name, flush=True)

def breakdown(folder, lanes):
    # 同じ thread の包含関係から区間の木を作り、完結した intrinsic_reward の区間の中だけを集計する。
    events = {}
    for path in sorted(folder.glob("events*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["thread"], int(row["ns_since_start"]), int(row["exec_time_ns"]), row["name"])
                events[key] = {"name": row["name"], "start": key[1], "duration": key[2], "thread": key[0]}
    ordered = sorted(events.values(), key=lambda e: (e["thread"], e["start"], -e["duration"], e["name"]))
    stack, thread = [], None
    for event in ordered:
        if event["thread"] != thread:
            stack, thread = [], event["thread"]
        while stack and event["start"] >= stack[-1]["start"] + stack[-1]["duration"]:
            stack.pop()
        inside = bool(stack) and event["start"] + event["duration"] <= stack[-1]["start"] + stack[-1]["duration"]
        event["parent"] = stack[-1] if inside else None
        event["children_ns"] = 0
        if inside:
            stack[-1]["children_ns"] += event["duration"]
        stack.append(event)

    def root_of(event):
        node = event
        while node is not None and not node["name"].endswith(".intrinsic_reward"):
            node = node["parent"]
        return node

    def path_of(event):
        names = []
        node = event
        while node is not None:
            names.append(node["name"])
            if node["name"].endswith(".intrinsic_reward"):
                break
            node = node["parent"]
        return " > ".join(reversed(names))

    roots = [e for e in ordered if e["name"].endswith(".intrinsic_reward") and e["duration"] > 0]
    if not roots:
        raise RuntimeError(f"Tracy CSV in {folder}: complete intrinsic_reward zones actual=0, expected > 0")
    total_ns = sum(root["duration"] for root in roots)
    rows = {}
    for event in ordered:
        if root_of(event) is None:
            continue
        item = rows.setdefault(path_of(event), {"count": 0, "inclusive_ns": 0, "self_ns": 0})
        item["count"] += 1
        item["inclusive_ns"] += event["duration"]
        item["self_ns"] += event["duration"] - event["children_ns"]
    for item in rows.values():
        item.update(inclusive_ms_per_intake=item["inclusive_ns"] / len(roots) / 1e6,
                    self_ms_per_intake=item["self_ns"] / len(roots) / 1e6,
                    inclusive_percent=item["inclusive_ns"] / total_ns * 100, self_percent=item["self_ns"] / total_ns * 100)
    updates = [e for e in ordered if e["name"].endswith("::UpdateFromBatch") and e["duration"] > 0]
    span_ns = max(r["start"] + r["duration"] for r in roots) - min(r["start"] for r in roots)
    result = {"intrinsic_intakes": len(roots), "observation_count": len(roots) * lanes,
              "intrinsic_ms_per_intake": total_ns / len(roots) / 1e6,
              "intrinsic_percent_of_span": total_ns / span_ns * 100,
              "update_from_batch_ms_per_call": (sum(e["duration"] for e in updates) / len(updates) / 1e6) if updates else None,
              "update_from_batch_calls": len(updates), "paths": dict(sorted(rows.items())),
              "note": "CPU host ranges from a separate profiled Run; not GPU kernel time"}
    grid.save(folder / "breakdown.json", result)
    print(f"BREAKDOWN {folder.name}: intakes={len(roots)} intrinsic_ms_per_intake={result['intrinsic_ms_per_intake']:.3f}")
    # 親子の並びのまま、区間の名前を字下げして表示する。
    for path, item in sorted(rows.items()):
        names = path.split(" > ")
        print(f"  {item['inclusive_ms_per_intake']:8.3f} ms incl {item['self_ms_per_intake']:8.3f} ms self "
              f"{item['inclusive_percent']:6.2f}%  {'  ' * (len(names) - 1)}{names[-1]}")
    return result

def choose_window(pilots, lanes):
    # いちばん速い群でも目標の秒数を測れる区間を、lane の数の倍数に切り上げる。
    fastest = min(item["measurement_seconds"] for item in pilots)
    return max(2 * PILOT_WINDOW, math.ceil(PILOT_WINDOW * FASTEST_TARGET_SECONDS / fastest / lanes) * lanes)


def runner_processes():
    # 休止済みの利用者の Run だけを許し、別の測定との重複を防ぐ。
    command = "Get-CimInstance Win32_Process -Filter \"Name LIKE 'AnetRLRunner%.exe'\" | ForEach-Object { $_.ProcessId }"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, check=True, **grid.hidden())
    return {int(line) for line in result.stdout.splitlines() if line.strip()}


def check_idle(plan):
    unexpected = runner_processes() - set(plan["paused_pids"])
    if unexpected:
        raise RuntimeError(f"Other Runner processes active: {sorted(unexpected)}")


def paused_cpu(plan):
    # 休止した Run が測定中に動き始めた場合に、その採取を受入へ使わない。
    with paused_cpu_reader(plan) as read:
        return read()


@contextmanager
def paused_cpu_reader(plan):
    # 同じプロセスの HANDLE を保持し、低負荷の OS API で CPU を読む。
    if not plan["paused_pids"]:
        yield lambda: {}
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    handles = {}
    try:
        for pid in plan["paused_pids"]:
            handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            handles[str(pid)] = handle

        def read():
            result = {}
            for pid, handle in handles.items():
                # 終了済みの HANDLE から読める値を、生存中の CPU と誤認しない。
                if kernel.WaitForSingleObject(handle, 0) != 0x102:
                    raise RuntimeError(f"Paused Runner PID={pid} is not alive; expected running process")
                created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
                if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                              ctypes.byref(system), ctypes.byref(user)):
                    raise ctypes.WinError(ctypes.get_last_error())
                result[pid] = (((system.dwHighDateTime << 32) | system.dwLowDateTime)
                               + ((user.dwHighDateTime << 32) | user.dwLowDateTime)) / 1e7
            return result

        yield read
    finally:
        for handle in handles.values():
            kernel.CloseHandle(handle)


def select_paused_sample(samples, elapsed):
    # CPU の値では選ばず、開始前と証明できる最後の時刻だけで選ぶ。
    previous = samples[0]["cpu"] if samples else {}
    previous_offset = -1.0
    for sample in samples:
        cpu = sample["cpu"]
        offset = sample["offset_seconds"]
        if (not math.isfinite(offset) or offset < 0 or offset < previous_offset
                or set(cpu) != set(previous)
                or any(not math.isfinite(value) or value < 0 or value < previous[pid]
                       for pid, value in cpu.items())):
            raise RuntimeError(f"Invalid CPU sample={sample}; expected complete nonnegative monotonic counters and time")
        previous, previous_offset = cpu, offset
    eligible = [sample for sample in samples if math.isfinite(sample["offset_seconds"])
                and 0 <= sample["offset_seconds"] + 0.01 < elapsed]
    if not eligible:
        raise RuntimeError(f"No CPU sample before measurement elapsed={elapsed}; expected offset + 0.01 < elapsed")
    return max(eligible, key=lambda sample: sample["offset_seconds"])


def paused_run_state(plan):
    # 学習停止は CPU 時間だけで推測せず、Run の記録が進んでいないことも確かめる。
    result = {}
    for pid, folder in plan["paused_runs"].items():
        root = Path(folder)
        result[pid] = {}
        for path in (root / "metrics.jsonl", root / (root.name + ".log")):
            stat = path.stat()
            result[pid][path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def verify_paused_run(plan, record):
    # 実測の全区間を含む CPU 検査と、Run 全体の記録の停止を両方要求する。
    if not plan["paused_pids"]:
        return
    if record.get("paused_guard_error"):
        raise RuntimeError("Paused CPU guard failed: " + record["paused_guard_error"])
    cpu = record["paused_guard_cpu_seconds"]
    end = record["paused_guard_end_experience"]
    if end < WARM + record["window_experiences"]:
        raise RuntimeError(f"Paused CPU guard does not cover measurement: end={end}; "
                           f"expected end >= {WARM + record['window_experiences']}")
    if record["paused_run_state_before"] != record["paused_run_state_after"]:
        raise RuntimeError(f"Paused Run metrics or log changed during measurement: before={record['paused_run_state_before']}; "
                           f"after={record['paused_run_state_after']}; expected unchanged")
    # Runner の時計は子プロセス起動後に 0 から始まる。起動前を原点にした
    # CPU 読み取り完了までの秒数が実測開始の Runner 経過秒数より短ければ、
    # 原点の差も含めて保守的に開始前だと証明できる。ファイルの出力遅延には依らない。
    offset = record["paused_guard_start_offset_seconds"]
    elapsed = record["measurement_start_elapsed_seconds"]
    if (not math.isfinite(offset) or not math.isfinite(elapsed) or offset < 0
            or offset + 0.01 >= elapsed):
        raise RuntimeError(f"Paused CPU guard does not cover measurement: sample offset={offset}, "
                           f"Runner measurement start elapsed={elapsed}; expected offset + 0.01 < elapsed")
    if (set(cpu) != {str(pid) for pid in plan["paused_pids"]}
            or any(not math.isfinite(delta) or delta < 0 or delta > PAUSED_CPU_LIMIT for delta in cpu.values())):
        raise RuntimeError(f"Paused Runner used CPU during guarded measurement: {cpu}; expected <= {PAUSED_CPU_LIMIT} seconds")


def prepare(out, packages, paused_pids, stage=1, paused_runs=()):
    # 機能検証で保存した実行体を固定し、測定専用の設定から可塑性の計算を外す。
    if out.exists():
        raise FileExistsError(out)
    if len(paused_pids) != len(paused_runs):
        raise ValueError(f"Each --paused-pid requires one --paused-run in the same order; "
                         f"actual pids={len(paused_pids)}, runs={len(paused_runs)}")
    if len(set(paused_pids)) != len(paused_pids):
        raise ValueError(f"Duplicate --paused-pid values={paused_pids}; expected unique process IDs")
    plan = {"paused_pids": paused_pids, "packages": {}, "build": "Release", "stage": stage,
            "paused_runs": {str(pid): str(Path(run).resolve()) for pid, run in zip(paused_pids, paused_runs)},
            "paused_cpu_guard": {"sample_interval_seconds": 1.0, "limit_seconds": PAUSED_CPU_LIMIT,
                                 "covers_measurement_start": WARM, "checks_run_files_for_full_process": True,
                                 "requires_sample_offset_before_runner_elapsed": True,
                                 "clock_rounding_margin_seconds": 0.01}}
    paused_run_state(plan)
    check_idle(plan)
    out.mkdir(parents=True)
    shutil.copytree(packages / "before/package/config", out / "config")
    removed = []
    for path in sorted((out / "config").glob("*.txt")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        kept = []
        for line in lines:
            if "metrics.scalar." in line and ("plasticity_" in line or "[34_agent_plasticity/" in line):
                removed.append({"file": path.name, "line": line})
            else:
                kept.append(line)
        path.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    for label in (("stage1", "stage2") if stage == 2 else ("before", "stage1")):
        package = packages / ("stage1-accepted" if stage == 2 and label == "stage1" else label) / "package"
        binary = (package / "Release/AnetRLRunner.exe").resolve()
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        actual = grid.digest(binary)
        if actual != manifest["sha256"]["Release/AnetRLRunner.exe"]:
            raise RuntimeError(f"Saved package hash mismatch: {binary}")
        plan["packages"][label] = {"binary": str(binary), "sha256": actual}
    if stage == 2:
        # 機能受入を終えた実行体だけを性能比較へ渡す。
        accepted = json.loads((packages / "stage2/functional-acceptance.json").read_text(encoding="utf-8"))
        if (not {11, 12, 14, 15}.issubset(accepted["criteria_passed"])
                or accepted["runner_sha256"]["Release"] != plan["packages"]["stage2"]["sha256"]):
            raise RuntimeError("Stage2 functional acceptance does not match the selected Release Runner")
    plan.update(config_sha256={p.name: grid.digest(p) for p in (out / "config").glob("*.txt")},
                plasticity_definitions_removed=removed,
                source_packages=str(packages.resolve()), warm_experiences=WARM,
                q_warmup=Q_WARMUP, rnd_warmup=RND_WARMUP, sets=STAGE2_SETS if stage == 2 else SETS)
    grid.save(out / "plan.json", plan)
    print("PREPARED", out, "removed_plasticity_definitions=" + str(len(removed)), flush=True)


def build(out, name, set_name, arm, window, stage=1):
    # 計算の構成は基準と同じにし、測定用の上書きだけを最後の profile へ置く。
    config_arm = "taiga" if set_name == "taiga" else ("on" if stage == 2 else arm)
    if set_name == "grid":
        text = grid.config(out, name, "on", window).replace("prd116_gp_", "prd130_perf_")
        text = text.replace(f"{grid.WARM + window + 2 * grid.LANES}", str(WARM + window + 2 * grid.LANES))
        profile = "run.@prd116_perf"
    elif set_name in ("legacy", "taiga"):
        text = atari.config(out, name, config_arm, window, EVAL_TAGS).replace("prd116_ap_", "prd130_perf_")
        profile = atari.PROFILE
    else:
        profile = "run.@prd130_perf"
        chain = MAIN_CHAIN if config_arm == "on" else MAIN_CHAIN.replace(">run.@rnd", "")
        args = BASE_ARGS + (RND_ARGS + VD_ARGS if config_arm == "on" else [])
        text = "\n".join([f"$include <{(out / 'config/_main.txt').as_posix()}>",
                           f"$include <{(out / 'config/Atari.txt').as_posix()}>"]
                          + [profile + " : " + item for item in args]
                          + ["run.$ = " + chain + ">" + profile]) + "\n"
    lanes = 256 if set_name == "grid" else (128 if set_name == "main" else 64)
    end = WARM + window + 2 * lanes
    overrides = ["app.$ = app.batchrun > P1", "app.show_error_dialog = false", "app.train_auto_start = true",
        "app.eval_panel.auto_start = false", "app.save_agent_on_close = false",
        f"app.runs_dir = {(out / 'runs').as_posix()}", f"app.run_name = run_{{t}}_prd130_perf_{name}",
        f"app.exp_exit_step = {end}", f"app.batchrun.exp_exit_step = {end}",
        "app.train_panel.fps = 0", "app.batchrun.train_panel.fps = 0", "app.batchrun.eval_panel.fps = 0",
        "metrics.image.interval = 0", "backend.torch_num_threads = 1", "run.seed = 1",
        "agent.device = " + ("cpu" if set_name == "grid" else "cuda"),
        f"DefaultDQNAgent.learner.update_warmup_steps = {Q_WARMUP}",
        f"metrics.scalar.[{CLOCK}] = elapse_hour @train $runner $exp_step interval:{lanes}"]
    if arm != "off":
        overrides.append(f"DefaultDQNAgent.intrinsic.rnd.warmup_steps = {RND_WARMUP}")
    eval_tags = ("eval", "eval_target", "eval_panel") if set_name == "grid" else EVAL_TAGS
    overrides += [f"run.eval_schedule.[{tag}].interval = 0" for tag in eval_tags]
    lines = text.splitlines()
    lines[-1:-1] = [profile + " : " + item for item in overrides]
    return "\n".join(lines) + "\n"


def verify_config(run, set_name, arm, stage=1):
    # 解決後の実効値と登録ログで、可塑性の計算や評価を測っていないことを確かめる。
    values = read_dump(run)
    expected = {"run.train.num_envs": str(256 if set_name == "grid" else (128 if set_name == "main" else 64)),
                "backend.torch_num_threads": "1", "DefaultDQNAgent.learner.update_warmup_steps": str(Q_WARMUP),
                "DefaultDQNAgent.intrinsic.enabled": "false" if arm == "off" else "true",
                "agent.device": "cpu" if set_name == "grid" else "cuda"}
    if arm != "off":
        expected.update({"DefaultDQNAgent.intrinsic.rnd.warmup_steps": str(RND_WARMUP),
                         "DefaultDQNAgent.intrinsic.rnd.statistics_mode": "variance_feedback" if set_name == "taiga" else "whiten"})
    if set_name == "main" and (arm == "on" or stage == 2):
        expected["DefaultDQNAgent.intrinsic.value_mode"] = "decomposed"
    if set_name == "taiga":
        expected.update({"DefaultDQNAgent.intrinsic.observation": "state",
                         "DefaultDQNAgent.intrinsic.learn_batch_size": "1",
                         "DefaultDQNAgent.intrinsic.rnd.multiplier_mode": "unclipped"})
    problems = {key: sorted(values.get(key, {"missing"})) for key, value in expected.items()
                if values.get(key) != {value}}
    if set_name == "taiga":
        # 数値の表記揺れは許すが、Taiga の値そのものは置き換えず検査する。
        for key, expected_value in {"DefaultDQNAgent.intrinsic.beta": 0.00005,
                                    "DefaultDQNAgent.intrinsic.learn_ratio": 1.0,
                                    "DefaultDQNAgent.intrinsic.rnd.adam_eps": 1e-5}.items():
            actual = values.get(key, set())
            try:
                valid = len(actual) == 1 and float(next(iter(actual))) == expected_value
            except ValueError:
                valid = False
            if not valid:
                problems[key] = sorted(actual)
    active = [key for key, value in values.items() if key.startswith("run.eval_schedule.[")
              and key.endswith(".interval") and value != {"0"}]
    plasticity = [key for key in values if key.startswith("metrics.scalar.[34_agent_plasticity/")]
    log_text = (run / (run.name + ".log")).read_text(encoding="utf-8", errors="replace")
    if problems or active or plasticity or re.search(r"Notifier:.*plasticity_|\[[EF]\]", log_text):
        raise RuntimeError(f"Invalid performance config/log: {run}; values={problems}; eval={active}; plasticity={plasticity}")


def execute(out, name, set_name, arm, window, tools=None):
    # 休止プロセスの同一性を保ち、例外でも OS の HANDLE を解放する。
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    with paused_cpu_reader(plan) as read_cpu:
        return execute_with_cpu_reader(out, name, set_name, arm, window, tools, read_cpu)


def execute_with_cpu_reader(out, name, set_name, arm, window, tools, read_cpu):
    # 他の測定を重ねず、同じ経験区間の時計の差を取る。Tracy は別の Run だけに接続する。
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    stage = plan.get("stage", 1)
    spec = plan["sets"][set_name]
    if arm not in spec["arms"]:
        raise ValueError(f"Invalid measurement arm={arm}; expected one of {spec['arms']}")
    check_idle(plan)
    label = arm if stage == 2 or set_name == "grid" else "stage1"
    package = plan["packages"][label]
    binary = Path(package["binary"])
    if grid.digest(binary) != package["sha256"]:
        raise RuntimeError("Runner hash changed: " + str(binary))
    for filename, expected in plan["config_sha256"].items():
        if grid.digest(out / "config" / filename) != expected:
            raise RuntimeError("Measurement config changed: " + filename)
    paused_before = read_cpu()
    if set(paused_before) != {str(pid) for pid in plan["paused_pids"]}:
        raise RuntimeError("Paused Runner CPU counters are unavailable: " + str(plan["paused_pids"]))
    paused_state_before = paused_run_state(plan)
    folder = out / "cases" / name
    folder.mkdir(parents=True, exist_ok=False)
    config = folder / "run.txt"
    config.write_text(build(out, name, set_name, arm, window, stage), encoding="utf-8", newline="\n")
    environment = validation.make_environment()
    environment["TRACY_PORT"] = "18086"
    if set_name == "grid":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    command = [str(binary), "--config", str(config)]
    record = {"name": name, "arm": arm, "set": set_name, "label": label, "command": command,
              "binary_sha256": package["sha256"], "config_sha256": grid.digest(config),
              "window_experiences": window, "profiled": tools is not None,
              "paused_run_state_before": paused_state_before,
              "paused_cpu_guard": plan["paused_cpu_guard"],
              "environment_sha256": validation.hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
              "backend_environment": {key: environment.get(key) for key in ("OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                                        "CUBLAS_WORKSPACE_CONFIG", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TRACY_PORT")}}
    grid.save(folder / "execution.json", record)
    print("START", name, flush=True)
    started = time.monotonic()
    capture = capture_log = None
    cpu_samples = []

    def sample_paused_cpu():
        if not plan["paused_pids"] or record.get("paused_guard_error"):
            return
        try:
            # 読み取り完了時刻を残し、採用する始点は終了後に時刻条件だけで決める。
            cpu = read_cpu()
            cpu_samples.append({"offset_seconds": time.monotonic() - started, "cpu": cpu})
        except Exception as error:
            # 自分の Runner の終了を待ってから失敗を返し、背景の採取を残さない。
            record["paused_guard_error"] = str(error)

    next_report = started + 60
    with (folder / "launcher.log").open("w", encoding="utf-8") as log:
        sample_paused_cpu()
        process = subprocess.Popen(command, cwd=REPO / "apps/runner", env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, **grid.hidden())
        record["pid"] = process.pid
        grid.save(folder / "execution.json", record)
        if tools:
            while process.poll() is None:
                found = list((out / "runs").glob("run_*_prd130_perf_" + name))
                sample_paused_cpu()
                if found and grid.clock_progress(found[0]) >= WARM:
                    capture_log = (folder / "capture.log").open("w", encoding="utf-8")
                    capture = subprocess.Popen([str(tools / "tracy-capture.exe"), "-a", "127.0.0.1", "-p", "18086", "-o",
                        str(folder / "capture.tracy"), "-s", "8", "-m", "5"], stdout=capture_log,
                        stderr=subprocess.STDOUT, **grid.hidden())
                    break
                time.sleep(0.5)
        while process.poll() is None:
            try:
                process.wait(timeout=1 if plan["paused_pids"] else 60)
            except subprocess.TimeoutExpired:
                sample_paused_cpu()
                if time.monotonic() >= next_report:
                    print("RUNNING", name, flush=True)
                    next_report = time.monotonic() + 60
    found = list((out / "runs").glob("run_*_prd130_perf_" + name))
    record.update(exit_code=process.returncode, process_seconds=time.monotonic() - started,
                  run=str(found[0]) if len(found) == 1 else None)
    try:
        paused_after = read_cpu()
    except Exception as error:
        paused_after = {}
        record["paused_guard_error"] = str(error)
    record["paused_cpu_samples"] = cpu_samples
    record["paused_cpu_end"] = paused_after
    record["paused_cpu_seconds"] = {pid: value - paused_before.get(pid, value) for pid, value in paused_after.items()}
    record["paused_run_state_after"] = paused_run_state(plan)
    record["paused_guard_end_experience"] = grid.clock_progress(found[0]) if len(found) == 1 else 0
    if len(found) == 1 and process.returncode == 0:
        points = {int(row["step"]): row["value"] for row in grid.rows(found[0]) if row.get("tag") == CLOCK}
        record["measurement_start_elapsed_seconds"] = points[WARM] * 3600
        if plan["paused_pids"] and not record.get("paused_guard_error"):
            grid.save(folder / "execution.json", record)
            sample = select_paused_sample(cpu_samples, record["measurement_start_elapsed_seconds"])
            record["paused_guard_start_offset_seconds"] = sample["offset_seconds"]
            record["paused_guard_start_cpu"] = sample["cpu"]
            record["paused_guard_cpu_seconds"] = {pid: value - sample["cpu"][pid] for pid, value in paused_after.items()}
    grid.save(folder / "execution.json", record)
    if process.returncode or len(found) != 1:
        raise RuntimeError(f"Run failed: {name}, exit={process.returncode}, runs={found}")
    verify_paused_run(plan, record)
    run = found[0]
    verify_config(run, set_name, arm, stage)
    if tools:
        if capture is None:
            raise RuntimeError("Tracy capture was not started: " + name)
        capture.wait(timeout=120)
        capture_log.close()
        if capture.returncode:
            raise RuntimeError("Tracy capture failed: " + name)
        export_tracy(folder, tools)
        breakdown(folder, 256 if set_name == "grid" else (128 if set_name == "main" else 64))
    seconds = (points[WARM + window] - points[WARM]) * 3600
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"Invalid elapsed time: {seconds}")
    if arm != "off":
        counts = [row["value"] for row in grid.rows(run) if row.get("tag") == "56_agent_in/32_learn_count"]
        if not counts or max(counts) <= 0:
            raise RuntimeError("RND learning did not advance: " + name)
        record["learn_count_max"] = max(counts)
    record["measurement_seconds"] = seconds
    grid.save(folder / "execution.json", record)
    print("END", name, "seconds=" + str(round(seconds, 3)), flush=True)
    return record


def run_set(out, set_name, tools=None, stage=1):
    # Pilot で経験数を固定したあと、指定の ABBA の 3 巡を保存する。
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    if stage != plan.get("stage", 1):
        raise ValueError(f"Measurement stage={stage} differs from prepared stage={plan.get('stage', 1)}")
    spec = plan["sets"][set_name]
    if tools:
        pilots = json.loads((out / ("pilot_" + set_name + ".json")).read_text(encoding="utf-8"))
        for arm in spec["arms"] if stage == 2 or set_name == "grid" else ("on",):
            execute(out, set_name + "_profile_" + arm, set_name, arm,
                    max(pilots["window_experiences"], 4 * PILOT_WINDOW), tools)
        return
    pilots = [execute(out, set_name + "_pilot_" + arm, set_name, arm, PILOT_WINDOW) for arm in spec["arms"]]
    lanes = 256 if set_name == "grid" else (128 if set_name == "main" else 64)
    window = choose_window(pilots, lanes)
    grid.save(out / ("pilot_" + set_name + ".json"), {"window_experiences": window, "runs": pilots})
    records = []
    for index, arm in enumerate(spec["order"]):
        records.append(execute(out, f"{set_name}_r{index + 1:02d}_{arm}", set_name, arm, window))
        grid.save(out / ("measurements_" + set_name + ".json"), records)
    summary = summarize_records(set_name, records, stage)
    grid.save(out / ("summary_" + set_name + ".json"), summary)
    print("SUMMARY", set_name, json.dumps(summary), flush=True)
    if not summary["accepted"]:
        raise RuntimeError("Performance acceptance failed: " + set_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "measure", "profile"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--packages", type=Path, default=REPO / ".scratch/prd130/acceptance-cuda")
    parser.add_argument("--paused-pid", type=int, action="append", default=[])
    parser.add_argument("--paused-run", type=Path, action="append", default=[])
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--set", choices=tuple(dict.fromkeys((*SETS, *STAGE2_SETS))), default="main")
    parser.add_argument("--tracy-tools", type=Path, default=REPO / ".scratch/prd116/tracy-0.13.1/tools")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.out.resolve(), args.packages.resolve(), args.paused_pid, args.stage, args.paused_run)
    else:
        run_set(args.out.resolve(), args.set, args.tracy_tools.resolve() if args.phase == "profile" else None, args.stage)


EVAL_TAGS = ("eval", "eval_panel", "eval_target", "greedy", "greedy_single")

if __name__ == "__main__":
    main()
