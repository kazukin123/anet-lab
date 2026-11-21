"""PRD121: 基準の保存と、既定5構成の全 scalar 比較。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import datetime
import statistics

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[5]
ROOT = REPO / ".scratch/prd121"
BASELINE = ROOT / "baseline"
INPUTS = ROOT / "inputs"
CASES = {"Atari_rnd": ("Atari", True), "LunarLander_rnd": ("LunarLander", True),
         "LunarLander": ("LunarLander", False), "GridMaze": ("GridMaze", False),
         "DropMerge": ("DropMerge", False)}


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load("prd121_common", "src/core/base/testdata/prd109/run_validation.py")
NORMALIZER = load("prd121_normalizer", "src/core/base/testdata/prd084/compare_metrics.py")


def snapshot(config):
    # 保存済みの実行体を上書きせず、構成ごとに保存する。
    destination = BASELINE / config
    destination.mkdir(parents=True, exist_ok=False)
    for path in (REPO / "apps/runner/bin" / config).iterdir():
        if path.suffix.lower() in (".exe", ".dll"):
            shutil.copy2(path, destination / path.name)
    COMMON.save(destination / "hashes.json", {
        path.name: COMMON.digest(path) for path in destination.iterdir() if path.is_file()})
    print("SNAPSHOT", destination, flush=True)


def write_frozen(path, text):
    # 再実行時も同じ入力だけを使い、採取済みの入力の変化を拒否する。
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError("Frozen input changed: " + str(path))
    else:
        path.write_text(text, encoding="utf-8", newline="\n")


def prepare():
    INPUTS.mkdir(parents=True, exist_ok=True)
    frozen = INPUTS / "baseline-config"
    if not frozen.exists():
        shutil.copytree(BASELINE / "config", frozen)
    if not (INPUTS / "resolved-before/LunarLander.json").exists():
        info = COMMON.execute([str(BASELINE / "Debug/AnetRLRunner-gui-test.exe"), "[prd109_config]"],
            INPUTS / "capture", cwd=REPO, env={**os.environ, "PRD109_CONFIG_OUTPUT": str(INPUTS)})
        if info["exit_code"]:
            raise RuntimeError("Configuration capture failed")
    for name, (env, rnd) in CASES.items():
        source = (frozen / (env + ".txt")).read_text(encoding="utf-8-sig")
        raw = json.loads((INPUTS / "resolved-before" / (env + ".json")).read_text(encoding="utf-8"))
        chains = re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)", source)
        chain = chains[-1].strip() if chains and env != "GridMaze" else ""
        if rnd:
            chain += (" > " if chain else "") + "run.@rnd"
        values = {"app.$": "app.batchrun > P1", "app.show_error_dialog": "false",
            "app.train_auto_start": "true", "app.eval_panel.auto_start": "false",
            "app.save_agent_on_close": "true", "app.runs_dir": (ROOT / "runs").as_posix(),
            "run.seed": "1", "run.train.num_envs": "8", "backend.torch_num_threads": "1",
            "backend.$": "backend.@deterministic", "backend.deterministic_warn_only": "false",
            "agent.device": "cpu" if env in ("GridMaze", "LunarLander") else "cuda",
            "DefaultDQNAgent.learner.replay_batch_size": "16",
            "DefaultDQNAgent.learner.replay_capacity": "4096",
            "DefaultDQNAgent.learner.update_warmup_steps": "256",
            "DefaultDQNAgent.learner.replay_ratio": "1", "DefaultDQNAgent.learner.use_amp": "false",
            "app.exp_exit_step": "8704", "app.batchrun.exp_exit_step": "8704"}
        if env == "Atari":
            values["E1.game"] = "qbert"
        if rnd:
            values["DefaultDQNAgent.intrinsic.learn_batch_size"] = "8"
            values["DefaultDQNAgent.intrinsic.rnd.warmup_steps"] = "256"
        tags = {match[1] for key in raw if (match := re.match(r"^run\.eval\.\[([^]]+)\]\.", key))}
        values.update({"run.eval_schedule.[" + tag + "].interval": "0" for tag in sorted(tags)})
        values.update({"run.eval_schedule.[eval].interval": "4096",
            "run.eval_schedule.[eval].warmup_count": "0", "run.eval_schedule.[eval].offset": "0",
            "run.eval_schedule.[eval].use_background": "false", "run.eval_schedule.[eval].wait_on_exit": "true",
            "run.eval.[eval].eval_batch_size": "1", "run.eval.[eval].eval_episodes": "1"})
        profile = "run.@prd121_validation"
        chain += (" > " if chain else "") + profile
        lines = ["$include <" + (frozen / "_main.txt").as_posix() + ">",
                 "$include <" + (frozen / (env + ".txt")).as_posix() + ">"]
        lines += [profile + " : " + key + " = " + value for key, value in values.items()]
        write_frozen(INPUTS / (name + ".txt"), "\n".join(lines + ["run.$ = " + chain]) + "\n")
    print("PREPARED", len(CASES), flush=True)


def runs(label, before, cases):
    binary = BASELINE / "Debug/AnetRLRunner.exe" if before else REPO / "apps/runner/bin/Debug/AnetRLRunner.exe"
    results = {}
    for name in cases:
        folder = ROOT / label / name
        suffix = "p121_" + label + "_" + name.lower()
        info = COMMON.execute([str(binary), "--config", str(INPUTS / (name + ".txt")),
                               "app.run_name=run_{t}_" + suffix], folder)
        matches = list((ROOT / "runs").glob("run_*_" + suffix))
        results[name] = info
        if info["exit_code"] == 0 and len(matches) == 1:
            normalized = folder / "normalized.jsonl"
            NORMALIZER.normalize(matches[0], normalized, include_noisy=True)
            results[name].update(run=str(matches[0]), config_sha256=COMMON.digest(INPUTS / (name + ".txt")),
                scalar_sha256=COMMON.digest(normalized), points=normalized.read_bytes().count(b"\n"))
        else:
            results[name]["failure"] = "Runner failed or unique Run not found"
        COMMON.save(ROOT / (label + "-summary.json"), results)
    if any(value.get("failure") or value["exit_code"] for value in results.values()):
        raise RuntimeError("One or more Runs failed; see summary")


def compare(label, reference):
    results = {}
    for name in CASES:
        first, second = (ROOT / run / name / "normalized.jsonl" for run in (reference, label))
        results[name] = {"exact_match": first.read_bytes() == second.read_bytes(),
            "before_sha256": COMMON.digest(first), "after_sha256": COMMON.digest(second)}
    COMMON.save(ROOT / (label + "-comparison.json"), results)
    print(json.dumps(results, indent=2), flush=True)
    if not all(value["exact_match"] for value in results.values()):
        raise RuntimeError("Scalar comparison differs")


def legacy_checkpoints(label):
    # 変更前Runの定数スケーラcheckpointを、新しい実行体の公開起動口から読む。
    baseline = json.loads((ROOT / "baseline-summary.json").read_text(encoding="utf-8"))
    binary = REPO / "apps/runner/bin/Debug/AnetRLRunner.exe"
    results = {}
    for name in ("Atari_rnd", "DropMerge"):
        checkpoint = Path(baseline[name]["run"]) / "agent_close.anet"
        original_hash = COMMON.digest(checkpoint)
        environment = CASES[name][0]
        raw = json.loads((INPUTS / "resolved-before" / (environment + ".json")).read_text(encoding="utf-8"))
        tags = {match[1] for key in raw if (match := re.match(r"^run\.eval\.\[([^]]+)\]\.", key))}
        args = [str(binary), "--config", str(INPUTS / (name + ".txt")),
            "app.run_name=run_{t}_p121_legacy_" + label + "_" + name.lower(),
            "DefaultDQNAgent.auto_load_file=" + checkpoint.as_posix(),
            "DefaultDQNAgent.learner.enabled=false", "app.exp_exit_step=8",
            "app.batchrun.exp_exit_step=8", "app.save_agent_on_close=false"]
        args += ["run.eval_schedule.[" + tag + "].interval=0" for tag in sorted(tags)]
        info = COMMON.execute(args, ROOT / (label + "-legacy") / name)
        info.update(checkpoint=str(checkpoint), checkpoint_sha256=original_hash,
            checkpoint_unchanged=original_hash == COMMON.digest(checkpoint))
        results[name] = info
        COMMON.save(ROOT / (label + "-legacy-summary.json"), results)
    if any(value["exit_code"] or not value["checkpoint_unchanged"] for value in results.values()):
        raise RuntimeError("Legacy checkpoint load failed; see summary")


def check_config(label):
    # PRD072の固定commitのgoldenは保持し、現用17入力の追加キーだけを別に比較する。
    folder = ROOT / (label + "-config")
    folder.mkdir(parents=True, exist_ok=False)
    reference = folder / "reference-config"
    shutil.copytree(REPO / "apps/runner/config", reference)
    for name in ("agent.txt", "metrics_scalar.txt"):
        source = subprocess.check_output(["git", "show", "HEAD:apps/runner/config/" + name], cwd=REPO)
        (reference / name).write_bytes(source)
    defaults = {"intrinsic.value_mode": "combined", "learner.intrinsic.discount_mode": "shared",
        "learner.intrinsic.terminal_mode": "episodic", "learner.intrinsic.loss_weight": "1.0",
        "learner.intrinsic.priority_lambda": "1.0", "learner.intrinsic.extrinsic_bootstrap_mode": "combined",
        "learner.intrinsic.feature_gradient": "merge", "intrinsic.value.head.input_mode": "shared",
        "intrinsic.value.head.use_dueling": "true", "intrinsic.value.head.init.mode": "xavier",
        "intrinsic.value.head.init.manual_gain": "0", "intrinsic.value.head.init.constant_val": "0",
        "intrinsic.value.head.init.trunc_std": "0.02", "intrinsic.value.head.init.trunc_a": "-2",
        "intrinsic.value.head.init.trunc_b": "2", "intrinsic.value.head.projection.noisy.enabled": "false",
        "intrinsic.value.head.projection.noisy.sigma0": "0.5", "intrinsic.reward_scaler.use_dynamic_scaling": "false",
        "intrinsic.reward_scaler.dynamic_mode": "reward_std", "intrinsic.reward_scaler.constant_scale": "1.0",
        "intrinsic.reward_scaler.use_auto_post_scale": "false", "intrinsic.reward_scaler.manual_post_scale": "1.0",
        "intrinsic.reward_scaler.reference_q_std": "30.0", "intrinsic.reward_scaler.epsilon": "1e-8",
        "intrinsic.reward_scaler.use_clipping": "false", "intrinsic.reward_scaler.clip_range": "10",
        "reward_scaler.dynamic_mode": "reward_std"}
    # 古いmanifestの廃止済みRun profileは再導入せず、現用の17構成を固定する。
    cases = [
        ("atari-default", "Atari", ""), ("atari-rnd", "Atari", "run.@rnd"),
        ("atari-skip", "Atari", "run.@skip"), ("atari-rainbow", "Atari", "run.@rainbow_like"),
        ("atari-taiga", "Atari", "run.@rnd > run.@rnd_taiga"),
        ("dropmerge-default", "DropMerge", ""), ("dropmerge-stratified", "DropMerge", "run.@iqn32_stratified"),
        ("dropmerge-qr", "DropMerge", "run.@qr51_control"),
        ("lunarlander-default", "LunarLander", ""), ("lunarlander-repro", "LunarLander", "run.@repro"),
        ("lunarlander-rnd", "LunarLander", "run.@rnd"),
        ("gridmaze-default", "GridMaze", ""), ("gridmaze-rnd", "GridMaze", "run.@rnd"),
        ("cartpole-default", "CartPole", ""), ("gridmaze-muzero", "GridMaze_muzero", ""),
        ("imagecls-default", "ImageCls", ""), ("imagecls-resnet", "ImageCls", "run.@resnet18ish_hr")]
    result = {"reference_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "cases": {}}
    original_binary = REPO / "out/test/Debug/anet-core-base-test.exe"
    binary = folder / original_binary.name
    shutil.copy2(original_binary, binary)
    for name, environment, chain in cases:
        resolved = []
        for version, config in (("before", reference), ("after", REPO / "apps/runner/config")):
            stem = name + "-" + version
            input_file, output = folder / (stem + ".txt"), folder / (stem + ".json")
            lines = ["$include <" + (config / "_main.txt").as_posix() + ">",
                "$include <" + (config / (environment + ".txt")).as_posix() + ">"]
            if chain:
                lines.append("run.$ = " + chain)
            input_file.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            info = COMMON.execute([str(binary), "[prd117-resolve]"], folder / (stem + "-logs"), cwd=REPO,
                env={**os.environ, "PATH": str(original_binary.parent) + os.pathsep + os.environ["PATH"],
                    "ANET_PRD117_RESOLVE_INPUT": str(input_file), "ANET_PRD117_RESOLVE_OUTPUT": str(output)})
            if info["exit_code"]:
                raise RuntimeError("Configuration resolution failed: " + stem)
            resolved.append(json.loads(output.read_text(encoding="utf-8")))
        before, after = resolved
        old, new = before["config"], after["config"]
        roots = {key[:-len("intrinsic.enabled")] for key in old if key.endswith(".intrinsic.enabled")}
        def allowed(key, value):
            if key.endswith(".intrinsic.action_beta_mode") and value == "learner":
                actor_prefix = key[:-len("intrinsic.action_beta_mode")]
                if any(old_key.startswith(actor_prefix) for old_key in old):
                    return True
            for root in roots:
                if not key.startswith(root):
                    continue
                suffix = key[len(root):]
                if suffix in defaults and value == defaults[suffix]:
                    return True
                if re.fullmatch(r"actor\.\[[^]]+\]\.intrinsic\.action_beta_mode", suffix) and value == "learner":
                    return True
            return False
        added = {key: value for key, value in new.items() if key not in old}
        changed = {key: {"before": value, "after": new.get(key)} for key, value in old.items() if new.get(key) != value}
        unexpected = {key: value for key, value in added.items() if not allowed(key, value)}
        metrics_equal = before["scalar_defs"] == after["scalar_defs"]
        passed = bool(added) and not changed and not unexpected and metrics_equal
        result["cases"][name] = {"passed": passed, "added": added, "changed": changed,
            "unexpected_added": unexpected, "metrics_equal": metrics_equal}
        COMMON.save(folder / "comparison.json", result)
        print("CONFIG", name, "PASS" if passed else "FAIL", "added=" + str(len(added)), flush=True)
    if not all(case["passed"] for case in result["cases"].values()):
        raise RuntimeError("Current configuration comparison differs; see comparison.json")


def ensure_measurement_idle():
    # 別Runは停止せず、競合する実行と指定時間外では測定を始めない。
    now = datetime.datetime.now().astimezone()
    if 1 <= now.hour < 9:
        raise RuntimeError("Performance measurement waits until 09:00 local time: " + now.isoformat())
    processes = subprocess.check_output(["powershell", "-NoProfile", "-Command",
        "Get-Process | Where-Object { $_.ProcessName -like 'AnetRLRunner*' -or "
        "$_.ProcessName -in @('anet-core-base-test','cl','ninja') } | "
        "Select-Object ProcessName,Id | ConvertTo-Json -Compress"], text=True).strip()
    if processes:
        raise RuntimeError("Performance measurement waits for other Runs/builds/tests: " + processes)


def benchmark(label, config, rounds):
    if rounds < 3:
        raise ValueError("rounds must be >= 3")
    ensure_measurement_idle()
    folder = ROOT / (label + "-performance")
    folder.mkdir(parents=True, exist_ok=False)
    frozen = folder / "config"
    shutil.copytree(REPO / "apps/runner/config", frozen)
    binary = REPO / "apps/runner/bin" / config / "AnetRLRunner.exe"
    raw = json.loads((INPUTS / "resolved-before/Atari.json").read_text(encoding="utf-8"))
    tags = {match[1] for key in raw if (match := re.match(r"^run\.eval\.\[([^]]+)\]\.", key))}
    source = (frozen / "Atari.txt").read_text(encoding="utf-8-sig")
    chains = re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)", source)
    chain = chains[-1].strip() if chains else ""
    chain += (" > " if chain else "") + "run.@rnd > run.@prd121_performance"
    values = {"app.$": "app.batchrun > P1", "app.show_error_dialog": "false",
        "app.train_auto_start": "true", "app.eval_panel.auto_start": "false", "app.save_agent_on_close": "false",
        "app.runs_dir": (ROOT / "runs").as_posix(), "app.exp_exit_step": "131072", "app.batchrun.exp_exit_step": "131072",
        "run.seed": "1", "run.train.num_envs": "128", "backend.torch_num_threads": "1",
        "backend.$": "backend.@non-deterministic", "agent.device": "cuda", "E1.game": "qbert",
        "DefaultDQNAgent.learner.replay_batch_size": "256", "DefaultDQNAgent.learner.replay_capacity": "65536",
        "DefaultDQNAgent.learner.update_warmup_steps": "4096", "DefaultDQNAgent.learner.replay_ratio": "1",
        "DefaultDQNAgent.intrinsic.learn_batch_size": "32", "DefaultDQNAgent.intrinsic.rnd.warmup_steps": "4096",
        "DefaultDQNAgent.intrinsic.value.head.input_mode": "shared",
        "DefaultDQNAgent.learner.intrinsic.feature_gradient": "merge"}
    values.update({"run.eval_schedule.[" + tag + "].interval": "0" for tag in sorted(tags)})
    common = ["$include <" + (frozen / "_main.txt").as_posix() + ">", "$include <" + (frozen / "Atari.txt").as_posix() + ">"]
    common += ["run.@prd121_performance : " + key + " = " + value for key, value in values.items()]
    for mode in ("combined", "decomposed"):
        write_frozen(folder / (mode + ".txt"), "\n".join(common + [
            "run.@prd121_performance : DefaultDQNAgent.intrinsic.value_mode = " + mode, "run.$ = " + chain]) + "\n")
    report = {"binary": str(binary), "binary_sha256": COMMON.digest(binary), "rounds": rounds,
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
        "conditions": values, "measurements": []}
    for round_index in range(1, rounds + 1):
        for mode in ("combined", "decomposed"):
            ensure_measurement_idle()
            suffix = "p121_perf_" + label + "_" + str(round_index) + "_" + mode
            name = str(round_index) + "-" + mode
            started = datetime.datetime.now().astimezone().isoformat()
            info = COMMON.execute([str(binary), "--config", str(folder / (mode + ".txt")),
                "app.run_name=run_{t}_" + suffix], folder / name)
            runs_found = list((ROOT / "runs").glob("run_*_" + suffix))
            if info["exit_code"] or len(runs_found) != 1:
                COMMON.save(folder / "summary.json", report)
                raise RuntimeError("Performance Run failed: " + name)
            run = runs_found[0]
            points = []
            with (run / "metrics.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("type") == "scalar" and row.get("tag") == "90_perf/90_elapse_hour" and row["step"] >= 16384:
                        points.append((row["step"], row["value"]))
            if len(points) < 2 or points[-1][1] <= points[0][1]:
                raise RuntimeError("Performance elapsed-time window missing: " + name)
            seconds = (points[-1][1] - points[0][1]) * 3600
            measurement = {"round": round_index, "mode": mode, "started": started, "run": str(run),
                "execution": info, "from_exp_step": points[0][0], "to_exp_step": points[-1][0],
                "elapsed_seconds": seconds, "experiences_per_second": (points[-1][0] - points[0][0]) / seconds}
            report["measurements"].append(measurement)
            COMMON.save(folder / "summary.json", report)
            ensure_measurement_idle()
            if COMMON.digest(binary) != report["binary_sha256"]:
                raise RuntimeError("Performance binary changed during measurement")
    report["median_seconds"] = {mode: statistics.median(m["elapsed_seconds"] for m in report["measurements"] if m["mode"] == mode)
        for mode in ("combined", "decomposed")}
    report["decomposed_over_combined"] = report["median_seconds"]["decomposed"] / report["median_seconds"]["combined"]
    COMMON.save(folder / "summary.json", report)
    print("PERFORMANCE", report["median_seconds"], "ratio", report["decomposed_over_combined"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "prepare", "before", "after", "compare", "config", "legacy", "benchmark"))
    parser.add_argument("--config", choices=("Debug", "Release"), default="Release")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--case", nargs="+", choices=tuple(CASES), default=list(CASES))
    args = parser.parse_args()
    if args.mode == "snapshot":
        snapshot(args.config)
    elif args.mode == "prepare":
        prepare()
    elif args.mode == "compare":
        compare(args.label, args.reference)
    elif args.mode == "config":
        check_config(args.label)
    elif args.mode == "legacy":
        legacy_checkpoints(args.label)
    elif args.mode == "benchmark":
        benchmark(args.label, args.config, args.rounds)
    else:
        runs(args.label, args.mode == "before", args.case)


if __name__ == "__main__":
    main()
