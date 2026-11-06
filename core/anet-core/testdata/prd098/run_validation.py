"""PRD098: short sequential Runs, exact comparison, and unexecuted long-run recipes."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / ".scratch/prd098/config-20261024-r2"
ENVS = ("GridMaze", "LunarLander", "Atari", "DropMerge")
RUNNER = REPO / "apps/runner/bin/Debug/AnetRLRunner.exe"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def settings(env, before=False, enabled=False, long=False, seed=1):
    root = OUT / "baseline-config" if before else REPO / "apps/runner/config"
    raw = json.loads((OUT / "resolved-before" / (env + ".json")).read_text(encoding="utf-8"))
    source = (root / (env + ".txt")).read_text(encoding="utf-8-sig")
    chains = re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)", source)
    chain = chains[-1].strip() if chains and env != "GridMaze" else ""
    overrides = ["app.$ = app.batchrun > P1", "app.show_error_dialog = false",
        "app.train_auto_start = true", "app.eval_panel.auto_start = false", "app.save_agent_on_close = false",
        f"app.runs_dir = {(OUT / 'runs').as_posix()}", f"run.seed = {seed}",
        "backend.torch_num_threads = 1"]
    if env == "Atari":
        overrides += ["E1.game = qbert"]
    if env == "DropMerge":
        # 保存済み nzrep の現行 Noisy 構成。旧 Run artifact は編集しない。
        overrides += ["net.block.[HeadFC1024].noisy.enabled = true", "DefaultDQNAgent.head.projection.noisy.enabled = true",
            "DefaultDQNAgent.nn_forward.[act_lane].noisy.mode = sample",
            "DefaultDQNAgent.nn_forward.[act_lane].noisy.hold = call",
            "DefaultDQNAgent.nn_forward.[act_lane].noisy.share = sample",
            "DefaultDQNAgent.actor.[train].nn_forward_key = act_lane",
            "DefaultDQNAgent.learner.munchausen.enabled = true",
            "DefaultDQNAgent.learner.munchausen.log_policy_mode = target",
            "DefaultDQNAgent.learner.munchausen.entropy_tau = 0.03",
            "DefaultDQNAgent.learner.use_double_dqn = false"]
    metric_chains = re.findall(r"(?m)^metrics\.scalar\.\$\s*=\s*([^#\n]+)", source)
    metrics = metric_chains[-1].strip() + " > metrics.scalar.@noisy"
    if enabled:
        # 有効化と指標はenv所有の一つのRun profileで選ぶ。変種はexecuteのCLI引数へ分離する。
        chain += (" > " if chain else "") + "run.@skip"
        metrics += " > metrics.scalar.@skip"
    else:
        overrides += ["DefaultDQNAgent.skip.enabled = false"]
    if long:
        for tag in (("eval_target", "eval_target_greedy") if env == "DropMerge" else ("eval",)):
            if enabled:
                for suffix, key in (("decision", "skip_decision_rate"), ("length", "skip_length_mean")):
                    metric_tag = "50_agent_skip/" + tag.replace("eval_target", "et") + "_" + suffix
                    overrides += [f"metrics.scalar.[{metric_tag}] = {key} @train $action_info $eval.[{tag}]"]
            if tag == "eval_target_greedy":
                overrides += ["run.eval.[eval_target_greedy].actor = eval_target_greedy",
                    "run.eval.[eval_target_greedy].eval_batch_size = 10", "run.eval.[eval_target_greedy].eval_episodes = 10",
                    "run.eval_schedule.[eval_target_greedy].interval = 10000",
                    "DefaultDQNAgent.actor.[eval_target_greedy].network = target",
                    "DefaultDQNAgent.actor.[eval_target_greedy].policy.policy_type = Greedy",
                    "DefaultDQNAgent.actor.[eval_target_greedy].nn_forward_key = mu_only"]
    overrides += ["metrics.scalar.$ = " + metrics]
    if not long:
        tags = sorted({match[1] for key in raw if (match := re.match(r"^run\.eval\.\[([^]]+)\]\.", key))})
        overrides += [f"run.eval_schedule.[{tag}].interval = 0" for tag in tags]
        overrides += ["backend.$ = backend.@deterministic", "backend.deterministic_warn_only = false",
            "agent.device = " + ("cpu" if env in ("GridMaze", "LunarLander") else "cuda"),
            "run.train.num_envs = 8", "DefaultDQNAgent.learner.replay_batch_size = 16",
            "DefaultDQNAgent.learner.replay_capacity = 4096", "DefaultDQNAgent.learner.update_warmup_steps = 256",
            "DefaultDQNAgent.learner.replay_ratio = 1", "DefaultDQNAgent.learner.use_amp = false",
            "app.exp_exit_step = 8704", "app.batchrun.exp_exit_step = 8704",
            "metrics.scalar.[90_perf/98_prd098_time] = elapse_hour @learn $learn_step $runner interval:32"]
    profile = "run.@prd098_validation"
    chain += (" > " if chain else "") + profile
    return [f"$include <{root.as_posix()}/_main.txt>", f"$include <{root.as_posix()}/{env}.txt>"] + [
        profile + " : " + line for line in overrides] + ["run.$ = " + chain]


def execute(name, env, before=False, enabled=False, performance=False, overrides=()):
    destination = OUT / name
    destination.mkdir(parents=True, exist_ok=True)
    binary = OUT / "baseline-bin/AnetRLRunner.exe" if before else RUNNER
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    record = destination / "execution.json"
    lines = settings(env, before, enabled)
    if performance:
        # 実運用の B/N、BF16、prefetch を保ち、warmup後の64..256更新を比較する。
        raw = json.loads((OUT / "resolved-before" / "Atari.json").read_text())
        for key in ("run.train.num_envs", "DefaultDQNAgent.learner.replay_batch_size", "DefaultDQNAgent.learner.use_amp",
                    "DefaultDQNAgent.learner.use_amp_bf16"):
            lines.append("run.@prd098_validation : " + key + " = " + raw[key])
        batch = int(raw["DefaultDQNAgent.learner.replay_batch_size"].replace(",", ""))
        budget = 256 + 320 * batch
        lines += [f"run.@prd098_validation : app.exp_exit_step = {budget}",
                  f"run.@prd098_validation : app.batchrun.exp_exit_step = {budget}"]
    cfg = destination / "run.txt"
    config_text = "\n".join(lines) + "\n"
    config_digest = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    overrides_digest = hashlib.sha256(json.dumps(list(overrides)).encode("utf-8")).hexdigest()
    # 現在のincludeはconfigルート内のtxt。未参照分も含めて保守的にfingerprintを取る。
    config_root = OUT / "baseline-config" if before else REPO / "apps/runner/config"
    if not config_root.is_dir(): raise RuntimeError("Missing config input directory: " + str(config_root))
    inputs_hash = hashlib.sha256()
    for path in sorted(config_root.rglob("*.txt")):
        inputs_hash.update(path.relative_to(config_root).as_posix().encode("utf-8") + b"\0")
        inputs_hash.update(hashlib.sha256(path.read_bytes()).digest())
    config_inputs_digest = inputs_hash.hexdigest()
    if record.exists():
        saved = json.loads(record.read_text(encoding="utf-8"))
        if saved.get("exit_code") == 0 and saved.get("binary_sha256") == digest and saved.get("config_sha256") == config_digest and saved.get("config_inputs_sha256") == config_inputs_digest and saved.get("overrides_sha256") == overrides_digest:
            print("REUSE", name, flush=True)
            return Path(saved["run"])
        raise RuntimeError("Existing run record needs a fresh --label: " + str(record))
    cfg.write_text(config_text, encoding="utf-8", newline="\n")
    suffix = "prd098_" + name
    args = [str(binary), "--config", str(cfg), "app.run_name=run_{t}_" + suffix] + list(overrides)
    info = {"binary_sha256": digest, "config_sha256": config_digest, "config_inputs_sha256": config_inputs_digest, "overrides_sha256": overrides_digest, "arguments": args}
    save(record, info)
    print("START", name, flush=True)
    start = time.monotonic()
    with (destination / "launcher.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(args, cwd=REPO / "apps/runner", stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        info["pid"] = process.pid
        save(record, info)
        code = process.wait()
    runs = list((OUT / "runs").glob("run_*_" + suffix))
    info.update(exit_code=code, seconds=time.monotonic() - start, run=str(runs[0]) if len(runs) == 1 else None)
    save(record, info)
    print("END", name, code, round(info["seconds"], 1), flush=True)
    if code or len(runs) != 1: raise RuntimeError("Run failed: " + name)
    return runs[0]


def rows(run):
    path = run / "metrics.jsonl"
    opener = open
    if not path.exists(): path, opener = run / "metrics.jsonl.gz", gzip.open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream]


def compare(label, env):
    spec = importlib.util.spec_from_file_location("prd084", REPO / "core/anet-core/testdata/prd084/compare_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    normalized = []
    for side in ("before", label):
        folder = OUT / (side + "_" + env)
        record = json.loads((folder / "execution.json").read_text())
        path = folder / "normalized.jsonl"
        module.normalize(record["run"], str(path), True)
        normalized.append(path.read_bytes())
    report = {"exact_match": normalized[0] == normalized[1], "points": normalized[1].count(b"\n"),
              "sha256": hashlib.sha256(normalized[1]).hexdigest()}
    save(OUT / (label + "_" + env) / "comparison.json", report)
    if not report["exact_match"]: raise RuntimeError("Learning metrics mismatch: " + env)


def prepare_long(checkpoint):
    commands = []
    for stage, env, budget in ((1, "GridMaze", 500000), (2, "LunarLander", 2000000), (3, "Atari", 10000000), (4, "DropMerge", 50000000)):
        if stage == 4 and checkpoint is None: continue
        arms = ("off", "stop", "merge", "action_in") if stage == 1 else ("off", "stop", "action_in") if stage == 3 else ("off", "stop") if stage == 4 else ("off", "stop", "merge")
        for seed in ((1,) if stage in (3, 4) else (1, 2, 3)):
            for arm in arms:
                lines = settings(env, enabled=arm != "off", long=True, seed=seed)
                lines += [f"run.@prd098_validation : app.exp_exit_step = {budget}",
                          f"run.@prd098_validation : app.batchrun.exp_exit_step = {budget}"]
                if stage == 1: lines += ["run.@prd098_validation : GridMazeEnv.width = 6", "run.@prd098_validation : GridMazeEnv.height = 10"]
                if stage == 4: lines += [f"run.@prd098_validation : DefaultDQNAgent.auto_load_file = {checkpoint.as_posix()}"]
                path = OUT / "long-config" / f"stage{stage}_{arm}_seed{seed}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
                # 変種は設定ファイルへ埋め込まず、実行時の葉として記録する。
                overrides = ["DefaultDQNAgent.learner.skip.feature_gradient=merge"] if arm == "merge" else []
                if arm == "action_in":
                    overrides += ["DefaultDQNAgent.skip.network_mode=action_in",
                                  "DefaultDQNAgent.skip.net.branch.[skip_feature].bind=skip_context, action"]
                commands.append({"stage": stage, "arm": arm, "seed": seed,
                    "arguments": [str(RUNNER), "--config", str(path)] + overrides})
    save(OUT / "long-config/commands.json", commands)
    save(OUT / "long-config/status.json", {"executed": False, "stages_prepared": [1, 2, 3] + ([4] if checkpoint else []),
        "stage4": "Prepared" if checkpoint else "Pass --checkpoint with the selected 50M primary checkpoint to prepare the two branches."})


def main():
    global RUNNER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "after", "smoke", "performance", "performance-off", "prepare-long", "compare"))
    parser.add_argument("--env", choices=ENVS, nargs="+", default=list(ENVS))
    parser.add_argument("--label", default="after")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Runner CLI key=value override; repeat for variants")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--build-config", choices=("Debug", "RelWithDebInfo", "Release"), default="Debug",
                        help="Current Runner build configuration; the saved baseline remains Debug")
    parser.add_argument("--order", choices=("off-on-on-off", "on-off-off-on"), default="off-on-on-off")
    args = parser.parse_args()
    if args.stage in ("before", "after", "compare", "performance-off") and args.build_config != "Debug":
        parser.error("The saved baseline is Debug; this stage requires --build-config Debug")
    RUNNER = REPO / "apps/runner/bin" / args.build_config / "AnetRLRunner.exe"
    if args.stage == "prepare-long":
        if args.checkpoint and not args.checkpoint.is_file(): parser.error("Checkpoint does not exist")
        prepare_long(args.checkpoint)
        return
    if args.stage not in ("before", "compare"):
        # 検証中の再ビルドから切り離し、同じ呼び出しの全条件に同じバイナリを使う。
        folder = OUT / (args.label + "-bin")
        folder.mkdir(exist_ok=True)
        source = RUNNER
        RUNNER = folder / source.name
        if RUNNER.exists():
            if RUNNER.read_bytes() != source.read_bytes(): raise RuntimeError("Binary changed; use a fresh --label")
        else:
            shutil.copy2(source, RUNNER)
            for dll in source.parent.glob("*.dll"): os.link(dll, folder / dll.name)
    errors = []
    for env in (("Atari",) if args.stage in ("performance", "performance-off") else args.env):
        try:
            if args.stage == "compare": compare(args.label, env)
            elif args.stage in ("performance", "performance-off"):
                # 異なる順序でも有効・無効の実際の腕から平均を取る。
                # OFFの変更前後比較は同じ条件・保存済み旧バイナリでABBAとする。
                off_comparison = args.stage == "performance-off"
                arms = (False, True, True, False) if off_comparison or args.order == "off-on-on-off" else (True, False, False, True)
                times = []
                for arm in arms:
                    run = execute(f"{args.label}_perf_{len(times)}", env, before=off_comparison and not arm,
                                  enabled=not off_comparison and arm, performance=True, overrides=args.overrides)
                    points = {int(row["step"]): row["value"] for row in rows(run) if row.get("tag") == "90_perf/98_prd098_time"}
                    times.append((points[256] - points[64]) * 3600 / 192)
                    print("MEASURE", len(times) - 1, "seconds_per_update", times[-1], flush=True)
                ratio = sum(value for value, arm in zip(times, arms) if arm) / sum(value for value, arm in zip(times, arms) if not arm)
                order = [("after" if arm else "before") if off_comparison else ("on" if arm else "off") for arm in arms]
                report = {"seconds_per_update": times, "order": order, "ratio": ratio,
                          "comparison": "after_off/before_off" if off_comparison else "on/off", "build_config": args.build_config}
                if not off_comparison: report["pass"] = ratio <= 1.3
                save(OUT / (args.label + "_performance.json"), report)
                if not off_comparison and ratio > 1.3: raise RuntimeError(f"Performance ratio {ratio:.3f} exceeds 1.3")
            else:
                side = "before" if args.stage == "before" else args.label
                name = side + ("_smoke_" if args.stage == "smoke" else "_") + env
                run = execute(name, env, before=args.stage == "before", enabled=args.stage == "smoke", overrides=args.overrides)
                if args.stage == "after": compare(side, env)
                if args.stage == "smoke":
                    values = [row for row in rows(run) if row.get("tag", "").startswith("50_agent_skip/") and row.get("value") is not None]
                    tags = sorted({row["tag"] for row in values})
                    if len(tags) < 9 or not all(math.isfinite(row["value"]) for row in values): raise RuntimeError("Missing or nonfinite skip metrics")
                    definitions = json.loads((run / "json/metrics.scalar.defs.json").read_text(encoding="utf-8-sig"))["data"]
                    required = {"01_decision_rate", "02_skip_length_mean", "03_skip_length_rate.[01]", "04_skip_q_gap",
                                "11_loss", "12_td_abs_mean", "13_sample_k_mean", "14_kmax_mean", "15_grad_norm"}
                    required = {"50_agent_skip/" + suffix for suffix in required}
                    if not required.issubset(definitions) or not required.issubset(tags):
                        raise RuntimeError("Missing required skip metric definition or values")
                    save(OUT / name / "skip-metrics.json", {"tags": tags, "points": len(values), "definitions_checked": sorted(required)})
        except Exception as error:
            errors.append(env + ": " + str(error))
            print("ERROR", errors[-1], flush=True)
    save(OUT / (args.stage + "_" + args.label + "_summary.json"), {"errors": errors})
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
