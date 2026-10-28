"""PRD103 の現行設定 smoke と Release 性能比較。ログと解決済み入力を保存する。"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_validation import module, save, ROOT, REPO

OUT = REPO / ".scratch/prd103/current"
PROFILE = "run.@prd098_validation : "


def resolve(config, destination, name):
    target = destination / (name + ".json")
    environment = dict(os.environ, ANET_PRD103_INPUT=str(config), ANET_PRD103_OUTPUT=str(target))
    with (destination / (name + ".log")).open("w", encoding="utf-8") as log:
        subprocess.run([str(REPO / "core/anet-core/bin/Debug/anet-core-test.exe"), "[prd103-resolve]"],
            cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(target.read_text(encoding="utf-8"))


def rows(run):
    path = run / "metrics.jsonl"
    opener = open
    if not path.exists():
        path = path.with_suffix(".jsonl.gz")
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line in stream:
            yield json.loads(line)


def execute(stage, label, env, mode=None, prepare=False):
    name = f"{label}_{env}" + ("_" + mode if mode else "")
    destination = OUT / name
    destination.mkdir(parents=True, exist_ok=False)
    previous = module(ROOT.parent / "prd098/run_validation.py", "prd098")
    previous.OUT = ROOT
    lines = previous.settings(env, enabled=True)
    # 古いsession名の一覧を流用せず、今回の入力を解決して停止対象を決める。
    lines = [line.replace((ROOT / "runs").as_posix(), (OUT / "runs").as_posix())
        for line in lines if "run.eval_schedule.[" not in line]
    if stage == "performance":
        lines[-1] = lines[-1].replace(" > run.@skip", " > run.@btrnet > run.@skip")
        for key, value in {"run.train.num_envs": 128, "DefaultDQNAgent.learner.replay_batch_size": 256,
            "DefaultDQNAgent.learner.use_amp": "true", "DefaultDQNAgent.learner.use_amp_bf16": "true",
            "DefaultDQNAgent.learner.use_rb_prefetch": "true", "app.exp_exit_step": 82176,
            "app.batchrun.exp_exit_step": 82176, "DefaultDQNAgent.learner.skip.feature_gradient": mode}.items():
            lines.append(PROFILE + f"{key} = {value}")
    else:
        lines += [PROFILE + "run.eval.[greedy].eval_batch_size = 1",
            PROFILE + "run.eval.[greedy].eval_episodes = 1",
            PROFILE + "run.eval_schedule.[greedy].use_background = false"]
        if env == "Atari":
            lines += [PROFILE + "run.eval.[greedy].env.max_episode_frames = 400"]
    config = destination / "run.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    raw = resolve(config, destination, "sessions")
    slots = sorted({m[1] for key in raw if (m := re.match(r"run\.eval\.\[([^]]+)\]\.", key))})
    lines += [PROFILE + f"run.eval_schedule.[{slot}].interval = " +
        ("256" if stage == "smoke" and slot == "greedy" else "0") for slot in slots]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    resolved = resolve(config, destination, "resolved")
    assert resolved["DefaultDQNAgent.skip.enabled"] == "true"
    if stage == "performance":
        for key, expected in {"run.train.num_envs": "128", "DefaultDQNAgent.learner.replay_batch_size": "256",
            "DefaultDQNAgent.learner.use_amp": "true", "DefaultDQNAgent.learner.use_amp_bf16": "true",
            "DefaultDQNAgent.learner.use_rb_prefetch": "true", "DefaultDQNAgent.learner.skip.feature_gradient": mode}.items():
            assert resolved[key] == expected, (key, resolved[key])
    binary = REPO / "apps/runner/bin" / ("Release" if stage == "performance" else "Debug") / "AnetRLRunner.exe"
    suffix = "t103_" + name
    args = [str(binary), "--config", str(config), "app.run_name=run_{t}_" + suffix]
    record = {"arguments": args, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "evaluation_slots": slots}
    save(destination / "execution.json", record)
    if prepare:
        print("PREPARED", name, flush=True)
        return record
    print("START", name, flush=True)
    started = time.monotonic()
    with (destination / "launcher.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(args, cwd=REPO / "apps/runner", stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
    runs = list((OUT / "runs").glob("run_*_" + suffix))
    record.update(exit_code=result.returncode, seconds=time.monotonic() - started,
        run=str(runs[0]) if len(runs) == 1 else None)
    save(destination / "execution.json", record)
    if result.returncode or len(runs) != 1:
        raise RuntimeError(f"Run failed: {name}")
    return audit(stage, env, destination, record)


def audit(stage, env, destination, record):
    run = Path(record["run"])
    resolved = json.loads((destination / "resolved.json").read_text(encoding="utf-8"))
    scalars = [row for row in rows(run) if row.get("type") == "scalar"]
    if stage == "performance":
        timing = {row["step"]: row["value"] for row in scalars if row["tag"] == "90_perf/98_prd098_time"}
        record["seconds_per_update"] = (timing[256] - timing[64]) * 3600 / 192
    else:
        tags = ["50_agent_skip/" + suffix for suffix in ("21_greedy_decision_rate", "22_greedy_length_mean",
            "23_greedy_length_rate.[01]", "03_skip_length_rate.[01]")]
        if env != "Atari":
            tags += ["21_eval/09_greedy_reward"]
        definitions = [row for row in rows(run) if row.get("tag") == "metrics.scalar.defs"]
        assert definitions, "Missing metrics.scalar.defs"
        record["metrics"] = {}
        for tag in tags:
            points = [row for row in scalars if row["tag"] == tag]
            assert points and any(row["value"] is not None for row in points), tag
            assert tag in json.dumps(definitions), ("definition", tag)
            if tag.startswith(("50_agent_skip/21", "50_agent_skip/22", "50_agent_skip/23")):
                assert definitions[-1]["data"][tag]["runner"] == "train", definitions[-1]["data"][tag]
                assert definitions[-1]["data"][tag]["step_axis"] == "exp_step"
            record["metrics"][tag] = {"count": len(points), "first": points[0], "last": points[-1]}
        text = (run / "config/config_data.txt").read_text(encoding="utf-8-sig")
        assert "run.eval.[greedy]" in text and "DefaultDQNAgent.actor.[greedy]" in text
        assert "greedy_dist" not in text
        if env == "GridMaze":
            assert resolved["DefaultDQNAgent.actor.[train].skip.policy.policy_type"] == "EpsilonGreedy"
        module(ROOT.parent / "prd084/compare_metrics.py", "normalize").normalize(str(run), str(destination / "normalized.jsonl"), True)
    save(destination / "execution.json", record)
    print("END", destination.name, round(record["seconds"], 1), record.get("seconds_per_update"), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "performance"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--audit", action="store_true", help="検証だけを既存Runへ再適用する")
    parser.add_argument("--prepare", action="store_true", help="入力を生成・解決し、学習は開始しない")
    parser.add_argument("--env", nargs="+", choices=("GridMaze", "LunarLander", "Atari"))
    args = parser.parse_args()
    if args.audit and args.prepare:
        parser.error("--audit and --prepare are mutually exclusive")
    if args.stage == "performance" and not (args.audit or args.prepare):
        # 学習やbuildとの競合を混ぜた比を受入結果として残さない。
        active = subprocess.check_output(["powershell", "-NoProfile", "-Command",
            "Get-Process | Where-Object { $_.ProcessName -match '^(AnetRLRunner.*|anet-core-test|cl|ninja|cmake)$' } | Select-Object -ExpandProperty ProcessName"], text=True).strip()
        if active:
            raise RuntimeError("Performance measurement blocked by active processes: " + active)
    results, errors = [], []
    cases = [(env, None) for env in (args.env or ("GridMaze", "LunarLander", "Atari"))] if args.stage == "smoke" else [
        ("Atari", mode) for mode in ("merge", "split", "split", "merge")]
    for i, (env, mode) in enumerate(cases):
        try:
            if args.audit:
                name = f"{args.label}_{i}_{env}" + ("_" + mode if mode else "")
                destination = OUT / name
                results.append(audit(args.stage, env, destination,
                    json.loads((destination / "execution.json").read_text(encoding="utf-8"))))
            else:
                results.append(execute(args.stage, f"{args.label}_{i}", env, mode, args.prepare))
        except Exception as error:
            errors.append(str(error))
            print("FAIL", str(error), flush=True)
    report = {"results": results, "errors": errors}
    if args.stage == "performance" and not errors and not args.prepare:
        report["split_merge_ratio"] = sum(results[i]["seconds_per_update"] for i in (1, 2)) / sum(
            results[i]["seconds_per_update"] for i in (0, 3))
    save(OUT / (args.label + "-summary.json"), report)
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
