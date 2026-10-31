"""PRD104の段0・段1を入力固定して順次実行し、段2は準備だけ行う。"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

from common import OUT, REPO, ROOT, batch_leaves, digest, execute, load, resolve, rows, save, write_config

CHECKPOINTS = {
    "off": "run_20261025-231207_a5_qb_aln_btreps_s1",
    "merge": "run_20261027-004945_a5_qb_aln_t98mrg_s1",
    "stop": "run_20261027-175541_a5_qb_aln_t98stp_s1",
    "lw03": "run_20261028-071900_a5_qb_aln_t98lw03_s1",
    "fallback": "run_20261028-215839_a5_qb_aln_t103spl_s1",
}
ARMS = ("off", "merge", "action_change", "episode_end", "control", "one_step", "episode_end_split")
PREFIX = "DefaultDQNAgent."


def properties(path):
    # 保存したdumpはデータとして読み、選択宣言として実行し直さない。
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def current_chain(env):
    # 長時間の比較中に現用設定が変わっても、固定した選択を使う。
    source = (OUT / "current-config" / (env + ".txt")).read_text(encoding="utf-8-sig")
    return re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)", source)[-1].strip()


def arm_leaves(arm):
    # 腕の差を明示的な葉に閉じ、ネットワークと既存の診断は共有する。
    leaves = {PREFIX + "learner.skip.feature_gradient": "merge"}
    if arm in ("action_change", "episode_end", "control", "one_step", "episode_end_split"):
        leaves.update({PREFIX + "actor.[train].skip.max_length_mode": "single",
                       PREFIX + "actor.[eval].skip.max_length_mode": "single"})
        leaves[PREFIX + "learner.skip.target_window"] = (
            "decision" if arm == "one_step" else "episode_end" if arm.startswith("episode_end") else "action_change")
    if arm == "control":
        leaves[PREFIX + "learner.skip.feature_gradient"] = "stop"
    if arm == "episode_end_split":
        leaves[PREFIX + "learner.skip.feature_gradient"] = "split"
    return leaves


def prepare():
    # 一度解決した入力と実行体を、長時間の比較中は固定する。
    destination = OUT / "experiments"
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(REPO / "apps/runner/config", OUT / "current-config")
    shutil.copy2(ROOT / "arms.txt", OUT / "current-config/prd104-arms.txt")
    runner = REPO / "apps/runner/bin/Release/AnetRLRunner.exe"
    shutil.copy2(runner, destination / "AnetRLRunner.exe")
    template = load("prd103/prepare_experiments.py", "prd104_templates")
    recipes = []
    checkpoint_root = REPO / "apps/runner/workspaces/Atari5-01/runs"
    for arm, run_name in CHECKPOINTS.items():
        run = checkpoint_root / run_name
        checkpoint = run / "agent_close.anet"
        case = destination / ("stage0_" + arm)
        case.mkdir()
        old = properties(run / "config/config_data.txt")
        leaves = batch_leaves(40)
        # PRD103と同じCLI葉の集合を、当該Runの保存値から読み直す。
        source_leaves = template.sample_efficiency()
        for key in source_leaves:
            canonical = PREFIX + key[3:] if key.startswith("A2.") else key
            if key not in old and canonical not in old:
                raise RuntimeError(f"Missing source CLI leaf: {run_name} {key}")
            leaves[key] = old.get(key, old.get(canonical))
        # 元CLIのrun.$が選んだ形状・前処理も葉に展開し、現在の連鎖で同じ重みを読む。
        source_profile_keys = ("AtariEnv.resize_mode", "net.block.[AtariHeadFC512].noisy.enabled",
            PREFIX + "head.projection.noisy.enabled", PREFIX + "net.branch.[main_feature].structure",
            PREFIX + "net.branch.[tau_embedding].structure")
        for key in source_profile_keys:
            leaves[key] = old[key]
        leaves.update({"E1.game": "qbert", PREFIX + "learner.enabled": "false",
                       PREFIX + "auto_load_file": checkpoint.as_posix(), "run.train.num_envs": 1,
                       PREFIX + "learner.skip.feature_gradient": old.get(PREFIX + "learner.skip.feature_gradient", "merge"),
                       PREFIX + "learner.skip.loss_weight": old.get(PREFIX + "learner.skip.loss_weight", "1"),
                       "run.eval_schedule.[eval].interval": 0, "run.eval_schedule.[eval_target].interval": 0,
                       "run.eval_schedule.[greedy].interval": 1000000000,
                       "run.eval_schedule.[greedy].use_background": "false",
                       "run.eval.[greedy].eval_episodes": 200,
                       "metrics.trace.[53_evalg/episode]": "$eval.[greedy] @episode_end $env_step game_score game_len game_frames ram_metric.[1]"})
        if arm != "off":
            leaves.update({PREFIX + "actor.[greedy_single].$": PREFIX + "actor.[greedy]",
                           PREFIX + "actor.[greedy_single].skip.max_length_mode": "single",
                           "run.eval.[greedy_single].$": "run.eval.[greedy]",
                           "run.eval.[greedy_single].actor": "greedy_single",
                           "run.eval_schedule.[greedy_single].$": "run.eval_schedule.[greedy]",
                           "metrics.scalar.[53_evalg/60_single_score_mean]": "$eval.[greedy_single] @session_end $env mean.game_score",
                           "metrics.scalar.[53_evalg/62_single_game_len_mean]": "$eval.[greedy_single] @session_end $env mean.game_len",
                           "metrics.trace.[53_evalg/single_episode]": "$eval.[greedy_single] @episode_end $env_step game_score game_len game_frames ram_metric.[1] ram_metric.[2] ram_metric.[3]"})
        else:
            leaves[PREFIX + "skip.enabled"] = "false"
        config = write_config(case, "Atari", current_chain("Atari"), leaves)
        actual = resolve(config, case)
        assert actual[PREFIX + "learner.enabled"] == "false"
        assert actual["run.eval.[greedy].eval_episodes"] == "200"
        assert all(actual[key] == old[key] for key in source_profile_keys)
        if arm != "off":
            assert actual[PREFIX + "actor.[greedy_single].skip.max_length_mode"] == "single"
        save(case / "source.json", {"run": str(run), "checkpoint": str(checkpoint),
                                     "checkpoint_sha256": digest(checkpoint) if checkpoint.exists() else None,
                                     "source_values": old, "copied_cli_leaves": {k: leaves[k] for k in source_leaves},
                                     "copied_profile_leaves": {k: leaves[k] for k in source_profile_keys}})
        recipes.append({"stage": 0, "arm": arm, "directory": str(case), "config": str(config),
                        "checkpoint_exists": checkpoint.exists()})

    # 段1は同じビルドで7腕を3seedずつ生成する。
    for seed in (1, 2, 3):
        for arm in ARMS:
            case = destination / f"stage1_{arm}_s{seed}"
            case.mkdir()
            chain = "run.@munch" if arm == "off" else current_chain("LunarLander")
            if arm in ("action_change", "episode_end", "control", "episode_end_split"):
                chain += " > run.@skip_aux"
            leaves = batch_leaves(2000000, seed)
            leaves.update(arm_leaves(arm))
            config = write_config(case, "LunarLander", chain, leaves)
            actual = resolve(config, case)
            assert actual[PREFIX + "skip.enabled"] == ("false" if arm == "off" else "true")
            assert actual["run.train.num_envs"] == "256"
            assert actual[PREFIX + "skip.max_length"] == "4"
            assert actual[PREFIX + "learner.skip.feature_gradient"] == leaves[PREFIX + "learner.skip.feature_gradient"]
            recipes.append({"stage": 1, "arm": arm, "seed": seed, "directory": str(case), "config": str(config)})

    # 段2は条件付きの入力だけを残し、runの対象に含めない。
    for arm in ("action_change", "episode_end"):
        for seed in (1, 2):
            case = destination / f"stage2_{arm}_s{seed}"
            case.mkdir()
            leaves = batch_leaves(50000000, seed)
            leaves.update(template.sample_efficiency())
            leaves.update(arm_leaves(arm))
            leaves["E1.game"] = "qbert"
            config = write_config(case, "Atari", template.SAMPLE_CHAIN + " > run.@skip_aux", leaves)
            resolve(config, case)
            recipes.append({"stage": 2, "arm": arm, "seed": seed, "directory": str(case), "config": str(config),
                            "condition": "PRD104段1の結果による。教材閾値と見込みありの判定はPRD104 §7。今回実行しない。"})
    for recipe in recipes:
        recipe["config_sha256"] = digest(Path(recipe["config"]))
    manifest = {"runner_sha256": digest(destination / "AnetRLRunner.exe"),
                "inputs": {str(p.relative_to(OUT)): digest(p) for p in sorted((OUT / "current-config").rglob("*.txt"))},
                "arms_sha256": digest(ROOT / "arms.txt"), "recipes": recipes}
    save(destination / "recipes.json", manifest)
    print("PREPARED", len(recipes), "recipes", flush=True)


def runners():
    result = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW)
    result.check_returncode()
    return [row for row in csv.reader(io.StringIO(result.stdout)) if row and row[0].lower().startswith("anetrlrunner")]


def wait_idle():
    # 既存Runの終了を待つ。待つために停止や再起動は行わない。
    while True:
        active = runners()
        if not active:
            time.sleep(15)
            if not runners():
                return
        save(OUT / "experiments/waiting.json", {"active": active, "time": time.time()})
        time.sleep(30)


def metric_values(run, tag):
    return [point["value"] for point in rows(run) if point.get("type") == "scalar"
            and point["tag"] == tag and math.isfinite(point["value"])]


def episode_summary(run, tag):
    # 採用された200ゲームから、得点と面の進みの両方を満たす突破率を作る。
    episodes = [p["data"] for p in rows(run) if p.get("type") == "trace" and p["tag"] == tag]
    assert len(episodes) == 200, (tag, len(episodes))
    passed = sum(p["game_score"] >= 29000 and p["ram_metric.[1]"] >= 8 for p in episodes)
    return {"episodes": len(episodes), "breakthrough_rate": passed / len(episodes),
            "score_mean": statistics.mean(p["game_score"] for p in episodes),
            "game_len_mean": statistics.mean(p["game_len"] for p in episodes)}


def run_all(through_stage=1):
    manifest = json.loads((OUT / "experiments/recipes.json").read_text(encoding="utf-8"))
    runner = OUT / "experiments/AnetRLRunner.exe"
    assert digest(runner) == manifest["runner_sha256"]
    runtime_manifest = json.loads((OUT / "current-runtime-manifest.json").read_text(encoding="utf-8"))
    for name, expected in runtime_manifest.items():
        assert digest(OUT / name) == expected, name
    assert digest(OUT / "current-release-bin/AnetRLRunner.exe") == manifest["runner_sha256"]
    for name, expected in manifest["inputs"].items():
        assert digest(OUT / name) == expected, name
    results = []
    # 保存済みの結果は再利用し、独立した未実施Runを順次進める。
    for recipe in manifest["recipes"]:
        if recipe["stage"] > through_stage:
            continue
        if recipe["stage"] == 0 and recipe["arm"] == "fallback" and any(
                r["stage"] == 0 and r["arm"] not in ("off", "fallback") and r.get("passed") for r in results):
            continue
        case = Path(recipe["directory"])
        if (case / "result.json").exists():
            results.append(json.loads((case / "result.json").read_text(encoding="utf-8")))
            continue
        wait_idle()
        record = dict(recipe)
        try:
            assert digest(Path(recipe["config"])) == recipe["config_sha256"], recipe["config"]
            if recipe["stage"] == 0:
                source = json.loads((case / "source.json").read_text(encoding="utf-8"))
                checkpoint = Path(source["checkpoint"])
                record["checkpoint_sha256"] = digest(checkpoint) if checkpoint.exists() else None
            # 凍結exeと同じビルドのDLLを使う。DLLはビルド中に更新しない。
            import os
            os.environ["PATH"] = str(OUT / "current-release-bin") + os.pathsep + os.environ["PATH"]
            run = execute(case, Path(recipe["config"]),
                          "t104_" + case.name, binary=runner)
            record["run"] = str(run)
            if recipe["stage"] == 0:
                full = metric_values(run, "53_evalg/10_game_score_mean")
                assert len(full) == 1, full
                record["full_score"] = full[0]
                record["full_episodes"] = episode_summary(run, "53_evalg/episode")
                if recipe["arm"] != "off":
                    single = metric_values(run, "53_evalg/60_single_score_mean")
                    assert len(single) == 1, single
                    record["single_score"] = single[0]
                    record["single_episodes"] = episode_summary(run, "53_evalg/single_episode")
            else:
                policy = metric_values(run, "21_eval/02_policy_reward")
                assert policy, "No policy reward"
                record["auc"] = statistics.mean(policy)
                ema = [p["step"] for p in rows(run) if p.get("type") == "scalar"
                       and p["tag"] == "21_eval/04_policy_reward_ema" and p["value"] >= 200]
                record["first_ema200"] = min(ema) if ema else None
                record["policy_points"] = len(policy)
            record["passed"] = True
        except Exception as error:
            record.update(passed=False, error=str(error))
        save(case / "result.json", record)
        results.append(record)
        save(OUT / "experiments/results.json", results)
        summarize(results)
    summarize(results)
    if (through_stage == 1 and sum(r["stage"] == 1 and r.get("passed", False) for r in results) != 21) or not any(
            r["stage"] == 0 and r["arm"] != "off" and r.get("passed") for r in results) or not any(
            r["stage"] == 0 and r["arm"] == "off" and r.get("passed") for r in results):
        raise SystemExit("Some experiments failed; see results.json")


def summarize(results):
    # 同じcheckpointの評価差を無効との差で割り、分母が正でない場合はgを作らない。
    stage0 = [r for r in results if r["stage"] == 0 and r.get("passed")]
    off = next((r["full_score"] for r in stage0 if r["arm"] == "off"), None)
    for result in stage0:
        if result["arm"] == "off" or off is None:
            continue
        denominator = off - result["full_score"]
        result["g_denominator"] = denominator
        result["g"] = (result["single_score"] - result["full_score"]) / denominator if denominator > 0 else None
        result["reading"] = ("評価では無効に遅れていない" if denominator <= 0 else
                             "固定が主因" if result["g"] >= 0.5 else
                             "固定とQの遅れ" if result["g"] >= 0.2 else
                             "Qの遅れ" if result["g"] > -0.2 else "固定は評価では得")
    # 段1のAUCはseed単位で保ち、3本そろった腕だけを比較する。
    arms = {}
    for arm in ARMS:
        group = [r for r in results if r["stage"] == 1 and r["arm"] == arm and r.get("passed")]
        values = [r["auc"] for r in group]
        arms[arm] = {"count": len(values), "aucs": values, "seeds": [r["seed"] for r in group],
                     "mean": statistics.mean(values) if values else None,
                     "min": min(values) if values else None, "max": max(values) if values else None}
    comparisons = {}
    for arm in ARMS:
        if arms[arm]["count"] != 3:
            continue
        comparisons[arm] = {}
        for reference in ("off", "merge", "episode_end"):
            a, b = arms[arm], arms[reference]
            if b["count"] == 3:
                comparisons[arm][reference] = {"above": a["min"] > b["max"],
                    "at_least": a["mean"] >= b["min"], "overlap": max(a["min"], b["min"]) <= min(a["max"], b["max"])}
    # 読めた本数に応じてPRDの合意した多数決を適用する。
    readings = [r["reading"] for r in stage0 if "reading" in r and r["g"] is not None]
    from collections import Counter
    counts = Counter(readings)
    aggregate = "読みが定まらない"
    interpreted = [r for r in stage0 if "reading" in r]
    if len(interpreted) == 1:
        aggregate = "1本だけの読み: " + interpreted[0]["reading"]
    elif counts and counts.most_common(1)[0][1] >= 2:
        aggregate = counts.most_common(1)[0][0]
    save(OUT / "experiments/summary.json", {"stage0": stage0, "stage0_reading": aggregate,
                                           "stage1": arms, "comparisons": comparisons,
                                           "stage1_complete": all(a["count"] == 3 for a in arms.values()),
                                           "stage2_executed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--through-stage", type=int, choices=(0, 1), default=1,
                        help="Last stage to execute; stage2 is never executed")
    args = parser.parse_args()
    if args.action != "run" and args.through_stage != 1:
        parser.error("--through-stage is only supported with run")
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run_all(args.through_stage)
    else:
        summarize(json.loads((OUT / "experiments/results.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
