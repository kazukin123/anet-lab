"""PRD106 の集中面・通過率・bootstrap を評価の episode trace から計算する。"""
from collections import Counter
import statistics

import numpy as np


def ended_levels(episodes):
    # クリア数8だけでは面8突破にならず、得点の条件も満たしたときだけ9へ進む。
    return np.array([9 if item["ram_metric.[1]"] >= 8 and item["game_score"] >= 29000
                     else min(8, int(item["ram_metric.[1]"]) + 1) for item in episodes])


def summarize(episodes, excluded=()):
    ended = ended_levels(episodes)
    levels, concentrated = [], []
    passage = 1.0
    for level in range(1, 9):
        reached = int((ended >= level).sum())
        failures = int((ended == level).sum())
        # 元の G の同じ得点の本数で決め、bootstrap の各回では選び直さない。
        scores = Counter(item["game_score"] for item, final in zip(episodes, ended) if final == level)
        largest = max(scores.values(), default=0)
        if reached and largest / reached >= 0.1:
            concentrated.append(level)
        rate = failures / reached if reached else None
        levels.append({"level": level, "reached": reached, "ended": failures, "rate": rate,
                       "largest_same_score_count": largest})
        if level not in excluded:
            if rate is None:
                raise ValueError(f"No games reached level {level}; passage is undefined")
            passage *= 1 - rate
    return {"episodes": len(episodes), "breakthrough": float((ended == 9).mean()),
            "score_mean": statistics.mean(item["game_score"] for item in episodes),
            "passage": passage, "concentrated": concentrated, "excluded": list(excluded), "levels": levels}


def interval(values):
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap statistic is undefined; cannot calculate its confidence interval")
    return np.quantile(values, (0.025, 0.975)).tolist()


def bootstrap(conditions, excluded, draws, seed):
    random = np.random.default_rng(seed)
    result = {}
    # 条件ごとは独立に選び直し、同じ回の率から比・和・差をまとめて作る。
    for name in sorted(conditions):
        ended = ended_levels(conditions[name])
        sampled = ended[random.integers(len(ended), size=(draws, len(ended)))]
        passage = np.ones(draws)
        for level in range(1, 9):
            if level in excluded:
                continue
            reached = (sampled >= level).sum(axis=1)
            failures = (sampled == level).sum(axis=1)
            passage *= np.divide(reached - failures, reached, out=np.full(draws, np.nan), where=reached > 0)
        result[name] = {"passage": passage, "breakthrough": (sampled == 9).mean(axis=1)}
    return result


def collect_runs():
    from common import OUT, REPO, digest, properties, rows
    import json
    from pathlib import Path

    records = {}

    def add(build, checkpoint, condition, seed, run, origin, expected):
        actual_config = properties(run / "config/config_data.txt")
        common = {"run.seed": str(seed), "app.save_agent_on_close": "false",
                  "run.eval.[greedy].eval_batch_size": "16", "run.eval.[greedy].eval_episodes": "200",
                  "DefaultDQNAgent.actor.[greedy].policy.epsilon": "0",
                  "DefaultDQNAgent.skip.max_length": "10", "AtariEnv.frame_skip": "4",
                  "AtariEnv.repeat_action_probability": "0.25", "AtariEnv.noop_max": "30",
                  "AtariEnv.max_episode_frames": "108,000"}
        for key, value in common.items():
            assert actual_config.get(key) == value, (run, key, actual_config.get(key), value)
        for key, value in expected.items():
            assert actual_config.get(key) == value, (run, key, actual_config.get(key), value)
        prefix = "DefaultDQNAgent.actor.[greedy]."
        if condition in ("G", "NQ", "NS", "NB", "TB"):
            assert not any(key.startswith(prefix + "skip.policy.") for key in actual_config), run
        if prefix + "skip.nn_forward_key" not in expected:
            assert prefix + "skip.nn_forward_key" not in actual_config, run
        episodes, scalars = [], {}
        for row in rows(run):
            if row.get("type") == "trace" and row.get("tag") == "53_evalg/episode":
                episodes.append(row["data"])
            elif row.get("type") == "scalar":
                scalars[row["tag"]] = row["value"]
        if len(episodes) != 200:
            raise ValueError(f"Expected 200 episodes: {run}; actual={len(episodes)}")
        key = (build, checkpoint, condition, seed)
        if key in records:
            raise ValueError(f"Duplicate evaluation: {key}")
        metric_file = run / "metrics.jsonl"
        if not metric_file.exists():
            metric_file = run / "metrics.jsonl.gz"
        records[key] = {"episodes": episodes, "run": str(run), "origin": origin,
                        "metrics_sha256": digest(metric_file),
                        "decision_rate": scalars.get("50_agent_skip/21_greedy_decision_rate"),
                        "length_mean": scalars.get("50_agent_skip/22_greedy_length_mean")}

    # PRD §10 で指定された実装前の G/TQ/TS/TB を、元の Run を変えずに読む。
    old = REPO / ".scratch/evaldiag_noise_tau"
    for checkpoint in ("spl", "mrg"):
        for seed in (1, 2):
            for label, condition in (("g", "G"), ("ta", "TQ"), ("ts", "TS"), ("tb", "TB")):
                suffix = f"evnt_p106_{checkpoint}_{label}" + ("_e2" if seed == 2 else "")
                matches = list((old / "runs").glob("run_*_" + suffix))
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous old evaluation: {suffix}")
                if matches:
                    selected = json.loads((old / "cases" / ("p106_" + checkpoint + "_" + label
                        + ("_e2" if seed == 2 else "")) / "resolved.json").read_text(encoding="utf-8"))
                    prefix = "DefaultDQNAgent.actor.[greedy]."
                    expected = {"run.seed": str(seed), "DefaultDQNAgent.learner.enabled": "false",
                                "DefaultDQNAgent.skip.enabled": "true", "DefaultDQNAgent.head.projection.noisy.enabled": "true",
                                "DefaultDQNAgent.auto_load_file": selected["DefaultDQNAgent.auto_load_file"],
                                "run.eval.[greedy].eval_episodes": "200", prefix + "policy.policy_type": "EpsilonGreedy",
                                prefix + "nn_forward_key": "mu_only", prefix + "skip.max_length_mode": "full",
                                prefix + "policy.tau_rule.sample_mode": "random" if label in ("ta", "tb") else "fixed"}
                    for key, value in expected.items():
                        assert selected.get(key) == value, (suffix, key)
                    if label in ("ta", "ts"):
                        expected[prefix + "skip.policy.tau_rule.sample_mode"] = "fixed" if label == "ta" else "random"
                        assert selected[prefix + "skip.policy.tau_rule.sample_mode"] == expected[prefix + "skip.policy.tau_rule.sample_mode"]
                    else:
                        assert not any(key.startswith(prefix + "skip.policy.") for key in selected)
                    add("old", checkpoint, condition, seed, matches[0], "PRD106 section 10 existing evaluation", expected)
    for path in sorted((OUT / "evaluation").glob("*/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        execution = json.loads((path.parent / "execution.json").read_text(encoding="utf-8"))
        assert execution["exit_code"] == 0
        assert execution["binary_sha256"] == record["binary_sha256"]
        actual_config = properties(Path(record["run"]) / "config/config_data.txt")
        assert Path(actual_config["DefaultDQNAgent.auto_load_file"]).resolve() == Path(record["checkpoint_file"]).resolve()
        assert digest(Path(record["checkpoint_file"])) == record["checkpoint_sha256"]
        add(record["build"], record["checkpoint"], record["condition"], record["seed"],
            Path(record["run"]), str(path), record["expected"])
    return records


def checkpoint_report(records, checkpoint, tau_build, tau_base):
    result = {"checkpoint": checkpoint, "noise": None, "tau": None, "tau_independent": None,
              "additional_seed2": [], "pending": []}
    for family, build, names, baseline in (
        ("noise", "new", ("G", "NQ", "NS", "NB"), "G"),
        ("tau", tau_build, ("G", "TQ", "TS", "TB", "TB_indep"), tau_base)):
        required = tuple(dict.fromkeys((baseline,) + names))
        seeds = [seed for seed in (1, 2) if all((build, checkpoint, name, seed) in records for name in required)]
        if not seeds:
            result["pending"].append(family)
            continue
        conditions = {name: [item for seed in seeds for item in records[build, checkpoint, name, seed]["episodes"]]
                      for name in required}
        q, s, both = names[1:4]
        reading = analyze_read(conditions, baseline, q, s, both)
        reading["build"] = build
        reading["seeds"] = seeds
        reading["baseline"] = baseline
        reading["runs"] = {name: [{key: value for key, value in records[build, checkpoint, name, seed].items()
                                    if key != "episodes"} for seed in seeds] for name in required}
        result[family] = reading
        if family == "tau":
            result["tau_independent"] = analyze_read(conditions, baseline, q, s, "TB_indep")
            result["tau_concentration_selected_after_observation"] = build == "old"
        if checkpoint == "spl" and seeds != [1, 2]:
            result["pending"].append(family + " seed2")
        if checkpoint == "mrg" and seeds == [1] and reading["read"] == "決めきれない":
            result["additional_seed2"].append({"build": build, "conditions": list(required)})
    return result


def main():
    import argparse
    from common import OUT, save

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tau-build", choices=("old", "new"), default="old")
    parser.add_argument("--tau-base", choices=("G", "Gp"), default="G")
    args = parser.parse_args()
    destination = OUT / "analysis" / (args.label + ".json")
    if destination.exists():
        raise RuntimeError("Analysis already exists; use a new label")
    records = collect_runs()
    report = {checkpoint: checkpoint_report(records, checkpoint, args.tau_build, args.tau_base)
              for checkpoint in ("spl", "mrg")}
    save(destination, report)
    for checkpoint, data in report.items():
        for family in ("noise", "tau"):
            reading = data[family]
            if reading:
                print(checkpoint, family, "seeds=", reading["seeds"], "read=", reading["read"],
                      "impact=", reading["impact"], flush=True)
        print(checkpoint, "pending=", data["pending"], "additional_seed2=", data["additional_seed2"], flush=True)


def analyze_read(conditions, base, q, s, both, draws=10000, seed=106):
    excluded = summarize(conditions[base])["concentrated"]
    points = {name: summarize(episodes, excluded) for name, episodes in conditions.items()}
    sampled = bootstrap(conditions, excluded, draws, seed)
    for name, point in points.items():
        for measure in ("passage", "breakthrough"):
            point[measure + "_ci"] = interval(sampled[name][measure])
    denominator = points[base]["passage"] - points[both]["passage"]
    denominator_draws = sampled[base]["passage"] - sampled[both]["passage"]
    denominator_ci = interval(denominator_draws)
    result = {"excluded": excluded, "conditions": points, "draws": draws, "bootstrap_seed": seed,
              "denominator": {"value": denominator, "ci": denominator_ci}, "ratios": None}
    # 全体の突破率の差は比の有効な回に絞らず、すべての回を使う。
    result["impact"] = {"value": points[q]["breakthrough"] - points[both]["breakthrough"],
                        "ci": interval(sampled[q]["breakthrough"] - sampled[both]["breakthrough"])}
    result["passage_impact"] = {"value": points[q]["passage"] - points[both]["passage"],
                                "ci": interval(sampled[q]["passage"] - sampled[both]["passage"])}
    low, high = result["impact"]["ci"]
    result["impact_read"] = "skip側を切ると上がる" if low > 0 else "skip側を切ると下がる" if high < 0 else "差が見えない"
    result["excluded_ratio_draw_fraction"] = float((denominator_draws <= 0).mean())
    if denominator_ci[0] <= 0:
        result["read"] = "ランダムさで通過率が上がった" if denominator_ci[1] < 0 else "害が再現しない"
        return result
    keep = denominator_draws > 0
    q_draws = (sampled[base]["passage"] - sampled[q]["passage"])[keep] / denominator_draws[keep]
    s_draws = (sampled[base]["passage"] - sampled[s]["passage"])[keep] / denominator_draws[keep]
    q_point = (points[base]["passage"] - points[q]["passage"]) / denominator
    s_point = (points[base]["passage"] - points[s]["passage"]) / denominator
    result["ratios"] = {name: {"value": value, "ci": interval(samples)} for name, value, samples in (
        ("Q", q_point, q_draws), ("S", s_point, s_draws), ("sum", q_point + s_point, q_draws + s_draws),
        ("S_minus_Q", s_point - q_point, s_draws - q_draws), ("recovery_Q", 1 - q_point, 1 - q_draws))}
    q_ci, s_ci = result["ratios"]["Q"]["ci"], result["ratios"]["S"]["ci"]
    if q_ci[0] <= 0.5 <= q_ci[1] or s_ci[0] <= 0.5 <= s_ci[1]:
        result["read"] = "決めきれない"
    else:
        result["read"] = {(True, False): "Q側が主因", (False, True): "skip側が主因",
                          (True, True): "どちらの経路でも下がる", (False, False): "2つがそろったときに下がる"}[
                              (q_ci[0] > 0.5, s_ci[0] > 0.5)]
    return result


if __name__ == "__main__":
    main()
