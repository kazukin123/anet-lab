"""PRD101: 段階ごとの決定的Runと全学習scalarの厳密比較。"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / ".scratch/prd101"
ACTOR_KEYS = {"epsilon", "uqe_tau", "epsilon_lane_min", "epsilon_lane_max", "uqe_epsilon",
              "uqe_epsilon_lane_min", "uqe_epsilon_lane_max", "uqe_tau_lane_min", "uqe_tau_lane_max"}
AGENT_KEYS = {"per_beta", "learning_rate", "weight_decay", "soft_update_tau", "gamma",
              "munchausen_alpha", "munchausen_entropy_tau"}


def records(run):
    path = run / "metrics.jsonl"
    opener = open
    if not path.exists():
        path, opener = run / "metrics.jsonl.gz", gzip.open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        yield from (json.loads(line) for line in stream)


def normalized(run):
    rows = list(records(run))
    definitions = next(row["data"] for row in rows
                       if row.get("type") == "json" and row.get("tag") == "metrics.scalar.defs")
    excluded = {tag for tag, definition in definitions.items()
                if (definition.get("target") == "actor" and definition.get("source_key") in ACTOR_KEYS)
                or (definition.get("target") == "agent" and definition.get("source_key") in AGENT_KEYS)}
    values = []
    for row in rows:
        if row.get("type") != "scalar":
            continue
        tag = row["tag"]
        # 時間・資源は学習数値ではない。学習scalarの例外は上のPRD明示一覧だけ。
        if tag.startswith(("90_perf/", "91_resource/", "92_resource/")) or tag in excluded:
            continue
        value = row["value"]
        values.append((tag, int(row["step"]), "NaN" if value is None else float(value).hex()))
    values.sort()
    return values, sorted(excluded)


def compare(reference, actual):
    left, old_excluded = normalized(reference)
    right, new_excluded = normalized(actual)
    data = json.dumps(right, separators=(",", ":")).encode()
    return {"exact_match": left == right, "before_points": len(left), "after_points": len(right),
            "sha256": hashlib.sha256(data).hexdigest(), "excluded_before": old_excluded,
            "excluded_after": new_excluded,
            "first_differences": [{"before": a, "after": b} for a, b in zip(left, right) if a != b][:10]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage1", "stage2"))
    parser.add_argument("--env", choices=("GridMaze", "LunarLander", "Atari", "DropMerge"))
    parser.add_argument("--label")
    args = parser.parse_args()
    # PRD098の確定済み4環境・seed・短予算を再利用し、基準と同一条件で実行する。
    spec = importlib.util.spec_from_file_location("prd098_validation",
        REPO / "core/anet-core/testdata/prd098/run_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    summary_path = OUT / (args.stage + "-summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for env in ((args.env,) if args.env else module.ENVS):
        try:
            name = (args.label or args.stage) + "_" + env
            overrides = ("DefaultDQNAgent.actor.[train].policy.spatial.uqe.epsilon.last=0.0001",) if env in ("DropMerge", "LunarLander") else ()
            actual = module.execute(name, env, overrides=overrides)
            reference_name = ("before_" if args.stage == "stage1" else "stage1_") + env
            reference = Path(json.loads((OUT / reference_name / "execution.json").read_text())["run"])
            if args.stage == "stage2":
                previous = json.loads((OUT / "stage1-summary.json").read_text())[env]
                reference = Path(previous.get("run", str(reference)))
            report = compare(reference, actual)
            report["run"] = str(actual)
            module.save(OUT / name / "comparison.json", report)
            summary[env] = report
            print(env, "PASS" if report["exact_match"] else "FAIL", flush=True)
        except Exception as error:
            summary[env] = {"error": str(error)}
            print(env, "ERROR", str(error), flush=True)
        module.save(OUT / (args.stage + "-summary.json"), summary)
    raise SystemExit(0 if all(row.get("exact_match") for row in summary.values()) else 1)


if __name__ == "__main__":
    main()