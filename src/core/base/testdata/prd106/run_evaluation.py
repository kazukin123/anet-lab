"""PRD106 の評価専用入力を固定し、指定された条件だけを実行する。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from common import OUT, REPO, ROOT, digest, execute, properties, resolve, rows, save

G = "DefaultDQNAgent.actor.[greedy]."
OLD = REPO / ".scratch/evaldiag_noise_tau"
CONDITIONS = {
    "G": {},
    "NQ": {G + "nn_forward_key": "act_sample", G + "skip.nn_forward_mode": "dedicated", G + "skip.nn_forward_key": "mu_only"},
    "NS": {G + "skip.nn_forward_mode": "dedicated", G + "skip.nn_forward_key": "act_sample"},
    "NB": {G + "nn_forward_key": "act_sample"},
    "TQ": {G + "policy.tau_rule.sample_mode": "random", G + "skip.policy.tau_rule.sample_mode": "fixed"},
    "TS": {G + "skip.policy.tau_rule.sample_mode": "random"},
    "TB": {G + "policy.tau_rule.sample_mode": "random"},
    "TB_indep": {G + "policy.tau_rule.sample_mode": "random", G + "skip.policy.tau_rule.sample_mode": "random"},
    "Gp": {G + "skip.policy.tau_rule.sample_mode": "fixed"},
}


def freeze():
    # 後続のビルドから実行体を分離し、既存の凍結先は上書きしない。
    binary = OUT / "current-release-bin"
    config = OUT / "current-config"
    if binary.exists() or config.exists():
        raise RuntimeError("PRD106 evaluation runtime already exists; refusing to overwrite")
    binary.mkdir()
    for path in (REPO / "apps/runner/bin/Release").iterdir():
        if path.is_file() and path.suffix.lower() in (".exe", ".dll"):
            shutil.copy2(path, binary / path.name)
    shutil.copytree(REPO / "apps/runner/config", config)
    shutil.copy2(ROOT.parent / "prd104/arms.txt", config / "prd104-arms.txt")
    save(OUT / "evaluation-runtime.json", {path.relative_to(OUT).as_posix(): digest(path)
         for base in (binary, config) for path in sorted(base.rglob("*")) if path.is_file()})
    print("PRD106 evaluation runtime frozen", flush=True)


def prepare(build, checkpoint, condition, seed):
    name = f"{build}_{checkpoint}_{condition}_s{seed}"
    destination = OUT / "evaluation" / name
    if destination.exists():
        # 準備済みの入力を再利用できるのは、同じ組と全hashを確認できるときだけ。
        record = json.loads((destination / "recipe.json").read_text(encoding="utf-8"))
        assert (record["build"], record["checkpoint"], record["condition"], record["seed"]) == (build, checkpoint, condition, seed)
        for key in ("binary", "config", "checkpoint_file"):
            hash_key = "checkpoint_sha256" if key == "checkpoint_file" else key + "_sha256"
            assert digest(Path(record[key])) == record[hash_key], (name, key)
        return destination, record
    destination.mkdir(parents=True, exist_ok=False)
    # 評価専用の既存レシピを使い、skip の構造と checkpoint の対応を維持する。
    original = OLD / "cases" / f"p106_{checkpoint}_g" / "run.txt"
    base = OUT / "current-config" if build == "new" else OLD / "config"
    lines = [line.replace((OLD / "config").as_posix(), base.as_posix())
             for line in original.read_text(encoding="utf-8").splitlines()]
    leaves = {"run.seed": seed, G + "nn_forward_key": "mu_only", G + "policy.tau_rule.sample_mode": "fixed"}
    if build == "new":
        leaves[G + "skip.nn_forward_mode"] = "shared"
    leaves.update(CONDITIONS[condition])
    lines += [f"run.@prd104 : {key} = {value}" for key, value in leaves.items()]
    config = destination / "run.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    selected = resolve(config, destination, binary=OUT / "baseline-bin/anet-core-test.exe")
    expected = {"DefaultDQNAgent.learner.enabled": "false", "DefaultDQNAgent.skip.enabled": "true",
                "app.save_agent_on_close": "false", "run.eval.[greedy].eval_episodes": "200",
                G + "policy.policy_type": "EpsilonGreedy", G + "skip.max_length_mode": "full",
                G + "nn_forward_key": str(leaves[G + "nn_forward_key"]),
                G + "policy.tau_rule.sample_mode": str(leaves[G + "policy.tau_rule.sample_mode"])}
    if build == "new":
        expected[G + "skip.nn_forward_mode"] = leaves[G + "skip.nn_forward_mode"]
    for key, value in CONDITIONS[condition].items():
        expected[key] = value
    for key, value in expected.items():
        assert selected.get(key) == value, (name, key, selected.get(key), value)
    if G + "skip.nn_forward_key" not in leaves:
        assert G + "skip.nn_forward_key" not in selected
    if G + "skip.policy.tau_rule.sample_mode" not in leaves:
        assert not any(key.startswith(G + "skip.policy.") for key in selected)
    binary = OUT / ("current-release-bin" if build == "new" else "baseline-release-bin") / "AnetRLRunner.exe"
    checkpoint_file = Path(selected["DefaultDQNAgent.auto_load_file"])
    assert checkpoint_file.is_file()
    record = {"build": build, "checkpoint": checkpoint, "condition": condition, "seed": seed,
              "config": str(config), "binary": str(binary), "binary_sha256": digest(binary),
              "checkpoint_file": str(checkpoint_file), "checkpoint_sha256": digest(checkpoint_file),
              "config_sha256": digest(config), "expected": expected}
    save(destination / "recipe.json", record)
    return destination, record


def run_one(item):
    destination, record = item
    binary = Path(record["binary"])
    config = Path(record["config"])
    assert digest(binary) == record["binary_sha256"]
    assert digest(config) == record["config_sha256"]
    assert digest(Path(record["checkpoint_file"])) == record["checkpoint_sha256"]
    execution_path = destination / "execution.json"
    if execution_path.exists():
        # 成功済みの結果だけを再利用し、失敗・実行途中の要求を黙って再実行しない。
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if execution.get("exit_code") != 0 or not (destination / "result.json").exists():
            raise RuntimeError(f"Evaluation is incomplete or failed; refusing an automatic retry: {destination}")
        result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
        assert all(result[key] == value for key, value in record.items())
        assert execution["binary_sha256"] == record["binary_sha256"]
        assert len([row for row in rows(Path(result["run"]))
                    if row.get("type") == "trace" and row.get("tag") == "53_evalg/episode"]) == 200
        return result["run"]
    actual = execute(destination, config, "t106_eval_" + destination.name, "Release", binary)
    actual_config = properties(actual / "config/config_data.txt")
    for key, value in record["expected"].items():
        assert actual_config.get(key) == value, (actual, key, actual_config.get(key), value)
    # 採用された200件だけを使い、不完全な Run は集計へ渡さない。
    episodes = [row["data"] for row in rows(actual)
                if row.get("type") == "trace" and row.get("tag") == "53_evalg/episode"]
    assert len(episodes) == 200, (actual, len(episodes))
    save(destination / "result.json", dict(record, run=str(actual), episodes=200))
    return str(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare", "run"))
    parser.add_argument("--build", choices=("old", "new"), default="new")
    parser.add_argument("--checkpoints", nargs="+", choices=("spl", "mrg"), default=("spl", "mrg"))
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=("G", "NQ", "NS", "NB"))
    parser.add_argument("--seed", type=int, choices=(1, 2))
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
        return
    # 古いビルドへ新しいキーを渡して黙って無視させない。
    if args.build == "old" and any(name in ("NQ", "NS") for name in args.conditions):
        parser.error("NQ and NS require the new build")
    if args.build == "new":
        manifest = json.loads((OUT / "evaluation-runtime.json").read_text(encoding="utf-8"))
        for name, expected in manifest.items():
            assert digest(OUT / name) == expected, name
    recipes = []
    for checkpoint in args.checkpoints:
        seeds = (args.seed,) if args.seed else ((1, 2) if checkpoint == "spl" else (1,))
        for seed in seeds:
            for condition in args.conditions:
                recipes.append(prepare(args.build, checkpoint, condition, seed))
    if args.action == "prepare":
        print("PREPARED", len(recipes), "cases; no Runs started", flush=True)
        return
    # 実時間は性能の比較に使わず、2条件まで同じGPUで評価する。
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(run_one, recipes):
            print("EVALUATED", result, flush=True)


if __name__ == "__main__":
    main()
