"""保存した変更前設定をPRD宣言で移行し、7環境の全キーと文字列表記を比較する。"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "core/anet-core/testdata/prd072"))
from profiled_migration import migrate_values

OUT = REPO / ".scratch/prd101"
ACTOR_METRICS = ("epsilon_lane_min", "epsilon_lane_max", "uqe_epsilon", "uqe_epsilon_lane_min",
                 "uqe_epsilon_lane_max", "uqe_tau_lane_min", "uqe_tau_lane_max")

def stage1_expected(before, env):
    expected = migrate_values(before, learning=False)
    # D15: 共有baseのspatial epsilonは時間のepsilonとは独立した0.4/0.01。
    if env != "Atari":
        expected["DefaultDQNAgent.actor.[train].policy.spatial.epsilon.first"] = "0.4"
        expected["DefaultDQNAgent.actor.[train].policy.spatial.epsilon.last"] = "0.01"
    # 元agent.txtのbare既定0.05は、上位linearの指定で消さずに休眠保持する。
    root = "DefaultDQNAgent.actor.[train].policy.uqe.epsilon"
    if expected.get(root + ".type") == "linear":
        expected[root] = "0.05"
    if "metrics.scalar.[32_agent_base/05_epsilon_learn]" in before:
        for key in ACTOR_METRICS:
            expected[f"metrics.scalar.[32_agent_base/{key}]"] = f"$actor {key} @learn $exp_step"
    return expected

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help=".scratch/prd101配下の採取済みディレクトリ名")
    parser.add_argument("--stage2", action="store_true")
    args = parser.parse_args()
    summary = {}
    for env in ("Atari", "CartPole", "DropMerge", "GridMaze", "GridMaze_muzero", "ImageCls", "LunarLander"):
        before = json.loads((OUT / "resolved-before" / (env + ".json")).read_text(encoding="utf-8"))
        actual = json.loads((OUT / args.snapshot / (env + ".json")).read_text(encoding="utf-8"))
        expected = stage1_expected(before, env)
        if args.stage2:
            expected = migrate_values(expected)
            if "metrics.scalar.[32_agent_base/05_epsilon_learn]" in before:
                for key in ("learning_rate", "weight_decay", "soft_update_tau", "gamma", "munchausen_alpha", "munchausen_entropy_tau"):
                    expected[f"metrics.scalar.[32_agent_base/{key}]"] = f"$agent {key} @learn $exp_step"
        differences = [{"key": key, "expected": expected.get(key), "actual": actual.get(key)}
                       for key in sorted(expected.keys() | actual.keys()) if expected.get(key) != actual.get(key)]
        summary[env] = {"keys": len(actual), "errors": differences}
        print(env, "keys=", len(actual), "errors=", len(differences))
    (OUT / (args.snapshot + "-comparison.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(any(row["errors"] for row in summary.values()))

if __name__ == "__main__":
    main()
