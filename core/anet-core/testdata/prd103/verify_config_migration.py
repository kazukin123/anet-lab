"""PRD103 の設定を全キー・文字列値で比較し、意図した移行だけを許可する。"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
OUT = REPO / ".scratch/prd103"


def normalize(data, env):
    # 旧名の書き換えは比較用の写しだけに行い、保存した入力は保持する。
    def rename(value):
        value = re.sub(r"03_skip_length_rate\.\[(\d)\]", r"03_skip_length_rate.[0\1]", value)
        return value.replace("greedy_dist", "greedy") if env == "Atari" else value
    return {rename(key): rename(value) for key, value in data.items()}


def allowed(key, before, after, env):
    # 既存値の変更は改名で正規化済み。新しい宣言だけを、所有者の範囲で許可する。
    if before is not None or after is None:
        return False
    if key == "DefaultDQNAgent.learner.skip.sample_weight":
        return after == "uniform"
    if key.startswith("metrics.scalar.[50_agent_skip/"):
        return any(key.startswith("metrics.scalar.[50_agent_skip/" + tag) for tag in (
            "17_skip_primary_grad_norm]", "18_skip_primary_clip_ratio]", "21_greedy_decision_rate]",
            "22_greedy_length_mean]", "23_greedy_length_rate.["))
    if env != "Atari" and any(key.startswith(prefix) for prefix in (
            "DefaultDQNAgent.actor.[greedy].", "run.eval.[greedy].", "run.eval_schedule.[greedy].")):
        return True
    if env != "Atari" and key in {f"metrics.scalar.[21_eval/{tag}]" for tag in (
            "09_greedy_reward", "10_greedy_reward_ema", "11_greedy_ep_steps", "12_greedy_ep_steps_max")}:
        return True
    if env == "GridMaze":
        expected = {"policy_type": "EpsilonGreedy", "epsilon.start": "0.60", "epsilon.end": "0.05"}
        return any(key == "DefaultDQNAgent.actor.[train].skip.policy." + suffix and after == value
            for suffix, value in expected.items())
    return False


def main():
    # defaultと明示的なskip選択を別々に比較し、無関係な群や既定値の脱落を検出する。
    reports = {}
    for source in sorted((OUT / "config-before").glob("*.json")):
        env = source.name.split("-")[0]
        before = normalize(json.loads(source.read_text(encoding="utf-8")), env)
        after = json.loads((OUT / "config-after" / source.name).read_text(encoding="utf-8"))
        changes = {key: [before.get(key), after.get(key)] for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)}
        unexpected = {key: values for key, values in changes.items() if not allowed(key, *values, env)}
        reports[source.stem] = {"before_keys": len(before), "after_keys": len(after),
            "changes": changes, "unexpected": unexpected}
        print(source.stem, "changes=", len(changes), "unexpected=", len(unexpected), flush=True)
    if len(reports) != 8:
        raise RuntimeError(f"Expected eight configuration snapshots; found {len(reports)}")
    (OUT / "config-comparison.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raise SystemExit(any(report["unexpected"] for report in reports.values()))


if __name__ == "__main__":
    main()
