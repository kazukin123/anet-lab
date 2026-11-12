"""PRD123 の5キー以外に、解決済み設定の変更が無いことを確かめる。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / ".scratch/prd123/baseline-20261111"
EXPECTED = {
    "DefaultDQNAgent.intrinsic.use_done_mask": "true",
    "DefaultDQNAgent.intrinsic.rnd.warmup_mode": "learn_predictor",
    "DefaultDQNAgent.intrinsic.rnd.error_norm.average_mode": "cumulative",
    "DefaultDQNAgent.intrinsic.rnd.error_norm.ema_steps": "1,000,000",
    "DefaultDQNAgent.intrinsic.rnd.error_norm.use_centering": "true",
}


def main():
    before_files = {p.name for p in (ROOT / "resolved-before").glob("*.json")}
    after_files = {p.name for p in (ROOT / "resolved-after").glob("*.json")}
    if not before_files or before_files != after_files:
        raise RuntimeError(f"Configuration snapshot sets differ: before={sorted(before_files)}, after={sorted(after_files)}")
    report = {}
    for name in sorted(before_files):
        before = json.loads((ROOT / "resolved-before" / name).read_text(encoding="utf-8"))
        after = json.loads((ROOT / "resolved-after" / name).read_text(encoding="utf-8"))
        # 追加だけでなく、既存の全キーと文字列値を厳密に照合する。
        added = {key: after[key] for key in after.keys() - before.keys()}
        removed = sorted(before.keys() - after.keys())
        changed = sorted(key for key in before.keys() & after.keys() if before[key] != after[key])
        passed = added == EXPECTED and not removed and not changed
        report[name] = {"added": added, "removed": removed, "changed": changed, "passed": passed}
        print(name, "added", len(added), "changed", len(changed), "removed", len(removed), "PASS" if passed else "FAIL")
    (ROOT / "config_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not all(entry["passed"] for entry in report.values()):
        raise RuntimeError("Resolved configuration changed beyond the five PRD123 defaults")


if __name__ == "__main__":
    main()
