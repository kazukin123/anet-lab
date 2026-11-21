"""Compare captured PRD061 module configs; never updates the baseline."""
import argparse
import json
from pathlib import Path


def destinations(key):
    if key.startswith("eval_policy."):
        return [key.replace("eval_policy.", f"actor.[{tag}].policy.", 1) for tag in ("eval", "eval_target")]
    if key == "bf16.actor":
        return [f"actor.[{tag}].bf16" for tag in ("train", "eval", "eval_target")]
    for old, new in (("train_policy.", "actor.[train].policy."), ("train_actor.", "actor.[train]."),
                     ("actor.temp_", "actor.[train].temp_"), ("action_policy.", "actor.[train].policy.")):
        if key.startswith(old):
            return [key.replace(old, new, 1)]
    return [key]


def compare(before, after):
    report = []
    for source in sorted(before.glob("*.json")):
        expected = json.loads(source.read_text(encoding="utf-8"))
        actual = json.loads((after / source.name).read_text(encoding="utf-8"))
        assert expected["agent"] == actual["agent"], source.name
        mapped = {new: value for key, value in expected["typed"].items() for new in destinations(key)}
        differences = [{"key": key, "before": value, "after": actual["typed"].get(key, "MISSING")}
                       for key, value in mapped.items() if actual["typed"].get(key, "MISSING") != value]
        added = {key: value for key, value in actual["typed"].items() if key not in mapped}
        report.append({"input": source.stem, "differences": differences, "added_catalog_fields": added})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    count = sum(len(item["differences"]) for item in result)
    print(f"inputs={len(result)}, typed_differences={count}")
    for item in result:
        for difference in item["differences"]:
            print(item["input"], difference)
    raise SystemExit(1 if count or len(result) != 17 else 0)
