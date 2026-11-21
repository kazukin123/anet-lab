"""保存したPRD104着手前の入力で、既定6構成を完全一致比較する。"""
import argparse
import json
import re

from common import OUT, REPO, ROOT, digest, execute, load, resolve, save

CASES = (("GridMaze", False), ("LunarLander", False), ("Atari", False),
         ("DropMerge", False), ("GridMaze", True), ("LunarLander", True))


def run(stage, label, env, enabled):
    destination = OUT / "validation" / f"{label}_{env}_{'on' if enabled else 'off'}"
    destination.mkdir(parents=True, exist_ok=False)
    # 短縮条件だけを既存資材から再利用し、保存先と基準入力は今回のものにする。
    previous = load("prd098/run_validation.py", "prd104_previous")
    previous.OUT = OUT
    lines = previous.settings(env, before=True, enabled=enabled)
    if env == "Atari":
        lines[-1] = lines[-1].replace(" > run.@prd098_validation", " > run.@btrnet > run.@prd098_validation")
    config = destination / "run.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    # run.@skipが追加したsessionも解決後に列挙し、既定には無い評価まで停止する。
    resolver = OUT / "baseline-bin/anet-core-test.exe"
    selected = resolve(config, destination, "sessions", resolver)
    slots = sorted({m[1] for key in selected if (m := re.match(r"run\.eval\.\[([^]]+)\]\.", key))})
    lines += [f"run.@prd098_validation : run.eval_schedule.[{slot}].interval = 0" for slot in slots]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    resolved = resolve(config, destination, "resolved", resolver)
    assert all(resolved[f"run.eval_schedule.[{slot}].interval"] == "0" for slot in slots)
    current = OUT / "current-debug-bin/AnetRLRunner.exe"
    if not current.exists():
        current = REPO / "apps/runner/bin/Debug/AnetRLRunner.exe"
    binary = OUT / "baseline-bin/AnetRLRunner.exe" if stage == "before" else current
    actual = execute(destination, config, "t104_validation_" + destination.name, "Debug", binary)
    normalizer = load("prd084/compare_metrics.py", "prd104_normalizer")
    normalizer.normalize(str(actual), str(destination / "normalized.jsonl"), True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "after", "compare"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference", default="before")
    parser.add_argument("--env", nargs="+", choices=sorted({env for env, _ in CASES}))
    args = parser.parse_args()
    failures = []
    for env, enabled in CASES:
        if args.env and env not in args.env:
            continue
        try:
            if args.stage != "compare":
                run(args.stage, args.label, env, enabled)
                continue
            suffix = f"{env}_{'on' if enabled else 'off'}"
            before = OUT / f"validation/{args.reference}_{suffix}/normalized.jsonl"
            after = OUT / f"validation/{args.label}_{suffix}/normalized.jsonl"
            report = {"exact_match": before.read_bytes() == after.read_bytes(),
                      "points": after.read_bytes().count(b"\n"), "sha256": digest(after)}
            save(after.parent / "comparison.json", report)
            assert report["exact_match"], suffix
            print(suffix, report, flush=True)
        except Exception as error:
            failures.append({"env": env, "enabled": enabled, "error": str(error)})
    save(OUT / "validation" / f"{args.label}-{args.stage}-summary.json", {"failures": failures})
    if failures:
        raise SystemExit(json.dumps(failures, ensure_ascii=False))


if __name__ == "__main__":
    main()
