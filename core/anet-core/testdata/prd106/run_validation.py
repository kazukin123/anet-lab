"""PRD106 の既定7構成を同じ保存入力で前後比較する。"""
import argparse
import json
import re

from common import OUT, REPO, digest, execute, load, resolve, save

CASES = (("GridMaze", False, False), ("LunarLander", False, False), ("Atari", False, False),
         ("DropMerge", False, False), ("GridMaze", True, False), ("LunarLander", True, False),
         ("LunarLander", True, True))


def run(stage, label, env, enabled, noisy):
    suffix = f"{env}_{'on' if enabled else 'off'}" + ("_noisy" if noisy else "")
    destination = OUT / "validation" / f"{label}_{suffix}"
    destination.mkdir(parents=True, exist_ok=False)
    previous = load("prd098/run_validation.py", "prd106_previous_validation")
    previous.OUT = OUT
    lines = previous.settings(env, before=True, enabled=enabled)
    # Atari は BTR、追加の LunarLander は NoisyNet と skip の両方を有効にする。
    if env == "Atari":
        lines[-1] = lines[-1].replace(" > run.@prd098_validation", " > run.@btrnet > run.@prd098_validation")
    if noisy:
        lines[-1] = lines[-1].replace(" > run.@prd098_validation", " > run.@noisy > run.@prd098_validation")
    config = destination / "run.txt"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    resolver = OUT / "baseline-bin/anet-core-test.exe"
    selected = resolve(config, destination, "sessions", resolver)
    slots = sorted({match[1] for key in selected if (match := re.match(r"run\.eval\.\[([^]]+)\]\.", key))})
    lines += [f"run.@prd098_validation : run.eval_schedule.[{slot}].interval = 0" for slot in slots]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    actual_config = resolve(config, destination, "resolved", resolver)
    assert all(actual_config[f"run.eval_schedule.[{slot}].interval"] == "0" for slot in slots)
    if noisy:
        assert actual_config["DefaultDQNAgent.head.projection.noisy.enabled"] == "true"
        assert actual_config["DefaultDQNAgent.actor.[train].nn_forward_key"] == "act_sample"
    binary = OUT / "baseline-bin/AnetRLRunner.exe" if stage == "before" else REPO / "apps/runner/bin/Debug/AnetRLRunner.exe"
    actual = execute(destination, config, "t106_validation_" + destination.name, "Debug", binary)
    normalizer = load("prd084/compare_metrics.py", "prd106_normalizer")
    normalizer.normalize(str(actual), str(destination / "normalized.jsonl"), True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "after", "compare"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference", default="before")
    args = parser.parse_args()
    failures = []
    for env, enabled, noisy in CASES:
        suffix = f"{env}_{'on' if enabled else 'off'}" + ("_noisy" if noisy else "")
        try:
            if args.stage != "compare":
                run(args.stage, args.label, env, enabled, noisy)
                continue
            before = OUT / f"validation/{args.reference}_{suffix}/normalized.jsonl"
            after = OUT / f"validation/{args.label}_{suffix}/normalized.jsonl"
            report = {"exact_match": before.read_bytes() == after.read_bytes(),
                      "points": after.read_bytes().count(b"\n"), "sha256": digest(after)}
            save(after.parent / "comparison.json", report)
            assert report["exact_match"], suffix
            print(suffix, report, flush=True)
        except Exception as error:
            failures.append({"case": suffix, "error": str(error)})
    save(OUT / "validation" / f"{args.label}-{args.stage}-summary.json", {"failures": failures})
    if failures:
        raise SystemExit(json.dumps(failures, ensure_ascii=False))


if __name__ == "__main__":
    main()
