"""PRD128: PRD125 の検証を呼び、設定記録とログも比較する。"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[5]
ROOT = REPO / ".scratch/prd128/acceptance"
SPEC = importlib.util.spec_from_file_location(
    "prd128_validation_base", REPO / "src/core/base/testdata/prd125/run_validation.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
BASE.ROOT, BASE.BASELINE, BASE.INPUTS = ROOT, ROOT / "baseline", ROOT / "inputs"
ORIGINAL_DEFINITIONS, ORIGINAL_PREPARE = BASE.definitions, BASE.prepare
ORIGINAL_EXPECTED, ORIGINAL_AUDIT = BASE.expected_values, BASE.audit
ORIGINAL_REPEAT, ORIGINAL_COMPARE = BASE.repeat_case, BASE.compare_case
EXTRA = {
    "rnd_LunarLander_b8": {"env": "LunarLander", "device": "cpu", "precision": "fp32", "rnd": True},
    "rnd_Atari_b8": {"env": "Atari", "device": "cuda", "precision": "fp32", "rnd": True},
    "diagnostics_DropMerge_explicit_sources": {"env": "DropMerge", "device": "cuda", "precision": "fp32", "diagnostics": True},
}
DIAGNOSTIC_PREFIXES = ("34_agent_plasticity/", "35_agent_churn/", "46_agent_replay_fit/",
                       "47_agent_noisy/", "48_agent_attention/", "49_agent_gradnoise/")
PERFORMANCE_RND = False


def base_definitions():
    return {key: value for key, value in ORIGINAL_DEFINITIONS().items() if value["precision"] == "fp32"}


def definitions():
    return {**base_definitions(), **EXTRA}


def expected_values(case):
    values = ORIGINAL_EXPECTED(case)
    if case.get("rnd"):
        values["DefaultDQNAgent.intrinsic.enabled"] = "true"
        values["DefaultDQNAgent.intrinsic.learn_batch_size"] = "8"
        values["DefaultDQNAgent.intrinsic.rnd.warmup_steps"] = "256"
    if case.get("performance") and PERFORMANCE_RND:
        values["DefaultDQNAgent.intrinsic.enabled"] = "true"
        values["DefaultDQNAgent.intrinsic.rnd.warmup_steps"] = "256"
    return values


def write_frozen(path, content):
    # 採取済みの入力は、後続の実行で書き換えない。
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError("Frozen input changed: " + str(path))
    else:
        path.write_text(content, encoding="utf-8", newline="\n")


def prepare():
    # 元の 31 組は原本の生成を使い、追加構成だけを拡張する。
    BASE.definitions = base_definitions
    try:
        ORIGINAL_PREPARE()
    finally:
        BASE.definitions = definitions
    for name, case in EXTRA.items():
        lines = (BASE.INPUTS / ("default_" + case["env"] + ".txt")).read_text(encoding="utf-8").splitlines()
        if case.get("rnd"):
            chain, _, validation = lines[-1].rpartition(" > ")
            lines[-1] = chain + " > run.@rnd > " + validation
            lines.insert(-1, "run.@prd125_validation : app.save_agent_on_close = true")
            lines.insert(-1, "run.@prd125_validation : DefaultDQNAgent.intrinsic.learn_batch_size = 8")
            lines.insert(-1, "run.@prd125_validation : DefaultDQNAgent.intrinsic.rnd.warmup_steps = 256")
        if case.get("diagnostics"):
            index = next(i for i, line in enumerate(lines) if ": metrics.scalar.$ = " in line)
            lines[index] = "run.@prd125_validation : metrics.scalar.$ = " + " > ".join(
                "metrics.scalar.@" + profile for profile in ("baseline", "full", "replay_fit", "gradnoise", "noisy", "attention"))
            # full の旧 Agent 指標は取得元が未指定で、未知キー時に Env まで探索する。
            # Agent 群のタグは取得先を明示し、未提供キーは既存の未知キー診断で扱う。
            catalog = (BASE.INPUTS / "baseline-config/metrics_scalar.txt").read_text(encoding="utf-8-sig")
            for tag, value in re.findall(r"(?m)^metrics\.scalar\.@full\.\[([^\]]+)\]\s*\?=\s*([^\n]+)", catalog):
                value = value.split("#", 1)[0].strip()
                if "$" not in value:
                    lines.insert(-1, "run.@prd125_validation : metrics.scalar.[" + tag + "] = " + value + " $agent")
        write_frozen(BASE.INPUTS / (name + ".txt"), "\n".join(lines) + "\n")
    BASE.COMMON.save(BASE.INPUTS / "cases.json", definitions())
    return definitions()


def audit(run, normalized, case):
    ORIGINAL_AUDIT(run, normalized, case)
    if case.get("rnd"):
        points = [json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines()]
        if not any(tag == "56_agent_in/32_learn_count" and float.fromhex(value) > 0 for tag, _, value in points):
            raise RuntimeError("RND auxiliary learning did not execute: " + str(run))
    if case.get("diagnostics"):
        points = [json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines()]
        finite = {tag for tag, _, value in points if math.isfinite(float.fromhex(value))}
        missing = [prefix for prefix in DIAGNOSTIC_PREFIXES if not any(tag.startswith(prefix) for tag in finite)]
        if missing:
            raise RuntimeError("Missing finite diagnostic groups: " + ", ".join(missing))


def artifact_records(normalized):
    run = Path(json.loads(Path(normalized).with_name("inputs.json").read_text(encoding="utf-8"))["run"])
    config = (run / "config/DefaultDQNAgent.txt").read_bytes().hex()
    rows = BASE.SETTINGS.rows(run)
    config_json = [row.get("value", row.get("data")) for row in rows
                   if row.get("type") == "json" and row.get("tag") == "DefaultDQNAgent"]
    if not config_json:
        raise RuntimeError("Missing DefaultDQNAgent JSON record: " + str(run))
    warnings = []
    for line in (run / (run.name + ".log")).read_text(encoding="utf-8-sig").splitlines():
        match = re.search(r"\[(?:W|E)\]|\b(?:WARN|ERROR)\b", line)
        if match:
            message = line[match.start():].replace(str(run), "<RUN_PATH>").replace(run.as_posix(), "<RUN_PATH>")
            warnings.append(message.replace(run.name, "<RUN_NAME>"))
    return {"config": config, "config_json": config_json, "warnings": warnings}


def compare_artifacts(first, second):
    before, after = artifact_records(first), artifact_records(second)
    matches = {key: before[key] == after[key] for key in before}
    BASE.COMMON.save(Path(second).with_name("artifact-comparison.json"), {
        "before": before, "after": after, "matches": matches})
    return matches


def repeat_case(label, case):
    report = ORIGINAL_REPEAT(label, case)
    if "first" in report:
        try:
            report["artifact_matches"] = compare_artifacts(report["first"], report["second"])
            report["exact_match"] &= all(report["artifact_matches"].values())
        except Exception as error:
            report.update(exact_match=False, artifact_error=str(error))
    return report


def compare_case(label, reference):
    report = ORIGINAL_COMPARE(label, reference)
    if "after" in report:
        try:
            report["artifact_matches"] = compare_artifacts(report["first"], report["after"])
            report["exact_match"] &= all(report["artifact_matches"].values())
        except Exception as error:
            report.update(exact_match=False, artifact_error=str(error))
    return report


BASE.definitions, BASE.expected_values, BASE.prepare = definitions, expected_values, prepare
BASE.audit, BASE.repeat_case, BASE.compare_case = audit, repeat_case, compare_case


def checkpoint(label, reference):
    prepare()
    reference_data = json.loads((ROOT / (reference + "-summary.json")).read_text(encoding="utf-8"))
    original = Path(reference_data["rnd_LunarLander_b8"]["first"])
    run = Path(json.loads(original.with_name("inputs.json").read_text(encoding="utf-8"))["run"])
    saved = run / "agent_close.anet"
    if not saved.is_file():
        raise FileNotFoundError(saved)
    lines = (BASE.INPUTS / "rnd_LunarLander_b8.txt").read_text(encoding="utf-8").splitlines()
    lines.insert(-1, "run.@prd125_validation : DefaultDQNAgent.auto_load_file = " + saved.as_posix())
    write_frozen(BASE.INPUTS / "checkpoint.txt", "\n".join(lines) + "\n")
    # 同じ旧 checkpoint と同じ凍結入力から、新旧をそれぞれ続ける。
    before = BASE.execute(BASE.BASELINE / "Debug/AnetRLRunner.exe", label + "_before", "checkpoint",
                          definitions()["rnd_LunarLander_b8"])
    after = BASE.execute(REPO / "apps/runner/bin/Debug/AnetRLRunner.exe", label + "_after", "checkpoint",
                         definitions()["rnd_LunarLander_b8"])
    report = {"scalar_match": before.read_bytes() == after.read_bytes(),
              "artifact_matches": compare_artifacts(before, after), "checkpoint_sha256": BASE.COMMON.digest(saved),
              "before": str(before), "after": str(after)}
    BASE.COMMON.save(ROOT / (label + "-summary.json"), report)
    if not report["scalar_match"] or not all(report["artifact_matches"].values()):
        raise RuntimeError("Checkpoint continuation mismatch")
    print("CHECKPOINT", label, "passed", flush=True)


def performance(label, rounds, rnd):
    global PERFORMANCE_RND
    if not rnd:
        return BASE.performance(label, rounds)
    # 通常の性能入力を保持し、RND 版は独立した凍結ディレクトリを使う。
    prepare()
    original_inputs, original_settings_out = BASE.INPUTS, BASE.SETTINGS.OUT
    original_prepare, original_settings = BASE.prepare, BASE.SETTINGS.settings
    directory = ROOT / "inputs-performance-rnd"
    directory.mkdir(exist_ok=True)
    for name in ("baseline-config", "resolved-before"):
        if not (directory / name).exists():
            shutil.copytree(original_inputs / name, directory / name)
    def rnd_settings(*args, **kwargs):
        lines = original_settings(*args, **kwargs)
        chain, _, validation = lines[-1].rpartition(" > ")
        lines[-1] = chain + " > run.@rnd > " + validation
        return lines
    BASE.INPUTS = BASE.SETTINGS.OUT = directory
    BASE.prepare = lambda: definitions()
    BASE.SETTINGS.settings = rnd_settings
    PERFORMANCE_RND = True
    try:
        BASE.performance(label, rounds)
    finally:
        BASE.INPUTS, BASE.SETTINGS.OUT = original_inputs, original_settings_out
        BASE.prepare, BASE.SETTINGS.settings = original_prepare, original_settings
        PERFORMANCE_RND = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "prepare", "repeat", "audit-reference", "compare", "checkpoint", "suite", "performance"))
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--case", nargs="+", choices=tuple(definitions()))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rnd", action="store_true")
    args = parser.parse_args()
    if args.mode == "snapshot":
        BASE.snapshot()
    elif args.mode == "prepare":
        prepare()
    elif args.mode == "repeat":
        BASE.repeat(args.label, args.case or tuple(definitions()), args.workers)
    elif args.mode == "compare":
        BASE.compare(args.label, args.reference, args.case or tuple(definitions()), args.workers)
    elif args.mode == "checkpoint":
        checkpoint(args.label, args.reference)
    elif args.mode == "audit-reference":
        destination = ROOT / (args.label + "-summary.json")
        if destination.exists():
            raise FileExistsError(destination)
        available = {}
        for source in args.sources or [args.reference]:
            available.update(json.loads((ROOT / (source + "-summary.json")).read_text(encoding="utf-8")))
        # 不成立だった旧構成は記録を保持し、現用の全構成が揃った基準だけを作る。
        reports = {case: available[case] for case in definitions()}
        for case, report in reports.items():
            for result in (report["first"], report["second"]):
                run = Path(json.loads(Path(result).with_name("inputs.json").read_text(encoding="utf-8"))["run"])
                audit(run, Path(result), definitions()[case])
            report["artifact_matches"] = compare_artifacts(report["first"], report["second"])
            report["exact_match"] &= all(report["artifact_matches"].values())
        BASE.COMMON.save(destination, reports)
        print("AUDIT", len(reports), "passed", sum(report["exact_match"] for report in reports.values()), flush=True)
        if not all(report["exact_match"] for report in reports.values()):
            raise RuntimeError("Baseline artifact mismatch")
    elif args.mode == "suite":
        BASE.suite(args.label)
    else:
        if args.rounds < 3:
            parser.error("--rounds must be >=3")
        performance(args.label, args.rounds, args.rnd)


if __name__ == "__main__":
    main()
