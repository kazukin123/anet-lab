"""PRD125: 凍結した入力で更新の共通化の前後を比較する。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / ".scratch/prd125"
BASELINE = ROOT / "baseline"
INPUTS = ROOT / "inputs-r2"
MODES = ("none", "qr", "iqn")
# PER、TBO、skip、Munchausen の分かれ目。PRD125 §5.4 の 9 組。
VARIANTS = {
    "base": (True, False, False, None),
    "no_per": (False, False, False, None),
    "tbo": (True, True, False, None),
    "skip": (True, False, True, None),
    "target": (True, False, False, "target"),
    "online": (True, False, False, "online"),
    "online_reuse": (True, False, False, "online_reuse"),
    "online_skip": (True, False, True, "online"),
    "target_tbo_skip": (True, True, True, "target"),
}


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load("prd125_common", "core/anet-core/testdata/prd109/run_validation.py")
SETTINGS = load("prd125_settings", "core/anet-core/testdata/prd098/run_validation.py")
NORMALIZER = load("prd125_normalizer", "core/anet-core/testdata/prd084/compare_metrics.py")


def snapshot():
    # 実行体と依存 DLL の写しを保存し、以後のビルドから独立させる。
    BASELINE.mkdir(parents=True, exist_ok=False)
    for config in ("Debug", "Release"):
        source = REPO / "apps/runner/bin" / config
        destination = BASELINE / config
        destination.mkdir()
        for path in source.iterdir():
            if path.suffix.lower() in (".exe", ".dll"):
                shutil.copy2(path, destination / path.name)
    tests = BASELINE / "tests"
    tests.mkdir()
    for name in ("dqn_based_agent_test.cpp", "dqn_munchausen_test.cpp"):
        shutil.copy2(REPO / "core/anet-core/src" / name, tests / name)
    core = BASELINE / "core"
    core.mkdir()
    for path in (REPO / "core/anet-core/bin/Debug").iterdir():
        if path.suffix.lower() in (".exe", ".dll"):
            shutil.copy2(path, core / path.name)
    shutil.copytree(REPO / "apps/runner/config", BASELINE / "config")
    COMMON.save(BASELINE / "manifest.json", {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
        "sha256": {path.relative_to(BASELINE).as_posix(): COMMON.digest(path)
                   for path in BASELINE.rglob("*") if path.is_file()},
    })
    print("SNAPSHOT", BASELINE, flush=True)


def definitions():
    cases = {"default_" + env: {"env": env, "device": device, "precision": "fp32"}
             for env, device in (("GridMaze", "cpu"), ("LunarLander", "cpu"),
                                 ("Atari", "cuda"), ("DropMerge", "cuda"))}
    for mode in MODES:
        for variant in VARIANTS:
            cases[mode + "_" + variant] = {
                "env": "LunarLander", "device": "cpu", "precision": "fp32",
                "mode": mode, "variant": variant}
    for mode in MODES:
        for device in ("cpu", "cuda"):
            cases["bf16_" + mode + "_" + device] = {
                "env": "LunarLander", "device": device, "precision": "bf16",
                "mode": mode, "variant": "target_tbo_skip"}
    cases["bf16_Atari_cuda"] = {"env": "Atari", "device": "cuda", "precision": "bf16"}
    return cases


def expected_values(case):
    values = {"run.seed": "1", "run.train.num_envs": "8", "agent.device": case["device"],
              "DefaultDQNAgent.learner.replay_batch_size": "16",
              "DefaultDQNAgent.learner.update_warmup_steps": "256",
              "DefaultDQNAgent.learner.replay_ratio": "1",
              "DefaultDQNAgent.learner.use_amp": str(case["precision"] == "bf16").lower()}
    if case["precision"] == "bf16":
        values["DefaultDQNAgent.learner.use_amp_bf16"] = "true"
    if case.get("performance"):
        values.update({"run.train.num_envs": "128",
            "DefaultDQNAgent.quantile_mode": "iqn",
            "DefaultDQNAgent.learner.replay_batch_size": "256",
            "DefaultDQNAgent.learner.use_rb_prefetch": "true",
            "DefaultDQNAgent.learner.munchausen.enabled": "true",
            "DefaultDQNAgent.learner.use_double_dqn": "false",
            "DefaultDQNAgent.skip.enabled": "false"})
    if "variant" in case:
        per, tbo, skip, munch = VARIANTS[case["variant"]]
        values.update({"DefaultDQNAgent.quantile_mode": case["mode"],
            "DefaultDQNAgent.learner.use_per": str(per).lower(),
            "DefaultDQNAgent.learner.use_tbo": str(tbo).lower(),
            "DefaultDQNAgent.skip.enabled": str(skip).lower(),
            "DefaultDQNAgent.learner.munchausen.enabled": str(munch is not None).lower(),
            "DefaultDQNAgent.learner.munchausen.log_policy_mode": munch or "target"})
    return values


def prepare():
    # 短縮条件の解決結果を、保存した設定だけから一度採る。
    INPUTS.mkdir(parents=True, exist_ok=True)
    if not (INPUTS / "baseline-config").exists():
        shutil.copytree(BASELINE / "config", INPUTS / "baseline-config")
    if not (INPUTS / "resolved-before/LunarLander.json").exists():
        info = COMMON.execute(
            [str(BASELINE / "Debug/AnetRLRunner-gui-test.exe"), "[prd109_config]"],
            INPUTS / "capture", cwd=REPO,
            env={**os.environ, "PRD109_CONFIG_OUTPUT": str(INPUTS)})
        if info["exit_code"]:
            raise RuntimeError("Frozen configuration capture failed")
    SETTINGS.OUT = INPUTS
    cases = definitions()
    for name, case in cases.items():
        lines = SETTINGS.settings(case["env"], before=True)
        if "variant" in case:
            per, tbo, skip, munch = VARIANTS[case["variant"]]
            # 既定の M1（IQN 診断 7 値）を保ち、必要な診断群を末尾へ足す。
            metrics_line = next(line for line in lines if ": metrics.scalar.$ = " in line)
            metrics = metrics_line.split(" = ", 1)[1] + " > metrics.scalar.@replay_fit"
            if munch:
                metrics += " > metrics.scalar.@munchausen"
            if skip:
                metrics += " > metrics.scalar.@skip"
            extra = {**expected_values(case), "metrics.scalar.$": metrics,
                     "DefaultDQNAgent.learner.use_double_dqn": "false",
                     "DefaultDQNAgent.$": "@baseline > @" + ("iqn" if case["mode"] == "iqn" else "qr") + " > @heavy > A1 > @bf16 > A2",
                     "DefaultDQNAgent.net.$": "net.@" + ("iqn" if case["mode"] == "iqn" else "qr"),
                     "DefaultDQNAgent.net.body.output.[skip_context]":
                         "fusion" if case["mode"] == "iqn" else "main_feature"}
            if case["mode"] == "none":
                resolved = json.loads((INPUTS / "resolved-before/LunarLander.json").read_text(encoding="utf-8"))
                extra.update({key: "EpsilonGreedy" for key, value in resolved.items()
                              if key.startswith("DefaultDQNAgent.actor.")
                              and key.endswith(".policy.policy_type") and value == "UQE"})
        else:
            extra = {} if case["precision"] == "fp32" else expected_values(case)
        for key, value in extra.items():
            lines.insert(-1, "run.@prd098_validation : " + key + " = " + value)
        text = "\n".join(lines).replace("prd098", "prd125") + "\n"
        config = INPUTS / (name + ".txt")
        if config.exists():
            if config.read_text(encoding="utf-8") != text:
                raise RuntimeError("Frozen input changed: " + str(config))
        else:
            config.write_text(text, encoding="utf-8", newline="\n")
    COMMON.save(INPUTS / "cases.json", cases)
    return cases


def audit(run, normalized, case):
    # 解決済み dump と実際の点を検査し、指定だけあって経路を通らない比較を拒む。
    text = (run / "config/config_data.txt").read_text(encoding="utf-8-sig")
    resolved = dict(re.findall(r"(?m)^\s*([^#=\n]+?)\s*=\s*(.*?)\s*$", text))
    for key, value in expected_values(case).items():
        if resolved.get(key) != value:
            raise RuntimeError(f"Unexpected config {key}: {resolved.get(key)!r}, expected {value!r}")
    points = [json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines()]
    if not points:
        raise RuntimeError("Missing scalar points")
    if "variant" not in case:
        return
    tags = {point[0] for point in points}
    finite = {tag for tag, _, value in points if math.isfinite(float.fromhex(value))}
    prefixes = ["46_agent_replay_fit/"]
    if VARIANTS[case["variant"]][3]:
        prefixes.append("36_agent_munchausen/")
    for prefix in prefixes:
        if not any(tag.startswith(prefix) for tag in finite):
            raise RuntimeError("Missing finite diagnostic: " + prefix)
    if case["mode"] == "iqn":
        prefixes = [f"36_agent_iqn/{index:02d}" for index in range(1, 8)]
        definitions_data = json.loads((run / "json/metrics.scalar.defs.json").read_text(encoding="utf-8-sig"))["data"]
        if not all(any(tag.startswith(prefix) for tag in definitions_data) for prefix in prefixes):
            raise RuntimeError("Missing IQN diagnostic definitions")
        # PER 無効時の初回行診断は未成立で、既存 Observer は NaN を出力しない。
        required = prefixes if VARIANTS[case["variant"]][0] else prefixes[:3]
        if not all(any(tag.startswith(prefix) for tag in tags) for prefix in required):
            raise RuntimeError("Missing IQN diagnostic tags")
        if not any(tag.startswith(tuple(required)) for tag in finite):
            raise RuntimeError("Missing finite IQN diagnostics")


def execute(binary, label, case, verification=None):
    config = INPUTS / (case + ".txt")
    if not config.is_file():
        raise FileNotFoundError(config)
    suffix = "tmp_125_" + label + "_" + case
    folder = ROOT / label / case
    info = COMMON.execute(
        [str(binary), "--config", str(config), "app.run_name=run_{t}_" + suffix], folder)
    runs = list((INPUTS / "runs").glob("run_*_" + suffix))
    if info["exit_code"] or len(runs) != 1:
        raise RuntimeError("Run failed: " + suffix)
    normalized = folder / "normalized.jsonl"
    NORMALIZER.normalize(runs[0], normalized, include_noisy=True)
    audit(runs[0], normalized, verification or definitions()[case])
    COMMON.save(folder / "inputs.json", {"run": str(runs[0]), "config_sha256": COMMON.digest(config)})
    return normalized


def repeat_case(label, case):
    try:
        first = execute(BASELINE / "Debug/AnetRLRunner.exe", label + "_1", case)
        second = execute(BASELINE / "Debug/AnetRLRunner.exe", label + "_2", case)
        result = {"exact_match": first.read_bytes() == second.read_bytes(),
                  "points": second.read_bytes().count(b"\n"),
                  "first": str(first), "second": str(second), "config_case": case}
        print("REPEAT", case, result, flush=True)
        return result
    except Exception as error:
        # 各組の失敗は残し、独立した組を引き続き確かめる。
        print("FAILED", case, str(error), flush=True)
        return {"exact_match": False, "error": str(error), "config_case": case}


def repeat(label, selected, workers):
    if (ROOT / (label + "-summary.json")).exists() or any((ROOT / (label + suffix)).exists() for suffix in ("_1", "_2")):
        raise FileExistsError("Use a fresh --label: " + label)
    cases = prepare()
    reports = {}
    # CPU の組だけを並行し、CUDA と性能測定は独立して順に流す。
    cpu = [case for case in selected if cases[case]["device"] == "cpu"]
    gpu = [case for case in selected if cases[case]["device"] == "cuda"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(repeat_case, label, case): case for case in cpu}
        for future in as_completed(pending):
            reports[pending[future]] = future.result()
            COMMON.save(ROOT / (label + "-summary.json"), reports)
    for case in gpu:
        reports[case] = repeat_case(label, case)
        COMMON.save(ROOT / (label + "-summary.json"), reports)
    # BF16 の LunarLander だけは D14 の順で CUDA を試す。FP32 は除外しない。
    for case in selected:
        if cases[case]["precision"] != "bf16" or reports[case]["exact_match"]:
            continue
        attempts = [reports[case]]
        if cases[case]["device"] == "cpu":
            fallback = case.removesuffix("cpu") + "cuda"
            attempts.append(repeat_case(label, fallback))
        chosen = next((attempt for attempt in attempts if attempt["exact_match"]), attempts[-1])
        reports[case] = {**chosen, "attempts": attempts, "excluded_bf16": not chosen["exact_match"]}
        COMMON.save(ROOT / (label + "-summary.json"), reports)
    if not all(item["exact_match"] or item.get("excluded_bf16", False) for item in reports.values()):
        raise RuntimeError("Baseline repeatability failed")


def suite(label):
    folder = ROOT / label
    folder.mkdir(parents=True, exist_ok=False)
    binary = BASELINE / "core/anet-core-test.exe" if label == "baseline-suite" else REPO / "core/anet-core/bin/Debug/anet-core-test.exe"
    xml = folder / "core.xml"
    info = COMMON.execute([str(binary), "--reporter", "junit", "--out", str(xml)],
                          folder / "process", cwd=REPO)
    report = {**info, "failures": COMMON.failures(xml)}
    COMMON.save(folder / "suite.json", report)
    print("SUITE", label, report, flush=True)
    if label != "baseline-suite":
        baseline = json.loads((ROOT / "baseline-suite/suite.json").read_text(encoding="utf-8"))
        if baseline["failures"] != report["failures"]:
            raise RuntimeError("Default suite failure set changed")


def compare_case(label, reference):
    case = reference["config_case"]
    try:
        before = Path(reference["first"])
        inputs = json.loads((before.parent / "inputs.json").read_text(encoding="utf-8"))
        if inputs["config_sha256"] != COMMON.digest(INPUTS / (case + ".txt")):
            raise RuntimeError("Comparison input differs from baseline: " + case)
        after = execute(REPO / "apps/runner/bin/Debug/AnetRLRunner.exe", label, case)
        result = {"exact_match": before.read_bytes() == after.read_bytes(),
                  "before_points": before.read_bytes().count(b"\n"),
                  "after_points": after.read_bytes().count(b"\n"),
                  "before_sha256": COMMON.digest(before), "after_sha256": COMMON.digest(after),
                  "first": str(before), "after": str(after), "config_case": case}
        print("COMPARE", case, result, flush=True)
        return result
    except Exception as error:
        print("FAILED", case, str(error), flush=True)
        return {"exact_match": False, "error": str(error), "config_case": case}


def compare(label, reference_label, selected, workers):
    if (ROOT / (label + "-summary.json")).exists() or (ROOT / label).exists():
        raise FileExistsError("Use a fresh --label: " + label)
    cases = prepare()
    reference = json.loads((ROOT / (reference_label + "-summary.json")).read_text(encoding="utf-8"))
    if any(not reference[case]["exact_match"] and not reference[case].get("excluded_bf16", False)
           for case in selected):
        raise RuntimeError("Baseline repeatability has not passed")
    reports = {case: {"excluded_bf16": True, "reason": reference[case]}
               for case in selected if reference[case].get("excluded_bf16", False)}
    included = [case for case in selected if case not in reports]
    cpu = [case for case in included if cases[reference[case]["config_case"]]["device"] == "cpu"]
    gpu = [case for case in included if case not in cpu]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(compare_case, label, reference[case]): case for case in cpu}
        for future in as_completed(pending):
            reports[pending[future]] = future.result()
            COMMON.save(ROOT / (label + "-summary.json"), reports)
    for case in gpu:
        reports[case] = compare_case(label, reference[case])
        COMMON.save(ROOT / (label + "-summary.json"), reports)
    if not all(item.get("exact_match", False) or item.get("excluded_bf16", False)
               for item in reports.values()):
        raise RuntimeError("Learning scalar mismatch")


def performance(label, rounds):
    if (ROOT / (label + "-summary.json")).exists() or (ROOT / (label + "_warmup")).exists():
        raise FileExistsError("Use a fresh --label: " + label)
    active = subprocess.check_output([
        "powershell", "-NoProfile", "-Command",
        "Get-Process | Where-Object { $_.ProcessName -match '^(AnetRLRunner.*|anet-core-test|cl|ninja|cmake)$' } | Select-Object -ExpandProperty ProcessName"],
        text=True).strip()
    if active:
        raise RuntimeError("Performance measurement blocked by active processes: " + active)
    prepare()
    case = {"env": "Atari", "device": "cuda", "precision": "bf16", "performance": True}
    lines = SETTINGS.settings("Atari", before=True)
    chain, _, validation = lines[-1].rpartition(" > ")
    lines[-1] = chain + " > run.@btrnet > " + validation
    extra = {**expected_values(case), "app.exp_exit_step": "82176", "app.batchrun.exp_exit_step": "82176",
             "DefaultDQNAgent.learner.munchausen.log_policy_mode": "target"}
    for key, value in extra.items():
        lines.insert(-1, "run.@prd098_validation : " + key + " = " + value)
    text = "\n".join(lines).replace("prd098", "prd125") + "\n"
    config = INPUTS / "performance.txt"
    if config.exists() and config.read_text(encoding="utf-8") != text:
        raise RuntimeError("Frozen performance input changed")
    if not config.exists():
        config.write_text(text, encoding="utf-8", newline="\n")
    binaries = {"before": BASELINE / "Release/AnetRLRunner.exe",
                "after": REPO / "apps/runner/bin/Release/AnetRLRunner.exe"}
    # 捨て Run のあと、各巡で基準→変更後を同じ入力で流す。
    execute(binaries["before"], label + "_warmup", "performance", case)
    results = []
    for round_index in range(1, rounds + 1):
        for side, binary in binaries.items():
            normalized = execute(binary, f"{label}_{round_index}_{side}", "performance", case)
            inputs = json.loads((normalized.parent / "inputs.json").read_text(encoding="utf-8"))
            timing = {row["step"]: row["value"] for row in SETTINGS.rows(Path(inputs["run"]))
                      if row.get("type") == "scalar" and row["tag"] == "90_perf/98_prd125_time"}
            seconds = (timing[256] - timing[64]) * 3600 / 192
            results.append({"round": round_index, "side": side, "seconds_per_update": seconds,
                            "run": inputs["run"], "binary_sha256": COMMON.digest(binary)})
            COMMON.save(ROOT / (label + "-summary.json"), {"results": results})
            print("PERFORMANCE", side, round_index, seconds, flush=True)
    before = [item["seconds_per_update"] for item in results if item["side"] == "before"]
    after = [item["seconds_per_update"] for item in results if item["side"] == "after"]
    report = {"rounds": rounds, "results": results, "before_median": statistics.median(before), "before_max": max(before),
              "after_median": statistics.median(after), "passed": statistics.median(after) <= max(before)}
    COMMON.save(ROOT / (label + "-summary.json"), report)
    if not report["passed"]:
        raise RuntimeError("Performance regression beyond baseline variation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "prepare", "repeat", "suite", "compare", "performance"))
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--case", nargs="+", choices=tuple(definitions()))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.mode == "performance" and args.rounds < 3:
        parser.error("performance requires --rounds >= 3")
    selected = args.case or [case for case, values in definitions().items()
                            if values["precision"] == "fp32" or values["device"] == "cpu"
                            or case == "bf16_Atari_cuda"]
    if args.mode == "snapshot":
        snapshot()
    elif args.mode == "suite":
        suite(args.label)
    elif args.mode == "compare":
        compare(args.label, args.reference, selected, args.workers)
    elif args.mode == "performance":
        performance(args.label, args.rounds)
    elif args.mode == "prepare":
        cases = prepare()
        for case in selected:
            print("PREPARED", case, cases[case], flush=True)
    else:
        repeat(args.label, selected, args.workers)


if __name__ == "__main__":
    main()
