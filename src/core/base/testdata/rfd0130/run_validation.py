"""RFD0130: 凍結した実行体で機能を比較する。性能の測定は行わない。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[5]
ROOT = REPO / ".scratch/rfd0130/acceptance-cuda"
PROFILE = "run.@rfd0130_validation"
CASES = ("off", "feedback", "cumulative", "ema", "collect", "cuda1", "cuda2")
FEATURE_VARIANCE = "56_agent_in/14_target_feature_var"


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_module("rfd0130_common", "src/core/base/testdata/rfd0109/run_validation.py")
METRICS = load_module("rfd0130_metrics", "src/core/base/testdata/rfd0084/compare_metrics.py")


def snapshot(label):
    # 実行体と設定をコピーし、更新しない外部 DLL だけをハードリンクする。
    target = ROOT / label / "package"
    target.mkdir(parents=True, exist_ok=False)
    for name, source in (("Debug", REPO / "apps/runner/bin/Debug"),
                         ("Release", REPO / "apps/runner/bin/Release"),
                         ("core", REPO / "out/test/Debug")):
        destination = target / name
        destination.mkdir()
        for path in source.iterdir():
            if path.suffix.lower() == ".exe":
                shutil.copy2(path, destination / path.name)
            elif path.suffix.lower() == ".dll":
                (destination / path.name).hardlink_to(path)
    shutil.copytree(REPO / "apps/runner/config", target / "config")
    sources = target / "sources"
    sources.mkdir()
    for name in ("intrinsic_rnd.cpp", "intrinsic_rnd.hpp", "scaler_impl.cpp", "scaler_impl.hpp"):
        shutil.copy2(REPO / "src/core/base/src" / name, sources / name)
    shutil.copy2(REPO / "src/core/base/include/anet/scaler.hpp", sources / "scaler.hpp")
    COMMON.save(target / "manifest.json", {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
        "sha256": {path.relative_to(target).as_posix(): COMMON.digest(path)
                   for path in target.rglob("*") if path.is_file()},
    })
    print("SNAPSHOT", target, flush=True)


def prepare():
    # 両側で同じ設定を読み、出荷の連鎖と Taiga の 8 値を凍結する。
    destination = ROOT / "inputs"
    destination.mkdir(exist_ok=False)
    config = ROOT / "before/package/config"
    lunar = (config / "LunarLander.txt").read_text(encoding="utf-8-sig")
    common = (config / "common.txt").read_text(encoding="utf-8-sig")
    chain = re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)", lunar)[-1].strip()
    atari = (config / "Atari.txt").read_text(encoding="utf-8-sig")
    taiga = re.findall(r"(?m)^run\.@rnd_taiga\s*:\s*(DefaultDQNAgent\.intrinsic\.[^\n#]+)", atari)
    if len(taiga) != 8:
        raise RuntimeError(f"Atari Taiga settings count={len(taiga)}; expected=8")
    tags = sorted(set(re.findall(r"run\.eval(?:_schedule)?\.\[([^]]+)\]", lunar + common)))
    for case in CASES:
        selected = chain + (" > run.@rnd" if case != "off" else "") + " > " + PROFILE
        settings = ["app.$ = app.batchrun > P1", "app.show_error_dialog = false",
                    "app.train_auto_start = true", "app.eval_panel.auto_start = false",
                    "app.save_agent_on_close = true", "app.train_panel.fps = 0",
                    "app.batchrun.train_panel.fps = 0", "metrics.image.interval = 0",
                    "app.exp_exit_step = 100000", "app.batchrun.exp_exit_step = 100000",
                    "backend.$ = backend.@deterministic", "backend.deterministic_warn_only = false",
                    "backend.torch_num_threads = 1", "run.seed = 1",
                    "agent.device = cuda"]
        settings += [f"run.eval_schedule.[{tag}].interval = 0" for tag in tags]
        if case == "feedback":
            settings += [value.strip() for value in taiga]
        if case == "ema":
            settings += ["DefaultDQNAgent.intrinsic.rnd.error_norm.average_mode = ema"]
        if case == "collect":
            settings += ["DefaultDQNAgent.intrinsic.rnd.warmup_mode = collect_errors"]
        text = [f"$include <{config.as_posix()}/_main.txt>", f"$include <{config.as_posix()}/LunarLander.txt>"]
        text += [PROFILE + " : " + value for value in settings]
        text += ["run.$ = " + selected]
        (destination / (case + ".txt")).write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    COMMON.save(destination / "manifest.json", {p.name: COMMON.digest(p) for p in destination.glob("*.txt")})
    print("PREPARED", destination, flush=True)


def make_environment():
    # 保存した旧実行体を含め、検証用の一時スイッチを親から継承しない。
    return {key: value for key, value in dict(os.environ, OMP_NUM_THREADS="1").items()
            if not key.startswith("ANET_RFD")}


def run_case(label, case, checkpoint=None):
    # 稼働中の評価 Run は操作せず、専用の出力先で機能だけを確かめる。
    suffix = ("load_" if checkpoint else "") + case
    destination = ROOT / label / suffix
    config = ROOT / "inputs" / (case + ".txt")
    binary = ROOT / label / "package/Debug/AnetRLRunner.exe"
    run_name = "run_p130_" + label + "_" + suffix
    runs = ROOT / label / "runs"
    command = [str(binary), "--config", str(config), "app.run_name=" + run_name,
               "app.runs_dir=" + runs.as_posix()]
    if checkpoint:
        command += ["DefaultDQNAgent.auto_load_file=" + checkpoint.as_posix()]
    environment = make_environment()
    info = COMMON.execute(command, destination, timeout=7200, env=environment)
    run = runs / run_name
    # 秘密の値をログへ写さず、実際に渡した環境と再現に効く値を追えるようにする。
    backend_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER", "PYTORCH_CUDA_ALLOC_CONF", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    COMMON.save(destination / "inputs.json", {
        "config_sha256": COMMON.digest(config), "run": str(run),
        "environment_sha256": hashlib.sha256(json.dumps(environment, sort_keys=True).encode("utf-8")).hexdigest(),
        "backend_environment": {key: environment.get(key) for key in backend_keys},
        "removed_test_env_keys": sorted(key for key in os.environ if key.startswith("ANET_RFD")),
        "checkpoint": str(checkpoint) if checkpoint else None})
    if info["exit_code"] or not (run / "agent_close.anet").is_file():
        raise RuntimeError(f"Functional Run {label}/{suffix} failed: exit={info['exit_code']}, run={run}")
    return run


def recorded_run(label, case):
    # 過去の採取結果を改名せず、実行時の記録を正本にする。
    return Path(json.loads((ROOT / label / case / "inputs.json").read_text(encoding="utf-8"))["run"])


def normalize_run(run, output, exclude_prefixes=()):
    # 時間・資源だけの記録は、一致していても機能検証の根拠にしない。
    METRICS.normalize(run, output, include_noisy=True, exclude_prefixes=exclude_prefixes)
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    if summary["point_count"] == 0:
        raise RuntimeError("Empty learning scalar metrics: " + str(run))
    return COMMON.digest(output)


def compare(left_label, right_label, case, checkpoint=False):
    # 継続も同じ採り方で比較し、参照する Run の種類だけを明示する。
    suffix = ("load_" if checkpoint else "") + case
    folder = ROOT / (left_label + "-vs-" + right_label) / suffix
    folder.mkdir(parents=True, exist_ok=False)
    hashes = []
    for index, label in enumerate((left_label, right_label)):
        run = recorded_run(label, suffix)
        output = folder / (str(index) + ".jsonl")
        hashes.append(normalize_run(run, output,
            (FEATURE_VARIANCE,) if not checkpoint and case == "feedback" and left_label == "before" else ()))
    COMMON.save(folder / "comparison.json", {"exact_match": hashes[0] == hashes[1], "sha256": hashes})
    if hashes[0] != hashes[1]:
        raise RuntimeError(f"Scalar checksum mismatch: {left_label}/{right_label}/{case}")


def repeat_cuda(label):
    # 同じ構成の 2 回は、時間・資源以外の全 scalar 点を比較する。
    hashes = []
    for case in ("cuda1", "cuda2"):
        output = ROOT / label / (case + ".jsonl")
        hashes.append(normalize_run(recorded_run(label, case), output))
    COMMON.save(ROOT / label / "cuda-repeat.json", {"exact_match": hashes[0] == hashes[1], "sha256": hashes})
    if hashes[0] != hashes[1]:
        raise RuntimeError("CUDA repeat scalar mismatch: " + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("snapshot", "prepare", "run", "compare", "compare-load", "load", "repeat"))
    parser.add_argument("--label", default="before")
    parser.add_argument("--reference", default="before")
    parser.add_argument("--case", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if args.phase == "snapshot":
        snapshot(args.label)
        return
    if args.phase == "prepare":
        prepare()
        return
    if args.phase == "repeat":
        repeat_cuda(args.label)
        return
    errors = []
    for case in args.case:
        try:
            if args.phase in ("compare", "compare-load"):
                compare(args.reference, args.label, case, checkpoint=args.phase == "compare-load")
            elif args.phase == "load":
                if case not in ("cumulative", "ema", "feedback"):
                    raise ValueError("Checkpoint case must be cumulative, ema, or feedback")
                checkpoint = recorded_run(args.reference, case) / "agent_close.anet"
                run_case(args.label, case, checkpoint)
            else:
                run_case(args.label, case)
            print("PASS", args.phase, case, flush=True)
        except Exception as error:
            errors.append(f"{case}: {error}")
            print("FAIL", errors[-1], flush=True)
    COMMON.save(ROOT / args.label / (args.phase + "-" + "-".join(args.case) + "-summary.json"), {"errors": errors})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
