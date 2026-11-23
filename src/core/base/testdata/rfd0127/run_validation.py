"""RFD0127: 変更前の設定を固定した決定的Runと全scalarの比較。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[5]
OUT = REPO / ".scratch/prd127/baseline-20261112"
CASES = ("GridMaze", "LunarLander", "Atari", "DropMerge", "collect", "feedback")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "after", "compare"))
    parser.add_argument("--label", default="after-r1")
    parser.add_argument("--case", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--retry-incomplete", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("rfd0098_validation", REPO / "src/core/base/testdata/rfd0098/run_validation.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.OUT = OUT
    binary = OUT / "baseline-bin/AnetRLRunner.exe"
    if args.stage == "after":
        source = REPO / "apps/runner/bin/Debug/AnetRLRunner.exe"
        folder = OUT / (args.label + "-bin")
        folder.mkdir(exist_ok=True)
        binary = folder / source.name
        if binary.exists() and digest(binary) != digest(source):
            raise RuntimeError("Binary changed; choose a fresh --label")
        if not binary.exists():
            shutil.copy2(source, binary)
            for dll in source.parent.glob("*.dll"):
                (folder / dll.name).hardlink_to(dll)
    config_root = OUT / "baseline-config"
    inputs = hashlib.sha256()
    for path in sorted(config_root.rglob("*.txt")):
        inputs.update(path.relative_to(config_root).as_posix().encode() + b"\0")
        inputs.update(hashlib.sha256(path.read_bytes()).digest())
    errors = []
    for case in args.case:
        name = ("before" if args.stage == "before" else args.label) + "_" + case
        try:
            if args.stage == "compare":
                driver.compare(args.label, case)
                definitions = []
                for side in ("before", args.label):
                    info = json.loads((OUT / f"{side}_{case}/execution.json").read_text(encoding="utf-8"))
                    definitions.append(json.loads((Path(info["run"]) / "json/metrics.scalar.defs.json").read_text(encoding="utf-8-sig")))
                if definitions[0] != definitions[1]:
                    raise RuntimeError("Metric definitions differ")
                print("PASS scalar checksum and definitions", case, flush=True)
                continue
            env = "LunarLander" if case in ("collect", "feedback") else case
            lines = driver.settings(env, before=True)
            enabled = case in ("collect", "feedback")
            if enabled:
                chain, _, validation = lines[-1].rpartition(" > ")
                lines[-1] = chain + " > run.@rnd > " + validation
            overrides = ["DefaultDQNAgent.intrinsic.enabled=" + str(enabled).lower()]
            if enabled:
                resolved = json.loads((OUT / "resolved-before/LunarLander.json").read_text(encoding="utf-8"))
                overrides += ["run.train.num_envs=" + resolved["run.train.num_envs"], "app.exp_exit_step=100000", "app.batchrun.exp_exit_step=100000", "app.save_agent_on_close=true"]
            if case == "feedback":
                # Atari の run.@rnd_taiga の8値を、同じLunarLanderへ明示する。
                overrides += ["DefaultDQNAgent.intrinsic.observation=state", "DefaultDQNAgent.intrinsic.learn_ratio=1",
                    "DefaultDQNAgent.intrinsic.learn_batch_size=1", "DefaultDQNAgent.intrinsic.beta=0.00005",
                    "DefaultDQNAgent.intrinsic.rnd.multiplier_mode=unclipped",
                    "DefaultDQNAgent.intrinsic.rnd.statistics_mode=variance_feedback",
                    "DefaultDQNAgent.intrinsic.rnd.warmup_steps=0", "DefaultDQNAgent.intrinsic.rnd.adam_eps=1e-5"]
            config_text = "\n".join(lines) + "\n"
            fingerprint = {"binary_sha256": digest(binary), "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                "config_inputs_sha256": inputs.hexdigest(), "overrides_sha256": hashlib.sha256(json.dumps(overrides).encode()).hexdigest()}
            destination = OUT / name
            record = destination / "execution.json"
            if record.exists():
                saved = json.loads(record.read_text(encoding="utf-8"))
                if saved.get("exit_code") == 0 and saved.get("run") and all(saved.get(k) == v for k, v in fingerprint.items()):
                    print("REUSE", name, flush=True)
                    continue
                if not args.retry_incomplete:
                    raise RuntimeError("Incomplete or changed record; use --retry-incomplete or a fresh --label")
                archived = OUT / (name + "_incomplete_" + str(time.time_ns()))
                if not destination.resolve().is_relative_to(OUT.resolve()) or not archived.resolve().is_relative_to(OUT.resolve()):
                    raise RuntimeError("Record archive leaves the validation directory")
                destination.rename(archived)
            destination.mkdir(parents=True, exist_ok=True)
            config = destination / "run.txt"
            config.write_text(config_text, encoding="utf-8", newline="\n")
            # 試行ごとの名前で、中断前のRunを誤って今回の成果物に選ばない。
            suffix = "p127_" + str(time.time_ns())
            command = [str(binary), "--config", str(config), "app.run_name=run_{t}_" + suffix] + overrides
            info = dict(fingerprint, arguments=command)
            driver.save(record, info)
            print("START", name, flush=True)
            start = time.monotonic()
            with (destination / "launcher.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=REPO / "apps/runner", stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                info["pid"] = process.pid
                driver.save(record, info)
                code = process.wait()
            runs = list((OUT / "runs").glob("run_*_" + suffix))
            info.update(exit_code=code, seconds=time.monotonic() - start, run=str(runs[0]) if len(runs) == 1 else None)
            driver.save(record, info)
            print("END", name, code, round(info["seconds"], 1), flush=True)
            if code or len(runs) != 1:
                raise RuntimeError("Run failed")
            print("PASS", case, info["run"], flush=True)
        except Exception as error:
            errors.append(case + ": " + str(error))
            print("ERROR", errors[-1], flush=True)
    driver.save(OUT / (args.stage + "_" + args.label + "_" + "-".join(args.case) + "_summary.json"), {"errors": errors})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
