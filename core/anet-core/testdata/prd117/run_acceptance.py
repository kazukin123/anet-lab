"""Runの構成を実際のConfigManagerで解決し、通常実行の受入を確かめる。"""

import json
import hashlib
import argparse
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / ".scratch/prd117/acceptance"
ENVIRONMENTS = ("Atari", "CartPole", "DropMerge", "GridMaze", "GridMaze_muzero", "ImageCls", "LunarLander")


def migrate_legacy_rnd_input() -> Path:
    source_root = ROOT / "apps/runner/workspaces/AtariRnd-01/config"
    destination_root = OUTPUT / "legacy-rnd-current"
    records = []
    for source in sorted(source_root.rglob("*.txt")):
        relative = source.relative_to(source_root)
        if relative.parts[0] != "frozen" and relative.as_posix() != "rnd_explore.txt":
            continue
        original = source.read_bytes()
        output = []
        expanded = 0
        removed = 0
        for line in original.decode("utf-8-sig").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                output.append(line)
                continue
            if "metrics.image.histgram.interval" in line or re.search(r"heatmap\.probe\.[xy]\.auto_scale_mode", line):
                removed += 1
                continue
            line = line.replace("mergin_x", "margin_x").replace("mergin_y", "margin_y")
            # 左辺のprofile境界を現行の表記へ揃える。値は変えない。
            assignment = re.search(r"\?=|=", line)
            if assignment:
                left, right = line[:assignment.start()], line[assignment.start():]
                if ":" not in left:
                    left = re.sub(r"(@[^\s.:]+)\.", r"\1 : ", left, count=1)
                line = left + right
            if "intrinsic.rnd.net." in line:
                output.append(line.replace("intrinsic.rnd.net.", "intrinsic.rnd.target.net."))
                output.append(line.replace("intrinsic.rnd.net.", "intrinsic.rnd.predictor.net."))
                expanded += 1
            else:
                output.append(line)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        current = ("\n".join(output) + "\n").encode("utf-8")
        destination.write_bytes(current)
        records.append({"source": str(source), "copy": str(destination),
                        "source_sha256": hashlib.sha256(original).hexdigest(),
                        "copy_sha256": hashlib.sha256(current).hexdigest(),
                        "shared_rnd_lines_expanded": expanded, "unused_image_keys_removed": removed})
    (destination_root / "migration.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return destination_root / "rnd_explore.txt"


def batch_chains() -> list[dict]:
    matrix = {}
    for filename in ("11_batch_run.bat", "12_batch_run.bat", "13_batch_run.bat"):
        lines = (ROOT / "apps" / filename).read_text(encoding="cp932").splitlines()
        uses = [(index, re.search(r"call\s+:use_ws\s+\S+\s+(\S+)", line, re.I))
                for index, line in enumerate(lines)]
        uses = {index: match.group(1) for index, match in uses if match}
        environment = next(iter(uses.values()))
        variables = {}
        for index, line in enumerate(lines):
            environment = uses.get(index, environment)
            match = re.match(r'\s*SET\s+"([^=]+)=(.*)"\s*$', line, re.I)
            if not match:
                continue
            key, value = match.groups()
            value = re.sub(r"%([A-Za-z0-9_]+)%", lambda item: variables.get(item[1].upper(), item[0]), value)
            variables[key.upper()] = value
            if not value.startswith("run.$="):
                continue
            chain = value.removeprefix("run.$=")
            if "%" in chain:
                raise RuntimeError(f"Unresolved batch chain: {filename}:{index + 1}: {chain}")
            entry = matrix.setdefault((environment, chain),
                                      {"config": environment, "chain": chain, "sources": []})
            entry["sources"].append(f"apps/{filename}:{index + 1} ({key})")
    return list(matrix.values())


def resolve(input_path: Path, output_path: Path) -> dict:
    environment = os.environ.copy()
    environment["ANET_PRD117_RESOLVE_INPUT"] = str(input_path)
    environment["ANET_PRD117_RESOLVE_OUTPUT"] = str(output_path)
    result = subprocess.run(
        [str(ROOT / "core/anet-core/bin/Debug/anet-core-test.exe"), "[prd117-resolve]"],
        cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    log = output_path.with_suffix(".log")
    log.write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Config resolution failed: exit={result.returncode}, log={log}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def prepare_matrix() -> list[dict]:
    prepared = OUTPUT / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    cases = [{"config": name + ".txt", "chain": "", "sources": ["default " + name]}
             for name in ENVIRONMENTS]
    cases.extend(batch_chains())
    legacy_input = migrate_legacy_rnd_input()
    for index, case in enumerate(cases, 1):
        name = f"{index:02d}-{Path(case['config']).stem}"
        input_path = prepared / (name + ".txt")
        output_path = prepared / (name + ".json")
        config_directory = ROOT / "apps/runner/config"
        if "run.@rnd_nn" in case["chain"].split(">"):
            text = f"$include <{legacy_input.as_posix()}>\n"
            case["migration"] = str(legacy_input.parent / "migration.json")
        else:
            text = (f"$include <{(config_directory / '_main.txt').as_posix()}>\n"
                    f"$include <{(config_directory / case['config']).as_posix()}>\n")
        if case["chain"]:
            text += f"run.$ = {case['chain']}\n"
        input_path.write_text(text, encoding="utf-8", newline="\n")
        case.update({"id": index, "name": name, "input": str(input_path), "resolved": str(output_path)})
        try:
            metadata = resolve(input_path, output_path)
        except RuntimeError as error:
            case["resolve_error"] = str(error)
            print(f"Unresolved {name}: {error}", flush=True)
            continue
        data = metadata["config"]
        learn_tags = [tag for tag, definition in metadata["scalar_defs"].items()
                      if definition["event"] == "learn" and definition["scope"] == "train"]
        print(f"Prepared {name}: agent={data.get('agent.class_id')}, lanes={data.get('run.train.num_envs')}, "
              f"warmup={data.get('DefaultDQNAgent.learner.update_warmup_steps')}, learn_tags={len(learn_tags)}", flush=True)
    (OUTPUT / "matrix.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cases


def integer(value: str) -> int:
    return int(value.replace(",", ""))


def prepare_eval_case(cases: list[dict]) -> dict:
    default = cases[0]
    metadata = json.loads(Path(default["resolved"]).read_text(encoding="utf-8"))
    chain = next(item["chain"] for item in metadata["resolution"]["selections"] if item["key"] == "run.$")
    text = f"$include <{Path(default['input']).as_posix()}>\n"
    text += "run.$ = " + " > ".join(item["term"] for item in chain) + " > run.@evalonly\n"
    case = {"id": 27, "name": "27-Atari-evalonly", "config": "Atari.txt", "eval_only": True,
            "chain": " > ".join(item["term"] for item in chain) + " > run.@evalonly",
            "sources": ["default Atari + run.@evalonly"],
            "input": str(OUTPUT / "prepared/27-Atari-evalonly.txt"),
            "resolved": str(OUTPUT / "prepared/27-Atari-evalonly.json")}
    Path(case["input"]).write_text(text, encoding="utf-8", newline="\n")
    resolve(Path(case["input"]), Path(case["resolved"]))
    return case


def run_case(case: dict) -> dict:
    metadata = json.loads(Path(case["resolved"]).read_text(encoding="utf-8"))
    data = metadata["config"]
    learn_tags = {tag for tag, definition in metadata["scalar_defs"].items()
                  if definition["event"] == "learn" and definition["scope"] == "train"}
    intervals = [definition["interval"] for tag, definition in metadata["scalar_defs"].items()
                 if tag in learn_tags and definition["interval"] > 0]
    lanes = integer(data["run.train.num_envs"])
    agent = data["agent.class_id"]
    if case.get("eval_only"):
        # 最初の通知がstep 0とは限らないので、実行中の評価間隔を越えてから通常停止する。
        eval_intervals = [integer(value) for key, value in data.items()
                          if key.startswith("run.eval_schedule.[") and key.endswith("].interval")
                          and integer(value) > 0]
        if not eval_intervals:
            raise RuntimeError("Eval-only acceptance requires an active evaluation schedule")
        budget = min(eval_intervals) + lanes
    elif agent == "DefaultDQNAgent":
        warmup = integer(data.get(agent + ".learner.update_warmup_steps", "0"))
        batch = integer(data.get(agent + ".learner.replay_batch_size", "128"))
        cadence = integer(data.get(agent + ".learner.update_interval", "1"))
        budget = max(warmup, batch) + lanes * (min(intervals, default=1) * cadence + 64)
    elif agent == "MuZeroAgent":
        budget = lanes * (200 + min(intervals, default=1))
    else:
        budget = lanes * (min(intervals, default=1) + 16)
    runs = OUTPUT / "runs"
    destination = runs / case["name"]
    attempt = 1
    while destination.exists():
        attempt += 1
        destination = runs / f"{case['name']}-attempt{attempt}"
    destination.mkdir(parents=True, exist_ok=True)
    input_path = destination / "native-input.txt"
    text = f"$include <{Path(case['input']).as_posix()}>\n"
    if case.get("eval_only"):
        candidates = sorted((OUTPUT / "runs").glob("01-Atari*/acceptance.json"),
                            key=lambda path: path.stat().st_mtime, reverse=True)
        source = next((path.parent for path in candidates if json.loads(path.read_text())["passed"]), None)
        if source is None:
            raise RuntimeError("Atari evalonly requires a passing default Atari Run checkpoint")
        checkpoint = source / "agent_close.anet"
        if not checkpoint.is_file():
            raise RuntimeError(f"Default Atari checkpoint is missing: {checkpoint}")
        text += f"DefaultDQNAgent.auto_load_file = {checkpoint.as_posix()}\n"
    for prefix in ("app", "app.online", "app.batchrun"):
        for key, value in {"exp_exit_step": str(budget), "exp_pause_step": "-1",
                           "train_pause_step": "-1", "show_error_dialog": "false",
                           "train_panel.fps": "0", "eval_panel.fps": "0"}.items():
            text += f"{prefix}.{key} = {value}\n"
    text += f"app.runs_dir = {runs.as_posix()}\napp.run_name = {destination.name}\n"
    input_path.write_text(text, encoding="utf-8", newline="\n")
    command = [str(ROOT / "apps/runner/bin/Debug/AnetRLRunner.exe"), "--config", str(input_path)]
    for prefix in ("app", "app.online", "app.batchrun"):
        for key, value in {"exp_exit_step": str(budget), "exp_pause_step": "-1",
                           "train_pause_step": "-1", "show_error_dialog": "false",
                           "train_panel.fps": "0", "eval_panel.fps": "0",
                           "runs_dir": runs.as_posix(), "run_name": destination.name}.items():
            command.append(f"{prefix}.{key}={value}")
    started = time.monotonic()
    with (destination / "launch.log").open("wb") as console:
        process = subprocess.Popen(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
        # 通常停止はアプリ自身のexp_exit_stepに任せ、強制終了で成功にしない。
        code = process.wait()
    return assess_case(case, metadata, destination, command, input_path, budget, code, time.monotonic() - started)


def assess_case(case: dict, metadata: dict, destination: Path, command: list[str],
                input_path: Path, budget: int, code: int, elapsed: float | None) -> dict:
    data = metadata["config"]
    learn_tags = {tag for tag, definition in metadata["scalar_defs"].items()
                  if definition["event"] == "learn" and definition["scope"] == "train"}
    log_path = destination / (destination.name + ".log")
    logs = log_path.read_text(encoding="utf-8-sig", errors="replace") if log_path.is_file() else ""
    records = []
    learn_steps = [int(value) for value in re.findall(
        r"\btrain_step=\d+\s+exp_step=\d+\s+learn_step=(\d+)\b", logs)]
    metrics_path = destination / "metrics.jsonl"
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("type") == "scalar" and record.get("tag") in learn_tags:
                records.append(record)
            definition = metadata["scalar_defs"].get(record.get("tag"), {})
            value = record.get("value")
            if (record.get("type") == "scalar" and definition.get("target") == "runner"
                    and definition.get("scope") == "train" and definition.get("source_key") == "learn_step"
                    and isinstance(value, (int, float))):
                learn_steps.append(value)
    warnings = [line for line in logs.splitlines() if ": [W] " in line]
    errors = [line for line in logs.splitlines() if ": [E] " in line]
    drained = "Draining background observers:" in logs
    saved = not data.get("app.save_agent_on_close", "true") == "true" or "Finished saving agent:" in logs
    session_ends = [line for line in logs.splitlines() if ": session end " in line]
    observed_learn_step = max(learn_steps, default=0)
    reached = bool(records) if learn_tags else observed_learn_step >= 1
    if case.get("eval_only"):
        reached = bool(session_ends)
    result = {"case": case, "command": command, "input": str(input_path), "exp_exit_step": budget,
              "exit_code": code, "elapsed_sec": elapsed,
              "learn_scalar_count": len(records), "first_learn_scalar": records[:1],
              "observed_learn_step": observed_learn_step, "session_ends": session_ends,
              "warnings": warnings, "errors": errors, "drained": drained, "saved": saved,
              "log": str(log_path), "passed": code == 0 and reached and not warnings and not errors and drained and saved}
    (destination / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{case['name']}: exit={code}, learn={len(records)}, WARN={len(warnings)}, passed={result['passed']}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the already resolved native matrix")
    parser.add_argument("--summarize", action="store_true", help="Report the latest saved attempt for each matrix case")
    parser.add_argument("--only", type=int, nargs="+", help="Run selected matrix IDs")
    args = parser.parse_args()
    if args.summarize:
        cases = json.loads((OUTPUT / "matrix.json").read_text(encoding="utf-8"))
        latest = {}
        for path in sorted((OUTPUT / "runs").glob("*/acceptance.json"), key=lambda item: item.stat().st_mtime):
            result = json.loads(path.read_text(encoding="utf-8"))
            latest[result["case"]["id"]] = {"result": str(path), "passed": result["passed"],
                "exit_code": result["exit_code"], "learn_scalar_count": result["learn_scalar_count"],
                "warning_count": len(result["warnings"]), "errors": result["errors"]}
        report = [{"id": case["id"], "name": case["name"], **latest.get(case["id"], {"passed": False, "pending": True})}
                  for case in cases]
        (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Passed: {sum(item['passed'] for item in report)}/{len(report)}; "
              f"pending: {sum(item.get('pending', False) for item in report)}", flush=True)
        for item in report:
            if not item["passed"]:
                print(json.dumps(item), flush=True)
        return 0 if report and all(item["passed"] for item in report) else 1
    if not args.run:
        return 1 if any("resolve_error" in case for case in prepare_matrix()) else 0
    cases = json.loads((OUTPUT / "matrix.json").read_text(encoding="utf-8"))
    if not args.only or 27 in args.only:
        cases = [case for case in cases if case["id"] != 27]
        cases.append(prepare_eval_case(cases))
        (OUTPUT / "matrix.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    if args.only:
        cases = [case for case in cases if case["id"] in args.only]
    results = [run_case(case) for case in cases]
    return 0 if results and all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
