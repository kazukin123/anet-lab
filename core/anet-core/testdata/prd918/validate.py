"""PRD918 の設定移行と短い Run の受入を確認する。"""

import argparse
import gzip
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUTPUT = REPO / ".scratch/prd918"


def static_checks():
    # bat の各評価専用変数は、指定したタグごとに初回評価を明示する。
    variables = {"11": ("EV2X2",), "12": ("EVONLY", "GEVONLY", "GSESBIG"), "13": ("EVTAU",)}
    for number, names in variables.items():
        path = REPO / f"apps/{number}_batch_run.bat"
        raw = path.read_bytes()
        assert b"\n" not in raw.replace(b"\r\n", b""), path
        source = raw.decode("cp932")
        for name in names:
            line = next(line for line in source.splitlines() if line.startswith(f'SET "{name}='))
            tags = re.findall(r"run\.eval_schedule\.\[([^]]+)\]\.interval=(\d+)", line)
            assert tags, name
            for tag, interval in tags:
                assert f"run.eval_schedule.[{tag}].warmup_count=0" in line, (name, tag)
                assert int(interval) == (1000 if name == "EV2X2" else 10), (name, tag, interval)
        if number == "11":
            assert "run.eval_schedule.[eval_target].offset=0" in source
        if number == "12":
            for name in ("GSESBIG",):
                line = next(line for line in source.splitlines() if line.startswith(f'SET "{name}='))
                for tag in ("greedy", "greedy_single"):
                    assert f"run.eval_schedule.[{tag}].offset=0" in line, (name, tag)
            twses = next(line for line in source.splitlines() if line.startswith('SET "TWSES='))
            assert "run.eval_schedule.[greedy_single].interval=1000 " in twses
            assert "run.eval_schedule.[greedy_single].offset=50" in twses
    # 過去の Run の写しを含めず、PRD A8 の追跡対象を検索する。
    patterns = [
        ("互いに素|同位相", "apps/runner/config"),
        ("learn_step 0", "apps"),
        (r"run\.eval_schedule\.\[[a-z_]+\]\.interval *=? *(1009|53|503|11|12)\b", "apps"),
    ]
    for pattern, directory in patterns:
        result = subprocess.run(["git", "grep", "-n", "-E", pattern, "--", directory],
            cwd=REPO, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 1, result.stdout or result.stderr
    assert not subprocess.check_output(["git", "diff", "--", "core/anet-core/include/anet/util.hpp"], cwd=REPO)
    print("PASS: bat encoding, migrated values, A1 and A8")


def config_diff():
    # 全キーの差を残し、今回の schedule 以外に変化が無いことを確かめる。
    changes = {}
    for before_path in sorted((OUTPUT / "config-before").glob("*.json")):
        after_path = OUTPUT / "config-after" / before_path.name
        before = json.loads(before_path.read_text(encoding="utf-8-sig"))
        after = json.loads(after_path.read_text(encoding="utf-8-sig"))
        diff = {key: {"before": before.get(key), "after": after.get(key)}
            for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}
        assert all(key.startswith("run.eval_schedule.") for key in diff), (before_path.name, diff)
        changes[before_path.stem] = diff
    assert len(changes) == 7, changes.keys()
    (OUTPUT / "config-diff.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("PASS: full-key comparison of 7 environments; see .scratch/prd918/config-diff.json")


def short_run(configuration):
    # 設定は共通 main と LunarLander の既定を重ね、予算と記録先だけを明示する。
    destination = OUTPUT if configuration == "Debug" else OUTPUT / configuration
    destination.mkdir(parents=True, exist_ok=True)
    config = destination / "lunarlander.txt"
    assert not config.exists(), "Use a fresh output directory for a new acceptance Run"
    config.write_text(
        f"$include <{(REPO / 'apps/runner/config/_main.txt').as_posix()}>\n"
        f"$include <{(REPO / 'apps/runner/config/LunarLander.txt').as_posix()}>\n",
        encoding="utf-8", newline="\n")
    binary = REPO / "apps/runner/bin" / configuration / "AnetRLRunner.exe"
    args = [str(binary), "--config", str(config), "app.$=app.batchrun>P1",
        f"app.runs_dir={(destination / 'runs').as_posix()}", "app.run_name=run_{t}_prd918_ll",
        "app.batchrun.exp_exit_step=300000", "app.save_agent_on_close=false", "app.show_error_dialog=false"]
    # 既存の経験数指標だけを毎イベント記録し、最終遷移まで予算を消費したか確かめる。
    args.append("metrics.scalar.[90_perf/82_exp_step]=exp_step @train $runner interval:1")
    startup = None
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
    with (destination / "short-run-console.log").open("w", encoding="utf-8") as output:
        process = subprocess.run(args, cwd=REPO, stdout=output, stderr=subprocess.STDOUT,
            timeout=3600, startupinfo=startup)
    assert process.returncode == 0, process.returncode
    runs = list((destination / "runs").glob("run_*_prd918_ll"))
    assert len(runs) == 1, runs
    run = runs[0]
    logs = list(run.glob("*.log"))
    source = "\n".join(path.read_text(encoding="utf-8-sig", errors="replace") for path in logs)
    session_steps = {}
    for tag, interval, offset in (("eval_target", 200, 0), ("eval", 200, 0), ("greedy", 1000, 50)):
        assert f"eval.[{tag}]: scheduled (interval={interval}, offset={offset}, warmup_count=1," in source, tag
        pattern = rf"eval\.\[{tag}\]: session end learn_step=(\d+)"
        ended = [int(value) for value in re.findall(pattern, source)]
        assert ended and min(ended) >= interval + offset, (tag, ended)
        session_steps[tag] = [int(value) for value in re.findall(rf"eval\.\[{tag}\]: session start learn_step=(\d+)", source)]
    assert not set(session_steps["greedy"]) & (set(session_steps["eval"]) | set(session_steps["eval_target"])), session_steps
    # TrainEvent は遷移の完了後、カウンタ加算前に通知されるので最後の batch 分を足す。
    resolved = (run / "config/config_data.txt").read_text(encoding="utf-8-sig")
    num_envs = int(re.search(r"^run\.train\.num_envs\s*=\s*([\d,]+)", resolved, re.M)[1].replace(",", ""))
    metrics = run / "metrics.jsonl"
    opener = open
    if not metrics.exists():
        metrics = metrics.with_suffix(".jsonl.gz")
        opener = gzip.open
    with opener(metrics, "rt", encoding="utf-8-sig") as stream:
        recorded = [record["value"] for line in stream
            if (record := json.loads(line)).get("type") == "scalar"
            and record.get("tag") == "90_perf/82_exp_step" and record.get("value") is not None]
    assert recorded, "Missing experience-counter records"
    last_event_exp_step = int(recorded[-1])
    final_exp_step = last_event_exp_step + num_envs
    assert last_event_exp_step < 300000 <= final_exp_step, (last_event_exp_step, final_exp_step)
    result = {"run": str(run), "exit_code": process.returncode, "session_steps": session_steps,
        "last_event_exp_step": last_event_exp_step, "final_exp_step": final_exp_step,
        "arguments": args}
    (destination / "short-run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("static", "config-diff", "short-run"))
    parser.add_argument("--configuration", choices=("Debug", "Release", "RelWithDebInfo"), default="Debug")
    options = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if options.mode == "short-run":
        short_run(options.configuration)
    else:
        {"static": static_checks, "config-diff": config_diff}[options.mode]()
