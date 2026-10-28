"""Verify final OFF and representative ON series and summarize all six measurements."""
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
FOLLOWUP = "--followup" in sys.argv[2:]
# 保存した実行体を基準にし、旧基準の取り違えや測定後の置換を検出する。
EXE_HASHES = {side: hashlib.sha256((ROOT.parent / side / "bin/AnetRLRunner.exe").read_bytes()).hexdigest()
              for side in ("before", "after")}
EXPECTED_OFF = {
    "atari_default": (78986, "9adcb0b9e1e5d6cb95b1666ca6716284f30a684b1ba294066b71c6008e93fb29"),
    "atari_efficiency": (122332, "5696fb4de42f8e308ec0740887697971d98f356dfd4f3c9147b42879132be57e"),
    "lunarlander_cpu": (88184, "b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5"),
}


def config_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            values[key] = value
    return values


def load_record(name):
    record = json.loads((ROOT / (name + ".execution.json")).read_text(encoding="utf-8-sig"))
    assert record["exit_code"] == 0, name
    assert record["seed"] == 1, name
    side = "before" if name.startswith("on_before_") else "after"
    assert record["sha256"].lower() == EXE_HASHES[side], name
    assert record["experience_budget"] == (256 if name == "old_checkpoint" else 500000), name
    series = ROOT / (name + ".metrics.jsonl")
    record["checksum"] = hashlib.sha256(series.read_bytes()).hexdigest()
    record["point_count"] = len(series.read_text(encoding="utf-8").splitlines())
    run = Path(record["run"])
    record["resolved_config"] = config_values(run / "config/config_data.txt")
    scalar = defaultdict(list)
    with (run / "metrics.jsonl").open(encoding="utf-8-sig") as source:
        for line in source:
            item = json.loads(line)
            if item.get("type") == "scalar":
                scalar[item["tag"]].append(item)
    return record, scalar


off = {}
for name, (count, checksum) in EXPECTED_OFF.items():
    record, _ = load_record(name)
    assert (record["point_count"], record["checksum"]) == (count, checksum), name
    off[name] = record
checkpoint, _ = load_record("old_checkpoint")
checkpoint_log = "".join((ROOT / ("old_checkpoint." + stream + ".log")).read_text(encoding="utf-8-sig") for stream in ("stdout", "stderr"))
checkpoint_run = Path(checkpoint["run"])
checkpoint_log += (checkpoint_run / (checkpoint_run.name + ".log")).read_text(encoding="utf-8-sig")
assert "LoadNetwork: model_size=1276270" in checkpoint_log
assert "LoadNetwork: learner_size=1192239" in checkpoint_log

runs = []
ordered = []
for repeat in range(1, 2 if FOLLOWUP else 4):
    pair = []
    for side in (("after",) if FOLLOWUP else ("before", "after")):
        name = "on_after_review" if FOLLOWUP else f"on_{side}_{repeat}"
        record, scalar = load_record(name)
        record["side"], record["repeat"] = side, repeat
        config = record["resolved_config"]
        assert config["run.train.num_envs"] == "128" and config["run.seed"] == "1"
        assert config["E1.game"] == "breakout"
        assert config["DefaultDQNAgent.learner.use_amp"] == "true"
        assert config["net.block.[AtariHeadFC512Def].force_fp32"] == "true"
        assert config["DefaultDQNAgent.learner.munchausen.log_policy_mode"] == "target"
        for key, hold in (("ref_actor", "episode"), ("ref_learner", "call")):
            prefix = f"DefaultDQNAgent.nn_forward.[{key}].noisy."
            assert config[prefix + "mode"] == "sample"
            assert config[prefix + "hold"] == hold and config[prefix + "share"] == "sample"
        assert config["DefaultDQNAgent.actor.[train].nn_forward_key"] == "ref_actor"
        assert all(config[f"DefaultDQNAgent.learner.{purpose}.nn_forward_key"] == "ref_learner" for purpose in ("current", "target"))
        assert config["run.eval_schedule.[greedy].interval"] == "0"
        noisy = {tag: points for tag, points in scalar.items() if tag.startswith("47_agent_noisy/")}
        assert len(noisy) == 7, (name, list(noisy))
        declarations = [key for key in config if key.startswith("metrics.scalar.[47_agent_noisy/")]
        assert len(declarations) == 11 and sum("_feature_" in key for key in declarations) == 4
        for tag, points in noisy.items():
            assert "_feature_" not in tag
            assert sum(isinstance(point["value"], (float, int)) and math.isfinite(point["value"]) for point in points) >= 2
        assert len(scalar["52_eval2/10_game_score_mean"]) >= 2
        throughput = scalar["90_perf/12_exp_step_per_sec"]
        active = [current["value"] for previous, current in zip(throughput, throughput[1:]) if previous["step"] >= 200000]
        assert active, name
        elapsed = [point for point in scalar["90_perf/90_elapse_hour"] if point["step"] >= 200000]
        assert len(elapsed) >= 2
        record["active_exp_per_second"] = statistics.mean(active)
        record["active_points"] = len(active)
        record["active_seconds"] = (elapsed[-1]["value"] - elapsed[0]["value"]) * 3600
        record["active_step_range"] = [elapsed[0]["step"], elapsed[-1]["step"]]
        assert record["warmup_observed_seconds"] is not None
        record["post_warmup_cpu_mib"] = record["post_warmup_peak_private_bytes"] / 1048576
        runs.append(record)
        ordered.append(record)
        pair.append(record)
    if FOLLOWUP:
        reference_root = Path(__file__).resolve().parents[4] / ".scratch/prd084/forward-refactor/after"
        reference = json.loads((reference_root / "on_before_1.execution.json").read_text(encoding="utf-8-sig"))
        assert pair[0]["point_count"] == 109941
        assert pair[0]["checksum"] == "5c3acbb6eba510e89210025a81249b42cde1019362f42257aa70947be31b7bd6"
        original = config_values(Path(reference["run"]) / "config/config_data.txt")
        current = pair[0]["resolved_config"]
        changes = {key: [original.get(key), current.get(key)] for key in original.keys() | current.keys()
                   if original.get(key) != current.get(key) and key not in {"app.run_name", "app.runs_dir"}}
        assert not changes, changes
        assert pair[0]["gpu_identity"] == reference["gpu_identity"], "GPU differs from representative reference"
    else:
        assert pair[0]["checksum"] == pair[1]["checksum"]
        assert pair[0]["point_count"] == pair[1]["point_count"]
        changes = {key: [pair[0]["resolved_config"].get(key), pair[1]["resolved_config"].get(key)]
                   for key in pair[0]["resolved_config"].keys() | pair[1]["resolved_config"].keys()
                   if pair[0]["resolved_config"].get(key) != pair[1]["resolved_config"].get(key) and key != "app.run_name"}
        assert not changes, changes
assert len({record["checksum"] for record in runs}) == 1, "Seed-identical repetitions differ"
assert len({json.dumps(record["gpu_identity"]) for record in runs}) == 1, "GPU differs"
for previous, current in zip(ordered, ordered[1:]):
    assert previous["end"] <= current["start"], "Measurements overlap"

keys = ("elapsed_seconds", "active_seconds", "active_exp_per_second", "post_warmup_cpu_mib", "post_warmup_device_extra_mib")
medians = {} if FOLLOWUP else {side: {key: statistics.median(record[key] for record in runs if record["side"] == side) for key in keys}
           for side in ("before", "after")}
report = {"off": off, "old_checkpoint": checkpoint, "on": runs, "medians": medians,
          "acceptance": "numeric series identical; no improvement gate",
          "memory_scope": "post-warmup process private bytes and whole-device peak-minus-start sampled every 5 seconds"}
(ROOT / "acceptance-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
lines = ["# Forward 事前準備の検証", "", ("OFF は旧基準と完全一致。最終 ON 単発は保存済み代表Runの109941点と一致。" if FOLLOWUP else "OFF は旧基準と完全一致。代表 ON は Noisy 指標を含む数値系列が全6 Runで一致。"), "",
         "| Run | 全体秒 | 200k後の測定区間秒 | 経験/s | CPU MiB | GPU追加 MiB |",
         "|---|---:|---:|---:|---:|---:|"]
for record in runs:
    lines.append(f"| {record['case']} | {record['elapsed_seconds']:.3f} | {record['active_seconds']:.3f} | {record['active_exp_per_second']:.1f} | {record['post_warmup_cpu_mib']:.2f} | {record['post_warmup_device_extra_mib']:.2f} |")
if not FOLLOWUP:
    lines.extend(["", "## 中央値", "", "| 側 | 全体秒 | 200k後の測定区間秒 | 経験/s | CPU MiB | GPU追加 MiB |", "|---|---:|---:|---:|---:|---:|"])
for side, values in medians.items():
    lines.append(f"| {side} | {values['elapsed_seconds']:.3f} | {values['active_seconds']:.3f} | {values['active_exp_per_second']:.1f} | {values['post_warmup_cpu_mib']:.2f} | {values['post_warmup_device_extra_mib']:.2f} |")
lines.append("\nGPU はデバイス全体の起動前との差。CPU はプロセス private bytes。メモリは5秒間隔で warmup 到達後だけ採取する。区間秒は最初と最後の200k以後のメトリクスの差で、採択した step 範囲を JSON に保存する。")
(ROOT / "acceptance-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"off_cases": len(off), "on_runs": len(runs), "on_checksum": runs[0]["checksum"], "medians": medians}))
