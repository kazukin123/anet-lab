"""Verify all review regression and integration runs against saved baselines."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

def validate_execution_record(case, record):
    """Validate saved run completion, seed and the case-specific experience budget."""
    assert record["exit_code"] == 0 and record["seed"] == 1, (case, record)
    # OFF回帰の固定基準は維持し、結合Runだけ成立値不足時の延長を許す。
    if case in ("nz_learning", "nz_reference"):
        assert record["experience_budget"] >= 500000, (case, record)
    else:
        assert record["experience_budget"] == (256 if case == "old_checkpoint" else 500000), (case, record)


def main():
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[4]
    helpers = Path(__file__).resolve().parent
    cases = ["atari_default", "atari_efficiency", "lunarlander_cpu", "nz_learning", "nz_reference", "old_checkpoint"]
    records = {}
    runs = {}
    for case in cases:
        record = json.loads((root / (case + ".execution.json")).read_text(encoding="utf-8-sig"))
        validate_execution_record(case, record)
        matches = [Path(record["run"])] if "run" in record else [
            p for p in (root / "runs").iterdir() if p.name.endswith("prd084_review_" + case) and (p / "agent_close.anet").is_file()]
        assert len(matches) == 1, (case, matches)
        run = matches[0]
        assert (run / "agent_close.anet").is_file(), (case, "Missing close checkpoint")
        records[case] = record
        runs[case] = run
    assert len({record["sha256"] for record in records.values()}) == 1
    assert hashlib.sha256((root / "AnetRLRunner.exe").read_bytes()).hexdigest().upper() == records[cases[0]]["sha256"]
    assert len(list(root.glob("*.dll"))) == 50
    results = {}
    def invoke(tool, *arguments):
        process = subprocess.run([sys.executable, str(helpers / tool), *map(str, arguments)],
            capture_output=True, text=True, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(tool + ": " + process.stdout + process.stderr)
        return json.loads(process.stdout)
    for case in cases[:3]:
        old_normalized = repo / ".scratch/prd084/old" / (case + ".metrics.jsonl")
        new_normalized = root / (case + ".metrics.jsonl")
        normalization = invoke("compare_metrics.py", "normalize", runs[case], new_normalized)
        metrics = invoke("compare_metrics.py", "compare", old_normalized, new_normalized)
        old_summary = json.loads(old_normalized.with_suffix(".summary.json").read_text(encoding="utf-8-sig"))
        old_run = Path(old_summary["run_dir"])
        settings = invoke("compare_resolved_settings.py", old_run / "config/config_data.txt", runs[case] / "config/config_data.txt")
        results[case] = {"run": str(runs[case]), "normalization": normalization, "metrics": metrics, "settings": settings}
    for case in ["nz_learning", "nz_reference"]:
        results[case] = invoke("verify_integration.py", case, runs[case])
    checkpoint_arguments = records["old_checkpoint"]["arguments"]
    checkpoint = next(Path(argument.split("=", 1)[1]) for argument in checkpoint_arguments
        if argument.startswith("DefaultDQNAgent.auto_load_file="))
    expected_hash = "43674EF8A169FA7C16A388FC4F2532C9D4F4537F3C5B1AF397606994B22D35FE"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper() == expected_hash
    load_log = (runs["old_checkpoint"] / (runs["old_checkpoint"].name + ".log")).read_text(encoding="utf-8-sig")
    assert "LoadNetwork: model_size=1276270" in load_log
    assert "LoadNetwork: learner_size=1192239" in load_log
    results["old_checkpoint"] = {"run": str(runs["old_checkpoint"]), "source": str(checkpoint),
        "sha256": expected_hash, "exit_code": records["old_checkpoint"]["exit_code"]}
    report = {"executable_sha256": records[cases[0]]["sha256"], "results": results,
        "acceptance": "OFF numeric points and existing string values match; integrations and old checkpoint pass"}
    (root / "acceptance-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "acceptance": report["acceptance"]}))


if __name__ == "__main__":
    main()
