"""Exercise timing-tool behavior with synthetic artifacts; never launch Runner."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

repo = Path(__file__).resolve().parents[4]
script = repo / "core/anet-core/testdata/prd116/run_performance.py"
spec = importlib.util.spec_from_file_location("grid_timing", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
passed = []


def check_failure(label, action, terms):
    try:
        action()
    except (RuntimeError, FileExistsError) as error:
        assert all(term in str(error) for term in terms), (label, str(error), terms)
    else:
        raise AssertionError(label + ": expected failure")
    passed.append(label)


def cli(out, phase="summarize"):
    return subprocess.run([str(repo / ".venv/Scripts/python.exe"), str(script), phase, "--out", str(out)],
                          capture_output=True, text=True, encoding="utf-8")


with tempfile.TemporaryDirectory(prefix="prd116-perf-check-") as temporary:
    root = Path(temporary)
    # CLIの正常集計と、入力契約・記録保護の失敗を合成artifactで確かめる。
    valid = root / "summary"
    valid.mkdir()
    records = [{"name": str(index), "arm": arm, "profiled": False, "window_experiences": 256,
                "binary_sha256": "fixed", "measurement_seconds": 2 if arm == "off" else 3}
               for index, arm in enumerate(module.ORDER)]
    path = valid / "measurements.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    result = cli(valid)
    assert result.returncode == 0, result.stderr
    summary = json.loads((valid / "summary.json").read_text(encoding="utf-8"))
    assert summary["on_off_ratio"] == 1.5 and summary["medians_seconds"] == {"off": 2, "on": 3}
    passed.append("CLI normal summary")
    before = path.read_bytes(), (valid / "summary.json").read_bytes()
    result = cli(valid, "measure")
    assert result.returncode != 0 and str(valid) in result.stderr and "fresh" in result.stderr, result.stderr
    assert before == (path.read_bytes(), (valid / "summary.json").read_bytes())
    passed.append("CLI recorded measurements retained")
    for label, changed, terms in [
        ("CLI count", records[:1], ["actual=1", "expected=12"]),
        ("CLI budget", [dict(item, window_experiences=512) if index == 0 else item
                         for index, item in enumerate(records)], ["256", "512", "exactly one"]),
        ("CLI binary", [dict(item, binary_sha256="other") if index == 0 else item
                         for index, item in enumerate(records)], ["fixed", "other", "exactly one"])]:
        path.write_text(json.dumps(changed), encoding="utf-8")
        result = cli(valid)
        assert result.returncode != 0 and all(term in result.stderr for term in terms), (label, result.stderr)
        passed.append(label)

    def run_case(label, terms, *, bad_binary=False, exit_code=0, boundary=True,
                 seconds=1, learn=1, tools=None):
        out = root / label
        (out / "bin").mkdir(parents=True)
        binary = out / "bin/AnetRLRunner.exe"
        binary.write_bytes(b"synthetic-not-executable")
        expected = "wrong" if bad_binary else module.digest(binary)
        module.save(out / "plan.json", {"binary_sha256": expected})
        folder = out / "runs" / ("run_synthetic_prd116_gp_" + label)
        folder.mkdir(parents=True)
        rows = [{"tag": module.CLOCK, "step": module.WARM, "value": 0}]
        if boundary:
            rows.append({"tag": module.CLOCK, "step": module.WARM + 256, "value": seconds / 3600})
        rows.append({"tag": "56_agent_in/32_learn_count", "value": learn})
        (folder / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        class FinishedProcess:
            returncode = exit_code
            pid = 1

            def poll(self):
                return self.returncode

        with patch.object(module, "idle"), patch.object(module.subprocess, "Popen", return_value=FinishedProcess()):
            check_failure(label, lambda: module.execute(out, label, "on", 256, tools), terms)

    # 実RunnerとCIM問い合わせを差し替え、失敗診断の識別子・実値・期待値だけを検証する。
    run_case("binary", ["binary", "actual=", "expected=wrong"], bad_binary=True)
    run_case("runner", ["runner", "exit_code=7", "expected_exit_code=0", "run_count=1"], exit_code=7)
    run_case("boundary", ["boundary", "16384", "16640", "missing="], boundary=False)
    run_case("clock", ["clock", "seconds=0.0", "finite and > 0"], seconds=0)
    run_case("learning", ["learning", "actual=0", "expected > 0"], learn=0)
    run_case("capture", ["capture", "actual=not started", "16384"], tools=root / "tools")
    profile = root / "empty-profile"
    profile.mkdir()
    (profile / "facade_events.csv").write_text(
        "name,ns_since_start,exec_time_ns,thread\n", encoding="utf-8")
    check_failure("profile roots", lambda: module.profile_summary(profile),
                  [str(profile), "actual=0", "expected > 0"])

print("PASS", len(passed), "behaviors:", ", ".join(passed))
