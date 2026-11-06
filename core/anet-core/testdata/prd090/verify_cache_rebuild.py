"""PRD090の隔離ビルド・再構築比較・優先反映待ちを再実行する。"""
import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
FIXTURES = Path(__file__).resolve().parent
DEFAULT_MASTER = ROOT / "apps/runner/workspaces/DropMerge-02/runs/run_20261006-033243_dm_fb_munch_tau3e-2_ndtpen20_nstep3_100m/metrics.jsonl.gz"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("validate", "new", "baseline", "priority", "report"))
parser.add_argument("--area", type=Path, default=ROOT / "apps/metrics-viewer/target/prd090-performance")
parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
parser.add_argument("--baseline", type=Path, default=ROOT / "apps/metrics-viewer/target/prd090-baseline")
parser.add_argument("--maven", default=shutil.which("mvn.cmd") or "C:/dev/apache-maven-3.9.15/bin/mvn.cmd")
parser.add_argument("--java", default="C:/Program Files/Eclipse Adoptium/jdk-25.0.3.9-hotspot/bin/java.exe")
args = parser.parse_args()
area = args.area.resolve()
area.mkdir(parents=True, exist_ok=True)

def remove_trial(path):
    resolved = path.resolve()
    if resolved.parent != area or resolved.name not in ("trial", "priority-trial"):
        raise RuntimeError(f"Refusing removal outside isolated trials: {resolved}")
    shutil.rmtree(resolved)

def prepare_input():
    inputs = area / "input"
    inputs.mkdir(exist_ok=True)
    copied = inputs / "metrics.jsonl.gz"
    if not copied.exists(): shutil.copy2(args.master.resolve(), copied)
    raw = inputs / "metrics.jsonl"
    if not raw.exists():
        with gzip.open(copied, "rb") as source, raw.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
    return copied, raw

def classpath(project):
    build = project / "target"
    return str(build / "benchmark") + ";" + str(build / "classes") + ";" + (build / "classpath.txt").read_text().strip()

def build_new():
    project = area / "new-project"
    if project.exists():
        raise RuntimeError(f"New build snapshot already exists; preserve or move it first: {project}")
    project.mkdir()
    shutil.copytree(ROOT / "apps/metrics-viewer/src", project / "src")
    shutil.copy2(ROOT / "apps/metrics-viewer/pom.xml", project / "pom.xml")
    # copied POMの../../ legal入力も検証領域内へ置く。
    for filename in ("LICENSE", "NOTICE"):
        shutil.copy2(ROOT / filename, project.parent.parent / filename)
    sources = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((project / "src").rglob("*")) if path.is_file()}
    (area / "new-source-sha256.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    with (area / "new-build.log").open("w", encoding="utf-8") as output:
        subprocess.run([args.maven, "-f", str(project / "pom.xml"), "-DskipTests", "package",
                        "dependency:build-classpath", "-Dmdep.outputFile=target/classpath.txt"],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    compile_benchmarks(project)
    # 再実行でも、測定後のコード同一性を集計時に検証できるようfreeze情報を残す。
    (area / "ingestion-measured-source-sha256.json").write_text(json.dumps(sources, indent=2), encoding="utf-8", newline="\n")
    classes = {str(path.relative_to(project / "target/classes")): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (project / "target/classes").rglob("*.class") if not path.name.startswith("IngestScheduler")}
    (area / "ingestion-measured-class-sha256.json").write_text(json.dumps(classes, indent=2), encoding="utf-8", newline="\n")
    return project

def compile_benchmarks(project, include_priority=True):
    build = project / "target"
    (build / "benchmark").mkdir(exist_ok=True)
    compiler = str(Path(args.java).with_name("javac.exe"))
    fixtures = [str(FIXTURES / "CacheRebuildBenchmark.java")]
    if include_priority:
        fixtures.append(str(FIXTURES / "PriorityLatencyBenchmark.java"))
    subprocess.run([compiler, "--release", "17", "-cp", classpath(project), "-d", str(build / "benchmark"),
                    *fixtures], check=True)

def run_java(project, classname, parameters, log):
    with log.open("w", encoding="utf-8") as output:
        subprocess.run([args.java, "-Xms512m", "-Xmx2g", "-cp", classpath(project),
                        "io.github.kazukin123.anetlab.metricsviewer.service." + classname, *map(str, parameters)],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith(("BENCH ", "DATA ", "PRIORITY ", "LATENCY ")): print(line, flush=True)

def report():
    groups = {}
    reference = None
    trials = {}
    for variant in ("baseline", "new"):
        for kind in ("raw", "gzip"):
            measurements = []
            for index in range(1, 4):
                name = f"{variant}-{kind}-{index}"
                lines = (area / (name + ".log")).read_text(encoding="utf-8").splitlines()
                bench = next(line for line in lines if line.startswith("BENCH "))
                values = dict(part.split("=", 1) for part in bench.split()[1:])
                data = [line for line in lines if line.startswith("DATA ")]
                if len(data) != 5: raise RuntimeError(f"Incomplete trial: {name}")
                if reference is None: reference = data
                if data != reference: raise RuntimeError(f"Persistent data differs: {name}")
                measurements.append(float(values["timeSeconds"]))
                if variant == "new":
                    completed = next(line for line in reversed(lines) if "Metrics cache completed:" in line)
                    fields = dict(part.split("=", 1) for part in completed.split() if "=" in part)
                    values["committedProgress"] = {key: fields[key] for key in (
                        "lines", "validPoints", "scope", "elapsedMs", "processingMs", "prepareMs",
                        "readParseWriteMs", "flushStatsMs", "metadataMs", "commitMs", "resourcesMs",
                        "readerMs", "parseMs", "l0Ms", "lodMs", "jsonMs", "linesPerSecond", "pointsPerSecond")}
                trials[name] = values
            groups[f"{variant}-{kind}"] = {"seconds": measurements, "medianSeconds": statistics.median(measurements),
                "minSeconds": min(measurements), "maxSeconds": max(measurements),
                "populationStddevSeconds": statistics.pstdev(measurements)}
    improvements = {}
    for kind in ("raw", "gzip"):
        before = groups[f"baseline-{kind}"]["medianSeconds"]
        after = groups[f"new-{kind}"]["medianSeconds"]
        improvements[kind] = 100.0 * (before - after) / before
        if after >= before: raise RuntimeError(f"Rebuild median did not improve: {kind}")
    priority_lines = (area / "priority.log").read_text(encoding="utf-8").splitlines()
    priority = next(line for line in priority_lines if line.startswith("PRIORITY "))
    priority_fields = dict(part.split("=",1) for part in priority.split()[1:])
    if float(priority_fields["p95Ms"]) > 10000.0: raise RuntimeError("Priority p95 exceeded 10 seconds")
    result = {"jvm": {"executable": args.java, "options": ["-Xms512m", "-Xmx2g"], "release": 17},
        "sourceBytes": {path.name: path.stat().st_size for path in (area / "input").iterdir()},
        "factor": 4, "groups": groups, "improvementPercent": improvements, "trials": trials,
        "persistentDataEqual": True, "data": reference, "priority": priority,
        "heapPeakDefinition": "sum of each heap pool peak used bytes; upper bound, not simultaneous peak"}
    # 最終結果から入力と凍結ソースを再識別できるようにする。hash時間は計測時間に含めない。
    result["inputSha256"] = {}
    for path in sorted((area / "input").iterdir()):
        with path.open("rb") as stream:
            result["inputSha256"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    measured_classes = json.loads((area / "ingestion-measured-class-sha256.json").read_text(encoding="utf-8"))
    classes_root = area / "new-project/target/classes"
    changed = [name for name, digest in measured_classes.items()
               if hashlib.sha256((classes_root / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f"Rebuild execution classes changed after measurements: {changed}")
    result["rebuildCodeVerification"] = {"nonSchedulerMainClassesEqual": True, "verifiedClasses": len(measured_classes)}
    result["rebuildMeasuredManifestSha256"] = hashlib.sha256((area / "ingestion-measured-source-sha256.json").read_bytes()).hexdigest()
    result["newSourceManifestSha256"] = hashlib.sha256((area / "new-source-sha256.json").read_bytes()).hexdigest()
    result["runtimeVersion"] = subprocess.run([args.java, "-version"], capture_output=True, text=True, check=True).stderr.strip()
    result["sqliteWriteCacheTargetBytes"] = 16 * 1024 * 1024
    result["sqliteNativeMemoryExcludedFromHeap"] = True
    result["priorityLatencySamplesMilliseconds"] = [
        float(dict(part.split("=", 1) for part in line.split()[1:])["milliseconds"])
        for line in priority_lines if line.startswith("LATENCY ")]
    validation = (area / "all-tests-package-final.log").read_text(encoding="utf-8")
    summary = re.findall(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", validation)[-1]
    if "BUILD SUCCESS" not in validation or any(int(value) for value in summary[1:]):
        raise RuntimeError("Final validation did not pass completely")
    result["validation"] = dict(zip(("tests", "failures", "errors", "skipped"), map(int, summary)))
    result["validation"]["finishedAt"] = re.findall(r"Finished at: ([^\r\n]+)", validation)[-1]
    result["validation"]["totalTime"] = re.findall(r"Total time: ([^\r\n]+)", validation)[-1].strip()
    (area / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if args.mode == "validate":
    module = ROOT / "apps/metrics-viewer"
    temporary_pom = module / "pom-prd090-check.xml"
    if temporary_pom.exists(): raise RuntimeError(f"Refusing to overwrite POM: {temporary_pom}")
    text = (module / "pom.xml").read_text(encoding="utf-8")
    text = text.replace("<build>", "<build>\n        <directory>target/prd090-validation</directory>", 1)
    text = text.replace("<finalName>metrics-viewer</finalName>", "<finalName>metrics-viewer-prd090</finalName>")
    temporary_pom.write_text(text, encoding="utf-8", newline="\n")
    try:
        with (area / "all-tests-package-final.log").open("w", encoding="utf-8") as output:
            subprocess.run([args.maven, "-f", str(temporary_pom), "package"],
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        print("Validation succeeded: " + str(area / "all-tests-package-final.log"))
    finally:
        temporary_pom.unlink()
elif args.mode == "report":
    report()
else:
    copied, raw = prepare_input()
    trial = area / ("priority-trial" if args.mode == "priority" else "trial")
    if trial.exists(): raise RuntimeError(f"A trial exists; do not overlap measurements: {trial}")
    if args.mode == "priority":
        project = area / "new-project"
        compile_benchmarks(project)
        trial.mkdir()
        run_java(project, "PriorityLatencyBenchmark", [trial, copied], area / "priority.log")
        remove_trial(trial)
    else:
        project = args.baseline.resolve() if args.mode == "baseline" else build_new()
        if args.mode == "baseline":
            compile_benchmarks(project, include_priority=False)
        for kind, source in (("raw", raw), ("gzip", copied)):
            for index in range(1, 4):
                trial.mkdir()
                os.link(source, trial / source.name)
                run_java(project, "CacheRebuildBenchmark", [trial], area / f"{args.mode}-{kind}-{index}.log")
                remove_trial(trial)
