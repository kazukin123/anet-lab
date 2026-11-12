"""PRD125: 共通の target の式を一時的に壊し、復元まで検査する。"""
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prd125_validation", HERE / "run_validation.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
REPO = validation.REPO
SOURCE = REPO / "core/anet-core/src/dqn_based_agent.cpp"
MUTATIONS = {
    "omit_h": (
        "const auto target = config_.use_tbo ? TransformH(raw_target) : raw_target;",
        "const auto target = raw_target;"),
    "omit_mask": (
        "const auto raw_target = returns + gamma_n * mask * bootstrap_real;",
        "const auto raw_target = returns + gamma_n * bootstrap_real;"),
    "mask_bonus": (
        "returns = returns + bonus;",
        "returns = returns + bonus * mask;"),
}


def replace_once(before, after):
    # 対象の一行だけを往復し、ほかの作業の編集を巻き戻さない。
    text = SOURCE.read_text(encoding="utf-8")
    if text.count(before) != 1:
        raise RuntimeError("Mutation source must occur once: " + before)
    SOURCE.write_text(text.replace(before, after), encoding="utf-8", newline="\n")


def build_and_test(folder):
    # cmd の引用符を subprocess の引数変換へ通さず、MSVC の標準入口を呼ぶ。
    folder.mkdir(parents=True, exist_ok=False)
    build_file = folder / "build.cmd"
    build_file.write_bytes((
        '@echo off\r\ntitle %~nx0\r\n'
        'call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\VsDevCmd.bat" '
        '-arch=x64 -host_arch=x64\r\n'
        'if errorlevel 1 exit /b %errorlevel%\r\n'
        'cmake --build --preset x64-Debug --target anet-core-test\r\n').encode("cp932"))
    build = validation.COMMON.execute([
        os.environ["COMSPEC"], "/d", "/c", str(build_file)],
        folder / "build", cwd=REPO)
    if build["exit_code"]:
        raise RuntimeError("Mutation build failed: " + str(folder))
    xml = folder / "core.xml"
    test = validation.COMMON.execute([
        str(REPO / "core/anet-core/bin/Debug/anet-core-test.exe"), "[prd125]",
        "--reporter", "junit", "--out", str(xml)], folder / "test", cwd=REPO)
    return {"build": build, "test": test, "failures": validation.COMMON.failures(xml)}


def main():
    # label を必須にし、過去の破壊検査とログを上書きしない。
    if len(sys.argv) != 2 or Path(sys.argv[1]).name != sys.argv[1]:
        raise SystemExit("Usage: run_mutations.py LABEL")
    folder = validation.ROOT / sys.argv[1]
    folder.mkdir(parents=True, exist_ok=False)
    report = {}
    for name, (before, after) in MUTATIONS.items():
        print("MUTATION", name, flush=True)
        replace_once(before, after)
        try:
            broken = build_and_test(folder / name / "broken")
        finally:
            replace_once(after, before)
            restored = build_and_test(folder / name / "restored")
        passed = bool(broken["test"]["exit_code"] and broken["failures"])
        passed = passed and restored["test"]["exit_code"] == 0 and not restored["failures"]
        report[name] = {"broken": broken, "restored": restored, "passed": passed}
        (folder / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8", newline="\n")
        print("RESULT", name, passed, flush=True)
        if not passed:
            raise RuntimeError("Mutation was not detected or restoration failed: " + name)


if __name__ == "__main__":
    main()
