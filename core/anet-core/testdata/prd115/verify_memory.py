"""PRD115: 既存Debug構成で実物の画像生成をASan/CRT確保フックで検証する。"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
PATHS = ("empty", "time", "histogram", "sweep")
BASELINE_BLOB = "1fff8bfb5629b1e23b4d732f3ddb3a7991286707"


def windows_args(command):
    """生成済みMSVCコマンドの引用符をWindows自身の規則で読む。"""
    shell = ctypes.windll.shell32
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    count = ctypes.c_int()
    result = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not result:
        raise ctypes.WinError()
    try:
        return [result[i] for i in range(count.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(result, ctypes.c_void_p))


def run(command, folder, name, *, environment=None, timeout=300):
    try:
        result = subprocess.run(command, cwd=ROOT / "out/build/x64-Debug",
                                env=environment, text=True, encoding="utf-8",
                                errors="replace", capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        result = subprocess.CompletedProcess(command, 124, output, f"Timed out after {timeout}s\n")
    (folder / (name + ".log")).write_text(
        subprocess.list2cmdline(command) + "\n" + result.stdout + result.stderr,
        encoding="utf-8", newline="\n")
    print(f"{name}: exit={result.returncode}", flush=True)
    return result


def build(source, folder, asan):
    build_dir = ROOT / "out/build/x64-Debug"
    entries = json.loads((build_dir / "compile_commands.json").read_text(encoding="utf-8"))
    entry = next(e for e in entries if e["file"].replace("\\", "/").endswith("/heat_map.cpp"))
    args = windows_args(entry["command"])
    compiler = args[0]
    flags = [arg for arg in args[1:-1]
             if not arg.startswith(("/Fo", "/Fd", "/Fp", "/Yu", "/FI"))
             and arg not in ("-c", "/c", "/RTC1", "/MP")]
    if asan:
        # 既存の未instrumentライブラリとSTLのコンテナ注釈だけをそろえる。ヒープ検査は有効。
        flags += ["/fsanitize=address", "/D_DISABLE_STRING_ANNOTATION", "/D_DISABLE_VECTOR_ANNOTATION"]
    objects = []
    headers = sorted((ROOT / "core/anet-core/include").rglob("*.hpp"))
    headers += sorted((ROOT / "core/anet-core/src").glob("*.hpp"))
    header_hash = hashlib.sha256(b"".join(path.read_bytes() for path in headers)).digest()
    for name, path in (("heat_map", source), ("memory_probe", Path(__file__).with_name("memory_probe.cpp"))):
        obj = folder / (name + ".obj")
        signature = hashlib.sha256(path.read_bytes() + header_hash
                                   + subprocess.list2cmdline([compiler, *flags]).encode()).hexdigest()
        signature_path = folder / (name + ".sha256")
        if obj.is_file() and signature_path.is_file() and signature_path.read_text() == signature:
            objects.append(str(obj))
            continue
        result = run([compiler, *flags, "/Fo" + str(obj),
                      "/Fd" + str(folder / (name + ".pdb")), "/c", str(path)],
                     folder, "compile-" + name)
        if result.returncode:
            raise RuntimeError(f"Compilation failed: {folder / ('compile-' + name + '.log')}")
        signature_path.write_text(signature, encoding="ascii")
        objects.append(str(obj))

    ninja = (build_dir / "build.ninja").read_text(encoding="utf-8")
    block = re.search(r"(?m)^build [^\n]*anet-core-test\.exe: [^\n]*\n((?:[ \t].*\n)*)", ninja)
    if not block:
        raise RuntimeError("Cannot find the anet-core-test link block")
    libraries = re.search(r"(?m)^  LINK_LIBRARIES = (.*)$", block[1])
    if not libraries:
        raise RuntimeError("Cannot find the existing link libraries")
    libs = windows_args("link " + libraries[1])[1:]
    exe = folder / "image_memory_probe.exe"
    result = run([compiler, "/nologo", *objects, "/Fe" + str(exe),
                  *(["/fsanitize=address"] if asan else []),
                  "/link", "/INCREMENTAL:NO", "/DEBUG", *libs], folder, "link")
    if result.returncode:
        raise RuntimeError(f"Link failed: {folder / 'link.log'}")
    return exe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "current"), required=True)
    parser.add_argument("--asan", action="store_true")
    parser.add_argument("--case", choices=PATHS, action="append")
    parser.add_argument("--dump-only", action="store_true")
    options = parser.parse_args()
    folder = ROOT / ".scratch/prd115" / ("memory-" + options.phase + ("-asan" if options.asan else "-crt"))
    folder.mkdir(parents=True, exist_ok=True)
    source = (ROOT / ".scratch/prd115/baseline_heat_map.cpp" if options.phase == "baseline"
              else ROOT / "core/anet-core/src/heat_map.cpp")
    if options.phase == "baseline" and not source.is_file():
        source.write_bytes(subprocess.check_output(["git", "show", BASELINE_BLOB], cwd=ROOT))
    if not source.is_file():
        raise RuntimeError(f"Source does not exist: {source}")
    if options.phase == "baseline":
        blob = subprocess.check_output(["git", "hash-object", str(source)], cwd=ROOT, text=True).strip()
        if blob != BASELINE_BLOB:
            raise RuntimeError(f"Baseline source mismatch: expected={BASELINE_BLOB} actual={blob}")
    exe = build(source, folder, options.asan)
    environment = os.environ.copy()
    environment["PATH"] = str(ROOT / "core/anet-core/bin/Debug") + os.pathsep + environment["PATH"]
    if options.asan:
        environment["ASAN_OPTIONS"] = "alloc_dealloc_mismatch=1:abort_on_error=0"
    results = []
    for case in options.case or PATHS:
        modes = ("dump",) if options.dump_only else (("render",) if options.asan else (("dump",) if options.phase == "baseline" else ("dump", "fail")))
        for mode in modes:
            result = run([str(exe), case, mode], folder, case + "-" + mode,
                         environment=environment, timeout=120)
            output = result.stdout + result.stderr
            if options.asan and options.phase == "baseline" and not options.dump_only:
                passed = result.returncode != 0 and "alloc-dealloc-mismatch" in output
            elif mode == "fail":
                passed = result.returncode == 0 and "INJECTED_PIXEL_ALLOCATION_FAILURE" in output
            else:
                passed = result.returncode == 0
            results.append({"case": case, "mode": mode, "exit": result.returncode, "passed": passed})
    report = {"phase": options.phase, "asan": options.asan, "dump_only": options.dump_only,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "results": results}
    (folder / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
