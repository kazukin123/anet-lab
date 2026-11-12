"""子プロセスだけにffmpegの代役を渡し、公開MetricsLoggerの動画経路を検証する。"""

import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    output = root / ".scratch" / "prd117" / "video-fallback"
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("fake_ffmpeg.cpp")
    executable = output / "ffmpeg.exe"
    object_path = output / "fake_ffmpeg.obj"
    vsdev = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat")
    command = (
        f'call "{vsdev}" -arch=x64 -host_arch=x64 && '
        f'cl /nologo /EHsc /std:c++20 /utf-8 /D_CRT_SECURE_NO_WARNINGS '
        f'"{source}" /Fe:"{executable}" /Fo:"{object_path}"'
    )
    compiled = subprocess.run(
        f'cmd /d /s /c "{command}"', cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (output / "compile.log").write_bytes(compiled.stdout)
    if compiled.returncode:
        print(f"Fake ffmpeg build failed; see {output / 'compile.log'}")
        return compiled.returncode
    trace = output / "codecs.txt"
    trace.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["PATH"] = str(output) + os.pathsep + environment.get("PATH", "")
    environment["PRD117_FFMPEG_TRACE"] = str(trace)
    environment["PRD117_VIDEO_TEST_ROOT"] = str(output / "runs")
    tested = subprocess.run(
        [str(root / "core/anet-core/bin/Debug/anet-core-test.exe"), "[.prd117_video_fallback]"],
        cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120,
    )
    (output / "test.log").write_bytes(tested.stdout)
    print(tested.stdout.decode("utf-8", errors="replace"))
    print(f"Codec trace: {trace}")
    return tested.returncode


if __name__ == "__main__":
    sys.exit(main())
