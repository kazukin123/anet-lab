"""PRD106 着手前の実行体・設定を hash とともに保存する。上書きしない。"""
import os
import shutil
import subprocess

from common import OUT, REPO, digest, save


def main():
    # 実験で使う Release も保存し、Debug の前後比較とは別の組として残す。
    OUT.mkdir(parents=True, exist_ok=True)
    names = ("baseline-bin", "baseline-release-bin", "baseline-config", "resolved-before")
    if any((OUT / name).exists() for name in names):
        raise RuntimeError("PRD106 baseline already exists; refusing to overwrite")
    for build, name in (("Debug", "baseline-bin"), ("Release", "baseline-release-bin")):
        destination = OUT / name
        destination.mkdir()
        for source in (REPO / "apps/runner/bin" / build).iterdir():
            if source.is_file() and source.suffix.lower() in (".exe", ".dll"):
                shutil.copy2(source, destination / source.name)
    shutil.copy2(REPO / "core/anet-core/bin/Debug/anet-core-test.exe", OUT / "baseline-bin/anet-core-test.exe")
    shutil.copytree(REPO / "apps/runner/config", OUT / "baseline-config")
    environment = dict(os.environ, ANET_PRD093_OUTPUT=str(OUT / "resolved-before"),
                       ANET_PRD093_CONFIG_ROOT=str(OUT / "baseline-config"))
    with (OUT / "baseline-config.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(OUT / "baseline-bin/anet-core-test.exe"), "[prd093-config-snapshot]"],
                       cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    manifest = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "files": {path.relative_to(OUT).as_posix(): digest(path)
                          for name in names for path in sorted((OUT / name).rglob("*")) if path.is_file()}}
    save(OUT / "baseline-manifest.json", manifest)
    print("PRD106 baseline saved:", len(manifest["files"]), "files", flush=True)


if __name__ == "__main__":
    main()
