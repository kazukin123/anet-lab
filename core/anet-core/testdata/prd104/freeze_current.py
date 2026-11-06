"""長時間のGPU空き待ちに備え、レビュー済みDebug/Releaseの実行体とDLLを固定する。"""
import shutil

from common import OUT, REPO, digest, save


def main():
    destinations = {build: OUT / ("current-" + build.lower() + "-bin") for build in ("Debug", "Release")}
    if any(path.exists() for path in destinations.values()):
        raise RuntimeError("Current runtime already exists; refusing to overwrite")
    # executableと全runtime DLLを一組にし、以後のrepo側ビルドから切り離す。
    for build, destination in destinations.items():
        destination.mkdir()
        for source in (REPO / "apps/runner/bin" / build).iterdir():
            if source.is_file() and source.suffix.lower() in (".exe", ".dll"):
                shutil.copy2(source, destination / source.name)
        shutil.copy2(REPO / "core/anet-core/bin" / build / "anet-core-test.exe", destination / "anet-core-test.exe")
    manifest = {str(path.relative_to(OUT)).replace("\\", "/"): digest(path)
                for destination in destinations.values() for path in sorted(destination.iterdir())}
    save(OUT / "current-runtime-manifest.json", manifest)
    print("CURRENT RUNTIMES FROZEN", len(manifest), "files", flush=True)


if __name__ == "__main__":
    main()
