"""PRD103 の変更前実行体・設定を一組として保存する。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def main():
    # 既存snapshotを上書きせず、コード・設定変更前の一組を保存する。
    destinations = [ROOT / name for name in ("baseline-bin", "baseline-config", "resolved-before")]
    if any(path.exists() for path in destinations):
        raise RuntimeError("PRD103 baseline already exists; refusing to overwrite")
    binary, config, resolved = destinations
    binary.mkdir()
    for source in (REPO / "apps/runner/bin/Debug").iterdir():
        if source.is_file() and source.suffix.lower() in (".exe", ".dll"):
            shutil.copy2(source, binary / source.name)
    shutil.copytree(REPO / "apps/runner/config", config)

    # 既存の公開ConfigManagerテストで解決値を採取する。学習は起動しない。
    environment = dict(os.environ, ANET_PRD093_OUTPUT=str(resolved), ANET_PRD093_CONFIG_ROOT=str(config))
    with (ROOT / "baseline-config.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([str(REPO / "core/anet-core/bin/Debug/anet-core-test.exe"),
            "[prd093-config-snapshot]"], cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Configuration snapshot failed: exit={result.returncode}; see baseline-config.log")

    # 実行体と入力の同一性を後から検証できる形で固定する。
    manifest = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "files": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            for directory in destinations for path in sorted(directory.rglob("*")) if path.is_file()}}
    (ROOT / "baseline-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PRD103 baseline saved: {len(manifest['files'])} files; commit={manifest['commit']}")


if __name__ == "__main__":
    main()
