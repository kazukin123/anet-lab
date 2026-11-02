"""PRD104の変更前の実行体と入力を、既存の基準を上書きせず保存する。"""
import json
import os
from pathlib import Path
import shutil
import subprocess

from common import digest

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / ".scratch/prd104"


def main():
    # 実行体と設定は一組で固定し、実行中の別名runnerには触れない。
    OUT.mkdir(parents=True, exist_ok=True)
    destinations = [OUT / name for name in ("baseline-bin", "baseline-config", "resolved-before")]
    if any(directory.exists() for directory in destinations):
        raise RuntimeError("PRD104 baseline already exists; refusing to overwrite")
    binary, config, resolved = destinations
    binary.mkdir()
    for source in (REPO / "apps/runner/bin/Debug").iterdir():
        if source.is_file() and source.suffix.lower() in (".exe", ".dll"):
            shutil.copy2(source, binary / source.name)
    shutil.copy2(REPO / "core/anet-core/bin/Debug/anet-core-test.exe", binary / "anet-core-test.exe")
    shutil.copytree(REPO / "apps/runner/config", config)

    # 公開ConfigManagerの既存snapshotテストで解決値も採取する。
    environment = dict(os.environ, ANET_PRD093_OUTPUT=str(resolved), ANET_PRD093_CONFIG_ROOT=str(config))
    with (OUT / "baseline-config.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(binary / "anet-core-test.exe"), "[prd093-config-snapshot]"], cwd=REPO,
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    manifest = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "files": {str(p.relative_to(OUT)).replace("\\", "/"): digest(p)
                          for directory in destinations for p in sorted(directory.rglob("*")) if p.is_file()}}
    (OUT / "baseline-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PRD104 baseline saved: {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
