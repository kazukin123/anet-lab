"""PRD106 の検証先を分離し、既存の公開 resolver と実行記録を使う。"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
OUT = REPO / ".scratch/prd106"


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT.parent / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = load("prd104/common.py", "prd106_previous_common")
previous.OUT = OUT
save = previous.save
digest = previous.digest
resolve = previous.resolve
execute = previous.execute
rows = previous.rows


def properties(path):
    # 実効 dump はデータとして読み、プロファイルの選択はやり直さない。
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result
