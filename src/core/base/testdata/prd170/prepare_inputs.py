"""PRD103 の履歴設定を保持し、PRD170 の変更前後へ同じ受入入力を用意する。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

REPO = Path(__file__).resolve().parents[5]
CASES = (("GridMaze", False), ("LunarLander", False), ("Atari", False),
         ("DropMerge", False), ("GridMaze", True), ("LunarLander", True))


def prepare_ui(destination):
    config = destination / "config"
    config.mkdir(parents=True, exist_ok=False)
    for path in (REPO / "apps/runner/config").glob("*.txt"):
        shutil.copy2(path, config / path.name)
    for env, label in (("ImageCls", "img"), ("Atari", "atari")):
        lines = [f"$include <{config.as_posix()}/_main.txt>", f"$include <{config.as_posix()}/{env}.txt>",
                 "app.$ = app.online > P1", "app.train_auto_start = false", "app.eval_panel.auto_start = false",
                 "app.eval_panel.fps = 1", "app.eval_panel.step_per_frame = 100", "app.save_agent_on_close = false",
                 "app.show_error_dialog = false", "app.train_panel.fps = 0",
                 "app.runs_dir = " + (destination / "runs").as_posix(),
                 "app.run_name = run_{t}_p170_" + label]
        (destination / (label + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("PREPARED UI / GPU memory inputs:", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / ".scratch/prd170/validation")
    parser.add_argument("--ui-output", type=Path)
    args = parser.parse_args()
    if args.ui_output is not None:
        prepare_ui(args.ui_output.resolve())
        return
    destination = args.output.resolve()
    if (destination / "baseline-config").exists():
        raise RuntimeError("Frozen input already exists: " + str(destination))
    source = REPO / "src/core/base/testdata/prd103"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "baseline-config", destination / "baseline-config")
    shutil.copytree(source / "resolved-before", destination / "resolved-before")
    removed = []
    # PRD117 で廃止した未読キーだけを受入用コピーから除く。履歴原本へは書かない。
    obsolete = re.compile(r"\.(?:auto_scale_mode|histgram\.interval)\s*[?]?=")
    for path in (destination / "baseline-config").glob("*.txt"):
        original = path.read_bytes()
        lines = original.decode("utf-8-sig").splitlines()
        removed.extend({"file": path.name, "line": line} for line in lines if obsolete.search(line))
        text = "\n".join(line for line in lines if not obsolete.search(line)) + "\n"
        bom = b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
        path.write_bytes(bom + text.encode("utf-8"))
    path = REPO / "src/core/base/testdata/prd098/run_validation.py"
    spec = importlib.util.spec_from_file_location("prd098_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = destination
    for env, enabled in CASES:
        lines = module.settings(env, before=True, enabled=enabled)
        if env == "Atari":
            lines[-1] = lines[-1].replace(" > run.@prd098_validation", " > run.@btrnet > run.@prd098_validation")
        name = env + ("_on" if enabled else "_off") + ".txt"
        (destination / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest = {"removed_unread_keys": removed, "config_sha256": {
        str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*.txt"))}}
    (destination / "input-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PREPARED", len(CASES), "cases; removed", len(removed), "unread historical keys:", destination)


if __name__ == "__main__":
    main()
