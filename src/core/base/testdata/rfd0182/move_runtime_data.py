"""Move the previous Runner workspace and appdata directories into data/."""

import argparse
import os
from pathlib import Path


def child_directories(path):
    return sorted((item for item in path.iterdir() if item.is_dir())) if path.is_dir() else []


def counts(path):
    workspaces = child_directories(path)
    return len(workspaces), sum(len(child_directories(workspace / "runs")) for workspace in workspaces)


def expected(root):
    old = child_directories(root / "apps/runner/workspaces")
    new = child_directories(root / "data/workspaces")
    names = {os.path.normcase(child.name) for child in (*old, *new)}
    runs = sum(len(child_directories(workspace / "runs")) for workspace in (*old, *new))
    return len(names), runs


def entries(path):
    if not path.exists():
        return []
    if not path.is_dir():
        raise NotADirectoryError(path)
    return sorted(path.iterdir())



def verify_paths(source, destination, root):
    if not source.is_dir():
        raise NotADirectoryError(f"Expected a directory to rename: {source}")
    for path in (source, destination):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Path leaves the repository: {path}")
        relative = path.relative_to(root)
        for parent in (root.joinpath(*relative.parts[:n]) for n in range(1, len(relative.parts) + 1)):
            if parent.is_symlink() or parent.is_junction():
                raise ValueError(f"Cannot rename through a symbolic link or junction: {parent}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    existing = next(parent for parent in destination.parents if parent.exists())
    if source.stat().st_dev != existing.stat().st_dev:
        raise ValueError(f"Directories are on different drives: {source} -> {destination}")


def plan(root):
    root = root.resolve(strict=True)
    old_workspaces = root / "apps/runner/workspaces"
    workspaces = root / "data/workspaces"
    old_entries = entries(old_workspaces)
    entries(workspaces)
    if old_workspaces.is_dir() and not workspaces.exists():
        # 旧ルート全体の改名なら、タグセットも同じディレクトリに残る。
        moves = [(old_workspaces, workspaces)]
    else:
        moves = [(source, workspaces / source.name) for source in old_entries]
        for source, _ in moves:
            if not source.is_dir():
                raise ValueError(
                    f"Cannot move a loose workspace file by renaming directories: {source}. "
                    f"Set aside {workspaces} before migrating the whole workspace directory.")
    appdata = root / "apps/runner/appdata"
    if appdata.exists() or appdata.is_symlink() or appdata.is_junction():
        moves.append((appdata, root / "data/appdata"))
    for source, destination in moves:
        verify_paths(source, destination, root)
    return moves


def apply(moves, root, before):
    if os.name != "nt":
        raise OSError("Apply requires Windows directory renames.")
    completed = []
    try:
        for source, destination in moves:
            verify_paths(source, destination, root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            completed.append((source, destination))
        actual = counts(root / "data/workspaces")
        if actual != before:
            raise RuntimeError(f"Workspace and Run counts changed: expected {before}, found {actual}")
    except Exception:
        # 改名済みのものを逆順で戻す。コピーと削除は行わない。
        for source, destination in reversed(completed):
            verify_paths(destination, source, root)
            destination.rename(source)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--apply", action="store_true", help="Rename the directories after stopping all Runs.")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    moves = plan(root)
    before = expected(root)
    for source, destination in moves:
        print(f"{source} -> {destination}")
    print(f"Before: {before[0]} workspaces and {before[1]} Run directories.")
    if args.apply:
        apply(moves, root, before)
        after = counts(root / "data/workspaces")
        print(f"After: {after[0]} workspaces and {after[1]} Run directories.")
    else:
        print("Preview only. Use --apply on a stopped installation to perform the renames.")


if __name__ == "__main__":
    main()
