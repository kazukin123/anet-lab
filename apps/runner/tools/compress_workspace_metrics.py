#!/usr/bin/env python3
"""Compress every direct Run metrics.jsonl in one workspace."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import sys
import uuid


RAW_NAME = "metrics.jsonl"
GZIP_NAME = "metrics.jsonl.gz"
COPY_CHUNK_SIZE = 1024 * 1024

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x00000080
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


@dataclass(frozen=True)
class RunPlan:
    run_dir: Path
    raw_path: Path | None
    gzip_path: Path | None
    action: str
    source_size: int = 0
    reason: str = ""


@dataclass
class Summary:
    compressed: int = 0
    replaced: int = 0
    already_compressed: int = 0
    skipped: int = 0
    failed: int = 0
    input_bytes: int = 0
    output_bytes: int = 0

    @property
    def has_blockers(self) -> bool:
        return self.skipped > 0 or self.failed > 0


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _open_stable_source(path: Path):
    """Open a source while allowing readers and denying concurrent writers."""
    if os.name != "nt":
        raise RuntimeError("Safe metrics compression is supported only on Windows.")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        GENERIC_READ,
        FILE_SHARE_READ,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == INVALID_HANDLE_VALUE:
        error_code = ctypes.get_last_error()
        raise OSError(error_code, ctypes.FormatError(error_code), str(path))

    import msvcrt

    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        kernel32.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, "rb", closefd=True)


def _has_complete_tail(stream, size: int) -> bool:
    if size == 0:
        return True
    stream.seek(-1, os.SEEK_END)
    complete = stream.read(1) == b"\n"
    stream.seek(0)
    return complete


def _estimated_gzip_size(source_size: int) -> int:
    return source_size + ((source_size // 16_384) + 1) * 5 + 64


def inspect_workspace(workspace_root: Path) -> tuple[Path, list[RunPlan], int]:
    root = workspace_root.resolve(strict=True)
    runs_dir = (root / "runs").resolve(strict=True)
    if not runs_dir.is_dir() or not _path_is_within(runs_dir, root):
        raise ValueError(f"Workspace runs directory is invalid: {runs_dir}")

    plans: list[RunPlan] = []
    required_free_bytes = 0
    for entry in sorted(runs_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not entry.is_dir():
            continue
        resolved_run = entry.resolve(strict=True)
        if not _path_is_within(resolved_run, runs_dir):
            plans.append(RunPlan(entry, None, None, "skip", reason="run path escapes runs directory"))
            continue

        raw = entry / RAW_NAME
        compressed = entry / GZIP_NAME
        has_raw = raw.is_file()
        has_gzip = compressed.is_file()
        if has_raw and not _path_is_within(raw.resolve(strict=True), resolved_run):
            plans.append(RunPlan(entry, raw, compressed if has_gzip else None, "skip",
                                 reason="metrics.jsonl escapes the Run directory"))
            continue
        if has_gzip and not _path_is_within(compressed.resolve(strict=True), resolved_run):
            plans.append(RunPlan(entry, raw if has_raw else None, compressed, "skip",
                                 reason="metrics.jsonl.gz escapes the Run directory"))
            continue
        if not has_raw:
            if has_gzip:
                plans.append(RunPlan(entry, None, compressed, "already-compressed"))
            continue

        try:
            with _open_stable_source(raw) as source:
                source_size = os.fstat(source.fileno()).st_size
                if not _has_complete_tail(source, source_size):
                    plans.append(
                        RunPlan(entry, raw, compressed if has_gzip else None, "skip", source_size,
                                "metrics.jsonl does not end with a newline")
                    )
                    continue
        except OSError as error:
            plans.append(
                RunPlan(entry, raw, compressed if has_gzip else None, "skip", reason=f"source is in use: {error}")
            )
            continue

        action = "replace" if has_gzip else "compress"
        plans.append(RunPlan(entry, raw, compressed if has_gzip else None, action, source_size))
        required_free_bytes = max(required_free_bytes, _estimated_gzip_size(source_size))

    return runs_dir, plans, required_free_bytes


def _format_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def print_preflight(workspace_root: Path, runs_dir: Path, plans: list[RunPlan], required_free_bytes: int) -> bool:
    print(f"[INFO] Workspace: {workspace_root}")
    print(f"[INFO] Runs directory: {runs_dir}")
    print("[INFO] Preflight:")
    for plan in plans:
        detail = f" ({_format_bytes(plan.source_size)})" if plan.source_size else ""
        reason = f": {plan.reason}" if plan.reason else ""
        print(f"  [{plan.action.upper()}] {plan.run_dir.name}{detail}{reason}")

    free_bytes = shutil.disk_usage(runs_dir).free
    enough_space = free_bytes >= required_free_bytes
    print(
        f"[INFO] Temporary space: required={_format_bytes(required_free_bytes)}, "
        f"free={_format_bytes(free_bytes)}"
    )
    if not enough_space:
        print("[ERROR] Insufficient free space for the largest temporary gzip file.")
    if not plans:
        print("[INFO] No Run metrics files were found.")
    return enough_space


def _copy_to_gzip(source, temporary_path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    source.seek(0)
    with temporary_path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw_output, mtime=0) as gzip_output:
            while chunk := source.read(COPY_CHUNK_SIZE):
                digest.update(chunk)
                byte_count += len(chunk)
                gzip_output.write(chunk)
        raw_output.flush()
        os.fsync(raw_output.fileno())
    return digest.hexdigest(), byte_count


def _verify_gzip(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(COPY_CHUNK_SIZE):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _restore_after_delete_failure(gzip_path: Path, backup_path: Path | None) -> None:
    gzip_path.unlink(missing_ok=True)
    if backup_path is not None and backup_path.exists():
        os.replace(backup_path, gzip_path)


def _delete_source(path: Path) -> None:
    path.unlink()


def compress_run(plan: RunPlan) -> int:
    assert plan.raw_path is not None
    gzip_path = plan.run_dir / GZIP_NAME
    temporary_path = plan.run_dir / f".{GZIP_NAME}.{uuid.uuid4().hex}.tmp"
    backup_path = plan.run_dir / f".{GZIP_NAME}.{uuid.uuid4().hex}.bak" if gzip_path.exists() else None

    try:
        # 圧縮と検証が終わるまで、他readerだけを許可してwriter・削除を排除する。
        with _open_stable_source(plan.raw_path) as source:
            current_size = os.fstat(source.fileno()).st_size
            if current_size != plan.source_size:
                raise RuntimeError(
                    f"source size changed after preflight: expected={plan.source_size}, actual={current_size}"
                )
            if not _has_complete_tail(source, current_size):
                raise RuntimeError("metrics.jsonl no longer ends with a newline")

            source_hash, source_bytes = _copy_to_gzip(source, temporary_path)
            gzip_hash, gzip_bytes = _verify_gzip(temporary_path)
            if (source_hash, source_bytes) != (gzip_hash, gzip_bytes):
                raise RuntimeError("gzip verification mismatch")

        # source handleを閉じてからgzipを配置し、元JSONLの削除失敗時は既存gzipへ戻す。
        if backup_path is not None:
            os.replace(gzip_path, backup_path)
        os.replace(temporary_path, gzip_path)
        try:
            _delete_source(plan.raw_path)
        except OSError:
            _restore_after_delete_failure(gzip_path, backup_path)
            raise

        if backup_path is not None:
            try:
                backup_path.unlink(missing_ok=True)
            except OSError as error:
                print(f"[WARN] Old gzip backup could not be removed: path={backup_path}: {error}")
        return gzip_path.stat().st_size
    except Exception:
        temporary_path.unlink(missing_ok=True)
        if backup_path is not None and backup_path.exists() and not gzip_path.exists():
            os.replace(backup_path, gzip_path)
        raise


def execute(plans: list[RunPlan], enough_space: bool, dry_run: bool) -> Summary:
    summary = Summary()
    for plan in plans:
        if plan.action == "already-compressed":
            summary.already_compressed += 1
        elif plan.action == "skip":
            summary.skipped += 1

    actionable = [plan for plan in plans if plan.action in ("compress", "replace")]
    summary.input_bytes = sum(plan.source_size for plan in actionable)
    if not enough_space and actionable:
        summary.skipped += len(actionable)
        return summary
    if dry_run:
        return summary

    for plan in actionable:
        try:
            output_size = compress_run(plan)
        except Exception as error:
            summary.failed += 1
            print(f"[ERROR] Failed: run={plan.run_dir.name}: {error}")
            continue
        if plan.action == "replace":
            summary.replaced += 1
        else:
            summary.compressed += 1
        summary.output_bytes += output_size
        print(f"[INFO] Completed: run={plan.run_dir.name}, output={_format_bytes(output_size)}")
    return summary


def print_summary(summary: Summary, dry_run: bool) -> None:
    mode = "DRY-RUN" if dry_run else "RESULT"
    print(f"[INFO] {mode} summary:")
    print(
        "  compressed={0.compressed}, replaced={0.replaced}, already-compressed={0.already_compressed}, "
        "skipped={0.skipped}, failed={0.failed}".format(summary)
    )
    print(f"  input={_format_bytes(summary.input_bytes)}, output={_format_bytes(summary.output_bytes)}")


def choose_action(input_stream=sys.stdin) -> str:
    while True:
        print("Execute compression? [YES/NO/DRY-RUN]: ", end="", flush=True)
        answer = input_stream.readline()
        if answer == "":
            return "NO"
        answer = answer.strip()
        if answer in ("YES", "NO", "DRY-RUN"):
            return answer
        print("[ERROR] Enter exactly YES, NO, or DRY-RUN.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress every direct Run metrics.jsonl in one workspace.")
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runs_dir, plans, required_free_bytes = inspect_workspace(args.workspace_root)
        enough_space = print_preflight(args.workspace_root.resolve(), runs_dir, plans, required_free_bytes)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[ERROR] Preflight failed: {error}")
        return 1

    action = "DRY-RUN" if args.dry_run else choose_action()
    if action == "NO":
        print("[INFO] Compression cancelled. No files were changed.")
        return 2
    dry_run = action == "DRY-RUN"
    if not dry_run:
        print("[INFO] Rechecking workspace state before compression.")
        try:
            runs_dir, plans, required_free_bytes = inspect_workspace(args.workspace_root)
            enough_space = print_preflight(args.workspace_root.resolve(), runs_dir, plans, required_free_bytes)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"[ERROR] Preflight recheck failed: {error}")
            return 1
    summary = execute(plans, enough_space, dry_run)
    print_summary(summary, dry_run)
    return 1 if summary.has_blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
