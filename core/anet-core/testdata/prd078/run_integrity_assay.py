#!/usr/bin/env python3
"""PRD078 ReplayBuffer integrity assay runner."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


CASE_COUNT = 384
DEFAULT_SEED = 20260919
SNAPSHOT_STARTED_RE = re.compile(r"Replay integrity snapshot: pushed=(\d+)")
SNAPSHOT_COMPLETED_RE = re.compile(r"Replay integrity snapshot complete: pushed=(\d+)")
PASSED_RE = re.compile(
    r"Replay integrity passed: case=(\d+) seed=(\d+) snapshots=(\d+) "
    r"covered_keys=(\d+) checked_samples=(\d+) seconds=([0-9.eE+-]+)"
)


@dataclass(frozen=True)
class Result:
    case: int
    status: str
    exit_code: int | None
    seconds: float
    last_completed_snapshot: int | None
    failed_snapshot: int | None
    log_path: Path
    snapshots: int | None = None
    covered_keys: int | None = None
    checked_samples: int | None = None


def parse_cases(value: str) -> list[int]:
    """単独番号または包含範囲の列を、重複のないcase番号列へ変換する。"""
    cases: list[int] = []
    seen: set[int] = set()
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            raise argparse.ArgumentTypeError("--cases contains an empty item")
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"invalid --cases item: {token!r}; expected N or A-B"
            )
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first > last:
            raise argparse.ArgumentTypeError(
                f"invalid descending --cases range: {token!r}"
            )
        for case in range(first, last + 1):
            if not 1 <= case <= CASE_COUNT:
                raise argparse.ArgumentTypeError(
                    f"case must be in 1-{CASE_COUNT}: {case}"
                )
            if case in seen:
                raise argparse.ArgumentTypeError(f"duplicate case: {case}")
            seen.add(case)
            cases.append(case)
    if not cases:
        raise argparse.ArgumentTypeError("--cases must select at least one case")
    return cases


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("--timeout-seconds must be greater than zero")
    return seconds


def uint32(value: str) -> int:
    seed = int(value)
    if not 0 <= seed <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("--seed must be in 0-4294967295")
    return seed


def make_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected PRD078 ReplayBuffer integrity assay cases."
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=repo_root / "core" / "anet-core" / "bin" / "Debug" / "anet-core-test.exe",
        help="Catch2 test executable (default: Debug anet-core-test.exe)",
    )
    parser.add_argument(
        "--cases",
        type=parse_cases,
        default=parse_cases(f"1-{CASE_COUNT}"),
        help=f"comma-separated N or inclusive A-B selections (default: 1-{CASE_COUNT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (default: .scratch/prd078/<timestamp>)",
    )
    parser.add_argument("--seed", type=uint32, default=DEFAULT_SEED)
    parser.add_argument("--timeout-seconds", type=positive_seconds, default=300.0)
    return parser


def parse_progress(output: str) -> tuple[int | None, int | None]:
    started = [int(value) for value in SNAPSHOT_STARTED_RE.findall(output)]
    completed = [int(value) for value in SNAPSHOT_COMPLETED_RE.findall(output)]
    last_started = started[-1] if started else None
    last_completed = completed[-1] if completed else None
    failed = last_started if last_started != last_completed else None
    return last_completed, failed


def run_case(
    repo_root: Path,
    executable: Path,
    output_dir: Path,
    case: int,
    seed: int,
    timeout_seconds: float,
) -> Result:
    log_path = output_dir / f"case-{case:03d}.log"
    command = [
        str(executable),
        "[integrity_assay]",
        "--section",
        f"case {case}",
        "--rng-seed",
        str(seed),
    ]
    started_at = time.perf_counter()
    exit_code: int | None = None
    status = "failed"
    output = ""
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        exit_code = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as error:
        status = "timeout"
        raw_output = error.stdout or ""
        output = raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else raw_output
        output += f"\nRunner timeout after {timeout_seconds:g} seconds.\n"
    except OSError as error:
        status = "launch_error"
        output = f"Runner failed to launch test executable: {error}\n"
    seconds = time.perf_counter() - started_at

    marker = PASSED_RE.search(output)
    snapshots = covered_keys = checked_samples = None
    if status not in {"timeout", "launch_error"}:
        if exit_code != 0:
            status = "failed"
        elif marker is None or int(marker.group(1)) != case or int(marker.group(2)) != seed:
            status = "completion_marker_missing"
        else:
            status = "passed"
            snapshots = int(marker.group(3))
            covered_keys = int(marker.group(4))
            checked_samples = int(marker.group(5))

    last_completed, failed_snapshot = parse_progress(output)
    log_path.write_text(output, encoding="utf-8", newline="\n")
    return Result(
        case=case,
        status=status,
        exit_code=exit_code,
        seconds=seconds,
        last_completed_snapshot=last_completed,
        failed_snapshot=failed_snapshot,
        log_path=log_path,
        snapshots=snapshots,
        covered_keys=covered_keys,
        checked_samples=checked_samples,
    )


def write_results_csv(output_dir: Path, results: list[Result]) -> None:
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "case",
                "status",
                "exit_code",
                "seconds",
                "last_completed_snapshot",
                "failed_snapshot",
                "log_path",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case,
                    result.status,
                    "" if result.exit_code is None else result.exit_code,
                    f"{result.seconds:.6f}",
                    "" if result.last_completed_snapshot is None else result.last_completed_snapshot,
                    "" if result.failed_snapshot is None else result.failed_snapshot,
                    result.log_path.relative_to(output_dir).as_posix(),
                ]
            )


def write_report(
    output_dir: Path,
    executable: Path,
    cases_text: str,
    seed: int,
    timeout_seconds: float,
    results: list[Result],
) -> None:
    passed = [result for result in results if result.status == "passed"]
    failed = [result for result in results if result.status != "passed"]
    lines = [
        "# PRD078 ReplayBuffer integrity assay",
        "",
        f"- 実行体: `{executable}`",
        f"- case指定: `{cases_text}`",
        f"- seed: `{seed}`",
        f"- case timeout: `{timeout_seconds:g}` 秒",
        f"- 結果: {len(passed)} passed / {len(failed)} failed / {len(results)} total",
        f"- 実行時間合計: {sum(result.seconds for result in results):.3f} 秒",
        f"- 完了snapshot合計: {sum(result.snapshots or 0 for result in passed)}",
        f"- 被覆key合計: {sum(result.covered_keys or 0 for result in passed)}",
        f"- 検査sample合計: {sum(result.checked_samples or 0 for result in passed)}",
        "",
    ]
    if failed:
        lines.extend(
            [
                "## 失敗",
                "",
                "| case | status | exit code | 最終完了地点 | 失敗地点 | log |",
                "|---:|---|---:|---:|---:|---|",
            ]
        )
        for result in failed:
            lines.append(
                "| {case} | {status} | {exit_code} | {last} | {failed_at} | `{log}` |".format(
                    case=result.case,
                    status=result.status,
                    exit_code="" if result.exit_code is None else result.exit_code,
                    last="" if result.last_completed_snapshot is None else result.last_completed_snapshot,
                    failed_at="" if result.failed_snapshot is None else result.failed_snapshot,
                    log=result.log_path.relative_to(output_dir).as_posix(),
                )
            )
        lines.append("")
    else:
        lines.extend(["全選択caseが完走し、完了標記を出力しました。", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    parser = make_parser(repo_root)
    args = parser.parse_args()
    executable = args.exe if args.exe.is_absolute() else (repo_root / args.exe)
    executable = executable.resolve()
    if not executable.is_file():
        parser.error(f"test executable does not exist: {executable}")

    cases_text = ",".join(str(case) for case in args.cases)
    output_dir = args.output
    if output_dir is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = repo_root / ".scratch" / "prd078" / timestamp
    elif not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    results: list[Result] = []
    print(f"PRD078 integrity assay: {len(args.cases)} case(s), seed={args.seed}")
    print(f"Output: {output_dir}")
    for position, case in enumerate(args.cases, start=1):
        result = run_case(
            repo_root,
            executable,
            output_dir,
            case,
            args.seed,
            args.timeout_seconds,
        )
        results.append(result)
        write_results_csv(output_dir, results)
        print(
            f"[{position}/{len(args.cases)}] case {case}: {result.status} "
            f"({result.seconds:.3f}s)"
        )

    write_report(
        output_dir,
        executable,
        cases_text,
        args.seed,
        args.timeout_seconds,
        results,
    )
    failed_count = sum(result.status != "passed" for result in results)
    print(f"Summary: {len(results) - failed_count} passed, {failed_count} failed")
    print(f"Report: {output_dir / 'report.md'}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())
