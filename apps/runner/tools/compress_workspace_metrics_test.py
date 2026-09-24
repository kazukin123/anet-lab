#!/usr/bin/env python3

import gzip
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import compress_workspace_metrics as subject


class CompressWorkspaceMetricsTest(unittest.TestCase):
    def make_workspace(self, root: Path) -> Path:
        workspace = root / "workspace"
        (workspace / "runs").mkdir(parents=True)
        return workspace

    def add_run(self, workspace: Path, name: str, contents: bytes | None, gzip_contents: bytes | None = None) -> Path:
        run_dir = workspace / "runs" / name
        run_dir.mkdir()
        if contents is not None:
            (run_dir / subject.RAW_NAME).write_bytes(contents)
        if gzip_contents is not None:
            with gzip.open(run_dir / subject.GZIP_NAME, "wb") as stream:
                stream.write(gzip_contents)
        return run_dir

    def test_compresses_replaces_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.make_workspace(Path(temp_dir))
            raw_run = self.add_run(workspace, "run_b", b'{"value":1}\n')
            replace_run = self.add_run(workspace, "run_a", b'{"value":2}\n', b"old\n")

            _, plans, _ = subject.inspect_workspace(workspace)
            summary = subject.execute(plans, enough_space=True, dry_run=False)

            self.assertEqual((summary.compressed, summary.replaced, summary.failed), (1, 1, 0))
            for run_dir, expected in ((raw_run, b'{"value":1}\n'), (replace_run, b'{"value":2}\n')):
                self.assertFalse((run_dir / subject.RAW_NAME).exists())
                with gzip.open(run_dir / subject.GZIP_NAME, "rb") as stream:
                    self.assertEqual(stream.read(), expected)

            _, second_plans, _ = subject.inspect_workspace(workspace)
            second = subject.execute(second_plans, enough_space=True, dry_run=False)
            self.assertEqual(second.already_compressed, 2)

    def test_skips_incomplete_tail_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.make_workspace(Path(temp_dir))
            run_dir = self.add_run(workspace, "run_partial", b'{"value":1}')

            _, plans, _ = subject.inspect_workspace(workspace)
            summary = subject.execute(plans, enough_space=True, dry_run=False)

            self.assertEqual(summary.skipped, 1)
            self.assertEqual((run_dir / subject.RAW_NAME).read_bytes(), b'{"value":1}')
            self.assertFalse((run_dir / subject.GZIP_NAME).exists())

    def test_compresses_empty_metrics_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.make_workspace(Path(temp_dir))
            run_dir = self.add_run(workspace, "run_empty", b"")
            _, plans, _ = subject.inspect_workspace(workspace)

            summary = subject.execute(plans, enough_space=True, dry_run=False)

            self.assertEqual(summary.compressed, 1)
            with gzip.open(run_dir / subject.GZIP_NAME, "rb") as stream:
                self.assertEqual(stream.read(), b"")

    def test_dry_run_and_insufficient_space_do_not_modify_files(self):
        for enough_space in (True, False):
            with self.subTest(enough_space=enough_space), tempfile.TemporaryDirectory() as temp_dir:
                workspace = self.make_workspace(Path(temp_dir))
                run_dir = self.add_run(workspace, "run", b"{}\n")
                _, plans, _ = subject.inspect_workspace(workspace)

                summary = subject.execute(plans, enough_space=enough_space, dry_run=True)

                self.assertTrue((run_dir / subject.RAW_NAME).exists())
                self.assertFalse((run_dir / subject.GZIP_NAME).exists())
                self.assertEqual(summary.skipped, 0 if enough_space else 1)

    def test_verification_failure_preserves_raw_and_existing_gzip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.make_workspace(Path(temp_dir))
            run_dir = self.add_run(workspace, "run", b"new\n", b"old\n")
            _, plans, _ = subject.inspect_workspace(workspace)

            with mock.patch.object(subject, "_verify_gzip", return_value=("wrong", 1)):
                summary = subject.execute(plans, enough_space=True, dry_run=False)

            self.assertEqual(summary.failed, 1)
            self.assertEqual((run_dir / subject.RAW_NAME).read_bytes(), b"new\n")
            with gzip.open(run_dir / subject.GZIP_NAME, "rb") as stream:
                self.assertEqual(stream.read(), b"old\n")

    def test_source_delete_failure_restores_existing_gzip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.make_workspace(Path(temp_dir))
            run_dir = self.add_run(workspace, "run", b"new\n", b"old\n")
            _, plans, _ = subject.inspect_workspace(workspace)

            with mock.patch.object(subject, "_delete_source", side_effect=OSError("busy reader")):
                summary = subject.execute(plans, enough_space=True, dry_run=False)

            self.assertEqual(summary.failed, 1)
            self.assertEqual((run_dir / subject.RAW_NAME).read_bytes(), b"new\n")
            with gzip.open(run_dir / subject.GZIP_NAME, "rb") as stream:
                self.assertEqual(stream.read(), b"old\n")

    def test_confirmation_accepts_only_exact_trimmed_values(self):
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            answer = subject.choose_action(io.StringIO("yes\nMAYBE\n DRY-RUN \n"))
        self.assertEqual(answer, "DRY-RUN")
        self.assertIn("Enter exactly YES, NO, or DRY-RUN", output.getvalue())

    def test_busy_source_is_reported_as_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.make_workspace(Path(temp_dir))
            self.add_run(workspace, "run", b"{}\n")

            with mock.patch.object(subject, "_open_stable_source", side_effect=OSError("busy")):
                _, plans, _ = subject.inspect_workspace(workspace)

            self.assertEqual(plans[0].action, "skip")
            self.assertIn("source is in use", plans[0].reason)

    def test_main_supports_no_yes_and_dry_run_outcomes(self):
        root = Path("workspace")
        skip_plan = subject.RunPlan(root / "runs" / "run", None, None, "skip", reason="busy")
        cases = (
            ("NO", [], False, 2),
            ("DRY-RUN", [], False, 0),
            ("YES", [skip_plan], False, 1),
            (None, [], True, 0),
        )
        for answer, plans, cli_dry_run, expected in cases:
            with self.subTest(answer=answer, cli_dry_run=cli_dry_run):
                arguments = ["--workspace-root", str(root)]
                if cli_dry_run:
                    arguments.append("--dry-run")
                with (
                    mock.patch.object(
                        subject, "inspect_workspace", return_value=(root / "runs", plans, 0)
                    ) as inspect,
                    mock.patch.object(subject, "print_preflight", return_value=True),
                    mock.patch.object(subject, "print_summary"),
                    mock.patch.object(subject, "choose_action", return_value=answer or "NO") as choose,
                ):
                    result = subject.main(arguments)
                self.assertEqual(result, expected)
                if cli_dry_run:
                    choose.assert_not_called()
                self.assertEqual(inspect.call_count, 2 if answer == "YES" else 1)

    def test_yes_executes_the_rechecked_plan(self):
        root = Path("workspace")
        first_plan = subject.RunPlan(root / "runs" / "old", root / "old", None, "compress", 1)
        rechecked_plan = subject.RunPlan(root / "runs" / "new", None, None, "skip", reason="busy")
        with (
            mock.patch.object(
                subject,
                "inspect_workspace",
                side_effect=(
                    (root / "runs", [first_plan], 1),
                    (root / "runs", [rechecked_plan], 0),
                ),
            ),
            mock.patch.object(subject, "print_preflight", return_value=True),
            mock.patch.object(subject, "choose_action", return_value="YES"),
            mock.patch.object(subject, "execute", return_value=subject.Summary()) as execute,
            mock.patch.object(subject, "print_summary"),
        ):
            result = subject.main(["--workspace-root", str(root)])

        self.assertEqual(result, 0)
        execute.assert_called_once_with([rechecked_plan], True, False)


if __name__ == "__main__":
    unittest.main()
