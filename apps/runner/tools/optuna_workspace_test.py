#!/usr/bin/env python3

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import dropmerge_optuna as subject
import optuna
import optuna_common


class OptunaWorkspaceTest(unittest.TestCase):
    def test_dry_run_writes_workspace_config_under_workspace_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            runner_root = repo_root / "apps" / "runner"
            workspace_root = runner_root / "workspaces" / "study_ws"
            workspace_config = workspace_root / "config" / "_main.txt"
            workspace_config.parent.mkdir(parents=True)
            workspace_config.write_text("$include <DropMerge.txt>\n", encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = subject.main([
                    "dry-run",
                    "--repo-root", str(repo_root),
                    "--workspace", "study_ws",
                    "--study-name", "workspaceSmoke",
                    "--trial-name", "t00000",
                ])

            self.assertEqual(result, 0)
            document = json.loads(output.getvalue())
            config_path = Path(document["context"]["config_path"])
            self.assertEqual(
                config_path,
                workspace_root / "runs" / "workspaceSmoke_t00000" / "trial" / "config.txt",
            )
            config_text = config_path.read_text(encoding="utf-8")
            self.assertIn(f'$include "{workspace_config.as_posix()}"', config_text)
            self.assertLess(config_text.index("$include <_main.txt>"), config_text.index(str(workspace_config.as_posix())))
            self.assertLess(config_text.index(str(workspace_config.as_posix())), config_text.index("$include <DropMerge_optuna.txt>"))
            self.assertFalse((workspace_root / "optuna").exists())

    def test_run_commands_reject_removed_runs_dir_argument(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for command in ("dry-run", "run-trial", "run-study"):
                with self.subTest(command=command):
                    with self.assertRaises(SystemExit) as raised:
                        subject.main([
                            command,
                            "--repo-root", temp_dir,
                            "--study-name", "legacy",
                            "--runs-dir", "runs_optuna",
                        ])
                    self.assertEqual(raised.exception.code, 2)

    def test_workspace_forbidden_characters_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workspaces_root = repo_root / "apps" / "runner" / "workspaces"
            for value in ("bad#name", "bad//name", "bad;", r"\\server\share", "//server/share"):
                with self.subTest(value=value):
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors):
                        result = subject.main([
                            "dry-run",
                            "--repo-root", str(repo_root),
                            "--workspace", value,
                            "--study-name", "invalidWorkspace",
                        ])

                    self.assertEqual(result, 2)
                    self.assertIn("Invalid --workspace", errors.getvalue())
                    self.assertFalse(workspaces_root.exists())

    def test_run_study_missing_workspace_config_has_no_output_side_effects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workspace_root = repo_root / "apps" / "runner" / "workspaces" / "missing_config"
            workspace_root.mkdir(parents=True)

            result = subject.main([
                "run-study",
                "--repo-root", str(repo_root),
                "--workspace", "missing_config",
                "--study-name", "missingConfig",
                "--n-trials", "0",
            ])

            self.assertEqual(result, 2)
            self.assertFalse((workspace_root / "runs").exists())
            self.assertFalse((workspace_root / "optuna").exists())

    def test_cleanup_missing_explicit_storage_does_not_create_parent_or_require_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            storage_path = repo_root / "outside" / "missing.db"
            errors = io.StringIO()

            with contextlib.redirect_stderr(errors):
                result = subject.main([
                    "cleanup-running",
                    "--repo-root", str(repo_root),
                    "--workspace", "missing-workspace",
                    "--study-name", "missingStudy",
                    "--storage", str(storage_path),
                    "--dry-run",
                ])

            self.assertEqual(result, 2)
            self.assertIn("Storage file does not exist", errors.getvalue())
            self.assertFalse(storage_path.parent.exists())

    def test_run_study_uses_workspace_optuna_outputs_and_workspace_restart_attrs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workspace_root = repo_root / "apps" / "runner" / "workspaces" / "study_ws"
            config_file = workspace_root / "config" / "_main.txt"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("$include <DropMerge.txt>\n", encoding="utf-8")

            completed = subprocess.run([
                    sys.executable,
                    str(Path(subject.__file__)),
                    "run-study",
                    "--repo-root", str(repo_root),
                    "--workspace", "study_ws",
                    "--study-name", "workspaceStudy",
                    "--n-trials", "0",
                    "--heartbeat-interval-sec", "0",
                ], capture_output=True, text=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            optuna_dir = workspace_root / "optuna"
            storage_path = optuna_dir / "optuna.db"
            self.assertTrue(storage_path.is_file())
            self.assertTrue((optuna_dir / "artifacts").is_dir())
            self.assertTrue((optuna_dir / "harness.log").is_file())
            self.assertFalse((workspace_root / "runs").exists())

            storage = optuna.storages.RDBStorage(url=f"sqlite:///{storage_path.as_posix()}")
            try:
                study = optuna.load_study(study_name="workspaceStudy", storage=storage)
                self.assertEqual(study.user_attrs["last_workspace"], "study_ws")
                self.assertNotIn("last_runs_dir", study.user_attrs)
                restart_args = study.user_attrs["00_last_run_study_args"]
                self.assertIn("--workspace study_ws", restart_args)
                self.assertNotIn("--runs-dir", restart_args)
            finally:
                storage.remove_session()
                storage.engine.dispose()

    def test_run_study_rejects_outputs_outside_workspace_optuna_bucket_without_side_effects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workspace_root = repo_root / "apps" / "runner" / "workspaces" / "dm_opt"
            config_file = workspace_root / "config" / "_main.txt"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("$include <DropMerge.txt>\n", encoding="utf-8")
            cases = (
                ("--storage", str(workspace_root.parent / "dm_opt2" / "optuna.db")),
                ("--storage", str(workspace_root / "runs" / "wrong.db")),
                ("--storage", str(workspace_root / "optuna" / ".." / "escape.db")),
                ("--storage", str(workspace_root / "optuna")),
                ("--optuna-artifact-dir", str(workspace_root / "config")),
                ("--optuna-artifact-dir", str(workspace_root / "optuna")),
            )

            for option, value in cases:
                with self.subTest(option=option, value=value):
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors):
                        result = subject.main([
                            "run-study",
                            "--repo-root", str(repo_root),
                            "--workspace", "dm_opt",
                            "--study-name", "outsideOutput",
                            "--n-trials", "0",
                            option, value,
                        ])
                    self.assertEqual(result, 2)
                    self.assertIn("workspace optuna directory", errors.getvalue())
                    self.assertFalse((workspace_root / "optuna").exists())
                    self.assertFalse((workspace_root / "runs").exists())

    def test_run_study_rejects_storage_artifact_target_conflict_without_side_effects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workspace_root = repo_root / "apps" / "runner" / "workspaces" / "dm_opt"
            config_file = workspace_root / "config" / "_main.txt"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("$include <DropMerge.txt>\n", encoding="utf-8")
            shared_path = workspace_root / "optuna" / "shared"

            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                result = subject.main([
                    "run-study",
                    "--repo-root", str(repo_root),
                    "--workspace", "dm_opt",
                    "--study-name", "conflictingOutput",
                    "--n-trials", "0",
                    "--storage", str(shared_path),
                    "--optuna-artifact-dir", str(shared_path / "artifacts"),
                ])

            self.assertEqual(result, 2)
            self.assertIn("Invalid output paths", errors.getvalue())
            self.assertFalse((workspace_root / "optuna").exists())
            self.assertFalse((workspace_root / "runs").exists())

    def test_summarize_invalid_source_does_not_create_target_or_require_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            source_storage = repo_root / "source" / "missing.db"
            source_artifact = repo_root / "source" / "missing-artifacts"
            target_storage = repo_root / "target" / "summary.db"
            target_artifact = repo_root / "target" / "artifacts"
            errors = io.StringIO()

            with contextlib.redirect_stderr(errors):
                result = subject.main([
                    "summarize-study",
                    "--repo-root", str(repo_root),
                    "--workspace", "missing-workspace",
                    "--source-study-name", "sourceStudy",
                    "--source-storage", str(source_storage),
                    "--source-artifact-dir", str(source_artifact),
                    "--target-storage", str(target_storage),
                    "--target-artifact-dir", str(target_artifact),
                ])

            self.assertEqual(result, 2)
            self.assertIn("Source storage is not a regular file", errors.getvalue())
            self.assertFalse(target_storage.parent.exists())
            self.assertFalse(target_artifact.exists())

    def test_storage_parser_treats_sqlite_url_and_bare_path_equally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            storage_path = repo_root / "apps" / "runner" / "workspaces" / "ws" / "optuna" / "optuna.db"
            bare = optuna_common.storage_path_from_text(repo_root, str(storage_path))
            url = optuna_common.storage_path_from_text(repo_root, f"sqlite:///{storage_path.as_posix()}")
            self.assertEqual(bare, url)
            invalid_values = (
                f"sqlite:///{storage_path.parent.as_posix()}/bad#name.db",
                f"sqlite:///{storage_path.parent.as_posix()}//duplicate.db",
                r"\\server\share\optuna.db",
                "postgresql://localhost/optuna",
            )
            for value in invalid_values:
                with self.subTest(value=value):
                    with self.assertRaises(optuna_common.TrialExecutionError):
                        optuna_common.storage_path_from_text(repo_root, value)

    def test_cleanup_explicit_storage_succeeds_without_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            storage_path = repo_root / "source" / "optuna.db"
            storage_path.parent.mkdir()
            storage_url = f"sqlite:///{storage_path.as_posix()}"
            storage = optuna.storages.RDBStorage(url=storage_url)
            study = optuna.create_study(study_name="cleanupStudy", storage=storage)
            running_trial = study.ask()
            storage.remove_session()
            storage.engine.dispose()
            del study
            del storage

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(subject.__file__)),
                    "cleanup-running",
                    "--repo-root", str(repo_root),
                    "--workspace", "missing-workspace",
                    "--study-name", "cleanupStudy",
                    "--storage", str(storage_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            verification_storage = optuna.storages.RDBStorage(url=storage_url)
            try:
                reloaded = optuna.load_study(study_name="cleanupStudy", storage=verification_storage)
                self.assertEqual(reloaded.trials[running_trial.number].state.name, "FAIL")
            finally:
                del reloaded
                verification_storage.remove_session()
                verification_storage.engine.dispose()

    def test_summarize_source_types_fail_before_target_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            source_root = repo_root / "source"
            source_root.mkdir()
            target_storage = repo_root / "target" / "summary.db"
            target_artifact = repo_root / "target" / "artifacts"

            source_storage_directory = source_root / "storage-directory"
            source_storage_directory.mkdir()
            source_artifact_directory = source_root / "artifacts"
            source_artifact_directory.mkdir()
            source_storage_file = source_root / "optuna.db"
            source_storage_file.touch()
            source_artifact_file = source_root / "artifact-file"
            source_artifact_file.touch()

            cases = (
                (source_storage_directory, source_artifact_directory, "Source storage"),
                (source_storage_file, source_artifact_file, "Source artifact"),
            )
            for source_storage, source_artifact, expected_error in cases:
                with self.subTest(expected_error=expected_error):
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors):
                        result = subject.main([
                            "summarize-study",
                            "--repo-root", str(repo_root),
                            "--workspace", "missing-workspace",
                            "--source-study-name", "sourceStudy",
                            "--source-storage", str(source_storage),
                            "--source-artifact-dir", str(source_artifact),
                            "--target-storage", str(target_storage),
                            "--target-artifact-dir", str(target_artifact),
                        ])

                    self.assertEqual(result, 2)
                    self.assertIn(expected_error, errors.getvalue())
                    self.assertFalse(target_storage.parent.exists())
                    self.assertFalse(target_artifact.exists())

    def test_run_study_artifact_init_failure_precedes_db_run_and_log_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workspace_root = repo_root / "apps" / "runner" / "workspaces" / "study_ws"
            config_file = workspace_root / "config" / "_main.txt"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("$include <DropMerge.txt>\n", encoding="utf-8")

            with mock.patch.object(
                optuna_common.ArtifactStore,
                "create_required_context",
                side_effect=optuna_common.TrialExecutionError("artifact init failed"),
            ):
                result = subject.main([
                    "run-study",
                    "--repo-root", str(repo_root),
                    "--workspace", "study_ws",
                    "--study-name", "artifactFailure",
                    "--n-trials", "0",
                ])

            self.assertEqual(result, 2)
            self.assertFalse((workspace_root / "optuna" / "optuna.db").exists())
            self.assertFalse((workspace_root / "optuna" / "harness.log").exists())
            self.assertFalse((workspace_root / "runs").exists())

    def test_symlink_escape_from_workspace_optuna_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workspace_root = repo_root / "apps" / "runner" / "workspaces" / "study_ws"
            config_file = workspace_root / "config" / "_main.txt"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("$include <DropMerge.txt>\n", encoding="utf-8")
            optuna_dir = workspace_root / "optuna"
            optuna_dir.mkdir()
            outside = repo_root / "outside"
            outside.mkdir()
            link = optuna_dir / "escape"
            try:
                os.symlink(outside, link, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Directory symlink is unavailable: {error}")

            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                result = subject.main([
                    "run-study",
                    "--repo-root", str(repo_root),
                    "--workspace", "study_ws",
                    "--study-name", "symlinkEscape",
                    "--n-trials", "0",
                    "--storage", str(link / "escape.db"),
                ])

            self.assertEqual(result, 2)
            self.assertIn("workspace optuna directory", errors.getvalue())
            self.assertFalse((outside / "escape.db").exists())

    def test_summarize_omitted_targets_inherit_explicit_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            source_storage = repo_root / "source" / "optuna.db"
            source_storage.parent.mkdir()
            source_storage.touch()
            source_artifact = repo_root / "source" / "artifacts"
            source_artifact.mkdir()
            args = subject.build_parser().parse_args([
                "summarize-study",
                "--repo-root", str(repo_root),
                "--workspace", "missing-workspace",
                "--source-study-name", "sourceStudy",
                "--source-storage", str(source_storage),
                "--source-artifact-dir", str(source_artifact),
            ])

            subject.RUNTIME.prepare_summarize_paths(args)

            self.assertEqual(args.target_storage, args.source_storage)
            self.assertEqual(args.target_artifact_dir, args.source_artifact_dir)

    def test_invalid_run_study_arguments_do_not_create_workspace_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workspace_root = repo_root / "apps" / "runner" / "workspaces" / "study_ws"
            config_file = workspace_root / "config" / "_main.txt"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("$include <DropMerge.txt>\n", encoding="utf-8")

            result = subject.main([
                "run-study",
                "--repo-root", str(repo_root),
                "--workspace", "study_ws",
                "--study-name", "invalidArgs",
                "--n-trials", "-1",
            ])

            self.assertEqual(result, 2)
            self.assertFalse((workspace_root / "optuna").exists())
            self.assertFalse((workspace_root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
