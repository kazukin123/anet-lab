from pathlib import Path
import tempfile
import unittest

import move_runtime_data as migration


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.area = tempfile.TemporaryDirectory(prefix="anet-rfd0182-")
        self.addCleanup(self.area.cleanup)
        self.root = Path(self.area.name).resolve()

    def folder(self, path):
        result = self.root / path
        result.mkdir(parents=True, exist_ok=True)
        return result

    def test_dry_run_leaves_workspaces_and_appdata_in_place(self):
        source = self.folder("apps/runner/workspaces/cartpole/runs/run-a")
        self.folder("apps/runner/appdata")
        plan = migration.plan(self.root)
        self.assertEqual(len(plan), 2)
        self.assertEqual(migration.expected(self.root), (1, 1))
        self.assertTrue(source.is_dir())
        self.assertFalse((self.root / "data").exists())

    def test_renames_the_whole_workspace_root_and_keeps_tag_sets(self):
        source = self.folder("apps/runner/workspaces/cartpole/runs/run-a")
        (source / "metrics.jsonl").write_bytes(b"metrics")
        self.folder("apps/runner/workspaces/atari/runs/run-b")
        old_state = self.folder("apps/runner/appdata")
        (old_state / "last_workspace.txt").write_bytes(b"state")
        tag_sets = self.root / "apps/runner/workspaces/metrics_viewer_tag_sets.json"
        tag_sets.write_bytes(b"user tag sets")
        plan = migration.plan(self.root)
        before = migration.expected(self.root)
        migration.apply(plan, self.root, before)
        self.assertEqual(migration.counts(self.root / "data/workspaces"), (2, 2))
        self.assertEqual((self.root / "data/workspaces" / tag_sets.name).read_bytes(), b"user tag sets")
        self.assertEqual((self.root / "data/workspaces/cartpole/runs/run-a/metrics.jsonl").read_bytes(), b"metrics")
        self.assertEqual((self.root / "data/appdata/last_workspace.txt").read_bytes(), b"state")
        self.assertFalse(source.exists())

    def test_detects_existing_destination_before_any_rename(self):
        first = self.folder("apps/runner/workspaces/a/runs/run-a")
        colliding = self.folder("apps/runner/workspaces/b/runs/run-b")
        config = self.folder("data/workspaces/b/config") / "_main.txt"
        config.write_bytes(b"shipped")
        with self.assertRaises(FileExistsError):
            migration.plan(self.root)
        self.assertTrue(first.is_dir())
        self.assertTrue(colliding.is_dir())
        self.assertFalse((self.root / "data/workspaces/a").exists())

    def test_merges_disjoint_workspaces_by_directory_rename(self):
        self.folder("apps/runner/workspaces/first/runs/run-a")
        self.folder("data/workspaces/second/runs/run-b")
        plan = migration.plan(self.root)
        migration.apply(plan, self.root, migration.expected(self.root))
        self.assertEqual(migration.counts(self.root / "data/workspaces"), (2, 2))

    def test_stops_on_a_tag_sets_file_with_an_existing_destination_root(self):
        workspaces = self.folder("apps/runner/workspaces")
        (workspaces / "metrics_viewer_tag_sets.json").write_bytes(b"user tag sets")
        self.folder("data/workspaces")
        with self.assertRaisesRegex(ValueError, "Cannot move a loose workspace file"):
            migration.plan(self.root)
        self.assertTrue((workspaces / "metrics_viewer_tag_sets.json").is_file())



if __name__ == "__main__":
    unittest.main()
