"""PRD095 の台帳CLIを、正本と分離した写しで検証する。"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[4]
CLI = REPO / 'reports/tools/registry.py'


class RegistryCliTest(unittest.TestCase):
    def setUp(self):
        # 台帳と分類だけを写し、CLIのcwdを一時repoへ向ける。
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'reports/registry').mkdir(parents=True)
        (self.root/'reports/tools').mkdir(parents=True)
        shutil.copyfile(REPO/'reports/tools/categories.json', self.root/'reports/tools/categories.json')
        self.records = []
        self.add_record('F-0001', 'apps/runner/tools/probe.py')
        self.add_record('F-0002', 'apps/runner/src/RunnerFrame.cpp')
        self.add_record('F-0003', 'apps/runner/config/common.txt')

    def add_record(self, ident, file, status='open'):
        # CLIに必要な実レコードを保存する。
        self.records.append(dict(id=ident, file=file, lens='refactor', title=ident+' title',
                                 summary='summary', scenario='scenario', priority='P2',
                                 confidence='high', status=status, last_seen='2026-10-22'))
        (self.root/'reports/registry/findings.jsonl').write_text(
            ''.join(json.dumps(r)+'\n' for r in self.records), encoding='utf-8')

    def cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(CLI), *args], cwd=self.root,
                              text=True, encoding='utf-8', capture_output=True)

    def test_category_cli_filters_real_records(self):
        result = self.cli('list', '--category', '311_dev_tools')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('F-0001', result.stdout)
        self.assertNotIn('F-0002', result.stdout)
        self.assertNotIn('F-0003', result.stdout)

    def test_test_suffix_is_removed_for_every_extension(self):
        self.add_record('F-0004', 'core/anet-core/include/anet/nn_test.hpp')
        result = self.cli('list', '--category', '315_nn')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('F-0004', result.stdout)
        self.assertNotIn('F-0001', result.stdout)

    def test_runner_category_excludes_tools_and_config(self):
        result = self.cli('list', '--category', '110_runner_app')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('F-0002', result.stdout)
        self.assertNotIn('F-0001', result.stdout)
        self.assertNotIn('F-0003', result.stdout)

    def test_unknown_category_fails_without_listing_records(self):
        result = self.cli('list', '--category', 'unknown_key')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertIn('unknown_key', result.stderr)
        self.assertIn('311_dev_tools', result.stderr)
        self.assertIn('110_runner_app', result.stderr)

    def map_page(self, folder, key):
        path = self.root/'docs/wiki'/folder/(key+'.md')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# map\n\n## 台帳の指摘\n\n台帳のツールが作り直す。手で直さない。\n\n'
                        '<!-- registry:begin -->\nold\n<!-- registry:end -->\n\nKEEP AFTER\n', encoding='utf-8')
        return path

    def test_build_updates_views_and_only_registry_sections(self):
        runner = self.map_page('apps', '110_runner_app')
        tools = self.map_page('categories', '311_dev_tools')
        before = runner.read_text(encoding='utf-8').split('<!-- registry:begin -->')[0]
        self.add_record('F-0004', 'apps/runner/tools/probe_test.py', 'stale')
        self.add_record('F-0005', 'apps/runner/src/Old.cpp', 'fixed')
        result = self.cli('build')
        self.assertEqual(result.returncode, 0, result.stderr)
        app_text = runner.read_text(encoding='utf-8')
        cat_text = tools.read_text(encoding='utf-8')
        self.assertEqual(app_text.split('<!-- registry:begin -->')[0], before)
        self.assertTrue(app_text.endswith('<!-- registry:end -->\n\nKEEP AFTER\n'))
        for ident in ['F-0001', 'F-0002', 'F-0003', 'F-0004']:
            self.assertIn(ident, app_text)
        for ident in ['F-0001', 'F-0004']:
            self.assertIn(ident, cat_text)
        for ident in ['F-0002', 'F-0003', 'F-0005']:
            self.assertNotIn(ident, cat_text)
        self.assertNotIn('F-0005', app_text)
        self.assertNotIn('summary', app_text)
        self.assertNotIn('scenario', app_text)
        self.assertIn('../../../reports/registry/findings.md', app_text)
        for file in ['findings.md', 'findings.html']:
            self.assertTrue((self.root/'reports/registry'/file).is_file())

    def test_invalid_markers_leave_every_output_unchanged(self):
        self.map_page('apps', '110_runner_app')
        bad = self.map_page('categories', '311_dev_tools')
        for file in ['findings.md', 'findings.html']:
            (self.root/'reports/registry'/file).write_text('KEEP '+file, encoding='utf-8')
        variants = [
            '<!-- registry:begin -->\nmissing end',
            '<!-- registry:begin -->\n<!-- registry:begin -->\n<!-- registry:end -->',
            '<!-- registry:end -->\n<!-- registry:begin -->',
            'inline <!-- registry:begin -->\n<!-- registry:end -->',
        ]
        for markers in variants:
            with self.subTest(markers=markers):
                bad.write_text(markers, encoding='utf-8')
                before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
                result = self.cli('build')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('311_dev_tools.md', result.stderr)
                self.assertEqual({p: p.read_bytes() for p in before}, before)


if __name__ == '__main__':
    unittest.main()
