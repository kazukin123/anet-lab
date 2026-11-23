#!/usr/bin/env python3

import subprocess
import tempfile
import unittest
from pathlib import Path

from stage_issue_memos import FrontMatterError, merge_memos, other_changes, plan, show, stage

BASE = '''---
id: issue-20260101-abcdef
status: 調査中
priority: 中
resolution:
tags: []
---

## 背景

元の本文
'''


def edit(text, *pairs):
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new)
    return text


class MergeMemosTest(unittest.TestCase):
    def test_moves_only_memo_keys(self):
        work = edit(BASE, ('status: 調査中', 'status: 対応中'), ('priority: 中', 'priority: 高'),
                    ('tags: []\n', 'tags:\n  - NGU関連\n'), ('元の本文', '書き足した本文'))
        merged, changed = merge_memos(BASE, work)
        self.assertEqual(changed, ['priority', 'tags'])
        self.assertEqual(merged, edit(BASE, ('priority: 中', 'priority: 高'), ('tags: []\n', 'tags:\n  - NGU関連\n')))
        self.assertEqual(other_changes(BASE, work), ['status', '本文'])

    def test_no_memo_change_keeps_base(self):
        work = edit(BASE, ('status: 調査中', 'status: 対応中'))
        self.assertEqual(merge_memos(BASE, work), (BASE, []))

    def test_inserts_missing_key_after_nearest_previous_key(self):
        work = edit(BASE, ('resolution:\n', 'resolution:\nmemo: あとで見る\n'))
        merged, changed = merge_memos(BASE, work)
        self.assertEqual(changed, ['memo'])
        self.assertEqual(merged, work)

    def test_removes_key_removed_in_work(self):
        work = edit(BASE, ('priority: 中\n', ''))
        merged, changed = merge_memos(BASE, work)
        self.assertEqual(changed, ['priority'])
        self.assertEqual(merged, work)

    def test_keeps_line_endings_of_base(self):
        base = BASE.replace('\n', '\r\n')
        work = edit(BASE, ('tags: []\n', 'tags:\n  - RND\n'))
        merged, _ = merge_memos(base, work)
        self.assertEqual(merged, edit(BASE, ('tags: []\n', 'tags:\n  - RND\n')).replace('\n', '\r\n'))

    def test_rejects_missing_or_unclosed_front_matter(self):
        with self.assertRaises(FrontMatterError):
            merge_memos('## 本文だけ\n', BASE)
        with self.assertRaises(FrontMatterError):
            merge_memos('---\npriority: 中\n', BASE)

    def test_rejects_duplicated_key(self):
        with self.assertRaises(FrontMatterError):
            merge_memos(BASE, edit(BASE, ('priority: 中\n', 'priority: 中\npriority: 高\n')))

    def test_show_values(self):
        self.assertEqual(show('tags:\n  - NGU関連\n  - RND\n'), '[NGU関連, RND]')
        self.assertEqual(show('priority: 高\n'), '高')
        self.assertEqual(show('memo:\n'), '（空）')
        self.assertEqual(show(None), '（無し）')


class GitIndexTest(unittest.TestCase):
    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, capture_output=True, check=True).stdout

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'core.autocrlf', 'false')
        folder = self.root / 'docs/management/issues'
        folder.mkdir(parents=True)
        self.memo = 'docs/management/issues/issue-20260101-abcdef.md'
        self.status = 'docs/management/issues/issue-20260101-123456.md'
        (self.root / self.memo).write_bytes(BASE.encode('utf-8'))
        (self.root / self.status).write_bytes(BASE.replace('abcdef', '123456').encode('utf-8'))
        self.git('add', '--', self.memo, self.status)

    def tearDown(self):
        self.temp.cleanup()

    def test_stages_memo_keys_and_leaves_the_rest_in_work_tree(self):
        work = edit(BASE, ('status: 調査中', 'status: 対応中'), ('priority: 中', 'priority: 高'))
        (self.root / self.memo).write_bytes(work.encode('utf-8'))
        status_work = edit(BASE.replace('abcdef', '123456'), ('status: 調査中', 'status: 対応中'))
        (self.root / self.status).write_bytes(status_work.encode('utf-8'))
        status_index = self.git('show', ':' + self.status)

        entries = plan(self.root)
        staged = stage(self.root, entries)

        self.assertEqual([entry.path for entry in staged], [self.memo])
        self.assertEqual(self.git('show', ':' + self.memo).decode('utf-8'), edit(BASE, ('priority: 中', 'priority: 高')))
        self.assertEqual(self.git('show', ':' + self.status), status_index)
        self.assertEqual((self.root / self.memo).read_bytes().decode('utf-8'), work)
        self.assertEqual(sorted(self.git('diff', '--name-only').decode('utf-8').split()), sorted([self.memo, self.status]))

    def test_restricts_to_given_ids(self):
        work = edit(BASE, ('priority: 中', 'priority: 高'))
        (self.root / self.memo).write_bytes(work.encode('utf-8'))
        self.assertEqual([entry.path for entry in plan(self.root, ['123456'])], [])
        self.assertEqual([entry.path for entry in plan(self.root, ['abcdef'])], [self.memo])

    def test_broken_issue_stops_before_staging(self):
        (self.root / self.memo).write_bytes(edit(BASE, ('priority: 中', 'priority: 高')).encode('utf-8'))
        (self.root / self.status).write_bytes(b'## frontmatter \xe3\x81\xaa\xe3\x81\x97\n')
        with self.assertRaises(FrontMatterError):
            plan(self.root)
        self.assertEqual(self.git('show', ':' + self.memo).decode('utf-8'), BASE)


if __name__ == '__main__':
    unittest.main()
