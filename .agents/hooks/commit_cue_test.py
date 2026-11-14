#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from commit_cue import STATE_DIR, cue_text

SCRIPT = Path(__file__).with_name('commit_cue.py')
SESSION = 'session-1'


class CommitCueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git('init', '-q')
        self.commit('最初')

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        r = subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid',
                            '-c', 'commit.gpgsign=false', *args],
                           cwd=self.repo, capture_output=True, check=True)
        return r.stdout.decode('utf-8').strip()

    def commit(self, subject):
        self.git('commit', '-q', '--allow-empty', '-m', subject)
        return self.git('rev-parse', '--short', 'HEAD')

    def run_hook(self, payload):
        r = subprocess.run([sys.executable, str(SCRIPT)], input=json.dumps(payload).encode('utf-8'),
                           capture_output=True)
        return r.returncode, r.stdout.decode('utf-8')

    def test_first_prompt_says_nothing_and_counts_from_there(self):
        self.commit('前からある変更')
        self.assertEqual(cue_text(self.repo, SESSION), '')
        h = self.commit('変更')
        self.assertEqual(cue_text(self.repo, SESSION).splitlines()[:2], [
            '[コミットの知らせ] 前の発言のあとに、このブランチへコミットが 1 件入った（新しい順）:',
            f'- {h} 変更',
        ])

    def test_no_change_says_nothing(self):
        cue_text(self.repo, SESSION)
        self.assertEqual(cue_text(self.repo, SESSION), '')

    def test_lists_up_to_three_new_commits_newest_first(self):
        cue_text(self.repo, SESSION)
        hashes = [self.commit(f'変更{i}') for i in range(1, 5)]
        self.assertEqual(cue_text(self.repo, SESSION).splitlines(), [
            '[コミットの知らせ] 前の発言のあとに、このブランチへコミットが 4 件入った（新しい順）:',
            f'- {hashes[3]} 変更4',
            f'- {hashes[2]} 変更3',
            f'- {hashes[1]} 変更2',
            '- ほか 1 件',
            '応答で触れなくてよい。覚えているコミットの状態は古い。言うときは git status・git log を見直す。',
        ])
        self.assertEqual(cue_text(self.repo, SESSION), '')

    def test_one_new_commit_has_no_remainder_line(self):
        cue_text(self.repo, SESSION)
        h = self.commit('変更')
        self.assertEqual(cue_text(self.repo, SESSION).splitlines()[1:], [
            f'- {h} 変更',
            '応答で触れなくてよい。覚えているコミットの状態は古い。言うときは git status・git log を見直す。',
        ])

    def test_sessions_are_independent(self):
        cue_text(self.repo, SESSION)
        self.commit('変更')
        self.assertEqual(cue_text(self.repo, 'session-2'), '')
        self.assertNotEqual(cue_text(self.repo, SESSION), '')

    def test_reset_back_says_nothing(self):
        first = self.git('rev-parse', 'HEAD')
        self.commit('変更')
        cue_text(self.repo, SESSION)
        self.git('reset', '-q', '--hard', first)
        self.assertEqual(cue_text(self.repo, SESSION), '')
        self.commit('次の変更')
        self.assertIn('次の変更', cue_text(self.repo, SESSION))

    def test_hook_prints_context_as_json(self):
        payload = {'session_id': SESSION, 'cwd': str(self.repo), 'hook_event_name': 'UserPromptSubmit', 'prompt': 'x'}
        self.assertEqual(self.run_hook(payload), (0, ''))
        self.commit('変更')
        code, out = self.run_hook(payload)
        self.assertEqual(code, 0)
        output = json.loads(out)['hookSpecificOutput']
        self.assertEqual(output['hookEventName'], 'UserPromptSubmit')
        self.assertIn('変更', output['additionalContext'])

    def test_hook_says_nothing_when_it_cannot_tell(self):
        # git のリポジトリでない場所。
        with tempfile.TemporaryDirectory() as plain:
            self.assertEqual(self.run_hook({'session_id': SESSION, 'cwd': plain}), (0, ''))
        # パスになりうる session_id。
        self.assertEqual(self.run_hook({'session_id': '../x', 'cwd': str(self.repo)}), (0, ''))
        # 前に見た HEAD が今のリポジトリに無い（git の失敗）。記録を外から作って起こす。
        (self.repo / '.git' / STATE_DIR).mkdir()
        (self.repo / '.git' / STATE_DIR / SESSION).write_text('0' * 40, encoding='utf-8')
        self.assertEqual(self.run_hook({'session_id': SESSION, 'cwd': str(self.repo)}), (0, ''))


if __name__ == '__main__':
    unittest.main()
