#!/usr/bin/env python3

import json
import subprocess
import sys
import unittest
from pathlib import Path

from search_guard import blocked

SCRIPT = Path(__file__).with_name('search_guard.py')

# 止めるコマンドと、そのとき返す起点。
STOPPED = [
    # 2026-11-14 に 4 時間以上残った grep と、同じ呼び出しの find（Git Bash の / から）。
    ('cd C:/dev/anet-lab; grep -rn "wxID_HIGHEST *=" --include=defs.h / 2>/dev/null | head -2 || true; '
     'find / -maxdepth 6 -name defs.h -path "*wx*" 2>/dev/null | head -3', '/'),
    ('find / -maxdepth 6 -type d -name mermaid -path "*node_modules*" 2>/dev/null | head -3', '/'),
    ('find /c -maxdepth 6 -type d -name runs', '/c'),
    ('rg -l wxID_HIGHEST /c', '/c'),
    ('find ~ -name "*.jsonl"', '~'),
    ('timeout 30 grep -r foo /', '/'),
    # PowerShell の書き方。
    ('Get-ChildItem C:\\ -Recurse -Filter defs.h', 'C:\\'),
    ('Get-ChildItem -Path C:\\ -Recurse -Filter defs.h', 'C:\\'),
    ('gci "C:\\" -r -Include *.h', 'C:\\'),
    ('dir C:\\ /s', 'C:\\'),
]

# 通すコマンド。
PASSED = [
    # repo の中や、場所を絞った検索。
    'grep -n foo /c/dev/anet-lab/AGENTS.md',
    'find . -name "*.md"',
    'grep -rl x ~/.claude/projects',
    'Get-ChildItem -Recurse -Filter *.md docs',
    # 探す文字列が / のもの。
    'grep -rn "/" src',
    'rg "path: /" docs',
    'grep -rnE "\'/\'|\\"/\\"" src/apps/metrics-viewer/src/main/resources/static/*.js',
    # 引用符の中の \| と \" で区切らない。
    'grep -rn "StampFly\\.pid\\|\\.pid\\." . ; grep -rn "PID \\|Filter " *.cpp',
    'grep -rn "paused\\"\\|resumed\\"\\|paused automatically" src/apps/runner/src/*.cpp | head',
    # 行の継続。
    'grep -rnE "TERA|PH1" \\\n  --include="*.md" README_kit.md kit_policy.md',
    # 引用符が閉じていないとき、空白で分けた切れ端の \ を / と見なさない。
    'grep -rn "PID \\ Filter src',
    # 再帰でない・1 段まで。
    'ls /',
    'ls -r /',
    'find / -maxdepth 0 >/dev/null',
    # ヒアドキュメントの中身は見ない。
    "cat > x.py << 'EOF'\nimport os\nos.system('find / -name x')\nEOF\npython x.py",
]


class SearchGuardTest(unittest.TestCase):
    def run_hook(self, payload):
        r = subprocess.run([sys.executable, str(SCRIPT)], input=payload, capture_output=True)
        return r.returncode, r.stdout.decode('utf-8')

    def test_stops_recursive_search_from_disk_root(self):
        for command, target in STOPPED:
            with self.subTest(command=command):
                self.assertEqual(blocked(command), target)

    def test_passes_narrowed_or_non_recursive_commands(self):
        for command in PASSED:
            with self.subTest(command=command):
                self.assertIsNone(blocked(command))

    def test_hook_denies_with_reason(self):
        payload = {'tool_name': 'Bash', 'tool_input': {'command': 'find / -name defs.h'}}
        code, out = self.run_hook(json.dumps(payload).encode('utf-8'))
        self.assertEqual(code, 0)
        output = json.loads(out)['hookSpecificOutput']
        self.assertEqual(output['hookEventName'], 'PreToolUse')
        self.assertEqual(output['permissionDecision'], 'deny')
        self.assertIn('（/）', output['permissionDecisionReason'])

    def test_hook_says_nothing_when_it_passes_or_cannot_tell(self):
        # 通すコマンド。
        ok = {'tool_name': 'Bash', 'tool_input': {'command': 'git status --short'}}
        self.assertEqual(self.run_hook(json.dumps(ok).encode('utf-8')), (0, ''))
        # command の無い入力と、JSON でない入力。
        self.assertEqual(self.run_hook(json.dumps({'tool_name': 'Bash'}).encode('utf-8')), (0, ''))
        self.assertEqual(self.run_hook(b'not json'), (0, ''))


if __name__ == '__main__':
    unittest.main()
