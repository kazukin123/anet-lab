#!/usr/bin/env python3
"""発言の前に、前の発言のあとに入ったコミットを AI に知らせる hook（UserPromptSubmit）。

commit は人間が AI の知らないところで行うので、AI の覚えている「まだコミットしていない」はすぐ古くなる
（AGENTS.md の「コミットの状態を言わない」）。Claude Code（.claude/settings.json）と
Codex（.codex/hooks.json）の両方から呼ぶ。

入力の JSON の session_id ごとに、前に見た HEAD を .git の下（anet-commit-cue/）に覚え、
そこから今の HEAD までに新しいコミットがあれば、件名を新しい順に最大 3 件出す。
そのセッションの初めての発言、変化なし、reset で戻ったとき、失敗したときは何も出さない。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

MAX_SUBJECTS = 3
STATE_DIR = 'anet-commit-cue'
SESSION_ID = re.compile(r'[A-Za-z0-9_.-]{1,128}')


def git(cwd, *args):
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, timeout=5)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode('utf-8', 'replace'))
    return r.stdout.decode('utf-8').strip()


def cue_text(cwd, session_id):
    # 今の HEAD と、このセッションが前の発言で見た HEAD を読む。
    head = git(cwd, 'rev-parse', 'HEAD')
    state = Path(cwd) / git(cwd, 'rev-parse', '--git-path', STATE_DIR) / session_id
    old = state.read_text(encoding='utf-8').strip() if state.exists() else ''

    # 比べる前に今の HEAD を書いておく。このあと git が失敗しても、同じ知らせを次の発言で出し直さない。
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(head, encoding='utf-8')

    # 初めての発言、変化なし、reset で戻った（前の HEAD から辿れる新しいコミットが無い）ときは出さない。
    if not old or old == head:
        return ''
    count = int(git(cwd, 'rev-list', '--count', f'{old}..{head}'))
    if count == 0:
        return ''

    # 新しいコミットの件名を、新しい順に MAX_SUBJECTS 件まで並べる。
    subjects = git(cwd, 'log', f'-{MAX_SUBJECTS}', '--format=%h %s', f'{old}..{head}').splitlines()
    lines = [f'[コミットの知らせ] 前の発言のあとに、このブランチへコミットが {count} 件入った（新しい順）:']
    lines += [f'- {s}' for s in subjects]
    if count > len(subjects):
        lines.append(f'- ほか {count - len(subjects)} 件')
    lines.append('応答で触れなくてよい。覚えているコミットの状態は古い。言うときは git status・git log を見直す。')
    return '\n'.join(lines)


def main():
    # 入力を読んで知らせの文を作る。hook の失敗で発言を止めないよう、ここで起きた例外はすべて捨てて何も出さない。
    try:
        data = json.loads(sys.stdin.buffer.read().decode('utf-8'))
        session_id = str(data.get('session_id', ''))
        if not SESSION_ID.fullmatch(session_id):  # 記録のファイル名に使うので、パスになりうる値は受けない
            return
        text = cue_text(data.get('cwd') or '.', session_id)
    except Exception:
        return
    if not text:
        return

    # Claude Code と Codex が共通に受ける形（hookSpecificOutput.additionalContext）で出す。
    out = {
        'hookSpecificOutput': {
            'hookEventName': data.get('hook_event_name') or 'UserPromptSubmit',
            'additionalContext': text,
        }
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=True))


if __name__ == '__main__':
    main()
