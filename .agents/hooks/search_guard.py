#!/usr/bin/env python3
"""ディスク全体を起点にした再帰の検索を、実行の前に止める hook（PreToolUse）。

Claude Code（.claude/settings.json）と Codex（.codex/hooks.json）の両方から呼ぶ。stdin の JSON の
tool_input.command を、パイプ・;・&& で区切り（引用符の中は区切らない）、再帰の検索（grep -r・rg・find・
du・tree・ls -R・Get-ChildItem -Recurse・dir /s・where /r）の起点が `/`・ドライブ直下・ホームなどなら、
permissionDecision: deny と理由を出す。止めないときと、読めない入力や失敗のときは何も出さない
（hook の失敗でコマンドを止めない）。
"""
import json
import re
import shlex
import sys

# 起点にしてはいけない場所（小文字、区切りは /、末尾の / を外した形）。
ROOTS = {
    '/', '~', '$home', '${home}', '/home',
    '/c', '/d', '/e', '/mnt', '/mnt/c', '/mnt/d', '/cygdrive/c', '/cygdrive/d',
    'c:', 'd:', 'e:', '/c/users', 'c:/users',
}
# PowerShell の書き方で見る形（ドライブ直下とホーム）。
DRIVE = re.compile(r'^(?:[A-Za-z]:[\\/]*|\$env:(?:userprofile|homepath|systemdrive)[\\/]*)$', re.I)
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2[ \t]*(?=\n|$)", re.S)
OPERATOR = set('|&;()<>')
PREFIX = ('timeout', 'nice', 'command', 'exec', 'time')
REASON = ('ディスク全体（{target}）を起点にした再帰の検索は止めた。終わらないことがあり、'
          'ツールの打ち切りや停止のあとも検索のプロセスが残って、端末全体が重くなる。'
          '探す場所を絞る（AGENTS.md の「AI エージェントの作業ルール」の「探すとき」）。')


def is_root(arg, posix):
    raw = arg.strip('"\'')
    if not posix:
        # PowerShell の書き方は \ を区切りとして残すぶん、引用符の中の \" を読み違えやすいので、ドライブの形だけを見る
        return bool(DRIVE.match(raw))
    # \ だけの語は、引用符が閉じずに空白で分けたときの切れ端（`grep -rn "PID \ Filter src` の \ など）で、
    # ディスクの / ではない。/ に置き換える前に外す
    if raw and set(raw) <= {'\\'}:
        return False
    # 区切りを / に、大文字を小文字にそろえ、末尾の / を外して比べる（/ だけは / のまま残す）
    a = raw.replace('\\', '/').lower()
    a = '/' if a and set(a) == {'/'} else a.rstrip('/')
    return a in ROOTS


def segments(command, posix):
    """コマンドを、パイプ・;・&& などで区切った単語の並びに分ける。ヒアドキュメントの中身は見ない。"""
    # ヒアドキュメントの中身はコマンドではない（書き出すスクリプトの本文など）ので消す。行の継続（\ と改行）は
    # 1 つのコマンドの続きなので空白に、残る改行はコマンドの区切りなので ; に置き換える
    text = HEREDOC.sub(' ', command).replace('\r', '').replace('\\\n', ' ').replace('\n', ' ; ')
    # 引用符を守って単語に分ける。| ; & ( ) < > は単語と分けて取り出す
    lex = shlex.shlex(text, posix=posix, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        toks = list(lex)
    except ValueError:
        # 引用符が閉じていない（途中で切れたコマンドなど）ときは、空白で分けるだけにして読み続ける
        toks = text.split()
    # 区切りの記号の並び（| や && や 2> の >）で単語の並びを切る
    seg = []
    for t in toks:
        if t and set(t) <= OPERATOR:
            if seg:
                yield seg
            seg = []
        else:
            seg.append(t)
    if seg:
        yield seg


def non_options(args):
    return [a for a in args if not a.startswith('-')]


def pattern_given(args, no_pattern_options=()):
    """探す文字列をオプションで渡しているか（-e・-f のほか、探す文字列を取らない no_pattern_options を含む）。

    渡していれば、オプションでない語は全部が起点。渡していなければ、オプションでない最初の語は探す文字列。
    """
    return any(a in ('-e', '-f', *no_pattern_options) or a.startswith(('--regexp', '--file')) for a in args)


def targets(words):
    """再帰の検索なら、起点の引数の並びを返す。検索でなければ None。"""
    # パスと .exe を外したコマンド名で見分ける（/usr/bin/grep・grep.exe も grep）
    cmd = words[0].strip('"\'').lower().replace('\\', '/').rsplit('/', 1)[-1].removesuffix('.exe')
    args = words[1:]
    if cmd in ('grep', 'egrep', 'fgrep'):
        if not any(a in ('--recursive', '--dereference-recursive') or
                   (re.fullmatch(r'-[A-Za-z]+', a) and ('r' in a or 'R' in a)) for a in args):
            return None
        rest = non_options(args)
        return rest if pattern_given(args) else rest[1:]  # 先頭は探す文字列
    if cmd == 'rg':
        rest = non_options(args)
        return rest if pattern_given(args, no_pattern_options=('--files',)) else rest[1:]
    if cmd == 'find':
        if '-maxdepth' in args:
            i = args.index('-maxdepth')
            if i + 1 < len(args) and args[i + 1].isdigit() and int(args[i + 1]) <= 1:
                return None  # ls と同じ深さ
        # find の起点は、式（-name などの - で始まる語、( や !）より前に並ぶ語
        out = []
        for a in args:
            if a.startswith('-') or a in ('(', '!', '\\('):
                break
            out.append(a)
        return out
    if cmd in ('du', 'tree'):
        return non_options(args)
    if cmd == 'ls':
        recursive = any(re.fullmatch(r'-[A-Za-z]*R[A-Za-z]*', a) or a.lower() == '-recurse' for a in args)
        return non_options(args) if recursive else None
    if cmd in ('get-childitem', 'gci', 'dir'):
        if not any(a.lower() in ('-recurse', '-r', '/s') for a in args):
            return None
        # -Filter などは次の語が値（defs.h など）なので起点から外す。-Path は名前を付けただけなので、次の語を起点に残す
        out, skip = [], False
        for a in args:
            if skip:
                skip = False
            elif a.lower() in ('-filter', '-include', '-exclude', '-depth', '-name'):
                skip = True
            elif not a.startswith('-') and a.lower() != '/s':
                out.append(a)
        return out
    if cmd == 'where':
        lower = [a.lower() for a in args]
        if '/r' in lower:
            # where /r <起点> <名前> の形なので、/r の次の 1 語が起点
            i = lower.index('/r')
            return args[i + 1:i + 2]
    return None


def blocked(command):
    """止める起点を返す。止めなければ None。bash の書き方と PowerShell の書き方の両方で読む。"""
    for posix in (True, False):
        for words in segments(command, posix):
            # `timeout 30 grep ...` のような前置きを、そのオプションと秒数ごと外す
            while words and words[0] in PREFIX:
                words = words[1:]
                while words and (words[0].startswith('-') or re.fullmatch(r'\d+(\.\d+)?[smh]?', words[0])):
                    words = words[1:]
            if not words:
                continue
            for a in targets(words) or []:
                if is_root(a, posix):
                    return a.strip('"\'')
    return None


def main():
    # 入力を読んで判定する。例外はすべて捨てて何も出さない（hook の失敗でコマンドを止めない）
    try:
        data = json.loads(sys.stdin.buffer.read().decode('utf-8'))
        command = (data.get('tool_input') or {}).get('command')
        target = blocked(command) if isinstance(command, str) else None
    except Exception:
        return
    # 止めるときだけ deny と理由を出す。理由は AI に返り、場所を絞って出し直す手がかりになる
    if target is not None:
        print(json.dumps({'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': REASON.format(target=target),
        }}, ensure_ascii=True))


if __name__ == '__main__':
    main()
