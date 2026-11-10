#!/usr/bin/env python3
"""課題の管理外のメモ（priority・memo・tags）の変更だけを、欄の単位で index に載せる。

管理外のメモは開発者が Obsidian などで直接書く欄で、AI は書かない（anet-create-issue の「管理外のメモ」）。
同じ課題のファイルに AI の作業の変更（status、本文など）があっても、この 3 つの欄の変更だけを
index に載せれば、開発者の手入れを分けてコミットできる。

引数なしで、載せる変更と、載せずに残すほかの変更を課題ごとに示す。--stage で index に載せる。
作業ツリーは変えない。index の中身を土台にするので、先に stage した変更は保つ。
課題の ID（末尾の 6 桁だけでもよい）を並べると、その課題だけを扱う。
"""
import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

MEMO_KEYS = ('priority', 'memo', 'tags')
ISSUES = 'docs/management/issues/'
ISSUE_FILE = re.compile(r'issue-\d{8}-[0-9a-f]{6}\.md')
KEY_LINE = re.compile(r'([A-Za-z_][A-Za-z0-9_-]*):(?=[ \t\r\n]|$)')


class FrontMatterError(ValueError):
    pass


def split_front_matter(text):
    """先頭の frontmatter を (開きの行, 欄の塊の並び, 閉じの行から後) に分ける。

    欄の塊は (欄の名前, 中身の文字列)。字下げした行・リストの項目・空行は、直前の欄の塊に含める。
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip('\r\n') != '---':
        raise FrontMatterError('frontmatter がありません')
    end = next((i for i in range(1, len(lines)) if lines[i].rstrip('\r\n') == '---'), None)
    if end is None:
        raise FrontMatterError('frontmatter が閉じていません')
    blocks = []
    for line in lines[1:end]:
        match = KEY_LINE.match(line)
        if match:
            blocks.append((match[1], [line]))
        elif blocks:
            blocks[-1][1].append(line)
        else:
            raise FrontMatterError(f'最初の欄より前に行があります: {line.rstrip()!r}')
    names = [name for name, _ in blocks]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise FrontMatterError(f'同じ欄が 2 つあります: {duplicated}')
    return lines[0], [(name, ''.join(block)) for name, block in blocks], ''.join(lines[end:])


def merge_memos(base, work, keys=MEMO_KEYS):
    """base（index の中身）に work（作業ツリーの中身）の keys の欄だけを移した中身と、移した欄の並びを返す。

    欄の書き方（Obsidian がリストを箇条書きに直すなど）は work のまま運び、改行だけ base に合わせる。
    base に無い欄は、work でその欄より前にある欄のうち base にもある最も近い欄の後ろに入れる。
    """
    head, merged, rest = split_front_matter(base)
    _, work_blocks, _ = split_front_matter(work)
    eol = '\r\n' if '\r\n' in base else '\n'
    work_names = [name for name, _ in work_blocks]
    work_map = dict(work_blocks)
    changed = []
    for key in keys:
        names = [name for name, _ in merged]
        old = dict(merged).get(key)
        new = work_map.get(key)
        if new is not None:
            new = new.replace('\r\n', '\n').replace('\n', eol)
        if old == new:
            continue
        changed.append(key)
        if new is None:
            del merged[names.index(key)]
        elif old is not None:
            merged[names.index(key)] = (key, new)
        else:
            before = [name for name in work_names[:work_names.index(key)] if name in names]
            merged.insert(names.index(before[-1]) + 1 if before else 0, (key, new))
    return head + ''.join(block for _, block in merged) + rest, changed


def other_changes(base, work, keys=MEMO_KEYS):
    """keys 以外で base と work が違う欄の名前の並び。本文が違えば最後に「本文」を足す。"""
    _, base_blocks, base_rest = split_front_matter(base)
    _, work_blocks, work_rest = split_front_matter(work)
    plain = lambda text: None if text is None else text.replace('\r\n', '\n')
    base_map, work_map = dict(base_blocks), dict(work_blocks)
    names = [name for name, _ in base_blocks] + [name for name, _ in work_blocks if name not in base_map]
    result = [name for name in names
              if name not in keys and plain(base_map.get(name)) != plain(work_map.get(name))]
    if plain(base_rest) != plain(work_rest):
        result.append('本文')
    return result


def show(block):
    """欄の塊を、一覧で読める 1 行の値にする。"""
    if block is None:
        return '（無し）'
    lines = block.splitlines()
    first = lines[0].split(':', 1)[1].strip()
    items = [line.strip().removeprefix('- ').strip() for line in lines[1:] if line.strip()]
    if items and not first:
        return '[' + ', '.join(items) + ']'
    return ' '.join([first] + items) if first or items else '（空）'


def git(root, *args, data=None):
    return subprocess.run(['git', *args], cwd=root, input=data, capture_output=True, check=True).stdout


@dataclass
class Entry:
    path: str
    base: str
    work: str
    merged: str = ''
    changed: list = field(default_factory=list)
    others: list = field(default_factory=list)


def plan(root, ids=()):
    """index と作業ツリーで違う課題のファイルを読み、載せる中身を作る。1 件でも読めなければ例外で止める。"""
    out = git(root, 'diff', '--name-only', '-z', '--', ISSUES).decode('utf-8')
    paths = [path for path in out.split('\0')
             if path.startswith(ISSUES) and ISSUE_FILE.fullmatch(path[len(ISSUES):])]
    if ids:
        paths = [path for path in paths if any(path[len(ISSUES):-3].endswith(i) for i in ids)]
    entries, errors = [], []
    for path in paths:
        stages = git(root, 'ls-files', '-s', '--', path).decode('utf-8').splitlines()
        if len(stages) != 1:
            errors.append(f'{path}: 衝突の解消が済んでいません')
            continue
        try:
            work = (root / path).read_bytes().decode('utf-8')
        except FileNotFoundError:
            continue  # 消したファイルは管理外のメモの手入れではない。
        entry = Entry(path, git(root, 'show', ':' + path).decode('utf-8'), work)
        try:
            entry.merged, entry.changed = merge_memos(entry.base, entry.work)
            entry.others = other_changes(entry.base, entry.work)
        except FrontMatterError as error:
            errors.append(f'{path}: {error}')
            continue
        entries.append(entry)
    if errors:
        raise FrontMatterError('\n'.join(errors))
    return entries


def stage(root, entries):
    """移した中身を index に載せ、載った中身が作った中身と一致することを確かめる。違えば元に戻して止める。"""
    staged = []
    for entry in entries:
        if not entry.changed:
            continue
        mode, old_sha = git(root, 'ls-files', '-s', '--', entry.path).decode('utf-8').split()[:2]
        data = entry.merged.encode('utf-8')
        sha = git(root, 'hash-object', '-w', '--no-filters', '--stdin', data=data).decode('ascii').strip()
        git(root, 'update-index', '--cacheinfo', f'{mode},{sha},{entry.path}')
        if git(root, 'show', ':' + entry.path) != data:
            git(root, 'update-index', '--cacheinfo', f'{mode},{old_sha},{entry.path}')
            raise RuntimeError(f'{entry.path}: index に載った中身が作った中身と違うので、元に戻しました')
        staged.append(entry)
    return staged


def report(entries):
    for entry in entries:
        if not entry.changed:
            continue
        base_map = dict(split_front_matter(entry.base)[1])
        work_map = dict(split_front_matter(entry.work)[1])
        print(entry.path[len(ISSUES):-3])
        for key in entry.changed:
            print(f'  {key}: {show(base_map.get(key))} → {show(work_map.get(key))}')
        if entry.others:
            print(f'  載せずに残す: {"・".join(entry.others)}')
    rest = [entry for entry in entries if not entry.changed]
    if rest:
        print(f'管理外のメモの変更が無い課題: {len(rest)} 件（' + '、'.join(e.path[len(ISSUES):-3] for e in rest) + '）')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='課題の管理外のメモ（priority・memo・tags）の変更だけを index に載せる')
    parser.add_argument('ids', nargs='*', help='扱う課題の ID（末尾の 6 桁だけでもよい）。省くと全部')
    parser.add_argument('--stage', action='store_true', help='index に載せる。省くと示すだけ')
    args = parser.parse_args()
    root = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel').decode('utf-8').strip())
    try:
        entries = plan(root, args.ids)
    except FrontMatterError as error:
        print(f'読めない課題があるので、何も載せずに止めました。\n{error}', file=sys.stderr)
        return 1
    if not any(entry.changed for entry in entries):
        print('管理外のメモの変更はありません。')
        return 0
    report(entries)
    if args.stage:
        staged = stage(root, entries)
        print(f'index に載せた課題: {len(staged)} 件（作業ツリーは変えていない）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
