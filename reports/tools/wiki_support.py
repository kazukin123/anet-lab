"""wikiと台帳で使う機械処理。本文の意味判断はskillを実行するエージェントが行う。"""
from pathlib import Path
import json
import re


class CategoryRules:
    def __init__(self, root):
        # 分類定義を定義順で保持し、統計ツールの独自規則とは分ける。
        config = json.loads((Path(root)/'reports/tools/categories.json').read_text(encoding='utf-8'))
        self.categories = config['categories']
        self.patterns = [(c['key'], [re.compile(p) for p in c['match']]) for c in self.categories]

    def category_of(self, path):
        # 最初に当たるカテゴリだけを返す。
        path = re.sub(r"_test(?=\.[^/.]+$)", "", path.replace("\\", "/"))
        for key, patterns in self.patterns:
            if any(p.search(path) for p in patterns):
                return key
        return None


def application_paths(key):
    # アプリケーションのページは分類keyより広い置き場全体を持つ。
    return {'110_runner_app': ('apps/runner/',),
            '120_metrics_viewer': ('apps/metrics-viewer/', 'viewers/metrics-viewer/')}[key]

# 材料の集合はPRD095の契約どおりに限定する。英訳と派生ビューは含めない。
MATERIAL_GLOBS = ('docs/experiments/**/*.md', 'reports/*.md',
                  'docs/memo/**/*_10prd.md', 'docs/design/*.jp.md', 'docs/adr/*.md')


def git(root, *args):
    # ファイル名を壊さないよう、呼び出し側で必要なときはNUL区切りを使う。
    import subprocess
    return subprocess.check_output(['git', '-c', 'core.quotepath=false', *args],
                                   cwd=root, encoding='utf-8')


def material_paths(root):
    # 集合は作業ツリーを含む。reports直下でも索引は正本の材料にしない。
    root = Path(root)
    return sorted({p.relative_to(root).as_posix() for pattern in MATERIAL_GLOBS
                   for p in root.glob(pattern) if p.is_file() and p.relative_to(root).as_posix() != 'reports/INDEX.md'})


def page_paths(root):
    return sorted(p for folder in ('subjects', 'mechanisms', 'apps', 'categories')
                  for p in (Path(root)/'docs/wiki'/folder).glob('*.md'))


def changed_paths(root, revision, prefixes):
    # revision対作業ツリーの差に未追跡を加える。
    tracked = git(root, 'diff', '--name-only', '-z', revision, '--', *prefixes).split('\0')
    untracked = git(root, 'ls-files', '--others', '--exclude-standard', '-z', '--', *prefixes).split('\0')
    return set(tracked + untracked) - {''}


def log_state(root):
    # fixは全文を読んだ証拠ではない。ingest/writebackだけを採用する。
    text = (Path(root)/'docs/wiki/log.md').read_text(encoding='utf-8-sig')
    start = re.search(r'^\d{4}-\d{2}-\d{2} start rev:([0-9a-f]+)', text, re.M)
    if not start:
        raise ValueError('docs/wiki/log.md: expected a start revision')
    lint = re.findall(r'^\d{4}-\d{2}-\d{2} lint rev:([0-9a-f]+)', text, re.M)
    records = {}
    for match in re.finditer(r'^\d{4}-\d{2}-\d{2} (?:ingest|writeback) (.+?) blob:([0-9a-f]{40})', text, re.M):
        records[match[1]] = match[2]
    return start[1], lint[-1] if lint else start[1], records


def broken_links(root):
    # コード例の記法と外部URLは除き、正本へのMarkdownリンクを検査する。
    from urllib.parse import unquote
    root = Path(root).resolve()
    sources = material_paths(root)
    result = []
    for page in page_paths(root):
        text = re.sub(r'```[^\n]*\n[\s\S]*?```', '', page.read_text(encoding='utf-8-sig'))
        text = re.sub(r'`[^`\n]+`', '', text)
        # 機械一覧は再生成で修復する。出典リンクの前提条件には数えない。
        text = re.sub(r'(?ms)^## (?:ファイル|PRD の一覧|台帳の指摘)\r?\n.*?(?=^## |\Z)', '', text)
        for line in text.splitlines():
            for match in re.finditer(r'\]\(([^)]+)\)', line):
                target = match[1]
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                    continue
                path = unquote(target.split('#', 1)[0])
                if (page.parent/path).exists():
                    continue
                name = Path(path).name
                candidates = [p for p in sources if Path(p).name == name]
                if not candidates:
                    slug = re.sub(r'^\d+_', '', name)
                    candidates = [p for p in sources if re.sub(r'^\d+_', '', Path(p).name) == slug]
                result.append(dict(page=page.relative_to(root).as_posix(), target=target,
                                   candidates=candidates, annotated='（出典のリンク切れ）' in line))
    return result


def scan(root):
    # 開始revision以後の材料だけをblobで比較し、点検の自己出力を候補にしない。
    root = Path(root).resolve()
    start, checkpoint, records = log_state(root)
    changed = changed_paths(root, start, ['docs/experiments', 'reports', 'docs/memo', 'docs/design', 'docs/adr'])
    known_blobs = set(records.values())
    pending = []
    moved = []
    for path in material_paths(root):
        if path not in changed:
            continue
        blob = git(root, 'hash-object', '--', path).strip()
        if records.get(path) == blob:
            continue
        if blob in known_blobs and not path.startswith('docs/memo/done/'):
            moved.append(dict(path=path, blob=blob))
            continue
        pending.append(dict(path=path, blob=blob, state='更新あり' if path in records else '未取り込み'))
    pending.sort(key=lambda row: (row['state'] != '更新あり', row['path']))
    changed_pages = changed_paths(root, checkpoint, ['docs/wiki'])
    pages = [p.relative_to(root).as_posix() for p in page_paths(root)]
    index = (root/'docs/wiki/index.md').read_text(encoding='utf-8-sig')
    unindexed = [p for p in pages if p.removeprefix('docs/wiki/') not in index]
    links = broken_links(root)
    return dict(start=start, checkpoint=checkpoint, pending=pending, moved=moved,
                changed_pages=sorted(set(pages) & changed_pages), broken_links=links,
                unindexed=unindexed,
                eligible=bool(pending or any(not r['annotated'] for r in links)))


def replace_section(text, title, body):
    # 人が書く節と台帳の区画はそのまま保持する。
    pattern = r'(?ms)(^## ' + re.escape(title) + r'\r?\n).*?(?=^## |\Z)'
    if not re.search(pattern, text):
        raise ValueError('Missing wiki section: '+title)
    return re.sub(pattern, lambda m: m[1]+'\n'+body.rstrip()+'\n\n', text, count=1)


def map_files(root, key, folder, rules):
    # ファイル集合はignoredを除く追跡・未追跡の全体で、統計のexcludedは適用しない。
    files = sorted(set(git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0'))-{''})
    if folder == 'apps':
        return [p for p in files if any(p.startswith(prefix) for prefix in application_paths(key))]
    return [p for p in files if rules.category_of(p) == key]


def file_rows(root, page, files, rules, app_key=None):
    # 異なるディレクトリは混ぜず、同名hpp/cppと対応テストだけを1行にまとめる。
    import os
    buckets = {}
    for path in files:
        bucket = rules.category_of(path) if app_key else None
        stem = re.sub(r'_test$', '', Path(path).stem)
        group = str(Path(path).parent/stem)
        buckets.setdefault(bucket, {}).setdefault(group, []).append(path)
    labels = {c['key']: c['label'] for c in rules.categories}
    order = ([app_key] if app_key else [None]) + [c['key'] for c in rules.categories if c['key'] != app_key]
    if app_key:
        order.append(None)
    lines = ['点検が作り直す。手で直さない。', '']
    for bucket in order:
        if bucket not in buckets:
            continue
        if app_key and bucket != app_key:
            lines.extend(['### '+labels.get(bucket, '機能カテゴリなし'), ''])
        for group, paths in sorted(buckets[bucket].items()):
            links = [f'[{Path(path).name}]({Path(os.path.relpath(Path(root)/path, page.parent)).as_posix()})'
                     for path in sorted(paths, key=lambda p: ('_test' in Path(p).stem, p))]
            lines.append('- '+' / '.join(links))
        lines.append('')
    return '\n'.join(lines)


def prd_rows(root, page, files):
    # コミットscopeの複数PRD番号も拾う。コマンド長を抑えて履歴を読む。
    import os
    numbers = set()
    for offset in range(0, len(files), 80):
        subjects = git(root, 'log', '--format=%s', '--', *files[offset:offset+80])
        for line in subjects.splitlines():
            match = re.match(r'^[^()]+\(([^)]+)\):', line)
            if match:
                numbers.update(re.findall(r'PRD(\d{3})', match[1]))
    rows = ['点検が作り直す。手で直さない。', '']
    unresolved = []
    for number in sorted(numbers):
        matches = sorted((Path(root)/'docs/memo').rglob(number+'_*_10prd.md'))
        if len(matches) != 1:
            # 共有番号と消えた番号はリンク無しで残し、点検側へ報告する。
            rows.append(f'- PRD{number} — （PRD を一意に決められない）')
            unresolved.append(dict(prd='PRD'+number, matches=len(matches)))
            continue
        source = matches[0]
        title = next((line.lstrip('# ').strip() for line in source.read_text(encoding='utf-8-sig').splitlines()
                      if line.startswith('# ')), 'PRD'+number)
        target = Path(os.path.relpath(source, page.parent)).as_posix()
        rows.append(f'- [PRD{number}]({target}) — {title}')
    return ('\n'.join(rows) if numbers else '\n'.join(rows)+'なし'), unresolved


def regenerate_maps(root):
    # 全ページの生成結果を揃えてから、機械的な2節だけを書き戻す。
    root = Path(root).resolve()
    rules = CategoryRules(root)
    updates = []
    unresolved_prds = []
    for folder in ['apps', 'categories']:
        for page in sorted((root/'docs/wiki'/folder).glob('*.md')):
            files = map_files(root, page.stem, folder, rules)
            text = page.read_bytes().decode('utf-8')
            result = replace_section(text, 'ファイル', file_rows(root, page, files, rules,
                                                                   page.stem if folder == 'apps' else None))
            prds, unresolved = prd_rows(root, page, files)
            result = replace_section(result, 'PRD の一覧', prds)
            unresolved_prds.extend(dict(page=page.relative_to(root).as_posix(), **row)
                                   for row in unresolved)
            if result != text:
                updates.append((page, result))
    for page, text in updates:
        page.write_bytes(text.encode('utf-8'))
    return dict(updated=[p.relative_to(root).as_posix() for p, _ in updates],
                unresolved_prds=unresolved_prds)

def main():
    # skill内部の読み取り補助。利用者向けの起動入口は2つのskillに保つ。
    import argparse
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Internal mechanical support for wiki skills')
    parser.add_argument('action', choices=['scan', 'maps'])
    args = parser.parse_args()
    try:
        print(json.dumps(scan(Path.cwd()) if args.action == 'scan' else regenerate_maps(Path.cwd()), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()