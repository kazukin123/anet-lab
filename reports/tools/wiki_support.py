"""wikiとissueの機械処理。本文の意味判断はskillを実行するエージェントが行う。"""
from io import StringIO
from pathlib import Path
import json
import re
from urllib.parse import unquote

import yaml


class CategoryRules:
    def __init__(self, root):
        # 分類定義を定義順で保持し、統計ツールの独自規則とは分ける。
        config = json.loads((Path(root)/'reports/tools/categories.json').read_text(encoding='utf-8'))
        self.categories = config['categories']
        self.patterns = [(c['key'], [re.compile(p) for p in c['match']]) for c in self.categories]

    def category_of(self, path):
        # 最初に当たるカテゴリだけを返す。
        path = path.replace("\\", "/")
        source_path = re.sub(r"_test(?=\.[^/.]+$)", "", path)
        for key, patterns in self.patterns:
            if any(p.search(path) or p.search(source_path) for p in patterns):
                return key
        return None


def application_paths(key):
    # アプリケーションのページは分類keyより広い置き場全体を持つ。
    return {'110_runner_app': ('src/apps/runner/', 'apps/runner/'),
            '120_metrics_viewer': ('src/apps/metrics-viewer/', 'apps/metrics-viewer/')}[key]

# 材料の集合はRFD0095の契約どおりに限定する。英訳と派生ビューは含めない。
MATERIAL_GLOBS = ('docs/experiments/**/*.md', 'reports/*.md',
                  'docs/memo/**/*_10rfd.md', 'docs/design/*.jp.md', 'docs/adr/*.md')


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


SUMMARY_DIR = Path('docs/wiki/summary')
CONCEPTS_DIR = Path('docs/wiki/concepts')
KNOWLEDGE_FOLDERS = ('subjects', 'methods', 'measurement', 'phenomena')
MAP_FOLDERS = ('apps', 'framework')
KNOWLEDGE_HEADINGS = ('今の結論', '分かっていること', '横断的知見', '見立て', '分かっていないこと', '撤回・更新', '関連')
SUMMARY_KINDS = {'experiments': '実験記録', 'reports': '報告', 'rfd': 'RFD', 'design': '設計文書', 'adr': 'ADR'}
CONCEPT_KINDS = {'subjects': '題材', 'methods': '手法', 'measurement': '測り方', 'phenomena': '現象',
                 'apps': 'アプリケーション', 'framework': 'フレームワーク'}
HEADER_KEYS = {'title', 'page_kind', 'material', 'material_kind', 'concepts', 'aliases', 'description', 'updated'}
YAML_MERGE_TAG = 'tag:yaml.org,2002:merge'


def get_summary_path(material):
    # 実験記録は階層を保ち、保管場所が変わる RFD はファイル名で引く。
    path = Path(material)
    if path.parts[:2] == ('docs', 'experiments'):
        return SUMMARY_DIR / 'experiments' / path.relative_to('docs/experiments')
    group = {'reports': 'reports', 'memo': 'rfd', 'design': 'design', 'adr': 'adr'}
    kind = path.parts[1] if path.parts[0] == 'docs' else 'reports'
    return SUMMARY_DIR / group[kind] / path.name


def read_summary_blob(path):
    # 本文に引用された別の blob を拾わないよう、「元の材料」の節だけを見る。
    content = read_page_body(path)
    section = re.search(r'(?ms)^## 元の材料[ \t]*\n(.*?)(?=^## |\Z)', content)
    matches = re.findall(r'\bblob:([0-9a-f]{40})\b', section[1]) if section else []
    return matches[0] if len(matches) == 1 else None


def page_paths(root):
    return sorted(p for folder in KNOWLEDGE_FOLDERS + MAP_FOLDERS
                  for p in (Path(root) / CONCEPTS_DIR / folder).glob('*.md'))


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


def iter_markdown_links(text):
    # フェンスとインラインのコード例を除き、空白のある山括弧パスも読む。
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r'^\s{0,3}(\x60{3,}|~{3,})(.*)$', line)
        if fence:
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        visible = re.sub(r'(\x60+).*?\1', lambda match: ' ' * len(match[0]), line)
        for match in re.finditer(r"""\]\(\s*(?:<([^>\n]+)>|((?:[^()\s]|\([^()\n]*\))+))(?:\s+(?:"[^"]*"|'[^']*'|\([^()]*\)))?\s*\)""", visible):
            yield number, line, match[1] or match[2], match.end()


def iter_local_links(text):
    for number, line, target, _ in iter_markdown_links(text):
        if is_local_link(target):
            yield number, line, target, unquote(target.partition('#')[0])


def is_local_link(target):
    return not target.startswith(('#', '//')) and not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target)


def index_properties(root, index):
    # 目録の別名は末尾の括弧から取り、その前の説明を取り出す。
    entries = {}
    index_dir = root / 'docs/wiki'
    for line in index.splitlines():
        if not line.lstrip().startswith('- ['):
            continue
        for _, _, target, end in iter_markdown_links(line):
            if not is_local_link(target):
                break
            page = (index_dir / unquote(target.partition('#')[0])).resolve()
            if page.is_relative_to(root / CONCEPTS_DIR):
                description = line[end:].partition(' — ')[2]
                prefix, _, tail = description.rpartition('（別名: ')
                aliases = re.match(r'((?:[^（）]|（[^（）]*）)*)）', tail)
                if aliases:
                    entry = (aliases[1].split(', ') if aliases[1] else [], prefix.strip())
                else:
                    entry = ([], description.strip())
                entries.setdefault(page, []).append(entry)
            break
    return entries


def yaml_scalar(value):
    # 課題の書式に合わせ、YAML で文字列として同じ値を読めるなら引用符を省く。
    if value is None or value == '':
        return ''
    if not any(char in value for char in ('"', '\\', ': ', ' #')):
        try:
            if yaml.safe_load(value) == value:
                return value
        except yaml.YAMLError:
            pass
    return json.dumps(value, ensure_ascii=False)


def split_header(text):
    # 本文は元の改行のまま返し、区切りだけを見てヘッダを分ける。
    opening = re.match(r'^---[ \t]*(?:\r?\n|\Z)', text)
    if not opening:
        return '', text
    closing = re.compile(r'^---[ \t]*(?:\r?\n|\Z)', re.M).search(text, opening.end())
    if not closing:
        raise ValueError('unclosed YAML header')
    return text[opening.end():closing.start()], text[closing.end():]


def read_page_body(page):
    # 閉じていないヘッダは scan で別途警告し、既存の本文点検も続ける。
    text = page.read_text(encoding='utf-8-sig')
    try:
        return split_header(text)[1]
    except ValueError:
        return text


def yaml_node_end(events, event):
    # 入れ子の値も、パーサーが示す終端まで飛ばす。
    if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
        for child in events:
            if isinstance(child, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                return child.end_mark
            yaml_node_end(events, child)
    return event.end_mark


def block_key_lines(tokens):
    # 明示キーは「?」とキーの値が別行になり得る。キー自体でなく区切りの位置から取る。
    starts = (yaml.BlockMappingStartToken, yaml.BlockSequenceStartToken,
              yaml.FlowMappingStartToken, yaml.FlowSequenceStartToken)
    ends = (yaml.BlockEndToken, yaml.FlowMappingEndToken, yaml.FlowSequenceEndToken)
    depth = 0
    for token in tokens:
        if isinstance(token, starts):
            depth += 1
        elif isinstance(token, ends):
            depth -= 1
        elif depth == 1 and isinstance(token, yaml.KeyToken):
            yield token.start_mark.line


def yaml_anchor_names(entries, token_type):
    return {token.value for entry in entries for token in yaml.scan(entry) if isinstance(token, token_type)}


def yaml_nodes_by_position(root):
    # 参照は元のノードと同じ実体を持つ。循環参照でも一度だけ調べる。
    nodes = {}
    pending = [root]
    while pending:
        node = pending.pop()
        position = (type(node), node.start_mark.index)
        if position in nodes:
            continue
        nodes[position] = node
        if isinstance(node, yaml.SequenceNode):
            pending.extend(node.value)
        elif isinstance(node, yaml.MappingNode):
            pending.extend(item for pair in node.value for item in pair)
    return nodes


def retained_anchor_entries(header, managed, custom, merges):
    names = yaml_anchor_names(custom, yaml.AliasToken)
    managed_names = yaml_anchor_names(managed, yaml.AnchorToken)
    if names.isdisjoint(managed_names):
        return custom

    # 管理外の断片から参照される定義を、依存する値ごと YAML ノードから出し直す。
    # 管理外の断片にある定義と名前が重ならないよう、依存する値には別の名前を与える。
    original = yaml.compose(header, Loader=yaml.SafeLoader)
    nodes = yaml_nodes_by_position(original)
    moved_names = names & (managed_names | yaml_anchor_names(merges, yaml.AnchorToken))
    node_types = {yaml.ScalarEvent: yaml.ScalarNode, yaml.SequenceStartEvent: yaml.SequenceNode,
                  yaml.MappingStartEvent: yaml.MappingNode}
    anchors = {event.anchor: nodes[node_types[type(event)], event.start_mark.index] for event in yaml.parse(header)
               if type(event) in node_types and event.anchor in moved_names}
    values = yaml.SequenceNode('tag:yaml.org,2002:seq', list(anchors.values()))
    key = yaml.ScalarNode('tag:yaml.org,2002:str', 'title')
    shadow = yaml.MappingNode('tag:yaml.org,2002:map', [(key, values)])

    # 古い値は現在の title で上書きされるマージ値に置く。既存のマージも同じキーにまとめる。
    sources = [shadow]
    for merge_key, value in reversed(original.value):
        if merge_key.tag == YAML_MERGE_TAG:
            sources.extend(value.value if isinstance(value, yaml.SequenceNode) else [value])
    merge_value = shadow if len(sources) == 1 else yaml.SequenceNode('tag:yaml.org,2002:seq', sources)

    prefix = 'wiki_header_'
    original_names = yaml_anchor_names([header], yaml.AnchorToken)
    while any(name.startswith(prefix) for name in original_names):
        prefix += '_'
    stream = StringIO()
    dumper = yaml.SafeDumper(stream, allow_unicode=True)
    dumper.ANCHOR_TEMPLATE = prefix + '%03d'
    dumper.open()
    dumper.anchor_node(merge_value)
    dumper.anchors.update({node: name for name, node in anchors.items()})
    dumper.emit(yaml.DocumentStartEvent())
    dumper.serialize_node(merge_value, None, None)
    dumper.emit(yaml.DocumentEndEvent())
    dumper.close()
    merged = '<<:\n' + ''.join('  ' + line for line in stream.getvalue().splitlines(keepends=True))
    return [merged, *(entry for entry in custom if entry not in merges)]


def custom_header_entries(header):
    # YAML の値を再出力すると引用符やコメントが変わるので、各キーの原文を抜き出す。
    loaded = yaml.safe_load(header)
    if loaded is None:
        return []
    if not isinstance(loaded, dict):
        raise ValueError('YAML header must be a mapping')
    events = iter(yaml.parse(header))
    next(events)  # ストリームの開始
    next(events)  # 文書の開始
    mapping = next(events)
    lines = header.splitlines(keepends=True)
    key_lines = iter(block_key_lines(yaml.scan(header)))
    custom = []
    managed = []
    merges = []
    for key in events:
        if isinstance(key, yaml.MappingEndEvent):
            break
        key_end = yaml_node_end(events, key)
        value = next(events)
        end = yaml_node_end(events, value)
        entries = managed if isinstance(key, yaml.ScalarEvent) and key.value in HEADER_KEYS else custom
        if mapping.flow_style:
            # インラインマッピングのコンマを外し、ブロックの区切りに必要な空白を補う。
            prefix = header[key.start_mark.index:value.start_mark.index]
            if prefix.endswith(':') and value.start_mark.index < end.index:
                prefix += ' '
            if key_end.index == value.start_mark.index:
                prefix += ':'
            entry = prefix + header[value.start_mark.index:end.index] + '\n'
        else:
            first = next(key_lines)
            last = end.line + (end.column > mapping.start_mark.column)
            block = lines[first:max(first + 1, last)]
            keep_trailing_lines = (isinstance(value, yaml.ScalarEvent) and value.style in ('|', '>')
                                   and re.search(r'[|>](?:[1-9]?\+|\+[1-9]?)', lines[value.start_mark.line]))
            while not keep_trailing_lines and len(block) > 1 and (not block[-1].strip() or block[-1].startswith('#')):
                block.pop()
            entry = ''.join(line[mapping.start_mark.column:] if line.strip() else line for line in block)
        entries.append(entry)
        if isinstance(key, yaml.ScalarEvent) and (key.tag == YAML_MERGE_TAG or
                                                  (key.tag is None and key.style is None and key.value == '<<')):
            merges.append(entry)
    return retained_anchor_entries(header, managed, custom, merges)


def render_header(properties, custom):
    # キーとリストの順序は、正本を読み取った順序で固定する。
    lines = ['---']
    for key, value in properties.items():
        if isinstance(value, list):
            lines.append(f'{key}:' if value else f'{key}: []')
            lines.extend(f'  - {yaml_scalar(item)}' for item in value)
        else:
            scalar = value if key == 'updated' else yaml_scalar(value)
            lines.append(f'{key}: {scalar}' if scalar else f'{key}:')
    return '\n'.join(lines + ['']) + ''.join(custom) + '---\n'


def page_headers(root, materials, index, log):
    # サマリの置き場は scan と共有する。概念リンクの表示名には本文の題名を使う。
    summaries = {p: [] for p in sorted((root / SUMMARY_DIR).rglob('*.md'))}
    for material in materials:
        page = root / get_summary_path(material)
        if page in summaries:
            summaries[page].append(material)
    concepts = set((root / CONCEPTS_DIR).rglob('*.md'))
    entries = index_properties(root, index)
    log_rows = [(line[:10], line.partition('→')[2]) for line in reversed(log.splitlines())
                if re.match(r'^\d{4}-\d{2}-\d{2} ', line)]

    # ページ単位で不備を集め、書き込みの可否と scan の警告に使う。
    pages = {}
    titles = {}
    errors = {}
    concept_links = {}
    warnings = []
    for page in [*summaries, *sorted(concepts)]:
        if not page.is_file():
            continue
        text = page.read_bytes().decode('utf-8')
        bom = '\ufeff' if text.startswith('\ufeff') else ''
        try:
            old_header, body = split_header(text.removeprefix(bom))
            if page in summaries:
                section = re.search(r'(?ms)^## 主要な概念[ \t]*\r?\n(.*?)(?=^## |\Z)', body)
                links = []
                for _, _, target, _ in iter_markdown_links(section[1] if section else ''):
                    dest = ((page.parent / unquote(target.partition('#')[0])).resolve()
                            if is_local_link(target) else None)
                    if dest in concepts and dest.is_file():
                        links.append(dest)
                    else:
                        warnings.append(dict(page=page.relative_to(root).as_posix(), target=target,
                                             rule='主要な概念のリンク'))
                concept_links[page] = links
            title = re.search(r'(?m)^# ([^\r\n]+)', body)
            if title and title[1].strip():
                titles[page] = title[1].strip()
            custom = custom_header_entries(old_header)
            base, kinds = (SUMMARY_DIR, SUMMARY_KINDS) if page in summaries else (CONCEPTS_DIR, CONCEPT_KINDS)
            folder = page.relative_to(root / base).parts[0]
            if page not in titles:
                raise ValueError('missing title')
            if folder not in kinds:
                raise ValueError('unrecognized page folder')
            if len(summaries.get(page, [])) > 1:
                raise ValueError('multiple materials map to this summary')
            if len(entries.get(page, [])) > 1:
                raise ValueError('multiple index entries point to this page')
        except yaml.YAMLError as error:
            errors[page] = f'invalid YAML header: {error}'
        except ValueError as error:
            errors[page] = str(error)
        else:
            pages[page] = (text, bom, body, custom, kinds[folder])

    changes = {}
    for page, (text, bom, body, custom, kind) in pages.items():
        properties = {'title': titles[page]}
        name = page.relative_to(root).as_posix()
        if page in summaries:
            materials_for_page = summaries[page]
            material = materials_for_page[0] if materials_for_page else None
            material_link = None
            if material:
                material_path = Path(material)
                label = material_path.with_suffix('').name
                if material.startswith('docs/experiments/'):
                    label = material_path.relative_to('docs/experiments').with_suffix('').as_posix()
                material_link = f'[[{material}|{label}]]'
            links = [f'[[{dest.relative_to(root).as_posix()}|{titles[dest]}]]'
                     for dest in concept_links[page] if dest in titles]
            properties.update(page_kind='サマリ', material=material_link, material_kind=kind, concepts=links)
        else:
            aliases, description = entries.get(page, [([], '')])[0]
            properties.update(page_kind=kind, aliases=aliases, description=description)
        # 末尾から照合し、区切りと後ろの注記に依存せず更新日を決める。
        properties['updated'] = next((date for date, targets in log_rows if name in targets), None)
        header = render_header(properties, custom)
        if custom:
            # 管理キーに結び付いた YAML のアンカーなどを失う場合、壊れた出力を書かない。
            try:
                yaml.safe_load(header.removesuffix('---\n'))
            except yaml.YAMLError as error:
                errors[page] = f'invalid generated YAML header: {error}'
                continue
        after = bom + header + body
        if text != after:
            changes[page] = (after, bool(re.match(r'^---[ \t]*\r?\n', text.removeprefix(bom))))
    return changes, errors, warnings


def regenerate_headers(root):
    # 全ページを作ってから書き込み、差分のないページは開かない。
    root = root.resolve()
    try:
        index = (root / 'docs/wiki/index.md').read_text(encoding='utf-8-sig')
        log = (root / 'docs/wiki/log.md').read_text(encoding='utf-8-sig')
    except FileNotFoundError as error:
        raise ValueError(f'{Path(error.filename).relative_to(root).as_posix()}: required file is missing') from error
    changes, errors, _ = page_headers(root, material_paths(root), index, log)
    if errors:
        raise ValueError('\n'.join(f'{page.relative_to(root).as_posix()}: {reason}' for page, reason in errors.items()))
    for page, (after, _) in changes.items():
        page.write_bytes(after.encode('utf-8'))
    return {'updated': [page.relative_to(root).as_posix() for page in changes]}


def find_incoming_link_warnings(root):
    # 対象は Git 管理下または ignore 外の文書。wiki 自身の出典は broken_links が扱う。
    wiki = root / 'docs/wiki'
    names = git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
                '--', 'docs', 'reports', '.agents', '.claude', 'AGENTS.md', 'README.md')
    candidates = {}
    for page in wiki.rglob('*.md'):
        candidates.setdefault(page.name, []).append(page.relative_to(root).as_posix())
    warnings = []
    for name in sorted(set(names.split('\0'))):
        if not name.endswith('.md') or name.startswith('docs/wiki/'):
            continue
        source = root / name
        if not source.is_file():
            continue
        body = source.read_text(encoding='utf-8-sig')
        for number, _, target, path in iter_local_links(body):
            if 'wiki' not in Path(path).parts:
                continue
            dest = (source.parent / path).resolve()
            if dest.is_relative_to(wiki) and not dest.exists():
                warnings.append(dict(page=name, line=number, target=target, rule='外からのリンク切れ',
                                     candidates=sorted(candidates.get(dest.name, []))))
    return warnings


def broken_links(root):
    # コード例の記法と外部URLは除き、正本へのMarkdownリンクを検査する。
    root = Path(root).resolve()
    sources = material_paths(root)
    result = []
    for page in page_paths(root):
        text = read_page_body(page)
        # 機械一覧は再生成で修復する。出典リンクの前提条件には数えない。
        text = re.sub(r'(?ms)^## (?:ファイル|RFD の一覧|課題)\r?\n.*?(?=^## |\Z)', '', text)
        for _, line, target, path in iter_local_links(text):
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
    # 材料全体をblobで比較し、点検の自己出力を候補にしない。
    root = Path(root).resolve()
    start, checkpoint, records = log_state(root)
    implemented = implemented_rfds(issue_records(root))
    # 本文が同じまま実装が済んだRFDも、一度は意味を取り込み直す。
    log = (root/'docs/wiki/log.md').read_text(encoding='utf-8-sig')
    # 実装済みの日は課題の判定や解決のたびに動く導いた値なので、取り込み済みかはパスと blob だけで突き合わせる。
    acknowledged = {(m[1],m[2]) for m in re.finditer(r'^\d{4}-\d{2}-\d{2} (?:ingest|writeback) (.+?) blob:([0-9a-f]{40}) implemented:\d{4}-\d{2}-\d{2}',log,re.M)}
    known_blobs = set(records.values())
    pending = []
    moved = []
    materials = material_paths(root)
    expected = {root / get_summary_path(path) for path in materials}
    warnings = [dict(page=p.relative_to(root).as_posix(), rule='材料の無いサマリ')
                for p in sorted((root / SUMMARY_DIR).rglob('*.md')) if p not in expected]
    warnings.extend(find_incoming_link_warnings(root))
    for path in materials:
        blob = git(root, 'hash-object', '--', path).strip()
        judged = implemented.get(path)
        if judged and (path,blob) not in acknowledged:
            pending.append(dict(path=path,blob=blob,state='実装済み',implemented=judged))
            continue
        if path in records and records[path] != blob:
            pending.append(dict(path=path, blob=blob, state='更新あり'))
            continue
        summary = root / get_summary_path(path)
        if not summary.is_file():
            pending.append(dict(path=path, blob=blob, state='サマリ無し'))
            continue
        if read_summary_blob(summary) != blob:
            pending.append(dict(path=path, blob=blob, state='サマリ要更新'))
            continue
        if records.get(path) == blob:
            continue
        if blob in known_blobs and not path.startswith('docs/memo/done/'):
            moved.append(dict(path=path, blob=blob))
            continue
        pending.append(dict(path=path, blob=blob, state='未取り込み'))
    pending.sort(key=lambda row: (row['state'] != '更新あり', row['path']))
    changed_pages = changed_paths(root, checkpoint, ['docs/wiki'])
    pages = [p.relative_to(root).as_posix() for p in page_paths(root)]
    index = (root/'docs/wiki/index.md').read_text(encoding='utf-8-sig')
    unindexed = [p for p in pages if p.removeprefix('docs/wiki/') not in index]
    links = broken_links(root)
    warnings.extend(check_page_format(root, index))
    # ヘッダの書式に関する警告は、点検を回す条件には加えない。
    header_changes, header_errors, concept_warnings = page_headers(root, materials, index, log)
    for page, (_, has_header) in header_changes.items():
        header_errors[page] = 'YAML header differs from generated content' if has_header else 'missing YAML header'
    warnings.extend(dict(page=page.relative_to(root).as_posix(), rule='YAML ヘッダ', reason=reason)
                    for page, reason in sorted(header_errors.items()))
    warnings.extend(concept_warnings)
    return dict(start=start, checkpoint=checkpoint, pending=pending, moved=moved,
                changed_pages=sorted(set(pages) & changed_pages), broken_links=links,
                unindexed=unindexed, warnings=warnings,
                eligible=bool(pending or any(not r['annotated'] for r in links)))


def check_page_format(root, index):
    # 記法の問題は位置を返すだけで、点検の起動条件を変えない。
    warnings = []
    def warn(path, line, rule):
        warnings.append(dict(page=path.relative_to(root).as_posix(), line=line, rule=rule))

    for page in page_paths(root):
        body = read_page_body(page)
        headings = re.findall(r'^## (.+)$', body, re.M)
        if page.parent.name in MAP_FOLDERS:
            if ('変えるときの条件' not in headings or headings.count('外の事例') != 1
                    or headings.index('外の事例') != headings.index('変えるときの条件') + 1):
                warn(page, 1, 'トピック別まとめの外の事例')
            continue
        if headings != list(KNOWLEDGE_HEADINGS):
            warn(page, 1, '知見の節')
        section = None
        subheadings = []
        conclusion_lines = 0
        early_item = False
        for number, line in enumerate(body.splitlines(), 1):
            if line.startswith('## '):
                section = line[3:]
                continue
            if section == '今の結論' and line.strip():
                conclusion_lines += 1
                if conclusion_lines == 6 or len(line) > 200:
                    warn(page, number, '今の結論の長さ')
            elif section == '分かっていること':
                if line.startswith('### '):
                    subheadings.append(line[4:])
                elif line.startswith('- ') and not subheadings:
                    early_item = True
            elif section == '横断的知見' and line.startswith('- '):
                citations = re.split(r'出典[:：]', line, maxsplit=1)
                paths = {(page.parent / path).resolve() for _, _, _, path in iter_local_links(citations[-1])}
                if len(citations) < 2 or len(paths) < 2:
                    warn(page, number, '横断的知見の出典')
            elif section == '見立て' and line.startswith('- '):
                if not re.search(r'状態[:：]\s*未検証(?=$|[\s、。・,）)｜|])', line):
                    warn(page, number, '見立ての状態')
        if subheadings != ['anet の知見', '外の事例'] or early_item:
            warn(page, 1, '知見の小見出し')
    for number, line in enumerate(index.splitlines(), 1):
        description = line.partition(' — ')[2]
        if len(re.split(r'[（(]別名[:：]', description)[0].strip()) > 200:
            warn(root / 'docs/wiki/index.md', number, '目録の説明の長さ')
    for page in sorted((root / SUMMARY_DIR).rglob('*.md')):
        if len(read_page_body(page)) > 8000:
            warn(page, 1, 'サマリの長さ')
    return warnings


def replace_section(text, title, body):
    # 人が書く節と課題の区画はそのまま保持する。
    pattern = r'(?ms)(^## ' + re.escape(title) + r'\r?\n).*?(?=^## |\Z)'
    if not re.search(pattern, text):
        raise ValueError('Missing wiki section: '+title)
    return re.sub(pattern, lambda m: m[1]+'\n'+body.rstrip()+'\n\n', text, count=1)


def map_files(root, key, folder, rules):
    # ファイル集合はignoredを除く追跡・未追跡の全体で、統計のexcludedは適用しない。
    files = sorted(set(git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0'))-{''})
    files = [path for path in files if (Path(root)/path).is_file()]
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


# 件名の番号の記号。RFD に続く 4 桁と、RFD0195 より前の件名に残る PRD に続く 3 桁（git の履歴は書き換えないので、
# 旧い記号を読む処理は期間を限らず持つ）。どちらも、すぐ後ろに数字が続かないものだけを記号とする。
SUBJECT_NUMBER = re.compile(r'RFD(\d{4})(?!\d)|PRD(\d{3})(?!\d)')
SUBJECT_REFERENCES = re.compile(r'(?:\s+(?:RFD\d{4}(?!\d)|PRD\d{3}(?!\d)|[0-9a-f]{6}|#\d+))+\s*$')


def subject_rfd_numbers(subject):
    # 過去の形のscopeと、今の形の件名末尾の参照の並びからRFD番号を読む。本題の文中の番号は拾わない。
    # 番号は値で扱い、0 埋めの 4 桁の文字列で返す（PRD195 と RFD0195 は同じ 0195）。
    def numbers_in(text):
        return {f'{int(new or old):04d}' for new, old in SUBJECT_NUMBER.findall(text)}
    numbers = set()
    scope = re.match(r'^[^()]+\(([^)]+)\):', subject)
    if scope:
        numbers.update(numbers_in(scope[1]))
    references = SUBJECT_REFERENCES.search(subject)
    if references:
        numbers.update(numbers_in(references[0]))
    return numbers


RENUMBER_PREFIX = re.compile(r'^> \d{4}-\d{2}-\d{2} 番号を')
# 振り直しの行は 2 つの形を読む。RFD0195 より前に書いた行は 3 桁と（PRDnnn）の形のまま残り、道具が読み続ける。
RENUMBER_LINES = (re.compile(r'^> (\d{4}-\d{2}-\d{2}) 番号を (\d{4}) から (\d{4}) へ振り直した（RFD\d{4}）。$'),
                  re.compile(r'^> (\d{4}-\d{2}-\d{2}) 番号を (\d{3}) から (\d{3}) へ振り直した（PRD\d{3}）。$'))
RENUMBER_FORMAT = ('"> YYYY-MM-DD 番号を NNNN から NNNN へ振り直した（RFDNNNN）。" '
                   'or the form before RFD0195 "> YYYY-MM-DD 番号を NNN から NNN へ振り直した（PRDNNN）。"')


def renumber_declarations(root):
    # RFD の題のあと最初の見出しまでにある振り直しの行を読み、(旧い番号, 振り直した日, RFD のパス) を返す（RFD0131 D7・§5.3）。
    root = Path(root)
    result = []
    for path in sorted((root/'docs/memo').rglob('*_10rfd.md')):
        name = path.relative_to(root).as_posix()
        # 1 行目は題。本文やコードの枠に書いた日付の例を読まないよう、最初の見出しで止める。
        lines = []
        for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines()[1:], start=2):
            if line.startswith('#'):
                break
            if not RENUMBER_PREFIX.match(line):
                continue
            match = next((m for m in (p.match(line) for p in RENUMBER_LINES) if m), None)
            if not match:
                raise ValueError(f'{name}:{number}: renumber line={line!r}; expected {RENUMBER_FORMAT}')
            # 番号は形によらず値で比べるので、0 埋めの 4 桁にそろえて持つ。
            lines.append((match[1], f'{int(match[2]):04d}', f'{int(match[3]):04d}', number))
        # 1 つの RFD の振り直しは、日付の順に旧い番号から新しい番号へつながり、最後がファイルの番号になる。
        seen = set()
        for day, _, _, number in lines:
            if day in seen:
                raise ValueError(f'{name}:{number}: renumber date={day}; expected one renumber line per date')
            seen.add(day)
        chain = sorted(lines)
        for (_, _, new, _), (_, old, _, number) in zip(chain, chain[1:]):
            if old != new:
                raise ValueError(f'{name}:{number}: renumber from {old}; expected from {new} (the previous renumber line)')
        if chain and chain[-1][2] != path.name[:4]:
            raise ValueError(f'{name}:{chain[-1][3]}: renumber to {chain[-1][2]}; expected the file number {path.name[:4]}')
        result.extend((old, day, path) for day, old, _, _ in lines)
    return result


def commit_history(root):
    # 全コミットの日付・件名と、触ったファイルを今のパスで返す。改名をたどるので、ファイルを移したあとも、
    # 移す前のコミットが今のパスに当たる（RFD0182 でソースを src/ へ移したあとも、RFD の一覧に移す前の RFD を残す）。
    # Git は新しい順に出すので、改名を見るたびに、それより古いコミットの旧いパスを今のパスへ写す。
    log = git(root, 'log', '-M', '--name-status', '-z', '--date=short', '--format=%x01%ad%x09%s')
    current = {}
    history = []
    for chunk in log.split('\x01')[1:]:
        header, _, body = chunk.partition('\0')
        date, _, subject = header.partition('\t')
        tokens = body.split('\0')
        touched = set()
        i = 0
        while i < len(tokens):
            status = tokens[i].strip()
            if not status:
                i += 1
            elif status[0] in 'RC' and i+2 < len(tokens):
                now = current.get(tokens[i+2], tokens[i+2])
                touched.add(now)
                if status[0] == 'R':
                    current[tokens[i+1]] = now
                i += 3
            elif i+1 < len(tokens):
                touched.add(current.get(tokens[i+1], tokens[i+1]))
                i += 2
            else:
                break
        history.append((date, subject, touched))
    return history


def rfd_rows(root, page, files, pending=None, declarations=None, history=None):
    # 件名の番号はコミットの日付と組で持つ（振り直しの行の日付と比べるため）。
    import datetime
    import os
    if history is None:
        history = commit_history(root)
    page_files = set(files)
    mentions = set()
    for date, subject, touched in history:
        if touched & page_files:
            mentions.update((number, date) for number in subject_rfd_numbers(subject))
    # これから作るコミットも、触るファイルがページに当たれば履歴と同じに数える。日付は道具を回した日とする。
    if pending and pending['files'] & set(files):
        today = datetime.date.today().isoformat()
        mentions.update((number, today) for number in subject_rfd_numbers(pending['subject']))
    if declarations is None:
        declarations = renumber_declarations(root)
    # 番号と日付の組を RFD に当てる。振り直しの行が先で、無ければファイル名の番号で探す（RFD0131 §5.3）。
    sources = set()
    unresolved_counts = {}
    for number, date in sorted(mentions):
        declared = sorted({path for old, day, path in declarations if old == number and day >= date})
        matches = declared or sorted((Path(root)/'docs/memo').rglob(number+'_*_10rfd.md'))
        if len(matches) == 1:
            sources.add(matches[0])
        else:
            # 共有番号と消えた番号はリンク無しで残し、点検側へ報告する。
            unresolved_counts.setdefault(number, len(matches))
    # 同じ RFD に当たった番号は 1 行にまとめ、今の番号の順に並べる。
    entries = []
    for source in sources:
        number = source.name[:4]
        title = next((line.lstrip('# ').strip() for line in source.read_text(encoding='utf-8-sig').splitlines()
                      if line.startswith('# ')), 'RFD'+number)
        target = Path(os.path.relpath(source, page.parent)).as_posix()
        entries.append((number, target, f'- [RFD{number}]({target}) — {title}'))
    entries.extend((number, '', f'- RFD{number} — （RFD を一意に決められない）')
                   for number in unresolved_counts)
    rows = ['点検が作り直す。手で直さない。', '', *(row for _, _, row in sorted(entries))]
    unresolved = [dict(rfd='RFD'+number, matches=count) for number, count in sorted(unresolved_counts.items())]
    return ('\n'.join(rows) if mentions else '\n'.join(rows)+'なし'), unresolved


def issue_records(root):
    # issueの単票だけを読み、READMEと派生INDEXは入力にしない。
    records = []
    for path in sorted((Path(root)/'docs/management/issues').glob('issue-*.md')):
        text = path.read_text(encoding='utf-8-sig')
        match = re.match(r'\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)', text, re.S)
        if not match:
            raise ValueError(f'{path}: expected YAML frontmatter')
        try:
            record = yaml.safe_load(match[1])
        except yaml.YAMLError as error:
            raise ValueError(f'{path}: frontmatter: expected valid YAML: {error}') from error
        except ValueError as error:
            # 日付型の構築に失敗したときは、値を構築しない構文木から欄を特定する。
            from datetime import date
            mapping = yaml.compose(match[1],Loader=yaml.SafeLoader)
            if isinstance(mapping,yaml.MappingNode):
                for key,node in mapping.value:
                    if key.value in ('raised','resolved','ai_judged') and node.tag=='tag:yaml.org,2002:timestamp':
                        try:
                            date.fromisoformat(node.value)
                        except ValueError:
                            raise ValueError(f'{path}: {key.value}={node.value!r}; expected a valid YYYY-MM-DD calendar date') from error
            raise ValueError(f'{path}: frontmatter={match[1]!r}; expected valid YAML values and calendar dates: {error}') from error
        if not isinstance(record,dict):
            raise ValueError(f'{path}: frontmatter={record!r}; expected a mapping')
        validate_issue(root,path,record)
        # 対応内容の節（次の見出しまで）に行があるかを、状態との食い違いの検出に使う。
        section = re.search(r'(?ms)^## 対応内容[ \t]*\r?$(.*?)(?=^## |\Z)',text[match.end():])
        record['_progress'] = bool(section and section[1].strip())
        record['_path'] = path
        records.append(record)
    return records


ISSUE_TYPES = ('機能追加','機能改善','性能改善','リファクタリング','不具合','不整合','調査')
ISSUE_DOMAINS = ('アルゴリズム','学習基盤','UI','Env','基盤','開発')
ISSUE_RESOLUTIONS = ('対応済み','対応しない','重複','再現しない','不要になった')
ISSUE_ID = r'issue-\d{8}-[0-9a-f]{6}'


def link_name(value):
    # Obsidian のリンク "[[名前|別名]]" から別名とパスを外し、指す先のファイル名（拡張子なし）を返す。
    # リンクの形でない値は None を返し、呼び出し側が不正として止める。
    if not isinstance(value,str):
        return None
    match = re.fullmatch(r'\[\[([^\[\]|]+)(?:\|[^\[\]]*)?\]\]', value.strip())
    return match[1].strip().split('/')[-1] if match else None


def validate_issue(root, path, record):
    # 認識する欄の型と値を、派生ファイルの作成より先に検証する。
    # priority・memo・tags は開発者が直接書く管理外の欄なので、検証せず、派生ファイルにも出さない。
    def require(key, valid, expected):
        if not valid:
            raise ValueError(f'{path}: {key}={record.get(key)!r}; expected {expected}')
    for key, values in (
        ('type',ISSUE_TYPES),
        ('domain',ISSUE_DOMAINS),
        ('ai_judge',('未確認','残件有り','残件無し','判定不能')),
        ('status',('未着手','調査中','対応中','中断中','保留','次版','解決')),
        ('kind',('コード','文書','ハーネス')),
        ('phase',('RFD','実装','監査','ハーネス保守','その他'))):
        require(key,key in record and record[key] in values,values)
    keys = {c['key'] for c in CategoryRules(root).categories}
    for key in ('area','docs','depends'):
        value = record.get(key)
        require(key,isinstance(value,list) and all(isinstance(v,str) for v in value),'a list of strings')
    require('area',all(value in keys for value in record['area']),sorted(keys))
    for value in record['docs']:
        target = (Path(root)/value).resolve()
        valid = not Path(value).is_absolute() and target.is_relative_to(Path(root).resolve()) and target.is_file()
        require('docs',valid,'existing repository-relative document paths')
    require('id',isinstance(record.get('id'),str) and re.fullmatch(r'issue-\d{8}-[0-9a-f]{6}',record['id']) is not None and record['id']==path.stem,'the issue filename without .md')
    # 課題と RFD を指す欄は Obsidian のリンクで書き、以後の処理には指す先の名前で渡す。
    names = [link_name(v) for v in record['depends']]
    require('depends',all(n is not None and re.fullmatch(ISSUE_ID,n) for n in names),'Obsidian links to issues such as "[[issue-YYYYMMDD-xxxxxx]]"')
    record['depends'] = names
    parent = record.get('parent')
    if parent in (None,''):
        record['parent'] = None
    else:
        name = link_name(parent)
        require('parent',name is not None and re.fullmatch(ISSUE_ID,name) is not None,'an Obsidian link to an issue such as "[[issue-YYYYMMDD-xxxxxx]]" or an empty value')
        record['parent'] = name
    rfd = record.get('rfd')
    if rfd in (None,''):
        record['rfd'] = None
    else:
        # 課題が指せるのは、作業中の RFD が置かれる docs/memo 直下の RFD だけである。
        name = link_name(rfd)
        target = Path(root)/'docs/memo'/(str(name)+'.md')
        require('rfd',name is not None and re.fullmatch(r'\d{4}_.+_10rfd',name) is not None and target.is_file(),
                'an Obsidian link to an existing RFD directly under docs/memo such as "[[NNNN_slug_10rfd]]" or an empty value')
        record['rfd'] = 'docs/memo/'+name+'.md'
        require('docs',record['rfd'] not in record['docs'],'no copy of the RFD that rfd already points to')
    # 完了の理由は解決の課題だけが持ち、解決でなくなった課題に古い理由を残さない。
    if record['status'] == '解決':
        require('resolution',record.get('resolution') in ISSUE_RESOLUTIONS,ISSUE_RESOLUTIONS)
    else:
        require('resolution',record.get('resolution') in (None,''),'an empty value unless status is 解決')
    for key in ('title','scope'):
        require(key,isinstance(record.get(key),str) and bool(record[key].strip()),'a nonempty string')
    from datetime import date
    for key in ('raised','resolved','ai_judged'):
        value = record.get(key)
        if key != 'raised' and value in (None,''):
            continue
        # YAMLの日付型も受け、整数や週日付を暗黙に読み替えない。
        require(key,re.fullmatch(r'\d{4}-\d{2}-\d{2}',str(value)) is not None,'a YYYY-MM-DD date')
        try:
            date.fromisoformat(str(value))
        except (ValueError,TypeError):
            require(key,False,'a YYYY-MM-DD date')
        record[key] = str(value)
    # 未確認は判定していない印なので日付を持たず、ほかの4値は判定した日付を持つ。
    require('ai_judged',(record['ai_judge']=='未確認')==(record.get('ai_judged') in (None,'')),'empty when ai_judge is 未確認, otherwise a date')
    # WSJFの元の値は離散的な見立てであり、端数を黙って補正しない。
    from decimal import Decimal, ROUND_HALF_UP
    import math
    scores = [record.get(k) for k in ('value','urgency','risk','size')]
    if record['status'] in ('次版','解決') and all(v is None for v in scores) and record.get('wsjf') is None:
        return
    for key,value in zip(('value','urgency','risk','size'),scores):
        require(key,type(value) is int and value in (1,2,3,5,8,13),'1, 2, 3, 5, 8, or 13')
    expected = float((Decimal(sum(scores[:3]))/Decimal(scores[3])).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))
    score = record.get('wsjf')
    require('wsjf',type(score) in (int,float) and math.isfinite(score) and score==expected,str(expected)+' from (value + urgency + risk) / size')


def implemented_rfds(records):
    # RFD の実装済みは RFD 自身にも課題の欄にも持たせず、その RFD を rfd で指す課題から導く（RFD0120 §5.2）。
    groups = {}
    for record in records:
        if record.get('rfd'):
            groups.setdefault(record['rfd'],[]).append(record)
    result = {}
    for path, issues in groups.items():
        # やらないと決めた仕事が残る RFD は、指示の全部が実装されていないので実装済みにしない。
        if any(r['status'] == '次版' or (r['status'] == '解決' and r.get('resolution') in ('対応しない','不要になった')) for r in issues):
            continue
        # 直す仕事が元から無い課題（重複・再現しない）は数えない。
        counted = [r for r in issues if not (r['status'] == '解決' and r.get('resolution') in ('重複','再現しない'))]
        # 解決は開発者が決めた事実なので、判定によらず済みとして数える。
        if not counted or not all(r['ai_judge'] == '残件無し' or r['status'] == '解決' for r in counted):
            continue
        # 表示と記録に使う日は、判定日と解決日の最も新しい日。どちらも無ければ起票日で補う。
        days = [d for r in counted for d in (r.get('ai_judged'), r.get('resolved')) if d]
        result[path] = max(days) if days else max(str(r['raised']) for r in counted)
    return result


def issue_order(records):
    # 絞り込む前の全件で、待たれている課題のWSJFを前提へ伝える。
    by_id = {r['id']: r for r in records}
    scores = {r['id']: r.get('wsjf') for r in records}
    active = {'対応中','調査中','中断中','未着手'}
    for record in records:
        if record['status'] not in active or record.get('wsjf') is None:
            continue
        stack = list(record['depends'])
        seen = set()
        while stack:
            dependency = stack.pop()
            if dependency not in by_id or dependency in seen:
                continue
            seen.add(dependency)
            if scores[dependency] is None or scores[dependency] < record['wsjf']:
                scores[dependency] = record['wsjf']
            stack.extend(by_id[dependency]['depends'])
    result = []
    for record in records:
        # 循環している課題も捨てず、待ちとして一覧と注意へ出す。
        stack = list(record['depends'])
        seen = set()
        cyclic = False
        while stack:
            key = stack.pop()
            if key == record['id']:
                cyclic = True
                break
            if key in by_id and key not in seen:
                seen.add(key)
                stack.extend(by_id[key]['depends'])
        waiting = [key for key in record['depends'] if key not in by_id or by_id[key]['status'] != '解決']
        group = {'保留':3,'次版':4,'解決':5}.get(record['status'],2 if waiting or cyclic else 1)
        result.append(dict(record,_waiting=waiting,_group=group,_score=scores[record['id']],_cyclic=cyclic))
    return sorted(result,key=lambda r:(r['_group'],r['status'] not in ('対応中','調査中'),r['_score'] is None,
                                      -(r['_score'] or 0),str(r['raised']),r['id']))


def issue_table(root, page, records, all_records=None):
    # 同じ列とリンクの作り方をINDEXとトピック別まとめで使う。
    # 位置の番号の列は持たない。1 件の増減や並べ替えで、ほかの行まで書き換わってコミットの差分が膨らむため。
    import os
    lines = ['| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for record in records:
        target = Path(os.path.relpath(record['_path'], page.parent)).as_posix()
        score = record.get('_score',record.get('wsjf'))
        wsjf = '—' if score is None else f'{score:.1f}'
        if score is not None and (record.get('wsjf') is None or score > record['wsjf']):
            wsjf += '（引継ぎ）'
        waiting = []
        for key in record.get('_waiting',[]):
            source = Path(root)/'docs/management/issues'/(key+'.md')
            link = Path(os.path.relpath(source,page.parent)).as_posix()
            waiting.append(f'[{key}]({link})' if source.exists() else key+'（無い）')
        # RFD の列は、ファイル名の先頭の番号を表示し、RFD 本文へのリンクにする。
        rfd = '—'
        if record.get('rfd'):
            rfd_link = Path(os.path.relpath(Path(root)/record['rfd'],page.parent)).as_posix()
            rfd = f"[RFD{Path(record['rfd']).name[:4]}]({rfd_link})"
        cells = [f"[{record['id']}]({target})", record['status'], wsjf, ', '.join(waiting) or '—',
                 record['kind'], record['phase'], ', '.join(record['area']) or '—', record['domain'], record['type'], rfd,
                 record['ai_judge'], record['title']]
        lines.append('| '+' | '.join(str(cell).replace('|','\\|').replace('\n',' ') for cell in cells)+' |')
    if records:
        lines += ['', ' / '.join(f'{label} {sum(r.get("_group")==group for r in records)}' for group,label in enumerate(('着手できる','待ち','保留','次版','解決'),1))]
        for record in records:
            if record.get('_cyclic'):
                target = Path(os.path.relpath(record['_path'],page.parent)).as_posix()
                lines.append(f"- 注意: [{record['id']}]({target}) の依存が循環している。")
        # 絞り込みで表から外れた前提も、題とリンクを添えて開けるようにする。
        shown = {r['id'] for r in records}
        related = {key for r in records for key in r.get('_waiting',[]) if key not in shown}
        for record in all_records or records:
            if record['id'] in related:
                target = Path(os.path.relpath(record['_path'],page.parent)).as_posix()
                lines.append(f"- 関連する課題: [{record['id']}]({target}) — {record['title']}")
    return '\n'.join(lines) if records else '課題はありません。'


def direct_rfds(root):
    # 作業中の RFD は docs/memo の直下に置く。過去の置き場（done/ など）は含めない。
    return sorted(p for p in (Path(root)/'docs/memo').glob('*_10rfd.md') if re.match(r'\d{4}_',p.name))


def rfd_title(path):
    # RFD の最初の見出しを題として使い、無ければファイル名にする。
    return next((line.lstrip('# ').strip() for line in path.read_text(encoding='utf-8-sig').splitlines()
                 if line.startswith('# ')), path.stem)


def decision_sections(root, page, records):
    # 状態を開発者が決める課題を、判定と状態の組み合わせから拾う。状態は書き換えない。
    import os
    waiting = [r for r in records if r['status'] != '解決' and r['ai_judge'] == '残件無し']
    # 解決より前の判定は、解決を決めたときに見ていたものなので候補にしない。
    reopen = [r for r in records if r['status'] == '解決' and r['ai_judge'] == '残件有り'
              and (r.get('resolved') in (None,'') or r['ai_judged'] >= r['resolved'])]
    # 未着手なのに対応内容に行があるものは、行の置き場か状態が違うことが多いので、確かめる候補として出す。
    started = [r for r in records if r['status'] == '未着手' and r.get('_progress')]
    def table(rows, header, values, order=lambda r:(r['ai_judged'],r['id'])):
        if not rows:
            return '該当する課題はありません。'
        lines = ['| '+' | '.join(header)+' |', '|'+'---|'*len(header)]
        for record in sorted(rows,key=order):
            target = Path(os.path.relpath(record['_path'], page.parent)).as_posix()
            cells = [f"[{record['id']}]({target})", *values(record), record['title']]
            lines.append('| '+' | '.join(str(cell).replace('|','\\|').replace('\n',' ') for cell in cells)+' |')
        return '\n'.join(lines)
    # どの課題の rfd からも指されない直下の RFD は、一覧に載る課題を持たないので名指しで出す。
    referenced = {r['rfd'] for r in records if r.get('rfd')}
    orphans = [p for p in direct_rfds(root) if p.relative_to(root).as_posix() not in referenced]
    if orphans:
        lines = ['| RFD | 題 |', '|---|---|']
        for path in orphans:
            target = Path(os.path.relpath(path, page.parent)).as_posix()
            lines.append(f'| [{path.stem}]({target}) | '+rfd_title(path).replace('|','\\|')+' |')
        orphan_table = '\n'.join(lines)
    else:
        orphan_table = '該当する RFD はありません。'
    return '\n'.join([
        '## 解決待ち', '',
        'AI の判定が「残件無し」で、解決でない課題。解決にするかは開発者が決める（コミットの準備で持ちかける）。', '',
        table(waiting, ['ID','状態','AIの判定','判定日','課題'], lambda r:[r['status'],r['ai_judge'],r['ai_judged']]), '',
        '## 再開候補', '',
        '解決の日付以後に（解決の日付が無ければ日付によらず）AI の判定が「残件有り」になった解決の課題。状態を戻すかは開発者が決める。', '',
        table(reopen, ['ID','解決日','判定日','課題'], lambda r:[r.get('resolved') or '—',r['ai_judged']]), '',
        '## 課題の無い RFD', '',
        'docs/memo 直下の RFD のうち、どの課題の rfd からも指されないもの。その RFD の仕事の課題を作るか、題材の課題の rfd に書く。', '',
        orphan_table, '',
        '## 対応内容のある未着手の課題', '',
        '状態が未着手なのに、対応内容に行がある課題。行の置き場（対応でない行は anet-create-issue の「起票のあとの追記」の節へ）と状態を確かめる。開発者が未着手へ戻した課題のように、どちらも正しければそのままでよい（状態は開発者が決める）。', '',
        table(started, ['ID','起票日','課題'], lambda r:[r['raised']], order=lambda r:r['id'])])


def issue_section(text, body, path):
    # 区画外の本文を保ち、全出力の検査が終わるまでは書かない。
    begin, end = '<!-- issues:begin -->', '<!-- issues:end -->'
    if text.count(begin) != 1 or text.count(end) != 1 or not re.search(r'(?m)^'+re.escape(begin)+r'\r?$',text) or not re.search(r'(?m)^'+re.escape(end)+r'\r?$',text) or text.index(begin) >= text.index(end):
        raise ValueError(f'{path}: expected one ordered issues begin/end marker pair on separate lines')
    left, rest = text.split(begin,1)
    _, right = rest.split(end,1)
    return left+begin+'\n'+body+'\n'+end+right


def regenerate_maps(root, pending_subject=None, pending_files=()):
    # 全ページとINDEXの生成結果を揃えてから書き戻す。
    # pending_* はコミットの準備が渡す予定のコミット。RFD の一覧の行を、原因のコミットに入れるため。
    root = Path(root).resolve()
    pending = (dict(subject=pending_subject, files={p.replace('\\', '/') for p in pending_files})
               if pending_subject else None)
    rules = CategoryRules(root)
    records = issue_order(issue_records(root))
    # RFD の振り直しの行は全ページで同じなので、書く前に 1 回だけ読んで検査する（崩れていれば何も書かずに止まる）。
    declarations = renumber_declarations(root)
    # 履歴も全ページで同じなので、1 回だけ読む。
    history = commit_history(root)
    index = root/'docs/management/issues/INDEX.md'
    current = [r for r in records if r['status'] not in ('次版','解決')]
    index_text = ('# 課題の一覧\n\n道具が作り直す。手で直さない。既定は本版（次版・解決を除く）。\n\n'+issue_table(root,index,current)+'\n\n'
                  +decision_sections(root,index,records)+'\n')
    updates = [] if index.is_file() and index.read_bytes() == index_text.encode('utf-8') else [(index,index_text)]
    unresolved_rfds = []
    for folder in MAP_FOLDERS:
        for page in sorted((root / CONCEPTS_DIR / folder).glob('*.md')):
            files = map_files(root, page.stem, folder, rules)
            text = page.read_bytes().decode('utf-8')
            result = replace_section(text, 'ファイル', file_rows(root, page, files, rules,
                                                                   page.stem if folder == 'apps' else None))
            rfds, unresolved = rfd_rows(root, page, files, pending, declarations, history)
            result = replace_section(result, 'RFD の一覧', rfds)
            selected = [r for r in records if page.stem in r['area'] and r['status'] not in ('次版','解決')]
            result = issue_section(result, issue_table(root,page,selected,records), page.relative_to(root))
            unresolved_rfds.extend(dict(page=page.relative_to(root).as_posix(), **row)
                                   for row in unresolved)
            if result != text:
                updates.append((page, result))
    for page, text in updates:
        page.parent.mkdir(parents=True,exist_ok=True)
        page.write_bytes(text.encode('utf-8'))
    return dict(updated=[p.relative_to(root).as_posix() for p, _ in updates],
                unresolved_rfds=unresolved_rfds)

def main():
    # 点検、一覧の再生成、ヘッダの再生成は別の操作として扱う。
    import argparse
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Internal mechanical support for wiki skills')
    parser.add_argument('action', choices=['scan', 'maps', 'headers'])
    parser.add_argument('--pending-subject', help='maps: これから作るコミットの件名（コミットメッセージ案）')
    parser.add_argument('--pending-files', nargs='+', default=[], metavar='PATH',
                        help='maps: そのコミットの対象ファイル（repo 相対）')
    args = parser.parse_args()
    if (args.pending_subject or args.pending_files) and (args.action != 'maps' or not args.pending_subject
                                                         or not args.pending_files):
        parser.error('--pending-subject と --pending-files は maps で組にして渡す')
    try:
        root = Path.cwd()
        if args.action == 'scan':
            result = scan(root)
        elif args.action == 'headers':
            result = regenerate_headers(root)
        else:
            result = regenerate_maps(root, args.pending_subject, args.pending_files)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
