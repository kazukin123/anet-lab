"""wikiとissueの機械処理。本文の意味判断はskillを実行するエージェントが行う。"""
from pathlib import Path
import json
import re

import yaml


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
        text = re.sub(r'(?ms)^## (?:ファイル|PRD の一覧|課題)\r?\n.*?(?=^## |\Z)', '', text)
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
    # 材料全体をblobで比較し、点検の自己出力を候補にしない。
    root = Path(root).resolve()
    start, checkpoint, records = log_state(root)
    implemented = implemented_prds(issue_records(root))
    # 本文が同じまま実装が済んだPRDも、一度は意味を取り込み直す。
    log = (root/'docs/wiki/log.md').read_text(encoding='utf-8-sig')
    # 実装済みの日は課題の判定や解決のたびに動く導いた値なので、取り込み済みかはパスと blob だけで突き合わせる。
    acknowledged = {(m[1],m[2]) for m in re.finditer(r'^\d{4}-\d{2}-\d{2} (?:ingest|writeback) (.+?) blob:([0-9a-f]{40}) implemented:\d{4}-\d{2}-\d{2}',log,re.M)}
    known_blobs = set(records.values())
    pending = []
    moved = []
    for path in material_paths(root):
        blob = git(root, 'hash-object', '--', path).strip()
        judged = implemented.get(path)
        if judged and (path,blob) not in acknowledged:
            pending.append(dict(path=path,blob=blob,state='実装済み',implemented=judged))
            continue
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


def subject_prd_numbers(subject):
    # 過去の形のscopeと、今の形の件名末尾の参照の並びからPRD番号を読む。本題の文中の番号は拾わない。
    numbers = set()
    scope = re.match(r'^[^()]+\(([^)]+)\):', subject)
    if scope:
        numbers.update(re.findall(r'PRD(\d{3})', scope[1]))
    references = re.search(r'(?:\s+(?:PRD\d{3}|[0-9a-f]{6}|#\d+))+\s*$', subject)
    if references:
        numbers.update(re.findall(r'PRD(\d{3})', references[0]))
    return numbers


def prd_rows(root, page, files):
    # コマンド長を抑えて履歴を読む。
    import os
    numbers = set()
    for offset in range(0, len(files), 80):
        subjects = git(root, 'log', '--format=%s', '--', *files[offset:offset+80])
        for line in subjects.splitlines():
            numbers.update(subject_prd_numbers(line))
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
        ('kind',('成果物','ハーネス')),
        ('phase',('PRD','実装','監査','ハーネス保守','その他'))):
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
    # 課題と PRD を指す欄は Obsidian のリンクで書き、以後の処理には指す先の名前で渡す。
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
    prd = record.get('prd')
    if prd in (None,''):
        record['prd'] = None
    else:
        # 課題が指せるのは、作業中の PRD が置かれる docs/memo 直下の PRD だけである。
        name = link_name(prd)
        target = Path(root)/'docs/memo'/(str(name)+'.md')
        require('prd',name is not None and re.fullmatch(r'\d{3}_.+_10prd',name) is not None and target.is_file(),
                'an Obsidian link to an existing PRD directly under docs/memo such as "[[NNN_slug_10prd]]" or an empty value')
        record['prd'] = 'docs/memo/'+name+'.md'
        require('docs',record['prd'] not in record['docs'],'no copy of the PRD that prd already points to')
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


def implemented_prds(records):
    # PRD の実装済みは PRD 自身にも課題の欄にも持たせず、その PRD を prd で指す課題から導く（PRD120 §5.2）。
    groups = {}
    for record in records:
        if record.get('prd'):
            groups.setdefault(record['prd'],[]).append(record)
    result = {}
    for path, issues in groups.items():
        # やらないと決めた仕事が残る PRD は、指示の全部が実装されていないので実装済みにしない。
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
    # 同じ列とリンクの作り方をINDEXと地図で使う。
    import os
    lines = ['| # | ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for number, record in enumerate(records, 1):
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
        # PRD の列は、ファイル名の先頭の番号を表示し、PRD 本文へのリンクにする。
        prd = '—'
        if record.get('prd'):
            prd_link = Path(os.path.relpath(Path(root)/record['prd'],page.parent)).as_posix()
            prd = f"[PRD{Path(record['prd']).name[:3]}]({prd_link})"
        cells = [number, f"[{record['id']}]({target})", record['status'], wsjf, ', '.join(waiting) or '—',
                 record['kind'], record['phase'], ', '.join(record['area']) or '—', record['domain'], record['type'], prd,
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


def direct_prds(root):
    # 作業中の PRD は docs/memo の直下に置く。過去の置き場（done/ など）は含めない。
    return sorted(p for p in (Path(root)/'docs/memo').glob('*_10prd.md') if re.match(r'\d{3}_',p.name))


def prd_title(path):
    # PRD の最初の見出しを題として使い、無ければファイル名にする。
    return next((line.lstrip('# ').strip() for line in path.read_text(encoding='utf-8-sig').splitlines()
                 if line.startswith('# ')), path.stem)


def decision_sections(root, page, records):
    # 状態を開発者が決める課題を、判定と状態の組み合わせから拾う。状態は書き換えない。
    import os
    waiting = [r for r in records if r['status'] != '解決' and r['ai_judge'] == '残件無し']
    # 解決より前の判定は、解決を決めたときに見ていたものなので候補にしない。
    reopen = [r for r in records if r['status'] == '解決' and r['ai_judge'] == '残件有り'
              and (r.get('resolved') in (None,'') or r['ai_judged'] >= r['resolved'])]
    def table(rows, header, values):
        if not rows:
            return '該当する課題はありません。'
        lines = ['| '+' | '.join(header)+' |', '|'+'---|'*len(header)]
        for number, record in enumerate(sorted(rows,key=lambda r:(r['ai_judged'],r['id'])), 1):
            target = Path(os.path.relpath(record['_path'], page.parent)).as_posix()
            cells = [number, f"[{record['id']}]({target})", *values(record), record['title']]
            lines.append('| '+' | '.join(str(cell).replace('|','\\|').replace('\n',' ') for cell in cells)+' |')
        return '\n'.join(lines)
    # どの課題の prd からも指されない直下の PRD は、一覧に載る課題を持たないので名指しで出す。
    referenced = {r['prd'] for r in records if r.get('prd')}
    orphans = [p for p in direct_prds(root) if p.relative_to(root).as_posix() not in referenced]
    if orphans:
        lines = ['| # | PRD | 題 |', '|---|---|---|']
        for number, path in enumerate(orphans, 1):
            target = Path(os.path.relpath(path, page.parent)).as_posix()
            lines.append(f'| {number} | [{path.stem}]({target}) | '+prd_title(path).replace('|','\\|')+' |')
        orphan_table = '\n'.join(lines)
    else:
        orphan_table = '該当する PRD はありません。'
    return '\n'.join([
        '## 解決待ち', '',
        'AI の判定が「残件無し」で、解決でない課題。解決にするかは開発者が決める（コミットの準備で持ちかける）。', '',
        table(waiting, ['#','ID','状態','AIの判定','判定日','課題'], lambda r:[r['status'],r['ai_judge'],r['ai_judged']]), '',
        '## 再開候補', '',
        '解決の日付以後に（解決の日付が無ければ日付によらず）AI の判定が「残件有り」になった解決の課題。状態を戻すかは開発者が決める。', '',
        table(reopen, ['#','ID','解決日','判定日','課題'], lambda r:[r.get('resolved') or '—',r['ai_judged']]), '',
        '## 課題の無い PRD', '',
        'docs/memo 直下の PRD のうち、どの課題の prd からも指されないもの。その PRD の仕事の課題を作るか、題材の課題の prd に書く。', '',
        orphan_table])


def issue_section(text, body, path):
    # 区画外の本文を保ち、全出力の検査が終わるまでは書かない。
    begin, end = '<!-- issues:begin -->', '<!-- issues:end -->'
    if text.count(begin) != 1 or text.count(end) != 1 or not re.search(r'(?m)^'+re.escape(begin)+r'\r?$',text) or not re.search(r'(?m)^'+re.escape(end)+r'\r?$',text) or text.index(begin) >= text.index(end):
        raise ValueError(f'{path}: expected one ordered issues begin/end marker pair on separate lines')
    left, rest = text.split(begin,1)
    _, right = rest.split(end,1)
    return left+begin+'\n'+body+'\n'+end+right


def regenerate_maps(root):
    # 全ページとINDEXの生成結果を揃えてから書き戻す。
    root = Path(root).resolve()
    rules = CategoryRules(root)
    records = issue_order(issue_records(root))
    index = root/'docs/management/issues/INDEX.md'
    current = [r for r in records if r['status'] not in ('次版','解決')]
    index_text = ('# 課題の一覧\n\n道具が作り直す。手で直さない。既定は本版（次版・解決を除く）。\n\n'+issue_table(root,index,current)+'\n\n'
                  +decision_sections(root,index,records)+'\n')
    updates = [] if index.is_file() and index.read_bytes() == index_text.encode('utf-8') else [(index,index_text)]
    unresolved_prds = []
    for folder in ['apps', 'categories']:
        for page in sorted((root/'docs/wiki'/folder).glob('*.md')):
            files = map_files(root, page.stem, folder, rules)
            text = page.read_bytes().decode('utf-8')
            result = replace_section(text, 'ファイル', file_rows(root, page, files, rules,
                                                                   page.stem if folder == 'apps' else None))
            prds, unresolved = prd_rows(root, page, files)
            result = replace_section(result, 'PRD の一覧', prds)
            selected = [r for r in records if page.stem in r['area'] and r['status'] not in ('次版','解決')]
            result = issue_section(result, issue_table(root,page,selected,records), page.relative_to(root))
            unresolved_prds.extend(dict(page=page.relative_to(root).as_posix(), **row)
                                   for row in unresolved)
            if result != text:
                updates.append((page, result))
    for page, text in updates:
        page.parent.mkdir(parents=True,exist_ok=True)
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
