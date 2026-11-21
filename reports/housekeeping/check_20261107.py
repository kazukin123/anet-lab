"""今回の文書整備を読み取り専用で検査する。"""
from pathlib import Path
import json
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports/tools'))
import wiki_support as wiki


def check_wiki():
    # 全ページの導線と出典を検査し、消えた名前は判断用の候補だけ返す。
    index = (ROOT / 'docs/wiki/index.md').read_text(encoding='utf-8-sig')
    problems, no_description = [], []
    absent_sections = {'選ばなかった形': 0, '変えるときの条件': 0}
    code_files = subprocess.check_output(['git', 'ls-files', '-z', '--', 'core', 'apps'], cwd=ROOT).decode('utf-8').split('\0')
    corpus = '\n'.join((ROOT / p).read_text(encoding='utf-8', errors='ignore') for p in code_files if p and (ROOT/p).is_file() and Path(p).suffix in {'.hpp', '.cpp', '.txt', '.java', '.js', '.py'})
    missing_names = set()
    for p in wiki.page_paths(ROOT):
        text = p.read_text(encoding='utf-8-sig')
        rel = p.relative_to(ROOT).as_posix()
        if p.relative_to(ROOT/'docs/wiki').as_posix() not in index:
            problems.append([rel, 'not indexed'])
        if '[wiki の目録](../index.md)' not in text.split('## ', 1)[0]:
            problems.append([rel, 'missing breadcrumb'])
        sections = re.split(r'^## (.+)\n', text, flags=re.M)
        for title, body in zip(sections[1::2], sections[2::2]):
            if title in {'ファイル', 'PRD の一覧', '課題'}:
                continue
            if title == '概要・責務' and '設計文書に記述なし' in body:
                no_description.append(rel)
            if title in absent_sections and re.search(r'^記述なし[。.]?$', body, re.M):
                absent_sections[title] += 1
            required = title in {'分かっていること', '撤回・更新', '概要・責務', '成立の経緯', '選ばなかった形', '変えるときの条件', '関係'}
            for line in body.splitlines():
                if required and line.startswith('- ') and not any(s in line for s in ['記述なし', 'なし']):
                    if not re.search(r'\]\([^)]+\)', line):
                        problems.append([rel, title, line[:100]])
            for name in re.findall(r'`([^`\n]+)`', body):
                if not name.endswith(('.py', '.md', '.cpp', '.hpp', '.java', '.txt')) and re.fullmatch(r'(?:[A-Z][A-Za-z0-9]*(?:Agent|Actor|Learner|Runner|Observer|Controller|Store|Manager|Buffer|Network)|[a-z_]+(?:\.[a-z_]+)+)', name) and name not in corpus:
                    missing_names.add(name)
    return dict(problems=problems, broken_links=wiki.broken_links(ROOT), no_description=no_description, absent_sections=absent_sections, missing_names=sorted(missing_names))


def check_translations():
    results = []
    for name in ['140_observability', '200_dqn_agents', '212_rnd_lifelong_novelty']:
        jp = ROOT / 'docs/design' / (name + '.jp.md')
        en = jp.with_name(name + '.en.md')
        a, b = jp.read_text(encoding='utf-8-sig'), en.read_text(encoding='utf-8-sig')
        blocks = lambda s: re.findall(r'^```[^\n]*\n.*?^```', s, re.M | re.S)
        outside = re.sub(r'^```[^\n]*\n.*?^```', '', b, flags=re.M | re.S)
        headings = lambda s: re.findall(r'^(#{1,6}) ', s, re.M)
        tables = lambda s: [len(t.splitlines()) for t in re.findall(r'(?:^\|.*\n)+', s, re.M)]
        links = re.findall(r'\]\(([^)]+)\)', outside)
        broken = []
        for link in links:
            if re.match(r'https?://', link):
                continue
            path, _, anchor = unquote(link).partition('#')
            target = (en.parent / path) if path else en
            if not target.exists():
                broken.append(link)
            elif anchor and target.suffix == '.md':
                titles = re.findall(r'^#{1,6} (.+)$', target.read_text(encoding='utf-8-sig'), re.M)
                slugs = [re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-') for title in titles]
                if anchor not in slugs:
                    broken.append(link)
        blob = subprocess.check_output(['git', 'hash-object', str(jp)], cwd=ROOT).decode().strip()
        results.append(dict(file=en.name, source_blob=blob, header_matches=('blob:'+blob) in b.splitlines()[0],
                            headings_match=headings(a)==headings(b), headings=len(headings(b)),
                            blocks_match=blocks(a)==blocks(b), blocks=len(blocks(b)),
                            tables_match=tables(a)==tables(b), table_rows=tables(b),
                            japanese_outside_blocks=[x for x in outside.splitlines() if re.search(r'[ぁ-んァ-ヶ一-龯]', x)],
                            broken_links=broken, crlf=b'\r' in en.read_bytes(), bom=en.read_bytes().startswith(b'\xef\xbb\xbf')))
    return results


def check_surveys():
    # 目次・内部リンク・引用の書式と保存形式を、今回の文書だけに照合する。
    results = []
    names = ['llm_assisted_rl_resurvey_2026-11-07.md',
             'llm_assisted_rl_reproducibility_2026-11-07.md',
             'llm_reward_shaping_guarantees_2026-11-07.md',
             'llm_generated_control_and_environments_2026-11-07.md',
             'llm_rl_small_models_and_costs_2026-11-07.md',
             'llm_rl_exploration_information_2026-11-07.md',
             'llm_rl_algorithm_discovery_2026-11-07.md',
             'llm_rl_language_grounding_2026-11-07.md',
             'llm_rl_reading_guide_2026-11-07.md',
             'llm_rl_goals_and_reflection_2026-11-07.md',
             'llm_rl_action_advice_2026-11-07.md',
             'llm_rl_distillation_and_planning_2026-11-07.md']
    for name in names:
        path = ROOT / 'reports' / name
        body = path.read_text(encoding='utf-8')
        outside = re.sub(r'^```[^\n]*\n.*?^```', '', body, flags=re.M | re.S)
        headings = re.findall(r'^#{2,3} (.+)$', outside, re.M)
        toc = body.split('## 目次\n', 1)[1].split('\n## ', 1)[0]
        toc_titles = re.findall(r'\[([^\]]+)\]\(#[^)]+\)', toc)
        broken, quotation_errors = [], []
        for link in re.findall(r'\]\(([^)]+)\)', outside):
            if re.match(r'https?://', link):
                continue
            file, _, anchor = unquote(link).partition('#')
            target = path.parent / file if file else path
            if not target.exists():
                broken.append(link)
            elif anchor and target.suffix == '.md':
                titles = re.findall(r'^#{1,6} (.+)$', target.read_text(encoding='utf-8-sig'), re.M)
                slugs = [re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-') for title in titles]
                if anchor not in slugs:
                    broken.append(link)
        for match in re.finditer(r'^```text\n(.*?)\n```\n([^\n]+)', body, re.M | re.S):
            quote, source = match.groups()
            original, separator, translation = quote.partition('\n\n')
            if not separator or not re.search(r'[ぁ-んァ-ヶ一-龯]', translation):
                quotation_errors.append('missing translation: ' + original[:60])
            if not re.match(r'\[.+\]\(https?://[^)]+\)$', source):
                quotation_errors.append('missing source: ' + source)
            if len(original.split()) > 25:
                quotation_errors.append('long quote: ' + original[:60])
        results.append(dict(file=name, toc_matches=toc_titles == [h for h in headings if h != '目次'],
                            broken_links=broken, quotation_errors=quotation_errors,
                            headings=len(headings), bytes=path.stat().st_size,
                            crlf=b'\r' in path.read_bytes(), bom=path.read_bytes().startswith(b'\xef\xbb\xbf')))
    return results


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    result = check_surveys() if 'surveys' in sys.argv else check_translations() if 'translations' in sys.argv else check_wiki()
    print(json.dumps(result, ensure_ascii=False, indent=2))
