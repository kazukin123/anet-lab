"""issueから派生一覧への公開CLIを、一時repoだけで検証する。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

REPO = Path(__file__).resolve().parents[4]
HELPER = REPO/'reports/tools/wiki_support.py'


class IssueViewsTest(unittest.TestCase):
    def setUp(self):
        # 本物のindexとrefを共有しない、一時repoを用意する。
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env = dict(os.environ)
        self.env.pop('GIT_INDEX_FILE', None)
        self.env.pop('GIT_DIR', None)
        self.env.pop('GIT_WORK_TREE', None)
        self.env['PYTHONUTF8'] = '1'
        self.git('init','-q')
        self.write('reports/tools/categories.json',(REPO/'reports/tools/categories.json').read_text(encoding='utf-8'))
        self.git('add','reports/tools/categories.json')
        self.git('-c','user.name=PRD114 test','-c','user.email=prd114@example.invalid','commit','-qm','test fixture')
        head = self.git('rev-parse','HEAD').strip()
        self.write('docs/wiki/index.md','# wiki\n')
        self.write('docs/wiki/log.md','2026-11-03 start rev:'+head+'\n')
        self.page('categories/311_dev_tools')

    def git(self,*args):
        return subprocess.check_output(['git',*args],cwd=self.root,env=self.env,encoding='utf-8')

    def write(self,path,text):
        target = self.root/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(text,encoding='utf-8',newline='\n')
        return target

    def page(self,key):
        return self.write('docs/wiki/'+key+'.md','# map\n\n## 概要・責務\nKEEP PROSE\n\n## ファイル\nold\n\n## PRD の一覧\nold\n\n## 課題\n<!-- issues:begin -->\nold\n<!-- issues:end -->\nKEEP TAIL\n')

    @staticmethod
    def link(name):
        # 課題・PRD を指す欄は、Obsidian のリンクの形で書く。
        return '[['+name+']]'

    def issue(self,suffix='000001',drop=(),**changes):
        data = dict(id='issue-20261103-'+suffix,title='課題 '+suffix,status='未着手',raised='2026-11-03',phase='監査',scope='test',kind='成果物',
                    area=['311_dev_tools'],type='リファクタリング',domain='開発',ai_judge='未確認',ai_judged=None,docs=[],prd=None,priority='中',depends=[],parent=None,
                    value=5,urgency=5,risk=3,size=5,wsjf=2.6,resolved=None,resolution=None,memo=None,tags=[])
        data.update(changes)
        for key in drop:
            data.pop(key)
        self.write('docs/management/issues/'+data['id']+'.md','---\n'+yaml.safe_dump(data,allow_unicode=True,sort_keys=False)+'---\n\n## 背景\nfixture\n')
        return data['id']

    def cli(self,action='maps',success=True):
        result = subprocess.run([sys.executable,'-B',str(HELPER),action],cwd=self.root,env=self.env,encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr) if success else self.assertNotEqual(result.returncode,0)
        return json.loads(result.stdout) if success else result.stderr

    def test_maps_builds_index_and_issue_section_from_issue(self):
        # 単票から、INDEXと地図の両方への生成経路を通す。
        issue_id = self.issue()
        self.cli()
        # 2つの一覧への掲載と、地図の区画外の本文の保持を確かめる。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        page = (self.root/'docs/wiki/categories/311_dev_tools.md').read_text(encoding='utf-8')
        self.assertIn(issue_id,index)
        self.assertIn(issue_id,page)
        self.assertIn('KEEP PROSE',page)
        self.assertTrue(page.endswith('KEEP TAIL\n'))

    def test_index_shows_domain_type_and_prd_link_columns(self):
        # 仕事の種類・分類・PRD のリンクを持つ課題から、一覧の列を作る経路を通す。
        prd = 'docs/memo/120_probe_10prd.md'
        self.write(prd,'# probe\n')
        issue_id = self.issue(type='機能追加',domain='UI',prd=self.link('120_probe_10prd'))
        self.cli()
        # 列は対象・工程・領域・分類・種類・PRD の順で、PRD の列は PRD へのリンクになる。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        self.assertIn('\n| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |\n',index)
        row = next(line for line in index.splitlines() if issue_id+'](' in line)
        self.assertIn('| 成果物 | 監査 | 311_dev_tools | UI | 機能追加 | [PRD120](../../memo/120_probe_10prd.md) | 未確認 |',row)

    def test_adding_an_issue_leaves_other_rows_unchanged(self):
        # 高い点数と低い点数の 2 件で一覧と地図を作り、その間に並ぶ課題をあとから足す。
        self.issue('000001',value=13,urgency=13,risk=13,size=3,wsjf=13.0,ai_judge='残件無し',ai_judged='2026-11-05')
        self.issue('000002',value=1,urgency=1,risk=1,size=3,wsjf=1.0,ai_judge='残件無し',ai_judged='2026-11-07')
        self.cli()
        outputs = [self.root/'docs/management/issues/INDEX.md',self.root/'docs/wiki/categories/311_dev_tools.md']
        rows = lambda path: [line for line in path.read_text(encoding='utf-8').splitlines() if line.startswith('| [issue-')]
        before = {path:rows(path) for path in outputs}
        added = self.issue('000003',value=5,urgency=5,risk=5,size=3,wsjf=5.0,ai_judge='残件無し',ai_judged='2026-11-06')
        self.cli()
        # 位置の番号を持たないので、足した課題の行が間に入っても、前からある行の文字は変わらない。
        for path in outputs:
            with self.subTest(path=path.name):
                after = rows(path)
                self.assertEqual([line for line in after if added+'](' not in line],before[path])
                self.assertEqual(len(after),len(before[path])+(2 if path.name == 'INDEX.md' else 1))

    def test_index_defaults_to_current_work_without_closed_issues(self):
        # 本版と、既定の一覧から除く2つの状態を混在させる。
        active = self.issue()
        deferred = self.issue('000002',status='次版')
        resolved = self.issue('000003',status='解決',resolved='2026-11-03',resolution='対応済み')
        self.cli()
        # 引数なしの生成では、本版だけがINDEXへ現れる。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        self.assertIn(active,index)
        self.assertNotIn(deferred,index)
        self.assertNotIn(resolved,index)

    def index_section(self,heading):
        # INDEXの見出しから次の見出しまでを、その欄の本文として切り出す。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        self.assertEqual(index.count('\n## '+heading+'\n'),1,index)
        return index.split('\n## '+heading+'\n',1)[1].split('\n## ',1)[0]

    def test_index_lists_issues_judged_done_but_not_resolved_as_waiting(self):
        # 判定が済みを示すか、状態が解決かの組み合わせを並べる。
        waiting = self.issue('000001',ai_judge='残件無し',ai_judged='2026-11-05')
        deferred = self.issue('000002',status='次版',ai_judge='残件無し',ai_judged='2026-11-05')
        remaining = self.issue('000003',ai_judge='残件有り',ai_judged='2026-11-05')
        resolved = self.issue('000004',status='解決',resolved='2026-11-06',resolution='対応済み',ai_judge='残件無し',ai_judged='2026-11-05')
        self.cli()
        # 解決でない課題だけが、次版も含めて解決待ちへ出る。
        section = self.index_section('解決待ち')
        self.assertIn(waiting,section)
        self.assertIn(deferred,section)
        self.assertNotIn(remaining,section)
        self.assertNotIn(resolved,section)
        self.assertIn('該当する課題はありません。',self.index_section('再開候補'))

    def test_index_lists_resolved_issues_judged_remaining_after_resolution_as_reopen_candidates(self):
        # 解決の日付と判定の日付の前後を変えた解決済みの課題を並べる。
        later = self.issue('000001',status='解決',resolved='2026-11-05',resolution='対応済み',ai_judge='残件有り',ai_judged='2026-11-06')
        same_day = self.issue('000002',status='解決',resolved='2026-11-05',resolution='対応済み',ai_judge='残件有り',ai_judged='2026-11-05')
        stale = self.issue('000003',status='解決',resolved='2026-11-09',resolution='対応済み',ai_judge='残件有り',ai_judged='2026-11-04')
        active = self.issue('000004',ai_judge='残件有り',ai_judged='2026-11-06')
        self.cli()
        # 解決の日付以後に残件有りとなったものだけを再開候補へ出し、状態は書き換えない。
        section = self.index_section('再開候補')
        self.assertIn(later,section)
        self.assertIn(same_day,section)
        self.assertNotIn(stale,section)
        self.assertNotIn(active,section)
        self.assertIn('該当する課題はありません。',self.index_section('解決待ち'))
        record = (self.root/'docs/management/issues'/(later+'.md')).read_text(encoding='utf-8')
        self.assertIn('status: 解決',record)

    def test_dependency_inherits_wsjf_before_ordering_and_links_waiting_issue(self):
        # 自分の点数が低い前提と、その前提を待つ高得点の課題を用意する。
        prerequisite = self.issue('000003',value=1,urgency=1,risk=1,size=3,wsjf=1.0)
        independent = self.issue('000002',value=5,urgency=5,risk=5,size=3,wsjf=5.0)
        dependent = self.issue('000001',depends=[self.link(prerequisite)],value=13,urgency=13,risk=13,size=3,wsjf=13.0)
        self.cli()
        # 前提の順位には継承が効き、待つ課題から前提へのリンクが残る。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        rows = [line for line in index.splitlines() if line.startswith('| ') and '](' in line]
        self.assertIn(prerequisite,rows[0])
        self.assertIn('13.0（引継ぎ）',rows[0])
        self.assertIn(independent,rows[1])
        self.assertIn(dependent,rows[2])
        self.assertIn('['+prerequisite+']('+prerequisite+'.md)',rows[2])

    def test_priority_and_memo_are_free_developer_memos_kept_out_of_outputs(self):
        # 開発者が Obsidian などで直接書く 2 つの欄に、決まりの無い値を入れる。
        text_memo = self.issue('000001',priority='あとで',memo='人間メモ XYZ')
        list_memo = self.issue('000002',priority=None,memo=['メモ ABC','PQR'])
        self.cli()
        # 道具は止まらず、どちらの値も一覧と地図に出さない。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        page = (self.root/'docs/wiki/categories/311_dev_tools.md').read_text(encoding='utf-8')
        for output in (index,page):
            self.assertIn(text_memo,output)
            self.assertIn(list_memo,output)
            self.assertNotIn('優先度',output)
            for value in ('あとで','XYZ','ABC','PQR'):
                self.assertNotIn(value,output)

    def test_priority_does_not_change_the_order(self):
        # WSJF が同じ 2 件で、新しいほうに高い priority を付ける。
        newer = self.issue('000001',raised='2026-11-04',priority='高')
        older = self.issue('000002',raised='2026-11-03',priority='低')
        self.cli()
        # priority ではなく、起票日の古い順に並ぶ。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        rows = [line for line in index.splitlines() if line.startswith('| ') and '](' in line]
        self.assertIn(older,rows[0])
        self.assertIn(newer,rows[1])

    def test_obsidian_style_frontmatter_is_accepted(self):
        # Obsidian のプロパティ欄が書き直す形（引用符なし、日付と整数の WSJF、1 行 1 項目のリスト、空の値）で書く。
        dependency = self.issue('000002')
        self.write('docs/management/issues/issue-20261103-000001.md','\n'.join([
            '---','id: issue-20261103-000001','title: Obsidian の書き方の課題','status: 未着手','raised: 2026-11-03','phase: 監査',
            'scope: "core/x.cpp :: Foo::Bar"','kind: 成果物','area:','  - 311_dev_tools','type: リファクタリング','domain: 開発','ai_judge: 残件有り','ai_judged: 2026-11-09',
            'docs: []','prd:','priority: 高','depends:','  - "[['+dependency+']]"','parent:','value: 3','urgency: 2','risk: 1','size: 3','wsjf: 2','resolved:','resolution:','memo:','tags: []',
            '---','','## 背景','fixture','']))
        self.cli()
        # 値は同じに読まれ、整数で書いた WSJF も一覧では小数 1 桁で出る。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        row = next(line for line in index.splitlines() if 'issue-20261103-000001](' in line)
        self.assertIn('| 2.0 |',row)
        self.assertIn('Obsidian の書き方の課題',row)

    def outputs(self):
        # 失敗の前後で比べる派生出力の中身を集める。
        paths = [self.root/'docs/management/issues/INDEX.md',self.root/'docs/wiki/categories/311_dev_tools.md',self.root/'docs/wiki/log.md']
        return {path:path.read_bytes() for path in paths}

    def assert_rejected(self,field,**changes):
        # 1 件の課題だけを書き換え、欄の名前を示して止まり、どの出力も変えないことを確かめる。
        before = self.outputs()
        self.issue(**changes)
        error = self.cli(success=False)
        self.assertIn('issue-20261103-000001.md',error)
        self.assertIn(field,error)
        self.assertIn('expected',error)
        self.assertEqual(self.outputs(),before)

    def test_old_type_values_and_implemented_judgement_are_rejected(self):
        # 正常な出力を作ってから、なくした値を 1 つずつ書く。
        self.issue()
        self.cli()
        for value in ('索引','ずれ','改善','案'):
            with self.subTest(type=value):
                self.assert_rejected('type',type=value)
        with self.subTest(ai_judge='実装済み'):
            self.assert_rejected('ai_judge',ai_judge='実装済み',ai_judged='2026-11-09')
        # 7 つの種類の値は、どれも受け付ける。
        for value in ('機能追加','機能改善','性能改善','リファクタリング','不具合','不整合','調査'):
            with self.subTest(type=value,accepted=True):
                self.issue(type=value)
                self.cli()

    def test_issue_and_prd_fields_require_obsidian_links(self):
        # 依存先と PRD を置き、正常な出力を作る。
        self.write('docs/memo/120_probe_10prd.md','# probe\n')
        prerequisite = self.issue('000002')
        self.issue()
        self.cli()
        # ID やパスだけの書き方は止まる。
        for field,changes in (('depends',dict(depends=[prerequisite])),('parent',dict(parent=prerequisite)),
                              ('prd',dict(prd='docs/memo/120_probe_10prd.md')),('prd',dict(prd='120_probe_10prd'))):
            with self.subTest(field=field,value=changes[field]):
                self.assert_rejected(field,**changes)
        # 別名つき・パスつきのリンクは、末尾の名前で読む。
        self.issue(depends=[self.link('issues/'+prerequisite+'|前提')],parent=self.link(prerequisite+'|親'),
                   prd=self.link('docs/memo/120_probe_10prd|120'))
        self.cli()
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        row = next(line for line in index.splitlines() if 'issue-20261103-000001](' in line)
        self.assertIn('['+prerequisite+']('+prerequisite+'.md)',row)
        self.assertIn('[PRD120](../../memo/120_probe_10prd.md)',row)

    def test_prd_must_point_to_an_existing_prd_directly_under_memo(self):
        # 直下の PRD、過去の置き場の PRD、PRD でない文書を置く。
        self.write('docs/memo/120_probe_10prd.md','# probe\n')
        self.write('docs/memo/done/090_old_10prd.md','# old\n')
        self.write('docs/memo/120_probe_20impl.md','# plan\n')
        self.issue(prd=self.link('120_probe_10prd'))
        self.cli()
        # 過去の置き場・実在しない PRD・PRD でない文書は止まり、docs に同じ PRD を重ねても止まる。
        for changes in (dict(prd=self.link('090_old_10prd')),dict(prd=self.link('121_missing_10prd')),
                        dict(prd=self.link('120_probe_20impl'))):
            with self.subTest(prd=changes['prd']):
                self.assert_rejected('prd',**changes)
        with self.subTest(docs='same PRD'):
            self.assert_rejected('docs',prd=self.link('120_probe_10prd'),docs=['docs/memo/120_probe_10prd.md'])

    def test_resolution_is_required_only_for_resolved_issues(self):
        # 解決の課題に理由がある正常な出力を作る。
        self.issue(status='解決',resolved='2026-11-03',resolution='対応済み')
        self.cli()
        # 解決で理由が無いか決まりに無い値、解決でないのに理由がある場合は止まる。
        for changes in (dict(status='解決',resolved='2026-11-03',resolution=None),
                        dict(status='解決',resolved='2026-11-03',resolution='直した'),
                        dict(status='未着手',resolution='対応済み')):
            with self.subTest(**{k:str(v) for k,v in changes.items()}):
                self.assert_rejected('resolution',**changes)
        # 5 つの理由は、どれも受け付ける。
        for value in ('対応済み','対応しない','重複','再現しない','不要になった'):
            with self.subTest(resolution=value,accepted=True):
                self.issue(status='解決',resolved='2026-11-03',resolution=value)
                self.cli()

    def stats(self,success=True):
        # 統計の公開 CLI を一時 repo へ向けて回す。
        args = [sys.executable,'-B',str(REPO/'reports/tools/stats.py'),'--repo',str(self.root),'--no-cloc','--no-blame','--no-runs','--quiet']
        result = subprocess.run(args,cwd=self.root,env=self.env,encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr) if success else self.assertNotEqual(result.returncode,0)
        return json.loads((self.root/'reports/stats/stats.json').read_text(encoding='utf-8')) if success else result.stderr

    def test_prd_without_issue_is_listed_in_index_and_stops_stats(self):
        # 課題から指される PRD と、どの課題からも指されない PRD を直下に置く。
        self.write('docs/memo/120_probe_10prd.md','# PRD120: 指される\n')
        self.write('docs/memo/121_orphan_10prd.md','# PRD121: 指されない\n')
        self.issue(prd=self.link('120_probe_10prd'))
        # 一覧の道具は止まらず、指されない PRD だけを欄に出す。
        self.cli()
        section = self.index_section('課題の無い PRD')
        self.assertIn('[121_orphan_10prd](../../memo/121_orphan_10prd.md)',section)
        self.assertIn('PRD121: 指されない',section)
        self.assertNotIn('120_probe',section)
        # 統計は、指されない PRD を名指しして止まる。
        self.assertIn('121_orphan_10prd.md',self.stats(success=False))
        # 課題から指されれば、欄は空になり、統計も止まらない。
        self.issue('000002',prd=self.link('121_orphan_10prd'))
        self.cli()
        self.assertIn('該当する PRD はありません。',self.index_section('課題の無い PRD'))
        self.stats()

    def test_invalid_new_issue_fields_fail_without_changing_any_output(self):
        # 正常な生成結果を保存し、不正入力を渡す前の状態を固定する。
        self.issue()
        self.cli()
        paths = [self.root/'docs/management/issues/INDEX.md',self.root/'docs/wiki/categories/311_dev_tools.md',self.root/'docs/wiki/log.md']
        before = {path:path.read_bytes() for path in paths}
        # 追加欄を1つずつ壊し、拒否理由と全出力の不変を確かめる。
        for field,value in [('type','unknown'),('domain','unknown'),('domain',None),('domain',''),('ai_judge','unknown'),('ai_judge',None),('ai_judge',''),('area',['unknown']),('docs',['docs/missing.md'])]:
            with self.subTest(field=field,value=value):
                self.issue(**{field:value})
                error = self.cli(success=False)
                self.assertIn('issue-20261103-000001.md',error)
                self.assertIn(field,error)
                self.assertIn('expected',error)
                self.assertEqual({path:path.read_bytes() for path in paths},before)

    def test_ai_judged_must_match_whether_ai_judge_is_a_judgement(self):
        # 正常な生成結果を保存し、判定と日付の組み合わせだけを壊す。
        self.issue()
        self.cli()
        paths = [self.root/'docs/management/issues/INDEX.md',self.root/'docs/wiki/categories/311_dev_tools.md',self.root/'docs/wiki/log.md']
        before = {path:path.read_bytes() for path in paths}
        # 未確認に日付がある組と、判定に日付が無い組を、どちらも拒否する。
        for judge,judged in (('未確認','2026-11-09'),('残件有り',None),('残件無し',''),('判定不能',None)):
            with self.subTest(ai_judge=judge,ai_judged=judged):
                self.issue(ai_judge=judge,ai_judged=judged)
                error = self.cli(success=False)
                self.assertIn('ai_judged',error)
                self.assertEqual({path:path.read_bytes() for path in paths},before)
        # 4 つの値は、判定の有無に合った日付となら受け付ける。
        for judge,judged in (('未確認',None),('残件有り','2026-11-09'),('残件無し','2026-11-09'),('判定不能','2026-11-09')):
            with self.subTest(ai_judge=judge,ai_judged=judged,accepted=True):
                self.issue(ai_judge=judge,ai_judged=judged)
                self.cli()

    def test_active_wsjf_crosses_held_prerequisite_without_using_held_own_score(self):
        # 保留を間に挟み、保留自身の点数は活動中の課題より高くする。
        prerequisite = self.issue('000003',value=1,urgency=1,risk=1,size=3,wsjf=1.0)
        held = self.issue('000002',status='保留',depends=[self.link(prerequisite)],value=13,urgency=13,risk=13,size=3,wsjf=13.0)
        self.issue(depends=[self.link(held)],value=5,urgency=5,risk=5,size=3,wsjf=5.0)
        self.cli()
        # 末端の前提へ届くのは、活動中の課題を元にした点数だけである。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        row = next(line for line in index.splitlines() if '](' in line and prerequisite in line)
        self.assertIn('5.0（引継ぎ）',row)
        self.assertNotIn('13.0',row)

    def test_invalid_date_formats_fail_without_changing_any_output(self):
        # YAMLが日付として読む標準形で、派生ファイルを先に作る。
        from datetime import date
        self.issue(raised=date(2026,11,3))
        self.cli()
        paths = [self.root/'docs/management/issues/INDEX.md',self.root/'docs/wiki/categories/311_dev_tools.md',self.root/'docs/wiki/log.md']
        before = {path:path.read_bytes() for path in paths}
        # 3つの日付欄へ、整数・週日付・実在しない日付をそれぞれ渡す。
        for field in ('raised','resolved','ai_judged'):
            for value in (20261104,'2026-W45-3','2026-02-30'):
                with self.subTest(field=field,value=value):
                    changes = {field:value}
                    if field=='resolved':
                        changes['status'] = '解決'
                        changes['resolution'] = '対応済み'
                    if field=='ai_judged':
                        changes['ai_judge'] = '残件有り'
                    self.issue(**changes)
                    error = self.cli(success=False)
                    # 拒否理由の欄名と期待形式を示し、どの派生出力も更新しない。
                    self.assertIn(field,error)
                    self.assertIn('YYYY-MM-DD',error)
                    self.assertEqual({path:path.read_bytes() for path in paths},before)

    def test_same_blob_prd_is_pending_once_when_its_issues_become_done(self):
        # 起草時の本文を取り込み済みにし、同じblobのPRDと、それを指す課題を用意する。
        prd = 'docs/memo/115_probe_10prd.md'
        self.write(prd,'# probe\n')
        blob = self.git('hash-object','--',prd).strip()
        log = self.root/'docs/wiki/log.md'
        log.write_text(log.read_text(encoding='utf-8')+f'2026-11-03 ingest {prd} blob:{blob} → なし\n',encoding='utf-8')
        self.issue(prd=self.link('115_probe_10prd'))
        self.assertEqual(self.cli('scan')['pending'],[])
        # 本文を変えずに課題を残件無しと判定し、新しい取り込み候補になることを見る。
        self.issue(prd=self.link('115_probe_10prd'),ai_judge='残件無し',ai_judged='2026-11-03')
        pending = self.cli('scan')
        self.assertTrue(pending['eligible'])
        self.assertEqual([r['path'] for r in pending['pending']],[prd])
        self.assertEqual(pending['pending'][0]['implemented'],'2026-11-03')
        # 実装済みとして取り込んだあとは、判定し直して日付が動いても、候補へ戻らない。
        log.write_text(log.read_text(encoding='utf-8')+f'2026-11-03 ingest {prd} blob:{blob} implemented:2026-11-03 → なし\n',encoding='utf-8')
        self.assertEqual(self.cli('scan')['pending'],[])
        self.issue(prd=self.link('115_probe_10prd'),ai_judge='残件無し',ai_judged='2026-11-10')
        self.assertEqual(self.cli('scan')['pending'],[])

    def test_prd_is_implemented_only_when_all_its_work_is_done(self):
        # 取り込み済みの本文を持つPRDを置き、課題の組み合わせごとに、scan が実装済みとして拾うかを見る。
        prd = 'docs/memo/115_probe_10prd.md'
        self.write(prd,'# probe\n')
        blob = self.git('hash-object','--',prd).strip()
        log = self.root/'docs/wiki/log.md'
        log.write_text(log.read_text(encoding='utf-8')+f'2026-11-03 ingest {prd} blob:{blob} → なし\n',encoding='utf-8')
        link = self.link('115_probe_10prd')
        done = dict(prd=link,ai_judge='残件無し',ai_judged='2026-11-05')
        resolved = lambda reason,day: dict(prd=link,status='解決',resolved=day,resolution=reason)
        cases = [
            # 行2: やらないと決めた仕事（次版・対応しない・不要になった）が残れば、実装済みにしない。
            ('次版が残る',[done,dict(prd=link,status='次版')],None),
            ('対応しないが残る',[done,resolved('対応しない','2026-11-06')],None),
            ('不要になったが残る',[done,resolved('不要になった','2026-11-06')],None),
            # 行3: 直す仕事が元から無い課題だけなら、実装済みにしない。
            ('重複と再現しないだけ',[resolved('重複','2026-11-06'),resolved('再現しない','2026-11-07')],None),
            # 行4: 除いたあとがすべて済みなら実装済みで、日は判定日と解決日の最も新しい日。
            ('残件無しと対応済みと重複',[done,resolved('対応済み','2026-11-06'),resolved('重複','2026-11-09')],'2026-11-06'),
            ('対応済みの解決に残件有りが付いても解決として数える',[dict(resolved('対応済み','2026-11-04'),ai_judge='残件有り',ai_judged='2026-11-07')],'2026-11-07'),
            # 行5: 段階の途中（片方が残件有り）や未確認が残れば、実装済みにしない。
            ('段階の途中',[done,dict(prd=link,ai_judge='残件有り',ai_judged='2026-11-05')],None),
            ('未確認が残る',[done,dict(prd=link)],None),
        ]
        for name,issues,expected in cases:
            with self.subTest(case=name):
                for path in (self.root/'docs/management/issues').glob('issue-*.md'):
                    path.unlink()
                for number,changes in enumerate(issues,1):
                    self.issue(f'{number:06d}',**changes)
                pending = [r for r in self.cli('scan')['pending'] if r['path'] == prd]
                self.assertEqual([r.get('implemented') for r in pending],[expected] if expected else [])

    def test_unquoted_invalid_calendar_dates_report_field_without_changing_output(self):
        # 正常な出力を保存し、YAMLが日付型へ変換する前の入力を直接作る。
        self.issue()
        self.cli()
        paths = [self.root/'docs/management/issues/INDEX.md',self.root/'docs/wiki/categories/311_dev_tools.md',self.root/'docs/wiki/log.md']
        before = {path:path.read_bytes() for path in paths}
        for field in ('raised','resolved','ai_judged'):
            with self.subTest(field=field):
                self.issue()
                path = self.root/'docs/management/issues/issue-20261103-000001.md'
                text = path.read_text(encoding='utf-8')
                import re
                self.write(path.relative_to(self.root),re.sub(r'(?m)^'+field+r':.*$',field+': 2026-02-30',text))
                # 構築途中の失敗でも、欄と入力値を特定して全出力を保つ。
                error = self.cli(success=False)
                for detail in (path.name,field,'2026-02-30','YYYY-MM-DD'):
                    self.assertIn(detail,error)
                self.assertEqual({path:path.read_bytes() for path in paths},before)

    def test_stats_counts_issue_fields_and_derives_prd_states_from_their_issues(self):
        # 未追跡の現行PRDと過去のdoneを置き、PRDごとに状態の組み合わせの違う課題で指す。
        prds = {number:f'{number}_probe_10prd' for number in (111,112,113,114,115,116)}
        for name in prds.values():
            self.write(f'docs/memo/{name}.md','# probe\n')
        self.write('docs/memo/done/090_legacy_10prd.md','# legacy\n')
        resolved = lambda reason: dict(status='解決',resolved='2026-11-03',resolution=reason)
        issues = [
            (111,dict(resolved('対応しない'),type='機能追加',domain='UI')),        # すべて解決で、対応済みが無い → 見送り
            (112,dict(resolved('対応済み'),type='不具合',domain='Env')),
            (112,dict(resolved('重複'))),                                           # すべて解決 → 解決
            (113,dict(resolved('対応済み'))),
            (113,dict(status='次版')),                                              # 解決か次版だけ → 次版
            (114,dict(status='対応中')),
            (114,dict(status='調査中')),
            (114,dict(status='保留')),                                              # 最も進んだもの → 対応中
            (115,dict(status='保留')),
            (115,dict()),                                                           # 保留がある → 保留
            (116,dict()),                                                           # それ以外 → 未着手
        ]
        for number,(prd,changes) in enumerate(issues,1):
            self.issue(f'{number:06d}',prd=self.link(prds[prd]),**changes)
        data = self.stats()
        # 課題の種類・分類・完了の理由と、現行PRDの状態、過去の保管場所を別々に集計している。
        self.assertNotIn('registry',data['health'])
        self.assertEqual(data['health']['issues']['by_type'],{'機能追加':1,'不具合':1,'リファクタリング':9})
        self.assertEqual(data['health']['issues']['by_domain'],{'UI':1,'Env':1,'開発':9})
        self.assertEqual(data['health']['issues']['by_resolution'],{'対応しない':1,'対応済み':2,'重複':1})
        self.assertEqual(data['process']['by_status'],{'見送り':1,'解決':1,'次版':1,'対応中':1,'保留':1,'未着手':1})
        self.assertEqual(data['process']['historical_folders']['done'],1)

    def test_stats_counts_free_priority_values_without_failing(self):
        # 管理外のメモの priority に、決まりの無い値・空・リスト・欄の無いものを混ぜる。
        self.issue('000001',priority='高')
        self.issue('000002',priority='あとで')
        self.issue('000003',priority=None)
        self.issue('000004',drop=('priority',))
        self.issue('000005',priority=['a','b'])
        self.issue('000006',priority='低',status='解決',resolved='2026-11-03',resolution='対応済み')
        args = [sys.executable,'-B',str(REPO/'reports/tools/stats.py'),'--repo',str(self.root),'--no-cloc','--no-blame','--no-runs','--quiet']
        result = subprocess.run(args,cwd=self.root,env=self.env,encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        # 本版の課題だけを、空と欄の無いものは「（無し）」、リストは文字にして数える。
        data = json.loads((self.root/'reports/stats/stats.json').read_text(encoding='utf-8'))
        self.assertEqual(data['health']['issues']['by_priority'],{'高':1,'あとで':1,'（無し）':2,"['a', 'b']":1})

    def test_multi_area_issue_does_not_expand_application_scope(self):
        # 2つの機能カテゴリとアプリ自身の課題を同時に置く。
        metric = self.page('categories/317_metrics')
        app = self.page('apps/110_runner_app')
        multi = self.issue(area=['311_dev_tools','317_metrics'])
        own = self.issue('000002',area=['110_runner_app'])
        self.cli()
        # 複数areaは拾い、アプリの課題区画へ機能カテゴリを拡大しない。
        self.assertIn(multi,metric.read_text(encoding='utf-8'))
        self.assertNotIn(multi,app.read_text(encoding='utf-8'))
        self.assertIn(own,app.read_text(encoding='utf-8'))

    def test_stats_allows_issues_of_two_prds_to_reference_the_same_design_document(self):
        # 別々のPRDを指す課題から、同じ設計文書も docs で参照させ、1 本のPRDを 2 件の課題で指す。
        shared = 'docs/design/160_tools.jp.md'
        self.write(shared,'# shared design\n')
        for number,suffixes in ((115,('000001','000003')),(116,('000002',))):
            self.write(f'docs/memo/{number}_probe_10prd.md','# probe\n')
            for suffix in suffixes:
                self.issue(suffix,prd=self.link(f'{number}_probe_10prd'),docs=[shared])
        # 文書の共有と、1 本のPRDを指す課題が複数あることは、統計を止めない。PRDそのものは2件として数えられる。
        self.assertEqual(self.stats()['process']['prd_count'],2)

    def test_bad_marker_in_later_page_leaves_all_outputs_unchanged(self):
        # 先に正常な出力を作り、あとで読む地図の目印だけを壊す。
        self.issue()
        self.cli()
        page = self.page('categories/317_metrics')
        page.write_text(page.read_text(encoding='utf-8').replace('<!-- issues:end -->','<!-- broken -->'),encoding='utf-8')
        paths = [self.root/'docs/management/issues/INDEX.md',self.root/'docs/wiki/categories/311_dev_tools.md',page]
        before = {path:path.read_bytes() for path in paths}
        # 後続ページで失敗しても、INDEXと先行ページへ書き戻していない。
        self.assertIn('317_metrics.md',self.cli(success=False))
        self.assertEqual({path:path.read_bytes() for path in paths},before)

    def test_regeneration_is_idempotent_and_does_not_report_or_write_changes(self):
        # 正常な生成後の内容と更新時刻を保存する。
        self.issue()
        self.cli()
        paths = [self.root/'docs/management/issues/INDEX.md',self.root/'docs/wiki/categories/311_dev_tools.md']
        before = {path:(path.read_bytes(),path.stat().st_mtime_ns) for path in paths}
        # 同じ入力で作り直しても、更新の報告もファイルの書き込みも無い。
        self.assertEqual(self.cli()['updated'],[])
        self.assertEqual({path:(path.read_bytes(),path.stat().st_mtime_ns) for path in paths},before)

    def test_maps_excludes_deleted_tracked_files(self):
        # Gitでは追跡中だが、作業ツリーで退役させたファイルを用意する。
        path = self.write('reports/tools/retired.py','# retired\n')
        self.git('add','reports/tools/retired.py')
        self.git('-c','user.name=PRD114 test','-c','user.email=prd114@example.invalid','commit','-qm','fixture file')
        path.unlink()
        self.cli()
        # 生成したファイル一覧に、存在しない退役資材を載せない。
        page = (self.root/'docs/wiki/categories/311_dev_tools.md').read_text(encoding='utf-8')
        self.assertNotIn('retired.py',page)

    def test_cycle_and_missing_dependency_are_reported_without_losing_rows(self):
        # 相互の依存と、実在しない依存先を持つ課題を別々に置く。
        first = self.issue(depends=[self.link('issue-20261103-000002')])
        second = self.issue('000002',depends=[self.link(first)])
        third = self.issue('000003',depends=[self.link('issue-20261103-ffffff')])
        self.cli()
        # すべての課題を残し、注意と待ちの区分で問題を示す。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        for key in (first,second,third):
            self.assertIn(key,index)
        self.assertIn('循環',index)
        self.assertIn('（無い）',index)
        self.assertIn('待ち 3',index)

    def test_cycle_through_resolved_issue_is_still_waiting(self):
        # 直接の依存先が解決済みでも、依存をたどると自分へ戻る形にする。
        first = self.issue(depends=[self.link('issue-20261103-000002')])
        self.issue('000002',status='解決',resolved='2026-11-03',resolution='対応済み',depends=[self.link(first)])
        self.cli()
        # 閉じた課題は一覧に出さず、残る課題を循環中の待ちとして示す。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        self.assertIn('循環',index)
        self.assertIn('着手できる 0 / 待ち 1',index)


if __name__ == '__main__':
    unittest.main()
