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

    def issue(self,suffix='000001',**changes):
        data = dict(id='issue-20261103-'+suffix,title='課題 '+suffix,status='未着手',raised='2026-11-03',phase='監査',scope='test',kind='成果物',
                    area=['311_dev_tools'],type='改善',ai_judge='未確認',ai_judged=None,docs=[],priority='中',depends=[],parent=None,
                    value=5,urgency=5,risk=3,size=5,wsjf=2.6,resolved=None)
        data.update(changes)
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

    def test_index_defaults_to_current_work_without_closed_issues(self):
        # 本版と、既定の一覧から除く2つの状態を混在させる。
        active = self.issue()
        deferred = self.issue('000002',status='次版')
        resolved = self.issue('000003',status='解決',resolved='2026-11-03')
        self.cli()
        # 引数なしの生成では、本版だけがINDEXへ現れる。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        self.assertIn(active,index)
        self.assertNotIn(deferred,index)
        self.assertNotIn(resolved,index)

    def test_dependency_inherits_wsjf_before_ordering_and_links_waiting_issue(self):
        # 自分の点数が低い前提と、その前提を待つ高得点の課題を用意する。
        prerequisite = self.issue('000003',value=1,urgency=1,risk=1,size=3,wsjf=1.0)
        independent = self.issue('000002',value=5,urgency=5,risk=5,size=3,wsjf=5.0)
        dependent = self.issue('000001',depends=[prerequisite],value=13,urgency=13,risk=13,size=3,wsjf=13.0)
        self.cli()
        # 前提の順位には継承が効き、待つ課題から前提へのリンクが残る。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        rows = [line for line in index.splitlines() if line.startswith('| ') and '](' in line]
        self.assertIn(prerequisite,rows[0])
        self.assertIn('13.0（引継ぎ）',rows[0])
        self.assertIn(independent,rows[1])
        self.assertIn(dependent,rows[2])
        self.assertIn('['+prerequisite+']('+prerequisite+'.md)',rows[2])

    def test_invalid_new_issue_fields_fail_without_changing_any_output(self):
        # 正常な生成結果を保存し、不正入力を渡す前の状態を固定する。
        self.issue()
        self.cli()
        paths = [self.root/'docs/management/issues/INDEX.md',self.root/'docs/wiki/categories/311_dev_tools.md',self.root/'docs/wiki/log.md']
        before = {path:path.read_bytes() for path in paths}
        # 追加欄を1つずつ壊し、拒否理由と全出力の不変を確かめる。
        for field,value in [('type','unknown'),('ai_judge','unknown'),('ai_judge',None),('ai_judge',''),('area',['unknown']),('docs',['docs/missing.md'])]:
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
        for judge,judged in (('未確認','2026-11-09'),('残件有り',None),('残件無し',''),('判定不能',None),('実装済み',None)):
            with self.subTest(ai_judge=judge,ai_judged=judged):
                self.issue(ai_judge=judge,ai_judged=judged)
                error = self.cli(success=False)
                self.assertIn('ai_judged',error)
                self.assertEqual({path:path.read_bytes() for path in paths},before)
        # 5 つの値は、判定の有無に合った日付となら受け付ける。
        for judge,judged in (('未確認',None),('残件有り','2026-11-09'),('残件無し','2026-11-09'),('判定不能','2026-11-09'),('実装済み','2026-11-09')):
            with self.subTest(ai_judge=judge,ai_judged=judged,accepted=True):
                self.issue(ai_judge=judge,ai_judged=judged)
                self.cli()

    def test_active_wsjf_crosses_held_prerequisite_without_using_held_own_score(self):
        # 保留を間に挟み、保留自身の点数は活動中の課題より高くする。
        prerequisite = self.issue('000003',value=1,urgency=1,risk=1,size=3,wsjf=1.0)
        held = self.issue('000002',status='保留',depends=[prerequisite],value=13,urgency=13,risk=13,size=3,wsjf=13.0)
        self.issue(depends=[held],value=5,urgency=5,risk=5,size=3,wsjf=5.0)
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
                    if field=='ai_judged':
                        changes['ai_judge'] = '残件有り'
                    self.issue(**changes)
                    error = self.cli(success=False)
                    # 拒否理由の欄名と期待形式を示し、どの派生出力も更新しない。
                    self.assertIn(field,error)
                    self.assertIn('YYYY-MM-DD',error)
                    self.assertEqual({path:path.read_bytes() for path in paths},before)

    def test_same_blob_prd_is_pending_when_index_issue_becomes_implemented(self):
        # 起草時の本文を取り込み済みにし、同じblobのPRDを用意する。
        prd = 'docs/memo/115_probe_10prd.md'
        self.write(prd,'# probe\n')
        blob = self.git('hash-object','--',prd).strip()
        log = self.root/'docs/wiki/log.md'
        log.write_text(log.read_text(encoding='utf-8')+f'2026-11-03 ingest {prd} blob:{blob} → なし\n',encoding='utf-8')
        self.issue(type='索引',docs=[prd])
        self.assertEqual(self.cli('scan')['pending'],[])
        # 本文を変えずに実装済みと判定し、新しい取り込み候補になることを見る。
        self.issue(type='索引',docs=[prd],ai_judge='実装済み',ai_judged='2026-11-03')
        pending = self.cli('scan')
        self.assertTrue(pending['eligible'])
        self.assertEqual([r['path'] for r in pending['pending']],[prd])
        self.assertEqual(pending['pending'][0]['implemented'],'2026-11-03')
        # 判定日も取り込んだあとは、再度候補へ戻らない。
        log.write_text(log.read_text(encoding='utf-8')+f'2026-11-03 ingest {prd} blob:{blob} implemented:2026-11-03 → なし\n',encoding='utf-8')
        self.assertEqual(self.cli('scan')['pending'],[])

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

    def test_stats_counts_issue_types_and_prd_states_from_untracked_working_files(self):
        # 未追跡の現行PRDと過去のdoneを、索引issueの人の状態と組み合わせる。
        self.write('docs/memo/115_probe_10prd.md','# active\n')
        self.write('docs/memo/116_closed_10prd.md','# closed\n')
        self.write('docs/memo/done/117_legacy_10prd.md','# legacy\n')
        self.issue(type='索引',status='保留',docs=['docs/memo/115_probe_10prd.md'])
        self.issue('000002',type='索引',status='解決',resolved='2026-11-03',docs=['docs/memo/116_closed_10prd.md'])
        # 統計の公開CLIを一時repoへ向け、実データを再生成する。
        args = [sys.executable,'-B',str(REPO/'reports/tools/stats.py'),'--repo',str(self.root),'--no-cloc','--no-blame','--no-runs','--quiet']
        result = subprocess.run(args,cwd=self.root,env=self.env,encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        # 課題の性質・現行PRDの状態・過去の保管場所を別々に集計している。
        data = json.loads((self.root/'reports/stats/stats.json').read_text(encoding='utf-8'))
        self.assertNotIn('registry',data['health'])
        self.assertEqual(data['health']['issues']['by_type'],{'索引':2})
        self.assertEqual(data['process']['by_status'],{'保留':1,'解決':1})
        self.assertEqual(data['process']['historical_folders']['done'],1)

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

    def test_stats_allows_two_prd_indexes_to_reference_the_same_design_document(self):
        # 別々のPRDの索引から、同じ設計文書も参照させる。
        shared = 'docs/design/160_tools.jp.md'
        self.write(shared,'# shared design\n')
        for number,suffix in ((115,'000001'),(116,'000002')):
            prd = f'docs/memo/{number}_probe_10prd.md'
            self.write(prd,'# probe\n')
            self.issue(suffix,type='索引',docs=[prd,shared])
        # 統計CLIで、文書の共有がPRDの一対一制約に引っ掛からないことを見る。
        args = [sys.executable,'-B',str(REPO/'reports/tools/stats.py'),'--repo',str(self.root),'--no-cloc','--no-blame','--no-runs','--quiet']
        result = subprocess.run(args,cwd=self.root,env=self.env,encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        # PRDそのものは2件として数えられる。
        data = json.loads((self.root/'reports/stats/stats.json').read_text(encoding='utf-8'))
        self.assertEqual(data['process']['prd_count'],2)

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
        first = self.issue(depends=['issue-20261103-000002'])
        second = self.issue('000002',depends=[first])
        third = self.issue('000003',depends=['issue-20261103-ffffff'])
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
        first = self.issue(depends=['issue-20261103-000002'])
        self.issue('000002',status='解決',resolved='2026-11-03',depends=[first])
        self.cli()
        # 閉じた課題は一覧に出さず、残る課題を循環中の待ちとして示す。
        index = (self.root/'docs/management/issues/INDEX.md').read_text(encoding='utf-8')
        self.assertIn('循環',index)
        self.assertIn('着手できる 0 / 待ち 1',index)


if __name__ == '__main__':
    unittest.main()
