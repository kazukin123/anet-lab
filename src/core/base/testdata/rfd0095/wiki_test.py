"""wiki skill の機械処理を利用側の呼び出しから検証する。"""
from pathlib import Path
import json
import os
import importlib.util
from unittest.mock import patch
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

REPO = Path(__file__).resolve().parents[5]
HELPER = REPO/'reports/tools/wiki_support.py'


class WikiSupportTest(unittest.TestCase):
    def setUp(self):
        # 全Git呼出しに専用indexを継承させ、HEADのtreeで初期化する。
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        env=patch.dict(os.environ,{'GIT_INDEX_FILE':str(Path(temp.name)/'index')})
        env.start()
        self.addCleanup(env.stop)
        subprocess.run(['git','read-tree','HEAD'],cwd=REPO,check=True,capture_output=True)

    def test_scan_reports_untracked_material_without_writing(self):
        # 実repoのGit読み取り経路を通し、作成した1ファイルだけ片付ける。
        path = REPO/'docs/experiments'/('rfd0095_probe_'+uuid.uuid4().hex+'.md')
        path.write_text('# probe\n\n観測したこと。\n', encoding='utf-8')
        self.addCleanup(path.unlink)
        log_before = (REPO/'docs/wiki/log.md').read_bytes()
        result = subprocess.run([sys.executable,'-B',str(HELPER),'scan'],cwd=REPO,
                                encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertIn(path.relative_to(REPO).as_posix(),[r['path'] for r in data['pending']])
        self.assertEqual((REPO/'docs/wiki/log.md').read_bytes(),log_before)

    def test_scan_reports_material_unchanged_since_start_and_clears_recorded_blob(self):
        # 開始時から変わらない既存材料も、全体走査の公開CLIで検出する。
        root = self.fixture()
        path = 'docs/adr/0001-default-dqn-tbo-scope.md'
        blob = self.material(root, path, (REPO/path).read_text(encoding='utf-8'))
        data = self.run_helper(root)
        self.assertIn(path, [row['path'] for row in data['pending']])
        with (root/'docs/wiki/log.md').open('a', encoding='utf-8') as log:
            log.write('2026-10-23 ingest '+path+' blob:'+blob+' → なし\n')
        self.assertNotIn(path, [row['path'] for row in self.run_helper(root)['pending']])

    def test_fixture_selects_its_own_git_index(self):
        root=self.fixture()
        env=dict(os.environ)
        env.pop('GIT_INDEX_FILE',None)
        original=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],
                                             cwd=REPO,text=True,env=env).strip())
        if not original.is_absolute():
            original=REPO/original
        original=original.resolve()
        # 環境変数の指定値だけでなく、Gitが実際に参照するindexを確認する。
        actual=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],
                                           cwd=root,text=True).strip())
        if not actual.is_absolute():
            actual=root/actual
        actual=actual.resolve()
        self.assertNotEqual(actual,original)
        self.assertTrue(actual.is_file())
        files=subprocess.check_output(['git','ls-files','-z'],cwd=root,text=True)
        self.assertIn('reports/tools/stats.py',files.split('\0'))

    def fixture(self):
        # object DBとrefは読むだけ。indexはsetUpで分離した一時ファイルを使う。
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        gitdir = subprocess.check_output(['git','rev-parse','--absolute-git-dir'],cwd=REPO,text=True).strip()
        (root/'.git').write_text('gitdir: '+gitdir+'\n',encoding='utf-8')
        (root/'docs/wiki').mkdir(parents=True)
        (root/'docs/wiki/index.md').write_text('# index\n',encoding='utf-8')
        rev = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        (root/'docs/wiki/log.md').write_text('2026-10-22 start rev:'+rev+'\n',encoding='utf-8')
        return root

    def material(self, root, path, text):
        dest = root/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(text,encoding='utf-8')
        return subprocess.check_output(['git','hash-object','--',str(dest)],cwd=REPO,text=True).strip()

    def run_helper(self, root, action='scan'):
        result = subprocess.run([sys.executable,'-B',str(HELPER),action],cwd=root,
                                encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_blob_state_ignores_fix_and_treats_done_moves_as_pending(self):
        root = self.fixture()
        a=self.material(root,'docs/experiments/changed_probe.md','# changed')
        b=self.material(root,'reports/moved_probe.md','# moved')
        c=self.material(root,'docs/memo/done/0998_probe_10rfd.md','# done')
        d=self.material(root,'reports/already_probe.md','# already')
        with (root/'docs/wiki/log.md').open('a',encoding='utf-8') as log:
            log.write('2026-10-22 ingest docs/experiments/changed_probe.md blob:'+'0'*40+' → なし\n')
            log.write('2026-10-22 fix docs/experiments/changed_probe.md blob:'+a+' → なし\n')
            log.write('2026-10-22 ingest reports/old_probe.md blob:'+b+' → なし\n')
            log.write('2026-10-22 ingest docs/memo/0998_probe_10rfd.md blob:'+c+' → なし\n')
            log.write('2026-10-22 writeback reports/already_probe.md blob:'+d+' → なし\n')
        data=self.run_helper(root)
        self.assertEqual([r['path'] for r in data['pending']],
                         ['docs/experiments/changed_probe.md','docs/memo/done/0998_probe_10rfd.md'])
        self.assertEqual(data['pending'][0]['state'],'更新あり')
        self.assertEqual([r['path'] for r in data['moved']],['reports/moved_probe.md'])

    def test_map_regeneration_includes_untracked_group_without_touching_prose(self):
        root=self.fixture()
        (root/'reports/tools').mkdir(parents=True)
        shutil.copyfile(REPO/'reports/tools/categories.json',root/'reports/tools/categories.json')
        shutil.copytree(REPO/'docs/memo',root/'docs/memo',dirs_exist_ok=True)
        for name in ['Rfd0095Probe.hpp','Rfd0095Probe.cpp','Rfd0095Probe_test.cpp']:
            self.material(root,'src/apps/runner/src/'+name,'// probe')
        page=root/'docs/wiki/apps/110_runner_app.md'
        page.parent.mkdir(parents=True)
        page.write_text('# app\n\n## 概要・責務\nKEEP PROSE\n\n## ファイル\nold\n\n'
                        '## RFD の一覧\nold\n\n## 課題\n<!-- issues:begin -->\nold\n<!-- issues:end -->\nKEEP OUTSIDE\n',encoding='utf-8')
        self.run_helper(root,'maps')
        text=page.read_text(encoding='utf-8')
        row=next(line for line in text.splitlines() if 'Rfd0095Probe.hpp' in line)
        self.assertIn('Rfd0095Probe.cpp',row)
        self.assertIn('Rfd0095Probe_test.cpp',row)
        self.assertIn('KEEP PROSE',text)
        self.assertTrue(text.endswith('KEEP OUTSIDE\n'))
        self.assertIn('課題はありません。',text)
        first=text
        self.run_helper(root,'maps')
        self.assertEqual(page.read_text(encoding='utf-8'),first)

    def test_only_unannotated_source_links_make_housekeeping_eligible(self):
        root=self.fixture()
        blob=self.material(root,'docs/memo/done/0097_probe_link_10rfd.md','# source')
        with (root/'docs/wiki/log.md').open('a',encoding='utf-8') as log:
            log.write('2026-10-22 ingest docs/memo/done/0097_probe_link_10rfd.md blob:'+blob+' → なし\n')
        page=root/'docs/wiki/apps/110_runner_app.md'
        page.parent.mkdir(parents=True)
        page.write_text('# app\n\n## 概要・責務\n- claim [RFD](../../memo/0999_probe_link_10rfd.md)\n\n'
                        '## ファイル\n- [gone](../../../src/apps/runner/src/Gone.cpp)\n',encoding='utf-8')
        data=self.run_helper(root)
        self.assertEqual(len(data['broken_links']),1)
        self.assertEqual(data['broken_links'][0]['candidates'],['docs/memo/done/0097_probe_link_10rfd.md'])
        self.assertTrue(data['eligible'])
        page.write_text(page.read_text(encoding='utf-8').replace('0999_probe_link_10rfd.md)',
                        '0999_probe_link_10rfd.md)（出典のリンク切れ）'),encoding='utf-8')
        self.assertFalse(self.run_helper(root)['eligible'])
        self.material(root,'docs/management/issues/INDEX.md','# derived')
        self.material(root,'reports/housekeeping/runs.jsonl','{}')
        self.assertFalse(self.run_helper(root)['eligible'])

    def test_maps_keep_running_and_report_unresolved_rfd_numbers(self):
        spec=importlib.util.spec_from_file_location('rfd0095_wiki_support',HELPER)
        helper=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        for number, count in [('931',0),('999',2)]:
            with self.subTest(number=number, sources=count), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                (root/'reports/tools').mkdir(parents=True)
                shutil.copyfile(REPO/'reports/tools/categories.json',root/'reports/tools/categories.json')
                for n in range(count):
                    source=root/f'docs/memo/0{number}_probe{n}_10rfd.md'
                    source.parent.mkdir(parents=True,exist_ok=True)
                    source.write_text('# probe\n',encoding='utf-8')
                pages=['docs/wiki/apps/110_runner_app.md','docs/wiki/categories/311_dev_tools.md']
                for q in pages:
                    page=root/q;page.parent.mkdir(parents=True,exist_ok=True)
                    page.write_text('# map\n\n## ファイル\nold\n\n## RFD の一覧\nold\n\n'
                                    '## 課題\n<!-- issues:begin -->\nold\n<!-- issues:end -->\nKEEP\n',encoding='utf-8')
                self.material(root,'src/apps/runner/src/Rfd0095Probe.cpp','// probe')
                self.material(root,'apps/tools/rfd0095_probe.py','# probe')
                # 履歴は外部境界のGitだけ差し替える。本文生成と全ページ更新を通す。
                # 件名は RFD0195 より前の scope の形（PRDnnn）で書き、旧い記号の番号も解決できない番号として報告することを見る。
                def history(_root,*args):
                    return (self.log_output([('2026-11-01',f'feat(PRD{number}): probe')],
                                            ['src/apps/runner/src/Rfd0095Probe.cpp','apps/tools/rfd0095_probe.py'])
                            if args[0]=='log' else
                            'src/apps/runner/src/Rfd0095Probe.cpp\0apps/tools/rfd0095_probe.py\0')
                with patch.object(helper,'git',side_effect=history):
                    result=helper.regenerate_maps(root)
                self.assertEqual(set(result['updated']),set(pages)|{'docs/management/issues/INDEX.md'})
                self.assertEqual(result['unresolved_rfds'],[
                    {'page':q,'rfd':'RFD0'+number,'matches':count} for q in pages])
                for q in pages:
                    text=(root/q).read_text(encoding='utf-8')
                    self.assertIn('- RFD0'+number+' — （RFD を一意に決められない）',text)
                    self.assertNotIn('[RFD0'+number+']',text)
                    self.assertTrue(text.endswith('KEEP\n'))

    def test_maps_count_pending_commit_only_on_pages_it_touches(self):
        # コミットの準備が渡す予定のコミットは、触るファイルが当たるページの RFD の一覧にだけ足す。
        spec=importlib.util.spec_from_file_location('rfd0095_wiki_support',HELPER)
        helper=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'reports/tools').mkdir(parents=True)
            shutil.copyfile(REPO/'reports/tools/categories.json',root/'reports/tools/categories.json')
            source=root/'docs/memo/0997_pending_probe_10rfd.md'
            source.parent.mkdir(parents=True)
            source.write_text('# pending probe\n',encoding='utf-8')
            pages=['docs/wiki/apps/110_runner_app.md','docs/wiki/categories/311_dev_tools.md']
            for q in pages:
                page=root/q;page.parent.mkdir(parents=True,exist_ok=True)
                page.write_text('# map\n\n## ファイル\nold\n\n## RFD の一覧\nold\n\n'
                                '## 課題\n<!-- issues:begin -->\nold\n<!-- issues:end -->\nKEEP\n',encoding='utf-8')
            self.material(root,'src/apps/runner/src/Rfd0095Probe.cpp','// probe')
            self.material(root,'apps/tools/rfd0095_probe.py','# probe')
            # 履歴は空。予定のコミットだけが RFD 番号を持つ。
            def history(_root,*args):
                return ('' if args[0]=='log' else
                        'src/apps/runner/src/Rfd0095Probe.cpp\0apps/tools/rfd0095_probe.py\0')
            with patch.object(helper,'git',side_effect=history):
                helper.regenerate_maps(root,'feat(RunnerApp): probe。 RFD0997 #3',['src\\apps\\runner\\src\\Rfd0095Probe.cpp'])
            app=(root/pages[0]).read_text(encoding='utf-8')
            tools=(root/pages[1]).read_text(encoding='utf-8')
            self.assertIn('- [RFD0997](../../memo/0997_pending_probe_10rfd.md) — pending probe',app)
            self.assertNotIn('RFD0997',tools)
            # 予定を渡さなければ、履歴に無い番号は出ない。
            with patch.object(helper,'git',side_effect=history):
                helper.regenerate_maps(root)
            self.assertNotIn('RFD0997',(root/pages[0]).read_text(encoding='utf-8'))

    def renumber_root(self, rfds):
        # RFD0131 の振り直しの行を持つ RFD と、RFD の一覧を作り直すページを一時の置き場に作る。
        spec=importlib.util.spec_from_file_location('rfd0131_wiki_support',HELPER)
        helper=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        (root/'reports/tools').mkdir(parents=True)
        shutil.copyfile(REPO/'reports/tools/categories.json',root/'reports/tools/categories.json')
        for path,text in rfds.items():
            self.material(root,path,text)
        page='docs/wiki/apps/110_runner_app.md'
        (root/page).parent.mkdir(parents=True,exist_ok=True)
        (root/page).write_text('# map\n\n## ファイル\nold\n\n## RFD の一覧\nold\n\n'
                               '## 課題\n<!-- issues:begin -->\nold\n<!-- issues:end -->\n',encoding='utf-8')
        self.material(root,'src/apps/runner/src/Rfd0131Probe.cpp','// probe')
        return helper,root,page

    @staticmethod
    def log_output(commits, files):
        # git log -M --name-status -z の出力の形を作る。どのコミットも files を変えたことにする。
        return ''.join('\x01'+date+'\t'+subject+'\0\n'+''.join('M\0'+f+'\0' for f in files)
                       for date, subject in commits)

    def renumber_maps(self, helper, root, history, pending=None):
        # 履歴は外部境界の Git だけを差し替え、日付と件名の組の行を、プローブのファイルを変えたコミットとして返す。
        commits=[tuple(line.split('\t',1)) for line in history.splitlines() if line]
        log=self.log_output(commits,['src/apps/runner/src/Rfd0131Probe.cpp'])
        def git(_root,*args):
            return (log if args[0]=='log' else 'src/apps/runner/src/Rfd0131Probe.cpp\0')
        with patch.object(helper,'git',side_effect=git):
            if pending:
                return helper.regenerate_maps(root,pending,['src/apps/runner/src/Rfd0131Probe.cpp'])
            return helper.regenerate_maps(root)

    def test_maps_list_rfd_numbers_of_four_digit_files(self):
        # 件名の末尾の RFDnnnn を読み、0 埋めの 4 桁のファイル名の文書を「RFD の一覧」に載せる（RFD0195 D2・D17）。
        helper,root,page=self.renumber_root({'docs/memo/0997_rfd_probe_10rfd.md':'# rfd probe\n'})
        result=self.renumber_maps(helper,root,'2026-11-24\tfeat(RunnerApp): probe。 RFD0997 #3\n')
        self.assertEqual(result['unresolved_rfds'],[])
        text=(root/page).read_text(encoding='utf-8')
        self.assertIn('## RFD の一覧\n\n点検が作り直す。手で直さない。\n\n'
                      '- [RFD0997](../../memo/0997_rfd_probe_10rfd.md) — rfd probe\n',text)

    def test_maps_read_old_prd_and_new_rfd_numbers_by_value(self):
        # RFD0195 より前の件名の PRDnnn（3 桁）も読み、値で同じ 4 桁の文書に当てる。scope と末尾の並びの両方から読み、
        # RFD195（3 桁）と PRD0195（旧い記号の 4 桁）は番号の記号に当たらないので読まない（RFD0195 D17・§9.1）。
        helper,root,page=self.renumber_root({
            'docs/memo/0997_old_probe_10rfd.md':'# old symbol\n',
            'docs/memo/0918_scope_probe_10rfd.md':'# scope\n',
            'docs/memo/0195_tail_probe_10rfd.md':'# tail\n',
            'docs/memo/0196_none_probe_10rfd.md':'# none\n'})
        result=self.renumber_maps(helper,root,'2026-11-22\tdocs(RunnerApp): c。 RFD196 PRD0196 #7\n'
                                               '2026-11-21\tfix(PRD918): b。 RFD0195 #7\n'
                                               '2026-11-20\tfeat(RunnerApp): a。 PRD997 #3\n')
        self.assertEqual(result['unresolved_rfds'],[])
        text=(root/page).read_text(encoding='utf-8')
        self.assertIn('- [RFD0195](../../memo/0195_tail_probe_10rfd.md) — tail\n'
                      '- [RFD0918](../../memo/0918_scope_probe_10rfd.md) — scope\n'
                      '- [RFD0997](../../memo/0997_old_probe_10rfd.md) — old symbol\n',text)
        self.assertNotIn('0196',text)

    def test_maps_read_old_and_new_renumber_lines_by_value(self):
        # 振り直しの行は、当時のまま残る旧い形（3 桁と（PRDnnn））と新しい形（4 桁と（RFDnnnn））の両方を読み、
        # 番号を値で比べる。行の日付と同じ日のコミットの旧い番号も行で当て、何度も振り直した文書の旧い番号は、
        # 途中の番号の文書ではなく行を持つ今の文書に当てる（RFD0195 D17・§9.1、ADR 0077）。
        helper,root,page=self.renumber_root({
            'docs/memo/0158_old_form_probe_10rfd.md':
                '# old form\n\n> 2026-11-17 番号を 997 から 158 へ振り直した（PRD131）。\n',
            'docs/memo/0160_new_form_probe_10rfd.md':
                '# new form\n\n> 2029-01-05 番号を 0998 から 0160 へ振り直した（RFD0300）。\n',
            'docs/memo/0132_chain_probe_10rfd.md':
                '# chain\n\n> 2026-11-03 番号を 999 から 901 へ振り直した（PRD114）。\n'
                '> 2026-11-17 番号を 901 から 132 へ振り直した（PRD131）。\n',
            'docs/memo/0901_reused_probe_10rfd.md': '# reused 901\n'})
        result=self.renumber_maps(helper,root,'2029-01-05\tfeat(RunnerApp): new。 RFD0998 #3\n'
                                               '2026-11-17\tfeat(RunnerApp): same day。 PRD997 #3\n'
                                               '2026-11-10\tfeat(RunnerApp): middle。 PRD901 #3\n'
                                               '2026-10-01\tfeat(RunnerApp): first。 PRD999 #3\n')
        self.assertEqual(result['unresolved_rfds'],[])
        text=(root/page).read_text(encoding='utf-8')
        self.assertIn('- [RFD0132](../../memo/0132_chain_probe_10rfd.md) — chain\n'
                      '- [RFD0158](../../memo/0158_old_form_probe_10rfd.md) — old form\n'
                      '- [RFD0160](../../memo/0160_new_form_probe_10rfd.md) — new form\n',text)
        self.assertNotIn('0901_reused',text)
        # 旧い番号と新しい番号の桁を混ぜた行は、どちらの形にも合わないので止まる。
        helper,root,page=self.renumber_root({
            'docs/memo/0158_mixed_probe_10rfd.md':
                '# mixed\n\n> 2026-11-17 番号を 997 から 0158 へ振り直した（PRD131）。\n'})
        with self.assertRaises(ValueError) as caught:
            self.renumber_maps(helper,root,'2026-11-10\tfeat(RunnerApp): a。 PRD997 #3\n')
        self.assertIn('docs/memo/0158_mixed_probe_10rfd.md:3:',str(caught.exception))

    def test_maps_follow_renames_to_keep_rfds_of_moved_files(self):
        # ファイルを移したあとも、移す前のコミットの RFD を、今のパスが当たるページの RFD の一覧に残す（課題 a8d777）。
        # 関係の無いファイルのコミットは数えない。
        helper,root,page=self.renumber_root({
            'docs/memo/0046_old_probe_10rfd.md':'# old probe\n',
            'docs/memo/0182_move_probe_10rfd.md':'# move probe\n',
            'docs/memo/0045_other_probe_10rfd.md':'# other probe\n'})
        log=('\x012026-11-19\trefactor(build): move。 RFD0182 #6\0\n'
             'R100\0core/runner/Rfd0131Probe.cpp\0src/apps/runner/src/Rfd0131Probe.cpp\0'
             '\x012026-10-01\tfeat(RunnerApp): old。 RFD0046 #3\0\nM\0core/runner/Rfd0131Probe.cpp\0'
             '\x012026-09-01\tfeat(RunnerApp): other。 RFD0045 #3\0\nM\0core/other/Other.cpp\0')
        def git(_root,*args):
            return log if args[0]=='log' else 'src/apps/runner/src/Rfd0131Probe.cpp\0'
        with patch.object(helper,'git',side_effect=git):
            result=helper.regenerate_maps(root)
        self.assertEqual(result['unresolved_rfds'],[])
        text=(root/page).read_text(encoding='utf-8')
        self.assertIn('- [RFD0046](../../memo/0046_old_probe_10rfd.md) — old probe',text)
        self.assertIn('- [RFD0182](../../memo/0182_move_probe_10rfd.md) — move probe',text)
        self.assertNotIn('RFD0045',text)

    def test_maps_resolve_old_number_through_renumber_line(self):
        # 振り直しの行の日付以前のコミットの旧い番号は、振り直した RFD の今の番号の行になる。
        helper,root,page=self.renumber_root({
            'docs/memo/0158_eval_probe_10rfd.md':
                '# RFD0158: eval probe\n\n> 2026-11-17 番号を 997 から 158 へ振り直した（PRD131）。\n\n## 1. 問題\n'})
        result=self.renumber_maps(helper,root,'2026-11-10\tfeat(RunnerApp): probe。 RFD0997 #3\n')
        self.assertEqual(result['unresolved_rfds'],[])
        text=(root/page).read_text(encoding='utf-8')
        self.assertIn('- [RFD0158](../../memo/0158_eval_probe_10rfd.md) — RFD0158: eval probe',text)
        self.assertNotIn('RFD0997',text)

    def test_maps_merge_old_and_new_numbers_of_one_rfd(self):
        # 今の番号と旧い番号で同じ RFD に当たったら、1 行にまとめる。
        helper,root,page=self.renumber_root({
            'docs/memo/0158_eval_probe_10rfd.md':
                '# eval probe\n\n> 2026-11-17 番号を 997 から 158 へ振り直した（PRD131）。\n'})
        result=self.renumber_maps(helper,root,'2026-11-20\tfix(RunnerApp): b。 RFD0158 #4\n'
                                               '2026-11-10\tfeat(RunnerApp): a。 RFD0997 #3\n')
        self.assertEqual(result['unresolved_rfds'],[])
        text=(root/page).read_text(encoding='utf-8')
        self.assertEqual(text.count('](../../memo/0158_eval_probe_10rfd.md)'),1)

    def test_maps_send_later_commits_of_reused_number_to_the_new_file(self):
        # 9xx を再び使ったあとは、振り直しの日より後のコミットの番号を、新しいファイルの RFD に当てる。
        helper,root,page=self.renumber_root({
            'docs/memo/0158_eval_probe_10rfd.md':
                '# old eval\n\n> 2026-11-17 番号を 997 から 158 へ振り直した（PRD131）。\n',
            'docs/memo/0997_new_probe_10rfd.md': '# new probe\n'})
        result=self.renumber_maps(helper,root,'2029-05-01\tfeat(RunnerApp): new。 RFD0997 #3\n'
                                               '2026-11-17\tfeat(RunnerApp): old。 RFD0997 #3\n')
        self.assertEqual(result['unresolved_rfds'],[])
        text=(root/page).read_text(encoding='utf-8')
        self.assertIn('- [RFD0158](../../memo/0158_eval_probe_10rfd.md) — old eval',text)
        self.assertIn('- [RFD0997](../../memo/0997_new_probe_10rfd.md) — new probe',text)

    def test_maps_report_old_number_declared_by_two_rfds_as_unresolved(self):
        # 同じ旧い番号の振り直しの行を 2 本以上の RFD が持つと（RFD0114 の 999 など）、一意に決められない。
        helper,root,page=self.renumber_root({
            'docs/memo/0911_a_probe_10rfd.md': '# a\n\n> 2026-11-03 番号を 999 から 911 へ振り直した（PRD114）。\n',
            'docs/memo/0931_b_probe_10rfd.md': '# b\n\n> 2026-11-03 番号を 999 から 931 へ振り直した（PRD114）。\n'})
        result=self.renumber_maps(helper,root,'2026-10-24\tdocs(RFD0999): x #3\n')
        self.assertEqual(result['unresolved_rfds'],[{'page':page,'rfd':'RFD0999','matches':2}])
        self.assertIn('- RFD0999 — （RFD を一意に決められない）',(root/page).read_text(encoding='utf-8'))

    def test_maps_stop_on_broken_renumber_lines(self):
        # 形の崩れた行、同じ日付の行が 2 つ、旧い番号と新しい番号がつながらない RFD では、どのページも書かずに止まる。
        cases={
            'malformed': ('docs/memo/0158_eval_probe_10rfd.md',
                          '# x\n\n> 2026-11-17 番号を 997 から 158 へ振り直した。\n',':3:'),
            'same_date': ('docs/memo/0158_eval_probe_10rfd.md',
                          '# x\n\n> 2026-11-17 番号を 999 から 997 へ振り直した（PRD114）。\n'
                          '> 2026-11-17 番号を 997 から 158 へ振り直した（PRD131）。\n',':4:'),
            'broken_chain': ('docs/memo/0158_eval_probe_10rfd.md',
                             '# x\n\n> 2026-11-17 番号を 997 から 159 へ振り直した（PRD131）。\n',':3:'),
        }
        for name,(path,text,line) in cases.items():
            with self.subTest(name):
                helper,root,page=self.renumber_root({path:text})
                with self.assertRaises(ValueError) as caught:
                    self.renumber_maps(helper,root,'2026-11-10\tfeat(RunnerApp): a。 RFD0997 #3\n')
                self.assertIn(path+line,str(caught.exception))
                self.assertIn('old',(root/page).read_text(encoding='utf-8'))

    def test_maps_ignore_renumber_like_lines_after_first_heading(self):
        # 本文やコードの枠にある同じ形の行は、振り直しの行として読まず、検査もしない。
        helper,root,page=self.renumber_root({
            'docs/memo/0158_eval_probe_10rfd.md':
                '# x\n\n> 2026-11-17 番号を 997 から 158 へ振り直した（PRD131）。\n\n## 1. 例\n\n'
                '> 2026-11-18 番号を 998 から 158 へ振り直した（PRD131）。\n> 2026-11-19 番号を 9xx から\n'})
        result=self.renumber_maps(helper,root,'2026-11-10\tfeat(RunnerApp): a。 RFD0998 #3\n')
        self.assertEqual(result['unresolved_rfds'],[{'page':page,'rfd':'RFD0998','matches':0}])

    def test_maps_rejects_pending_subject_without_files(self):
        root=self.fixture()
        for extra in (['--pending-subject','feat(DQN): x。 RFD0997 #3'],['--pending-files','a.cpp']):
            with self.subTest(extra=extra):
                result=subprocess.run([sys.executable,'-B',str(HELPER),'maps',*extra],cwd=root,
                                      encoding='utf-8',capture_output=True)
                self.assertEqual(result.returncode,2)
                self.assertIn('--pending-subject と --pending-files',result.stderr)

    def test_only_reports_index_is_excluded_from_materials(self):
        root=self.fixture()
        self.material(root,'reports/INDEX.md','# derived index')
        self.material(root,'docs/experiments/INDEX.md','# experiment observations')
        data=self.run_helper(root)
        self.assertEqual([row['path'] for row in data['pending']],['docs/experiments/INDEX.md'])


if __name__ == '__main__':
    unittest.main()
