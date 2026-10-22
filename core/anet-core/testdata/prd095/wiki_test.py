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

REPO = Path(__file__).resolve().parents[4]
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
        path = REPO/'docs/experiments'/('prd095_probe_'+uuid.uuid4().hex+'.md')
        path.write_text('# probe\n\n観測したこと。\n', encoding='utf-8')
        self.addCleanup(path.unlink)
        log_before = (REPO/'docs/wiki/log.md').read_bytes()
        result = subprocess.run([sys.executable,'-B',str(HELPER),'scan'],cwd=REPO,
                                encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertIn(path.relative_to(REPO).as_posix(),[r['path'] for r in data['pending']])
        self.assertEqual((REPO/'docs/wiki/log.md').read_bytes(),log_before)

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
        c=self.material(root,'docs/memo/done/998_probe_10prd.md','# done')
        d=self.material(root,'reports/already_probe.md','# already')
        with (root/'docs/wiki/log.md').open('a',encoding='utf-8') as log:
            log.write('2026-10-22 ingest docs/experiments/changed_probe.md blob:'+'0'*40+' → なし\n')
            log.write('2026-10-22 fix docs/experiments/changed_probe.md blob:'+a+' → なし\n')
            log.write('2026-10-22 ingest reports/old_probe.md blob:'+b+' → なし\n')
            log.write('2026-10-22 ingest docs/memo/998_probe_10prd.md blob:'+c+' → なし\n')
            log.write('2026-10-22 writeback reports/already_probe.md blob:'+d+' → なし\n')
        data=self.run_helper(root)
        self.assertEqual([r['path'] for r in data['pending']],
                         ['docs/experiments/changed_probe.md','docs/memo/done/998_probe_10prd.md'])
        self.assertEqual(data['pending'][0]['state'],'更新あり')
        self.assertEqual([r['path'] for r in data['moved']],['reports/moved_probe.md'])

    def test_map_regeneration_includes_untracked_group_without_touching_prose(self):
        root=self.fixture()
        (root/'reports/tools').mkdir(parents=True)
        shutil.copyfile(REPO/'reports/tools/categories.json',root/'reports/tools/categories.json')
        shutil.copytree(REPO/'docs/memo',root/'docs/memo',dirs_exist_ok=True)
        for name in ['Prd095Probe.hpp','Prd095Probe.cpp','Prd095Probe_test.cpp']:
            self.material(root,'apps/runner/src/'+name,'// probe')
        page=root/'docs/wiki/apps/110_runner_app.md'
        page.parent.mkdir(parents=True)
        page.write_text('# app\n\n## 概要・責務\nKEEP PROSE\n\n## ファイル\nold\n\n'
                        '## PRD の一覧\nold\n\n## 台帳の指摘\nKEEP REGISTRY\n',encoding='utf-8')
        self.run_helper(root,'maps')
        text=page.read_text(encoding='utf-8')
        row=next(line for line in text.splitlines() if 'Prd095Probe.hpp' in line)
        self.assertIn('Prd095Probe.cpp',row)
        self.assertIn('Prd095Probe_test.cpp',row)
        self.assertIn('KEEP PROSE',text)
        self.assertTrue(text.endswith('## 台帳の指摘\nKEEP REGISTRY\n'))
        first=text
        self.run_helper(root,'maps')
        self.assertEqual(page.read_text(encoding='utf-8'),first)

    def test_only_unannotated_source_links_make_housekeeping_eligible(self):
        root=self.fixture()
        blob=self.material(root,'docs/memo/done/097_probe_link_10prd.md','# source')
        with (root/'docs/wiki/log.md').open('a',encoding='utf-8') as log:
            log.write('2026-10-22 ingest docs/memo/done/097_probe_link_10prd.md blob:'+blob+' → なし\n')
        page=root/'docs/wiki/apps/110_runner_app.md'
        page.parent.mkdir(parents=True)
        page.write_text('# app\n\n## 概要・責務\n- claim [PRD](../../memo/999_probe_link_10prd.md)\n\n'
                        '## ファイル\n- [gone](../../../apps/runner/src/Gone.cpp)\n',encoding='utf-8')
        data=self.run_helper(root)
        self.assertEqual(len(data['broken_links']),1)
        self.assertEqual(data['broken_links'][0]['candidates'],['docs/memo/done/097_probe_link_10prd.md'])
        self.assertTrue(data['eligible'])
        page.write_text(page.read_text(encoding='utf-8').replace('999_probe_link_10prd.md)',
                        '999_probe_link_10prd.md)（出典のリンク切れ）'),encoding='utf-8')
        self.assertFalse(self.run_helper(root)['eligible'])
        self.material(root,'reports/registry/findings.md','# derived')
        self.material(root,'reports/housekeeping/runs.jsonl','{}')
        self.assertFalse(self.run_helper(root)['eligible'])

    def test_maps_keep_running_and_report_unresolved_prd_numbers(self):
        spec=importlib.util.spec_from_file_location('prd095_wiki_support',HELPER)
        helper=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        for number, count in [('931',0),('999',2)]:
            with self.subTest(number=number, sources=count), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                (root/'reports/tools').mkdir(parents=True)
                shutil.copyfile(REPO/'reports/tools/categories.json',root/'reports/tools/categories.json')
                for n in range(count):
                    source=root/f'docs/memo/{number}_probe{n}_10prd.md'
                    source.parent.mkdir(parents=True,exist_ok=True)
                    source.write_text('# probe\n',encoding='utf-8')
                pages=['docs/wiki/apps/110_runner_app.md','docs/wiki/categories/311_dev_tools.md']
                for q in pages:
                    page=root/q;page.parent.mkdir(parents=True,exist_ok=True)
                    page.write_text('# map\n\n## ファイル\nold\n\n## PRD の一覧\nold\n\n'
                                    '## 台帳の指摘\nKEEP\n',encoding='utf-8')
                # 履歴は外部境界のGitだけ差し替える。本文生成と全ページ更新を通す。
                def history(_root,*args):
                    return (f'feat(PRD{number}): probe\n' if args[0]=='log' else
                            'apps/runner/src/Prd095Probe.cpp\0apps/runner/tools/prd095_probe.py\0')
                with patch.object(helper,'git',side_effect=history):
                    result=helper.regenerate_maps(root)
                self.assertEqual(set(result['updated']),set(pages))
                self.assertEqual(result['unresolved_prds'],[
                    {'page':q,'prd':'PRD'+number,'matches':count} for q in pages])
                for q in pages:
                    text=(root/q).read_text(encoding='utf-8')
                    self.assertIn('- PRD'+number+' — （PRD を一意に決められない）',text)
                    self.assertNotIn('[PRD'+number+']',text)
                    self.assertTrue(text.endswith('## 台帳の指摘\nKEEP\n'))

    def test_only_reports_index_is_excluded_from_materials(self):
        root=self.fixture()
        self.material(root,'reports/INDEX.md','# derived index')
        self.material(root,'docs/experiments/INDEX.md','# experiment observations')
        data=self.run_helper(root)
        self.assertEqual([row['path'] for row in data['pending']],['docs/experiments/INDEX.md'])


if __name__ == '__main__':
    unittest.main()
