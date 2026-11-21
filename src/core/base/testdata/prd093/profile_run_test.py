"""NVTXのprobe範囲を同じRunの更新区間へ対応付ける回帰テスト。"""
from pathlib import Path
import sqlite3
import json
import tempfile
import unittest

from profile_run import read_capture_cost, add_recorded_run_cost

class ProfileTests(unittest.TestCase):
    def test_inclusive_cost_uses_the_same_update_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.sqlite'
            with sqlite3.connect(path) as connection:
                connection.executescript('CREATE TABLE StringIds(id INTEGER, value TEXT); CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER);')
                connection.execute('INSERT INTO StringIds VALUES(1, ?)',('Learner::CaptureAttentionEntropy',))
                connection.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)',[
                    (0,10_000_000,'Learner::CaptureAttentionEntropy',None),
                    (1_000_000_000,1_020_000_000,None,1),
                    (2_000_000_000,2_030_000_000,'Learner::CaptureAttentionEntropy',None),
                    (1_000_000,3_000_000,'CustomTransformerEncoderLayer::MeasureAttentionEntropy',None)])
            connection.close()
            result=read_capture_cost(path,[0,503,1006],10.0,25.0)
            self.assertAlmostEqual(result['capture_seconds_total'],0.06)
            self.assertAlmostEqual(result['profile_capture_seconds'],0.03)
            self.assertAlmostEqual(result['profile_capture_ratio'],0.003)
            self.assertAlmostEqual(result['capture_ratio_process'],0.06/25)
            self.assertEqual(result['capture_count'],3)
            self.assertAlmostEqual(result['nested_scopes']['CustomTransformerEncoderLayer::MeasureAttentionEntropy']['seconds'],0.002)

    def test_missing_capture_cannot_be_reported_as_zero_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.sqlite'
            connection=sqlite3.connect(path)
            connection.executescript('CREATE TABLE StringIds(id INTEGER, value TEXT); CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER);')
            connection.execute('INSERT INTO NVTX_EVENTS VALUES(0,1000,?,NULL)',('Learner::CaptureAttentionEntropy',))
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError,'capture count'):
                read_capture_cost(path,[0,503,1006],10.0,25.0)

    def test_invalid_clock_and_negative_duration_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.sqlite'
            connection=sqlite3.connect(path)
            connection.executescript('CREATE TABLE StringIds(id INTEGER, value TEXT); CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER);')
            connection.execute('INSERT INTO NVTX_EVENTS VALUES(1000,0,?,NULL)',('Learner::CaptureAttentionEntropy',))
            connection.commit(); connection.close()
            with self.assertRaisesRegex(ValueError,'duration'):
                read_capture_cost(path,[1006],10.0,25.0)
            for seconds in (0,-1,float('nan')):
                with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                    read_capture_cost(path,[1006],seconds,25.0)

    def test_whole_recorded_run_ratio_keeps_cold_probe_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            rows=[{'tag':'90_perf/90_elapse_hour','step':10,'value':12/3600},
                {'tag':'90_perf/99_prd093_elapse_hour','step':1006,'value':10/3600}]
            (path/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            result={'run':str(path),'capture_seconds_total':0.6}
            add_recorded_run_cost(result)
            self.assertAlmostEqual(result['recorded_run_clock_seconds'],12)
            self.assertAlmostEqual(result['capture_ratio_recorded_run'],0.05)

if __name__=='__main__': unittest.main()
