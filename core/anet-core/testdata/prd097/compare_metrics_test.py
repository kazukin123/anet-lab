"""PRD097 comparison contract checks."""
import json
from pathlib import Path
import tempfile
import unittest
from compare_metrics import compare, read_run

class ComparisonTest(unittest.TestCase):
    def test_only_documented_differences_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a';b=Path(directory)/'b';a.mkdir();b.mkdir()
            rows=[{'type':'scalar','tag':'47_agent_noisy/test','step':1,'value':1.0},
                {'type':'scalar','tag':'42_env/30_lives_mean','step':1,'value':3.0},
                {'type':'trace','tag':'episode','step':1,'lane':0,'data':{'score':1.0},'timestamp':'old'}]
            def save(path,data): (path/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in data),encoding='utf-8')
            save(a,rows);changed=json.loads(json.dumps(rows));changed[1]['value']=0.0;changed[2]['timestamp']='new';save(b,changed)
            self.assertFalse(compare(a,b)['exact_match']);self.assertTrue(compare(a,b,True)['exact_match'])
            changed[0]['value']=2.0;save(b,changed);self.assertFalse(compare(a,b,True)['exact_match'])
            changed[0]['value']=1.0;changed[2]['data']['score']=2.0;save(b,changed);self.assertFalse(compare(a,b,True)['exact_match'])
            changed[2]['data']['score']=1.0;changed[1]['step']=2;save(b,changed);self.assertFalse(compare(a,b,True)['exact_match'])
            save(b,rows+[rows[2]]);self.assertFalse(compare(a,b)['exact_match'])

    def test_pipeline_edges_only_for_migrated_declarations(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a';b=Path(directory)/'b'
            for run in (a,b):
                (run/'config').mkdir(parents=True);(run/'json').mkdir()
                (run/'config/run.txt').write_text('run.train.runner_type = pipeline\nrun.train.num_envs = 2\n')
                (run/'config/app.txt').write_text('app.exp_exit_step = 8\n')
            tag='42_env/score'
            old={tag:{'target':'env','event':'train','source_key':'mean.score','step_axis':'exp_step'}}
            new={tag:{**old[tag],'target':'env_step','event':'episode_end'}}
            for run,defs in ((a,old),(b,new)):
                (run/'json/metrics.scalar.defs.json').write_text(json.dumps({'data':defs}))
            rows=[{'type':'scalar','tag':tag,'step':4,'value':1.0}]
            (a/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            edges=[{'type':'scalar','tag':tag,'step':x,'value':2.0} for x in (0,6)]
            def save(rows): (b/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save(rows+edges)
            self.assertFalse(compare(a,b)['exact_match'])
            self.assertTrue(compare(a,b,allow_migrated_edges=True)['exact_match'])
            save(rows+edges+[{'type':'scalar','tag':tag,'step':2,'value':3.0}])
            self.assertFalse(compare(a,b,allow_migrated_edges=True)['exact_match'])
            save(rows+[{'type':'scalar','tag':'other','step':8,'value':3.0}])
            self.assertFalse(compare(a,b,allow_migrated_edges=True)['exact_match'])

    def test_performance_uses_last_reported_coordinate(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);(run/'config').mkdir()
            (run/'config/app.txt').write_text('app.exp_exit_step = 1000\n')
            rows=[{'type':'scalar','tag':'90_perf/12_exp_step_per_sec','step':step,'value':value}
                for step,value in [(100,1),(200,10),(300,20),(400,30)]]
            (run/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            _,summary=read_run(run)
            self.assertEqual(summary['performance_median'],25)
            self.assertEqual(summary['performance_last_step'],400)

if __name__=='__main__': unittest.main()
