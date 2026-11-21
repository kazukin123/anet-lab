"""PRD094 acceptance comparator must detect numerical and contract changes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from compare_metrics import compare
from compare_configs import compare as compare_configs

class ComparatorTest(unittest.TestCase):
    def test_post_review_config_contract(self):
        with tempfile.TemporaryDirectory(prefix='prd094-config-') as directory:
            root = Path(directory)
            def write(name, entries, output=None):
                run = root / name
                (run / 'config').mkdir(parents=True)
                rows = {'app.runs_dir': output or root.as_posix(), 'run.seed': '1', **entries}
                (run / 'config/config_data.txt').write_text(
                    ''.join(f'{key} = {value}\n' for key, value in rows.items()), encoding='utf-8')
                return run
            before = write('before', {'trace.key': 'value with spaces'})
            same = write('same', {'trace.key': 'value with spaces'})
            self.assertTrue(compare_configs(before, same, 'atari1', existing_only=True)['exact_match'])
            for name, entries in (
                ('added', {'trace.key': 'value with spaces', 'extra': '1'}),
                ('removed', {}), ('changed', {'trace.key': 'value with space'})):
                with self.subTest(change=name):
                    self.assertFalse(compare_configs(before, write(name, entries), 'atari1', existing_only=True)['exact_match'])
            self.assertFalse(compare_configs(before, write('bad_dir', {'trace.key': 'value with spaces'}, '/wrong'),
                                             'atari1', existing_only=True)['exact_match'])

    def test_contract_and_numeric_differences(self):
        with tempfile.TemporaryDirectory(prefix='prd094-compare-') as directory:
            root=Path(directory)
            records=[{'type':'scalar','tag':'agent/loss','step':16,'value':1.0},
                     {'type':'trace','tag':'episode','step':16,'lane':0,'timestamp':'before','data':{'reward':2.0}}]
            def write(name, rows, definition=1, warning='same'):
                run=root/name;run.mkdir();(run/'json').mkdir()
                (run/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf8')
                (run/'json/metrics.scalar.defs.json').write_text(json.dumps({'data':definition}),encoding='utf8')
                (run/(name+'.log')).write_text('12:00:00: [W] '+warning+'\n',encoding='utf8')
                return run
            before=write('before',records)
            same=copy.deepcopy(records);same[1]['timestamp']='after'
            same.append({'type':'scalar','tag':'90_perf/speed','step':16,'value':999})
            self.assertTrue(compare(before,write('same',same))['exact_match'])
            changes=[]
            changed=copy.deepcopy(records);changed[0]['value']=2.0;changes.append(changed)
            changed=copy.deepcopy(records);changed[0]['step']=17;changes.append(changed)
            changed=copy.deepcopy(records);changed[1]['lane']=1;changes.append(changed)
            changed=copy.deepcopy(records);changed[1]['data']['reward']=3.0;changes.append(changed)
            changed=copy.deepcopy(records);changed.append(copy.deepcopy(changed[0]));changes.append(changed)
            for i,rows in enumerate(changes):
                with self.subTest(change=i):
                    self.assertFalse(compare(before,write('different'+str(i),rows))['exact_match'])
            self.assertFalse(compare(before,write('definition',records,definition=2))['exact_match'])
            self.assertFalse(compare(before,write('warning',records,warning='different'))['exact_match'])

if __name__=='__main__': unittest.main()
