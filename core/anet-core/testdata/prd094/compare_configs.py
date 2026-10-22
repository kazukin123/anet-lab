"""Compare all resolved configuration keys and strings for PRD094 acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ROOT = Path(os.environ.get('PRD094_ACCEPTANCE_ROOT',str(REPO / '.scratch/prd094/acceptance-20261021')))
CASES = ('atari1', 'dropmerge', 'lunarlander', 'gridmaze', 'actorapprox')
COLUMNS = ('$runner exp_step label:step', '$runner step_in_episode label:ep_step',
           '$action_info action', '$env_step reward', '$agent_step q_sa label:Q_sa',
           '$agent_step state_value label:V_s', '$env_step done', '$env_step truncated label:trunc',
           '$agent_step approx_priority label:prio',
           '$agent_step approx_priority display:pct label:prio_pct')

def read(run):
    data = {}
    for line in (Path(run) / 'config/config_data.txt').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        key, value = line.split(' = ', 1)
        if key in data: raise ValueError('Duplicate resolved key: ' + key)
        data[key] = value
    return data

def compare(old_run, new_run, case, existing_only=False):
    old, new = read(old_run), read(new_run)
    prefix = 'app.eval_trace_panel.'
    expected = {prefix+'max_rows': '1000', prefix+'pct_window': '10000'}
    expected.update({prefix+f'columns.[{i}]': value for i, value in enumerate(COLUMNS, 1)})
    if case == 'atari1':
        expected.update({prefix+'columns.[20]': '$env_step game_score', prefix+'columns.[21]': '$env_step lives'})
    if existing_only:
        expected = {}
    added = {key: new[key] for key in new.keys()-old.keys()}
    removed = sorted(old.keys()-new.keys())
    changed = {key: [old[key], new[key]] for key in old.keys() & new.keys()
               if key != 'app.runs_dir' and old[key] != new[key]}
    # 出力先だけは各世代のRun親に一致することを別に検査する。
    output_dirs_match = all(data.get('app.runs_dir') == Path(run).parent.as_posix()
                            for data, run in ((old, old_run), (new, new_run)))
    def checksum(data):
        normalized = {key: ('<run_dir>' if key == 'app.runs_dir' else data[key]) for key in old if key in data}
        return hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {'exact_match': not removed and not changed and added == expected and output_dirs_match,
            'existing_keys': len(old), 'new_keys': len(new), 'added': dict(sorted(added.items())),
            'removed': removed, 'unexpected_changes': changed, 'output_dirs_match': output_dirs_match,
            'old_existing_sha256': checksum(old), 'new_existing_sha256': checksum(new)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', default='after-v1')
    parser.add_argument('--existing-only', action='store_true', help='Require identical keys for post-review comparisons')
    args = parser.parse_args()
    results = []
    for case in CASES:
        before = json.loads((ROOT/'before'/f'{case}_1.execution.json').read_text(encoding='utf-8'))
        after = json.loads((ROOT/args.after/f'{case}_1.execution.json').read_text(encoding='utf-8'))
        if before.get('exit_code') != 0 or after.get('exit_code') != 0:
            raise RuntimeError('Configuration comparison requires completed Runs: ' + case)
        result = compare(before['run'], after['run'], case, args.existing_only)
        results.append({'case': case, **result})
        print(case, result['exact_match'], 'existing', result['existing_keys'], 'added', len(result['added']))
    (ROOT/args.after/'config-equivalence.json').write_text(json.dumps(results, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    raise SystemExit(0 if all(row['exact_match'] for row in results) else 1)
