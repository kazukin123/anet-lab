"""PRD099の実行体と入力を隔離・固定して決定的な前後比較を行う。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / '.scratch/prd099/acceptance'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prd094'))
from compare_metrics import compare


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_case(label, explicit):
    # 実行体と設定をケースごとに固定し、既存の完了記録は上書きしない。
    target = ROOT / label
    target.mkdir(parents=True, exist_ok=False)
    binary_dir = target / 'bin'
    binary_dir.mkdir()
    source = REPO / '.scratch/prd099/bin/Release'
    binary = binary_dir / 'AnetRLRunner.exe'
    shutil.copy2(source / binary.name, binary)
    for dll in source.glob('*.dll'):
        os.link(dll, binary_dir / dll.name)
    shutil.copytree(REPO / 'apps/runner/config', target / 'source-config')
    runs = target / 'runs'
    runs.mkdir()
    # コピーした設定だけを参照し、短い学習を含む決定的なCPU条件を明示する。
    lines = [f'$include <{target.as_posix()}/source-config/_main.txt>',
             f'$include <{target.as_posix()}/source-config/Atari.txt>']
    values = ['app.$=app.batchrun>P1', 'app.show_error_dialog=false',
              f'app.runs_dir={runs.as_posix()}', 'run.seed=1',
              'backend.$=backend.@deterministic', 'backend.deterministic_warn_only=false',
              'backend.torch_num_threads=1', 'run.train.runner_type=pipeline',
              'agent.device=cpu', 'env.device=cpu', 'run.eval_device=cpu',
              'run.train.num_envs=4', 'env.worker_type=1',
              'E1.game=qbert', 'AtariEnv.max_episode_frames=256',
              'DefaultDQNAgent.learner.replay_capacity=8192',
              'DefaultDQNAgent.learner.update_warmup_steps=256',
              'DefaultDQNAgent.learner.replay_batch_size=32',
              'DefaultDQNAgent.learner.replay_ratio=0.25',
              'DefaultDQNAgent.learner.use_amp=false',
              'DefaultDQNAgent.learner.use_amp_bf16=false']
    for tag in ['eval_target', 'eval', 'greedy_dist']:
        values.append(f'run.eval_schedule.[{tag}].interval=0')
    if explicit:
        values.append('AtariEnv.resize_mode=adaptive_average')
    lines += ['run.@prd099 : ' + value for value in values]
    lines.append('run.$=run.@v5_iqn_impala_x2>run.@prd099')
    cfg = target / 'config.txt'
    cfg.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # 名前と予算をケースに固定し、実行した引数と実行体のhashを先に保存する。
    args = [str(binary), '--config', str(cfg), f'app.run_name=run_{{t}}_099_{label}',
            'app.exp_exit_step=16384', 'app.batchrun.exp_exit_step=16384']
    record = {'label': label, 'arguments': args, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'seed': 1, 'budget': 16384, 'explicit': explicit}
    save(target / 'execution.json', record)
    print('START', label, flush=True)
    started = time.monotonic()
    # 既存のRunを操作せず隔離実行体を逐次実行し、標準出力とエラーを別々に残す。
    with (target / 'stdout.log').open('w', encoding='utf-8') as out, (target / 'stderr.log').open('w', encoding='utf-8') as err:
        process = subprocess.run(args, cwd=REPO / 'apps/runner', stdout=out, stderr=err,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    # 終了状態とRunの一意性を検証し、失敗したケースを比較へ渡さない。
    found = list(runs.glob('run_*'))
    record.update(exit_code=process.returncode, seconds=time.monotonic()-started,
                  run=str(found[0]) if len(found)==1 else None)
    save(target / 'execution.json', record)
    print('END', label, 'exit', process.returncode, 'seconds', round(record['seconds'], 1), flush=True)
    if process.returncode or len(found)!=1:
        raise RuntimeError('Acceptance Run failed: ' + label)
    return found[0]


def saved_run(label):
    record = json.loads((ROOT / label / 'execution.json').read_text(encoding='utf-8'))
    if record['exit_code'] != 0:
        raise RuntimeError('Baseline Run did not complete')
    return Path(record['run'])


def config_values(run):
    return dict(line.split(' = ',1) for line in (run/'config/config_data.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip())


def check_pair(old, new):
    # 時刻・性能・資源だけを除く既存の全点比較を再利用する。
    result = compare(old, new)
    a, b = config_values(old), config_values(new)
    ignored = {'app.runs_dir', 'app.run_name'}
    changed = {key: [a[key], b[key]] for key in a.keys() & b.keys() if key not in ignored and a[key] != b[key]}
    removed = sorted(a.keys()-b.keys())
    added = {key: b[key] for key in b.keys()-a.keys()}
    expected_added = {'AtariEnv.resize_mode': 'adaptive_average'} if 'AtariEnv.resize_mode' not in a and 'AtariEnv.resize_mode' in b else {}
    result['config'] = {'changed': changed, 'removed': removed, 'added': added,
                        'exact_existing_values': not changed and not removed and added == expected_added}
    result['exact_match'] &= result['config']['exact_existing_values']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['before', 'after'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    # 変更前は同条件を2回実行し、基準そのものの再現性を先に確認する。
    if args.phase == 'before':
        first = run_case('before1', False)
        second = run_case('before2', False)
        result = check_pair(first, second)
        save(ROOT/'baseline-comparison.json', result)
        print('BASELINE', result['exact_match'], 'points', result['new']['points'], flush=True)
        return 0 if result['exact_match'] else 1
    # 変更後は省略・明示の両方を同じ基準へ照合する。一方が失敗しても他方を残す。
    errors = []
    for label, explicit in [('after-default', False), ('after-explicit', True)]:
        try:
            run = run_case(label, explicit)
            result = check_pair(saved_run('before1'), run)
            save(ROOT/(label+'-comparison.json'), result)
            print('COMPARE', label, result['exact_match'], 'points', result['new']['points'], flush=True)
            if not result['exact_match']:
                errors.append(label)
        except Exception as error:
            errors.append(label + ': ' + str(error))
            print('ERROR', errors[-1], flush=True)
    # 各比較結果を残したうえで全ケースの失敗を集計し、終了コードへ反映する。
    save(ROOT/'summary.json', {'errors': errors})
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
