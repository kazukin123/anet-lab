"""PRD098 round2: baseline設定、OFF等価性、CLI変種とGridMaze学習の検証。"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import run_validation as validation


def compare_configs(root):
    # 直接依頼の表で許した変更だけを過去側へ適用し、全キー・文字列値を比較する。
    additions = {
        'skip.enabled': 'false', 'net.body.output.[skip_context]': 'main_feature',
        'skip.net.branch.[skip_feature].bind': 'skip_context', 'skip.net.branch.[skip_feature].bind_concat_dim': '-1',
        'skip.net.branch.[skip_feature].structure': 'SkipFC1 > SkipReLU > SkipFC2 > SkipReLU',
        'skip.net.body.output.[features]': 'skip_feature', 'skip.network_mode': 'action_out',
        'skip.head.use_dueling': 'false', 'learner.skip.feature_gradient': 'stop',
        'learner.skip.loss_weight': '1.0', 'learner.skip.priority_lambda': '0'}
    blocks = {'net.block.[SkipFC1].type': 'Linear', 'net.block.[SkipFC1].linear.out_features': '128',
              'net.block.[SkipFC2].type': 'Linear', 'net.block.[SkipFC2].linear.out_features': '128',
              'net.block.[SkipReLU].type': 'ReLU'}
    reports = {}
    for before in sorted((root / 'resolved-before').glob('*.json')):
        old = json.loads(before.read_text(encoding='utf-8-sig'))
        after = json.loads((root / 'resolved-after' / before.name).read_text(encoding='utf-8-sig'))
        expected = {}
        for key, value in old.items():
            if re.match(r'net\.block\.\[Skip(?:Linear128|Flatten|Cosine|ReLU)\]', key):
                continue
            if key.endswith('.head.dueling'):
                key = key.removesuffix('.dueling') + '.use_dueling'
                value = {'none': 'false', 'standard': 'true', 'q_shared': 'true'}[value]
            expected[key] = value
        expected.update({'DefaultDQNAgent.' + key: value for key, value in additions.items()})
        expected.update(blocks)
        env = before.stem
        if env in validation.ENVS:
            expected['DefaultDQNAgent.skip.max_length'] = {'GridMaze': '7', 'LunarLander': '4', 'Atari': '10', 'DropMerge': '10'}[env]
        if env in ('LunarLander', 'Atari', 'DropMerge'):
            expected['DefaultDQNAgent.net.body.output.[skip_context]'] = 'fusion' if env == 'LunarLander' else 'iqn_fusion'
        differences = {key: {'expected': expected.get(key), 'actual': after.get(key)}
                       for key in sorted(expected.keys() | after.keys()) if expected.get(key) != after.get(key)}
        reports[env] = {'before_keys': len(old), 'after_keys': len(after), 'differences': differences}
    validation.save(root / 'config-comparison.json', reports)
    if len(reports) != 7 or any(report['differences'] for report in reports.values()):
        raise RuntimeError('Unexpected configuration differences; see config-comparison.json')
    print('PASS seven OFF-normalized environment snapshots', flush=True)


def freeze_runner(root, label, build_config):
    # 同一matrixの途中で実行体が変わらないよう固定する。
    source = validation.REPO / 'apps/runner/bin' / build_config / 'AnetRLRunner.exe'
    folder = root / (label + '-bin')
    folder.mkdir(exist_ok=True)
    runner = folder / source.name
    if runner.exists() and runner.read_bytes() != source.read_bytes():
        raise RuntimeError('Binary changed; choose a fresh --label')
    if not runner.exists():
        shutil.copy2(source, runner)
        for dll in source.parent.glob('*.dll'):
            os.link(dll, folder / dll.name)
    validation.RUNNER = runner


def check_profile(run, variant, width_j1=False, trace_q=False):
    # CLIの変種が解決結果へ反映され、廃止した選択葉が無いことを確かめる。
    data = json.loads((run / 'json/config_data.json').read_text(encoding='utf-8-sig'))['data']
    prefix = 'DefaultDQNAgent.'
    assert data[prefix + 'skip.enabled'] == 'true'
    assert not any('$+' in key for key in data)
    assert data[prefix + 'skip.head.use_dueling'] == ('true' if variant == 'dueling' else 'false')
    assert data[prefix + 'learner.skip.feature_gradient'] == ('merge' if variant == 'merge' else 'stop')
    assert data[prefix + 'skip.network_mode'] == ('action_in' if variant == 'action_in' else 'action_out')
    limit = int(data[prefix + 'skip.max_length'])
    dot = (run / 'dot/skip.net.detail.dot').read_text(encoding='utf-8-sig')
    if variant == 'action_in':
        assert 'action' in re.split(r'\s*[, *]\s*', data[prefix + 'skip.net.branch.[skip_feature].bind'])
        assert 'input: action' in dot
    # IQNは融合済み特徴を渡し、skip側へ独立したtau枝を増やさない。
    if data[prefix + 'quantile_mode'] == 'iqn':
        assert data[prefix + 'net.body.output.[skip_context]'] in ('fusion', 'iqn_fusion')
        assert not any(key.startswith(prefix + 'skip.net.branch.[skip_tau]') for key in data)
        input_line = next(line for line in dot.splitlines() if '<B>input: skip_context' in line)
        shape = re.search(r'>shape</TD><TD[^>]*>(\[[^<]+\])', input_line)[1]
        assert ',' in shape, shape
    if not width_j1:
        assert all(f'app.eval_trace_panel.columns.[{i}]' in data for i in (30, 31))
    if trace_q:
        assert all(f'skip_q.[{j}]' in data[f'app.eval_trace_panel.columns.[{31+j}]'] for j in range(1, limit + 1))
    # 幅は解決値だけでなく、次層の実際の入力次元もDOTで確認する。
    if width_j1:
        assert limit == 1 and data['net.block.[SkipFC1].linear.out_features'] == '256'
        assert any('SkipFC1' in line and 'out_features</TD><TD ALIGN="RIGHT">256' in line for line in dot.splitlines())
        assert any('SkipFC2' in line and 'in_features</TD><TD ALIGN="RIGHT">256' in line for line in dot.splitlines())
    required = {'50_agent_skip/' + name for name in ('01_decision_rate', '02_skip_length_mean',
        '03_skip_length_rate.[01]', '04_skip_q_gap', '11_loss', '12_td_abs_mean', '13_sample_k_mean', '14_kmax_mean', '15_grad_norm')}
    values = [row for row in validation.rows(run) if row.get('tag') in required and row.get('value') is not None]
    definitions = json.loads((run / 'json/metrics.scalar.defs.json').read_text(encoding='utf-8-sig'))['data']
    assert required.issubset(definitions) and required.issubset({row['tag'] for row in values})
    assert all(math.isfinite(row['value']) for row in values)
    # 分位平均と分布の軸、およびskipパラメータのログを区別して確認する。
    if data[prefix + 'quantile_mode'] != 'none':
        mean_line = next(line for line in dot.splitlines() if '<B>output: skip_q</B>' in line)
        line = next(line for line in dot.splitlines() if '<B>output: skip_q_dist</B>' in line)
        mean_shape = re.search(r'>shape</TD><TD[^>]*>\[([^<]+)\]', mean_line)[1].replace(' ', '').split(',')
        shape = re.search(r'>shape</TD><TD[^>]*>\[([^<]+)\]', line)[1].replace(' ', '').split(',')
        assert len(mean_shape) == (1 if variant == 'action_in' else 2) and mean_shape[-1] == str(limit), mean_line
        assert shape[:-1] == mean_shape, line
        if data[prefix + 'quantile_mode'] == 'qr':
            assert shape[-1] == data[prefix + 'qr.num_quantiles'], line
        else:
            assert shape[-1] in ('-1', '?'), line
    log = next(run.glob('run_*.log')).read_text(encoding='utf-8-sig')
    assert 'SKIP MODEL SHAPE DUMP' in log and 'skip.' in log
    dump = log[log.index('========== MODEL SHAPE DUMP =========='):log.index('Number of online network parameter tensors:')]
    parameters = re.findall(r'\[I\] ([^\n]+?) : (\[[^\n]*\])', dump)
    skip_parameters = sum(name.startswith('skip.') for name, _ in parameters)
    tensor_counts = [int(value) for value in re.findall(r'Number of (?:online|target) network parameter tensors: (\d+)', log)]
    assert skip_parameters > 0 and tensor_counts == [len(parameters), len(parameters)], (parameters, tensor_counts)
    validation.save(run.parent.parent / (run.name + '-profile-check.json'),
        {'profile': variant, 'mode': data[prefix + 'skip.network_mode'], 'metric_points': len(values),
         'action_dot_checked': variant == 'action_in', 'trace_q_configured': trace_q,
         'skip_parameter_tensors': skip_parameters, 'total_parameter_tensors': tensor_counts[0],
         'width_j1_checked': width_j1, 'distribution_dot_checked': data[prefix + 'quantile_mode'] != 'none'})


def summarize_learning(root, arms):
    # 1seedの動作確認として評価の推移を記録し、一般的な優劣の閾値は追加しない。
    reports = {}
    for arm, run in arms.items():
        tags, positive_steps = {}, []
        definitions = json.loads((run / 'json/metrics.scalar.defs.json').read_text(encoding='utf-8-sig'))['data']
        assert definitions['21_eval/01_target_reward']['step_axis'] == 'exp_step'
        for row in validation.rows(run):
            if row.get('value') is not None and math.isfinite(row['value']):
                tags.setdefault(row['tag'], []).append(row['value'])
                if row['tag'] == '21_eval/01_target_reward' and row['value'] > 0:
                    positive_steps.append(row['step'])
        selected = ('21_eval/01_target_reward', '21_eval/05_target_ep_steps', '50_agent_skip/01_decision_rate',
                    '50_agent_skip/02_skip_length_mean', '50_agent_skip/11_loss')
        reports[arm] = {tag: {'first': tags[tag][0], 'last': tags[tag][-1],
                             'mean': sum(tags[tag])/len(tags[tag]), 'max': max(tags[tag]), 'points': len(tags[tag])} for tag in selected}
        # 到達時点も残し、最終値だけで学習の立ち上がりを評価しない。
        reports[arm]['first_positive_target_exp_step'] = min(positive_steps) if positive_steps else None
        reports[arm]['run'] = str(run)
    validation.save(root / 'learning-summary.json', reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('config', 'after', 'profiles', 'learning'))
    parser.add_argument('--root', type=Path, default=validation.OUT)
    parser.add_argument('--label', default='r2_after1')
    parser.add_argument('--build-config', choices=('Debug', 'Release'), default='Debug')
    args = parser.parse_args()
    root = args.root.resolve()
    validation.OUT = root
    # 設定だけの比較はRunnerを起動しない。OFFの比較はDebug同士に限定する。
    if args.stage == 'config':
        compare_configs(root)
        return
    if args.stage == 'after' and args.build_config != 'Debug':
        parser.error('The saved OFF baseline is Debug')
    freeze_runner(root, args.label, args.build_config)
    errors, arms = [], {}
    original_settings = validation.settings
    for env in (('GridMaze',) if args.stage == 'learning' else validation.ENVS):
        if args.stage == 'after':
            try:
                validation.execute(args.label + '_' + env, env)
                validation.compare(args.label, env)
                print('PASS exact metrics', env, flush=True)
            except Exception as error:
                errors.append(env + ': ' + str(error))
            continue
        # 長い学習の再確認は指定されたGridMaze seed1・500kの2腕だけを実行する。
        if args.stage == 'learning':
            def settings(env, before=False, enabled=False):
                return original_settings(env, before, enabled, long=True, seed=1) + [
                    'run.@prd098_validation : GridMazeEnv.width = 6', 'run.@prd098_validation : GridMazeEnv.height = 10',
                    'run.@prd098_validation : app.exp_exit_step = 500000',
                    'run.@prd098_validation : app.batchrun.exp_exit_step = 500000',
                    'run.@prd098_validation : app.save_agent_on_close = true']
            validation.settings = settings
            variants = ('default', 'action_in')
        else:
            def settings(*a, **kw):
                return original_settings(*a, **kw) + ['run.@prd098_validation : app.exp_exit_step = 1280',
                    'run.@prd098_validation : app.batchrun.exp_exit_step = 1280']
            validation.settings = settings
            variants = ('default', 'merge', 'dueling', 'action_in')
        for variant in variants:
            overrides = {'default': [], 'merge': ['DefaultDQNAgent.learner.skip.feature_gradient=merge'],
                         'dueling': ['DefaultDQNAgent.skip.head.use_dueling=true'],
                         'action_in': ['DefaultDQNAgent.skip.network_mode=action_in',
                                      'DefaultDQNAgent.skip.net.branch.[skip_feature].bind=skip_context, action']}[variant]
            trace_q = args.stage == 'profiles' and env == 'GridMaze' and variant == 'default'
            if trace_q:
                overrides += [f'app.eval_trace_panel.columns.[{31+j}]=$action_info skip_q.[{j}] label:q{j}' for j in range(1, 8)]
            name = args.label + '_' + env + '_' + variant
            try:
                run = validation.execute(name, env, enabled=True, overrides=overrides)
                check_profile(run, variant, trace_q=trace_q)
                if args.stage == 'learning':
                    arms[variant] = run
                print('PASS', args.stage, env, variant, flush=True)
            except Exception as error:
                errors.append(name + ': ' + str(error))
                print('ERROR', errors[-1], flush=True)
    # J=1はenvの長さ別指標j>1を選ばず、9指標と1本のQ列で確認する。
    if args.stage == 'profiles':
        overrides = ['DefaultDQNAgent.skip.enabled=true', 'DefaultDQNAgent.skip.max_length=1',
            'net.block.[SkipFC1].linear.out_features=256',
            'metrics.scalar.$=metrics.scalar.@baseline > M1 > metrics.scalar.@noisy > metrics.scalar.@skip',
            'app.eval_trace_panel.columns.[32]=$action_info skip_q.[1] label:q1']
        try:
            run = validation.execute(args.label + '_GridMaze_width_j1', 'GridMaze', enabled=False, overrides=overrides)
            check_profile(run, 'default', width_j1=True, trace_q=True)
            print('PASS width override and explicit J=1', flush=True)
        except Exception as error:
            errors.append('width_j1: ' + str(error))
    if args.stage == 'learning' and len(arms) == 2:
        summarize_learning(root, arms)
    # 独立した条件を最後まで走査してから、全体の終了状態を返す。
    validation.save(root / (args.label + '_' + args.stage + '-summary.json'), {'errors': errors})
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
