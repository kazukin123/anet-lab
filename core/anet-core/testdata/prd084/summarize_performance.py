"""Summarize paired Atari measurements without improvement gates."""
import datetime, json, statistics, sys
from collections import defaultdict
from pathlib import Path
root=Path(sys.argv[1])
results=[]
for path in sorted(root.glob('*.execution.json')):
    record=json.loads(path.read_text(encoding='utf-8-sig'))
    assert record.get('exit_code')==0, path
    suffix='prd084_perf_'+path.name.removesuffix('.execution.json')
    runs=[p for p in (root/'runs').iterdir() if p.name.endswith(suffix)]
    assert len(runs)==1, (path,runs)
    run=runs[0]
    config={}
    for line in (run/'config/config_data.txt').read_text(encoding='utf-8-sig').splitlines():
        if ' = ' in line:
            key,value=line.split(' = ',1);config[key]=value
    for schedule in ['eval','eval_target','greedy_dist']:
        assert config[f'run.eval_schedule.[{schedule}].interval']=='0', (path,schedule)
    assert config['run.train.num_envs']=='128' and config['run.seed']=='1'
    assert config['E1.game']=='breakout'
    declarations=[k for k in config if k.startswith('metrics.scalar.[47_agent_noisy/')]
    expected_count=0 if record['pair']=='diagnostics' and record['side']=='b' else 11
    assert len(declarations)==expected_count, (path,len(declarations))
    # CLI の同一キーは最後の指定が実効値。ベースと比較軸の上書きを照合する。
    arguments={}
    for argument in record['arguments']:
        if '=' in argument and not argument.startswith('app.run_name=') and not argument.split('=',1)[0].endswith('.$'):
            key,value=argument.split('=',1);arguments[key]=value
    for key,value in arguments.items():
        assert config.get(key)==value, (path,key,value,config.get(key))
    scalar=defaultdict(list)
    for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines():
        item=json.loads(line)
        if item.get('type')=='scalar':scalar[item['tag']].append(item)
    # warmup をまたぐ最初の区間を除き、区間全体が学習中の値だけを使う。
    throughput_points=scalar['90_perf/12_exp_step_per_sec']
    active=[current['value'] for previous,current in zip(throughput_points,throughput_points[1:]) if previous['step']>=200000]
    # ON/OFF が宣言だけでなく、実際の probe 実行へ反映されたことを確認する。
    probe_tags=['47_agent_noisy/09_q_abs_diff','47_agent_noisy/10_action_disagreement_ratio','47_agent_noisy/11_action_entropy']
    record['probe_metric_points']={tag:len(scalar[tag]) for tag in probe_tags}
    if record['pair']=='diagnostics':
        if record['side']=='a':assert all(record['probe_metric_points'].values()), path
        else:assert not any(record['probe_metric_points'].values()), path
    record['run']=str(run)
    record['full_run_exp_per_second']=record['experience_budget']/record['elapsed_seconds']
    record['active_metric_exp_per_second_mean']=statistics.mean(active) if active else None
    record['active_metric_points']=len(active)
    assert active, (path, 'No post-warmup throughput measurement')
    record['resolved_config']=config
    record['resolved_settings']={k:v for k,v in config.items() if k.startswith(('DefaultDQNAgent.nn_forward.', 'DefaultDQNAgent.actor.[train].nn_forward_key','DefaultDQNAgent.learner.use_amp','DefaultDQNAgent.head.projection.', 'net.block.[AtariHeadFC512'))}
    results.append(record)
expected_pairs={'off','actor_share','learner_share','hold_episode','hold_count32','sn_spectral','sn_cap','precision','diagnostics','learning_sn_none','learning_sn_spectral','learning_sn_spectral_cap'}
if len(sys.argv)>2:
    selected=set(sys.argv[2].split(','))
    assert selected and selected<=expected_pairs, selected
    expected_pairs=selected
assert {r['pair'] for r in results}==expected_pairs
repetitions=sorted({r['repeat'] for r in results})
assert len(repetitions)>=2 and repetitions==list(range(1,len(repetitions)+1))
assert len(results)==len(expected_pairs)*len(repetitions)*2, len(results)
# 開始・終了記録で逐次実行と A/B の交互順を検証する。
ordered_results=sorted(results,key=lambda r:r['start'])
for previous,current in zip(ordered_results,ordered_results[1:]):
    assert datetime.datetime.fromisoformat(previous['end'])<=datetime.datetime.fromisoformat(current['start']), (previous['pair'],current['pair'])
for name in expected_pairs:
    ordered_pair=[r for r in ordered_results if r['pair']==name]
    assert [(r['repeat'],r['side']) for r in ordered_pair]==[(repeat,side) for repeat in repetitions for side in ['a','b']], name
assert len({r['executable_sha256'] for r in results})==1
assert len({r['experience_budget'] for r in results})==1
assert len({(r['pair'],r['repeat'],r['side']) for r in results})==len(results)
# 比較軸以外の解決済み値が変わっていないことを全キーで確認する。
actor_prefix='DefaultDQNAgent.nn_forward.[perf_actor].noisy.'
body_prefix='net.block.[AtariHeadFC512Def].'
allowed_changes={
    'off':{'A2.head.projection.noisy.enabled','DefaultDQNAgent.head.projection.noisy.enabled',body_prefix+'noisy.enabled'},
    'actor_share':{actor_prefix+'share'},
    'learner_share':{'DefaultDQNAgent.nn_forward.[perf_learner].noisy.share'},
    'hold_episode':{actor_prefix+'hold'},
    'hold_count32':{actor_prefix+'hold',actor_prefix+'count'},
    'sn_spectral':{body_prefix+'weight_norm.mode'},
    'sn_cap':{body_prefix+'weight_norm.mode'},
    'precision':{body_prefix+'force_fp32'},
    'diagnostics':set(),
}
for mode in ['none','spectral','spectral_cap']:
    allowed_changes['learning_sn_'+mode]={'net.block.[AtariHeadFC512].noisy.enabled','DefaultDQNAgent.head.projection.noisy.enabled'}
axis_differences={}
for name in expected_pairs:
    for repeat in repetitions:
        a=next(r['resolved_config'] for r in results if r['pair']==name and r['repeat']==repeat and r['side']=='a')
        b=next(r['resolved_config'] for r in results if r['pair']==name and r['repeat']==repeat and r['side']=='b')
        differences={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k) and k!='app.run_name'}
        permitted=allowed_changes[name]
        unexpected={k:v for k,v in differences.items() if k not in permitted and not(name=='diagnostics' and k.startswith('metrics.scalar.[47_agent_noisy/'))}
        assert not unexpected, (name,repeat,unexpected)
        assert differences, (name,repeat,'Comparison axis was not applied')
        assert a['DefaultDQNAgent.learner.update_warmup_steps']==b['DefaultDQNAgent.learner.update_warmup_steps']=='200,000'
        if name not in axis_differences:axis_differences[name]=differences
        else:assert differences==axis_differences[name], (name,repeat,'Axis differs between repetitions')
for record in results:del record['resolved_config']
pairs={}
for name in sorted({r['pair'] for r in results}):
    sides={}
    for side in ['a','b']:
        rows=[r for r in results if r['pair']==name and r['side']==side]
        assert len(rows)==len(repetitions)
        sides[side]={key:statistics.mean(r[key] for r in rows) for key in ['elapsed_seconds','active_metric_exp_per_second_mean','full_run_exp_per_second','peak_private_bytes','peak_working_set_bytes','device_extra_mib']}
    pairs[name]={'a':sides['a'],'b':sides['b'],'b_minus_a':{k:sides['b'][k]-sides['a'][k] for k in sides['a']}}
    pairs[name]['median']={side:{key:statistics.median(r[key] for r in results if r['pair']==name and r['side']==side)
        for key in sides[side]} for side in ['a','b']}

report={'runs':results,'pairs':pairs,'axis_differences':axis_differences,'acceptance':'all runs exit0; no speed or score improvement gate',
        'memory_scope':'CPU private/working-set process peaks; GPU whole-device peak-minus-start, sampled every 5 seconds; not allocator-attributed',
        'budget':f"{results[0]['experience_budget']} experiences, unchanged 200000 warmup and learning schedules; {len(repetitions)} repetitions in alternating A/B order"}
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
lines=['# PRD084 性能測定','', f"Breakout / seed=1 / 128 env / {results[0]['experience_budget']:,}経験。各比較は A/B を{len(repetitions)}反復。速度・スコア改善は合否条件にしない。",'',
       'GPU はデバイス全体の5秒間隔の使用量から起動前の値を引く。CPU はプロセスの peak private bytes。経験/s は起動から終了までの全体値、学習中経験/s は warmup 200k 後のメトリクス平均。', '', 'OFF 行は A=Noisy ON / B=Noisy OFF。その他の比較は README の行対応を参照。','',
       '| 比較 | A 秒 | B 秒 | A 経験/s | B 経験/s | A 学習中経験/s | B 学習中経験/s | CPU追加 MiB B-A | GPU追加 MiB B-A |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,pair in pairs.items():
    a,b,d=pair['a'],pair['b'],pair['b_minus_a']
    lines.append(f"| {name} | {a['elapsed_seconds']:.2f} | {b['elapsed_seconds']:.2f} | {a['full_run_exp_per_second']:.1f} | {b['full_run_exp_per_second']:.1f} | {a['active_metric_exp_per_second_mean']:.1f} | {b['active_metric_exp_per_second_mean']:.1f} | {d['peak_private_bytes']/1048576:.2f} | {d['device_extra_mib']:.2f} |")
if len(sys.argv)>2:
    lines+=['','## Run ごとの値','',
        '| 比較 | 反復 | 側 | 秒 | 学習中経験/s | CPU peak MiB | GPU追加 MiB |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in ordered_results:
        lines.append(f"| {r['pair']} | {r['repeat']} | {r['side']} | {r['elapsed_seconds']:.3f} | {r['active_metric_exp_per_second_mean']:.1f} | {r['peak_private_bytes']/1048576:.2f} | {r['device_extra_mib']:.2f} |")
    lines+=['','## 3反復の中央値','',
        '| 比較 | 側 | 秒 | 学習中経験/s | CPU peak MiB | GPU追加 MiB |',
        '|---|---|---:|---:|---:|---:|']
    for name,pair in sorted(pairs.items()):
        for side,m in pair['median'].items():
            lines.append(f"| {name} | {side} | {m['elapsed_seconds']:.3f} | {m['active_metric_exp_per_second_mean']:.1f} | {m['peak_private_bytes']/1048576:.2f} | {m['device_extra_mib']:.2f} |")
(root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'runs':len(results),'pairs':len(pairs),'acceptance':report['acceptance']}))
