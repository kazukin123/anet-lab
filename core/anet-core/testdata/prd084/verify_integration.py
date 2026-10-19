import json, math, sys
from collections import defaultdict
from pathlib import Path
case, directory = sys.argv[1:]
run = Path(directory)
rows = defaultdict(list)
for line in (run / 'metrics.jsonl').read_text(encoding='utf-8').splitlines():
    item = json.loads(line)
    if item.get('type') == 'scalar': rows[item['tag']].append(item)
noisy = {tag: values for tag, values in rows.items() if tag.startswith('47_agent_noisy/')}
assert len(noisy) == 7, list(noisy)
# ScalarMetricObserver は既知 NaN を出力しない。4 feature 行の購読宣言と成立値の不在を別に確認する。
declarations = [line for line in (run/'config/config_data.txt').read_text(encoding='utf-8-sig').splitlines() if line.startswith('metrics.scalar.[47_agent_noisy/')]
assert len(declarations)==11, declarations
assert sum('_feature_' in line for line in declarations)==4
counts = {}
for tag, values in noisy.items():
    finite = [v for v in values if isinstance(v.get('value'), (float,int)) and math.isfinite(v['value'])]
    counts[tag] = {'points':len(values), 'finite':len(finite)}
    if '_feature_' in tag:
        assert not finite, (tag,finite[:2])
    else:
        assert len(finite)>=2, (tag,len(finite))
eval_rows = rows['52_eval2/10_game_score_mean']
assert len(eval_rows)>=2, len(eval_rows)
config = {}
for line in (run/'config/config_data.txt').read_text(encoding='utf-8-sig').splitlines():
    if ' = ' in line:
        key,value=line.split(' = ',1);config[key]=value
# 新契約では外側の AMP を継承し、Body だけを選択的に FP32 にする。
assert config['DefaultDQNAgent.learner.use_amp']=='true'
assert config['DefaultDQNAgent.learner.use_amp_bf16']=='true'
# 元チェーンの Actor は FP32。参照 profile が AMP 設定を上書きしないことも確認する。
assert config['DefaultDQNAgent.actor.[train].policy.use_amp']=='false'
assert 'A2.learner.use_amp' not in config
assert 'A2.actor.[train].policy.use_amp' not in config
assert 'A2.target_policy.use_amp' not in config
for purpose in ['current','target']:
    assert config[f'DefaultDQNAgent.learner.{purpose}.nn_forward_key']=='act_sample'
assert config['DefaultDQNAgent.nn_forward.[act_sample].noisy.hold']=='call'
assert config['DefaultDQNAgent.nn_forward.[act_sample].noisy.share']=='batch'
if case=='nz_reference':
    assert config['net.block.[AtariHeadFC512Def].force_fp32']=='true'
    actor_key=config['DefaultDQNAgent.actor.[train].nn_forward_key']
    prefix=f'DefaultDQNAgent.nn_forward.[{actor_key}].noisy.'
    assert config[prefix+'hold']=='episode' and config[prefix+'share']=='sample'
else:
    assert config['DefaultDQNAgent.actor.[train].nn_forward_key']=='act_sample'
result={'case':case,'run':str(run),'noisy':counts,'evaluation_points':len(eval_rows),'feature_population':'empty; known NaN is omitted by ScalarMetricObserver; contract-test evidence required','acceptance':'pass'}
output=run.parent.parent/(case+'.acceptance.json')
output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
