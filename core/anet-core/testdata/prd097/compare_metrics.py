"""PRD097: compare every scalar and trace record without rewriting Run artifacts."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

EXCLUDED = ('90_perf/', '91_resource/', '92_resource/')
LIVES = frozenset(('42_env/30_lives_mean', '42_env/31_lives_mean_ema',
    '51_eval1/30_lives_mean', '51_eval1/31_lives_mean_ema',
    '52_eval2/30_lives_mean', '52_eval2/31_lives_mean_ema', '53_evalg/30_lives_mean'))

def exact(value):
    # 保存された数値の符号付きゼロも残し、trace の配列順序を維持する。
    if isinstance(value, float): return {'float': value.hex()}
    if isinstance(value, list): return [exact(v) for v in value]
    if isinstance(value, dict): return {k: exact(v) for k,v in sorted(value.items())}
    return value

def read_run(run, allow_lives=False, excluded_edges=None):
    source=Path(run)/'metrics.jsonl'
    opener=open
    if not source.is_file(): source=source.with_suffix('.jsonl.gz'); opener=gzip.open
    records=[]; counts=Counter(); last_step=0; speeds=[]
    with opener(source,'rt',encoding='utf-8-sig') as stream:
        for line in stream:
            obj=json.loads(line); kind=obj.get('type'); tag=obj.get('tag','')
            if kind not in ('scalar','trace') or tag.startswith(EXCLUDED):
                if kind=='scalar' and tag=='90_perf/12_exp_step_per_sec' and obj['value'] is not None:
                    speeds.append((obj['step'],obj['value']))
                continue
            step=int(obj['step'])
            if kind=='scalar' and excluded_edges and step in excluded_edges.get(tag,()): continue
            counts[kind+':'+tag]+=1
            if kind=='scalar':
                value=None if allow_lives and tag in LIVES else obj['value']
                value=None if value is None else float(value).hex()
                row=(kind,tag,step,value)
            else: row=(kind,tag,step,int(obj['lane']),exact(obj['data']))
            records.append(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False))
    # 異なるtag間の書込順に依存せず、同じ行の重複はそのまま数える。
    records.sort()
    if speeds:
        last_step=max(step for step,_ in speeds)
    import statistics
    speed=statistics.median([value for step,value in speeds if step>last_step/2]) if speeds else None
    payload='\n'.join(records).encode('utf-8')
    return records,{'points':len(records),'sha256':hashlib.sha256(payload).hexdigest(),
        'counts':dict(sorted(counts.items())),'performance_median':speed,'performance_last_step':last_step}

def migrated_edges(old,new):
    # 定義の差で指定された移行tagだけを認め、他tagや内側の座標は一切除外しない。
    def definitions(run):
        obj=json.loads((Path(run)/'json/metrics.scalar.defs.json').read_text(encoding='utf-8-sig'))
        return obj.get('data',obj)
    def config(run,name):
        return dict(line.split(' = ',1) for line in (Path(run)/'config'/name).read_text(encoding='utf-8-sig').splitlines() if ' = ' in line)
    run_config=config(old,'run.txt')
    if run_config['run.train.runner_type']!='pipeline': return {}
    n=int(run_config['run.train.num_envs']);budget=int(config(old,'app.txt')['app.exp_exit_step'].replace(',',''))
    steps=(budget+n-1)//n
    a=definitions(old);b=definitions(new);edges={}
    for tag,d in a.items():
        e=b.get(tag,{})
        if d.get('target')=='env' and d.get('event')=='train' and e.get('target')=='env_step' and e.get('event')=='episode_end':
            if d['step_axis']!=e['step_axis'] or d['source_key']!=e['source_key']:
                raise RuntimeError('Migrated metric changed meaning: '+tag)
            axis=d['step_axis']
            if axis not in ('exp_step','train_step'): raise RuntimeError('Unsupported migrated step axis: '+axis)
            edges[tag]=[0,(steps-1)*n] if axis=='exp_step' else [0,steps-1]
    return edges


def compare(old,new,allow_lives=False,allow_migrated_edges=False):
    edges=migrated_edges(old,new) if allow_migrated_edges else {}
    a,sa=read_run(old,allow_lives,edges); b,sb=read_run(new,allow_lives,edges)
    result={'exact_match':a==b,'allow_lives_values':allow_lives,'excluded_migrated_edges':edges,'old':sa,'new':sb}
    if a!=b:
        left=Counter(a)-Counter(b); right=Counter(b)-Counter(a)
        result['old_only']=[json.loads(row) for row in list(left)[:10]]
        result['new_only']=[json.loads(row) for row in list(right)[:10]]
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old');parser.add_argument('new');parser.add_argument('--allow-lives-values',action='store_true')
    parser.add_argument('--allow-migrated-edges',action='store_true')
    parser.add_argument('--output')
    args=parser.parse_args();result=compare(args.old,args.new,args.allow_lives_values,args.allow_migrated_edges)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output: Path(args.output).write_text(text+'\n',encoding='utf-8')
    print(json.dumps({'exact_match':result['exact_match'],'old_points':result['old']['points'],'new_points':result['new']['points']}))
    raise SystemExit(0 if result['exact_match'] else 1)
