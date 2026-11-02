"""PRD094: compare every scalar and trace record without rewriting Run artifacts."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

EXCLUDED = ('90_perf/', '91_resource/', '92_resource/')

def exact(value):
    # 保存された数値の符号付きゼロも残し、trace の配列順序を維持する。
    if isinstance(value, float): return {'float': value.hex()}
    if isinstance(value, list): return [exact(v) for v in value]
    if isinstance(value, dict): return {k: exact(v) for k,v in sorted(value.items())}
    return value

def read_run(run):
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
            counts[kind+':'+tag]+=1
            if kind=='scalar':
                value=obj['value']
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

def compare(old,new):
    a,sa=read_run(old);b,sb=read_run(new)
    result={'exact_match':a==b,'old':sa,'new':sb}
    if a!=b:
        result['old_only']=[json.loads(x) for x in list(Counter(a)-Counter(b))[:10]]
        result['new_only']=[json.loads(x) for x in list(Counter(b)-Counter(a))[:10]]
    for name in ('metrics.scalar.defs','metrics.trace.defs'):
        def load(run):
            p=Path(run)/'json'/(name+'.json')
            return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else None
        result[name]=load(old)==load(new)
        result['exact_match'] &= result[name]
    def warnings(run):
        root=Path(run)
        source=root/(root.name+'.log')
        # 出力元の時刻だけを外し、本文・順序・件数は維持する。
        return [line.split('[W]',1)[1] for line in source.read_text(encoding='utf-8-sig').splitlines() if '[W]' in line]
    result['warnings_match']=warnings(old)==warnings(new)
    result['exact_match'] &= result['warnings_match']
    return result
