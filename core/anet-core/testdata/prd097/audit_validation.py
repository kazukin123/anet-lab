"""Audit PRD097 resolved settings, artifacts and frozen snapshot hashes."""
import argparse
import hashlib
import gzip
import math
import json
from pathlib import Path
from run_validation import ROOT, CASES, BUDGET, write_json, target_for, baseline_repeats, after_repeat, comparison_name
from compare_metrics import LIVES


def config(path):
    return dict(line.split(' = ',1) for line in path.read_text(encoding='utf-8-sig').splitlines() if ' = ' in line)


def expect(values,key,value):
    actual=values.get(key)
    if actual!=str(value): raise RuntimeError(f'{key}: expected={value} actual={actual}')


def finite_loss_count(run):
    # タグの存在だけでなく、実学習で成立したlossを確認する。
    path=Path(run)/'metrics.jsonl'
    opener=open
    if not path.is_file(): path=path.with_suffix('.jsonl.gz');opener=gzip.open
    count=0
    with opener(path,'rt',encoding='utf-8-sig') as stream:
        for line in stream:
            row=json.loads(line)
            value=row.get('value')
            if row.get('type')=='scalar' and row.get('tag')=='38_agent_loss/01_loss' and value is not None and math.isfinite(value): count+=1
    return count


def audit(side, output_name="audit.json"):
    target=target_for(side)
    results=[]
    for case in CASES:
        comparison=json.loads((target/comparison_name(case)).read_text(encoding='utf-8'))
        if not comparison['exact_match']: raise RuntimeError(case+': comparison mismatch')
        for repeat in (baseline_repeats(case) if side=='before' else (after_repeat(case),)):
            execution=json.loads((target/f'{case}_{repeat}.execution.json').read_text(encoding='utf-8'))
            if execution['exit_code']!=0: raise RuntimeError(case+': Run failed')
            if case=='dropmerge_serial' and 'run.train.runner_type=serial' not in execution['arguments']:
                raise RuntimeError(case+': runner was not specified on CLI')
            run=Path(execution['run']);folder=run/'config'
            expect(config(folder/'run.txt'),'run.seed',1)
            expect(config(folder/'run.txt'),'run.train.runner_type','serial' if case=='dropmerge_serial' else 'pipeline')
            expect(config(folder/'app.txt'),'app.exp_exit_step',BUDGET)
            backend=config(folder/'backend.txt')
            expect(backend,'backend.deterministic_algorithms','true')
            expect(backend,'backend.deterministic_warn_only','false')
            if case=='lunarlander':
                expect(config(folder/'agent.txt'),'agent.device','cpu')
                expect(config(folder/'run.txt'),'run.eval_device','cpu')
            if case.startswith('atari'):
                train=config(folder/'env.train.txt')
                expect(train,'AtariEnv.game','breakout')
                expect(train,'AtariEnv.episodic_life','true')
                expect(train,'AtariEnv.fire_reset','true' if case=='atari1' else 'false')
                resolved=config(folder/'config_data.txt')
                for tag in ('eval_target','eval','greedy'):
                    expect(resolved,f'run.eval_schedule.[{tag}].interval',1000)
                    expect(resolved,f'run.eval_schedule.[{tag}].use_background','false')
                    expect(resolved,f'run.eval.[{tag}].eval_batch_size',2)
                    expect(resolved,f'run.eval.[{tag}].eval_episodes',2)
                    expect(config(folder/f'env.{tag}.txt'),f'run.eval.[{tag}].env.episodic_life','false')
                counts=comparison['old' if repeat==baseline_repeats(case)[0] and side=='before' else 'new']['counts']
                if counts.get('scalar:38_agent_loss/01_loss',0)<=0 or finite_loss_count(run)<=0:
                    raise RuntimeError(case+': no finite learning loss')
                for tag in LIVES:
                    if counts.get('scalar:'+tag,0)<=0: raise RuntimeError(case+': missing '+tag)
                for group in ('51_eval1','52_eval2','53_evalg'):
                    if counts.get('trace:'+group+'/episode',0)!=4: raise RuntimeError(case+': evaluation did not finish: '+group)
            results.append({'case':case,'repeat':repeat,'run':str(run),'exit_code':0,'resolved_contract':True})
    # DLL・入力設定・実効設定を含め、保存したファイルをストリームでハッシュする。
    paths=list((target/'bin').glob('*'))+list((target/'source-config').rglob('*'))+list((target/'config').glob('*.txt'))
    for item in results: paths+=list((Path(item['run'])/'config').glob('*.txt'))
    hashes={}
    for path in sorted(set(paths)):
        if path.is_file():
            with path.open('rb') as stream: value=hashlib.file_digest(stream,'sha256').hexdigest()
            hashes[path.relative_to(target).as_posix()]=value
    write_json(target/output_name,{'accepted':True,'runs':results,'sha256':hashes})
    print(json.dumps({'side':side,'accepted':True,'runs':len(results),'hashed_files':len(hashes)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('side',choices=['before','after']);parser.add_argument('--output-name',default='audit.json');args=parser.parse_args();audit(args.side,args.output_name)
