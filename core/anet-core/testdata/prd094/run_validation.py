"""PRD094 isolated, sequential before/after acceptance Runs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from compare_metrics import compare

REPO=Path(__file__).resolve().parents[4]
ROOT=Path(os.environ.get('PRD094_ACCEPTANCE_ROOT',str(REPO/'.scratch/prd094/acceptance-20261021')))
CASES=('atari1','dropmerge','lunarlander','gridmaze','actorapprox')
FILES={'atari1':'Atari.txt','dropmerge':'DropMerge.txt','lunarlander':'LunarLander.txt',
       'gridmaze':'GridMaze.txt','actorapprox':'LunarLander.txt'}
BUDGET=65536
FILES['actorapprox']='LunarLander.txt'


def baseline_repeats(case): return (1,2)

def after_repeat(case): return 1

def comparison_name(case): return case+'.comparison.json'


def target_for(side):
    label=os.environ.get('PRD094_AFTER_NAME','after-v1') if side=='after' else 'before'
    if not label or Path(label).name!=label or label in ('.','..','before') and side=='after':
        raise RuntimeError('PRD094_AFTER_NAME must be a fresh directory name under the acceptance root')
    return ROOT/label

def write_json(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(side):
    target=target_for(side);target.mkdir(parents=True,exist_ok=True)
    binary=target/'bin/AnetRLRunner.exe'
    if not binary.exists():
        (target/'bin').mkdir(exist_ok=True)
        source=Path(os.environ.get('PRD094_BINARY_DIR',str(REPO/'apps/runner/bin/Release')))
        shutil.copy2(source/'AnetRLRunner.exe',binary)
        for dll in source.glob('*.dll'): os.link(dll,target/'bin'/dll.name)
        source_config=REPO/'apps/runner/config'
        shutil.copytree(source_config,target/'source-config')
        if side=='after':
            for name in ('Atari.txt','DropMerge.txt','GridMaze.txt'):
                shutil.copy2(REPO/'apps/runner/config'/name,target/'source-config'/name)
        write_json(target/'binary.json',{'sha256':digest(binary),'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()})
    if digest(binary)!=json.loads((target/'binary.json').read_text(encoding='utf-8'))['sha256']: raise RuntimeError('Saved executable changed')
    if side=='after' and digest(binary)!=digest(Path(os.environ.get('PRD094_BINARY_DIR',str(REPO/'apps/runner/bin/Release')))/'AnetRLRunner.exe'):
        raise RuntimeError('Release executable changed after snapshot; select a fresh PRD094_AFTER_NAME')
    (target/'config').mkdir(exist_ok=True);(target/'runs').mkdir(exist_ok=True)
    return target,binary

def config(target,case):
    # 設定一式を固定し、出力先と実行用の明示指定だけを重ねる。
    source=(target/'source-config').as_posix()
    lines=[f'$include <{source}/_main.txt>',f'$include <{source}/{FILES[case]}>']
    overrides=['app.$=app.batchrun>P1','app.show_error_dialog=false',f'app.runs_dir={target.as_posix()}/runs',
      'run.seed=1','backend.$=backend.@deterministic','backend.deterministic_warn_only=false',
      'run.train.runner_type=pipeline',
      'run.eval_schedule.[eval_target].use_background=false','run.eval_schedule.[eval].use_background=false']
    chain='run.@prd094'
    if case.startswith('atari'):
        protocol='classic'
        chain='run.@v5_iqn_impala_x2>run.@prd094'
        overrides += [f'AtariEnv.$=AtariEnv.@{protocol}>E1','E1.game=breakout']
        for tag in ['eval_target','eval','greedy_dist']:
            overrides += [f'run.eval_schedule.[{tag}].use_background=false',f'run.eval_schedule.[{tag}].interval=1000',
                      f'run.eval.[{tag}].eval_batch_size=2',f'run.eval.[{tag}].eval_episodes=2']
    if case in ('lunarlander','actorapprox'):
        chain='run.@cpu>run.@prd094'
        overrides+=['agent.device=cpu','env.device=cpu','run.eval_device=cpu']
    if case=='actorapprox': overrides += ['DefaultDQNAgent.learner.use_per=true','DefaultDQNAgent.learner.per_initial_priority_mode=actor_approx']
    lines += ['run.@prd094 : '+line for line in overrides]+['run.$='+chain]
    path=target/'config'/f'{case}.txt';content='\n'.join(lines)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8')!=content: raise RuntimeError('Frozen acceptance configuration changed: '+str(path))
    else: path.write_text(content,encoding='utf-8')
    return path

def run(side,case,repeat):
    target,binary=prepare(side);name=f'{case}_{repeat}';record=target/f'{name}.execution.json'
    if record.exists():
        saved=json.loads(record.read_text(encoding='utf-8'))
        if saved.get('exit_code')==0 and saved.get('sha256')==digest(binary) and Path(saved['run']).exists():
            print('REUSE',side,name,flush=True);return saved
        raise RuntimeError('Existing incomplete run record: '+str(record))
    cfg=config(target,case); short={'atari1':'atari','dropmerge':'drop','lunarlander':'lunar','gridmaze':'maze','actorapprox':'approx'}[case]
    suffix=f'094_{short}{repeat}'
    args=[str(binary),'--config',str(cfg),f'app.run_name=run_{{t}}_{suffix}',f'app.exp_exit_step={BUDGET}',f'app.batchrun.exp_exit_step={BUDGET}']
    data={'case':case,'side':side,'repeat':repeat,'sha256':digest(binary),'arguments':args,'seed':1,'budget':BUDGET,'cwd':str(REPO/'apps/runner'),'start':datetime.datetime.now().isoformat()}
    write_json(record,data);print('START',side,name,flush=True);start=time.monotonic()
    with (target/f'{name}.stdout.log').open('w',encoding='utf-8') as out,(target/f'{name}.stderr.log').open('w',encoding='utf-8') as err:
        proc=subprocess.Popen(args,cwd=REPO/'apps/runner',stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        data['pid']=proc.pid;write_json(record,data);code=proc.wait()
    matches=list((target/'runs').glob('run_*_'+suffix))
    data.update(exit_code=code,elapsed_seconds=time.monotonic()-start,run=str(matches[0]) if len(matches)==1 else None)
    write_json(record,data);print('END',side,name,'exit',code,'seconds',round(data['elapsed_seconds'],1),flush=True)
    if code!=0 or len(matches)!=1: raise RuntimeError('Run failed: '+name)
    return data

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('side',choices=['before','after']);parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));args=parser.parse_args()
    errors=[];summary=[]
    for case in args.cases:
        try:
            if args.side=='before':
                first,second=baseline_repeats(case)
                a=run('before',case,first);b=run('before',case,second);result=compare(a['run'],b['run'])
            else:
                first=baseline_repeats(case)[0]
                old=json.loads((ROOT/'before'/f'{case}_{first}.execution.json').read_text(encoding='utf-8'));b=run('after',case,after_repeat(case));result=compare(old['run'],b['run'])
            write_json(target_for(args.side)/comparison_name(case),result)
            summary.append({'case':case,'exact_match':result['exact_match'],'points':result['new']['points']})
            print('COMPARE',case,result['exact_match'],'points',result['new']['points'],flush=True)
            if not result['exact_match']: errors.append(case+': numeric mismatch')
        except Exception as error:
            errors.append(case+': '+str(error));print('ERROR',errors[-1],flush=True)
    write_json(target_for(args.side)/'summary.json',{'cases':summary,'errors':errors});raise SystemExit(1 if errors else 0)
