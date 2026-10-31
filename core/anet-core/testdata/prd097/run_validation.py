"""PRD097 isolated, sequential before/after acceptance Runs."""
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
ROOT=REPO/'.scratch/prd097/implementation-20261020/protocol-fixed'
CASES=('atari1','atari2','dropmerge','lunarlander','gridmaze','imagecls','dropmerge_serial')
FILES={'atari1':'Atari.txt','atari2':'Atari.txt','dropmerge':'DropMerge.txt',
       'dropmerge_serial':'DropMerge.txt','lunarlander':'LunarLander.txt','gridmaze':'GridMaze.txt','imagecls':'ImageCls.txt'}
BUDGET=512000


def baseline_repeats(case):
    # 旧serial1/2はfile指定だったため、CLIを明示するD7受入は別番号に保存する。
    return (3,4) if case=='dropmerge_serial' else (1,2)


def after_repeat(case): return 3 if case=='dropmerge_serial' else 1


def comparison_name(case):
    return ('dropmerge_serial_cli' if case=='dropmerge_serial' else case)+'.comparison.json'


def target_for(side):
    label=os.environ.get('PRD097_AFTER_NAME','after-d7-v1') if side=='after' else 'before'
    if not label or Path(label).name!=label or label in ('.','..','before') and side=='after':
        raise RuntimeError('PRD097_AFTER_NAME must be a fresh directory name under the acceptance root')
    return ROOT/label

def write_json(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(side):
    target=target_for(side);target.mkdir(parents=True,exist_ok=True)
    binary=target/'bin/AnetRLRunner.exe'
    if not binary.exists():
        (target/'bin').mkdir(exist_ok=True)
        source=REPO/'apps/runner/bin/Release'
        shutil.copy2(source/'AnetRLRunner.exe',binary)
        for dll in source.glob('*.dll'): shutil.copy2(dll,target/'bin'/dll.name)
        source_config=ROOT/'before/source-config' if side=='after' else REPO/'apps/runner/config'
        shutil.copytree(source_config,target/'source-config')
        if side=='after':
            for name in ('Atari.txt','DropMerge.txt','GridMaze.txt'):
                shutil.copy2(REPO/'apps/runner/config'/name,target/'source-config'/name)
        write_json(target/'binary.json',{'sha256':digest(binary),'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()})
    if digest(binary)!=json.loads((target/'binary.json').read_text(encoding='utf-8'))['sha256']: raise RuntimeError('Saved executable changed')
    if side=='after' and digest(binary)!=digest(REPO/'apps/runner/bin/Release/AnetRLRunner.exe'):
        raise RuntimeError('Release executable changed after snapshot; select a fresh PRD097_AFTER_NAME')
    (target/'config').mkdir(exist_ok=True);(target/'runs').mkdir(exist_ok=True)
    return target,binary

def config(target,case):
    # 設定一式を固定し、出力先と実行用の明示指定だけを重ねる。
    source=(target/'source-config').as_posix()
    lines=[f'$include <{source}/_main.txt>',f'$include <{source}/{FILES[case]}>']
    overrides=['app.$=app.batchrun>P1','app.show_error_dialog=false',f'app.runs_dir={target.as_posix()}/runs',
      'run.seed=1','backend.$=backend.@deterministic','backend.deterministic_warn_only=false',
      'run.train.runner_type='+('serial' if case=='dropmerge_serial' else 'pipeline'),
      'run.eval_schedule.[eval_target].use_background=false','run.eval_schedule.[eval].use_background=false']
    chain='run.@prd097'
    if case.startswith('atari'):
        protocol='classic' if case=='atari1' else '100k'
        chain='run.@v5_iqn_impala_x2>run.@prd097'
        overrides += [f'AtariEnv.$=AtariEnv.@{protocol}>E1','E1.game=breakout']
        for tag in ['eval_target','eval','greedy']:
            overrides += [f'run.eval_schedule.[{tag}].use_background=false',f'run.eval_schedule.[{tag}].interval=1000',
                      f'run.eval.[{tag}].eval_batch_size=2',f'run.eval.[{tag}].eval_episodes=2']
    if case=='lunarlander':
        chain='run.@cpu>run.@prd097'
        overrides+=['agent.device=cpu','env.device=cpu','run.eval_device=cpu']
    lines += ['run.@prd097 : '+line for line in overrides]+['run.$='+chain]
    path=target/'config'/f'{case}.txt';content='\n'.join(lines)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8')!=content: raise RuntimeError('Frozen acceptance configuration changed: '+str(path))
    else: path.write_text(content,encoding='utf-8')
    return path

def run(side,case,repeat):
    target,binary=prepare(side);name=f'{case}_{repeat}';record=target/f'{name}.execution.json'
    if record.exists():
        saved=json.loads(record.read_text(encoding='utf-8'))
        if case=='dropmerge_serial' and 'run.train.runner_type=serial' not in saved.get('arguments',[]):
            raise RuntimeError('Historical serial run did not specify the runner on CLI; use a fresh repeat')
        if saved.get('exit_code')==0 and saved.get('sha256')==digest(binary) and Path(saved['run']).exists():
            print('REUSE',side,name,flush=True);return saved
        raise RuntimeError('Existing incomplete run record: '+str(record))
    cfg=config(target,case); short={'atari1':'atari','atari2':'a100','dropmerge':'drop','lunarlander':'lunar','gridmaze':'maze','imagecls':'img','dropmerge_serial':'drops'}[case]
    suffix=f'097_{short}{repeat}'
    args=[str(binary),'--config',str(cfg),f'app.run_name=run_{{t}}_{suffix}',f'app.exp_exit_step={BUDGET}',f'app.batchrun.exp_exit_step={BUDGET}']
    if case=='dropmerge_serial': args.append('run.train.runner_type=serial')
    data={'case':case,'side':side,'repeat':repeat,'sha256':digest(binary),'arguments':args,'seed':1,'budget':BUDGET,'start':datetime.datetime.now().isoformat()}
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
                old=json.loads((ROOT/'before'/f'{case}_{first}.execution.json').read_text(encoding='utf-8'));b=run('after',case,after_repeat(case));result=compare(old['run'],b['run'],True,True)
            write_json(target_for(args.side)/comparison_name(case),result)
            summary.append({'case':case,'exact_match':result['exact_match'],'points':result['new']['points']})
            print('COMPARE',case,result['exact_match'],'points',result['new']['points'],flush=True)
            if not result['exact_match']: errors.append(case+': numeric mismatch')
        except Exception as error:
            errors.append(case+': '+str(error));print('ERROR',errors[-1],flush=True)
    write_json(target_for(args.side)/'summary-d7.json',{'cases':summary,'errors':errors});raise SystemExit(1 if errors else 0)
