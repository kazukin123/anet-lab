"""Nsight SystemsのNVTX範囲と同じRunの時計からprobe費用を算出する。"""
from contextlib import closing
import sqlite3
import math

CAPTURE='Learner::CaptureAttentionEntropy'

def read_capture_cost(database, steps, window_seconds, process_seconds):
    if not all(math.isfinite(value) and value>0 for value in (window_seconds,process_seconds)):
        raise ValueError('Profiling clock duration must be finite and positive')
    # 文字列が直書きでも登録済みIDでも、同じ範囲名として扱う。
    with closing(sqlite3.connect(database)) as connection:
        rows=connection.execute('''SELECT n.start,n.end FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON n.textId=s.id
            WHERE COALESCE(n.text,s.value)=? AND n.end IS NOT NULL ORDER BY n.start''',(CAPTURE,)).fetchall()
        phases=connection.execute('''SELECT COALESCE(n.text,s.value),COUNT(*),SUM(n.end-n.start)/1e9
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
            WHERE COALESCE(n.text,s.value) IN (?,?) AND n.end IS NOT NULL GROUP BY COALESCE(n.text,s.value)''',
            ('CustomTransformerEncoderLayer::MeasureAttentionEntropy','anet::ComputeAttentionEntropy')).fetchall()
    if len(rows)!=len(steps) or not rows:
        raise ValueError(f'NVTX capture count={len(rows)} expected={len(steps)}')
    durations=[(end-start)/1e9 for start,end in rows]
    if any(seconds<0 for seconds in durations): raise ValueError('Invalid NVTX range duration')
    total=sum(durations)
    # 503の結果時刻より後、1006の結果時刻までに含まれるのは1006側のprobe。
    steady=sum(seconds for step,seconds in zip(steps,durations) if 503<step<=1006)
    return {'nested_scopes':{name:{'count':count,'seconds':seconds} for name,count,seconds in phases},
        'capture_count':len(rows),'capture_seconds_by_step':dict(zip(steps,durations)),
        'capture_seconds_total':total,'capture_ratio_process':total/process_seconds,
        'profile_capture_seconds':steady,'profile_capture_ratio':steady/window_seconds,
        'profile_window_seconds':window_seconds}



def add_recorded_run_cost(summary):
    import gzip
    import json
    from pathlib import Path

    # 初回を含む全probeの費用を、同じRunの最終通常時計点までの時間でも報告する。
    source=Path(summary['run'])/'metrics.jsonl'; opener=open
    if not source.exists(): source=source.with_suffix('.jsonl.gz'); opener=gzip.open
    recorded=0.0; last_probe_clock=0.0
    with opener(source,'rt',encoding='utf-8-sig') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('tag')=='90_perf/90_elapse_hour': recorded=max(recorded,float(row['value'])*3600)
            elif row.get('tag')=='90_perf/99_prd093_elapse_hour': last_probe_clock=max(last_probe_clock,float(row['value'])*3600)
    if not math.isfinite(recorded) or recorded<=0 or recorded<last_probe_clock:
        raise ValueError('Recorded Run clock does not cover all probe measurements')
    summary['recorded_run_clock_seconds']=recorded
    summary['capture_ratio_recorded_run']=summary['capture_seconds_total']/recorded
    summary['recorded_run_clock_note']='Last ordinary Run clock point; may precede shutdown, and includes all captured probes.'


def capture_run(nsys, env):
    # 本線の学習設定を保ち、profiling付きRunを通常の時間比較から分離する。
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import time
    import run_validation as validation
    from validate_run import analyze

    root=validation.ROOT; out=validation.OUT
    name='profile_'+env.lower(); directory=out/name
    running=subprocess.run(['powershell','-NoProfile','-Command',
        '@(Get-Process AnetRLRunner -ErrorAction SilentlyContinue).Count'],capture_output=True,text=True,check=True)
    if int(running.stdout.strip()): raise RuntimeError('Another Runner is open; profiling was not started')
    directory.mkdir()
    config=directory/'input.txt'
    config.write_text('\n'.join(validation.smoke_settings(name,env,True))+'\n',encoding='utf-8',newline='\n')
    executable=root/'apps/runner/bin/Release/AnetRLRunner.exe'
    report=directory/'nvtx.nsys-rep'; database=directory/'nvtx.sqlite'
    command=[str(nsys),'profile','--trace=nvtx','--sample=none','--cpuctxsw=none',
        '--show-output=true','--wait=primary','--force-overwrite=false','--output='+str(directory/'nvtx'),
        str(executable),'--config',str(config)]
    environment=os.environ.copy(); environment.pop('CUDA_VISIBLE_DEVICES',None)
    startup=subprocess.STARTUPINFO(); startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow=0
    start=time.perf_counter()
    with (directory/'capture.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=root/'apps/runner',env=environment,stdout=log,
            stderr=subprocess.STDOUT,startupinfo=startup)
    cli_seconds=time.perf_counter()-start
    metadata={'command':command,'exit_code':result.returncode,'profile_cli_seconds':cli_seconds,
        'executable_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
    validation.save(directory/'execution.json',metadata)
    if result.returncode or not report.exists(): raise RuntimeError('NVTX report was not generated: '+name)
    runs=sorted((out/'runs').glob('run_*_prd093_'+name))
    if not runs: raise RuntimeError('Profiling Run output is missing: '+name)
    summary=analyze(runs[-1],True,env=='Atari')
    with (directory/'export.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(nsys),'export','--type=sqlite','--output='+str(database),str(report)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    steps=sorted(int(step) for step in next(iter(summary['points'].values())))
    cost=read_capture_cost(database,steps,summary['wall_seconds_503_1006'],cli_seconds)
    cost['capture_ratio_cli']=cost.pop('capture_ratio_process')
    summary.update(cost,profile_report=str(report),profile_status='measured NVTX host inclusive ranges',
        profile_cli_seconds=cli_seconds,denominator='same profiled Run: clock(1006)-clock(503)',
        synchronization='Probe includes synchronous D2H and may wait for earlier GPU work; NVTX is host elapsed time, not GPU kernel time.')
    add_recorded_run_cost(summary)
    validation.save(directory/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    import argparse
    from pathlib import Path
    import run_validation as validation
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nsys',type=Path,required=True)
    arguments=parser.parse_args()
    if not arguments.nsys.is_file(): parser.error('nsys executable not found')
    profiles={env:capture_run(arguments.nsys,env) for env in ('DropMerge','Atari')}
    path=validation.OUT/'performance-summary.json'
    if path.exists():
        import json
        reports=json.loads(path.read_text(encoding='utf-8'))
        for report in reports:
            profile=profiles[report['env']]
            for key in ('profile_capture_seconds','profile_capture_ratio','profile_window_seconds','profile_status'):
                report[key]=profile[key]
            report['profile_run']=profile['run']
            report['profile_capture_seconds_total']=profile['capture_seconds_total']
            report['profile_recorded_run_clock_seconds']=profile['recorded_run_clock_seconds']
            report['profile_capture_ratio_recorded_run']=profile['capture_ratio_recorded_run']
        validation.save(path,reports)
