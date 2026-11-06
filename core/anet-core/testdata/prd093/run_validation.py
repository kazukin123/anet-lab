"""PRD093 の現用設定比較、決定論 CPU 比較、両 smoke と費用測定。"""
import argparse
import ctypes
from ctypes import wintypes
import math
import os
import shutil
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from validate_run import analyze

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/".scratch/prd093"

def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")

class MemoryCounters(ctypes.Structure):
    _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
        ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
         "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage")]

def gpu_memory():
    result=subprocess.run(["nvidia-smi","--id=0","--query-gpu=memory.used","--format=csv,noheader,nounits"],
        capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return float(result.stdout.strip())

def execute(name, executable, lines, monitor_gpu=False, allow_parallel_cpu=False):
    # ユーザー許可の小型CPU検証だけをCUDA不可視で並走させ、本線GPU/性能は排他を保つ。
    result=subprocess.run(["powershell","-NoProfile","-Command","@(Get-Process AnetRLRunner -ErrorAction SilentlyContinue).Count"],capture_output=True,text=True,check=True)
    environment=os.environ.copy()
    if allow_parallel_cpu:
        if monitor_gpu or "agent.device = cpu" not in lines or "run.eval_device = cpu" not in lines:
            raise ValueError("Concurrent validation requires an explicitly CPU-only recipe")
        environment["CUDA_VISIBLE_DEVICES"]="-1"
    elif int(result.stdout.strip()):
        raise RuntimeError("Another Runner is open; validation was not started")
    record_path=OUT/(name+".execution.json")
    if record_path.exists(): raise FileExistsError("Existing evidence must be retained: "+str(record_path))
    config=OUT/(name+".txt")
    config.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    startup=subprocess.STARTUPINFO(); startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow=0
    memory=ctypes.WinDLL("psapi").GetProcessMemoryInfo
    memory.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]; memory.restype=wintypes.BOOL
    record={"name":name,"peak_working_set_bytes":0,"peak_private_bytes":0,
        "executable_sha256":hashlib.sha256(executable.read_bytes()).hexdigest(),
        "config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
        "allow_parallel_cpu":allow_parallel_cpu,"cuda_visible_devices":environment.get("CUDA_VISIBLE_DEVICES")}
    if monitor_gpu: record["gpu_before_mib"]=record["gpu_device_peak_mib"]=gpu_memory()
    start=time.perf_counter(); next_gpu=start
    with (OUT/(name+".launcher.log")).open("w",encoding="utf-8") as log:
        process=subprocess.Popen([str(executable),"--config",str(config)],cwd=ROOT/"apps/runner",stdout=log,stderr=subprocess.STDOUT,startupinfo=startup,env=environment)
        while process.poll() is None:
            counters=MemoryCounters(); counters.cb=ctypes.sizeof(counters)
            if memory(int(process._handle),ctypes.byref(counters),counters.cb):
                record["peak_working_set_bytes"]=max(record["peak_working_set_bytes"],counters.PeakWorkingSetSize)
                record["peak_private_bytes"]=max(record["peak_private_bytes"],counters.PrivateUsage)
            if monitor_gpu and time.perf_counter()>=next_gpu:
                record["gpu_device_peak_mib"]=max(record["gpu_device_peak_mib"],gpu_memory()); next_gpu=time.perf_counter()+1
            time.sleep(0.1)
    records=sorted((OUT/"runs").glob("run_*_prd093_"+name))
    record.update(exit_code=process.returncode,wall_seconds=time.perf_counter()-start,run=str(records[-1]) if records else None)
    save(record_path,record)
    if process.returncode or not records: raise RuntimeError("Validation failed: "+name)
    print(json.dumps(record),flush=True)
    return records[-1]


def settings(name, env, config_root):
    # 実在する評価定義だけを無効化し、未知tagのscheduleを追加しない。
    resolved=json.loads((OUT/"before-current"/(env+".json")).read_text(encoding="utf-8"))
    eval_tags=sorted({match[1] for key in resolved if (match:=re.match(r"^run\.eval\.\[([^]]+)\]\.",key))})
    return [f"$include <{config_root.as_posix()}/_main.txt>",f"$include <{config_root.as_posix()}/{env}.txt>",
        "app.$ = app.batchrun > P1", "app.show_error_dialog = false", "app.train_auto_start = true",
        "app.eval_panel.auto_start = false", "app.save_agent_on_close = false",
        f"app.runs_dir = {(OUT/'runs').as_posix()}", f"app.run_name = run_{{t}}_prd093_{name}",
        "run.seed = 931"]+[f"run.eval_schedule.[{tag}].interval = 0" for tag in eval_tags]


def cpu(name, before=False, enabled=False):
    config_root=OUT/"before/apps/runner/config" if before else ROOT/"apps/runner/config"
    lines=settings(name,"LunarLander",config_root)+[
        "agent.device = cpu", "run.eval_device = cpu", "backend.$ = backend.@deterministic",
        "backend.deterministic_warn_only = false", "backend.torch_num_threads = 1",
        "run.train.num_envs = 8", "DefaultDQNAgent.learner.replay_batch_size = 16",
        "DefaultDQNAgent.learner.replay_capacity = 4096", "DefaultDQNAgent.learner.update_warmup_steps = 256",
        "DefaultDQNAgent.learner.replay_ratio = 1", "DefaultDQNAgent.learner.use_amp = false",
        "DefaultDQNAgent.qr.num_quantiles = 8", "DefaultDQNAgent.learner.attention_entropy.probe.batch_size = 16",
        "app.exp_exit_step = 16640", "app.batchrun.exp_exit_step = 16640",
        "run.@prd093_cpu : DefaultDQNAgent.net.branch.[main_feature].structure = Flatten > P93FC > ReLU > P93Res > P93TF > Flatten",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93FC].type = Linear",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93FC].linear.out_features = 32",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93Res].type = Reshape",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93Res].reshape.dims = 4 8",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93TF].type = TransformerEncoder",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93TF].tf.d_model = 8",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93TF].tf.nhead = 2",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93TF].tf.num_layers = 4",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93TF].tf.dim_feedforward = 16",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93TF].tf.hidden_dropout_rate = 0.1",
        "run.@prd093_cpu : DefaultDQNAgent.net.block.[P93TF].tf.attn_dropout_rate = 0.1",
        "run.@prd093_cpu : metrics.scalar.$ = metrics.scalar.@baseline > metrics.scalar.@noisy > "+("metrics.scalar.@attention > " if enabled else "")+"M1 > M2",
        "run.$ = run.@cpu > run.@noisy > run.@prd093_cpu"]
    executable=(OUT/"before" if before else ROOT)/"apps/runner/bin/Debug/AnetRLRunner.exe"
    run=execute(name,executable,lines,allow_parallel_cpu=True)
    if enabled:
        definitions=json.loads((run/"json/metrics.scalar.defs.json").read_text(encoding="utf-8-sig"))["data"]
        expected={f"48_agent_attention/{i+1:02d}_entropy_layer{i}":f"attention_entropy.[{i}]" for i in range(4)}
        actual={tag:definition["source_key"] for tag,definition in definitions.items() if tag.startswith("48_agent_attention/")}
        if actual!=expected: raise ValueError("CPU attention definitions are incomplete")
        values={tag:{} for tag in expected}; source=run/"metrics.jsonl"; opener=open
        if not source.exists(): source=run/"metrics.jsonl.gz"; opener=gzip.open
        with opener(source,"rt",encoding="utf-8-sig") as stream:
            for line in stream:
                row=json.loads(line)
                if row.get("type")=="scalar" and row.get("tag") in values:
                    value=row.get("value")
                    if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
                        raise ValueError("Invalid CPU attention value: "+str(row))
                    values[row["tag"]][int(row["step"])]=value
        if any(not {503,1006}<=points.keys() for points in values.values()): raise ValueError("CPU attention measurement points are missing")
        save(OUT/"cpu-attention-values.json",values)
    return run

def compare():
    path=ROOT/"core/anet-core/testdata/prd084/compare_metrics.py"
    spec=importlib.util.spec_from_file_location("prd084_compare",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    outputs=[]
    for name in ("before_cpu","after_off_cpu","after_on_cpu"):
        record=json.loads((OUT/(name+".execution.json")).read_text())
        destination=OUT/(name+".normalized.jsonl")
        module.normalize(record["run"],str(destination),True,["48_agent_attention/"])
        outputs.append(destination)
    for left,right in zip(outputs,outputs[1:]):
        try: module.compare(str(left),str(right))
        except SystemExit as result:
            if result.code: raise RuntimeError("Learning metrics differ")
    save(OUT/"cpu-comparison.json",{"match":True})

def snapshot_compare():
    # 改名単独の比較。履歴資料のキーや値を書き換えない。
    reports=[]
    for source in (OUT/"before-current").glob("*.json"):
        old=json.loads(source.read_text(encoding="utf-8")); new=json.loads((OUT/"after-rename"/source.name).read_text(encoding="utf-8"))
        expected={key.replace("learner.plasticity.feature_key","learner.final_feature_branch_key").replace("ImageClsAgent.plasticity.feature_key","ImageClsAgent.final_feature_branch_key"):value for key,value in old.items()}
        delta=[key for key in sorted(expected.keys()|new.keys()) if expected.get(key)!=new.get(key)]
        reports.append({"input":source.name,"keys":len(new),"differences":delta})
    save(OUT/"rename-comparison.json",reports)
    if len(reports)!=7 or any(row["differences"] for row in reports): raise RuntimeError("Unexpected config changes")

def rename_snapshot():
    # 改名だけの設定入力を分離し、追加profileによる差分と混ぜずに比較する。
    source=OUT/"before/apps/runner/config"; destination=OUT/"rename-config"
    if destination.exists(): raise FileExistsError(destination)
    shutil.copytree(source,destination)
    for path in destination.glob("*.txt"):
        text=path.read_text(encoding="utf-8-sig").replace("learner.plasticity.feature_key","learner.final_feature_branch_key").replace("ImageClsAgent.plasticity.feature_key","ImageClsAgent.final_feature_branch_key")
        path.write_text(text,encoding="utf-8",newline="\n")
    environment=os.environ.copy(); environment["ANET_PRD093_OUTPUT"]=str(OUT/"after-rename")
    environment["ANET_PRD093_CONFIG_ROOT"]=str(destination)
    with (OUT/"rename-snapshot.log").open("w",encoding="utf-8") as log:
        subprocess.run([str(ROOT/"core/anet-core/bin/Debug/anet-core-test.exe"),"[prd093-config-snapshot]"],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    snapshot_compare()

def progress_rows():
    return ["metrics.scalar.[90_perf/98_prd093_learn_step] = learn_step @learn $learn_step $runner interval:1",
        "metrics.scalar.[90_perf/99_prd093_elapse_hour] = elapse_hour @learn $learn_step $runner interval:503"]

def smoke_settings(name, env, enabled):
    config_root=ROOT/"apps/runner/config"
    text=(config_root/(env+".txt")).read_text(encoding="utf-8-sig")
    defaults=re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)",text)
    chain=defaults[-1].strip() if defaults else ""
    if env=="Atari": chain += (" > " if chain else "")+"run.@vitva"
    if enabled: chain += (" > " if chain else "")+"run.@attention"
    # 保存した既定のB/RR/warmupから1006点を越える予算を求める。学習スケジュールは縮めない。
    raw=json.loads((OUT/"before-current"/(env+".json")).read_text(encoding="utf-8"))
    def number(key): return float(raw["DefaultDQNAgent.learner."+key].replace(",",""))
    batch=number("replay_batch_size"); ratio=number("replay_ratio"); warmup=number("update_warmup_steps")
    if ratio<=0: raise ValueError("This acceptance recipe requires a positive default replay_ratio")
    envs=int(raw["run.train.num_envs"])
    budget=int(warmup+math.ceil(1088*batch/ratio/envs)*envs)
    lines=settings(name,env,config_root)+["agent.device = cuda",f"app.exp_exit_step = {budget}",f"app.batchrun.exp_exit_step = {budget}"]+progress_rows()
    # 既定Runがapp名・終了予算・評価scheduleを上書きするため、検証条件は末尾profileで固定する。
    profile="run.@prd093_validation"
    includes=[line for line in lines if line.startswith("$include")]
    overrides=[f"{profile} : {line}" for line in lines if not line.startswith("$include")]
    chain += (" > " if chain else "")+profile
    return includes+overrides+["run.$ = "+chain]

def smoke(environments=("DropMerge","Atari")):
    for env in environments:
        name="smoke_"+env.lower()
        run=execute(name,ROOT/"apps/runner/bin/Debug/AnetRLRunner.exe",smoke_settings(name,env,True),True)
        report=analyze(run,True,env=="Atari"); save(OUT/(name+".summary.json"),report)
        with (OUT/(name+".tags.txt")).open("w",encoding="utf-8") as log:
            subprocess.run([sys.executable,str(ROOT/"viewers/metrics-tools/inspect_run.py"),"tags",str(run)],cwd=ROOT,stdout=log,check=True)

def performance():
    # 同じReleaseでOFF/ONを交互に実行し、503..1006の同一更新区間を比べる。
    reports=[]
    for env in ("DropMerge","Atari"):
        pair=[]
        for enabled in (False,True):
            name="perf_"+env.lower()+("_on" if enabled else "_off")
            run=execute(name,ROOT/"apps/runner/bin/Release/AnetRLRunner.exe",smoke_settings(name,env,enabled),True)
            report=analyze(run,enabled,env=="Atari"); save(OUT/(name+".summary.json"),report); pair.append(report)
        reports.append({"env":env,"off":pair[0],"on":pair[1],
            "wall_change_ratio":pair[1]["wall_seconds_503_1006"]/pair[0]["wall_seconds_503_1006"]-1,
            "profile_capture_seconds":None,"profile_capture_ratio":None,
            "profile_status":"pending NVTX/Tracy capture; wall change is not the sum of diagnostic ranges"})
    save(OUT/"performance-summary.json",reports)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["before-cpu","cpu","compare","rename-compare","rename-snapshot","smoke","performance"])
    parser.add_argument("--env",choices=["DropMerge","Atari"],help="Select one smoke environment when resuming")
    arguments=parser.parse_args()
    stage=arguments.stage
    if arguments.env and stage!="smoke": parser.error("--env requires the smoke stage")
    OUT.mkdir(parents=True,exist_ok=True)
    if stage=="before-cpu": cpu("before_cpu",True)
    elif stage=="cpu": cpu("after_off_cpu"); cpu("after_on_cpu",enabled=True); compare()
    elif stage=="compare": compare()
    elif stage=="rename-compare": snapshot_compare()
    elif stage=="rename-snapshot": rename_snapshot()
    elif stage=="smoke": smoke((arguments.env,) if arguments.env else ("DropMerge","Atari"))
    else: performance()
