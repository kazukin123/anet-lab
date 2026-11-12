"""PRD092のCPU比較、本線smoke、OFF/ON各3回の性能測定を再現する。"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
EVIDENCE=ROOT/".scratch/prd092"
SOURCE=ROOT/"apps/runner/workspaces/DropMerge-02/runs/run_20261014-080249_dm_fb_munch_tau3e-2_ndtpen20_nzrep_c100_b512rr2_20m"
WARMUP=503
END=1509
PROGRESS="90_perf/98_prd092_learn_step"
CLOCK="90_perf/99_prd092_elapse_hour"
IDLE_RUNNER_PIDS=set()
RELEASE_EXECUTABLE=ROOT/"apps/runner/bin/Release/AnetRLRunner.exe"

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

VALIDATOR=module("prd092_validate",Path(__file__).with_name("validate_run.py"))

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")

def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def gradient_rows():
    rows=[]
    prefix="metrics.scalar.@gradnoise : "
    for line in (ROOT/"apps/runner/config/metrics_scalar.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(prefix):
            rows.append("metrics.scalar."+line[len(prefix):].replace(" ?= "," = ",1))
    if len(rows)!=15:
        raise ValueError("Expected exactly fifteen gradnoise definitions")
    return rows

def progress_rows():
    return [f"metrics.scalar.[{PROGRESS}] = learn_step @learn $learn_step $runner interval:1",
            f"metrics.scalar.[{CLOCK}] = elapse_hour @learn $learn_step $runner interval:503"]

def ensure_exclusive():
    # 既存Runは停止しない。人間が停止中と確認したPIDだけを明示的に許容する。
    check=subprocess.run(["powershell","-NoProfile","-Command",
        "@(Get-CimInstance Win32_Process -Filter \"Name='AnetRLRunner.exe'\" | Select-Object ProcessId,ExecutablePath) | ConvertTo-Json -Compress"],
        capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW,check=True)
    found=json.loads(check.stdout) if check.stdout.strip() else []
    if isinstance(found,dict):
        found=[found]
    active=[entry for entry in (found or []) if entry["ProcessId"] not in IDLE_RUNNER_PIDS]
    if active:
        raise RuntimeError("An unconfirmed Runner is open; close or confirm it idle before validation: "+json.dumps(active))

class MemoryCounters(ctypes.Structure):
    _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
        ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
         "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage")]

MEMORY=ctypes.WinDLL("psapi").GetProcessMemoryInfo
MEMORY.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
MEMORY.restype=wintypes.BOOL

def gpu_used():
    result=subprocess.run(["nvidia-smi","--id=0","--query-gpu=memory.used","--format=csv,noheader,nounits"],
        capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW,check=True)
    return float(result.stdout.strip())

def execute(name,executable,settings,monitor_gpu=False):
    ensure_exclusive()
    config=EVIDENCE/(name+".txt")
    config.write_text("\n".join(settings)+"\n",encoding="utf-8",newline="\n")
    args=[str(executable),"--config",str(config)]
    startup=subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=0
    record={"name":name,"arguments":args,"confirmed_idle_runner_pids":sorted(IDLE_RUNNER_PIDS),"executable_sha256":hash_file(executable),
            "config_sha256":hash_file(config),"peak_working_set_bytes":0,"peak_private_bytes":0,
            "gpu_sampling_seconds":0.5,"gpu_measurement":"device 0 memory.used sampled by nvidia-smi"}
    if monitor_gpu:
        record["gpu_before_mib"]=gpu_used()
        record["peak_gpu_used_mib"]=record["gpu_before_mib"]
    with (EVIDENCE/(name+".launcher.log")).open("w",encoding="utf-8") as log:
        start=time.perf_counter()
        process=subprocess.Popen(args,cwd=ROOT/"apps/runner",stdout=log,stderr=subprocess.STDOUT,startupinfo=startup)
        next_gpu=start
        while process.poll() is None:
            counters=MemoryCounters()
            counters.cb=ctypes.sizeof(counters)
            if MEMORY(int(process._handle),ctypes.byref(counters),counters.cb):
                record["peak_working_set_bytes"]=max(record["peak_working_set_bytes"],counters.PeakWorkingSetSize)
                record["peak_private_bytes"]=max(record["peak_private_bytes"],counters.PrivateUsage)
            if monitor_gpu and time.perf_counter()>=next_gpu:
                record["peak_gpu_used_mib"]=max(record["peak_gpu_used_mib"],gpu_used())
                next_gpu=time.perf_counter()+0.5
            time.sleep(0.1)
        record["exit_code"]=process.returncode
        record["process_wall_seconds"]=time.perf_counter()-start
    candidates=sorted((EVIDENCE/"runs").glob("run_*_prd092_"+name))
    if candidates:
        record["run"]=str(candidates[-1])
    write_json(EVIDENCE/(name+".execution.json"),record)
    if record["exit_code"]!=0 or "run" not in record:
        raise RuntimeError("Validation run failed: "+name)
    logfile=Path(record["run"])/(Path(record["run"]).name+".log")
    if "Draining background observers" not in logfile.read_text(encoding="utf-8-sig"):
        raise RuntimeError("Incomplete shutdown log: "+name)
    print(json.dumps({key:record[key] for key in ("name","exit_code","process_wall_seconds","run")}),flush=True)
    return record

def cpu():
    before=EVIDENCE/"cpu-before.txt"
    template=before.read_text(encoding="utf-8")
    template=template.replace((EVIDENCE/"before/config").as_posix(),(ROOT/"apps/runner/config").as_posix())
    records=[]
    for side in ("off","on"):
        name="after_"+side+"_cpu"
        settings=template.replace("prd092_before_cpu","prd092_"+name).splitlines()+progress_rows()
        if side=="on":
            settings += ["run.@prd092_gradnoise : metrics.scalar.$ = metrics.scalar.@baseline > metrics.scalar.@iqn_search_p0 > metrics.scalar.@munchausen > metrics.scalar.@noisy > metrics.scalar.@gradnoise > M1",
                         "run.$ = run.@cpu > run.@noisy > run.@prd092_gradnoise"]
        record=execute(name,ROOT/"apps/runner/bin/Debug/AnetRLRunner.exe",settings)
        if side=="on":
            write_json(EVIDENCE/"cpu-on-summary.json",VALIDATOR.analyze(Path(record["run"]),record["exit_code"],require_all=True))
        records.append(record)
    baseline_result=json.loads((EVIDENCE/"before-cpu-result.json").read_text(encoding="utf-8"))
    if baseline_result["exit_code"]!=0:
        raise RuntimeError("Before baseline did not complete")
    baseline=sorted((EVIDENCE/"runs").glob("run_*_prd092_before_cpu"))[-1]
    compare=module("prd084_compare",Path(__file__).parents[1]/"prd084/compare_metrics.py")
    files=[]
    for label,run in [("before",baseline)]+[(record["name"],Path(record["run"])) for record in records]:
        output=EVIDENCE/(label+".normalized.jsonl")
        compare.normalize(str(run),str(output),True,["49_agent_gradnoise/"])
        files.append(output)
    for old,new in zip(files,files[1:]):
        try:
            compare.compare(str(old),str(new))
        except SystemExit as result:
            if result.code:
                raise RuntimeError("Existing numeric metrics changed")
    compare_config=Path(__file__).parents[1]/"prd084/compare_resolved_settings.py"
    with (EVIDENCE/"resolved-settings.json").open("w",encoding="utf-8") as log:
        subprocess.run([sys.executable,str(compare_config),str(baseline/"config/config_data.txt"),
                        str(Path(records[0]["run"])/"config/config_data.txt")],stdout=log,check=True)
    write_json(EVIDENCE/"cpu-comparison.json",{"match":True,"runs":[str(baseline)]+[x["run"] for x in records]})

def main_settings(name,budget,enabled,save=False):
    lines=["$include <"+(SOURCE/"config/config_data.txt").as_posix()+">",
        "DefaultDQNAgent.auto_load_file="+(SOURCE/"agent_close.anet").as_posix(),
        "agent.device=cuda","run.seed=920","DefaultDQNAgent.learner.replay_batch_size=256",
        "DefaultDQNAgent.learner.replay_ratio=1","DefaultDQNAgent.learner.grad_noise.num_micro_batches=8",
        "DefaultDQNAgent.learner.grad_noise.ema_alpha=0.05","app.train_auto_start=true",
        "app.show_error_dialog=false","app.eval_panel.auto_start=false",
        "app.save_agent_on_close="+str(save).lower(),"app.exp_exit_step="+str(budget),
        "app.batchrun.exp_exit_step="+str(budget),"app.runs_dir="+(EVIDENCE/"runs").as_posix(),
        "app.run_name=run_{t}_prd092_"+name,
        "run.eval_schedule.[eval].interval=0","run.eval_schedule.[eval_target].interval=0",
        "run.eval_schedule.[eval_target_greedy].interval=0"]+progress_rows()
    if enabled:
        lines+=gradient_rows()
    return lines

def smoke():
    # warmup境界のbatch分を見込み、実更新505回以上とinterval 503の2点を確保する。
    budget=25600+512*256
    record=execute("smoke",ROOT/"apps/runner/bin/Debug/AnetRLRunner.exe",main_settings("smoke",budget,True,True),True)
    summary=VALIDATOR.analyze(Path(record["run"]),record["exit_code"],require_all=True)
    if summary["updates"]<505:
        raise RuntimeError("Smoke requires at least 505 actual learner updates")
    write_json(EVIDENCE/"smoke-summary.json",summary)

def performance():
    write_json(EVIDENCE/"performance-input.json",{"checkpoint":str(SOURCE/"agent_close.anet"),
        "checkpoint_sha256":hash_file(SOURCE/"agent_close.anet"),"warmup_learn_steps":WARMUP,
        "measured_learn_steps":END-WARMUP,"batch_size":256,"replay_ratio":1,
        "num_micro_batches":8,"interval":503,"ema_alpha":0.05})
    records=[]
    for repetition in range(1,4):
        for enabled in (False,True):
            name=f"perf_{repetition}_{'on' if enabled else 'off'}"
            record=execute(name,RELEASE_EXECUTABLE,
                main_settings(name,25600+(END+3)*256,enabled),True)
            summary=VALIDATOR.analyze(Path(record["run"]),record["exit_code"],require_all=enabled,start_step=WARMUP,end_step=END)
            record["measurement"]=summary
            record["enabled"]=enabled
            records.append(record)
            write_json(EVIDENCE/(name+".summary.json"),summary)
            write_json(EVIDENCE/"performance-runs.json",records)
    result={}
    for enabled,label in [(False,"off"),(True,"on")]:
        selected=[x for x in records if x["enabled"]==enabled]
        seconds=[x["measurement"]["timing"]["wall_seconds"] for x in selected]
        result[label]={"seconds":seconds,"median_seconds":statistics.median(seconds),"min_seconds":min(seconds),"max_seconds":max(seconds),
            "peak_working_set_bytes":[x["peak_working_set_bytes"] for x in selected],
            "peak_private_bytes":[x["peak_private_bytes"] for x in selected],
            "peak_gpu_used_mib":[x["peak_gpu_used_mib"] for x in selected]}
    result["median_increase"]=result["on"]["median_seconds"]/result["off"]["median_seconds"]-1
    for key in ["peak_working_set_bytes","peak_private_bytes","peak_gpu_used_mib"]:
        result[key+"_median_delta"]=statistics.median(result["on"][key])-statistics.median(result["off"][key])
    write_json(EVIDENCE/"performance-summary.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["cpu","smoke","performance"])
    parser.add_argument("--idle-runner-pid",type=int,action="append",default=[],help="PID explicitly confirmed idle by the user")
    parser.add_argument("--release-executable",type=Path,default=RELEASE_EXECUTABLE)
    args=parser.parse_args()
    IDLE_RUNNER_PIDS=set(args.idle_runner_pid)
    RELEASE_EXECUTABLE=args.release_executable.resolve()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    {"cpu":cpu,"smoke":smoke,"performance":performance}[args.stage]()
