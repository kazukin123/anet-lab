"""PRD092の正常完了Runで、購読ごとの未成立率と同一学習区間の実時間を集計する。"""
import argparse
import gzip
import json
import math
from pathlib import Path

PREFIX="49_agent_gradnoise/"
PROGRESS="90_perf/98_prd092_learn_step"
CLOCK="90_perf/99_prd092_elapse_hour"
RAW_TAGS={"31_signal_adam_raw","32_variance_adam_raw","33_signal_raw","34_variance_raw",
          "35_signal_adam_feature_raw","36_variance_adam_feature_raw",
          "37_signal_adam_readout_raw","38_variance_adam_readout_raw"}

EXPECTED_KEYS={
    "01_scale_adam":"grad_noise_scale_adam", "02_scale_adam_feature":"grad_noise_scale_adam_feature",
    "03_scale_adam_readout":"grad_noise_scale_adam_readout", "05_ratio_adam":"grad_noise_ratio_adam",
    "11_scale":"grad_noise_scale", "21_signal_adam":"grad_noise_signal_adam", "22_variance_adam":"grad_noise_variance_adam",
    "31_signal_adam_raw":"grad_noise_signal_adam_raw", "32_variance_adam_raw":"grad_noise_variance_adam_raw",
    "33_signal_raw":"grad_noise_signal_raw", "34_variance_raw":"grad_noise_variance_raw",
    "35_signal_adam_feature_raw":"grad_noise_signal_adam_feature_raw",
    "36_variance_adam_feature_raw":"grad_noise_variance_adam_feature_raw",
    "37_signal_adam_readout_raw":"grad_noise_signal_adam_readout_raw",
    "38_variance_adam_readout_raw":"grad_noise_variance_adam_readout_raw"}

def scheduled_points(steps,interval):
    if interval<1:
        raise ValueError("interval must be positive")
    result=[]
    previous=None
    for step in steps:
        bucket=step//interval
        if previous is None or bucket>previous:
            result.append(step)
            previous=bucket
    return result

def analyze(run,exit_code,empty_groups=(),require_all=False,start_step=None,end_step=None):
    if exit_code!=0:
        raise ValueError("Only a normally completed run (exit_code=0) can be analyzed")
    definitions=json.loads((run/"json/metrics.scalar.defs.json").read_text(encoding="utf-8-sig"))["data"]
    definitions={tag:definition for tag,definition in definitions.items() if tag.startswith(PREFIX)}
    if require_all:
        actual={tag.removeprefix(PREFIX):definition["source_key"] for tag,definition in definitions.items()}
        if actual!=EXPECTED_KEYS:
            raise ValueError("Gradient-noise tag/source-key wiring differs from the fifteen-key contract")
        logfile=run/(run.name+".log")
        if not logfile.is_file() or "Draining background observers" not in logfile.read_text(encoding="utf-8-sig"):
            raise ValueError("Missing normal shutdown log")
    source=run/"metrics.jsonl"
    opener=open
    if not source.is_file():
        source=run/"metrics.jsonl.gz"
        opener=gzip.open
    steps=[]
    values={tag:{} for tag in definitions}
    clocks={}
    with opener(source,"rt",encoding="utf-8-sig") as stream:
        for line in stream:
            row=json.loads(line)
            if row.get("type")!="scalar":
                continue
            tag=row["tag"]
            if tag==PROGRESS:
                steps.append(int(row["step"]))
            elif tag==CLOCK:
                clocks[int(row["step"])]=float(row["value"])*3600
            elif tag in values and isinstance(row.get("value"),(int,float)) and math.isfinite(row["value"]):
                values[tag][int(row["step"])]=row["value"]
    steps=sorted(set(steps))
    if not steps:
        raise ValueError("Missing per-update progress records; cannot infer scheduled points")
    report={"run":str(run),"exit_code":exit_code,"updates":len(steps),"first_step":steps[0],"last_step":steps[-1],"metrics":{}}
    for tag,definition in definitions.items():
        if (definition["event"],definition["step_axis"],definition["target"])!=("learn","learn_step","update_result"):
            raise ValueError(f"Unexpected subscription: {tag}: {definition}")
        if require_all and definition["interval"]!=503:
            raise ValueError(f"Unexpected interval: {tag}")
        empty=any(f"_{group}" in definition["source_key"] for group in empty_groups)
        planned=[] if empty else scheduled_points(steps,definition["interval"])
        recorded=sum(step in values[tag] for step in planned)
        report["metrics"][tag]={"key":definition["source_key"],"interval":definition["interval"],
            "planned":len(planned),"finite":recorded,
            "unavailable_rate":(len(planned)-recorded)/len(planned) if planned else None}
        if require_all and tag.removeprefix(PREFIX) in RAW_TAGS and not empty and recorded==0:
            raise ValueError(f"No finite raw measurement: {tag}")
    if start_step is not None or end_step is not None:
        if start_step not in clocks or end_step not in clocks or end_step<=start_step:
            raise ValueError("Missing fixed measurement clock boundaries")
        report["timing"]={"start_step":start_step,"end_step":end_step,"learn_steps":end_step-start_step,
            "wall_seconds":clocks[end_step]-clocks[start_step]}
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--exit-code",type=int,required=True)
    parser.add_argument("--empty-group",choices=["feature","readout"],action="append",default=[])
    parser.add_argument("--require-all",action="store_true")
    parser.add_argument("--start-step",type=int)
    parser.add_argument("--end-step",type=int)
    args=parser.parse_args()
    print(json.dumps(analyze(args.run,args.exit_code,args.empty_group,args.require_all,args.start_step,args.end_step),indent=2))
