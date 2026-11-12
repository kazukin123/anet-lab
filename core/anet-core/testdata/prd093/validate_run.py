"""正常完了したPRD093 Runの配線、503/1006の測定点と同一区間時間を検証する。"""
import gzip
import json
import math
from pathlib import Path

PREFIX="48_agent_attention/"
PROGRESS="90_perf/98_prd093_learn_step"
CLOCK="90_perf/99_prd093_elapse_hour"
EXPECTED={f"{PREFIX}{i+1:02d}_entropy_layer{i}":f"attention_entropy.[{i}]" for i in range(4)}

def analyze(run, enabled=True, atari=False):
    run=Path(run)
    definitions=json.loads((run/"json/metrics.scalar.defs.json").read_text(encoding="utf-8-sig"))["data"]
    attention={tag:value for tag,value in definitions.items() if tag.startswith(PREFIX)}
    expected=EXPECTED if enabled else {}
    if {tag:value["source_key"] for tag,value in attention.items()}!=expected:
        raise ValueError("Attention definitions differ from the four-layer contract")
    for definition in attention.values():
        for key,value in {"event":"learn","step_axis":"learn_step","target":"update_result","scope":"train","interval":503}.items():
            if definition.get(key)!=value:
                raise ValueError("Invalid attention definition field: "+key)
    if atari:
        keys={value["source_key"] for value in definitions.values()}
        if not any(key.startswith("munchausen_") for key in keys) or not any(key.startswith("replay_fit_") for key in keys):
            raise ValueError("Atari default diagnostics were lost")
    logfile=run/(run.name+".log")
    if "Draining background observers" not in logfile.read_text(encoding="utf-8-sig"):
        raise ValueError("Normal shutdown log is missing")
    source=run/"metrics.jsonl"; opener=open
    if not source.exists(): source=run/"metrics.jsonl.gz"; opener=gzip.open
    points={tag:{} for tag in expected}; clocks={}; updates=[]
    with opener(source,"rt",encoding="utf-8-sig") as stream:
        for line in stream:
            row=json.loads(line)
            if row.get("type")!="scalar": continue
            tag=row["tag"]; step=int(row["step"])
            if tag==PROGRESS: updates.append(step)
            elif tag==CLOCK: clocks[step]=float(row["value"])*3600
            elif tag in points:
                value=row.get("value")
                if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:
                    raise ValueError("Attention value outside [0,1]: "+tag)
                if step in points[tag]: raise ValueError("Duplicate attention measurement point")
                points[tag][step]=value
    if not updates or max(updates)<1006 or 503 not in clocks or 1006 not in clocks:
        raise ValueError("Required learning interval 503..1006 was not reached")
    for tag,values in points.items():
        if not {503,1006}<=values.keys(): raise ValueError("Missing required attention measurement: "+tag)
    return {"run":str(run),"updates":len(updates),"max_learn_step":max(updates),
        "wall_seconds_503_1006":clocks[1006]-clocks[503],"points":points}
