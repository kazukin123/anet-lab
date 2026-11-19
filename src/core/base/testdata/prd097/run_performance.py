"""PRD097 alternating performance acceptance for five fixed configurations."""
import argparse
import json
import statistics
import re
from run_validation import ROOT, run, write_json, target_for
from compare_metrics import read_run

CASES=('gridmaze','lunarlander','dropmerge','dropmerge_serial','atari1')


def measure(case):
    target=target_for('after');target.mkdir(parents=True,exist_ok=True)
    plan=target/f'{case}.performance-plan.json'
    if plan.exists(): repeats=json.loads(plan.read_text(encoding='utf-8'))['repeats']
    else:
        used=[int(match.group(1)) for path in (ROOT/'before').glob(case+'_*.execution.json')
            if (match:=re.fullmatch(re.escape(case)+r'_(\d+)\.execution\.json',path.name))]
        first=max([200]+used)+1
        repeats=list(range(first,first+5));write_json(plan,{'repeats':repeats})
    results=[]
    for index,repeat in enumerate(repeats):
        # 同じ構成のbefore/afterを1組ずつ実行し、過去世代のRunを混ぜない。
        for side in ('before','after'):
            execution=run(side,case,repeat)
            _,metrics=read_run(execution['run'])
            if metrics['performance_median'] is None: raise RuntimeError('No performance samples: '+execution['run'])
            results.append({'side':side,'repeat':repeat,'execution':execution,
                'performance_median':metrics['performance_median'],
                'last_reported_step':metrics['performance_last_step']})
        if index>=2:
            values=[item['performance_median'] for item in results if item['side']=='before']
            spread=max(values)/min(values)-1
            if spread<=0.02: break
    before=statistics.median(item['performance_median'] for item in results if item['side']=='before')
    after=statistics.median(item['performance_median'] for item in results if item['side']=='after')
    status='inconclusive' if spread>0.02 else ('pass' if after/before>=0.98 else 'fail')
    result={'case':case,'runs':results,'before_median':before,'after_median':after,
        'before_spread':spread,'ratio':after/before,'status':status}
    write_json(target/f'{case}.performance.json',result)
    print(json.dumps({key:value for key,value in result.items() if key!='runs'}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));args=parser.parse_args()
    # 短い構成から検証し、安定した不合格を検出した世代では残りを走らせない。
    # 修正後は新しい世代ですべての構成を改めて測る。
    results=[]
    for case in CASES:
        if case not in args.cases: continue
        result=measure(case);results.append(result)
        if result['status']=='fail': break
    write_json(target_for('after')/'performance.json',{'cases':results})
    return 1 if any(item['status']=='fail' for item in results) else 0


if __name__=='__main__': raise SystemExit(main())
