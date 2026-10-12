"""Compare all existing resolved string values, allowing only new keys and artifact locations."""
import json, sys
from pathlib import Path

def read(path):
    values={}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if ' = ' in line:
            key,value=line.split(' = ',1)
            if key in values: raise ValueError('Duplicate resolved key: '+key)
            values[key]=value
    return values

if __name__=='__main__':
    old,new=map(read,sys.argv[1:3])
    locations={'app.run_name','app.runs_dir'}
    differences={key:{'old':value,'new':new.get(key)} for key,value in old.items()
                 if key not in locations and new.get(key)!=value}
    result={'old_keys':len(old),'new_keys':len(new),'location_keys':sorted(locations),
            'new_only_keys':sorted(new.keys()-old.keys()),'differences':differences,'match':not differences}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(bool(differences))
