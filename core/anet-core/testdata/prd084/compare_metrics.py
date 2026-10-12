"""Normalize numeric baseline metrics and compare every retained point."""
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

EXCLUDED_PREFIXES = ('90_perf/', '91_resource/', '92_resource/', '47_agent_noisy/')

def normalize(run_text, output_text, include_noisy=False):
    run = Path(run_text)
    source = run / 'metrics.jsonl'
    opener = open
    if not source.is_file():
        source = run / 'metrics.jsonl.gz'
        opener = gzip.open
    prefixes = EXCLUDED_PREFIXES[:-1] if include_noisy else EXCLUDED_PREFIXES
    records = []
    excluded = Counter()
    with opener(source, 'rt', encoding='utf-8-sig') as stream:
        for line in stream:
            obj = json.loads(line)
            if obj.get('type') != 'scalar':
                continue
            tag = obj['tag']
            if tag.startswith(prefixes):
                excluded[tag] += 1
                continue
            value = obj['value']
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f'Non-numeric scalar: {tag}')
            records.append((tag, int(obj['step']), 'NaN' if value is None else float(value).hex()))
    records.sort()
    output = Path(output_text)
    canonical = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in records)
    output.write_text(canonical, encoding='utf-8', newline='\n')
    result = {'run_dir': str(run), 'source': str(source), 'point_count': len(records),
              'sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
              'counts': dict(sorted(Counter(row[0] for row in records).items())),
              'excluded_counts': dict(sorted(excluded.items()))}
    output.with_suffix('.summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'point_count': result['point_count'], 'sha256': result['sha256'],
                      'tag_count': len(result['counts']), 'excluded_tags': list(excluded)}))

def compare(old_text, new_text):
    old = Path(old_text).read_text(encoding='utf-8').splitlines()
    new = Path(new_text).read_text(encoding='utf-8').splitlines()
    old_counts = Counter(json.loads(line)[0] for line in old)
    new_counts = Counter(json.loads(line)[0] for line in new)
    delta = {tag: {'old': old_counts[tag], 'new': new_counts[tag]}
             for tag in sorted(old_counts.keys() | new_counts.keys()) if old_counts[tag] != new_counts[tag]}
    result = {'old_points': len(old), 'new_points': len(new), 'count_mismatches': delta,
              'exact_match': old == new, 'first_differences': []}
    for index, (left, right) in enumerate(zip(old, new)):
        if left != right:
            result['first_differences'].append({'index': index, 'old': json.loads(left), 'new': json.loads(right)})
            if len(result['first_differences']) == 10:
                break
    print(json.dumps(result))
    raise SystemExit(0 if old == new else 1)

if __name__ == '__main__':
    if len(sys.argv) not in (4, 5) or (len(sys.argv) == 5 and sys.argv[1] != 'normalize'):
        raise SystemExit('Usage: compare_metrics.py normalize RUN OUTPUT [--include-noisy] | compare OLD NEW')
    if sys.argv[1] == 'normalize':
        if len(sys.argv) == 5 and sys.argv[4] != '--include-noisy':
            raise SystemExit('Expected --include-noisy')
        normalize(*sys.argv[2:4], include_noisy=len(sys.argv) == 5)
    elif sys.argv[1] == 'compare':
        compare(*sys.argv[2:])
    else:
        raise SystemExit('Expected normalize or compare')
