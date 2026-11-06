"""CHANGELOGの対応表を使い、固定された過去入力だけを新契約へ移す検証用変換。"""
from pathlib import Path
import re
from types import MappingProxyType


def migration_rules():
    # 廃止名を現用testdataへ重複保持せず、人向けの対応表を正本にする。
    changelog = Path(__file__).resolve().parents[4] / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8-sig")
    section = text.split("### PRD101 ProfiledValue のキー対応", 1)[1].split("\n##", 1)[0]
    rows = re.findall(r"^\| `([^`]+)` \| `([^`]+)` \|$", section, re.MULTILINE)
    if not rows:
        raise RuntimeError("Missing PRD101 migration table in CHANGELOG.md")
    profiles, fields = {}, {}
    for old, new in rows:
        root, _, field = new.rpartition(".")
        if field in ("start", "end", "steps"):
            profiles[old] = (root, field)
        else:
            fields[old] = new
    return profiles, fields


names, other = (MappingProxyType(rules) for rules in migration_rules())

def key_format(key):
    match = re.match(r'^(.*?\.@[\w-]+)\.(.+)$', key)
    return match[1] + ' : ' + match[2] if match else key


def migrate(text, learning=True):
    if learning:
        for old, new in other.items():
            text = text.replace(old, new)
    rows = text.splitlines(True)
    groups = {}
    for line in rows:
        match = re.match(r'\s*([^#=\n]+?)\s*(\?=|=)\s*([^#\n]*)', line)
        if not match:
            continue
        key = re.sub(r'\s*:\s*', '.', match[1].strip())
        for old, (profile, field) in names.items():
            if not learning and profile == 'per_beta':
                continue
            if key.endswith('.' + old):
                groups.setdefault((key[:-len(old)-1], profile), {})[field] = match[3].strip()

    def constant(prefix, profile):
        group = groups.get((prefix, profile), {})
        steps = group.get('steps')
        if profile == 'per_beta':
            return steps == '0'
        eval_names = ('[eval]', '[eval_target]', '[eval_panel]', '[greedy]', '@eval_base', '@greedy')
        implicit_constant = ((profile == 'uqe.epsilon' and '@baseline' in prefix)
                             or any(name in prefix for name in eval_names)
                             or (group.get('start') is not None and group.get('start') == group.get('end')))
        return steps == '0' or (steps is None and implicit_constant)

    out = []
    for line in rows:
        match = re.match(r'(\s*)([^#=\n]+?)\s*(\?=|=)\s*([^#\n]*)(#[^\n]*)?\n?$', line)
        if not match:
            for old, new in other.items():
                line = line.replace(old, new)
            out.append(line)
            continue
        indent, key, op, value, comment = match.groups()
        key = re.sub(r'\s*:\s*', '.', key.strip())
        value = value.strip()
        comment = (' ' + comment) if comment else ''
        found = False
        for old, (profile, field) in names.items():
            if not learning and profile == 'per_beta':
                continue
            if not key.endswith('.' + old):
                continue
            prefix = key[:-len(old)-1]
            root = prefix + '.' + profile
            found = True
            entries = []
            is_constant = constant(prefix, profile)
            if field == 'steps':
                entries.append((root + '.type', 'constant' if value == '0' else 'linear'))
                if value != '0':
                    entries.append((root + '.steps', value))
            elif field == 'start':
                entries.append((root if is_constant else root + '.start', value))
            elif not is_constant:
                entries.append((root + '.end', value))
            if profile in ('epsilon', 'uqe.epsilon', 'uqe.tau') and field in ('start', 'end') and 'target_policy' not in prefix:
                spatial_value = value
                # baselineの両端はPRD D15の値とし、実験値はenv側へ残す。
                if prefix == 'DefaultDQNAgent.@baseline.actor.[train].policy' and profile == 'epsilon':
                    spatial_value = '0.4' if field == 'start' else '0.01'
                entries.append((prefix + '.spatial.' + profile + ('.first' if field == 'start' else '.last'), spatial_value))
            for index, (new_key, new_value) in enumerate(entries):
                out.append(indent + key_format(new_key) + ' ' + op + ' ' + new_value + (comment if index == 0 else '') + '\n')
            break
        if found:
            continue
        changed = False
        for old, new in other.items():
            if not learning and new == 'learner.learning_rate':
                continue
            if key.endswith('.' + old):
                key = key[:-len(old)] + new
                changed = True
        if changed:
            out.append(indent + key_format(key) + ' ' + op + ' ' + value + comment + '\n')
        else:
            out.append(line)
    return ''.join(out)


def migrate_values(values, learning=True):
    """旧宣言から新宣言を構成する。resolverの実行結果から期待値を作らない。"""
    source = ''.join(key+' = '+str(value)+'\n' for key,value in values.items())
    result={}
    for line in migrate(source, learning=learning).splitlines():
        match=re.match(r'(.+?)\s*(?:\?=|=)\s*(.*)',line)
        if match:result[re.sub(r'\s*:\s*','.',match[1].strip())]=match[2].strip()
    return result
