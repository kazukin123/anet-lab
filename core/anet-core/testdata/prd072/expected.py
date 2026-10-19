"""固定入力の宣言からP7の記録期待値を作る（resolver実行結果は読まない）。"""

import json
from pathlib import Path
import re
import shlex

from migrate import normalize


def read_config(path, values, defaults):
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("$include"):
            name = re.fullmatch(r"\$include\s+<(.+)>", line)[1]
            included = path.parent / name
            read_config(included, values, defaults)
        elif "=" in line:
            key, value = line.split("=", 1)
            weak = key.rstrip().endswith("?")
            key = normalize(key.rstrip()[:-1] if weak else key)
            if not weak or key not in values or key in defaults:
                values[key] = value.strip()
                if weak:
                    defaults.add(key)
                else:
                    defaults.discard(key)


def material(key):
    return any(part.startswith("@") for part in key.split("."))


def chain(key, value):
    owner = key[:-2]
    if owner.split(".")[-1].startswith("@"):
        owner = owner.rpartition(".")[0]
    return [{"term": term, "resolved": owner + "." + term if term.startswith("@") and "." not in term else term}
            for term in (s.strip() for s in value.split(">")) if term]


def expected_resolution(config, item, baseline):
    values, defaults = {}, set()
    read_config(config / "_main.txt", values, defaults)
    read_config(config / (item["env"] + ".txt"), values, defaults)
    cli = dict(token.split("=", 1) for token in shlex.split(item["cli"]))
    values.update(cli)
    defaults.difference_update(cli)
    before_run = dict(values)
    before_defaults = set(defaults)
    run_leaves = {}
    entries = []
    if "run.$" in values:
        run = chain("run.$", values["run.$"])
        entries.append({"key": "run.$", "chain": run})
        for term in run:
            prefix = term["resolved"] + "."
            supplied = {key[len(prefix):]: value for key, value in list(values.items()) if key.startswith(prefix)}
            values.update(supplied)
            defaults.difference_update(supplied)
            for key, value in supplied.items():
                if not material(key) and not key.endswith(".$"):
                    run_leaves[key] = (value, term["resolved"])
        values.update(cli)
    declarations = {key: chain(key, value) for key, value in values.items() if key.endswith(".$") and key != "run.$"}
    seen = set()

    def visit(key):
        if key in seen:
            return
        seen.add(key)
        terms = declarations[key]
        entries.append({"key": key, "chain": terms})
        for term in terms:
            prefix = term["resolved"] + "."
            for child in declarations:
                if child.startswith(prefix) and not material(child[len(prefix):]):
                    visit(child)

    # 固定17入力の参照は定義済みのprefixを使う。複雑な生成在庫はM13/A09で別に検証する。
    for key in declarations:
        if not material(key):
            visit(key)
    # 固定入力の宣言から当該葉を読む。実行結果を期待値の根拠にしない。
    def leaf(key, inputs, weak, path=()):
        if key in path:
            raise RuntimeError(f"Expected value dependency cycle: {path + (key,)}")
        if key in inputs and key not in weak:
            return inputs[key]
        candidates = [(decl[:-2], terms) for decl, terms in declarations.items()
                      if key.startswith(decl[:-2] + ".") and not material(key[len(decl[:-2]) + 1:])
                      and not any(key.startswith(term["resolved"] + ".") for term in terms)]
        for owner, terms in sorted(candidates, key=lambda pair: pair[0].count("."), reverse=True):
            for term in reversed(terms):
                source_key = term["resolved"] + key[len(owner):]
                found = leaf(source_key, inputs, weak, path + (key,))
                if found is not None:
                    return found
        return inputs.get(key)
    overrides = []
    for key, (value, by) in run_leaves.items():
        previous_input, previous_weak = dict(values), set(defaults)
        previous_input.pop(key, None)
        if key in before_run:
            previous_input[key] = before_run[key]
        if key in before_defaults:
            previous_weak.add(key)
        else:
            previous_weak.discard(key)
        previous = leaf(key, previous_input, previous_weak)
        previous = "" if previous is None else previous
        if previous != value:
            overrides.append({"key": key, "by": by, "from": previous, "to": value})
    return {"schema_version": 1, "selections": entries, "overrides": overrides,
            "references": sorted(baseline["resolution"]["references"], key=lambda entry: (entry["source"], entry["target"]))}


def verify_expectations(repo, root, manifest):
    for item in manifest["inputs"]:
        directory = Path(__file__).parent
        baseline = json.loads((directory / "baseline" / (item["id"] + ".json")).read_text(encoding="utf-8"))
        expected = expected_resolution(root / "migrated/apps/runner/config", item, baseline)
        stored = json.loads((directory / "resolution" / (item["id"] + ".json")).read_text(encoding="utf-8"))
        if expected != stored:
            raise RuntimeError(f"Resolution expectation differs from declarations: {item['id']}")
