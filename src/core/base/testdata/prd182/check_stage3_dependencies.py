"""Check the direct library dependencies and test include paths required by PRD182."""

import argparse
import json
from pathlib import Path
import re

BASE = "anet-core-base"
TESTING = "anet-core-testing"
RUNNER = "AnetRLRunner"
RUNNER_TEST = "AnetRLRunner-gui-test"
NODE = re.compile(r'^\s*"([^"]+)"\s+\[\s*label\s*=\s*"([^"\\]+)')
EDGE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')
OBJECT_TARGET = re.compile(r"CMakeFiles[/\\]([^/\\]+)\.dir[/\\]")
INCLUDES = re.compile(r'(?:^|\s)(?:/I|-I|-isystem\s+)(?:"([^"]+)"|([^\s]+))')


def is_component(name):
    return name.startswith(("anet-core-", "anet-algo-", "anet-env-")) or name in {RUNNER, RUNNER_TEST}


def allowed_dependency(owner, dependency):
    if owner == BASE:
        return False
    if owner == TESTING:
        return dependency == BASE
    if owner == RUNNER:
        return dependency == BASE or dependency.startswith(("anet-algo-", "anet-env-")) and not dependency.endswith("-test")
    if owner == RUNNER_TEST:
        return dependency in {TESTING, RUNNER} or allowed_dependency(RUNNER, dependency)
    if owner.endswith("-test"):
        return dependency in {BASE, TESTING, owner.removesuffix("-test")}
    if owner.startswith(("anet-algo-", "anet-env-")):
        return dependency == BASE
    return False


def graph_violations(text):
    labels = {}
    edges = []
    for line in text.splitlines():
        node = NODE.match(line)
        edge = EDGE.match(line)
        if node:
            labels[node[1]] = node[2]
        elif edge:
            edges.append((edge[1], edge[2]))
    families = ("anet-algo-dqn", "anet-algo-muzero", "anet-algo-imagecls")
    required = {BASE, BASE + "-test", TESTING, RUNNER, RUNNER_TEST, *families, *(name + "-test" for name in families)}
    results = [f"Missing build target: {name}" for name in sorted(required - set(labels.values()))]
    for source, target in edges:
        # CMake attributes a target's own post-build file expressions to the target.
        if source == target:
            continue
        owner, dependency = labels.get(source, ""), labels.get(target, "")
        if is_component(owner) and is_component(dependency) and not allowed_dependency(owner, dependency):
            results.append(f"Forbidden dependency: {owner} -> {dependency}")
    return results


def include_violations(commands, private_paths):
    results = set()
    permitted = {BASE, BASE + "-test", TESTING}
    for row in commands:
        command = row["command"]
        match = OBJECT_TARGET.search(command)
        if not match:
            continue
        owner = match[1]
        if not is_component(owner) or owner in permitted:
            continue
        for path in INCLUDES.finditer(command):
            include = (Path(row["directory"]) / (path[1] or path[2])).resolve()
            if any(include.is_relative_to(private_path) for private_path in private_paths):
                results.add(f"Private base include path: {owner} -> {include}")
    return sorted(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphviz", type=Path)
    parser.add_argument("compile_commands", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[5]
    violations = graph_violations(args.graphviz.read_text(encoding="utf-8"))
    private_paths = tuple((root / "src/core/base" / directory).resolve() for directory in ("src", "src_testing"))
    commands = json.loads(args.compile_commands.read_text(encoding="utf-8"))
    violations.extend(include_violations(commands, private_paths))
    for violation in violations:
        print(violation)
    print(f"Dependency or include path violations: {len(violations)}")
    raise SystemExit(bool(violations))


if __name__ == "__main__":
    main()
