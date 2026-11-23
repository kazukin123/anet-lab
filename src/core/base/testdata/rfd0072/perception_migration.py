"""固定された RFD0072 の入力だけを RFD0161 の設定に移す。"""

import re

from profiled_migration import key_format


_STACK = "stu" "cker"
_NORMALIZE = "DefaultDQNAgent.@baseline.obs_norm."
_RUNNING_FIELDS = {
    "use_centering",
    "epsilon",
    "use_robust_update",
    "robust_warmup_count",
    "robust_std_threshold",
    "post_process_type",
    "post_process_threshold",
}


def migrate_perception(text):
    output = []
    for line in text.splitlines(keepends=True):
        match = re.fullmatch(r"([^#=\n]+?)\s*(\?=|=)\s*([^#\n]*)\s*(?:#[^\n]*)?\n?", line.lstrip())
        if match is None:
            output.append(line)
            continue
        key, op, value = match.groups()
        key = re.sub(r"\s*:\s*", ".", key.strip())
        value = value.strip()
        owner, separator, field = key.rpartition(f".{_STACK}.")
        declarations = None
        if separator:
            if field == "stack_count":
                declarations = [(f"{owner}.perception.block.[stack].count", value)]
            elif field == "use_stacker" and value == "true":
                declarations = []
                if owner == "DefaultDQNAgent.@baseline":
                    declarations.extend([
                        (f"{owner}.perception.structure", "stack > norm"),
                        (f"{owner}.perception.block.[stack].type", "FrameStack"),
                    ])
            elif field == "stack_keys" and not value:
                declarations = [(f"{owner}.perception.block.[stack].keys.mode", "all")]
        elif key.startswith(_NORMALIZE):
            field = key[len(_NORMALIZE):]
            root = "DefaultDQNAgent.@baseline.perception.block.[norm]"
            if field == "pass_through" and value == "true":
                declarations = [(f"{root}.type", "Normalize"), (f"{root}.mode", "none")]
            elif field == "use_dynamic_scaling" and value == "true":
                declarations = []
            elif field in {"use_clipping", "clip_range"}:
                declarations = [(f"{root}.{mode}.{field}", value) for mode in ("constant", "running")]
            elif field in _RUNNING_FIELDS:
                declarations = [(f"{root}.running.{field}", value)]
        else:
            output.append(line)
            continue
        if declarations is None:
            raise ValueError(f"Unexpected Perception declaration in the frozen input: {key} {op} {value}")
        output.extend(f"{key_format(replacement)} {op} {setting}\n" for replacement, setting in declarations)
    return "".join(output)
