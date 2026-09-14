"""共通・ベース・デフォルトは既定葉、実験・選択は個別指定として揃える。"""

import re


ENV_FILES = {"Atari.txt", "DropMerge.txt", "LunarLander.txt", "CartPole.txt",
             "GridMaze.txt", "GridMaze_muzero.txt", "ImageCls.txt"}
BASE_FILES = {"nn.txt", "nn_cnx.txt", "metrics_scalar.txt", "metrics_image.txt"}
CONFIG_FILES = ENV_FILES | BASE_FILES | {"common.txt", "agent.txt", "DropMerge_optuna.txt"}


ATARI_AGENT_DEFAULTS = {
    "DefaultDQNAgent.net.branch.[value_stream].structure",
    "DefaultDQNAgent.net.branch.[adv_stream].structure",
    "DefaultDQNAgent.net.body.output.[features]",
}


def normalize(key):
    return re.sub(r"\s+", "", key).replace(":", ".")


def assignment_role(filename, key):
    """現用設定の用途分類。resolverの名前規則ではなく、設定編集・監査だけに使う。"""
    if filename not in CONFIG_FILES:
        return None
    if key.endswith(".$") or key == "$":
        return "selection declaration"
    if key.startswith("run.@"):
        return "explicit Run assignment"
    if filename == "DropMerge_optuna.txt":
        if key in ("DropMergeEnv.seed_mode", "DropMergeEnv.global_seed"):
            return "intentional Optuna seed override"
        return None
    if filename in ENV_FILES:
        if re.match(r"[A-Z][A-Za-z0-9]*Env\.", key) and "@" not in key:
            return "environment default definition"
        if filename == "Atari.txt" and key in ATARI_AGENT_DEFAULTS:
            return "environment agent default definition"
        return "explicit experiment assignment"
    if filename in BASE_FILES:
        return "shared base definition"
    # 名前を持つ選択肢のうち、ベースとして提供する定義だけを既定葉にする。
    if re.search(r"\.(?:@?baseline|@eval_base)\.", key):
        return "base profile definition"
    if key.startswith("net."):
        return "reusable network definition"
    if filename == "common.txt":
        return "explicit choice profile" if "@" in key else "shared default definition"
    if filename == "agent.txt":
        return "explicit choice profile"
    return "explicit experiment assignment"


def expected_operator(role):
    return "?=" if role and ("base definition" in role or "default definition" in role
                               or role in ("base profile definition", "reusable network definition")) else "="


def migrate_text(filename, text):
    """演算子だけを整え、右辺・既存コメント・重複順・BOM・改行は保持する。"""
    result, changed = [], []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        active = line.split("#", 1)[0]
        match = re.match(r"^(\s*[^=]+?)(\?=|=)", active)
        if match:
            key = normalize(match[1]).lstrip("\ufeff")
            role = assignment_role(filename, key)
            operator = expected_operator(role)
            if role is not None and match[2] != operator:
                line = line[:match.start(2)] + operator + line[match.end(2):]
                changed.append(dict(line=number, key=key, before=match[2], after=operator, reason=role))
        result.append(line)
    return "".join(result), changed
