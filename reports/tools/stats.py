#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""anet-stats: anet-lab の定点統計を git 履歴とローカル Run フォルダから再生成する。

使い方(リポジトリルートで実行):
  python reports/tools/stats.py                       # 全部生成
  python reports/tools/stats.py --weeks 16            # 表に出す週数
  python reports/tools/stats.py --no-cloc --no-blame  # 速い縮退モード(規模の comment 分離と行年代を省く)
  python reports/tools/stats.py --no-runs             # ローカル Run フォルダを見ない

出力(既定 reports/stats/):
  stats.json      全期間の集計。毎回丸ごと再生成する
  trend.md        人が読む表。毎回再生成
  trend.html      1 ファイル完結のチャート付きビュー。毎回再生成
  summary.md      digest を書く LLM 向けの短い要約と insight flag。毎回再生成
  runs_seen.jsonl Run 台帳。唯一の蓄積ファイル(Run フォルダは消えるため)
  cache/          blame と cloc のキャッシュ。消しても再生成できる(gitignore 対象)

設計:
  正本は git 履歴とローカル Run フォルダ。分類は reports/tools/categories.json。
  コミット数は見出しにしない。稼働日・行の流量・Run 起動数を主軸にする。
  生成物(実験 config ダンプ、archify HTML、testdata 等)は除外し、1 コミット 1 ファイル 2000 行超は bulk として別枠。
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import hashlib
import html
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

CODE_EXT = (".cpp", ".hpp", ".h", ".cu", ".cuh", ".py", ".java", ".cmake", ".ts", ".tsx")
CPP_EXT = (".cpp", ".hpp", ".h", ".cu", ".cuh")
TYPE_RE = re.compile(r"^﻿?([A-Za-z]+)(\([^)]*\))?[:：]")
TOPIC_RE = re.compile(r"#(\d+)")
IDENT_RE = re.compile(r"\b(?:class|struct)\s+([A-Z][A-Za-z0-9_]{5,})\b")
IDENT_STOP = {"Config", "Result", "Options", "Params", "Impl", "Entry", "Record", "Status", "Context", "Handler"}


# ----------------------------------------------------------------------------- helpers
def sh(args, cwd, check=False):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(args)}\n{r.stderr}")
    return r.stdout


def git(repo, *args, check=False):
    return sh(["git", *args], repo, check=check)


def week_label(d: dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def week_start(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def week_name(start: dt.date) -> str:
    """人が読む週表記。ISO 週番号は日本では通じないので週開始日(月曜)で示す。"""
    return f"{start.isoformat()}週"


def iter_weeks(first: dt.date, last: dt.date):
    cur = week_start(first)
    end = week_start(last)
    while cur <= end:
        yield week_label(cur), cur
        cur += dt.timedelta(days=7)


def read_text(repo, path):
    try:
        with open(os.path.join(repo, path), encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(n):
    return f"{n:,}" if isinstance(n, int) else (f"{n:.2f}" if isinstance(n, float) else str(n))


# ----------------------------------------------------------------------------- config
class Rules:
    def __init__(self, repo):
        with open(os.path.join(repo, "reports", "tools", "categories.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.excluded = [re.compile(p) for p in self.cfg["excluded"]]
        self.areas = [(n, re.compile(p)) for n, p in self.cfg["areas"]]
        self.cats = []
        for c in self.cfg["categories"]:
            self.cats.append((c["key"], c.get("label", c["key"]), c.get("topic"), c.get("test_target", True),
                              [re.compile(p) for p in c["match"]]))
        self.cat_label = {k: l for k, l, *_ in self.cats}
        self.cat_test_target = {k: t for k, _, _, t, _ in self.cats}
        self.test_infra = set(self.cfg.get("test_infra", []))
        self.boundary = dt.date.fromisoformat(self.cfg["boundary"])
        self.bulk = int(self.cfg.get("bulk_lines_per_file", 2000))
        self.large = int(self.cfg.get("large_file_lines", 800))
        self.done_convention = dt.date.fromisoformat(self.cfg.get("done_convention_date", "2026-08-21"))
        self.topics = {int(k): v for k, v in self.cfg.get("topics", {}).items()}
        self.run_globs = self.cfg.get("run_folder_globs", [])

    def is_excluded(self, path):
        return any(r.search(path) for r in self.excluded)

    def area_of(self, path):
        for n, r in self.areas:
            if r.search(path):
                return n
        return "other"

    def category_of(self, path):
        base = os.path.basename(path)
        if base in self.test_infra:
            return "test_infra"
        q = re.sub(r"_test\.cpp$", ".cpp", path)
        for k, _, _, _, rs in self.cats:
            if any(r.search(q) for r in rs):
                return k
        return "uncategorized"


# ----------------------------------------------------------------------------- flow (git log)
def collect_commits(repo):
    raw = git(repo, "log", "--numstat", "--date=short", "--format=%x00%H%x1f%ad%x1f%s")
    commits = []
    for block in raw.split("\x00")[1:]:
        head, _, body = block.partition("\n")
        parts = head.split("\x1f")
        if len(parts) != 3:
            continue
        sha, date, subj = parts
        m = TYPE_RE.match(subj)
        files = []
        for line in body.splitlines():
            p = line.split("\t")
            if len(p) != 3 or p[0] == "-":
                continue
            files.append((int(p[0]), int(p[1]), p[2]))
        commits.append({"sha": sha, "date": dt.date.fromisoformat(date), "subject": subj.lstrip("﻿"),
                        "type": (m.group(1).lower() if m else "other"),
                        "topics": [int(t) for t in TOPIC_RE.findall(subj)], "files": files})
    commits.sort(key=lambda c: c["date"])
    return commits


CODE_AREAS = {"core", "tests", "envs", "config", "runner", "viewer", "build"}
DOC_AREAS = {"docs", "memo"}


def aggregate(commits, rules: Rules):
    A = {"days": set(), "commits": 0, "add": collections.Counter(), "del": collections.Counter(), "bulk": 0,
         "bulk_files": collections.Counter(), "type_lines": collections.Counter(), "topic_lines": collections.Counter(),
         "code_commits": 0, "code_with_docs": 0, "sizes": [], "files_touched": set()}
    for c in commits:
        A["days"].add(c["date"])
        A["commits"] += 1
        touched = set()
        total = 0
        for add, dele, path in c["files"]:
            if rules.is_excluded(path) or add > rules.bulk or dele > rules.bulk:
                A["bulk"] += add + dele
                A["bulk_files"][path] += add + dele
                continue
            ar = rules.area_of(path)
            touched.add(ar)
            A["files_touched"].add(path)
            A["add"][ar] += add
            A["del"][ar] += dele
            total += add + dele
            A["type_lines"][c["type"]] += add + dele
            for t in c["topics"]:
                A["topic_lines"][t] += add + dele
        A["sizes"].append(total)
        if touched & CODE_AREAS:
            A["code_commits"] += 1
            if touched & DOC_AREAS:
                A["code_with_docs"] += 1
    return A


def summarize_agg(A, rules: Rules):
    code_add = sum(A["add"][a] for a in CODE_AREAS)
    code_del = sum(A["del"][a] for a in CODE_AREAS)
    docs_add = sum(A["add"][a] for a in DOC_AREAS)
    docs_del = sum(A["del"][a] for a in DOC_AREAS)
    top_type = A["type_lines"].most_common(1)
    top_topic = A["topic_lines"].most_common(1)
    return {
        "active_days": len(A["days"]), "commits": A["commits"],
        "code_add": code_add, "code_del": code_del, "docs_add": docs_add, "docs_del": docs_del,
        "test_add": A["add"]["tests"], "test_del": A["del"]["tests"],
        "config_add": A["add"]["config"], "config_del": A["del"]["config"],
        "add_by_area": dict(A["add"]), "del_by_area": dict(A["del"]),
        "bulk": A["bulk"], "bulk_top": A["bulk_files"].most_common(5),
        "type_lines": dict(A["type_lines"]), "topic_lines": {str(k): v for k, v in A["topic_lines"].items()},
        "top_type": top_type[0][0] if top_type else None,
        "top_topic": (top_topic[0][0], rules.topics.get(top_topic[0][0], "")) if top_topic else None,
        "code_commits": A["code_commits"], "code_with_docs": A["code_with_docs"],
        "median_lines_per_commit": int(statistics.median(A["sizes"])) if A["sizes"] else 0,
        "files_touched": len(A["files_touched"]),
    }


def collect_flow(commits, rules: Rules, today: dt.date):
    if not commits:
        return {"weeks": [], "churn90": {}}
    by_week = collections.defaultdict(list)
    for c in commits:
        by_week[week_label(c["date"])].append(c)
    weeks = []
    for label, start in iter_weeks(commits[0]["date"], today):
        s = summarize_agg(aggregate(by_week.get(label, []), rules), rules)
        s["week"] = label
        s["week_name"] = week_name(start)
        s["start"] = start.isoformat()
        s["partial"] = start + dt.timedelta(days=6) >= today
        weeks.append(s)
    churn = collections.Counter()
    since = today - dt.timedelta(days=90)
    for c in commits:
        if c["date"] >= since:
            for _, _, p in c["files"]:
                churn[p] += 1
    return {"weeks": weeks, "churn90": dict(churn)}


# ----------------------------------------------------------------------------- size
def tracked_files(repo):
    return [p for p in git(repo, "ls-files").split("\n") if p]


def collect_size(repo, rules: Rules, files, use_cloc):
    per_file = {}
    cloc_used = False
    if use_cloc and shutil.which("cloc"):
        raw = sh(["cloc", "--vcs=git", "--by-file", "--json", "--quiet", "core", "apps", "viewers", "docs", ".github"], repo)
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            data = {}
        for path, v in data.items():
            if path in ("header", "SUM"):
                continue
            p = path.replace("\\", "/")
            p = p[2:] if p.startswith("./") else p
            per_file[p] = {"code": v.get("code", 0), "comment": v.get("comment", 0), "blank": v.get("blank", 0), "language": v.get("language", "")}
        cloc_used = bool(per_file)
    if not cloc_used:
        for p in files:
            if p.endswith(CODE_EXT):
                per_file[p] = {"code": count_lines(read_text(repo, p)), "comment": 0, "blank": 0, "language": os.path.splitext(p)[1]}
    by_area = collections.defaultdict(lambda: {"files": 0, "code": 0, "comment": 0})
    by_lang = collections.defaultdict(lambda: {"files": 0, "code": 0, "comment": 0})
    for p, v in per_file.items():
        if rules.is_excluded(p) or not p.endswith(CODE_EXT + (".md",)):
            continue
        a = rules.area_of(p)
        by_area[a]["files"] += 1
        by_area[a]["code"] += v["code"]
        by_area[a]["comment"] += v["comment"]
        by_lang[v["language"]]["files"] += 1
        by_lang[v["language"]]["code"] += v["code"]
        by_lang[v["language"]]["comment"] += v["comment"]
    # config: 行数とキー数(コメント/空行を除く)
    cfg_lines = cfg_keys = 0
    for p in files:
        if p.startswith("apps/runner/config/") and p.endswith(".txt"):
            for line in read_text(repo, p).splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                cfg_lines += 1
                if re.match(r"^[^#=]+?(\?=|=)", s):
                    cfg_keys += 1
    jp = [p for p in files if re.match(r"^docs/design/.*\.jp\.md$", p)]
    en = [p for p in files if re.match(r"^docs/design/.*\.en\.md$", p)]
    docs = {"design_jp_files": len(jp), "design_jp_lines": sum(count_lines(read_text(repo, p)) for p in jp),
            "design_en_files": len(en), "design_en_lines": sum(count_lines(read_text(repo, p)) for p in en),
            "adr_files": len([p for p in files if p.startswith("docs/adr/") and p.endswith(".md")]),
            "context_lines": count_lines(read_text(repo, "CONTEXT.md"))}
    tests = {
        "cpp_test_cases": sum(int(x.rsplit(":", 1)[1]) for x in git(repo, "grep", "-c", "-E", r"^\s*(TEST_CASE|SCENARIO|TEMPLATE_TEST_CASE)\s*\(", "--", "*_test.cpp").split("\n") if ":" in x),
        "java_test_methods": sum(int(x.rsplit(":", 1)[1]) for x in git(repo, "grep", "-c", "-E", r"@Test\b", "--", "*.java").split("\n") if ":" in x),
        "python_test_functions": sum(int(x.rsplit(":", 1)[1]) for x in git(repo, "grep", "-c", "-E", r"^\s*def test_", "--", "*.py").split("\n") if ":" in x),
    }
    return {"cloc": cloc_used, "by_area": dict(by_area), "by_language": dict(by_lang),
            "config": {"lines": cfg_lines, "keys": cfg_keys}, "docs": docs, "test_counts": tests, "per_file_code": {p: v["code"] for p, v in per_file.items()}}


# ----------------------------------------------------------------------------- tests (L0/L1 × category × era)
def blame_era(repo, path, blob, boundary: dt.date, cache_dir):
    cache = os.path.join(cache_dir, "blame", f"{blob}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    raw = git(repo, "blame", "--line-porcelain", "-w", "--", path)
    pre = post = 0
    for line in raw.split("\n"):
        if line.startswith("author-time "):
            d = dt.date.fromtimestamp(int(line[12:]))
            if d < boundary:
                pre += 1
            else:
                post += 1
    res = {"pre": pre, "post": post}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(res, f)
    return res


def collect_tests(repo, rules: Rules, files, churn90, use_blame, cache_dir):
    blobs = {}
    for line in git(repo, "ls-files", "-s").split("\n"):
        p = line.split("\t")
        if len(p) == 2:
            blobs[p[1]] = p[0].split()[1]
    cpp = [p for p in files if p.endswith(CPP_EXT) and (p.startswith("core/") or p.startswith("apps/runner/src/"))
           and not p.endswith("pch.hpp") and not rules.is_excluded(p)]
    texts = {p: read_text(repo, p) for p in cpp}
    loc = {p: count_lines(t) for p, t in texts.items()}
    test_files = [p for p in cpp if rules.area_of(p) == "tests"]
    prod_files = [p for p in cpp if p not in test_files]
    test_text = "\n".join(texts[p] for p in test_files)
    stems = {}
    for p in cpp:
        stems.setdefault(os.path.basename(p).rsplit(".", 1)[0], []).append(p)
    cat = collections.defaultdict(lambda: {"prod_files": 0, "prod_loc": 0, "test_loc": 0, "direct": 0, "any": 0,
                                           "pre_lines": 0, "post_lines": 0, "pre_files": 0, "pre_tested": 0, "post_files": 0, "post_tested": 0})
    for p in test_files:
        cat[rules.category_of(p)]["test_loc"] += loc[p]
    file_rows = []
    era_mode = "blame" if use_blame else "none"
    for p in prod_files:
        stem = os.path.basename(p).rsplit(".", 1)[0]
        direct = any(q.endswith("/" + stem + "_test.cpp") for q in test_files)
        idents = set(IDENT_RE.findall(texts[p]))
        for h in stems.get(stem, []):
            idents |= set(IDENT_RE.findall(texts[h]))
        idents -= IDENT_STOP
        indirect = bool(re.search(r"\b" + re.escape(stem) + r"\.hpp\b", test_text)) or any(re.search(r"\b" + i + r"\b", test_text) for i in idents)
        c = rules.category_of(p)
        C = cat[c]
        C["prod_files"] += 1
        C["prod_loc"] += loc[p]
        if direct:
            C["direct"] += 1
        tested = direct or indirect
        if tested:
            C["any"] += 1
        era = None
        if use_blame and p in blobs:
            e = blame_era(repo, p, blobs[p], rules.boundary, cache_dir)
            C["pre_lines"] += e["pre"]
            C["post_lines"] += e["post"]
            era = "pre" if e["pre"] >= e["post"] else "post"
            C[era + "_files"] += 1
            if tested:
                C[era + "_tested"] += 1
        file_rows.append({"path": p, "category": c, "loc": loc[p], "direct": direct, "indirect": indirect,
                          "era": era, "churn90": churn90.get(p, 0)})
    untested = [r for r in file_rows if not (r["direct"] or r["indirect"]) and rules.cat_test_target.get(r["category"], True)]
    untested.sort(key=lambda r: r["loc"] * (1 + r["churn90"]), reverse=True)
    tot = {"prod_files": 0, "prod_loc": 0, "test_loc": 0, "direct": 0, "any": 0, "pre_files": 0, "pre_tested": 0, "post_files": 0, "post_tested": 0, "pre_lines": 0, "post_lines": 0}
    for k, C in cat.items():
        if not rules.cat_test_target.get(k, True) or k == "test_infra":
            continue
        for f in tot:
            tot[f] += C[f]
    # Java / Python は L0 のみ
    def lines_of(pred):
        return sum(count_lines(read_text(repo, p)) for p in files if pred(p))
    other = {
        "java": {"prod": lines_of(lambda p: p.endswith(".java") and "/src/main/" in p),
                 "test": lines_of(lambda p: p.endswith(".java") and "/src/test/" in p)},
        "python": {"prod": lines_of(lambda p: p.endswith(".py") and not re.search(r"(_test\.py$|/test_[^/]+\.py$)", p) and (p.startswith("apps/") or p.startswith("viewers/") or p.startswith("reports/"))),
                   "test": lines_of(lambda p: p.endswith(".py") and re.search(r"(_test\.py$|/test_[^/]+\.py$)", p) is not None)},
    }
    return {"era_mode": era_mode, "boundary": rules.boundary.isoformat(), "categories": dict(cat), "total": tot,
            "untested": untested[:40], "files": file_rows, "other_languages": other}


# ----------------------------------------------------------------------------- health / process / activity
def grep_count(repo, pattern, pathspecs, rev=None):
    args = ["grep", "-c", "-E", pattern]
    if rev:
        args.append(rev)
    args += ["--", *pathspecs]
    total = 0
    for line in git(repo, *args).split("\n"):
        if ":" in line:
            try:
                total += int(line.rsplit(":", 1)[1])
            except ValueError:
                pass
    return total


def rev_before(repo, date: dt.date):
    return git(repo, "rev-list", "-1", f"--before={date.isoformat()}T23:59:59", "HEAD").strip()


def collect_health(repo, rules: Rules, size, files, today):
    todo_now = grep_count(repo, r"\b(TODO|FIXME)\b", ["*.cpp", "*.hpp", "*.py", "*.java"])
    rev4 = rev_before(repo, today - dt.timedelta(days=28))
    todo_4w = grep_count(repo, r"\b(TODO|FIXME)\b", ["*.cpp", "*.hpp", "*.py", "*.java"], rev4) if rev4 else None
    large = []
    for p, n in size["per_file_code"].items():
        if n >= rules.large and p.endswith(CODE_EXT) and not rules.is_excluded(p):
            large.append((n, p))
    large.sort(reverse=True)
    reg = {"open": 0, "by_priority": collections.Counter(), "by_lens": collections.Counter(), "by_status": collections.Counter()}
    rp = os.path.join(repo, "reports", "registry", "findings.jsonl")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reg["by_status"][r.get("status", "?")] += 1
                if r.get("status") == "open":
                    reg["open"] += 1
                    reg["by_priority"][r.get("priority", "?")] += 1
                    reg["by_lens"][r.get("lens", "?")] += 1
    reg = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in reg.items()}
    # config キー数の 4 週前比較
    keys_4w = None
    if rev4:
        keys_4w = 0
        for p in files:
            if p.startswith("apps/runner/config/") and p.endswith(".txt"):
                for line in git(repo, "show", f"{rev4}:{p}").splitlines():
                    s = line.strip()
                    if s and not s.startswith("#") and re.match(r"^[^#=]+?(\?=|=)", s):
                        keys_4w += 1
    return {"todo_now": todo_now, "todo_4w_ago": todo_4w, "large_files": large[:15], "large_threshold": rules.large,
            "registry": reg, "config_keys_4w_ago": keys_4w, "rev_4w_ago": rev4[:7] if rev4 else None}


def collect_process(repo, rules: Rules, files, flow, today):
    prds = [p for p in files if re.match(r"^docs/memo/(done/|frozen/|dropped/)?\d{3}_.*_10prd\.md$", p)]
    created_m = collections.Counter()
    state_m = {"done": collections.Counter(), "frozen": collections.Counter(), "dropped": collections.Counter()}
    lead = []
    batch_moved = 0
    rows = []
    for p in prds:
        state = "active"
        m = re.match(r"^docs/memo/(done|frozen|dropped)/", p)
        if m:
            state = m.group(1)
        cr = git(repo, "log", "--follow", "--diff-filter=A", "--format=%ad", "--date=short", "--", p).split()
        st = git(repo, "log", "--format=%ad", "--date=short", "--", p).split()
        created = dt.date.fromisoformat(cr[-1]) if cr else None
        moved = dt.date.fromisoformat(st[-1]) if st else None
        if created:
            created_m[created.strftime("%Y-%m")] += 1
        if state != "active" and moved:
            state_m[state][moved.strftime("%Y-%m")] += 1
            if state == "done" and created:
                if moved == rules.done_convention:
                    batch_moved += 1
                elif moved > rules.done_convention:
                    lead.append((moved - created).days)
        rows.append({"path": p, "state": state, "created": created.isoformat() if created else None, "moved": moved.isoformat() if moved else None})
    root = [p for p in prds if re.match(r"^docs/memo/\d{3}_", p)]
    wip = len([p for p in root if re.match(r"^docs/memo/0\d{2}_", p)])
    backlog = len([p for p in root if re.match(r"^docs/memo/9\d{2}_", p)])
    months = sorted(set(created_m) | set(state_m["done"]) | set(state_m["frozen"]) | set(state_m["dropped"]))
    monthly = [{"month": m, "created": created_m[m], "done": state_m["done"][m], "frozen": state_m["frozen"][m], "dropped": state_m["dropped"][m]} for m in months]
    status = [l for l in git(repo, "status", "--porcelain").split("\n") if l.strip()]
    diff = git(repo, "diff", "--shortstat").strip() + " " + git(repo, "diff", "--cached", "--shortstat").strip()
    ins = sum(int(x) for x in re.findall(r"(\d+) insertion", diff))
    dele = sum(int(x) for x in re.findall(r"(\d+) deletion", diff))
    return {"prd_count": len(prds), "wip": wip, "backlog_9xx": backlog, "monthly": monthly,
            "lead_time_days": {"n": len(lead), "median": statistics.median(lead) if lead else None,
                                "p25": statistics.quantiles(lead, n=4)[0] if len(lead) >= 4 else None,
                                "p75": statistics.quantiles(lead, n=4)[2] if len(lead) >= 4 else None,
                                "batch_moved_on_convention_day": batch_moved, "valid_after": rules.done_convention.isoformat()},
            "uncommitted": {"entries": len(status), "insertions": ins, "deletions": dele}, "prds": rows}


def collect_activity(repo, rules: Rules, flow, today, use_runs, out_dir):
    weeks = flow["weeks"]
    streak = 0
    for w in reversed(weeks):
        if w["partial"] and w["commits"] == 0:
            continue
        if w["commits"] > 0:
            streak += 1
        else:
            break
    ledger_path = os.path.join(out_dir, "runs_seen.jsonl")
    seen = {}
    if os.path.exists(ledger_path):
        with open(ledger_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    seen[r["run"]] = r
    new = []
    if use_runs:
        for pat in rules.run_globs:
            for d in glob.glob(os.path.join(repo, pat)):
                name = os.path.basename(d)
                if name in seen:
                    continue
                m = re.search(r"(\d{4})(\d{2})(\d{2})", name)
                if not m:
                    continue
                try:
                    started = dt.date(int(m[1]), int(m[2]), int(m[3]))
                except ValueError:
                    continue
                rel = os.path.relpath(d, repo).replace("\\", "/")
                r = {"run": name, "path": rel, "started": started.isoformat(), "first_seen": today.isoformat()}
                seen[name] = r
                new.append(r)
        if new:
            with open(ledger_path, "a", encoding="utf-8") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    runs_by_week = collections.Counter()
    for r in seen.values():
        runs_by_week[week_label(dt.date.fromisoformat(r["started"]))] += 1
    for w in weeks:
        w["runs"] = runs_by_week.get(w["week"], 0)
    return {"streak_weeks": streak, "runs_total": len(seen), "runs_new_in_ledger": len(new), "runs_by_week": dict(runs_by_week),
            "runs_dates": sorted(r["started"] for r in seen.values()),
            "ledger": os.path.relpath(ledger_path, repo).replace("\\", "/")}


# ----------------------------------------------------------------------------- digest period & flags
def digest_period(out_dir, today):
    path = os.path.join(out_dir, "digest.md")
    last_to = None
    if os.path.exists(path):
        for m in re.finditer(r"^## (\d{4}-\d{2}-\d{2}) 〜 (\d{4}-\d{2}-\d{2})", read_text(out_dir, "digest.md"), re.M):
            last_to = dt.date.fromisoformat(m.group(2))
    since = (last_to + dt.timedelta(days=1)) if last_to else today - dt.timedelta(days=28)
    return since, today, last_to


def build_flags(stats):
    flags = []
    weeks = [w for w in stats["flow"]["weeks"] if not w["partial"]]
    if not weeks:
        return flags
    lcw = weeks[-1]
    l4 = weeks[-4:]
    code_add4 = sum(w["code_add"] for w in l4)
    test_add4 = sum(w["test_add"] for w in l4)
    if lcw["code_del"] > lcw["code_add"] and lcw["code_del"] > 300:
        flags.append({"kind": "cleanup_week", "level": "good", "text": f"{lcw.get('week_name', lcw['week'])}はコードの削除({lcw['code_del']:,})が追加({lcw['code_add']:,})を上回る掃除週"})
    if code_add4 > 500:
        ratio = test_add4 / max(1, code_add4 - test_add4)
        if ratio < 0.15:
            flags.append({"kind": "test_investment_low", "level": "watch", "text": f"直近 4 週のテスト投資率 {ratio:.2f}(テスト追加行 / 本体追加行)が低い"})
        elif ratio >= 0.5:
            flags.append({"kind": "test_investment_high", "level": "good", "text": f"直近 4 週のテスト投資率 {ratio:.2f} は厚い"})
    h = stats["health"]
    if h.get("config_keys_4w_ago") is not None:
        d = stats["size"]["config"]["keys"] - h["config_keys_4w_ago"]
        if d >= 100:
            flags.append({"kind": "config_growth", "level": "watch", "text": f"設定キーが 4 週で +{d}(現在 {stats['size']['config']['keys']})"})
    if h.get("todo_4w_ago") is not None and h["todo_now"] - h["todo_4w_ago"] >= 5:
        flags.append({"kind": "todo_growth", "level": "info", "text": f"TODO/FIXME が 4 週で +{h['todo_now'] - h['todo_4w_ago']}(参考値)"})
    if stats["process"]["wip"] > 6:
        flags.append({"kind": "wip_high", "level": "watch", "text": f"直下の 0xx PRD が {stats['process']['wip']} 本(WIP 多め)"})
    cc = sum(w["code_commits"] for w in l4)
    cd = sum(w["code_with_docs"] for w in l4)
    if cc >= 8 and cd / cc < 0.3:
        flags.append({"kind": "docs_coupdate_low", "level": "watch", "text": f"直近 4 週でコード変更コミットのうち設計文書も同時更新したのは {cd}/{cc}"})
    if stats["activity"]["streak_weeks"] >= 4:
        flags.append({"kind": "streak", "level": "good", "text": f"{stats['activity']['streak_weeks']} 週連続で稼働"})
    ut = [u for u in stats["tests"]["untested"] if u.get("era") in ("pre", None)]
    if ut:
        u = ut[0]
        flags.append({"kind": "untested_pre_era", "level": "info", "text": f"未テストで最大は {u['path']}({u['loc']} 行、{u['era'] or '年代不明'})"})
    if stats["period"]["bulk"] > 0:
        top = ", ".join(f"{p}({n:,})" for p, n in stats["period"]["bulk_top"][:3])
        flags.append({"kind": "bulk", "level": "info", "text": f"期間中の bulk 除外 {stats['period']['bulk']:,} 行: {top}"})
    return flags


# ----------------------------------------------------------------------------- outputs
def write_summary(stats, out_dir):
    P = stats["period"]
    T = stats["tests"]
    cats = T["categories"]
    thin = sorted(((k, v) for k, v in cats.items() if v["prod_loc"] > 300 and k not in ("test_infra",)), key=lambda kv: kv[1]["test_loc"] / max(1, kv[1]["prod_loc"]))[:3]
    lab = stats["meta"]["category_labels"]
    lines = [f"# anet-stats summary ({stats['meta']['generated']}, HEAD {stats['meta']['head']})", ""]
    if P.get("empty"):
        lines += [f"**digest は追記しない**: 前回 digest が {P['last_digest_to']} まで書かれており、それ以降の新しい期間が無い(同日の再実行)。"
                  f" 期間を指定して書き直す場合は `--since <YYYY-MM-DD>` を使う。以下の規模・テスト・プロセスの数字は現在値なので参照してよい。", ""]
    lines += [f"期間: {P['since']} 〜 {P['until']} (前回 digest: {P['last_digest_to'] or '無し'})", "",
             "## 期間の活動",
             f"- 稼働日 {P['active_days']}、コミット {P['commits']}、Run 起動 {P['runs']}、連続稼働 {stats['activity']['streak_weeks']} 週",
             f"- コード 追加 {P['code_add']:,} / 削除 {P['code_del']:,}(うちテスト 追加 {P['test_add']:,})、docs 追加 {P['docs_add']:,} / 削除 {P['docs_del']:,}、config 追加 {P['config_add']:,} / 削除 {P['config_del']:,}",
             f"- 主種別 {P['top_type']}、主 Topic {('#%s %s' % tuple(P['top_topic'])) if P['top_topic'] else '-'}、コード変更コミットの docs 同時更新 {P['code_with_docs']}/{P['code_commits']}",
             f"- PRD: 直下 0xx {stats['process']['wip']} 本、9xx {stats['process']['backlog_9xx']} 本、未コミット {stats['process']['uncommitted']['entries']} 件(+{stats['process']['uncommitted']['insertions']:,}/-{stats['process']['uncommitted']['deletions']:,})",
             "", "## テスト充足(C++)",
             f"- 全体 test/prod {T['total']['test_loc'] / max(1, T['total']['prod_loc']):.2f}、被テスト(直接) {T['total']['direct']}/{T['total']['prod_files']}、被テスト(間接込み) {T['total']['any']}/{T['total']['prod_files']}",
             ]
    if T["era_mode"] == "blame":
        lines.append(f"- 年代(境界 {T['boundary']}): 以前ファイル {T['total']['pre_tested']}/{T['total']['pre_files']} 被テスト、以後 {T['total']['post_tested']}/{T['total']['post_files']}、以前に書かれた行の割合 {T['total']['pre_lines'] / max(1, T['total']['pre_lines'] + T['total']['post_lines']):.0%}")
    lines.append("- 薄いカテゴリ: " + "、".join(f"{lab.get(k, k)} {v['test_loc'] / max(1, v['prod_loc']):.2f}" for k, v in thin))
    lines.append("- 未テスト上位: " + "、".join(f"{u['path']}({u['loc']}行,{u['era'] or '?'})" for u in T["untested"][:5]))
    lines += ["", "## insight flags"] + [f"- [{f['level']}] {f['text']}" for f in stats["flags"]] + ["",
              "## 参考", f"- TODO/FIXME {stats['health']['todo_now']}(4 週前 {stats['health']['todo_4w_ago']})、台帳 open {stats['health']['registry']['open']}",
              f"- 直近の完了週: " + ", ".join(f"{w.get('week_name', w['week'])}: 稼働 {w['active_days']}日 / コード +{w['code_add']:,} -{w['code_del']:,} / Run {w.get('runs', 0)}" for w in [x for x in stats['flow']['weeks'] if not x['partial']][-4:])]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def write_trend_md(stats, out_dir, nweeks):
    M = stats["meta"]
    lab = M["category_labels"]
    weeks = stats["flow"]["weeks"][-nweeks:]
    rows = []
    for w in weeks:
        tt = w["top_topic"]
        rows.append([w.get("week_name", w["week"]) + (" (途中)" if w["partial"] else ""), w["active_days"], w["commits"], w.get("runs", 0),
                     f"{w['code_add']:,} / {w['code_del']:,}", f"{w['test_add']:,}", f"{w['docs_add']:,} / {w['docs_del']:,}",
                     f"{w['bulk']:,}", w["top_type"] or "-", (f"#{tt[0]} {tt[1]}" if tt else "-"), f"{w['code_with_docs']}/{w['code_commits']}", w["median_lines_per_commit"]])
    out = [f"# anet-lab trend", "", f"生成 {M['generated']}、HEAD `{M['head']}`、境界日 {stats['tests']['boundary']}、cloc {'使用' if stats['size']['cloc'] else '未使用'}、年代 {stats['tests']['era_mode']}。",
           "正本は git 履歴と Run フォルダ。この表は毎回再生成される。コミット数は参考列で、稼働日と行の流量と Run 起動数を主軸に読む。", "",
           "## 活動と流量(週次)", "", "週は開始日(月曜)で示す。", "",
           md_table(["週", "稼働日", "commits", "Run 起動", "コード 追加 / 削除", "うちテスト追加", "docs 追加 / 削除", "bulk 除外", "主種別", "主 Topic", "docs 同時更新", "粒度中央値"], rows), "",
           f"連続稼働 {stats['activity']['streak_weeks']} 週。Run 台帳 {stats['activity']['runs_total']} 本(今回追加 {stats['activity']['runs_new_in_ledger']})。", ""]
    # 規模
    S = stats["size"]
    srows = [[a, v["files"], f"{v['code']:,}", f"{v['comment']:,}", (f"{v['comment'] / max(1, v['code']):.2f}" if S["cloc"] else "-")] for a, v in sorted(S["by_area"].items(), key=lambda kv: -kv[1]["code"])]
    out += ["## 規模(現在)", "", md_table(["領域", "ファイル", "code 行", "comment 行", "comment/code"], srows), "",
            f"config: {S['config']['lines']:,} 行、キー {S['config']['keys']:,}。設計文書 jp {S['docs']['design_jp_files']} 本 {S['docs']['design_jp_lines']:,} 行、en {S['docs']['design_en_files']} 本 {S['docs']['design_en_lines']:,} 行。ADR {S['docs']['adr_files']}。",
            f"テスト数: C++ TEST_CASE {S['test_counts']['cpp_test_cases']}、Java @Test {S['test_counts']['java_test_methods']}、Python test_ {S['test_counts']['python_test_functions']}。", ""]
    # テスト
    T = stats["tests"]
    trows = []
    for k, v in sorted(T["categories"].items()):
        if k == "test_infra":
            continue
        ratio = v["test_loc"] / max(1, v["prod_loc"]) if v["prod_loc"] else 0
        era = (f"{v['pre_tested']}/{v['pre_files']}", f"{v['post_tested']}/{v['post_files']}", f"{v['pre_lines'] / max(1, v['pre_lines'] + v['post_lines']):.0%}") if T["era_mode"] == "blame" else ("-", "-", "-")
        trows.append([lab.get(k, k) + ("" if stats["meta"]["test_target"].get(k, True) else " (対象外)"), v["prod_files"], f"{v['prod_loc']:,}", f"{v['test_loc']:,}", f"{ratio:.2f}", f"{v['direct']}/{v['prod_files']}", f"{v['any']}/{v['prod_files']}", era[0], era[1], era[2]])
    tot = T["total"]
    trows.append(["合計(対象外を除く)", tot["prod_files"], f"{tot['prod_loc']:,}", f"{tot['test_loc']:,}", f"{tot['test_loc'] / max(1, tot['prod_loc']):.2f}", f"{tot['direct']}/{tot['prod_files']}", f"{tot['any']}/{tot['prod_files']}",
                  f"{tot['pre_tested']}/{tot['pre_files']}" if T["era_mode"] == "blame" else "-", f"{tot['post_tested']}/{tot['post_files']}" if T["era_mode"] == "blame" else "-", "-"])
    ol = T["other_languages"]
    trows.append(["Java (Viewer)", "-", f"{ol['java']['prod']:,}", f"{ol['java']['test']:,}", f"{ol['java']['test'] / max(1, ol['java']['prod']):.2f}", "-", "-", "-", "-", "-"])
    trows.append(["Python (tools)", "-", f"{ol['python']['prod']:,}", f"{ol['python']['test']:,}", f"{ol['python']['test'] / max(1, ol['python']['prod']):.2f}", "-", "-", "-", "-", "-"])
    out += ["## テスト充足(機能カテゴリ × 年代)", "",
            f"L0 = test 行 / prod 行。L1 直接 = 隣に `_test.cpp` がある。L1 間接 = 公開クラス名かヘッダ名がテストに出現(誤検出あり)。年代は git blame の行日付で境界 {T['boundary']} の前後。", "",
            md_table(["カテゴリ", "prod ファイル", "prod 行", "test 行", "L0", "L1 直接", "L1 間接込み", "以前の被テスト", "以後の被テスト", "以前の行割合"], trows), "",
            "### テスト負債(未テストの prod ファイル、LOC × churn 順)", "",
            md_table(["LOC", "churn 90日", "ファイル", "カテゴリ", "年代"], [[u["loc"], u["churn90"], u["path"], lab.get(u["category"], u["category"]), u["era"] or "?"] for u in T["untested"][:15]]), ""]
    # 健全性
    H = stats["health"]
    out += ["## 健全性", "",
            f"- TODO/FIXME {H['todo_now']}(4 週前 {H['todo_4w_ago']})。参考値。",
            f"- 台帳 open {H['registry']['open']}。優先度別 {H['registry']['by_priority']}、観点別 {H['registry']['by_lens']}。",
            f"- {H['large_threshold']} 行以上のファイル {len(H['large_files'])} 本: " + ", ".join(f"{p}({n:,})" for n, p in H["large_files"][:8]), ""]
    # プロセス
    Pp = stats["process"]
    out += ["## プロセス", "",
            md_table(["月", "PRD 起票", "done", "frozen", "dropped"], [[m["month"], m["created"], m["done"], m["frozen"], m["dropped"]] for m in Pp["monthly"][-8:]]), "",
            f"- 直下 0xx {Pp['wip']} 本、9xx バックログ {Pp['backlog_9xx']} 本、PRD 総数 {Pp['prd_count']}。",
            f"- リードタイム(done 規約 {Pp['lead_time_days']['valid_after']} より後に完了した {Pp['lead_time_days']['n']} 本): 中央値 {Pp['lead_time_days']['median']} 日、p25 {Pp['lead_time_days']['p25']}、p75 {Pp['lead_time_days']['p75']}。規約導入日の一括移動 {Pp['lead_time_days']['batch_moved_on_convention_day']} 本は除外。",
            f"- 未コミット {Pp['uncommitted']['entries']} 件、+{Pp['uncommitted']['insertions']:,} / -{Pp['uncommitted']['deletions']:,}。", ""]
    out += ["## insight flags", ""] + [f"- [{f['level']}] {f['text']}" for f in stats["flags"]] + [""]
    with open(os.path.join(out_dir, "trend.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))


HTML_TEMPLATE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>anet-lab trend</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#fff;--fg:#1b1b1b;--muted:#666;--line:#ddd;--bar:#4c78a8;--bar2:#e45756;--bar3:#54a24b;--good:#2a7;--watch:#d80;--info:#579}
@media (prefers-color-scheme: dark){:root{--bg:#141618;--fg:#e6e6e6;--muted:#9aa;--line:#333;--bar:#7aa6d6;--bar2:#f08080;--bar3:#7fcf7a}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1200px;margin:0 auto}
h1{font-size:20px;margin:8px 0}h2{font-size:16px;margin:24px 0 8px;border-bottom:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{cursor:pointer;user-select:none}
.bar{display:inline-block;height:10px;background:var(--bar);vertical-align:middle}.bar2{background:var(--bar2)}.bar3{background:var(--bar3)}
.muted{color:var(--muted)}.flag{padding:2px 6px;border-radius:4px;margin:2px 0;display:inline-block}
.good{background:var(--good);color:#fff}.watch{background:var(--watch);color:#fff}.info{background:var(--info);color:#fff}
svg{width:100%;height:120px}
</style></head><body>
<h1>anet-lab trend <span class="muted" id="meta"></span></h1>
<div id="flags"></div>
<h2>活動と流量(週次)</h2><div id="spark"></div><table id="weeks"></table>
<h2>テスト充足(機能カテゴリ × 年代)</h2><table id="tests"></table>
<h2>テスト負債</h2><table id="debt"></table>
<h2>プロセス</h2><table id="process"></table>
<script id="data" type="application/json">__DATA__</script>
<script>
const S=JSON.parse(document.getElementById('data').textContent);
const lab=S.meta.category_labels;
document.getElementById('meta').textContent=`生成 ${S.meta.generated} / HEAD ${S.meta.head} / 期間 ${S.period.since} 〜 ${S.period.until}`;
document.getElementById('flags').innerHTML=S.flags.map(f=>`<span class="flag ${f.level}">${f.text}</span>`).join(' ');
function table(id,heads,rows){const t=document.getElementById(id);t.innerHTML='<tr>'+heads.map((h,i)=>`<th data-i="${i}">${h}</th>`).join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(c=>`<td>${c}</td>`).join('')+'</tr>').join('');
 t.querySelectorAll('th').forEach(th=>th.onclick=()=>{const i=+th.dataset.i;const rs=[...t.querySelectorAll('tr')].slice(1);const num=v=>parseFloat(String(v).replace(/[^0-9.\\-]/g,''));
 const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';rs.sort((a,b)=>{const x=a.children[i].textContent,y=b.children[i].textContent;const nx=num(x),ny=num(y);const c=(isNaN(nx)||isNaN(ny))?x.localeCompare(y):nx-ny;return asc?c:-c});rs.forEach(r=>t.appendChild(r));});}
const W=S.flow.weeks.slice(-__NWEEKS__);
const mx=Math.max(1,...W.map(w=>w.code_add+w.code_del));
const sw=800,bw=sw/W.length;let svg=`<svg viewBox="0 0 ${sw} 120" preserveAspectRatio="none">`;
W.forEach((w,i)=>{const h1=100*w.code_add/mx,h2=100*w.code_del/mx;const wn=w.week_name||w.week;svg+=`<rect x="${i*bw+2}" y="${110-h1}" width="${bw/2-2}" height="${h1}" fill="var(--bar)"><title>${wn} 追加 ${w.code_add}</title></rect><rect x="${i*bw+bw/2}" y="${110-h2}" width="${bw/2-2}" height="${h2}" fill="var(--bar2)"><title>${wn} 削除 ${w.code_del}</title></rect>`;});
svg+='</svg>';document.getElementById('spark').innerHTML=svg+'<div class="muted">コード追加(青)と削除(赤)の週次。バーにカーソルで値。</div>';
table('weeks',['週','稼働日','commits','Run','コード+','コード-','テスト+','docs+','docs-','bulk','主種別','主Topic','docs同時','粒度'],W.map(w=>[(w.week_name||w.week)+(w.partial?' (途中)':''),w.active_days,w.commits,w.runs||0,w.code_add,w.code_del,w.test_add,w.docs_add,w.docs_del,w.bulk,w.top_type||'-',w.top_topic?('#'+w.top_topic[0]+' '+w.top_topic[1]):'-',`${w.code_with_docs}/${w.code_commits}`,w.median_lines_per_commit]));
const T=S.tests;const trows=Object.entries(T.categories).filter(([k])=>k!=='test_infra').sort().map(([k,v])=>{const r=v.test_loc/Math.max(1,v.prod_loc);const pre=v.pre_lines/Math.max(1,v.pre_lines+v.post_lines);
 return [lab[k]||k,v.prod_files,v.prod_loc,v.test_loc,`<span class="bar" style="width:${Math.min(100,r*60)}px"></span> ${r.toFixed(2)}`,`${v.direct}/${v.prod_files}`,`${v.any}/${v.prod_files}`,T.era_mode==='blame'?`${v.pre_tested}/${v.pre_files}`:'-',T.era_mode==='blame'?`${v.post_tested}/${v.post_files}`:'-',T.era_mode==='blame'?(pre*100).toFixed(0)+'%':'-'];});
table('tests',['カテゴリ','prodファイル','prod行','test行','L0','L1直接','L1間接込み','以前の被テスト','以後の被テスト','以前の行割合'],trows);
table('debt',['LOC','churn90','ファイル','カテゴリ','年代'],T.untested.slice(0,20).map(u=>[u.loc,u.churn90,u.path,lab[u.category]||u.category,u.era||'?']));
table('process',['月','PRD起票','done','frozen','dropped'],S.process.monthly.slice(-12).map(m=>[m.month,m.created,m.done,m.frozen,m.dropped]));
</script></body></html>
"""


def write_trend_html(stats, out_dir, nweeks):
    slim = dict(stats)
    slim = {k: v for k, v in stats.items() if k != "size"}
    slim["size"] = {k: v for k, v in stats["size"].items() if k != "per_file_code"}
    slim["tests"] = {k: v for k, v in stats["tests"].items() if k != "files"}
    slim["process"] = {k: v for k, v in stats["process"].items() if k != "prds"}
    data = json.dumps(slim, ensure_ascii=False).replace("</", "<\\/")
    out = HTML_TEMPLATE.replace("__DATA__", data).replace("__NWEEKS__", str(nweeks))
    with open(os.path.join(out_dir, "trend.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(out)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="anet-lab 定点統計")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--out", default=os.path.join("reports", "stats"))
    ap.add_argument("--weeks", type=int, default=12)
    ap.add_argument("--no-cloc", action="store_true")
    ap.add_argument("--no-blame", action="store_true")
    ap.add_argument("--no-runs", action="store_true")
    ap.add_argument("--since", default=None, help="digest 期間の開始日(既定は前回 digest の翌日、無ければ 28 日前)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    repo = os.path.abspath(args.repo)
    out_dir = os.path.join(repo, args.out)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(out_dir, exist_ok=True)
    today = dt.date.today()
    rules = Rules(repo)
    files = tracked_files(repo)

    commits = collect_commits(repo)
    flow = collect_flow(commits, rules, today)
    since, until, last_to = digest_period(out_dir, today)
    if args.since:
        since = dt.date.fromisoformat(args.since)
    period = summarize_agg(aggregate([c for c in commits if since <= c["date"] <= until], rules), rules)
    period.update({"since": since.isoformat(), "until": until.isoformat(), "last_digest_to": last_to.isoformat() if last_to else None,
                   "empty": since > until})

    size = collect_size(repo, rules, files, not args.no_cloc)
    tests = collect_tests(repo, rules, files, flow["churn90"], not args.no_blame, cache_dir)
    health = collect_health(repo, rules, size, files, today)
    process = collect_process(repo, rules, files, flow, today)
    activity = collect_activity(repo, rules, flow, today, not args.no_runs, out_dir)
    period["runs"] = sum(1 for d in activity["runs_dates"] if since <= dt.date.fromisoformat(d) <= until)
    period["prds_done"] = sum(1 for r in process["prds"] if r["state"] == "done" and r["moved"] and since <= dt.date.fromisoformat(r["moved"]) <= until)

    stats = {"meta": {"generated": today.isoformat(), "head": git(repo, "rev-parse", "--short", "HEAD").strip(), "repo": repo,
                      "category_labels": rules.cat_label, "test_target": rules.cat_test_target, "topics": {str(k): v for k, v in rules.topics.items()},
                      "weeks_in_tables": args.weeks},
             "period": period, "flow": flow, "size": size, "tests": tests, "health": health, "process": process, "activity": activity}
    stats["flags"] = build_flags(stats)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1, default=str)
    write_trend_md(stats, out_dir, args.weeks)
    write_trend_html(stats, out_dir, args.weeks)
    summary = write_summary(stats, out_dir)
    if not args.quiet:
        print(summary)
        print(f"\nwrote: {os.path.relpath(out_dir, repo)}/{{stats.json, trend.md, trend.html, summary.md}}")


if __name__ == "__main__":
    main()
