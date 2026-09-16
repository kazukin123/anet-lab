#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""anet-audit の指摘台帳(reports/registry/findings.jsonl)を扱う。

正本は findings.jsonl(1 行 1 レコード、追記と状態更新のみ)。findings.md と findings.html は派生ビューで毎回再生成する。

使い方(リポジトリルートで実行):
  python reports/tools/registry.py validate
  python reports/tools/registry.py build
  python reports/tools/registry.py add --file new_records.jsonl [--run RUN_ID]
  python reports/tools/registry.py add --json '{...1 レコード...}' [--run RUN_ID]
  python reports/tools/registry.py touch --ids F-0003,F-0007 --run RUN_ID        # 再検証して依然として存在
  python reports/tools/registry.py set --id F-0003 --status fixed --note "..." [--run RUN_ID]
  python reports/tools/registry.py list [--status open] [--lens refactor] [--path core/anet-core/src/replay] [--limit 50]
  python reports/tools/registry.py fingerprint --lens refactor --file core/anet-core/src/x.cpp --symbol Foo::Bar

レコード(必須: lens, title, file, summary, scenario, priority, confidence。他は任意):
  id            F-0001 形式。add が採番する
  fingerprint   sha1(lens|file|symbol) の先頭 12 桁。行番号は含めない(行はずれる)
  lens          refactor | defect | architecture | drift | test-gap | backlog | process
  category      観点内の分類タグ(docs/リファクタリング観点.txt の項目名など)
  title         1 行
  file          リポジトリ相対パス(コード・文書どちらも可)
  symbol        クラス/関数/節など。無ければ空文字
  evidence      [{"path":..., "line":..., "note":...}]  最終確認時の根拠
  summary       何が問題か(1〜3 文)
  scenario      具体的な失敗シナリオ、または具体的な smell(「可能性がある」だけの推測は不可)
  fix_hint      対処方針の一言(任意)
  priority      P0 | P1 | P2 | P3
  confidence    high | med | low
  status        open | fixed | wontfix | stale | promoted
  topic         GitHub Topic issue 番号(任意)
  first_seen / last_seen / updated  YYYY-MM-DD
  run_id        最後に触った実行の識別子(例 2026-09-17-claude)
  history       [{"date","status","note"}]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import sys

LENSES = {"refactor", "defect", "architecture", "drift", "test-gap", "backlog", "process"}
PRIORITIES = ["P0", "P1", "P2", "P3"]
CONFIDENCES = {"high", "med", "low"}
STATUSES = {"open", "fixed", "wontfix", "stale", "promoted"}
REQUIRED = ["lens", "title", "file", "summary", "scenario", "priority", "confidence"]


def repo_root():
    return os.path.abspath(os.getcwd())


def paths():
    root = repo_root()
    d = os.path.join(root, "reports", "registry")
    return d, os.path.join(d, "findings.jsonl"), os.path.join(d, "findings.md"), os.path.join(d, "findings.html")


def fingerprint(lens, file, symbol):
    key = f"{lens.strip().lower()}|{file.strip().replace(chr(92), '/').lower()}|{(symbol or '').strip().lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def load(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"findings.jsonl {n} 行目が JSON として読めない: {e}")
    return recs


def save(path, recs):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def validate_record(r, known_fps=None):
    errs = []
    for k in REQUIRED:
        if not r.get(k):
            errs.append(f"必須項目 {k} が無い")
    if r.get("lens") not in LENSES:
        errs.append(f"lens が不正: {r.get('lens')}")
    if r.get("priority") not in PRIORITIES:
        errs.append(f"priority が不正: {r.get('priority')}")
    if r.get("confidence") not in CONFIDENCES:
        errs.append(f"confidence が不正: {r.get('confidence')}")
    if r.get("status", "open") not in STATUSES:
        errs.append(f"status が不正: {r.get('status')}")
    if not isinstance(r.get("evidence", []), list):
        errs.append("evidence は配列")
    return errs


def next_id(recs):
    mx = 0
    for r in recs:
        try:
            mx = max(mx, int(str(r.get("id", "F-0")).split("-")[1]))
        except (IndexError, ValueError):
            pass
    return f"F-{mx + 1:04d}"


def cmd_validate(args):
    _, jl, _, _ = paths()
    recs = load(jl)
    bad = 0
    fps = {}
    for r in recs:
        errs = validate_record(r)
        fp = r.get("fingerprint")
        if fp in fps:
            errs.append(f"fingerprint 重複: {fps[fp]}")
        fps[fp] = r.get("id")
        if errs:
            bad += 1
            print(f"{r.get('id', '?')}: " + "; ".join(errs))
    print(f"{len(recs)} レコード、問題 {bad} 件")
    return 1 if bad else 0


def cmd_add(args):
    _, jl, _, _ = paths()
    recs = load(jl)
    by_fp = {r.get("fingerprint"): r for r in recs}
    today = dt.date.today().isoformat()
    new = []
    if args.json:
        new = [json.loads(args.json)]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            new = [json.loads(l) for l in f if l.strip()]
    else:
        new = [json.loads(l) for l in sys.stdin if l.strip()]
    added, dup, reopened = [], [], []
    for r in new:
        errs = validate_record(r)
        if errs:
            print(f"skip ({r.get('title', '?')}): " + "; ".join(errs))
            continue
        fp = fingerprint(r["lens"], r["file"], r.get("symbol", ""))
        r["fingerprint"] = fp
        if fp in by_fp:
            ex = by_fp[fp]
            ex["last_seen"] = today
            ex["updated"] = today
            if args.run:
                ex["run_id"] = args.run
            if r.get("evidence"):
                ex["evidence"] = r["evidence"]
            if ex.get("status") == "stale":
                ex["status"] = "open"
                ex.setdefault("history", []).append({"date": today, "status": "open", "note": "再検出で stale から復帰"})
                reopened.append(ex["id"])
            else:
                dup.append(ex["id"])
            continue
        r.setdefault("id", next_id(recs))
        r.setdefault("status", "open")
        r.setdefault("symbol", "")
        r.setdefault("evidence", [])
        r.setdefault("first_seen", today)
        r["last_seen"] = today
        r["updated"] = today
        if args.run:
            r["run_id"] = args.run
        r.setdefault("history", [{"date": today, "status": "open", "note": "初出"}])
        recs.append(r)
        by_fp[fp] = r
        added.append(r["id"])
    save(jl, recs)
    print(f"added {len(added)}: {', '.join(added)}")
    if dup:
        print(f"already known (last_seen 更新) {len(dup)}: {', '.join(dup)}")
    if reopened:
        print(f"reopened {len(reopened)}: {', '.join(reopened)}")
    return 0


def cmd_touch(args):
    _, jl, _, _ = paths()
    recs = load(jl)
    today = dt.date.today().isoformat()
    ids = set(x.strip() for x in args.ids.split(",") if x.strip())
    n = 0
    for r in recs:
        if r.get("id") in ids:
            r["last_seen"] = today
            r["updated"] = today
            if args.run:
                r["run_id"] = args.run
            n += 1
    save(jl, recs)
    print(f"touched {n}")
    return 0


def cmd_set(args):
    _, jl, _, _ = paths()
    recs = load(jl)
    today = dt.date.today().isoformat()
    if args.status not in STATUSES:
        raise SystemExit(f"status が不正: {args.status}")
    for r in recs:
        if r.get("id") == args.id:
            r["status"] = args.status
            r["updated"] = today
            if args.run:
                r["run_id"] = args.run
            r.setdefault("history", []).append({"date": today, "status": args.status, "note": args.note or ""})
            save(jl, recs)
            print(f"{args.id}: {args.status}")
            return 0
    raise SystemExit(f"{args.id} が無い")


def cmd_list(args):
    _, jl, _, _ = paths()
    recs = load(jl)
    out = []
    for r in recs:
        if args.status and r.get("status") != args.status:
            continue
        if args.lens and r.get("lens") != args.lens:
            continue
        if args.path and not r.get("file", "").startswith(args.path):
            continue
        out.append(r)
    out.sort(key=lambda r: (PRIORITIES.index(r.get("priority", "P3")) if r.get("priority") in PRIORITIES else 9, r.get("id", "")))
    for r in out[: args.limit]:
        print(f"{r.get('id')} {r.get('status'):8} {r.get('priority')} {r.get('confidence'):4} {r.get('lens'):12} {r.get('file')}{('::' + r['symbol']) if r.get('symbol') else ''}  {r.get('title')}  [last_seen {r.get('last_seen')}]")
    print(f"{len(out)} 件")
    return 0


def build_md(recs, today):
    by_status = {}
    for r in recs:
        by_status.setdefault(r.get("status", "?"), []).append(r)
    open_recs = sorted(by_status.get("open", []), key=lambda r: (PRIORITIES.index(r.get("priority", "P3")) if r.get("priority") in PRIORITIES else 9, -(["low", "med", "high"].index(r.get("confidence", "low"))), r.get("id", "")))
    recent_cut = (dt.date.fromisoformat(today) - dt.timedelta(days=14)).isoformat()
    recent = [r for r in recs if (r.get("updated") or "") >= recent_cut]
    lines = ["# 指摘台帳(派生ビュー)", "", f"生成 {today}。正本は findings.jsonl。この文書は毎回再生成される。", "",
             "## 概況", "",
             "| 状態 | 件数 |", "|---|---|"] + [f"| {s} | {len(v)} |" for s, v in sorted(by_status.items())] + ["",
             "open の優先度別: " + ", ".join(f"{p} {sum(1 for r in open_recs if r.get('priority') == p)}" for p in PRIORITIES), "",
             "open の観点別: " + ", ".join(f"{l} {sum(1 for r in open_recs if r.get('lens') == l)}" for l in sorted(LENSES)), "",
             "## 最近の変化(14 日)", ""]
    if recent:
        lines += ["| id | 更新 | 状態 | 優先度 | 観点 | 場所 | タイトル |", "|---|---|---|---|---|---|---|"]
        for r in sorted(recent, key=lambda r: r.get("updated", ""), reverse=True):
            lines.append(f"| {r.get('id')} | {r.get('updated')} | {r.get('status')} | {r.get('priority')} | {r.get('lens')} | `{r.get('file')}`{('::' + r['symbol']) if r.get('symbol') else ''} | {r.get('title')} |")
    else:
        lines.append("変化なし。")
    lines += ["", "## open(優先度順)", ""]
    if open_recs:
        lines += ["| id | 優先度 | 確度 | 観点 | 分類 | 場所 | タイトル | 初出 | 最終確認 |", "|---|---|---|---|---|---|---|---|---|"]
        for r in open_recs:
            lines.append(f"| {r.get('id')} | {r.get('priority')} | {r.get('confidence')} | {r.get('lens')} | {r.get('category', '')} | `{r.get('file')}`{('::' + r['symbol']) if r.get('symbol') else ''} | {r.get('title')} | {r.get('first_seen')} | {r.get('last_seen')} |")
        lines += ["", "### 詳細", ""]
        for r in open_recs:
            ev = "; ".join(f"{e.get('path')}:{e.get('line')} {e.get('note', '')}".strip() for e in r.get("evidence", []) if isinstance(e, dict))
            lines += [f"#### {r.get('id')} {r.get('priority')} {r.get('title')}", "",
                      f"- 観点/分類: {r.get('lens')} / {r.get('category', '')}", f"- 場所: `{r.get('file')}`{('::' + r['symbol']) if r.get('symbol') else ''}",
                      f"- 要約: {r.get('summary')}", f"- シナリオ: {r.get('scenario')}"]
            if r.get("fix_hint"):
                lines.append(f"- 対処: {r['fix_hint']}")
            if ev:
                lines.append(f"- 根拠: {ev}")
            lines.append("")
    else:
        lines.append("open は無し。")
    closed = [r for r in recs if r.get("status") in ("fixed", "wontfix", "promoted", "stale")]
    lines += ["", "## クローズ済み・保留", ""]
    if closed:
        lines += ["| id | 状態 | 優先度 | 観点 | 場所 | タイトル | 更新 |", "|---|---|---|---|---|---|---|"]
        for r in sorted(closed, key=lambda r: r.get("updated", ""), reverse=True):
            lines.append(f"| {r.get('id')} | {r.get('status')} | {r.get('priority')} | {r.get('lens')} | `{r.get('file')}` | {r.get('title')} | {r.get('updated')} |")
    else:
        lines.append("無し。")
    return "\n".join(lines) + "\n"


HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>anet-lab findings</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
:root{--bg:#fff;--fg:#1b1b1b;--muted:#666;--line:#ddd;--acc:#4c78a8}
@media (prefers-color-scheme: dark){:root{--bg:#141618;--fg:#e6e6e6;--muted:#9aa;--line:#333;--acc:#7aa6d6}}
body{margin:0 auto;padding:16px;max-width:1300px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
h1{font-size:20px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
th{cursor:pointer;user-select:none;white-space:nowrap}input,select{margin:0 6px 8px 0;padding:4px}.muted{color:var(--muted)}
.P0{color:#c00;font-weight:600}.P1{color:#d60}.P2{color:#37a}.P3{color:var(--muted)}details{margin:0}summary{cursor:pointer}
</style></head><body><h1>anet-lab findings <span class="muted" id="meta"></span></h1>
<div><input id="q" placeholder="検索(タイトル/場所/要約)" size="40">
<select id="fs"><option value="">状態: 全部</option><option selected>open</option><option>fixed</option><option>wontfix</option><option>stale</option><option>promoted</option></select>
<select id="fl"><option value="">観点: 全部</option><option>refactor</option><option>defect</option><option>architecture</option><option>drift</option><option>test-gap</option><option>backlog</option><option>process</option></select>
<select id="fp"><option value="">優先度: 全部</option><option>P0</option><option>P1</option><option>P2</option><option>P3</option></select>
<span class="muted" id="count"></span></div>
<table id="t"><thead><tr><th data-k="id">id</th><th data-k="status">状態</th><th data-k="priority">優先度</th><th data-k="confidence">確度</th><th data-k="lens">観点</th><th data-k="category">分類</th><th data-k="file">場所</th><th data-k="title">タイトル</th><th data-k="first_seen">初出</th><th data-k="last_seen">最終確認</th></tr></thead><tbody></tbody></table>
<script id="data" type="application/json">__DATA__</script><script>
const R=JSON.parse(document.getElementById('data').textContent);document.getElementById('meta').textContent=`${R.length} 件 / 生成 __TODAY__`;
let sortK='priority',asc=true;const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function render(){const q=document.getElementById('q').value.toLowerCase(),fs=document.getElementById('fs').value,fl=document.getElementById('fl').value,fp=document.getElementById('fp').value;
 let rows=R.filter(r=>(!fs||r.status===fs)&&(!fl||r.lens===fl)&&(!fp||r.priority===fp)&&(!q||[r.title,r.file,r.symbol,r.summary,r.category].join(' ').toLowerCase().includes(q)));
 rows.sort((a,b)=>{const x=a[sortK]??'',y=b[sortK]??'';const c=String(x).localeCompare(String(y));return asc?c:-c});
 document.getElementById('count').textContent=`${rows.length} 件表示`;
 document.querySelector('#t tbody').innerHTML=rows.map(r=>`<tr><td>${esc(r.id)}</td><td>${esc(r.status)}</td><td class="${esc(r.priority)}">${esc(r.priority)}</td><td>${esc(r.confidence)}</td><td>${esc(r.lens)}</td><td>${esc(r.category)}</td><td><code>${esc(r.file)}</code>${r.symbol?'::'+esc(r.symbol):''}</td><td><details><summary>${esc(r.title)}</summary><div>${esc(r.summary)}</div><div class="muted">${esc(r.scenario)}</div>${r.fix_hint?'<div>対処: '+esc(r.fix_hint)+'</div>':''}${(r.evidence||[]).map(e=>`<div class="muted">${esc(e.path)}:${esc(e.line)} ${esc(e.note||'')}</div>`).join('')}</details></td><td>${esc(r.first_seen)}</td><td>${esc(r.last_seen)}</td></tr>`).join('');}
document.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(sortK===k)asc=!asc;else{sortK=k;asc=true}render()});
['q','fs','fl','fp'].forEach(id=>document.getElementById(id).oninput=render);render();
</script></body></html>
"""


def cmd_build(args):
    _, jl, md, ht = paths()
    recs = load(jl)
    today = dt.date.today().isoformat()
    with open(md, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_md(recs, today))
    data = json.dumps(recs, ensure_ascii=False).replace("</", "<\\/")
    with open(ht, "w", encoding="utf-8", newline="\n") as f:
        f.write(HTML.replace("__DATA__", data).replace("__TODAY__", today))
    print(f"built findings.md / findings.html from {len(recs)} records")
    return 0


def cmd_fingerprint(args):
    print(fingerprint(args.lens, args.file, args.symbol or ""))
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="anet-audit 指摘台帳")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("build")
    a = sub.add_parser("add")
    a.add_argument("--file")
    a.add_argument("--json")
    a.add_argument("--run")
    t = sub.add_parser("touch")
    t.add_argument("--ids", required=True)
    t.add_argument("--run")
    s = sub.add_parser("set")
    s.add_argument("--id", required=True)
    s.add_argument("--status", required=True)
    s.add_argument("--note")
    s.add_argument("--run")
    l = sub.add_parser("list")
    l.add_argument("--status")
    l.add_argument("--lens")
    l.add_argument("--path")
    l.add_argument("--limit", type=int, default=100)
    fp = sub.add_parser("fingerprint")
    fp.add_argument("--lens", required=True)
    fp.add_argument("--file", required=True)
    fp.add_argument("--symbol")
    args = ap.parse_args()
    return {"validate": cmd_validate, "build": cmd_build, "add": cmd_add, "touch": cmd_touch, "set": cmd_set, "list": cmd_list, "fingerprint": cmd_fingerprint}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
