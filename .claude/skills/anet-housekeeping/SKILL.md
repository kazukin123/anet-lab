---
name: anet-housekeeping
description: Run anet-lab's routine upkeep (fixed-point stats, integrated audit, design-doc translation, survey queue, commit planning, atlas refresh) autonomously within a token and time budget, choosing which tasks to run and in what order. Use when the user invokes $anet-housekeeping or /anet-housekeeping, optionally with a budget such as auto, S, M, L or 40%, and optional until=, only=, skip=, dry-run.
---

# ANET Housekeeping

空いた予算で、リポジトリの定常整備を人の判断なしに回す。何を回すかは候補表と予算から機械的に決める。個々の作業は各スキルが行い、このスキルは選定・実行・記録だけを担う。

## 引数

`$anet-housekeeping [budget] [until=<分>] [only=a,b] [skip=a,b] [dry-run]`

| 引数 | 意味 | 既定 |
|---|---|---|
| budget | `auto` / `S` / `M` / `L` / `NN%` | Claude Code デスクトップで get_usage が使えれば `auto`、それ以外は `S` |
| until | 使ってよい残り時間(分) | auto ならリセットまでの時間から 10 分引いた値。それ以外は制限なし |
| only / skip | 候補名のカンマ区切り | なし |
| dry-run | 選定結果だけ報告して実行しない | なし |

## 予算の読み方

1. **Claude Code デスクトップ**: ツール `mcp__ccd_session_mgmt__get_usage` を呼ぶ。`plan.status` が `ok` なら、`plan.windows` の各 `percentUsed` から 残率 = min(100 − percentUsed) を取り、`5-hour limit` の `resetsIn` を残時間にする。`ok` でなければ auto は使えないので `S` に落とす。
2. **Codex**: 残量 API が無い。budget 引数が無ければ `S`。
3. 消費してよい率: `auto` = 残率 − 予備 5%。`S` = 5%、`M` = 15%、`L` = 30%。`NN%` はその値。残率が読めるときは、いずれも 残率 − 5% を上限にする。
4. 時間: `until` 指定 > auto の残時間 − 10 分 > 制限なし。

## 候補と前提条件

| 名前 | スキル | 前提条件 | 単位 | 初期見積(消費率 / 分) |
|---|---|---|---|---|
| stats | anet-stats | `reports/stats/stats.json` が無い、または `meta.generated` が 7 日以上前 | 1 回 | 3% / 5 |
| commit-plan | prepare-commit | `git status --porcelain` が空でない | 1 回 | 4% / 5 |
| audit | anet-audit | 常に可 | 観点 1 × 範囲 1 | 10% / 15 |
| translate | anet-translate-docs | 未訳、続き、または訳元更新のある `docs/design/*.jp.md` がある(判定は anet-translate-docs の「対象の決め方」) | ファイル 1 本 | 8% / 10 |
| survey | anet-survey-queue | `reports/INDEX.md` のキューに状態 `ready` の行がある | テーマ 1 つ | 30% / 40 |
| atlas | anet-archify-atlas | `archify` スキルが存在し、`docs/archify/README.md` に記録された revision から `git diff --stat <rev> HEAD -- core apps` の変更行が 3,000 を超える | 1 回 | 20% / 30 |

既定の優先順は stats, commit-plan, audit, translate, survey, atlas。

## 見積もりの較正

`reports/housekeeping/runs.jsonl` に過去の実測がある。同じ `name` の直近 5 件の `pct` と `minutes` の中央値を見積もりに使い、無ければ初期見積を使う。`pct` が null の行は分の較正にだけ使う。

## 手順

1. 引数と予算を確定し、`run_id = <YYYY-MM-DD>-<claude|codex>-<HHMM>` を決める。
2. 候補ごとに前提条件を確認し、通ったものを優先順に並べる。`only` / `skip` を適用する。
3. 選定: 残予算と残時間に収まる限り上から取る。translate は残りが許せば複数ファイル(1 本ずつ別単位)。1 つも入らなければ「予算不足」と報告して終わる。
4. `dry-run` ならここで選定結果を報告して終わる。
5. 各候補を順に実行する。
   - 開始時刻と、取れれば使用率を記録する。
   - 対象スキルの `SKILL.md` を読み、その手順に従う。Claude Code では Skill ツールで呼び出してもよい。
   - 終了時刻と使用率を記録し、`runs.jsonl` に 1 行追記する。失敗しても行は残す(`outcome: failed`)。
6. 各単位の後で予算を読み直す。残率が予備 5% を割る、または残時間が次の見積もりに足りなければ、そこで止める。
7. 完了報告を出す。

## runs.jsonl

1 行 1 単位。

```json
{"run_id": "2026-09-17-claude-2130", "tool": "claude", "name": "audit", "unit": "refactor × 318_rl_core", "started": "2026-09-17T21:30", "ended": "2026-09-17T21:44", "minutes": 14, "pct_before": 40, "pct_after": 49, "pct": 9, "outcome": "done", "note": "新規 3 件"}
```

`outcome` は `done` / `partial` / `skipped` / `failed`。使用率が取れないときは `pct_before` `pct_after` `pct` を null にする。

## 禁止

- git の書き込み操作(add、commit、push、checkout、stash、branch、reset)。人間が行う。
- 質問して止まること。判断が要る点は既定値で進め、完了報告に「判断した点」として書く。
- コードの変更(v1)。各スキルの出力先と `reports/housekeeping/` 以外への書き込み。
- 予算が読めないのに `auto` を仮定すること。

## 完了報告

日本語で 10 行程度。予算(入力、残率、残時間)、選んだ候補と順序、各候補の結果と出力パス、次回に回した候補、判断した点。
