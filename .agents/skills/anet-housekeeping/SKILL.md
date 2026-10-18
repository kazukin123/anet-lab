---
name: anet-housekeeping
description: Run anet-lab's routine upkeep (fixed-point stats, integrated audit, design-doc translation, survey queue, harness map refresh, atlas refresh) within a token and time budget, proposing tasks and their order for user approval before running the approved plan autonomously. Use when the user invokes $anet-housekeeping or /anet-housekeeping, optionally with a budget such as auto, S, M, L or 40%, and optional until=, only=, skip=, dry-run.
---

# ANET Housekeeping

空いた予算で、リポジトリの定常整備を回す。候補表と予算から実行案を選定し、実行対象についてユーザーの了承を得てから開始する。了承された範囲内は自律的に進める。個々の作業は各スキルが行い、このスキルは選定・了承確認・実行・記録を担う。

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
| audit | anet-audit | 常に可 | 観点 1 × 範囲 1 | 10% / 15 |
| translate | anet-translate-docs | 未訳、続き、または訳元更新のある `docs/design/*.jp.md` がある(判定は anet-translate-docs の「対象の決め方」) | ファイル 1 本 | 8% / 10 |
| survey | anet-survey-queue | `reports/INDEX.md` のキューに状態 `ready` の行がある | テーマ 1 つ | 30% / 40 |
| harness-map | anet-harness-map | `archify` スキルが存在し、`docs/agents/README.md` が無い、または記録された revision から `git diff --stat <rev> HEAD -- AGENTS.md CLAUDE.md .agents/skills .claude/skills docs/agents docs/design core/anet-core/testdata .github/workflows` が空でない | 1 回 | 15% / 20 |
| atlas | anet-archify-atlas | `archify` スキルが存在し、`docs/archify/README.md` に記録された revision から `git diff --stat <rev> HEAD -- core apps` の変更行が 3,000 を超える | 1 回 | 20% / 30 |

既定の優先順は stats, audit, translate, survey, harness-map, atlas。

## 見積もりの較正

`reports/housekeeping/runs.jsonl` に過去の実測がある。同じ `name` の直近 5 件の `pct` と `minutes` の中央値を見積もりに使い、無ければ初期見積を使う。`pct` が null の行は分の較正にだけ使う。

## 手順

1. 引数と予算を確定し、`run_id = <YYYY-MM-DD>-<claude|codex>-<HHMM>` を決める。
2. 候補ごとに前提条件を確認し、通ったものを優先順に並べる。`only` / `skip` を適用する。
3. 選定: 残予算と残時間に収まる限り上から取る。translate は残りが許せば複数ファイル(1 本ずつ別単位)。1 つも入らなければ「予算不足」と報告して終わる。
4. `dry-run` ならここで選定結果を報告して終わる。
5. 実行案として対象・範囲(観点、ファイル、本数など)、順序、消費率と所要時間の見積もり、出力先を提示し、その内容で実行してよいかユーザーに確認する。了承が得られるまで待ち、各候補の実行や成果物・実行ログへの書き込みは始めない。
   - 了承前は、選定と見積もりに必要な読み取り専用の調査だけを行う。
   - スキルの呼び出し、予算の指定、対象・順序の調整だけを実行開始の了承とみなさない。提示した実行案への了承と実行開始の意思が明確になってから進める。
   - 同じ実行案への明確な了承が会話内ですでに得られている場合は、重ねて確認しない。了承後に対象や範囲を追加・拡大する場合は、その変更について了承を得る。
6. 了承された候補を順に実行する。
   - 開始時刻と、取れれば使用率を記録する。
   - 対象スキルの `SKILL.md` を読み、その手順に従う。Claude Code では Skill ツールで呼び出してもよい。
   - 終了時刻と使用率を記録し、`runs.jsonl` に 1 行追記する。失敗しても行は残す(`outcome: failed`)。
7. 各単位の後で予算を読み直す。残率が予備 5% を割る、または残時間が次の見積もりに足りなければ、そこで止める。
8. 完了報告を出す。

## runs.jsonl

1 行 1 単位。

```json
{"run_id": "2026-09-17-claude-2130", "tool": "claude", "name": "audit", "unit": "refactor × 318_rl_core", "started": "2026-09-17T21:30", "ended": "2026-09-17T21:44", "minutes": 14, "pct_before": 40, "pct_after": 49, "pct": 9, "outcome": "done", "note": "新規 3 件"}
```

`outcome` は `done` / `partial` / `skipped` / `failed`。使用率が取れないときは `pct_before` `pct_after` `pct` を null にする。

`harness-map` の `note` には anet-harness-map の完了報告の件数(死んだ動線 / 実体コピーのドリフト / shadow / 移設候補の節数と行数合計)を書く。

## 禁止

- git の書き込み操作(add、commit、push、checkout、stash、branch、reset)。人間が行う。
- 実行対象の了承を得ずに作業を開始すること。了承後の範囲内で判断が要る細部は既定値で進め、完了報告に「判断した点」として書く。
- コードの変更(v1)。各スキルの出力先と `reports/housekeeping/` 以外への書き込み。
- 予算が読めないのに `auto` を仮定すること。

## 完了報告

日本語で 10 行程度。予算(入力、残率、残時間)、選んだ候補と順序、各候補の結果と出力パス、次回に回した候補、判断した点。
