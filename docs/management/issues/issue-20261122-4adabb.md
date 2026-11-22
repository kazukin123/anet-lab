---
id: issue-20261122-4adabb
title: PRD072 の既定葉検査が移転前の core を探索して停止する
status: 未着手
raised: 2026-11-22
phase: 実装
scope: PRD072 の既定葉検査による現行リポジトリの設定とソースの静的検査
kind: コード
area:
  - 311_dev_tools
  - 316_config
type: 不具合
domain: 開発
ai_judge: 未確認
ai_judged:
docs:
  - docs/memo/done/072_config_selection_final_value_10prd.md
  - docs/memo/191_rnd_raw_error_and_recent_learning_20impl.md
prd:
priority: 中
depends: []
parent:
value: 3
urgency: 3
risk: 2
size: 2
wsjf: 4
resolved:
resolution:
memo:
tags: []
---

## 背景

PRD191 の設定移行を検証中に、既定葉の `?=` と `=` の使い分けを点検する PRD072 の検査が停止した。実装工程の終わりに起票候補として示し、開発者が起票を選んだ。

## 事象

[既定葉検査](../../../src/core/base/testdata/prd072/check_default_leaves.py) は `rg` の探索先に `apps` と移転前の `core` を指定している。リポジトリのルートから下記を実行すると、`rg` が `core` を開けずに終了コード 2 を返す。検査は `RuntimeError` を送出し、終了コード 1 で止まった。

```powershell
.\.venv\Scripts\python.exe -B src/core/base/testdata/prd072/check_default_leaves.py
```

## 影響

現行ツリーで既定葉検査を実行しても、集計と違反一覧を取得できない。PRD191 では検査の一時コピーで探索先を `src` に読み替えて確認した。[PRD191 実装記録](../../memo/191_rnd_raw_error_and_recent_learning_20impl.md)

## 対応方針

探索先を現行の配置に合わせる。リポジトリのルートで標準の実行手順を通し、`apps` と `src` が検査されて集計が出ることを確かめる。

## 対応内容
