---
id: issue-20261122-4adabb
title: RFD0072 の既定葉検査が移転前の core を探索して停止する
status: 解決
raised: 2026-11-22
phase: 実装
scope: RFD0072 の既定葉検査による現行リポジトリの設定とソースの静的検査
kind: コード
area:
  - 311_dev_tools
  - 316_config
type: 不具合
domain: 開発
ai_judge: 残件無し
ai_judged: 2026-11-28
docs:
  - docs/memo/done/0072_config_selection_final_value_10rfd.md
  - docs/memo/0191_rnd_raw_error_and_recent_learning_20impl.md
rfd: "[[0206_core_and_app_small_fixes_10rfd]]"
priority: 中
depends: []
parent:
value: 3
urgency: 3
risk: 2
size: 2
wsjf: 4
resolved: 2026-11-28
resolution: 対応済み
memo:
tags: []
---

## 背景

PRD191 の設定移行を検証中に、既定葉の `?=` と `=` の使い分けを点検する PRD072 の検査が停止した。実装工程の終わりに起票候補として示し、開発者が起票を選んだ。

## 事象

[既定葉検査](../../../src/core/base/testdata/rfd0072/check_default_leaves.py) は `rg` の探索先に `apps` と移転前の `core` を指定している。リポジトリのルートから下記を実行すると、`rg` が `core` を開けずに終了コード 2 を返す。検査は `RuntimeError` を送出し、終了コード 1 で止まった。

```powershell
.\.venv\Scripts\python.exe -B src/core/base/testdata/prd072/check_default_leaves.py
```

- 2026-11-23 RFD0195: 実装の検証（[実装 レビュー観点](../../quality/impl.md)の項番 52。出荷設定のコメントの行だけを変えたことの確かめ）で回したところ、実装の前の状態でも実装のあとでも、同じく `rg` が `core` を開けずに止まり、回せなかった。RFD0195 で検証資材のフォルダを改名したので、検査の今のパスは `src/core/base/testdata/rfd0072/check_default_leaves.py` である。[実装記録](../../memo/0195_document_term_rfd_and_4digit_numbers_20impl.md)

## 影響

現行ツリーで既定葉検査を実行しても、集計と違反一覧を取得できない。PRD191 では検査の一時コピーで探索先を `src` に読み替えて確認した。[PRD191 実装記録](../../memo/0191_rnd_raw_error_and_recent_learning_20impl.md)

## 対応方針

探索先を現行の配置に合わせる。リポジトリのルートで標準の実行手順を通し、`apps` と `src` が検査されて集計が出ることを確かめる。

## 対応内容

- 2026-11-27 探索先は 125075dc で現在の配置へ対応済み。標準検査が apps と src を検査し、4 行を報告して終了値 1。検査のテスト 16 件が通った。 [RFD0206 実装メモ](../../memo/0206_core_and_app_small_fixes_20impl.md)
$1
- 2026-11-28 状態を `対応中` から `解決` へ変更。コミットの準備で解決の候補を選んだ。AI 判定は「残件無し」。現行の配置で既定キー検査が動作し、検査のテスト 16 件が通ることを確認した。
