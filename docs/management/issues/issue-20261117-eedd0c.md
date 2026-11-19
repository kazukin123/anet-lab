---
id: issue-20261117-eedd0c
title: Evaluation Trace の単一行選択テストが合成ドラッグ後に失敗する
status: 調査中
raised: 2026-11-17
phase: 実装
scope: Runner GUI の Evaluation Trace と単一行選択の回帰テスト
kind: 成果物
area:
  - 110_runner_app
type: 不具合
domain: UI
ai_judge: 未確認
ai_judged:
docs:
  - docs/memo/170_nn_trace_capture_unification_20impl.md
  - docs/design/160_applications_and_tools.jp.md
prd:
priority: 中
depends: []
parent:
value: 3
urgency: 8
risk: 2
size: 5
wsjf: 2.6
resolved:
resolution:
memo:
tags: []
---

## 背景

PRD170 の実装で、変更前後の既定テスト一式を比較した際に見つかった。PRD170 は nn_trace の capture 統合を対象とし、この失敗は着手前からあるため修正範囲に含めなかった。工程末に候補として提示し、開発者がこの件の新規起票を選んだ。

## 事象

- [EvalTracePanel_test.cpp](../../../src/apps/runner/src/EvalTracePanel_test.cpp) の `Trace keeps one selected row and fixed row heights`（`[trace_ui_single_row]`）が失敗する。
- テストは3行を追加したGridで、0行目のセルに合成した左ボタン押下を送り、押したまま1行目・2行目へ移動し、2行目でボタンを離す。その直後の `REQUIRE(ui.panel->IsTraceMode())` が false になる。
- PRD170 の着手前と最終実装後の通常 GUI 一式を、同じ Catch seed `3413744419` で実行すると、両方とも39 cases中38 pass / 1 fail、894 assertions中893 pass / 1 fail。失敗する条件は同じで、着手前は576行目、最終実装後は581行目だった。
- 実画面の同じ操作が失敗するか、合成イベントの準備に原因があるかは未確認。製品の選択処理とテスト側のどちらを直すべきかは、まだ特定していない。
- ローカルの証拠は `.scratch/prd170/gui-before.log`、`gui-after-results.log`、`gui-final-results.log`。これらはGit管理外であり、結果の要約は[PRD170実装記録](../../memo/170_nn_trace_capture_unification_20impl.md)にも残っている。

## 影響

通常 GUI 一式が非0で終了し、単一行選択と固定行高の回帰確認を最後まで通せない。この既存失敗と、新しい変更による失敗を区別して確認する手間がかかる。

## 対応方針

同じ seed の通常 GUI 一式と対象テストを比較し、実画面の選択結果、Gridの座標・表示状態、合成イベントが通る経路を照合する。原因を特定してから製品またはテスト側を修正し、既存の単一行選択・行高の検査を維持したまま通常一式の成功を確認する。assertionを削除したり、失敗するテストを既定一式から除いたりして解決扱いにしない。

## 対応内容
