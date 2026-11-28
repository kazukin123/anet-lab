---
title: 終了時の背景評価セッションは完走を既定とし、設定または上限時間による協調キャンセルで抜けられるようにして、部分集計は設けない
page_kind: サマリ
material: "[[docs/adr/0045-eval-session-drain-on-exit-wait-or-discard.md|0045-eval-session-drain-on-exit-wait-or-discard]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# 終了時の背景評価セッションは完走を既定とし、設定または上限時間による協調キャンセルで抜けられるようにして、部分集計は設けない

## 元の材料

[ADR 0045](../../../adr/0045-eval-session-drain-on-exit-wait-or-discard.md) blob:acdf0d4865e8bdad6df997f5c0a6db4cee9bafec

## 要点

- 学習末に起動した背景評価を取り逃さないよう、終了時は実行中のセッションの完走を既定にする。
- 待たない設定または全体の上限時間に達したときは協調キャンセルする。
- ワーカーが停止する前に Logger や RunManager を破棄しない。
- スロット別の上限だと終了待ちの合計が増えるため、終了経路全体の上限を使う。
- 打ち切ったセッションは部分集計の scalar を出さず、完了済みの個別 trace を残す。

## 内容

### 停止と記録

終了期限まで背景の評価を完了させる。待ち切れない場合は評価ループがステップの境界で停止要求を見る。単に future を待たない方式では、動作中のワーカーが破棄済みの記録先に触れうる。

### 部分結果

採用本数がそろわない平均を確定すると、同じ Run 内で評価時点ごとの母数が変わる。完了済み個体の trace は残し、セッション単位の値は確定させない。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
