---
title: 強制行動は `MakeAction` の入力とし、Actor の出力を実行された行動にする
page_kind: サマリ
material: "[[docs/adr/0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action.md|0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
updated: 2026-11-26
---
# 強制行動は `MakeAction` の入力とし、Actor の出力を実行された行動にする

## 元の材料

[ADR 0083](../../../adr/0083-forced-actions-are-make-action-input-and-actor-output-is-executed-action.md) blob:a153c4018746702b869becb009984539eddcca89

## 要点

- 強制行動は Runner の後処理から、lane ごとの MakeAction 入力へ移す。
- Actor が方策を評価した後で強制指定を反映する。
- Actor の出力が環境で実行する行動になる。
- 優先度ヒント、skip の判断、直前行動は同じ最終行動から求める。
- 行動を書き換えるたびに aux と派生値を作り直す専用経路を不要にする。

## 主要な概念

- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
