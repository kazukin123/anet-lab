---
title: 行動選択に添える入力は Runner の step と MakeAction の引数で都度渡し、core に要求の登録を持たない
page_kind: サマリ
material: "[[docs/adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md|0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-26
---
# 行動選択に添える入力は Runner の step と MakeAction の引数で都度渡し、core に要求の登録を持たない

## 元の材料

[ADR 0072](../../../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md) blob:b8f63d16b72e4e6da52045634c5224e1b4dbe29a

## 要点

- 行動選択に添える要求は step ごとの MakeActionOptions で渡す。
- step を進める側が決め、Runner が Actor へ渡す。
- 可視化を Run の途中で開閉しても、その回の要求を反映できる。
- Runner や Actor に購読数を登録する形では、引数から見えない状態が出力を変えてしまう。
- 入力を引数で渡し、結果を Observer が受け取る分担を維持する。

## 主要な概念

- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
