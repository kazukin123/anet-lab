---
title: Perception の stack の対象のキーは、1 枚でも必ず stack 軸を持つ
page_kind: サマリ
material: "[[docs/adr/0086-perception-stack-target-keys-always-carry-stack-axis.md|0086-perception-stack-target-keys-always-carry-stack-axis]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# Perception の stack の対象のキーは、1 枚でも必ず stack 軸を持つ

## 元の材料

[ADR 0086](../../../adr/0086-perception-stack-target-keys-always-carry-stack-axis.md) blob:22fd741595a09d49b79147f45d0abac91b803f61

## 要点

- 対象観測の stack 軸は1枚のときも長さ1として残す。
- 行動選択、可視化、Network の入力仕様、Sample の形をそろえる。
- Network 定義の先頭で stack 軸をどう扱うかを明示する。
- 枚数が1なら軸を省く方式では次元の意味が設定で変わる。
- 後の変更で設定元は Agent 全体となり、暗黙の1枚を加える役割は並びから DefaultDQN へ移った。

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
