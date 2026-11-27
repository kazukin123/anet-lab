---
title: 観測 sweep / 可視化の NN 呼び出しを TensorDictFunction に一本化する
page_kind: サマリ
material: "[[docs/adr/0002-tensordict-function-unify.md|0002-tensordict-function-unify]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-26
---
# 観測 sweep / 可視化の NN 呼び出しを TensorDictFunction に一本化する

## 元の材料

[ADR 0002](../../../adr/0002-tensordict-function-unify.md) blob:08c8ecf818f0695e170af5ff4ee6024d8f29a749

## 要点

- 複数の観測キーを受ける状態スイープは TensorDictFunction を使う。
- 格子を作るベクトル観測だけを変え、他のキーは固定値とともに辞書で渡す。
- 従来の単一 Tensor 用入口は全 Agent で値を返さない状態だった。
- 単一キーへの変換アダプターでは将来の格子観測を運べないため旧入口を撤去する。
- ヒートマップと Agent の NN 呼び出しが同じ観測契約を使う。
- 辞書経由でも policy/target Network の Head を選べるようにし、従来の conv2d 用経路だけに限定しない。

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
