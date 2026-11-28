---
title: 勾配ノイズスケールは更新自身の batch を小バッチに分けて学習経路の中で測り、BF16 も対象にする
page_kind: サマリ
material: "[[docs/adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md|0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/phenomena/gradient_noise_batch.md|勾配ノイズとbatch]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# 勾配ノイズスケールは更新自身の batch を小バッチに分けて学習経路の中で測り、BF16 も対象にする

## 元の材料

[ADR 0054](../../../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md) blob:b8cccc2e9eb1ceccd85235628ffd15934986f503

## 要点

- 勾配ノイズスケールには実際の更新に使う重み付き要素損失を使う。
- 同じ損失グラフの batch を小分けし、追加 forward を行わず小バッチ勾配を得る。
- この方法は PER・IS 重み・分位サンプル・Munchausen を含む実更新の条件を保つ。
- 測定は乱数・optimizer state・本来の勾配を変更しない。
- BF16 を対象に含め、GradScaler の必要な FP16 は対象から外す。

## 主要な概念

- [勾配ノイズとbatch](../../concepts/phenomena/gradient_noise_batch.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
