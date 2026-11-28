---
title: WeightInitConfig.mode の文字列化
page_kind: サマリ
material: "[[docs/adr/0008-weight-init-mode-string.md|0008-weight-init-mode-string]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-28
---
# WeightInitConfig.mode の文字列化

## 元の材料

[ADR 0008](../../../adr/0008-weight-init-mode-string.md) blob:2b6ea3e16a071dcd4db668b84dd0482bc3583ec4

## 要点

- 重みの初期化は WeightInitConfig.mode で手法名を指定する。
- 従来は5つの手法を整数に対応させていた。
- 切断正規分布を追加すると番号から手法を判別しにくくなる。
- default、xavier、he、orthogonal、constant、trunc_normal を識別する。
- 旧数値の読み替えはせず、不明な値は明示的に失敗させる。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
