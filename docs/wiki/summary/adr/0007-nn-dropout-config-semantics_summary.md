---
title: NN dropout 設定名と `Drop.p` 改名
page_kind: サマリ
material: "[[docs/adr/0007-nn-dropout-config-semantics.md|0007-nn-dropout-config-semantics]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-28
---
# NN dropout 設定名と `Drop.p` 改名

## 元の材料

[ADR 0007](../../../adr/0007-nn-dropout-config-semantics.md) blob:b5e5f6d197f6c39457a6728e895725c3ebdcf96a

## 要点

- dropout は適用する位置と種類が設定名から区別できるようにする。
- ResBlock の res.dropout_rate はチャネル単位の Dropout2d を指す。
- 構造ブロック Drop の dropout_rate は要素単位の dropout を指す。
- Transformer では hidden、attention weights、stochastic depth の各確率を別に持つ。
- 旧 Drop.p を dropout_rate に改め、互換の読み替えは設けない。
- 旧キーだけが残った設定では Drop は既定の dropout_rate=0.0 で動くため、管理中の設定も移す。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
