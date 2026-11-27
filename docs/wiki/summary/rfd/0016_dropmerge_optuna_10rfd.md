---
title: DropMerge Optuna 対応 仕様書
page_kind: サマリ
material: "[[docs/memo/done/0016_dropmerge_optuna_10rfd.md|0016_dropmerge_optuna_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-26
---
# DropMerge Optuna 対応 仕様書

## 元の材料

[DropMerge Optuna 対応 仕様書](../../../memo/done/0016_dropmerge_optuna_10rfd.md) blob:f87e221e5f914674fc2c36ac6fdd47078ec7ed64

## 要点

- DropMerge の ResNet と ViT の構造を Optuna で比較する。
- 学習条件を固定し、token 数・表現幅・層数などの構造と計算コストを探索する。
- パラメータ数だけでは実行負荷を表せないため、複数の cost_budget で候補を比べる。
- 有望な構成は seed を変えて再評価し、複数 seed の集計値でも比較できる。
- Runner の成果物は手動 Run と分け、Metrics Viewer で候補を横断して読めるようにする。

## 内容

### 目的

DropMerge の ResNet + ViT 系 NN 構成について、個人環境で回せる範囲の Optuna 探索を導入する。

### 背景

`cost_budget` は実時間そのものではなく、Transformer 計算量の事前 proxy である。trial の duration や step/sec は同一 GPU 並列の干渉を受けるため、v1 の primary score には入れない。

### 対象範囲

proxy 探索で backend deterministic や DropMergeEnv seed 固定を使う場合は、生成 config ではなく Optuna 専用 config 側で指定する。

## 主要な概念

- [DropMerge](../../concepts/subjects/dropmerge.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
