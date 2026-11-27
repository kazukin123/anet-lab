---
title: NN モジュール dropout / DropPath サポート
page_kind: サマリ
material: "[[docs/memo/done/0024_nn_dropout_droppath_10rfd.md|0024_nn_dropout_droppath_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-26
---
# NN モジュール dropout / DropPath サポート

## 元の材料

[NN モジュール dropout / DropPath サポート](../../../memo/done/0024_nn_dropout_droppath_10rfd.md) blob:1b78cb275875f590217a4593a523ea8f9f610422

## 要点

- Food-101 の汎化を検証するため、ブロック内部の dropout と残差枝の DropPath を設定で変えられるようにする。
- ResBlock は畳み込み間でチャネル単位の dropout を使える。
- TransformerEncoder は注意・FFN・残差の dropout と DropPath を扱う。
- 確率は0以上1未満とし、既定の0では従来のネットワークを保つ。
- 学習・評価の切替に従って発動する。確率を変えると乱数消費も変わるため、再現性は同一設定で比較する。

## 内容

### Context（背景・目的）

DropPath とは: 残差 `out = x + F(x)` の 枝 F(x) のみをサンプル単位で確率的にゼロ化する。skip 接続が残るのでブロック出力は 恒等 x（出力ゼロではない）。これにより「サンプルごとにネットの深さが変わる」= Stochastic Depth となり、深い ResNet/ViT の過学習に効く主力レバー。eval 時は全枝 ON（no-op）。

### 確定した設計判断

設定キー命名: `_rate` 接尾辞で率を示し、`dropout` / `droppath` を各 1 語表記で統一。"2d" は conv 文脈で自明なので付けない。

### 前提事実

`register_module` 経由で登録した submodule に train/eval が再帰伝播するため、新規 dropout も `register_module` すれば自動追従。

### 設計方針

mask 形状は `x.dim()` から動的生成（[N,C,H,W] と [N,S,D] 両対応）。RNG は torch 既定ジェネレータ（`manual_seed` / ADR 0006 `setDeterministicAlgorithms` 尊重）。有効化すると RNG 消費が変わり dropout 無し run とは bit 一致しない（同一 config では再現可能）。
最終 `norm_`（`TransformerEncoderModule` 側）に dropout は入れない。

### 受け入れ基準

ビルド緑（x64-Debug、`anet-core-test`）、既存テスト緑（特に既存 `[transformer][sdpa]` 群が不変）。
既存挙動不変: 全 rate 既定 0.0 のとき、ビルド前と数値・チェックポイント名（`layer_*.self_attn.*` 等）が不変。

## 主要な概念

- [ImageCls](../../concepts/subjects/imagecls.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
