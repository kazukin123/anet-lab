---
title: net.config_profile によるブロック群パラメータ補間機構
page_kind: サマリ
material: "[[docs/memo/done/0029_config_profile_param_interp_10rfd.md|0029_config_profile_param_interp_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
updated: 2026-11-26
---
# net.config_profile によるブロック群パラメータ補間機構

## 元の材料

[net.config_profile によるブロック群パラメータ補間機構](../../../memo/done/0029_config_profile_param_interp_10rfd.md) blob:91bbc4f3df2a3d2ff045ac2314c38ce858bf7820

## 要点

- ブロックの深さに応じて DropPath などの値を変える場合、従来は出現箇所ごとに設定を書き分ける必要があった。
- net.config_profile で同じグループのブロックに増減の規則を指定する。
- NetworkStructBuilder がブロックを出現順に数えて、構築時に値を展開する。
- グループ内の通し番号と総数を用い、複数のステージにまたがる深さも表せる。
- 同じブロック定義を再利用しつつ、構築後の各ブロックに異なる値を配る。

## 内容

### Context（背景・目的）

anet-lab の NN は宣言的 DSL（`net.block` で部品定義、`net.branch` で組み立て順）で記述する。ここで ConvNeXt の droppath 線形補間のように「ブロック群に沿ってパラメータを段階的に変える」ケースを表現しようとすると、現状は各ブロックインスタンスを個別定義して値を手書きするしかない（18ブロックなら18個）。
anet-lab の宣言的 DSL に、この「スカラー1個の補間ポリシー + 構築時の自動展開」を持ち込む。実装の要は `NetworkStructBuilder`（DSL を解釈してブロック列を生成する層、PyTorch でいうモデル構築コードに相当）である。

### 確定した設計判断

`$` との棲み分け: 将来構想の `$`（config レイヤーでの静的変数置換）とは記号で分ける。config_profile は「NN構造依存の動的計算（出現順 `i/N`）」であり、`$` の静的置換とは本質的に別機構。同じ記号に乗せず、別記号 `@` にするのが設計的に正しい。
汎用機構（任意の double フィールド × 補間ルール）として設計するが、初手の実証は CNBlock droppath のみ。`type=linear` のみ実装する。

### 前提事実

基準コミット: HEAD `0598b1b`。`nn.hpp` / `nn_impl.cpp` / `config.hpp` / `nn_test.cpp` は未コミット変更なし。

### 設計方針

読み取り後は `ConfigProfile` interface へ object 化し、`ValidateConfig()` で型固有の必須項目を検証する。初期実装では `LinearConfigProfile` のみを作る。

### 受け入れ基準

マーカーを含まない既存 branch（ResNet18ish/ViT 等）が無影響で回帰なし（パス1でマーカー検出0なら従来と完全同一のブロック生成になること）。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [ImageCls](../../concepts/subjects/imagecls.md)
