---
title: "ImageCls 過学習対策②: 最終段 Flatten → Global Average Pooling 2D（GAP2D）"
page_kind: サマリ
material: "[[docs/memo/done/0023_imagecls_gap_head_10rfd.md|0023_imagecls_gap_head_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
updated: 2026-11-26
---
# ImageCls 過学習対策②: 最終段 Flatten → Global Average Pooling 2D（GAP2D）

## 元の材料

[ImageCls 過学習対策②: 最終段 Flatten → Global Average Pooling 2D（GAP2D）](../../../memo/done/0023_imagecls_gap_head_10rfd.md) blob:0281aea43b7fc3d20f39be8413fa4495442a42a9

## 要点

- FoodResNet13 の終端では28×28の特徴を Flatten し、10万次元を超えるベクトルを Linear に渡していた。
- Global Average Pooling で各チャネルの空間平均を取ってから分類に使う。
- 空間方向を1×1に縮約し、全結合層へ渡す入力次元とパラメータ数を抑える。
- Linear の入力幅は従来の構築処理で推論する。
- 任意の出力サイズや注意による pooling は、この過学習対策の対象から外した。

## 内容

### 分類ヘッドの入力とパラメータ数

FoodResNet13 の最終特徴は128チャネル、各28×28画素で、Flatten 後の分類層は100,352入力×101クラス、約1,014万の重みを持っていた。GAP2D でチャネルごとに空間平均を取ると128入力×101クラス、約1.3万の重みになる。空間位置ごとに別の重みを学習する余地を減らして、学習精度と評価精度の開きが縮まるかを調べる。

ブロック順は GAP2D、既存の dropout（確率0.2）、LinearOut。dropout が作用するのは平均後の128特徴で、LinearOut は構築時に入力幅を自動推論する。その設定を個別に変更せずに Flatten を置き換えられる。

## 主要な概念

- [ImageCls](../../concepts/subjects/imagecls.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
