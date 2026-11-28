---
title: "ImageCls 過学習対策①: train 専用データ拡張（水平フリップ + RandomResizedCrop）"
page_kind: サマリ
material: "[[docs/memo/done/0022_imagecls_augmentation_10rfd.md|0022_imagecls_augmentation_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-28
---
# ImageCls 過学習対策①: train 専用データ拡張（水平フリップ + RandomResizedCrop）

## 元の材料

[ImageCls 過学習対策①: train 専用データ拡張（水平フリップ + RandomResizedCrop）](../../../memo/done/0022_imagecls_augmentation_10rfd.md) blob:088978d489ac753b179a4a3f321fe3374c7e39fc

## 要点

- Food-101 では高い学習精度に対して評価精度が低く、入力処理は224×224へのリサイズだけだった。
- 水平反転と RandomResizedCrop を学習画像に適用する。
- 評価に使う画像は従来の処理で読み込み、評価の基準を一定にする。
- 拡張の有効化と強さは ImageClsEnv の設定で調整する。
- 既定を無効とし、同じ学習条件で拡張の有無を検証できるようにする。

## 内容

### 学習時だけ適用する画像拡張

Food-101 では224×224へのリサイズ以外の変換がなく、学習と評価で精度に開きがあった。ImageClsEnv の学習経路で確率0.5の水平反転と RandomResizedCrop 相当の変換を行う。料理の主要部分を切り落としすぎないよう、切り出す面積の割合は0.7〜1.0とし、出力を元の入力サイズに戻す。評価時には同じ画像を従来の方法で読み込む。

### 乱数と構成の比較

画像を供給する ImageDataSource に別の乱数器を持たせると、同じ seed で結果を再現するために二箇所の乱数状態を管理することになる。そこで拡張の抽選には Env 側の乱数を使う。拡張の設定は既定を無効とし、Food-101 の学習用 baseline で有効にして、同じビルドの設定変更だけで比較する。

## 主要な概念

- [ImageCls](../../concepts/subjects/imagecls.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
