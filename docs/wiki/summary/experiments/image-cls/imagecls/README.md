---
title: ImageClsAgent / ImageCls 探索記録
page_kind: サマリ
material: "[[docs/experiments/image-cls/imagecls/README.md|image-cls/imagecls/README]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
updated: 2026-11-26
---
# ImageClsAgent / ImageCls 探索記録

## 元の材料

[ImageClsAgent / ImageCls 探索記録](../../../../../experiments/image-cls/imagecls/README.md) blob:8a06960353efd94d7b2515c91183bacee9b6904c

## 要点

- ImageCls 環境で Food-101 を学習した Run の索引である。
- 2026-06-25〜07-09 のスクラッチ学習 campaign は完了と記録されている。
- 比較対象には Body と ConvNeXt の stem・batch がある。
- 過学習対策、学習率の cosine 制御、Mixup / CutMix を同じ campaign からたどれる。
- 学習予算と BF16 の比較条件・結果は campaign 本文にある。

## 主要な概念

- [ImageCls](../../../../concepts/subjects/imagecls.md)
