# ImageClsAgent 探索記録

## 元の材料

[ImageClsAgent 探索記録](../../../../experiments/image-cls/README.md) blob:7659aa38d5b0b2396682b7f739cc77c4642f8e59

## 要点

- ImageClsAgent の設定を軸に実験を引く索引である。
- 対象は Food-101 のスクラッチ学習である。
- Body は ResNet・ConvNeXt・Hybrid・ViT の比較を含む。
- 過学習に対してデータ拡張・GAP、Mixup / CutMix を検討した記録へ進める。
- ConvNeXt の stem・batch、cosine による学習率制御、BF16 の記録は環境別索引を経由して引く。

## 主要な概念

- [ImageCls](../../../concepts/subjects/imagecls.md)
