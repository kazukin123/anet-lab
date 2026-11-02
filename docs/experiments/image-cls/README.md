# ImageClsAgent 探索記録

ImageClsAgent 系の探索記録です。Body の構成、LR の制御、Mixup/CutMix、精度（BF16）など、Agent 側の条件を軸に整理します。


## 知見のまとめ

- [wikiの知見](../../wiki/subjects/imagecls.md)から題材・機構と正本をたどる。

## Env 別記録

| Env | 記録 | 概要 |
|---|---|---|
| ImageCls | [ImageCls 探索記録](imagecls/README.md) | Food-101 のスクラッチ学習。過学習対策（データ拡張・GAP）、Body の比較（ResNet・ConvNeXt・Hybrid・ViT）、ConvNeXt の stem と batch、cosine、Mixup/CutMix、BF16 |
