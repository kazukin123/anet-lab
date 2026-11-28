---
title: self-attention を ATen `scaled_dot_product_attention` 直接呼び出しに置き換える
page_kind: サマリ
material: "[[docs/adr/0004-sdpa-attention-via-aten.md|0004-sdpa-attention-via-aten]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/313_profiling.md|313 プロファイリング]]"
updated: 2026-11-28
---
# self-attention を ATen `scaled_dot_product_attention` 直接呼び出しに置き換える

## 元の材料

[ADR 0004](../../../adr/0004-sdpa-attention-via-aten.md) blob:b1995c7e29f7f1447bae1890127ae5fe996a8ca1

## 要点

- Transformer の自己注意では、中間注意行列、等値判定の GPU 同期、不要な注意重みの計算が負荷になった。
- 既存の MultiheadAttention に重みを保持し、forward で ATen の SDPA を呼ぶ。
- 専用 Linear に置き換える案と異なり、パラメーター名を維持できる。
- Flash・省メモリ・math の適用経路は dtype などの条件で変わる。
- 演算順に差があるため、厳密な比較や切り戻しでは従来経路を利用する。
- 測定した構成では自己注意が encoder layer の65%、encoder が学習 forward の63%を占めた。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
