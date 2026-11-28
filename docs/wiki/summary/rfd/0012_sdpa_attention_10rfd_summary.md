---
title: SDPA / FlashAttention 化（`CustomTransformerEncoderLayer` の self-attention 経路差し替え）仕様書
page_kind: サマリ
material: "[[docs/memo/done/0012_sdpa_attention_10rfd.md|0012_sdpa_attention_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/313_profiling.md|313 プロファイリング]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# SDPA / FlashAttention 化（`CustomTransformerEncoderLayer` の self-attention 経路差し替え）仕様書

## 元の材料

[SDPA / FlashAttention 化（`CustomTransformerEncoderLayer` の self-attention 経路差し替え）仕様書](../../../memo/done/0012_sdpa_attention_10rfd.md) blob:e0e370089fe4ceef26bec9156d471fa2354c8d99

## 要点

- Transformer の自己注意で、LibTorch の MultiheadAttention が多数の演算に分かれていた。
- CustomTransformerEncoderLayer の自己注意を scaled_dot_product_attention で計算できるようにする。
- 既存 submodule の射影重みを利用し、チェックポイントのパラメータ名と初期化を保つ。
- Pre-LN と Post-LN の双方で同じ自己注意の意味を保つ。
- tf.use_sdpa で旧経路と切り替え、同じ重みで結果と実行時間を比べる。

## 内容

### Context（背景・目的）

Tracy / NSight(launch_blocking=OFF) では CPU 側 NVTX が backpressure 待ちを吸って見かけが歪む（`attn_norm` が 1.273ms に膨張する等）。
`torch::equal` 同期: C++ 版 `multi_head_attention_forward` は self-attention 判定を `torch::equal(query,key) && torch::equal(key,value)`（`activation.h:683`）で行う。Python の `query is key` ポインタ恒等短絡が C++ には無いため、同一テンソルを 3 つ渡しても eq カーネル + `all()` リダクション + `.item()` D2H が走る。1 層 2 回 ×4 層＝8 回/forward の冗長カーネル兼同期点。

### 設計方針

A/B 計測と安全な切り戻しのため config フラグ `tf.use_sdpa`（既定 true）で旧 nn::MHA 経路と切替可能にする（FusedAdamW の `use_fused_optimizer` 前例に倣う）。submodule はどちらのモードでも生成・登録されるため、フラグはチェックポイントに影響しない。

### 外部仕様（config 追加）

未指定時は `ANET_READ_CONFIG` が構造体既定値 `true` を採用（既存読み込みフローと同じ）。

### 検証方針

CUDA 経路: `torch::cuda::is_available()` 時、CUDA 上でも 1.〜2. を実施（half 精度での flash 経路を含めるなら autocast 下でも実行）。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
