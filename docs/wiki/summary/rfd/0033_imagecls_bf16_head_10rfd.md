---
title: ImageCls BF16(AMP)対応 + 汎用 LinearHead による Head 化
page_kind: サマリ
material: "[[docs/memo/done/0033_imagecls_bf16_head_10rfd.md|0033_imagecls_bf16_head_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-26
---
# ImageCls BF16(AMP)対応 + 汎用 LinearHead による Head 化

## 元の材料

[ImageCls BF16(AMP)対応 + 汎用 LinearHead による Head 化](../../../memo/done/0033_imagecls_bf16_head_10rfd.md) blob:ce2243bfab72c313866691bd7f1746067b9ecde1

## 要点

- ImageCls の GPU 計算が学習時間を占める状況で、BF16 autocast により forward を高速化する。
- ネットワークの body と FP32 の Head の境界を使えるよう、分類の最終射影を汎用 LinearHead に移す。
- クラス数は Env の離散行動数から決め、最終射影の重み初期化は既存の方式を保つ。
- BF16 のマスタースイッチは既定で無効とし、全 FP32 の比較条件を残す。
- DQN 用の既存 Head まで一律に置き換える作業は対象外とする。

## 内容

### 確定した設計判断

汎用 `LinearHead` / `LinearHeadFactory` を `nn_heads`（`anet::nn`）へ追加。
DQN 側 `LinearHeadFactory`（`anet::rl::dqn`, `dqn_based_heads.*`）は一切触らない。namespace が異なり衝突しない。

### 前提事実

`at::autocast::set_autocast_enabled/dtype` の RAII ガード。`~Autocast` で前値へ復元。
`NetworkHead` は `Forward(feature_dict)` のみの単純 IF。
`Network::Clone` は `head_factory_` を保持し clone 時に head を再構築。

### 設計方針

`nn_heads.cpp`: DQN 版 `LinearHead` と等価。差分は 出力キーを `output_key_` に（DQN は "q" 固定）。
重み初期化は `WeightInitializer::Initialize(linear_, init_config)`。
device は actor の `device_`（CPU eval 対応）。softmax/argmax は head 後 FP32 logits で従来どおり。

## 主要な概念

- [ImageCls](../../concepts/subjects/imagecls.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
