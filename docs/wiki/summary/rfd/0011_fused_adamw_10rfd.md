---
title: FusedAdamW（ATen `_fused_adamw_` 直接呼び出し）導入 仕様書
page_kind: サマリ
material: "[[docs/memo/done/0011_fused_adamw_10rfd.md|0011_fused_adamw_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-26
---
# FusedAdamW（ATen `_fused_adamw_` 直接呼び出し）導入 仕様書

## 元の材料

[FusedAdamW（ATen `_fused_adamw_` 直接呼び出し）導入 仕様書](../../../memo/done/0011_fused_adamw_10rfd.md) blob:52239363dbbaa9ad23a521d4612b1b75cc53cbff

## 要点

- 小型 MLP では AdamW のパラメータごとの CUDA カーネル起動が学習時間を占めていた。
- LibTorch C++ API に fused 指定がないため、利用中の ATen の fused AdamW 演算を呼ぶ。
- CPU と CUDA のディスパッチを持つ演算を、Optimizer の更新経路に組み込む。
- 既存の zero_grad は既定で grad を未定義にしており、勾配の初期化には追加の最適化を入れない。
- 更新結果とプロファイルを比較し、計算の差とカーネル起動数の差を分けて確かめる。

## 内容

### Context（背景・目的）

DQN Learner の `optimizer_->step()` が学習ループのボトルネック。Tracy 実測で optimizer_step 区間が支配的、CUDA_LAUNCH_BLOCKING 実測でカーネル起動律速と確認済み。LunarLander 級の小型 MLP では param テンソル ~10 個 × per-param 約 12 演算 ≈ 毎 step 100 回超のカーネル起動が発生する。
zero_grad は対象外と判明: LibTorch 2.11 では `Optimizer::zero_grad(bool set_to_none = true)` がデフォルト true で、既存の `optimizer_->zero_grad()` は既にカーネル起動ゼロ（grad を undefined にリセットするだけ）。変更しない。

### 外部仕様（config 追加）

Rainbow は `RainbowAgentConfig` が同じ `LearnerConfig` を使う場合のみ同様に読み込みを追加（TBO の前例にならい、公開しない判断でもよい。最小変更を優先）。

### 検証方針

実測（ユーザー実施）: LunarLander run で Tracy の optimizer_step 区間と steps/sec を前後比較。`use_fused_optimizer = false` で従来挙動に戻ることも確認。

### Out of Scope

`grad_scale` / `found_inf` の fused カーネル直渡し（grad clip 使用時に利得が消えるため。ADR 参照）。
foreach 手組みの AdamW 実装（fused 一本。ADR 参照）。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
