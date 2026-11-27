---
title: Spectral Normalization は自前 power iteration + u/v named buffer で実装し、σ は毎 forward その場の W で再計算する
page_kind: サマリ
material: "[[docs/adr/0032-spectral-norm-self-impl-buffer-semantics.md|0032-spectral-norm-self-impl-buffer-semantics]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/methods/target_update.md|Target更新の比較]]"
updated: 2026-11-26
---
# Spectral Normalization は自前 power iteration + u/v named buffer で実装し、σ は毎 forward その場の W で再計算する

## 元の材料

[ADR 0032](../../../adr/0032-spectral-norm-self-impl-buffer-semantics.md) blob:b78dfbd4759684c0e8fc6977c34a514df9698385

## 要点

- Spectral Normalization は重みの最大特異値を power iteration で近似する。
- 反復に使う u/v を named buffer に置き、使用時の重みと組み合わせて σ を毎 forward 計算する。
- 各ブロックは共通設定で無効、spectral、spectral_cap を選べる。
- 推定に使う乱数を専用化し、初期化の再現性を維持する。
- 無効時には従来の学習系列と checkpoint の一致を検証する。

## 内容

### 正規化の意味

通常の spectral は W を推定特異値で割るため、初期の重みがゼロでは定義できない。spectral_cap は分母を max(1, abs(σ)) とし、小さな重みをそのまま使える。zero-init を使う ResBlock などの条件で両者が異なる。

### 重みと状態

正規化済み重みを永続化する形では、soft update や異なる用途の forward 後に古い推定値を使う危険がある。保存するのは u/v で、σ は使用時の W から取り直す。反復のノルムが小さい場合はベクトルを更新せず、zero-init で反復がゼロのまま固定されるのを避ける。

### 確認と適用外

構築時にはモードに応じて σ の非有限・退化を検出する。power iteration は推定であり、上限法が常に厳密なスペクトル境界を保証するわけではない。層ごとの上書きや反復回数の追加設定は先送りした。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [可塑性指標と成績](../../concepts/phenomena/plasticity.md)
- [Target更新の比較](../../concepts/methods/target_update.md)
