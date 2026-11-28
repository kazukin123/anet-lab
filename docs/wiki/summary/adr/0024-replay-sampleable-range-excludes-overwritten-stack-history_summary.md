---
title: ReplayBufferのsample可能範囲は、ring上書きで失われたstack履歴を必要とするtransitionを除外する
page_kind: サマリ
material: "[[docs/adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md|0024-replay-sampleable-range-excludes-overwritten-stack-history]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
updated: 2026-11-28
---
# ReplayBufferのsample可能範囲は、ring上書きで失われたstack履歴を必要とするtransitionを除外する

## 元の材料

[ADR 0024](../../../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md) blob:ba3d00cd53bd31b5c2a2659105a105e67a9a9b0d

## 要点

- リングの上書き後、最古の経験付近で stack に必要な過去フレームが失われる。
- 未来側の確定済み範囲 ready と、過去フレームもそろう sampleable を区別する。
- 折り返した後は各 lane の下限から stack_count − 1 件を抽選候補から除く。
- episode の開始時の padding は維持し、上書きで失われた履歴には使わない。
- 均一抽選、PER、Size、可視化は同じ候補集合を参照し、初期優先度の確定と eviction 統計は ready を使う。

## 内容

### 抽選の下限

wrap 前は履歴用の除外幅を設けず、起動直後や実エピソード先頭での padding を保つ。wrap 後に上書きした frame を未来から読み戻すことは、正常な episode 境界の padding と区別する。

### 統計との境界

追い出す経験は履歴を失ってから実際に上書きされる。eviction 統計まで sampleable を基準にすると wrap 後に数えられないため、未来側が確定した ready を使う。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
