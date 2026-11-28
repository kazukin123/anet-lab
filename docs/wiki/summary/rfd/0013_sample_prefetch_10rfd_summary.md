---
title: Sample+To プリフェッチ（ReplayBuffer サンプリング + H2D を GPU 学習と overlap）仕様書
page_kind: サマリ
material: "[[docs/memo/done/0013_sample_prefetch_10rfd.md|0013_sample_prefetch_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# Sample+To プリフェッチ（ReplayBuffer サンプリング + H2D を GPU 学習と overlap）仕様書

## 元の材料

[Sample+To プリフェッチ（ReplayBuffer サンプリング + H2D を GPU 学習と overlap）仕様書](../../../memo/done/0013_sample_prefetch_10rfd.md) blob:648db2e9c49f0c586ce5b0b333112dc1afe44c8c

## 要点

- プロファイルでは ReplayBuffer の Sample と CPU から GPU への転送が、それぞれ約6.5ms の GPU 待ちを作っていた。
- 現在のバッチを学習する間に、次のバッチのサンプリングと転送を進める。
- 先読みの対象には PER の抽選と経験の収集、ExperienceSamples のデバイス転送を含める。
- 次の学習は転送済みのサンプルを受け取り、準備が済むまで待つ。
- 学習処理と転送に重なりが生じるかを、直列実行とプロファイルで比較する。

## 内容

### Context（背景・目的）

`ExperienceSamples::To`（device 転送）~6.5ms/step：CPU→GPU の H2D。現状 `non_blocking=false`（同期）かつ default stream 固定のため非同期化されておらず、Sample と同程度に遅い。
スレッド化は ReplayBuffer 公開インターフェース単位（`Sample()` 丸ごと）+ `To()` をセットで。Learner は内部に手を突っ込まない。スレッド安全性は ReplayBuffer の責務として内部カプセル化。

### 設計方針

`use_rb_prefetch=false` 時は現行の同期 Sample+To 経路（A/B・切り戻し用）。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
