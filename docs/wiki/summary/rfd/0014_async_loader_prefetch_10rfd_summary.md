---
title: PrefetchingReplayBuffer リファクタ（Learner 埋め込み prefetch を decorator へ分離 + 再現性修正）仕様書
page_kind: サマリ
material: "[[docs/memo/done/0014_async_loader_prefetch_10rfd.md|0014_async_loader_prefetch_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
updated: 2026-11-28
---
# PrefetchingReplayBuffer リファクタ（Learner 埋め込み prefetch を decorator へ分離 + 再現性修正）仕様書

## 元の材料

[PrefetchingReplayBuffer リファクタ（Learner 埋め込み prefetch を decorator へ分離 + 再現性修正）仕様書](../../../memo/done/0014_async_loader_prefetch_10rfd.md) blob:3fa979c818c186ba5e848e806a4631cab0481fff

## 要点

- Learner の内部にあった Sample とデバイス転送の先読みを、ReplayBuffer を包む PrefetchingReplayBuffer に分ける。
- 先読み後の Push は寿命を確保した入力を FIFO worker に渡す。
- 優先度の更新と次バッチの先読みの順番を決め、同じ seed の学習がスレッドの偶然の順序に依存しないようにする。
- 抽選とデータ取得を呼び出し側と worker に分割する案は採らず、内部実装の追加の境界を避ける。
- 先読みを使う学習経路で同じ入力からの再現と処理の重なりを確かめる。

## 内容

### Context（背景・目的）

prefetch を 共通 decorator `PrefetchingReplayBuffer`（`ReplayBuffer` を wrap し、`ReplayBuffer` インターフェースを継承して提供する）に分離。`Learner` を薄くする。
overlap（`Sample+To` prefetch）と速度（8.8→7.3h 近傍）は維持。

### 前提事実

注：013 ADR は「Extract はロックフリー（リング距離で安全）」としていたがこれは誤り（最古 valid スロットは write cursor 直前で次 Push で上書きされ得る）。現実装は storage shared/exclusive lock で Push⇄Extract を排他しており正しい。本リファクタでもこの lock を維持する。
`RainbowAgentConfig` は現在 `learner.use_rb_prefetch = false` に固定しており、今回も現状維持する。

### 設計方針

prefetch 状態（future, pending）は `Sample`(const) から触るため `mutable`（既存 mutable mutex と同様）。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
