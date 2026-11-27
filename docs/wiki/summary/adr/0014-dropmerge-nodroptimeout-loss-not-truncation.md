---
title: DropMergeのNoDropTimeoutをtruncationではなく敗着(done+罰)とする
page_kind: サマリ
material: "[[docs/adr/0014-dropmerge-nodroptimeout-loss-not-truncation.md|0014-dropmerge-nodroptimeout-loss-not-truncation]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
updated: 2026-11-26
---
# DropMergeのNoDropTimeoutをtruncationではなく敗着(done+罰)とする

## 元の材料

[ADR 0014](../../../adr/0014-dropmerge-nodroptimeout-loss-not-truncation.md) blob:647e8556c463dc3af36ab7089fa595de148c1237

## 要点

- ショットクロックは NOOP を選ぶと進み、残量は観測に含まれる。
- 時間切れは行動の帰結なので、打ち切りではなく罰を伴う敗着とする。
- 罰がゼロだと危険な盤面で合法な DROP を避ける誘因が生じる。
- 打ち切りへの切り替えは観測のスカラー次元も変え、比較条件が混ざる。
- 罰の初期値は game_over_penalty と同じ −10 を起点とし、両者を独立に調整できる。
- 罰を有効にする際は、真に詰んだ盤面を NoLegalDrop として確実に受理する仕組みも用意する。正しい投了まで時間切れの敗着にしてしまうのを防ぐ。

## 主要な概念

- [DropMerge](../../concepts/subjects/dropmerge.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
