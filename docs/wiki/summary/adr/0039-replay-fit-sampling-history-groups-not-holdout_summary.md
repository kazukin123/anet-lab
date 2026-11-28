---
title: replay 当てはまり診断は held-out 分割ではなく既存の抽選履歴フラグで群分けし、評価条件を学習条件から独立させる
page_kind: サマリ
material: "[[docs/adr/0039-replay-fit-sampling-history-groups-not-holdout.md|0039-replay-fit-sampling-history-groups-not-holdout]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# replay 当てはまり診断は held-out 分割ではなく既存の抽選履歴フラグで群分けし、評価条件を学習条件から独立させる

## 元の材料

[ADR 0039](../../../adr/0039-replay-fit-sampling-history-groups-not-holdout.md) blob:d753d4f2b57692f00aade94a18584322865d76f2

## 要点

- 学習済み replay への当てはまりは、抽選されたことがある群と未抽選群で測る。
- 既存の履歴フラグを使い、診断のために遷移を学習対象から外さない。
- 両群と実際の PER バッチを同じ FP32・eval・固定分位点・target 式で比較する。
- 母数と平均年齢を記録し、若い遷移の偏りを踏まえて読む。
- 群の TD 平均の比を、そのまま記憶力の差と解釈できない。

## 内容

### 比較の設計

学習から排除した held-out 集合を作ると学習系列そのものが変わる。測定時点で抽選可能な遷移を分類し、未抽選・抽選済み各群と実バッチに同じ target 推定を適用する。

### 読み方

未抽選群には若い遷移が偏ることがある。群平均の差を解釈する際は群の大きさと年齢を併記し、経験の古さを統制した比較と混同しない。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
