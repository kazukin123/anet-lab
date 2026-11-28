---
title: 観測メトリクスを目標値に合わせるハイパラ調整案
page_kind: サマリ
material: "[[docs/memo/0152_metric_targeted_hyperparam_control_10rfd.md|0152_metric_targeted_hyperparam_control_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# 観測メトリクスを目標値に合わせるハイパラ調整案

## 元の材料

[RFD0152　観測メトリクスを目標値に合わせるハイパラ調整案](../../../memo/0152_metric_targeted_hyperparam_control_10rfd.md) blob:2f48e4c853842b32abb5eafc9768f3b981682efa

## 要点

- メトリクスの目標値に合わせた調整は暫定案である。どの指標と設定を組み合わせるかは未決である。
- 初期優先度を変えた Atari 比較で、未抽選で追い出された遷移の割合が変化し、壁突破率にも差が出た。
- 制御に使うメトリクスは成績との対応や、設定変更に対する動き方を実測で確かめる。
- タグ・設定キー・目標値・ゲイン・発火間隔を指定し、遅い積分制御で値を更新する案である。
- 正の優先度は対数空間で調整し、観測値が追従する時間より十分に長い間隔で更新する。

## 内容

### Context / Problem Statement

ハイパラはすべて開ループの定数である。値は人間が別環境・別予算で得た経験から選び、Run 中は動かない。以下 4 点はすべて実測事象である。
「最小化すればよい」ではない。 内部最適があるということは、上限や下限へ張り付ければ済む問題ではなく、制御が意味を持つということでもある。

### 効く事実

観測量は既に出ている。 `39_agent_per` / `37_agent_qtd` / `34_agent_plasticity` / `35_agent_churn` 群は Observer 経由で毎 update 取得できる。新規計測は不要。

### 前提として先に要る作業

制御を書く前に、設定値を横に振って `(観測量, 成績)` の対応を取る較正キャンペーンが要る。これは既存の群振り運用（`apps/1x_batch_run*.bat`）でそのまま実施できる。この較正が済むまで制御則は書けない。

### 提案

`(観測量タグ, 設定キー, 目標値, ゲイン, 発火間隔)` を config で宣言し、Learner が遅い積分制御で設定値を更新する。
制御は log 空間で行う（優先度は正の量でスケールが数桁動くため）。発火間隔は観測量自身の時定数より十分遅くする。

### 制御観測量の選定基準

`probe_*` 群が落ちるのは ④ より ① による。決定打は活性関数の最小差での対照で、LeakyReLU 群と ReLU 群は churn・系統差率・target 不一致が区別できないのに壁突破率が 1.4 倍違った（2026-09-03、実験記録）。差が出ることと成績を予測することは別である。

### `soft_update_tau` を制御対象に加える根拠

成績順（0.008 > hard > 0.001）と単調に対応するのは `13_tgtΔ`（逆順）・`11_q_max`（順）・`tgtΔ/q_gap`（逆順）の 3 つだけで、`01_churn`・`11_target_disagreement`・`41_probe_dormant_ratio`・`17_q_gap` 単独は対応しない。

### `probe_*` 群を制御対象から外す根拠

可塑性指標が成績を予測しないことは、本キャンペーンで 6 例が確定している（SN 群 / VA+pool6 / fuse2304 / hard125 / btrstruct / LeakyReLU）。文献側も同方向で、可塑性喪失は単一の測定量に帰着できないとされ、dormant neuron が駆動因か症状かは未決とされている。条件 ① を満たさないので目標値を置けない。

### 候補を採る場合の受入条件

較正が先: `per_initial_priority` を 3 点以上振り、`07_evicted_unsampled_ratio` と壁突破率の対応が単峰であることを実測する。これが取れないと目標値を置けないので最初に確認する。

### スコープ外

`per_initial_priority` 以外の対の実装。候補表は挙げるが、較正が済んでいるのは 1 対もない。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
