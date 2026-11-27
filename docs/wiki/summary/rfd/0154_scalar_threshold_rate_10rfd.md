---
title: スカラーが閾値を超えた割合を測る仕組み
page_kind: サマリ
material: "[[docs/memo/0154_scalar_threshold_rate_10rfd.md|0154_scalar_threshold_rate_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
updated: 2026-11-26
---
# スカラーが閾値を超えた割合を測る仕組み

## 元の材料

[RFD0154　スカラーが閾値を超えた割合を測る仕組み](../../../memo/0154_scalar_threshold_rate_10rfd.md) blob:83bff0ebde9fcfe6df88073a04831f326341c39f

## 要点

- 閾値を超えた割合の集計は暫定案で、比較演算、集計の単位と各部品の受け持ちは未決である。
- Breakout の1画面突破率と DropMerge の成功率に、スカラーと閾値を比較して割合を出す共通点があった。
- ゲームごとに専用実装を増やさずに、元のスカラーと条件から割合を定義する案である。
- 実行時の観測側で集計する方式と、保存済みのデータから Viewer で集計する方式を比較する。
- 集約済みの平均からは個々の結果が閾値を超えたか戻せないため、入力として残す粒度を明示する。

## 内容

### Context / Problem Statement

DropMerge のスイカ到達率。`ep_max_rank` は `max`（`42_env/01`）と `mean`（`61`）しか出しておらず、「Rank 11 以上に到達した割合」が無い。Atari の 432 と同じ性質の指標なのに、閾値がコードに無いので出せない。

### 案

3 層に分ける。Observation 層が本体、Viewer 層と inspect_run は同じ導出を後付けで提供する。
NaN 契約が要点。 `NaN >= N` は false なので素直に比較すると NaN が 0.0 になり、集約の分母が「完了した env 数」ではなく num_envs になって率が壊れる。必ず `std::isnan` を先に見て NaN を透過させる。

### 着手条件

現時点では着手しない。 Atari 側は 2026-09-05 のローカル実装で当面の用が足りている。
DropMerge でスイカ到達率が実際に要るとき。 そこで 3 度目の手書きをするくらいなら案 1 を通す。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Breakout](../../concepts/subjects/breakout.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
