---
title: 個体メトリクスは trace チャネルで出し、評価セッションは採用 episode ごとの EpisodeEndEvent と SessionEndEvent を分けて発火する
page_kind: サマリ
material: "[[docs/adr/0037-metrics-trace-channel-and-session-end-event.md|0037-metrics-trace-channel-and-session-end-event]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
updated: 2026-11-26
---
# 個体メトリクスは trace チャネルで出し、評価セッションは採用 episode ごとの EpisodeEndEvent と SessionEndEvent を分けて発火する

## 元の材料

[ADR 0037](../../../adr/0037-metrics-trace-channel-and-session-end-event.md) blob:341c051a17d8e06e913bcb999392bb842feccac5

## 要点

- 集約 scalar では各エピソードの score と長さの組を復元できない。
- 個体を識別する trace を設け、1件に1つの記録を対応させる。
- 評価では採用エピソードごとに EpisodeEndEvent、集約確定時に SessionEndEvent を出す。
- 評価の集約 scalar はセッション末へ、個体の trace はエピソード末へ結び付ける。
- 後の変更ではエピソード末の値を終了した Step の結果から読み、Env の最新値に依存しない。

## 内容

### 個体の記録

scalar の step に lane を埋め込むと整数軸や集約が壊れる。trace では lane と各値を同じレコードにまとめ、スコア・長さなど同時に起きた情報を保持する。

### イベントと保存

設定した N 本のエピソードを待つ場合も各個体の終了を別々に通知し、最後にセッション集約を確定する。個体値の列をデコレーターに蓄積して終了時に一括で取り出す口は設けない。

### 解析と回帰

定義レコードは scalar と trace を別名で出し、解析側は既存 Run の旧名も読めるようにする。値が既知でも非有限なら列を残して null とし、読めないキーとは区別する。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
