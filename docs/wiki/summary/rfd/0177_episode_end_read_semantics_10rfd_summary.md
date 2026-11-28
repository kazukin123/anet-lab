---
title: エピソード終了時に読むメトリクスの対象
page_kind: サマリ
material: "[[docs/memo/0177_episode_end_read_semantics_10rfd.md|0177_episode_end_read_semantics_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# エピソード終了時に読むメトリクスの対象

## 元の材料

[RFD0177　エピソード終了時に読むメトリクスの対象](../../../memo/0177_episode_end_read_semantics_10rfd.md) blob:4532a7dd66c5cb36969550586ad6ca3b55f78e7d

## 要点

- エピソード終了時に読む値の時点をそろえる暫定案で、対象の広さや移行方法は再グリルで決める。
- EpisodeEndEvent は終了 lane ごとに発行されるが、現在状態を返す Env は自動リセット後の値を返す場合がある。
- 同時に複数 lane が終わるとき、Runner の集計がどの終了分までを含むかも明確にする必要がある。
- 暫定案では終了時に現在状態を読む指定を構築時に検出し、遷移結果は env_step から読むようにする。
- Runner の集計は同じ step に終わったエピソードをすべて加えてから通知する案を扱う。
- 現用設定で問題の宣言があるかと、後から不整合を生じうる契約とを区別して検証する。

## 内容

### 目的と問題

今のリポジトリの設定には、どちらに当たる宣言も無い（`@episode_end` の scalar は 0 行。`@episode_end` の trace は Atari の 4 行で、097 が `$env_step` へ移す）。実害は今は無いが、宣言すると黙って違う値になる。
`@episode_end` で読む値が、どの target でも、終わったエピソード（その Step）の値になる。そうならない読み方は、構築のときに止まる。

### 候補を採る場合の成功の測り方の目安

単体テストで、同じ step に 2 lane が終わるとき、`$runner mean.episode_return @episode_end` が 2 本の平均を 1 回だけ記録し、EMA を 1 回だけ更新する。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
