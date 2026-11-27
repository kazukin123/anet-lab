---
title: DefaultDQNAgent の Atari 探索記録
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/README.md|default-dqn/atari/README]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
updated: 2026-11-26
---
# DefaultDQNAgent の Atari 探索記録

## 元の材料

[DefaultDQNAgent の Atari 探索記録](../../../../../experiments/default-dqn/atari/README.md) blob:cbfadf8f613c06ee74fcdc77cd9a1230cacbe310

## 要点

- Atari の campaign を時期・主題・状態から引ける。各比較の得点・Run 名・評価条件はリンク先の実験記録にある。
- 実時間優先とサンプル効率優先は別の目的を持つ。BTR NN 構成の RR2 は、どちらの選択にも影響しないため未測定のままとした。
- ViT ハイブリッドは CNN 以外の基本構成として常設し、Breakout の成績だけでは採否を決めない。
- 複製がない比較の小差は順位付けに使わず、判断が必要な比較に限って複製を取る。
- 索引では、完了・継続に加えて、途中停止した Run と次のステージを保留した比較も区別する。

## 内容

### campaign

8月は Breakout / Pong の基準、Atari-5 の横断比較、可塑性の測定、replay ratio と target 更新を扱う。9月は BTR の構成差、環境数と lane 窓、実時間優先・サンプル効率優先の比較へ進む。10月以降は NoisyNet、kung_fu_master のボス、qbert の初期値、TempoRL の学習と評価をゲーム別に引ける。11月の Montezuma と kung_fu_master の記録は、RND の誤差・尺度とゲーム内の出来事を扱う。

### 知見のまとめ

題材をまたぐ結果は Atari の wiki ページから、得点と条件は個々の campaign からたどる。

### 独自の分析

ViT を残す判断、BTR NN で RR2 を追加探索しない判断、小差を判定するときにだけ複製を取る方針を記録した。いずれも探索予算の使い方を定めるもので、未測定の条件に性能の順位は付けていない。

## 主要な概念

- [Atari 横断](../../../../concepts/subjects/atari.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [Replay ratio と lane 窓](../../../../concepts/methods/replay_ratio_lane_window.md)
- [Breakout](../../../../concepts/subjects/breakout.md)
