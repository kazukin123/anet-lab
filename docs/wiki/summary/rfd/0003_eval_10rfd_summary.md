---
title: EpisodeEvalObserver の Eval メトリクス強化 — 構造分析と修正方針
page_kind: サマリ
material: "[[docs/memo/done/0003_eval_10rfd.md|0003_eval_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# EpisodeEvalObserver の Eval メトリクス強化 — 構造分析と修正方針

## 元の材料

[EpisodeEvalObserver の Eval メトリクス強化 — 構造分析と修正方針](../../../memo/done/0003_eval_10rfd.md) blob:7bf28eec7d6c0574b741b9ca57e649d89503e19d

## 要点

- 評価エピソードの総報酬に加え、Env と Agent の統計を既存の scalar 設定で選べるようにする。
- 従来の EpisodeEvalObserver は評価実行と統計の伝搬を抱え、学習用イベントの経路を共有しにくかった。
- 評価を実行する側とメトリクスを取り出す Observer の責務を分ける。
- 末尾の改訂で、学習と評価を共通の EpisodeEndEvent で通知する方針にそろえた。
- 設定はイベント、Runner の範囲、値を取る対象の3軸を独立に選ぶ。

## 内容

### 評価値をどこで集めるか

旧経路は評価結果を単一の float として運び、評価時の Env や Agent の別の scalar を読む場所がなかった。評価の実行、イベントの通知、値の記録を分ければ、コードを追加しなくても既存のメトリクス設定から記録対象を選べる。

### 最終の方針

当初の案では評価専用イベントと専用タグの構文を設けていた。末尾の改訂で共通の EpisodeEndEvent を採り、同じイベント・Runner の範囲・対象フィールドの組み合わせで学習と評価を記述する形に改めた。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
