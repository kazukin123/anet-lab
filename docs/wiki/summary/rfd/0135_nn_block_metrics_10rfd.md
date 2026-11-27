---
title: NN ブロック別の勾配と更新量を測る案
page_kind: サマリ
material: "[[docs/memo/0135_nn_block_metrics_10rfd.md|0135_nn_block_metrics_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-26
---
# NN ブロック別の勾配と更新量を測る案

## 元の材料

[RFD0135　NN ブロック別の勾配と更新量を測る案](../../../memo/0135_nn_block_metrics_10rfd.md) blob:f9dbea5674649f13ad0c920a74375ffdee45a185

## 要点

- 背景・目的・基本方針までが合意範囲である。実際の記録経路、CPU 同期や重みの更新量の取り方は未調査だった。
- 全体の grad_norm だけではどの層が学習を支配しているかを特定できない。
- 最初の候補としてブロックごとの勾配ノルム、全体に対する寄与率、重みノルム、更新比率を測る。
- 絶対的な勾配ノルムはパラメータ数にも依存するため、重みに対する更新量の比で層どうしを比べる。
- 最初の計測を backward 後のパラメータ走査にまとめ、forward の通常経路への負荷を避ける。

## 内容

### Context

現状、NN 分析は grad_norm など 全体集約スカラーで傾向は掴めるが、内部のブロック単位の挙動が見えない。
過去に簡易コードで確認したところ、grad_norm の大半が TransformerEncoder の FFN 部由来だった（観察止まり。定量・定常監視はされていない）。

### 基本方針

キーをフルパスにするのは、encoder の `linear` と head の `linear` のような 文脈衝突をパターンで分離するため（末端名だけでは不可）。

### 前提事実

調査途中でツール結果に混線が発生したため、行番号が未確認のもの（online/target net の保持場所等）は本書から除外し、次段階で取り直す。

### これから詰める論点

CPU 同期: 現状 grad_norm は optimizer step 後まで `.item()` を遅延。ブロック分割で同期点が N 倍にならないよう、全 group の norm を 1 tensor に stack して 1 回で吸い上げる設計が要る。

### 後続

番号 999 はバックログ仮番号。確定時に正式採番する。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
