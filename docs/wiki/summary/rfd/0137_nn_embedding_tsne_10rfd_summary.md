---
title: NN 内部表現を2次元へ投影する可視化案
page_kind: サマリ
material: "[[docs/memo/frozen/0137_nn_embedding_tsne_10rfd.md|0137_nn_embedding_tsne_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
updated: 2026-11-28
---
# NN 内部表現を2次元へ投影する可視化案

## 元の材料

[RFD0137　NN 内部表現を2次元へ投影する可視化案](../../../memo/frozen/0137_nn_embedding_tsne_10rfd.md) blob:351cf4ede1cdc860c5ee8a3507321b97d949ca50

## 要点

- この可視化案は凍結中である。NN の内部表現を可視化する必要が具体化したときに再開し、方式を選ぶ。
- 学習済み Network が状態をどのようにまとめているかを、t-SNE や PCA の2次元図で比較する案である。
- 入力画像や価値などの補助情報を使って、近くに配置された状態を人が解釈できるようにする。
- 対象の層、取得する標本、次元削減の方式と処理場所を設計上の選択肢として残す。
- 学習中の通常処理に与える負荷と、再現可能な条件での比較を考慮する。

## 内容

### Context / 何を作りたいか

対象は Space Invaders 1 本のみ。学習済み DQN に実時間 2 時間プレイさせて経験した状態を集める。
その集合に t-SNE をかけて 2 次元散布にし、各点を DQN が予測した状態価値 V（= max_a Q(s,a)） で着色（濃赤=高、濃青=低）。

### 分解：4 つの関心と充足状況

t-SNE 図は独立した 4 関心の合成であり、新規実装が必要なのは ③ だけである。

### 問題

埋め込みは回転・スケール・クラスタ配置が実行ごとに変わる。interval で連番出力して動画にすると「動いて見えるが意味がない」図ができる。原典 Fig.4 も学習済みエージェントの 1 枚であって時系列ではない。
Dueling head の場合、head 内部の中間層も「最終隠れ層」の候補になりうる。原典 DQN は Dueling ではないので厳密対応は無い。

### 候補を採る場合の目標契約

既定 OFF。config で明示的に有効化しない限り hot path に一切乗らない。
dump 先は Run フォルダ内（`runs/<name>/embed/`）。Run フォルダだけで完結し、外部に主データを置かない。
`Network` への追加は予約 key 1 本のみ。既存の forward 経路・数値挙動を変えない。

### スコープ外

埋め込みの定量評価（trustworthiness、kNN 一致率など）

### 未調査事項

原典が t-SNE 前に PCA を挟んでいたか（van der Maaten の標準手順ではあるが、DQN 論文 Methods での明記は未確認）

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
