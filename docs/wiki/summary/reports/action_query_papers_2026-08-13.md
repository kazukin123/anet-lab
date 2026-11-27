---
title: Action Query 関連論文サーベイ
page_kind: サマリ
material: "[[reports/action_query_papers_2026-08-13.md|action_query_papers_2026-08-13]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/action_query_attention.md|Action queryとattentionの証拠]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
updated: 2026-11-26
---
# Action Query 関連論文サーベイ

## 元の材料

[Action Query 関連論文サーベイ](../../../../reports/action_query_papers_2026-08-13.md) blob:7cdf6b745c861545ac9d5eda92fda0b8787d83d9

## 要点

- AQT は離散行動ごとに query を作り、画像の空間特徴を cross-attention で読む。
- 実験では Rainbow を土台に 51 atoms を用いる。decoder 出力に同じ小さな advantage head を各 action へ適用するため、巨大な `flattened_state → actions × atoms` 全結合とは異なる。ただし value branch は encoder 全体を flatten しており、AQT 自体も flatten を完全には排除していない。また Atari で AQT 単体が常に baseline を上回ったわけではなく、ゲームによっては改善が確認されなかった。
- Query2Label のように出力に固定の意味がある query と、DETR の object query は対応付けが違う。
- Pooling by Multihead Attention（PMA）は、`k` 個の learned seed vectors から入力集合へ attention し、`k` 個の出力を作る。
- ACT の query は未来の行動時刻を表し、離散行動ごとの Q 値を読む AQT と出力の意味が異なる。

## 内容

### 結論

一方、AQT は C51 系であり、`action feature × tau feature → quantile value` という IQN 結合は扱わない。DropMerge で検討している `[B,59,D] × [B,K,D] → [B,59,K]` は、AQT の action-query decoder を IQN へ拡張する設計になる。

### 論文一覧

引用数は OpenAlex 等の索引を 2026-08-13 に取得した参考値である。プレプリントと出版版の重複、索引差があるため影響度の概算にのみ使う。

### 1. Action Q-Transformer

DropMerge へそのまま移せない点は、AQT が `hidden_dim = action_count × 32` としていること、C51 であって QR/IQN ではないこと、提案目的が主として attention の可視化であることである。

### 3. Perceiver IO

Perceiver IO は、望む出力要素ごとに query を一つ作り、query 自身にその出力の意味を持たせる一般形を提示する。

## 主要な概念

- [Action queryとattentionの証拠](../../concepts/methods/action_query_attention.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
