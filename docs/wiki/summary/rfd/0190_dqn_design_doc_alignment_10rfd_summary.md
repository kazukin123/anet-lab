---
title: DQN 系 Agent の設計文書を現行実装に合わせる
page_kind: サマリ
material: "[[docs/memo/0190_dqn_design_doc_alignment_10rfd.md|0190_dqn_design_doc_alignment_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-28
---
# DQN 系 Agent の設計文書を現行実装に合わせる

## 元の材料

[RFD0190　DQN 系 Agent の設計文書を現行実装に合わせる](../../../memo/0190_dqn_design_doc_alignment_10rfd.md) blob:75fc4e043159ed155448fff929143ca468075bab

## 要点

- 設計文書200の監査で、設定キー・用語・参照先と実際のコードの違いが見つかった。
- RFD0182 による配置変更後のソース、出荷設定、用語集、ADR を参照して記述を直す。
- Munchausen の選択、メトリクスのプロファイル、DQN 設定型などを現用の名前と対応づける。
- 文書上の型名 `TauGenerator` を実在する自由関数 `GenerateTaus` に合わせ、コードのエラー文・引数検証関数・テスト名にも同じ表記を使う。
- 英訳は訳元の blob を使って専用の翻訳スキルで更新する。今回の修正範囲は和文である。

## 内容

### 監査の対象

DQN 系の設計文書には、廃止された型名、存在しないメトリクスのプロファイル、ソースを再配置する前の参照先が残っていた。読者が設定例やエラー表示から実装を探せるよう、和文の設計文書と関連する ADR の注記を直す。

### 和文の説明と実装の対応

設計200のコードマップから DQN 設定型、内的報酬、フレームスタックの実装を引けるようにする。`run.@munch`、`metrics.scalar.@full` など、設定に実在する綴りを使う。スキップ長、Bellman ターゲット、時間で変わる τ は用語集に合わせる。

### 設計判断への案内

DQN の判断を扱う ADR は該当する設計の節と対応させ、価値分解の二つの ADR は詳細を持つ設計214に載せる。後から変わった前提は元の ADR の判断を残して注記する。

### コードの範囲と受入

τ の生成関数の名前に合わせ、エラー文4か所、引数検証関数の定義と呼び出し、テスト名14本を変更対象に含める。旧名がソースに残らず、τ の既存テストが通ることを確かめる。英訳の更新は、和文を直した後に開発者が翻訳スキルを起動して行う。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
