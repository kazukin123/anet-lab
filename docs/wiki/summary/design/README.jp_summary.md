---
title: ANET ドキュメント
page_kind: サマリ
material: "[[docs/design/README.jp.md|README.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/framework/326_ppo.md|326 Agent個別実装 PPO系列]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-28
---
# ANET ドキュメント

## 元の材料

[ANET ドキュメント](../../../design/README.jp.md) blob:66411560a61cbe444474f1b3b40e7022aa538295

## 要点

- 設計文書の入口から、ANET の全体像、利用手順、共通機能、具象実装の資料を引ける。
- 0xx は概要・Run の利用と分析・開発環境のガイドを扱う。
- 1xx は実行基盤、Agent、方策ベースの学習の共通部品、Env、Network、可観測性など共通の設計を扱う。
- 2xx は DQN、PPO、Metrics Viewer など具象実装の仕様を扱う。
- 日本語版には .jp.md、対応する英語版には同じ番号・基底名と .en.md を使う。

## 内容

### 3. ドキュメント一覧

- 文書番号は 10 刻みを基本とし、後から文書を挿入できる余地を残します。`0xx` は全体概要・利用・開発環境、`1xx` はフレームワーク共通設計、`2xx` は具象実装の仕様を表します。ファイル名の `.jp.md` は日本語版を表し、英語版は同じ番号・基底名の `.en.md` として追加します。
- 機能カテゴリ別設計ガイドに 115「方策ベースの学習の共通部品」（行動の分布、advantage の推定、Head の合成。Agent・フレームワーク開発者向け）がある。
- 具象実装仕様に 230「PPO 系 Agent」（PPOAgent、Network の共有と分離、rollout と更新、設定、メトリクス。PPO 系 Agent 開発者と方策ベースの手法を学ぶ利用者向け）がある。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [RunnerApp](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [326 Agent個別実装 PPO系列](../../concepts/framework/326_ppo.md)
- [311 開発ツール](../../concepts/framework/311_dev_tools.md)
