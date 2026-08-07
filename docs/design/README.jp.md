# ANET ドキュメント

> 主たる観点: 全体構成（機能単位・行程単位の文書への案内）

## 1. はじめに

### 1.1 目的

この文書は、ANET の利用・分析・開発に関するドキュメントの入口です。フレームワークの概要を短く示し、詳細を扱う文書を一覧化します。

### 1.2 対象読者

- ANET で学習や評価を実行する利用者
- Run の結果を分析する利用者
- ANET 本体、Agent、Env、アプリケーションを変更する開発者

### 1.3 記載範囲

ANET の概要と `docs/design/` 配下の文書インデックスを記載します。設定、操作、実装設計の詳細は、一覧から各文書を参照してください。

## 2. ANET の概要

ANET は、libtorch を基盤とする C++20 の強化学習実験フレームワークです。Env と Agent を設定から組み立て、wxWidgets 製の `AnetRLRunner` で学習と評価を実行します。実行中の状態は GUI で確認でき、Run ごとに記録したメトリクスは Java/Spring 製の Metrics Viewer や補助ツールで分析できます。

現時点で確認している実行環境は Windows 11 x64 と NVIDIA CUDA の組み合わせです。他の OS と CPU-only 構成は、コード上の経路が存在する場合も含めて、フレームワーク全体としては未検証です。全体像と環境要件は [ANET フレームワーク全体概要](010_framework_overview.jp.md)を参照してください。

## 3. ドキュメント一覧

文書番号は 10 刻みを基本とし、後から文書を挿入できる余地を残します。`0xx` は全体概要・利用・開発環境、`1xx` はフレームワーク共通設計、`2xx` は具象実装の仕様を表します。ファイル名の `.jp.md` は日本語版を表し、英語版は同じ番号・基底名の `.en.md` として追加します。

### 3.1 全体概要

| 番号 | 文書 | 主な観点 | 対象読者 |
|---:|---|---|---|
| - | この文書 | 初めに、システム概要、文書インデックス | 全員 |
| 010 | [ANET フレームワーク全体概要](010_framework_overview.jp.md) | 基本概念、全体構成、機能一覧、主要処理フロー | 全員 |

### 3.2 ユーザーガイド

ユーザーガイドは、利用者が行う工程の順に説明します。

| 番号 | 文書 | 主な観点 | 対象読者 |
|---:|---|---|---|
| 020 | [Run 実行ガイド](020_user_guide_run.jp.md) | 設定、起動、画面、基本操作、Run 成果物 | 実行利用者 |
| 030 | [Run 分析ガイド](030_user_guide_analysis.jp.md) | メトリクス、Metrics Viewer、Run 比較、外部分析ツール | 分析利用者 |

### 3.3 開発ガイド

| 番号 | 文書 | 主な観点 | 対象読者 |
|---:|---|---|---|
| 040 | [開発環境構築ガイド](040_development_environment.jp.md) | Windows/MSVC、依存ライブラリ、ビルド、テスト | 開発者 |

### 3.4 機能カテゴリ別設計ガイド

設計ガイドは機能カテゴリ単位で分け、各文書の中で関係する処理工程を時系列に説明します。

| 番号 | 文書 | 主な観点 | 対象読者 |
|---:|---|---|---|
| 100 | [実行基盤と設定](100_runtime_and_configuration.jp.md) | 起動、設定解決、Run 構築、ライフサイクル | フレームワーク開発者 |
| 110 | [Agent と学習](110_agents_and_learning.jp.md) | Agent、Actor、Learner の共通契約と所有権 | Agent・フレームワーク開発者 |
| 120 | [環境](120_environments.jp.md) | Env、BatchEnv、Reset、Step、Env 実装 | Env・フレームワーク開発者 |
| 130 | [ニューラルネットワーク](130_neural_networks.jp.md) | NetworkModel、モジュール、forward、optimizer | NN・Agent 開発者 |
| 140 | [可観測性](140_observability.jp.md) | Event、Observer、メトリクス、可視化、profiling | フレームワーク・分析機能開発者 |
| 150 | [ReplayBuffer](150_replay_buffer.jp.md) | Experience、N-step、PER、転送、prefetch | Agent・性能改善担当者 |
| 160 | [アプリケーションとツール](160_applications_and_tools.jp.md) | Runner GUI、Metrics Viewer、補助ツール | アプリケーション開発者 |

### 3.5 具象実装仕様

具象実装仕様は、共通設計の上に成り立つ個別の実装単位について、部品構成、設定、データ構造、内部契約を説明します。Agent ファミリと、独立した実行単位を持つアプリケーションの両方をここに置きます。

| 番号 | 文書 | 主な観点 | 対象読者 |
|---:|---|---|---|
| 200 | [DQN 系 Agent](200_dqn_agents.jp.md) | DefaultDQN、Rainbow、DQN 共通部品、学習・同期・PER | DQN 系 Agent 開発者 |
| 210 | [Metrics Viewer](210_metrics_viewer.jp.md) | 取り込み、キャッシュ DB、range query、描画、設定、依存 | Metrics Viewer 開発者 |

## 4. 関連文書

- [プロジェクト README](../../README.md)
- [ドメイン用語集](../../CONTEXT.md)
- [Agent 実装の所有権ガイドライン](../ownership_guideline.md)
- [設計判断記録](../adr/)
- [実装計画・検討メモ](../memo/)
- [ANET 概要紹介 PDF](../anet_overview_ja.pdf)
