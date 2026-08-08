# anet-lab

[![Windows Build](https://github.com/kazukin123/anet-lab/actions/workflows/windows-ci.yml/badge.svg)](https://github.com/kazukin123/anet-lab/actions/workflows/windows-ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## 概要

- libtorchを基盤としたC++による強化学習実装
- wxWidgetsによるGUIフロントエンド
- Java/SpringによるWebベースのMetricsViewer

勉強と趣味で作っているので不安定です。

## ドキュメント

日本語のドキュメント入口は[docs/design/README.jp.md](docs/design/README.jp.md)です。
フレームワーク全体像、ユーザーガイド、開発環境、機能カテゴリ別・具象Agent別の設計ガイドを参照できます。

| 場所 | 役割 |
|---|---|
| [docs/design/](docs/design/) | 現在の構成、利用方法、実装上のcontractを説明する |
| [CONTEXT.md](CONTEXT.md) | プロジェクト内で使うドメイン用語の正本 |
| [docs/adr/](docs/adr/) | 採用した設計判断と理由を記録する |
| [docs/memo/](docs/memo/) | 未実装の要求、検討中の設計、実装計画を記録する |
| [docs/ownership_guideline.md](docs/ownership_guideline.md) | Agent系State/Resourceの所有権規則 |
| Doxygen | class、method、公開APIの詳細 |

現行動作の最終的な根拠はコード、設定、テストとする。
設計ガイドへ将来案を現行仕様として混在させず、変更前の計画は`docs/memo/`、採用後も残す判断理由は`docs/adr/`へ置く。

## ビルド手順

Windows/MSVC、libtorch、wxWidgets、CUDA、Metrics Viewerを含む開発環境と検証手順は、
[開発環境構築ガイド](docs/design/040_development_environment.jp.md)を参照してください。

## 実行手順

設定ファイルの選択、Runnerの起動、画面操作、Run成果物は、
[Run実行ユーザーガイド](docs/design/020_user_guide_run.jp.md)を参照してください。

## 対応状況/予定

### DONE

* 学習アルゴリズム
	* 強化学習基本クラス群
	* Rainbow Agent
	* QR-DQN
	* 1D-Conv/2D-Conv
	* Reward Scaling
	* Observation Normalization
	* TransformerEncoder
* メトリクス
	* MetricsViewer.java(Scalarメトリクスのリアルタイム表示)
	* ヒストグラムやヒートマップの時系列動画出力（ffmpeg利用）
	* Conv2D動画可視化
	* TensorBoard連携（ブリッジpyスクリプト経由）
* AP機能
	* CUDA対応
	* マルチスレッドによるN環境並列実行
	* 設定によるNN定義
	* 設定によるメトリクス定義
* Env実装
	* CartPoleEnv
	* LunarLanderEnv
	* DropMergeEnv

### DOING

* 学習アルゴリズム
	* ①MuZero試作(オリジナル版ベース、除外：Categorical分布、PER、Reanalyze、Batched MCTS）
* Env実装
	* DropMergeEnv評価＆調整
* メトリクス
* AP改善
	* ②フレームワーク構成変更
	* ③学習セッション断面の保存と読込
	* ④Dict Observation Space (Vector / Image / Token)

### TODO

* 学習アルゴリズム
	* ActionMasking
	* マルチエージェント対応
	* Ape-X Exploration(分散ε)
	* Transformer
	* MuZero
	* SWA
	* Swish活性化関数
* メトリクス
	* MetricsViewer.java：間引データ連携対応
	* MetricsViewer.java：フェーズ1.5：ハイパラ表示対応
	* MetricsViewer.java：フェーズ1.5：Loadingスケジュール最適化
	* 閾値値等の基準横線出力対応
	* ヒートマップ：Config対応
	* ヒートマップ：凡例出力対応
	* ヒートマップ：MetricsViewer統合
	* TB:add_hparams対応
* AP改善
	* Envのメトリクス取得
	* EvalRunnerのseed指定
	* PolicyコピーによるEval性能向上
	* EnvRunnerのスレッド化
	* RB:NextStateのメモリ削減
	* EvalPanelに自動アクション選択を追加
	* NNパラメータの保存と読込
	* 同じクラスでインスタンス別の設定キー
	* TODOコメントいれまくり
	* constいれまくり
	* GPU対応向けソース可読性向上
	* ヒートマップ：サンプリング量で書き出しタイミング制御

* ### DONE (自分用メモ)

* 学習アルゴリズム
	* Orthogonal Initialization
	* Adam Optimizer's Epsilon Parameter optimization
	* Replay Ratio対応
* AP改善
	* 設定の継承読み込み
	* Runフォルダに設定内容ダンプを残す
	* seed値指定
	* コマンドライン引数でRun名を指定
	* 学習と推論の同時分離表示


### SUSPENDED
* 学習アルゴリズム
	* DDPG対応
	* TD3対応
	* SAC対応
	* Adaptive Stabilized DQN (AS-DQN)：unstable_ema
	* AS-DQN：過安定制御(stagnant)
	* AS-DQN：勾配／損失ベース
	* Adaptive Grad Control
	* Adaptive α-schedule
	* AlphaZero-Lite
* メトリクス
	* ヒートマップ：ヒストグラムTB対応
* AP改善

## ライセンス

特記のない本リポジトリのファーストパーティ部分は、[Apache License 2.0](LICENSE)で提供します。
著作権表示は[NOTICE](NOTICE)を参照してください。

`third_party/`、`licenses/`、およびファイル内で別のライセンスが明記された部分には、
それぞれのライセンス条件が適用されます。

## 備考

このプロジェクトは、OpenAI ChatGPT (GPT-5) の技術的支援を受けながら開発を進めています。<br>
Developed with technical assistance from OpenAI ChatGPT (GPT-5).

## 参考文献
