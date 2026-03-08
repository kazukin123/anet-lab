# anet-lab

[![Windows Build](https://github.com/kazukin123/anet-lab/actions/workflows/windows-ci.yml/badge.svg)](https://github.com/kazukin123/anet-lab/actions/workflows/windows-ci.yml)

## 概要

- libtorchを基盤としたC++による強化学習実装
- wxWidgetsによるGUIフロントエンド
- Java/SpringによるWebベースのMetricsViewer

勉強と趣味で作っているので不安定です。
  
## ビルド手順
TODO：ビルド手順や依存ライブラリを書く

## 実行手順
TODO：実行手順を書く

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
	* MuZero（まずは検討用に試作）
* Env実装
	* DropMergeEnv評価＆調整
* メトリクス
* AP改善
	* ①フレームワーク構成変更
	* ②学習セッション断面の保存と読込
	* ③Dict Observation Space (Vector / Image / Token)

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

## 備考

このプロジェクトは、OpenAI ChatGPT (GPT-5) の技術的支援を受けながら開発を進めています。<br>
Developed with technical assistance from OpenAI ChatGPT (GPT-5).

## 参考文献
