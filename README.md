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
	* 強化学習基本インタフェース 
	* DQNAgent
	* CartPoleEnvGUI
	* LunarLanderEnvGUI
* メトリクス
	* MetricsViewer.java：フェーズ1.5(基本機能、差分ロード)
	* TensorBoard対応（ブリッジpyスクリプト経由）
	* Ploty＋pythonによる独自グラフ出力
	* ヒストグラムやヒートマップの時系列動画出力（ffmpeg利用）
* AP機能
	* CUDA対応
	* N環境スレッド対応
	* 設定による柔軟なメトリクス定義
	* 学習と推論の同時分離表示

### DOING

* 学習アルゴリズム
	* Replay Ratio対応
	* Observation Normalization
* メトリクス
* AP改善
	* 評価RunnerでENVとAPの個別RunMode指定

### TODO

* 学習アルゴリズム
	* SWA
	* CNN
	* Transformer
	* AlphaZero-Lite
	* MuZero
* メトリクス
	* MetricsViewer.java：フェーズ1.5：ハイパラ表示対応
	* MetricsViewer.java：フェーズ1.5：Loadingスケジュール最適化
	* 閾値値等の基準横線出力対応
	* ヒートマップ：Config対応
	* ヒートマップ：凡例出力対応
	* ヒートマップ：MetricsViewer統合
	* ヒートマップ：ヒストグラムTB対応
	* TB:add_hparams対応
* AP改善
	* コマンドライン引数でRun名を指定
	* 同じクラスでインスタンス別の設定キー
	* TODOコメントいれまくり
	* constいれまくり
	* GPU対応向けソース可読性向上
	* ヒートマップ：サンプリング量で書き出しタイミング制御

* ### DONE (自分用メモ)

* 学習アルゴリズム
	* Orthogonal Initialization
	* Adam Optimizer's Epsilon Parameter optimization
* AP改善
	* 設定の継承読み込み
	* Runフォルダに設定内容ダンプを残す
	* seed値指定

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
* メトリクス
* AP改善

## 備考

このプロジェクトは、OpenAI ChatGPT (GPT-5) の技術的支援を受けながら開発を進めています。<br>
Developed with technical assistance from OpenAI ChatGPT (GPT-5).

## 参考文献
