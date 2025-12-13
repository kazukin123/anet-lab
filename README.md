# anet-lab

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
	* Double DQN Agent実装
	* CartPole環境APP実装
	* LunarLander環境APP実装
* メトリクス
	* MetricsViewer.java：フェーズ1.5：基本機能、差分ロード
	* TensorBoard対応（ブリッジpyスクリプト経由）
	* Ploty＋pythonによる独自グラフ出力
	* ヒストグラムやヒートマップの時系列動画出力（ffmpeg利用）
* AP機能
	* CUDAによる高速処理
	* N環境スレッド対応
	* seed値指定
	* 設定でメトリクス定義
	* 設定の継承読み込み
	* 学習と推論の表示分離

### DOING

* 学習アルゴリズム
* メトリクス
* AP改善
	* 評価RunnerでENVとAPの個別RunMode指定
	* 同じクラスでインスタンス別の設定キー

### TODO

* 学習アルゴリズム
	* Rainbow対応
* メトリクス
	* MetricsViewer.java：フェーズ1.5：ハイパラ表示対応
	* MetricsViewer.java：フェーズ1.5：Loadingスケジュール最適化
	* 閾値値等の基準横線出力対応
	* ヒートマップ：凡例出力対応
	* ヒートマップ：MetricsViewer統合
	* ヒートマップ：ヒストグラムTB対応
	* TB:add_hparams対応
* AP改善
	* コマンドライン引数でRun名を指定
	* Runフォルダに設定内容ダンプを残す
	* TODOコメントいれまくり
	* constいれまくり
	* GPU対応向けソース可読性向上
	* ヒートマップ：サンプリング量で書き出しタイミング制御

### SUSPENDED
* 学習アルゴリズム
	* DDPG対応
	* TD3対応
	* SAC対応
	* AS-DQN：ハイパラ調整
	* Adaptive Stabilized DQN (AS-DQN)：unstable_ema
	* AS-DQN：過安定制御(stagnant) 
	* AS-DQN：勾配／損失ベース
	* Adaptive Grad Control
	* Adaptive α-schedule
* メトリクス
* AP改善
	* 起動時にRun名をプロンプト

## 備考

このプロジェクトは、OpenAI ChatGPT (GPT-5) の技術的支援を受けながら開発を進めています。<br>
Developed with technical assistance from OpenAI ChatGPT (GPT-5).

## 参考文献
