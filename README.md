# anet-lab

## 概要

- libtorchを基盤としたC++による強化学習実装
- wxWidgetsによるGUIフロントエンド
- Java/SpringによるWebベースのMetricsViewer

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
* メトリクス
	* MetricsViewer.java：フェーズ1.0(基本機能)
	* TensorBoard対応（ブリッジpyスクリプト経由）
	* Ploty＋pythonによる独自グラフ出力
	* ヒストグラムやヒートマップの時系列動画出力（ffmpeg利用）
* AP改善
	* 設定情報をいい感じにメトリクスとして記録
	* 設定ファイル

### DOING

* 学習アルゴリズム
* メトリクス
* AP改善
	* リファクタリング：クラス/メソッド構成の適正化
	* リファクタリング：DDQNAgent実装整理
	* リファクタリング：命名規約揃え

### TODO

* 学習アルゴリズム
	* DDPG対応
	* TD3対応
	* SAC対応
* メトリクス
	* MetricsViewer.java：フェーズ1.5
	* 閾値値等の基準横線出力対応
	* ヒートマップ：凡例出力対応
	* ヒートマップ：MetricsViewer統合
	* ヒートマップ：ヒストグラムTB対応
	* TB:add_hparams対応
* AP改善
	* ReplayBufferの実装最適化（最初からTensorを保持しておく）
	* N環境とバッチ実行を前提としたAPI見直し
	* Tensorチェック用アサーション

	* コマンドライン引数でRun名を指定
	* Runフォルダに設定内容ダンプを残す
	* 設定の継承読み込み
	* TODOコメントいれまくり
	* GPU対応向けソース可読性向上
	* ヒートマップ：サンプリング量で書き出しタイミング制御

### SUSPENDED
* 学習アルゴリズム
	* AS-DQN：ハイパラ調整
	* Adaptive Stabilized DQN (AS-DQN)：unstable_ema
	* AS-DQN：過安定制御(stagnant) 
	* AS-DQN：勾配／損失ベース
	* Adaptive Grad Control
	* Adaptive α-schedule
* メトリクス
* AP改善
	* マルチスレッド対応（危険）
	* 起動時にRun名をプロンプト
	* 学習と推論の表示分離

## 備考

このプロジェクトは、OpenAI ChatGPT (GPT-5) の技術的支援を受けながら開発を進めています。<br>
Developed with technical assistance from OpenAI ChatGPT (GPT-5).

## 参考文献
