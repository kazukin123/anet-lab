# CartPoleRLGUI

## 概要
libtorchを基盤とした強化学習ライブラリ、およびwxWidgetsを利用したそれらのGUIフロントエンドです。

## ビルド手順
TODO：ビルド手順や依存ライブラリを書く

## 対応状況

### DONE

* 学習アルゴリズム
	* Double DQN対応
	* Replay Buffer対応
	* Adaptive Stabilized DQN (AS-DQN)：基本ロジック
	* CartPole環境APP実装
* メトリクス
	* TensorBoard対応
	* Ploty＋pythonによる独自グラフ出力
	* ヒストグラムやヒートマップの動画出力対応（ffmpeg利用）
* AP改善
	* 設定情報をいい感じにメトリクスとして記録
	* 設定ファイル

### DOING

* 学習アルゴリズム
* メトリクス
	* MetricsViewer.java：基本機能対応
* AP改善

### TODO

* 学習アルゴリズム
	* DDPG対応
	* TD3対応
	* SAC対応
* メトリクス
	* MetricsViewer.java：複数Run表示対応
	* MetricsViewer.java：UIレイアウト改善対応
	* 閾値値等の基準横線出力対応
	* ヒートマップ：凡例出力対応
	* ヒートマップ：MetricsViewer対応
	* ヒートマップ：ヒストグラムTB対応
	* TB:add_hparams対応
* AP改善
	* Runフォルダに設定内容ダンプを残す
	* 設定の継承読み込み
	* コマンドライン引数でRun名を指定
	* リファクタリング：命名規約揃え
	* リファクタリング：メソッド分割の適正化
	* リファクタリング：DQNAgent周りの整理
	* TODOコメントいれまくり

	* Tensorチェック用マクロ
	* EnvironmentとAgentのバッチ対応精査
	* N環境想定のTensor仕様整理
	* TensorShape アサーション

	* コマンドラインオプションで設定上書き
 	* loggerの引数順番をSummaryWriter同様に戻す?
	* 報酬バーをRewardのスケールに合わせる
	* GPU対応向けソース可読性向上
	* リファクタリング：EMAを簡単に使えるようにするクラス
	* ReplayBufferの実装最適化（最初からTensorを保持しておく）
	* HeatMapのサンプリング量で書き出しタイミング制御

## SUSPENDED
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

## メトリクスメモ

## 参考文献
