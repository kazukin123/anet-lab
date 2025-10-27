# CartPoleRLGUI

## 概要

## ビルド方法

## DONE

* 学習改善
	* Double DQN対応
* メトリクス
	* TensorBoard対応
	* Plotyによる独自グラフ出力
* AP改善
	* 設定情報をいい感じにメトリクスとして記録

## DOING

* 学習改善
	* Replay Buffer
* メトリクス
	* ヒートマップ：基本対応
	* ヒートマップ：スウィープ対応
	* ヒートマップ：ヒストグラム対応（1D、1D＋時系列）
* AP改善
	* 設定ファイル導入

## TODO

* 学習改善
	* Adaptive Grad Control
	* 学習率減衰
* メトリクス
	* メトリクスタグの適正階層化 
	* ヒストグラム出力対応
	* add_hparams対応
	* ヒートマップ：動画化対応
	* ヒートマップ：凡例出力対応
	* ヒートマップ：MetricsViewer対応
	* ヒートマップ：TensorBoard対応
* AP改善
	* 起動時にRun名をプロンプト
	* 学習と推論の表示分離
	* loggerの引数順番をSummaryWriter同様に戻す
	* 報酬バーをRewardのスケールに合わせる
	* GPU対応向け可読性向上
	* コマンドラインオプションで設定上書き
	* logs -> runs?
	* リファクタリング

## 参考文献
