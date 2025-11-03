# CartPoleRLGUI

## 概要

## ビルド方法

## DONE

* 学習アルゴリズム
	* Double DQN対応
	* Replay Buffer
	* Adaptive Stabilized DQN (AS-DQN)：基本ロジック
	* AS-DQN：アクション偏り統計対応（CartPole以外の応用には適用性が低い）
	* AS-DQN：Adaptive ε-schedule
* メトリクス
	* TensorBoard対応
	* Plotyによる独自グラフ出力
	* メトリクスタグの適正階層化 
	* ヒストグラム出力対応
* AP改善
	* 設定情報をいい感じにメトリクスとして記録
	* ヒートマップ：基本対応
	* ヒートマップ：スウィープ対応
	* ヒートマップ：ヒストグラム対応（1D、1D＋時系列）
	* 設定ファイル導入
	* ReplayBuffer貯め中にログが出ないので軸がずれている問題
	* BUG:TimeHeatMapのUnlimitedがスクロールしてしまう
	* ヒートマップ：動画化対応

## DOING

* 学習アルゴリズム
	* AS-DQN：ハイパラ調整
* メトリクス
* AP改善

## TODO

* 学習アルゴリズム
	* AS-DQN：過安定制御(stagnant) 
	* AS-DQN：勾配／損失ベース
	* Adaptive Grad Control
	* Adaptive α-schedule
* メトリクス
	* 閾値値等の基準横線出力対応
	* ヒートマップ：凡例出力対応
	* ヒートマップ：MetricsViewer対応
	* ヒートマップ：ヒストグラムTB対応
	* TB:add_hparams対応
* AP改善
	* Runフォルダに設定内容ダンプを残す
	* 設定の継承読み込み
	* MetricsViewer.java
	* コマンドライン引数でRun名を指定
	* リファクタリング：命名規約揃え
	* リファクタリング：メソッド分割の適正化

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

## 劣後
* 学習アルゴリズム
	* Adaptive Stabilized DQN (AS-DQN)：unstable_ema
* メトリクス
* AP改善
	* マルチスレッド対応（危険）
	* logs -> runs?
	* 起動時にRun名をプロンプト
	* 学習と推論の表示分離

## メトリクスメモ

### q_var

| 観測パターン	|状態	|解釈
|---------------|-------|-----|
| q_var が徐々に減少 or 安定	|正常学習中	| 続行 |
| q_var が周期的に波打ち始める	|崩壊前兆	|対策チャンス
| q_var が急上昇して戻らない	|破綻（学習壊れた）	|ε↑ or τ↓ or LR↓、最悪再初期化|

### 自動安定制御

 | パラメータ | 役割	| 何を抑える？ | 効果の即効性 |見た目の変化 |
 |------------|---------|--------------|--------------|-------------|
 | τ (soft update)	| Q_targetの更新速度	 | Q値の不安定化（振動/爆発）	   | 中〜遅い | Q値 TimeHistogram が安定 |
 | ε (exploration)	| 探索 / 経験の多様性	 | ReplayBuffer の偏り（行動固着） | 即効	  | Action TimeHistogram が変わる |

| 状態	|ε の動き|	τ の動き	|結果|
|-------|---------|-------------|----|
|行動が固着	|ε↑	|τは維持	|ReplayBufferに多様性回復 → 行動揺れ復活 → 崩壊回避|
|Q値が振動	|εは維持	|τ↓	|Q値の波動が止まる → TimeHistogram の縞模様が消える|
|学習安定	|ε徐減	|τ維持 	|スムーズに収束|

 | タグ |計算式| 意味｜ 何を抑える？ | 効果の即効性 |見た目の変化 |
 |------------|---------|--------------|--------------|-------------|
 | q_std	  | 現在の標準偏差 | 瞬間的な揺れの大きさ | Q値の不安定化（振動/爆発）	   | 中〜遅い | Q値 TimeHistogram が安定 |
 | q_std_ema  | q_stdの移動平均| 長期的な安定レベル | 即効	  | Action TimeHistogram が変わる |
 | q_std_z    | 	 | ReplayBuffer の偏り（行動固着） | 即効	  | Action TimeHistogram が変わる |


## 参考文献
