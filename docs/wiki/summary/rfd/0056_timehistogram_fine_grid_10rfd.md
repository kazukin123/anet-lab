# TimeHistogram fine-bin保持・描画時集約 RFD

## 元の材料

[TimeHistogram fine-bin保持・描画時集約 RFD](../../../memo/0056_timehistogram_fine_grid_10rfd.md) blob:dcde72015eff2f56c0c4be039196e55b66eaeff6

## 要点

- DropMerge の計測では TimeHistogram の再構築が22.37秒中16.26秒を占め、学習スループットが落ちた。
- 全件の生値を保持して繰り返し分け直す処理を、細かい bin に蓄積する方式へ変える。
- 表示に必要な粗さへの集約は描画時に行う。
- ReplayBuffer 上の値を集める CPU コストと、ヒストグラム再構築のコストを分けて調べる。
- 描画した分布と学習への時間負荷を、同じメトリクス条件で比較する。

## 内容

### 解決方針

`TimeHistogramObserverConfig`に`sample_mode`(enum: Accumulate | Snapshot、既定Accumulate)を追加する。
stride間引き(決定的)とし、乱択はしない(同seed再現性を壊さない)。RBのindex順は挿入順のため、strideサンプルは全時期を等間隔にカバーし分布推定として偏らない。

### 実装上の決定

Scale統合を算術平均(和ではなく)にするのは、AutoNormValue無効時にも統合フレームの明度が跳ねないようにするため。AutoNormValue有効時はどのみちフレーム正規化されるため差は出ない。
`TimeHistogramObserver`のmutex(OnLearn/GetImageData間)は現行のまま。TimeHistogramに新たな共有可変状態を追加しない。描画がOnLearnと同一ロックで直列化される構造も維持する(描画は数msに軽量化されるためロック保持は問題にならない)。

### 受入条件

`TimeHistogram`に生値保持(`frames_raw_`/`cur_raw_`)が存在せず、保持メモリが`max_frames × fine_bins × 4B`+定数を上限とする。
`histgram.sample_mode = snapshot`で、probe取得とAddBatchがframe境界のみになる(mock検証)。既定はaccumulateで現行挙動と一致する。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
