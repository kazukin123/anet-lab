# TempoRLのSTOP×Singleと無効のbit比較

## 元の材料

[TempoRLのSTOP×Singleと無効のbit比較](../../../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-stop-single-bit-compare.md) blob:353fef41cd12e146dfbf6f0d279a6bfbab760414

## 要点

- skipの学習の中身はQ側へ届いていない。
- 乱数とtargetのforwardを分けると、STOP×Singleは無効と全点一致した。
- パッチは無効の経路を変えていない。 パッチなしの無効とパッチ版の無効は10万exp stepまで21,987点が全点一致した。`[greedy]` sessionを足しても一致は崩れなかった（21,987点）。
- 同じseedで順番だけが違うと、差の向きが区間で入れ替わる。
- STOP対照の差は、偶然で起きる範囲に入る。
- 差は最初の学習から出て、warmupの教材は同じだった。

## 内容

### 条件と判定方法

STOPの群は`learner.skip.feature_gradient=stop`、`learner.skip.target_window=action_change`、すべてのActorで`skip.max_length_mode=single`、`[greedy]` sessionつきである。`sv_stop_w1L`と`run_20261029-162638_t104r1_ctl_s1`の実効configで違うのは、決定的モードの3キー、あとから足された`skip.nn_forward_mode`（5 Actor）、metricsとappのキーで、ほかに実行体のビルドが違う。

### 差の経路

skipのTD誤差は`CombinePriorityErrors`で`priority_lambda > 0`のときだけPERの優先度へ入り、このRunでは0である。skipのNetworkと方策の乱数の種は名前つきで作るので、skipの有無でreplayとLearnerの種はずれず、Q側の乱数列も進まない。Singleはスキップ長を常に1にする。ActorのskipのforwardはSingleでも動くが、その結果は長さの選択に使わない。

### 同じseedで順番だけが違うときの報酬

seed 1、決定的モード、`21_eval/02_policy_reward`。パッチ版のSTOPはパッチ版の無効と全点同じ値である。

### 過去のRunとの関係

無効は5本と数える。`t98_ll_off_s3`と`t104r1_off_s3`は、ビルドが違うが評価の312点がすべて一致した。s1とs2は14,848 exp stepの評価点から違う。

並べ替えの割合は、8本から3本を選ぶ56通りのうち、選んだ3本の平均と残り5本の平均の差が観測以上になる割合である。統計的有意差の検定ではない。

### パッチ

窓長の抽選と、窓の target 用の τ の抽選をそれぞれ別の乱数列に移した。窓の target は Q の target と別の forward で計算した。この条件で STOP×Single と無効を seed 1 の100万 exp step まで比較した。

### 解釈の限界と残る問い

seed 1だけで、範囲は100万exp stepまでである。1〜2Mは測っていない。

「更新の分布を変えない」はコードの読みである。10/29のSTOP対照は決定的モードではなく、同じRunをbit比較したものではない。

### Runと実効config

記録は `loss_weight` の違いを比べた Run と、乱数列・target の計算を分けて STOP×Single と無効を比べた Run の設定を掲載している。実時間を測った間は DropMerge も同時に動いていた。

比較に使った過去のRunの実効configは、Singleの7群比較とTrainの繰り返しの時期と長さの比較にある。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [LunarLander](../../../../concepts/subjects/lunarlander.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
