# 外的と内的の価値の分解

> 主たる観点: DefaultDQNAgentの報酬成分、Head、行動選択、学習、診断、保存。

## 1. はじめに

### 1.1 目的と範囲

DefaultDQNAgentは、外的報酬と内的報酬を足して1本のQを学ぶ`combined`と、別々のHeadで学ぶ`decomposed`を持つ。既定は`combined`である。TD、QR、IQNに対応する。内的報酬の推定器は[RND](212_rnd_lifelong_novelty.jp.md)、共通の行動選択と学習は[DQN系Agent](200_dqn_agents.jp.md)、成分の保存と抽選は[ReplayBuffer](150_replay_buffer.jp.md)を参照する。

### 1.2 責任の境界

Replayは成分数、報酬、割引率、収益を扱い、外的・内的、β、終端の意味を持たない。DefaultDQNAgentが成分0を外的、成分1を内的として扱う。NetworkのHeadはそれぞれの値を出し、βによる結合とTBOの扱いはDQN側が所有する。

## 2. 構成と報酬の流れ

### 2.1 Head

`decomposed`では、既存のTD・QR・IQN Headを外的と内的に1つずつ組む。Bodyとforwardの実行器は共通で、Headのパラメータは別である。Dueling、初期化、通常の射影とNoisyの射影をHeadごとに選べる。分布の表現は外的・内的で共通である。

| 出力 | 外的 | 内的 |
|---|---|---|
| 行動価値 | `extrinsic_q` | `intrinsic_q` |
| 分位点 | `extrinsic_q_dist` | `intrinsic_q_dist` |
| Duelingの内訳 | `extrinsic_v`、`extrinsic_a`など | `intrinsic_v`、`intrinsic_a`など |

Network自身は分解時に`q`と`q_dist`を返さない。DQNの結合処理が、そのforwardで使うβにより両方を足す。行動の方策、既存のQのメトリクス、診断の読み手は、結合後の`q`・`q_dist`を読む。

内的Headの`input_mode=shared`は、外的Headと同じBody出力を読む。通常は`features`、Duelingの専用出力を使う場合は`value_feature`と`adv_feature`である。`dedicated`は、対応する`intrinsic_features`、または`intrinsic_value_feature`と`intrinsic_adv_feature`を読む。Bodyの`net.body.output`で公開する。IQNの入力はτを混ぜたrank 3の特徴を必要とする。欠けた鍵や不正なshapeは構築時に拒否する。

### 2.2 報酬の成分

外的報酬は、既存の外的報酬スケーラを通す。内的報酬は、RNDの倍率から1を引き、`intrinsic.use_done_mask`によるdoneの扱いを適用したあと、内的報酬スケーラを通す。

| モード | Replayへ渡す報酬 | Replayの成分数 |
|---|---|---|
| `combined` | 外的 + β × 内的 | 1 |
| `decomposed` | 外的、βを掛けない内的 | 2 |

分解時のβは、Replayの保存値に入らない。取り込みの入口で評価した学習のβを、今回のLearnerの更新へ渡す。Actorは自分のstepで行動のβを評価する。報酬をdoneで0にする設定と、内的価値のbootstrapを終端で切る設定は独立である。

### 2.3 結合とTBO

実空間では`Q = Q_ext + β Q_int`である。QRとIQNは同じτの分位点どうしを足す。TBOが有効なら、各Headの出力へhの逆変換を掛け、実空間で足し、hを掛け直す。変換済みの値を直接足さない。

## 3. 学習

### 3.1 目標値

外的・内的のn-step収益をそれぞれ`R_ext`・`R_int`、実際のstep数をnとすると、目標値は次のとおりである。

```text
y_ext = R_ext + gamma_ext^n (1 - done) Q_ext,target(s_next, a_ext)
y_int = R_int + gamma_int^n (1 - done_int) Q_int,target(s_next, a_combined)
```

Double DQNは行動をonline側、価値をtarget側で評価する。無効時は行動もtarget側で選ぶ。内的の行動は結合したQから選ぶ。外的の行動は`extrinsic_bootstrap_mode=combined`なら同じ行動、`extrinsic`なら外的のQから選ぶ。

`discount_mode=shared`は外的の割引率を使う。`separate`は明示した内的の割引率を使い、scheduleも選べる。定数の成分はPush時の収益を保存し、scheduleの成分だけ抽選後に保存したstep報酬から収益を作り直す。

`terminal_mode=episodic`では、内的もdoneでbootstrapを切る。`non_episodic`では、終端後の最初の観測から内的だけbootstrapする。doneとtruncatedが同時に真なら両方とも切る。truncatedだけなら両方ともbootstrapする。TBOは完成した実空間の目標値へ適用する。

### 3.2 Munchausen

Munchausen項とエントロピー項は外的Headだけに置く。内的Headは次状態の内的Qの期待値だけを使う。

```text
y_ext = R_ext + bonus + gamma_ext^n (1 - done) sum(pi_ext * (Q_ext,target - tau log pi_ext))
y_int = R_int + gamma_int^n (1 - done_int) sum(pi_combined * Q_int,target)
```

`pi_combined`は結合したQから作る。外的の`pi_ext`とbonusは、外的のbootstrapモードが`extrinsic`なら外的のQ、それ以外は結合したQから作る。`log_policy_mode`によるtarget、online、current出力の再利用の規則は通常のDQNと共通である。bonusは先頭の遷移へ1回だけ加え、終端のマスクを掛けない。

割引率共有・episodic・combined bootstrapでは、`y_ext + β y_int`が、足した報酬で1本のQを学ぶ目標値と一致する。TBOではhを掛ける前の実空間で成り立つ。

### 3.3 損失、優先度、最適化

損失は`L_ext + loss_weight * L_int`、PERの誤差は`abs(delta_ext) + priority_lambda * abs(delta_int)`である。Headごとに既存のTD・QR・IQNの損失と誤差を計算し、PERの下限と上限は足したあとに1回だけ適用する。Huberの折れ目、分位点のκ、TD誤差のclipは共通である。損失の重みは正、優先度の重みは0以上とする。

`feature_gradient=merge`は損失を足して1回backwardし、全体をclipして1つのAdam状態で更新する。`split`は同じforwardのgraphから外的と重み付き内的の勾配を別々に取り、損失ごとにclipし、別々のAdam状態による更新を足す。共有部分のweight decayは外的側で1回、各Headはそれぞれの更新で1回だけ掛ける。Adamが歩幅をそろえるため、splitでは損失の重みの影響が小さくなる。

## 4. 設定

### 4.1 分解と行動

| キー（DefaultDQNAgent配下） | 値・既定 | 条件 |
|---|---|---|
| `intrinsic.value_mode` | `combined`、`decomposed`。既定`combined` | 分解にはintrinsic有効、skip無効が必要 |
| `learner.intrinsic.discount_mode` | `shared`、`separate`。既定`shared` | separateはgammaの明示が必要 |
| `learner.intrinsic.gamma` | 0より大きく1未満。既定なし | 定数またはschedule |
| `learner.intrinsic.terminal_mode` | `episodic`、`non_episodic`。既定`episodic` | 内的のbootstrap |
| `learner.intrinsic.loss_weight` | 既定1、有限かつ正 | 損失の重み |
| `learner.intrinsic.priority_lambda` | 既定1、有限かつ0以上 | 優先度の重み |
| `learner.intrinsic.extrinsic_bootstrap_mode` | `combined`、`extrinsic`。既定`combined` | 外的のbootstrap方策 |
| `learner.intrinsic.feature_gradient` | `merge`、`split`。既定`merge` | 最適化の分け方 |
| `intrinsic.value.head.input_mode` | `shared`、`dedicated`。既定`shared` | Body出力の選択 |
| `intrinsic.value.head.use_dueling` | 既定true | 内的HeadのDueling |
| `intrinsic.value.head.init.*`、`projection.*` | 通常Headと同じ決まり | 内的Headの初期化と射影 |
| `actor.[name].intrinsic.action_beta_mode` | `learner`、`fixed`。既定`learner` | 行動のβ |
| `actor.[name].intrinsic.action_beta` | 有限かつ0以上。既定なし | fixedは明示が必要 |

sharedで書いた内的gamma、learnerで書いたaction_beta、combinedで書いた分解専用値は使わない。認識する値は型・値域を検証し、休眠と未使用の診断は共通の設定方針に従う。不正な値や、分解を要求しながらHeadを構築できない構成をcombinedへ置き換えない。

### 4.2 報酬スケーラ

`reward_scaler.dynamic_mode`と`intrinsic.reward_scaler.dynamic_mode`は`reward_std`（既定）または`return_std`を取る。`use_dynamic_scaling=true`で有効になる。内的の既定は動的処理・自動倍率・clipが無効、定数倍率と手動倍率が1の恒等変換である。

`return_std`はlaneごとに`R = gamma * R + reward`を保持し、Rの標準偏差で今回のrewardを割る。episode境界でRを0にせず、resetで全laneを0にする。統計部品、epsilon、手動倍率、clipはreward_stdと共通である。外的は外的gamma、内的は分解時の実効的な内的gammaのstep 0の値を使う。gammaのscheduleでRの割引率を途中変更しない。

return_stdと`use_auto_post_scale=true`の同時指定は拒否する。Traceは統計を更新しない変換を使う。内的スケーラはcombinedでも使える。RND推定器内部の誤差の正規化とは別の処理である。

## 5. 診断

### 5.1 Head別の値と学習

| 読み取り元 | 鍵 | 内容 |
|---|---|---|
| `$agent_step` | `extrinsic_q_sa`、`intrinsic_q_sa` | 選んだ行動のHead別Q |
| `$agent_step` | `extrinsic_state_value`、`intrinsic_state_value` | Head別の状態価値 |
| `$agent` | `extrinsic_loss`、`intrinsic_loss` | 重みを掛ける前の各損失 |
| `$agent` | `extrinsic_q_sa_mean`、`intrinsic_q_sa_mean` | 選んだ行動の各Qの平均 |
| `$agent` | `extrinsic_head_grad_norm`、`intrinsic_head_grad_norm` | clip前の各Headの最終射影の勾配norm |
| `$agent` | `extrinsic_grad_norm`、`intrinsic_grad_norm` | splitでの損失ごとのclip前の勾配norm |
| `$agent` | `intrinsic_q_share` | 選んだ行動の絶対値で見た内的の割合 |
| `$agent` | `intrinsic.reward_scaler.scale` | 内的報酬スケーラの倍率 |

`intrinsic_q_share`は`abs(β Q_int) / (abs(Q_ext) + abs(β Q_int))`の標本平均で、両方0の標本は0とする。split専用のnormはmergeでNaN、それ以外のHead別メトリクスとTraceの4鍵はcombinedでNaNとする。内的スケーラの倍率はintrinsic無効時にNaNとする。値がまだ成立していないときもNaNを返す。既存のq_sa、state_value、Qの学習メトリクスは結合した値である。

### 5.2 行動選択への影響

Actorは、各Headの平均の実空間Qと、そのActorのβから次の4値を作る。分布型は個々の分位点を逆変換してから平均する。UQEやThompsonのスコア、ε探索、強制行動には依存しない。

| laneの鍵 | lane平均の鍵 | 評価session累積の鍵 | 値 |
|---|---|---|---|
| `extrinsic_q_argmax_disagreement` | `extrinsic_q_argmax_disagreement_rate` | `extrinsic_q_session_argmax_disagreement_rate` | 結合Qと外的Qの1位が違えば1 |
| `extrinsic_q_regret` | `extrinsic_q_regret_mean` | `extrinsic_q_session_regret_mean` | 行動を変えたことで譲った外的Q |
| `extrinsic_q_gap` | `extrinsic_q_gap_mean` | `extrinsic_q_session_gap_mean` | 外的Qの1位と2位の差 |
| `intrinsic_q_gap` | `intrinsic_q_gap_mean` | `intrinsic_q_session_gap_mean` | βを掛けた内的Qの1位と2位の差 |

分解が有効なら購読の有無に依らず毎step作る。sessionはlane×stepで数え、同期で累積をリセットする。空sessionとcombinedはNaNとする。1行動なら不一致とregretは0、2つのgapはNaNとする。βが0なら不一致、regret、内的gapは0（1行動のgapはNaN）となる。laneの鍵は範囲外のlaneを拒否する。

lane平均は`@train $action_info`、評価の累積は`$eval.[eval] $actor @session_end`と`$eval.[eval_target] $actor @session_end`で読む。任意の組`metrics.scalar.@value_decomposition`を選択チェーンへ足して使う。既定のRunプロファイルとEvaluation Traceの既定列には足さない。

### 5.3 Actor Qヒント

hintは常時float32の5列で、順序は外的Q_sa、外的V_s、Munchausen項、内的Q_sa、内的V_sである。combinedでは先頭2列が従来のQ_sa・V_s、末尾2列は0となる。5列以外は拒否する。行動を差し替えた場合は外的Q_sa、内的Q_sa、Munchausen項を選び直し、状態価値は保つ。追加のforwardはしない。

初期優先度は各成分の収益と割引、hintから各Headの誤差を見積もり、学習後のPERと同じ重み付き絶対値の和へ下限と上限を1回適用する。

non_episodicの終端でも、初期優先度の内的の見積もりはbootstrapなしとする。その時点では終端後のhintがまだないためである。

## 6. 保存と復元

Networkの保存には両Headを含む。splitは二つのAdam状態を保存する。保存側と読込み側のvalue_modeが異なる場合、または分解時のfeature_gradientが異なる場合は拒否する。報酬スケーラは外的・内的とも、種類と動的な統計、return_stdのlane別Rを保存する。

旧い定数倍率・combinedのcheckpointは読める。動的状態のない旧checkpointは定数の種類として扱い、動的スケーラへの読込みでは種類不一致として拒否する。保存した種類と現在の種類が異なる場合も拒否する。return_stdのlane数が異なる場合は統計だけ引き継ぎ、Rを0にし、対象と保存側・現在のlane数を1回WARNする。観測正規化の状態の保存はこの機能の範囲外である。

## 7. テストと拡張時の確認事項

受入手順と再実行の入口は[testdata/rfd0121](../../src/core/base/testdata/rfd0121/README.md)、計画と検証の記録は[実装メモ](../memo/0121_extrinsic_intrinsic_value_decomposition_20impl.md)を参照する。

新しい方策や診断forwardを足すときは、Headの出力へ結合処理を適用してからq/q_distを読む。新しい報酬成分をReplayへ足すときは、1成分のshapeと計算順、定数収益の保存、schedule成分だけの再計算、初期優先度、prefetchの割引率の寿命を確認する。成分が複数のときの窓規則は未対応として構築時に拒否する。

内的Headの重みもNetworkのreadout側に入る。可塑性・Noisy・policy churn・Replay fitの診断を拡張するときは、結合したQを使う経路と、各Headのパラメータを扱う経路を区別する。

## 8. 関連文書

- [Replay の報酬成分と割引率 ADR](../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)（§2.2・§3.1）
- [内的価値 Head の置き場 ADR](../adr/0074-intrinsic-value-head-lives-in-primary-network-as-composite-head.md)（§2.1）
