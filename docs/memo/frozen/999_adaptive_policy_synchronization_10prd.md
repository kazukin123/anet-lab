# Train Actor Adaptive Policy Synchronization PRD 草案

> 凍結中(再開条件: 現行 ProfiledValue の一方向 cosine を長期 Run で評価し、adaptive が固定 profile を上回る根拠が得られたら)

> 番号 999。backlog / 検討草案。
> 関連: `036_train_actor_periodic_snapshot_10prd.md`、`035_approx_actor_priority_per_10prd.md`。
> 状態: 問題、観測、設計候補を忘れないための暫定記録。設定名、drift定義、採用方式、実装時期は未確定。

## 背景

PRD 036では、DefaultDQN Train ActorがLearner online networkを直接共有せず、privateなnetwork snapshotを一定期間使う機能を追加した。
同期周期は`ProfiledValue<step_t>`で表現され、`constant`、`linear`、`cosine`、`cosine_restart`、`phased`を選択できる。
profileの評価軸は`exp_step`、同期周期とsnapshot ageの単位は`train_step`である。

この機能の目的は、複数ENVを一括forwardするTrain Actorで、episode途中にもLearner更新によりnetworkが変化する影響を切り分けることである。
特にDropMergeでは、near-greedy laneがNOOPを継続するNEET現象をTrain側でも経験し、Evalだけで発生する状態を減らせるかを調べている。

一方、単一のbatched Actor networkを全ENV laneが共有する現在の構成では、ENVごとのepisode終端に合わせてnetworkを切り替えられない。
全laneを同時に同期する周期snapshotは、episode単位の方策固定ではなく、globalなpiecewise-stationary behavior networkを作る近似である。

## 現時点の観測

DropMergeの50M Runでは、次の傾向が観測された。

| 条件 | Eval Target EMA 40-50M | Eval Policy EMA 40-50M | Q max real | Train NoDrop | Eval NoDrop |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared network (`snap000`) | 約1387 | 約1322 | 約6.125 | 0 | 15 |
| fixed interval 200 | 約1271 | 約1238 | 約5.757 | 4 | 3 |
| fixed interval 400 | 約1224 | 約1198 | 約5.439 | 15 | 3 |

この結果から、少なくとも次は観測事実として扱う。

- snapshot intervalを長くすると、Train側でNoDropを経験する件数は増えた。
- interval 200と400では、Eval側NoDropはshared networkより少なかった。
- intervalを長くすると、50Mまでの報酬とQ値の立ち上がりは遅くなった。
- TD、gradient、PERに明確な破綻は見られず、学習停止ではなく成長速度の差に見える。
- fixed interval 200のRunでは、shared networkより実時間が約3%増え、throughput EMAが約4%低下した。network copy頻度との因果は未確定である。

ただし、50Mは長期Runの最終到達点を判断するには短い。次は未確認であり、前提にしてはならない。

- intervalが大きいほど最終Eval成績が高くなること。
- Eval NoDropの減少がsnapshotの効果であり、episode長や報酬水準の違いによる見かけではないこと。
- 50M時点の成長差が700Mから1000Mでも残ること。

想定される関係は単調ではない。intervalを長くするとbehavior networkの連続性が増す一方でActor-Learner lagも増えるため、最終成績には中間の適正範囲が存在する可能性がある。

## 問題

### 1. 固定`train_step`周期は意味が実験条件に依存する

同じintervalでも、実質的なstalenessは次に依存する。

- `replay_ratio`
- `replay_batch_size`
- `num_envs`
- learner updateの実行頻度
- learning rateとoptimizerによる1 updateあたりのparameter変化
- ENVのepisode長

現在のDropMerge設定では1 `train_step`あたり概算2.5 learner updateであるため、interval 200は最大約500 learner update、interval 400は最大約1000 learner updateのlagになる。
この換算はハイパーパラメータ変更で変わる。

また、平均episode長が約220 stepの場合、interval 200では多くのepisodeが同期境界を跨ぐ。
interval 400でもepisode開始位置によっては同期を跨ぐため、固定周期だけではepisode内network一貫性を保証できない。

### 2. 小周期と大周期のトレードオフをopen-loop値だけで決めている

- 小周期はLearnerに近い経験を生成するが、networkの時間的一貫性が弱い。
- 大周期は同じnetworkによる連続経験を増やすが、古いbehaviorによる経験をReplayBufferへ供給する。
- 小周期ほどnetwork copy回数が増え、実時間性能を悪化させる可能性がある。

モデル変化が大きい序盤と、parameter更新が小さくなる可能性がある習熟後で、同じ固定周期を使う必然性はない。

### 3. `ProfiledValue`は時間に対するopen-loop制御である

一方向のcosineなどにより「序盤は短く、後半は長く」は表現できる。この方向は次の理由で妥当な初期仮説である。

- 序盤はparameter driftが大きい可能性があり、短いintervalでstalenessを抑えたい。
- 習熟後はepisodeが長くなるため、intervalも延ばさないとepisode内同期回数が増える。

しかし、`exp_step`はmodel maturityやActor-Learner距離を直接表さない。学習停滞、急激な方策変化、load後の再開などに追従できない。

## 目的

1. 固定時間ではなく、Actor snapshotとLearner online networkの差に基づいて同期を判断できるようにする。
2. model driftが大きいときは短周期、driftが小さいときは長周期になる制御を可能にする。
3. 最小snapshot寿命を設け、毎stepのnetwork変化をそのまま追従するshared networkとは区別する。
4. 最大snapshot ageを設け、drift推定が過小でも無期限に古いnetworkを使わない。
5. v1候補では追加action forwardを要求せず、既存Train throughputへの影響を抑える。
6. 既存の`ProfiledValue`周期をfallback、hard limit、または比較用open-loop modeとして維持する。
7. Serial/Pipelineや分散マシン構成に依存しないAgent/Actor境界の機能として設計する。

## 非目標

- ENV laneごとに異なるnetwork snapshotを持ち、episode終端で厳密に切り替えること。
- snapshot versionごとにENV batchを分割し、複数回forwardすること。
- Eval Actorの同期契約を変更すること。
- target networkの同期周期を変更すること。
- NoDropTimeoutを必ず解消すること。
- Ape-X、R2D2、IMPALAの分散実行構成をそのまま導入すること。
- adaptive制御を報酬向上の保証として扱うこと。

## 暫定解決方針

### 1. min age、drift、max ageによる同期

基本候補は次の判定とする。

```text
if snapshot_age >= min_interval
   and (actor_learner_drift >= drift_threshold
        or snapshot_age >= max_interval):
    synchronize_snapshot()
```

- `min_interval`: behavior networkを最低限継続使用する期間。
- `drift_threshold`: Learnerとの差が許容範囲を超えたと判断する閾値。
- `max_interval`: drift推定に関係なく同期するhard limit。
- 強制`Actor::Sync()`は従来どおり全条件を無視して即時同期する。

この方式では、1 updateあたりの変化が大きい序盤は早くthresholdへ到達し、変化が小さい後半はintervalが自然に長くなることを期待する。
ただし、その性質はdrift指標とoptimizer挙動に依存するため、実測で確認する。

### 2. drift指標候補

候補を実装優先順ではなく、検討対象として列挙する。

#### A. Learner parameter update量の累積

snapshot同期時点以降のparameter update量をLearner側で累積し、Actorが同期判定に使う。

- action用の追加network forwardが不要。
- ENVやepisode長への直接依存がない。
- gradient normとlearning rateの単純積はAdam系optimizerの実parameter deltaと一致しない。
- 全parameterの実delta計測は追加走査と同期コストを伴う。

#### B. parameter subsetの距離

一部のlayerまたは固定sampleのparameterだけを比較し、normalized distanceを近似する。

- full parameter scanより安価にできる可能性がある。
- sample対象がpolicy変化を代表する保証がない。
- BatchNorm等のbuffer driftも別途考える必要がある。

#### C. probe state上のpolicy/Q divergence

固定またはReplay由来のprobe stateに対してsnapshotとonline networkを評価し、greedy action disagreement、Q順位相関、Q倍率差などを使う。

- 実際のaction選択差を直接測れる。
- 追加forward、probe state管理、device転送が必要になる。
- 毎actionではなく低頻度の診断・同期候補時だけ評価する案は残す。

#### D. Eval報酬やNoDrop発生率

主同期基準にはしない。

- 遅延が大きく、低頻度で、seed差とカオス性の影響が強い。
- 報酬低下後に同期しても手遅れになりやすい。
- adaptive制御の効果判定metricとしては利用する。

### 3. `ProfiledValue`との関係

既存機能を削除せず、次の段階を想定する。

1. `profiled`: PRD 036の現行動作。固定、linear、cosine、phasedでopen-loop制御する。
2. `adaptive`: drift thresholdで早期同期し、profiled値を`max_interval`として使う候補。
3. 必要なら`min_interval`も`ProfiledValue`化するが、初期実装では設定自由度を増やしすぎない。

暫定実験としては、一方向cosineの`100 -> 200`を200M `exp_step`で変化させ、その後200を維持する案がある。
これはadaptive方式ではなく、adaptive実装前に「序盤のlagを抑え、成熟後にsnapshot寿命を延ばす」仮説を検証するopen-loop比較である。
`cosine_restart`はlagを周期的に急変させるため、採用根拠が得られるまで使用しない。

## 所有権と依存方向

所有権は`docs/ownership_guideline.md`に従う。

- source online network、optimizer、共有drift集計ResourceはAgentが所有する。
- learner updateに伴うdrift Stateは、実際に更新するLearner側moduleが所有する。
- snapshot network、snapshot age、最終同期時点、同期理由はActor State / Actor-private Resourceとする。
- ActionPolicyからLearnerへの依存は作らない。
- ActorへLearner pointerを渡さず、Agentが所有する小さなread-only同期状態または明示interfaceを介する。
- shared stateの読み書きは既存Agent mutexとの関係を明文化し、action pathへ不要なexclusive lockを追加しない。

drift Stateの具体的な型、更新タイミング、archive対象かどうかは未決事項とする。

## 設定候補

設定名は未確定。意味だけを暫定記録する。

```ini
DefaultDQNAgent.train_actor.sync_mode = profiled | adaptive
DefaultDQNAgent.train_actor.sync_interval.* = <existing ProfiledValue>
DefaultDQNAgent.train_actor.adaptive.min_interval = ...
DefaultDQNAgent.train_actor.adaptive.drift_threshold = ...
DefaultDQNAgent.train_actor.adaptive.drift_metric = update_norm | parameter_subset | policy_probe
```

検討事項:

- `sync_interval`をadaptive時の`max_interval`と読むか、別キーへ分けるか。
- interval単位を`train_step`のまま維持するか、learner update ageも選べるようにするか。
- mode変更時も既存PRD036の設定成果物・後方互換性を維持できるか。
- 不正な閾値、非finite値、`min_interval > max_interval`はfail-fastする。

## メトリクス候補

既存:

- `train_actor_snapshot_interval`
- `train_actor_snapshot_age`

追加候補:

- `train_actor_snapshot_learner_update_age`
- `train_actor_snapshot_drift`
- `train_actor_snapshot_sync_count`
- `train_actor_snapshot_sync_reason`相当のreason別count
- `train_actor_snapshot_copy_time`
- `train_actor_episode_crossed_snapshot_ratio`

`train_actor_episode_crossed_snapshot_ratio`は問題を直接表すが、ENV laneごとのepisodeとsnapshot versionの対応追跡が必要になる。
Actor単体にENV lifecycle責務を持たせず、既存metadata経路で観測可能かを先に確認する。

adaptive採用判断では、次も併せて見る。

- 長期Eval Target / Policy rewardの終盤窓と傾き
- Train/Eval NoDrop率をepisode数で正規化した値
- 報酬崩壊と回復回数
- Q、TD、gradientの長期drift
- PER actor priorityを併用する場合のActor-Learner priority一致度
- `exp_step_per_sec`と実所要時間

## 検証方針

### 単体・結合テスト候補

1. `min_interval`未満ではdriftが大きくても同期しない。
2. `min_interval`到達後、drift threshold超過で同期する。
3. driftが小さくても`max_interval`到達時に同期する。
4. 強制`Sync()`はadaptive条件に関係なく同期する。
5. 同期後にage、drift基準、sync reasonが正しくresetされる。
6. Serial/Pipelineで同じstep入力に対する同期判定境界が一致する。
7. adaptive無効時はPRD036のprofiled動作を維持する。
8. v1が追加action forwardを行わないことをmock networkのforward回数で確認する。
9. configの不正mode、非finite threshold、範囲逆転をfail-fastする。

### Run比較候補

長期最終成績を目的とし、50M到達速度だけで採否を決めない。

1. shared network control
2. fixed interval 200
3. cosine 100 -> 200 / 200M
4. adaptive候補

200Mから300Mを中間確認点とし、最終判断は700Mから1000Mの終盤評価を複数Runで比較する。
途中評価では、差が単なる学習遅延か、Eval傾き・崩壊率・NoDrop率の改善へ変化しているかを確認する。

## 一般事例との位置付け

- Ape-XはActor networkをLearnerから周期的に更新するが、adaptive drift thresholdを中心機構にはしていない。
- R2D2はparameter lag、representational drift、recurrent state stalenessを問題として扱い、Actor parameter update intervalに固定400 environment stepsを使用する。
- IMPALAはtrajectory開始時にActor方策を更新し、残るpolicy lagをV-traceで補正する。
- SEED RLはcentralized inferenceによりActor側parameter配布とlagを減らす方向である。
- Adaptive Policy Synchronizationは、central learnerとのpolicy divergenceが大きくなったworkerだけが更新を要求する方式を提案している。ただし新しく限定的な事例であり、一般的なDQN標準手法とはみなさない。

参考:

- Horgan et al., "Distributed Prioritized Experience Replay": <https://arxiv.org/abs/1803.00933>
- Kapturowski et al., "Recurrent Experience Replay in Distributed Reinforcement Learning": <https://openreview.net/forum?id=r1lyTjAqYX>
- Espeholt et al., "IMPALA": <https://proceedings.mlr.press/v80/espeholt18a>
- Espeholt et al., "SEED RL": <https://openreview.net/forum?id=rkgvXlrKwH>
- Lafuente-Mercado, "Adaptive Policy Synchronization for Scalable Reinforcement Learning": <https://arxiv.org/abs/2507.10990>

## 未決事項

1. 最初のadaptive drift指標を何にするか。
2. Adam系optimizerで追加full parameter scan無しに、十分意味のあるupdate量を取得できるか。
3. drift thresholdをnetwork規模・reward scale・learning rateから独立した無次元量にできるか。
4. adaptive制御の`min_interval` / `max_interval`をどのstep軸で定義するか。
5. `ProfiledValue`をmax ageとして再利用するか、adaptive専用設定へ分けるか。
6. snapshot Stateをarchiveへ保存するか。PRD036同様にloadをfresh run扱いとするか。
7. Actor priority `actor_approx`併用時に、snapshot lagがpriority品質へ与える影響をどう分離するか。
8. episode途中のnetwork切替率を追加forward無しで観測できるか。
9. fixed interval 200のthroughput低下が同期copy頻度に起因するか。
10. global adaptive syncで十分か、最終的にper-lane snapshot versionが必要か。

## 採用判断

本PRDは現時点では実装承認ではない。次の順で判断する。

1. 現行`ProfiledValue`による一方向cosineを長期Run候補として評価する。
2. snapshot interval、learner update age、drift、episode跨ぎをどう観測するか決める。
3. 追加forward無しで成立するdrift指標を小規模prototypeまたは計測で検証する。
4. fixed / cosine / adaptiveの長期最終成績、崩壊率、NoDrop率、実時間を比較する。
5. adaptiveが固定profileを上回る根拠が得られた場合だけ正式番号を採番し、実装PRDへ昇格する。
