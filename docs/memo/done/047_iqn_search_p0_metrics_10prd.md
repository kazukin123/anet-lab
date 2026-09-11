# PRD 047: IQN探索P0診断メトリクス

- 起票日: 2026-08-11
- 状態: implementation ready
- 対象: DefaultDQN IQN、DropMerge / LunarLander のベースライン探索
- Topic Issue: ハイパラ探索 `#22`、DQN `#20`
- 関連: PRD 001（IQN）、PRD 035（PER priority source）、PRD 044（IQN stratified τ sampling）

## Problem Statement

IQN探索のベースラインRunでは、最終報酬だけでなく、次の問いを同一Runのartifactから判断できる必要がある。

- DropMergeでDouble Suikaへ到達しているか、また何回生成しているか。
- Policy側のτ数 `K` が行動選択を安定させるのに十分か。
- Learner側のcurrent quantile数 `N` とtarget quantile数 `M` が、priority信号を評価するのに十分か。
- ReplayBufferへ投入された遷移が初めてLearner更新を受ける時点で、IQNのpairwise TD誤差がpriorityへどの程度反映されているか。
- PERが初期priorityへ偏りすぎず、十分な有効サンプル数を保っているか。
- 診断を有効にしたときも、長時間のDropMerge探索に必要なthroughputを維持できるか。

現状の観測だけでは、これらを十分に分離できない。

- `q_std` はreturn distributionの幅であり、τサンプリングによる推定誤差そのものではない。
- IQNのpriorityに使う平均TDは、quantile pair間の大きな誤差が正負で相殺された場合に、その大きさを隠し得る。
- UQEのrisk scoreに基づくmarginだけでは、期待値としてのfull Qとrisk-sensitiveな行動選択の差を判定できない。
- Double Suika生成数はEnv内部では把握できるが、Run比較に使える直接のscalar契約がない。
- PERの詳細metricは主に`metrics.scalar.full`へ含まれており、ベースライン探索で必要な項目だけを低負荷で有効化しにくい。

このPRDでは、学習挙動を変更せず、DropMerge成果、IQNの`K/N/M`、初回Learner priority更新、PER健全性、throughputをP0として観測するための契約を定める。

## Solution

既存のEnv、Policy、Learner、PER、Runnerの公開scalar経路を拡張し、IQN探索向けのP0診断metricを追加する。診断は既存forwardとIQN loss計算で得られるTensorを再利用し、学習loss、priority、sampling probability、action、RNG消費を変えない。

また、既存の`metrics.scalar.full`から探索判断に必要なmetricだけを選んだ`metrics.scalar.iqn_search_p0`グループを追加する。DropMergeとLunarLanderのIQN探索設定はこのグループを合成し、`metrics.scalar.full`全体を有効化せずにP0診断を記録できるようにする。

### DropMerge成果metric

`DropMergeEnv::GetScalar()`に次のキーを追加する。

| Metric | 値 | 記録契約 |
|---|---|---|
| `ep_double_suika_created` | 終了したepisode内のDouble Suika生成数 | episode終了時のみ値を返し、それ以外は`NaN` |
| `ep_double_suika_achieved` | 生成数が1以上なら`1`、それ以外は`0` | episode終了時のみ値を返し、それ以外は`NaN` |

Train/Evalの双方で`exp_step`軸に記録する。生成数はraw値を残し、達成有無は達成率としてEMAを残す。episode終了外の`NaN`は非イベントを0件として扱わないための契約である。

### Policy K診断metric

Policy側では、行動選択に用いた既存のrisk quantileと、既存のfull-distribution queryが返した`full_q_values`を再利用する。

#### `iqn_policy_margin_mc_ratio`

IQN+UQEにおける上位2行動のscore gapを、各行動のquantile推定scaleで正規化する。

batch行`b`について、UQE score上位2行動を`a1`、`a2`、risk quantile数を`K`とし、次を計算する。

```text
s[b,a] = std_k(risk_quantiles[b,a,k]) / sqrt(K)
gap[b] = uqe_values[b,a1] - uqe_values[b,a2]
ratio[b] = gap[b] / (sqrt(s[b,a1]^2 + s[b,a2]^2) + 1e-6)
iqn_policy_margin_mc_ratio = mean_b(ratio[b])
```

標準偏差はfloat32・不偏分散で計算する。`random`ではMonte Carlo平均の安定度、`fixed` / `stratified`では積分解像度のproxyとして解釈する。IQN+UQE以外、または`K < 2`では`NaN`とする。

#### `iqn_uqe_full_q_argmax_disagreement`

```text
mean_b(argmax(uqe_values[b,:]) != argmax(full_q_values[b,:]))
```

UQEによるrisk-sensitiveな選択と、full distributionの期待値による選択が食い違うbatch比率を表す。full-distribution queryがない場合は`NaN`とする。

#### `action_full_q_margin.[i]`

action index `i`ごとに、full Qの他行動に対するmarginを記録する。

```text
mean_b(full_q_values[b,i] - max_{a != i}(full_q_values[b,a]))
```

これにより、NOOPを含む各行動の期待値側marginをUQE marginから分離する。full-distribution queryがない場合は`NaN`とする。解決後のaction数に対してindexが不正な場合は、対象indexと有効範囲を含むエラーでfail-fastする。

### Learner N/M・priority診断metric

current quantileを`z[b,i]`、target quantileを`y[b,j]`とし、`delta[b,i,j] = y[b,j] - z[b,i]`とする。current quantile数は`N`、target quantile数は`M`である。

各batch行のscaleを次のように定義する。

```text
current_scale[b] = std_i(z[b,i]) / sqrt(N)
target_scale[b]  = std_j(y[b,j]) / sqrt(M)
```

標準偏差はfloat32・不偏分散で計算する。`N < 2`ではcurrent scaleとそれを必要とするratio、`M < 2`ではtarget scaleとそれを必要とするratioを`NaN`とする。

| Metric | 計算 | 意味 |
|---|---|---|
| `iqn_current_mc_scale` | `mean_b(current_scale[b])` | current側のτ有限本数による推定scale |
| `iqn_target_mc_scale` | `mean_b(target_scale[b])` | target側のτ有限本数による推定scale |
| `iqn_priority_mc_ratio` | `mean_b(abs(mean_i(z[b,i]) - mean_j(y[b,j])) / (sqrt(current_scale[b]^2 + target_scale[b]^2) + 1e-6))` | 現行priority信号がτ推定scaleに対してどの程度大きいか |
| `iqn_first_priority_mc_ratio` | 初回Learner更新行だけの`iqn_priority_mc_ratio` | Replay投入後、最初の更新時点におけるpriority信号の識別度 |
| `iqn_first_pair_abs_td` | 初回Learner更新行に対する`mean_ij(abs(delta[b,i,j]))`の平均 | 相殺前のpairwise TD誤差の大きさ |
| `iqn_first_cancellation_ratio` | 初回Learner更新行に対する`clamp(1 - abs(mean_ij(delta[b,i,j])) / (mean_ij(abs(delta[b,i,j])) + 1e-6), 0, 1)`の平均 | pairwise誤差が平均TDで相殺される割合 |
| `iqn_first_quantile_loss_norm` | 現行IQN sample lossを`N`で除算し、初回Learner更新行だけで平均 | `N`探索に伴うloss総和scaleの変化を除いた初回loss |
| `per_sample_initial_count` | minibatch内の初回Learner更新行数 | `per_sample_initial_ratio`の分母規模と初回系metricの成立性 |

「初回Learner更新行」は、sample時点のpriority sourceが`fixed_initial`、`max_initial`、`actor_initial`のいずれかである行とする。`none`と`learner_updated`は除外する。該当行が0件の場合、`iqn_first_*` metricは`NaN`とし、`per_sample_initial_count`は`0`を記録する。

PER無効時はpriority sourceを`none`として扱うため、`per_sample_initial_count`は`0`、`iqn_first_*` metricは`NaN`となる。`iqn_current_mc_scale`、`iqn_target_mc_scale`、`iqn_priority_mc_ratio`はPERの有効・無効に依存せず、IQN quantileが得られる場合は同じ数値契約で計算する。

TBO有効時も、これらの診断は現行priorityと同じh空間で計測する。priority式やTBOの空間変換自体は変更しない。

### `metrics.scalar.iqn_search_p0`グループ

新しい`metrics.scalar.iqn_search_p0`グループへ、既存の`metrics.scalar.full`から次のmetricだけを登録する。

- `per_sample_initial_ratio`
- `replaybuffer.per.initial_mass_ratio`
- `replaybuffer.per.last_evicted_never_sampled_ratio`
- `per_batch_prio_mean`
- `per_prio_cv`
- `per_prio_ess_ratio`
- `per_is_ess_ratio`
- `per_prio_clip_ratio`
- `exp_step_per_sec`
- `elapse_hour`

DropMergeとLunarLanderのIQN探索設定はこのグループを合成する。新規IQN診断metricは各Agent設定のscalar選択へ、DropMerge成果metricはDropMergeのTrain/Eval設定へ登録する。

Learner/PER系metricは`exp_step`軸、原則`interval: 100`で出力する。初回系metricは全イベントでEMA状態を更新し、100 step間隔で出力する。raw値とEMAの無条件な二重記録は行わず、判断に必要な集約だけを選択する。`metrics.scalar.full`全体は有効化しない。

## User Stories

1. As an IQN experimenter, I want to record Double Suika creation counts directly, so that I can evaluate terminal-quality progress without decoding it from another Env metric.
2. As an IQN experimenter, I want to record Double Suika achievement as an EMA rate, so that I can compare rare-event success across baseline Runs.
3. As a DropMerge analyst, I want non-terminal steps to emit `NaN` for episode outcomes, so that missing episode events are not counted as failures.
4. As an experimenter, I want Train and Eval outcome metrics on the same `exp_step` axis, so that I can compare learning progress and evaluation quality.
5. As an IQN experimenter, I want a normalized Policy margin metric, so that I can judge whether `K` is large enough relative to τ-sampling uncertainty.
6. As an IQN experimenter, I want the Policy metric to distinguish `random` from `fixed` / `stratified` interpretation, so that I do not mistake an integration-resolution proxy for measured randomness.
7. As an UQE experimenter, I want to see disagreement between risk-sensitive and full-Q argmax, so that I can identify when UQE changes the selected action.
8. As a DropMerge analyst, I want full-Q margins per action, so that I can separate NOOP expectation from UQE risk margin.
9. As a configuration author, I want invalid action indices to fail fast, so that a malformed metric selection cannot silently corrupt a Run.
10. As an IQN experimenter, I want separate current and target τ scale metrics, so that I can explore `N` and `M` independently.
11. As an IQN experimenter, I want priority signal normalized by current/target τ scale, so that I can tell signal strength from finite-τ estimation scale.
12. As a PER experimenter, I want the normalized priority signal specifically on first Learner updates, so that I can assess whether initial samples receive an informative first priority.
13. As a PER experimenter, I want pairwise absolute TD error on first updates, so that I can observe distributional error before cancellation.
14. As a PER experimenter, I want a cancellation ratio on first updates, so that I can detect cases where mean TD understates pairwise IQN error.
15. As an IQN experimenter, I want first-update quantile loss normalized by `N`, so that changing `N` does not create a misleading loss-scale comparison.
16. As an analyst, I want the number of first-update rows alongside their ratio, so that I can judge whether a first-update aggregate has enough samples.
17. As a replay-system maintainer, I want the first-update mask to follow priority-source identity, so that fixed, max, and actor initial sources share one explicit contract.
18. As a TBO experimenter, I want diagnostics measured in the same h space as priority, so that metric values remain comparable with the mechanism they explain.
19. As an experimenter, I want unsupported or statistically undefined diagnostics to return `NaN`, so that dashboards do not present fabricated zeroes.
20. As a baseline runner, I want a compact P0 metric group, so that I can enable the required PER and throughput observations without enabling `metrics.scalar.full`.
21. As a baseline runner, I want Learner/PER metrics at a bounded output interval, so that long DropMerge Runs remain practical.
22. As a performance maintainer, I want diagnostics to reuse existing forward and loss tensors, so that metrics do not add model inference or pairwise Tensor construction.
23. As a performance maintainer, I want scalar readback packed into existing synchronization points, so that each metric does not introduce a GPU synchronization.
24. As an algorithm maintainer, I want diagnostics detached from gradients, so that enabling metrics cannot change loss, priority, action, sampling, or RNG behavior.
25. As a reproducibility reviewer, I want the metric group visible in resolved Run config, so that `config/config_data.txt` proves which observations were active.
26. As a test maintainer, I want metric contracts verified through public Env, Agent, and configuration paths, so that tests do not introduce production APIs solely for instrumentation.
27. As a long-run experimenter, I want P0 metrics to cost no more than 2% throughput in a short paired smoke, so that diagnostic coverage does not invalidate exploration speed assumptions.

## Implementation Decisions

1. IQN loss計算内に既に存在する`B x N x M`の`delta`を再利用し、診断のためにpairwise Tensorを再生成しない。
2. 診断値は学習グラフからdetachする。学習loss、PER priority、sampling probability、ReplayBuffer更新値、action、RNG消費は一切変更しない。
3. Learner診断のdevice-to-host転送は1本のpacked readbackへまとめる。PER有効時は既存priority readbackへ同梱し、metricごとの`.item()`や追加同期を発生させない。
4. Policy診断は既存forwardのrisk/full quantileを再利用し、診断用forwardを追加しない。複数scalarは1回のlazy CPU materializeへまとめる。
5. 実装は既存DQN機能グループと`DropMergeEnv`内へ収める。新規componentファイル、設定flag、ADRは追加しない。
6. `BatchUpdateResult`、`DQNActionInfo`、`DropMergeEnv::GetScalar()`のキーを新しい観測interfaceとする。既存キーの名称と意味は変更しない。
7. metricの有効化は既存scalar metric catalogと設定合成で行う。C++側から外部catalogファイルを読む依存は追加しない。
8. 実装時に`CONTEXT.md`へ「初回Learner priority更新」を追加する。DQN設計、可観測性、Run分析ガイドへ式、計測空間、`NaN`条件、`fixed` / `stratified`での解釈を反映する。
9. τ生成値のhistogram、mean、min/maxは追加しない。τ配置そのものはPRD 044の生成契約と単体テストで保証する。
10. 公開設定キー、checkpoint形式、TensorDict契約は変更しない。

## Testing Decisions

### 数値契約

- IQN診断計算をCPU Tensorの手計算値と比較する。`N != M`、相殺率`0`、相殺率`1`、sample lossの`/N`正規化を含める。
- float32・不偏分散によるscaleを検証する。
- first maskが`fixed_initial`、`max_initial`、`actor_initial`を含み、`none`、`learner_updated`を除外することを検証する。
- first maskが0件の場合に`iqn_first_*`が`NaN`、`per_sample_initial_count`が`0`になることを検証する。
- `K < 2`、`N < 2`、`M < 2`、full query不在、非IQN、IQN+UQE以外、PER無効時について、各metricの値または`NaN`契約を検証する。

### Policy・Env公開契約

- UQE score gap、full-Q argmaxの一致/不一致、NOOPを含む`action_full_q_margin.[i]`を既存ActionInfo metricテストへ追加する。
- 不正action indexがfail-fastし、エラーに指定indexと有効範囲を含むことを検証する。
- Double Suika生成数、達成有無、episode終了外の`NaN`を、Envの`Reset()`、`Step()`、`GetScalar()`から検証する。test-only production APIは追加しない。

### 非干渉・性能契約

- metric追加前後でIQN loss、raw priority、ReplayBufferへ反映されるpriority、action、RNG列が不変であることを回帰テストする。
- forward回数が増えず、診断用pairwise計算が重複しないことをfocused testまたはprobeで確認する。
- CPU、およびCUDA利用可能時にshape、dtype、device契約を維持することを検証する。
- BF16学習時も診断集約がfloat32で行われることを検証する。
- 同一binary・同一条件でP0 metric group OFF/ONの短時間paired smokeを行い、`exp_step_per_sec`低下が2%以内であることを受け入れ基準とする。2%を超えた場合はreadbackと同期設計を修正する。

### 設定・総合検証

- metric catalogをC++テストの外部ファイル依存へ戻さず、設定解決smokeとtag重複検査でDropMerge/LunarLanderへの登録を確認する。
- Debug buildを通す。
- DQNのIQN診断・action-policy metricに関するfocused testを通す。
- DropMergeEnv testを通す。
- `git diff --check`を通す。

## Out of Scope

- 実際のPER priority式の変更。
- 固定初期priority `0.3`、`actor_approx`、`per_alpha`、`per_beta`の変更または探索。
- `K`、`N`、`M`、BatchSize、ReplayRatio、学習率、終端stepの変更または採用判断。
- τ配置方式の変更、PRD 044の実装、既定modeの変更。
- quantile crossing、Spearman相関、repeated-forward jitter、calibrationなどのP1/P2診断metric。
- Metrics Viewer UI、Optuna scoring、artifactサイズmetric。
- 実験Runの実施、結果分析、ハイパラ採用判断。
- コード、設定、設計文書、実験記録の変更。本PRD作成タスクではこのファイルだけを新設する。

## Further Notes

- このPRDのmetricは診断用であり、`K/N/M`やPER方式の最適値を直接決めるものではない。報酬、Double Suika成果、PER健全性、throughputと合わせて判断する。
- `iqn_priority_mc_ratio`は現行priority式を変更せず、その平均TD信号を有限τのscaleと比較する。`iqn_first_pair_abs_td`と`iqn_first_cancellation_ratio`は、将来の「初回優先度のquantileペア考慮」を議論するための観測根拠である。
- `fixed` / `stratified`ではforwardごとのMonte Carlo分散を直接測る指標ではないため、`*_mc_*`は積分解像度proxyとして解釈する。この制約をRun分析ガイドにも明記する。
- Run分析ではRun名や編集後の設定ではなく、artifactの`config/config_data.txt`を実効設定の正本とする。
- 実装は本PRDとは別タスクで行い、その際にコード、設定、設計文書、テストを同じ変更内で整合させる。
- 本PRDの作成では外部Issue作成、stage、commit、pushを行わない。
