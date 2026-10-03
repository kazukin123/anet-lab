# PRD 048: 分位tail探索診断メトリクス

- 起票日: 2026-08-11
- 更新日: 2026-08-12
- 状態: implementation ready
- 対象: DefaultDQN QR / IQN、DropMergeの探索診断
- Topic Issue: 可視化/メトリクス `#13`、DQN `#20`、ハイパラ探索 `#22`
- 関連: PRD 047（IQN探索P0診断メトリクス）、ADR 0019（IQN UQE scoreとfull distribution query）、ADR 0010（Actor priority mean-Q近似）
- 参考: Mavrin et al., [Distributional Reinforcement Learning for Efficient Exploration](https://proceedings.mlr.press/v97/mavrin19a/mavrin19a.pdf)

## Problem Statement

DefaultDQNはQR / IQNのreturn distributionを学習し、IQN+UQEではrisk queryとfull distribution queryを同一forwardから取得できる。PRD 047により、UQE scoreとfull-Qのargmax不一致、Policy marginの有限tau scale、Learnerのpriority signalと有限tau scale、PER健全性を観測できるようになった。

一方、DLTVを起点に探索方法を検討するには、現在のメトリクスだけでは次の点を判断できない。

- 選択した行動のreturn distributionが、medianより上側へどの程度広がっているか。
- 上側だけでなく、medianより下側のリスクがどの程度大きいか。
- 下側tailを負のbonusとして扱った場合に、risk-neutralな行動選択がどの程度変わり得るか。
- 出力されたquantileがtau順に単調であり、upper / lower tailという解釈が妥当か。
- quantile crossingが多数の浅い局所逆転なのか、選択行動にも残る深い逆転なのか。
- upper-tail幅が現行PER raw priorityと同じ経験を強調しており、追加の探索信号として冗長か。

DLTVのupper-tail variabilityは、分布が非対称な場合に全分散よりも探索側の幅を直接捉えられる。ただし、upper-tail幅だけではdownside riskを捉えられず、return distributionの幅にはparametric uncertaintyとintrinsic uncertaintyが混在する。また、分布情報を直ちにPolicy scoreや内的報酬へ組み込むと、QR / IQNの基本評価と探索方式の効果が分離できなくなる。

まず6個の診断メトリクスだけを追加し、QR / IQNの基本評価を変更せず、上側tail、下側tail、quantile ordering、その深さ、PERとの関係をRun artifactへ残せる観測契約が必要である。

## Solution

QR / IQNに共通するtau順のquantile列から、medianを基準とする上下対称のtail幅を定義する。これをPolicy側のfull distributionとLearner側のcurrent distributionへ適用し、6個のscalar keyとして公開する。

本PRDはDLTVを再現しない。DLTVのupper-tail variabilityを観測の出発点とし、下側も同じ定義で測れる対称な診断へ拡張する。メトリクスはPolicy score、loss、PER priority、ReplayBuffer sampling probabilityへ反映しない。

### 共通tail定義

tau順に並んだquantile列を次のように置く。

\[
z_0, z_1, \ldots, z_{K-1}
\]

`K >= 2`とし、`h = floor(K / 2)`とする。median `m`、lower index集合`L`、upper index集合`U`を次のように定義する。

\[
m =
\begin{cases}
(z_{h-1} + z_h) / 2 & (K\text{が偶数}) \\
z_h & (K\text{が奇数})
\end{cases}
\]

\[
L = \{0, \ldots, h-1\}, \qquad
U = \{K-h, \ldots, K-1\}
\]

`K`が奇数の場合、中央のquantile `z_h`は上下どちらにも含めない。上下で常に同数のquantileを使い、次のtruncated standard deviationを求める。

\[
\sigma_+ = \sqrt{\frac{1}{h}\sum_{i \in U}(z_i-m)^2}, \qquad
\sigma_- = \sqrt{\frac{1}{h}\sum_{i \in L}(m-z_i)^2}
\]

`sigma_+`と`sigma_-`はvarianceではなくstandard deviationとして公開する。Q値と同じ単位になり、将来bonusまたはpenalty候補と比較しやすいためである。

quantile crossingが存在しても、tailの所属は予測値による再sortではなくtau順で決める。`quantile_crossing_ratio`を併記し、tau順のtail解釈が壊れていないかを別に観測する。

### `policy_upper_truncated_std`

Policyのfull distributionを`z[b,a,i]`、最終的に実行する行動を`a_exec[b]`とする。

\[
policy\_upper\_truncated\_std
= \operatorname{mean}_b\left(\sigma_+(z[b,a_{exec}[b],:])\right)
\]

- 値域は`0`以上、単位はQ値と同じ。
- epsilon、invalid-action補正などにより`WithAction()`で行動が差し替えられた場合は、差し替え後の行動を使う。
- 選択行動が「良い結果側へ化ける幅」を持つかを観測するが、それをparametric uncertaintyと断定しない。

### `policy_lower_truncated_std`

\[
policy\_lower\_truncated\_std
= \operatorname{mean}_b\left(\sigma_-(z[b,a_{exec}[b],:])\right)
\]

- 値域は`0`以上、単位はQ値と同じ。
- `policy_upper_truncated_std`と同じ最終実行行動を使う。
- 選択行動のdownside幅を観測するが、Policyへ負のbonusを適用しない。
- tail asymmetryは新しいscalarを追加せず、必要に応じてRun分析時に次式から導出する。

\[
A = \frac{\sigma_+ - \sigma_-}{\sigma_+ + \sigma_- + \epsilon}
\]

### `lower_risk_full_q_argmax_disagreement`

各actionのfull distribution平均とlower-tail penalty scoreを次のように定義する。

\[
\mu[b,a] = \operatorname{mean}_i z[b,a,i]
\]

\[
s_{lower}[b,a] = \mu[b,a] - \sigma_-(z[b,a,:])
\]

\[
lower\_risk\_full\_q\_argmax\_disagreement
= \operatorname{mean}_b
\mathbf{1}\left[
\arg\max_a \mu[b,a]
\ne
\arg\max_a s_{lower}[b,a]
\right]
\]

- 値域は`0`から`1`。
- penalty係数は診断用に`1`で固定し、新しい設定キーは追加しない。
- tie時の行動index決定は既存の`argmax`規則に従う。
- 既存の`iqn_uqe_full_q_argmax_disagreement`を置換しない。既存metricは現在のUQEとfull-Qの差、本metricは仮想的なlower-risk scoreとfull-Qの差を測る。

### `quantile_crossing_ratio`

Policyのfull distributionに対し、隣接するtauで値が降下する組の割合を求める。

\[
quantile\_crossing\_ratio
= \frac{1}{B A (K-1)}
\sum_{b,a,i=0}^{K-2}
\mathbf{1}[z[b,a,i] > z[b,a,i+1]]
\]

- 値域は`0`から`1`。
- 同値はcrossingに含めない。
- 全batch、全action、全隣接tauを同じ重みで集約する。
- 値が大きい場合、上下tail幅を分位関数の形として解釈する信頼度が低いと判断する。

### `policy_selected_crossing_depth_p90_ratio`

Policyのfull distributionから最終実行actionのquantile列を取り出し、隣接tauのcrossing深度をaction内分布rangeで正規化する。lane `b`の選択action分布を次のように置く。

\[
\tilde z[b,i] = z[b,a_{exec}[b],i]
\]

\[
d[b,i] = \max(\tilde z[b,i]-\tilde z[b,i+1],0), \qquad
r[b] = \max_i \tilde z[b,i]-\min_i \tilde z[b,i]
\]

`r[b] > 0`かつ`d[b,i] > 0`となる隣接pairだけについて、正規化深度の集合を作る。

\[
V_b = \left\{\frac{d[b,i]}{r[b]} \mid d[b,i] > 0\right\}
\]

各laneでは、`V_b`を昇順に並べたnearest-rank 90 percentileを使う。`n=|V_b|`、1始まりのindexを`k=ceil(0.9n)`とし、`V_b`の第`k`要素を`P90_b`とする。crossingがない場合、または`r[b] = 0`の場合は`P90_b = 0`とする。

\[
policy\_selected\_crossing\_depth\_p90\_ratio
= \operatorname{mean}_b(P90_b)
\]

- 値域は`0`から`1`で、無次元量とする。
- `quantile_crossing_ratio`が全actionの発生頻度を測るのに対し、本metricは最終実行actionで生じた深いcrossingを強調する。
- positive crossingだけをpercentile母集団とするため、crossing件数の多寡とは別に深さを解釈する。crossingなしを正常値`0`として扱う。
- epsilon、invalid-action補正などにより`WithAction()`で行動が差し替えられた場合は、差し替え後の行動を使う。
- 値によってquantile列を再sortしない。sortするのはpositive crossing深度のpercentile選択だけである。
- TBO有効時はh-space内でrange正規化する。real-spaceへ逆変換しない。
- 本metricは深いcrossingの経過観察用であり、Policy score、loss、priorityへ反映しない。
- 公開scalarは各action event内のlane別p90をbatch平均した値であり、Run全期間のcrossing sampleをpoolして求めたp90ではない。既定tagのEMAは、このevent scalarを時系列平滑化する。

### `upper_tail_priority_spearman`

Learner minibatchの各sampleについて、経験に記録されたactionのcurrent quantile列から`upper_tail_std[b]`を求める。QRではheadのquantile index順、IQNでは`current_taus`を昇順に並べた順序へcurrent quantileを同じpermutationで並べ替えてから共通tail定義を適用する。

現行のraw PER priorityを次のように置く。

\[
p[b] = \operatorname{clip}(|\delta_{mean}[b]| + per\_eps)
\]

clip無効時は上限を適用しない。`per_alpha`を適用する前の、ReplayBufferへ渡すraw priorityを使う。

\[
upper\_tail\_priority\_spearman
= \operatorname{Corr}
\left(
\operatorname{rank}_{avg}(upper\_tail\_std),
\operatorname{rank}_{avg}(p)
\right)
\]

- 値域は`-1`から`1`。
- tieには平均順位を割り当てる。
- PER無効、minibatch sizeが`2`未満、どちらかの順位列の分散が`0`の場合は`NaN`。
- 相関は現在のPER samplingで取得されたminibatch内の条件付き相関であり、ReplayBuffer全体の相関ではない。
- 高い正相関はupper-tail幅と現行priorityの選別が近いことを示す。低相関または負相関だけでは、upper-tail幅が有用な探索信号であることを意味しない。

### QR / IQNの入力契約

- QR Policyでは、固定quantile headが返す`q_quantiles`をfull distributionとして使う。
- IQN Policyでは、`full_distribution_query.enabled=true`かつ`sample_mode=fixed`の`full_q_quantiles`だけを使う。
- IQNのfull queryが無効、またはfixed以外の場合、Policy側5metricは`NaN`を返す。
- Learner側はQR / IQNとも既存current forwardの、経験actionに対応するquantile列を使う。
- `K < 2`または必要なTensorが未定義の場合、該当metricは`NaN`を返す。
- TBO有効時も、Policy scoreとPER priorityの計測空間に合わせてh-spaceで算出する。real-spaceへ逆変換しない。

### DropMergeの既定tag

新規scalar keyと既定tagは6個だけ追加する。Policy側は探索Policyを評価する`eval2`だけへ登録し、`eval1`へ複製しない。

| Tag | Scalar key | Event / aggregation |
|---|---|---|
| `52_eval2/56_policy_upper_truncated_std_ema` | `policy_upper_truncated_std` | `eval2` action、EMA `0.01`、`interval:100` |
| `52_eval2/57_policy_lower_truncated_std_ema` | `policy_lower_truncated_std` | `eval2` action、EMA `0.01`、`interval:100` |
| `52_eval2/58_lower_risk_full_q_argmax_disagreement_ema` | `lower_risk_full_q_argmax_disagreement` | `eval2` action、EMA `0.01`、`interval:100` |
| `52_eval2/59_quantile_crossing_ratio_ema` | `quantile_crossing_ratio` | `eval2` action、EMA `0.01`、`interval:100` |
| `52_eval2/60_policy_selected_crossing_depth_p90_ratio_ema` | `policy_selected_crossing_depth_p90_ratio` | `eval2` action、EMA `0.01`、`interval:100` |
| `36_agent_quantile/01_upper_tail_priority_spearman_ema` | `upper_tail_priority_spearman` | Learner update、EMA `0.01`、`interval:100` |

raw版、Train Policy版、`eval1`版、LunarLander版は既定登録しない。

## User Stories

1. As a distributional RL experimenter, I want to observe the upper-tail width of the selected action, so that I can judge whether the learned distribution contains meaningful upside information.
2. As a risk-sensitive RL experimenter, I want to observe the lower-tail width of the selected action, so that I can distinguish upside potential from downside risk.
3. As a DropMerge analyst, I want upper and lower tail widths to use the same median-centered definition, so that their asymmetry can be compared without a definition mismatch.
4. As a DropMerge analyst, I want tail widths in Q-value units, so that I can compare them with existing Q margins and candidate bonus magnitudes.
5. As an exploration researcher, I want a hypothetical lower-risk action score diagnostic, so that I can see whether a downside penalty could materially change action selection before changing the Policy.
6. As an exploration researcher, I want the lower-risk coefficient fixed at one for this diagnostic, so that Runs remain comparable without adding another hyperparameter.
7. As an IQN experimenter, I want the existing UQE/full-Q disagreement and the new lower-risk/full-Q disagreement to remain separate, so that upside exploration and downside avoidance are not conflated.
8. As a quantile-model maintainer, I want to measure adjacent quantile crossing, so that I can tell whether upper and lower tail interpretations are trustworthy.
9. As a quantile-model maintainer, I want equal adjacent values excluded from crossing, so that flat regions are not reported as ordering violations.
10. As a PER experimenter, I want the upper-tail width ranked against the actual raw PER priority, so that I can evaluate whether both signals select similar experiences.
11. As a PER experimenter, I want Spearman ties handled with average ranks, so that clipping-induced equal priorities do not receive arbitrary ordering.
12. As a PER experimenter, I want undefined Spearman cases reported as `NaN`, so that constant or insufficient batches are not presented as zero correlation.
13. As a QR experimenter, I want the diagnostics to use the existing fixed quantile head, so that the metrics do not depend on adopting IQN.
14. As an IQN experimenter, I want Policy diagnostics based on a fixed full-range query, so that comparisons do not mix distribution shape with newly sampled tau placement.
15. As an IQN experimenter, I want random current taus ordered together with their predicted values in the Learner diagnostic, so that upper-tail membership follows tau rather than output position.
16. As an action-processing maintainer, I want selected-action tail metrics to follow the final action after `WithAction()`, so that metrics describe the action actually sent to the Env.
17. As a TBO experimenter, I want tail and priority diagnostics in the same h-space as the mechanisms they explain, so that no nonlinear inverse transform changes their relationship.
18. As a baseline runner, I want exactly six new default tags, so that the metrics view remains compact.
19. As a baseline runner, I want Policy metrics only on `eval2`, so that target evaluation and Policy evaluation are not duplicated by default.
20. As a LunarLander experimenter, I want no automatic metric registration change, so that the current LunarLander baseline remains unchanged.
21. As a performance maintainer, I want diagnostics to reuse existing forward tensors, so that no model inference is added.
22. As a performance maintainer, I want Policy scalar readback shared by one lazy packed transfer, so that separate tags do not add separate GPU synchronization points.
23. As a performance maintainer, I want Learner tail data transferred with the existing PER readback, so that the Spearman metric does not add a new wait boundary.
24. As an algorithm maintainer, I want all diagnostics detached from the learning graph, so that action, loss, priority, sampling, and RNG behavior remain unchanged.
25. As a reproducibility reviewer, I want the six tags visible in resolved Run configuration, so that `config/config_data.txt` proves whether the diagnostics were active.
26. As an analyst, I want unavailable inputs to produce `NaN`, so that QR/IQN configuration differences do not silently become fabricated zeroes.
27. As an experimenter, I want throughput impact checked with a paired smoke, so that diagnostic overhead does not invalidate long-running comparisons.
28. As an algorithm designer, I want these values to remain observations only, so that QR/IQN baseline evaluation can finish before introducing DLTV, intrinsic rewards, or learned priorities.
29. As a quantile-model maintainer, I want the selected action's normalized crossing-depth p90, so that persistent deep crossings can be distinguished from numerous shallow local crossings during baseline observation.

## Implementation Decisions

1. 上下tail幅、median、crossingを計算する共通の純粋Tensor helperをDQN機能グループ内へ置き、PolicyとQR / IQN Learnerから再利用する。新しいcomponentファイルは作らない。
2. 共通helperはtau順の分布を入力とし、値による再sortを行わない。IQN Learnerだけは既存`current_taus`の昇順permutationをcurrent quantileへ適用してから渡す。
3. Policy側は、QRでは既存`q_quantiles`、IQNではfixed full queryの`full_q_quantiles`を診断用full distributionとして選択する。診断のためのforwardやtau生成を追加しない。
4. Policy診断payloadは、最終行動を後からgatherできるper-lane / per-actionの上下tail幅、診断元full distributionへのdetached Tensor alias、full distribution全体から求めるdisagreement / crossingを保持する。aliasのためにquantile値を複製しない。5個のscalarは初回参照時に1本へpackしてCPU materializeし、同じActionInfo内でcacheを共有する。
5. 行動差し替え後のActionInfoは同じ診断payloadを再利用するが、scalar cacheは引き継がず、差し替え後の行動から上下tail幅とcrossing深度p90を再集約する。
6. `DQNActionInfo::GetScalar()`の新しい観測interfaceは`policy_upper_truncated_std`、`policy_lower_truncated_std`、`lower_risk_full_q_argmax_disagreement`、`quantile_crossing_ratio`、`policy_selected_crossing_depth_p90_ratio`とする。
7. QR / IQN Learnerは既存current forwardから、経験actionのupper-tail幅をsample単位でfloat32集約する。学習graphからdetachし、quantile regression lossへ接続しない。
8. Learnerのsample単位upper-tail幅は、PER有効時だけ既存raw priorityと同じpacked device-to-host readbackへ同梱する。readback完了後、CPU上で平均順位tie補正とSpearman相関を計算する。
9. `BatchUpdateResult::GetScalar()`の新しい観測interfaceは`upper_tail_priority_spearman`とする。PER無効時は計算用readbackを増やさず`NaN`を返す。
10. raw priorityの値、clip件数、ReplayBufferへ渡すpriority配列、`per_alpha`、sampling tree更新順序は変更しない。
11. 診断集約はfloat32で行う。AMP / BF16のquantile Tensorをそのまま分散集約しない。
12. 公開設定キー、checkpoint形式、ReplayBuffer item、Actor priority hint、公開TensorDict schemaは変更しない。
13. metric登録はDropMergeへ6本だけ追加し、すべてEMA `0.01`、`interval:100`とする。raw、Train、`eval1`、LunarLanderへの重複登録は行わない。
14. 既存scalar keyの名称、値、`NaN`条件を変更しない。PRD 047とADRは履歴として書き換えない。
15. 実装時はDQN設計、可観測性、Run分析ガイドへ式、Q-space、`NaN`条件、Spearmanのsampling bias、crossingがtail解釈へ与える制約を反映する。
16. 新しい探索Policy、内的報酬、PER priority方式を追加しないため、新規ADRは要求しない。実装中に本PRDの観測契約を越える設計判断が必要になった場合だけ、別PRDまたはADRへ分離する。
17. `policy_selected_crossing_depth_p90_ratio`のpositive-depth抽出とnearest-rank選択は、Policy側5metricのいずれかを初めて参照した時だけdevice上で行う。毎action生成時にはpercentile sortを追加せず、算出したscalarは既存Policy診断と同じpacked readbackへ同梱する。

## Testing Decisions

テストは内部payloadの配置ではなく、公開`GetScalar()`、公開Learner更新経路、設定解決結果から観測できる振る舞いを検証する。production codeへtest-only APIを追加しない。

### 数値契約

- 偶数列`[0, 1, 2, 3]`について、medianが`1.5`、上下とも`sqrt(1.25)`になることを検証する。
- 上側長尾、下側長尾の分布で、対応するtruncated stdだけが大きくなることを検証する。
- 奇数`K`で中央quantileが上下集合から除外され、上下の要素数が一致することを検証する。
- `K = 2`を最小有効ケース、`K < 2`を`NaN`ケースとして検証する。
- full-Q argmaxと`mean - sigma_-` argmaxが一致するケース、不一致になるケース、tieを含むケースを検証する。
- crossingなし、一部crossing、全隣接crossing、隣接同値を含む分布について、手計算した比率と一致することを検証する。
- 選択actionの列`[0, 2, 1, 4, 2]`について、positive crossingの正規化深度が`[0.25, 0.5]`、nearest-rank p90が`0.5`になることを検証する。
- 選択actionにcrossingがない場合、および全quantileが同値でrangeが`0`の場合、`policy_selected_crossing_depth_p90_ratio`が`0`になることを検証する。`K < 2`は他のPolicy側quantile診断と同じく`NaN`とする。
- Spearmanが完全同順で`+1`、完全逆順で`-1`になることを検証する。
- Spearmanの両側にtieがあるケースで平均順位による期待値と一致することを検証する。
- minibatch sizeが`2`未満、upper-tail幅が定数、raw priorityが定数、PER無効の場合に`NaN`になることを検証する。
- priority clipでtieが発生した場合、clip後の実際のraw priority順位を使うことを検証する。

### Policy公開契約

- QR Policyが既存`q_quantiles`から5metricを返すことを検証する。
- IQN Policyがfixed full queryから5metricを返すことを検証する。
- IQN full query無効時、およびfull queryがfixed以外の場合に5metricが`NaN`になることを検証する。
- 複数action / 複数laneで、選択actionに対応する上下tail幅がbatch平均されることを検証する。
- 複数action / 複数laneで、最終実行actionごとのcrossing-depth p90を先に求め、そのlane平均が返ることを検証する。
- `WithAction()`後は上下tail幅とcrossing-depth p90が差し替え後のactionへ追従し、disagreement / crossing ratioはfull distribution全体の値を維持することを検証する。
- 5つのscalarを複数回、異なる順序で取得しても同じ値を返すことを検証する。

### Learner公開契約

- QR Learnerが固定quantile index順、IQN Learnerが`current_taus`順に並べたcurrent quantileからupper-tail幅を求めることを検証する。
- IQNのtau入力順を入れ替えても、tauとquantileを同じpermutationで並べ替えた結果が一致することを検証する。
- `upper_tail_priority_spearman`が、ReplayBufferへ実際に渡されるraw priorityと同じclip後の値を使うことを検証する。
- TBO有効時にtail幅とraw priorityの両方がh-spaceのまま比較されることを検証する。

### 非干渉・性能契約

- metric追加前後でPolicy action、IQN / QR loss、raw priority、ReplayBufferへ反映されるpriority、RNG列が不変であることを回帰テストする。
- Policy / Learnerともforward回数が増えないことをfocused testまたはprobeで確認する。
- Policyの5metricが1つのCPU cacheを共有し、Learnerのsample単位tail幅が既存PER readbackへ同梱されることを確認する。
- Policyのpercentile sortがPolicy側5metricの初回参照時だけ実行され、action生成時とcache再参照時には実行されないことをfocused testまたはprofile probeで確認する。
- CPU、およびCUDA利用可能時にshape、dtype、device契約を維持することを検証する。
- AMP / BF16有効時も診断結果がfloat32で集約されることを検証する。
- 同一binary・同一条件で診断追加前後の短時間paired smokeを行い、`exp_step_per_sec`低下が2%以内であることを受け入れ基準とする。2%を超えた場合は、追加forwardではなくsort、集約、readback payload、同期位置を先に調査する。

### 設定・総合検証

- DropMergeの解決済みscalar設定に、新規scalar keyが6個、既定tagが6本だけ存在することを検査する。
- 5本が`eval2`、1本がLearnerに属し、すべてEMA `0.01`、`interval:100`であることを検証する。
- `eval1`、Train Policy、LunarLanderに新規tagが登録されていないことを検証する。
- Debug buildを通す。
- DQN ActionInfo、QR Learner、IQN Learnerのfocused testを通す。
- 設定解決smokeとtag重複検査を通す。
- `git diff --check`を通す。

## Out of Scope

- DLTVの探索bonusまたは時間減衰scheduleの実装。
- `mean + sigma_+`、`mean - sigma_-`、上下を混合したscoreによる実際のPolicy変更。
- NGU、RND、ICMなどの内的報酬、探索専用Policy、複数Policyの混合。
- 内的報酬または探索bonusを生成するサブネットワーク。
- FQF、PQR、QUOTA、risk-conditioned Policyの実装。
- PER priority式、`per_alpha`、`per_beta`、initial priority、priority sourceの変更。
- upper-tail幅を直接PER priorityとして使用すること、またはpriorityを出力する学習サブネットワーク。
- parametric uncertaintyとintrinsic uncertaintyをensemble等で分離すること。
- `tail_asymmetry`、全分散、upper-risk/full-Q disagreement、repeated-forward jitter、calibrationの追加scalar。
- selected-action crossing ratio、全action crossing-depth percentile、crossing-depth mean / RMS、raw crossing-depthの追加scalar。
- metricのTrain Policy版、`eval1`版、raw版、LunarLander既定tag。
- Metrics Viewer UI、Optuna objective、Run比較ロジックの変更。
- QR / IQNのハイパラ探索、探索方式の採用判断、実験Runの実施。
- PRD 047または既存ADRの履歴修正。
- コード、設定、設計文書、実験記録の変更。本PRD更新タスクではこのファイルだけを変更する。

## Further Notes

- `sigma_+`と`sigma_-`は学習済みreturn distributionの形を測る。単一QR / IQN networkの分布幅だけからparametric uncertaintyとintrinsic uncertaintyを分離できない。
- Policy側metricはfixed full queryを使うため、IQNのrisk queryに使ったtau集合そのものの幅ではない。行動scoreと独立した観測用full distributionの形を測る。
- `quantile_crossing_ratio`が高いRunでは、upper / lower tailを分位関数の領域として解釈する前にquantile orderingの問題を確認する。
- `quantile_crossing_ratio`は全actionの発生頻度、`policy_selected_crossing_depth_p90_ratio`は選択actionのconditional depthを測る。ratioが横ばいでもp90が低下すれば浅い局所crossingへ収束している可能性があり、ratioが低下してp90が上昇すれば少数の深いcrossingが残っている可能性がある。
- `upper_tail_priority_spearman`はPERにより既に偏って抽出されたminibatch内の相関である。ReplayBuffer全体の冗長性や、upper-tail priorityを採用した場合の性能を直接示さない。
- Spearmanが高い場合は、upper-tail幅が現行mean-TD priorityと似た経験を順位付けしている可能性が高い。低い場合は信号が異なるだけであり、探索や学習に有益とは限らない。
- 上下tail幅とlower-risk disagreementは、将来の探索bonus、risk penalty、探索専用Policyを検討するための観測材料である。本PRDの段階では方策へ載せない。
- Run分析では6metricだけで採用判断せず、同一step範囲の報酬、Double Suika成果、既存UQE/full-Q disagreement、PER健全性、throughputと合わせて解釈する。
- Run名や編集後の設定ではなく、artifactの`config/config_data.txt`を実効設定の正本とする。
- 実装は本PRDとは別タスクで行い、その際にコード、DropMerge設定、設計文書、テストを同じ変更内で整合させる。
- 本PRDの更新では外部Issue作成、stage、commit、pushを行わない。
