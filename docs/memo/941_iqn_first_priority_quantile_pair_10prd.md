# PRD 999: IQN初回Learner priorityのquantile-pair方式

- 起票日: 2026-08-24
- 状態: 暫定・要設計レビュー（implementation readyではない）
- 対象: DefaultDQN IQN、PERの初回Learner priority更新
- Topic Issue: ハイパラ探索 `#22`、DQN `#20`
- 関連: PRD 001（IQN）、PRD 035（PER priority source）、PRD 044（IQN τ sampling）、PRD 047（IQN探索P0診断metric）

> 本書は探索候補を忘れないための暫定PRDである。主候補の式と責務境界は記録するが、設定名、採用式、適用範囲は実装着手前に再レビューする。

## Problem Statement

現行IQN Learnerは、experienceのpriority更新にcurrent quantileの平均とtarget quantileの平均の差を使う。current quantileを`z[b,i]`、target quantileを`y[b,j]`、pairwise TDを`delta[b,i,j] = y[b,j] - z[b,i]`とすると、clip前のpriority信号は次に相当する。

```text
mean_td[b] = abs(mean_i(z[b,i]) - mean_j(y[b,j]))
           = abs(mean_ij(delta[b,i,j]))
```

これは期待値TDとして明快であり、通常のDQN priorityとの対応もよい。一方、正負のpairwise TDが混在すると、個々のquantile pairに大きな誤差があっても平均時に相殺される。

特にReplayBuffer投入後の遷移が初めてLearner更新を受ける時点では、次の懸念がある。

- Double Suika達成間際のような希少遷移は経験数が少なく、初回更新でpriorityの識別力を失うと再sample機会を逃しやすい。
- IQNのcurrent/target τは有限本数であり、ランダムまたはstratifiedなτ集合に対して、平均TDだけでは分布上の局所的なずれを表せない。
- PERが遷移をsampleしても、最初のLearner priority更新で大きなpairwise誤差が相殺されれば、次回以降のsampling massが小さくなり得る。

PRD 047では、この議論の観測根拠として次を実装済みである。

```text
pair_abs_td[b] = mean_ij(abs(delta[b,i,j]))
cancellation[b] = clamp(
    1 - abs(mean_ij(delta[b,i,j])) / (pair_abs_td[b] + 1e-6),
    0,
    1)
```

ただし、`pair_abs_td`をpriorityに使えば常に改善するとは限らない。独立にsampleした同じ分布同士でも`E[abs(Y-Z)]`は0にならず、return distribution自体が広い遷移を、学習誤差が大きい遷移として過大評価する可能性がある。また、本方式は有限τ集合内の相殺を抑えるものであり、希少遷移へ割り当てられたτ集合そのものが不十分な問題を解消するものではない。τ被覆はstratified等のsampling方式と別に評価する必要がある。

## Solution

IQNの初回Learner priority更新行に限り、現行の平均TD priorityとquantile-pair由来priorityを切り替えられる実験設定を追加する。

暫定の主候補は`pair_abs_td`方式とする。

```text
control_signal[b] = abs(mean_ij(delta[b,i,j]))
pair_signal[b]    = mean_ij(abs(delta[b,i,j]))

signal[b] = initial_update[b]
    ? pair_signal[b]
    : control_signal[b]

raw_priority[b] = clip_if_enabled(signal[b] + per_eps)
leaf_priority[b] = pow(raw_priority[b], per_alpha)
```

`initial_update[b]`は、sample時点の優先度sourceが`fixed_initial`、`max_initial`、`actor_initial`のいずれかである行とする。`none`と`learner_updated`は含めない。この定義はPRD 047および`CONTEXT.md`の「初回Learner priority更新」と一致させる。

初回更新以外は現行の平均TD priorityを維持する。IQN loss、target、optimizer、初期priority方式、Replay sampling、importance sampling weightは変更しない。

### 暫定設定契約

設定名は実装前レビュー対象だが、意味を固定するため暫定的に次を置く。

```text
learner.iqn.first_priority_update_mode = mean_td | pair_abs_td
```

| 値 | 意味 |
|---|---|
| `mean_td` | 現行方式。初回を含む全Learner更新で平均TDを使う |
| `pair_abs_td` | 初回Learner priority更新行だけ`mean_ij(abs(delta))`を使い、それ以外は平均TDを使う |

- 既定値は`mean_td`とし、既存Runの挙動を変えない。
- `pair_abs_td`は`quantile_mode=iqn`かつ`use_per=true`でのみ有効とする。
- 未知値、または対応しないmodeとの組み合わせは、設定キー、指定値、許容値または必要条件を含むエラーでfail-fastする。
- 設定値を`fixed` / `max` / `actor_approx`へ暗黙変換しない。これは初期priority投入方式ではなく、初回Learner priority更新式の選択である。

## User Stories

1. As an IQN experimenter, I want first Learner priority updates to optionally use pairwise TD magnitude, so that positive and negative quantile errors do not cancel before PER ranking.
2. As a DropMerge experimenter, I want rare transitions to retain a chance of high priority when their distributional error is large, so that expectation-level cancellation does not immediately suppress them.
3. As a PER experimenter, I want the new formula applied only to first Learner priority updates, so that I can isolate the hypothesis without changing all steady-state priority updates.
4. As a baseline runner, I want the existing mean-TD formula to remain the default, so that saved baselines and established configurations keep their behavior.
5. As an experimenter, I want fixed, max, and actor-approx initial sources to share the same first-update definition, so that the formula does not depend on how the transition first entered the SumTree.
6. As an analyst, I want `per_sample_initial_count` alongside first-update metrics, so that I can judge whether a comparison contains enough first-update rows.
7. As an analyst, I want `iqn_first_pair_abs_td` and `iqn_first_cancellation_ratio` retained, so that I can distinguish larger pairwise error from mere changes in reward or Q scale.
8. As an analyst, I want the mean-TD diagnostic retained when pair priority is enabled, so that I can measure how much the new rule departs from the control signal.
9. As a TBO experimenter, I want the pairwise signal calculated in the same h-space as the current Learner priority, so that the comparison does not mix transformed and real return spaces.
10. As a performance maintainer, I want the existing `B x N x M` IQN loss tensor reused, so that the experiment adds no forward and no duplicate pairwise tensor.
11. As a GPU performance maintainer, I want the initial-source mask and priority choice to stay on device, so that no new CPU synchronization appears in the Learner hot path.
12. As an algorithm maintainer, I want loss and optimizer inputs unchanged, so that priority A/B results are not confounded by a different gradient objective.
13. As a replay maintainer, I want existing generation checks and stale-update rejection unchanged, so that a new priority formula cannot update an overwritten slot.
14. As a configuration author, I want unsupported combinations to fail fast, so that QR or PER-disabled Runs cannot silently ignore an intended IQN experiment.
15. As an experimenter, I want `per_eps`, clipping, and `per_alpha` applied exactly once through the existing priority pipeline, so that only the pre-transform signal changes.
16. As a researcher, I want the distribution-width inflation risk documented, so that a high pair priority is not automatically interpreted as better error estimation.
17. As a researcher, I want τ coverage and priority aggregation treated as separate hypotheses, so that stratified sampling benefits are not attributed to the pair formula.
18. As a test maintainer, I want hand-computable mixed-source batches, so that first-only selection and numerical formulas can be verified without depending on a full training Run.

## Implementation Decisions

### 1. 初回更新の判定

- sample時点の`per_priority_sources[B]`から`fixed_initial | max_initial | actor_initial`を選ぶ。
- `none | learner_updated`には現行平均TDを使う。
- 同一replay itemが同じminibatchへ重複して現れた場合は、各行のsample時点sourceに従う。ReplayBufferの重複更新順序とgeneration検査は変更しない。
- stale updateは従来どおり棄却する。計算時に初回行だったことを理由に、上書き済みslotへpriorityを適用しない。

### 2. quantile-pair信号

- IQN loss計算で既に生成する`delta[B,N,M]`から、PRD 047で既に計算済みの`pair_abs_td[B]`を再利用する。
- pairwise tensorの再生成、追加forward、追加target計算を行わない。
- `pair_abs_td`はfloat32かつ学習graphからdetachした値を使う。
- `N != M`を許容し、全`N x M`ペアを同じ重みで平均する。
- τの値による追加重み、上位pairだけを使うtop-k、percentile、最大値は暫定MVPへ含めない。

### 3. priority pipeline

- 初回行では`pair_abs_td`、その他の行では現行`abs(mean(z)-mean(y))`をdevice上の行単位選択で合成する。
- 合成後の非負signalへ、現行と同じ`per_eps`、optional clip、`per_alpha`を同じ順序で一度だけ適用する。
- `per_prio_clip_ratio`、raw priority、SumTree leaf priority、priority sourceの意味は変更しない。
- generation一致の適用後は、方式にかかわらずsourceを`learner_updated`へ更新する。
- optimizer step前にpriority内容を準備し、optimizer step後にReplayBufferへ反映する現行順序を維持する。

### 4. loss・学習との非干渉

- IQN quantile Huber lossとimportance sampling weightは変更しない。
- priority signalをlossへ逆流させない。
- action、τ生成、RNG消費、target network更新、Replay sampling probabilityの算出式は、priority値が変わった結果を除いて変更しない。
- checkpoint形式、Replay artifact形式、priority source enumは変更しない。

### 5. 観測

- PRD 047の`iqn_first_pair_abs_td`、`iqn_first_cancellation_ratio`、`iqn_first_priority_mc_ratio`、`iqn_first_quantile_loss_norm`、`per_sample_initial_count`を第一の観測根拠とする。
- 暫定MVPでは新しいmetricを必須にしない。既存metricで適用後raw priorityとの対応が追えないと判明した場合だけ、初回行のclip前/後priority集約をP1として追加検討する。
- Run artifactの実効設定からmodeを判別できるようにする。Run名だけを方式の正本にしない。

### 6. 文書責務

- 実装時は`CONTEXT.md`の「初回Learner priority更新」を維持し、「初期priority」と混同しない説明を加える。
- DQN設計文書へ式、適用source、TBO空間、初回以外のfallbackを記載する。
- ReplayBuffer設計文書ではpriority sourceと適用順序が不変であることを記載する。
- Run分析ガイドへ、pair方式が相殺を抑える一方でreturn distribution幅をpriorityへ含め得ることを記載する。
- 新規componentやADRの要否は実装計画時に再判定する。現時点では既存IQN loss/priority境界内へ収まる見込みである。

## Testing Decisions

### 数値契約

- 手計算できる`N != M`のTensorで`mean_td`と`pair_abs_td`を検証する。
- 正負が完全相殺し、`mean_td=0`かつ`pair_abs_td>0`になるケースを検証する。
- 全pairが同符号かつ同じ大きさで、両方式が一致するケースを検証する。
- `per_eps=0`、clip無効、clip境界未満・等値・超過、`per_alpha=0`を含め、既存変換が一度だけ適用されることを検証する。
- BF16学習時もpriority集約がfloat32であることを検証する。

### 初回mask

- 1 minibatchへ`fixed_initial`、`max_initial`、`actor_initial`、`none`、`learner_updated`を混在させ、最初の3種だけpair方式になることを検証する。
- 初回行0件では全行が現行平均TDと一致することを検証する。
- duplicate item、generation一致、stale itemについて既存ReplayBuffer更新契約を維持することを検証する。

### 非干渉

- mode切替前後でIQN sample loss、勾配、optimizer入力、current/target quantile、τ列、actionが一致することを検証する。
- `mean_td` modeのraw priorityとReplayBuffer leafが現行実装と完全一致する回帰テストを置く。
- forward回数、target計算回数、pairwise tensor生成回数が増えないことをfocused testまたはprobeで確認する。
- 追加の`.item()`、device-to-host readback、待機境界がないことを確認する。

### 設定・総合

- 未知mode、非IQN、PER無効との不正な組み合わせをfail-fastで検証する。
- IQN+PERの`mean_td` / `pair_abs_td`を終了step 1相当でsmokeし、NN構築、Learner更新、priority反映まで到達することを確認する。
- Debug build、DQN IQN/PER focused test、ReplayBuffer focused test、`git diff --check`を通す。
- 同一binaryの短時間A/Bで、追加処理による`exp_step_per_sec`低下2%以内を目安とする。

## Out of Scope

- `per_initial_priority_mode`の`fixed`、`max`、`actor_approx`契約変更。
- ReplayBuffer投入時の初期priority計算変更。
- 初回以外を含む全Learner更新へのpair方式適用。
- QR-DQN、通常DQN、MuZeroのpriority式変更。
- IQN quantile Huber loss、target計算、TBO変換、importance sampling weightの変更。
- `pair_abs_td`以外のtop-k、max、percentile、quantile Huber loss、Wasserstein、energy distance等の採用。
- `K/N/M`、τ sampling mode、PER alpha/beta、BatchSize、ReplayRatio、学習率の探索。
- 既定modeの変更、既存Run artifactの読み替え、実験Runの実施と採否判断。
- Metrics Viewer UI、Optuna objective、外部Issue作成。
- コード、設定、設計文書、実験記録の変更。本起票では本PRDファイルだけを新設する。

## Further Notes

### 実装前に決める事項

1. 設定名を`learner.iqn.first_priority_update_mode`で確定するか。
2. 主候補を`pair_abs_td`のまま採用するか、`iqn_first_quantile_loss_norm`由来の方式も同じ実装で比較可能にするか。
3. 初回更新だけに限定するか、後続実験として全Learner更新方式も用意するか。
4. 分布幅によるpriority inflationを、報酬成績、`iqn_first_cancellation_ratio`、priority分布、PER ESS、initial sample量から十分に判定できるか。
5. 適用後の初回raw priority専用metricを追加する必要があるか。

### 採否判断の観点

- pair方式によって初回priorityの識別が増えても、PER ESSの崩壊、priority clip率の増加、同じ高分散遷移への過集中が起きる場合は採用しない。
- DropMerge成果が改善してもthroughput低下やseed依存が大きい場合は、同条件複数Runまたは短い別Env傾向確認を要求する。
- `iqn_first_cancellation_ratio`が低い領域では両方式の差が小さいはずであり、その場合は追加複雑性に見合わない可能性がある。
- 本方式は「外れτを完全に防ぐ」ものではない。τ集合の被覆はstratified/antithetic等、priority集約は本PRDとして別軸で評価する。

本PRDは忘却防止の暫定起票であり、実装開始条件は上記未決事項のレビューと、既存RunにおけるPRD 047 metricの再確認である。
