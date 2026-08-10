# 044: IQN stratified τ sampling

- 起票: 2026-08-10
- 対象: DefaultDQNAgent の IQN tau配置方式
- 状態: implementation ready
- 関連: `001_iqn_10prd.md`、ADR 0018、`CONTEXT.md`「taus」「tau配置方式」

## Problem Statement

現行の IQN は、tau配置方式として `random` と `fixed` を提供している。

`random` は各 forward で τ を独立に一様抽出するため、IQN の連続的な分位関数学習に必要なランダム性を持つ。一方で、有限個の τ が特定区間へ偏り、1回の minibatch 内で分位範囲の一部が未被覆になることがある。DropMerge の Double Suika 達成間際のような希少遷移では、その遷移が ReplayBuffer から選ばれる回数自体が少ないため、偏った τ 集合が割り当てられると学習機会を十分に活用できない懸念がある。

`fixed` は指定範囲を等幅区間へ分割し、その中点を使うため全域を安定して被覆するが、forward ごとの τ のランダム性を失う。これは QR 寄せの control として有用だが、IQN が持つ連続分位関数の確率的な学習という性質を弱める。

長時間を要する DropMerge の探索では、τ 数や ReplayBuffer/PER の設定を変える前に、総 τ 数と Tensor shape、forward 回数、概算計算量を維持したまま、被覆とランダム性を両立する比較候補が必要である。

## Solution

IQN の `sample_mode` に `stratified` を追加し、現行契約を `random | fixed | stratified` とする。

`stratified` は、指定された τ 範囲を `K = num_taus` 個の等幅区間へ分け、各区間から1点ずつ独立に一様抽出する。これにより、1回の forward で全区間を必ず被覆しながら、各点は forward ごとに変化する。

共通範囲 `[τ_min, τ_max]` に対する定義は次のとおりとする。

```text
u[b,i] ~ U[0,1)
τ[b,i] = τ_min + ((i + u[b,i]) / K) * (τ_max - τ_min)
```

per-env 下限 `τ_min[b]` を使う場合も同じ相対配置を用いる。

```text
τ[b,i] = τ_min[b] + ((i + u[b,i]) / K) * (τ_max - τ_min[b])
```

出力本数は常に `K` とし、既存の `random` / `fixed` と同じ `(B,K)` shape、`float32` dtype、指定 device を維持する。既存の既定値は変更せず、Train Policy と Learner は `random`、Eval Policy と target Policy は `fixed` のままとする。

## User Stories

1. As an RL experimenter, I want one τ sample from every equal-width stratum, so that each sampled transition covers the full configured quantile range.
2. As an RL experimenter, I want τ positions to vary between forwards, so that IQN retains stochastic sampling instead of becoming a fixed-grid model.
3. As a DropMerge experimenter, I want rare replayed transitions to avoid large uncovered τ regions, so that scarce learning opportunities are used more consistently.
4. As an RL experimenter, I want `stratified` to use the same total `num_taus` as other modes, so that sampling-mode comparisons do not silently change network workload.
5. As an RL experimenter, I want `random`、`fixed`、`stratified` to share the same configuration field, so that I can run one-axis A/B comparisons.
6. As an RL experimenter, I want current、target-value、Train Policy、Eval Policy、target Policy、full-distribution query to use the same mode vocabulary, so that mode semantics do not depend on the call site.
7. As an RL experimenter, I want common-range and per-env lower-bound generation to follow the same stratification rule, so that UQE and spatial exploration do not introduce a different sampling definition.
8. As an RL experimenter, I want the generated τ values ordered by stratum, so that coverage is inspectable and deterministic properties can be tested directly.
9. As an RL experimenter, I want identical seeds and inputs to reproduce identical stratified τ tensors, so that experiment setup can be diagnosed consistently.
10. As an RL experimenter, I want different seeds to produce different within-stratum positions, so that the new mode is genuinely stochastic.
11. As a performance-conscious user, I want random values generated in one batched operation on the target device, so that the new mode adds no CPU synchronization or per-element host loop.
12. As a maintainer, I want existing `GenerateTaus` callers and `TauRuleConfig` fields to remain unchanged, so that the new mode stays inside the current `TauGenerator` responsibility.
13. As a maintainer, I want invalid mode strings rejected before training, so that misspelled experiment settings do not silently fall back to another behavior.
14. As a maintainer, I want the runtime validation boundary to reject unknown modes too, so that direct component use obeys the same contract as configuration construction.
15. As a maintainer, I want current `random` and `fixed` behavior preserved, so that existing configs and experiment artifacts retain their meaning.
16. As a maintainer, I want `fixed` to continue consuming no RNG state, so that deterministic evaluation and existing RNG-sequence tests remain valid.
17. As an experiment analyst, I want the selected mode recorded in the resolved Run config, so that Run names or later source-config edits are not mistaken for the effective setting.
18. As an experiment analyst, I want sampling correctness proven by focused tests rather than hot-path metrics, so that long DropMerge Runs do not pay avoidable measurement cost.
19. As an experiment planner, I want literal fixed-plus-random concatenation excluded from this change, so that no extra split ratio or hidden τ-count increase confounds the first sampling comparison.
20. As an experiment planner, I want τ-count and PER exploration deferred, so that the effect of the placement rule can be isolated first.

## Implementation Decisions

1. `sample_mode` の許容値へ `stratified` を追加し、現行契約を `random | fixed | stratified` とする。文字列フィールドは維持し、新しいmode専用の設定fieldは追加しない。
2. `stratified` は `K` 個の等幅stratumごとに独立な `U[0,1)` を1点生成する。全batch行に共通のrandom shiftを使わず、`u[b,i]` は行・stratumごとに独立とする。
3. `num_taus` は全modeで出力本数を意味する。`stratified` でも出力は `(B,K)` であり、fixed点や追加random点を連結して `K` を超えない。
4. 共通下限overloadとper-env下限overloadは同じ正規化位置 `(i + u[b,i]) / K` を使う。per-env版は各行の `[τ_min[b], τ_max]` へ写像する。
5. 正の幅を持つ範囲では、各行の点はstratum順に厳密な昇順となる。`τ_min == τ_max` の退化範囲では既存方式と同様に全点が同値となり、暗黙のclampや別modeへのfallbackは行わない。
6. 乱数は既存の `RandomGenerator` から指定device用generatorを取得し、`(B,K)` のTensorとして一括生成する。同一seed・同一呼び出し順・同一入力では同じ結果を返す。
7. CPUへのmaterialize、device間転送、要素単位のhost loopを追加しない。既存の `TauGenerator` profile範囲内で計測可能な状態を維持する。
8. `GenerateTaus` の2つのoverload、引数、戻り値契約は変更しない。Policy、Learner、full-distribution queryの呼び出し側にmode別処理を分散させず、生成方式の分岐は `TauGenerator` 内へ閉じ込める。
9. DefaultDQNの全tau ruleは `stratified` を受理する。既定値は一切変更せず、既存configの解決結果を維持する。
10. 未知modeは設定構築時に該当keyと指定値、許容値 `random, fixed, stratified` を含めてfail-fastする。`TauGenerator` の直接呼び出しも同じ許容値集合でfail-fastする。
11. `random` のiid一様抽出、`fixed` のmidpoint gridとRNG非消費契約は変更しない。互換alias、旧値変換、WARN、暗黙fallbackは追加しない。
12. `TauRuleConfig` のserializationやRunの解決済みconfigには、既存fieldを通じて `stratified` がそのまま記録される。分析時はRun artifactの `config/config_data.txt`を実効設定の正本とする。
13. 新しいτ統計metricは追加しない。生成方式の正しさは単体テストで保証し、学習効果は既存のreward、max-rank、Double Suika、Q、TD error、loss、NOOP、throughput指標で評価する。
14. 実装時にdomain glossaryの「tau配置方式」とDQN設計文書を `random / fixed / stratified` の契約へ更新する。新規componentやADRは作らない。

## Testing Decisions

- テストは内部の演算手順ではなく、生成Tensorと設定境界から観測できる契約を検証する。
- 既存の `[dqn][iqn][tau]` テスト群を先例とし、`TauGenerator` の両overloadに対する focused testを追加する。
- 共通範囲では各行・各stratumにちょうど1点が入り、正の範囲幅ではstratum順に厳密な昇順となり、全点が指定範囲内であることを検証する。
- per-env下限では、各行固有の範囲へ正しく写像され、各行の全stratumが被覆されることを検証する。下限が上限と等しい行も既存の退化範囲契約に従うことを確認する。
- 同一seedの独立した `RandomGenerator` から同じTensorが得られ、異なるseedからは異なるTensorが得られることを検証する。
- `stratified` がRNGを消費することと、`fixed` が引き続きRNGを消費しないことを、後続の `random` 出力との比較で検証する。
- 既存 `random` のiid抽出、shape、範囲、再現性テストを維持し、方式追加による挙動変更がないことを確認する。
- CPUでは `(B,K)`、`float32`、指定範囲、指定deviceを検証する。CUDA利用可能時は両overloadで出力がCUDA上に留まり、shapeとdtypeを維持することを検証する。
- 既存の `[dqn][iqn][config]` テストを拡張し、Train/Eval/target Policy、各full-distribution query、Learner current/targetの全tau ruleが `stratified` を受理することを検証する。
- 未知modeの設定と直接呼び出しはfail-fastし、エラーに指定値と `random, fixed, stratified` が含まれることを検証する。
- 既定configのmodeとtau数が変更されていないことを既存のdefault testで継続確認する。
- focused testは `[dqn][iqn][tau]` と `[dqn][iqn][config]` を実行し、実装範囲に応じてDefaultDQN関連テストを追加実行する。

## Out of Scope

- fixed midpoint点とiid random点を連結するliteral mixed mode。
- fixed/randomの本数、比率、交互配置を指定する新しい設定。
- `num_taus` の意味変更、modeによる暗黙の出力本数増加。
- τ数 `N`、`M`、`K` の探索や既定値変更。
- PER、ReplayBuffer、batch size、replay ratio、termination、reward、network architectureの変更。
- 既存 `random` / `fixed` の改名、削除、意味変更。
- Eval/targetを `stratified` へ切り替える既定値変更。
- τの分布やstratum被覆を記録する新しいruntime metric。
- DropMerge、LunarLanderその他Envでの実験Runと採用判断。
- 新規component、公開API拡張、checkpoint payload変更、新規ADR。

## Further Notes

- `stratified` はfixed midpointを常に含む方式ではない。等幅区間を必ず1点ずつ被覆し、その区間内の位置だけをランダム化することで、総τ数を増やさずにQR寄せの被覆とIQNの確率性を両立する候補である。
- 最初の実験比較では `random`、`fixed`、`stratified` 以外の設定を揃える。特に `num_taus`、minibatch size、replay ratio、終端step、seed、Env、NN、PERを同時に変更しない。
- `stratified` の機構成立と成績改善は分けて判断する。生成契約と設定反映が正常でも、DropMergeの立ち上がりや終盤成績が改善するとは限らない。
- Run開始後は生成された `config/config_data.txt` でPolicyとLearnerの各 `sample_mode` を確認する。Run名や実行後に編集されたsource configを実効設定として扱わない。
- 長時間比較へ進む前に短いsmokeで設定解決、有限なloss/Q/gradient、throughput、正常closeを確認する。その後の評価gateとseed数はIQN探索記録側で決定し、本PRDの実装契約へ混ぜない。
