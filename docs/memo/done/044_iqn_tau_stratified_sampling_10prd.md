# 044: IQN tau sampling modes(stratified / systematic / antithetic)

- 起票: 2026-08-10
- 更新: 2026-08-12(stratified 単独 → 3 mode 追加へ改訂)
- 対象: DefaultDQNAgent の IQN tau配置方式
- 状態: implementation ready
- 関連: `001_iqn_10prd.md`、ADR 0018、`CONTEXT.md`「taus」「tau配置方式」
- 参考資料: `reports/tau_sampling_modes_survey.md` と同ディレクトリの中間レポート4本(τ サンプリング方式の文献・実装サーベイ。未コミットの作業成果物であり、本PRDは単体で self-contained に読めるよう要約を第2節に含む)

## Problem Statement

現行の IQN は、tau配置方式として `random` と `fixed` を提供している。

`random` は各 forward で τ を独立に一様抽出するため、IQN の連続的な分位関数学習に必要なランダム性を持つ。一方で、有限個の τ が特定区間へ偏り、1回の minibatch 内で分位範囲の一部が未被覆になることがある。DropMerge の Double Suika 達成間際のような希少遷移では、その遷移が ReplayBuffer から選ばれる回数自体が少ないため、偏った τ 集合が割り当てられると学習機会を十分に活用できない懸念がある。

`fixed` は指定範囲を等幅区間へ分割し、その中点を使うため全域を安定して被覆するが、forward ごとの τ のランダム性を失う。これは QR 寄せの control として有用だが、IQN が持つ連続分位関数の確率的な学習という性質を弱める。

`random` の偏りには、性質の異なる2つの成分がある。①点集合が局所に固まり分位範囲の一部が未被覆になる成分(被覆の問題)、②点集合全体が上位側または下位側へ寄り、Q 推定が片側へ振れる成分(対称性の問題)。この2成分は別の処方で潰せる: 被覆は層化(各区間1点保証)で、対称性は鏡映ペア(範囲中点対称の強制)で解消できる。単一 mode の追加では、改善が観測されたときにどちらの成分が原因だったのかを切り分けられない。

長時間を要する DropMerge の探索では、τ 数や ReplayBuffer/PER の設定を変える前に、総 τ 数と Tensor shape、forward 回数、概算計算量を維持したまま、被覆とランダム性の軸を独立に動かせる比較候補群が必要である。

## 背景サーベイ要約

2026-08-12 に実施した文献・実装サーベイの要点(詳細と出典一覧は `reports/tau_sampling_modes_survey.md`):

- 分布RL文献で確立している τ 配置は3系統のみ: 固定一様 grid(QR-DQN, https://arxiv.org/abs/1710.10044)、iid 一様サンプル(IQN, https://arxiv.org/abs/1806.06923)、proposal network による学習(FQF, https://arxiv.org/abs/1911.02140)。DSAC-Ma(https://arxiv.org/abs/2004.14547)が fixed / random / net を同一アルゴリズム内で直接比較し、random を採用した。
- 層化・QMC・jitter を分布RLの τ に適用した先行は、論文・主要実装(Dopamine / DQN Zoo / pfrl / Tianshou / d3rlpy 等はいずれも学習時 τ は素朴な一様乱数)・GitHub 議論のいずれにも見つからなかった(英語圏調査による負の結果)。実装側で確認できた一様乱数以外のバリエーションは「評価時の等間隔 linspace 切替」「行動選択 τ へのリスク歪曲」「FQF 系」の3種のみ。
- 間接的な支持材料: QEMRL(ICML 2023, https://arxiv.org/abs/2307.16152)は「IQN の一様サンプル τ は、QR-DQN の等間隔 τ が満たす分散低減の十分条件を満たさない」と明示。NDQFN(https://arxiv.org/abs/2105.06696)は「IQN/FQF の毎 iteration の τ 再サンプルは分布ベース探索ボーナスを極めて不安定にする」と批判。FQF 提案元の Microsoft Research は「サンプルされた fraction は最良とは限らない」を FQF の動機として明言。τ の取り方が推定分散・安定性に影響するという認識は複数ソースにあるが、解として層化系は検討されていない。
- 統計学側には対応する問題と理論保証がある: stratified は層化抽出(比例配分の層化は iid より分散を増やさない無条件保証。Owen, Monte Carlo theory, https://artowen.su.domains/mc/)および粒子フィルタの stratified resampling に対応。systematic は粒子フィルタの systematic resampling(無条件保証はないが経験的に最良とされる)および拡散モデル VDM の timestep sampler(https://arxiv.org/abs/2107.00630, `t^i = mod(u_0 + i/k, 1)`)に対応。antithetic は単調な被積分関数に対して分散低減が保証される対蹠変量に対応する。理想的な分位関数はτについて単調だが、現行IQNの学習済み出力は単調性を強制しないため、Q推定への効果は条件付き理論を背景に実験で評価する。

## モード概観

5 mode は2つの比較軸を構成する。被覆軸は **random → stratified → systematic → fixed**、対称性軸は **random → antithetic** とする。antithetic は範囲中点対称を強制する一方でstratum被覆を保証しないため、被覆軸上でstratifiedとの前後関係を持たない。共通記法: `K = num_taus`、正規化位置 `v` を生成してから `τ = τ_min + v·(τ_max − τ_min)` へ写像する。

| mode | 正規化位置 v の定義 | 被覆 | ランダム性(自由度/行) | RNG消費/行 | 行内順序 |
|---|---|---|---|---|---|
| random | `v[b,i] = u[b,i]` | 保証なし | K | K | 非順序 |
| antithetic | 前半 `M=⌊K/2⌋` 個は `u[b,j]`、後半は `1−u[b,j]`(奇数 K は末尾に独立 u を1個) | 保証なし(鏡映ペアのみ範囲中点対称) | ⌈K/2⌉ | ⌈K/2⌉ | 非順序(対称レイアウト) |
| stratified | `v[b,i] = (i + u[b,i]) / K` | 各層1点(隣接ギャップ < 2/K) | K | K | 厳密昇順 |
| systematic | `v[b,i] = (i + u[b]) / K`(u は行ごと1個) | 完全等間隔 | 1 | 1 | 厳密昇順 |
| fixed | `v[i] = (i + 0.5) / K`(midpoint) | 完全等間隔+位相固定 | 0 | 0 | 厳密昇順 |

- **random**(既存): 各点が独立一様。偏りの両成分(未被覆・片寄り)を持つ。
- **antithetic**(新規): random の変種。乱数を半分だけ引き、残りを `1−u` で鏡映する。偶数 K では点集合全体が範囲中点に関して対称になり、τ の標本平均が範囲中点に固定される。奇数 K では末尾の独立サンプルを除く鏡映ペアだけが対称となる。単調な分位関数には対蹠変量の分散低減理論が適用できるが、現行IQNは出力の単調性を強制しないため、実際のQ/target推定への効果は実験で評価する。局所クラスタ(テール未被覆)は防げない。
- **stratified**(新規): 範囲を K 等分し各層から独立に1点。全層の被覆を保証しつつ層内位置はランダム。比例配分層化として iid より分散を増やさない無条件保証を持つ。
- **systematic**(新規): stratified の層内乱数を行内で1個に共有した特殊化。点間隔が正確に `(τ_max−τ_min)/K` に固定され、位相のみランダム。被覆は最強、ランダム性は最小。
- **fixed**(既存): midpoint grid。RNG 非消費。

## Solution

IQN の `sample_mode` に `stratified`、`systematic`、`antithetic` を追加し、現行契約を `random | fixed | stratified | systematic | antithetic` とする。

共通範囲 `[τ_min, τ_max]` に対する各 mode の定義は次のとおりとする。

```text
u ~ U[0,1)  (すべて対象 device 上で一括生成)

stratified:  τ[b,i] = τ_min + ((i + u[b,i]) / K) * (τ_max - τ_min)
             u[b,i] は行・stratum ごとに独立((B,K) 個)

systematic:  τ[b,i] = τ_min + ((i + u[b]) / K) * (τ_max - τ_min)
             u[b] は行ごとに独立に1個((B,1) 個)。batch 全行での共有はしない

antithetic:  M = floor(K/2)、u[b,j] は (B, ceil(K/2)) 個
             τ[b,j]   = τ_min + u[b,j]       * (τ_max - τ_min)   (j = 0..M-1)
             τ[b,M+j] = τ_min + (1 - u[b,j]) * (τ_max - τ_min)   (j = 0..M-1)
             K が奇数のとき τ[b,K-1] は独立な u による通常の一様サンプル
```

per-env 下限 `τ_min[b]` を使う場合も同じ正規化位置を用い、各行の `[τ_min[b], τ_max]` へ写像する。antithetic の鏡映は正規化空間で行うため、対称中心は行ごとの範囲中点 `(τ_min[b] + τ_max) / 2` となる。

数式上、stratified/systematicの正規化位置は `[0,1)`、antitheticは鏡映値`1-u`により`[0,1]`を取り得る。出力はfloat32であるため、stratified/systematicも丸めによって保存値が`τ_max`と等しくなる場合がある。新3 modeの保存Tensorは`τ_min <= τ <= τ_max`を契約とし、上端回避のclampや再抽出は行わない。また、float32で隣接位置を区別できない極狭範囲では同値を許容する。

出力本数は常に `K` とし、全 mode で既存と同じ `(B,K)` shape、`float32` dtype、指定 device を維持する。既存の既定値は変更せず、Train Policy と Learner は `random`、Eval Policy と target Policy は `fixed` のままとする。

## User Stories

1. As an RL experimenter, I want one τ sample from every equal-width stratum in `stratified`, so that each sampled transition covers the full configured quantile range.
2. As an RL experimenter, I want τ positions to vary between forwards in all three new modes, so that IQN retains stochastic sampling instead of becoming a fixed-grid model.
3. As a DropMerge experimenter, I want rare replayed transitions to avoid large uncovered τ regions under `stratified` and `systematic`, so that scarce learning opportunities are used more consistently.
4. As an RL experimenter, I want `systematic` to produce exactly equally spaced τ with a per-row random phase, so that coverage is maximal while per-forward randomness is preserved at minimal RNG cost.
5. As an RL experimenter, I want `antithetic` to produce mirrored pairs symmetric around the range midpoint, so that even-K samples have a fixed mean and variance reduction can be evaluated under the learned IQN's observed monotonicity.
6. As an RL experimenter, I want all modes to use the same total `num_taus`, so that sampling-mode comparisons do not silently change network workload.
7. As an RL experimenter, I want the five modes to share the same configuration field, so that I can run one-axis A/B comparisons along the randomness–coverage spectrum.
8. As an RL experimenter, I want current、target-value、Train Policy、Eval Policy、target Policy、full-distribution query to use the same mode vocabulary, so that mode semantics do not depend on the call site.
9. As an RL experimenter, I want common-range and per-env lower-bound generation to follow the same normalized-position rule per mode, so that UQE and spatial exploration do not introduce a different sampling definition.
10. As an RL experimenter, I want `stratified` and `systematic` τ ordered ascending and `antithetic` laid out in a documented mirrored structure, so that coverage and symmetry are inspectable and deterministic properties can be tested directly.
11. As an RL experimenter, I want identical seeds and inputs to reproduce identical τ tensors in every mode, so that experiment setup can be diagnosed consistently.
12. As an RL experimenter, I want different seeds to produce different random positions in the three new modes, so that they are genuinely stochastic.
13. As a performance-conscious user, I want random values generated in one batched operation on the target device in every mode, so that no mode adds CPU synchronization or per-element host loops.
14. As a maintainer, I want existing `GenerateTaus` callers and `TauRuleConfig` fields to remain unchanged, so that the new modes stay inside the current `TauGenerator` responsibility.
15. As a maintainer, I want invalid mode strings rejected before training, so that misspelled experiment settings do not silently fall back to another behavior.
16. As a maintainer, I want the runtime validation boundary to reject unknown modes too, so that direct component use obeys the same contract as configuration construction.
17. As a maintainer, I want current `random` and `fixed` behavior preserved, so that existing configs and experiment artifacts retain their meaning.
18. As a maintainer, I want `fixed` to continue consuming no RNG and each new mode's RNG consumption documented as part of its contract, so that RNG-sequence expectations across modes are explicit and testable.
19. As an experiment analyst, I want the selected mode recorded in the resolved Run config, so that Run names or later source-config edits are not mistaken for the effective setting.
20. As an experiment analyst, I want sampling correctness proven by focused tests rather than hot-path metrics, so that long DropMerge Runs do not pay avoidable measurement cost.
21. As an experiment planner, I want the mode set to expose separate coverage and symmetry axes, so that an observed improvement can be attributed to coverage (`stratified`/`systematic`) or symmetry (`antithetic`) rather than to an entangled mixture.
22. As an experiment planner, I want literal fixed-plus-random concatenation excluded from this change, so that no extra split ratio or hidden τ-count increase confounds the sampling comparison.
23. As an experiment planner, I want τ-count and PER exploration deferred, so that the effect of the placement rule can be isolated first.

## Implementation Decisions

1. `sample_mode` の許容値へ `stratified`、`systematic`、`antithetic` を追加し、現行契約を `random | fixed | stratified | systematic | antithetic` とする。文字列フィールドは維持し、新しいmode専用の設定fieldは追加しない。
2. `stratified` は `K` 個の等幅stratumごとに独立な `U[0,1)` を1点生成する。全batch行に共通のrandom shiftを使わず、`u[b,i]` は行・stratumごとに独立とする。
3. `systematic` は行ごとに単一の `u[b] ~ U[0,1)` を生成し、行内の全stratumへbroadcastする。batch全行での `u` 共有は行わない(行間で τ 集合が相関すると minibatch 内の loss 推定誤差が相関するため。Decision 2 と同じ理由)。
4. `antithetic` のレイアウトは契約として固定する: 前半 `M = floor(K/2)` 点が `u[b,j]`、後半 `M` 点が `1 - u[b,j]`(同じ `j` 順)、`K` が奇数なら末尾1点は独立な一様サンプル。出力のソートは行わない。鏡映は正規化空間で行い、対称中心は共通範囲版で `(τ_min + τ_max)/2`、per-env版で行ごとの `(τ_min[b] + τ_max)/2` となる。
5. `num_taus` は全modeで出力本数を意味する。どのmodeでも出力は `(B,K)` であり、fixed点や追加random点を連結して `K` を超えない。
6. 共通下限overloadとper-env下限overloadは同じ正規化位置を使う。per-env版は各行の `[τ_min[b], τ_max]` へ写像する。
7. 正の幅を持ち、float32で隣接位置を区別できる範囲では、`stratified` と `systematic` の各行の点はstratum順に厳密な昇順となる。極狭範囲では丸めによる同値を許容し、非減少を契約とする。`antithetic` に順序契約はない(Decision 4 の対称レイアウトのみ)。`τ_min == τ_max` の退化範囲では全modeで既存方式と同様に全点が同値となり、暗黙のclampや別modeへのfallbackは行わない。
8. modeごとのRNG消費量を契約として明記する: `random` と `stratified` は行あたり `K` 個、`systematic` は行あたり `1` 個、`antithetic` は行あたり `ceil(K/2)` 個、`fixed` は `0` 個。同一seedでもmodeを変更すると後続の乱数列が変わるのは仕様であり、互換化は行わない。
9. 乱数は既存の `RandomGenerator` から指定device用generatorを取得し、各modeが必要とする個数を1回の一括生成で得る。同一seed・同一呼び出し順・同一入力では同じ結果を返す。CPUへのmaterialize、device間転送、要素単位のhost loopを追加しない。既存の `TauGenerator` profile範囲内で計測可能な状態を維持する。
10. `GenerateTaus` の2つのoverload、引数、戻り値契約は変更しない。Policy、Learner、full-distribution queryの呼び出し側にmode別処理を分散させず、生成方式の分岐は `TauGenerator` 内へ閉じ込める。
11. DefaultDQNの全tau ruleは3つの新modeを受理する。既定値は一切変更せず、既存configの解決結果を維持する。
12. 未知modeは設定構築時に該当keyと指定値、許容値 `random, fixed, stratified, systematic, antithetic` を含めてfail-fastする。`TauGenerator` の直接呼び出しも同じ許容値集合でfail-fastする。
13. `random` のiid一様抽出、`fixed` のmidpoint gridとRNG非消費契約は変更しない。互換alias、旧値変換、WARN、暗黙fallbackは追加しない。
14. `TauRuleConfig` のserializationやRunの解決済みconfigには、既存fieldを通じて選択modeがそのまま記録される。分析時はRun artifactの `config/config_data.txt` を実効設定の正本とする。
15. 新しいτ統計metricは追加しない。生成方式の正しさは単体テストで保証し、学習効果は既存のreward、max-rank、Double Suika、Q、TD error、loss、NOOP、throughput指標で評価する。
16. 実装時にdomain glossaryの「tau配置方式」とDQN設計文書を `random / fixed / stratified / systematic / antithetic` の5 mode契約へ更新する。新規componentやADRは作らない。

## Testing Decisions

- テストは内部の演算手順ではなく、生成Tensorと設定境界から観測できる契約を検証する。
- 既存の `[dqn][iqn][tau]` テスト群を先例とし、`TauGenerator` の両overloadに対する focused testを3 mode分追加する。
- `stratified`: 共通範囲では各行・各stratumにちょうど1点が入り、代表的な正の範囲幅ではstratum順に厳密な昇順となり、全点が指定範囲内であることを検証する。極狭範囲では非減少を許容する。
- `systematic`: 代表的な範囲で隣接点の差が全行・全位置で定数 `(τ_max − τ_min)/K` であること、位相 `u` が行ごとに独立(同一batch内の行間で異なり得る)こと、厳密昇順、各stratumに1点、全点範囲内を検証する。極狭範囲では非減少を許容する。
- `antithetic`: `τ[b,j] + τ[b,M+j] == τ_min + τ_max`(浮動小数許容誤差付き)が全ペアで成り立つこと、奇数 `K` の末尾点が対称拘束を持たない独立サンプルであること、全点範囲内を検証する。per-env版では対称中心が行ごとの `(τ_min[b] + τ_max)/2` であることを検証する。
- per-env下限では、各modeが各行固有の範囲へ正しく写像され、被覆・等間隔・対称の各性質が行単位で成り立つことを検証する。下限が上限と等しい行も既存の退化範囲契約に従うことを確認する。
- 同一seedの独立した `RandomGenerator` から同じTensorが得られ、異なるseedからは異なるTensorが得られることを、3 mode すべてで検証する。
- modeごとのRNG消費量(`stratified`: B×K、`systematic`: B、`antithetic`: B×⌈K/2⌉)を、後続の `random` 出力との比較で検証する。`fixed` が引き続きRNGを消費しないことも同方式で検証する。
- 既存 `random` のiid抽出、shape、範囲、再現性テストを維持し、方式追加による挙動変更がないことを確認する。
- CPUでは `(B,K)`、`float32`、指定範囲、指定deviceを全modeで検証する。CUDA利用可能時は両overloadで出力がCUDA上に留まり、shapeとdtypeを維持することを検証する。
- 既存の `[dqn][iqn][config]` テストを拡張し、Train/Eval/target Policy、各full-distribution query、Learner current/targetの全tau ruleが3つの新modeを受理することを検証する。
- 未知modeの設定と直接呼び出しはfail-fastし、エラーに指定値と `random, fixed, stratified, systematic, antithetic` が含まれることを検証する。
- 既定configのmodeとtau数が変更されていないことを既存のdefault testで継続確認する。
- focused testは `[dqn][iqn][tau]` と `[dqn][iqn][config]` を実行し、実装範囲に応じてDefaultDQN関連テストを追加実行する。

## Out of Scope

- fixed midpoint点とiid random点を連結するliteral mixed mode(統計学の defensive mixture に相当する形。比率という交絡軸が増えるため本比較から除外)。
- fixed/randomの本数、比率、交互配置を指定する新しい設定。
- QMC/低差異列(Sobol、Halton等)による τ 生成。分布RLでは未踏の将来候補としてサーベイ要約に記録するに留める。
- loss駆動の τ 重点サンプリング(τ ビンごとのloss履歴から非一様サンプル+IS重み補正。FQF-lite)と、FQF型 fraction proposal network。いずれも hot path へのフィードバック経路や追加networkを要するため別PRDとする。
- `num_taus` の意味変更、modeによる暗黙の出力本数増加。
- τ数 `N`、`M`、`K` の探索や既定値変更。
- PER、ReplayBuffer、batch size、replay ratio、termination、reward、network architectureの変更。
- 既存 `random` / `fixed` の改名、削除、意味変更。
- Eval/targetを新modeへ切り替える既定値変更。
- τの分布やstratum被覆を記録する新しいruntime metric。
- DropMerge、LunarLanderその他Envでの実験Runと採用判断。
- 新規component、公開API拡張、checkpoint payload変更、新規ADR。

## Further Notes

- 3 mode を同時に追加するのは、被覆軸 `random → stratified → systematic → fixed` と対称性軸 `random → antithetic` を分けて比較するためである。`stratified`/`systematic` が有効で `antithetic` が無効なら改善の由来は被覆(テール保証)、逆なら Q 推定の対称化と切り分けられる。単独追加ではこの帰属ができない。
- `antithetic` の分散低減理論は単調な分位関数を条件として Q/target の期待値推定に適用できる。現行IQNはquantile crossingを許すため、実モデルでの効果とQR lossの勾配への効果はいずれも経験的検証事項である。この区別を評価時に混同しない。
- `stratified` はfixed midpointを常に含む方式ではない。等幅区間を必ず1点ずつ被覆し、その区間内の位置だけをランダム化する。`systematic` はさらに層内位置を行内で共有し、等間隔性まで固定する。
- 最初の実験比較では `sample_mode` 以外の設定を揃える。特に `num_taus`、minibatch size、replay ratio、終端step、seed、Env、NN、PERを同時に変更しない。
- 各modeの機構成立と成績改善は分けて判断する。生成契約と設定反映が正常でも、DropMergeの立ち上がりや終盤成績が改善するとは限らない。
- Run開始後は生成された `config/config_data.txt` でPolicyとLearnerの各 `sample_mode` を確認する。Run名や実行後に編集されたsource configを実効設定として扱わない。
- 長時間比較へ進む前に短いsmokeで設定解決、有限なloss/Q/gradient、throughput、正常closeを確認する。その後の評価gate、mode間の実施順、seed数はIQN探索記録側で決定し、本PRDの実装契約へ混ぜない。
