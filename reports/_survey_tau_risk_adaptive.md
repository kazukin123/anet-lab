# 分布強化学習における τ(quantile fraction)の非一様分布・distortion・適応的選択 — 学術サーベイ

Date: 2026-08-12
Scope: 分位点ベースの分布強化学習(QR-DQN / IQN 系)において、quantile fraction τ を一様分布以外からサンプリングする・distortion risk measure で歪める・学習中に適応的に選ぶ研究。risk-sensitive RL と適応的 τ サンプリングが中心。被引用数はすべて Semantic Scholar(2026/08/12 時点、概数)。

## 目次

1. [原典 — IQN と distortion risk measure β(τ)](#1-原典--iqn-と-distortion-risk-measure-βτ)
   - Implicit Quantile Networks for Distributional Reinforcement Learning(IQN)
2. [CVaR 系 — τ を下位区間に制限する方式](#2-cvar-系--τ-を下位区間に制限する方式)
   - Worst Cases Policy Gradients(WCPG)
   - Being Optimistic to Be Conservative: Quickly Learning a CVaR Policy
   - Conservative Offline Distributional Reinforcement Learning(CODAC)
3. [楽観的 τ 選択・上位分位による探索](#3-楽観的-τ-選択上位分位による探索)
   - QUOTA: The Quantile Option Architecture for Reinforcement Learning
   - Distributional Reinforcement Learning for Efficient Exploration(DLTV)
4. [適応的リスク調整 — 学習中に τ 分布・distortion を自動調整](#4-適応的リスク調整--学習中に-τ-分布distortion-を自動調整)
   - Automatic Risk Adaptation in Distributional Reinforcement Learning(ARA)
   - Adaptive Risk-Tendency: Nano Drone Navigation with Distributional Reinforcement Learning(ART-IQN)
   - DRL-ORA: Distributional Reinforcement Learning with Online Risk Adaption
   - Tactical Optimism and Pessimism for Deep Reinforcement Learning(TOP)
5. [Thompson sampling 的な return 分布の利用](#5-thompson-sampling-的な-return-分布の利用)
   - The Potential of the Return Distribution for Exploration in RL
   - Exploration by Distributional Reinforcement Learning
6. [τ 自体の学習(fraction proposal)](#6-τ-自体の学習fraction-proposal)
   - Fully Parameterized Quantile Function for Distributional Reinforcement Learning(FQF)
7. [総括](#7-総括)
8. [調査限界](#8-調査限界)
9. [ソース一覧](#9-ソース一覧)

---

## 1. 原典 — IQN と distortion risk measure β(τ)

### Implicit Quantile Networks for Distributional Reinforcement Learning(IQN)

- 著者・所属: Will Dabney, Georg Ostrovski, David Silver, Rémi Munos(いずれも DeepMind, London)[IQN (2018/06)]
- 発表会場・年: ICML 2018(PMLR v80)
- 被引用数: 約 722(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/1806.06923 / https://proceedings.mlr.press/v80/dabney18a.html

IQN は分位関数を暗黙的に学習するネットワークで、基本形では τ ~ U([0,1]) をサンプルし、その τ における分位値 Z_τ(x,a) を return 分布からのサンプルとして用いる。risk-sensitive 化は「τ のサンプリング分布を distortion risk measure β で歪める」ことで定式化される。行動価値は β(τ) を通した期待値 Q_β として定義され、方策はその greedy 選択になる。

```
Q_β(x,a) := E_{τ~U([0,1])} [ Z_{β(τ)}(x,a) ] ... π_β(x) = arg max_{a∈A} Q_β(x,a)

Q_β(x,a) は τ~U([0,1]) のもとでの Z_{β(τ)}(x,a) の期待値として定義され、方策 π_β は Q_β を最大化する行動を選ぶ。
```

[IQN (2018/06)]

評価された distortion は CPW・Wang・Pow・CVaR・Norm の 5 種で、論文中の定義は CPW(η,τ) = τ^η / (τ^η + (1−τ)^η)^{1/η}、Wang(η,τ) = Φ(Φ^{−1}(τ) + η)(Φ は標準正規 CDF)、Pow(η,τ) = τ^{1/(1+|η|)}(η≥0 のとき。負なら 1−(1−τ)^{1/(1+|η|)})、CVaR(η,τ) = ητ、Norm(η) は U([0,1]) からの η 個のサンプルの平均で τ を作る、というもの [IQN (2018/06)]。CVaR(η,τ) = ητ は τ ~ U([0,1]) を U([0,η]) に線形に写すので、τ のサンプリングを下位区間 [0,η] に制限することと同じである(β の定義式からの直接の帰結)[IQN (2018/06)]。

---

## 2. CVaR 系 — τ を下位区間に制限する方式

### Worst Cases Policy Gradients(WCPG)

- 著者・所属: Yichuan Charlie Tang, Jian Zhang, Ruslan Salakhutdinov(いずれも Apple Inc.)[WCPG (2019/11)]
- 発表会場・年: CoRL 2019(PMLR v100)
- 被引用数: 約 88(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/1911.03618 / http://proceedings.mlr.press/v100/tang20a/tang20a.pdf

WCPG は actor-critic 構成で CVaR_α を最適化するが、α を固定せず「α を追加入力として条件付けた単一ネットワーク」を学習し、訓練中は α をエピソードごとにランダムサンプルする(論文では α を [0.01, 1.0] からランダムに引く。Algorithm 1 では α ~ U(0,1) と記載)[WCPG (2019/11), Section 4 / Algorithm 1]。これにより推論時に α を変えるだけでリスク態度を連続的に切り替えられる。

```
During inference, π can output different actions given the same exact state s, conditioned on the setting of α. Intuitively, a small α leads to conservative actions while a larger α leads to more aggressive actions.

推論時、π は α の設定に条件付けられて、全く同じ状態 s に対して異なる行動を出力できる。直観的には、小さい α は保守的な行動を、大きい α はより積極的な行動をもたらす。
```

[WCPG (2019/11)]

CVaR の定義は CVaR_α ≐ E_{p_π}[R | R ≤ pcntl(α)](pcntl(α) は return 分布の α パーセンタイル)で、critic がガウス分布仮定の場合は CVaR_α = Q − (φ(α)/Φ(α))·√Υ という閉形式を使う [WCPG (2019/11)]。

### Being Optimistic to Be Conservative: Quickly Learning a CVaR Policy

- 著者・所属: Ramtin Keramati(Stanford, ICME)、Christoph Dann(CMU / Stanford)、Alex Tamkin(Stanford)、Emma Brunskill(Stanford)[Keramati+ (2019/11)]
- 発表会場・年: AAAI 2020(vol.34, pp.4436–4443, DOI 10.1609/aaai.v34i04.5870)
- 被引用数: 約 91(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/1911.01546 / https://ojs.aaai.org/index.php/AAAI/article/view/5870

CVaR 最適方策をサンプル効率よく学習するために「不確実性に直面したときの楽観主義」を分布 Bellman 作用素に持ち込んだ研究。τ のサンプリング分布を変えるのではなく、推定 CDF 自体を訪問回数ベースのボーナス c/√n(s,a) だけ下方シフトさせる楽観的作用素を定義し、その分布に対する CVaR_α を最大化する行動 a* ← argmax_a CVaR_α(F̃_a) を選ぶ [Keramati+ (2019/11)]。

```
By shifting the cumulative distribution function down, this operator essentially puts probability mass from the lower tail to the highest possible value Vmax.

累積分布関数を下方にシフトすることで、この作用素は実質的に確率質量を下側テールから最大可能値 Vmax へ移す。
```

[Keramati+ (2019/11)]

CVaR は CVaR_α(X) := sup_ν {ν − (1/α)·E[(ν−X)+]}、連続分布では CVaR_α(X) = E[X | X ≤ F^{−1}(α)] として定義される(すなわち分位関数の下位 α 区間の平均)[Keramati+ (2019/11)]。

### Conservative Offline Distributional Reinforcement Learning(CODAC)

- 著者・所属: Yecheng Jason Ma, Dinesh Jayaraman, Osbert Bastani(いずれも University of Pennsylvania)[CODAC (2021/07)]
- 発表会場・年: NeurIPS 2021
- 被引用数: 約 114(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/2107.06106 / https://proceedings.neurips.cc/paper/2021/hash/a05d886123a54de3ca4b0985b718fb9b-Abstract.html

オフライン設定で、out-of-distribution 行動に対して予測分位値そのものをペナルティ付けする保守的分布 RL。リスク回避目的は τ の重み分布 g で表現され、g を下位区間上の一様分布にとると CVaR 目的(下位 ξ パーセンタイルのみ考慮)になる。実験では CVaR_0.1、すなわち τ ∈ [0, 0.1] 上の積分を用いる [CODAC (2021/07)]。

```
g=Uniform([0,ξ]) corresponds to the CVaR [...] objective, where only the bottom ξ-percentile of the return is considered.

g = Uniform([0,ξ]) は CVaR 目的に対応し、return の下位 ξ パーセンタイルのみが考慮される。
```

[CODAC (2021/07)]

(引用中の [...] は原文の文献参照番号を省略した箇所。)

---

## 3. 楽観的 τ 選択・上位分位による探索

### QUOTA: The Quantile Option Architecture for Reinforcement Learning

- 著者・所属: Shangtong Zhang(University of Alberta)、Borislav Mavrin(Huawei Noah's Ark Lab)、Linglong Kong(University of Alberta)、Bo Liu(Auburn University)、Hengshuai Yao(Huawei Noah's Ark Lab)[QUOTA (2018/11)]
- 発表会場・年: AAAI 2019
- 被引用数: 約 33(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/1811.02073 / https://ojs.aaai.org/index.php/AAAI/article/view/4527

平均でなく「特定の分位」に対して greedy に行動選択する探索法。どの分位を使うかは option 框組の高位方策がオンラインで学習する(N 個の分位を M 個の窓に分け、各 option は窓内 K 個の分位の平均に基づいて行動を提案する)[QUOTA (2018/11)]。

```
A high quantile represents an optimistic estimation of the action value, and action selection based on a high quantile indicates an optimistic exploration strategy. A low quantile represents a pessimistic estimation of the action value, and action selection based on a low quantile indicates a pessimistic exploration strategy.

高い分位は行動価値の楽観的推定を表し、高い分位に基づく行動選択は楽観的探索戦略を意味する。低い分位は行動価値の悲観的推定を表し、低い分位に基づく行動選択は悲観的探索戦略を意味する。
```

[QUOTA (2018/11)]

### Distributional Reinforcement Learning for Efficient Exploration(DLTV)

- 著者・所属: Borislav Mavrin(University of Alberta / Huawei Noah's Ark)、Hengshuai Yao(Huawei)、Linglong Kong(University of Alberta)、Kaiwen Wu、Yaoliang Yu(University of Waterloo)。arXiv 版著者欄には Shangtong Zhang(Oxford、Huawei インターン時の研究)も含まれるが、PMLR 掲載版の著者は上記 5 名 [DLTV (2019/05)]
- 発表会場・年: ICML 2019(PMLR 97:4424–4434)
- 被引用数: 約 107(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/1905.06125 / https://proceedings.mlr.press/v97/mavrin19a.html

QR-DQN の学習済み分位から「左側切断分散」(中央値より上側の分位のみで計算する分散。中央値を使うのは統計的頑健性のため)を探索ボーナスとして使い、パラメトリック不確実性の減衰理論に合わせた減衰スケジュール c_t = c√(log t / t) を掛けて intrinsic な(環境由来の)不確実性の影響を抑える手法 [DLTV (2019/05)]。Atari 49 ゲームで QR-DQN 比 483% の平均累積報酬ゲインを報告している [DLTV (2019/05)]。

```
By using the upper quantiles of the estimated distribution, we estimate an optimistic exploration bonus for QR-DQN.

推定された分布の上側分位を用いることで、QR-DQN のための楽観的探索ボーナスを推定する。
```

[DLTV (2019/05)]

---

## 4. 適応的リスク調整 — 学習中に τ 分布・distortion を自動調整

### Automatic Risk Adaptation in Distributional Reinforcement Learning(ARA)

- 著者・所属: Frederik Schubert, Theresa Eimer, Bodo Rosenhahn, Marius Lindauer(いずれも Leibniz University Hannover)[ARA (2021/06)]
- 発表会場・年: arXiv 2021。ICML 2021 Workshop "Reinforcement Learning for Real Life" でポスター発表 [ICML2021-WS (2021/07)]
- 被引用数: 約 10(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/2106.06317 / https://icml.cc/virtual/2021/13017

静的なリスクレベル(CVaR の α)は環境条件が変わると最適でなくなることを示した上で、Random Network Distillation(RND)誤差を状態依存の不確実性シグナルとして distortion のパラメータに流し込み、各ステップでリスクレベルを動的に選ぶ。具体的には β_ARA(τ) = β_CVaR(τ, ψ(u))(u は正規化した RND 誤差、ψ(u) = e^{−u})という形で CVaR distortion の α を毎ステップ決める。ベースは DSAC(Distributional Soft Actor-Critic)で、Q_β(s,a) = E_τ[Z_{β(τ)}(s,a)] という IQN 型の distortion 適用 [ARA (2021/06)]。

```
we use the RND error for the parameter α in the distortion function β

distortion 関数 β のパラメータ α に RND 誤差を用いる。
```

[ARA (2021/06)]

locomotion 環境で、失敗率を最大 1/7 に低減し、未知環境への汎化性能を最大 14% 改善したと報告している [ARA (2021/06)]。

### Adaptive Risk-Tendency: Nano Drone Navigation with Distributional Reinforcement Learning(ART-IQN)

- 著者・所属: Cheng Liu, Erik-Jan van Kampen, Guido C.H.E. de Croon(いずれも Delft University of Technology)[ART-IQN (2022/03)]
- 発表会場・年: ICRA 2023(DOI 10.1109/ICRA48891.2023.10160324)
- 被引用数: 約 26(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/2203.14749

IQN ベースで、CVaR を「τ のサンプリング区間の制限」として実装することを明示している論文。リスクレベル α 自体は、学習した return 分布の下側テール条件付き分散(右側切断分散 RTV と等価)を intrinsic 不確実性として、exponentially weighted average forecasting(EWAF)でオンライン適応させる(重み更新のフィードバック f = RTV_t − RTV_{t−1})[ART-IQN (2022/03)]。

```
CVaR is applied to IQN by modifying τ~∼U[0,1] to τ~∼U[0,α], where α is the CVaR value

CVaR は、τ~ ∼ U[0,1] を τ~ ∼ U[0,α] に変更することで IQN に適用される。ここで α は CVaR の値である。
```

[ART-IQN (2022/03)]

適応リスク傾向のエージェントは、リスク中立・リスク回避固定のベースラインより優れた性能を達成し、最も有効なリスク傾向は状態によって異なると報告している [ART-IQN (2022/03)]。

### DRL-ORA: Distributional Reinforcement Learning with Online Risk Adaption

- 著者・所属: Yupeng Wu, Wenyun Li, Wenjie Huang, Chin Pang Ho(所属は論文 HTML 上で明記を確認できず — citation not confirmed)
- 発表会場・年: arXiv プレプリント(2023 初出、v5 は 2026/02)。査読付き会場への採録は確認できず(citation not confirmed)
- 被引用数: 0(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/2310.05179

epistemic リスクレベルの選択自体をオンライン学習問題として定式化した枠組み。リスクレベルの選択は Follow-The-Leader 型アルゴリズムによるグリッドサーチで行い、実験では CVaR を α_min = 0.1 とした候補集合 𝒜 = [0.1, 1] 上で選ぶ。implicit な aleatory 不確実性に対しては β(q) = q(リスク中立)とし、epistemic 側にのみリスク考慮を入れる構成 [DRL-ORA (2023/10)]。

```
This framework quantifies both epistemic and implicit aleatory uncertainties in a unified manner and dynamically adjusts the epistemic risk levels by solving a total variation minimization problem online.

この枠組みは epistemic と implicit な aleatory の両不確実性を統一的に定量化し、全変動最小化問題をオンラインで解くことにより epistemic リスクレベルを動的に調整する。
```

[DRL-ORA (2023/10), abstract]

### Tactical Optimism and Pessimism for Deep Reinforcement Learning(TOP)

- 著者・所属: Ted Moskovitz(Gatsby Unit, UCL)、Jack Parker-Holder(University of Oxford)、Aldo Pacchiano(Microsoft Research)、Michael Arbel(Université Grenoble Alpes, Inria, CNRS)、Michael I. Jordan(UC Berkeley)[TOP (2021/02)]
- 発表会場・年: NeurIPS 2021
- 被引用数: 約 75(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/2102.03765 / https://proceedings.neurips.cc/paper/2021/hash/6abcc8f24321d1eb8c95855eab78ee95-Abstract.html

分位表現の distributional critic(aleatoric)とアンサンブル(epistemic)を分離し、belief 分布の分位を q = q_mean + β·q_σ と構成して、楽観度 β(β≥0 で楽観、β<0 で悲観)の選択を多腕バンディットとしてオンラインで解く。楽観・悲観のどちらが良いかは環境依存(HalfCheetah では楽観、Hopper では悲観が優位)であり、推定バイアスの度合いは環境・最適化の段階・文脈の関数として変動する、というのが適応化の動機 [TOP (2021/02)]。

```
each bandit arm represents a particular value of β, and we consider D experts making recommendations from a discrete set of values {β_d}

各バンディットのアームは特定の β の値を表し、離散的な値集合 {β_d} から推薦を行う D 個のエキスパートを考える。
```

[TOP (2021/02)]

---

## 5. Thompson sampling 的な return 分布の利用

τ を確率的に引くこと自体を Thompson sampling として定式化した研究は今回確認できなかった(調査限界を参照)。近接する系統として、return 分布(または価値分布のパラメータの事後分布)からサンプルして行動選択する研究を挙げる。

### The Potential of the Return Distribution for Exploration in RL

- 著者・所属: Thomas M. Moerland, Joost Broekens, Catholijn M. Jonker(いずれも Delft University of Technology)[Moerland+ (2018/06)]
- 発表会場・年: arXiv 2018(EWRL 2018 の論文としても掲載: https://ewrl.wordpress.com/wp-content/uploads/2018/09/ewrl_14_2018_paper_22.pdf )
- 被引用数: 約 10(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/1806.04242

学習した return 分布からの Thompson sampling で探索する研究。決定論的環境では return 分布は方策自身によって誘導されるため、方策は変更可能である以上、この分布に対して楽観的に行動することに意味がある、という論拠を与えている [Moerland+ (2018/06)]。

```
Thompson sampling [...], which takes a sample z_a ∼ p(Z|s,a) for each action and picks the action with the highest draw.

Thompson sampling は、各行動について z_a ∼ p(Z|s,a) のサンプルを 1 つ取り、最も高い値を引いた行動を選ぶ。
```

[Moerland+ (2018/06)]

(引用中の [...] は原文の文献参照(Thompson 1933)を省略した箇所。)

### Exploration by Distributional Reinforcement Learning

- 著者・所属: Yunhao Tang, Shipra Agrawal(いずれも Columbia University IEOR)[Tang-Agrawal (2018/05)]
- 発表会場・年: IJCAI 2018(DOI 10.24963/ijcai.2018/376)
- 被引用数: 約 34(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/1805.01907

こちらは τ ではなく「価値分布ネットワークのパラメータ θ の(近似)事後分布」からサンプルして行動選択する posterior sampling 型の探索。行動選択はパラメータを 1 つ引いてから期待値 greedy で行う [Tang-Agrawal (2018/05)]。

```
In state s_t, we sample a parameter from θ∼q_ϕ(θ) and choose action a_t=arg max_a E[Z_θ(s_t,a)].

状態 s_t において、θ ∼ q_ϕ(θ) からパラメータをサンプルし、行動 a_t = arg max_a E[Z_θ(s_t, a)] を選ぶ。
```

[Tang-Agrawal (2018/05)]

---

## 6. τ 自体の学習(fraction proposal)

### Fully Parameterized Quantile Function for Distributional Reinforcement Learning(FQF)

- 著者・所属: Derek Yang(UC San Diego)、Li Zhao(Microsoft Research)、Zichuan Lin(Tsinghua University)、Tao Qin(Microsoft Research)、Jiang Bian(Microsoft Research)、Tie-Yan Liu(Microsoft Research)[FQF (2019/11)]
- 発表会場・年: NeurIPS 2019
- 被引用数: 約 185(Semantic Scholar、2026/08/12 確認)
- URL: https://arxiv.org/abs/1911.02140 / https://proceedings.neurips.cc/paper_files/paper/2019/hash/f471223d1a1614b58a7dc45c9d01df19-Abstract.html

τ をランダムサンプルするのではなく、状態(-行動)ごとに τ の離散集合そのものを生成する fraction proposal network を導入し、分位値ネットワークと同時に学習する。fraction proposal network は、近似分布と真の分布の 1-Wasserstein 距離が最小になるように訓練される。「τ を非一様に配置する」ことをリスク目的でなく分布近似精度の目的で行う研究であり、今回の調査範囲では τ 分布を学習で偏らせる研究として最も直接的なもの [FQF (2019/11)]。

```
Unlike QR-DQN and IQN where quantile fractions are fixed or sampled and only the corresponding quantile values are parameterized, both quantile fractions and corresponding quantile values in our algorithm are parameterized.

quantile fraction が固定またはサンプルされ、対応する分位値のみがパラメータ化される QR-DQN や IQN と異なり、我々のアルゴリズムでは quantile fraction と対応する分位値の両方がパラメータ化される。
```

[FQF (2019/11)]

---

## 7. 総括

収集したソースに書かれている範囲で、τ の非一様な扱いは次の系統に整理できる。

- 固定 distortion による τ の再パラメータ化: IQN が β(τ)(CPW / Wang / Pow / CVaR / Norm)で τ ~ U([0,1]) を歪める定式化を与えた原典 [IQN (2018/06)]。
- τ の下位区間への制限(CVaR): β(τ) = ητ は τ ∈ [0,η] の一様サンプリングと等価 [IQN (2018/06)]。ART-IQN はこれを「τ~ ∼ U[0,1] を U[0,α] に変更」と明示的に実装 [ART-IQN (2022/03)]。CODAC は τ の重み分布 g = Uniform([0,ξ]) として CVaR を表現 [CODAC (2021/07)]。Keramati らは τ 側でなく CDF 自体を count-based に下方シフトする楽観作用素で CVaR 学習を加速 [Keramati+ (2019/11)]。
- リスクレベル(α や β)の適応: 訓練中に α を一様サンプルして条件付ける WCPG [WCPG (2019/11)]、RND 誤差で α を毎ステップ決める ARA [ARA (2021/06)]、テール分散を EWAF で追跡する ART-IQN [ART-IQN (2022/03)]、オンライン学習(FTL)で α を選ぶ DRL-ORA [DRL-ORA (2023/10)]、楽観度 β をバンディットで選ぶ TOP [TOP (2021/02)]。
- 上位分位による楽観探索: どの分位で greedy になるかを option 高位方策が学習する QUOTA [QUOTA (2018/11)]、上側分位から計算した左側切断分散を減衰スケジュール付き探索ボーナスにする DLTV [DLTV (2019/05)]。
- τ 配置自体の学習: FQF の fraction proposal network(1-Wasserstein 最小化)[FQF (2019/11)]。
- Thompson sampling 系は「τ のサンプリング」ではなく「return 分布またはパラメータ事後分布からのサンプリング」として実現されている [Moerland+ (2018/06)] [Tang-Agrawal (2018/05)]。

## 8. 調査限界

- 被引用数はすべて Semantic Scholar の値(2026/08/12 確認)。Google Scholar とは集計方法が異なり、数値が乖離する可能性がある(例: IQN の 722 は Semantic Scholar 値)。Google Scholar 側の数値は未確認。
- 「τ に対する明示的な importance sampling(提案分布からサンプルして重み補正する形)」および「τ 依存の loss 重み付けで訓練時の τ 分布を意図的に偏らせる研究」は、今回の Web 検索では直接該当する論文を確認できなかった(citation not confirmed)。最も近いのは τ の配置自体を学習する FQF [FQF (2019/11)]。
- 「τ を確率変数として Thompson sampling と解釈する研究」も直接は確認できなかった。検索中に「Jiang et al. (2023) が分位推定器のアンサンブルに基づく Thompson sampling を提案」という言及が二次ソース(arXiv:2406.12241 の検索要約)にあったが、一次ソースを特定・確認できていない(citation not confirmed)。
- DRL-ORA の著者所属は arXiv HTML 上で確認できなかった。また同論文の査読付き会場への採録有無も未確認。
- ARA の正式な発表形態は arXiv プレプリント+ICML 2021 ワークショップ(Reinforcement Learning for Real Life)ポスターであり、本会議採録ではない。
- 原文引用は ar5iv / arXiv HTML 経由で取得したため、数式まわりの文字表現が原文 PDF と細部で異なる可能性がある(取得時の Unicode 変換による)。数式の引用は最小限にとどめた。
- DLTV の著者リストは arXiv 版(6 名、Shangtong Zhang を含む)と PMLR 掲載版(5 名)で異なって観測された。本文では両方を記載した。

## 9. ソース一覧

[IQN, 2018/06] Will Dabney, Georg Ostrovski, David Silver, Rémi Munos. "Implicit Quantile Networks for Distributional Reinforcement Learning." ICML 2018 (PMLR v80). https://arxiv.org/abs/1806.06923

[QUOTA, 2018/11] Shangtong Zhang, Borislav Mavrin, Linglong Kong, Bo Liu, Hengshuai Yao. "QUOTA: The Quantile Option Architecture for Reinforcement Learning." AAAI 2019. https://arxiv.org/abs/1811.02073

[DLTV, 2019/05] Borislav Mavrin, Hengshuai Yao, Linglong Kong, Kaiwen Wu, Yaoliang Yu. "Distributional Reinforcement Learning for Efficient Exploration." ICML 2019 (PMLR 97:4424–4434). https://arxiv.org/abs/1905.06125

[FQF, 2019/11] Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, Tie-Yan Liu. "Fully Parameterized Quantile Function for Distributional Reinforcement Learning." NeurIPS 2019. https://arxiv.org/abs/1911.02140

[WCPG, 2019/11] Yichuan Charlie Tang, Jian Zhang, Ruslan Salakhutdinov. "Worst Cases Policy Gradients." CoRL 2019 (PMLR v100). https://arxiv.org/abs/1911.03618

[Keramati+, 2019/11] Ramtin Keramati, Christoph Dann, Alex Tamkin, Emma Brunskill. "Being Optimistic to Be Conservative: Quickly Learning a CVaR Policy." AAAI 2020. https://arxiv.org/abs/1911.01546

[ARA, 2021/06] Frederik Schubert, Theresa Eimer, Bodo Rosenhahn, Marius Lindauer. "Automatic Risk Adaptation in Distributional Reinforcement Learning." arXiv preprint / ICML 2021 Workshop on Reinforcement Learning for Real Life. https://arxiv.org/abs/2106.06317

[ICML2021-WS, 2021/07] ICML 2021 Virtual Site. "Automatic Risk Adaptation in Distributional Reinforcement Learning" (poster page, Workshop: Reinforcement Learning for Real Life). https://icml.cc/virtual/2021/13017

[TOP, 2021/02] Ted Moskovitz, Jack Parker-Holder, Aldo Pacchiano, Michael Arbel, Michael I. Jordan. "Tactical Optimism and Pessimism for Deep Reinforcement Learning." NeurIPS 2021. https://arxiv.org/abs/2102.03765

[CODAC, 2021/07] Yecheng Jason Ma, Dinesh Jayaraman, Osbert Bastani. "Conservative Offline Distributional Reinforcement Learning." NeurIPS 2021. https://arxiv.org/abs/2107.06106

[Moerland+, 2018/06] Thomas M. Moerland, Joost Broekens, Catholijn M. Jonker. "The Potential of the Return Distribution for Exploration in RL." arXiv preprint (EWRL 2018). https://arxiv.org/abs/1806.04242

[Tang-Agrawal, 2018/05] Yunhao Tang, Shipra Agrawal. "Exploration by Distributional Reinforcement Learning." IJCAI 2018. https://arxiv.org/abs/1805.01907

[ART-IQN, 2022/03] Cheng Liu, Erik-Jan van Kampen, Guido C.H.E. de Croon. "Adaptive Risk-Tendency: Nano Drone Navigation in Cluttered Environments with Distributional Reinforcement Learning." ICRA 2023. https://arxiv.org/abs/2203.14749

[DRL-ORA, 2023/10] Yupeng Wu, Wenyun Li, Wenjie Huang, Chin Pang Ho. "DRL-ORA: Distributional Reinforcement Learning with Online Risk Adaption." arXiv preprint. https://arxiv.org/abs/2310.05179
