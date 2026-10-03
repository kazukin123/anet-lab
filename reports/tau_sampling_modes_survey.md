# Survey: 分布強化学習における τ(quantile fraction)のサンプリング/配置方式

Date: 2026-08-12
Scope: quantile ベース分布強化学習(QR-DQN / IQN / FQF 系)における τ の生成・配置・分布の決め方を、(1) 原典論文の系譜、(2) risk-sensitive・適応的 τ 選択、(3) モンテカルロ分散低減理論と隣接分野の適用例、(4) 主要フレームワーク実装、の4角度から調査した。被引用数は原則 Semantic Scholar(2026-08-12 確認、概数)、取得不能時は OpenAlex で補完または「未確認」と明記。詳細版は同ディレクトリの中間レポート4本(`_survey_tau_core_papers.md` / `_survey_tau_risk_adaptive.md` / `_survey_tau_mc_theory.md` / `_survey_tau_implementations.md`)を参照。

## Table of Contents

1. [τ 生成方式の全体分類](#1-τ-生成方式の全体分類)
2. [原典論文における τ 配置の系譜](#2-原典論文における-τ-配置の系譜)
   - 2.1 QR-DQN: 固定一様 grid(midpoint)
   - 2.2 IQN: τ ∼ U[0,1] の iid サンプリングと N/N′ ablation
   - 2.3 FQF: fraction proposal network による τ の学習
   - 2.4 Non-crossing 系(NC-QR-DQN / NDQFN): τ ではなく quantile 値側への制約
   - 2.5 連続制御系(TQC / DSAC / D4PG)と DSAC-Ma の fixed/random/net 直接比較
3. [risk-sensitive・適応的な τ 分布の変形](#3-risk-sensitive適応的な-τ-分布の変形)
   - 3.1 distortion risk measure β(τ): CPW / Wang / Pow / CVaR / Norm
   - 3.2 CVaR 系: τ の下位区間への制限
   - 3.3 適応的リスク調整: α・β を学習中に自動調整
   - 3.4 楽観的分位選択による探索: QUOTA / DLTV
   - 3.5 Thompson sampling 系(τ ではなく分布・事後からのサンプル)
4. [モンテカルロ分散低減理論と隣接分野での層化・QMC 適用](#4-モンテカルロ分散低減理論と隣接分野での層化qmc-適用)
   - 4.1 一般理論: stratified / LHS / antithetic / QMC の保証
   - 4.2 機械学習での適用例: VDM / Improved DDPM / NeRF / RQMC 方策勾配
   - 4.3 分位点推定の統計学における層化・重点サンプリング
5. [分布RL の τ への stratified/QMC 適用の探索結果(負の結果を含む)](#5-分布rl-の-τ-への-stratifiedqmc-適用の探索結果負の結果を含む)
6. [主要フレームワーク実装の現状](#6-主要フレームワーク実装の現状)
7. [批判・懸念(ソースが自己申告・相互批判したもの)](#7-批判懸念ソースが自己申告相互批判したもの)
8. [総合評価](#8-総合評価)
9. [調査の限界](#9-調査の限界)
10. [ソース一覧](#10-ソース一覧)

## 1. τ 生成方式の全体分類

収集したソース全体から、τ の生成・利用方式は次の系統に整理できる(各行の根拠は括弧内の節に引用)。

| # | 系統 | 代表 | τ の決め方 | 節 |
|---|---|---|---|---|
| A | 固定一様 grid(midpoint) | QR-DQN、TQC、NC-QR-DQN | τ̂_i = (i+0.5)/N 相当の固定中点。RNG 非消費 | 2.1, 2.5 |
| B | iid 一様サンプリング | IQN、NDQFN、DSAC-Ma、M-IQN、EQR | 毎 forward で τ ∼ U([0,1]) | 2.2 |
| C | 評価時のみ決定的 grid へ切替 | pfrl、d3rlpy(実装慣行) | 学習時 rand / 評価時 linspace | 6 |
| D | distortion による τ の歪み | IQN(CPW/Wang/Pow/CVaR/Norm)、DI-engine | τ ∼ U([0,1]) を β(τ) で写像(行動選択側のみ) | 3.1 |
| E | τ の区間制限(CVaR) | ART-IQN、CODAC | τ ∼ U([0,α]) / 重み g=Uniform([0,ξ]) | 3.2 |
| F | リスクレベルの適応 | WCPG、ARA、ART-IQN、DRL-ORA、TOP | α・β を条件付け/オンライン学習で決定 | 3.3 |
| G | 分位の選択を探索方策化 | QUOTA、DLTV | どの分位で greedy かを高位方策・ボーナスで決定 | 3.4 |
| H | τ 自体の学習 | FQF | fraction proposal network(W1 最小化) | 2.3 |
| I | 層化・QMC 系 | (分布RL では未発見) | 隣接分野に理論・適用例のみ | 4, 5 |
| J | τ を使わないパラメタ化 | C51/D4PG(categorical)、DSAC-Duan(Gaussian) | quantile fraction が存在しない | 2.5 |

## 2. 原典論文における τ 配置の系譜

### 2.1 QR-DQN: 固定一様 grid(midpoint)

- Will Dabney (DeepMind), Mark Rowland (Univ. of Cambridge), Marc G. Bellemare (Google Brain), Rémi Munos (DeepMind)。AAAI 2018。被引用数 1031(Semantic Scholar、2026-08-12)。https://arxiv.org/abs/1710.10044

QR-DQN は分布を N 個の等重み Dirac で表し、各出力を固定 grid の中点 τ̂_i に対応させる。τ は学習中に変化しない。

```
These quantile midpoints will be denoted by τ̂_i = (τ_{i-1} + τ_i)/2 for 1 ≤ i ≤ N.

これらの quantile 中点を、1 ≤ i ≤ N に対して τ̂_i = (τ_{i-1} + τ_i)/2 と表記する。
```
[QR-DQN (2017/10), Section "Approximately Minimizing Wasserstein"]

### 2.2 IQN: τ ∼ U[0,1] の iid サンプリングと N/N′ ablation

- Will Dabney, Georg Ostrovski, David Silver, Rémi Munos(全員 DeepMind)。ICML 2018。被引用数 722(Semantic Scholar、2026-08-12)。https://arxiv.org/abs/1806.06923

IQN は online 側 τ と target 側 τ′ を U([0,1]) から独立にサンプルし、pairwise TD 誤差で quantile regression loss を構成する。

```
For two samples τ, τ′ ∼ U([0,1]), and policy π_β, the sampled temporal difference (TD) error at step t is

2 つのサンプル τ, τ′ ∼ U([0,1]) と方策 π_β に対して、ステップ t におけるサンプルされた TD 誤差は(次式で与えられる)
```
[IQN (2018/06), Section 3, Eq. (2) 直前]

τ サンプル本数の ablation(N: online、N′: target、1〜64)では、N は初期性能に劇的な効果、N′ は 8 以降は長期性能への影響が最小、N = N′ = 8 で改善の大部分を達成、方策側 K=32 は感度なしと報告されている。

```
As expected, we found that N has a dramatic effect on early performance, shown by the continual improvement in score as the value increases.

予想通り、N は初期性能に劇的な効果を持つことが分かった。これは値を増やすにつれてスコアが継続的に改善することに示されている。
```
[IQN (2018/06), N/N′ ablation の段落]

```
Overall, while using more samples for both distributions is generally favorable, N = N′ = 8 appears to be sufficient to achieve the majority of improvements offered by IQN for long-term performance, with variation past this point largely insignificant.

全体として、両分布でより多くのサンプルを使うことは概ね好ましいが、長期性能については N = N′ = 8 で IQN のもたらす改善の大部分を達成するのに十分であるように見え、この点を超えた変動はほぼ有意でない。
```
[IQN (2018/06), N/N′ ablation の段落]

```
In an informal evaluation, we did not find IQN to be sensitive to K, the number of samples used for the policy, and have fixed it at K=32 for all experiments.

非公式な評価では、方策に使うサンプル数 K に対して IQN が敏感であるとは分からず、全実験で K=32 に固定した。
```
[IQN (2018/06), Section 3.1]

### 2.3 FQF: fraction proposal network による τ の学習

- Derek Yang (UC San Diego), Li Zhao, Tao Qin, Jiang Bian, Tie-Yan Liu (Microsoft Research), Zichuan Lin (Tsinghua Univ.)。NeurIPS 2019。被引用数 185(Semantic Scholar、2026-08-12)。https://arxiv.org/abs/1911.02140

FQF は τ をサンプルせず、状態ごとに softmax 出力の累積和として単調増加する τ 集合を生成する fraction proposal network を導入し、1-Wasserstein 距離最小化で学習する。「τ の配置を学習で非一様化する」研究として最も直接的である。

```
Unlike QR-DQN and IQN where quantile fractions are fixed or sampled and only the corresponding quantile values are parameterized, both quantile fractions and corresponding quantile values in our algorithm are parameterized.

quantile fraction が固定またはサンプルされ、対応する分位値のみがパラメータ化される QR-DQN や IQN と異なり、我々のアルゴリズムでは quantile fraction と対応する分位値の両方がパラメータ化される。
```
[FQF (2019/11)]

```
Let τ_i = Σ_{j=0}^{i−1} q_j, i ∈ [0, N], then straightforwardly we have τ_i < τ_j for ∀ i < j and τ_0 = 0, τ_N = 1 in our fraction proposal network.

τ_i = Σ_{j=0}^{i−1} q_j(i ∈ [0, N])とおくと、我々の fraction proposal network では直ちに ∀i<j で τ_i < τ_j、かつ τ_0 = 0, τ_N = 1 が成り立つ。
```
[FQF (2019/11), Section 3.4]

安定化のため proposal network の学習率は quantile value network より大幅に小さく設定される(sweep の結果 2.5e-9)。

```
we sweep the learning rate of fraction proposal network among (0, 2.5e-5) and finally fix this learning rate as 2.5e-9.

fraction proposal network の学習率を (0, 2.5e-5) の範囲で sweep し、最終的に 2.5e-9 に固定した。
```
[FQF (2019/11), Appendix(実験設定)]

FQF 提案元の Microsoft Research 公式ブログ(Li Zhao, Principal Researcher, MSR Asia, 2019-12-18)は、IQN のランダム τ の限界を FQF の動機として明言している。

```
"the sampled quantile fractions aren't necessarily the best quantile fractions."
"This motivated us to find the learning targets—that is, the quantile fractions—that would result in the least approximation error."

サンプリングされた quantile fraction が最良の fraction とは限らない。これが、近似誤差を最小にする学習ターゲット(=quantile fraction)を見つけようという動機になった。
```
[MSR FQF blog (2019/12), 本文]

### 2.4 Non-crossing 系(NC-QR-DQN / NDQFN): τ ではなく quantile 値側への制約

NC-QR-DQN(Fan Zhou, Jianing Wang, Xingdong Feng; 上海財経大学。NeurIPS 2020。被引用数 15、OpenAlex、2026-08-12。https://proceedings.neurips.cc/paper/2020/file/b6f8dc086b2d60c5856e4ff517060392-Paper.pdf)は、固定 τ grid のまま、softmax 累積和×非負スロープの構成で quantile 値の非交差(non-crossing)を保証する。τ 自体は変更しない。

```
Since α(s,a) is non-negative and {ψ_{i,a}}'s are non-decreasing, the non-crossing property of the N q_i(s,a)'s is automatically satisfied.

α(s,a) が非負で {ψ_{i,a}} が非減少であるため、N 個の q_i(s,a) の非交差性は自動的に満たされる。
```
[NC-QR (2020/12), Section 3.3]

NDQFN(Fan Zhou, Zhoufan Zhu, Qi Kuang, Liwen Zhang; 上海財経大学。IJCAI 2021。被引用数 23、Semantic Scholar、2026-08-12。https://arxiv.org/abs/2105.06696)は、固定 supporting points 上の非負増分累積で非減少 quantile 関数を構成しつつ、学習時の評価点 τ は IQN と同様に毎 iteration 一様再サンプルする。

```
Following the idea of IQN, two random sets of quantile fractions τ = {τ_1,⋯,τ_{N_1}}, τ′ = {τ′_1,⋯,τ′_{N_2}} are independently drawn from a uniform distribution U(0,1) at each training iteration.

IQN の考え方に従い、2 つのランダムな quantile fraction 集合 τ と τ′ を、各学習 iteration で一様分布 U(0,1) から独立に抽出する。
```
[NDQFN (2021/05), Section 4]

### 2.5 連続制御系(TQC / DSAC / D4PG)と DSAC-Ma の fixed/random/net 直接比較

TQC(Kuznetsov et al.; Samsung AI Center Moscow。ICML 2020。被引用数 285、Semantic Scholar、2026-08-12。https://arxiv.org/abs/2005.04269)は QR-DQN 型の固定中点 τ_m = (2m−1)/2M を使い、複数 critic の atom をプール・ソートして上側を切り捨てる(truncation)ことで過大評価を制御する。τ 配置そのものは固定 grid のままである [TQC (2020/05), Section 2.3, 3.2]。

D4PG(Barth-Maron et al.; DeepMind。ICLR 2018。https://arxiv.org/abs/1804.08617)は categorical(C51 系)パラメタ化で quantile fraction を持たず、DSAC-Duan(Duan et al.; Tsinghua。IEEE TNNLS 2022。被引用数 326、Semantic Scholar、2026-08-12。https://arxiv.org/abs/2001.02811)は return 分布を Gaussian としてモデル化するため、いずれも τ という自由度が存在しない [D4PG (2018/04), Appendix A] [DSAC-Duan (2020/01), Section V-A]。

DSAC-Ma(Xiaoteng Ma et al.; Tsinghua ほか。JAIR 採録。https://arxiv.org/abs/2004.14547)は、fixed(QR-DQN 式)・random(IQN 式)・net(FQF 式)の 3 方式を同一アルゴリズム内で直接比較した数少ない例で、random を採用した。

```
random sampling (Dabney et al. 2018b) has better performance and fewer parameters.

ランダムサンプリング(Dabney et al. 2018b)の方が性能が良く、パラメータも少ない。
```
[DSAC-Ma (2020/04), Section 5.1]

## 3. risk-sensitive・適応的な τ 分布の変形

### 3.1 distortion risk measure β(τ): CPW / Wang / Pow / CVaR / Norm

IQN 原典は、行動選択に使う τ のサンプリング分布を distortion risk measure β で歪めることで risk-sensitive 方策を定式化した。評価された distortion は CPW・Wang・Pow・CVaR・Norm の 5 種である [IQN (2018/06)]。

```
Q_β(x,a) := E_{τ~U([0,1])} [ Z_{β(τ)}(x,a) ] ... π_β(x) = arg max_{a∈A} Q_β(x,a)

Q_β(x,a) は τ~U([0,1]) のもとでの Z_{β(τ)}(x,a) の期待値として定義され、方策 π_β は Q_β を最大化する行動を選ぶ。
```
[IQN (2018/06)]

CVaR(η,τ) = ητ は τ ∼ U([0,1]) を U([0,η]) へ線形に写すため、τ のサンプリングを下位区間に制限することと等価である(β の定義式からの直接の帰結)[IQN (2018/06)]。

### 3.2 CVaR 系: τ の下位区間への制限

ART-IQN(Cheng Liu, Erik-Jan van Kampen, Guido de Croon; TU Delft。ICRA 2023。被引用数 約26、Semantic Scholar、2026-08-12。https://arxiv.org/abs/2203.14749)は、CVaR を「τ のサンプリング区間の制限」として実装することを明示している。

```
CVaR is applied to IQN by modifying τ~∼U[0,1] to τ~∼U[0,α], where α is the CVaR value

CVaR は、τ~ ∼ U[0,1] を τ~ ∼ U[0,α] に変更することで IQN に適用される。ここで α は CVaR の値である。
```
[ART-IQN (2022/03)]

CODAC(Ma, Jayaraman, Bastani; Univ. of Pennsylvania。NeurIPS 2021。被引用数 約114、Semantic Scholar、2026-08-12。https://arxiv.org/abs/2107.06106)は、リスク回避目的を τ 上の重み分布 g で表現し、g = Uniform([0,ξ]) を CVaR に対応させる(オフライン RL の保守化)[CODAC (2021/07)]。一方 Keramati et al.(Stanford/CMU。AAAI 2020。被引用数 約91、Semantic Scholar、2026-08-12。https://arxiv.org/abs/1911.01546)は τ 側でなく推定 CDF 自体を訪問回数ベースで下方シフトする楽観作用素により CVaR 方策の学習を加速する [Keramati+ (2019/11)]。

### 3.3 適応的リスク調整: α・β を学習中に自動調整

固定リスクレベルを避け、α や楽観度 β を動的に決める研究群が確認できた。

- WCPG(Tang, Zhang, Salakhutdinov; Apple。CoRL 2019。被引用数 約88。https://arxiv.org/abs/1911.03618): α を訓練中エピソードごとにランダムサンプルして条件付け、推論時に α で行動のリスク態度を連続的に切り替える [WCPG (2019/11), Section 4 / Algorithm 1]。
- ARA(Schubert et al.; Leibniz Univ. Hannover。ICML 2021 WS。被引用数 約10。https://arxiv.org/abs/2106.06317): Random Network Distillation の誤差を CVaR distortion の α に毎ステップ流し込む [ARA (2021/06)]。
- ART-IQN: 下側テール条件付き分散を exponentially weighted average forecasting で追跡し α をオンライン適応 [ART-IQN (2022/03)]。
- DRL-ORA(Wu, Li, Huang, Ho。arXiv、査読採録未確認。被引用数 0。https://arxiv.org/abs/2310.05179): epistemic リスクレベルの選択を Follow-The-Leader 型オンライン学習で解く [DRL-ORA (2023/10), abstract]。
- TOP(Moskovitz et al.; UCL/Oxford/MSR/UC Berkeley。NeurIPS 2021。被引用数 約75。https://arxiv.org/abs/2102.03765): 楽観度 β の選択を多腕バンディットとして解く。楽観・悲観のどちらが良いかは環境依存というのが適応化の動機である [TOP (2021/02)]。

```
each bandit arm represents a particular value of β, and we consider D experts making recommendations from a discrete set of values {β_d}

各バンディットのアームは特定の β の値を表し、離散的な値集合 {β_d} から推薦を行う D 個のエキスパートを考える。
```
[TOP (2021/02)]

### 3.4 楽観的分位選択による探索: QUOTA / DLTV

QUOTA(Zhang et al.; Univ. of Alberta / Huawei。AAAI 2019。被引用数 約33。https://arxiv.org/abs/1811.02073)は、平均でなく特定の分位に対して greedy に行動し、どの分位を使うかを option 高位方策がオンライン学習する。

```
A high quantile represents an optimistic estimation of the action value, and action selection based on a high quantile indicates an optimistic exploration strategy.

高い分位は行動価値の楽観的推定を表し、高い分位に基づく行動選択は楽観的探索戦略を意味する。
```
[QUOTA (2018/11)]

DLTV(Mavrin et al.; Univ. of Alberta / Huawei。ICML 2019。被引用数 約107。https://arxiv.org/abs/1905.06125)は、上側分位から計算した左側切断分散を減衰スケジュール付き探索ボーナスとして使う [DLTV (2019/05)]。

### 3.5 Thompson sampling 系(τ ではなく分布・事後からのサンプル)

τ を確率的に引くこと自体を Thompson sampling として定式化した研究は確認できなかった(第 9 節)。近接系統として、学習した return 分布からサンプルして行動選択する Moerland et al.(TU Delft。arXiv/EWRL 2018。https://arxiv.org/abs/1806.04242)、価値分布ネットワークのパラメータ事後分布からサンプルする Tang & Agrawal(Columbia。IJCAI 2018。https://arxiv.org/abs/1805.01907)がある [Moerland+ (2018/06)] [Tang-Agrawal (2018/05)]。

## 4. モンテカルロ分散低減理論と隣接分野での層化・QMC 適用

### 4.1 一般理論: stratified / LHS / antithetic / QMC の保証

比例配分の層化抽出(1 次元で「K 等分ビンに 1 点ずつ」= jittered sampling と同型)は、iid モンテカルロより分散が大きくなり得ないという無条件の保証を持つ。

```
Equation (8.13) allows us to show that stratified sampling with proportional allocation cannot have larger variance than ordinary MC sampling.

式 (8.13) により、比例配分の層化抽出は通常のモンテカルロサンプリングより大きな分散を持ち得ないことが示せる。
```
[owen-mc-ch8 (2018), §8.4 p.13]

```
stratified sampling can never increase variance. In fact, stratification always reduces variance unless the right-hand sum is exactly 0.

層化抽出が分散を増やすことは決してない。実際、右辺の和が厳密に 0 でない限り、層化は常に分散を低減する。
```
[pbrt-13.8 (2018), §13.8.1]

Latin hypercube sampling は被積分関数の加法的成分の分散を漸近的に消去し(Stein 1987)、有限標本でも Var ≤ σ²/(n−1) の保証を持つ(Owen 1997)[owen-mc-ch10 (2013), Proposition 10.1, 10.4]。対蹠変量(antithetic variates)には無条件保証がなく、最悪で分散 2 倍・単調関数なら低減保証となる [owen-mc-ch8 (2018), §8.2]。QMC(Sobol'/Halton 等の low-discrepancy 列)は Koksma–Hlawka 型上界によりほぼ O(n⁻¹) の誤差減衰を示す [hickernell-qmc (2025/02)]。

### 4.2 機械学習での適用例: VDM / Improved DDPM / NeRF / RQMC 方策勾配

「有限サンプルで [0,1] 上の期待値を近似する」同型の問題に対し、隣接分野では層化・低差異列が実際に使われている。

Variational Diffusion Models(Kingma et al.; Google Research。NeurIPS 2021。被引用数 約1,570)は、拡散損失の timestep サンプリングに low-discrepancy sampler を使い分散を低減した。共有オフセット 1 個による等間隔配置で、各 1/k 幅のビンに 1 点が入る。

```
When processing a minibatch of k examples, instead of sampling timesteps independently, we sample a single uniform random number u₀∼U[0,1] and then set tⁱ=mod(u₀+i/k,1).

k 個の例からなるミニバッチを処理するとき、timestep を独立にサンプリングする代わりに、一様乱数 u₀∼U[0,1] を 1 個だけサンプリングし、tⁱ = mod(u₀ + i/k, 1) と設定する。
```
[kingma-vdm (2021/07), Appendix I.1]

Improved DDPM(Nichol & Dhariwal; OpenAI。ICML 2021。被引用数 約5,728)は同じ問題に重点サンプリング(p_t ∝ √E[L_t²])で対処した [nichol-iddpm (2021/02), §3.3]。NeRF(Mildenhall et al.; UC Berkeley ほか。ECCV 2020)はレンダリング積分の求積点を「等間隔ビン内 1 点一様サンプル」で取ることを stratified sampling と呼んで採用している。

```
Instead, we use a stratified sampling approach where we partition [tn,tf] into N evenly-spaced bins and then draw one sample uniformly at random from within each bin.

その代わりに我々は層化サンプリングを使う。[tn, tf] を N 個の等間隔ビンに分割し、各ビンの中から一様ランダムに 1 サンプルを引く。
```
[mildenhall-nerf (2020/03), §4]

RL 一般では、方策勾配・actor-critic の行動サンプリングへ randomized QMC を適用した Arnold et al.(AISTATS 2022)がある [arnold-rqmc-rl (2022), Abstract]。

### 4.3 分位点推定の統計学における層化・重点サンプリング

分位点推定への分散低減は金融リスク管理(VaR)・信頼性解析で確立している: delta-gamma 近似で誘導した重点+層化サンプリング(Glasserman, Heidelberger, Shahabuddin; Management Science 2000)[glasserman-var (2000), Abstract]、メタモデル誘導の適応的 controlled stratification(Cannamela, Garnier, Iooss; Annals of Applied Statistics 2008。被引用数 約90)[cannamela-strat (2008), Abstract]、RQMC 分位点推定の中心極限定理(Kaplan et al.; WSC 2019)[kaplan-rqmc-quantile (2019/12), Abstract]。

## 5. 分布RL の τ への stratified/QMC 適用の探索結果(負の結果を含む)

分布RL(IQN/QR-DQN/FQF 系)の τ サンプリングに層化・jitter・low-discrepancy 列を明示的に適用した論文・実装・記事は、英語 Web 検索の範囲では見つからなかった(2026-08-12 時点。検索クエリ 4 種は `_survey_tau_mc_theory.md` §3.3 に記録)。この不在自体が本調査の主要な成果である [_survey_tau_mc_theory (2026/08), §3.3]。

最近接の査読付き研究は QEMRL(Kuang, Zhu, Zhang, Zhou。ICML 2023。被引用数 約4。https://arxiv.org/abs/2307.16152)で、Q 値推定量(分位平均)の分散低減の十分条件が QR-DQN の等間隔 τ では成り立つが IQN の一様サンプル τ では成り立たないことを明示的に指摘している。ただし提案は推定量(重み付け)側の改良であり、τ のサンプリング方式自体の置き換えではない。

```
IQN does not satisfy the sufficient condition zτi = −zτN−i since τ is sampled from a uniform distribution, rather than evenly spaced as in QDRL.

IQN は十分条件 zτi = −zτN−i を満たさない。τ が、QDRL(QR-DQN)のような等間隔ではなく、一様分布からサンプルされるためである。
```
[kuang-qemrl (2023/07), Appendix E.1]

## 6. 主要フレームワーク実装の現状

公式・第三者の主要実装(Dopamine JAX/TF、DeepMind DQN Zoo、rlax、Acme、pfrl、Tianshou、d3rlpy、SB3-contrib、DI-engine、toshikwa/fqf-iqn-qrdqn.pytorch、BY571/IQN-and-Extensions)を横断確認した結果、IQN の学習時 τ はすべて素朴な一様乱数であり、stratified・ソート・quasi-random 等の工夫は確認できなかった [_survey_tau_implementations (2026/08), §9.4]。代表例として DeepMind 公式リファレンス実装:

```python
def _sample_tau(
    rng_key: parts.PRNGKey,
    shape: Tuple[int, ...],
) -> jnp.ndarray:
  """Samples tau values uniformly between 0 and 1."""
  return jax.random.uniform(rng_key, shape=shape)
```
[google-deepmind/dqn_zoo master (2026/08 参照), dqn_zoo/iqn/agent.py]

一様乱数以外に実在するバリエーションは次の 3 系統のみである [_survey_tau_implementations (2026/08), §9.4]:

1. 評価/決定的行動時の等間隔 linspace 切替 — pfrl(`act_deterministically=True` 時)、d3rlpy(train/eval 分岐)。

```python
def _make_taus(
    batch_size: int, n_quantiles: int, training: bool, device: torch.device
) -> torch.Tensor:
    if training:
        taus = torch.rand(batch_size, n_quantiles, device=device)
    else:
        taus = torch.linspace(start=0, end=1, steps=n_quantiles, ...)
```
[takuseno/d3rlpy master (2026/08 参照), d3rlpy/models/torch/q_functions/iqn_q_function.py(一部省略)]

2. 行動選択側 τ へのリスク歪曲(beta function)適用 — DI-engine の `QuantileHead` [opendilab/DI-engine main (2026/08 参照), ding/model/common/head.py]。
3. τ 自体の学習(FQF 系 fraction proposal network、log_softmax→cumsum→中点 detach)— Tianshou / DI-engine / toshikwa。実装系譜は toshikwa(ku2482)/fqf-iqn-qrdqn.pytorch が事実上の共通祖先(Tianshou が docstring で明記)[tianshou discrete nets (2026/08)]。

τ サンプリングの偏りや stratified 化を主題にした GitHub issue・実務者ブログも見つからなかった [_survey_tau_implementations (2026/08), §10.6, §11]。

## 7. 批判・懸念(ソースが自己申告・相互批判したもの)

FQF の自己申告: fraction proposal network の収束保証がなく、追加ネットワークにより IQN 比約 20% 低速である。

```
However, one side effect of the full parameterization in FQF is that the training speed is decreased. With same settings, FQF is roughly 20% slower than IQN due to the additional fraction proposal network.

しかし、FQF における完全パラメータ化の副作用の一つは学習速度の低下である。同一設定で、FQF は追加の fraction proposal network のために IQN よりおよそ 20% 遅い。
```
[FQF (2019/11), Discussion/Conclusion]

NDQFN による IQN/FQF 批判: 毎 iteration の τ 再サンプルは、固定分位位置を前提とする分布ベース探索ボーナス(DLTV 型)を極めて不安定にする。

```
the original DLTV method requires all the quantile locations to be fixed while IQN or FQF resample the quantile locations at each training iteration and the bonus term could be extremely unstable

元の DLTV 法は全ての quantile 位置が固定されていることを要求するが、IQN や FQF は各学習 iteration で quantile 位置を再サンプルするため、ボーナス項は極めて不安定になり得る
```
[NDQFN (2021/05), Section 1]

NC-QR-DQN による固定 grid 系批判: 非交差保証がないと方策探索の方向が歪み、最適行動の選択が学習エポック間で大きく変動する [NC-QR (2020/12), Section 1]。FQF 自身も「学習中の quantile fraction の選択はどの程度重要か」を open question として残している [FQF (2019/11), Discussion/Conclusion]。

## 8. 総合評価

- τ の配置方式として文献上確立しているのは「固定一様 grid(QR-DQN 系)」「iid 一様サンプル(IQN 系)」「学習(FQF)」の 3 つであり、両者の中間である「被覆保証つきランダム化(層化/jitter/低差異列)」は、分布RL の τ に限っては論文・主要実装・実務者議論のいずれにも確認できなかった(第 5, 6 節)。
- 一方でその中間形は、モンテカルロ分散低減の一般理論では「比例配分の層化は分散を増やさない」という無条件保証を持つ標準技法であり(第 4.1 節)、拡散モデルの timestep・NeRF の求積点など「[0,1] 上の期待値を少数サンプルで近似する」同型問題では実際に採用されている(第 4.2 節)。
- 分布RL 側にも接点となる指摘は存在する: QEMRL は IQN の一様 τ が等間隔 τ の持つ分散低減条件を満たさないことを明示し(第 5 節)、NDQFN は τ 再サンプルの分散が探索ボーナスを不安定化させると批判し(第 7 節)、MSR は FQF の動機として「サンプルされた fraction は最良とは限らない」と述べている(第 2.3 節)。τ の取り方が推定分散・安定性に影響するという認識は複数ソースにあるが、その解として提示されてきたのは「固定化」「学習(FQF)」「本数の増加(IQN ablation)」であり、サンプリング分布の層化はいずれのソースでも検討されていない。
- τ 分布の非一様化そのものは risk-sensitive 文脈で豊富な先行がある(distortion β(τ)、CVaR の区間制限、適応的リスクレベル)。これらは「どの分位を重視するか」を目的関数側で変える系統であり、「全域を均等に被覆する」目的の層化とは動機が直交する(第 3 節)。

## 9. 調査の限界

- 被引用数は Semantic Scholar(2026-08-12)の概数。レート制限で取得できなかったもの(D4PG、NC-QR-DQN、DSAC-Ma、NeRF、McKay 1979、Glasserman 2000、Arnold 2022)は OpenAlex 補完または「未確認」。OpenAlex は arXiv 版と出版版のレコード分割により過小に出る傾向がある。
- 「見つからなかった」系の結論(τ への層化/QMC 適用、τ の importance sampling、τ の Thompson sampling 解釈、stratified τ の GitHub issue)は英語 Web 検索+ヒットした論文フルテキスト確認に基づく。arXiv 全文検索・GitHub コード検索 API・中国語文献は未走査であり、悉皆調査ではない。
- 実装コードの参照はすべて default branch の 2026-08-12 時点取得であり、commit hash は未固定。
- 原文引用の多くは ar5iv / arXiv HTML 変換テキスト由来で、数式表記は Unicode/ASCII に転記した(語句は原文通り)。
- FQF の「学習された fraction が最終的にほぼ一様に退化する」という後続の再現報告は一次ソースを確認できなかった(citation not confirmed)。
- 各中間レポート固有の限界は `_survey_tau_core_papers.md` §10、`_survey_tau_risk_adaptive.md` §8、`_survey_tau_mc_theory.md` §5、`_survey_tau_implementations.md` 冒頭凡例を参照。

## 10. ソース一覧

[QR-DQN, 2017/10] Will Dabney, Mark Rowland, Marc G. Bellemare, Rémi Munos. "Distributional Reinforcement Learning with Quantile Regression." AAAI 2018. https://arxiv.org/abs/1710.10044

[IQN, 2018/06] Will Dabney, Georg Ostrovski, David Silver, Rémi Munos. "Implicit Quantile Networks for Distributional Reinforcement Learning." ICML 2018. https://arxiv.org/abs/1806.06923

[FQF, 2019/11] Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, Tie-Yan Liu. "Fully Parameterized Quantile Function for Distributional Reinforcement Learning." NeurIPS 2019. https://arxiv.org/abs/1911.02140

[NC-QR, 2020/12] Fan Zhou, Jianing Wang, Xingdong Feng. "Non-crossing quantile regression for deep reinforcement learning." NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/file/b6f8dc086b2d60c5856e4ff517060392-Paper.pdf

[NDQFN, 2021/05] Fan Zhou, Zhoufan Zhu, Qi Kuang, Liwen Zhang. "Non-decreasing Quantile Function Network with Efficient Exploration for Distributional Reinforcement Learning." IJCAI 2021. https://arxiv.org/abs/2105.06696

[TQC, 2020/05] Arsenii Kuznetsov, Pavel Shvechikov, Alexander Grishin, Dmitry Vetrov. "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics." ICML 2020. https://arxiv.org/abs/2005.04269

[D4PG, 2018/04] Gabriel Barth-Maron, Matthew W. Hoffman, David Budden, Will Dabney, Dan Horgan, Dhruva TB, Alistair Muldal, Nicolas Heess, Timothy Lillicrap. "Distributed Distributional Deterministic Policy Gradients." ICLR 2018 (Poster). https://arxiv.org/abs/1804.08617

[DSAC-Ma, 2020/04] Xiaoteng Ma, Junyao Chen, Li Xia, Jun Yang, Qianchuan Zhao, Zhengyuan Zhou. "DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning." arXiv(JAIR 採録). https://arxiv.org/abs/2004.14547

[DSAC-Duan, 2020/01] Jingliang Duan, Yang Guan, Shengbo Eben Li, Yangang Ren, Qi Sun, Bo Cheng. "Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors." IEEE Trans. Neural Networks Learn. Syst. 33(11), 2022. https://arxiv.org/abs/2001.02811

[WCPG, 2019/11] Yichuan Charlie Tang, Jian Zhang, Ruslan Salakhutdinov. "Worst Cases Policy Gradients." CoRL 2019. https://arxiv.org/abs/1911.03618

[Keramati+, 2019/11] Ramtin Keramati, Christoph Dann, Alex Tamkin, Emma Brunskill. "Being Optimistic to Be Conservative: Quickly Learning a CVaR Policy." AAAI 2020. https://arxiv.org/abs/1911.01546

[CODAC, 2021/07] Yecheng Jason Ma, Dinesh Jayaraman, Osbert Bastani. "Conservative Offline Distributional Reinforcement Learning." NeurIPS 2021. https://arxiv.org/abs/2107.06106

[QUOTA, 2018/11] Shangtong Zhang, Borislav Mavrin, Linglong Kong, Bo Liu, Hengshuai Yao. "QUOTA: The Quantile Option Architecture for Reinforcement Learning." AAAI 2019. https://arxiv.org/abs/1811.02073

[DLTV, 2019/05] Borislav Mavrin, Hengshuai Yao, Linglong Kong, Kaiwen Wu, Yaoliang Yu. "Distributional Reinforcement Learning for Efficient Exploration." ICML 2019. https://arxiv.org/abs/1905.06125

[ARA, 2021/06] Frederik Schubert, Theresa Eimer, Bodo Rosenhahn, Marius Lindauer. "Automatic Risk Adaptation in Distributional Reinforcement Learning." arXiv / ICML 2021 Workshop RL4RealLife. https://arxiv.org/abs/2106.06317

[ART-IQN, 2022/03] Cheng Liu, Erik-Jan van Kampen, Guido C.H.E. de Croon. "Adaptive Risk-Tendency: Nano Drone Navigation in Cluttered Environments with Distributional Reinforcement Learning." ICRA 2023. https://arxiv.org/abs/2203.14749

[DRL-ORA, 2023/10] Yupeng Wu, Wenyun Li, Wenjie Huang, Chin Pang Ho. "DRL-ORA: Distributional Reinforcement Learning with Online Risk Adaption." arXiv preprint. https://arxiv.org/abs/2310.05179

[TOP, 2021/02] Ted Moskovitz, Jack Parker-Holder, Aldo Pacchiano, Michael Arbel, Michael I. Jordan. "Tactical Optimism and Pessimism for Deep Reinforcement Learning." NeurIPS 2021. https://arxiv.org/abs/2102.03765

[Moerland+, 2018/06] Thomas M. Moerland, Joost Broekens, Catholijn M. Jonker. "The Potential of the Return Distribution for Exploration in RL." arXiv (EWRL 2018). https://arxiv.org/abs/1806.04242

[Tang-Agrawal, 2018/05] Yunhao Tang, Shipra Agrawal. "Exploration by Distributional Reinforcement Learning." IJCAI 2018. https://arxiv.org/abs/1805.01907

[owen-mc-ch8, 2018] Art B. Owen. "Monte Carlo theory, methods and examples, Chapter 8: Variance reduction." オンライン教科書 (Stanford University). https://artowen.su.domains/mc/Ch-var-basic.pdf

[owen-mc-ch10, 2013] Art B. Owen. "Monte Carlo theory, methods and examples, Chapter 10: Advanced variance reduction." オンライン教科書 (Stanford University). https://artowen.su.domains/mc/Ch-var-adv.pdf

[pbrt-13.8, 2018] Matt Pharr, Wenzel Jakob, Greg Humphreys. "Physically Based Rendering, 3rd ed., §13.8 Careful Sample Placement." https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Careful_Sample_Placement

[hickernell-qmc, 2025/02] Fred J. Hickernell, Nathan Kirk, Aleksei G. Sorokin. "Quasi-Monte Carlo Methods: What, Why, and How?" arXiv:2502.03644. https://arxiv.org/abs/2502.03644

[buchholz-qmcvi, 2018/07] Alexander Buchholz, Florian Wenzel, Stephan Mandt. "Quasi-Monte Carlo Variational Inference." ICML 2018. https://arxiv.org/abs/1807.01604

[kingma-vdm, 2021/07] Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho. "Variational Diffusion Models." NeurIPS 2021. https://arxiv.org/abs/2107.00630

[nichol-iddpm, 2021/02] Alex Nichol, Prafulla Dhariwal. "Improved Denoising Diffusion Probabilistic Models." ICML 2021. https://arxiv.org/abs/2102.09672

[mildenhall-nerf, 2020/03] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." ECCV 2020. https://arxiv.org/abs/2003.08934

[arnold-rqmc-rl, 2022] Sébastien M. R. Arnold, Pierre L'Ecuyer, Liyu Chen, Yi-Fan Chen, Fei Sha. "Policy Learning and Evaluation with Randomized Quasi-Monte Carlo." AISTATS 2022. https://proceedings.mlr.press/v151/arnold22a.html

[glasserman-var, 2000] Paul Glasserman, Philip Heidelberger, Perwez Shahabuddin. "Variance Reduction Techniques for Estimating Value-at-Risk." Management Science 46(10). https://pubsonline.informs.org/doi/10.1287/mnsc.46.10.1349.12274

[cannamela-strat, 2008] Claire Cannamela, Josselin Garnier, Bertrand Iooss. "Controlled stratification for quantile estimation." Annals of Applied Statistics 2(4). https://arxiv.org/abs/0802.2426

[kaplan-rqmc-quantile, 2019/12] Zachary T. Kaplan, Yajuan Li, Marvin K. Nakayama, Bruno Tuffin. "Randomized Quasi-Monte Carlo for Quantile Estimation." Winter Simulation Conference 2019. https://web.njit.edu/~marvin/papers/wsc19-rqmc.pdf

[kuang-qemrl, 2023/07] Qi Kuang, Zhoufan Zhu, Liwen Zhang, Fan Zhou. "Variance Control for Distributional Reinforcement Learning." ICML 2023. https://arxiv.org/abs/2307.16152

[MSR FQF blog, 2019/12] Li Zhao. "Finding the best learning targets automatically: Fully Parameterized Quantile Function for distributional RL." Microsoft Research Blog. https://www.microsoft.com/en-us/research/blog/finding-the-best-learning-targets-automatically-fully-parameterized-quantile-function-for-distributional-rl/

[dqn_zoo iqn, 2026/08] Google DeepMind. "dqn_zoo/iqn/agent.py (master)." GitHub. https://github.com/google-deepmind/dqn_zoo/blob/master/dqn_zoo/iqn/agent.py

[d3rlpy iqn, 2026/08] Takuma Seno. "d3rlpy/models/torch/q_functions/iqn_q_function.py (master)." GitHub. https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/models/torch/q_functions/iqn_q_function.py

[opendilab/DI-engine head, 2026/08] OpenDILab. "ding/model/common/head.py (main)." GitHub. https://github.com/opendilab/DI-engine/blob/main/ding/model/common/head.py

[tianshou discrete nets, 2026/08] THU-ML. "tianshou/utils/net/discrete.py (master)." GitHub. https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/discrete.py

[_survey_tau_core_papers, 2026/08] 本調査中間レポート(基礎系譜)。reports/_survey_tau_core_papers.md

[_survey_tau_risk_adaptive, 2026/08] 本調査中間レポート(risk-sensitive・適応系)。reports/_survey_tau_risk_adaptive.md

[_survey_tau_mc_theory, 2026/08] 本調査中間レポート(MC 分散低減理論)。reports/_survey_tau_mc_theory.md

[_survey_tau_implementations, 2026/08] 本調査中間レポート(フレームワーク実装)。reports/_survey_tau_implementations.md
