# Survey: モンテカルロ積分の分散低減サンプリング理論と quantile/τ サンプリングへの適用

Date: 2026-08-12
Scope: (1) モンテカルロ積分の分散低減サンプリング技法の一般理論(stratified / LHS / antithetic / QMC / jittered)、(2) 機械学習の期待値推定への適用事例、(3) 分布強化学習の τ サンプリングへの QMC/stratified/jitter 適用の有無、(4) 分位点推定における層化・重点サンプリングの統計学的知見。

## Table of Contents

1. [一般理論: 分散低減サンプリング技法](#1-一般理論-分散低減サンプリング技法)
   - 1.1 層化抽出 (stratified sampling)
   - 1.2 Latin hypercube sampling (LHS)
   - 1.3 対蹠変量 (antithetic variates)
   - 1.4 準モンテカルロ / low-discrepancy 列 (Sobol', Halton, Koksma–Hlawka)
   - 1.5 Jittered sampling(CG レンダリング文脈)
2. [機械学習の期待値推定への適用事例](#2-機械学習の期待値推定への適用事例)
   - 2.1 変分推論への QMC 適用 (Buchholz et al. 2018)
   - 2.2 拡散モデルの timestep サンプリング(low-discrepancy / importance sampling)
   - 2.3 NeRF のレイ上サンプリングにおける stratified sampling
   - 2.4 強化学習の方策勾配への RQMC 適用 (Arnold et al. 2022)
3. [分布RLの τ サンプリングへの QMC/stratified/jitter 適用(探索結果)](#3-分布rlの-τ-サンプリングへの-qmcstratifiedjitter-適用探索結果)
   - 3.1 現状: IQN 系の τ は一様サンプリング、QR-DQN は等間隔固定
   - 3.2 最近接の研究: τ の等間隔性と分散低減条件を論じた QEMRL
   - 3.3 明示的に QMC/stratified/jittered τ を使った研究: 見つからなかった
4. [分位点推定における層化・重点サンプリングの統計学的知見](#4-分位点推定における層化重点サンプリングの統計学的知見)
5. [調査の限界](#5-調査の限界)
6. [ソース一覧](#6-ソース一覧)

---

## 1. 一般理論: 分散低減サンプリング技法

一般理論の主典拠として、Art Owen(Stanford University)のオンライン教科書 "Monte Carlo theory, methods and examples" の第8章(基本の分散低減)・第10章(発展的分散低減)を用いる。教科書級ソースであり、各命題に原典(Stein 1987, Owen 1997 等)への帰属が明記されている。

### 1.1 層化抽出 (stratified sampling)

層化抽出は、積分域を排反な層(strata)に分割し、各層から所定数のサンプルを取る方法である。比例配分(proportional allocation, nj = nωj)の層化は、通常の iid モンテカルロより分散が大きくなることはない。これは層内分散と層間分散への分解(σ² = σ²_W + σ²_B)から従う。

```
Equation (8.13) allows us to show that stratified sampling with proportional allocation cannot have larger variance than ordinary MC sampling.

式 (8.13) により、比例配分の層化抽出は通常のモンテカルロサンプリングより大きな分散を持ち得ないことが示せる。
```
[owen-mc-ch8 (2018), §8.4 p.13]

レンダリング系の標準教科書 PBRT(Physically Based Rendering, 3rd ed.)も同じ結論を、モンテカルロ積分の文脈で明言している(層平均がすべて等しい場合のみ分散低減がゼロになる)。

```
stratified sampling can never increase variance. In fact, stratification always reduces variance unless the right-hand sum is exactly 0.

層化抽出が分散を増やすことは決してない。実際、右辺の和が厳密に 0 でない限り、層化は常に分散を低減する。
```
[pbrt-13.8 (2018), §13.8.1]

### 1.2 Latin hypercube sampling (LHS)

LHS は d 次元の各座標軸を n 等分し、各軸のすべての層に 1 点ずつ入るようにサンプルを配置する「全軸同時 1 次元層化」である。導入は McKay, Beckman, Conover (1979, Technometrics)。Owen の教科書の章末注は、原論文の内容(重要変数を知らなくても自動的に層化されること、各入力に単調な関数では iid より分散が小さいことの証明)を次のように要約している。

```
They introduced Latin hypercube sampling as a way to explore computationally the input space of a function, pointing out that it automatically stratifies on the important variables without the user having to know which those are. They prove that Var(µ̂LHS) ⩽ Var(µ̂IID) whenever the function being sampled is monotone in each of its d input variables.

彼らは Latin hypercube sampling を、関数の入力空間を計算的に探索する方法として導入し、ユーザーがどれが重要変数かを知らなくても自動的に重要変数上で層化されることを指摘した。彼らは、サンプリング対象の関数が d 個の入力変数それぞれについて単調であれば常に Var(µ̂LHS) ⩽ Var(µ̂IID) となることを証明している。
```
[owen-mc-ch10 (2013), §10.3 章末注]

LHS の漸近分散は、被積分関数の「加法的成分」を完全に除去した残差のみで決まる(Stein 1987 の結果)。

```
Proposition 10.1. ... Then Var(µ̂LHS) = (1/n) ∫ e(x)² dx + o(1/n), where e(x) = f(x) − f_add(x). Proof. Stein (1987). Proposition 10.1 shows us that the additive part of f does not contribute to the asymptotic variance in Latin hypercube sampling.

命題 10.1. …このとき Var(µ̂LHS) = (1/n)∫e(x)²dx + o(1/n)。ここで e(x) = f(x) − f_add(x)(加法的最良近似からの残差)。証明は Stein (1987)。命題 10.1 は、f の加法的部分が LHS の漸近分散に寄与しないことを示している。
```
[owen-mc-ch10 (2013), §10.3 Proposition 10.1]

さらに LHS は「iid よりも大きく悪化することはない」という有限標本の保証を持つ(Owen 1997 の結果)。

```
Proposition 10.4. ... For n ⩾ 2 let X1, ..., Xn be a Latin hypercube sample. Then Var(µ̂LHS) ⩽ σ²/(n − 1). Proof. Owen (1997). The point of Proposition 10.4 is that while Latin hypercube sampling can be much better than IID sampling, it cannot be much worse.

命題 10.4. … n ⩾ 2 に対し X1,…,Xn を Latin hypercube サンプルとすると Var(µ̂LHS) ⩽ σ²/(n − 1)。証明は Owen (1997)。命題 10.4 の要点は、LHS は IID サンプリングよりはるかに良くなり得る一方で、大きく悪くなることはあり得ない、という点である。
```
[owen-mc-ch10 (2013), §10.4 Proposition 10.4]

### 1.3 対蹠変量 (antithetic variates)

対蹠変量は、サンプル x とその「反対点」x̃(例: U(0,1)^d なら 1−x)のペアで平均を取る方法で、Hammersley and Morton (1956) により導入された。

```
Antithetic sampling was introduced by Hammersley and Morton (1956).

対蹠サンプリングは Hammersley and Morton (1956) によって導入された。
```
[owen-mc-ch8 (2018), 章末注]

対蹠変量の分散は Var(µ̂_anti) = (σ²/n)(1+ρ)(ρ はペア間相関)であり、層化と異なり無条件の保証はない。最良で分散ゼロ、最悪で 2 倍になる。分散低減の十分条件は各入力についての単調性である。

```
In the best case, antithetic sampling gives the exact answer from just one pair of function evaluations. In the worst case it doubles the variance. Both cases do arise.

最良の場合、対蹠サンプリングはたった 1 組の関数評価から正確な答えを与える。最悪の場合は分散を 2 倍にする。どちらの場合も実際に起こり得る。
```
[owen-mc-ch8 (2018), §8.2 p.7]

```
Monotonicity of f is a safe harbor: if f is monotone then we're sure antithetic sampling will reduce the variance.

f の単調性は安全圏である: f が単調であれば、対蹠サンプリングが分散を低減することが保証される。
```
[owen-mc-ch8 (2018), §8.2 p.7]

なお対蹠サンプリングは被積分関数の「奇成分」の分散寄与を消去し「偶成分」の寄与を 2 倍にする、という分解(式 8.4)も同章に示されている(spiky な被積分関数では悪化し得る)。

### 1.4 準モンテカルロ / low-discrepancy 列 (Sobol', Halton, Koksma–Hlawka)

準モンテカルロ(QMC)は iid サンプルを決定的な low-discrepancy 点列に置き換える。誤差は「被積分関数の変動 × 点集合の discrepancy」で上から抑えられる(Koksma–Hlawka 型不等式)。近年のレビュー(Hickernell, Kirk, Sorokin 2025, Illinois Institute of Technology)は再生核ヒルベルト空間の枠組みでこの不等式を提示している。

```
|μ(f)−μ̂ₙ(f)| ≤ discrepancy({xᵢ}ᵢ₌₀ⁿ⁻¹, K) · variation(f, K)

|μ(f)−μ̂ₙ(f)| ≤ 点集合の discrepancy × 関数の variation(Koksma–Hlawka 型誤差上界)
```
[hickernell-qmc (2025/02), eq.(25) 周辺]

同レビューは、QMC の誤差がほぼ O(n⁻¹) で減衰し、単純モンテカルロの O(n⁻¹ᐟ²) に対して大幅な改善となることを述べる。

```
The error of qMC methods decays nearly like O(n⁻¹), which for this example corresponds to a reduction in error of several orders of magnitude compared to simple MC

qMC 法の誤差はほぼ O(n⁻¹) で減衰し、この例では単純 MC と比べて数桁の誤差低減に相当する
```
[hickernell-qmc (2025/02)]

代表的な low-discrepancy 列として Halton 列(基数の異なる van der Corput 列の組)とSobol' 列(基数 2 の digital sequence の最初期の構成)が挙げられている。

```
The Halton sequence is defined in terms of the van der Corput sequences for different bases

Halton 列は、異なる基数の van der Corput 列によって定義される
```
[hickernell-qmc (2025/02)]

```
The earliest instance is due to Sobol'

(digital sequence の)最初期の実例は Sobol' による
```
[hickernell-qmc (2025/02)]

原典の書誌情報: Sobol' (1967) "On the distribution of points in a cube and the approximate evaluation of integrals" USSR Computational Mathematics and Mathematical Physics 7(4):86–112、Halton (1960) "On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals" Numerische Mathematik 2:84–90(いずれも書誌のみ確認、原文引用は未取得)。

### 1.5 Jittered sampling(CG レンダリング文脈)

jittered sampling(規則格子の各セル内で 1 点をランダムに揺らす)は、1 次元の [0,1] を n 等分して各ビンから 1 点取る層化抽出と同型の構成である。CG では Cook (1986, Pixar, ACM Transactions on Graphics) が導入し、規則サンプリングのエイリアシングをノイズに変換する効果を示した。

```
In this paper it is shown that these artifacts are not an inherent part of point sampling, but a consequence of using regularly spaced samples. If the samples occur at appropriate nonuniformly spaced locations, frequencies above the Nyquist limit do not alias, but instead appear as noise of the correct average intensity.

本論文では、これらのアーティファクト(エイリアシング)が点サンプリングに固有のものではなく、等間隔サンプルを使うことの帰結であることを示す。適切な非一様位置でサンプルを取れば、Nyquist 限界を超える周波数はエイリアスにならず、正しい平均強度のノイズとして現れる。
```
[cook-tog (1986/01), Abstract]

```
Jittering, or adding noise to sample locations, is a form of stochastic sampling that can be used to approximate a Poisson disk distribution.

ジッタリング、すなわちサンプル位置にノイズを加えることは、Poisson disk 分布を近似するために使える確率的サンプリングの一形態である。
```
[cook-tog (1986/01), §4.1]

なお PBRT §13.8 は、この jittered(= 層化)アプローチの分散低減保証を 1.1 節の引用のとおりモンテカルロ積分の言葉で与えている。

## 2. 機械学習の期待値推定への適用事例

### 2.1 変分推論への QMC 適用 (Buchholz et al. 2018)

Buchholz(ENSAE-CREST), Wenzel(TU Kaiserslautern), Mandt(Disney Research)は、モンテカルロ変分推論(MCVI)の確率的勾配の分散低減に QMC を適用した(ICML 2018)。被引用数: 約70(Semantic Scholar, 2026-08-12 確認)。

```
qmc replaces N i.i.d. samples from a uniform probability distribution by a deterministic sequence of samples of length N. This sequence covers the underlying random variable space more evenly than i.i.d. draws, reducing the variance of the gradient estimator.

QMC は一様分布からの N 個の i.i.d. サンプルを、長さ N の決定的なサンプル列で置き換える。この列は下層の確率変数空間を i.i.d. サンプルよりも均等に被覆し、勾配推定量の分散を低減する。
```
[buchholz-qmcvi (2018/07), Abstract]

### 2.2 拡散モデルの timestep サンプリング(low-discrepancy / importance sampling)

Variational Diffusion Models(Kingma, Salimans, Poole, Ho; Google Research, NeurIPS 2021)は、拡散損失の timestep t のミニバッチサンプリングに low-discrepancy sampler を使い、損失推定量の分散を低減した。被引用数: 約1,570(Semantic Scholar, 2026-08-12 確認)。この構成は本レポート 1.1 節の層化(各 1/k 幅のビンに 1 点)と同じ被覆を、共有オフセット u₀ 1 個で実現するものである。

```
When processing a minibatch of k examples, instead of sampling timesteps independently, we sample a single uniform random number u₀∼U[0,1] and then set tⁱ=mod(u₀+i/k,1).

k 個の例からなるミニバッチを処理するとき、timestep を独立にサンプリングする代わりに、一様乱数 u₀∼U[0,1] を 1 個だけサンプリングし、tⁱ = mod(u₀ + i/k, 1) と設定する。
```
[kingma-vdm (2021/07), Appendix I.1]

Improved DDPM(Nichol and Dhariwal, OpenAI, ICML 2021)は同じ「timestep 上の期待値のサンプル近似」の分散問題に対し、層化ではなく重点サンプリング(p_t ∝ √E[L_t²])で対処した。被引用数: 約5,728(Semantic Scholar, 2026-08-12 確認)。

```
we hypothesized that sampling t uniformly causes unnecessary noise in the L_vlb objective.

t を一様にサンプリングすることが L_vlb 目的関数に不要なノイズを生じさせている、と我々は仮説を立てた。
```
[nichol-iddpm (2021/02), §3.3]

### 2.3 NeRF のレイ上サンプリングにおける stratified sampling

NeRF(Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng; UC Berkeley / Google Research / UC San Diego, ECCV 2020)は、ボリュームレンダリング積分の求積点を等間隔ビン内一様サンプル(= jittered / 層化)で取る。

```
Instead, we use a stratified sampling approach where we partition [tn,tf] into N evenly-spaced bins and then draw one sample uniformly at random from within each bin.

その代わりに我々は層化サンプリングを使う。[tn, tf] を N 個の等間隔ビンに分割し、各ビンの中から一様ランダムに 1 サンプルを引く。
```
[mildenhall-nerf (2020/03), §4]

NeRF 論文中での層化の動機は分散低減ではなく、最適化の過程で MLP が連続位置で評価されること(連続表現の学習)に置かれている点に注意。

```
Although we use a discrete set of samples to estimate the integral, stratified sampling enables us to represent a continuous scene representation because it results in the MLP being evaluated at continuous positions over the course of optimization.

積分の推定には離散的なサンプル集合を使うものの、層化サンプリングによって連続なシーン表現を表せる。最適化の過程で MLP が連続的な位置で評価されることになるからである。
```
[mildenhall-nerf (2020/03), §4]

### 2.4 強化学習の方策勾配への RQMC 適用 (Arnold et al. 2022)

分布RLではないが、強化学習の期待値推定(方策勾配・actor-critic)へ randomized QMC を適用した研究として Arnold, L'Ecuyer, Chen, Chen, Sha(AISTATS 2022, PMLR v151)がある。行動サンプリングの MC を low-discrepancy 点集合で置き換える。

```
combine policy gradient methods with Randomized Quasi-Monte Carlo, yielding variance-reduced formulations of policy gradient and actor-critic algorithms

方策勾配法を Randomized Quasi-Monte Carlo と組み合わせ、方策勾配および actor-critic アルゴリズムの分散低減版の定式化を得る
```
[arnold-rqmc-rl (2022), Abstract]

```
replacing Monte Carlo with Quasi-Monte Carlo yields significantly more accurate gradient estimates

モンテカルロを準モンテカルロで置き換えると、著しく高精度な勾配推定が得られる
```
[arnold-rqmc-rl (2022), Abstract]

## 3. 分布RLの τ サンプリングへの QMC/stratified/jitter 適用(探索結果)

### 3.1 現状: IQN 系の τ は一様サンプリング、QR-DQN は等間隔固定

IQN(Dabney, Ostrovski, Silver, Munos; DeepMind, ICML 2018)は、基底分布からのサンプル(τ)を分位値へ写す暗黙的分位関数を学習する。被引用数: 約722(Semantic Scholar, 2026-08-12 確認)。

```
By reparameterizing a distribution over the sample space, this yields an implicitly defined return distribution and gives rise to a large class of risk-sensitive policies.

サンプル空間上の分布を再パラメータ化することで、暗黙的に定義されたリターン分布が得られ、リスク感応的方策の大きなクラスが導かれる。
```
[dabney-iqn (2018/06), Abstract]

IQN の τ が current/target とも U([0,1]) からの一様サンプルであることは、2026 年の後続論文でも次のように記述されており、現在も標準のままである。

```
IQN samples current fractions {τi} for the quantile levels to be fitted at (s,a) and target fractions {τ'j} for bootstrapped next-state quantile targets, both from U([0,1])

IQN は、(s,a) でフィットする分位レベルの current fractions {τi} と、次状態のブートストラップ分位ターゲット用の target fractions {τ'j} を、いずれも U([0,1]) からサンプルする
```
[zhang-rqiqn (2026/05), §2.3]

一方 FQF(Yang, Zhao, Lin, Qin, Bian, Liu; NeurIPS 2019, 被引用数約185, Semantic Scholar 2026-08-12 確認)は、τ を「サンプリングする」のでなく fraction proposal network で学習する路線であり、分散低減サンプリング(層化/QMC)とは別方向の解である。

```
Existing distributional RL algorithms parameterize either the probability side or the return value side of the distribution function, leaving the other side uniformly fixed as in C51, QR-DQN or randomly sampled as in IQN.

既存の分布RLアルゴリズムは、分布関数の確率側か値側のどちらかをパラメータ化し、もう一方は C51 や QR-DQN のように一様固定にするか、IQN のようにランダムサンプルにしている。
```
[yang-fqf (2019/11), Abstract]

### 3.2 最近接の研究: τ の等間隔性と分散低減条件を論じた QEMRL

今回の調査で「τ の取り方(等間隔 vs 一様サンプル)と推定分散」を明示的に結び付けている唯一の査読付き論文は、Variance Control for Distributional Reinforcement Learning(Kuang, Zhu, Zhang, Zhou; ICML 2023)であった。被引用数: 約4(Semantic Scholar, 2026-08-12 確認)。同論文は Q 値推定量(分位平均)の分散を制御する QEM 推定量を提案し、その分散低減の十分条件が QR-DQN の等間隔 τ では成り立つが、IQN の一様サンプル τ では成り立たないことを指摘している。

```
Moreover, one important sufficient condition zτi = −zτN−i which ensures the reduction of variance does not hold in the IQN case as τ's are sampled from a uniform distribution. However, according to the simulation results in Table 4, the variance reduction still remains valid in practice.

さらに、分散低減を保証する重要な十分条件 zτi = −zτN−i は、τ が一様分布からサンプルされる IQN の場合には成り立たない。ただし Table 4 のシミュレーション結果によれば、実際には分散低減は依然として有効である。
```
[kuang-qemrl (2023/07), §5.2]

```
IQN does not satisfy the sufficient condition zτi = −zτN−i since τ is sampled from a uniform distribution, rather than evenly spaced as in QDRL.

IQN は十分条件 zτi = −zτN−i を満たさない。τ が、QDRL(QR-DQN)のような等間隔ではなく、一様分布からサンプルされるためである。
```
[kuang-qemrl (2023/07), Appendix E.1]

注意: 同論文は τ のサンプリング方式自体を層化や QMC に置き換える提案はしておらず、推定量(分位値の重み付け)側の改良である。

### 3.3 明示的に QMC/stratified/jittered τ を使った研究: 見つからなかった

以下のクエリ群で検索した範囲では、分布RL(IQN/QR-DQN/FQF 系)の τ サンプリングに Sobol/Halton などの low-discrepancy 列、層化抽出、jittered sampling を明示的に適用した論文・実装・記事は**見つからなかった**(2026-08-12 時点、WebSearch による英語検索)。

- "IQN stratified tau sampling quantile fraction low-discrepancy distributional reinforcement learning"
- "quasi-Monte Carlo" OR "Sobol" "distributional reinforcement learning" quantile tau sampling
- "IQN" OR "implicit quantile" "stratified" tau OR fractions sampling variance reduction reinforcement learning github
- github IQN tau "sobol" OR "halton" OR "stratified" quantile sampling implementation distributional RL

検索でヒットした IQN 実装(BY571/IQN-and-Extensions、DI-engine、deep_rl_zoo 等)はいずれも τ の一様サンプリングを実装しており、検索結果の要約でも QMC/層化 τ の実装は確認されなかった。2026 年の IQN 派生研究(zhang-rqiqn)も τ~U([0,1]) のみを扱い、stratified/QMC/jittered τ への言及はない(同論文フルテキスト HTML に対する確認による)。この「不在」が本節の主要な調査成果である。

参考(周辺ヒット): RL 一般への RQMC 適用は §2.4(Arnold et al. 2022、方策勾配の行動サンプリング)、分位点「推定」への RQMC/層化は §4 に存在するが、いずれも分布RLの τ には適用されていない。

## 4. 分位点推定における層化・重点サンプリングの統計学的知見

分位点(quantile)推定への分散低減は、金融リスク管理(VaR)と信頼性解析の分野で確立した文献群がある。

Glasserman(Columbia), Heidelberger(IBM Research), Shahabuddin(Columbia)は、損失分布の分位点である VaR の推定に対し、delta-gamma 近似で誘導した重点サンプリングと層化抽出の組合せが大きな分散低減をもたらすことを示した(Management Science 2000)。

```
The method employs a quadratic ("delta-gamma") approximation to the change in portfolio value to guide the selection of effective variance reduction techniques; specifically importance sampling and stratified sampling.

この方法は、ポートフォリオ価値変化の二次(「delta-gamma」)近似を用いて、有効な分散低減技法 — 具体的には重点サンプリングと層化抽出 — の選択を導く。
```
[glasserman-var (2000), Abstract]

Cannamela(CEA), Garnier(Université Paris VII), Iooss(CEA)は、縮約モデル(メタモデル)で誘導した「controlled stratification」による分位点推定の分散低減を分析した(Annals of Applied Statistics 2008)。被引用数: 約90(Semantic Scholar, 2026-08-12 確認)。

```
In this paper we propose and discuss variance reduction techniques for the estimation of quantiles of the output of a complex model with random input parameters. ... The different strategies are analyzed and the asymptotic variances are computed, which shows the benefit of an adaptive controlled stratification method.

本論文では、ランダムな入力パラメータを持つ複雑なモデルの出力の分位点推定に対する分散低減技法を提案・議論する。…各戦略を解析し漸近分散を計算した結果、適応的 controlled stratification 法の利点が示される。
```
[cannamela-strat (2008), Abstract]

RQMC を分位点推定そのものに使う研究もある。Kaplan, Li, Nakayama(NJIT), Tuffin(Inria Rennes)は、CDF 推定を介した分位点推定における 2 種の RQMC 推定量を比較し、片方について中心極限定理を確立した(Winter Simulation Conference 2019)。被引用数: 約5(Semantic Scholar, 2026-08-12 確認)。

```
We compare two approaches for quantile estimation via randomized quasi-Monte Carlo (RQMC) in an asymptotic setting where the number of randomizations for RQMC grows large but the size of the low-discrepancy point set remains fixed. ... In contrast, the second estimator does, and we establish a central limit theorem for it.

RQMC のランダム化回数が大きくなり low-discrepancy 点集合のサイズは固定という漸近設定で、ランダム化準モンテカルロ(RQMC)による分位点推定の 2 つのアプローチを比較する。…対照的に第 2 の推定量は(真の分位点に)収束し、我々はそれに対する中心極限定理を確立する。
```
[kaplan-rqmc-quantile (2019/12), Abstract]

なお、重点サンプリングによる分位点推定の先行研究として Glynn (1996) "Importance sampling for Monte Carlo estimation of quantiles" が上記文献群から引用されるが、原文は今回未入手(citation not confirmed)。

## 5. 調査の限界

- 被引用数はすべて Semantic Scholar API(2026-08-12 確認)による概数であり、Google Scholar とは大きく異なり得る。NeRF・McKay et al. (1979)・Glasserman et al. (2000)・Arnold et al. (2022) は API のレート制限により件数を取得できなかった(下記ソース一覧では「未確認」と記載。ただし NeRF・McKay 1979 はいずれも当該分野の代表的高被引用論文である、という定性的事実以上は本レポートでは主張しない)。
- McKay, Beckman, Conover (1979) と Stein (1987) の原文は paywall のため未入手であり、内容の引用は Owen 教科書による記述(帰属明記あり)を経由している。Sobol' (1967)、Halton (1960)、Hammersley and Morton (1956)、Glynn (1996) も同様に二次ソース経由(書誌のみ確認)。
- Caflisch (1998, Acta Numerica) はスキャン PDF のためテキスト抽出できず、引用を断念した(Koksma–Hlawka の典拠は Hickernell et al. 2025 で代替)。
- §3 の「見つからなかった」は英語 Web 検索(WebSearch)+ ヒットした論文フルテキストの確認に基づく。arXiv 全文検索・GitHub コード検索 API・中国語文献は未走査であり、悉皆調査ではない。
- kuang-qemrl(ICML 2023)の著者所属は未確認。zhang-rqiqn(arXiv 2026/05)は査読状況・所属未確認。
- kingma-vdm の引用は ar5iv 版、mildenhall-nerf の引用も ar5iv 版に基づく(出版版と字句が異なる可能性あり)。

## 6. ソース一覧

- [owen-mc-ch8, 2018] Art B. Owen. "Monte Carlo theory, methods and examples, Chapter 8: Variance reduction." オンライン教科書 (Stanford University). https://artowen.su.domains/mc/Ch-var-basic.pdf
- [owen-mc-ch10, 2013] Art B. Owen. "Monte Carlo theory, methods and examples, Chapter 10: Advanced variance reduction." オンライン教科書 (Stanford University). https://artowen.su.domains/mc/Ch-var-adv.pdf
- [pbrt-13.8, 2018] Matt Pharr, Wenzel Jakob, Greg Humphreys. "Physically Based Rendering, 3rd ed., §13.8 Careful Sample Placement." オンライン書籍. https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Careful_Sample_Placement
- [cook-tog, 1986/01] Robert L. Cook (Pixar). "Stochastic Sampling in Computer Graphics." ACM Transactions on Graphics 5(1):51–72. 被引用数約1,144 (Semantic Scholar, 2026-08-12). https://dl.acm.org/doi/10.1145/7529.8927
- [mckay-lhs, 1979] Michael D. McKay, Richard J. Beckman, William J. Conover. "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." Technometrics 21(2):239–245. 被引用数: 未確認(2026-08-12 時点で取得できず). https://www.tandfonline.com/doi/abs/10.1080/00401706.1979.10489755 (原文未入手・Owen 経由で引用)
- [stein-lhs, 1987] Michael Stein. "Large Sample Properties of Simulations Using Latin Hypercube Sampling." Technometrics 29(2):143–151. 被引用数約2,408 (Semantic Scholar, 2026-08-12). https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488205 (原文未入手・Owen 経由で引用)
- [hammersley-morton, 1956] J. M. Hammersley, K. W. Morton. "A new Monte Carlo technique: antithetic variates." Mathematical Proceedings of the Cambridge Philosophical Society 52:449–475. 被引用数約271 (Semantic Scholar, 2026-08-12). https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/new-monte-carlo-technique-antithetic-variates/69A9BBEDC6A4F1B1AF7E0764CD422E15 (原文未入手・Owen 経由で言及)
- [hickernell-qmc, 2025/02] Fred J. Hickernell, Nathan Kirk, Aleksei G. Sorokin (Illinois Institute of Technology). "Quasi-Monte Carlo Methods: What, Why, and How?" arXiv:2502.03644. https://arxiv.org/abs/2502.03644
- [buchholz-qmcvi, 2018/07] Alexander Buchholz (ENSAE-CREST), Florian Wenzel (TU Kaiserslautern), Stephan Mandt (Disney Research). "Quasi-Monte Carlo Variational Inference." ICML 2018, PMLR 80:668–677. 被引用数約70 (Semantic Scholar, 2026-08-12). https://arxiv.org/abs/1807.01604
- [kingma-vdm, 2021/07] Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho (Google Research). "Variational Diffusion Models." NeurIPS 2021. 被引用数約1,570 (Semantic Scholar, 2026-08-12). https://arxiv.org/abs/2107.00630
- [nichol-iddpm, 2021/02] Alex Nichol, Prafulla Dhariwal (OpenAI). "Improved Denoising Diffusion Probabilistic Models." ICML 2021. 被引用数約5,728 (Semantic Scholar, 2026-08-12). https://arxiv.org/abs/2102.09672
- [mildenhall-nerf, 2020/03] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng (UC Berkeley / Google Research / UC San Diego). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." ECCV 2020. 被引用数: 未確認(2026-08-12 時点で取得できず). https://arxiv.org/abs/2003.08934
- [arnold-rqmc-rl, 2022] Sébastien M. R. Arnold, Pierre L'Ecuyer, Liyu Chen, Yi-Fan Chen, Fei Sha. "Policy Learning and Evaluation with Randomized Quasi-Monte Carlo." AISTATS 2022, PMLR 151. 被引用数: 未確認(2026-08-12 時点で取得できず). https://proceedings.mlr.press/v151/arnold22a.html
- [dabney-iqn, 2018/06] Will Dabney, Georg Ostrovski, David Silver, Rémi Munos (DeepMind). "Implicit Quantile Networks for Distributional Reinforcement Learning." ICML 2018. 被引用数約722 (Semantic Scholar, 2026-08-12). https://arxiv.org/abs/1806.06923
- [yang-fqf, 2019/11] Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, Tie-Yan Liu. "Fully Parameterized Quantile Function for Distributional Reinforcement Learning." NeurIPS 2019. 被引用数約185 (Semantic Scholar, 2026-08-12). https://arxiv.org/abs/1911.02140
- [kuang-qemrl, 2023/07] Qi Kuang, Zhoufan Zhu, Liwen Zhang, Fan Zhou (所属未確認). "Variance Control for Distributional Reinforcement Learning." ICML 2023. 被引用数約4 (Semantic Scholar, 2026-08-12). https://arxiv.org/abs/2307.16152
- [zhang-rqiqn, 2026/05] Zhaofan Zhang, Minghao Yang, Rufeng Chen, Sihong Xie, Hui Xiong (所属未確認). "Quantile Geometry Regularization for Distributional Reinforcement Learning." arXiv:2605.08182 (査読状況未確認). https://arxiv.org/html/2605.08182v1
- [glasserman-var, 2000] Paul Glasserman (Columbia), Philip Heidelberger (IBM Research), Perwez Shahabuddin (Columbia). "Variance Reduction Techniques for Estimating Value-at-Risk." Management Science 46(10):1349–1364. 被引用数: 未確認(2026-08-12 時点で取得できず). https://pubsonline.informs.org/doi/10.1287/mnsc.46.10.1349.12274
- [cannamela-strat, 2008] Claire Cannamela (CEA), Josselin Garnier (Université Paris VII), Bertrand Iooss (CEA). "Controlled stratification for quantile estimation." Annals of Applied Statistics 2(4):1554–1580. 被引用数約90 (Semantic Scholar, 2026-08-12). https://arxiv.org/abs/0802.2426
- [kaplan-rqmc-quantile, 2019/12] Zachary T. Kaplan, Yajuan Li, Marvin K. Nakayama (NJIT), Bruno Tuffin (Inria Rennes). "Randomized Quasi-Monte Carlo for Quantile Estimation." Winter Simulation Conference 2019. 被引用数約5 (Semantic Scholar, 2026-08-12). https://web.njit.edu/~marvin/papers/wsc19-rqmc.pdf
- [sobol-1967, 1967] I. M. Sobol'. "On the distribution of points in a cube and the approximate evaluation of integrals." USSR Computational Mathematics and Mathematical Physics 7(4):86–112. (書誌のみ、原文未入手)
- [halton-1960, 1960] J. H. Halton. "On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals." Numerische Mathematik 2:84–90. (書誌のみ、原文未入手)
- [glynn-1996, 1996] Peter W. Glynn. "Importance sampling for Monte Carlo estimation of quantiles." Proc. 2nd International Workshop on Mathematical Methods in Stochastic Simulation and Experimental Design. (書誌のみ、原文未入手、citation not confirmed)
