# Survey: 分布強化学習における quantile fraction τ の選び方・配置方式の系譜(原典論文)

Date: 2026-08-12
Scope: quantile ベース分布強化学習の基礎手法の原典論文における τ(quantile fraction)の生成・配置方式。対象: QR-DQN / IQN / FQF / NC-QR-DQN / NDQFN / TQC / D4PG / DSAC(2系統)/ Munchausen-RL / EQR。各論文の書誌情報・被引用数(原則 Semantic Scholar、レート制限時は OpenAlex で補完。確認日 2026-08-12)・τ 方式の原文引用を収集した。

## Table of Contents

1. [系譜の概観](#1-系譜の概観)
2. [QR-DQN: 固定 quantile grid(midpoint)](#2-qr-dqn-固定-quantile-gridmidpoint)
3. [IQN: τ ∼ U[0,1] のランダムサンプリングと cosine embedding](#3-iqn-τ--u01-のランダムサンプリングと-cosine-embedding)
4. [FQF: fraction proposal network による τ の学習](#4-fqf-fraction-proposal-network-による-τ-の学習)
5. [Non-crossing 系: NC-QR-DQN と NDQFN](#5-non-crossing-系-nc-qr-dqn-と-ndqfn)
6. [TQC: 固定 τ + truncation(連続制御)](#6-tqc-固定-τ--truncation連続制御)
7. [連続制御系のその他: D4PG / DSAC](#7-連続制御系のその他-d4pg--dsac)
8. [τ の本数・配置が性能に与える影響の知見](#8-τ-の本数配置が性能に与える影響の知見)
9. [批判・限界(各ソースが自己申告・他ソースを批判したもの)](#9-批判限界各ソースが自己申告他ソースを批判したもの)
10. [調査の限界](#10-調査の限界)
11. [ソース一覧](#11-ソース一覧)

## 1. 系譜の概観

τ の生成方式は、原典論文ベースで次の 4 系統に整理できる(各主張の根拠は以降の各節の引用)。

| 方式 | 代表手法 | τ の決め方 | 学習対象か |
|---|---|---|---|
| 固定一様 grid(midpoint) | QR-DQN (2017)、NC-QR-DQN (2020)、TQC (2020) | τ̂_i = (τ_{i-1}+τ_i)/2、τ_i = i/N の中点。TQC は τ_m = (2m−1)/2M | 固定 |
| ランダムサンプリング | IQN (2018)、NDQFN (2021)、DSAC-Ma (2020)、M-IQN (2020)、EQR (2023) | τ ∼ U([0,1])(NDQFN は毎 iteration 再サンプル、DSAC-Ma は正規化してソート) | 固定分布からサンプル |
| 学習(proposal network) | FQF (2019) | softmax 出力の累積和で τ を生成し、W1 最小化で proposal network を学習 | 学習対象 |
| τ を使わない(quantile 系でない) | D4PG (2018)、DSAC-Duan (2020) | categorical 分布(C51 系)/ Gaussian パラメータ化のため quantile fraction が存在しない | — |

## 2. QR-DQN: 固定 quantile grid(midpoint)

- 著者: Will Dabney (DeepMind), Mark Rowland (University of Cambridge), Marc G. Bellemare (Google Brain), Rémi Munos (DeepMind) [QR-DQN (2017/10), 著者欄] [AAAI-Proceedings (2018/04)]
- 発表会場: AAAI-18 (Thirty-Second AAAI Conference on Artificial Intelligence, Vol. 32 No. 1, DOI: 10.1609/aaai.v32i1.11791) [AAAI-Proceedings (2018/04)]
- 被引用数: 1031(Semantic Scholar、2026-08-12 確認)[S2-QRDQN (2026/08)]
- URL: https://arxiv.org/abs/1710.10044

τ の生成方式: 分布を N 個の等重み Dirac(各重み 1/N)で表し、各出力 θ_i は固定 grid の中点 τ̂_i = (τ_{i-1}+τ_i)/2 に対応する quantile を推定する。τ は学習中に一切変化しない。

```
A quantile distribution Z_θ ∈ Z_Q maps each state-action pair (x, a) to a uniform probability distribution supported on {θ_i(x, a)}.

quantile 分布 Z_θ ∈ Z_Q は、各 state-action ペア (x, a) を {θ_i(x, a)} 上に台を持つ一様確率分布へ写像する。
```
[QR-DQN (2017/10), Section "Approximately Minimizing Wasserstein" 近傍]

```
Specifically, we take uniform weights, so that q_i = 1/N for each i = 1, …, N.

具体的には一様な重みを取り、各 i = 1, …, N に対して q_i = 1/N とする。
```
[QR-DQN (2017/10), Section "Approximately Minimizing Wasserstein"]

```
These quantile midpoints will be denoted by τ̂_i = (τ_{i-1} + τ_i)/2 for 1 ≤ i ≤ N.

これらの quantile 中点を、1 ≤ i ≤ N に対して τ̂_i = (τ_{i-1} + τ_i)/2 と表記する。
```
[QR-DQN (2017/10), Section "Approximately Minimizing Wasserstein"]

## 3. IQN: τ ∼ U[0,1] のランダムサンプリングと cosine embedding

- 著者: Will Dabney, Georg Ostrovski, David Silver, Rémi Munos(全員 DeepMind, London, UK)[IQN (2018/06), 著者欄]
- 発表会場: ICML 2018 (International Conference on Machine Learning) [S2-IQN (2026/08)]
- 被引用数: 722(Semantic Scholar、2026-08-12 確認)[S2-IQN (2026/08)]
- URL: https://arxiv.org/abs/1806.06923

τ の生成方式: online 側の τ と target 側の τ′ をそれぞれ一様分布 U([0,1]) から独立にサンプルし、その間の pairwise TD 誤差で quantile regression loss を構成する。

```
For two samples τ, τ′ ∼ U([0,1]), and policy π_β, the sampled temporal difference (TD) error at step t is

2 つのサンプル τ, τ′ ∼ U([0,1]) と方策 π_β に対して、ステップ t におけるサンプルされた TD 誤差は(次式で与えられる)
```
[IQN (2018/06), Section 3, Eq. (2) 直前]

τ の埋め込み: τ を n 次元 cosine 基底で埋め込み、状態埋め込みと結合(Hadamard 積)する。

```
φ_j(τ) := ReLU(Σ_{i=0}^{n−1} cos(π i τ) w_{ij} + b_j).

φ_j(τ) := ReLU(Σ_{i=0}^{n−1} cos(π i τ) w_{ij} + b_j)。(τ の cosine 埋め込みの定義)
```
[IQN (2018/06), Section 3.1, Eq. (4)]

N / N′(τ サンプル本数)の ablation: 詳細は[第 8 節](#8-τ-の本数配置が性能に与える影響の知見)に引用。方策側の τ サンプル数 K は 32 に固定された。

```
In an informal evaluation, we did not find IQN to be sensitive to K, the number of samples used for the policy, and have fixed it at K=32 for all experiments.

非公式な評価では、方策に使うサンプル数 K に対して IQN が敏感であるとは分からず、全実験で K=32 に固定した。
```
[IQN (2018/06), Section 3.1]

## 4. FQF: fraction proposal network による τ の学習

- 著者: Derek Yang (UC San Diego), Li Zhao (Microsoft Research), Zichuan Lin (Tsinghua University), Tao Qin (Microsoft Research), Jiang Bian (Microsoft Research), Tie-Yan Liu (Microsoft Research) [FQF (2019/11), 著者欄]
- 発表会場: NeurIPS 2019 (Neural Information Processing Systems) [S2-FQF (2026/08)]
- 被引用数: 185(Semantic Scholar、2026-08-12 確認)[S2-FQF (2026/08)]
- URL: https://arxiv.org/abs/1911.02140

τ の生成方式: fraction proposal network が状態ごとに softmax 出力 q_j を生成し、その累積和として単調増加する τ_i を得る(τ_0 = 0, τ_N = 1 が構成的に保証される)。

```
Let τ_i = Σ_{j=0}^{i−1} q_j, i ∈ [0, N], then straightforwardly we have τ_i < τ_j for ∀ i < j and τ_0 = 0, τ_N = 1 in our fraction proposal network.

τ_i = Σ_{j=0}^{i−1} q_j(i ∈ [0, N])とおくと、我々の fraction proposal network では直ちに ∀i<j で τ_i < τ_j、かつ τ_0 = 0, τ_N = 1 が成り立つ。
```
[FQF (2019/11), Section 3.4]

学習信号: proposal network は 1-Wasserstein 距離 W1 の最小化を目標とし、W1 自体は直接計算できないため ∂W1/∂τ_i の解析勾配を optimizer に渡す。真の quantile 関数の代わりに quantile value network を用いる。

```
we use the quantile value network F_{Z,w2}^{-1} with parameters w2 for current state and action as true quantile function.

現在の状態・行動に対する、パラメータ w2 を持つ quantile value network F_{Z,w2}^{-1} を、真の quantile 関数として用いる。
```
[FQF (2019/11), Section 3.2]

```
Instead, we use the grad_ys argument in the tensorflow operator tf.gradients to assign ∂W1/∂τ_i to the optimizer.

その代わりに、tensorflow の演算子 tf.gradients の grad_ys 引数を使って ∂W1/∂τ_i を optimizer に渡す。
```
[FQF (2019/11), Section 3.4]

安定化のための実装措置: proposal network の学習率は quantile value network より大幅に小さく設定される(sweep の結果 2.5e-9)。

```
The weights of the fraction proposal network are initialized so that initial probabilities are uniform as in QR-DQN, also the learning rates are relatively small compared with the quantile value network to keep the probabilities relatively stable while training.

fraction proposal network の重みは、初期確率が QR-DQN と同じく一様になるよう初期化され、また学習中に確率を比較的安定に保つため、学習率は quantile value network に比べて相対的に小さくされる。
```
[FQF (2019/11), Section 3.4]

```
we sweep the learning rate of fraction proposal network among (0, 2.5e-5) and finally fix this learning rate as 2.5e-9.

fraction proposal network の学習率を (0, 2.5e-5) の範囲で sweep し、最終的に 2.5e-9 に固定した。
```
[FQF (2019/11), Appendix(実験設定)]

自己申告された限界(収束保証なし・速度低下)は[第 9 節](#9-批判限界各ソースが自己申告他ソースを批判したもの)に引用。

## 5. Non-crossing 系: NC-QR-DQN と NDQFN

### 5.1 NC-QR-DQN (Zhou, Wang, Feng 2020)

- 著者: Fan Zhou, Jianing Wang, Xingdong Feng(School of Statistics and Management, Shanghai University of Finance and Economics)[NC-QR (2020/12), 著者欄]
- 発表会場: NeurIPS 2020(論文フッタに "34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada." と明記)[NC-QR (2020/12), p.1 脚注]
- 被引用数: 15(OpenAlex、2026-08-12 確認。Semantic Scholar はレート制限で未確認)[OpenAlex-NCQR (2026/08)]
- URL: https://proceedings.neurips.cc/paper/2020/file/b6f8dc086b2d60c5856e4ff517060392-Paper.pdf(arXiv 版は確認できず)

動機: 複数の quantile level を個別に回帰すると単調性の大域制約がなく、推定 quantile が交差(crossing)する。

```
One common problem of fitting quantile regressions at multiple percentiles is the non-monotonicity of the obtained quantile estimates. Much of this issue can be attributed to the fact that the quantile functions are estimated at different quantile levels separately without applying any global constraints to ensure monotonicity.

複数のパーセンタイルで quantile 回帰を当てはめる際の一般的な問題は、得られる quantile 推定値の非単調性である。この問題の多くは、単調性を保証する大域的制約を課さずに、異なる quantile level で quantile 関数を個別に推定していることに起因する。
```
[NC-QR (2020/12), Section 1]

τ の扱い: QR-DQN と同じく固定の非減少 quantile fraction 列を用い、各出力は中点 τ̂_i の quantile 推定である(τ 自体は変更しない)。

```
Let τ̃ = (τ_0, ..., τ_N) be a fixed sequence of N+1 non-decreasing quantile fractions, and Z_Θ be some pre-defined function space.

τ̃ = (τ_0, ..., τ_N) を N+1 個の非減少 quantile fraction の固定列とし、Z_Θ をある事前定義された関数空間とする。
```
[NC-QR (2020/12), Section 3.2]

単調性の実現機構: softmax logit の累積和 ψ_{i,a} と、非負スロープ α(s,a)・切片 β(s,a) を出す Scale Factor Network の線形変換で、quantile 値の非交差を構成的に保証する。

```
q_i(s,a) := α(s,a) × ψ_{i,a} + β(s,a), i = 0, ..., N−1; a = 1, ..., |A|.

q_i(s,a) := α(s,a) × ψ_{i,a} + β(s,a)(i = 0,...,N−1; a = 1,...,|A|)。(Eq. 19)
```
[NC-QR (2020/12), Section 3.3, Eq. (19)]

```
Since α(s,a) is non-negative and {ψ_{i,a}}'s are non-decreasing, the non-crossing property of the N q_i(s,a)'s is automatically satisfied.

α(s,a) が非負で {ψ_{i,a}} が非減少であるため、N 個の q_i(s,a) の非交差性は自動的に満たされる。
```
[NC-QR (2020/12), Section 3.3]

### 5.2 NDQFN (Zhou, Zhu, Kuang, Zhang 2021)

- 著者: Fan Zhou, Zhoufan Zhu, Qi Kuang, Liwen Zhang(Shanghai University of Finance and Economics)[NDQFN (2021/05), 著者欄]
- 発表会場: IJCAI 2021 (International Joint Conference on Artificial Intelligence) [S2-NDQFN (2026/08)]
- 被引用数: 23(Semantic Scholar、2026-08-12 確認)[S2-NDQFN (2026/08)]
- URL: https://arxiv.org/abs/2105.06696

動機: QR-DQN / IQN / FQF の理論的正当性は quantile 曲線の非減少性を前提とするが、大域制約がないため保証されない。

```
The theoretical validity of QR-DQN, IQN and FQF heavily depends on a prerequisite that the approximated quantile curve is non-decreasing. Unfortunately, since no global constraint is imposed when simultaneously estimating the quantile values at multiple locations, the monotonicity can not be ensured.

QR-DQN・IQN・FQF の理論的正当性は、近似された quantile 曲線が非減少であるという前提条件に強く依存している。不幸にも、複数の位置で quantile 値を同時推定する際に大域的制約が課されないため、単調性は保証できない。
```
[NDQFN (2021/05), Section 1/3]

構成: 固定の supporting points p 上で「基底値 + 非負増分の累積」として非減少 quantile 関数を構成する。p は端点をクリップした一様 grid。

```
By limiting the output range of g_a in (4) to be [0,∞), the obtained N+1 quantile estimates are non-decreasing.

(4) 式の g_a の出力範囲を [0,∞) に制限することで、得られる N+1 個の quantile 推定値は非減少になる。
```
[NDQFN (2021/05), Section 4]

```
we let p_i = i/N for i ∈ {1,⋯,N−1}, p_0 = 0.001 and p_N = 0.999

i ∈ {1,⋯,N−1} に対して p_i = i/N とし、p_0 = 0.001、p_N = 0.999 とする
```
[NDQFN (2021/05), Section 4(実装詳細)]

τ の扱い: 学習時の評価点 τ は IQN と同様に毎 iteration 一様分布から再サンプルする(supporting points p は固定のまま、任意 τ の値は補間で得る)。

```
Following the idea of IQN, two random sets of quantile fractions τ = {τ_1,⋯,τ_{N_1}}, τ′ = {τ′_1,⋯,τ′_{N_2}} are independently drawn from a uniform distribution U(0,1) at each training iteration.

IQN の考え方に従い、2 つのランダムな quantile fraction 集合 τ = {τ_1,⋯,τ_{N_1}} と τ′ = {τ′_1,⋯,τ′_{N_2}} を、各学習 iteration で一様分布 U(0,1) から独立に抽出する。
```
[NDQFN (2021/05), Section 4]

## 6. TQC: 固定 τ + truncation(連続制御)

- 著者: Arsenii Kuznetsov (Samsung AI Center Moscow), Pavel Shvechikov (Samsung AI Center Moscow; HSE), Alexander Grishin (Samsung AI Center Moscow; Samsung-HSE Lab), Dmitry Vetrov (Samsung AI Center Moscow; Samsung-HSE Lab) [TQC (2020/05), 著者欄]
- 発表会場: ICML 2020 (International Conference on Machine Learning) [S2-TQC (2026/08)]
- 被引用数: 285(Semantic Scholar、2026-08-12 確認)[S2-TQC (2026/08)]
- URL: https://arxiv.org/abs/2005.04269

τ の扱い: QR-DQN と同じ固定 midpoint grid を使う(論文の準備節で QR-DQN の説明として τ_m = (2m−1)/2M, m ∈ [1..M] が示され、TQC の各 critic はこの QR-DQN 型パラメータ化をそのまま用いる)。各 critic は M 個の等重み Dirac atom で分布を表す。

```
τ_m = (2m−1)/2M, m ∈ [1..M]

τ_m = (2m−1)/2M(m ∈ [1..M])。(quantile fraction の定義)
```
[TQC (2020/05), Section 2.3]

```
Each Z_ψn maps each (s,a) to a probability distribution Z_ψn(s,a) := (1/M) Σ_{m=1}^{M} δ(θ_ψn^m(s,a)), supported on atoms θ_ψn^1(s,a), …, θ_ψn^M(s,a).

各 Z_ψn は各 (s,a) を、atom θ_ψn^1(s,a), …, θ_ψn^M(s,a) 上に台を持つ確率分布 Z_ψn(s,a) := (1/M) Σ_{m=1}^{M} δ(θ_ψn^m(s,a)) に写像する。
```
[TQC (2020/05), Section 3.2]

truncation: N 個の critic の atom を全部プールしてソートし、大きい側を切り捨てて(小さい方 kN 個だけ残して)過大評価バイアスを制御する。

```
We pool atoms of distributions Z_ψ1(s′,a′), …, Z_ψN(s′,a′) into a set Z(s′,a′) := {θ_ψn^m(s′,a′) | n ∈ [1..N], m ∈ [1..M]} and denote elements sorted in ascending order by z_(i)(s′,a′).

分布 Z_ψ1(s′,a′), …, Z_ψN(s′,a′) の atom を集合 Z(s′,a′) := {θ_ψn^m(s′,a′) | n ∈ [1..N], m ∈ [1..M]} にプールし、昇順にソートした要素を z_(i)(s′,a′) と表記する。
```
[TQC (2020/05), Section 3.2]

```
The kN smallest elements of Z(s′,a′) define atoms y_i(s,a) := r(s,a) + γ[z_(i)(s′,a′) − α log π_φ(a′|s′)]

Z(s′,a′) の小さい方から kN 個の要素が、atom y_i(s,a) := r(s,a) + γ[z_(i)(s′,a′) − α log π_φ(a′|s′)] を定義する
```
[TQC (2020/05), Section 3.2, Eq. (11)]

## 7. 連続制御系のその他: D4PG / DSAC

### 7.1 D4PG (Barth-Maron et al. 2018) — quantile 系ではなく categorical

- 著者: Gabriel Barth-Maron, Matthew W. Hoffman, David Budden, Will Dabney, Dan Horgan, Dhruva TB, Alistair Muldal, Nicolas Heess, Timothy Lillicrap(全員 DeepMind, London, UK)[D4PG (2018/04), 著者欄]
- 発表会場: ICLR 2018 (Poster) [dblp-D4PG (2026/08)]
- 被引用数: OpenAlex では arXiv 版レコード 285 + 別レコード 78 に分割(2026-08-12 確認)。Semantic Scholar はレート制限で確認できず(citation not confirmed on Semantic Scholar)[OpenAlex-D4PG (2026/08)]
- URL: https://arxiv.org/abs/1804.08617

τ の扱い: D4PG の critic は quantile fraction を持たない。C51 系の categorical パラメータ化(固定 atom 上の離散分布)を採用しており、mixture of Gaussians も検討したが劣後した。

```
Following Bellemare et al. 2017, we first consider the categorical parameterization, a layer whose parameters are the logits ω_i of a discrete-valued distribution defined over a fixed set of atoms z_i.

Bellemare et al. 2017 に従い、まず categorical パラメータ化、すなわち固定 atom 集合 z_i 上に定義された離散値分布の logit ω_i をパラメータとする層を検討する。
```
[D4PG (2018/04), Appendix A]

```
While this is definitely a technique that is worth further exploration, we found in initial experiments that this choice of distribution under-performed the Categorical distribution by a fair margin.

これ(mixture of Gaussians)はさらなる探究に値する手法であることは確かだが、初期実験ではこの分布の選択は Categorical 分布にかなりの差で劣ることが分かった。
```
[D4PG (2018/04), Appendix C]

### 7.2 DSAC (Ma et al. 2020) — IQN 系のランダム τ を採用(fixed / random / net を比較)

- 著者: Xiaoteng Ma (Tsinghua Univ.), Junyao Chen, Li Xia (Sun Yat-Sen Univ.), Jun Yang (Tsinghua Univ.), Qianchuan Zhao (Tsinghua Univ.), Zhengyuan Zhou (NYU)(arXiv v3 時点の著者列)[DSAC-Ma (2020/04), arXiv abs ページ]
- 発表会場: arXiv コメント欄に "Accecpted by Journal of Artificial Intelligence Research"(JAIR 採録)と記載 [DSAC-Ma (2020/04), arXiv abs ページ Comments 欄]
- 被引用数: arXiv 版 30 + JAIR 版 24(OpenAlex はレコードが分割されている。2026-08-12 確認。Semantic Scholar はレート制限で未確認)[OpenAlex-DSAC-Ma (2026/08)]
- 掲載誌の補足: OpenAlex には Journal of Artificial Intelligence Research, Vol. 83 (2025) の雑誌版レコードが存在する [OpenAlex-DSAC-Ma (2026/08)]
- URL: https://arxiv.org/abs/2004.14547

τ の生成方式: fixed(QR-DQN 式 τ_i = i/N)・random(IQN 式サンプリング)・net(FQF 式 proposal network)の 3 方式を比較し、random を採用した。

```
With the basic idea of QR-DQN, quantile fractions are given by a group of fix values as τ_i = i/N, i = 0, …, N

QR-DQN の基本的な考え方では、quantile fraction は固定値の組 τ_i = i/N(i = 0, …, N)で与えられる
```
[DSAC-Ma (2020/04), Appendix B.1]

```
As we fix the number of quantile fractions and keep them in ascending order, we adapt the sampling as τ_0 = 0, τ_i = ε_i / Σ_{i=0}^{N−1} ε_i where ε_i ∼ U[0,1], i = 1, …, N

quantile fraction の本数を固定し昇順に保つため、サンプリングを τ_0 = 0、τ_i = ε_i / Σ_{i=0}^{N−1} ε_i(ε_i ∼ U[0,1]、i = 1, …, N)のように適応させる
```
[DSAC-Ma (2020/04), Appendix B.1]

```
random sampling (Dabney et al. 2018b) has better performance and fewer parameters.

ランダムサンプリング(Dabney et al. 2018b)の方が性能が良く、パラメータも少ない。
```
[DSAC-Ma (2020/04), Section 5.1]

τ の埋め込みは IQN と同じ cosine embedding を用いる。

```
φ_j(τ) := f(Σ_{i=1}^{n} cos(iπτ) w_{ij} + b_j)

φ_j(τ) := f(Σ_{i=1}^{n} cos(iπτ) w_{ij} + b_j)(cosine 埋め込み)
```
[DSAC-Ma (2020/04), Appendix B.3, Eq. (28)]

### 7.3 DSAC (Duan et al. 2020) — quantile を使わず Gaussian パラメータ化

- 著者: Jingliang Duan, Yang Guan, Shengbo Eben Li, Yangang Ren, Qi Sun, Bo Cheng(Tsinghua University, State Key Lab of Automotive Safety and Energy ほか)[DSAC-Duan (2020/01), 著者欄]
- 発表会場: IEEE Transactions on Neural Networks and Learning Systems, 33(11): 6584-6598 (2022) [dblp-DSAC-Duan (2026/08)]
- 被引用数: 326(Semantic Scholar、2026-08-12 確認)[S2-DSAC-Duan (2026/08)]
- URL: https://arxiv.org/abs/2001.02811

τ の扱い: この DSAC は quantile fraction を用いない。リターン分布を平均・分散を NN が出力する Gaussian としてモデル化する。

```
both the state-action return distribution and policy functions are modeled as Gaussian with mean and covariance given by neural networks (NNs).

state-action リターン分布と方策関数の両方を、平均と共分散をニューラルネットワーク(NN)が与える Gaussian としてモデル化する。
```
[DSAC-Duan (2020/01), Section V-A]

```
Different from existing distributional RL algorithms that learn a discrete return distribution..., the proposed DSAC is capable of learning a continuous return distribution.

離散的なリターン分布を学習する既存の分布強化学習アルゴリズムと異なり(中略)、提案する DSAC は連続的なリターン分布を学習できる。
```
[DSAC-Duan (2020/01), Section II/III(中略は原文の省略箇所)]

## 8. τ の本数・配置が性能に与える影響の知見

### 8.1 IQN の N / N′ ablation

IQN は τ サンプル本数 N(online 側)と N′(target 側)を 1〜64 で振る ablation を行い、N は初期性能に劇的な影響を持ち、N′ は N′=8 以降は長期性能への影響がほぼないと報告した。

```
As expected, we found that N has a dramatic effect on early performance, shown by the continual improvement in score as the value increases.

予想通り、N は初期性能に劇的な効果を持つことが分かった。これは値を増やすにつれてスコアが継続的に改善することに示されている。
```
[IQN (2018/06), N/N′ ablation の段落]

```
N′ affected performance very differently than expected: it had a strong effect on early performance, but minimal impact on long-term performance past N′=8.

N′ は予想とは大きく異なる影響を与えた。初期性能には強い効果を持ったが、N′=8 を超えると長期性能への影響は最小限だった。
```
[IQN (2018/06), N/N′ ablation の段落]

```
Overall, while using more samples for both distributions is generally favorable, N = N′ = 8 appears to be sufficient to achieve the majority of improvements offered by IQN for long-term performance, with variation past this point largely insignificant.

全体として、両分布でより多くのサンプルを使うことは概ね好ましいが、長期性能については N = N′ = 8 で IQN のもたらす改善の大部分を達成するのに十分であるように見え、この点を超えた変動はほぼ有意でない。
```
[IQN (2018/06), N/N′ ablation の段落]

なお、N=1 で DQN に近づくという仮説(分布 RL の効果は補助損失によるという説の検証)も同じ ablation の動機として明記されている。

```
We hypothesized that N, the number of samples of τ ∼ U([0,1]), would affect the sample complexity of IQN, with larger values leading to faster learning, and that with N=1 one would potentially approach the performance of DQN.

我々は、τ ∼ U([0,1]) のサンプル数 N が IQN のサンプル複雑性に影響し、大きい値ほど学習が速くなり、N=1 では DQN の性能に近づく可能性があると仮説を立てた。
```
[IQN (2018/06), N/N′ ablation の段落]

### 8.2 FQF: τ 選択の重要性を open question として提示

FQF は考察で「学習中の quantile fraction の選択はどの程度重要か」を未解決問題として挙げている。

```
More generally, how important is quantile fraction selection during training?

より一般に、学習中の quantile fraction の選択はどの程度重要なのか?
```
[FQF (2019/11), Discussion/Conclusion]

### 8.3 DSAC-Ma: fixed / random / net の直接比較

第 7.2 節に引用の通り、fixed(QR-DQN 式)・random(IQN 式)・net(FQF 式)を比較し「random sampling has better performance and fewer parameters」と結論した [DSAC-Ma (2020/04), Section 5.1]。

### 8.4 Munchausen-RL (M-IQN): τ 方式には触れず IQN をそのまま利用

- 著者: Nino Vieillard, Olivier Pietquin, Matthieu Geist [S2-MRL (2026/08)]
- 発表会場: NeurIPS 2020 [S2-MRL (2026/08)]
- 被引用数: 118(Semantic Scholar、2026-08-12 確認)[S2-MRL (2026/08)]
- URL: https://arxiv.org/abs/2007.14430

M-IQN は IQN の TD ターゲットにスケール付き log-policy 項を加えるのみで、τ のサンプリング方式自体への変更や知見は本文に見当たらなかった(調査範囲内)。

```
our core contribution stands in a very simple idea: optimizing for the immediate reward augmented by the scaled log-policy of the agent when using any TD scheme

我々の中核的貢献は非常に単純なアイデアにある。任意の TD 方式を使う際に、エージェントのスケール付き log-policy で増強した即時報酬を最適化することである
```
[M-RL (2020/07), Section 2]

```
We observe that M-IQN outperforms Rainbow, both in terms of mean and median scores, and thus defines the new state of the art.

M-IQN は平均・中央値スコアの両方で Rainbow を上回り、新たな state of the art を定めることが観察される。
```
[M-RL (2020/07), Section 4]

### 8.5 EQR(Dual Expectile-Quantile Regression): Huber 化による分布崩壊の指摘と一様サンプリングの踏襲

- 著者: Sami Jullien, Romain Deffayet, Jean-Michel Renders, Paul Groth, Maarten de Rijke [EQR (2023/05), arXiv abs ページ]
- 発表会場: UAI 2025 (Conference on Uncertainty in Artificial Intelligence) [EQR (2023/05), arXiv abs ページ] [S2-EQR (2026/08)]
- 被引用数: 0(Semantic Scholar、2026-08-12 確認)[S2-EQR (2026/08)]
- URL: https://arxiv.org/abs/2305.16877

主張: Huber 損失による quantile 回帰では分布推定の保証が消え、推定分布が平均に崩壊する。fraction 自体は IQN 同様に一様サンプルし、expectile fraction を quantile fraction へ変換する mapper を学習する。推定 fraction 数無限大の極限で分布 Bellman 作用素へ収束すると主張する。

```
by doing so, distributional guarantees vanish, and we empirically observe that the estimated distribution rapidly collapses to its mean.

そうする(Huber 損失を使う)ことで分布的保証は消失し、推定された分布が急速にその平均へ崩壊することを我々は経験的に観察する。
```
[EQR (2023/05), Section 1]

```
fractions (τ_i)_{i=1,…,N} ~ U([0,1])

fraction (τ_i)_{i=1,…,N} は U([0,1]) からサンプルする
```
[EQR (2023/05), Section 4.3 / Algorithm 1]

## 9. 批判・限界(各ソースが自己申告・他ソースを批判したもの)

### 9.1 FQF の自己申告: 収束保証なし・20% の速度低下

```
We cannot guarantee convergence of the fraction proposal network in deep neural networks where we involve quantile regression and Bellman update.

quantile 回帰と Bellman 更新を伴う深層ニューラルネットワークにおいて、fraction proposal network の収束は保証できない。
```
[FQF (2019/11), 本文(理論節)]

```
However, one side effect of the full parameterization in FQF is that the training speed is decreased. With same settings, FQF is roughly 20% slower than IQN due to the additional fraction proposal network.

しかし、FQF における完全パラメータ化の副作用の一つは学習速度の低下である。同一設定で、FQF は追加の fraction proposal network のために IQN よりおよそ 20% 遅い。
```
[FQF (2019/11), Discussion/Conclusion]

### 9.2 NDQFN による IQN/FQF 批判: τ 再サンプルが分布ベース探索ボーナスを不安定化

```
the original DLTV method requires all the quantile locations to be fixed while IQN or FQF resample the quantile locations at each training iteration and the bonus term could be extremely unstable

元の DLTV 法は全ての quantile 位置が固定されていることを要求するが、IQN や FQF は各学習 iteration で quantile 位置を再サンプルするため、ボーナス項は極めて不安定になり得る
```
[NDQFN (2021/05), Section 1]

```
how to ensure the monotonicity of the approximated quantile function in DRL still remains challenging, especially to some quantile value based algorithms such as IQN and FQF, which do not focus on fixed quantile locations during the training process

DRL において近似 quantile 関数の単調性をどう保証するかは依然として難しい。特に IQN や FQF のように、学習過程で固定された quantile 位置に着目しないタイプの quantile value ベースのアルゴリズムでは顕著である
```
[NDQFN (2021/05), Section 3]

### 9.3 NC-QR-DQN による固定 grid 系(QR-DQN)批判: crossing による行動選択の不安定化

```
Without the non-crossing guarantee, the direction of policy searching may be distorted and the selection of optimal actions greatly varies across training epochs.

非交差の保証がなければ、方策探索の方向は歪められ得るし、最適行動の選択は学習エポック間で大きく変動する。
```
[NC-QR (2020/12), Section 1]

### 9.4 EQR による Huber quantile 損失批判

第 8.5 節に引用(「distributional guarantees vanish ... collapses to its mean」)。

## 10. 調査の限界

- 被引用数は原則 Semantic Scholar API(2026-08-12)だが、レート制限で取得できなかった D4PG・NC-QR-DQN・DSAC-Ma は OpenAlex(同日)で補完した。OpenAlex は arXiv 版と出版版のレコードが分割されるため実際の総被引用数より小さく出る傾向がある(特に D4PG)。
- 原文引用は ar5iv / arXiv HTML 変換テキストからの抽出であり、数式表記は Unicode/ASCII に転記した(語句は原文通り、数式の添字表記のみ転記)。NC-QR-DQN のみ NeurIPS 公式 PDF から直接抽出しており転記精度が最も高い。
- IQN の N/N′ ablation の節番号は ar5iv 版の変換で "Section 3.1" と表示されたが、公式版 PDF の節番号と一致しない可能性があるため、位置表記は「N/N′ ablation の段落」とした。
- FQF の「学習された fraction が最終的にほぼ固定/一様になる」という後続の再現報告は、本調査では一次ソースを確認できなかった(citation not confirmed)。FQF 自身が書いているのは第 9.1 節の引用(収束保証なし・速度低下)まで。
- D4PG の発表会場は OpenReview がブラウザ認証で直接取得できなかったため、dblp レコード(ICLR (Poster) 2018)で代替確認した。
- DSAC-Duan の掲載誌は dblp レコード(IEEE TNNLS 33(11), 2022)で確認した。IEEE 公式ページでの直接確認は行っていない。
- EQR の被引用数 0 は Semantic Scholar のレコード(year 2023 表記)に基づく。Google Scholar では異なる可能性がある。

## 11. ソース一覧

- [QR-DQN, 2017/10] Will Dabney, Mark Rowland, Marc G. Bellemare, Rémi Munos. "Distributional Reinforcement Learning with Quantile Regression." AAAI 2018. https://arxiv.org/abs/1710.10044
- [AAAI-Proceedings, 2018/04] AAAI Conference Proceedings. "Distributional Reinforcement Learning With Quantile Regression." AAAI-18, Vol. 32 No. 1. https://ojs.aaai.org/index.php/AAAI/article/view/11791
- [IQN, 2018/06] Will Dabney, Georg Ostrovski, David Silver, Rémi Munos. "Implicit Quantile Networks for Distributional Reinforcement Learning." ICML 2018. https://arxiv.org/abs/1806.06923
- [FQF, 2019/11] Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, Tie-Yan Liu. "Fully Parameterized Quantile Function for Distributional Reinforcement Learning." NeurIPS 2019. https://arxiv.org/abs/1911.02140
- [NC-QR, 2020/12] Fan Zhou, Jianing Wang, Xingdong Feng. "Non-crossing quantile regression for deep reinforcement learning." NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/file/b6f8dc086b2d60c5856e4ff517060392-Paper.pdf
- [NDQFN, 2021/05] Fan Zhou, Zhoufan Zhu, Qi Kuang, Liwen Zhang. "Non-decreasing Quantile Function Network with Efficient Exploration for Distributional Reinforcement Learning." IJCAI 2021. https://arxiv.org/abs/2105.06696
- [TQC, 2020/05] Arsenii Kuznetsov, Pavel Shvechikov, Alexander Grishin, Dmitry Vetrov. "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics." ICML 2020. https://arxiv.org/abs/2005.04269
- [D4PG, 2018/04] Gabriel Barth-Maron, Matthew W. Hoffman, David Budden, Will Dabney, Dan Horgan, Dhruva TB, Alistair Muldal, Nicolas Heess, Timothy Lillicrap. "Distributed Distributional Deterministic Policy Gradients." ICLR 2018 (Poster). https://arxiv.org/abs/1804.08617
- [DSAC-Ma, 2020/04] Xiaoteng Ma, Junyao Chen, Li Xia, Jun Yang, Qianchuan Zhao, Zhengyuan Zhou. "DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning." arXiv(JAIR 採録とコメント欄に記載). https://arxiv.org/abs/2004.14547
- [DSAC-Duan, 2020/01] Jingliang Duan, Yang Guan, Shengbo Eben Li, Yangang Ren, Qi Sun, Bo Cheng. "Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors." IEEE Trans. Neural Networks Learn. Syst. 33(11), 2022. https://arxiv.org/abs/2001.02811
- [M-RL, 2020/07] Nino Vieillard, Olivier Pietquin, Matthieu Geist. "Munchausen Reinforcement Learning." NeurIPS 2020. https://arxiv.org/abs/2007.14430
- [EQR, 2023/05] Sami Jullien, Romain Deffayet, Jean-Michel Renders, Paul Groth, Maarten de Rijke. "Distributional Reinforcement Learning with Dual Expectile-Quantile Regression." UAI 2025. https://arxiv.org/abs/2305.16877
- [S2-QRDQN, 2026/08] Semantic Scholar API record for arXiv:1710.10044. https://api.semanticscholar.org/graph/v1/paper/arXiv:1710.10044
- [S2-IQN, 2026/08] Semantic Scholar API record for arXiv:1806.06923. https://api.semanticscholar.org/graph/v1/paper/arXiv:1806.06923
- [S2-FQF, 2026/08] Semantic Scholar API record for arXiv:1911.02140. https://api.semanticscholar.org/graph/v1/paper/arXiv:1911.02140
- [S2-TQC, 2026/08] Semantic Scholar API record for arXiv:2005.04269. https://api.semanticscholar.org/graph/v1/paper/arXiv:2005.04269
- [S2-NDQFN, 2026/08] Semantic Scholar API record for arXiv:2105.06696. https://api.semanticscholar.org/graph/v1/paper/arXiv:2105.06696
- [S2-MRL, 2026/08] Semantic Scholar API record for arXiv:2007.14430. https://api.semanticscholar.org/graph/v1/paper/arXiv:2007.14430
- [S2-EQR, 2026/08] Semantic Scholar API record for arXiv:2305.16877. https://api.semanticscholar.org/graph/v1/paper/arXiv:2305.16877
- [S2-DSAC-Duan, 2026/08] Semantic Scholar API record for arXiv:2001.02811. https://api.semanticscholar.org/graph/v1/paper/arXiv:2001.02811
- [OpenAlex-NCQR, 2026/08] OpenAlex work record. "Non-Crossing Quantile Regression for Distributional Reinforcement Learning" (NeurIPS 2020, cited_by_count=15). https://api.openalex.org/works?search=Non-crossing%20quantile%20regression%20reinforcement
- [OpenAlex-DSAC-Ma, 2026/08] OpenAlex work records (arXiv 版と JAIR Vol.83 版). https://api.openalex.org/works?search=DSAC%20Distributional%20Soft%20Actor%20Critic%20Risk-Sensitive
- [OpenAlex-D4PG, 2026/08] OpenAlex work records. "Distributed Distributional Deterministic Policy Gradients." https://api.openalex.org/works?search=Distributed%20Distributional%20Deterministic%20Policy%20Gradients
- [dblp-D4PG, 2026/08] dblp record. "Distributed Distributional Deterministic Policy Gradients." ICLR (Poster) 2018. https://dblp.org/search?q=Distributed+Distributional+Deterministic+Policy+Gradients
- [dblp-DSAC-Duan, 2026/08] dblp record. "Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors." IEEE Trans. Neural Networks Learn. Syst. 33(11), 2022. https://dblp.org/search?q=Distributional+Soft+Actor-Critic
