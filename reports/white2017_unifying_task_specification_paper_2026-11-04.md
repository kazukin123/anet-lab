# Unifying Task Specification in Reinforcement Learning — 詳細解説

- 著者: Martha White [p.1]
- 所属: Department of Computer Science, Indiana University [p.1, footnote 1]
- 発表: ICML 2017（第 34 回、シドニー）、Proceedings of Machine Learning Research vol. 70 [p.1]。頁は 3742–3750（PMLR の書誌ページによる）
- 年: 2017
- arXiv: 1609.01995。v1 は 2016 年 9 月 7 日、v4 は 2021 年 9 月 17 日（arXiv の書誌ページによる）。本解説の底本は v4 の PDF（16 ページ）である [p.1]
- URL: https://arxiv.org/abs/1609.01995 （PMLR 版: https://proceedings.mlr.press/v70/white17a.html ）

Date: 2026-11-04

表記の約束:

- 著作権に配慮し、論文の本文を再現せず、自分の言葉で要約する。逐語の引用はアブストラクトの節の 1 か所（15 語未満）だけとする。図表は転載せず番号とページで参照し、議論に要る数値だけを位置引用つきで挙げる（表全体は写さない）。原文と図表は papers/white2017_unifying_task_specification.pdf を参照されたい。
- 対象論文への参照は [p.ページ, 節・定理・図など] の形で位置だけを示す（ページは arXiv v4 の PDF の通しページ）。他の文献は [著者 (年)] と書き、末尾の出典リストに対応させる。
- 数式は LaTeX で書き、記号は使う前に定義する。(n) は論文の式番号を指す。式は本解説の記法で書き直しており、途中式はすべては載せていない。
- 章ごとの監査は未実施である。paper-details の章ごとの監査役は費用のため立てず、書き手が数値・式・位置引用を PDF の画像とテキスト抽出で読み合わせた。その後、全体を通した第三者照合を 1 回受け、指摘を PDF と計算で確かめて反映した。
- 本解説が独自に行った計算や照合は「本解説」と明記して論文の主張と区別する。原典の記載の食い違いは「論文内の記載の食い違い」にまとめ、本文からは「項目 N」で参照する。
- PMLR 版の本文との照合はしていない。書誌情報だけを公式の掲載ページで確認した。

アブストラクトの要約:

本論文の中心は、次の一節で説明される RL タスク形式である。

```
a unification through simple constructs including a generalization to transition-based discounting

遷移ベースの割引への一般化を含む、単純な構成要素による統一
```
[p.1, Abstract]

狙いは、環境のダイナミクスと学習の目的が結び付いているために、エピソード型と継続型の関係などが見えにくくなる問題を解くことにある。論文は例でこの形式の有用性を示し、Bellman 作用素と近似誤差の上界をこの形式へ拡張する [p.1, Abstract]。

## Table of Contents

- [1. はじめに（Introduction）](#1-はじめにintroduction)
- [2. 問題の一般的な定式化（Generalized problem formulation）](#2-問題の一般的な定式化generalized-problem-formulation)
  - [2.1 エピソード型と継続型の統一（Unifying episodic and continuing specification）](#21-エピソード型と継続型の統一unifying-episodic-and-continuing-specification)
  - [2.2 RL タスクとしての option（Options as RL tasks）](#22-rl-タスクとしての-optionoptions-as-rl-tasks)
  - [2.3 一般価値関数（General value functions）](#23-一般価値関数general-value-functions)
- [3. タクシー領域での実演（Demonstration in the taxi domain）](#3-タクシー領域での実演demonstration-in-the-taxi-domain)
- [4. 目的とアルゴリズム（Objectives and algorithms）](#4-目的とアルゴリズムobjectives-and-algorithms)
- [5. 一般化した理論的性質（Generalized theoretical properties）](#5-一般化した理論的性質generalized-theoretical-properties)
  - [5.1 近似誤差の上界（Approximation bound）](#51-近似誤差の上界approximation-bound)
  - [5.2 TD アルゴリズムの性質（Properties of TD algorithms）](#52-td-アルゴリズムの性質properties-of-td-algorithms)
- [6. 考察と結論（Discussion and conclusion）](#6-考察と結論discussion-and-conclusion)
- [A. 確率的な割引を含む定式化（More general formulation with probabilistic discounts）](#a-確率的な割引を含む定式化more-general-formulation-with-probabilistic-discounts)
- [B. 状態ベースと遷移ベースの割引の関係（Relationship between state-based and transition-based discounting）](#b-状態ベースと遷移ベースの割引の関係relationship-between-state-based-and-transition-based-discounting)
  - [B.1 等価性の結果（Equivalence result）](#b1-等価性の結果equivalence-result)
  - [B.2 遷移ベースの割引の利点（Advantages of transition-based discounting over state-based discounting）](#b2-遷移ベースの割引の利点advantages-of-transition-based-discounting-over-state-based-discounting)
  - [B.3 Theorem 2 の証明（Proof of Theorem 2）](#b3-theorem-2-の証明proof-of-theorem-2)
- [C. 割引と平均報酬（Discounting and average reward for control）](#c-割引と平均報酬discounting-and-average-reward-for-control)
- [D. アルゴリズム（Algorithms）](#d-アルゴリズムalgorithms)
- [E. 補題（Lemmas）](#e-補題lemmas)
- [F. emphatic アルゴリズムの収束（Convergence of emphatic algorithms for the RL task formalism）](#f-emphatic-アルゴリズムの収束convergence-of-emphatic-algorithms-for-the-rl-task-formalism)
- [G. emphatic 重みなしの遷移ベース trace の問題（Issues with transition-based trace without emphatic weighting）](#g-emphatic-重みなしの遷移ベース-trace-の問題issues-with-transition-based-trace-without-emphatic-weighting)
- [論文が述べている限界](#論文が述べている限界)
- [論文内の記載の食い違い](#論文内の記載の食い違い)
- [キューの論点への答え](#キューの論点への答え)
- [SMDP の PRD（956）の検討に効く点](#smdp-の-prd956の検討に効く点)
- [出典リスト](#出典リスト)

## 1. はじめに（Introduction）

RL のタスクは普通、遷移モデル・報酬モデル・割引を含む MDP として書かれ、終端に着くと開始状態へ戻るエピソード型と、定数 $\gamma<1$ で総報酬を有限に保つ継続型に分けられてきた [p.1, Section 1]。その後、より広い目的を書くために option [Sutton et al. (1999)]、状態ベースの割引 [Sutton (1995)]、[Sutton et al. (2011)]、関心度の関数 [Reza & Sutton (2010)]、[Sutton et al. (2016)] が足された。主な動機はオフポリシー学習と、Horde の一般価値関数 [Sutton et al. (2011)]、[White A. (2015)] である [p.1, Section 1]。

著者が問題にするのは、これらが別々の用語と定式化を持つため関係が見えにくいことである。option と一般価値関数の重なりや、エピソード型と継続型で収束証明 [Bertsekas & Tsitsiklis (1996)]、[Tsitsiklis & Van Roy (1997)]、[Sutton et al. (2009)] とアルゴリズムの書き方が分かれていることを例に挙げる [p.1, Section 1]。

本論文は、環境のダイナミクスの指定と、その中での目的の指定を切り離す形式を提案する。貢献は 2 つある [p.1, Section 1]。

1. option、一般価値関数、エピソード型・継続型を、状態や MDP に手を入れず割引関数の変更だけで書けることを示す。
2. 一般化したタスクの Bellman 作用素に新しい縮小の上界を証明し、エピソード型・継続型の既存の上界がその特殊例になることを示す。

## 2. 問題の一般的な定式化（Generalized problem formulation）

記号:

| 記号 | 意味 | 注記 |
|---|---|---|
| $\mathcal S$, $\mathcal A$, $n$ | 状態集合、行動集合、状態数 | 理論の節では有限（A1） |
| $\Pr(s,a,s')$ | 遷移確率 | MDP $(\mathcal S,\mathcal A,\Pr)$ の一部。タスクには含めない |
| $\pi(s,a)$, $\mathcal P$ | 方策とその集合 | |
| $r(s,a,s')$ | 報酬関数 | タスクごとに変えられる |
| $\gamma(s,a,s')$ | 遷移ベースの割引関数 | 値域は $[0,1]$。1 も 0 も含む |
| $\mathrm i(s)$ | 関心度 | 値域は $[0,\infty)$。その状態をどれだけ重視するか |
| $\gamma_{t+1}$ | $\gamma(s_t,a_t,s_{t+1})$ の略記 | 付録 D と同じ書き方 |
| $R_{t+1}$, $G_t$ | 報酬とリターン | |
| $\gamma_c$ | 定数の割引 | $[0,1)$ |

RL タスクは、MDP の上に載せる組 $(\mathcal P,r,\gamma,\mathrm i)$ である [p.2, Section 2]。同じ環境の上に、報酬の違う別のタスク（台所へ行く、会議室へ行く、など）を複数置ける。予測でも制御でも、ある方策のリターンを評価するのが中心になる [p.2, Section 2]。リターンは遷移ごとの割引の積で定義される。

$$G_t=\sum_{k=0}^{\infty}\Bigl(\prod_{j=1}^{k}\gamma_{t+j}\Bigr)R_{t+k+1}$$

空の積は 1 とする [p.2, Section 2]。したがって遷移の報酬には、その遷移自身の割引は掛からず、先行する遷移の割引の積だけが掛かる。すべての遷移で $\gamma=\gamma_c$ なら積は $\gamma_c^k$ になり、定数割引の設定に戻る。エピソードの終わりは $\gamma=0$ で表され、積がそこで 0 になって再帰が切れる [p.2, Section 2]。報酬と割引をさらに確率的にする一般化は付録 A に送られている [p.2, footnote 1]。

### 2.1 エピソード型と継続型の統一（Unifying episodic and continuing specification）

吸収状態を使う従来の定式化 [Sutton & Barto (1998)] や状態ベースの割引 [Sutton (1995)]、[Reza & Sutton (2010)]、[Sutton et al. (2011)]、[van Hasselt (2011), Section 2.1.1] は、状態集合や MDP を特別に扱う必要があり、タスク指定とダイナミクスを結び付けてしまう、と著者は論じる [p.2, Section 2.1]。

説明には 3 状態のチェーンを使う。開始は $s_1$、行動は right と left、報酬は毎ステップ $-1$、$s_3$ で right を取ると終わる。割引は特に断らなければ 1、関心度はすべて 1 である [p.2, Section 2.1]。Figure 1 [p.3, Figure 1] は、これを継続問題として書く 3 通り (a)〜(c) と、誤った表現 (d) を示す。

- (a) 吸収状態: 四角の状態に入ると報酬 0 でとどまる。リターンは定義できるが、次のエピソードへ移るには実際にはテレポートが要る [p.2, Section 2.1]。
- (b) 遷移ベース: $\gamma(s_3,\text{right},s_1)=0$ と置く。状態を足さず、普通に $s_1$ へ遷移するだけで同じリターンになる [p.2, Section 2.1]。
- (c) 仮想状態を足す状態ベース: $s_3$ から仮想状態 $s_4$ を経て $s_1$ へ移し、$\gamma_s(s_4)=0$ と置く。正しい更新になるが、定常分布が $s_4$ にも重みを持つので、$s_4$ の特徴を $s_1$ と揃えるか 0 にしないと関数近似の資源を消費して他の状態の近似を変えかねない [p.3, Section 2.1]。
- (d) 状態を足さない状態ベース: 正しくない。$\gamma_s(s_1)=0$ では、$s_1$ へ入る遷移のうち終わるもの（$s_3$ からの right）と終わらないもの（$s_2$ からの left）を次状態だけでは区別できない。$\gamma_s(s_3)=0$ では、切れる位置が $s_2\to s_3$ の入りにずれ、$s_3$ からの right で切れなくなる [p.3, Figure 1]（理由の言い方は本解説の整理）。

線形の TD で (b) を確かめる。特徴を $x:\mathcal S\to\mathbb R^d$、重みを $w$、trace を $e$、trace のパラメータを $\lambda$ とすると、$s_3$ から right を取った遷移では $\gamma_{t+1}=0$ なので、TD 誤差は $r_{t+1}-x(s_3)^\top w$ となり、次の trace は $x(s_1)$ だけになって過去が自動で消える [p.2, Section 2.1]。定常分布も元のエピソード問題と同じである [p.3, Section 2.1]。

タスク指定と MDP を分けたまま統一するには遷移ベースの割引が必要だ、というのが著者のまとめである [p.3, Section 2.1]。

### 2.2 RL タスクとしての option（Options as RL tasks）

Proposition 1: 方策、終了関数 $\beta:\mathcal S\to[0,1]$、開始集合 $\mathcal I$ の組である option [Sutton et al. (1999), Section 2] は、RL タスクとして等価に書ける [p.3, Proposition 1]。割引を $\gamma(s,a,s')=1-\beta(s')$、関心度を $\mathcal I$ の中で 1・外で 0 と置き、報酬は MDP のものを使う。

RL タスクは、終了を遷移ベースの割引へ、2 値の関心度を程度へ広げる点で option を一般化している。RL の部分タスクに対応する方策は macro-action として使え、semi-Markov 決定過程を定める [Sutton et al. (1999), Theorem 1]、[p.3, Section 2.2]。

### 2.3 一般価値関数（General value functions）

一般価値関数は世界についての予測的な問いを表すもので [Sutton et al. (2011)]、[Modayil et al. (2014)]、[White A. (2015)]、これも RL タスクとして書ける [p.3–4, Section 2.3]。ダイナミクスと表現を共有しながら問いごとに価値関数を持つ Horde では、問いを MDP から切り離せることが特に効く [p.4, Section 2.3]。

遷移ベースの割引は、遷移で起きる変化そのものを予測の対象にできる。例はタクシー領域で壁にぶつかる確率の予測で、動く行動が同じ状態にとどまる遷移で $\gamma(s,a,s)=0$ と置く [p.4, Section 2.3]。また次の 2 種類を区別できる [p.4, Section 2.3]。

- 硬い終了: $\gamma(s,a,s')=0$。
- 柔らかい終了: $\gamma(s,a,s')=\epsilon$（小さい正の値）。遷移の直後の価値を少しだけ反映させるので、遷移までは同等だがその後のリターンが大きく違う 2 つの方策を区別できる。

## 3. タクシー領域での実演（Demonstration in the taxi domain）

タクシー領域 [Dietterich (2000)]、[Diuk et al. (2008)] に、車の向きを足して使う。乗り場から行き先へ客を運び、報酬は毎ステップ $-1$（乗車と降車に成功した遷移は 0）である。右折・左折・後退には追加の $-0.05$、$-0.1$、$-0.2$ が掛かり、客を乗せているとこの追加分は 2 倍になる [p.4, Section 3]。状態は車の 2 座標、客の位置、行き先、向きの組である。方策と価値は十分な回数の反復で計算している [p.4, Section 3]。サンプルからの学習や関数近似の実験ではないと読めるが、これは本解説の読み取りで、論文はそう述べておらず、Figure 2 のキャプションは方策を learned とも書く [p.5, Figure 2]。

Figure 2 [p.5, Figure 2] は 5×5 の格子と、割引の違う 3 つの方策の軌道を示す。

- (b) 乗車の遷移で $\gamma=0.1$（柔らかい終了）、他は 0.99: 乗車後の向きまで見越して近づく最適な方策で、総報酬は $-7.7$ である。
- (c) 乗車の遷移で $\gamma=0$（硬い終了）: 乗り場までの曲がる費用だけを減らし、乗車後に左折が増える。総報酬は $-8$ である。
- (d) 状態ベースの $\gamma=0$（乗り場にいて客が車内の状態）: 乗車後に乗り場へ戻るだけで負のリターンを打ち切れるので、乗り場の周りを往復して降車しない。

Figure 2(e) は 100 ステップ・5000 回の結果である。一部を抜き出す。

- 乗降回数: 100 ステップで成功した乗車と降車の回数。
- 曲がる費用: 向きの変更で追加された費用の量。

| 割引 | 乗降回数 | 曲がる費用 |
|---|---|---|
| 遷移ベース（柔らかい） | 7.74 ± 0.03 | 5.54 ± 0.01 |
| 遷移ベース（硬い） | 7.73 ± 0.03 | 5.83 ± 0.01 |
| 状態ベース | 0.00 ± 0.00 | 18.8 ± 0.02 |
| 定数 $\gamma_c=0.8$ | 7.34 ± 0.03 | 4.62 ± 0.01 |

定数の割引は付録 C の式 (8) により平均報酬の目標に対応する。理論上は柔らかい終了と同じ解に至るはずだが、実際には数値誤差で準最適になった [p.4, Section 3]。Figure 2(e) で $\gamma_c$ を 0.1 から 0.99 まで振ると、0.7〜0.9 以外は乗降回数がほぼ 0 である。負の報酬の無限和では向きの小さな差を区別しにくいためで、部分目標を持てるほうが判断の連鎖と局所的な評価がしやすいことを示唆する、と著者は述べる。小さい $\gamma_c$ の有用性 [Jiang et al. (2015)] に加えて、部分タスクを作る $\gamma$ も学習を助ける手段だと主張する [p.4, Section 3]。

## 4. 目的とアルゴリズム（Objectives and algorithms）

記号:

| 記号 | 意味 |
|---|---|
| $P_\pi(s,s')$ | $\sum_a\pi(s,a)\Pr(s,a,s')$。方策のもとでの遷移行列 |
| $P_{\pi,\gamma}(s,s')$ | $\sum_a\pi(s,a)\Pr(s,a,s')\gamma(s,a,s')$。割引を掛けた遷移行列 |
| $r_\pi(s)$ | $\sum_a\pi(s,a)\sum_{s'}\Pr(s,a,s')r(s,a,s')$。期待報酬 |
| $v_\pi$ | 方策の価値（期待リターン） |
| $\lambda_c$ | 定数の trace パラメータ。bias と variance の調整に使う [Kearns & Singh (2000)] |
| $X$, $D$, $w$ | 特徴の行列（$n\times d$）、対角の重み行列、線形近似の重み |
| $d_\mu$, $d_\pi$ | 挙動方策 $\mu$ と方策 $\pi$ の定常分布 |

価値は $v_\pi=r_\pi+P_{\pi,\gamma}v_\pi$ を満たす [p.4, Section 4]。定数割引なら $P_{\pi,\gamma}=\gamma_cP_\pi$ である。本文は割引に焦点を当てるため trace を定数 $\lambda_c$ とし、遷移ベースの trace は付録に回す [p.5, Section 4]。一般化した Bellman 作用素は次のとおりである [p.5, (1)–(2)]。状態ベースの割引と trace までの一般化は先行研究にある [Sutton et al. (2016), Eq. 29]、[Sutton (1995)]、[Sutton & Barto (1998)]、[Reza & Sutton (2010)]、[Sutton et al. (2014)]、[Yu (2012)]、[p.4, Section 4; p.4–5, footnote 2]。

$$T^{(\lambda)}v=r^\lambda_\pi+P^\lambda_\pi v,\qquad P^\lambda_\pi=(1-\lambda_c)(I-\lambda_cP_{\pi,\gamma})^{-1}P_{\pi,\gamma},\qquad r^\lambda_\pi=(I-\lambda_cP_{\pi,\gamma})^{-1}r_\pi$$

これが期待 $\lambda$-リターンの再帰を解いた形であることは、$v_{\pi,\lambda}=r_\pi+P_{\pi,\gamma}\bigl[(1-\lambda_c)v+\lambda_cv_{\pi,\lambda}\bigr]$ を展開して確かめられる。無限和が収束するのは $P_{\pi,\gamma}$ の最大固有値が 1 未満だからで、その証明は Lemma 3 にある [p.5, Section 4; p.5, footnote 3]。

多くのアルゴリズムは、$\gamma_c$ や $\gamma_s(s_{t+1})$ を遷移ベースの値へ置き換えるだけで一般化できる。TD はステップ幅を $\alpha_t$ として次のようになる [p.5, (3)]。

$$\delta_t=r_{t+1}+\gamma_{t+1}x(s_{t+1})^\top w-x(s_t)^\top w,\qquad e_t=\gamma_t\lambda_ce_{t-1}+x(s_t),\qquad w_{t+1}=w_t+\alpha_t\delta_te_t$$

線形近似の不動点は $Aw=b$ で書ける [p.6, Section 4]。

$$A=X^\top D(I-\lambda_cP_{\pi,\gamma})^{-1}(I-P_{\pi,\gamma})X,\qquad b=X^\top D(I-\lambda_cP_{\pi,\gamma})^{-1}r_\pi$$

$D$ は普通 $\mathrm{diag}(d_\mu)$ で、オンポリシーでは $d_\mu=d_\pi$、関心度を入れると $\mathrm{diag}(d_\mu\circ\mathrm i)$ になる（$\circ$ は要素ごとの積）。emphatic TD（ETD）[Mahmood et al. (2015)]、[Sutton et al. (2016)] は $D=M=\mathrm{diag}(m)$ を使う。$m$ は $m^\top(I-P^\lambda_\pi)=(d_\mu\circ\mathrm i)^\top$ を満たすベクトルで（印字の形は項目 2）、この重みならオフポリシーでも $A$ は正定値になる [p.6, Section 4]。

## 5. 一般化した理論的性質（Generalized theoretical properties）

既存の上界はほとんど定数割引を仮定しており、ETD の解析 [Hallak et al. (2015)] も定数の $\gamma_c$ だった。本節は $P_{\pi,\gamma}$ の行列ノルムを使って両方の型を一度に扱う [p.6, Section 5]。

記号: 重み付きノルムを $\|v\|_D=\sqrt{v^\top Dv}$、誘導される行列ノルムを $\|P\|_D=\|D^{1/2}PD^{-1/2}\|_{\mathrm{sp}}$（$\mathrm{sp}$ は最大特異値。印字の形は項目 11）、$s_D=\|P^\lambda_\pi\|_D$、$\mathcal F_v$ を解の候補の部分空間（例えば $Xw$ の形）、$\Pi_D$ を $D$ ノルムでの $\mathcal F_v$ への射影、$D_\pi=\mathrm{diag}(d_\pi)$、$v^*$ を真の価値とする [p.6, Section 5]。

仮定 [p.6, Section 5]:

- A1: 状態と行動は有限。
- A2: $\mu$ と $\pi$ の不変分布 $d_\mu$、$d_\pi$ が一意に存在する（普通はエルゴード性で満たす）。
- A3: $\gamma(s,a,s')<1$ かつ $\pi(s,a)\Pr(s,a,s')>0$ である遷移が存在する。方策が割引 1 未満の遷移に実際に届くという条件である。
- A4: $\mathcal F_v$ の関数は、関心度が正の状態で 0 ならすべての状態で 0 である。

### 5.1 近似誤差の上界（Approximation bound）

Lemma 1: $D=D_\pi$ または $D=M$ のとき $s_D<1$ である [p.6, Lemma 1]。定数の $\gamma_c,\lambda_c$ での既知の結果 [Tsitsiklis & Van Roy (1997)]、[Hallak et al. (2015)] を遷移ベースへ広げたものである。証明の流れは 2 通りある。

- $D=M$: Jensen の不等式と、$P^\lambda_\pi$ の行和が 1 以下であること（Lemma 4）から $\|P^\lambda_\pi v\|_M^2\le\|v\|_M^2-v^\top\mathrm{diag}(d_\mu\circ\mathrm i)v$ を得る。最後の項が正であることを A4 に当たる条件で保証する [p.6–7, Lemma 1]。
- $D=D_\pi$: A3 を根拠に、ある $s<1$ と任意の非負ベクトル $v_+$ について $d_\pi P_{\pi,\gamma}v_+\le s\,d_\pi v_+$ が成り立つとし、$\|P^\lambda_\pi v\|_{D_\pi}^2\le\frac{s-s\lambda_c}{1-s\lambda_c}\|v\|_{D_\pi}^2$ を導く [p.7, Lemma 1]。この不等式は A3 だけからは出ず、$\lambda_c=0$ では Lemma 1 の主張そのものが成り立たない例がある（項目 3）。

Lemma 2: A1–A3 のもとで $T^{(\lambda)}$ は $D_\pi$ または $M$ で重み付けたノルムの縮小写像であり、$\Pi_DT^{(\lambda)}$ も唯一の不動点を持つ [p.7, Lemma 2]（主張の書き方は項目 1）。差を取ると $r$ が消えて $P^\lambda_\pi$ だけが残り、Lemma 1 が使える。

Theorem 1: $s_D<1$ なら射影の不動点 $v$ が存在し、次の誤差上界が成り立つ [p.7, Theorem 1, (4)]。

$$\|v-v^*\|_D\le(1-s_D)^{-1}\|\Pi_Dv^*-v^*\|_D$$

定数の $\gamma_c\in[0,1)$ と $\lambda_c\in[0,1]$、$D=D_\pi$ では $s_D\le\gamma_c(1-\lambda_c)/(1-\gamma_c\lambda_c)$ となり、既知の上界 $(1-s_D)^{-1}\le(1-\gamma_c\lambda_c)/(1-\gamma_c)$ [Tsitsiklis & Van Roy (1997), Lemma 6] に帰着する [p.7, Theorem 1]。

遷移ベースの trace については、emphatic 重みなら同じ縮小が成り立つが、$d_\pi$ 重みでは成り立たない例があり、この点が emphatic 重みの柔軟さの根拠になる、と著者は述べる [p.8, Section 5.1]。詳細は付録 G にある。

### 5.2 TD アルゴリズムの性質（Properties of TD algorithms）

ETD の収束: 定数割引または状態ベースの ETD と ELSTD（$D=M$ の $A,b$ を使う最小二乗版）は、確率 1 の収束が示されている [Yu (2015)]。その証明の要は $A$ の正定値性による期待値での収束である。遷移ベースへの拡張として本論文が示すのは、この期待値での収束である（証明は付録 F）[p.8, Section 5.2]。

LSTD($\lambda$) の収束レート: 継続型の結果 [Tagorti & Scherrer (2015)] は、感度の補題 [Tagorti & Scherrer (2015), Lemma 1] の行列を $P^\lambda_\pi$ に置き換えると、定数 $(1-\lambda_c\gamma_c)/(1-\gamma_c)$ が $1/(1-s_D)$ に変わる形でエピソード型へ広がる。emphatic LSTD にも同様に広がり、レートは $s_M$ で決まる [p.8, Section 5.2]。

$s_D$ の読み方: 定数の場合の傾向（$\lambda_c$ が 1 に近づくか $\gamma_c$ が 0 に近づくと上界の定数が 1 へ下がる）は直観に合うが、遷移ごとに割引が大小する一般の場合は傾向を読みにくい [p.8, Section 5.2]。ランダム方策のタクシー領域の $s_D$ を Table 1 [p.8, Table 1] に示している。本文の所見は 3 点ある。乗車を硬い終了にしても柔らかい終了にしても $s_D$ は同じ。$\gamma_c=0.99$ より「エピソード型」のほうが少し小さい。終了を増やすより $\lambda_c$ を上げるほうがずっと効く。

$\lambda_c=0$ の値を例にとると、エピソード型のタクシーは 0.989、$\gamma_c=0.99$ は 0.990 で、$\lambda_c=0.99$ ではそれぞれ 0.483 と 0.497 に下がる。ランダムな終了を足す行は、本文では似た値と述べられるが、10% の状態からすべての経路で 5 ステップ以内に終わる行だけは $\lambda_c=0$ で 0.898 と目に見えて下がる [p.8, Table 1]。

本解説の照合: $\gamma_c=0.99$ の行は 5 列とも $\gamma_c(1-\lambda_c)/(1-\gamma_c\lambda_c)$ の値に一致する。$s_D=0.989$ のとき $(1-s_D)^{-1}$ は約 91 である。「エピソード型」の行で乗車以外の割引が 0.99 か 1 かは、本文に書かれていない。

## 6. 考察と結論（Discussion and conclusion）

論文の目的は RL タスク形式の使い方を直観と例で示すことで、技術的な貢献は付録に置いた部分もある。著者は改めて 4 点を挙げる。RL タスク形式と遷移ベースの割引の導入、状態ベースと遷移ベースの関係の明示、エピソード型と継続型に共通の近似上界、状態ベース・遷移ベースへ縮小の結果を広げるときの知見と問題点である [p.8, Section 6]。謝辞は遷移ベースの割引と確率的な割引の議論について Hado van Hasselt に向けられている [p.9, Acknowledgements]。

## A. 確率的な割引を含む定式化（More general formulation with probabilistic discounts）

報酬と割引を $\Pr(r,\gamma\mid s,a,s')$ から同時に出す形にしても、扱いはほとんど変わらない [p.10, Appendix A]。次状態 $s'$ を固定すれば $v_\pi(s')$ は定数なので、期待を取ると報酬と割引の項に分かれ、$\gamma(s,a,s')=\mathbb E[\gamma\mid s,a,s']$ と置いた同じ式に戻るからである。

## B. 状態ベースと遷移ベースの割引の関係（Relationship between state-based and transition-based discounting）

### B.1 等価性の結果（Equivalence result）

遷移の組 $(s,a,s')$ ごとに仮想状態 $f_{sas'}$ を足すと、遷移ベースの MDP を状態ベースの MDP へ写せる。状態数は $\lvert\mathcal A\rvert n^2+n$ になる。仮想状態の割引を $\gamma(s,a,s')$、他を 1 とし、報酬は仮想状態へ入るときに受け取り、仮想状態の方策は一様、線形近似では $f_{sas'}$ の特徴を $x(s')$ に揃える [p.10, Appendix B.1; p.11, Appendix B.3]。

Theorem 2: 元の状態へ正規化した定常分布は元の $d_\pi$ に一致し、元の状態の価値も一致する。さらに、どちらで最適化しても同じ方策になる [p.10, Theorem 2, (5); p.11, Theorem 2, (6)]（書き方は項目 6）。関数近似でも解が同じになるという意味で等価だが、大きい MDP で学習する実用面は無視している [p.10, Appendix B]。

### B.2 遷移ベースの割引の利点（Advantages of transition-based discounting over state-based discounting）

表現力が同じでも遷移ベースが望ましい理由を、著者は次のように挙げる [p.10–11, Appendix B.2]。

- 大きさ: 最悪で状態数が $\lvert\mathcal A\rvert n^2+n$ になる。
- 問題の定義: 割引の違う学習器ごとに誘導される MDP が変わり、変数や計算を共有しにくい。
- 手間: 仮想状態の保存と更新、特徴の設定が要る。
- 定常分布: 余分な状態が入り、再正規化が要る。
- オフポリシー: 多数の学習器を理論で扱うと MDP が別々になる [White A. (2015)]。
- アルゴリズムの書き方: エピソード型の true-online TD [van Seijen & Sutton (2014)]、継続型の低分散 ETD [Hallak et al. (2015)]、エピソードのループを持つ Sarsa [Sutton & Barto (1998), Figure 8.8] のように分かれていたものが、$\gamma_{t+1}$ を渡すだけの 1 つの形になる。trace を消す手順も不要になる。
- 実験の設計: エピソード数とステップ数の使い分けをやめ、ステップ数でそろえられる。エピソードの長さの打ち切りも、最大ステップ数の制限に置き換えられる。

### B.3 Theorem 2 の証明（Proof of Theorem 2）

仮想状態を含む遷移行列 $\bar P_\pi$ と分布 $\bar d_\pi$（正規化定数 $c$）を作り、$j\in\mathcal S$ と $j\in\mathcal F$ に分けて不変性 $\bar d_\pi\bar P_\pi=\bar d_\pi$ を確かめる。価値は同じ不動点方程式を満たし、重み付きの和は元の状態の和の $2/c$ 倍になるので、最適化の結果も同じになる [p.11–12, Appendix B.3]。状態ベースでエピソード問題を表せるという従来の理論の、定常分布の等価性が示されていない隙間を埋める、と著者は述べる [Yu (2015)]、[p.11, Appendix B.3]。

## C. 割引と平均報酬（Discounting and average reward for control）

定数割引では $d_\pi P_\pi=d_\pi$ を使うと次式が得られ、定数割引のリターンの重み付き和は平均報酬の定数倍にすぎない [p.12, (7)–(8)]。

$$d_\pi v_\pi=\frac{1}{1-\gamma_c}\,d_\pi r_\pi$$

ただしこの議論は遷移ベースには及ばない。割引が遷移によって違うとリターンの重み付けが一様でなくなり、最適方策の選択そのものが変わるからで、タクシー領域がその実演である [p.12, Appendix C]。

## D. アルゴリズム（Algorithms）

true-online TD($\lambda$) と ELSTDQ($\lambda$) の疑似コードがある [p.13, Algorithm 1; p.13, Algorithm 2]。後者は重点サンプリングを持つ LSTDQ の変種である [p.12–13, Appendix D]。どちらも入力として毎ステップ $\gamma_{t+1}$（と $\lambda_{t+1}$）を受け取る。エピソードの先頭で特別な初期化はしない。$\gamma_{t+1}=0$ なら trace が 0 になり、true-online TD の補正項が消えて普通の更新になるからである [p.13, Appendix D]。

$\gamma_{t+1}$ が入るのは次の場所である（本解説による整理）。

- Algorithm 1: TD 誤差のブートストラップの係数と、trace の減衰。
- Algorithm 2: 上と同じ 2 か所に加えて、emphasis の累積 $F\leftarrow\gamma_{t+1}F+\mathrm i(x_t)$。

## E. 補題（Lemmas）

Lemma 3: A3 のもとで $P_{\pi,\gamma}$ の最大固有値は 1 未満である [p.13, Lemma 3]。既約な $P_\pi$ の成分を減らす形で $P_{\pi,\gamma}$ を作り、固有値が下がることを示す。[Yu (2015)] の Assumption 2.1 を明示的に確かめたものでもある [p.13, Appendix E]。証明の道具は項目 5 を参照。

Lemma 4: 遷移ベースの trace $\lambda(s,a,s')$ を入れるため $P_{\pi,\gamma,\lambda}(s,s')=\sum_a\pi\Pr\gamma\lambda$、$P_{\pi,\gamma,1-\lambda}=P_{\pi,\gamma}-P_{\pi,\gamma,\lambda}$ と定義する。A3 のもとで $I-P_{\pi,\gamma,\lambda}$ は正則で、$P^\lambda_\pi=(I-P_{\pi,\gamma,\lambda})^{-1}P_{\pi,\gamma,1-\lambda}$ は非負で行和が 1 以下である [p.14, Lemma 4]。

## F. emphatic アルゴリズムの収束（Convergence of emphatic algorithms for the RL task formalism）

状態ベースの結果は B.1 の等価性で遷移ベースへ自動で移るはずだが、著者は書き方をそろえるため遷移ベースのまま証明する [p.14, Appendix F]。

Theorem 3: 線形近似で $X$ の列が独立、関心度が正、$M=\mathrm{diag}(m)$ のとき、$A=X^\top M(I-P^\lambda_\pi)X$ は正定値である [p.14, Theorem 3]。$M(I-P^\lambda_\pi)$ の対角が非負、非対角が非正、行和が非負、列和が正であることを示す [Sutton et al. (2016), Theorem 1]、[Yu (2015), Proposition C.1]。関心度が 0 の状態があるときは A4 のもとで [Yu (2015), Corollary C.1] の議論に従う [p.15, Appendix F]。

ELSTDQ は状態と行動の組を状態と見なす拡張で扱い、同じ性質が移る。Corollary 1 は行動価値の $A_q$ の正定値性を述べる [p.15, Corollary 1]。

## G. emphatic 重みなしの遷移ベース trace の問題（Issues with transition-based trace without emphatic weighting）

$d_\pi$ 重みのまま trace を遷移ベースにするには、trace の関数を制限する必要がある [p.15, Appendix G]。任意の非負ベクトルについて $d_\pi P_{\pi,\gamma,\lambda}v_+\le s_\lambda d_\pi v_+$、$d_\pi P_{\pi,\gamma,1-\lambda}v_+\le s_{1-\lambda}d_\pi v_+$ となる $s_\lambda$、$s_{1-\lambda}$ を取ると、Lemma 5 は $\|P^\lambda_\pi\|_{D_\pi}\le s_{1-\lambda}/(1-s_\lambda)$ を与える。縮小になるのは $s_{1-\lambda}<1-s_\lambda$ のときである [p.15, Lemma 5]。定数の trace ならこの条件は満たされる。

反例は遷移も方策も一様な 2 状態の MDP で、$\gamma_c=0.99$ とし、$\lambda$ を $s_1$ へ入るとき 0.9、$s_2$ へ入るとき 0 にする。$P^\lambda_\pi$ の 2 行はともに $(0.0893,\,0.8927)$ となり、縮小にならない [p.15–16, Appendix G]。

本解説の照合: この行列を計算すると同じ値になり、$\|P^\lambda_\pi\|_{D_\pi}$ は約 1.27 である。

## 論文が述べている限界

この節は本解説が追加したもので、論文自身が述べる限界を言い換えて集める。

- $s_D$ は定数の場合より解釈しにくい [p.8, Section 5.2]。
- $d_\pi$ 重みと遷移ベースの trace では縮小が保証されない [p.15–16, Appendix G]。
- 遷移ベースの ETD について論文が示す収束は期待値でのものである [p.8, Section 5.2]。
- 定数割引と平均報酬の対応は遷移ベースには及ばない [p.12, Appendix C]。
- Theorem 2 の等価性は、大きい MDP で学習する実用面を無視している [p.10, Appendix B]。
- タクシー領域では数値誤差が定数割引の結果を左右している [p.4, Section 3; p.5, Figure 2]。

## 論文内の記載の食い違い

この節も本解説が追加したもので、本解説による照合の結果である。項目 3 の Lemma 1 への疑問は第三者照合でも独立に確認された。他の項目が個別に確認されたとは限らない。

1. Lemma 2 の主張は $\|T^{(\lambda)}v\|_D<\|v\|_D$ と印字されているが、証明が示すのは 2 つのベクトルの差についての縮小である [p.7, Lemma 2]。$T^{(\lambda)}$ は定数項 $r^\lambda_\pi$ を持つので、印字のままでは $v=0$ で成り立たない。
2. $m$ の定義は $(I-P^\lambda_\pi)^{-1}(d\circ\mathrm i)$ と転置なしで印字されるが [p.6, Section 4; p.14, Theorem 3]、証明で使うのは $m^\top(I-P^\lambda_\pi)=(d\circ\mathrm i)^\top$ である [p.6, Lemma 1; p.15, Appendix F]。本文では証明に合わせた形で書いた。
3. Lemma 1 の $D_\pi$ 側の主張は、$\lambda_c=0$ では A1–A3 のもとでも成り立たないことがある [p.7, Lemma 1]。
   - 証明の不等式: $d_\pi P_{\pi,\gamma}v_+\le s\,d_\pi v_+$ は、次状態ごとに「流入の重みで平均した $\gamma$」が $s$ 以下であることと同じで、割引 1 未満の遷移が 1 つあるという A3 だけからは出ない。$\gamma=1$ の遷移でだけ入る状態があると破れる。
   - 不等式の破れと $s_{D_\pi}=1$ は別である: 確率 0.5 ずつの $s_1\to s_1,s_2$、$s_2\to s_1,s_3$、$s_3\to s_1,s_2$ で $s_1$ へ入る遷移だけ $\gamma=0.5$ の連鎖は、不等式が破れても $\lambda_c=0$ の $s_{D_\pi}$ は約 0.79 である。
   - $s_{D_\pi}=1$ の条件: $\lambda_c=0$ でこうなるのは、$\gamma=1$ でしか入れない状態の集まり $T$ があり、$T$ へ遷移しうる状態は必ず $T$ へ遷移するときである。$T$ が 1 状態なら、前の状態から確率 1 で入るときに当たる（本解説の計算。ランダムな連鎖 878 本でも一致した）。
   - 反例: $s_1\to s_2$（$\gamma=1$）、$s_2\to s_3$（$\gamma=1$）、$s_3\to s_1$（確率 0.5、$\gamma=0.5$）、$s_3\to s_3$（確率 0.5、$\gamma=1$）のエルゴードな連鎖は $T=\{s_2\}$ に当たり、$\lambda_c=0$ で $\|P_{\pi,\gamma}\|_{D_\pi}=1$ となって Theorem 1 の上界は意味を失う。同じ例で最大固有値は約 0.848、$\|P_{\pi,\gamma}\|_M$ は約 0.913 である。
   - 残ること: A3 と既約性のもとで $\|P^\lambda_\pi\|_{D_\pi}\le1$ かつ最大固有値は 1 未満である。ここから、$\lambda_c\in(0,1]$ では $s_{D_\pi}<1$（同じ例の $\lambda_c=0.5$ で約 0.794）、どの $\lambda_c$ でも $D_\pi(I-P^\lambda_\pi)$ の対称部分は正定値であることが言える（本解説の計算）。崩れるのは $\lambda_c=0$ で 1 回の作用ごとに必ず縮むという性質とそれに基づく上界で、$A$ の正定値性は残る。
4. Lemma 2 の仮定は A1–A3 だが、Lemma 1 の $M$ 側の証明は A4 に当たる条件を使う [p.7, Lemma 1; p.7, Lemma 2]。また A4 は $\mathcal F_v$ の関数についての条件なのに、Lemma 1 のノルムは $\mathbb R^n$ 全体の最大で定義される [p.6, Section 5; p.7, Lemma 1]。関心度に 0 があると主張が破れることがあり、3 状態の巡回（$s_3\to s_1$ だけ $\gamma=0.5$、他は 1）で $\mathrm i=(1,0,0)$、$\lambda_c=0$ とすると $\|P_{\pi,\gamma}\|_M=1$ である（本解説の計算）。関心度がすべて正ならこの問題は起きない。
5. Lemma 3 の証明は Courant–Fischer–Weyl の原理を非対称行列へ使っている [p.13, Lemma 3; p.13, footnote 4]。この原理は普通は対称行列についてのものである。ただし結論自体は、既約な非負行列の成分を減らすとスペクトル半径が下がるという Perron–Frobenius の標準的な事実からも得られる。また証明は $P_\pi$ の既約性を使うが、A3 の書き方にそれは現れない。
6. Theorem 2 は 2 か所に置かれ、p.10 は方策の一致まで述べて argmin で書き、p.11 は価値と定常分布だけを述べる [p.10, Theorem 2; p.11, Theorem 2]。報酬を最大化する設定なので、argmin は argmax の誤記の可能性がある。
7. 付録 G は $1-s_\lambda=0.9\gamma_c$、上界を $1.\bar1$ と書く [p.16, Appendix G]。$s_\lambda=0.9\gamma_c=0.891$ なら $1-s_\lambda=0.109$ で、Lemma 5 の上界は約 9.08 になる。$1.\bar1$ は $s_{1-\lambda}/s_\lambda$ の値である。反例の結論は $P^\lambda_\pi$ の直接計算で確かめられるので影響しない。
8. Section 5.1 はこの反例を状態ベースの割引の例と呼ぶが、例の割引は定数で、入る状態に依存するのは trace である [p.8, Section 5.1; p.15, Appendix G]。
9. 行動価値の遷移行列は $((s,a),(s,a'))$ と印字されるが、右辺に合わせれば $((s,a),(s',a'))$ である [p.15, Appendix F]。
10. 参考文献の [Reza & Sutton (2010)] と [Sutton et al. (2009)] の共著者は「Hamid Maei Reza」と印字されている [p.9, References]。本解説のラベルは論文の表記に合わせた。
11. 行列ノルムは p.6 で $\|D^{1/2}P^\lambda_\pi D^{1/2}\|_{\mathrm{sp}}$ と印字されるが、p.7 の計算は $D^{-1/2}$ を使う [p.6, Section 5; p.7, Theorem 1]。本文では p.7 に合わせた形で書いた。
12. Figure 1 のキャプションは「(c) について」と切り出すが、続く 2 つの場合の説明は (d) の図の内容である [p.3, Figure 1]。(c) が状態を足す理由の説明とも読めるので、誤記とは断定しない。

## キューの論点への答え

reports/INDEX.md のキューの問いに、論文の記述の範囲で答える。論文が扱っていない点と本解説の照合は、そう明記する。

1. 遷移ごとの割引 $\gamma(s,a,s')$ の定式化と、収束や安定性の条件
   - 定式化: 割引は MDP ではなくタスクの一部で、値域は $[0,1]$ である [p.2, Section 2]。リターンは割引の積で定義し、定数割引は $P_{\pi,\gamma}=\gamma_cP_\pi$ の特殊例になる [p.2, Section 2; p.4, Section 4]。Bellman 作用素と TD は $P_{\pi,\gamma}$ と $\gamma_{t+1}$ に置き換わる [p.5, (1)–(3)]。
   - 条件: 有限（A1）、一意な定常分布（A2）、方策が割引 1 未満の遷移へ届くこと（A3）、関心度と部分空間の整合（A4）である [p.6, Section 5]。A3 から $P_{\pi,\gamma}$ の最大固有値が 1 未満であることが出る [p.13, Lemma 3]。
   - 保証の範囲:

     | 対象 | 重み | 論文の結果 | 位置 |
     |---|---|---|---|
     | $T^{(\lambda)}$ の縮小、定数の trace | $D_\pi$、$M$ | 縮小（$D_\pi$ 側は $\lambda_c=0$ で反例あり。項目 3） | [p.7, Lemma 2] |
     | $T^{(\lambda)}$ の縮小、遷移ベースの trace | $M$ | 縮小 | [p.8, Section 5.1] |
     | 同上 | $D_\pi$ | 条件つき。反例あり | [p.15–16, Appendix G] |
     | 射影不動点の誤差 | $s_D<1$ の $D$ | $(1-s_D)^{-1}$ 倍の上界 | [p.7, Theorem 1] |
     | ETD と ELSTDQ の $A$ | $M$ | 正定値。期待値での収束 | [p.8, Section 5.2; p.14, Theorem 3; p.15, Corollary 1] |
     | LSTD($\lambda$) のレート | $D_\mu$、$M$ | 定数を $1/(1-s_D)$ へ置き換えて拡張 | [p.8, Section 5.2] |

   - 論文が扱っていない点: 結果はすべて固定した方策 $\pi$ の評価についてで、線形の関数近似までである。max を含む制御の作用素の縮小、Q 学習や DQN の収束、深層ネットワークは証明も実験もない。タクシー領域の制御は反復計算での実演である [p.4, Section 3]。リプレイの分布は $D_\pi$ でも $M$ でもないので、これらの重みの結果はそのままは当てはまらない（本解説の読み取り）。
2. エピソード型と継続型の統一
   - 方法: エピソードの終わりを、開始状態へ普通に遷移しつつ $\gamma=0$ を返す遷移として書く。吸収状態も仮想状態も足さない [p.2, Section 2.1]。
   - アルゴリズム: 違いは渡す $\gamma_{t+1}$ だけになり、trace は自動で消え、エピソード先頭の特別な処理が要らない [p.2, Section 2.1; p.11, Appendix B.2; p.13, Appendix D]。
   - 理論: 上界が $P^\lambda_\pi$ を通して両方を覆い、定数の場合の既知の結果に帰着する [p.6, Section 5; p.7, Theorem 1]。
   - 状態ベースとの関係: 状態を足さない状態ベースでは表せない [p.3, Figure 1]。仮想状態を足せば等価だが、状態数と手間が増える [p.10, Theorem 2; p.10–11, Appendix B.2]。
3. 「ある遷移では割り引かない（$\gamma=1$）」設計は扱えるか、TD 系の更新式で何が変わるか
   - 扱えるか: 値域に 1 が含まれ、形式上は扱える。チェーンの例は終了の遷移以外すべて $\gamma=1$ で、毎ステップ $-1$ の報酬を持つ [p.2, Section 2.1]。必要なのは、方策のもとで割引 1 未満の遷移へ届くことである [p.6, A3]。
   - 本解説の補足: Lemma 3 の証明は既約性を使う。既約でないなら、方策のもとで出られない状態の集まりのそれぞれに割引 1 未満の遷移が要り、その中がすべて $\gamma=1$ だと最大固有値は 1 になる。
   - TD の更新式で変わること:
     - TD 誤差のブートストラップが減衰しない。$\delta_t=r_{t+1}+x(s_{t+1})^\top w-x(s_t)^\top w$ である。
     - trace は $\gamma$ で減らず $\lambda$ でだけ減る。$e_{t+1}=\lambda_ce_t+x(s_{t+1})$ である（$\lambda_c=1$ なら減らない）。
     - ETD 系の $F$ は減衰しないので、$\gamma=1$ が続く間は関心度の分だけ増え続ける [p.13, Algorithm 2]（増えるという読み取りは本解説による）。
     - アルゴリズムの形は変わらない [p.11, Appendix B.2]。定常分布は $P_\pi$ だけで決まるので $\gamma$ によらない（A2 の定義からの本解説の読み取り）[p.6, Section 5]。
   - 理論で変わること: $s_D$ が 1 に近づくと上界 $(1-s_D)^{-1}$ が緩む [p.7, Theorem 1]。項目 3 のとおり、$\gamma=1$ でしか入れない状態へ前の状態から確率 1 で入るような場合は、$D_\pi$、$\lambda_c=0$ で $s_D=1$ になる（本解説の照合）。この場合も、オンポリシー（$D_\pi$）に限れば $A$ の正定値性は残り、期待値の TD は安定側である（本解説の計算）。
   - 論文が扱っていない点: 数値の実演で使う割引はタクシー領域の 0.99、0.1、0 などで [p.5, Figure 2]、$\gamma=1$ が続く連鎖の挙動を測った結果はない。不動点が定まるかも、max を含む場合については述べていない。

## SMDP の PRD（956）の検討に効く点

この節は論文の主張ではなく、本解説が論文と docs/memo/956_smdp_transition_duration_10prd.md を突き合わせた読み取りである。論文に経過時間 $\tau$ は出てこない。固定点の式と Munchausen の試算は本解説の計算で、実 Run では確かめていない。

- §3.1 の形はこの枠組みに載る。$\gamma(s,a,s')=\gamma_{\text{base}}^{\tau(s,a,s')}$ と置けば、$\tau=0$ が $\gamma=1$ に当たる。Figure 1(d) の理由で、$\tau$ は状態ではなく遷移の属性にする。$\tau$ が確率的でも付録 A の形で扱える。
- §3.1.2 には、報酬に掛けるのは先行遷移までの積で自分の $\tau$ を含めないこと、ブートストラップは窓全体の積で、終端は同じ係数の 0 であることを書くとよい。積は $\gamma^{\sum\tau}$ なので、係数ではなく $\sum\tau$ を保存すれば $\gamma$ を後から変えられる（§3.1.3 の「保存した係数」との選択）。
- §1 の「理論の裏づけ」は範囲を絞る。この論文が支えるのは定式化と固定方策の評価までで、DQN・QR・IQN の max とリプレイの分布は範囲外である。§5.2 の安定性と §5.4 の PER はこの論文では決まらないので、実験で確かめるか別の根拠を探す。
- DROP 単位は `use_no_drop_timeout_gameover = true` と組にする。true なら時間切れが `done`（$\gamma=0$）になり、観測に `no_drop_timeout_ratio` が入る（core/envs/dropmerge1/src/DropMergeEnv.cpp の 1224–1226 行、1353–1358 行）。false だと `truncated` になり観測にも入らないので、打ち切りをブートストラップする場合、NOOP を続ける方策には割引 1 未満の遷移が無く A3 が崩れる。
- 待つ罰は、値段付けだけでなく問題の解を定めるために要る。盤面が変わらない NOOP を自己ループと見て、最善の DROP の価値を $V_D$、NOOP の価値を $x$、NOOP の 1 step の報酬を $c$ とすると $x=c+\gamma\max(x,V_D)$ である。
  - $\gamma=1,c=0$: $x\ge V_D$ のすべてが固定点で、NOOP の価値が定まらない。
  - $\gamma=1,c<0$: $x=V_D+c$ に定まり、差は $\lvert c\rvert$ になる（今の $(1-\gamma)V_D$ の代わり）。
  - $\gamma=1,c>0$: 固定点が無く発散する。
  - ショットクロックがあれば連鎖は有限だが、$c=0$ では今の DROP と後の DROP が同点で、max の上振れが最大 100 step ぶん効く。よって §5.5 の罰は推定の揺れより大きくする。
- §5.3 の Munchausen は $\tau=0$ の遷移で上向きの項になる。スカラー Q で方策が同じ Q のソフトマックスのとき、自己ループでの正味の増分は $(1-\alpha)\tau_{\text{m}}(-\ln p)\ge 0$ である（$\tau_{\text{m}}$ は `entropy_tau`、$p$ は NOOP の方策確率）。既定の $\alpha=0.9$、$\tau_{\text{m}}=0.03$（apps/runner/config/agent.txt の 228–229 行）では $p=0.5$ で 1 step あたり約 0.0021 である。$p$ を固定して 100 step 積むと約 0.21 だが、これは粗い目安で、実際には Q が上がると $p$ も上がる（2 行動なら NOOP が 0.03 上がるだけで $p$ は約 0.73）。
  - Q と $p$ を一緒に動かすと、DROP と同値から始めた NOOP の 100 step の上昇は、DROP 1 本で 0.069、5 本で 0.113、8 本で 0.126、16 本で 0.146 である（$p$ は約 0.9）。差 $-0.10$、$-0.16$ から始めても、DROP 1 本と 8 本の計算では 100 step で NOOP が上回る。
  - DROP の Q にも log-policy の加点を入れる（DROP の次の価値は固定）と、同値から始めた差は 0.14〜0.22 になるが、NOOP 自身の上昇は 0.014〜0.022 で、大半は DROP の低下による。
  - いずれも §2 の健全な時期の差 0.10〜0.16 と同じ桁である。§2 の実測（正味の項は全窓で負）は全遷移の平均で、ここは $\gamma=1$ の自己ループなので条件が違う。
  - 罰 $c$ との釣り合いは $p=\exp\bigl(c/((1-\alpha)\tau_{\text{m}})\bigr)$ になる。UQE 方策など Q と方策がずれる構成ではこの式のままではない。
- 時間の数え方を変えると、学習の速さだけでなく最適方策も変わる（付録 C）。§5.6 の物差しはこの理由でも要る。同じ $\gamma$ の値でも見通しは違い、半減期は 231 step から 231 DROP に伸びる。
- フレーム単位はすべての遷移で $\gamma<1$ になるので、max を含む作用素でも最大ノルムの標準的な縮小の議論が使え、§5.2 の問題は DROP 単位に限られる。ただし長い見通しを保つよう $\gamma$ をフレームあたりへ直すと NOOP と DROP の差はさらに縮むので [Tallec et al. (2019)]、罰を DROP 単位だけに書いている §3.2.2 は見直す余地がある。

## 出典リスト

対象論文のほかは、対象論文の参考文献欄の書誌による（月は不明）。URL を確かめたのは対象論文だけで、他の文献の本文は本解説では読んでいない。

[White, 2021/09] Martha White. "Unifying Task Specification in Reinforcement Learning." arXiv:1609.01995v4（2021 年 9 月 17 日。ICML 2017, PMLR vol. 70, pp. 3742–3750 に収録）. https://arxiv.org/abs/1609.01995 （PMLR 版: https://proceedings.mlr.press/v70/white17a.html ）

[Bertsekas & Tsitsiklis, 1996] Dimitri P. Bertsekas, John N. Tsitsiklis. "Neuro-dynamic Programming." Athena Scientific.

[Dietterich, 2000] Thomas G. Dietterich. "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition." Journal of Artificial Intelligence Research.

[Diuk et al., 2008] Carlos Diuk, Andre Cohen, Michael L. Littman. "An object-oriented representation for efficient reinforcement learning." ICML.

[Hallak et al., 2015] Assaf Hallak, Aviv Tamar, Rémi Munos, Shie Mannor. "Generalized Emphatic Temporal Difference Learning: Bias-Variance Analysis." CoRR abs/1509.05172.

[Jiang et al., 2015] Nan Jiang, Alex Kulesza, Satinder P. Singh, Richard L. Lewis. "The Dependence of Effective Planning Horizon on Model Accuracy." AAMAS.

[Kearns & Singh, 2000] Michael J. Kearns, Satinder P. Singh. "Bias-Variance error bounds for temporal difference updates." COLT.

[Mahmood et al., 2015] Ashique Rupam Mahmood, Huizhen Yu, Martha White, Richard S. Sutton. "Emphatic temporal-difference learning." European Workshop on Reinforcement Learning.

[Modayil et al., 2014] Joseph Modayil, Adam White, Richard S. Sutton. "Multi-timescale nexting in a reinforcement learning robot." Adaptive Behavior.

[Reza & Sutton, 2010] Hamid Maei Reza（印字のまま）, Richard S. Sutton. "GQ(λ): A general gradient algorithm for temporal-difference prediction learning with eligibility traces." Conference on Artificial General Intelligence.

[Sutton, 1995] Richard S. Sutton. "TD models: Modeling the world at a mixture of time scales." ICML.

[Sutton & Barto, 1998] Richard S. Sutton, A. G. Barto. "Reinforcement Learning: An Introduction." MIT Press.

[Sutton et al., 1999] Richard S. Sutton, Doina Precup, Satinder Singh. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." Artificial Intelligence.

[Sutton et al., 2009] Richard S. Sutton, Hamid Maei Reza（印字のまま）, Doina Precup, Shalab Bhatnagar. "Fast gradient-descent methods for temporal-difference learning with linear function approximation." ICML.

[Sutton et al., 2011] Richard S. Sutton, Joseph Modayil, Michael Delp, Thomas Degris, Patrick M. Pilarski, Adam White, Doina Precup. "Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction." AAMAS.

[Sutton et al., 2014] Richard S. Sutton, Ashique Rupam Mahmood, Doina Precup, Hado van Hasselt. "A new Q(lambda) with interim forward view and Monte Carlo equivalence." ICML.

[Sutton et al., 2016] Richard S. Sutton, Ashique Rupam Mahmood, Martha White. "An emphatic approach to the problem of off-policy temporal-difference learning." Journal of Machine Learning Research.

[Tagorti & Scherrer, 2015] Manel Tagorti, Bruno Scherrer. "On the Rate of Convergence and Error Bounds for LSTD(λ)." ICML.

[Tsitsiklis & Van Roy, 1997] John N. Tsitsiklis, Benjamin Van Roy. "An analysis of temporal-difference learning with function approximation." IEEE Transactions on Automatic Control.

[van Hasselt, 2011] Hado Philip van Hasselt. "Insights in Reinforcement Learning." PhD thesis.

[van Seijen & Sutton, 2014] Harm van Seijen, Rich Sutton. "True online TD(lambda)." ICML.

[White A., 2015] Adam White. "Developing a predictive approach to knowledge." PhD thesis, University of Alberta（本文の表記は White, 2015）.

[Yu, 2012] Huizhen Yu. "Least Squares Temporal Difference Methods: An Analysis under General Conditions." SIAM Journal on Control and Optimization.

[Yu, 2015] Huizhen Yu. "On convergence of emphatic temporal-difference learning." COLT.

対象論文の参考文献欄に無く、「SMDP の PRD（956）の検討に効く点」だけで参照したもの:

[Tallec et al., 2019] Corentin Tallec, Léonard Blier, Yann Ollivier. "Making Deep Q-learning Methods Robust to Time Discretization." ICML 2019. https://arxiv.org/abs/1901.09732 （解説: reports/tallec2019_time_discretization_paper_2026-10-03.md）
