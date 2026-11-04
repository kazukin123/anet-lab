# Increasing the Action Gap: New Operators for Reinforcement Learning — 詳細解説

- 著者: Marc G. Bellemare, Georg Ostrovski, Arthur Guez, Philip S. Thomas, Rémi Munos [p.1]
- 所属: Google DeepMind。Thomas は Carnegie Mellon University へ移った旨の脚注がある [p.1, footnote ∗]
- 発表: AAAI 2016（Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 30, No. 1）[Crossref (2026/11)]
- 年: 2016（公開日は 2016-02-21 [Crossref (2026/11)]。arXiv の投稿は 2015-12-15 で、版は v1 のみ [arXiv 1512.04860 (2015/12)]）
- arXiv: 1512.04860 [cs.AI]
- DOI: 10.1609/aaai.v30i1.10303 [Crossref (2026/11)]
- URL: https://arxiv.org/abs/1512.04860 （AAAI 版: https://ojs.aaai.org/index.php/AAAI/article/view/10303）
- 底本: arXiv v1 の PDF（本文 7 ページ、参考文献 1 ページ、付録 6 ページ）

Date: 2026-10-03

書き直し・原典照合: 2026-11-04（本文・付録・式・図を PDF と読み合わせ。底本は上記 arXiv v1）

表記の約束:

- 著作権に配慮し、本解説は論文の内容を自分の言葉で要約する。逐語の引用は 3 節の 1 か所（15 語未満）だけにとどめ、図表は転載せず番号とページで参照し、Table 1 は議論に要る数値だけを挙げる。原文と図表は [papers/bellemare2016_increasing_action_gap.pdf](../papers/bellemare2016_increasing_action_gap.pdf) を参照されたい。
- 章ごとの監査は未実施。paper-details の章ごとの監査役は費用のため立てず、数値・式・位置引用は筆者が PDF と読み合わせただけである。
- 論文の節には番号がないので、見出しのない導入部を 1 節、付録を 9 節として出現順に番号を振り、位置を [p.X, Section N] と書く（p.X は arXiv v1 PDF のページ）。他の文献は [著者 (年)] と書き、末尾の出典リストに対応させる。
- 数式は自分の記法（期待値 $\mathbb{E}$、指示関数 $\mathbf{1}\lbrace\cdot\rbrace$）で書き直し、記号は使う前に定義する。(n) は論文の式番号を指す。途中の式は省いた。
- 「本解説の補足」「本解説の計算」「本解説の読み取り」は本解説が加えたもので、論文の記述ではない。anet-lab の文脈は末尾の「SMDP の PRD の検討に効く点」だけで扱う。
- AAAI 版の本文との照合はしていない。書誌は arXiv の書誌ページと Crossref で確かめた。

## 概要（Abstract）

最適方策を変えずに、最適でない行動の Q 値だけを下げて行動ギャップ（最適な行動と次善の行動の価値の差、action gap）を広げる作用素を提案する論文である。ギャップが広いと近似誤差や推定誤差で貪欲方策が崩れにくい、というのが論拠になっている。まず表形式向けの一貫ベルマン作用素を示し、離散化した連続系でも良い結果を報告する。次に、作用素が最適性を保つ十分条件を導いて作用素の族を定め、Baird のアドバンテージ学習の最適性をその系として示す。最後に Atari 2600 の 60 ゲームで効果を確かめる [p.1, Abstract]。

## Table of Contents

- [概要（Abstract）](#概要abstract)
- [1. はじめに（原文に見出しなし）](#1-はじめに原文に見出しなし)
- [2. 背景（Background）](#2-背景background)
- [3. 一貫ベルマン作用素（The Consistent Bellman Operator）](#3-一貫ベルマン作用素the-consistent-bellman-operator)
  - [3.1 集約法（Aggregation Methods）](#31-集約法aggregation-methods)
  - [3.2 Q 値補間（Q-Value Interpolation）](#32-q-値補間q-value-interpolation)
  - [3.3 自転車ドメインでの実験（Experiments on the Bicycle Domain）](#33-自転車ドメインでの実験experiments-on-the-bicycle-domain)
- [4. 収束する作用素の族（A Family of Convergent Operators）](#4-収束する作用素の族a-family-of-convergent-operators)
  - [4.1 Baird のアドバンテージ学習（Baird's Advantage Learning）](#41-baird-のアドバンテージ学習bairds-advantage-learning)
  - [4.2 持続的アドバンテージ学習（Persistent Advantage Learning）](#42-持続的アドバンテージ学習persistent-advantage-learning)
  - [4.3 怠惰な作用素（The Lazy Operator）](#43-怠惰な作用素the-lazy-operator)
- [5. Atari 2600 での実験結果（Experimental Results on Atari 2600）](#5-atari-2600-での実験結果experimental-results-on-atari-2600)
- [6. 未解決の問題（Open Questions）](#6-未解決の問題open-questions)
  - [6.1 最適性のためのより弱い条件（Weaker Conditions for Optimality）](#61-最適性のためのより弱い条件weaker-conditions-for-optimality)
  - [6.2 新しい作用素の統計的効率（Statistical Efficiency of New Operators）](#62-新しい作用素の統計的効率statistical-efficiency-of-new-operators)
  - [6.3 最大限に効率的な作用素（Maximally Efficient Operator）](#63-最大限に効率的な作用素maximally-efficient-operator)
- [7. 結び（Concluding Remarks）](#7-結びconcluding-remarks)
- [8. 謝辞（Acknowledgments）](#8-謝辞acknowledgments)
- [9. 付録（Appendix）](#9-付録appendix)
  - [9.1 理論的結果（Theoretical Results）](#91-理論的結果theoretical-results)
  - [9.2 実験の詳細: 自転車ドメイン（Experimental Details: Bicycle）](#92-実験の詳細-自転車ドメインexperimental-details-bicycle)
  - [9.3 実験の詳細: ALE（Experimental Details: ALE）](#93-実験の詳細-aleexperimental-details-ale)
- [論文が述べている限界](#論文が述べている限界)
- [論文内の表記の食い違い](#論文内の表記の食い違い)
- [キューの論点への答え](#キューの論点への答え)
- [SMDP の PRD の検討に効く点](#smdp-の-prd-の検討に効く点)
- [出典リスト](#出典リスト)

## 1. はじめに（原文に見出しなし）

価値ベースの強化学習は、作用素を繰り返し当てて価値関数を改善する。価値反復 [Bellman (1957)] はベルマン作用素 $\mathcal{T}$ を $V_{k+1}=\mathcal{T}V_k$ と当て続け、ダイナミクスが未知なら Q 関数 $Q_k$ とサンプルに基づく作用素で代用する（Q 学習 [Watkins (1989)]）。不動点が最適 Q 関数 $Q^*$ で、そこから最適方策 $\pi^*$ が決まる [p.1, Section 1]。

著者らが問題にするのは、$Q^*$ の最適でない行動の成分が定常方策の価値になっていないことである。状態 $x$ で最適でない行動 $a$ の $Q^*(x,a)$ は、再び $x$ に来たら $\pi^*(x)$ に乗り換える方策の価値であり、論文はこれを非一貫性と呼ぶ。全域で一貫させるのは現実的でないので、一次の対処にあたる小さな修正として一貫ベルマン作用素を導入する [p.1, Section 1]。

この修正で最適方策の集合は保たれ、最適でない行動の値は一般に下がるので、行動ギャップが広がる。著者らはこれを誤差への備え [Farahmand (2011)] と位置づけ、時間刻みの細かい系で特に重要とみる。例に挙がるのはビデオゲーム [Togelius et al. (2009)]、[Bellemare et al. (2013)]、リアルタイム市場 [Jiang and Powell (2015)]、ロボット [Riedmiller et al. (2009)]、[Hoburg and Tedrake (2009)]、[Deisenroth and Rasmussen (2011)]、[Sutton et al. (2011)] である。同じ発想は連続時間向けのアドバンテージ学習 [Baird (1999)] や、連続 MDP の離散化解法 [Munos and Moore (1998)]、[Munos and Moore (2002)]（極限はハミルトン・ヤコビ・ベルマン方程式 [Kushner and Dupuis (2001)]）にも見られるという [p.1, Section 1]。

後半では、縮小性も最適でない行動の値の収束も要らない、弱い十分条件を導く。既存の代替作用素 [Azar et al. (2011)]、[Bertsekas and Yu (2012)] よりも標準の不動点条件から大きく離れる、と著者らは主張する。実験は自転車 [Randlov and Alstrom (1998)] と ALE [Bellemare et al. (2013)] で、後者では DQN [Mnih et al. (2015)] の学習則だけを替える [p.1–2, Section 1]。

## 2. 背景（Background）

MDP の記号は次のとおり [p.2, Section 2]。

- $\mathcal{X}$, $\mathcal{A}$: 状態空間と、有限の行動空間。
- $P(\cdot\mid x,a)$, $R(x,a)$, $\gamma\in[0,1)$: 遷移確率、有界な報酬、割引率。
- $x'$: 次の状態。$\mathbb{E}_{x'}$ は $x'\sim P(\cdot\mid x,a)$ についての期待値。
- $Q$, $\mathcal{Q}$: Q 関数と、有界な Q 関数の空間。$V(x)=\max_a Q(x,a)$ と書く。$\tilde Q$ に対する $\tilde V$ なども同じ規則で書く。

ベルマン作用素は次の形で、上限ノルムで縮小的であり [Bertsekas and Tsitsiklis (1996)]、唯一の不動点が $Q^*$ である [p.2, Section 2]。

$$(\mathcal{T}Q)(x,a)=R(x,a)+\gamma\,\mathbb{E}_{x'}\Big[\max_{b}Q(x',b)\Big] \tag{1}$$

貪欲方策の Q 関数が $Q$ 自身に一致するのは $Q=Q^*$ のときだけである。論文は (1) の形で述べるが、結果は SARSA [Rummery and Niranjan (1994)]、方策評価 [Sutton (1988)]、fitted Q-iteration [Ernst et al. (2005)] にも広がり、提案作用素はどれも Q 学習型のサンプル版をもつとする [p.2, Section 2]。

## 3. 一貫ベルマン作用素（The Consistent Bellman Operator）

出発点は Figure 1 の 2 状態の例である [p.2, Figure 1]。状態 $x_1$ では、報酬 0 で留まる $a_2$（ケーキを食べない）か、報酬 1 を得て確率 1/2 で吸収状態 $x_2$ へ移る $a_1$（食べる）を選ぶ。$x_2$ の価値は正の定数 $\epsilon$ を使って $-2(1+\epsilon)/\gamma$ に置いてある。計算すると食べないのが最適で、$V^*(x_1)=0$、$Q^*(x_1,a_1)=-\epsilon$、ギャップは $\epsilon$ になる ((2))。ところが食べ続ける定常方策の価値は $-\epsilon/(1-\gamma/2)$ ((3)) で、$-\epsilon$ より低い。$Q^*(x_1,a_1)$ は「1 回だけ食べてあとは控える」非定常な方策の値であり、ベルマン作用素の出力が定常で決定的な方策の集合 $\Pi$ に収まらないことを示している [p.2, Section 3]。

この例では $\epsilon$ 程度の推定誤差で最悪の方策が貪欲になりうる [p.3, Section 3]。戻るたびに同じ行動をとるとした Q 関数 (4) ならギャップは $\epsilon/(1-\gamma/2)$ に広がるが、そこから有用な作用素は作れそうにない。そこで定常性を、1 ステップ先で同じ状態に戻る場合に限って入れる [p.3, Section 3]。

$$(\mathcal{T}_{\mathrm{C}}Q)(x,a)=R(x,a)+\gamma\,\mathbb{E}_{x'}\Big[\mathbf{1}\lbrace x'\neq x\rbrace\max_{b}Q(x',b)+\mathbf{1}\lbrace x'=x\rbrace\,Q(x,a)\Big] \tag{5}$$

次の状態が違えば (1) のまま、同じなら $\max$ の代わりに今の行動 $a$ の値を使う。著者らはこの変更の意味を次のように言う。

```
Effectively, our operator redefines the meaning of Q-values

実質的に、我々の作用素は Q 値の意味を定義し直す
```
[p.3, Section 3]

例の Q 値は、$x_2$ に移るまで食べ続けたときの期待収益に変わる [p.3, Section 3]。作用素 $\mathcal{T}'$ の良さは次の 2 つの性質で定義する。列は任意の $Q_0$ から $Q_{k+1}=\mathcal{T}'Q_k$ で作り、$V_k(x)=\max_b Q_k(x,b)$ とする。

- 最適性保存（定義 1）: $V_k(x)$ が $V^*(x)$ に収束し、$Q^*(x,a)<V^*(x)$ の行動は $\limsup_k Q_k(x,a)<V^*(x)$ のまま残る [p.3, Section 3, Definition 1]。
- ギャップ増大（定義 2）: 次の (6) が成り立つ。少なくとも 1 組の $(x,a)$ で狭義なら、狭義にギャップ増大的と呼ぶ [p.3, Section 3, Definition 2]。

$$\liminf_{k\to\infty}\big[V_k(x)-Q_k(x,a)\big]\;\geq\;V^*(x)-Q^*(x,a) \tag{6}$$

一貫ベルマン作用素は最適性保存的で、遷移グラフに直接のループがあるとギャップ増大的でもある。例では極限 $\tilde Q(x_1,a_1)$ が $-\epsilon/(1-\gamma/2)$ となり、誤差があっても順位が保たれやすい。論文はこれをギャップ拡大の第一の利点（推定誤差への頑健さ）とする [p.3, Section 3]。

### 3.1 集約法（Aggregation Methods）

(5) の指示関数は、自己遷移の確率がほぼ 0 の系や、特徴量でしか状態を見られず同一性を判定できない系では使いにくい。そこで論文は、表形式に近い性質をもつ集約スキーム [Bertsekas (2011)] なら作用素が矛盾なく定義できることを示す [p.3, Section 3.1]。

- $\mathcal{Z}$: 集約状態の集合（多重線形補間なら格子点）。
- $A(\cdot\mid x)$: 状態を集約状態の分布へ写す写像（補間の重み）。行動空間 $\mathcal{A}$ とは別。
- $D(\cdot\mid z)$: 集約状態を状態の分布へ写す写像。

$$(\mathcal{T}_{\mathrm{C}}Q)(z,a)=\mathbb{E}_{x\sim D(\cdot\mid z)}\Big[R(x,a)+\gamma\,\mathbb{E}_{x'}\mathbb{E}_{z'\sim A(\cdot\mid x')}\big[\mathbf{1}\lbrace z'\neq z\rbrace\max_{b}Q(z',b)+\mathbf{1}\lbrace z'=z\rbrace\,Q(z,a)\big]\Big] \tag{8}$$

内側を $\max_b Q(z',b)$ だけにしたものが集約ベルマン作用素 (7) である。$\mathcal{Z}$ が $\mathcal{X}$ の有限部分集合で $D$ が恒等写像なら、平均化器 [Gordon (1995)] やカーネル法 [Ormoneit and Sen (2002)] を含み、$A$ も恒等で状態が有限なら表形式 [Sutton and Barto (1998)] に戻る。$D$ と $A$ は集約状態の上に新しい MDP を誘導するので、(5) と (8) の違いは遷移核の読み方だけであり、$P$ が決定的な連続系に補間を使う場合 [Munos and Moore (1998)] でも意味をもつ [p.3, Section 3.1]。Figure 2 は 2 次元格子の多重線形補間の図で、セル内の点の値を四隅の格子点の重み付き平均で表す [p.4, Figure 2]。

### 3.2 Q 値補間（Q-Value Interpolation）

格子点以外の Q 値を (9) のように定めるにはモデルか $D$ の逆が要る。そこで格子点の Q 値をそのまま補間する $Q(x,a)=\mathbb{E}_{z\sim A(\cdot\mid x)}Q(z,a)$ を採り、次の状態をこの補間値で評価する作用素を $\mathcal{T}_{\mathrm{QVI}}$ とする。$\max$ の凸性から $\mathcal{T}_{\mathrm{QVI}}Q\leq\mathcal{T}Q$ である [p.4, Section 3.2]。一貫性を素直に入れると、$z$ に戻る重み $A(z\mid x')$ の分だけ差を引く形になる [p.4, Section 3.2]。

$$(\mathcal{T}'_{\mathrm{QVI}}Q)(z,a)=\mathbb{E}_{x\sim D(\cdot\mid z)}\Big[R(x,a)+\gamma\,\mathbb{E}_{x'}\max_{b}\Big(Q(x',b)-A(z\mid x')\big[Q(z,b)-Q(z,a)\big]\Big)\Big]$$

これは縮小写像だが $\mathcal{T}_{\mathrm{QVI}}$ 以下である保証がなく、最適性を保たない。そこで小さいほうをとる $\mathcal{T}_{\mathrm{CQVI}}Q=\min\lbrace\mathcal{T}_{\mathrm{QVI}}Q,\mathcal{T}'_{\mathrm{QVI}}Q\rbrace$ (10) を定義する。定理 1 の系として、これは最適性保存的かつギャップ増大的である [p.4, Section 3.2]。

### 3.3 自転車ドメインでの実験（Experiments on the Bicycle Domain）

- 課題: 自転車を倒さず、出発点の 1 km 北のゴールまで走らせる [Randlov and Alstrom (1998)]。1 ステップは 0.01 秒で、成功エピソードは 5 万ステップ以上になる。
- 表現: 状態は 6 次元の特徴で、4 つが角度と角速度、2 つがゴールに対する極座標 $(\psi,d)$ である。$10\times\cdots\times10$ の格子の多重線形補間と $\mathcal{T}_{\mathrm{CQVI}}$ を使う。
- 手順: 作用素の質だけを見るため、軌跡型の学習ではなく、全格子点へ同時に作用素を当てる価値反復を使う。次状態の期待値はサンプルで推定する [p.4, Section 3.3]。詳細は 9.2 節。

極限の価値は格子点で $V^*$ と一致するが、それ以外の状態では大きくずれることがあり、一般には一致しない。この問題の軌跡はほとんど格子点から離れた所を通る。$\psi$ の 1 セルは $\pi/5$ の幅で、1 ステップの変化はふつう $\pi/1000$ 未満である [p.4, Section 3.3]。

Figure 3 の上段は反復回数に対する転倒率とゴール到達率、下段は 100〜1000 反復の方策の軌跡である。一貫作用素の方策は早く転倒しなくなり、早くゴールに着く。ベルマン作用素の方策は初めゴールの周りを回り続ける [p.4, Section 3.3]、[p.5, Figure 3]。ゴール到達率が初めて 0.8 を超えるのは、一貫作用素が約 700 反復、ベルマン作用素が 3700〜3800 反復である（本解説の読み取り。図のベクターデータから求めた）。$8\times\cdots\times8$ の格子では差がさらに開く。著者らは、格子点で最適でない Q 値を下げたことでセル内の方策が良くなったと解釈し、ギャップと方策の質を結ぶ理論 [Farahmand (2011)] と整合するとして、第二の利点（Q 値補間から得る方策の改善）に数える [p.4, Section 3.3]。

## 4. 収束する作用素の族（A Family of Convergent Operators）

確率的な解釈をもたない近似、例えば線形近似 [Sutton (1996)]、局所重み付き回帰 [Atkeson (1991)]、ニューラルネットワーク [Tesauro (1995)]、情報理論的手法 [Veness et al. (2015)] にも広げられるか、が本節の問いで、論文は肯定する。以下の作用素は一般に縮小写像ではないが、ギャップ増大的で、Q 関数を厳密に表せれば最適性保存的である [p.5, Section 4]。

定理 1: (1) の $\mathcal{T}$ について、ある $\alpha\in[0,1)$ があり、すべての $Q,x,a$ で次を満たす $\mathcal{T}'$ は、最適性保存的かつギャップ増大的である。右の不等式が論文の条件 1、左が条件 2 にあたる [p.5, Section 4, Theorem 1]。

$$\mathcal{T}Q(x,a)-\alpha\big[V(x)-Q(x,a)\big]\;\leq\;\mathcal{T}'Q(x,a)\;\leq\;\mathcal{T}Q(x,a)$$

各反復で下げてよい量の上限は、現在の $Q_k$ の行動ギャップに届かない範囲である。途中で最適行動が貪欲行動より低く評価されることはあっても際限なくは続かず、最適行動 $a^*$ の $Q_k(x,a^*)$ は最後に $V^*(x)$ に届く。最適行動が複数あれば、保証されるのはそのうち 1 つである [p.5, Section 4]、[p.5, footnote 3]。著者らの知る限り、縮小性の議論に頼らない初めての収束の結果で、不動点の存在も仮定しない [p.5, Section 4]。

注意 1 は条件の必要性に近い主張で、$\mathcal{T}'Q>\mathcal{T}Q$ を許す作用素、または $\alpha=1$ の下界さえ割る作用素には、$\max_a$ の極限が $V^*$ にならない単一状態の例があるとする。$\alpha=1$ の等号の場合は未解決である [p.5, Section 4, Remark 1]。系 1 は $\mathcal{T}_{\mathrm{C}}$ (8) と $\mathcal{T}_{\mathrm{CQVI}}$ (10) の最適性保存を述べ、論文は $\mathcal{T}_{\mathrm{C}}$ が縮小写像でもあること、両者がギャップ増大的であることを添える [p.5, Section 4, Corollary 1]。

### 4.1 Baird のアドバンテージ学習（Baird's Advantage Learning）

アドバンテージ学習（A.L.）は、連続時間の問題に残差アルゴリズムを使う文脈で提案された [Baird (1999)]。時定数 $\Delta_t>0$、定数 $C>0$、$K=C\Delta_t$ を使って次のように書ける [p.5–6, Section 4.1]。

$$\mathcal{T}'Q(x,a)=\frac{1}{K}\Big[R(x,a)+\gamma^{\Delta_t}\,\mathbb{E}_{x'}V(x')+(K-1)V(x)\Big]$$

論文は $\Delta_t=1$、$\alpha=1-K$ とおき、同じ不動点をもつ次の形に書き直す。$1/K$ を掛けない分だけ実用上安定だろう、と著者らはみる [p.6, Section 4.1]。

$$\mathcal{T}_{\mathrm{AL}}Q(x,a)=\mathcal{T}Q(x,a)-\alpha\big[V(x)-Q(x,a)\big]$$

$\alpha\in[0,1)$ なら価値は一意に $V^*$ へ収束する [p.6, Section 4.1, Corollary 2]。$V$ と $A=Q-V$ を別々に持つアドバンテージ更新とは別物である [p.5, footnote 4]。

一貫ベルマン作用素との関係は次の書き換えで分かる。自己遷移の確率 $P(x\mid x,a)$ を使うと (5) は次の形になり、$\alpha$ を $(x,a)$ ごとに $\gamma P(x\mid x,a)$ としたアドバンテージ学習とみなせる [p.6, Section 4.1]。

$$\mathcal{T}_{\mathrm{C}}Q(x,a)=\mathcal{T}Q(x,a)-\gamma P(x\mid x,a)\big[V(x)-Q(x,a)\big]$$

### 4.2 持続的アドバンテージ学習（Persistent Advantage Learning）

時間分解能の高い問題では行動をあまり切り替えない方策が有利なことがあるとして、同じ行動の継続を優遇する作用素（P.A.L.）を定義する [p.6, Section 4.2]。

$$\mathcal{T}_{\mathrm{PAL}}Q(x,a)=\max\Big\lbrace\mathcal{T}_{\mathrm{AL}}Q(x,a),\ R(x,a)+\gamma\,\mathbb{E}_{x'}Q(x',a)\Big\rbrace$$

第 2 項は次の状態でも $a$ を続けたときの値で、$\mathcal{T}Q(x,a)-\gamma\,\mathbb{E}_{x'}[V(x')-Q(x',a)]$ と同じである [p.6, Section 4.2]。

### 4.3 怠惰な作用素（The Lazy Operator）

$\alpha\in[0,1)$ について、更新しても貪欲方策が変わらない範囲（$Q\leq\mathcal{T}Q\leq\alpha V+(1-\alpha)Q$）では Q 値を据え置き、それ以外は $\mathcal{T}Q$ にする作用素も定理 1 を満たす。不動点は多数ありうる。$\alpha=1$ は定理の範囲外だが最適性は保つ。ただし保証されるのは最適行動の 1 つだけである [p.6, Section 4.3]。

## 5. Atari 2600 での実験結果（Experimental Results on Atari 2600）

ALE [Bellemare et al. (2013)] は 1 フレーム 1/60 秒で、行動はふつう 4 フレームごとに選ぶ。個々の行動の影響が小さい点で自転車に近い、と論文はみる。DQN [Mnih et al. (2015)] は $\epsilon$-greedy で行動し、経験再生のサンプルで TD 誤差の二乗を減らす。論文はこの誤差の式だけを替える。観測した遷移 $(x,a,x')$ について次のとおりで、$\Delta_{\mathrm{PAL}}$ の第 2 項の係数は $\mathcal{T}_{\mathrm{PAL}}$ の $\gamma$ ではなく $\alpha$ にしてある [p.6, Section 5]。

$$\Delta Q=R(x,a)+\gamma V(x')-Q(x,a)$$

$$\Delta_{\mathrm{AL}}Q=\Delta Q-\alpha\big[V(x)-Q(x,a)\big]$$

$$\Delta_{\mathrm{PAL}}Q=\max\Big\lbrace\Delta_{\mathrm{AL}}Q,\ \Delta Q-\alpha\big[V(x')-Q(x',a)\big]\Big\rbrace$$

本解説の補足: どちらの式にも同じ状態かどうかの判定はない。$\Delta_{\mathrm{PAL}}$ は、今の状態のギャップと次の状態のギャップのうち、補正が小さくなるほうを選ぶ。

1 つ目の実験は Stochastic Minimal 設定である。ゲームごとの最小の行動集合を使い、死亡の情報を与えず、フレームごとに確率 $p=0.25$ で行動を退けて直前のフレームの行動を繰り返す。反応の遅れに近い操作で、DQN の性能も下がる難しい設定である。DQN、A.L.、P.A.L. を 1 億フレーム学習させ、$\alpha$ は 5 つの学習用ゲームで調整し、他の 55 ゲームで各 10 試行のテストをした。ゲームごとに学習後の評価スコアを対応のある t 検定（99% 信頼区間）で比べた [p.6, Section 5]、[p.11, Section 9.3]。

2 つ目は Original DQN 設定である。[Mnih et al. (2015)] に合わせて確率的な操作を使わず、ライフの喪失を終了として扱い、3 試行の平均で 2 億フレーム学習した [p.6, Section 5]、[p.11, Section 9.3]。Figure 4 は ASTERIX と SPACE INVADERS の学習曲線で、著者らはこれを例外でなく典型とする [p.6, Section 5]、[p.7, Figure 4]。ASTERIX では DQN が後半に大きく下がるのに提案作用素は下がらず、SPACE INVADERS では A.L. と P.A.L. がほぼ重なって DQN を上回る（本解説の読み取り）。

結果の表の見方は次のとおり。論文は通常のベルマン更新のエージェントを DQN とも Bellman とも呼ぶ（Table 1 の列名は Bellman）。

- 統計的に上回った・下回ったゲーム数: Stochastic Minimal の 60 ゲームでの t 検定の結果。
- スコア改善の中央値・平均: DQN に対する改善率をゲームを通して集計した値。Original DQN 設定の段落に書かれているので、その設定の値と読んだ（本解説の読み取り）。算出方法（どの時点のどのスコアを比べたか）は論文に書かれていない。
- Times Best: Table 1 で各列が最高値になったゲーム数。

| 設定 | 指標 | DQN | A.L. | P.A.L. | 位置 |
|---|---|---|---|---|---|
| Stochastic Minimal | DQN を統計的に上回ったゲーム数 / 60 | − | 37 | 35 | [p.6, Section 5] |
| Stochastic Minimal | DQN を統計的に下回ったゲーム数 / 60 | − | 1 | 1 | [p.6, Section 5] |
| Original DQN | スコア改善の中央値 | 基準 | 8.4% | 9.1% | [p.6, Section 5] |
| Original DQN | スコア改善の平均 | 基準 | 27.0% | 32.5% | [p.6, Section 5] |
| Original DQN | Times Best | 12 | 21 | 31 | [p.12, Table 1] |

下回ったゲームとして論文は ATLANTIS と JAMES BOND を挙げている。P.A.L. と A.L. の直接の比較では、P.A.L. が 16 ゲームで上回り 6 ゲームで下回った [p.6, Section 5]。

3 つ目は過大評価との関係を見る実験である。DQN の低迷の原因をベルマン更新の Q 関数の不安定さに求めた [van Hasselt et al. (2016)] にならい、学習済み DQN が SPACE INVADERS（Original DQN 設定）を $\epsilon=0.05$ でプレイした 1 エピソードの軌跡の上で、各作用素の Q 関数の $V(x)$ と行動ギャップを計算した。ギャップは長さ 50 の移動窓で平均している [p.6–7, Section 5]、[p.11, Section 9.3]、[p.13, Section 9.3]。Figure 5 では提案作用素のギャップがはっきり広く、価値の推定は低い [p.7, Figure 5]。著者らは低い推定のほうが現実的だとし、Double Q-learning [van Hasselt (2010)] でも同じ傾向が報告されていること、$\max$ 由来の正のバイアスへの対策 [Azar et al. (2011)]、[Lee et al. (2013)] の文脈にあることを挙げる。Q 値どうしが近いほどバイアスが強いので、ギャップの拡大はバイアスの低減と整合する。これが第三の利点になるという仮説である [p.7, Section 5]。

## 6. 未解決の問題（Open Questions）

### 6.1 最適性のためのより弱い条件（Weaker Conditions for Optimality）

本当に要るのは反復の極限 $\tilde Q$ の貪欲行動が最適であることだけで、最適価値の保存すら要らないはずである。定理 1 より弱い条件がありうるが、今の証明技法は届かないという [p.7, Section 6.1]。

### 6.2 新しい作用素の統計的効率（Statistical Efficiency of New Operators）

書き直したアドバンテージ学習は、遷移確率が未知か無関係なときの一貫ベルマン作用素の一般化とみなせる。そこからアドバンテージ学習に確率的な解釈があるか、最適行動の取り違えの確率で測ると一貫ベルマン作用素が通常の作用素に劣る場合があるか、を問う [p.7, Section 6.2]。

### 6.3 最大限に効率的な作用素（Maximally Efficient Operator）

どの最適性保存作用素を選ぶべきか、ある MDP に最も効率のよい作用素はあるか、価値の推定と同時にそれを探せるか、を問う [p.7, Section 6.3]。

## 7. 結び（Concluding Remarks）

総括は、ギャップが近似価値関数の貪欲方策の性能を大きく左右し、ベルマン作用素の小さな修正で性能が大きく伸びる、というものである。あわせて、古典的な Q 関数は実用の方策を得るには心もとないこと、方策と価値の伝統的な関係や価値の概念そのものを見直す余地があることを指摘する [p.7, Section 7]。

## 8. 謝辞（Acknowledgments）

議論に協力した Michael Bowling、Csaba Szepesvári、Craig Boutilier、Dale Schuurmans、Marty Zinkevich、Lihong Li、Thomas Degris、Joseph Modayil と匿名の査読者に謝意を述べている [p.7, Section 8]。

## 9. 付録（Appendix）

証明、自転車の詳細、ALE の詳細の 3 部からなる [p.9, Section 9]。主張と筋道、設定だけをまとめた。証明の各行と表の数値は PDF を参照されたい。

### 9.1 理論的結果（Theoretical Results）

補題 1: $\mathcal{T}'$ が (i) $\mathcal{T}'Q\leq\mathcal{T}Q$ と (ii) 貪欲行動では $\mathcal{T}'Q=\mathcal{T}Q$ を満たすなら、$V_k$ は収束し、極限は $V^*$ 以下である。(i) から $\limsup_k Q_k\leq Q^*$ が出る。(ii) を各反復の貪欲行動に使うと $V_{k+1}-V_k$ の下側が $\gamma^k$ で縮む形 (11) になり、後の下げ幅が幾何級数の裾で抑えられて、上極限と下極限が一致する [p.9, Section 9.1, Lemma 1]。

補題 2: 同じ条件で、$\lvert V_k\rvert$ は $V_0$ と $R$ の大きさと $1/(1-\gamma)$ で一様に抑えられる (12) [p.9, Section 9.1, Lemma 2]。

定理 2 は定理 1 の結論を式で明示した版である [p.9–10, Section 9.1, Theorem 2]。証明の筋道は次のとおり。$\tilde Q=\limsup_k Q_k$ とおくと、条件 1 と $\max$ と $\limsup$ の可換性から $\tilde Q\leq\mathcal{T}\tilde Q$ が出る。条件 2 の下界の上極限と補題 2（優収束定理）から $\tilde V\geq\max_a\mathcal{T}\tilde Q$ が出るので、$\tilde V$ は状態価値のベルマン方程式を満たし、$V^*$ に等しい。条件 1 から $\limsup_k Q_k(x,a)\leq Q^*(x,a)$ なので、最適でない行動は最適でないまま残り、ギャップ増大も (18) の形で同じことから従う [p.10, Section 9.1]。

系 3 は 2 つの一貫作用素が両方の性質をもつと述べる [p.10, Section 9.1, Corollary 3]。付録に証明があるのは補題 1・2 と定理 2 だけで、系 1〜3 の証明と注意 1 の反例の構成は載っていない（本解説の補足）。

### 9.2 実験の詳細: 自転車ドメイン（Experimental Details: Bicycle）

シミュレータの元の報酬はゴールから遠ざかる走行にも報酬が出て価値反復に向かないので、転倒で負の報酬、ゴールで 1.0、途中は $\psi$ の関数の小さい値という報酬に置き換えた（式は p.11）。割引率は 0.99 で、似た報酬でも定性的な結果は変わらないという [p.10–11, Section 9.2]。

作用素は集約状態上のサンプル版で、価値反復はステップサイズ $\eta$ の平均化形 $Q_{k+1}=(1-\eta)Q_k+\eta\,\mathcal{T}'Q_k$ を全 $z,a$ に同時に当てる。収束が速く、次状態のサンプルが 1 つで済むためで、スイープの結果 $\eta=0.1$ とした。格子は 6 変数の有界な範囲に置き、範囲外は端の値に丸める（範囲は p.11）[p.11, Section 9.2]。

Figure 6 はベルマン、一貫、A.L.、P.A.L.（ともに $\alpha=0.1$）を $8\times\cdots\times8$ と $10\times\cdots\times10$ の格子で比べる [p.11, Figure 6]。通常のベルマン作用素はゴールに至る解を見つけられず、一貫作用素は見つける。A.L. と P.A.L. もベルマンより良いが、小さい格子ではやや不安定である [p.11, Section 9.2]。図を見ると、解を見つけられないのは $8\times\cdots\times8$ の格子で、$10\times\cdots\times10$ では反復を重ねると到達する（本解説の読み取り）。

### 9.3 実験の詳細: ALE（Experimental Details: ALE）

DQN の構造は [Mnih et al. (2015)] に従う。フレームは ALE の 1 ステップ、タイムステップはひとまとめに扱う 4 フレームである。$p=0.25$ は ALE 0.5 の標準値である [p.11, Section 9.3]。60 ゲームのうち 5 ゲームは [Bellemare et al. (2013)] の表にないもので、Table 1 に † が付く [p.13, Section 9.3]。

Table 1 は 60 ゲームについて各作用素の最高スコアを並べた表で、ゲームごとの最良値を強調し、最下行に Times Best を置く [p.12, Table 1]。キャプションに設定の記載はなく、付録の Original DQN 設定の段落で参照されているので、その設定の結果と読んだ（本解説の読み取り）[p.11, Section 9.3]。

#### 9.3.1 DQN の実装詳細（DQN Implementation Details）

DQN は毎タイムステップ更新する方策ネットワークと、1 万タイムステップごとに更新するターゲットネットワークを持つ [Mnih et al. (2015)]。提案作用素では補正項も含めて誤差をターゲットネットワークで計算した。補正項を方策ネットワークで計算すると性能が下がったという [p.13, Section 9.3.1]。

#### 9.3.2 パラメータ選択（Parameter Selection）

$\alpha$ は ASTERIX、BEAM RIDER、PONG、SEAQUEST、SPACE INVADERS の 5 ゲームで、$\lbrace 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1\rbrace$ を Original DQN 設定で学習させ、最高スコアを見て手で選んだ（$\alpha=0$ は DQN に一致）。選んだ値は本文にも付録にも書かれていない [p.13, Section 9.3.2]。

Figure 7 は $\alpha$ に対する性能の図で、DQN を 1.0 とした相対値と読める。A.L. の山は ASTERIX、BEAM RIDER、SEAQUEST で $\alpha=0.9$、SPACE INVADERS で $0.7$ にあり、PONG はほとんど動かない。$\alpha=1$ の A.L. は 5 ゲームとも 1.0 を下回る（以上、本解説の読み取り）。キャプションは $\alpha=1$ が定理の条件外であること、SEAQUEST の P.A.L. の $\alpha=0.9$ の不自然な値を統計的な問題とみていることを断っている [p.14, Figure 7]。

## 論文が述べている限界

この節は本解説が追加したもので、論文自身が認めている限界を言い換えて集める。

- 定理 1 が最適な貪欲方策を保証するのは、Q 関数を厳密に表現できる場合に限られる [p.5, Section 4]。
- 最適行動が複数あるとき、正しく評価されることが保証されるのは 1 つだけである [p.5, footnote 3]、[p.6, Section 4.3]。
- $\alpha=1$ の等号の場合に発散例があるかは未解決である [p.5, Section 4, Remark 1]。定理 1 より弱い条件へは今の証明技法では広げられないようだという [p.7, Section 6.1]。
- 統計的効率と、どの作用素を選ぶべきかは未解決である [p.7, Section 6.2]、[p.7, Section 6.3]。
- 価値の推定が低くなったのをバイアス低減の結果とみる説明は仮説である [p.7, Section 5]。
- Stochastic Minimal では A.L. と P.A.L. がそれぞれ 1 ゲームで DQN を統計的に下回った [p.6, Section 5]。
- 自転車では、A.L. と P.A.L. は小さい格子でやや不安定に見える [p.11, Section 9.2]。
- Atari では、補正項を方策ネットワークで計算すると性能が落ちた [p.13, Section 9.3.1]。

## 論文内の表記の食い違い

この節も本解説が追加したもので、論文内の照合で見つけた不一致を挙げる。著者の意図は確かめていない。本解説の表記を片方に合わせた箇所はそう書く。最後の項目は表記の食い違いではなく、本解説の再計算の結果である。

- 定義 1 の条件式の左側は $V^*(x,a)$ と印字され、付録の定理 2 の対応する式では $V^*(x)$ である [p.3, Section 3, Definition 1]、[p.10, Section 9.1, Theorem 2]。本解説は後者に合わせた。
- 4 節は一貫ベルマン作用素を式番号 (7) で参照するが、(7) は集約ベルマン作用素で、一貫ベルマン作用素は (5) と (8) である [p.5, Section 4]、[p.3, Section 3.1]。
- 系 3 は $\mathcal{T}_{\mathrm{CQVI}}$ に (9) を添えるが、この作用素の式番号は (10) である。$\mathcal{T}_{\mathrm{C}}$ にも系 3 は (5) を、系 1 は (8) を添えている [p.10, Section 9.1, Corollary 3]、[p.4, Section 3.2]、[p.5, Section 4, Corollary 1]。
- 付録は行動ギャップの実験を Figure 4 を作るためと書くが、その結果は Figure 5 にある [p.13, Section 9.3]、[p.7, Figure 5]。
- Table 1 のキャプションは本文の Section 4 を参照するが、arXiv 版の本文に節番号はない。見出しのある節を数えると Atari の節が 4 番目になる [p.12, Table 1]。
- 付録のサンプル版 $\mathcal{T}'_{\mathrm{QVI}}$ では、和の中の次状態に添字 $i$ がない [p.11, Section 9.2]。
- 本文は $\alpha$ の最適化を Stochastic Minimal の段落で述べ、付録は Original DQN 設定で選んだと書く [p.6, Section 5]、[p.13, Section 9.3.2]。
- 本文の改善率（中央値 8.4%・9.1%、平均 27.0%・32.5%）は、Table 1 の値からの単純な比では再現しない。基本の計算は「キューの論点への答え」の 3 に置いた。$S$ を作用素のスコア、$B$ を Bellman のスコアとして、調整用の 5 ゲームや † の 5 ゲームを除いても、分母を $\max(\lvert S\rvert,\lvert B\rvert)$ や $\lvert B\rvert+1$ に変えても、中央値は A.L. 12.1% 以上、P.A.L. 14.0% 以上で本文の値に届かない。符号をそのまま使う $(S-B)/B$ では平均が 28.0%・33.7% と本文に近づくが、中央値は 12.1%・14.0% で合わない。

## キューの論点への答え

reports/INDEX.md のキューの問いに、論文の記述の範囲で答える。論文が扱っていない点はそう明記する。

1. 一貫ベルマン作用素と advantage learning の効き方
   - 一貫ベルマン作用素: 次の状態が今と同じなら、$\max$ の代わりに今の行動の値を使う ((5))。これは $\mathcal{T}Q$ から $\gamma P(x\mid x,a)[V(x)-Q(x,a)]$ を引く形に書ける [p.3, Section 3]、[p.6, Section 4.1]。貪欲行動では引く量が 0 なので価値 $V$ は動かず、貪欲でない行動だけが下がる（本解説の補足）。自己遷移がなければ (1) と同じで、ギャップが広がるのは直接のループがある場合である [p.3, Section 3]。
   - advantage learning: 同じ差し引きを定数 $\alpha$ で全遷移に当てる [p.6, Section 4.1]。$\alpha<1$ なら価値は $V^*$ に収束する [p.6, Section 4.1, Corollary 2]。
   - 大きさ（本解説の計算。論文にこの式はない）: $V_k\to V^*$ のもとで反復 $Q_{k+1}(x,a)=\mathcal{T}Q_k(x,a)-\alpha[V_k(x)-Q_k(x,a)]$ の極限をとると、次の形になる。一貫ベルマン作用素は $\alpha=\gamma P(x\mid x,a)$ を入れた形で、2 状態の例の $\epsilon\to\epsilon/(1-\gamma/2)$ と一致する。関数近似の誤差がない場合の値である。

     $$V^*(x)-\tilde Q(x,a)=\frac{V^*(x)-Q^*(x,a)}{1-\alpha}$$

   - 効用: 論文が挙げる利点は、推定誤差への頑健さ [p.3, Section 3]、補間で得る方策の改善 [p.4, Section 3.3]、過大評価バイアスの低減（仮説）[p.7, Section 5] の 3 つである。
   - 上限: $\alpha=1$ は定理の外で、Figure 7 の A.L. は 5 ゲームとも DQN を下回る（本解説の読み取り）[p.14, Figure 7]。
   - 論文が扱っていない点: Atari の改善が 3 つの利点のどれによるかは切り分けていない。定理は Q 関数を厳密に表せる場合の保証で、DQN の挙動を保証するものではない [p.5, Section 4]。
2. 関数近似と Atari で「同じ状態」をどう扱ったか
   - 表形式: 恒等の判定 $x'=x$ をそのまま使う [p.3, Section 3]。
   - 集約・格子: 判定を集約状態で行う（$z'=z$、(8)）[p.3, Section 3.1]。Q 値補間では 0/1 の判定でなく、$z$ に戻る補間の重み $A(z\mid x')$ の分だけ差し引く [p.4, Section 3.2]。$P$ が決定的でも意味をもつのはこのためである（本解説の補足）[p.3, Section 3.1]。
   - 一般の関数近似: 判定をやめる。指示関数が使えない場合があることを出発点に [p.3, Section 3.1]、$\gamma P(x\mid x,a)$ を定数 $\alpha$ に置き換えたアドバンテージ学習を、$P$ が未知か無関係なときの一般化と位置づける [p.6, Section 4.1]、[p.7, Section 6.2]。
   - Atari: 提案作用素は $\Delta_{\mathrm{AL}}$ と $\Delta_{\mathrm{PAL}}$ で、状態の一致は調べず、すべての遷移に同じ $\alpha$ の補正を掛ける [p.6, Section 5]。補正項はターゲットネットワークで計算する [p.13, Section 9.3.1]。
   - 論文が扱っていない点: 一貫ベルマン作用素そのものは Atari で走らせていない。比較は DQN、A.L.、P.A.L. の 3 つである [p.6, Section 5]。ハッシュや距離による近似的な同一判定も出てこない。
3. Atari での改善幅
   - Stochastic Minimal（1 億フレーム、各 10 試行）: DQN を統計的に上回ったのは 60 ゲーム中 A.L. が 37、P.A.L. が 35 で、下回ったのはそれぞれ 1 である。P.A.L. は A.L. に 16 ゲームで勝ち 6 ゲームで負ける。改善率は示されていない [p.6, Section 5]。
   - Original DQN（2 億フレーム、3 試行）: DQN に対するスコア改善の中央値が A.L. 8.4%、P.A.L. 9.1%、平均が A.L. 27.0%、P.A.L. 32.5% である [p.6, Section 5]。これらと Table 1 をこの設定の結果とするのは、段落の位置からの読み取りである。Table 1 の Times Best は DQN 12、A.L. 21、P.A.L. 31 で、PITFALL と TENNIS は 3 列とも同点なので重ねて数えた形になる（合計 64）[p.12, Table 1]。
   - 本解説の計算: Table 1 の値で $(S_{\mathrm{op}}-S_{\mathrm{B}})/\lvert S_{\mathrm{B}}\rvert$（$S_{\mathrm{op}}$ は作用素のスコア、$S_{\mathrm{B}}$ は Bellman のスコア）をとると、Bellman が 0 の 2 ゲームを除いても 0 として含めても、中央値は A.L. 12.7〜15.0%、P.A.L. 22.3〜23.4%、平均は A.L. 32.6〜33.7%、P.A.L. 41.5〜42.9% で、本文の値と一致しない。Table 1 は最高スコアの表なので、本文の改善率は別の集計だと考えられるが、算出方法は書かれていない。同じ計算で Bellman を上回るのは A.L. 45 ゲーム、P.A.L. 42 ゲームである。平均は一部のゲームに引っ張られる（A.L. の PRIVATE EYE は約 5.5 倍、P.A.L. の WIZARD OF WOR は約 4.3 倍）。
   - 留意点: 60 ゲームには $\alpha$ の調整に使った 5 ゲームが含まれる [p.6, Section 5]。選んだ $\alpha$ の値は書かれていない [p.13, Section 9.3.2]。Stochastic Minimal のゲーム別スコアの表はない。

## SMDP の PRD の検討に効く点

対象は [docs/memo/956_smdp_transition_duration_10prd.md](../docs/memo/956_smdp_transition_duration_10prd.md) である（PRD114 で 999 から改番）。論文は SMDP も DropMerge も扱っていないので、この節は論文の記述ではなく本解説の推論である。§ は PRD の節を指す。

1. 役割が違う。経過時間 $\tau$ は目的（何が最適か）を変えて、待つ値段と計画の見通しを分ける。行動ギャップを広げる作用素は最適方策を変えず（厳密な表現の場合、定理 1）、順位の逆転を減らすだけである。§1 の「本来の差 $(1-\gamma)V$ が揺れで入れ替わる」には直接効くが、§4 のストーリー 1 は満たせない。代わりに Env と replay の契約を変えず、$\tau$ の実装を待たずに比べられ、return の物差し（§5 の 6）も変わらない。
2. 効く大きさは $1/(1-\alpha)$ 倍である（「キューの論点への答え」の 1）。PRD の差 0.3% は $\alpha=0.9$ で 3%、$\alpha=0.99$ で 30% になる。厳密な自己遷移に一貫ベルマン作用素を当てると差は $V$ 全体に広がる。
3. 既存の Munchausen と同じ仕組みである。Munchausen のボーナスの差は clip にかからない範囲で $\alpha\times$（Q の差）で（[PRD067](../docs/memo/done/067_MunchausenRL_10prd.md) の「方策温度の効き方」）、A.L. の補正と同じ形をしている。§2 の Run は [$\alpha=0.9$](../docs/experiments/default-dqn/dropmerge/config/run_20261003-081944_dmfb_qr_munch.txt) で、健全な時期に差が約 5 倍（極限の 10 倍の内側）になったのに、NEET 期には順位が逆転した。差を広げるだけでは NEET を防げなかった実例として読める。したがって A.L. を新機構として足す前に、この結果の理由を見るほうが先である。ただし同じなのはボーナスの差の形までで、温度 $\tau_{\text{m}}>0$ の Munchausen はソフト値が $\max$ を上回るので（PRD067 の同じ節）、単独でも定理 1 の条件 1・2 を破りうる。定理の族に入るのは $\tau_{\text{m}}\to0$ の極限だけである [p.5, Section 4, Theorem 1]。Munchausen と A.L. を重ねるときの $\alpha$ の和が 1 未満という条件も、この極限でなければ十分ではない。なお PRD067 の拡大率 $(1+\alpha)$ は 1 回の適用の値で、反復の極限は $1/(1-\alpha)$ である。
4. $\tau$ と組み合わせるときの式の形が分かる。
   - §5 の 3: 補正項は今の状態の値で、$\gamma$ が掛からない。ボーナスは $\tau$ によらず残し、次の状態の soft value にだけ $\gamma^\tau$ を掛けるのが式の形に沿う。
   - §5 の 5: $\tau=0$ で次の状態がほぼ同じなら、待つ遷移の報酬を $c$（罰なら $c<0$）として $V-Q=-c/(1-\alpha)$ になる（本解説の計算）。$\tau_{\text{m}}>0$ の Munchausen では $V$ をソフト値に読み替えた同じ式になる。$\alpha=0.9$ では罰が 10 倍に効くので、罰の大きさは Munchausen の有無と一緒に決める。
   - §5 の 2: 定理は $\gamma<1$ が前提で、証明は $\gamma^k$ の縮み (11) に頼るので、$\tau=0$ の連鎖は範囲外である [p.2, Section 2]、[p.9, Section 9.1]。一貫ベルマン作用素の形は、$\tau=0$ の厳密な自己遷移で $\gamma^\tau P(x\mid x,a)=1$ となって条件を破り、待つ罰があると Q が際限なく下がる。$\tau=0$ とは組み合わせない。
5. $\tau$ を $\alpha$ に通す案がある。Baird の原式は時間 $\Delta_t$ を $\gamma^{\Delta_t}$ と $K=C\Delta_t$ の両方に持ち、論文はこれを $\Delta_t=1$ に固定している [p.5–6, Section 4.1]。$\tau$ を運ぶなら $\alpha=1-C\tau$（$0<C\tau\leq1$）として、1 フレームの NOOP で強く、長い DROP で弱く差を広げられる。[Tallec et al. (2019)] の時間刻みのスケーリングと同じ方向である。定理の $\alpha$ は上限が 1 未満なら $(x,a)$ に依存してよいと読める（一貫ベルマン作用素がその例）。未検証で、$\tau=0$ には使えない。
6. 持ち込むときの注意。
   - 「同じ状態」の判定は要らない。論文も NN では定数 $\alpha$ にしている。
   - P.A.L. は同じ行動の継続を優遇するので、NOOP の連打を抑えたい DropMerge とは向きが逆になりうる。Atari での順位をそのまま持ち込まない。
   - Q の意味が変わるので、§2 の「NOOP − 最善 DROP」は $\alpha$ の違う Run の間で直接比べられない。BF16 の丸めに対する余裕としての相対ギャップは成績の予測子にならなかったという記録がある（[wiki: breakout](../docs/wiki/subjects/breakout.md) の BF16 の項）。ギャップを広げる作用素どうしを比べた記録ではないが、判定はギャップではなく評価スコアで行う。
   - 論文の実験は 1-step の DQN だけで、QR・IQN、n-step、Double DQN、PER との組み合わせはない。§3.1 の 3 の分布の目標値の判断材料にはならない。

## 出典リスト

本論文（解説の対象）:

[本論文 AAAI 版, 2016/02] Marc G. Bellemare, Georg Ostrovski, Arthur Guez, Philip S. Thomas, Rémi Munos. "Increasing the Action Gap: New Operators for Reinforcement Learning." Proceedings of the AAAI Conference on Artificial Intelligence 30(1)（AAAI 2016）. https://doi.org/10.1609/aaai.v30i1.10303 （論文ページ: https://ojs.aaai.org/index.php/AAAI/article/view/10303 。今回は bot 確認で開けなかった）

[本論文 arXiv 版, 2015/12] Marc G. Bellemare, Georg Ostrovski, Arthur Guez, Philip S. Thomas, Rémi Munos. "Increasing the Action Gap: New Operators for Reinforcement Learning." arXiv:1512.04860v1 [cs.AI]. https://arxiv.org/abs/1512.04860

書誌の確認に使ったもの:

[arXiv 1512.04860, 2015/12] arXiv. 書誌ページ（投稿日と版の履歴）. https://arxiv.org/abs/1512.04860

[Crossref, 2026/11] Crossref. DOI 10.1609/aaai.v30i1.10303 の書誌レコード（2026-11-04 に参照）. https://api.crossref.org/works/10.1609/aaai.v30i1.10303

本論文が引用し、本解説で言及した文献（本論文の参考文献の順）:

[Atkeson, 1991] C. G. Atkeson. "Using locally weighted regression for robot learning." Proceedings of the 1991 IEEE International Conference on Robotics and Automation, pp. 958–963. https://doi.org/10.1109/ROBOT.1991.131713

[Azar et al., 2011] M. G. Azar, R. Munos, M. Ghavamzadeh, H. J. Kappen. "Speedy Q-learning." Advances in Neural Information Processing Systems 24. https://proceedings.neurips.cc/paper_files/paper/2011/hash/ab1a4d0dd4d48a2ba1077c4494791306-Abstract.html （本論文の参考文献では第 3 著者が Gavamzadeh と表記されている）

[Baird, 1999/05] L. C. Baird. "Reinforcement learning through gradient descent." Ph.D. Dissertation, Carnegie Mellon University（CMU-CS-99-132）. https://www.csd.cmu.edu/academics/doctoral/degrees-conferred/leemon-c-baird-iii

[Bellemare et al., 2013/06] M. G. Bellemare, Y. Naddaf, J. Veness, M. Bowling. "The Arcade Learning Environment: An evaluation platform for general agents." Journal of Artificial Intelligence Research 47:253–279. https://doi.org/10.1613/jair.3912

[Bellman, 1957] R. E. Bellman. "Dynamic programming." Princeton University Press. URL 未確認（1957 年の初版に対応する出版社のページは確認できなかった。2010 年の復刻版のページ: https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming）

[Bertsekas and Tsitsiklis, 1996] D. P. Bertsekas, J. N. Tsitsiklis. "Neuro-Dynamic Programming." Athena Scientific. http://athenasc.com/ndpbook.html

[Bertsekas and Yu, 2012/02] D. P. Bertsekas, H. Yu. "Q-learning and enhanced policy iteration in discounted dynamic programming." Mathematics of Operations Research 37(1):66–94. https://doi.org/10.1287/moor.1110.0532

[Bertsekas, 2011/08] D. P. Bertsekas. "Approximate policy iteration: A survey and some new methods." Journal of Control Theory and Applications 9(3):310–335. https://doi.org/10.1007/s11768-011-1005-3

[Deisenroth and Rasmussen, 2011] M. P. Deisenroth, C. E. Rasmussen. "PILCO: A model-based and data-efficient approach to policy search." Proceedings of the 28th International Conference on Machine Learning（ICML 2011）. https://icml.cc/2011/papers/323_icmlpaper.pdf

[Ernst et al., 2005/04] D. Ernst, P. Geurts, L. Wehenkel. "Tree-based batch mode reinforcement learning." Journal of Machine Learning Research 6:503–556. https://www.jmlr.org/papers/v6/ernst05a.html

[Farahmand, 2011] A. Farahmand. "Action-gap phenomenon in reinforcement learning." Advances in Neural Information Processing Systems 24. https://proceedings.neurips.cc/paper_files/paper/2011/hash/013d407166ec4fa56eb1e1f8cbe183b9-Abstract.html

[Gordon, 1995] G. Gordon. "Stable function approximation in dynamic programming." Proceedings of the Twelfth International Conference on Machine Learning（Machine Learning Proceedings 1995）, pp. 261–268. https://doi.org/10.1016/B978-1-55860-377-6.50040-2

[Hoburg and Tedrake, 2009/04] W. Hoburg, R. Tedrake. "System identification of post stall aerodynamics for UAV perching." Proceedings of the AIAA Infotech@Aerospace Conference. https://doi.org/10.2514/6.2009-1930

[Jiang and Powell, 2015/08] D. R. Jiang, W. B. Powell. "Optimal hour-ahead bidding in the real-time electricity market with battery storage using approximate dynamic programming." INFORMS Journal on Computing 27(3):525–543. https://doi.org/10.1287/ijoc.2015.0640

[Kushner and Dupuis, 2001] H. Kushner, P. G. Dupuis. "Numerical methods for stochastic control problems in continuous time." Springer. https://doi.org/10.1007/978-1-4613-0007-6

[Lee et al., 2013/04] D. Lee, B. Defourny, W. B. Powell. "Bias-corrected Q-learning to control max-operator bias in Q-learning." 2013 IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning（ADPRL）, pp. 93–99. https://doi.org/10.1109/ADPRL.2013.6614994

[Mnih et al., 2015/02] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. "Human-level control through deep reinforcement learning." Nature 518(7540):529–533. https://doi.org/10.1038/nature14236

[Munos and Moore, 1998] R. Munos, A. Moore. "Barycentric interpolators for continuous space & time reinforcement learning." Advances in Neural Information Processing Systems 11. https://proceedings.neurips.cc/paper_files/paper/1998/hash/d961e9f236177d65d21100592edb0769-Abstract.html

[Munos and Moore, 2002/11] R. Munos, A. Moore. "Variable resolution discretization in optimal control." Machine Learning 49(2–3):291–323. https://doi.org/10.1023/A:1017992615625

[Ormoneit and Sen, 2002/11] D. Ormoneit, Ś. Sen. "Kernel-based reinforcement learning." Machine Learning 49(2–3):161–178. https://doi.org/10.1023/A:1017928328829

[Randlov and Alstrom, 1998] J. Randlov, P. Alstrom. "Learning to drive a bicycle using reinforcement learning and shaping." Proceedings of the Fifteenth International Conference on Machine Learning. URL 未確認（公式の掲載ページは確認できなかった。二次索引の候補: https://mlanthology.org/icml/1998/randlv1998icml-learning/）

[Riedmiller et al., 2009/07] M. Riedmiller, T. Gabel, R. Hafner, S. Lange. "Reinforcement learning for robot soccer." Autonomous Robots 27(1):55–73. https://doi.org/10.1007/s10514-009-9120-4

[Rummery and Niranjan, 1994/09] G. A. Rummery, M. Niranjan. "On-line Q-learning using connectionist systems." Technical Report CUED/F-INFENG/TR166, Cambridge University Engineering Department. https://mi.eng.cam.ac.uk/reports/abstracts/rummery_tr166.html

[Sutton and Barto, 1998] R. S. Sutton, A. G. Barto. "Reinforcement learning: An introduction." MIT Press. https://mitpress.mit.edu/9780262193986/reinforcement-learning/

[Sutton et al., 2011/05] R. Sutton, J. Modayil, M. Delp, T. Degris, P. Pilarski, A. White, D. Precup. "Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction." Proceedings of the 10th International Conference on Autonomous Agents and Multiagent Systems（AAMAS 2011）, pp. 761–768. https://www.ifaamas.org/Proceedings/aamas2011/papers/A6_R70.pdf

[Sutton, 1988/08] R. S. Sutton. "Learning to predict by the methods of temporal differences." Machine Learning 3(1):9–44. https://doi.org/10.1007/BF00115009

[Sutton, 1996] R. S. Sutton. "Generalization in reinforcement learning: Successful examples using sparse coarse coding." Advances in Neural Information Processing Systems 8, pp. 1038–1044. https://proceedings.neurips.cc/paper_files/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html

[Tesauro, 1995/03] G. Tesauro. "Temporal difference learning and TD-Gammon." Communications of the ACM 38(3):58–68. https://doi.org/10.1145/203330.203343

[Togelius et al., 2009/09] J. Togelius, S. Karakovskiy, J. Koutník, J. Schmidhuber. "Super Mario evolution." 2009 IEEE Symposium on Computational Intelligence and Games, pp. 156–161. https://doi.org/10.1109/CIG.2009.5286481

[van Hasselt et al., 2016/03] H. van Hasselt, A. Guez, D. Silver. "Deep reinforcement learning with double Q-learning." Proceedings of the AAAI Conference on Artificial Intelligence 30(1). https://doi.org/10.1609/aaai.v30i1.10295 （arXiv 版: https://arxiv.org/abs/1509.06461）

[van Hasselt, 2010] H. van Hasselt. "Double Q-learning." Advances in Neural Information Processing Systems 23. https://proceedings.neurips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html

[Veness et al., 2015/02] J. Veness, M. G. Bellemare, M. Hutter, A. Chua, G. Desjardins. "Compress and control." Proceedings of the AAAI Conference on Artificial Intelligence 29(1). https://doi.org/10.1609/aaai.v29i1.9600 （arXiv 版: https://arxiv.org/abs/1411.5326）

[Watkins, 1989/05] C. Watkins. "Learning From Delayed Rewards." Ph.D. Dissertation, University of Cambridge. https://doi.org/10.17863/CAM.132164

SMDP の節でだけ参照したもの:

[Tallec et al., 2019/01] Corentin Tallec, Léonard Blier, Yann Ollivier. "Making Deep Q-learning Methods Robust to Time Discretization." ICML 2019. https://arxiv.org/abs/1901.09732 （詳細解説: [tallec2019_time_discretization_paper_2026-10-03.md](tallec2019_time_discretization_paper_2026-10-03.md)）

出典リストの年月は、掲載誌の号や公開日など出典側に月の記載があるものだけに月を付けた。会議論文と書籍は年だけにした。本文中の [著者 (年)] は、この一覧の同じ著者と年の項目を指す。arXiv と Crossref 以外の URL は下書きから引き継いだもので、今回は開いて確かめていない。参考文献の著者・題名・巻号は本論文の参考文献の節 [p.8] と読み合わせた。
