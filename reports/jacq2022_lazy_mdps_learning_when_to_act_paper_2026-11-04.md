# Lazy-MDPs: Towards Interpretable Reinforcement Learning by Learning When to Act — 詳細解説

- 著者: Alexis Jacq, Johan Ferret, Olivier Pietquin, Matthieu Geist [p.669]
- 所属: 4 人とも Google Research, Brain Team。Ferret は Inria（Scool Team）と CRIStAL, CNRS, Université de Lille にも所属する [p.669]
- 掲載: Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems（AAMAS 2022）, Online, May 9–13, 2022, IFAAMAS [p.669]。pp. 669–677（PDF に印字されたページ番号による）
- 年: 2022
- arXiv: 2203.08542（v1 のみ。2022 年 3 月 16 日、cs.LG、cs.AI）[arXiv 2203.08542 (2026/11)]
- URL: https://arxiv.org/abs/2203.08542 （AAMAS 版の PDF: https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2022/pdfs/p669.pdf ）

書誌の注記:

- キューで「未確認」だった名前は、著者・掲載・arXiv 番号ともに確かめた。AAMAS 版の題名は「Towards Interpretable RL」と略し、arXiv 版の題名がキューの名前と同じである [p.669] [arXiv 2203.08542 (2026/11)]。
- arXiv のコメント欄によれば、arXiv 版は AAMAS 2022 の 14 ページの拡張版で、Sec. 7.4 と付録 K を足したものである [arXiv 2203.08542 (2026/11)]。
- arXiv の Journal reference 欄は Autonomous Agents and Multi-Agent Systems (2022) と書く [arXiv 2203.08542 (2026/11)]。AAMAS 版の記載は上の会議録で [p.669]、雑誌への掲載は確かめていない。
- 版の違い（本解説の照合）: 要旨の結びは、AAMAS 版が DQN に近づきときに上回ると書き、arXiv 版は学習済みまたはランダムの既定で競争力のある性能と書く [p.669, Abstract] [arXiv p.1, Abstract]。節名は 3・4・5・6・7.3 で違う（AAMAS 版は Framework、Solving Lazy-MDPs、Setting the Cost、Learning When and How to Act、Doing less for better exploration。arXiv 版は Lazy-MDPs、Optimality in lazy-MDPs、Setting the cost of taking control、Learning when and how to act in Lazy-MDPs、Lazy exploration）。信用割り当ての段落の括弧書きは、AAMAS 版がパラメータは大幅に少ないがパラメトリックと書き、arXiv 版はノンパラメトリックと書く [p.670, Section 2] [arXiv p.3, Section 2]。結論の違いは 8 章に書く。本解説の見出しは AAMAS 版による。

Date: 2026-11-04

表記の約束:

- 著作権のため、論文の内容は筆者の言葉で要約する。逐語の引用は 2 章の 1 か所（14 語）だけで、アブストラクトも要約にした。論文は arXiv で CC BY-NC-SA 4.0 で公開されている [arXiv 2203.08542 (2026/11)]。paper-details スキルが求める「原文と訳のブロックを主体にする」形式と図表の画像の貼り付けからは、この理由で外れている。
- 図表は転載せず、番号で参照して内容を言葉で説明する。図の曲線から値を読み取らない。数値は本文とキャプションに印字されたものだけを使う。
- 章ごとの監査は未実施（費用のため）。数値と式は、筆者が下の (a)(b) の PDF のページ画像と読み合わせた。第三者照合の指摘を受けて (b) を全ページ読み直した。
- 読んだ版は次の 2 つで、どちらも全ページの画像を筆者が直接読んだ。
  - (a) AAMAS 版の PDF（9 ページ、付録なし）。https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2022/pdfs/p669.pdf
  - (b) arXiv v1 の PDF（26 ページ。本文 14 ページに参考文献と付録が続く）。https://arxiv.org/pdf/2203.08542 。AAMAS 版に無い Section 7.4 と付録 A〜K はこちらにある。Figure 12 の凡例は字が小さいので、同じ図の ar5iv の画像を拡大して読んだ。
- 位置の書き方: [p.671, Section 3] は (a) の印字ページ。(b) は [arXiv p.14, Section 7.4] と書く。式は (a) に番号が無いので、節と定理・性質の番号で指す。
- 式は記号を論文から置き換えて書き直した。対応は 3 章の表にある。
- 筆者が独自に加えた計算や説明は「本解説の補足」、論文内の記載どうしの照合は「本解説の照合」、推論は「本解説の推論」と明記し、論文の記述と区別する。
- 論文が引用する文献は [著者 (年)] で示し、書誌は出典リストに載せる。出典リストの書誌は論文の参考文献の印字に従い、URL は印字のあるものだけ書く。閲覧時点の内容によるものは [名称 (閲覧した年/月)] と書く。

用語: 論文の take control を「制御を取る」（元の行動空間の行動を選ぶこと）、lazy action を「任せる行動」、default policy を「既定の方策」と書く。lazy-MDP、lazy-gap、lazy 方策は論文の語のまま使う。

## アブストラクト（要約）

論文は、強化学習が扱ってきた「どう行動するか」と同じくらい「いつ行動するか」が重要だと主張する。人間が習慣的なふるまいと注意を払ったふるまいを行き来するのになぞらえ、MDP に、決定を既定の方策へ任せる行動を足し、それ以外の行動に罰を与えて、決定的な場面にだけ行動させる。この形式を lazy-MDP と呼び、価値関数・貪欲性・最適解を理論的に調べる。実験では、制御を取る状態が構成上そのまま見えるので一種の解釈可能性が得られると示す。既定の方策が準最適（一様ランダムを含む）でも、Atari で限られた状態でしか制御を取らずに DQN に近づき、ときに上回る [p.669, Abstract]。arXiv 版の要旨は結びの表現が違う（書誌の注記を参照）。

## Table of Contents

- [1. はじめに（Introduction）](#1-はじめにintroduction)
- [2. 関連研究（Related Work）](#2-関連研究related-work)
- [3. 枠組み（Framework）](#3-枠組みframework)
- [4. lazy-MDP を解く（Solving Lazy-MDPs）](#4-lazy-mdp-を解くsolving-lazy-mdps)
- [5. 罰の決め方（Setting the Cost）](#5-罰の決め方setting-the-cost)
- [6. いつどう行動するかの学習（Learning When and How to Act）](#6-いつどう行動するかの学習learning-when-and-how-to-act)
- [7. 実験（Experiments）](#7-実験experiments)
- [8. 結論（Conclusion）](#8-結論conclusion)
- [付録 A〜K（arXiv 版のみ）](#付録-ak-arxiv-版のみ)
- [論文内の記載の食い違い（本解説の照合）](#論文内の記載の食い違い本解説の照合)
- [論文に書かれていない点](#論文に書かれていない点)
- [確かめられなかった点](#確かめられなかった点)
- [キューの論点への答え](#キューの論点への答え)
- [PRD 958 の検討に効く点](#prd-958-の検討に効く点)
- [本解説で省略した内容](#本解説で省略した内容)
- [出典リスト](#出典リスト)

## 1. はじめに（Introduction）

MDP [Puterman (1994)] と RL は「どう行動するか」に答える枠組みで、論文はこれに対して「いつ、どう行動するか」を問う [p.669, Section 1]。動機は 2 つある。1 つは、結果を左右する状態はごく一部で、ほかの状態では行動が環境のふるまいほど効かない課題が多いこと（落下中の Tetris のピースの向きなど）。もう 1 つは、既存の方策の上で、制御を取る場面だけを学んで改善できることで、人とロボットの協調 [Meresht et al. (2020)] が例に挙がる。既定の方策の形は問わず、制御器、手作りの方策、プログラムなどでよいとされる [p.669, Section 1]。

MDP を作り直す代わりに、行動空間に任せる行動を足し、元の行動を選ぶ（制御を取る）たびに報酬から罰を引く [p.669, Section 1] [p.669, Figure 1]。任せる行動が離散なので簡単のため離散行動に絞るが、既定と学習の 2 つのアクターとそれぞれのクリティックを持てば連続制御にも適応できる、と述べる [p.669, Section 1]。

解釈可能性としては、既定の方策と lazy 方策の状態ごとの違いから、既定の方策を上回るのに何が要るか、または課題全体で何が要るかを読み取ることを指す。「なぜ」に答える説明 [Molnar (2020)] は狙わない [p.669–670, Section 1]。

貢献は、(1) lazy-MDP の形式、(2) 最適性が既定の方策と罰の値に依存することの理論的な特徴づけ、(3) 状態の解釈可能な分割が学ばれることと、罰で制御を減らしたときのスコアの変化を示す実験の 3 つで、探索の難しい課題では制御を減らすと性能が上がることもあるとする [p.670, Section 1]。

## 2. 関連研究（Related Work）

論文は、既定のふるまいをできるだけ採るよう方策を制約する考えは筆者らの知る限り新しいとし、次の分野との関係を整理する [p.670, Section 2]。

- 残差 RL [Johannink et al. (2019)] [Silver et al. (2018)]: 基の方策の行動に足す残差の方策を学ぶ。行動の和が要るので連続制御に限られ、lazy-MDP は離散制御に使える。
- 探索を意識した RL [Shani et al. (2019)]（[Van Seijen et al. (2009)] に関連づけて述べられる）: 現在の方策と固定の方策を線形に混ぜた力学の代理 MDP を解くもので、実際のふるまいが固定の混合だという前提の最適化である。lazy 方策は状態ごとに切り替えられる点が違う。
- 解釈可能な RL: 入力の影響を事後の勾配 [Wang et al. (2015)] [Zahavy et al. (2016)] や差分の推定 [Greydanus et al. (2018)] [Puri et al. (2019)] で測る研究、解釈可能なモデルやプログラムで模倣する研究 [Coppens et al. (2019)] [Liu et al. (2018)] [Verma et al. (2019a)] [Verma et al. (2019b)]、表現を理解する研究 [Zahavy et al. (2016)] [Topin & Veloso (2019)]、代表的な軌跡を選ぶ研究 [Amir & Amir (2018)] [Amitai & Amir (2021)]、選好を復元する研究 [Bica et al. (2021)] [Juozapaitis et al. (2019)] が挙がる。どれも決定的な状態の特定を明示的には扱わないとする。近い量の action gap [Bellemare et al. (2016b)] と importance advice [Torrey & Taylor (2013)] には 7.2 節で弱点を示す。
- 信用割り当て: 観測した収益の分解や再分配 [Arjona-Medina et al. (2019)] [Ferret et al. (2019)] [Hung et al. (2019)] [Raposo et al. (2021)] や、後知恵の情報の利用 [Ferret et al. (2021)] [Harutyunyan et al. (2019)] [Mesnard et al. (2021)]。lazy-MDP は結果ではなく性能の向上に結びつき、MDP の抽象化から来る点が違う。
- 時間抽象: オプション [Bacon et al. (2017)] [Barreto et al. (2019)] [Precup (2000)] [Sutton et al. (1999)] は開始状態の集合・方策・終了確率の 3 つ組で、一から学ぶのは難しく、1 行動のオプションに潰れやすく、通常の RL より効率が悪いことが多いとされる。マルコフジャンプ過程 [Huang et al. (2019)] は観測の頻度を制御する。skip-MDP [Biedenkapp et al. (2021)] は行動方策とスキップ方策に分解する。

skip-MDP について、論文は次のように評する。

```
"does not exactly learn when to act as it permanently plays a decided action"

「決めた行動を実行し続けるので、いつ行動するかを厳密には学んでいない」
```

[p.670, Section 2]

続けて論文は、行動を続ける必要がある場面が多いことは示せても、独立の方策へ任せる状態が無いので、決定が振る舞いと報酬に強く効く状態を示せないと述べる。動的な行動反復 [Lakshminarayanan et al. (2017)] [Sharma et al. (2017)] は概念的に近いが、MDP の抽象化としては定式化されていないとする [p.670, Section 2]。

- 正則化 RL [Geist et al. (2019)]: エントロピー正則化 [Neu et al. (2017)] は全状態でランダムさを促すが、一様ランダムを既定にした lazy 方策が完全にランダムになるのは一部の状態だけである。KL 正則化 [Vieillard et al. (2020)] は直前の方策の近くに留める。基準のふるまいから離れない別の方法に状態訪問分布の正則化 [Geist et al. (2021)] [Lee et al. (2019)] がある [p.670, Section 2]。

## 3. 枠組み（Framework）

記号を次のように置く。論文は $\gamma<1$ の無限ホライズンで述べるが、有限ホライズンにも適用できるとする [p.671, Section 3]。

| 本解説の記号 | 意味 | 論文の表記 |
| --- | --- | --- |
| $\mathcal{M}=(\mathcal{S},\mathcal{A},\gamma,r,P,\delta_0)$ | 基の MDP。$\mathcal{A}$ は離散の行動空間、$\delta_0$ は初期状態の分布 | $M=(\mathcal{S},\mathcal{A},\gamma,r,\mathcal{P},\delta_0)$ |
| $\mathcal{S}_{\mathrm{end}}$ | 吸収状態の集合。自分へ確定的に移り、報酬は 0 | $\mathcal{S}_{\mathrm{abs}}$ |
| $V^{\pi}$, $Q^{\pi}$ | 基の MDP での方策 $\pi$ の価値 | 同じ |
| $D^{\pi}(s,a)$ | 吸収状態に入るまでの割引ステップ数の期待値。5 章で使う | $Z^{\pi}$ |
| $\ell$ | 任せる行動 | $\bar{a}$ |
| $\pi_{\mathrm{d}}$ | 既定の方策。状態から $\mathcal{A}$ 上の分布への写像 | $\bar{\pi}$ |
| $\eta$ | 罰。実数 | 同じ |
| $\mathcal{A}^{\ell}$, $r^{\ell}$, $P^{\ell}$ | lazy-MDP の行動空間・報酬・遷移 | $\mathcal{A}_+$, $r_+$, $\mathcal{P}_+$ |
| $\mu$ | lazy-MDP の方策（$\mathcal{A}^{\ell}$ 上の分布） | $\pi_+$ |
| $\pi^{\mu}$ | $\mu$ を基の MDP で実行したときの方策 | $\pi$ |
| $V^{\ell,\mu}$, $Q^{\ell,\mu}$ | lazy-MDP での $\mu$ の価値（罰込み） | $V_+^{\pi_+}$, $Q_+^{\pi_+}$ |
| $C^{\mu}$ | 費用の関数。罰の割引和 | $C^{\pi_+}$ |
| $Q^{\mu}_{\mathcal{A}}(s,a)$ | lazy-MDP で $a\in\mathcal{A}$ を取る価値（罰込み） | $Q^{\pi_+}_{\setminus\bar{a}}$ |
| $\mu_{\mathcal{A}}$ | $\mu$ から任せる行動を除いて正規化した方策 | $\pi_{\setminus\bar{a}}$ |
| $\Delta_Q(s)$ | lazy-gap | $G_Q(s)$ |
| $\mathcal{H}(Q)$ | $Q$ から作る貪欲な方策 | $\mathcal{G}(Q)$ |
| $\mathcal{B}$ | lazy-MDP の Bellman 型の作用素 | $\mathcal{T}$ |

lazy-MDP は基の MDP に $\ell,\pi_{\mathrm{d}},\eta$ を足した組で、それ自体も MDP である。基の MDP の報酬を保ち、$\ell$ 以外の全行動で $-\eta$ を追加で受ける [p.671, Section 3]。$\mathcal{S},\gamma,\delta_0$ は変わらない。

$$\mathcal{A}^{\ell}=\mathcal{A}\cup\{\ell\}$$

$$
r^{\ell}(s,a)=\begin{cases} r(s,a)-\eta & a\in\mathcal{A}\\ \mathbb{E}_{b\sim\pi_{\mathrm{d}}(\cdot\mid s)}[r(s,b)] & a=\ell \end{cases}
\qquad
P^{\ell}(s'\mid s,a)=\begin{cases} P(s'\mid s,a) & a\in\mathcal{A}\\ \mathbb{E}_{b\sim\pi_{\mathrm{d}}(\cdot\mid s)}[P(s'\mid s,b)] & a=\ell \end{cases}
$$

[p.671, Section 3]

## 4. lazy-MDP を解く（Solving Lazy-MDPs）

この章の結論は 2 つで、制御を取る重要さを測る lazy-gap を見つけること（4.2・4.3）と、この値だけで制御を取るかを決めれば lazy-MDP で最適になること（4.4）である [p.671, Section 4]。

### 4.1 価値関数（Value functions）

$\mu$ が任せる行動を選ぶと実際の行動は $\pi_{\mathrm{d}}$ から引かれるので、基の MDP で実行される方策は $\pi^{\mu}(a\mid s)=\mu(a\mid s)+\mu(\ell\mid s)\,\pi_{\mathrm{d}}(a\mid s)$ である [p.671, Section 4.1]。lazy-MDP での価値は、基の MDP での価値と費用の和に分けられる。

$$V^{\ell,\mu}(s)=V^{\pi^{\mu}}(s)+C^{\mu}(s)$$

定理 1 は、費用が次の Bellman 方程式を満たすとする [p.671, Theorem 1]。

$$C^{\mu}(s)=-\eta\bigl(1-\mu(\ell\mid s)\bigr)+\gamma\,\mathbb{E}_{a\sim\pi^{\mu},\,s'\sim P(\cdot\mid s,a)}\bigl[C^{\mu}(s')\bigr]$$

一度も任せない方策の費用は最大で、$V^{\ell,\mu}=V^{\pi^{\mu}}-\eta/(1-\gamma)$ になる [p.671, Section 4.1]。

### 4.2 Q 関数（Q-functions）

$a\in\mathcal{A}$ を取る価値は、1 ステップ目の罰と将来の費用を含む [p.671, Section 4.2]。

$$Q^{\mu}_{\mathcal{A}}(s,a)=r(s,a)-\eta+\gamma\,\mathbb{E}_{s'}\bigl[V^{\ell,\mu}(s')\bigr]=Q^{\pi^{\mu}}(s,a)-\eta+\gamma\,\mathbb{E}_{s'}\bigl[C^{\mu}(s')\bigr]$$

性質 1 は、状態価値をこの Q で書く。$\mu_{\mathcal{A}}(a\mid s)=\mu(a\mid s)/(1-\mu(\ell\mid s))$ である [p.671, Property 1]。

$$V^{\ell,\mu}(s)=\bigl(1-\mu(\ell\mid s)\bigr)\,\mathbb{E}_{a\sim\mu_{\mathcal{A}}}\bigl[Q^{\mu}_{\mathcal{A}}(s,a)\bigr]+\mu(\ell\mid s)\Bigl(\mathbb{E}_{a\sim\pi_{\mathrm{d}}}\bigl[Q^{\mu}_{\mathcal{A}}(s,a)\bigr]+\eta\Bigr)$$

性質 2 は、任せる行動を含む Q を与える [p.671, Property 2]。

$$Q^{\ell,\mu}(s,a)=\begin{cases} Q^{\mu}_{\mathcal{A}}(s,a) & a\neq\ell\\ \mathbb{E}_{b\sim\pi_{\mathrm{d}}}\bigl[Q^{\mu}_{\mathcal{A}}(s,b)\bigr]+\eta & a=\ell \end{cases}$$

$+\eta$ が付くのは、$Q^{\mu}_{\mathcal{A}}$ が 1 ステップ目の罰を引いてあり、任せるとその罰を払わないからである（本解説の補足）。

### 4.3 貪欲性（Greediness）

貪欲な方策を作るのに便利な量が lazy-gap で、最良の行動の価値と、次の行動を既定の方策で引いたときの価値の期待値との差である。論文は、ここでの $Q$ は lazy-MDP の価値、つまり費用を織り込んだものだと明記する [p.672, Section 4.3]。

$$\Delta_Q(s)=\max_{a\in\mathcal{A}}Q(s,a)-\mathbb{E}_{a\sim\pi_{\mathrm{d}}}\bigl[Q(s,a)\bigr]$$

性質 3 と貪欲な方策 $\mathcal{H}(Q)$ は、次の規則である。argmax が複数なら一様に割る [p.672, Property 3] [p.672, Section 4.3]。

$$\mathcal{H}(Q)(\cdot\mid s)=\begin{cases} \text{$\arg\max_{a\in\mathcal{A}}Q(s,a)$ の上の一様分布} & \Delta_Q(s)>\eta\\ \ell \text{ に 1 点集中} & \text{それ以外} \end{cases}$$

この規則は $\ell$ の価値を読まないので、論文は $\mathcal{A}$ 上の Q から貪欲な方策を作れるとし、その Q を「基の MDP の Q 関数」と呼ぶ [p.672, Section 4.3]。ただし 4.4 節の不動点は罰込みの Q なので、基の MDP の価値関数 $Q^{*}$ を入れてよいという意味ではない（本解説の補足）。

### 4.4 最適性（Optimality）

作用素 $\mathcal{B}$ は、即時の報酬と罰に、次の状態の貪欲な方策のもとでの価値の平均を足す [p.672, Section 4.4]。

$$\mathcal{B}Q(s,a)=r(s,a)-\eta+\gamma\,\mathbb{E}_{s'\sim P(\cdot\mid s,a),\ a'\sim\mathcal{H}(Q)(\cdot\mid s')}\bigl[Q(s',a')\bigr]$$

$a'=\ell$ のときの $Q(s',\ell)$ を性質 2 で読むと、次のように書き直せる（本解説の補足。付録 E の証明は $a'=\ell$ をこの形で扱う [arXiv p.21, Appendix E]）。

$$\mathcal{B}Q(s,a)=r(s,a)-\eta+\gamma\,\mathbb{E}_{s'}\Bigl[\max\bigl\{\max_{a'\in\mathcal{A}}Q(s',a'),\ \mathbb{E}_{b\sim\pi_{\mathrm{d}}}[Q(s',b)]+\eta\bigr\}\Bigr]$$

定理 2 は、$\mathcal{B}$ が $\gamma$ 縮小で、lazy-MDP の最適方策 $\mu^{*}$ の $Q^{*}_{\mathcal{A}}:=Q^{\mu^{*}}_{\mathcal{A}}$ に収束するとする [p.672, Theorem 2]。系 1 は、$\mu^{*}$ が決定的で、制御を取るかが $\Delta^{*}:=\Delta_{Q^{*}_{\mathcal{A}}}$ と $\eta$ の比較で決まるとする。論文の系 1 の不等号は性質 3 と向きが逆に印字されている（食い違いの項目 1）[p.672, Corollary 1]。

## 5. 罰の決め方（Setting the Cost）

基の MDP が既知なら、任せる行動が少なくとも 1 状態で取られる最小の罰 $\eta_{\min}$ と、全状態では取られない最大の罰 $\eta_{\max}$ を求められる。2 つは次のように定義され、この範囲の $\eta$ で非自明な判断を学ぶ [p.672, Section 5]。

$$\eta_{\min}=\inf\{\eta>0:\exists s,\ \Delta^{*}(s)<\eta\},\qquad \eta_{\max}=\sup\{\eta>0:\exists s,\ \Delta^{*}(s)>\eta\}$$

### 5.1 $\eta_{\max}$

全状態で $\eta$ が lazy-gap 以上なら、最適方策は全部を任せて費用を払わない。よって $\eta_{\max}$ は、既定の方策のもとでの lazy-gap の最大値である。$Q^{\pi_{\mathrm{d}}}$ は基の MDP での既定の方策の Q である [p.672, Theorem 3]。

$$\eta_{\max}=\max_{s}\Delta_{Q^{\pi_{\mathrm{d}}}}(s)$$

### 5.2 $\eta_{\min}$

$\eta$ が全状態で lazy-gap 以下なら、最適なエージェントは基の MDP の最適方策 $\pi^{*}$ に従い、費用は割引ステップ数に比例する。$D^{\pi^{*}}$ は次の式を満たし、吸収状態に一度も入らなければ全状態で $1/(1-\gamma)$ になる [p.672, Section 5.2]。

$$D^{\pi^{*}}(s,a)=\mathbb{I}\{s\notin\mathcal{S}_{\mathrm{end}}\}+\gamma\,\mathbb{E}_{s'}\Bigl[\mathbb{E}_{b\sim\pi^{*}}\bigl[D^{\pi^{*}}(s',b)\bigr]\Bigr],\qquad C^{\pi^{*}}(s)=-\eta\,\mathbb{E}_{b\sim\pi^{*}}\bigl[D^{\pi^{*}}(s,b)\bigr]$$

実際の MDP は終端を持つので $D^{\pi^{*}}$ は状態と行動で変わり、定理 4 はこれを織り込む。$Q^{*}$ は基の MDP の最適な Q で、$\eta_{\min}\ge 0$ である [p.672, Theorem 4]。

$$\eta_{\min}=\min_{s}\max_{a}\frac{Q^{*}(s,a)-\mathbb{E}_{b\sim\pi_{\mathrm{d}}}[Q^{*}(s,b)]}{1+\mathbb{E}_{b\sim\pi^{*}}[D^{\pi^{*}}(s,b)]-\mathbb{E}_{b\sim\pi_{\mathrm{d}}}[D^{\pi^{*}}(s,b)]}$$

吸収状態が無ければ分母は 1 で、$\eta_{\min}=\min_{s}\Delta_{Q^{*}}(s)$ になる（本解説の補足）。論文は、$\eta<\eta_{\min}$ で任せる行動が選ばれず、$\eta>\eta_{\max}$ で常に選ばれることを実験で確かめたと述べる [p.672, Section 5.2] [arXiv p.23–24, Appendix H]。

## 6. いつどう行動するかの学習（Learning When and How to Act）

lazy-MDP は拡張した MDP なので、通常の RL アルゴリズムをそのまま使える。MDP を lazy-MDP に変換するだけで、アルゴリズムの動きを変えずに、いつどう行動するかを学ぶ [p.672, Section 6]。

- 罰の値: 5 章の上下限の式は、常に任せるのでも常に制御を取るのでもない最適方策を保証する。$\eta_{\max}$ は、既定の方策の価値を SARSA [Rummery & Niranjan (1994)] などで推定し、既知の状態やロールアウトで差の最大を取れば近似できる。$\eta_{\min}$ は $Q^{*}$ と $D^{\pi^{*}}$ が普通は手に入らないので、価値反復や Q 学習 [Watkins & Dayan (1992)] で近似する。$D$ の近似は、損失の報酬を全部 1 に置き換えて得る [p.672–673, Section 6]。
- 既定の方策: 一様ランダムは課題の知識が要らず、行動が本当に効く状態でだけ制御を取るという考えに合う最も単純な選択である。エントロピー正則化に近いが、ランダムさを促すのは一部の状態だけである。複雑な場面では、制御が多すぎると累積費用が高くなり探索を妨げかねない。残差学習のように既知の準最適な方策を置けば、その方策が明らかに悪い状態で制御を取る誘因が生まれる [p.673, Section 6]。
- lazy 方策の解釈: 制御を取る状態は少数になると期待され、専門知識で評価でき、基の方策と性能を比べて得が見合うかも確かめられる。特別な場合が 2 つある。既定が一様ランダムなら、ランダムより大きく得なときだけ行動するので、任せる状態は「ランダムでも構わない」という意味で重要でない。既定が最適と一様ランダムの混合なら、「たいてい最適」より大きく得なときだけ行動するので、行動する状態は「ほぼ最適では足りない」という意味で重要である [p.673, Section 6]。

## 7. 実験（Experiments）

問いは 3 つで、(1) 重要な場面で制御を取るようになるか、(2) 状態の分割は解釈できるか、(3) 罰を上げて制御を減らすと収益はどうなるかである [p.673, Section 7]。

### 7.1 重要な場面で制御を取る（Taking control when it matters）

Rivers and Bridges（R&B）: 3 本の川を滑りやすい橋で渡るグリッドワールドで、開始 S からゴール G へ向かう。水に落ちると報酬 −100、ゴールで +1 [p.673, Section 7.1] [p.673, Figure 2]。滑りやすさの代わりに、既定の方策を橋の上では一様ランダム、ほかでは最適とする [p.673, Section 7.1]。価値は収束まで価値反復で求める [arXiv p.24, Appendix I]。既定の方策が最適な状態があるので定理 4 から $\eta_{\min}=0$ で、$0<\eta\le\eta_{\max}$ の範囲で $\Delta^{*}$ が正なのは橋の上だけになる。つまり橋の上でだけ制御を取る [p.673, Section 7.1] [p.673, Figure 2]。

Key-Door-Treasure（KDT）[Oh et al. (2018)]: 4 部屋の課題 [Sutton & Barto (2018)] の変種で、鍵を取り、扉を開け、宝へ着く順に進み、報酬は宝だけである [p.674, Section 7.1]。既定の方策は一様ランダムで、$\eta$ を 0.008、0.02、0.05 と上げて価値反復で解く [p.674, Figure 3]。$\eta$ が大きいほど制御を取る状態は減り、部屋の間の移動と宝の近くに絞られる [p.674, Section 7.1]。

### 7.2 解釈可能性（Interpretability）

Atari [Bellemare et al. (2013)] の、タイミングが中心のゲームで、制御を取る状態を重要な状態とみなし、定性的にフレームで示す。DQN [Mnih et al. (2015)] の Dopamine [Castro et al. (2018)] の実装を使い、既定の方策は一様ランダムである [p.674, Section 7.2]。

- Breakout（$\eta=0.1$）: 球を打ち返さないと負ける直前にだけ制御を取る [p.675, Figure 4]。
- Ms Pacman（$\eta=0.5$）: 動けない序盤と、近くのボーナスを集める間は任せ、ゴーストが近づくと、パワーアップの取得、ゴーストを食べること、ボーナスの連続した収集で散発的に制御を取る [p.675, Figure 5]。
- Bowling（$\eta=0.04$）: 投げる準備のときだけ制御を取り、球に影響しなくなった投球後は任せる [p.675, Figure 6]。

論文は、制御のタイミングが直感に合うとまとめる [p.674, Section 7.2]。また KDT で、lazy-gap を action gap [Bellemare et al. (2016b)]（最良と 2 番目の行動価値の差）と importance advice [Torrey & Taylor (2013)]（最良と最悪の差）に比べる。lazy-gap は鍵・扉・宝の要所にだけ重要さを置き、action gap は最適な軌跡の全状態に一様に置き、importance advice は報酬への近さに支配される [p.674, Section 7.2] [arXiv p.24–25, Appendix J, Figure 11]。

### 7.3 探索のために手を抜く（Doing less for better exploration）

一様ランダムを既定にすると、ランダムでも構わない状態でランダムさが保たれ、安全なときだけ探索する形になる、というのが論文の説明である [p.674, Section 7.3]。

表形式の実験: KDT の左上の部屋に、小さな報酬の吸収状態（りんご）を足して局所解を作る。条件は表形式の Q 学習、学習率 0.5、$\varepsilon$ は 0.1 から 0 へ線形、$\gamma=0.99$、りんごの報酬 0.1、吸収状態に着かないエピソードは 1000 ステップで打ち切りで、結果は 100 seed の平均である [p.674, Section 7.3] [p.676, Section 7.3] [p.676, Figure 7]。$\eta=0$ は局所解に引かれ、$\eta=0.03$ は局所解を抜けて最適に収束し、$\eta=0.05$ は費用が高すぎて制御を取らない [p.676, Section 7.3] [p.676, Figure 7]。

Atari の実験: 探索の難しいゲームのうち報酬が密な Bank Heist、Frostbite、Ms Pacman、Zaxxon と、疎な Gravitar、Private Eye（分類は [Bellemare et al. (2016a)]）を使う。既定の方策は一様ランダム、$\eta$ は 0.005、0.01、0.02、0.05、0.1、0.2 の 6 値で、3 seed の平均である [p.676, Section 7.3]。Figure 8 の指標は次のとおりである [p.676, Figure 8]。

- 横軸: 制御を取った割合（lazy 方策が任せる行動を選ばなかった割合）。
- 縦軸: 標準の DQN のスコアに対する百分率。0% は報酬なし、100% は DQN と同じ報酬。

多くの場合、制御を減らしてもスコアはあまり下がらない。本文の例は、Gravitar が制御 10% 未満でほぼ 100%、Zaxxon が 30% 未満で 80%、Private Eye が 20% で 100% 超、Frostbite が低い罰で 200% である [p.676, Section 7.3]。

### 7.4 学習済みの方策の上での学習（arXiv 版のみ）

lazy-MDP は残差学習を離散行動に拡張したものと見なせるとして、既定の方策を確実に改善できるかを確かめる [arXiv p.14, Section 7.4]。

- 既定の方策: 漸近性能の 50% の DQN。
- 比較相手: 同じ重みから始める warm-restart の DQN と、一から学ぶ DQN。
- 指標: 60 ゲームを集計した人間基準の正規化スコアの中央値 [Mnih et al. (2015)]（Figure 9）。全ゲームの曲線は Figure 12。
- 結果: warm-restart は一から学ぶより速く伸びるが、不安定で最後は下がる。本文が良い罰として挙げる $\eta=0.05$ の lazy-MDP も一から学ぶより速く伸び、伸びは安定している。漸近的な得は無いが、同じエージェントなので予想どおりとする。探索の難しいゲーム（Frostbite、Zaxxon、Qbert、WizardOfWor）では、7.3 節と同じ現象として lazy-MDP のほうが良い方策になる。ゲームごとに最良の $\eta$ を選ぶと、一から学ぶ DQN に対してはっきり得がある。
- そのほか: lazy-MDP はプログラムや制御器も既定にできるが、warm-restart には互換の重みが要る。$\eta$ の自動選択は今後の課題である。

Figure 9 は $\eta$ が 0.05 とゲームごとの最良の 2 本を描く [arXiv p.14, Figure 9]。Figure 12 の凡例は 0.005、0.02、0.05、0.1、0.2 の 5 値と、本文に説明の無い unbundle という項目である [arXiv p.26, Figure 12]。seed の数は本文にもキャプションにも書かれていない。フレーム数も本文には無いが、Figure 9 の横軸は 200M まで印字されている [arXiv p.14, Figure 9]。

## 8. 結論（Conclusion）

論文は、学習の問題に合う MDP の抽象化として lazy-MDP を提案し、価値関数と最適性を調べ、RL で解けることを示した。元の MDP に戻すと解釈しやすく、制御の重要な状態が分かる。一様ランダムを既定にしても、DQN で一部の状態でだけ制御を取りつつ標準の MDP の方策に近い性能に達し、探索の難しい課題ではより高いスコアにも届く、とまとめる [p.676, Section 8]。arXiv 版の結論には、学習済みの方策を既定にすると warm restart より良い傾向があるという 1 文が加わる [arXiv p.14, Section 8]。

## 付録 A〜K（arXiv 版のみ）

位置は A が p.18、B が p.18–19、C が p.19–20、D が p.20、E が p.20–22、F が p.22、G が p.22–23、H が p.23–24、I が p.24、J が p.24–25、K が p.24 と p.26 である。

- A〜G: 定理 1、性質 1〜3、定理 2〜4 の証明。付録の中では番号が振り直され、性質 1〜3 は性質 4〜6、定理 3 は定理 5 として再掲される。A〜C は定義の展開である。D は、性質 3 の場合分けが $\Delta>\eta$ で制御を取る向きであることを示す。E の定理 2 の証明は拡張した MDP の作用素の縮小を使い、任せる行動の場合は $\eta$ が打ち消し合う。G の定理 4 の証明は分数形の最小値についての補題（系 2）を使う。E〜G の記載の問題は食い違いの項目 6〜8 に挙げる [arXiv p.18–23, Appendix A–G]。
- H（Effect of the cost）: $\eta$ を振って最適なエージェントが任せる行動を選ぶ頻度を測り、R&B と KDT で、既定の方策を一様ランダムと「常に 2 番目に良い行動」の 2 種にした 4 通りで、5 章の上下限を確かめる（Figure 10）[arXiv p.23–24, Appendix H]。
- I（Implementation details）: Atari は Dopamine の既定の構成とハイパーパラメータ（詳細は [Castro et al. (2018)]）。R&B と KDT は収束までの価値反復。探索の実験は環境の力学を知らない表形式の Q 学習で、学習の長さは、1000 エピソードを含む 100 ステップ、エピソードの最大長 1000 という趣旨で書かれる（食い違いの項目 4）[arXiv p.24, Appendix I]。
- J（Lazy-gap as a measure of state importance）: KDT の鍵なしと鍵ありで action gap、importance advice、$\eta=0.03$ と $0.05$ の lazy-gap を並べる（Figure 11）[arXiv p.24–25, Appendix J]。
- K（Atari curves）: 60 ゲームのスコアと、制御を取る割合の曲線（Figure 12）[arXiv p.24, Appendix K] [arXiv p.26, Figure 12]。

## 論文内の記載の食い違い（本解説の照合）

1. 系 1 の不等号: 系 1 は $\mu^{*}(\ell\mid s)>0$ と $\Delta^{*}(s)>\eta$ が同値だと印字する [p.672, Corollary 1]。arXiv 版も同じである [arXiv p.6, Corollary 1]。性質 3 と 4.3 節は $\Delta\le\eta$ のときに任せる行動を選ぶので [p.672, Section 4.3] [p.672, Property 3]、向きが逆である。付録 D の証明も $\Delta>\eta$ で制御を取る向きで [arXiv p.20, Appendix D]、5 章の $\eta_{\min}$・$\eta_{\max}$ の定義の説明も性質 3 の向きに合う [p.672, Section 5]。本解説は系 1 を誤植と読み、性質 3 を正とする。
2. 7.2 節の本文は対象の例に Pong、Breakout、Ms Pacman を挙げるが [p.674, Section 7.2]、図で示すのは Breakout、Ms Pacman、Bowling である [p.675, Figure 4–6]。
3. AAMAS 版の付録への参照は番号が ?? のまま印字され、付録は載っていない [p.672, Section 5.2] [p.673, Section 7] [p.674, Section 7.2]。中身は arXiv 版にある。
4. 探索の実験の長さ: 付録 I の、1000 エピソードを含む 100 ステップという記載は [arXiv p.24, Appendix I]、Figure 7 (d) の横軸の印字が 100 エピソード単位で 10 までなのと合わせると全体で 1000 エピソードとも読めるが、はっきりしない [p.676, Figure 7]。
5. R&B の $\eta$ の範囲は、本文が $0<\eta\le\eta_{\max}$、キャプションが $\eta<\eta_{\max}$ である [p.673, Section 7.1] [p.673, Figure 2]。
6. 付録 E: 任せる行動かどうかの両方の場合で、不等式の右辺に $\gamma$ が無いまま $\gamma$ 縮小と結論する [arXiv p.21, Appendix E]。左辺の $\gamma$ を右辺に残せば $\gamma$ 縮小になるので、結論は変わらない（本解説の補足）。
7. 付録 F: 定理 3 は $\Delta_{Q^{\pi_{\mathrm{d}}}}$ で述べるが、証明は $\Delta^{*}$ を使い、集合の条件も 5 章の定義の不等号ではなく等号で書く [arXiv p.22, Appendix F]。
8. 付録 G: $Q^{*}_{\mathcal{A}}$ を基の MDP の Q と $\eta D^{\pi^{*}}$ の差に置き換える式が、導出なしに置かれる [arXiv p.23, Appendix G]。

## 論文に書かれていない点

DQN の実装の細部は、本文にも付録 I にも書かれていない [arXiv p.8, Section 6] [arXiv p.24, Appendix I]。

- 任せる行動を Q の出力に足したのか、$\mathcal{A}$ 上の Q と 4.3 節の規則で扱ったのか。
- $\varepsilon$ を任せる行動を含めて当てるのか。
- 報酬のクリップと $\eta$ の順序。
- 任せる行動とフレームスキップの関係。
- 既定の方策の引き方。

## 確かめられなかった点

- Dopamine の既定値の中身は調べていない。
- 著者の公開コードは、論文にリンクが無く、検索でも見つからなかった。無いとは断定しない [Web 検索 (2026/11)]。

## キューの論点への答え

依頼時の reports/INDEX.md のキューにあった優先 15 の問いに、論文の記述の範囲で答える。今の INDEX ではこの行が消え、完了表に PRD 958 と書かれているので、問いの文は git の履歴（コミット d37da6f4 で追加）にだけ残る。

1. 既定の方策に任せる行動と罰 $\eta$ の定式化
   - 行動空間に任せる行動 $\ell$ を 1 つ足す。$\ell$ を選ぶと $\pi_{\mathrm{d}}$ から引いた行動が実行され、報酬と遷移は $\pi_{\mathrm{d}}$ の期待値になる。$\ell$ 以外を選ぶと報酬から $\eta$ を引く。これ自体が MDP である [p.671, Section 3]。
   - 罰の割引和は Bellman 方程式を満たす費用 $C^{\mu}$ で、lazy-MDP の価値は基の MDP の価値と費用の和である [p.671, Theorem 1]。
2. 学習した方策が切り替える条件（行動の価値の差と $\eta$ の関係）
   - 条件は $\Delta_Q(s)=\max_{a}Q(s,a)-\mathbb{E}_{a\sim\pi_{\mathrm{d}}}[Q(s,a)]>\eta$ で、満たせば argmax の行動で制御を取り、満たさなければ任せる [p.672, Property 3]。
   - 差を取る相手は既定の方策の Q の期待値で、2 番目の行動ではない [p.672, Section 4.3]。今の行動との差でもない（本解説の補足。論文はこの比較に触れていない）。7.2 節は、2 番目との差（action gap）や最悪との差（importance advice）と別物だと比べている [p.674, Section 7.2]。
   - $Q$ は lazy-MDP の Q で、費用を織り込む [p.672, Section 4.3]。不動点 $Q^{*}_{\mathcal{A}}$ で判定すると最適になる [p.672, Theorem 2] [p.672, Corollary 1]。
   - $\eta$ は報酬から直接引くので、Q の差と同じ単位である（本解説の補足）。意味のある範囲は $\eta_{\min}$ から $\eta_{\max}$ で [p.672, Section 5]、学習前に近似もできる [p.672–673, Section 6]。
3. 既定の方策に「直前の行動を続ける」を置けるか
   - 論文は扱っていない。$\pi_{\mathrm{d}}$ は状態だけの関数として定義され [p.671, Section 3]、使った既定の方策は、一様ランダム（KDT、Atari）、橋の上だけ一様ランダムで他は最適（R&B）、常に 2 番目に良い行動（付録 H）、漸近性能の 50% の DQN（7.4）である [p.673–676, Section 7] [arXiv p.14, Section 7.4] [arXiv p.23–24, Appendix H]。形を問わないとは述べるが、例は制御器、手作りの方策、プログラムである [p.669, Section 1] [arXiv p.14, Section 7.4]。行動の反復は他の研究の話としてだけ出てくる [p.670, Section 2]。
   - 定義の上では、状態を $\tilde{s}=(s,b)$（$b$ は直前の行動）に広げれば、$b$ に 1 点集中する方策として置ける。広げた問題も MDP なので定理はそのまま当てはまる（本解説の推論）。制御を取る枝の Q は $b$ に依らなくなり、$\tilde{Q}(s,a)$ と書けて、4.4 節の書き直しの式と性質 2 から次の形になる（本解説の推論）。

   $$\tilde{Q}(s,a)=r(s,a)-\eta+\gamma\,\mathbb{E}_{s'\sim P(\cdot\mid s,a)}\Bigl[\max\bigl\{\max_{a'}\tilde{Q}(s',a'),\ \tilde{Q}(s',a)+\eta\bigr\}\Bigr]$$

   切り替える条件は $\max_{a}\tilde{Q}(s,a)-\tilde{Q}(s,b)>\eta$ である。エピソードの最初は $b$ が無いので別に定める必要がある。乱数の MDP で価値反復を回すと、$(s,b)$ に広げて任せる行動を持たせた解とこの式の解の差は最大で約 $2.4\times10^{-13}$ だった（本解説の補足）。条件は 8 状態、4 行動、$\gamma=0.95$、$\eta=0.1$。遷移と報酬の生成法は 5 通りで、遷移は Dirichlet(1)、一様乱数の正規化、行き先 2 状態だけの疎の 3 種、報酬は $[0,1]$ または $[-1,1]$ の一様乱数である（一様乱数の遷移は $[0,1]$ の報酬だけ）。各 20 個で、Python の `random.Random(seed0+k)`（$k=0,\dots,19$、seed0 は 0 と 100）で作った。
4. DQN での実装と比べた env
   - 実装: MDP を lazy-MDP に変換すれば通常の RL をそのまま使えるとし [p.672, Section 6]、Dopamine の DQN の既定を使う [p.674, Section 7.2] [arXiv p.24, Appendix I]。出力の持ち方などの細部は書かれていない（「論文に書かれていない点」を参照）。
   - env は次のとおりである。表は本解説のまとめで、論文の表ではない。

| 節 | env | 既定の方策 | 学習 | $\eta$ | 平均 |
| --- | --- | --- | --- | --- | --- |
| 7.1 | R&B | 橋の上は一様ランダム、他は最適 | 価値反復 | $\eta_{\min}=0$ から $\eta_{\max}$ | なし |
| 7.1 | KDT | 一様ランダム | 価値反復 | 0.008、0.02、0.05 | なし |
| 付録 H | R&B、KDT | 一様ランダム、常に 2 番目 | 最適方策 | 振る | なし |
| 7.2 | Breakout、Ms Pacman、Bowling | 一様ランダム | DQN | 0.1、0.5、0.04 | 定性の図 |
| 7.3 | KDT＋りんご | 一様ランダム | 表形式 Q 学習 | 0、0.03、0.05 | 100 seed |
| 7.3 | Atari 6 ゲーム | 一様ランダム | DQN | 0.005〜0.2 の 6 値 | 3 seed |
| 7.4 | Atari 60 ゲーム | 漸近性能の 50% の DQN | DQN | 0.005、0.02、0.05、0.1、0.2（Figure 12）。Figure 9 は 0.05 と最良 | 記載なし |

## PRD 958 の検討に効く点

docs/memo/958_temporl_skip_followups_10prd.md（キューの行の 999 は起票時の番号で、2026-11-03 に 958 へ振り直された）の §2.2 と §9 の 1 に関わる点である。PRD は変更していない。この節は論文の記述ではなく本解説の推論で、数値確認はキューの論点 3 と同じ条件の乱数の MDP による。

1. §2.2 の説明を直す。壁打ちの「行動を変える価値の差が閾値を超えたときだけ切り替える」は論文の定義ではない。論文の差は最良の行動と既定の方策の期待値の差で、既定が「続ける」のときに限り、続けている行動との差になる。
2. D5 の G は既定を「続ける」に置いた lazy-gap と同じ形だが、保証は引き継がない。論文の定理 2 と系 1 は罰を込めて学んだ Q の話で、D5 は罰なしの primary の Q を使う。4.3 節の「基の MDP の Q 関数」という言い方に引かれないよう注意が要る。乱数の MDP で、罰なしの最適な Q に閾値を当てた規則は lazy の最適方策と判断が変わった（5 通りの生成法と 2 組の seed の 10 条件で、20 個中 9〜19 個。$(s,b)$ の組の平均 2.2〜9.5%、最大 9.4〜25%）。値は生成法で変わるので、言えるのは判断がずれることまでである。D5 には、規則の形を借りただけで最適性の根拠は無いと書くのがよい。
3. §3 と §8 の「行動空間と報酬を変える別の関心」のうち、既定が「続ける」なら避けられるのは行動空間の側だけである。キューの論点 3 の式なら、Q の出力も replay の形も変えず、Actor の規則と Q 側の目標値の 1 行で済む。報酬の側の変更は目標値の $-\eta$ として残る。また Q 側の学習を変えるので §3 の別の非ゴールに触れ、n-step、分布型、Munchausen、スキップとの併用は論文に無く未検討である。§8 の理由を書き直し、再開条件を置く候補になる。
4. D6 のうち、$\eta$ を絶対値で与えることと Q の差と同じ単位であることは論文と合う。「Actor の用途ごとに与える」は論文に無く、論文の $\eta$ は MDP の定義に入る 1 つの値である [p.671, Section 3]。§9 の 2（尺度）は論文では決まらない。値は Atari の格子が 0.005〜0.2、図の例が 0.04〜0.5、60 ゲームの Figure 12 が 0.005〜0.2 の 5 値で、本文が良い罰として挙げるのは 0.05 である。報酬のスケールとの関係は述べられず、自動選択は今後の課題である [p.676, Section 7.3] [p.675, Figure 4–6] [arXiv p.14, Section 7.4] [arXiv p.26, Figure 12]。
5. §7 のステージ I（PRD の表記は段 I）の $\eta$ の決め方に論文の根拠は借りられない。論文の $\eta_{\max}$ は既定の方策の Q から決まり、既定が「続ける」なら同じ行動を続け続ける方策の Q になる。ステージ I の百分位は学習した方策の Q の分布なので別の物差しである。論文の見せ方（$\eta$ を振り、制御の割合とスコアを並べる [p.676, Figure 8]）は、ステージ I の記録の形の参考になる。
6. 論文の Atari の結果を (I) の裏付けにしない。「続ける」を既定にした実験は無く、TempoRL との比較も関連研究の文章だけである [p.670, Section 2]。探索の効果を 7.3 節は一様ランダムの既定の副作用として説明するが [p.674, Section 7.3]、7.4 節は学習済みの DQN を既定にした場合も同じ現象として述べる [arXiv p.14, Section 7.4]。理由を一様ランダムに限れないので、「続ける」でどうなるかは論文からは分からない。
7. §9 の 3（G に UQE の risk score を使うか）に論文は答えない。読んだ範囲では不確かさを扱っていない。

## 本解説で省略した内容

- 本文の文章そのもの、図表の転載、Figure 8・9・10・11・12 の点や曲線の細部。
- 付録 A〜D の証明の中身と、付録 E〜G の式の展開。
- 参考文献のうち本解説で言及しなかった細目（編者や版など）。

## 出典リスト

本解説の対象:

[Jacq et al., 2022/05] Jacq, A., Ferret, J., Pietquin, O., and Geist, M. "Lazy-MDPs: Towards Interpretable RL by Learning When to Act." Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2022), pp. 669–677. 本文中では [p.X, ...] と位置だけで参照する. https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2022/pdfs/p669.pdf

[arXiv v1, 2022/03] Jacq, A., Ferret, J., Pietquin, O., and Geist, M. "Lazy-MDPs: Towards Interpretable Reinforcement Learning by Learning When to Act." arXiv:2203.08542v1（拡張版、26 ページ）. 本文中では [arXiv p.X, ...] と位置で参照する. https://arxiv.org/pdf/2203.08542

閲覧した資料:

[arXiv 2203.08542, 2026/11] arXiv. "2203.08542 の抄録ページ（書誌、コメント欄、Journal reference、ライセンス）." https://arxiv.org/abs/2203.08542

[ar5iv, 2026/11] arXiv の HTML 変換. Figure 12 の画像の凡例の拡大に使った. https://ar5iv.labs.arxiv.org/html/2203.08542

[Web 検索, 2026/11] 公開コードの検索。著者のリポジトリは見つからなかった。問い合わせ語は論文名、著者名、github、DQN、Dopamine。

リポジトリ内の資料:

[PRD958, 2026/11] anet-lab. "TempoRL の後続（長い j の不確かさ・閉ループの割り込み・全 k の一括学習）暫定 PRD." docs/memo/958_temporl_skip_followups_10prd.md

[キュー, 2026/10] anet-lab. "サーベイキュー（優先 15）." reports/INDEX.md。今の INDEX には行が無く、問いの文はコミット d37da6f4 の版にある.

論文が引用している文献（本解説で言及したもの。書誌は論文の参考文献の印字による）:

[Amir & Amir, 2018] Amir, D., and Amir, O. "Highlights: Summarizing agent behavior to people." International Conference on Autonomous Agents and Multiagent Systems.

[Amitai & Amir, 2021] Amitai, Y., and Amir, O. "\"I don't think so\": Disagreement-based policy summaries for comparing agents." arXiv:2102.03064.

[Arjona-Medina et al., 2019] Arjona-Medina, J. A., Gillhofer, M., Widrich, M., Unterthiner, T., Brandstetter, J., and Hochreiter, S. "Rudder: Return decomposition for delayed rewards." Advances in Neural Information Processing Systems.

[Bacon et al., 2017] Bacon, P.-L., Harb, J., and Precup, D. "The option-critic architecture." AAAI Conference on Artificial Intelligence.

[Barreto et al., 2019] Barreto, A., Borsa, D., Hou, S., Comanici, G., Aygün, E., Hamel, P., Toyama, D. K., Hunt, J. J., Mourad, S., Silver, D., et al. "The option keyboard: Combining skills in reinforcement learning." Advances in Neural Information Processing Systems.

[Bellemare et al., 2013] Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M. "The arcade learning environment: An evaluation platform for general agents." Journal of Artificial Intelligence Research.

[Bellemare et al., 2016a] Bellemare, M., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., and Munos, R. "Unifying count-based exploration and intrinsic motivation." Advances in Neural Information Processing Systems.

[Bellemare et al., 2016b] Bellemare, M. G., Ostrovski, G., Guez, A., Thomas, P., and Munos, R. "Increasing the action gap: New operators for reinforcement learning." AAAI Conference on Artificial Intelligence.

[Bica et al., 2021] Bica, I., Jarrett, D., Hüyük, A., and van der Schaar, M. "Learning \"what-if\" explanations for sequential decision-making." International Conference on Learning Representations.

[Biedenkapp et al., 2021] Biedenkapp, A., Rajan, R., Hutter, F., and Lindauer, M. "Towards TempoRL: learning when to act." International Conference on Machine Learning, BIG workshop.

[Castro et al., 2018] Castro, P. S., Moitra, S., Gelada, C., Kumar, S., and Bellemare, M. G. "Dopamine: A Research Framework for Deep Reinforcement Learning." arXiv:1812.06110. http://arxiv.org/abs/1812.06110

[Coppens et al., 2019] Coppens, Y., Efthymiadis, K., Lenaerts, T., Nowé, A., Miller, T., Weber, R., and Magazzeni, D. "Distilling deep reinforcement learning policies in soft decision trees." IJCAI/ECAI Workshop on Explainable Artificial Intelligence.

[Ferret et al., 2019] Ferret, J., Marinier, R., Geist, M., and Pietquin, O. "Self-attentional credit assignment for transfer in reinforcement learning." International Joint Conference on Artificial Intelligence.

[Ferret et al., 2021] Ferret, J., Pietquin, O., and Geist, M. "Self-imitation advantage learning." International Conference on Autonomous Agents and Multiagent Systems.

[Geist et al., 2019] Geist, M., Scherrer, B., and Pietquin, O. "A theory of regularized markov decision processes." International Conference on Machine Learning.

[Geist et al., 2021] Geist, M., Pérolat, J., Laurière, M., Elie, R., Perrin, S., Bachem, O., Munos, R., and Pietquin, O. "Concave utility reinforcement learning: the mean-field game viewpoint." arXiv:2106.03787.

[Greydanus et al., 2018] Greydanus, S., Koul, A., Dodge, J., and Fern, A. "Visualizing and understanding atari agents." International Conference on Machine Learning.

[Harutyunyan et al., 2019] Harutyunyan, A., Dabney, W., Mesnard, T., Gheshlaghi Azar, M., Piot, B., Heess, N., van Hasselt, H. P., Wayne, G., Singh, S., Precup, D., et al. "Hindsight credit assignment." Advances in Neural Information Processing Systems.

[Huang et al., 2019] Huang, Y., Kavitha, V., and Zhu, Q. "Continuous-time markov decision processes with controlled observations." Allerton Conference on Communication, Control, and Computing.

[Hung et al., 2019] Hung, C.-C., Lillicrap, T., Abramson, J., Wu, Y., Mirza, M., Carnevale, F., Ahuja, A., and Wayne, G. "Optimizing agent behavior over long time scales by transporting value." Nature Communications.

[Johannink et al., 2019] Johannink, T., Bahl, S., Nair, A., Luo, J., Kumar, A., Loskyll, M., Ojea, J. A., Solowjow, E., and Levine, S. "Residual reinforcement learning for robot control." International Conference on Robotics and Automation.

[Juozapaitis et al., 2019] Juozapaitis, Z., Koul, A., Fern, A., Erwig, M., and Doshi-Velez, F. "Explainable reinforcement learning via reward decomposition." IJCAI/ECAI Workshop on Explainable Artificial Intelligence.

[Lakshminarayanan et al., 2017] Lakshminarayanan, A., Sharma, S., and Ravindran, B. "Dynamic action repetition for deep reinforcement learning." AAAI Conference on Artificial Intelligence.

[Lee et al., 2019] Lee, L., Eysenbach, B., Parisotto, E., Xing, E., Levine, S., and Salakhutdinov, R. "Efficient exploration via state marginal matching." arXiv:1906.05274.

[Liu et al., 2018] Liu, G., Schulte, O., Zhu, W., and Li, Q. "Toward interpretable deep reinforcement learning with linear model u-trees." Joint European Conference on Machine Learning and Knowledge Discovery in Databases.

[Meresht et al., 2020] Meresht, V. B., De, A., Singla, A., and Gomez-Rodriguez, M. "Learning to switch between machines and humans." arXiv:2002.04258.

[Mesnard et al., 2021] Mesnard, T., Weber, T., Viola, F., Thakoor, S., Saade, A., Harutyunyan, A., Dabney, W., Stepleton, T., Heess, N., Guez, A., et al. "Counterfactual credit assignment in model-free reinforcement learning." International Conference on Machine Learning.

[Mnih et al., 2015] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., et al. "Human-level control through deep reinforcement learning." Nature.

[Molnar, 2020] Molnar, C. "Interpretable machine learning." Lulu.com.

[Neu et al., 2017] Neu, G., Jonsson, A., and Gómez, V. "A unified view of entropy-regularized markov decision processes." arXiv:1705.07798.

[Oh et al., 2018] Oh, J., Guo, Y., Singh, S., and Lee, H. "Self-imitation learning." International Conference on Machine Learning.

[Precup, 2000] Precup, D. "Temporal abstraction in reinforcement learning." University of Massachusetts Amherst.

[Puri et al., 2019] Puri, N., Verma, S., Gupta, P., Kayastha, D., Deshmukh, S., Krishnamurthy, B., and Singh, S. "Explain your move: Understanding agent actions using specific and relevant feature attribution." International Conference on Learning Representations.

[Puterman, 1994] Puterman, M. L. "Markov Decision Processes." Wiley.

[Raposo et al., 2021] Raposo, D., Ritter, S., Santoro, A., Wayne, G., Weber, T., Botvinick, M., van Hasselt, H., and Song, F. "Synthetic returns for long-term credit assignment." arXiv:2102.12425.

[Rummery & Niranjan, 1994] Rummery, G. A., and Niranjan, M. "Online Q-learning using connectionist systems."

[Shani et al., 2019] Shani, L., Efroni, Y., and Mannor, S. "Exploration conscious reinforcement learning revisited." International Conference on Machine Learning.

[Sharma et al., 2017] Sharma, S., Lakshminarayanan, A. S., and Ravindran, B. "Learning to repeat: Fine grained action repetition for deep reinforcement learning." International Conference on Learning Representations.

[Silver et al., 2018] Silver, T., Allen, K., Tenenbaum, J., and Kaelbling, L. "Residual policy learning." arXiv:1812.06298.

[Sutton & Barto, 2018] Sutton, R. S., and Barto, A. G. "Reinforcement Learning: An Introduction." The MIT Press.

[Sutton et al., 1999] Sutton, R. S., Precup, D., and Singh, S. "Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning." Artificial Intelligence.

[Topin & Veloso, 2019] Topin, N., and Veloso, M. "Generation of policy-level explanations for reinforcement learning." AAAI Conference on Artificial Intelligence.

[Torrey & Taylor, 2013] Torrey, L., and Taylor, M. "Teaching on a budget: Agents advising agents in reinforcement learning." International Conference on Autonomous Agents and Multi-agent Systems.

[Van Seijen et al., 2009] Van Seijen, H., Van Hasselt, H., Whiteson, S., and Wiering, M. "A theoretical and empirical analysis of expected sarsa." IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning.

[Verma et al., 2019a] Verma, A., Le, H. M., Yue, Y., and Chaudhuri, S. "Imitation-projected programmatic reinforcement learning." Advances in Neural Information Processing Systems.

[Verma et al., 2019b] Verma, A., Murali, V., Singh, R., Kohli, P., and Chaudhuri, S. "Programmatically interpretable reinforcement learning." International Conference on Machine Learning.

[Vieillard et al., 2020] Vieillard, N., Kozuno, T., Scherrer, B., Pietquin, O., Munos, R., and Geist, M. "Leverage the average: an analysis of kl regularization in rl." Advances in Neural Information Processing Systems.

[Wang et al., 2015] Wang, Z., Schaul, T., Hessel, M., Van Hasselt, H., Lanctot, M., and De Freitas, N. "Dueling network architectures for deep reinforcement learning." International Conference on Machine Learning.

[Watkins & Dayan, 1992] Watkins, C. J., and Dayan, P. "Q-learning." Machine Learning.

[Zahavy et al., 2016] Zahavy, T., Ben-Zrihem, N., and Mannor, S. "Graying the black box: Understanding dqns." International Conference on Machine Learning.
