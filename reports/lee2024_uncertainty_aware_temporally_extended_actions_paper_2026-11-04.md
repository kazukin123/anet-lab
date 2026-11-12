# Learning Uncertainty-Aware Temporally-Extended Actions — 詳細解説

- 著者: Joongkyu Lee, Seung Joon Park, Yunhao Tang, Min-hwan Oh [p.1]。Lee と Park は同等の貢献とされる [p.1, 脚注]。
- 所属: Lee と Oh は Seoul National University、Park は Samsung Research、Tang は Google DeepMind [p.1]。
- 掲載: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38 No. 12, pp. 13391–13399（AAAI 2024、Main Technical Track）[ML Anthology (2024)] [arXiv 2402.05439 (2024/02)]。
- 年: 2024
- arXiv: 2402.05439（v1、2024 年 2 月 8 日、cs.LG・stat.ML）[arXiv 2402.05439 (2024/02)]
- DOI: 10.1609/aaai.v38i12.29241 [ML Anthology (2024)]
- URL: https://arxiv.org/abs/2402.05439
- 書誌の確認: 依頼時のサーベイキュー（reports/INDEX.md の優先 14）にあった題名と AAAI 2024 は正しかった。その行は完了時に消え、今の INDEX には「済み」の表の本解説の行だけがある。
- UTE の公開コード: 論文内の GitHub のリンクは DDPG の基にした sfujim/TD3 だけで [p.21, DDPG Implementation Details and Additional Results]、UTE の実装へのリンクは無い。Web 検索 1 回でも見つからなかった。公開されていないと確かめたわけではない。

Date: 2026-11-04

表記の約束:

- 章ごとの監査は未実施である。paper-details の §7 の監査役は費用のため立てず、数値と式は筆者が PDF のページ画像と読み合わせた。
- 著作権のため、本解説は論文の内容を筆者の言葉で要約する。paper-details スキルの「原文の引用と訳のブロックを主体にする」形式からは外れている。逐語の引用は「Ensemble-based Uncertainty Quantification」の節の 1 か所（15 語未満）だけで、要旨の全文も引用しない。論文は arXiv で CC BY-NC-ND 4.0 として公開され [arXiv 2402.05439 (2024/02)]、PDF には AAAI の著作権表示がある [p.1, 脚注]。
- 図・表・アルゴリズムは転載せず、番号とページで参照して内容を言葉で説明する。表の値は議論に必要なものだけを書き、表全体は写さない。図の曲線から値を読み取らず、図中に数字として印字された値だけを使う。
- 読んだ版は arXiv v1 の PDF（全 24 ページ。https://arxiv.org/pdf/2402.05439 ）で、全ページを読んだ。節の構成の確認に HTML 版（https://arxiv.org/html/2402.05439v1 ）も使った。AAAI の公式版は開けなかった（末尾の「確かめられなかった点」）。
- 位置は [p.ページ, 見出し・図表・式] の形で示す。ページは v1 の PDF のページである。PDF の節に番号は無いので、見出しの名前で指す。
- 式は記号の一部を書き直し、対応する論文の式番号を添える。論文の記号との対応は「準備と記法」の表に載せる。
- 筆者が加えた計算や補足、論文内の記載どうしの照合、推論は「本解説の補足」「本解説の照合」「本解説の推論」と明記する。
- 論文が引用する他の文献は [著者 (年)] の形で示し、末尾の出典リストに個別に載せる。書誌は論文の参考文献 [p.8–9, References] によるもので、筆者は原典を読んでいない。

## 要旨（Abstract）

行動の繰り返しは行動空間の時間抽象の一種で、方策の学習を助ける。しかし従来の研究には、最適でない行動を繰り返すと性能が落ち、利点が打ち消されることがあるという大きな限界がある。論文はこれに対して UTE（Uncertainty-aware Temporal Extension）を提案する。UTE はアンサンブルで、行動を延長している間の不確かさを測る。これにより方策は、探索を重んじる動きと不確かさを避ける動きを、目的に応じて選べる。Gridworld と Atari 2600 の実験で既存の行動繰り返し手法を上回り、方策の学習の効率を高めたとする [p.1, Abstract]。要旨に名前が出るのはこの 2 つだが、本文には Chain MDP と Pendulum-v0 の実験もある [p.5, Chain MDP] [p.7, Control Problem: Pendulum-v0]。

## Table of Contents

- [要旨（Abstract）](#要旨abstract)
- [はじめに（Introduction）](#はじめにintroduction)
- [関連研究（Related Work）](#関連研究related-work)
- [準備と記法（Preliminaries and Notations）](#準備と記法preliminaries-and-notations)
- [不確かさを考える時間的延長（Uncertainty-aware Temporal Extension）](#不確かさを考える時間的延長uncertainty-aware-temporal-extension)
  - [時間的に延長した Q 学習（Temporally-extended Q-Learning）](#時間的に延長した-q-学習temporally-extended-q-learning)
  - [アンサンブルによる不確かさの定量化（Ensemble-based Uncertainty Quantification）](#アンサンブルによる不確かさの定量化ensemble-based-uncertainty-quantification)
  - [n-step の Q 学習（Multi-step Q-Learning）](#n-step-の-q-学習multi-step-q-learning)
  - [不確かさのパラメータの適応（Adaptive Uncertainty Parameter）](#不確かさのパラメータの適応adaptive-uncertainty-parameter)
- [実験（Experiments）](#実験experiments)
  - [Chain MDP](#chain-mdp)
  - [Gridworlds](#gridworlds)
  - [Atari 2600: Arcade Learning Environment](#atari-2600-arcade-learning-environment)
  - [制御問題: Pendulum-v0（Control Problem: Pendulum-v0）](#制御問題-pendulum-v0control-problem-pendulum-v0)
- [結論（Conclusion）](#結論conclusion)
- [謝辞（Acknowledgments）](#謝辞acknowledgments)
- [付録: ベースラインの詳細（Details of Baselines）](#付録-ベースラインの詳細details-of-baselines)
- [付録: UTE の実装の詳細（Implementation Details: UTE）](#付録-ute-の実装の詳細implementation-details-ute)
- [付録: 実験の詳細（Experiments Details）](#付録-実験の詳細experiments-details)
- [付録: 追加の実験結果（Further Experimental Results）](#付録-追加の実験結果further-experimental-results)
- [論文が述べている限界と、論文に無いこと](#論文が述べている限界と論文に無いこと)
- [論文内の記載の食い違い（本解説の照合）](#論文内の記載の食い違い本解説の照合)
- [キューの論点への答え](#キューの論点への答え)
- [PRD958 の検討に効く点](#prd958-の検討に効く点)
- [本解説で省略した内容と、確かめられなかった点](#本解説で省略した内容と確かめられなかった点)
- [出典リスト](#出典リスト)

## はじめに（Introduction）

時間抽象は、長い時間の幅をもつ複雑な課題を解く有望な手法とされる（[Fikes et al. (1972)]、[Dayan and Hinton (1992)]、[Parr and Russell (1997)]、[Sutton et al. (1999)]、[Precup (2000)]、[Bacon et al. (2017)]、[Barreto et al. (2019)]、[Machado et al. (2021)]）。階層型の強化学習は、一連の意思決定を下位の行動やサブタスクへ分ける。その代表が option の枠組み（[Sutton et al. (1999)]、[Precup (2000)]）である。時間的に延長した行動は、学習を速め、1 step の探索より有効に探索でき、上位の決定の回数を減らせるとされる（[Stolle and Precup (2002)]、[Biedenkapp et al. (2021)]）。人の学習の見方とも整合するという [Xia and Collins (2021)] [p.1, Introduction]。

行動の繰り返しは、その特殊な形である（[Lakshminarayanan et al. (2017)]、[Sharma et al. (2017)]、[Dabney et al. (2020)]、[Metelli et al. (2020)]、[Biedenkapp et al. (2021)]、[Park et al. (2021)]）。行動と一緒に、繰り返す長さをエージェントが学ぶ。選んだ行動に一定期間コミットすることで探索が深くなり、近傍でふらつく動きを抑えられるとする [p.1, Introduction]。固定回数の繰り返しは Atari でも使われてきた（[Mnih et al. (2015)]、[Machado et al. (2018)]）[p.1, 脚注 1]。

ただし繰り返すだけでは性能の向上は保証されない、というのが論文の問題意識である。危険な領域で無謀に繰り返せばゲームが終わるので、不確かさを避ける動きが役に立つ。逆に疎な報酬では、楽観が足りないと近傍に留まるので、探索寄りの動きが役に立つ。どちらの場合も長い時間の幅の価値の不確かさの扱いが要であり、適切な度合いは環境によって違うので適応的であるべきだとする。将来の不確かさを行動の繰り返しに取り入れた研究はこれまで無い、というのが著者らの認識である [p.1–2, Introduction]。

貢献は 3 点である [p.2, Introduction]。

- アンサンブルで不確かさを考えながら行動を繰り返す枠組みを示す。繰り返しに伴う不確かさの量を制御して、延長の長さと、どれだけ楽観的または悲観的であるべきかを学ぶ。
- 環境ごとに不確かさへの好みがあると示す。Chain MDP は不確かさを好み、Gridworlds は不確かさを避ける。
- 試した環境で、UTE が既存のベースライン（DAR、$\epsilon z$-Greedy、DQN、B-DQN）を最終の評価得点、学習の速さ、状態空間の被覆で上回るとする。

## 関連研究（Related Work）

時間抽象と行動の繰り返し: MDP に option を足すと Semi-Markov Decision Process（SMDP）になる。時間的に延長した行動を定式化したのが option の枠組みで（[Sutton et al. (1999)]、[Precup (2000)]、[Bacon et al. (2017)]）、価値の側では行動価値を一般化した option 価値（[Sutton et al. (1999)]）を、方策の側では option の確率を返す方策を考える。その手前の見方として、時間抽象を AI の行動に合う時間尺度の探索と捉える説明も挙がる [Precup (2000)] [p.2, Related Work]。

単純な option の 1 つが、基本の行動を一定回数繰り返すものである [Schoknecht and Riedmiller (2002)]。繰り返しは探索を深くし（[Dabney et al. (2020)]）、制御の粒度を下げて学習を効率化するとされる（[Lakshminarayanan et al. (2017)]、[Sharma et al. (2017)]、[Metelli et al. (2020)]、[Biedenkapp et al. (2021)]）。延長の長さは分布から引くか（[Dabney et al. (2020)]）、方策が返す（[Lakshminarayanan et al. (2017)]、[Sharma et al. (2017)]）。最も近いのは TempoRL [Biedenkapp et al. (2021)] で、behavior 方策が行動を、skip 方策が繰り返す長さを決める。著者らの直感は、ただ繰り返すだけでは足りず、繰り返しの途中で望ましくない状態に入り、「危険な」領域で破局的な失敗に至りうるというものである [p.2, Related Work]。

強化学習の不確かさ: 不確かさの定量化は近年の成果の柱とされる（[Osband et al. (2016)]、[Bellemare et al. (2016)]、[Badia et al. (2020)]、[Lee et al. (2022)]）。環境の確率性に由来して減らせない aleatoric と、学習の不完全さに由来して減らせる epistemic の 2 種がある。推定の主流はブートストラップで、[Osband et al. (2016)] の Bootstrapped DQN は共有の胴体（論文の語は torso）に $K$ 個の Head をもつ。Q 関数のアンサンブルで過大評価を抑える研究もある（[Anschel et al. (2017)]、[Peer et al. (2021)]）[p.2, Related Work]。

論文の選び方は UCB 型（[Auer et al. (2002)]、[Audibert et al. (2009)]）で、アンサンブルの平均に不確かさを足した値が最大のものを取る（[Chen et al. (2017)]、[Peer et al. (2021)]）。不確かさには、option 価値の複数の推定の分散を使い、著者らはこれで aleatoric と epistemic の両方を捉えられるとする。複数の推定は、共有ネットワークから伸びる Head が出す。各 Head はランダムに初期化し、ブートストラップする [p.2, Related Work]。

## 準備と記法（Preliminaries and Notations）

環境は MDP $\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$ [Puterman (2014)] で、option を組み込むために SMDP $\mathcal{M}_o:=\langle\mathcal{S},\Omega,P_o,R_o\rangle$（[Sutton et al. (1999)]、[Precup (2000)]）を考える。$P_o$ は option を取ったときの遷移、$R_o$ はその報酬である。option 上の方策 $\pi_\omega$ は状態ごとに option の分布を与え、目的は初期状態からの割引収益の期待値の最大化である [p.2–3, Preliminaries and Notations]。

option は一般に、開始からの履歴 $h$ に依存する。semi-Markov の option は開始集合、option 内の方策、終了関数の組である。論文はこのうち、行動 $a$ を $j$ 回取って終わる option を、どの状態でも開始でき、内部の方策が $a$ の one-hot で、履歴の長さが $j$ になると終わる組として定義する [p.3, Preliminaries and Notations]。

本解説で使う記号は次のとおりである。

| 本解説 | 論文 | 意味 |
|---|---|---|
| $o_{a,j}$ | $\omega_{aj}$ | 行動 $a$ を $j$ 回繰り返して終わる option |
| $\mathcal{J}=\lbrace 1,\dots,J\rbrace$ | $\mathcal{J}$、$J$ | 延長の長さ（extension length。TempoRL のスキップ長にあたる）の集合と上限 |
| $Q(s,a;\theta)$ | $Q^{\pi_\omega}(s,a;\theta)$ | 行動価値関数。以下、この側を Q 側と呼ぶ |
| $\tilde{Q}_b(s,a,j;\phi)$ | $\tilde{Q}^{\pi_\omega}_{(b)}(s,\omega_{aj};\phi)$ | $b$ 番目のメンバーの option 価値関数。以下、この側を skip 側と呼ぶ |
| $B$ | $B$ | メンバーの数。実験ではすべて 10 |
| $\bar{\mu}$、$\bar{\sigma}$ | $\hat{\mu}_{\pi_\omega}$、$\hat{\sigma}_{\pi_\omega}$ | メンバーの値の平均と標準偏差 |
| $\lambda$ | $\lambda$ | 不確かさのパラメータ。実数 |
| $m_{t,b}$ | $m_{t,b}$ | 時刻 $t$ のメンバー $b$ の 2 値のマスク |
| $\bar{\theta}$ | $\bar{\theta}$ | Q 側の target network のパラメータ |
| $G_t^{(n)}$ | $\sum_{k=0}^{n-1}\gamma^k r_{t+k}$ | $t$ から $n$ step の割引報酬の和 |
| $\pi_a$、$\pi_e$ | $\pi_a$、$\pi_e$ | 行動を選ぶ方策と、延長の長さを選ぶ方策 |

$j$ 回繰り返すと、区間の途中の組み合わせも含めて合計 $j(j+1)/2$ 本の skip 遷移が観測され、replay に入る [Biedenkapp et al. (2021)]。短い延長ほど観測が多く更新も多いので不確かさが小さく、長い延長は不確かさが大きくなる、と論文は見ている [p.3, Preliminaries and Notations]。

## 不確かさを考える時間的延長（Uncertainty-aware Temporal Extension）

### 時間的に延長した Q 学習（Temporally-extended Q-Learning）

Q 学習 [Watkins and Dayan (1992)] に基づき、最適な行動価値と option 価値を次で定義する [p.3, (1)] [p.3, (2)]。$s_{+j}$ は $s$ から $j$ step 後の状態である。

$$Q^*(s,a)=\mathbb{E}\left[R(s,a)+\gamma\max_{a'}Q^*(s_{+1},a')\right]$$

$$\tilde{Q}^*(s,a,j)=\mathbb{E}\left[R_o(s,o_{a,j})+\gamma^{j}\max_{a',j'}\tilde{Q}^*(s_{+j},a',j')\right]$$

実際には 2 つの別々のニューラルネットワーク（パラメータ $\theta$ と $\phi$）で近似する [p.3, Temporally-extended Q-Learning]。

option の分解: option 上の方策をそのまま学ぶと、行動空間が $|\mathcal{A}|$ から $|\mathcal{A}|\times|\mathcal{J}|$ へ広がり、Q の推定誤差が増幅する（[Thrun and Schwartz (1993)]、[Zahavy et al. (2018)]）。そこで方策を分解する [Biedenkapp et al. (2021)]。

$$\pi_\omega(o_{a,j}\mid s):=\pi_a(a\mid s)\cdot\pi_e(j\mid s,a)$$

$\pi_a$ を必ず先に引き、選んだ行動 $a$ を条件に $\pi_e$ が $j$ を選ぶ。探索の空間は $|\mathcal{A}|+|\mathcal{J}|$ に減る。ただし全組み合わせを探さないので準最適に収束しうる。最適でない行動を繰り返しすぎたり、近視的になりすぎたりする。論文は、これを $\pi_e$ の不確かさの度合いの制御で和らげられるとする [p.3–4, Option Decomposition]。

命題 1: 分解した最適方策について、状態価値は option 価値の最大とも行動価値の最大とも等しい [p.4, Proposition 1]。

$$V^*(s)=\max_{a,j}\tilde{Q}^*(s,a,j)=\max_{a}Q^*(s,a)$$

この結果から、式 (2) の目標値の $\max_{a',j'}\tilde{Q}^*$ の代わりに $\max_{a'}Q^*(s_{+j},a')$ を使う。同じ目標値を共有することで学習が安定するとする [p.4, Proposition 1]。

Algorithm 1 の流れ [p.3, Algorithm 1]: 入力は $\lambda$ と $B$ である。各判断で、$Q$ に対する ε-greedy で $a$ を選び、$\bar{\mu}$ と $\bar{\sigma}^2$ を計算し、次節の規則で $j$ を選び、$a$ を $j$ 回（または終端まで）実行する。延長の途中で打ち切る処理は無い。

### アンサンブルによる不確かさの定量化（Ensemble-based Uncertainty Quantification）

従来の行動繰り返し手法（[Lakshminarayanan et al. (2017)]、[Sharma et al. (2017)]、[Dabney et al. (2020)]、[Biedenkapp et al. (2021)]）は不確かさを考えずに延長するので失敗しやすい、と論文はいう。UTE は TempoRL の不確かさを考える版で、$\pi_e$ が $j$ を選ぶときに $\tilde{Q}$ の不確かさを考える [p.4, Ensemble-based Uncertainty Quantification]。

アンサンブル（[Osband et al. (2016)]、[Da Silva et al. (2020)]、[Bai et al. (2021)]）は、共有部から分かれる $B$ 個の独立な Head で構成する。各 Head は 1 つの option 価値を表し、ランダムに初期化され、経験バッファの異なるサンプルで学ぶ [p.4, Ensemble-based Uncertainty Quantification]。

B-DQN と違い、UTE は全 Head を同じ目標値で学ばせる。Head ごとに target をもつと、目的関数が非凸なので各 Head が別のモードへ収束しうる。すると不確かさの推定 $\bar{\sigma}$ が 0 へ収束できず、最適方策を学べない。この点で同じ目標値の使用を実装の要点の 1 つとする [p.4, Ensemble-based Uncertainty Quantification]。

状態と行動を与えて、$j$ ごとに平均と分散を集計する [p.4, (3)] [p.4, (4)]。

$$\bar{\mu}(s,a,j):=\frac{1}{B}\sum_{b=1}^{B}\tilde{Q}_b(s,a,j)$$

$$\bar{\sigma}^2(s,a,j):=\frac{1}{B}\sum_{b=1}^{B}\tilde{Q}_b(s,a,j)^2-\bar{\mu}(s,a,j)^2$$

延長方策 $\pi_e$ は決定的で、$\lambda\in\mathbb{R}$ を使う [p.4, Ensemble-based Uncertainty Quantification]。

$$j=\arg\max_{j'\in\mathcal{J}}\left\lbrace\bar{\mu}(s,a,j')+\lambda\,\bar{\sigma}(s,a,j')\right\rbrace$$

$\lambda$ は考える不確かさの度合いを表し、符号の意味を論文は次のように書く。

```
The positive λ induces more aggressive exploration, and the negative one causes uncertainty-averse exploration.

正の λ はより積極的な探索を生み、負の λ は不確かさを避ける探索を生む。
```

[p.4, Ensemble-based Uncertainty Quantification]

### n-step の Q 学習（Multi-step Q-Learning）

TempoRL が skip 側だけに使った n-step [Sutton (1988)] を、UTE は Q 側にも使う。replay から引いた $n$ step の遷移は、$n$ が現在の $\pi_e$ の出力 $j$ 以下なら目標の方策に自明に従っているので、重点サンプリングのようなオフポリシーの補正（[Harutyunyan et al. (2016)]）なしで $Q$ の更新に使える、というのが論文の議論である [p.4, Multi-step Q-Learning]。1 step の更新（式 (1)）の代わりに次を使う。

$$L_Q(\theta)=\mathbb{E}_{\tau_t\sim\mathcal{R}}\left[\left(Q(s_t,a_t;\theta)-G_t^{(n)}-\gamma^{n}\max_{a'}Q(s_{t+n},a';\bar{\theta})\right)^2\,\middle|\,n\le j\sim\pi_e\right]$$

$\mathcal{R}$ は replay である。n-step は報酬を速く伝え（[Watkins (1989)]、[Peng and Williams (1994)]）、過大評価も和らげる [Meng et al. (2021)]。$n$ は現在の $\pi_e$ が動的に決めるので、前もって決める必要がないとする [p.4, Multi-step Q-Learning]。

### 不確かさのパラメータの適応（Adaptive Uncertainty Parameter）

$\lambda$ を固定する代わりに、[Badia et al. (2020)] にならって非定常の多腕バンディットで選ぶ方法も提案する。あらかじめ決めた集合 $\Lambda$ から、エピソード $k$ の開始時に $\lambda_k$ を選び、そのエピソードの収益 $R_k(\lambda_k)$ を受け取る。報酬の信号が非定常なので、スライディングウィンドウ UCB に $\epsilon_{ucb}$-greedy の探索を組み合わせる。式は付録にある [p.4–5, Adaptive Uncertainty Parameter]。

## 実験（Experiments）

主な実験は 3 つである。まず正の $\lambda$ が探索を促し（Chain MDP）、負の $\lambda$ が不確かさを避ける（Gridworlds）という仮説を確かめ、次に Atari で、調整した $\lambda$ の効果と適応的な選択の効果を示す。公平のため、各アルゴリズムの最適な値を広い範囲で探したとする [p.5, Experiments]。

### Chain MDP

- 環境: 2 行動（左・右）の鎖 [Osband et al. (2016)]。左端で左を取ると紛らわしい小さな報酬 0.001、右端で右を取ると大きな報酬 1 を得るので、最適方策は右を取り続けることである [p.5, Chain MDP] [p.5, Figure 1]。
- エピソード: 長さ $N+8$ の固定（$N$ は鎖の長さ）で、1 エピソードの報酬は 0 から 10 である [p.5, Setup]。
- 比較: $\epsilon z$-Greedy [Dabney et al. (2020)]、TempoRL [Biedenkapp et al. (2021)]。$J=10$ を TempoRL と UTE に使う [p.5, Setup]。
- UTE の設定: 最も楽観的な $\lambda=+2.0$ を選んで比較する [p.5, Exploration-Favor]。
- 指標: 学習 1,000 エピソードの報酬の曲線下面積を正規化した AUC。1.0 に近いほど速く最適方策に達する。20 回の実行の結果である [p.5, Exploration-Favor] [p.5, Table 1]。

結果 [p.5, Table 1]: すべての鎖の長さで UTE が最も高い。差が大きいのは $N=50$ で、UTE 0.560、$\epsilon z$-Greedy 0.434、TempoRL 0.246。$N=70$ では 3 手法とも低く、UTE 0.191、TempoRL 0.052 である。論文は、これをより良い探索の表れとする。また UTE は最適報酬 10 に達した後の分散が TempoRL より小さく、探索的な $\pi_e$ でより多様なサンプルを集めるためだと説明する [p.5, Exploration-Favor]。

UTE は $\pi_e$ に ε-greedy を使わず、TempoRL は使う。$\pi_a$ の ε-greedy でランダムに行動を選んだとき、楽観的であればより深く探索でき、この環境には繰り返しの途中で終わる危険な領域がないので積極的な探索が有利だとする [p.5, Exploration-Favor]。

### Gridworlds

- 環境: $6\times10$ の Lava。左上から出発し、溶岩（報酬 −1）を避けてゴール（報酬 +1）に達する [p.5, Gridworlds] [p.5, Figure 2]。本文は Bridge と ZigZag を示す。
- 比較: DDQN [Van Hasselt et al. (2016)]、$\epsilon z$-Greedy、TempoRL [p.5, Gridworlds]。
- 学習: 全エージェント $3.0\times10^3$ エピソード。$\pi_a$ の ε は、全エピソードで 1.0 から 0.0 への線形減衰、対数減衰、$\epsilon=0.1$ の固定の 3 種。$J=7$。表形式ではなくニューラルネットワークで学ぶ [p.6, Setup]。
- 指標: 報酬の正規化 AUC、20 seed [p.6, Table 2]。

結果 [p.6, Table 2]: 6 通りの組で、UTE は 0.76〜0.92、DDQN は 0.34〜0.61、TempoRL は 0.12〜0.44、$\epsilon z$-Greedy は 0.36〜0.92 である。UTE はすべての組で最高で、Bridge の対数減衰だけ $\epsilon z$-Greedy と同じ 0.92 である。TempoRL はすべての組で DDQN を下回る（本解説の照合）。Table 2 の UTE の列は、Table 9 の $\lambda=-1.5$ の列と一致する [p.14, Table 9]（本解説の照合）。

論文の解釈: $\lambda$ を小さくするほど性能が上がることは、危険な領域のある環境で悲観が有利という主張を支える [p.6, Uncertainty-Averse]。TempoRL が原論文より大幅に悪いのは、表形式ではなく関数近似で Q を推定するためで、関数近似では過大評価が起きやすく（[Moskovitz et al. (2021)]）、最大の価値の延長を単純に選ぶと破局的になるとする [p.6, Uncertainty-Averse]。

被覆 [p.6, Coverage] [p.6, Figure 4]: $\lambda=-1.5$ の UTE は TempoRL よりかなり良く状態空間を覆う。悲観的な延長方策で準最適な行動の繰り返しが減り、長く生き残れるためだとする。

延長の長さの分布 [p.6, Distribution of Extension Length] [p.6, Figure 3]: $\lambda<0$ の UTE は TempoRL より繰り返しが少ない。長い延長の観測は更新にほとんど使われないので、遠い状態の価値は近い状態よりずっと不確かになり、悲観的な方策はあまり繰り返さなくなるという説明である。

### Atari 2600: Arcade Learning Environment

- 比較: DDQN、Fixed Repeat（$j=4$）、$\epsilon z$-Greedy、DAR [Lakshminarayanan et al. (2017)]、TempoRL、B-DQN [Osband et al. (2016)] の 6 つ [p.6, Atari 2600: Arcade Learning Environment]。
- 学習: $2.5\times10^6$ step（論文の言い方で、わずか 1000 万フレーム）[p.6, Setup]。
- ε: B-DQN 以外は最初の 200,000 step で線形に減らし、以後 0.01 に固定 [p.6, Setup]。
- 評価: 10,000 step ごとに 3 エピソード、ε = 0.001 [p.6, Setup]。
- 環境の設定: OpenAI Gym の Atari、フレームスキップ 4 [Bellemare et al. (2013)]。$J=10$ は [Biedenkapp et al. (2021)] に揃えた [p.7, 脚注 2]。
- 表の指標: 最後の 100,000 step の平均報酬と標準偏差 [p.7, Table 3]。曲線の影は 7 seed の標準偏差 [p.7, Figure 5]。
- $\lambda$ の扱い: 表の「UTE」の n-step の列はゲームごとに最良の $\lambda$ の値で、「Adaptive $\lambda$」の列はバンディットで選ぶ [p.7, Figure 5] [p.23, Table 12]。本文は Crazy Climber、Road Runner、Sea Quest、Up n Down の 4 ゲームを示し、残り 8 ゲームは付録にある [p.7, Table 3] [p.24, Table 13]。

結果: 4 ゲームとも UTE の n-step か Adaptive $\lambda$ が最高である [p.7, Table 3]。たとえば Road Runner は n-step 12323.2、Adaptive $\lambda$ 10353.3、TempoRL 8131.5、1-step の UTE 4935.6、DDQN 3277.0 である。論文は、$\lambda$ を環境に合わせて調整すれば既存の行動繰り返し手法や B-DQN を大きく上回るとする。Fixed Repeat は多くのゲームで学習に失敗し、延長方策を学ぶことが肝要だとする [p.7, Uncertainty-Awareness]。

n-step の効果: Q 側の n-step は多くのゲームで役立ち、平均で 30.2%（1.39 から 1.81）改善し、オフポリシーの補正が要らないという議論を実験で支えるとする [p.7, Effect of n-step Learning]。

適応的な $\lambda$: 固定より学習がやや遅く、$\lambda$ の最適化に追加のサンプルが要るためだと説明する。それでも他の手法を大きく上回り、$\lambda$ を前もって決める必要をなくせるとする [p.7, Adaptive Uncertainty Parameter λ]。

### 制御問題: Pendulum-v0（Control Problem: Pendulum-v0）

連続行動の Pendulum-v0 [Brockman et al. (2016)] で、$\pi_a$ に DDPG [Lillicrap et al. (2015)] を使い UTE-DDPG と呼ぶ。$\lambda$ は適応的に選ぶ。比較は DDPG、FiGAR [Sharma et al. (2017)]、t-DDPG（TempoRL の DDPG 版）[Biedenkapp et al. (2021)]。学習は $3\times10^4$ step、250 step ごとに評価し、最初の $10^3$ step は一様ランダムである [p.7, Setup]。$J$ は 2、4、6、8 を試す [p.8, Table 4]。曲線は 20 回の試行の平均と標準偏差である [p.23, Figure 14]。

結果 [p.8, Table 4]: 最後の 10,000 step の平均報酬で、DDPG は $J$ に関わらず −156.9。UTE-DDPG は $J=2$ で −152.6、$J=8$ で −165.0。t-DDPG は同じく −163.2、−175.3。FiGAR は $J=8$ で −1295.0 まで落ちる。UTE-DDPG は全 $J$ で 1 位か 2 位で、標準偏差も $J=8$ を除いて他より小さいと論文は述べ、不確かさを考えて慎重に繰り返すためと説明する [p.7–8, Robustness to Continuous Control Environment]。

## 結論（Conclusion）

不確かさを明示的に考えて行動を繰り返す手法を提案し、$\lambda$ で度合いを調整することで、Chain MDP、Gridworlds、Atari 2600、制御問題の各環境で他の手法を一貫して大きく上回ったとまとめる。著者らの知る限り、時間的に延長した行動で将来の不確かさを考える最初の深層強化学習アルゴリズムだとする [p.8, Conclusion]。

## 謝辞（Acknowledgments）

Seoul National University の研究プログラムと、韓国の National Research Foundation の助成への謝辞がある [p.8, Acknowledgments]。

## 付録: ベースラインの詳細（Details of Baselines）

- Fixed Repeat: DDQN で常に同じ回数繰り返す。$j=4$ [p.10, Fixed Repeat]。
- $\epsilon z$-Greedy [Dabney et al. (2020)]: 確率 $1-\epsilon$ で現在の方策に 1 step 従い、確率 $\epsilon$ で一様に行動を選んで zeta 分布から引いた長さだけ繰り返す。$\mu$ は Chain MDP と Gridworlds で 1.25（$\lbrace 1.25, 1.5, 2.0, 2.5, 3.0\rbrace$ の探索による）、Atari はゲームごとに $\lbrace 1.5, 1.75, 2.0, 2.25, 2.5\rbrace$ の最良である [p.10, Temporally-Extended ε-Greedy]。
- DAR [Lakshminarayanan et al. (2017)]: 出力を $2\times|\mathcal{A}|$ に複製し、前半は $r_1$ 回、後半は $r_2$ 回繰り返す。繰り返しの値を前もって決める必要があり、行動空間が倍になる [p.10, Dynamic Action Repetition]。
- TempoRL [Biedenkapp et al. (2021)]: behavior 方策と skip 方策が同じ時刻に決める平坦な階層で、behavior を先に引く。behavior 側は 1 step、skip 側は行動を変えないという条件の n-step で学ぶ [p.10, TempoRL]。
- B-DQN [Osband et al. (2016)]: 共有部に $K$ 個のブートストラップ Head をもち、各 Head が自分のサブサンプルで学ぶ。評価は多数決である [p.10, Bootstrapped DQN]。
- FiGAR [Sharma et al. (2017)]: 行動と繰り返しの 2 つの方策が、互いの決定を知らずに選ぶ [p.10, FiGAR]。

## 付録: UTE の実装の詳細（Implementation Details: UTE）

### option 価値のブートストラップとランダム初期化（Bootstrap with random initialization for option-value functions）

多様性を 2 つの仕組みで作る（[Efron (1982)]、[Osband et al. (2016)]）[p.10, Bootstrap with random initialization]。

1. メンバーごとのパラメータのランダム初期化。
2. メンバーごとに異なるサンプルで学ぶこと。各時刻で、メンバー $b$ の目的関数に 2 値のマスク $m_{t,b}$ を掛ける。マスクは Bernoulli 分布から引き、実験ではパラメータ $p=0.5$ を使う。

### 行動価値と option 価値の n-step の目標値（Multi-step target for both action- and option-value functions）

Double Q-learning [Van Hasselt et al. (2016)] を組み込むとしたうえで、損失を次のように書く [p.11, (5)] [p.11, (6)]。

$$L_Q(\theta)=\mathbb{E}_{\tau_t\sim\mathcal{R}}\left[\left(Q(s_t,a_t;\theta)-G_t^{(n)}-\gamma^{n}\max_{a'}Q(s_{t+n},a';\bar{\theta})\right)^2\,\middle|\,n\le j\sim\pi_e\right]$$

$$L_{\tilde{Q}}(\phi)=\mathbb{E}_{\tau_t\sim\mathcal{R}}\left[\sum_{b=1}^{B}m_{t,b}\left(\tilde{Q}_b(s_t,a_t,j;\phi)-G_t^{(j)}-\gamma^{j}\max_{a'}Q(s_{t+j},a';\bar{\theta})\right)^2\right]$$

- $\bar{\theta}$ は Q 側の target で、$\tau$ step ごとに online からコピーする [Mnih et al. (2015)]。
- skip 側の損失はメンバーの和で、平均ではない。目標値は全メンバーで同じである。
- Q 側の $n$ は現在の $\pi_e$ に従う必要がある。延長の長さが $\pi_e$ の出力以下のサンプルだけを使う [p.11, Multi-step target]。

### 同じ目標値を使うこと（The Same Target for both action- and option- value functions）

命題 1 の証明である [p.11, The Same Target]。option 内の方策が決定的に $a$ を返すので、$\tilde{Q}(s,a,j)$ は option 文脈での $a$ の価値 $Q_U(s,o_{a,j},a)$ に等しい（式 (7)）。すると $a$ と $j$ の最大を内外に分けて $\max_a Q^*(s,a)$ が得られる。

## 付録: 実験の詳細（Experiments Details）

全実験は GeForce RTX 3090 のクラスタで走らせ、Atari は 1000 万フレームに 13 時間かかった。手法ごとの実時間は書かれていない。Chain MDP と B-DQN の実装は [Touati et al. (2020)]、Gridworlds と Atari の設定と TempoRL の実装は [Biedenkapp et al. (2021)] に基づく [p.11, Experiments Details]。

Chain MDP [p.11, Chain MDP experiment] [p.12, Table 5]:

- Q 側: 16 ユニット、3 層の全結合、ReLU。
- UTE の skip 側: 同じ構成の 10 個のネットワーク（3 層、26 ユニット）。入力は状態と選んだ行動の連結。
- $\lambda$ の探索範囲: −2、−1、0、1、2。$J$: 10、15、20。$B=10$。ε は 1.0 から 0.001 へ $N\times100$ step。

Gridworlds [p.12, Gridworlds experiment] [p.12, Table 6]:

- Q 側: 50 ユニット、3 層、double DQN。
- TempoRL の skip 側: 3 層のネットワーク 1 個。UTE の skip 側: 同じ形を 10 個複製。入力は状態と選んだ行動の連結。
- $\lambda$ の探索範囲: −1.5、−1.0、−0.5。$J=7$。$B=10$。

Atari [p.12–13, Atari experiment] [p.13, Table 7]:

- 入力は 84×84 の 4 フレーム。DDQN は [Mnih et al. (2015)] の 3 層の畳み込み、512 ユニット、出力。
- TempoRL: 共有の構成で、延長の長さを 10 次元に埋め込み（論文の記述。「論文内の記載の食い違い」の項目 10）、最後の畳み込み層の出力と連結し、512 ユニットの全結合 2 層へ送る。
- UTE: TempoRL に似た構成で、連結したベクトルが 10 個の独立な Head へ分かれる。各 Head は 512 ユニットの全結合と、延長の長さごとの Q 値を出す全結合からなる。
- B-DQN: 同じ胴体に 10 個の Head。
- $\lambda$ の探索範囲: −1.5、−1.0、−0.5、−0.2、0.0、0.2、0.5、1.0。$J=10$。$B=10$。
- replay と延長用の replay はどちらも $5\times10^4$、バッチと延長用のバッチはどちらも 32、target の更新は 500。残りは Table 7。

## 付録: 追加の実験結果（Further Experimental Results）

### Chain MDP（付録）

$\lambda$ の影響 [p.13, Uncertainty Parameter] [p.14, Table 8]: 正の $\lambda$ ほど AUC が高く、鎖が長いほど差が大きい。$N=50$ で $\lambda=-2.0$ は 0.07、$\lambda=0.0$ は 0.37、$\lambda=+2.0$ は 0.67 である。$\lambda=0.0$ と TempoRL を比べると、$N=30$ は 0.62 と 0.74、$N=50$ は 0.37 と 0.25 で、どちらかが常に勝つわけではない（本解説の照合）。

$J$ の影響 [p.13, Maximum Repeat J] [p.15, Figure 6]: $J$ を増やすと繰り返しが増えるが、選択肢が増えて学習が遅くなり、小さすぎても性能を損なう。TempoRL は鎖が長いと $J$ に敏感で、UTE は頑健とする。

### Gridworlds（付録）

Cliff を加えた 3 環境の結果である。離散・決定的で $6\times10$、報酬は疎である [p.14, Gridworlds]。

- $\lambda$ の影響 [p.14, Table 9]: 負に大きいほど AUC が高い傾向で、$\lambda=-1.5$ がすべての組で UTE の最高である。探索範囲の端の値である（本解説の照合）。$\lambda=-0.5$ では、Bridge と ZigZag の対数減衰で $\epsilon z$-Greedy の $\mu=1.25$ を下回る。
- 学習曲線 [p.14, Learning curve] [p.17, Figure 9]: UTE は他より速く収束し、標準偏差も低い。最適でない固定 ε でも頑健である。TempoRL が素の DDQN より悪いことは [Biedenkapp et al. (2021)] の表形式の結果と矛盾し、関数近似では準最適な行動を選びやすいので不確かさを考える必要があると論文はいう。
- 被覆 [p.14, Coverage] [p.16, Figure 7]: TempoRL は DDQN より良くならず、UTE はかなり良い。
- 延長の長さの分布 [p.14, Distribution of extension length] [p.16, Figure 8]: TempoRL は 7 に近い長い延長を UTE より多く選ぶ。$\lambda$ を負に大きくするほど短い延長が増える。

### Atari 2600（付録）

DQN 正規化得点 [p.18, DQN-normalized score]: ゲームごとの最後の 100,000 step の平均報酬から次を計算する。DQN は DDQN を指す。人の得点で正規化しないのは、資源の都合で 1000 万フレームしか学習していないためである。

$$\mathrm{score}=\frac{\mathrm{agent}-\mathrm{random}}{\mathrm{DQN}-\mathrm{random}}$$

Table 10 の平均 [p.18, Table 10]（7 seed）: DDQN 1.00、Fixed-$j$ 0.56、$\epsilon z$-Greedy 1.28、DAR 0.35、TempoRL 1.29、B-DQN 1.03、UTE（1-step）1.34、UTE（n-step）1.81。表のキャプションによれば UTE は最良の $\lambda$ の値である。UTE は DDQN より 81%、TempoRL より 40% 高いとする。

生の得点で 12 ゲームを見ると [p.24, Table 13]、最高は UTE の 3 変種が 9 ゲーム（n-step 7、1-step 1、Adaptive $\lambda$ 1）、TempoRL が Freeway、$\epsilon z$-Greedy が Pong と Riverraid である（本解説の照合）。Adaptive $\lambda$ は TempoRL を 9 ゲームで上回り、$\lambda=0$ には 7 勝 1 分 4 敗（同値は Kangaroo の 661.0）[p.23, Table 12]、最良の固定 $\lambda$ を 3 ゲーム（Freeway、Pong、Sea Quest）で上回る（本解説の照合）。

ゲームごとの最良の $\lambda$ [p.18, Per-game Best Parameter] [p.23, Table 12]: 表の太字（最高値）を $\lambda$ ごとにまとめると次のとおりである（本解説の照合）。本文の記述との違いは「食い違い」の項目 1。

- $\lambda=-1.5$: Kangaroo
- $\lambda=-1.0$: Beam Rider、Qbert、Up n Down
- $\lambda=-0.5$: Crazy Climber、Road Runner
- $\lambda=-0.2$: Centipede、Freeway、Ms Pacman、Pong、Sea Quest
- $\lambda=+1.0$: Riverraid

本解説の照合（Table 12 と Table 13 から計算）:

- $\lambda=0$ の UTE は、TempoRL を 12 ゲーム中 10 ゲームで上回る（例外は Beam Rider と Freeway）。$\lambda=0$ は標準偏差を使わないので、この差は平均の選択、Q 側の n-step、$\pi_e$ の ε なしなど、$\bar{\sigma}$ 以外の違いから来る。ただし手法の違いを 1 つずつ外した比較は論文に無い。
- 最良の $\lambda$ と $\lambda=0$ の差は、12 ゲームすべてで最良の $\lambda$ の列の標準偏差より小さい。差が大きいのは Crazy Climber（8175.6 と 5033.9、標準偏差 5790.4）、Road Runner（12323.2 と 9788.2、標準偏差 4177.1）、Qbert（582.5 と 399.2、標準偏差 558.7）である。表の標準偏差が seed 間のものかはキャプションに明記が無い。
- UTE の列は 8 個の $\lambda$ からの事後の選択である。$\epsilon z$-Greedy も 5 個の $\mu$ からゲームごとに選ぶが、TempoRL、DDQN、DAR、B-DQN にこの種の探索は書かれていない。

バンディットの詳細 [p.18–20, Multi-arm Bandit for Choosing λ]: [Garivier and Moulines (2008)] と [Badia et al. (2020)] の付録 D に基づく。$i$ を腕（$\lambda$ の候補の番号）、$N_{\mathrm{arm}}$ を腕の数、$\tau$ を窓の幅、$\beta$ を探索の係数、$I_k$ をエピソード $k$ で選んだ腕、$R_k$ をその収益とする。窓の中の回数と平均を次で定める [p.19, (8)] [p.19, (9)]。

$$N_k^{\tau}(i)=\sum_{k'=\max(0,k-\tau)}^{k-1}\mathbf{1}(I_{k'}=i)$$

$$\hat{R}_k^{\tau}(i)=\frac{1}{N_k^{\tau}(i)}\sum_{k'=\max(0,k-\tau)}^{k-1}R_{k'}\,\mathbf{1}(I_{k'}=i)$$

式 (9) の和の上限は、原文では $k=1$ と印字されている（窓の無い平均の式も同じ）。式 (8) の上限と揃えて、本解説は $k-1$ に直して書いた（本解説の照合。「食い違い」の項目 15）。

最初の $N_{\mathrm{arm}}$ エピソードは各腕を 1 回ずつ選ぶ。以後は確率 $\epsilon_{ucb}$ で一様に腕を選び、それ以外は次で選ぶ [p.19, Multi-arm Bandit for Choosing λ]。

$$I_k=\arg\max_{i}\left\lbrace\hat{R}_{k-1}^{\tau}(i)+\beta\sqrt{\frac{\log(k-1)}{N_{k-1}^{\tau}(i)}}\right\rbrace$$

Atari の腕の集合は $\lbrace+1.0,+0.5,+0.2,0.0,-0.2,-0.5,-1.0,-1.5\rbrace$ の 8 個である [p.20, Multi-arm Bandit for Choosing λ]。$\tau$、$\beta$、$\epsilon_{ucb}$ の値は読んだ範囲に書かれていない。

Full Ensemble [p.20, Full Ensemble Model] [p.19, Figure 10]: UTE と B-DQN を組み合わせ、Q 側と skip 側の両方をアンサンブルにする。$\lambda$ は調整していない。他と同等以上で、報酬の疎な Road Runner で目立って良いとする。

Fixed Repeat の $j$ [p.20–21, Various Fixed Repeat] [p.20, Figure 11]: 5 seed の平均は図中に $j=2$ で 0.70、$j=4$ で 0.56、$j=8$ で 0.23 と印字され、$j$ が長いほど悪く、DQN の 1.0 に届かない。

アブレーション 1、分解 [p.21, Ablation 1] [p.20, Figure 12a]: 分解しないと探索の空間が $|\mathcal{A}|\times|J|$ になり破局的で、Beam Rider と Road Runner では負の得点になる。図中の平均は分解ありが 1.29、なしが 0.16。

アブレーション 2、Q 側の n-step [p.21, Ablation 2] [p.20, Figure 12b]: 図中の平均は n-step ありが 1.81、なしが 1.39。Road Runner や Centipede のような報酬が比較的疎なゲームで特に効くのは、報酬が速く伝わるためとする。

### DDPG の実装の詳細と追加結果（DDPG Implementation Details and Additional Results）

基の DDPG は公開コード（github.com/sfujim/TD3）で、学習 step 数と初期のランダム step 数だけを変えた。FiGAR と t-DDPG の延長方策には一定の ε-greedy を使い、UTE-DDPG は UCB 型の探索を組み込んでいるので使わない。Q 学習を DDPG の更新に置き換え、基のエージェントの Q 関数を skip 側の学習に使える。UTE-DDPG は特に小さい $J$（2、4）で速く、$J$ を増やしても安定し、t-DDPG より繰り返しが少ないとする [p.21, DDPG Implementation Details and Additional Results] [p.23, Figure 14]。

## 論文が述べている限界と、論文に無いこと

論文が述べていること:

- 繰り返しの最大長は大きければよいわけではなく、学習が遅くなりうる [p.13, Maximum Repeat J]。
- 適応的な $\lambda$ は固定より学習がやや遅い [p.7, Adaptive Uncertainty Parameter λ]。
- 分解は準最適に収束しうる [p.4, Option Decomposition]。
- Atari は資源の都合で 1000 万フレームに留めた [p.18, DQN-normalized score]。

読んだ範囲で論文に無いこと（本解説の照合）:

- メンバーの数 $B$ の感度。すべて 10 である [p.12, Table 5] [p.12, Table 6] [p.13, Table 7]。
- マスクの有無、Head ごとの target の有無を切り替えた実験。同じ目標値の利点は論証で、測定は無い [p.4, Ensemble-based Uncertainty Quantification]。
- $\bar{\sigma}$ が実際の価値の誤差と対応するかの測定。
- 学習と評価で $\lambda$ を変える設定。Algorithm 1 の入力の $\lambda$ は 1 つである [p.3, Algorithm 1]。
- 分布型の価値、PER、延長の途中の打ち切り。

## 論文内の記載の食い違い（本解説の照合）

1. Kangaroo: 本文は Kangaroo で正の $\lambda$ が良いとする [p.18, Per-game Best Parameter]。Table 12 の Kangaroo は $\lambda=-1.5$ の 728.5 が最高で、正の 3 値は 555.2、493.3、543.0 である。正の $\lambda$ が最高なのは Riverraid（$\lambda=+1.0$、938.0）である [p.23, Table 12]。
2. UTE（1-step）の平均: 本文と Figure 12b は 1.39 [p.7, Effect of n-step Learning] [p.20, Figure 12b]、Table 10 は 1.34 [p.18, Table 10]。Table 10 の行の平均を計算すると 1.34 である。30.2% は 1.39 からの計算で、1.34 からなら 35% である。
3. Table 10 の Average: 行の平均を計算すると、UTE（n-step）は 1.78（印字 1.81）、$\epsilon z$-Greedy は 1.26（印字 1.28）、DAR は 0.39（印字 0.35）である。他の列は一致する。81% と 40% は 1.81 による [p.18, Table 10]。
4. Freeway の $\epsilon z$-Greedy: Table 10 は 0.94 だが、Table 13 の 30.8 は DDQN の 27.1 より高く、定義どおりなら 1 を超える [p.18, Table 10] [p.24, Table 13]。
5. Chain MDP の $N=50$: Table 1 の UTE は 0.560、Table 8 の $\lambda=2.0$ は 0.67 である。他の鎖の長さは一致する [p.5, Table 1] [p.14, Table 8]。
6. Chain MDP の比較相手: 本文は DDQN も挙げるが、Table 1 と Table 8 に DDQN の列は無い [p.5, Chain MDP]。
7. Chain MDP の $J$: Table 5 は 10、15、20、Figure 6 は 10、20、30 である [p.12, Table 5] [p.15, Figure 6]。また AUC は全長で 1,000 エピソードとされるが [p.5, Exploration-Favor]、Figure 6 の $N=70$ は 5,000 エピソードまで描かれている。
8. Gridworlds の ε: Table 6 は ε を減らす step 数を 50 とするが、本文の 3 種のスケジュールと合わない [p.12, Table 6] [p.6, Setup]。Table 9 のキャプションは対数減衰と書くが、表には 3 種ある [p.14, Table 9]。
9. DAR の設定: $r_1=J$、$r_2=1$ [p.10, Dynamic Action Repetition] と、$r_1=1$、$r_2=10$ [p.13, Atari experiment] がある。
10. Atari の skip 側の入力: 付録は「延長の長さ」を 10 次元に埋め込むと書く [p.13, Atari experiment]。方策の定義 $\pi_e(j\mid s,a)$ [p.3, Option Decomposition] と、他の環境の入力が状態と行動の連結であること [p.11, Chain MDP experiment] [p.12, Gridworlds experiment] からは、行動の埋め込みを指すと読める（本解説の推論）。TempoRL の Atari の記述にも 10 ユニットの隠れ層があり、特徴量の環境では行動を処理する層である [TempoRL 解説 (2026/10), 4.2.2、付録 H]。
11. 式 (5)(6) の報酬: 和の中が $r_t$ だが、本文の損失は $r_{t+k}$ である [p.11, (5)] [p.4, Multi-step Q-Learning]。本解説は $r_{t+k}$ で書いた。
12. Double Q-learning: 組み込むと書くが、式の目標値は $\bar{\theta}$ の max で書かれている [p.11, Multi-step target]。
13. 記号と語: マスクのパラメータを $\beta$ と $p$ の両方で書く [p.10, Bootstrap with random initialization]。バンディットの腕の $\lambda$ を延長の長さと呼ぶ箇所がある [p.20, Multi-arm Bandit for Choosing λ]。Figure 12b の凡例の手法名は RiSAR である [p.20, Figure 12b]。
14. seed の数: Figure 12 は 5 seed、Table 10 は 7 seed だが、平均の 1.81 と 1.29 が同じ値で出る [p.20, Figure 12] [p.18, Table 10]。
15. バンディットの平均の和の上限: 式 (9) と窓の無い平均の式は $k=1$ と印字され、回数の式（式 (8) と窓の無い式）は $k-1$ である [p.19, (8)] [p.19, (9)]。本解説は式 (9) を $k-1$ に直して書いた。

## キューの論点への答え

依頼時のサーベイキュー（reports/INDEX.md の優先 14）の論点メモへの答えである。その行は完了時に消えて参照先が無いので、論点をここに写す。(1) skip 価値のアンサンブルの作り（メンバーの多様性の出し方、目標値を全メンバーで共有するか）。(2) 平均と標準偏差で j を選ぶ式と、λ の符号の意味と決め方（固定か自動調整か）。(3) TempoRL との差分と、比べた env。

### 1. skip 価値のアンサンブルの作り

- メンバーの数: すべての実験で 10 [p.12, Table 5] [p.12, Table 6] [p.13, Table 7]。
- 多様性の出し方: ランダム初期化と、Bernoulli（$p=0.5$）の 2 値マスクの 2 つ [p.10, Bootstrap with random initialization]。事前関数（randomized prior）の記述は無い。マスクの実体は損失への掛け算である [p.11, (6)]。
- 形: Chain MDP と Gridworlds はネットワークごと 10 個 [p.11, Chain MDP experiment] [p.12, Gridworlds experiment]。Atari は共有の胴体から 10 個の Head に分かれ、各 Head は 512 ユニットの全結合と出力の 2 層である [p.13, Atari experiment]。
- 目標値: 全メンバーで共有する。終点の価値は Q 側の target network の $\max_{a'}Q(s_{t+j},a';\bar{\theta})$ である [p.11, (6)]。理由は、Head ごとの target だと別のモードへ収束し、$\bar{\sigma}$ が 0 へ収束しないため、という論証である [p.4, Ensemble-based Uncertainty Quantification]。
- 損失: マスク付きのメンバーの和 [p.11, (6)]。
- 分からないこと: マスクが遷移ごとに固定か、更新のたびに引き直すか（明記は無いが、添字が $m_{t,b}$ なので遷移ごとと読む余地がある）。共有の胴体への勾配の調整の有無。

### 2. 平均と標準偏差で j を選ぶ式と λ

- 式: $j=\arg\max_{j'}\lbrace\bar{\mu}(s,a,j')+\lambda\bar{\sigma}(s,a,j')\rbrace$。$a$ は先に $Q$ の ε-greedy で選ぶ [p.3, Algorithm 1] [p.4, Ensemble-based Uncertainty Quantification]。
- $\pi_e$ は決定的で、ε を使わない [p.4, Ensemble-based Uncertainty Quantification] [p.5, Exploration-Favor]。
- 符号: 正が積極的な探索、負が不確かさを避ける探索 [p.4, Ensemble-based Uncertainty Quantification]。負にすると短い延長が増える [p.6, Figure 3] [p.16, Figure 8]。
- 決め方: 固定と自動調整の 2 通りがある。固定は環境ごとの探索で最良を選ぶ（Chain MDP は $+2.0$、Gridworlds は $-1.5$、Atari はゲームごと）[p.5, Exploration-Favor] [p.6, Coverage] [p.23, Table 12]。自動調整はエピソードごとの非定常バンディットで、報酬はエピソードの収益、スライディングウィンドウ UCB と $\epsilon_{ucb}$-greedy を使う [p.4–5, Adaptive Uncertainty Parameter] [p.19, Multi-arm Bandit for Choosing λ]。自動調整を使ったのは Atari と Pendulum-v0 である [p.7, Adaptive Uncertainty Parameter λ] [p.7, Control Problem: Pendulum-v0]。Chain MDP の $+2.0$ と Gridworlds の $-1.5$ はどちらも探索範囲の端である（本解説の照合）。

### 3. TempoRL との差分と比べた env

TempoRL の列は本論文の記述 [p.5, Exploration-Favor] [p.10, TempoRL] と [TempoRL 解説 (2026/10), Algorithm 1・付録 B] による。

| 観点 | TempoRL | UTE |
|---|---|---|
| skip 側の価値 | 1 組 | $B=10$ 組（初期化とマスクで多様化） |
| $j$ の選び方 | skip 価値に ε-greedy | $\bar{\mu}+\lambda\bar{\sigma}$ の argmax。ε なし |
| skip 側の目標値の終点 | Q 側の $\max_{a'}Q$ | 同じ。全メンバーで共有 |
| Q 側の学習 | 1 step | $n\le j$ の n-step |
| 行動と $j$ の分解 | 行動が先 | 同じ |
| Atari のネットワーク | 共有の構成 | 同じで、延長の流れを 10 Head にする |

終点の価値を Q 側で測ることは TempoRL にもあり、UTE の新しさは全メンバーでの共有である。論文が同じ目標値の比較相手に挙げるのは B-DQN である（本解説の照合）[p.4, Ensemble-based Uncertainty Quantification] [TempoRL 解説 (2026/10), キューの論点への答え 2]。

比べた env [p.5–8]: Chain MDP（$N$ = 10、30、50、70）、Lava の Gridworlds（Bridge、ZigZag、付録に Cliff。$6\times10$、$J=7$）、Atari 12 ゲーム（$2.5\times10^6$ step、$J=10$、7 seed）、Pendulum-v0（$J$ = 2、4、6、8）。TempoRL が比較に入るのは全 env で、Pendulum-v0 では t-DDPG である。

## PRD958 の検討に効く点

[PRD958](../docs/memo/958_temporl_skip_followups_10prd.md) の §9 の 1 に照らした、本解説の読みである。論文の主張ではない。PRD は変更していない。

合っていた点:

- $\bar{\mu}+\lambda\bar{\sigma}$ の形と $\lambda$ の符号（D3）、全メンバーで同じ目標値（D2）は論文と同じである。
- D5 の割り込みの閾値は Lazy-MDP の話で、この論文では確かめられない。UTE は延長の途中で打ち切らない。

直す候補:

1. D2 の「メンバーの違いは初期値だけ」は論文の構成より弱い。論文は初期値に加えて $p=0.5$ のマスクを使う。§8 で先送りした「bootstrap の印」は論文の標準の構成で、「事前関数」は論文に無い。初期値だけにするなら、anet の簡略化だと書く。マスクを外した実験は論文に無いので、初期値だけで $\bar{\sigma}$ が保てるかは分からない。
2. D3 の「既存の skip の方策に渡す」は論文と違う。論文の $j$ は argmax で ε を使わない。現行の GridMaze の `run.@skip` は学習の Actor が ε-greedy（0.60→0.05）なので、論文の再現には skip の方策を greedy にする群が要る。UQE に渡す形は論文に例が無く、§7 のステージ 0 の結果（UQE の楽観）とも重なる。
3. §3・§6 の 2・§7 の群の組み方は、負の $\lambda$ を評価の Actor にだけ置く。これは論文に根拠が無い。D3 と §5.1 のキーは `actor.[key]` なので学習の Actor にも負を書けるが、D3 の例も学習は 0 か正、評価は負である。論文の $\lambda$ は 1 つで、学習中の行動を変える。短い延長や被覆の改善は学習中の悲観で説明されている。§7 の U のステージに、学習と評価の両方を負にする群を足す候補がある。値は Lava の $-1.5$ が参考になる。
4. §6 の 1 の「UTE の範囲（5〜10）」は論文で確かめられない。$B$ は 10 だけで感度も無い。$E=5$ は anet の費用の判断と書き直す。
5. §5.5 の「最終層が $E$ 倍」は論文より軽い構成である。論文の Atari の Head は 512 ユニットの全結合と出力の 2 層、小さい env はネットワークごとである。多様性が足りなければ費用の見積もりと受入 4 の 1.1 倍に響く。
6. D4 のメンバー平均の loss は、論文のマスク付きの和と勾配の大きさが違う。論文は調整に触れていない。

判断の材料:

7. §8 の $\lambda$ の自動調整の再開条件（env ごとに最良の値が大きく違う）の根拠になるのは、Chain MDP の $+2.0$ と Gridworlds の $-1.5$ である。Atari のゲームごとの差は項目 8 のとおり標準偏差の内なので、根拠にしない。論文の自動調整は $\lambda=0$ に対して 12 ゲームで 7 勝 1 分 4 敗である。最良の固定値を上回るのは 3 ゲームだが、この相手は 8 値からの事後の最大である。$\tau$、$\beta$、$\epsilon_{ucb}$ の値は無い。エピソード単位で選ぶので、多数の lane を並べる anet にはそのまま載らない。
8. Atari で $\bar{\sigma}$ の効果を大きく見込まないほうがよい。Table 10 の平均は TempoRL 1.29、UTE（1-step、最良の $\lambda$）1.34、UTE（n-step）1.81 で、伸びの大半は Q 側の n-step の側にある。最良の $\lambda$ の上乗せも 12 ゲームすべてで標準偏差の内で、8 値からの事後の選択である。学習も 1000 万フレームなので、50M の qbert の比較（[実験記録](../docs/experiments/default-dqn/atari/2026-10-31_temporl-qbert-50m-learning.md)）には直接使えない。
9. Q 側の $n\le j$ の n-step は §3 の非ゴール（primary の学習を変えない）の外にある。論文の Atari の伸びの主因に見えるので、取り込むなら別 PRD の候補になる。[914](../docs/memo/914_decision_aligned_nstep_10prd.md) は「判断」の定義が別である。
10. D11 の判断（(U) を作らない）を覆す材料ではない。論文の Lava で TempoRL が DDQN を大きく下回る結果は過大評価で説明されているが、anet の 5×5 は skip の方策を ε-greedy にするだけで届いた（§7）。ε-greedy の平均と負の $\lambda$ を同じ条件で比べた実験は論文に無いので、(U) を再開するときはこの比較が要る。

## 本解説で省略した内容と、確かめられなかった点

省略した内容:

- Table 1〜13 の値のうち、議論に使わないもの。とくに Table 7 の残りの値、Table 11 の全値、Table 12 と Table 13 の大部分。
- 図の曲線の読み取り（Figure 5、6、9、10、13、14）。
- 参考文献の個別の内容。

確かめられなかった点:

- AAAI の公式版: ojs.aaai.org の記事ページがボット対策で開けず、v1 との差は確かめていない。巻号とページは ML Anthology と Web 検索の結果による。
- $\tau$、$\beta$、$\epsilon_{ucb}$ の値: 読んだ範囲に無い。
- マスクを遷移ごとに固定するか、更新ごとに引き直すか: 明記が無い。添字が $m_{t,b}$ なので遷移ごとと読む余地はある [p.10, Bootstrap with random initialization]（本解説の推論）。
- 評価時の $\lambda$: 固定の場合は 1 つの $\lambda$ を使うと読めるが明記は無く、適応的な場合に評価でどの値を使うかは書かれていない。
- Table 3、12、13 の標準偏差の定義と、「最後の 100,000 step」が評価の値か学習の値か: 明記が無い。
- Gridworlds の AUC が学習の値か評価の値か: 明記が無い。Chain MDP の Figure 6 は学習の報酬である [p.15, Figure 6]。
- UTE の公開コードの有無: 論文内にリンクは無く、検索 1 回でも見つからなかったが、公開されていないとは確かめていない。

## 出典リスト

本解説の対象:

[Lee et al., 2024/02] Lee, J., Park, S. J., Tang, Y., and Oh, M. "Learning Uncertainty-Aware Temporally-Extended Actions." Proceedings of the AAAI Conference on Artificial Intelligence, 38(12), 13391–13399（本文中では位置のみで参照。読んだのは arXiv v1）. https://arxiv.org/abs/2402.05439

書誌とリポジトリ内の資料:

[arXiv 2402.05439, 2024/02] arXiv abstract page（題名、著者、投稿日、コメント、分野、ライセンスを確認）. https://arxiv.org/abs/2402.05439

[ML Anthology, 2024] "Learning Uncertainty-Aware Temporally-Extended Actions." ML Anthology（AAAI の書誌、ページ、DOI を確認）. https://mlanthology.org/aaai/2024/lee2024aaai-learning

[TempoRL 解説, 2026/10] 本リポジトリの論文解説（TempoRL の目標値、方策、Atari の構成の照合に使用）. ../reports/biedenkapp2021_temporl_paper_2026-10-03.md

[PRD958, 2026/11] 本リポジトリの PRD「TempoRL の後続」（最後から 3 つ目の節でのみ参照）. ../docs/memo/958_temporl_skip_followups_10prd.md

論文が引用する文献（書誌は論文の参考文献 [p.8–9, References] による。原典は読んでいない。URL は論文に無いので載せない）:

[Anschel et al., 2017] Anschel, O., Baram, N., and Shimkin, N. "Averaged-DQN: Variance reduction and stabilization for deep reinforcement learning." ICML, 176–185.

[Audibert et al., 2009] Audibert, J.-Y., Munos, R., and Szepesvári, C. "Exploration–exploitation tradeoff using variance estimates in multi-armed bandits." Theoretical Computer Science, 410(19): 1876–1902.

[Auer et al., 2002] Auer, P., Cesa-Bianchi, N., and Fischer, P. "Finite-time analysis of the multiarmed bandit problem." Machine Learning, 47: 235–256.

[Bacon et al., 2017] Bacon, P.-L., Harb, J., and Precup, D. "The option-critic architecture." AAAI, volume 31.

[Badia et al., 2020] Badia, A. P., Piot, B., Kapturowski, S., Sprechmann, P., Vitvitskyi, A., Guo, Z. D., and Blundell, C. "Agent57: Outperforming the Atari human benchmark." ICML, 507–517.

[Bai et al., 2021] Bai, C., Wang, L., Han, L., Hao, J., Garg, A., Liu, P., and Wang, Z. "Principled exploration via optimistic bootstrapping and backward induction." ICML, 577–587.

[Barreto et al., 2019] Barreto, A., Borsa, D., Hou, S., Comanici, G., Aygün, E., Hamel, P., Toyama, D., Mourad, S., Silver, D., Precup, D., et al. "The option keyboard: Combining skills in reinforcement learning." NeurIPS 32.

[Bellemare et al., 2013] Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M. "The arcade learning environment: An evaluation platform for general agents." JAIR, 47: 253–279.

[Bellemare et al., 2016] Bellemare, M., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., and Munos, R. "Unifying count-based exploration and intrinsic motivation." NeurIPS 29.

[Biedenkapp et al., 2021] Biedenkapp, A., Rajan, R., Hutter, F., and Lindauer, M. "TempoRL: Learning When to Act." ICML 2021.

[Brockman et al., 2016] Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., and Zaremba, W. "OpenAI Gym." arXiv:1606.01540.

[Chen et al., 2017] Chen, R. Y., Schulman, J., Abbeel, P., and Sidor, S. "UCB and infogain exploration via Q-ensembles." arXiv:1706.01502.

[Da Silva et al., 2020] Da Silva, F. L., Hernandez-Leal, P., Kartal, B., and Taylor, M. E. "Uncertainty-aware action advising for deep reinforcement learning agents." AAAI, volume 34, 5792–5799.

[Dabney et al., 2020] Dabney, W., Ostrovski, G., and Barreto, A. "Temporally-Extended ε-Greedy Exploration." ICLR 2021.

[Dayan and Hinton, 1992] Dayan, P., and Hinton, G. E. "Feudal reinforcement learning." NeurIPS 5.

[Efron, 1982] Efron, B. "The jackknife, the bootstrap and other resampling plans." SIAM.

[Fikes et al., 1972] Fikes, R. E., Hart, P. E., and Nilsson, N. J. "Learning and executing generalized robot plans." Artificial Intelligence, 3: 251–288.

[Garivier and Moulines, 2008] Garivier, A., and Moulines, E. "On upper-confidence bound policies for non-stationary bandit problems." arXiv:0805.3415.

[Harutyunyan et al., 2016] Harutyunyan, A., Bellemare, M. G., Stepleton, T., and Munos, R. "Q(λ) with Off-Policy Corrections." ALT, 305–320.

[Lakshminarayanan et al., 2017] Lakshminarayanan, A., Sharma, S., and Ravindran, B. "Dynamic action repetition for deep reinforcement learning." AAAI, volume 31.

[Lee et al., 2022] Lee, S., Seo, Y., Lee, K., Abbeel, P., and Shin, J. "Offline-to-online reinforcement learning via balanced replay and pessimistic Q-ensemble." CoRL, 1702–1712.

[Lillicrap et al., 2015] Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., and Wierstra, D. "Continuous control with deep reinforcement learning." arXiv:1509.02971.

[Machado et al., 2018] Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M. J., and Bowling, M. "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." JAIR, 61: 523–562.

[Machado et al., 2021] Machado, M. C., Barreto, A., and Precup, D. "Temporal Abstraction in Reinforcement Learning with the Successor Representation." arXiv:2110.05740.

[Meng et al., 2021] Meng, L., Gorbet, R., and Kulić, D. "The effect of multi-step methods on overestimation in deep reinforcement learning." ICPR, 347–353.

[Metelli et al., 2020] Metelli, A. M., Mazzolini, F., Bisi, L., Sabbioni, L., and Restelli, M. "Control frequency adaptation via action persistence in batch reinforcement learning." ICML 2020, 6862–6873.

[Mnih et al., 2015] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. "Human-level control through deep reinforcement learning." Nature, 518(7540): 529–533.

[Moskovitz et al., 2021] Moskovitz, T., Parker-Holder, J., Pacchiano, A., Arbel, M., and Jordan, M. "Tactical optimism and pessimism for deep reinforcement learning." NeurIPS 34.

[Osband et al., 2016] Osband, I., Blundell, C., Pritzel, A., and Van Roy, B. "Deep exploration via bootstrapped DQN." NeurIPS 29, 4026–4034.

[Park et al., 2021] Park, S., Kim, J., and Kim, G. "Time Discretization-Invariant Safe Action Repetition for Policy Gradient Methods." NeurIPS 34.

[Parr and Russell, 1997] Parr, R., and Russell, S. "Reinforcement learning with hierarchies of machines." NeurIPS 10.

[Peer et al., 2021] Peer, O., Tessler, C., Merlis, N., and Meir, R. "Ensemble bootstrapping for Q-Learning." ICML, 8454–8463.

[Peng and Williams, 1994] Peng, J., and Williams, R. J. "Incremental multi-step Q-learning." Machine Learning Proceedings 1994, 226–232.

[Precup, 2000] Precup, D. "Temporal abstraction in reinforcement learning." University of Massachusetts Amherst.

[Puterman, 2014] Puterman, M. L. "Markov decision processes: discrete stochastic dynamic programming." John Wiley & Sons.

[Schoknecht and Riedmiller, 2002] Schoknecht, R., and Riedmiller, M. "Speeding-up reinforcement learning with multi-step actions." International Conference on Artificial Neural Networks, 813–818.

[Sharma et al., 2017] Sharma, S., Srinivas, A., and Ravindran, B. "Learning to repeat: Fine grained action repetition for deep reinforcement learning." ICLR 2017.

[Stolle and Precup, 2002] Stolle, M., and Precup, D. "Learning options in reinforcement learning." International Symposium on Abstraction, Reformulation, and Approximation, 212–223.

[Sutton, 1988] Sutton, R. S. "Learning to predict by the methods of temporal differences." Machine Learning, 3(1): 9–44.

[Sutton et al., 1999] Sutton, R. S., Precup, D., and Singh, S. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." Artificial Intelligence, 112(1-2): 181–211.

[Thrun and Schwartz, 1993] Thrun, S., and Schwartz, A. "Issues in using function approximation for reinforcement learning." Proceedings of the 1993 Connectionist Models Summer School, volume 6.

[Touati et al., 2020] Touati, A., Satija, H., Romoff, J., Pineau, J., and Vincent, P. "Randomized value functions via multiplicative normalizing flows." UAI, 422–432.

[Van Hasselt et al., 2016] Van Hasselt, H., Guez, A., and Silver, D. "Deep reinforcement learning with double Q-learning." AAAI, volume 30.

[Watkins, 1989] Watkins, C. J. C. H. "Learning from delayed rewards."

[Watkins and Dayan, 1992] Watkins, C. J., and Dayan, P. "Q-learning." Machine Learning, 8(3): 279–292.

[Xia and Collins, 2021] Xia, L., and Collins, A. G. "Temporal and state abstractions for efficient learning, transfer, and composition in humans." Psychological Review.

[Zahavy et al., 2018] Zahavy, T., Haroush, M., Merlis, N., Mankowitz, D. J., and Mannor, S. "Learn what not to learn: Action elimination with deep reinforcement learning." NeurIPS 31.
