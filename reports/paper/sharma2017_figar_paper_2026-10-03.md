# Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning — 詳細解説

- 著者: Sahil Sharma, Aravind S. Lakshminarayanan, Balaraman Ravindran [p.1]。arXiv の書誌ページでは第 2 著者の名前が Aravind Srinivas と表示される [arXiv 1702.06054 (2020/09)]。
- 所属: Indian Institute of Technology, Madras（インド、チェンナイ）[p.1]
- 発表: ICLR 2017 の会議論文 [p.1]
- 年: 2017
- arXiv: 1702.06054。v1 は 2017 年 2 月 20 日、v2 は 2020 年 9 月 21 日の投稿である [arXiv 1702.06054 (2020/09)]。本書は v2 の PDF（24 ページ）を読んだ。
- URL: https://arxiv.org/abs/1702.06054 。ICLR 2017 の OpenReview のページは https://openreview.net/forum?id=B1GOWV5eg である [ML Anthology (2017)]。

Date: 2026-10-03

表記の約束:

- 本書は論文の文章を逐語では転載しない。reports/ の論文解説の取り決め（著作物の逐語の引用は 1 文書につき 1 か所・15 語未満）に従ったもので、paper-details スキルが求める「原文の引用と訳のブロックを主体にする」形式からは外れている。各節の内容は、論文の位置を `[p.X, Section Y.Z]` の形で示したうえで、日本語で要約・説明した。逐語の引用は「要旨（Abstract）」の節の 1 か所だけで、要旨の全文の引用と全訳もしていない。原文は上の URL で確認できる。論文は arXiv の非独占的配布ライセンスで公開されている [arXiv 1702.06054 (2020/09)]。
- 図表は転載せず、番号とページで参照し（例: [p.7, Figure 2]、[p.13, Table 4]）、何を示すかを本書の言葉で説明した。議論に必要な数値だけを位置引用つきで本文に書き、表全体は転記していない。図表そのものは論文 PDF（[papers/sharma2017_figar.pdf](../papers/sharma2017_figar.pdf)）で確認できる。
- 式は LaTeX で、論文の記法に合わせて書いた。数値は論文の表記どおりに転記した。
- 本書が独自に行った計算・照合・読み取りは、そのことを明記して論文の主張と区別した。論文内で記載が食い違う箇所は「論文内の記載の食い違い（本書による照合）」の節にまとめ、本文からは「食い違いの項目 N」の形で参照する。参考文献の書誌の誤りは、出典リストの各項目の注記と、同じ節の末尾に記す。
- 論文が引用する他の文献は `[著者 (年)]` の形で示し、末尾の出典リストに個別に載せた。出典リストのラベルで日付が年だけのものは、月を特定していない。

## 要旨（Abstract）

要旨によれば、強化学習のアルゴリズムは、環境と相互作用して報酬という形のフィードバックを得ながら、逐次的な意思決定のための複雑な行動パターンを学べる。ただし従来のアルゴリズムは、行動の選択を相互作用の時間ステップごとに行ってきた [p.1, Abstract]。論文はこれに対して新しい枠組み FiGAR（Fine Grained Action Repetition）を提案する。FiGAR のエージェントが決めるものを、要旨は次のように書く。

```
"decide the action as well as the time scale of repeating it"

「行動に加えて、その行動を繰り返す時間スケールも決める」
```
[p.1, Abstract]

FiGAR は、行動空間での時間的抽象化を可能にすることで、方策の推定値を明示的に保持する深層強化学習（DRL）アルゴリズムであればどれでも改善できるとされる。有効性は、異なる領域の 3 つの方策探索アルゴリズムに FiGAR を組み合わせたときの性能向上として、実験で示される。Atari 2600 での Asynchronous Advantage Actor Critic（A3C）、MuJoCo での Trust Region Policy Optimization（TRPO）、自動車レースの TORCS での Deep Deterministic Policy Gradients（DDPG）である [p.1, Abstract]。

## Table of Contents

- [要旨（Abstract）](#要旨abstract)
- [1. はじめに（Introduction）](#1-はじめにintroduction)
- [2. 関連研究（Related Work）](#2-関連研究related-work)
- [3. 背景（Background）](#3-背景background)
  - [3.1 A3C（Asynchronous Advantage Actor Critic）](#31-a3casynchronous-advantage-actor-critic)
  - [3.2 TRPO（Trust Region Policy Optimization）](#32-trpotrust-region-policy-optimization)
  - [3.3 DDPG（Deep Deterministic Policy Gradients）](#33-ddpgdeep-deterministic-policy-gradients)
- [4. FiGAR: 細粒度の行動反復（Fine Grained Action Repetition）](#4-figar-細粒度の行動反復fine-grained-action-repetition)
  - [4.1 FiGAR の動作（How FiGAR operates）](#41-figar-の動作how-figar-operates)
  - [4.2 FiGAR-A3C](#42-figar-a3c)
  - [4.3 FiGAR-TRPO](#43-figar-trpo)
  - [4.4 FiGAR-DDPG](#44-figar-ddpg)
- [5. 実験設定と結果（Experimental Setup and Results）](#5-実験設定と結果experimental-setup-and-results)
  - [5.1 Atari 2600 での FiGAR-A3C（FiGAR-A3C on Atari 2600）](#51-atari-2600-での-figar-a3cfigar-a3c-on-atari-2600)
  - [5.2 MuJoCo タスクでの FiGAR-TRPO（FiGAR-TRPO on Mujoco Tasks）](#52-mujoco-タスクでの-figar-trpofigar-trpo-on-mujoco-tasks)
  - [5.3 TORCS での FiGAR-DDPG（FiGAR-DDPG on TORCS）](#53-torcs-での-figar-ddpgfigar-ddpg-on-torcs)
  - [5.4 行動反復集合の影響（Effect of Action Repetition Set on FiGAR）](#54-行動反復集合の影響effect-of-action-repetition-set-on-figar)
- [6. 結論・欠点・今後の課題（Conclusion, Shortcomings and Future Work）](#6-結論欠点今後の課題conclusion-shortcomings-and-future-work)
- [謝辞（Acknowledgments）](#謝辞acknowledgments)
- [付録 A. FiGAR-A3C の実験詳細（Appendix A）](#付録-a-figar-a3c-の実験詳細appendix-a)
  - [A3C 実験の詳細と結果（Experimental details and results）](#a3c-実験の詳細と結果experimental-details-and-results)
  - [A3C のアーキテクチャの詳細（Architecture details）](#a3c-のアーキテクチャの詳細architecture-details)
- [付録 B. Atari 2600 の追加実験（Appendix B）](#付録-b-atari-2600-の追加実験appendix-b)
  - [貪欲か確率的か（The choice of whether to be greedy or stochastic）](#貪欲か確率的かthe-choice-of-whether-to-be-greedy-or-stochastic)
  - [性能と速度のトレードオフ（Performance versus speed tradeoff）](#性能と速度のトレードオフperformance-versus-speed-tradeoff)
- [付録 C. FiGAR-TRPO の実験設定（Appendix C）](#付録-c-figar-trpo-の実験設定appendix-c)
  - [TRPO 実験の詳細（Experimental details）](#trpo-実験の詳細experimental-details)
  - [損失関数とアーキテクチャ（Loss function and Architecture）](#損失関数とアーキテクチャloss-function-and-architecture)
- [付録 D. FiGAR-DDPG の実験詳細（Appendix D）](#付録-d-figar-ddpg-の実験詳細appendix-d)
  - [DDPG 実験の詳細（Experimental details）](#ddpg-実験の詳細experimental-details)
  - [DDPG のアーキテクチャの詳細（Architectural details）](#ddpg-のアーキテクチャの詳細architectural-details)
- [付録 E. FiGAR の変種の詳細（Appendix E）](#付録-e-figar-の変種の詳細appendix-e)
- [付録 F. 反復方策の重要性（Appendix F）](#付録-f-反復方策の重要性appendix-f)
- [付録 G. FiGAR-TRPO の共有表現実験（Appendix G）](#付録-g-figar-trpo-の共有表現実験appendix-g)
- [論文が述べている限界](#論文が述べている限界)
- [論文内の記載の食い違い（本書による照合）](#論文内の記載の食い違い本書による照合)
- [キューの論点への答え](#キューの論点への答え)
- [出典リスト](#出典リスト)

## 1. はじめに（Introduction）

強化学習（RL）は、正しい決定を教師として与えられず、報酬という評価的なフィードバックだけから目標指向の逐次的意思決定を学ぶ枠組みで、問題をマルコフ決定過程（MDP）としてモデル化する [Sutton & Barto (1998)]。状態空間が指数的に大きいか連続である問題では、状態ごとに推定値を持つ表形式の手法が使えず、未知の状態への汎化が要る。論文は、深層ニューラルネットワークの表現学習 [LeCun et al. (2015)] と強化学習を組み合わせた手法が、Atari 2600 [Bellemare et al. (2013); Mnih et al. (2015); Schaul et al. (2015); Mnih et al. (2016)]、MuJoCo の物理シミュレーション課題 [Todorov et al. (2012); Lillicrap et al. (2015)]、ロボサッカー [Hausknecht et al. (2016)]、TORCS [Wymann et al. (2000); Mnih et al. (2016)] で有望な結果を示してきたことを出発点にする [p.1, Section 1]。

論文は、これらの深層強化学習（DRL）アルゴリズムに共通する特徴として、選んだ行動を固定の $k$ ステップのあいだ繰り返し実行することを挙げる [p.1, Section 1]。時刻 $t$ に取る行動を $a_t$、$k$ ステップごとの区切りの番号を $i$ とすると、これは次のように書ける。この $k$ はこの節だけの記号で、3.1 節では学習スレッドの数、4 節以降では行動決定の番号という別の意味で使われる。

$$a_1 = a_2 = \cdots = a_k,\qquad a_{k+1} = a_{k+2} = \cdots = a_{2k},\qquad a_{ik+1} = a_{ik+2} = \cdots = a_{(i+1)k}\quad (i \ge 0)$$

[p.1, Section 1]

論文は行動反復の利点として次の点を挙げる [p.1, Section 1]。

- 行動を $k$ ステップに 1 回だけ計算すればよいので高速に動き、リアルタイムで動作できる。
- ほかの利点の例として、行動の方策がなめらかになる。
- より重要な点として、同じ行動を $k$ 回繰り返すマクロ行動は方策に時間的抽象化を持ち込むものと解釈でき、時間的に離れた有利な状態のあいだの遷移を可能にする。これは [Lakshminarayanan et al. (2016)] と [Durugkar et al. (2016)] が示したとされる。

DRL の行動反復の時間スケールは、これまでほぼ固定されてきた [Mnih et al. (2015); Mnih et al. (2016); Schaul et al. (2015)]。DRL で時間スケールを動的に選ぶことを初めて試みた [Lakshminarayanan et al. (2016)] は、いくつかの Atari 2600 のゲームで大きな性能向上を示したが、時間スケールは 2 つだけ、実験は代表的な少数のゲームに限られ、手法は離散行動空間の課題に限定される [p.2, Section 1]。

FiGAR は、行動を選ぶ方策と行動反復回数を選ぶ方策を独立に分けた、構造化・因子分解された方策の表現によって、行動空間が連続か離散かによらず、どの DRL アルゴリズムにも、時間的に引き延ばしたマクロ行動の形で時間的抽象化を学ばせる枠組みである。論文は、2 つを独立に決めるので、[Vezhnevets et al. (2016)] や [Lakshminarayanan et al. (2016)] と違って行動空間が膨らまないと指摘する [p.2, Section 1]。

論文が挙げる貢献は 2 つである [p.2, Section 1]。

- 時間的抽象化のための因子分解された方策表現により、DRL アルゴリズムの汎用的な拡張を提案した。2 つの Atari 2600 のゲームで学習されたマクロ行動の列を図 1 [p.2, Figure 1] に示す。
- FiGAR が方策勾配型の DRL アルゴリズムの性能を改善することを、複数の領域で実証した。A3C [Mnih et al. (2016)] による 31 の Atari 2600 のゲーム、TRPO [Schulman et al. (2015)] による MuJoCo の 5 課題、DDPG [Lillicrap et al. (2015)] による TORCS である。Atari のゲーム数は 5.1 節と付録 A では 33 である（食い違いの項目 1）。

図 1 [p.2, Figure 1] は、FiGAR が学習した方策に現れる時間的抽象化の例を、Freeway と Seaquest の連続したゲーム画面で示す。画面のあいだの矢印は実行した行動、数字はその行動を繰り返したステップ数を表し、雷の印は発射、矢印と雷の組は移動しながらの発射を表す。Freeway では、車が通り過ぎるまで下向きの行動（この場面では何もしないのと同じ）で待ち、その後は長い反復で道路を渡る。Seaquest では、見つけた敵に長い反復で近づいて倒す [p.2, Figure 1]。図中の反復回数は、Freeway の例で 16〜29、Seaquest の例で 7〜18 である（本書による図の読み取り）。

## 2. 関連研究（Related Work）

FiGAR の中心にあるのは、必要なときにだけ決めるという一般的な考え方である。RL がふつう扱う領域の外にも似た発想があり、リアルタイムのニューラル機械翻訳では、各ステップで現在の文脈にもとづいて目標言語の新しいトークンを出すかどうかを決める [Gu et al. (2016); Satija & Pineau (2016)] [p.2, Section 2]。論文は他の研究との関係を次のように整理する [p.3, Section 2]。

- Transition Point Dynamic Programming（TPDP）[Buckland & Lawrence (1994)]: 連続な確率的動的システムの制御に要する学習時間とメモリを減らすよう、表形式の動的計画法を改変したもの。方策は遷移点と呼ぶ状態でだけ変わる。TPDP は、ある状態を遷移点の集合に加えるか除くかを Q 学習の変種で評価して、最適な遷移点の集合を学ぶ。FiGAR は遷移点をその場で学び、状態空間全体へ汎化するのに対し、TPDP は表形式で大規模な問題には使えない。
- Dynamic Frameskip Deep Q-network [Lakshminarayanan et al. (2016)]: DQN [Mnih et al. (2015)] に、時間スケールごとに同じ基本行動の出力の流れを別々に加えて、反復の時間スケールを動的に学ぶ。いくつかの Atari 2600 のゲームで大きく性能が上がったが、行動空間の膨張のおそれから多数の時間スケールを扱えず、離散行動空間に限られる。[Durugkar et al. (2016)] も同じ行動を異なる時間スケールで繰り返すマクロ行動を学ぶが、離散行動空間に限られ、性能向上も大きくない。
- Strategic Attentive Writer（STRAW）[Vezhnevets et al. (2016)]: マクロ行動と動的な行動計画を、強化学習の信号から直接学ぶ手法。観測ごとに 1 つの行動を出すのではなく複数ステップの行動計画を保持し、観測にもとづいて定期的に計画を更新し、再計画のあいだは計画に従う。論文は、STRAW は FiGAR より一般的な時間的抽象化を表すとしたうえで、FiGAR は STRAW を補完でき、反復の判断を計画の水準と基本行動の水準で階層的にできるとする。
- パラメータ化された方策 [Hausknecht & Stone (2016)]: DRL でパラメータ化された方策を扱った唯一の他の研究とされ、離散行動を連続値でパラメータ化する。FiGAR では実行の時間スケールが選んだ行動をパラメータ化するものと見なせ、離散または連続の行動を離散値でパラメータ化する。論文は、Atari の状態空間は [Hausknecht et al. (2016)]（1 節でロボサッカーの領域の例に挙がる Half Field Offense の文献で、[Hausknecht & Stone (2016)] とは別のもの）で扱われたものより複雑だとする。
- セミマルコフ決定過程（SMDP）[Duff (1995); Mahadevan et al. (1997); Dietterich (2000)]: 行動の完了に保持時間（holding time）がかかると仮定する、持続時間のある行動を持つ MDP。ふつう次状態への遷移の分布と保持時間（方策がある行動を出してから次の行動を出すまでの時間ステップ数）の分布でモデル化され、保持時間全体の報酬がその行動の信用割り当てになる。FiGAR では決定が行動と時間スケールの組なので、元の MDP は自明に SMDP に変換される。保持時間中は選んだ行動を繰り返すと決まっている分、保持時間中に何が起きるかを定めない SMDP より構造が多い。時間スケールの分布を出す方策の部分が保持時間の分布にあたり、保持時間中の報酬は SMDP と同様に適切な割引をかけて合計する。

## 3. 背景（Background）

この節の記号は次のとおりである。

共通の記号:

- $s_t$, $a_t$: 時刻 $t$ の状態と行動。
- $r_j$: 時刻 $j$ の報酬。
- $\gamma$: 割引率。0 と 1 のあいだの数で [Schulman et al. (2015), Section 2]、3.1 節と 4.2 節の目標値の式では、報酬にかける重みとして $\gamma$ のべき乗が現れる。

A3C の記号（3.1 節）:

- $\pi_{\theta_a}(a \mid s)$: パラメータ $\theta_a$ を持つ方策（actor）。状態 $s$ で行動 $a$ を選ぶ確率。
- $V_{\theta_c}(s)$: パラメータ $\theta_c$ を持つ状態価値関数（critic）。方策勾配の分散を減らすのに使う。式の中では、論文に合わせて添字を省き $V(s_t)$、$V(s_n)$ とも書く。
- $G_t$: 時刻 $t$ の収益の推定値。A3C では $n$ ステップ収益で推定し、$Q(s_t, a_t)$ の偏りのある推定になる。
- $Q(s_t, a_t)$: 状態 $s_t$ で行動 $a_t$ を取ったときの行動価値。
- $A(s_t, a_t)$: 状態 $s_t$ で行動 $a_t$ を取ることのアドバンテージ。$G_t - V(s_t)$ がその推定になる。
- $n$: $n$ ステップ収益の長さ。固定のハイパーパラメータ。付録 A の FiGAR-A3C の実験では 20 で [p.13, Appendix A]、FiGAR-A3C ではこの $n$ を時間ステップではなく決定ステップで数える [p.6, Section 4.2]。
- $\hat{V}(s_t)$: 現在の状態からの $n$ ステップ収益の推定値。critic の学習の目標値になる。
- $k$: 方策のコピーを非同期に実行する学習スレッドの数（1 節の反復回数 $k$ とは別の意味）。

TRPO の記号（3.2 節）:

- $\pi$, $\theta_{old}$: 現在の方策とそのパラメータ。
- $\tilde{\pi}$, $\tilde{\theta}$: 次の方策の候補とそのパラメータ。
- $L_{\theta_{old}}(\tilde{\theta})$: 候補の方策を評価する代理損失（surrogate loss）。
- $\eta(\pi)$: 方策 $\pi$ の期待割引収益（割引した報酬の和の期待値）。TRPO の原論文の定義による [Schulman et al. (2015), Section 2]。
- $\rho^{\pi}(s)$: 方策 $\pi$ の下での割引された状態訪問頻度（正規化しない）[Schulman et al. (2015), Section 2]。付録 C でも、正規化されていない割引状態訪問分布と説明される [p.20, Appendix C]。
- $A_{\pi}(s, a)$: 方策 $\pi$ のアドバンテージ関数。
- $D^{\max}_{KL}(\pi, \tilde{\pi})$: 状態ごとの KL ダイバージェンス $D_{KL}(\pi(\cdot \mid s) \,\|\, \tilde{\pi}(\cdot \mid s))$ の、状態についての最大値 [Schulman et al. (2015), Section 3]。新旧の方策がもっとも違う状態での違いの大きさ。
- $\delta$: KL ダイバージェンスの上限。1 回の更新で方策を動かしてよい範囲を決める。

DPG と DDPG の記号（3.3 節）:

- $\mu_\theta(s)$: パラメータ $\theta$ を持つ決定論的方策。状態から連続行動のベクトルを直接出す。
- $J(\mu_\theta)$: 方策の性能目的関数（期待割引収益）[Lever (2014), Section 2.1, Section 3.2]。
- $\rho^{\mu}(s)$: 方策 $\mu$ の下での割引された状態分布（正規化しない）[Lever (2014), Section 2.1, Section 3.2]。
- $Q^{\mu}(s, a)$: 方策 $\mu$ の行動価値関数。critic が近似する。
- $\mathbb{S}$: 状態空間。

このうち $\gamma$、$\eta(\pi)$、$\rho^{\pi}(s)$、$D^{\max}_{KL}$、$J$、$\rho^{\mu}(s)$ は論文の 3 節に定義がないので、引用文献の定義で補い、出典を添えた。

### 3.1 A3C（Asynchronous Advantage Actor Critic）

actor-critic のアルゴリズムでは、方策 $\pi_{\theta_a}(a \mid s)$ と価値関数 $V_{\theta_c}(s)$ をどちらもパラメータで表して方策勾配で更新し、価値関数の推定は方策勾配の分散の低減に使う [Sutton & Barto (1998)] [p.3, Section 3.1]。A3C [Mnih et al. (2016)] では、$k$ 本の学習スレッドがそれぞれ方策のコピーを非同期に動かし、非同期の $n$ ステップ収益にもとづく更新を一定間隔で中央のパラメータサーバへ送る。論文は、スレッドが状態空間の別々の部分を並行して探索しうることを、連続する更新の時間的な相関が断ち切られる理由に挙げる [p.4, Section 3.1]。

方策改善の目的関数は次のとおりである。

$$L(\theta_a) = \log \pi_{\theta_a}(a_t \mid s_t)\,\bigl(G_t - V(s_t)\bigr)$$

[p.4, Section 3.1]

価値関数は $n$ ステップの TD 誤差で更新する。A3C は $j \le n$ の $j$ ステップ収益を使い、論文は簡単のため $t \le n$ と仮定して目標値 $\hat{V}(s_t)$ を書く。次の式の $j$ は和の添字（時刻）で、「$j$ ステップ収益」の $j$（収益のステップ数）とは役割が違う（論文の表記のまま）。

$$L(\theta_c) = \bigl(\hat{V}(s_t) - V_{\theta_c}(s_t)\bigr)^2, \qquad \hat{V}(s_t) = \sum_{j=t}^{n-1} \gamma^{t-j} r_j + \gamma^{n-t} V(s_n)$$

[p.4, Section 3.1]

割引の指数は論文の表記どおり $t-j$ と記した（食い違いの項目 2）。方策と価値関数は深層ニューラルネットワークでパラメータ化される [p.4, Section 3.1]。

### 3.2 TRPO（Trust Region Policy Optimization）

TRPO [Schulman et al. (2015)] は、代理損失の制約つき最適化として方策を改善するアルゴリズムで、方策が単調に改善する理論的保証を持つ [p.4, Section 3.2]。次の方策の候補 $\tilde{\pi}$ に対する代理損失は次の式で、

$$L_{\theta_{old}}(\tilde{\theta}) = \eta(\pi) + \sum_{s} \rho^{\pi}(s) \sum_{a} \tilde{\pi}(a \mid s)\, A_{\pi}(s, a)$$

これを次の制約の下で最適化する。

$$D^{\max}_{KL}(\pi, \tilde{\pi}) \le \delta$$

[p.4, Section 3.2]

この KL の制約により、自明でない大きさの更新幅をとりつつ、新しい方策が現在の方策から大きく離れないようにできる [p.4, Section 3.2]。

### 3.3 DDPG（Deep Deterministic Policy Gradients）

決定論的方策勾配（DPG）の定理 [Lever (2014)] によれば、連続行動空間での決定論的方策 $\mu_\theta$ の性能目的関数 $J$ の、方策パラメータ $\theta$ についての勾配は式 (1) で与えられる。

$$\nabla_\theta J(\mu_\theta) = \int_{\mathbb{S}} \rho^{\mu}(s)\, \nabla_\theta \mu_\theta(s)\, \nabla_a Q^{\mu}(s, a)\big|_{a=\mu_\theta(s)}\, ds = \mathbb{E}_{s \sim \rho^{\mu}}\Bigl[\nabla_\theta \mu_\theta(s)\, \nabla_a Q^{\mu}(s, a)\big|_{a=\mu_\theta(s)}\Bigr] \tag{1}$$

[p.4, (1)]

この定理にもとづく DPG のモデルは、連続行動空間の行動ベクトルを出す actor と、ある状態で選ばれた行動を評価する critic $Q(s, a)$ からなる。DDPG [Lillicrap et al. (2015)] は、actor と critic を非線形のニューラルネットワークで近似して DPG を拡張したものである [p.4, Section 3.3]。

## 4. FiGAR: 細粒度の行動反復（Fine Grained Action Repetition）

この節の記号は次のとおりである（4.1〜4.4 節で使うものもまとめて挙げる）。

枠組みの構成要素:

- $Z$: FiGAR で拡張する元の DRL アルゴリズム。方策を明示的に持つものに限る。論文の例は A3C、TRPO、DDPG。
- $Z'$: $Z$ を FiGAR で拡張したアルゴリズム（FiGAR-$Z$ とも書く）。例は FiGAR-A3C。
- $W = \{w_1, w_2, \cdots, w_{|W|}\}$: $Z'$ が選べる行動反復回数の集合。従来の DRL は定数 $c$ だけからなる $W = \{c\}$ にあたる。5.1 節の Atari の実験では $W = \{1, 2, \cdots, 30\}$ [p.8, Section 5.1]。
- $|W|$: $W$ の要素数。反復ネットワークの出力の次元になる。5.1 節と 5.2 節では 30、5.3 節では 15 [p.8, Section 5.1] [p.9, Section 5.2] [p.9, Section 5.3]。

2 つの方策とネットワーク:

- $\pi_{\theta_a}$（$\pi_a$ とも書く）: 行動を選ぶ方策。パラメータは $\theta_a$。元の $Z$ が持っている方策そのもの。
- $\pi_{\theta_x}$（$\pi_x$ とも書く）: 行動反復回数を選ぶ方策。パラメータは $\theta_x$。FiGAR が新たに加える部分。
- $f_{\theta_a}$: $\pi_{\theta_a}$ を実現するネットワーク（行動ネットワーク）。
- $f_{\theta_x}$: $\pi_{\theta_x}$ を実現するネットワーク（反復ネットワーク）。出力のサイズは $|W|$。
- $f_{\theta_c}$: critic のネットワーク。FiGAR-A3C では状態価値 $V$、FiGAR-DDPG では行動価値 $Q$ を出す。

目的関数の記号:

- $L(\pi_a, s_t, a_t)$: $Z$ が $\pi_a$ を改善するのに使う目的関数。
- $L(\pi_x, s_t, a_t)$: 同じ $L$ を $\pi_x$ で評価したもの。
- $\ast$: 2 つの目的関数を結ぶ結合演算子。FiGAR-A3C ではスカラーの加算、FiGAR-TRPO では論文の言い方でいえばある意味でスカラーの乗算。
- $T(s_t, a_t)$: 全体の損失。$T(s_t, a_t) = L(\pi_x, s_t, a_t) \ast L(\pi_a, s_t, a_t)$。

実行の記号（4.1 節）:

- $(a_j, x_j)$: 行動決定（action decision）。実行する行動 $a_j$ と、それを実行するステップ数 $x_j \in W$ の組。
- $s_j$: $j$ 回の行動決定を終えた時点の状態。$s_0$ は最初の状態。
- $t = \sum_{i=0}^{k} x_i$: $a_0, \ldots, a_k$ をそれぞれ $x_0, \ldots, x_k$ ステップ実行し終えた時刻。次の行動決定はこの時刻に行う。

FiGAR-A3C の記号（4.2 節）:

- $A(s_j, a, x)$: 状態 $s_j$ で行動 $a$ を $x$ ステップ実行することのアドバンテージ。
- $y_k$: 反復回数の累積和。$y_0 = 0$、$y_k = y_{k-1} + x_k$（$k \ge 1$）なので $y_k = x_1 + \cdots + x_k$ で、$x_0$ は入らない（食い違いの項目 2）。
- $n$: critic の目標値の計算に使う行動決定の数。論文は、この目標値は $n$ 時間ステップではなく $n$ 個の決定ステップにもとづくと明記している [p.6, Section 4.2]。
- $r_k$: 状態 $s_k$ での行動決定 $(a_k, x_k)$ に対応する報酬。論文はこの式の $r_k$ を改めて定義していない。2 節では、保持時間中の報酬は SMDP と同様に適切な割引をかけて合計するとしている [p.3, Section 2]。
- $|A|$: 元の MDP の行動空間の大きさ。FiGAR の出力は $|A| + |W|$ 次元になり、行動と反復回数を素朴に組にすると $|A| \times |W|$ 個になる。

FiGAR-TRPO の記号（4.3 節）:

- $A$: 連続行動の次元数。$f_{\theta_a}(s_j)$ の出力の各次元が独立な連続値の行動を表す。4.2 節のアドバンテージ $A(\cdot)$ や行動空間の大きさ $|A|$ と同じ文字だが、別の意味である。
- $\theta_{a,old}, \theta_{x,old}$: 更新前（現在）の行動方策と反復方策のパラメータ。
- $\tilde{\theta}_a, \tilde{\theta}_x$: 更新後の候補のパラメータ。それぞれが表す候補の方策を $\tilde{\pi}_a$, $\tilde{\pi}_x$ と書く。
- $L_{\theta_{a,old},\theta_{x,old}}(\cdot)$: 現在の方策で集めた軌跡から作る TRPO の代理損失。
- $\beta_{ar}$: 中核の方策のパラメータと反復のパラメータの、相対的な学習率を調整する係数。付録 C の調整結果は 1.28 [p.19, Appendix C]。
- $\beta_{KL}$: 反復方策の KL 項の重み。$\pi_x$ と $\pi_a$ が新しい方策からどれだけ離れてよいかの、相対的な大きさを調整する。付録 C の調整結果は 0.64 [p.19, Appendix C]、付録 G の共有表現の設定では 0.16 [p.24, Appendix G]。
- $\pi_a$, $\pi_x$（4.3 節）: 実行する行動のガウス分布と、softmax による行動反復回数の多項分布。
- $D^{\max}_{KL}$, $\delta$: 3.2 節と同じく、状態について最大をとった KL ダイバージェンスと、その上限。

FiGAR-DDPG の記号（4.4 節）:

- $\mu(s)$: 決定論的方策。$f_{\theta_a}(s_j)$ が出す、行動空間と同じ大きさのベクトル。
- $f_{\theta_c}(s_j, a_j)$: critic。推定した行動価値 $Q(s_j, a_j)$ を 1 つの数として出す。
- $f(s_j, a_j, x_j)$: 状態 $s_j$ と行動決定 $(a_j, x_j)$ に対する FiGAR-DDPG の critic の値。4.4 節の式のとおり、$f_{\theta_c}$ に $f_{\theta_a}(s_j)$ と $f_{\theta_x}(s_j)$ の出力を入れて求める。論文はこの記号を改めて定義していない。
- $\epsilon$: 反復方策の ε-greedy 探索の確率のパラメータ。付録 D の TORCS の実験では、50000 学習ステップで 0.2 から 0 まで減衰させる [p.21, Appendix D]。学習は 40000 学習ステップで終わるので、書かれたとおりなら $\epsilon$ は 0 に達しない（食い違いの項目 11）。

FiGAR は DRL アルゴリズムに、実行のために選んだ行動を何ステップ繰り返すかを予測する能力を加えて、時間的抽象化をモデル化させる。この予測は環境の現在の状態を条件とする [p.4, Section 4]。方策を明示的に持つ任意の DRL アルゴリズム $Z$ を拡張でき、拡張後の $Z'$ は、互いに独立に切り離された 2 つの方策の成分、すなわち行動を選ぶ方策 $\pi_{\theta_a}$ と行動反復回数を選ぶ方策 $\pi_{\theta_x}$ を持つ [p.4–5, Section 4]。従来の DRL では $W = \{c\}$ で反復が固定されているのに対し、FiGAR の $Z'$ は $W = \{w_1, w_2, \cdots, w_{|W|}\}$ から反復回数を選ぶ。FiGAR の中心となる考えは、$Z$ が $\theta_a$ の更新に使う目的関数を、$\theta_x$ の更新にもそのまま使うことである [p.5, Section 4]。

アルゴリズム 1（Create FiGAR-Z）は、$Z$ と $W$ を受け取って $Z'$ を作る手順を定める。内容を手順として書くと次のようになる [p.5, Algorithm 1]。

1. $Z$ の行動方策 $\pi_a$ と、それを実現する行動ネットワーク $f_{\theta_a}(s_t)$ を取り出す。$s_t$ は時刻 $t$ の状態、$a_t$ はそこで取った行動である。
2. $Z$ が $\pi_a$ を改善するのに使う目的関数 $L(\pi_a, s_t, a_t)$ を取り出す。アルゴリズム 1 のこの行は目的関数の持ち主を A と印刷しているが、p.5 の本文に合わせて $Z$ と読んだ（食い違いの項目 3）。
3. 行動反復の方策 $\pi_x$ を作り、出力サイズ $|W|$ の反復ネットワーク $f_{\theta_x}(s_t)$ で実現する。
4. 同じ $L$ を $\pi_x$ で評価して $L(\pi_x, s_t, a_t)$ を得る。
5. 全体の損失を $T(s_t, a_t) = L(\pi_x, s_t, a_t) \ast L(\pi_a, s_t, a_t)$ とし、$T$、$f_{\theta_a}$、$f_{\theta_x}$ を返す。

### 4.1 FiGAR の動作（How FiGAR operates）

$Z'$ は解いている MDP を次のように進む [p.5, Section 4.1]。

- 最初の状態 $s_0$ で、実行する行動と実行するステップ数の組 $(a_0, x_0)$ を予測する。$a_0$ は $\pi_{\theta_a}(s_0)$、$x_0$ は $\pi_{\theta_x}(s_0)$ にもとづいて決める。この組を行動決定と呼ぶ。
- $j$ 回の行動決定の後の状態を $s_j$、そのとき選んだ行動と反復回数を $a_j$ と $x_j$ と書く。$x_j$ は許される反復回数の集合 $W$ の要素である。
- 時間ステップ 0 から $x_0$ まで $a_0$ を実行し、時刻 $x_0$ に、そのときの状態 $s_1$ と $(\pi_{\theta_a}(s_1), \pi_{\theta_x}(s_1))$ にもとづいて次の組 $(a_1, x_1)$ を決める。
- 一般に、$a_k$ を $x_k$ ステップ連続して実行した後、次の行動は時刻 $t = \sum_{i=0}^{k} x_i$ に、その時刻に見た状態 $s_{k+1}$ と $(\pi_{\theta_a}(s_{k+1}), \pi_{\theta_x}(s_{k+1}))$ にもとづいて決める。

### 4.2 FiGAR-A3C

A3C は、方策 $\pi(a \mid s_j)$ を表す $f_{\theta_a}(s_j)$ と、価値関数 $V(s_j)$ を表す $f_{\theta_c}(s_j)$ を使う。$\pi(a \mid s_j)$ は元の MDP の行動空間と同じ大きさのベクトル、$V(s_j)$ はスカラーである。FiGAR-A3C はこれを 3 点で変える [p.5–6, Section 4.2]。

変更 1（反復ネットワークの追加）: $f_{\theta_a}(s_j)$ と $f_{\theta_c}(s_j)$ に加えて、ネットワーク $f_{\theta_x}(s_j)$ を定義する。その出力は $W$ の要素の上の確率分布を表す $|W|$ 次元のベクトルで、この多項分布からサンプルした時間スケールが、$f_{\theta_a}(s_j)$ で決めた行動を繰り返す長さになる。actor は行動ネットワーク $f_{\theta_a}(s_j)$ と反復ネットワーク $f_{\theta_x}(s_j)$ の 2 つからなる [p.5, Section 4.2]。

変更 2（actor の目的関数）: actor の目的関数を次のように変える。アルゴリズム 1 の結合演算子 $\ast$ は、FiGAR-A3C ではスカラーの加算になる [p.6, Section 4.2]。

$$L(\theta_a, \theta_x) = \bigl(\log f_{\theta_a}(a \mid s_j) + \log f_{\theta_x}(x \mid s_j)\bigr)\, A(s_j, a, x)$$

[p.6, Section 4.2]

変更 3（critic の目標値）: critic の目的関数は同じ形のまま、目標値に使う価値の推定だけを次のように変える。状態 $s_k$ で行動 $a_k$ が $x_k$ 回繰り返されたとし、$y_0 = 0$、$y_k = y_{k-1} + x_k$（$k \ge 1$）と定める [p.6, Section 4.2]。

$$\hat{V}(s_j) = \sum_{k=j}^{n-1} \gamma^{y_{k-j}} r_k + \gamma^{y_{n-j}} V(s_n)$$

[p.6, Section 4.2]

割引の指数は論文の表記どおり $y_{k-j}$（$y$ の添字が $k-j$）と記した。この定義のままでは、指数が状態 $s_j$ からの経過時間ステップ数と一致するのは $j = 1$ のときだけである（食い違いの項目 2）。目標値の収益は、実行する行動が変わりうるステップ（決定ステップ）の $n$ 個にもとづくもので、$n$ 時間ステップにもとづくものではない [p.6, Section 4.2]。

論文は、変更 2 によって行動空間が $|W|$ だけ拡張されて次元が $|A| + |W|$ になり、方策を因子分解しているからこそパラメータ数が膨らまないと述べる。行動と反復回数を素朴に組にして行動空間を広げると、[Lakshminarayanan et al. (2016); Vezhnevets et al. (2016)] に見られる行動空間の膨張が起きる。$|W|$ 通りの反復の水準（STRAW では長さ $|W|$ の計画）を制御するには $|A| \times |W|$ 個の行動または行動価値をモデル化する必要があり、最終層の大きさが $|W|$ 倍になる [p.6, Section 4.2]。

### 4.3 FiGAR-TRPO

A3C の $f_{\theta_a}(s_j)$ は連続・離散どちらの行動も出せるが、論文は A3C を離散行動空間でだけ扱う。FiGAR-TRPO では、$f_{\theta_a}(s_j)$ の出力を $A$ 次元とし、各次元が独立な連続値の行動を表す場合を考える。確率的方策は対角共分散の多変量ガウス分布でモデル化し、平均と共分散のパラメータをまとめて $\theta_a$、平均と共分散を連結したベクトルを $f_{\theta_a}(s_j)$ とする [p.6, Section 4.3]。FiGAR-TRPO はこれを 2 点で変える。

変更 1（目的関数）: TRPO の目的関数 $L_{\theta_{old}}(\tilde{\theta})$ は、現在の方策に従って集めた軌跡から作る。FiGAR-TRPO では目的関数を次のように変える。$\theta_x$ は反復の分布を計算する部分ネットワーク $f_{\theta_x}$ のパラメータである。論文は、アルゴリズム 1 の $\ast$ が FiGAR-TRPO ではある意味でスカラーの乗算になると述べる [p.6, Section 4.3]。

$$L_{\theta_{a,old},\theta_{x,old}}(\tilde{\theta}_a) \times \Bigl(L_{\theta_{a,old},\theta_{x,old}}(\tilde{\theta}_x)\Bigr)^{\beta_{ar}}$$

[p.6, Section 4.3]

変更 2（KL 制約）: 新旧の方策の KL ダイバージェンスについての TRPO の制約を次のように変える。$\pi_a$ は実行する行動のガウス分布、$\pi_x$ は softmax による行動反復の多項分布である [p.6, Section 4.3]。

$$D^{\max}_{KL}(\pi_a, \tilde{\pi}_a) + \beta_{KL}\, D^{\max}_{KL}(\pi_x, \tilde{\pi}_x) \le \delta$$

[p.6, Section 4.3]

この損失関数の由来は付録 C で説明される [p.6, Section 4.3]。

### 4.4 FiGAR-DDPG

DDPG は、決定論的方策 $\mu(s)$ を表す $f_{\theta_a}(s_j)$（元の MDP の行動空間と同じ大きさのベクトル）と、推定した行動価値 $Q(s_j, a_j)$ を 1 つの数として出す critic $f_{\theta_c}(s_j, a_j)$ からなる。FiGAR-DDPG はこれを 3 点で変える [p.6–7, Section 4.4]。

変更 1（反復ネットワークの追加）: FiGAR-A3C と同様に $f_{\theta_x}$ を加え、方策全体 $(\pi_{\theta_a}, \pi_{\theta_x})$ をネットワークの組 $(f_{\theta_a}, f_{\theta_x})$ で計算する。DDPG [Lillicrap et al. (2015)] と同様に actor には損失関数がない。actor が出した方策をそのまま critic に入れ、critic が返す勾配に沿って方策を改善する。方策全体 $\pi$ は $\pi_a$ と $\pi_x$ のベクトルの連結で、その勾配も両者の勾配の連結になる [p.7, Section 4.4]。

変更 2（探索）: 反復の探索方策には、挙動方策（behavioral policy）としての反復方策の ε-greedy 版を使う。行動の部分 $f_{\theta_a}(s_j)$ は引き続き、Ornstein-Uhlenbeck 過程で生成した時間相関のあるノイズで探索する [Lillicrap et al. (2015)] [p.7, Section 4.4]。

変更 3（critic）: critic を次の式でモデル化する。$f_{\theta_x}$ は、critic が出す $f_{\theta_x}$ についての勾配を逆伝播して、$f_{\theta_a}$ とまったく同じやり方で学習する [p.7, Section 4.4]。

$$f(s_j, a_j, x_j) = f_{\theta_c}\bigl(s_j, f_{\theta_a}(s_j), f_{\theta_x}(s_j)\bigr)$$

[p.7, Section 4.4]

## 5. 実験設定と結果（Experimental Setup and Results）

実験は 3 つの問いに答えるよう設計された。(1) 異なる DRL アルゴリズムの FiGAR 拡張が動的な行動反復を使うことを学べるか、(2) FiGAR はさまざまな課題でアルゴリズムの性能にどう影響するか、(3) さまざまな種類の行動反復集合 $W$ で制御を学べるか、である [p.7, Section 5]。5.1〜5.3 節は最も単純な $W = \{1, 2, \cdots, |W|\}$ を使い、5.4 節で $W$ を変えた影響を調べる [p.7, Section 5]。

### 5.1 Atari 2600 での FiGAR-A3C（FiGAR-A3C on Atari 2600）

実験の設定は次のとおりである。

- 手法: FiGAR-A3C（A3C の方策に反復回数を選ぶ反復ネットワークを加え、4.2 節の actor の目的関数で両方を学習するもの）と、ベースラインの A3C [p.8, Section 5.1]。
- ハイパーパラメータ: 5 つのゲーム（Beamrider, Breakout, Pong, Seaquest, Space Invaders）で調整し、全ゲームで同じ値を使った [p.8, Section 5.1]。主な値は付録 A にある（学習率は $10^{-3}$ から線形減衰、$n = 20$、エントロピー正則化の係数 0.02）[p.13, Appendix A]。
- 行動反復集合: 論文は $W$ を最も重要なハイパーパラメータかもしれないとし、エージェントが未来を予測する能力への信頼を表すもので、領域に応じて選ぶべきだとする。ここでの目的は時間的抽象化を学べることを示すだけなので、$|W|$ は調整せず任意に 30 とし、$W = \{1, 2, 3, \cdots, 30\}$ とした [p.8, Section 5.1]。
- 学習量: FiGAR-A3C も A3C も 100 million 決定ステップ学習した [p.8, Section 5.1]。FiGAR-A3C は 2 段階で学習し、最初の 20 million ステップは反復回数を固定して通常の A3C の目的関数で行動方策と価値関数だけを学び、残りの 80 million ステップで反復ネットワークを含む全体を同時に学ぶ [p.13–14, Appendix A]。
- 評価: 最終的に学習された方策で評価した [p.8, Section 5.1]。評価方策は、確率 0.1 で行動と反復回数を出力分布からサンプルし、確率 0.9 で、行動と反復回数をそれぞれ独立に最大確率のものに決める。100 エピソードまたは 100000 ステップの少ない方で平均スコアを求める [p.14, Appendix A]。

改善の指標は次のとおりである [p.8, Section 5.1]。ここでの $f$ と $i$ は 5.1 節だけの記号で、4 節のネットワーク $f_{\theta_a}$ などや 4.4 節の critic の式の $f$、1 節と 4.1 節の添字 $i$ とは別のものである。

- $b$: A3C のスコア（ベースライン）。
- $f$: FiGAR-A3C のスコア。
- $i$: FiGAR-A3C の改善率。次の式で求め、図 2 は百分率で示す [p.7, Figure 2]。

$$i = \frac{f - b}{b}$$

[p.8, Section 5.1]

図 2 [p.7, Figure 2] は、33 ゲームそれぞれの改善率 $i$ を百分率で示す棒グラフである。棒は改善率の大きい順に並び、改善は青、低下は赤で描かれ、各棒に値が書かれている。低下の棒には大きさだけが書かれている（本書による図の読み取り）。棒に書かれた値のうち最大の 2 つは、Enduro の 91822 と Atlantis の 3768 である [p.7, Figure 2]。

Enduro と Atlantis の改善はそれぞれ 900× と 35× を超え、図 2 の縦軸は見やすさのため 1000% で切ってある [p.8, Section 5.1]。論文は、図 2 から FiGAR-A3C が 33 ゲーム中 26 ゲームで A3C を上回るとする [p.8, Section 5.1]。ただし図 2 で負でない棒は 25 本で、26 は付録 A の表 4 で FiGAR-A3C の方が高いゲームの数と一致する。図 2 の一部の値も表 4 から計算した値と合わない（いずれも食い違いの項目 4）。

問い (1) に答えるため、各行動反復回数が選ばれた割合を記録した。図 3 は任意に選んだゲームでの分布で、値は小数第 2 位に丸めてあるため、ゲームごとの合計が 1 にならないことがある。各ゲームは、図 2 の平均スコアの計算と同じ方策で 10 エピソードプレイした [p.8, Section 5.1]。

図 3 [p.8, Figure 3] は、8 つの列（Atlantis、Crazy Climber、Demon Attack、Demon Attack、Pong、Freeway、Tutankham、Wizard of Wor）について、反復回数を 3 刻みの区間（1-3, 4-6, …, 28-30）にまとめ、各区間が選ばれた割合を積み上げ棒と数値の表で示す。列名には Demon Attack が 2 回現れる（食い違いの項目 5）。拡張版は付録 B の表 7 [p.18, Table 7] である [p.8, Figure 3]。例えば Crazy Climber の列では、1-3 の区間が 0.55、10-12 の区間が 0.38 を占める [p.8, Figure 3]。

論文の読み取りでは、FiGAR-A3C は全体として短い反復を好むが、方策空間に時間的抽象化を作り出し、それは特に Pong と Crazy Climber で目立つ。その一部が図 1 である。一方で、時間的抽象化がいつもゲームプレイを助けるとは限らない（Demon Attack）[p.8, Section 5.1]。

なお、学習した $\pi_{\theta_a}$ で毎時間ステップ行動を予測し、評価時に $\pi_{\theta_x}$ を完全に捨てれば、FiGAR を深い探索の枠組みとして使えるとも考えられる。付録 F はこの使い方への実証的な反論で、$\pi_{\theta_x}$ が表す時間的抽象化がゲームプレイの性能に重要であることを示す [p.8–9, Section 5.1]。

### 5.2 MuJoCo タスクでの FiGAR-TRPO（FiGAR-TRPO on Mujoco Tasks）

この節は、FiGAR-TRPO が MuJoCo の物理シミュレーション課題をそれなりにうまく解けることを示す [p.9, Section 5.2]。設定は次のとおりである。

- 手法: FiGAR-TRPO（TRPO のガウス方策に反復方策を加え、4.3 節の目的関数と KL 制約で学習するもの）と TRPO [p.9, Section 5.2]。
- 課題と観測: MuJoCo の 5 課題（Ant, Hopper, Inverted Pendulum, Inverted Double Pendulum, Swimmer）[p.9, Table 1]。観測は低次元の特徴ベクトル [p.19, Appendix C]。
- 行動反復集合: FiGAR-A3C と同様に $|W|$ を任意に 30 とした [p.9, Section 5.2]。
- 学習: 方策全体 $(f_{\theta_a}, f_{\theta_x})$ を同時に学習する。TRPO の各最適化ステップの後に得た方策を、それまでで最良の方策と比べて全体の最良方策を求め、結果はこの最良方策のものである [p.9, Section 5.2]。比べる基準は、学習中に代理損失を作るための軌跡を集める段階で見た平均収益である [p.19, Appendix C]。方策改善ステップは 500 回 [p.19, Appendix C]。
- 評価: 最良方策を 100 エピソード評価した平均スコア [p.9, Section 5.2]。

表 1 [p.9, Table 1] は、5 課題それぞれについて FiGAR-TRPO と TRPO の 100 エピソードの平均スコアを並べ、括弧内に選ばれた行動反復回数の平均を添えた表である [p.9, Section 5.2]。各行で高い方のスコアが太字になっている（本書による表の読み取り）。FiGAR-TRPO が上回るのは Ant、Inverted Pendulum、Inverted Double Pendulum で、TRPO が上回るのは Hopper と Swimmer である。FiGAR-TRPO の平均反復回数は Ant（28.35）と Swimmer（10.51）で大きく、ほかの 3 課題では 1.00 か 1.01 である。Ant では TRPO の -161.93 に対して FiGAR-TRPO が 947.06、Swimmer では TRPO の 364.55 に対して FiGAR-TRPO が 337.48 である [p.9, Table 1]。

論文によれば、FiGAR は、わずかに最適性を失う代わりに実行がずっと速い方策か、反復しない場合と似た方策のどちらかを学び、性能はベースラインとおおむね互角である。TRPO は MuJoCo では手ごわいベースラインで、FiGAR は 5 課題中 3 課題で TRPO を上回るが、多くの課題で上回る幅はわずかである [p.9, Section 5.2]。論文は FiGAR-TRPO の学習した挙動の動画の URL も示している [p.9, Section 5.2] [FiGAR-TRPO 動画 (2016/11)]。

### 5.3 TORCS での FiGAR-DDPG（FiGAR-DDPG on TORCS）

設定は次のとおりである。

- 手法: FiGAR-DDPG（DDPG の決定論的方策に反復ネットワークを加え、4.4 節のとおり critic の勾配で両方を学ぶもの）と DDPG [p.9, Section 5.3]。
- 課題と観測: TORCS の自動車レース。観測は 29 次元の特徴ベクトル、行動は加速・ブレーキ・ステアリングの 3 つの連続行動 [p.21, Appendix D]。
- 行動反復集合: $|W|$ を任意に 15 とした [p.9, Section 5.3]。
- 学習量: ベースラインも FiGAR-DDPG も 40000 学習ステップ [p.21, Appendix D]。この学習ステップが行動決定と時間ステップのどちらを数えるのかは、論文に書かれていない。

指標は、FiGAR-DDPG と DDPG がそれぞれ得た総報酬（total reward）である [p.9, Section 5.3]。論文は、この値を求めた評価の手順（評価したエピソードの数など）を書いていない [p.9, Section 5.3] [p.21, Appendix D]。総報酬は FiGAR-DDPG が 557929.68、DDPG が 59519.70 である [p.9, Section 5.3]。

FiGAR-DDPG はレースの課題を問題なくこなし、シミュレータが止まるまでにコースを 20 周した。FiGAR-DDPG が学んだ方策は DDPG の方策よりなめらかだったことも観察されている [p.9, Section 5.3]。論文は運転の様子の動画の URL も示している [p.9, Section 5.3] [FiGAR-DDPG 動画 (年月不明)]。

### 5.4 行動反復集合の影響（Effect of Action Repetition Set on FiGAR）

この節は問い (3) に肯定的に答える。FiGAR-A3C がよく働いた $W = \{1, 2, \cdots, 30\}$ に特別な意味はなく、よい性能は他の行動反復集合にも引き継がれることを示す [p.9, Section 5.4]。設定は次のとおりである。

- 手法: 行動反復集合 $W$ だけが異なる FiGAR-A3C の変種（表 3）と、ベースラインの A3C [p.9, Section 5.4]。FiGAR-A3C は、A3C の方策に反復回数を選ぶ反復ネットワークを加えたものである。
- ハイパーパラメータ: 変種ごとには調整せず、FiGAR-30（$W = \{1, 2, \cdots, 30\}$ の変種で、5.1 節の FiGAR-A3C と同じ集合）用に調整した値をそのまま使った [p.9, Section 5.4] [p.10, Table 3]。
- ゲーム: Seaquest, Space Invaders, Asterix [p.10, Table 2]。
- 指標: 各変種の最終的な評価スコア（生のスコア）[p.9, Section 5.4] [p.22, Appendix E]。
- 学習量と評価: 5.4 節と付録 E は、変種の学習量も評価の手順も書いていない。付録 E の図 6 の横軸にも目盛りの値がない（本書による図の読み取り）[p.22, Figure 6]。なお、表 2 の FiGAR-30 の 3 つの値は表 4 の FiGAR-A3C の値と同じで、Baseline の Space Invaders と Asterix の値も表 4 の A3C と同じである。Seaquest の Baseline だけが異なる（本書による照合。食い違いの項目 6）[p.10, Table 2] [p.13, Table 4]。

変種は表 3 [p.10, Table 3] で、行動反復集合 $W$ によって定義される。名前の数字が 1 つの変種（FiGAR-20、FiGAR-30、FiGAR-50）は 1 からその数までのすべての整数を $W$ とする。FiGAR-30-50 と FiGAR-20-30 は、1 から後ろの数までの整数から前の数の個数を非復元抽出した集合を $W$ とする。FiGAR-P は 50 未満の素数の集合を $W$ とする [p.10, Table 3]。

表 2 [p.10, Table 2] は、6 つの変種と A3C ベースライン（Baseline の行）の生のスコアを、Seaquest、Space Invaders、Asterix の 3 列で並べた表である。列ごとの最高値が太字になっている（本書による表の読み取り）。棒グラフ版が付録 E の図 7 [p.22, Figure 7] である。ベースラインのスコアは Seaquest が 2769.40、Space Invaders が 1268.75、Asterix が 2364.00 で、6 つの変種のスコアはそれぞれ 14683.00〜22904.50、1828.90〜2310.70、7730.00〜11949.00 の範囲にある。最高値は Seaquest が FiGAR-50、Space Invaders が FiGAR-20-30、Asterix が FiGAR-30 である [p.10, Table 2]。Space Invaders の最高値は表 2 によるもので、図 7 では FiGAR-20 の棒が FiGAR-20-30 の棒と同じ 2.3 で、その前に置かれている（食い違いの項目 7）。どのゲームでも 6 つの変種はすべてベースラインを上回る（本書による表の読み取り。付録 E も同じ趣旨を述べる [p.22, Appendix E]）。

論文は、FiGAR はどの $W$ でも学習でき、FiGAR-30 用のハイパーパラメータを他の変種に使っても性能は大きくは落ちないとする。付録 E に、平均スコアの学習ステップに対する推移と、表 2 の棒グラフがある [p.9–10, Section 5.4]。表 2 の FiGAR-20 の値の一部は図 7 と合わない（食い違いの項目 7）。

## 6. 結論・欠点・今後の課題（Conclusion, Shortcomings and Future Work）

論文のまとめは次の 3 点である [p.10, Section 6]。

- 枠組みと主張: FiGAR は、方策空間で時間的抽象化を学ばせて方策最適化の DRL アルゴリズムを改善する軽量な枠組みで、A3C、TRPO、DDPG のような連続・離散の行動空間を扱う方策勾配型のアルゴリズムに一般に適用できる。行動の確率分布に、選んだ行動を繰り返す時間スケールの確率分布を加えた構造化された方策を持つ。著者らは、時間的に引き延ばしたマクロ行動の最適な列を見つけることで、FiGAR が複数の領域でよりよい制御方策を学び、方策勾配法と actor-critic 法を大きく改善できることを結果が示したとする。
- 決定論的な環境: Atari、TORCS、MuJoCo はほぼ決定論的で、環境の動力学の確率性が小さい。著者らは、こうした環境では FiGAR のエージェントが環境の動力学の潜在的なモデルを作り、死なずに大きな行動反復を実行できると予想し、Freeway をその例に挙げる。図 1 (a) [p.2, Figure 1] では、ニワトリが近づく車の速さを見積もり、車をぎりぎりでかわしながら道路を渡ってゴールに着く。
- 欠点と今後の課題: 行動反復（マクロ行動）を途中で止める能力は、特に確率的な環境で重要だが、論文の設定は一度実行を決めたマクロ行動を止めることを考えていない。実行中に環境が予想外に変わったときにはこの能力が必要で、マクロ行動を止める行動と開始する行動を動的な時間スケールの基本設定に加えれば、より頑健な方策になりうる。著者らは、この変更が Minecraft のようなより一般的な確率的世界でも機能しうると考え、今後の課題としている。

## 謝辞（Acknowledgments）

謝辞のうち、実装に関する事実だけを記す [p.11, Acknowledgments]。A3C の実験はオープンソースの実装 [miyosuda/async_deep_reinforce (2016/04)] を使った。TRPO の実装は [aravindr93/robustRL (2016/08)] を、DDPG の実装は [yanpanlau/DDPG-Keras-Torcs (2016/10)] を改変したものである。ハイパーパラメータの情報提供や議論への謝意と、計算資源の提供元への謝意は省いた。

## 付録 A. FiGAR-A3C の実験詳細（Appendix A）

### A3C 実験の詳細と結果（Experimental details and results）

学習の設定は次のとおりである [p.13, Appendix A]。

- アルゴリズム: LSTM を使う A3C の変種 [Mnih et al. (2016)]。
- 最適化: async-rmsprop [Mnih et al. (2016)]。ハイパーパラメータは [Mnih et al. (2016)] と同じ。
- 学習率: 初期値 $10^{-3}$ を 100 million ステップで 0 まで線形に減衰。
- $n$ ステップ収益の $n$: 20。
- エントロピー正則化: [Mnih et al. (2016)] と同様に探索のために使う。係数 $\beta$ は調整の結果 0.02 で、行動方策 $f_{\theta_a}$ と反復方策 $f_{\theta_x}$ の両方に適用した。

Atari 2600 のゲームは複雑なため、因子分解された方策をランダムな初期値から同時に学ぶより、段階的に学ぶ方がよかった [p.13, Appendix A]。手順は次の 2 段階である [p.13–14, Appendix A]。

- 第 1 段階: 通常の A3C の目的関数で、行動方策 $f_{\theta_a}$ と価値関数 $f_{\theta_c}$ を、固定の行動反復率で少数の反復だけ学習する。$f_{\theta_x}$ には勾配を逆伝播せず、反復回数の予測はすべて捨てる。
- 第 2 段階: $(f_{\theta_a}, f_{\theta_x}, f_{\theta_c})$ 全体を同時に学習する。論文は、この非定常な学習目標によって、方策全体 $(f_{\theta_a}, f_{\theta_x})$ を同時に学び始める前に、よい価値関数の推定器 $f_{\theta_c}$ とよい行動方策の推定器 $f_{\theta_a}$ が得られるとする。

この付録での「ステップ」の数え方には注意が要る。FiGAR が行動 $a_t$ を $x_t$ 時間ステップ実行すると決めるたびに、行動選択を 1 ステップ行ったと数える。反復回数が可変なので、学習時間は行動選択の回数で測り、この付録ではこれを単にステップと呼ぶ [p.14, Appendix A]。つまり 4.2 節の決定ステップにあたり、環境の時間ステップではない。5.1 節も同じ学習量を 100 million 決定ステップと書いている [p.8, Section 5.1]。第 1 段階は 20 million ステップ（グリッドサーチで決めたハイパーパラメータ）、第 2 段階は 80 million ステップで、ベースラインの A3C は 100 million ステップ（行動選択）学習した [p.14, Appendix A]。

方策の拡散と評価の設定は次のとおりである。

- 2 つの成分（$f_{\theta_a}$ と $f_{\theta_x}$）を探索するために大きなエントロピー正則化が必要で、その結果、学習された方策は望ましいより拡散したものになる [p.14, Appendix A]。
- 評価は 1 million ステップごとに行い、ε-greedy に似た方式をとる。確率 $\epsilon = 0.1$ で行動と反復回数をそれぞれの出力分布（$f_{\theta_a}$ と $f_{\theta_x}$）からサンプルし、確率 $1 - \epsilon$ で、行動と反復回数をそれぞれ独立に最大確率のものに決める [p.14, Appendix A]。付録 B はこれを、確率 0.1 でサンプルし、確率 0.9 で最適な行動と反復回数を選ぶ方式と言い換えている [p.15, Appendix B]。
- 100 エピソードまたは 100000 ステップの少ない方で平均スコアを求める [p.14, Appendix A]。

表 4 [p.13, Table 4] は、33 ゲームそれぞれについて、最終的な FiGAR-A3C と A3C の生のスコアを並べ、括弧内に信頼水準 0.95 の信頼区間（下限, 上限）を添えた表である。信頼区間は 100 エピソードのスコアの平均から計算した [p.14, Appendix A]。各行で高い方のスコアが太字になっている（本書による表の読み取り）。A3C の方が高いのは Chopper Command、Crazy Climber、Demon Attack、Name This Game、Phoenix、Q-bert、Road Runner の 7 ゲームで、残る 26 ゲームでは FiGAR-A3C の方が高い（本書による数え上げ）[p.13, Table 4]。差が大きい例として、Enduro は FiGAR-A3C が 707.80、A3C が 0.77、Atlantis は 6330600.00 と 163660.00 である。逆に Name This Game は 1752.60 と 12100.80、Demon Attack は 7499.30 と 26742.75 である [p.13, Table 4]。A3C の Seaquest の値（2799.60）は表 2 のベースラインの値と異なる（食い違いの項目 6）。

表 5 [p.14, Table 5] は、行動計画を保持して時間的抽象化を学ぶ競合手法 STRAW [Vezhnevets et al. (2016)] のスコアを、FiGAR と STRAW の両方が学習・評価された 6 ゲーム（Alien、Amidar、Breakout、Crazy Climber、Frostbite、Q-bert）について、STRAW と STRAW-e の 2 列で示す。STRAW のスコアは、成績上位 5 つの複製（replica）の平均である [p.14, Appendix A]。論文は STRAW-e を説明していない。STRAW の原論文は、特徴抽出器と計画モジュールのあいだに雑音のある通信路を入れた、構造化された探索のための版を STRAW-explorer（STRAWe）と呼んでいる [Vezhnevets et al. (2016), Section 3.1]。

論文は表 4 と表 5 から、FiGAR と STRAW の成績はおおむね互角で、Breakout では FiGAR が、Frostbite では STRAW が明らかに上回ると読む [p.14, Appendix A]。該当する値は、Breakout が FiGAR-A3C の 814.50 に対して STRAW 344、STRAW-e 386、Frostbite が FiGAR-A3C の 309.60 に対して STRAW 4394、STRAW-e 8108 である [p.13, Table 4] [p.14, Table 5]。

図 4 [p.15, Figure 4] は、FiGAR-A3C の性能の学習に伴う推移を、比較のための A3C の同じ指標とともに示す。100 エピソードの評価のあいだ最良のエピソードのスコアも記録し、学習した方策が到達しえた最良と比べてどれだけ劣るかを見るため、その推移も描く [p.14, Appendix A]。図は 33 ゲームそれぞれの小さなグラフからなり、横軸に学習の経過をとって、A3C-Average、A3C-Best、FiGAR-A3C-Average、FiGAR-A3C-Best の 4 本の線を描く（本書による図の読み取り）。

### A3C のアーキテクチャの詳細（Architecture details）

- 低水準のアーキテクチャは [Mnih et al. (2016)] と同じで、それは [Mnih et al. (2015)] と同じものである。ただし LSTM の前の隠れ層のサイズを、[Mnih et al. (2016)] の 512 ではなく 256 にした [p.14, Appendix A]。
- [Mnih et al. (2016)] と同様に、actor と critic は最後の 1 層以外を共有する。$f_{\theta_a}$、$f_{\theta_x}$、$f_{\theta_c}$ は最終層以外が共通で、最終層だけがそれぞれ異なる。$f_{\theta_a}$ と $f_{\theta_x}$ の出力の非線形性は多項分布を表す softmax、critic $f_{\theta_c}$ の出力は線形である [p.14, Appendix A]。

## 付録 B. Atari 2600 の追加実験（Appendix B）

この付録は、論文が選んだ評価方式の影響を調べる [p.15, Appendix B]。

### 貪欲か確率的か（The choice of whether to be greedy or stochastic）

付録 A の評価では、確率 0.1 で出力の確率分布からサンプルし、確率 0.9 で最適な行動と反復回数を選んだ。0.1 という値は恣意的に見えるので、最大のものを選ぶ方（0.1-greedy の方策）から、出力分布からサンプルする方（確率的な方策）へ少しずつ移したときの性能を調べた [p.15, Appendix B]。

論文によれば、図 5 から、常に方策の分布からサンプルしても、ほとんどのゲームで FiGAR-A3C の性能は A3C と比べて大きくは落ちない。大きく落ちるゲームについて、著者らは、学習された方策の分布（行動と反復回数）が拡散しているためだと考えている。そのうえで、評価方式は恣意的に見えても妥当だとする [p.15, Appendix B]。

図 5 [p.16, Figure 5] は、最終方策の分布からサンプルする確率を横軸にとって平均性能を描いた図で、各グラフの左寄りの点は貪欲な方策、右寄りの点は確率的な方策の平均性能を表す [p.16, Figure 5]。図は 33 ゲームそれぞれの小さなグラフからなり、A3C-Average と FiGAR-A3C-Average の 2 本の線を描く（本書による図の読み取り）。

### 性能と速度のトレードオフ（Performance versus speed tradeoff）

上の議論は、ゲームプレイの性能と速度をトレードオフする新しい方法につながる。図 3 のとおり FiGAR-A3C は時間的に引き延ばしたマクロ行動を使うことを学ぶが、多くのゲームで短い行動を好む。論文は、反復の分布 $\pi_{\theta_x}$ は拡散しているので（表 6）、その分布からサンプルすれば、おそらくゲームプレイの最適性と引き換えに、より大きな反復回数を選ぶはずだとする [p.16, Appendix B]。

3 つの表の作り方は次のとおりである。

- 表 6: 確率的な方策（$\pi_{\theta_a}$ も $\pi_{\theta_x}$ も常に分布からサンプル）で、10 エピソードまたは 100000 ステップの少ない方をプレイし、各反復回数が選ばれた割合を記録した [p.16, Appendix B]。
- 表 7: 図 3 の拡張版で、0.1-greedy の方策（確率 0.9 で貪欲な行動または反復回数を選び、確率 0.1 で出力分布からサンプル）で作った。完全に貪欲な方策（常に最適な行動と反復回数を選ぶ）と比べなかったのは、それが決定論的になり評価に向かないからだとする [p.17, Appendix B]。
- 表 8: 2 つの方策でのゲームごとの平均行動反復回数。表 6 と表 7 と同じエピソードから求めた [p.18, Appendix B]。

表 6 [p.17, Table 6] と表 7 [p.18, Table 7] は、ゲームごとに、反復回数を 3 刻みの 10 区間（1-3, 4-6, …, 28-30）にまとめ、各区間が選ばれた割合を並べた表である。表 6 は方策（$\pi_{\theta_a}$ と $\pi_{\theta_x}$ の両方）が完全に確率的なとき、表 7 は両方が 0.1-greedy のときのものである。ゲーム名には略記や表記の違いがあり、表 8 のゲーム名と次のように対応する（本書による対応づけ）。Chpr Cmd は Chopper Command、Crzy Clmbr は Crazy Climber、Dmn Attk は Demon Attack、Kung Fu は Kung Fu Master、NTG は Name This Game、Spc Invdr（表 7 では Spc Invdrs）は Space Invaders、Tutankham は Tutankhamun、Wzd of Wor は Wizard of Wor である。表 6 には Pong の行が 2 つある（食い違いの項目 8）。2 つの方策の違いが目立つ例として、Bowling では、確率的な方策で 28-30 の区間が 0.91 を占めるのに対し、0.1-greedy の方策では 4-6 の区間が 0.91 を占める [p.17, Table 6] [p.18, Table 7]。

論文は、FiGAR が、確率 1 で分布からサンプルして確率的にふるまうか、0.1-greedy にふるまって小さな確率でだけサンプルするかを選ぶことで、ゲームプレイの最適性と速度をトレードオフすることを学ぶと読む。表 6 と図 3 を比べると、確率的な方策ではすべてのゲームでより長い行動を使う傾向がはっきり見え、その代償は最適性の小さな損失（図 5）だとする [p.17, Appendix B]。

表 8 [p.19, Table 8] は、33 ゲームそれぞれについて、確率的な方策（Stochastic の列）と 0.1-greedy の方策（0.1-Greedy の列）での平均行動反復回数を並べた表である。2 つの方策の差が最も大きいのは Bowling で、確率的な方策では 28.64、0.1-greedy の方策では 5.13 である [p.19, Table 8]（差が最大であることは本書による照合）。

論文は、ほとんどのゲームで確率的な方策の方が速くプレイすることを学ぶが、図 5 のとおり最適性はいくらか失われうるとする [p.18, Appendix B]。表 8 で確率的な方策の値の方が小さいゲームが 1 つある（Centipede の 6.84 と 7.88。食い違いの項目 9）。

## 付録 C. FiGAR-TRPO の実験設定（Appendix C）

### TRPO 実験の詳細（Experimental details）

- 観測: FiGAR-TRPO も、対応するベースラインも、低次元の特徴ベクトルを観測とする [p.19, Appendix C]。
- TRPO の 2 つの段階: 第 1 段階（P1）で、現在の挙動方策 $\pi$ に従って $K$ 本の軌跡をサンプルし、代理損失を作る。第 2 段階（P2）で、新しい方策への KL ダイバージェンスの制約の下で代理損失の最適化を 1 ステップ行い、方策を改善する。方策改善ステップは 500 回行った [p.19, Appendix C]。
- $K$ のスケジュール: $K$ は学習の進みに応じて変わり、次の P1 での $K$ は直前の P1 の収益の線形関数として決める。直前の収益が大きければ、代理損失を作るエピソード数は少なくなる [p.19, Appendix C]。
- 最良方策と評価: P1 で見た平均収益を追跡して最良の方策を求め、100 エピソードで評価して、学習した方策の平均スコアとした [p.19, Appendix C]。
- ハイパーパラメータ: 最も重要なのは $\beta_{ar}$ と $\beta_{KL}$ である。集合 $\{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28\}$ でのグリッドサーチにより、$\beta_{ar} = 1.28$、$\beta_{KL} = 0.64$ を得た。調整は 5 課題すべてで行った [p.19, Appendix C]。

### 損失関数とアーキテクチャ（Loss function and Architecture）

アーキテクチャは次のとおりである [p.20, Appendix C]。

- 非線形性はすべて tanh。
- 平均ベクトル: 隠れ層 2 層（128, 64）のネットワーク（平均ネットワーク）で実現する。
- 標準偏差: 入力に依存しないパラメータの層（std-dev 層）で実現する。平均ネットワークの出力と std-dev 層を連結したものが、4 節の行動方策 $f_{\theta_a}$ になる。
- 行動反復の関数 $f_{\theta_x}$: 平均ネットワークに似た隠れ層 2 層のネットワーク（act-rep ネットワーク）で実現する。論文は平均ネットワークより小さい隠れ層としつつ、サイズを (128, 64) と書いている（食い違いの項目 10）。出力の非線形性は、$W$ の大きさに合わせたサイズ 30 の softmax 層である。FiGAR-TRPO のパラメータ数が TRPO より大きく増えないよう、このネットワークは小さくした。
- 平均ネットワーク、std-dev 層、act-rep ネットワークは、パラメータも層も共有しない（共有した場合の実験は付録 G）。

損失関数の記号は次のとおりである。

- $\pi_{\theta_{old}}$: 更新前の方策（挙動方策）。Single Path 法では、サンプリング分布 $q$ がこの方策そのものになり、これが Single Path 法を特徴づける [p.20, Appendix C]。
- $\pi_{\tilde{\theta}}$: 候補の方策。
- $\rho^{\theta}_{old}$: 更新前の方策の下での状態の分布。論文は $\rho$ を正規化されていない割引状態訪問分布と説明する [p.20, Appendix C]。
- $Q_{\theta_{old}}(s, a)$: 更新前の方策の下での行動価値。
- $\pi_{\theta_{a,old}}, \pi_{\theta_{x,old}}$: 更新前の行動方策と反復方策。$\pi_{\theta_a}, \pi_{\theta_x}$ は更新後の候補で、それぞれネットワーク $f_{\theta_a}, f_{\theta_x}$ と等しい。
- $\rho^{\theta_a,\theta_x}_{old}$: 更新前の 2 つの方策の下での状態の分布。
- $Q_{\theta_{a,old},\theta_{x,old}}(s, a, x)$: 更新前の 2 つの方策の下で、状態 $s$ で行動 $a$ を $x$ ステップ実行したときの行動価値。

Single Path 法で構成するときの TRPO の代理損失は次の形になる [Schulman et al. (2015)] [p.20, Appendix C]。

$$L_{\theta_{old}}(\tilde{\theta}) = \mathbb{E}_{s \sim \rho^{\theta}_{old},\, a \sim \pi_{\theta_{old}}}\left[\frac{\pi_{\tilde{\theta}}(a \mid s)}{\pi_{\theta_{old}}(a \mid s)}\, Q_{\theta_{old}}(s, a)\right]$$

[p.20, Appendix C]

FiGAR-TRPO のように因子分解された方策の代理損失は次のとおりである。ここで $s \sim \rho^{\theta_a,\theta_x}_{old}$、$a \sim \pi_{\theta_{a,old}}$、$x \sim \pi_{\theta_{x,old}}$ で、$\pi_{\theta_a} = f_{\theta_a}$、$\pi_{\theta_{a,old}} = f_{\theta_{a,old}}$、$\pi_{\theta_x} = f_{\theta_x}$、$\pi_{\theta_{x,old}} = f_{\theta_{x,old}}$ である [p.20, Appendix C]。

$$L_{\theta_{a,old},\theta_{x,old}}(\theta_a, \theta_x) = \mathbb{E}_{s,a,x}\left[\frac{\pi_{\theta_a}(a \mid s)}{\pi_{\theta_{a,old}}(a \mid s)}\,\frac{\pi_{\theta_x}(x \mid s)}{\pi_{\theta_{x,old}}(x \mid s)}\, Q_{\theta_{a,old},\theta_{x,old}}(s, a, x)\right]$$

[p.20, Appendix C]

確率の比がこのように 2 つの積に分かれるのは、行動方策 $f_{\theta_a}$ と反復方策 $f_{\theta_x}$ が独立な確率分布だからである。論文によれば、理論的に筋の通った実現法はこの損失 $L_{\theta_{a,old},\theta_{x,old}}(\theta_a, \theta_x)$ を最小化することだが、実際には次の緩めた目的関数を最適化する方がうまくいき、これが 4.3 節の FiGAR-TRPO の目的関数になる [p.20, Appendix C]。

$$L_{\theta_{a,old},\theta_{x,old}}(\tilde{\theta}_a) \times L_{\theta_{a,old},\theta_{x,old}}(\tilde{\theta}_x)^{\beta_{ar}}$$

[p.20, Appendix C]

なお、上の代理損失の式はどれも負号を含まない。本書が記号の確認に使った TRPO の arXiv 版は、$\eta$ を割引した報酬の和の期待値として定め、同じ形の目的関数を KL 制約の下で最大化している [Schulman et al. (2015), Section 2, Section 5]。これに照らすと、論文の言う「最小化」は向きが逆になる（本書による照合）。

## 付録 D. FiGAR-DDPG の実験詳細（Appendix D）

### DDPG 実験の詳細（Experimental details）

設定は次のとおりである [p.21, Appendix D]。

- 観測: 低次元（29 次元）の特徴ベクトル。
- 行動: 加速・ブレーキ・ステアリングの 3 つの連続行動。
- 行動反復集合: 主実験では $W$ のハイパーパラメータを任意に 15 とした。
- バッチ正規化: [Lillicrap et al. (2015)] と違い、有用でなかったので使わない。
- リプレイメモリ: サイズ 10000。
- ターゲットネットワーク: ソフト更新を $\tau = 0.001$ で行う。
- 探索: DDPG は off-policy の actor-critic なので十分な探索が必要である。行動方策の空間では Ornstein-Uhlenbeck 過程 [Lillicrap et al. (2015)] で探索する。反復方策の空間では、探索のための手だてを 2 つとる。1 つ目は学習時に ε-greedy 版の方策を使うことで、$\epsilon$ は 50000 学習ステップで 0.2 から 0 まで減衰する。2 つ目は、確率 $1 - \epsilon$ の場合に貪欲な反復回数を選ぶ代わりに、出力分布 $f_{\theta_x}(s)$ からサンプルすることである。
- 学習量: ベースラインも FiGAR-DDPG も 40000 学習ステップで、$\epsilon$ の減衰にかける 50000 学習ステップより短い（食い違いの項目 11）。この学習ステップが行動決定と時間ステップのどちらを数えるのかは、論文に書かれていない。

### DDPG のアーキテクチャの詳細（Architectural details）

- 隠れ層の非線形性は ReLU で、隠れ層の重みはすべて He の初期化 [He et al. (2015)] で初期化した [p.21, Appendix D]。
- actor: 隠れ層 2 層（300, 600）のネットワーク。2 層目の表現を $h_2$ とし、その上に 2 つの出力層を学習する。$f_{\theta_a}$ は $h_2$ をサイズ 3 の出力層で変換したもので、ステアリングの出力は tanh、加速とブレーキの出力は sigmoid の非線形性を使う。$f_{\theta_x}$ は $h_2$ をサイズ $|W|$ の softmax 出力層で変換したものである。actor の出力は $3 + |W| = 18$ 次元のベクトルになる [p.21, Appendix D]。
- critic: 状態ベクトル（29 次元）と行動ベクトル（18 次元）を入力とする、隠れ層 3 層（300, 600, 600）のネットワーク。[Lillicrap et al. (2015)] と同様に、行動は $f_{\theta_c}$ の 1 層目には入れず、2 層目の隠れ層から入れる。出力は線形で、[Lillicrap et al. (2015)] と同様に TD 誤差の目的関数で学習する [p.21, Appendix D]。

## 付録 E. FiGAR の変種の詳細（Appendix E）

5.4 節の FiGAR-A3C の変種（行動反復集合 $W$ だけが異なるもの。定義は表 3）について、学習曲線と最終スコアの棒グラフを示す付録である [p.9–10, Section 5.4] [p.10, Table 3]。

図 6 [p.22, Figure 6] は、Space Invaders、Asterix、Seaquest の 3 つのパネルに、6 つの変種と Baseline の学習曲線を重ねて描く（本書による図の読み取り）。キャプションは Sea Quest と Asterix の 2 ゲームの比較と書いている（食い違いの項目 12）。

論文は図 6 から、FiGAR-A3C は基本行動と反復回数という 2 つの行動空間を探索する必要があるにもかかわらず、どの変種でも、その探索のために学習の進みが遅くなってはいないとする [p.22, Appendix E]。

図 7 [p.22, Figure 7] は、表 2 の 3 ゲーム（Seaquest、Space Invaders、Asterix）のスコアを棒グラフにしたもので、ゲームごとに 6 つの変種と Baseline の棒を並べる。キャプションによれば、スコアは 1000 で割り、小数第 1 位に丸めてある [p.22, Figure 7]（棒の値と表 2 の値の関係は食い違いの項目 7）。

表 2 は各変種の最終的な評価スコアで、図 7 はそれを棒グラフにして、すべての変種がベースラインより有利であることを示す [p.22, Appendix E]。

## 付録 F. 反復方策の重要性（Appendix F）

この付録は、学習後の評価で反復方策 $\pi_{\theta_x}$ を使わず、毎時間ステップ $\pi_{\theta_a}$ だけで行動を選ぶ変種を、通常の FiGAR と比べる。この変種は反復率 1 にあたり、FiGAR-wo-$\pi_{\theta_x}$ と呼ばれる。多くのゲームでこの変種が劣ることから、論文は、$\pi_{\theta_x}$ が学んだ時間的抽象化は自明なものではなく、ゲームプレイの性能に効いていると論じる。評価方式は付録 A と同じである [p.23, Appendix F]。

表 9 [p.23, Table 9] は、ゲームごとに FiGAR（通常の FiGAR-A3C）と FiGAR-wo-$\pi_{\theta_x}$ のスコアを並べた表である。FiGAR の列の値は表 4 の FiGAR-A3C の列と同じである（本書による照合）。各行で高い方のスコアが太字になっているが、太字のない行が 4 つある。Chopper Command、Crazy Climber、Demon Attack、Name this Game で、どれも FiGAR の方が高い。4 つとも、表 4 で A3C が FiGAR-A3C を上回ったゲームである（本書による表の読み取り）[p.13, Table 4]。反復方策を捨てると大きく下がる例として、Pong は 20.32 から -21.00 に、Crazy Climber は 154177.00 から 210.00 になる。逆に反復方策を捨てた方が高い例として、Beam Rider は 2348.78 に対して 3732.00、Enduro は 707.80 に対して 941.10 である [p.23, Table 9]。

論文は、33 ゲーム中 24 ゲームで $\pi_{\theta_x}$ が時間的抽象化の学習を助け、FiGAR-wo-$\pi_{\theta_x}$ のエージェントより性能が大きく上がると述べる [p.23, Appendix F]。表 9 の行数と、FiGAR の方が高い行の数は、この記述と合わない（食い違いの項目 13）。

## 付録 G. FiGAR-TRPO の共有表現実験（Appendix G）

5.2 節と付録 C の FiGAR-TRPO の実験では、方策の成分 $f_{\theta_a}$ と $f_{\theta_x}$ は表現を共有しない。この付録は、両者が最終層以外のすべての層を共有する設定を調べ、そのエージェントを FiGAR-shared-TRPO と呼ぶ [p.24, Appendix G]。設定は次のとおりである。

- ハイパーパラメータ: 付録 C と同じ。ただし $\beta_{ar}$ と $\beta_{KL}$ は付録 C と同様のグリッドサーチで 5 課題すべてで調整し直し、$\beta_{ar} = 1.28$、$\beta_{KL} = 0.16$ を得た [p.24, Appendix G]。
- 学習と評価: 付録 C と同じ手順で、TRPO の 2 段階の学習を行い、P1 の平均収益で選んだ最良方策を 100 エピソードで評価する [p.24, Appendix G] [p.19, Appendix C]。

表 10 [p.24, Table 10] は、5 課題それぞれについて、FiGAR-TRPO、FiGAR-shared-TRPO、TRPO の最良方策の平均スコアを並べ、スコアに括弧つきの数を添えた表である。括弧内の数の意味は表 10 にも付録 G にも書かれていないが、表 1 と同じ形式なので、選ばれた行動反復回数の平均と読める（本書による読み取り。表 1 の括弧の説明は [p.9, Section 5.2]）。FiGAR-TRPO と TRPO の列は表 1 と同じ値である（本書による照合）。各行で最も高いスコアが太字になっている（本書による表の読み取り）。3 手法のうち FiGAR-shared-TRPO が最も高いのは Ant（1779.72）と Inverted Double Pendulum（9138.85）で、Hopper と Swimmer では TRPO が、Inverted Pendulum では FiGAR-TRPO が最も高い [p.24, Table 10]。FiGAR-TRPO と直接比べると、FiGAR-shared-TRPO のスコアは Ant・Inverted Double Pendulum・Swimmer で高く、Hopper・Inverted Pendulum で低い。平均反復回数は Ant で 28.35 から 7.99、Swimmer で 10.51 から 8.02 に下がり、Hopper では 1.00 から 2.07 に上がる（本書による表の読み取り）[p.24, Table 10]。

論文は、FiGAR-shared-TRPO は全体として FiGAR-TRPO よりそれほどよくならないとする。TRPO の実験のネットワークは隠れ層 2 層と浅いので、層の共有は学習される方策の最適性の点で小さな改善しかもたらさない、と著者らは考えている [p.24, Appendix G]。

## 論文が述べている限界

論文自身が欠点や留保として述べている事柄を、論文の言う範囲で集める。

- 一度実行を決めたマクロ行動を途中で止めることを考えていない。確率的な環境ではこの能力が重要で、止める行動と開始する行動の追加は今後の課題である [p.10, Section 6]。
- 実験した Atari、TORCS、MuJoCo は、環境の動力学の確率性が小さく、ほぼ決定論的である [p.10, Section 6]。
- STRAW は FiGAR より一般的な時間的抽象化を表すとし、FiGAR は STRAW を補完するものと位置づけている [p.3, Section 2]。
- 行動反復集合 $W$ は最も重要なハイパーパラメータかもしれず、領域に応じて選ぶべきだとしながら、実験では大きさを調整せずに任意に決めた（Atari と MuJoCo で 30、TORCS で 15）[p.8, Section 5.1] [p.9, Section 5.2] [p.9, Section 5.3]。
- MuJoCo では TRPO を 5 課題中 3 課題で上回るが、多くの課題で上回る幅はわずかである [p.9, Section 5.2]。
- 速さと引き換えに最適性を失うことがある。MuJoCo では、FiGAR は最適性をわずかに失う代わりに実行がずっと速い方策か、反復しない場合に近い方策のどちらかを学ぶ [p.9, Section 5.2]。Atari では、確率的な方策はより長い行動を使うが、ゲームプレイの最適性がいくらか失われる [p.16–18, Appendix B]。
- 時間的抽象化はいつもゲームプレイを助けるとは限らない（Demon Attack）[p.8, Section 5.1]。
- 2 つの方策の成分を探索するのに大きなエントロピー正則化が必要で、学習された方策は望ましいより拡散する [p.14, Appendix A]。確率的な評価で性能が大きく落ちるゲームについて、著者らはこの拡散が原因だと考えている [p.15, Appendix B]。
- Atari では、因子分解された方策をランダムな初期値から同時に学ぶより段階的に学ぶ方がよく、2 段階の学習をとった [p.13, Appendix A]。
- 5.4 節の変種には FiGAR-30 用に調整したハイパーパラメータをそのまま使い、変種ごとには調整していない [p.9, Section 5.4]。
- FiGAR-TRPO では、理論的に筋の通った結合した損失ではなく、実際にうまくいった緩めた目的関数を最適化した [p.20, Appendix C]。
- MuJoCo で 2 つの方策の成分に層を共有させても改善は小さく、著者らはネットワークが浅いためだと考えている [p.24, Appendix G]。

## 論文内の記載の食い違い（本書による照合）

本書が論文の本文・図・表を突き合わせて見つけた食い違いを、論文の位置の順に挙げる。原因の推測はしない。計算値は本書が論文の数値から求めたもので、小数第 3 位を四捨五入した。節の末尾に、論文の参考文献の書誌を論文の外の書誌と照合した結果を別に記す。

1. Atari 2600 のゲーム数: 1 節は A3C での実験を 31 ゲームと書く [p.2, Section 1]。5.1 節は 33 ゲーム中 26 ゲームで上回ると書き [p.8, Section 5.1]、付録 A は 33 ゲームとし [p.14, Appendix A]、表 4 も 33 行である [p.13, Table 4]。
2. 目標値の割引の指数（3.1 節と 4.2 節）:
   - 3.1 節: $\hat{V}(s_t)$ の式の第 1 項の指数は $t-j$ と印刷されており、和の範囲 $j \ge t$ では 0 以下になる [p.4, Section 3.1]。
   - 4.2 節: $\hat{V}(s_j)$ の式の指数は $y_{k-j}$ と $y_{n-j}$ で、$y_0 = 0$、$y_k = y_{k-1} + x_k$（$k \ge 1$）、$x_k$ は状態 $s_k$ で行動 $a_k$ を繰り返した回数と定義されている [p.6, Section 4.2]。この定義では $y_m = x_1 + \cdots + x_m$ で、$x_0$ は入らない。一方、4.1 節の手順では、$a_k$ を $x_k$ ステップ実行した後の次の決定は時刻 $\sum_{i=0}^{k} x_i$ に行われるので [p.5, Section 4.1]、$s_j$ から $s_k$ までの経過時間ステップ数は $x_j + \cdots + x_{k-1}$ である。2 節は、保持時間中の報酬を SMDP と同様に適切に割り引くとしている [p.3, Section 2]。$k = j$ の項の指数 $y_0 = 0$ はどの $j$ でも経過ステップ数と合うが、それ以外の項と $V(s_n)$ の項の指数が経過ステップ数と一致するのは $j = 1$ のときだけである（反復回数がたまたま等しい場合を除く）。例えば $j = 0$ では、$r_1$ の指数は $y_1 = x_1$ だが、$s_0$ から $s_1$ までは $x_0$ ステップである。$V(s_n)$ の指数 $y_n = x_1 + \cdots + x_n$ も、経過ステップ数 $x_0 + \cdots + x_{n-1}$ と合わない。
3. アルゴリズム 1 の目的関数の持ち主: アルゴリズム 1 は元のアルゴリズムを引数 $Z$ として受け取り、4 行目で行動方策を $Z$ の行動方策として取り出すが、$\pi_a$ を改善する目的関数を取り出す 6 行目では、目的関数の持ち主が $Z$ ではなく A と印刷されている [p.5, Algorithm 1]。p.5 の本文は、この目的関数を $Z$ が持つ方策 $\pi_{\theta_a}$ の更新に使うものとして述べている [p.5, Section 4]。
4. 図 2 と表 4: 表 4 の値から $i = (f-b)/b$ を百分率で計算すると、33 ゲーム中 29 ゲームで、図 2 の棒の値は計算値の小数点以下を切り捨てた値（負の値は絶対値の小数点以下を切り捨てた値）と一致する。一致しないのは次の 4 ゲームである [p.7, Figure 2] [p.13, Table 4]。
   - Beam Rider: 表 4 では FiGAR-A3C（2348.78）が A3C（2189.96）を上回り、計算値は +7.25% だが、図 2 では負の側の棒（大きさ 7）として描かれている。
   - Bank Heist: 図 2 は 91、表 4 からの計算値は 94.33。
   - Alien: 図 2 は 19、表 4 からの計算値は 15.85。
   - Seaquest: 図 2 は 552、表 4 からの計算値は 545.70。表 2 のベースラインの値（2769.40）を使うと 552.74 になる（項目 6）。

   表 4 で FiGAR-A3C が上回るゲームは 26 で、5.1 節の本文（33 ゲーム中 26 ゲームで上回る）と一致する [p.8, Section 5.1]。一方、図 2 で負でない棒は、0 の Frostbite を含めて 25 本である。また、5.1 節は図 2 の縦軸を 1000% で切ったとするが [p.8, Section 5.1]、Atlantis の棒（3768）は頭打ちの Enduro の棒より目に見えて短い。図 2 の縦軸には目盛りの値がないので、Seaquest（552）と Asterix（405）の棒の高さから目盛りを求めると、Enduro の棒は約 1000%、Atlantis の棒は約 800% の高さに描かれている（本書による図の読み取り）[p.7, Figure 2]。
5. 図 3 の列名: 列名に Demon Attack が 2 回現れる。2 つ目の列の値は、表 7 の Koolaid の行の値と一致する [p.8, Figure 3] [p.18, Table 7]。
6. A3C の Seaquest のスコア: 表 4 は 2799.60 [p.13, Table 4]、表 2 のベースライン（A3C）は 2769.40 である [p.10, Table 2]。Space Invaders（1268.75）と Asterix（2364.00）は両表で一致する。
7. 表 2 の FiGAR-20 と図 7: 表 2 の FiGAR-20 の Space Invaders（1929.50）と Asterix（7730.00）は、FiGAR-50 の行と同じ値である。図 7 の FiGAR-20 の棒は Space Invaders が 2.3、Asterix が 9.1 で、表 2 の値を 1000 で割った 1.93 と 7.73（小数第 1 位までなら 1.9 と 7.7）に合わない [p.10, Table 2] [p.22, Figure 7]。図 7 はゲームごとに棒を値の大きい順に並べており、Space Invaders では FiGAR-20 の棒（2.3）が、表 2 で最高の FiGAR-20-30 の棒（2.3）より先に置かれている（本書による図の読み取り）。また、FiGAR-20 の 2 本を除く 19 本の棒の値は、表 2 の値を 1000 で割って小数第 1 位未満を切り捨てた値と一致し、四捨五入した値とは 8 本で合わない（例: 14683.00 に対して図は 14.6）。キャプションは小数第 1 位に丸めたと書いている [p.22, Figure 7]。
8. 表 6 の Pong: 表 6 には Pong の行が 2 つあり（Enduro の次と Phoenix の次）、値が異なる。表 6 は 34 行、表 7 は 33 行である [p.17, Table 6] [p.18, Table 7]。
9. 確率的な方策の反復回数: 付録 B は、確率的な方策ではすべてのゲームでより長い行動を使う傾向が見えるとする [p.17, Appendix B]。表 8 で確率的な方策の平均反復回数が 0.1-greedy の方策より大きいのは 33 ゲーム中 32 ゲームで、Centipede だけは確率的な方策の方が小さい（6.84 と 7.88）[p.19, Table 8]。表 8 についての本文は、ほとんどのゲームと書いている [p.18, Appendix B]。
10. 付録 C の act-rep ネットワーク: 平均ネットワークより小さい隠れ層と書きながら、サイズを (128, 64) としており、平均ネットワークのサイズ (128, 64) と同じである [p.20, Appendix C]。
11. FiGAR-DDPG の探索の $\epsilon$ と学習量: 付録 D は、反復方策の ε-greedy 探索の $\epsilon$ を 50000 学習ステップで 0.2 から 0 まで減衰させたと書き、すぐ次の文で、学習はベースラインも FiGAR-DDPG も 40000 学習ステップ行ったと書く [p.21, Appendix D]。同じ「学習ステップ」で書かれた 2 つの数をそのまま比べると、$\epsilon$ は 0 に達する前に学習が終わる。
12. 図 6 のキャプション: Sea Quest と Asterix の 2 ゲームと書くが、図には Space Invaders・Asterix・Seaquest の 3 つのパネルがある [p.22, Figure 6]。
13. 表 9 の行数: 本文は 33 ゲーム中 24 ゲームで $\pi_{\theta_x}$ が性能を大きく上げると書く [p.23, Appendix F]。表 9 は 32 行（Q-bert の行がない）で、FiGAR の値が FiGAR-wo-$\pi_{\theta_x}$ を上回るのは 23 行である [p.23, Table 9]。

参考文献の書誌（論文の外の書誌との照合）: 次の 3 件は論文内の記載どうしの食い違いではなく、論文の参考文献の書誌が論文の外の書誌と合わないものである [p.11–12, References]。

- [Duff (1995)] として挙げられた Reinforcement learning methods for continuous-time markov decision problems の著者は、参考文献では Steven J Duff と書かれているが、NeurIPS の論文集では Steven Bradtke と Michael Duff である [Duff (1995)]。
- [Lever (2014)] として挙げられた Deterministic policy gradient algorithms の著者は、参考文献では Guy Lever だけだが、ICML 2014 の論文集では David Silver を筆頭とする 6 名である [Lever (2014)]。
- [Sutton & Barto (1998)] の書名は、参考文献では Introduction to reinforcement learning と書かれているが、書籍の書名は Reinforcement Learning: An Introduction である [Sutton & Barto (1998)]。

## キューの論点への答え

reports/INDEX.md のキューに書かれた問いに、論文の記述の範囲で答える。論文が扱っていない点はそう明記する。

1. 繰り返し回数の候補（集合 $W$）の選び方と、粒度による効果の差
   - 選び方: 論文は $W$ を最も重要なハイパーパラメータかもしれないとし、エージェントが未来を予測する能力への信頼を表すもので、領域に応じて選ぶべきだとする [p.8, Section 5.1]。具体的な選び方の手順は示していない。
   - 実験で使った値: 5.1〜5.3 節はどれも最も単純な $W = \{1, 2, \cdots, |W|\}$ を使い [p.7, Section 5]、$|W|$ は調整せず任意に決めた。Atari と MuJoCo では 30、TORCS では 15 である [p.8, Section 5.1] [p.9, Section 5.2] [p.9, Section 5.3]。どの領域についても 1 時間ステップの実時間の長さ（Atari では何フレームか）は書かれておらず [p.13–14, Appendix A] [p.19, Appendix C] [p.21, Appendix D]、反復回数が実時間でどれだけの長さにあたるかは論文からは決まらない。
   - $W$ を変えた比較: 5.4 節は、$W$ だけが異なる FiGAR-A3C の 6 つの変種を、Seaquest・Space Invaders・Asterix の 3 ゲームで A3C と比べた [p.9–10, Section 5.4]。$W$ は、上限の違う連続した整数（1〜20、1〜30、1〜50）、間引いた集合（1〜50 から 30 個、1〜30 から 20 個を非復元抽出）、50 未満の素数の 3 種類である [p.10, Table 3]。ハイパーパラメータは FiGAR-30 用に調整した値を使い回した [p.9, Section 5.4]。
   - 結果: 3 ゲームとも 6 つの変種すべてがベースラインを上回り [p.10, Table 2] [p.22, Appendix E]、最も高い変種はゲームごとに違った（Seaquest は FiGAR-50、Space Invaders は FiGAR-20-30、Asterix は FiGAR-30）[p.10, Table 2]。論文はここから、FiGAR はどの $W$ でも学習でき、FiGAR-30 用のハイパーパラメータのままでも性能は大きくは落ちないとする [p.9, Section 5.4]。どの粒度がよいかや、変種の差がどこから来るかは論じていない。表 2 には信頼区間などのばらつきの情報が無い（本書による表の読み取り）。FiGAR-20 の値の一部は図 7 と合わない。図 7 では Space Invaders の FiGAR-20 の棒が FiGAR-20-30 の棒と同じ 2.3 で先頭に置かれており、上の Space Invaders の最高値は表 2 にもとづく（食い違いの項目 7）。
   - 定義から分かること（本書による補足）: 50 未満の素数は 2 から 47 までの 15 個なので、FiGAR-P の $W$ は 1 を含まず、間隔もそろっていない。それでも FiGAR-P は 3 ゲームともベースラインを上回った [p.10, Table 2]。間引いた 2 つの変種について、論文は実際に抽出した集合を示していない [p.10, Table 3]。
   - 比べるときの物差し: 反復回数が可変なので、Atari の実験では学習量を環境の時間ステップではなく決定（行動選択）の回数で数え、FiGAR-A3C も A3C も 100 million 決定ステップ学習した [p.8, Section 5.1] [p.14, Appendix A]。環境の時間ステップ数をそろえた比較は無い。
2. 行動の一覧を平らに増やす方式と、繰り返しを別の方策で選ぶ方式の違い
   - 平らに増やす方式: 行動と反復回数を組にして行動空間を広げると、$|W|$ 通りの反復を扱うのに $|A| \times |W|$ 個の行動または行動価値をモデル化する必要があり、最終層が $|W|$ 倍になる。論文はこの膨張の例に Dynamic Frameskip Deep Q-network [Lakshminarayanan et al. (2016)]（以下、本書では DFDQN と略す）と STRAW [Vezhnevets et al. (2016)] を挙げ、STRAW では長さ $|W|$ の計画を扱うときに同じことが起きるとする [p.6, Section 4.2]。
   - DFDQN について: DFDQN は、DQN [Mnih et al. (2015)] に時間スケールごとに同じ基本行動の出力を別に加えて、反復の時間スケールを学ぶ。いくつかの Atari のゲームで大きく性能が上がったが、論文は、行動空間の膨張のおそれから多数の時間スケールを扱えないことと、離散行動空間に限られることを弱点に挙げ [p.3, Section 2]、時間スケールは 2 つだけで、実験したゲームも少数だったとする [p.2, Section 1]。
   - 別の方策で選ぶ方式（FiGAR）: 行動の方策 $\pi_{\theta_a}$ と反復回数の方策 $\pi_{\theta_x}$ を分け、どちらもその時点の状態から決める [p.4–5, Section 4] [p.5, Section 4.1]。出力は $|A| + |W|$ 次元で済み、論文はこの因子分解のおかげでパラメータ数が膨らまないとする [p.6, Section 4.2]。行動空間が連続でも離散でも使え [p.2, Section 1]、連続行動の TRPO と DDPG にも適用した [p.6, Section 4.3] [p.6–7, Section 4.4]。
   - 組の確率: FiGAR-A3C の actor の目的関数は、2 つの対数確率の和にアドバンテージを掛けた形である [p.6, Section 4.2]。FiGAR-TRPO の代理損失では確率の比が 2 つの積に分かれ、論文はこれを、2 つの方策が独立な確率分布だからと説明する [p.20, Appendix C]。
   - 本書の推論: 組の確率が 2 つの積なので、確率的な方策を使う A3C 版と TRPO 版では、同じ状態ならどの行動を選んでも反復回数の分布は同じになる。平らに増やす方式は組ごとに別の出力を持つので、行動ごとに違う反復の分布も表せる。論文はこの違いを論じていない。
   - FiGAR の学習で論文がとった手当て: 2 つの成分を探索するために大きなエントロピー正則化が要り、学習された方策は望ましいより拡散した [p.14, Appendix A]。Atari では、反復を固定して行動方策と価値関数だけを先に学ぶ 2 段階の学習をとった [p.13–14, Appendix A]。一方で、5.4 節の変種の学習曲線（図 6）について、2 つの行動空間を探索しても学習の進みは遅くならなかったとする [p.22, Appendix E]。
   - 論文が扱っていない点: 平らに増やす方式と FiGAR を同じ条件で比べる実験は無い。時間的抽象化を扱う他の手法との比較は、STRAW のスコアとの比較（表 5）だけである [p.14, Appendix A]。膨張の議論はすべての行動に反復を組み合わせる場合のもので [p.6, Section 4.2]、一部の行動にだけ反復を足す形は扱っていない。
3. 価値ベース（DQN 系）に入れるときの扱い
   - 論文の範囲: FiGAR の対象は方策を明示的に持つ DRL アルゴリズムで [p.1, Abstract] [p.4, Section 4]、具体化したのは方策勾配型の A3C・TRPO・DDPG だけである [p.5, Section 4] [p.10, Section 6]。DQN 系に入れる方法も、その実験も論文には無い。
   - 仕組みの前提: FiGAR の中心は、元のアルゴリズムが行動方策の改善に使う目的関数を、反復方策の更新にもそのまま使うことである [p.5, Section 4] [p.5, Algorithm 1]。DQN 系は行動価値を最大にする行動を選び、改善の目的関数を持つ方策を別に持たないので、この仕組みをそのまま当てはめる先がない（本書の推論）。
   - 論文が触れる価値ベースの形: DQN 系の構成として論文が具体的に説明しているのは DFDQN だけで、行動と反復の組の数だけ行動価値を持つ平らな形として扱い、多数の時間スケールを扱えないとする [p.3, Section 2] [p.6, Section 4.2]（問い 2）。このほかに論文は、表形式の動的計画法を改変した TPDP [Buckland & Lawrence (1994)] にも触れている。TPDP では方策が遷移点の状態でだけ変わり、遷移点の集合を Q 学習の変種で学ぶ。論文は FiGAR との違いを、遷移点をその場で学んで状態空間に汎化するか、表形式で大きな問題に使えないかに置いている [p.3, Section 2]。
   - 価値の推定の仕方: FiGAR-A3C の critic は状態価値 $V(s)$ だけを出し、行動と反復の組はアドバンテージ $A(s_j, a, x)$ として actor の目的関数に入る [p.5–6, Section 4.2]。組を入力にとって価値を出すネットワークは FiGAR-DDPG の critic で、4.4 節の式では、状態と、行動 $f_{\theta_a}(s_j)$ と、反復回数の分布 $f_{\theta_x}(s_j)$ を入力にとる [p.7, Section 4.4]。付録 D によれば、critic の入力は状態ベクトル（29 次元）と行動ベクトル（18 次元）で、この行動ベクトルは 3 次元の行動に $|W| = 15$ 次元の softmax の出力を連結したものである。critic は TD 誤差の目的関数で学ぶ [p.21, Appendix D]。
   - 反復をまたぐ目標値: 論文は、保持時間中の報酬を SMDP と同様に適切な割引をかけて合計するとし [p.3, Section 2]、FiGAR-A3C の critic の目標値では、和を決定ステップ $n$ 個でとり、割引の指数に反復回数の累積和 $y$ を使う [p.6, Section 4.2]。ただし式の添字のままでは、指数が経過時間ステップ数と一致するのは $j = 1$ のときだけである（食い違いの項目 2）。決定ごとの報酬 $r_k$ の作り方は式の中では定義されていない（4 節の記号の $r_k$ の項）。
   - replay と探索: FiGAR の中で replay を使う off-policy の構成は FiGAR-DDPG だけで、サイズ 10000 のリプレイメモリと、$\tau = 0.001$ でソフト更新するターゲットネットワークを使う。反復方策の探索は ε-greedy で、$\epsilon$ は 50000 学習ステップで 0.2 から 0 へ減衰し、確率 $1 - \epsilon$ の側でも貪欲に選ばず出力分布からサンプルする [p.21, Appendix D]。学習は 40000 学習ステップなので、書かれたとおりなら $\epsilon$ は 0 に達しない（食い違いの項目 11）。
   - 論文が書いていないこと: FiGAR-DDPG で、反復を含む遷移を replay にどう入れるか、critic の目標値で反復のステップ数をどう割り引くか、replay の遷移で critic を学ぶとき反復回数を分布と実行した値のどちらで与えるか（4.4 節の式は分布を入れる形で書かれている）は書かれていない [p.7, Section 4.4] [p.21, Appendix D]。
4. 繰り返しが長すぎるときの弊害
   - 論文が挙げる問題: 一度実行を決めたマクロ行動を途中で止められないこと。実行中に環境が予想外に変わったときには止める能力が要り、確率的な環境では特に重要だとする。止める行動と開始する行動を加える変更は今後の課題で、実験には入っていない [p.10, Section 6]。
   - 実験環境の性質: 実験した Atari・TORCS・MuJoCo はほぼ決定論的である。著者らは、そうした環境ではエージェントが動力学の潜在的なモデルを作り、死なずに長い反復を実行できると見込み、その例に Freeway を挙げる [p.10, Section 6] [p.2, Figure 1]。なお論文は、$W$ をエージェントが未来を予測する能力への信頼を表すものと位置づけている（問い 1）[p.8, Section 5.1]。環境の確率性を変えて長い反復の害を測る実験は無い。
   - 近似的に貪欲な評価での長さ: 0.1-greedy の評価では、FiGAR-A3C は全体として短い反復を好む [p.8, Section 5.1]。ゲームごとの平均反復回数は 1.26（Road Runner）から 14.25（Freeway）で、どのゲームも上限 30 の半分に届かない（本書による表の読み取り）[p.19, Table 8]。
   - 長い反復を多く使わせたとき: 出力分布から常にサンプルする確率的な評価では、33 ゲーム中 32 ゲームで平均反復回数が大きくなり（食い違いの項目 9）、差が最も大きい Bowling では 5.13 から 28.64 になる [p.19, Table 8]。論文はこれを、最適性を少し失う代わりに速くプレイするトレードオフとして読む [p.16–18, Appendix B]。ただし確率的な評価では行動もサンプルするので、性能の低下を反復の長さだけのせいとは言えない（本書の推論）。論文自身は、性能が大きく落ちたゲームの原因を、学習された分布の拡散に求めている [p.15, Appendix B]。
   - 時間的抽象化が助けにならなかった例: 論文は、時間的抽象化がいつもゲームプレイを助けるとは限らない例に Demon Attack を挙げる [p.8, Section 5.1]。表 4 では A3C の 26742.75 に対して FiGAR-A3C は 7499.30 である [p.13, Table 4]。この差が反復の長さによるのかは分析されていない。
   - 評価時に反復をやめた比較: 付録 F は、学習を終えたエージェントの評価で $\pi_{\theta_x}$ を捨て、毎時間ステップ $\pi_{\theta_a}$ で行動を選ぶ FiGAR-wo-$\pi_{\theta_x}$ を、通常の FiGAR と比べている。論文は、33 ゲーム中 24 ゲームで $\pi_{\theta_x}$ が性能を大きく上げたとする [p.23, Appendix F]（表 9 の行数との関係は食い違いの項目 13）。一方、表 9 には反復をやめた方が高い行が 9 つあり、例えば Beam Rider は 2348.78 と 3732.00、Enduro は 707.80 と 941.10 である（本書による表の読み取り）[p.23, Table 9]。論文はこの 9 行を論じていない。Demon Attack は反復をやめても 6661.00 で、A3C の 26742.75 に届かない [p.23, Table 9] [p.13, Table 4]。FiGAR-wo-$\pi_{\theta_x}$ は反復ありで学習した行動方策をそのまま使うので、この値は反復なしで学習した場合の性能ではない（本書の推論）。
   - MuJoCo: 論文は、FiGAR が、最適性を少し失う代わりに実行がずっと速い方策か、反復しない場合に近い方策のどちらかを学ぶとする [p.9, Section 5.2]。平均反復回数が大きいのは Ant（28.35）と Swimmer（10.51）で、Ant は TRPO を大きく上回り（947.06 と -161.93）、Swimmer は下回った（337.48 と 364.55）[p.9, Table 1]。
   - 上限を広げた変種: 30 を超える反復を含む $W$ を持つのは、FiGAR-50（1〜50）と FiGAR-P（最大 47）で、1〜50 から 30 個を抽出した FiGAR-30-50 も 30 を超える値を含みうる（実際に抽出した集合は示されていない）[p.10, Table 3]。3 つともベースラインを上回るが、6 つの変種の中での順位はゲームごとにまちまちである。FiGAR-50 は Seaquest で最も高く、Asterix で最も低い値（7730.00）、FiGAR-30-50 は Space Invaders で最も低く、Asterix で 2 番目、FiGAR-P は 3 ゲームとも 2〜3 番目である（本書による表の読み取り）[p.10, Table 2]。表 2 では FiGAR-20 の Asterix も 7730.00 だが、これは食い違いの項目 7 の箇所である。論文はこの変種間の差を論じていない。

## 出典リスト

本論文と書誌情報:

[Sharma et al., 2017/02] Sahil Sharma, Aravind S. Lakshminarayanan, Balaraman Ravindran. "Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning." ICLR 2017（arXiv:1702.06054。本書は 2020 年 9 月の v2 を参照）. https://arxiv.org/abs/1702.06054

[arXiv 1702.06054, 2020/09] arXiv. "Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning." arXiv. https://arxiv.org/abs/1702.06054（書誌ページ。投稿履歴・著者表示・ライセンスの表示を参照）

[ML Anthology, 2017] ML Anthology. "Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning." ML Anthology. https://mlanthology.org/iclr/2017/sharma2017iclr-learning/（ICLR 2017 の書誌ページ。OpenReview のページ https://openreview.net/forum?id=B1GOWV5eg へのリンクを含む。OpenReview のページ自体は確認画面が出たため本書では開いていない）

論文が引用する文献（本書で言及したもの）:

[Bellemare et al., 2013/06] Marc G. Bellemare, Yavar Naddaf, Joel Veness, Michael Bowling. "The Arcade Learning Environment: An Evaluation Platform for General Agents." Journal of Artificial Intelligence Research, pp. 253–279. https://doi.org/10.1613/jair.3912

[Buckland & Lawrence, 1994] Kenneth M. Buckland, Peter D. Lawrence. "Transition Point Dynamic Programming." Advances in Neural Information Processing Systems 6（NIPS 1993）. https://papers.neurips.cc/paper_files/paper/1993/hash/362e80d4df43b03ae6d3f8540cd63626-Abstract.html

[Dietterich, 2000] Thomas G. Dietterich. "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition." Journal of Artificial Intelligence Research. https://doi.org/10.1613/jair.639

[Duff, 1995] Steven Bradtke, Michael Duff. "Reinforcement Learning Methods for Continuous-Time Markov Decision Problems." Advances in Neural Information Processing Systems 7（NIPS 1994）. https://proceedings.neurips.cc/paper/1994/hash/07871915a8107172b3b5dc15a6574ad3-Abstract.html（論文の参考文献は著者を Steven J Duff、年を 1995 と記載している）

[Durugkar et al., 2016/06] Ishan P. Durugkar, Clemens Rosenbaum, Stefan Dernbach, Sridhar Mahadevan. "Deep Reinforcement Learning With Macro-Actions." arXiv:1606.04615. https://arxiv.org/abs/1606.04615

[Gu et al., 2016/10] Jiatao Gu, Graham Neubig, Kyunghyun Cho, Victor O.K. Li. "Learning to Translate in Real-time with Neural Machine Translation." arXiv:1610.00388. https://arxiv.org/abs/1610.00388

[Hausknecht & Stone, 2016/05] Matthew Hausknecht, Peter Stone. "Deep Reinforcement Learning in Parameterized Action Space." 4th International Conference on Learning Representations（arXiv:1511.04143）. https://arxiv.org/abs/1511.04143

[Hausknecht et al., 2016/05] Matthew Hausknecht, Prannoy Mupparaju, Sandeep Subramanian, Shivaram Kalyanakrishnan, Peter Stone. "Half Field Offense: An Environment for Multiagent Learning and Ad Hoc Teamwork." AAMAS Adaptive Learning Agents (ALA) Workshop. https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ALA16-hausknecht.pdf

[He et al., 2015/12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." Proceedings of the IEEE International Conference on Computer Vision, pp. 1026–1034. https://doi.org/10.1109/ICCV.2015.123

[Lakshminarayanan et al., 2016/05] Aravind S. Lakshminarayanan, Sahil Sharma, Balaraman Ravindran. "Dynamic Frame skip Deep Q Network." arXiv:1605.05365. https://arxiv.org/abs/1605.05365

[LeCun et al., 2015/05] Yann LeCun, Yoshua Bengio, Geoffrey Hinton. "Deep learning." Nature 521(7553), pp. 436–444. https://doi.org/10.1038/nature14539

[Lever, 2014/06] David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller. "Deterministic Policy Gradient Algorithms." Proceedings of the 31st International Conference on Machine Learning（PMLR 32）. https://proceedings.mlr.press/v32/silver14.html（論文の参考文献は著者を Guy Lever だけと記載している。本書は記号の定義の確認にこの論文集版を参照した）

[Lillicrap et al., 2015/09] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra. "Continuous control with deep reinforcement learning." arXiv:1509.02971. https://arxiv.org/abs/1509.02971

[Mahadevan et al., 1997/07] Sridhar Mahadevan, Nicholas Marchalleck, Tapas K. Das, Abhijit Gosavi. "Self-improving factory simulation using continuous-time average-reward reinforcement learning." Proceedings of the 14th International Conference on Machine Learning（ICML 1997）. URL 未確認

[Mnih et al., 2015/02] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis. "Human-level control through deep reinforcement learning." Nature. https://doi.org/10.1038/nature14236

[Mnih et al., 2016/06] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu. "Asynchronous Methods for Deep Reinforcement Learning." Proceedings of the 33rd International Conference on Machine Learning（PMLR 48）. https://proceedings.mlr.press/v48/mniha16.html

[Satija & Pineau, 2016/06] Harsh Satija, Joelle Pineau. "Simultaneous machine translation using deep reinforcement learning." ICML 2016 Workshop on Abstraction in Reinforcement Learning. URL 未確認

[Schaul et al., 2015/11] Tom Schaul, John Quan, Ioannis Antonoglou, David Silver. "Prioritized Experience Replay." 4th International Conference on Learning Representations（arXiv:1511.05952）. https://arxiv.org/abs/1511.05952

[Schulman et al., 2015/02] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel. "Trust Region Policy Optimization." CoRR abs/1502.05477（ICML 2015）. https://arxiv.org/abs/1502.05477（本書は記号の定義の確認にこの arXiv 版を参照した。付録 C の節で触れた目的関数の向きは、この arXiv 版の v5（2017 年 4 月）で確認した）

[Sutton & Barto, 1998] Richard S. Sutton, Andrew G. Barto. "Reinforcement Learning: An Introduction." MIT Press. http://incompleteideas.net/book/first/the-book.html（論文の参考文献は書名を Introduction to reinforcement learning と記載している）

[Todorov et al., 2012/10] Emanuel Todorov, Tom Erez, Yuval Tassa. "MuJoCo: A physics engine for model-based control." 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026–5033. https://doi.org/10.1109/IROS.2012.6386109

[Vezhnevets et al., 2016/12] Alexander (Sasha) Vezhnevets, Volodymyr Mnih, Simon Osindero, Alex Graves, Oriol Vinyals, John Agapiou, Koray Kavukcuoglu. "Strategic Attentive Writer for Learning Macro-Actions." Advances in Neural Information Processing Systems（NIPS 2016）, pp. 3486–3494. https://papers.neurips.cc/paper_files/paper/2016/file/c4492cbe90fbdbf88a5aec486aa81ed5-Paper.pdf（著者の順は論文の参考文献に合わせた。参考文献は 6 人目の Agapiou までを挙げて et al. とするので、7 人目の Kavukcuoglu は arXiv 版で補った。arXiv 版 https://arxiv.org/abs/1606.04695 は Agapiou を 3 人目に置く。STRAWe の説明は arXiv 版の Section 3.1 で確認した）

[Wymann et al., 2000] Bernhard Wymann, E. Espié, C. Guionneau, C. Dimitrakakis, R. Coulom, A. Sumner. "TORCS, The Open Racing Car Simulator." Software. http://torcs.sourceforge.net

実装と動画（論文が示すもの）:

[miyosuda/async_deep_reinforce, 2016/04] miyosuda. "async_deep_reinforce." GitHub. https://github.com/miyosuda/async_deep_reinforce（年月はリポジトリの作成日）

[aravindr93/robustRL, 2016/08] aravindr93. "robustRL." GitHub. https://github.com/aravindr93/robustRL（年月はリポジトリの作成日）

[yanpanlau/DDPG-Keras-Torcs, 2016/10] yanpanlau. "DDPG-Keras-Torcs." GitHub. https://github.com/yanpanlau/DDPG-Keras-Torcs（年月はリポジトリの作成日）

[FiGAR-TRPO 動画, 2016/11] Sahil Sharma. "inverted pendulum baseline." YouTube. http://youtu.be/JiaO2tBtH-k（論文 5.2 節が FiGAR-TRPO の学習した挙動の動画として示す URL。2026-10-03 時点で再生可能で、タイトルと公開年月は YouTube の表示による）

[FiGAR-DDPG 動画, 年月不明] 著者不明. "タイトル不明." YouTube. https://youtu.be/dX8J-sF-WX4（論文 5.3 節が FiGAR-DDPG の運転の動画として示す URL。2026-10-03 時点で再生できず、動画の中身は未確認）
