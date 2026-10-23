# Making Deep Q-learning Methods Robust to Time Discretization — 詳細解説

- 著者: Corentin Tallec、Léonard Blier、Yann Ollivier [p.1]
- 所属: TAckling the Underspecified, Université Paris Sud（Tallec、Blier）、Facebook Artificial Intelligence Research（Blier、Ollivier）[p.1]
- 発表: ICML 2019（Proceedings of the 36th International Conference on Machine Learning, PMLR vol. 97, pp. 6096–6104。PMLR の書誌ページによる）
- 年: 2019
- arXiv: 1901.09732（本解説の底本は v2、PDF 欄外の日付は 2019 年 1 月 29 日）[p.1]
- URL: https://arxiv.org/abs/1901.09732 （PMLR 版: https://proceedings.mlr.press/v97/tallec19a.html ）

Date: 2026-10-03

原典照合・修正: 2026-10-23（本文・付録・数式・図を確認。底本は上記 arXiv v2）

表記の約束:

- 著作権に配慮し、本解説は論文の内容を自分の言葉で説明する。逐語的な引用は 1 箇所（15 語未満）だけにとどめ、図・表・アルゴリズムは転載せず、番号とページで参照して内容を言葉で述べる。原文と図は papers/tallec2019_time_discretization.pdf を参照されたい。
- 対象論文への参照は [p.ページ, 節・定理・図・アルゴリズム・脚注など] の形で位置だけを示す（ページは arXiv 版 PDF の通しページ）。他の文献は [著者 (年)] と書き、末尾の出典リストに対応させる。
- 数式は LaTeX で書き、記号は使う前に定義する。(n) は論文の式番号を指す。式は必要に応じて本解説の表記で書き直しており、付録の途中式はすべては載せていない。
- PMLR 版の本文・付録との照合はしていない。書誌情報だけは公式掲載ページで確認した。

アブストラクトの要約:

深層強化学習（DRL）は、ハイパーパラメータ・実装の細部・環境の小さな違いに敏感で、これが実世界への適用を妨げている [p.1, Abstract]。著者らはそのうち、ほぼ連続時間で動く環境での時間離散化（1 秒あたりのフレーム数や制御の周期）への敏感さを重要な要因として取り上げる。実験では、DQN や DDPG といった Q 学習系の手法が時間刻みを小さくすると破綻することを示し、理論面では次の結果を証明する。

```
Formally, we prove that Q-learning does not exist in continuous time.

形式的には、連続時間では Q 学習が存在しないことを証明する。
```
[p.1, Abstract]

そのうえで、幅広い時間刻みで同程度の性能を保つオフポリシーのアルゴリズムを原理から導き、その頑健性を実験で確かめる [p.1, Abstract]。アブストラクトが根拠に挙げる文献は、敏感さについて [Henderson et al. (2017)] と [Zhang et al. (2018)]、DQN について [Mnih et al. (2015)]、DDPG について [Lillicrap et al. (2015)] である。

## Table of Contents

- [1. はじめに（Introduction）](#1-はじめにintroduction)
- [2. 関連研究（Related Work）](#2-関連研究related-work)
- [3. 準連続時間の強化学習（Near Continuous-Time Reinforcement Learning）](#3-準連続時間の強化学習near-continuous-time-reinforcement-learning)
  - [3.1 枠組み（Framework）](#31-枠組みframework)
  - [3.2 連続時間には Q 関数が存在しない（There is No Q-Function in Continuous Time）](#32-連続時間には-q-関数が存在しないthere-is-no-q-function-in-continuous-time)
- [4. 連続時間極限をもつ強化学習（Reinforcement Learning with a Continuous-Time Limit）](#4-連続時間極限をもつ強化学習reinforcement-learning-with-a-continuous-time-limit)
  - [4.1 Advantage Updating](#41-advantage-updating)
  - [4.2 時間刻みに不変な探索（Timestep-Invariant Exploration）](#42-時間刻みに不変な探索timestep-invariant-exploration)
  - [4.3 Deep Advantage Updating のアルゴリズム（Algorithms for Deep Advantage Updating）](#43-deep-advantage-updating-のアルゴリズムalgorithms-for-deep-advantage-updating)
  - [4.4 学習率のスケーリング（Scaling the Learning Rates）](#44-学習率のスケーリングscaling-the-learning-rates)
- [5. 実験（Experiments）](#5-実験experiments)
  - [共通の実験設定](#共通の実験設定)
  - [定性的実験: 方策と価値の可視化（Qualitative experiments: Visualizing policies and values）](#定性的実験-方策と価値の可視化qualitative-experiments-visualizing-policies-and-values)
  - [定量的実験（Quantitative experiments）](#定量的実験quantitative-experiments)
- [6. 考察（Discussion）](#6-考察discussion)
  - [価値関数の滑らかさ（Smoothness of the value function）](#価値関数の滑らかさsmoothness-of-the-value-function)
  - [メモリバッファの大きさ（Memory buffer size）](#メモリバッファの大きさmemory-buffer-size)
  - [準連続強化学習と RMSProp（Near-continuous reinforcement learning and RMSProp）](#準連続強化学習と-rmspropnear-continuous-reinforcement-learning-and-rmsprop)
- [7. 結論（Conclusion）](#7-結論conclusion)
  - [謝辞（Acknowledgments）](#謝辞acknowledgments)
- [A. 証明（Proofs）](#a-証明proofs)
  - [Lemma 1](#lemma-1)
  - [Theorem 6](#theorem-6)
  - [Theorem 7](#theorem-7)
  - [Theorem 8](#theorem-8)
  - [Theorem 9](#theorem-9)
  - [Theorem 10](#theorem-10)
- [B. 実装の詳細（Implementation details）](#b-実装の詳細implementation-details)
  - [B.1 全実験共通のハイパーパラメータ（Global hyperparameters）](#b1-全実験共通のハイパーパラメータglobal-hyperparameters)
  - [B.2 環境ごとのハイパーパラメータ（Environment dependent hyperparameters）](#b2-環境ごとのハイパーパラメータenvironment-dependent-hyperparameters)
- [C. 追加結果（Additional results）](#c-追加結果additional-results)
- [論文が述べている限界](#論文が述べている限界)
- [論文内の表記の食い違い](#論文内の表記の食い違い)
- [キューの論点への答え](#キューの論点への答え)
- [出典リスト](#出典リスト)

## 1. はじめに（Introduction）

DRL は完全情報ゼロサムゲーム [Silver et al. (2017)]、ビデオゲーム [OpenAI (2018b)]、[Mnih et al. (2015)]、デモなしの器用な物体操作 [OpenAI (2018a)] で成果を上げた。一方で設定や環境の違いへの敏感さ [Henderson et al. (2017)]、[Zhang et al. (2018)] とサンプル効率の悪さが実世界での利用を阻み、環境パラメータへの敏感さは不完全なシミュレータから実機への移行も難しくしている [p.1, Section 1]。

本論文はそのうち時間離散化への敏感さを扱う。著者の例では、観測と行動を毎秒 10 回から 50 回に増やす場面である。反応が速くなるのだから性能は上がりこそすれ下がらないはずだが、状態行動価値を推定する DQN [Mnih et al. (2015)] や DDPG [Lillicrap et al. (2015)] は時間刻みの変更にまったく頑健でない [p.1, Section 1]。直観的な理由は、1 つの行動を保つ時間が短いほど、その行動がリターンにもたらす差も小さくなることにある。最適行動価値 $Q^{\ast}(s,a)$（行動 $a$ をとってから最適に振る舞ったときの価値）でみると、行動どうしの差は時間刻みとともに縮み、割引率を適切に調整してもこの縮みは残る。時間刻みが無限小の極限では Q 関数は行動を区別できなくなる（Theorem 2）。この議論は連続時間強化学習 [Doya (2000)]、[Baird (1994)] の枠組みで厳密にできる [p.1, Section 1]。

連続時間を選ぶのは理論が明快になるからで、多数の小さな行動の積み重ねで価値が決まる設定にも問題意識は当てはまり、適切なスケーリングや再パラメータ化がなければ学習が失敗しうると著者は論じる [p.1, Section 1]。目標は、時間刻みの変更にできるだけ不変で、時間刻みを 0 に近づけた極限が存在するアルゴリズムであり、この要請からアーキテクチャ・探索方策・学習率のスケーリングが決まる。時間刻みのオーダーが分かれば、多くのハイパーパラメータの刻みへの依存関係・オーダーについて指針が得られる [p.1–2, Section 1]。

貢献は 3 点である [p.2, Section 1]。

1. [Baird (1994)] を土台に、準連続時間では Q 関数が V 関数へ縮退し、標準的な Q 学習がうまく振る舞わないことを形式的に示す。
2. 連続時間極限の解析から、アーキテクチャ・探索・学習率に制約を与える頑健なオフポリシーアルゴリズムを導く。
3. 標準的な Q 学習手法は時間刻みの変化で性能を落とすが、提案手法は頑健であることを実験で示す。

論文の「collapse」には 2 つの意味がある。アルゴリズムについては学習の破綻を指し、Q 関数については行動に依存しない V 関数に一致してしまうことを指す。本解説では前者を「破綻」、後者を「縮退」と書き分ける。

## 2. 関連研究（Related Work）

この節の $\delta t$ は時間刻み（1 ステップが表す物理時間）である。

出発点は [Baird (1994)] の Advantage Updating である。Baird は小さな時間刻みで Q 学習が破綻することを指摘し、価値の部分とアドバンテージの部分を $\delta t$ に応じて別々にスケーリングする方法を提案したが、検証は 2 次・線形の問題に限られていた [p.2, Section 2]。本論文はこれを 5 つの方向に広げる。アドバンテージの正規化を学習させずにパラメータ化 (27) で満たすこと、深層ネットワークで多様な環境に初めて適用すること（Deep Advantage Updating）、Q 学習の破綻と、適切にスケーリングした Advantage Updating が縮退しないことの一般的な証明、$\delta t$ に不変な探索、そして頑健性の厳しい実験的検証である [p.2, Section 2]。

対象はオフポリシー手法に限られる。A3C [Mnih et al. (2016)]、PPO [Schulman et al. (2017)]、TRPO [Schulman et al. (2015)] などのオンポリシー手法が特定の設定で不変になりうるかは範囲外とされる [p.2, Section 2]。Dueling architecture [Wang et al. (2015)] も価値とアドバンテージを分けるが、学習は両者を足した Q に対して通常どおり行うのでスケーリングは変わらず、小さな $\delta t$ で不変にはならない [p.2, Section 2]。

著者はこの問題がマルチスケール RL とは別物だと強調する。時間スケールが 1 つしかない環境でも起こり、Q の V への縮退は時間の連続性そのものから生じる数学的現象だからである [p.2, Section 2]。時空間がともに連続な強化学習は、最適制御や Hamilton–Jacobi–Bellman（HJB）方程式（連続時空間の価値関数を特徴づける偏微分方程式）との関係で研究されてきた。[Munos & Bourgine (1998)] は格子の各状態を十分に訪れるという仮定のもとで、離散化を細かくすると HJB 方程式の解に収束する格子ベースの方法を示したが、[Doya (2000)] によれば高次元では実用的でない。Doya 自身は advantage updating や環境の時間微分のモデル化を含む、連続時空間専用のアルゴリズムを扱った [p.2, Section 2]。本論文が目指すのは、離散・連続どちらの時間と空間にも使え、連続時間でも縮退しない汎用の深層 RL アルゴリズムである [p.2, Section 2]。

## 3. 準連続時間の強化学習（Near Continuous-Time Reinforcement Learning）

Section 3 の記号:

| 記号 | 意味 | 注記 |
|---|---|---|
| $\mathcal{S}=\mathbb{R}^d$, $\mathcal{A}$ | 状態集合と行動集合 | 行動は離散でも連続でもよい |
| $s_t$, $a_t$ | 物理時間 $t$ での状態と行動 | |
| $F$ | ダイナミクス $\mathcal{S}\times\mathcal{A}\to\mathcal{S}$ | 状態の時間変化率を与える |
| $\pi$ | 決定論的な方策 $\mathcal{S}\to\mathcal{A}$ | 活用（exploitation）方策 |
| $\delta t$ | 時間刻み | 1 ステップが表す物理時間 |
| $k$ | アルゴリズム時間（ステップ数） | 物理時間との関係は $t = k\,\delta t$ |
| $\mathcal{M}_{\delta t}$ | 時間刻み $\delta t$ で離散化した MDP | 行動を $\delta t$ のあいだ保ち、$\delta t$ ごとに観測・行動する |
| $r(s,a)$ | 連続時間の報酬関数 | 時間で積分されるので、単位時間あたりの報酬に当たる |
| $\gamma$ | 物理時間の単位あたりの割引率 | $0<\gamma<1$。付録の実験では 0.8 [p.16, Appendix B.1] |
| $\gamma_{\delta t}$, $r_{\delta t}$ | $\mathcal{M}_{\delta t}$ の 1 ステップあたりの割引率と報酬 | (5) で定義 |
| $\tau$ | 軌道 $(s_t,a_t)_t$ | 付録 B の $\tau$（target network の更新係数）とは別 |
| $V^\pi$ | 連続時間の価値関数 | |
| $V^\pi_{\delta t}$, $Q^\pi_{\delta t}$ | $\mathcal{M}_{\delta t}$ の価値関数と行動価値関数 | $Q^\pi_{\delta t}$ は最初の 1 ステップだけ行動 $a$ を保ち、その後 $\pi$ に従う |
| $O(\delta t)$, $o(\delta t)$ | $\delta t\to0$ で定数倍の $\delta t$ で抑えられる量、$\delta t$ より速く 0 に向かう量 | $o(1)$ は 0 に向かう量 |

多くの RL 環境は、連続時間の系を一定の時間刻みで区切ったものである。Mujoco の環境 [Lillicrap et al. (2015)] や OpenAI Gym の classic control [Brockman et al. (2016)] では刻みはシミュレータが決め、実機ではカメラのフレームレートやアクチュエータの応答速度が刻みの下限を決める [p.2, Section 3]。刻みは主に計算資源の制約なので、細かくしても性能は落ちないはずであり（最適化は難しくなりうる）、アルゴリズムは $\delta t\to0$ でも機能し続けるべきだと著者は論じる。刻みに不変なアルゴリズムは、時間ホライズンに関わるハイパーパラメータに良い既定値を与えることにもつながる [p.2–3, Section 3]。刻みの小さい離散化環境を、本論文は準連続（near-continuous）環境と呼ぶ [p.3, Section 3]。

### 3.1 枠組み（Framework）

連続時間 MDP は微分方程式

$$\frac{ds_t}{dt} = F(s_t, a_t) \tag{1}$$

で与えられ、エージェントは $a_t = \pi(s_t)$ で行動する [p.3, Section 3.1]。簡単のためダイナミクスと活用方策は決定論的とし、探索方策だけをランダムにする（連続時間でのランダム方策の定義には注意が要り、Section 4.2 で扱う）。脚注では、状態が確率微分方程式 $ds = F(s,a)\,dt + \Sigma(s,a)\,dB_t$（(2)。$B_t$ は多次元ブラウン運動、$\Sigma$ は確率項の係数行列で、原典はこれを共分散行列と呼ぶ）に従う場合にも結果は成り立つと考えているが、形式的な扱いは範囲外としている [p.3, footnote 1]。

$\mathcal{M}_{\delta t}$ では、状態 $s$ で行動 $a$ を選ぶと、$a$ を時間 $\delta t$ だけ保って (1) を解いた先の状態に移る [p.3, Section 3.1]。頑健であるための必要条件は $\delta t\to0$ でも機能し続けることなので、関係する量が意味のある極限をもつように定義していく。本文の定理は半形式的な主張で、厳密な主張と仮定は補足資料（arXiv 版では付録 A）にある [p.3, Section 3.1]。

#### リターンと割引率（Return and discount factor）

連続時間のリターンは [Doya (2000)] に従って

$$R(\tau) = \int_0^\infty \gamma^t\, r(s_t, a_t)\, dt \tag{3}$$

とし、これを幅 $\delta t$ の和で近似した離散化リターン

$$R_{\delta t}(\tau) = \sum_{k=0}^\infty \gamma^{k\delta t}\, r(s_{k\delta t}, a_{k\delta t})\, \delta t \tag{4}$$

を考える [p.3, Section 3.1]。これが $\mathcal{M}_{\delta t}$ の通常の割引リターンと一致するように、1 ステップあたりの割引率と報酬を

$$\gamma_{\delta t} = \gamma^{\delta t}, \qquad r_{\delta t} = \delta t\, r \tag{5}$$

と置く。$\delta t = 1$ のときは通常の離散時間の設定に戻る [p.3, Section 3.1]。

#### 物理時間とアルゴリズム時間、時間ホライズン（Physical time vs algorithmic time, time horizon）

時間の数え方には、ステップ数 $k$ と物理時間 $t = k\,\delta t$ の 2 つがある。一般に 1 ステップあたりの割引係数が $\gamma$ のエージェントが見通す範囲（時間ホライズン）はおよそ $1/(1-\gamma)$ ステップで、物理時間ではその $\delta t$ 倍になる [p.3, Section 3.1]。(5) のスケーリングのもとでは

$$\frac{\delta t}{1-\gamma^{\delta t}} = -\frac{1}{\log\gamma} + O(\delta t) \tag{6}$$

となり、物理時間のホライズンは $\delta t\to0$ でも一定に保たれる（論文はこれがおよそ $1/(1-\gamma)$ だとも書く。この最後の近似は $\gamma$ が 1 に近い場合に当たる）。逆に 1 ステップの割引率を固定したまま刻みを細かくすると、物理時間のホライズンは約 $\delta t/(1-\gamma)$ に縮んで 0 に向かい、エージェントはどんどん近視眼的になる [p.3, Section 3.1]。このため以降は、提案手法にもベースラインにも (5) の割引率を使う [p.3, Section 3.1]。

#### 価値関数（Value function）

連続時間の価値関数を $V^\pi(s) = \mathbb{E}_{\tau\sim\pi}[R(\tau)\mid s_0 = s]$（(7)–(8)）、$\mathcal{M}_{\delta t}$ の価値関数を $V^\pi_{\delta t}(s) = \mathbb{E}_{\tau\sim\pi}[R_{\delta t}(\tau)\mid s_0 = s]$（(9)–(10)）とすると、後者はベルマン方程式

$$V^\pi_{\delta t}(s) = r(s,\pi(s))\,\delta t + \gamma^{\delta t}\,\mathbb{E}\left[V^\pi_{\delta t}(s_{(k+1)\delta t}) \mid s_{k\delta t} = s\right] \tag{11}$$

を満たす [p.4, Section 3.1]。$\delta t\to0$ でのこの式の極限は、$V^\pi$ についての HJB 方程式 $r + \nabla_s V^\pi\cdot F = -\log(\gamma)\,V^\pi$ になる [Doya (2000)]、[p.4, footnote 2]。

Theorem 1: 滑らかさの仮定のもとで、$\delta t\to0$ のとき $V^\pi_{\delta t}(s)\to V^\pi(s)$ が成り立つ [p.4, Theorem 1]（厳密版は付録の Theorem 6）。軌道の近似に使う Lemma 1 は、固定行動での $s\mapsto F(s,a)$ と方策を入れた $s\mapsto F(s,\pi(s))$ の $C^1$ 性、有界性、$K$-Lipschitz 性を仮定する。その上で Theorem 6 は、報酬 $r$ の有界性と $s\mapsto r(s,\pi(s))$ の Lipschitz 連続性を要求する [p.10, Appendix A, Lemma 1, Theorem 6]。

### 3.2 連続時間には Q 関数が存在しない（There is No Q-Function in Continuous Time）

価値関数と違い、行動価値関数は連続時間ではうまく定義できない。1 回の行動が Q 関数に与える影響は $O(\delta t)$ しかなく、近似 Q 関数では行動の順位づけが難しくなる。この見方自体は [Baird (1994)] に非形式的に現れている [p.4, Section 3.2]。

Theorem 2: 滑らかさの仮定のもとで、すべての $(s,a)$ について $\delta t\to0$ のとき

$$Q^\pi_{\delta t}(s,a) = V^\pi_{\delta t}(s) + O(\delta t) \tag{12}$$

が成り立つ [p.4, Theorem 2]。

したがって厳密な連続時間では $Q^\pi = V^\pi$ となり、Q 関数は行動の良し悪しの情報をまったく持たない。これが連続時間の Q 学習は存在しないという主張の中身である [p.4, Section 3.2]。

証明の流れ [p.4, Section 3.2]: $\mathcal{M}_{\delta t}$ の Q 関数は

$$Q^\pi_{\delta t}(s,a) = r(s,a)\,\delta t + \gamma^{\delta t}\,\mathbb{E}_{s'\mid s,a}\left[V^\pi_{\delta t}(s')\right] \tag{13}$$

を満たし、(1) より次の状態は $s' = s + F(s,a)\,\delta t + o(\delta t)$ である (14)。$V^\pi_{\delta t}$ が状態について連続微分可能で勾配が状態 $s$ と小さい $\delta t$ に関して一様有界で、価値自体も $\delta t$ に関して一様有界なら $V^\pi_{\delta t}(s') = V^\pi_{\delta t}(s) + O(\delta t)$ となり (15)–(16)、これを (13) に入れて $r\,\delta t = O(\delta t)$、$\gamma^{\delta t} = 1 + O(\delta t)$ を使うと (12) が得られる (17)–(19)。行動 $a$ への依存は $O(\delta t)$ の項にしか残らない。

## 4. 連続時間極限をもつ強化学習（Reinforcement Learning with a Continuous-Time Limit）

Section 4 の記号:

| 記号 | 意味 | 注記 |
|---|---|---|
| $A^\pi_{\delta t}$, $A^\pi$ | スケールし直したアドバンテージ関数 (20) とその極限 | |
| $\pi'$ | 比較する別の方策 | |
| $\theta$, $\psi$, $\Theta=(\theta,\psi)$ | 価値近似器・アドバンテージ近似器のパラメータ | Section 5 の振り子の角度 $\theta$ とは別 |
| $V_\theta$ | 価値近似器 | |
| $\bar A_\psi$, $A_\psi$ | ネットワークの生の出力と、整合性条件を満たすよう補正したアドバンテージ近似器 | (27) |
| $\pi_\phi$ | 連続行動用の方策ネットワーク（パラメータ $\phi$） | |
| $z_{k\delta t}$, $z_t$ | 探索ノイズの列と、その連続時間の確率過程 | |
| $\kappa$, $\sigma$ | OU 過程の引き戻しの強さとノイズの大きさ | 実験では 7.5 と 1.5 [p.16, Appendix B.1] |
| $B_t$ | ブラウン運動 | |
| $\varepsilon$, $\pi^{\text{noise}}$ | ε-greedy でランダム行動をとる確率と、そのときに使うノイズ方策 | |
| $\delta Q_{\delta t}$ | ベルマン残差 (35) | |
| $\eta^V_{\delta t}$, $\eta^A_{\delta t}$ | $\theta$ と $\psi$ の学習率 | Theorem 5 では $\alpha\,\delta t^\beta$ の形 |

離散時間のアルゴリズムに連続時間極限をもたせるため、著者は 3 つの要素を組み合わせる。極限でも行動の順位の情報を保つ量を学習すること、極限でも意味をもつ探索を使うこと、パラメータの軌道が素直な極限をもつよう学習率をスケーリングすることである [p.4, Section 4]。

### 4.1 Advantage Updating

$\delta t$ が 0 でなければ $Q^\pi_{\delta t}$ はまだ行動に依存するが、近似誤差が $O(\delta t)$ より大きいと順位づけは誤差に支配される [p.4, Section 4.1]。そこで [Baird (1994)] に従い、Q と V の差を $\delta t$ で割ったスケール済みアドバンテージ

$$A^\pi_{\delta t}(s,a) = \frac{Q^\pi_{\delta t}(s,a) - V^\pi_{\delta t}(s)}{\delta t} \tag{20}$$

を導入する [p.4, Section 4.1]。差が $O(\delta t)$ なので割っても有限に残る。価値とアドバンテージとでは $\delta t$ に対する大きさのオーダーが違う、という点を明示的に分けた形である。

Theorem 3: 滑らかさの仮定のもとで、$A^\pi_{\delta t}(s,a)$ は $\delta t\to0$ で極限 $A^\pi(s,a)$ をもつ。この極限は行動の情報を保ち、$\pi'$ が $\pi$ より厳密に良ければ、ある状態 $s$ で $A^\pi(s,\pi'(s))>0$ となる [p.4, Theorem 3]（厳密版は付録の Theorem 7 と Theorem 8）。

#### $A^\pi$ の学習（Learning $A^\pi$）

(20) を書き直すと $Q^\pi_{\delta t} = V^\pi_{\delta t} + \delta t\,A^\pi_{\delta t}$ (21) なので、Q 関数の近似器を

$$Q_\Theta(s,a) = V_\theta(s) + \delta t\,A_\psi(s,a) \tag{22}$$

と分解して Sarsa や Q 学習を当てればよい [p.5, Section 4.1]。$V_\theta$ と $A_\psi$ を $\delta t$ と無関係に初期化しておけば、Q に占める状態の寄与と行動の寄与の比が最初から妥当になる。

ただしこの分解は一意でない。任意の関数 $f$ について $(V_\theta - f,\; A_\psi + f/\delta t)$ も同じ $Q_\Theta$ を与えるので、順位は変わらないものの、$\delta t\to0$ で数値的に不安定になったり、学習した $A_\psi$ を解釈できなくなったりしうる [p.5, Section 4.1]。そこで整合性条件 $V^\pi_{\delta t}(s) = Q^\pi_{\delta t}(s,\pi(s))$ (23)、言い換えれば

$$A_\psi(s,\pi(s)) = 0 \tag{24}$$

を課す。この条件のもとでは $Q_\Theta = Q^\pi_{\delta t}$ から $A_\psi = A^\pi_{\delta t}$ と $V_\theta = V^\pi_{\delta t}$ が従う (25)–(26)。実装では [Wang et al. (2015)] にならい、ネットワークの出力 $\bar A_\psi$ から

$$A_\psi(s,a) = \bar A_\psi(s,a) - \bar A_\psi(s,\pi(s)) \tag{27}$$

と定義して、(24) が構成上つねに成り立つようにする [p.5, Section 4.1]。

### 4.2 時間刻みに不変な探索（Timestep-Invariant Exploration）

アルゴリズム全体を刻みに不変にするには、探索も不変でなければならない [p.5, Section 4.2]。連続行動では、[Lillicrap et al. (2015)] が使った Ornstein–Uhlenbeck（OU）過程 [Uhlenbeck & Ornstein (1930)] のノイズを行動に足す方式が、すでにこの性質をもつ。

$$a_{k\delta t} = \pi(s_{k\delta t}) + z_{k\delta t}, \qquad dz_t = -\kappa\, z_t\, dt + \sigma\, dB_t \tag{28, 29}$$

$z_{k\delta t}$ は OU 過程の離散化で、その軌道は、0 へ引き戻される力のかかったブラウン運動のような連続時間の軌道に収束する [p.5, Section 4.2]。より一般に、状態や行動と独立なノイズ列 $z_{k\delta t}$ を使う探索 $a_{k\delta t} = \pi^{\text{explore}}_{\delta t}(s_{k\delta t}, z_{k\delta t})$ (30) は、$z_{k\delta t}$ がある連続な確率過程に法則収束（分布の意味で収束）すれば連続時間の極限をもつ [p.5, Section 4.2]。離散行動では、行動の数 $\#\mathcal{A}$ と同じ次元の OU ノイズでアドバンテージの値を揺らしてから、最大のものを選ぶ。

$$\pi^{\text{explore}}(s_{k\delta t}, z_{k\delta t}) = \operatorname*{argmax}_a \left(A_\psi(s_{k\delta t},a) + z_{k\delta t}[a]\right) \tag{31}$$

$z_{k\delta t}[a]$ はノイズの $a$ 番目の成分である [p.5, Section 4.2]。

これに対し ε-greedy は、刻みが小さいと毎ステップのランダムな行動が大数の法則で平均化され、実質的に探索しなくなる [p.5, Section 4.2]。

Theorem 4（本文の概略。滑らかさなどの条件は付録 Theorem 10 を参照）: 各ステップで確率 $\varepsilon$ だけノイズ方策 $\pi^{\text{noise}}$ の行動をとり、残りは決定論的な活用方策 $\pi$ に従う ε-greedy では、$\delta t\to0$ でエージェントの軌道が決定論的な方程式

$$\frac{ds_t}{dt} = (1-\varepsilon)\,F(s_t,\pi(s_t)) + \varepsilon\,\mathbb{E}_{a\sim\pi^{\text{noise}}(a\mid s)}F(s_t,a) \tag{32}$$

の解に収束する [p.5, Theorem 4]（厳密版は付録の Theorem 10）。右辺は 2 種類の行動によるダイナミクスの重み付き平均で、ランダム性が残らない。

### 4.3 Deep Advantage Updating のアルゴリズム（Algorithms for Deep Advantage Updating）

活用方策を貪欲方策 $\pi(s) = \operatorname{argmax}_{a'} A^\pi_{\delta t}(s,a')$ とすると、真の $V^\pi_{\delta t}$ と $A^\pi_{\delta t}$ は、ベルマン方程式と整合性条件

$$V^\pi_{\delta t}(s) + \delta t\,A^\pi_{\delta t}(s,a) = r\,\delta t + \gamma^{\delta t}\,\mathbb{E}_{s'}V^\pi_{\delta t}(s'), \qquad A^\pi_{\delta t}(s,\pi(s)) = 0 \tag{33, 34}$$

のただ 1 つの解になる [p.5–6, Section 4.3]。そこで $V_\theta$ と $A_\psi$ をこの 2 式を近似的に満たすよう学習する。行動についての最大化は、離散行動では厳密に計算し、連続行動では DDPG [Lillicrap et al. (2015)] と同様に方策ネットワーク $\pi_\phi$ で代用する。原典本文は $A_\psi(s,\pi_\phi(s))$ を最大化すると表現しているが、Algorithm 2 の actor 更新は、生の $\bar A_\psi$ の行動微分 $\partial_a\bar A_\psi(s,\pi_\phi(s))\,\partial_\phi\pi_\phi(s)$ を使う [p.6, Section 4.3; p.14, Algorithm 2]。同じ actor を基準に引いた (27) の両項をまとめて actor へ微分すると値が恒等的に 0 になるため、actor の改善方向と、critic の整合性を課す引き算とは区別する（本解説による式の照合）。

(34) は (27) によって自動的に満たされるので、学習するのは (33) の 2 乗残差の最小化だけである。遷移 $(s,a,r,s')$ は探索中の軌道から取っても、リプレイバッファ [Mnih et al. (2015)] から取ってもよい。残差と、それを使う勾配更新は次のとおりである [p.6, Section 4.3]。

$$\delta Q_{\delta t} = A_\psi(s,a)\,\delta t - \left(r\,\delta t + \gamma^{\delta t}V_\theta(s') - V_\theta(s)\right) \tag{35}$$

$$\theta \leftarrow \theta + \eta^V_{\delta t}\,\frac{\delta Q_{\delta t}}{\delta t}\,\partial_\theta V_\theta(s), \qquad \psi \leftarrow \psi + \eta^A_{\delta t}\,\frac{\delta Q_{\delta t}}{\delta t}\,\partial_\psi A_\psi(s,a) \tag{36, 37}$$

ここで (35)–(37) は原典の印字どおりである。ただし (35) は予測−目標の残差であり、それを加算する (36)–(37) は、直前の二乗残差最小化という説明と符号が一致しない。Algorithm 1 は同じ予測−目標に対応する勾配を optimizer に渡し、付録 (84)–(85) と (93)–(94) は目標−予測の向きで加算する [p.6, Algorithm 1, (35)–(37); p.12, (84)–(85), (93)–(94)]。実装上は、目標値への逆伝播を止めた最小化更新なら予測−目標の勾配を減算する。この符号の整理は本解説によるもので、原典の式番号つき表示を黙って訂正したものではない。

離散行動版の手順は Algorithm 1 [p.6, Algorithm 1] にまとめられている。要点を手順の形で書くと次のようになる。

1. 各エポックで、探索方策（OU ノイズ $\nu_{\delta t}$ を使う (31) の方式）により nb_steps 回行動し、遷移 $(s,a,r,d,s')$ をバッファ $\mathcal{D}$ に貯める。$d$ はエピソード終了の信号である。
2. 続いて nb_learn 回、$\mathcal{D}$ から $N$ 個の遷移を無作為に取り出す。
3. 各遷移 $i$ について、予測値 $Q^i = V_\theta(s^i) + \delta t\,\bigl(\bar A_\psi(s^i,a^i) - \max_{a'}\bar A_\psi(s^i,a')\bigr)$ と、目標値 $\tilde Q^i = r^i\,\delta t + (1-d^i)\,\gamma^{\delta t}\,V_\theta(s'^i)$ を計算する。予測値は (22) と (27) に貪欲方策を入れた形、目標値は (33) の右辺に終了時のマスクを掛けた形である。
4. 差 $Q^i - \tilde Q^i$ を $\delta t$ で割り、$\partial_\theta V_\theta(s^i)$ と、$\partial_\psi[\bar A_\psi(s^i,a^i)-\max_{a'}\bar A_\psi(s^i,a')]$ をそれぞれ掛けたバッチ平均を求める。目標値は定数として扱い、価値側・アドバンテージ側の最小化 optimizer が、それぞれ学習率 $\alpha^V\delta t$ と $\alpha^A\delta t$ でこの勾配を減算する。最大値基準を引く項もアドバンテージ側の微分対象に含まれる（Algorithm 1 の勾配式からの本解説による整理）。

### 4.4 学習率のスケーリング（Scaling the Learning Rates）

連続時間極限が存在するには、パラメータの離散的な軌道も $\delta t\to0$ で収束しなければならず、これが学習率のスケーリングを決める [p.6, Section 4.4]。目安として、準連続系では $s' = s + O(\delta t)$ なので $\delta Q_{\delta t} = O(\delta t)$ であり、更新に使う $\delta Q_{\delta t}/\delta t$ とその勾配は $O(1)$ になる。学習率が $O(\delta t)$ なら、パラメータは時間 $\delta t$ ごとに $O(\delta t)$ だけ動き、滑らかな軌道に近づくと期待できる [p.6, Section 4.4]。

Theorem 5: 準連続 MDP のある探索軌道に沿って (35)–(37) を繰り返し、学習率を $\eta^V_{\delta t} = \alpha^V\delta t^\beta$、$\eta^A_{\delta t} = \alpha^A\delta t^\beta$（$\beta\ge0$）とする。$\delta t\to0$ で、$\beta = 1$ ならパラメータの軌道は連続な軌道に収束し、$\beta > 1$ ならパラメータは初期値から動かず、$\beta < 1$ ならパラメータは任意に短い物理時間で発散しうる [p.6, Theorem 5]（厳密版は付録の Theorem 9）。

この仮定の下で、初期値に止まらない連続時間パラメータ極限を求めるには、学習率を $\delta t$ に比例させる必要がある。厳密版 Theorem 9 は連続行動空間、勾配とヘッセ行列が有界な $C^2$ 近似器、$C^1$ の探索行動軌道などを仮定する [p.12, Appendix A, Theorem 9]。離散行動の OU 探索、リプレイからのバッチ抽出、RMSprop を組み合わせた実験全体を、この定理が直接保証するわけではない（仮定と実験手順の本解説による照合）。こうしてスケーリングを施したアルゴリズムが Deep Advantage Updating（DAU）であり、離散行動版が Algorithm 1、連続行動版が補足資料（arXiv 版 PDF の Algorithm 2）である [p.6, Section 4.4; p.14, Algorithm 2]。

## 5. 実験（Experiments）

実験の目的は、DAU が個別の調整なしに広い範囲の時間刻みで働き、標準的な深層 Q 学習はそうでないことを示すことである。比較相手は連続行動では DDPG、離散行動では DQN で、オンポリシー手法は扱わない [p.6, Section 5]。

### 共通の実験設定

Section 5 の本文には小見出しがないので、2 つの実験に共通する設定を本解説がこの見出しの下にまとめる。

DAU は Algorithm 1 と、補足資料の連続行動版（arXiv 版では Algorithm 2）を使う。ベースラインの DDPG と DQN の擬似コードは付録の Algorithm 3・4 にある [p.7, Section 5; p.14, Algorithm 2; p.15, Algorithm 3; p.15, Algorithm 4]。ベースラインには 2 つの設定を試している [p.7, Section 5; p.16, Appendix B]。

- 設定 1（本文の図。キャプションの表記は unscaled）: 割引率だけを $\gamma^{\delta t}$ とスケーリングし、報酬の倍率と学習率は基準の刻み $\delta t_0 = 10^{-2}$ に合わせた値に固定する。どの $\delta t$ でも受け取る報酬の値は $r\,\delta t_0$ である。
- 設定 2（補足資料。キャプションの表記は scaled）: (5) に従って割引率と報酬をスケーリングし、RMSProp の学習率を $O(\delta t)$ にする。

設定 1 のほうがわずかに良い結果だったので本文に載せている [p.7, Section 5]。定量的な結果はすべて 5 回の実行の平均である [p.7, Section 5]。

図の軸は次のように読む。縦軸の Scaled return は、異なる刻みどうしを比べられるように (4) の離散化リターン $R_{\delta t}$ で表したリターンで、環境がもともと報酬を $\delta t$ 倍しない場合は報酬に $\delta t$ を掛けたものに当たる。横軸は常に物理時間（エージェントが環境とやりとりした時間）であり、ステップ数ではない [p.7, Section 5; p.7, footnote 3]。同じ物理時間でも、刻みが小さい実行ほど多くのステップを経ている。

全実験で共通の設定は付録 B.1 にある [p.16, Appendix B.1]。ネットワークは 256 ユニットの全結合層を 2 層（それぞれに LayerNorm と ReLU）重ねて出力層を付けたもので、方策ネットワークは出力に tanh を加え、環境によっては入力をそれまでのサンプルの平均と標準偏差で正規化する。バッファは容量 $10^6$ の循環バッファ、環境は 256 個を並列に動かして各 10 ステップ進め（学習の合間に計 2560 回の環境とのやりとり）、その後にバッチサイズ 256 の勾配ステップを 50 回行う。物理時間の割引率は 0.8 で、常に $\gamma^{\delta t}$ とスケーリングする。optimizer はモーメンタムなしの RMSprop で、設定値 $\alpha$ は $1-\delta t$（設定 1 のベースラインでは $1-\delta t_0$）である。この $\alpha$ は学習率の定数とは別の記号で、論文はその意味を定義していない。OU ノイズは $\kappa = 7.5$、$\sigma = 1.5$、学習率の定数は特に断りがなければ価値側（$\alpha^V$、$\alpha^A$ と DDPG・DQN の $\tilde\alpha^Q$）が 0.1、方策側（$\alpha^\pi$、$\tilde\alpha^\pi$）が 0.03、target network の更新係数は $\tau = 0.9$ である。論文は価値側の定数をまとめて $\alpha_1$、方策側を $\alpha_2$ と呼ぶ。

### 定性的実験: 方策と価値の可視化（Qualitative experiments: Visualizing policies and values）

設定: OpenAI Gym classic control の振り子の振り上げ（pendulum swing-up）を使う。状態は角度 $\theta$ と角速度 $\dot\theta$ の 2 次元（この $\theta$ は Section 4 のパラメータとは別）で、行動は連続である [p.7, Section 5; p.16, Appendix B.2]。DAU と設定 1 の DDPG を $\delta t$ = 0.01、0.005、0.002、0.001、0.0005 の 5 通りで訓練し、物理時間 4、8、12、16、20 分の時点で、位相平面の各点の価値と方策を画像にする [p.7, Figure 1; p.16, Appendix B.2]。Pendulum では方策側の学習率の定数を $\alpha_2 = 0.02$、$\tau = 0$ とする [p.16, Appendix B.2]。

結果について著者は、DAU では方策と価値関数がどの刻みでも調整なしに速く収束するのに対し、DDPG では学習の様子が刻みごとに大きく変わると報告している [p.7, Section 5]。

Figure 1 [p.7, Figure 1] は学習した価値関数の図である。左に DDPG、右に DAU のブロックがあり、各ブロックは行が $\delta t$、列が訓練時点の 5×5 の画像の格子になっている。各画像は横軸が角度、縦軸が角速度で、明るいほど価値が高い。方策の図と設定 2 の DDPG での図は付録 C にある。

### 定量的実験（Quantitative experiments）

設定: OpenAI Gym の Pendulum、Cartpole、BipedalWalker、Ant、Half-Cheetah で、DAU と設定 1 のベースラインを比べる [p.7, Section 5]。付録 B.2 では Cartpole だけが離散行動で、ほかは連続行動である。試した刻みは、Ant と Cheetah が 0.05 から 0.002 までの 5 段階（0.05、0.02、0.01、0.005、0.002）、Bipedal Walker が 0.01 から 0.001 までの 4 段階（0.01、0.005、0.002、0.001）、Cartpole と Pendulum が 0.01 から 0.0005 までの 5 段階（0.01、0.005、0.002、0.001、0.0005）である [p.16, Appendix B.2]。状態の正規化は Ant、Cheetah、Bipedal Walker で使い、$\alpha_2 = 0.02$ は Bipedal Walker、Cartpole、Pendulum で、$\tau = 0$ は Cartpole と Pendulum で使う。Bipedal Walker では、環境が返す報酬を $\delta t$ に比例しない形へ変更してある。これは環境側の報酬の変更であり、式 (5) の DAU 側の $r\delta t$ を無効にするという意味ではない（脚注と式の本解説による照合） [p.16, Appendix B.2; p.16, footnote 5]。

著者の所見を手法ごとにまとめる [p.7, Section 5]。DAU は Pendulum・Bipedal Walker・Ant ではどの刻みでも妥当な性能を保ち、刻みの変化にかなり強いが、Half-Cheetah では刻みに不変とまでは言えない。DDPG の性能は刻みに左右され、刻みを小さくしたときの変わり方は環境で異なる（Ant と Cheetah では低下し、Pendulum では学習が進むほど結果のばらつきが広がる）。Cartpole はノイズのため結果を読み取りにくい。Half-Cheetah について著者は、頭で歩く・背中で歩くといった準最適な動き方が並存して価値関数に不連続ができることを原因の候補に挙げ、Section 6 の議論につなげている。

Figure 2 [p.8, Figure 2] は 5 つの環境の学習曲線を並べた図で、縦軸が Scaled return、横軸が物理時間（単位は時間）である。凡例は青系を DDPG、赤系を DAU と一括表記する。ただし Section 5 冒頭と B.2 の設定では、唯一の離散行動環境 Cartpole の比較相手は DQN であり、この点は原典内で表記が一致しない [p.6, Section 5; p.16, Appendix B.2]。色の濃さが刻みの大きさを対数目盛りで表す（淡い側が 5.00e-04、濃い側が 5.00e-02）。横軸の長さは環境ごとに異なり、曲線の周りの帯が何を表すかは論文に書かれていない。

## 6. 考察（Discussion）

提案手法は理論上は刻みに不変で、単純な移動タスクなどでは実際に頑健さを改善したが、改善の余地が残る環境もある。著者は、この理論と実践のずれの原因候補を 3 つ挙げる [p.7–8, Section 6]。その前提として、Alg. 1 は時間刻み $\delta t$ を知っている必要があるが、フレームレート、アクチュエータや観測の周波数、物理シミュレータの刻みなどから通常は分かる、と述べている [p.8, Section 6]。

### 価値関数の滑らかさ（Smoothness of the value function）

証明は $V^\pi$ の連続微分可能性を仮定しているが、これは現実には破れることがある [p.8, Section 6]。著者は振り子の振り上げを例に、初期状態によっては目標に届く前に振り子を往復させる必要があり、往復が要る状態と要らない状態の境目で最適価値関数が不連続になる。その境目では行動のアドバンテージが無限小にならないので、方策が初期状態によって異なる動き方をする環境では、連続時間の解析はほとんど至るところでしか成り立たない、と説明する。この段落は著者の考察であり、動作の切り替わる任意の環境についての一般定理として述べたものではない。

### メモリバッファの大きさ（Memory buffer size）

Theorem 5 は 1 本の軌道から順にサンプルする場合の主張だが、実験では相関を避けるためにリプレイバッファからサンプルしている [p.8, Section 6]。著者はこの考察では、1 本の探索軌道で埋めていく固定サイズの循環バッファを念頭に置く。一方、実験設定 B.1 は 256 環境を並列に動かすとしている [p.16, Appendix B.1]。バッファは全刻みで同じ大きさ（容量 $10^6$ [p.16, Appendix B.1]）だったため、バッファ内のサンプルが物理時間でどれだけ古いかは刻みによって変わる。厳密にはこれで不変性が崩れている。物理時間で一定の記憶量になるよう、大きさを $1/\delta t$ に比例させる案（メモリは多く要る）が挙げられている。

### 準連続強化学習と RMSProp（Near-continuous reinforcement learning and RMSProp）

RMSProp [Tieleman & Hinton (2012)] は勾配を 2 次モーメントの移動平均の平方根で割るので、学習率のスケーリングと干渉しうる [p.8, Section 6]。決定論的な環境では勾配は典型的に $\delta t$ に対して $O(1)$ なので、適切な学習率のオーダーは変わらない。しかし確率的な準連続環境（(2) 式）では $\delta Q_{\delta t}/\delta t$ と勾配の分散が典型的には $O(1/\delta t)$ になり、バッチサイズが固定なら RMSProp は勾配を $O(\sqrt{\delta t})$ 倍にする。その場合、学習率は $\delta t$ ではなく $\sqrt{\delta t}$ に比例させれば足りる。

さらに一般論として、系のどの構成要素についても連続時間での解析をやり直すべきだと著者は述べる。例えば状態の記憶や部分観測のために再帰型モデルを使うなら、$\delta t\to0$ でも無限小でない物理時間のあいだ記憶を保てる必要がある [Tallec & Ollivier (2018)]、[p.8–9, Section 6]。

## 7. 結論（Conclusion）

Q 学習系の手法は小さな時間刻みで学習に失敗することが理論と実験の両面から示され、連続時間の解析にもとづいて時間刻みの変更への頑健性を改善した実用的なオフポリシー深層 RL アルゴリズムが作られ、その頑健性が実験で確かめられた、と著者はまとめる [p.9, Section 7]。

### 謝辞（Acknowledgments）

著者は Harsh Satija と Joelle Pineau に、有益な指摘への謝意を述べている [p.9, Acknowledgments]。参考文献欄は転記せず、引用された文献は末尾の出典リストに収めた。

## A. 証明（Proofs）

付録 A は本文の結果を仮定つきで述べ直して証明する。大半は微積分と微分方程式の数値解法の定石に沿っており、強化学習に固有の議論を使うのは Theorem 8 だけだと著者は述べる [p.10, Appendix A]。

付録の定理と本文の定理の対応は明記されていないが、内容からは次のように対応する（本解説による整理）。

- 本文 Theorem 1（$V^\pi_{\delta t}$ の収束）: 付録 Theorem 6
- 本文 Theorem 2（Q 関数の縮退）: 本文 Section 3.2 で証明済み
- 本文 Theorem 3（$A^\pi_{\delta t}$ の極限と行動の情報）: 付録 Theorem 7（極限の存在）と Theorem 8（方策改善）
- 本文 Theorem 4（ε-greedy の縮退）: 付録 Theorem 10
- 本文 Theorem 5（学習率のスケーリング）: 付録 Theorem 9

付録 A で新たに使う記号:

| 記号 | 意味 | 注記 |
|---|---|---|
| $s^k_{\delta t}$ | 各行動を $\delta t$ のあいだ保って得る離散化軌道の $k$ 番目の状態 | |
| $K$ | Lipschitz 定数 | Theorem 9 の証明の $K_{\delta t}$（ステップ数）とは別 |
| $C$ | 正の定数 | |
| $\lfloor x\rfloor$ | 床関数（$x$ 以下の最大の整数） | $\lfloor t/\delta t\rfloor$ は時刻 $t$ までのステップ数 |
| $L_r$ | $s\mapsto r(s,\pi(s))$ の Lipschitz 定数 | |
| $\lVert r\rVert_\infty$ | 報酬の絶対値の上限 | |
| $B(T)$ | Theorem 8 の証明の補助関数 | ブラウン運動 $B_t$ とは別 |
| $F^\theta$, $F^\psi$ | パラメータの連続時間軌道を定めるベクトル場 | 環境のダイナミクス $F$ とは別 |
| $f(s)$ | 確率的方策について平均したダイナミクス $\mathbb{E}_{a\sim\pi(s)}[F(s,a)]$ | Theorem 10 |
| $\xi$ | 行動のランダム性による誤差の和 | 期待値 0。その大きさを抑える定数 $\sigma$ は OU 過程の $\sigma$ とは別 |

### Lemma 1

主張: ダイナミクス（方策を入れたものも含む）が $C^1$ 級で有界かつ $K$-Lipschitz なら、各行動を $\delta t$ のあいだ保つ離散化軌道は、連続軌道 $ds_t/dt = F(s_t,\pi(s_t))$ (38) に各点収束し、誤差は

$$\left\lVert s_t - s^{\lfloor t/\delta t\rfloor}_{\delta t}\right\rVert \le \delta t\,\frac{C}{K}\,e^{Kt} \tag{40}$$

で抑えられる [p.10, Appendix A, Lemma 1]。

証明はオイラー法の収束の古典的な議論で、1 ステップあたりの誤差 $e^k_{\delta t}$ の漸化式 $e^{k+1}_{\delta t}\le(1+K\delta t)\,e^k_{\delta t} + C\delta t^2$ (45) を解いて、$e^k_{\delta t}\le(e^{K\delta t k}-1)\,\delta t\,C/K$ (48) を得る [p.10, Appendix A]。

### Theorem 6

主張: 報酬が有界で、$s\mapsto r(s,\pi(s))$ が $L_r$-Lipschitz なら、すべての状態で $V^\pi_{\delta t}(s) = V^\pi(s) + o(1)$ となる（本文 Theorem 1 の厳密版）[p.10, Appendix A, Theorem 6]。

証明では、離散和を区分的に一定な軌道の積分に直したときの係数が $\delta t\log\gamma/(\gamma^{\delta t}-1) = 1 + O(\delta t)$ であることを使い (58)–(59)、誤差の積分を時刻 $T$ で前後に分ける。後半は $2\lVert r\rVert_\infty\,\gamma^T/\log(1/\gamma)$ 以下に (65)、前半は定数倍を除いて $\delta t\int_0^T e^{(K+\log\gamma)t}\,dt$ 以下に (67) 抑える [p.11, Appendix A]。原典 (68) は $K+\log\gamma>0$ の形の評価を記し、(69) の指数には $K$ による除算が抜けている。以下は本解説による評価の整理である。$K+\log\gamma$ が正・0・負の場合、前半の上界はそれぞれ $O(\delta t\,e^{(K+\log\gamma)T})$、$O(\delta t\,T)$、$O(\delta t)$ となる。$T=-\log\delta t/K$ とすれば後半は $O(\delta t^{-\log\gamma/K})$ で、前半も各場合に 0 へ向かい、価値の収束という定性的結論は変わらない。

### Theorem 7

主張: $V^\pi$ と $V^\pi_{\delta t}$ が連続微分可能で、$V^\pi_{\delta t}$ の勾配とヘッセ行列が状態と $\delta t$ について一様有界、かつ $\partial_s V^\pi_{\delta t}\to\partial_s V^\pi$ と仮定すると、$A^\pi_{\delta t}$ は各点で

$$A^\pi(s,a) = r(s,a) + \partial_s V^\pi(s)\,F(s,a) + \log(\gamma)\,V^\pi(s) \tag{71}$$

に収束する [p.11, Appendix A, Theorem 7]。

証明は、Q のベルマン方程式 (72) で次の状態と $\gamma^{\delta t}$ を $\delta t$ について 1 次まで展開し (73)–(74)、(20) に代入して $\delta t\to0$ とするものである。ヘッセ行列の有界性は、2 次の剰余を $\delta t$ で評価するために必要だと注記されている [p.11, Appendix A; p.11, footnote 4]。同じ式は証明の最後に (76) として再び現れる。

### Theorem 8

主張: 2 つの方策 $\pi$、$\pi'$ について、$s\mapsto r(s,\pi(s))$ と $s\mapsto r(s,\pi'(s))$ がともに連続で $V^\pi$、$V^{\pi'}$ が連続微分可能とし、アドバンテージを (76) で定める。このとき、すべての状態で $A^\pi(s,\pi'(s))\ge0$ なら $V^\pi\le V^{\pi'}$ が成り立つ。また、すべての状態で $V^{\pi'}>V^\pi$ なら、$A^\pi(s',\pi'(s'))>0$ となる状態 $s'$ が存在する [p.11, Appendix A, Theorem 8]。

証明では、$\pi'$ に従って時刻 $T$ まで報酬を受け取り、その先を $\pi$ の価値で評価する量

$$B(T) = \int_0^T \gamma^t\, r(s_t,\pi'(s_t))\,dt + \gamma^T V^\pi(s_T) \tag{78}$$

を考える。$B(0) = V^\pi(s)$、$T\to\infty$ で $B(T)\to V^{\pi'}(s)$ であり、微分すると $\dot B(T) = \gamma^T A^\pi(s_T,\pi'(s_T))\ge0$ (81)–(82) なので $B$ は増加する [p.12, Appendix A]。付録冒頭で強化学習に固有とされた議論はこれである。

### Theorem 9

主張（本文 Theorem 5 の厳密版）: 行動空間を $\mathcal A=\mathbb R^{d_a}$（$d_a$ は行動の次元）とし、価値とアドバンテージの近似器を、勾配とヘッセ行列が有界な $C^2$ 級関数とし、探索行動の軌道を $C^1$ 級とする。学習率 $\alpha^V\delta t^\beta$、$\alpha^A\delta t^\beta$（$\beta\ge0$）で本文の勾配ステップを繰り返すと、$\delta t\to0$ で、$\beta=1$ なら各固定物理時刻で連続なパラメータ軌道に各点収束し、$\beta>1$ なら初期パラメータを値とする定数軌道へ収束し、$0\le\beta<1$ なら任意に短い物理時間で任意に大きくなりうる [p.12, Appendix A, Theorem 9]。

証明では、ベルマン残差に勾配を掛けたベクトル場 $F^\theta$、$F^\psi$ (84)–(85) を定め、1 回の更新が

$$\begin{pmatrix}\theta^{k+1}_{\delta t}\\ \psi^{k+1}_{\delta t}\end{pmatrix} = \begin{pmatrix}\theta^{k}_{\delta t}\\ \psi^{k}_{\delta t}\end{pmatrix} + \delta t^\beta\,F(\theta^k_{\delta t},\psi^k_{\delta t},s_{k\delta t},a_{k\delta t}) + O(\delta t^{\beta+1}) \tag{89}$$

と書けることを示す（$F$ は $F^\theta$ と $F^\psi$ を並べたもの）[p.12, Appendix A]。$\beta = 1$ ならこれは常微分方程式 $\frac{d}{dt}(\theta_t,\psi_t) = F$ (90) のオイラー法そのものであり、$\beta>1$ なら変化率が 0 に向かう (91)。どちらも Lemma 1 と同じ型の議論で収束が言える。$\beta<1$ については反例を作る。行動によらず $s_t = \sin t$ と動き、報酬が常に 0 の MDP（真の価値は 0）で $V_\theta(s) = \theta s$ を学習させると、$K_{\delta t} = \lfloor\delta t^{-(\beta+3)/4}\rfloor$ ステップのうちに $\log\theta$ が増えて発散するとする (92)–(100) [p.12–13, Appendix A]。ただし原典 (97)–(99) は添字と刻みのべきが整合せず、(99) には $\delta t^{(\beta-1)/3}$ と印刷されている。(94) からの本解説による再計算では、正の初期値・$\alpha>0$ のもとで $\log\theta^{K_{\delta t}}_{\delta t}-\log\theta^0_{\delta t}=\frac{\alpha}{2}K_{\delta t}^2\delta t^{\beta+1}(1+o(1))=\frac{\alpha}{2}\delta t^{(\beta-1)/2}(1+o(1))$ となる。原典の指数をそのまま計算結果とは扱えないが、$0\le\beta<1$ で発散しうるという定性的結論とは整合する。このステップ数に当たる物理時間 $K_{\delta t}\,\delta t\approx\delta t^{(1-\beta)/4}$ は $\beta<1$ なら 0 に向かうので、発散は物理時間でみて一瞬のうちに起こる。

### Theorem 10

主張: 行動を各刻みで確率的方策 $\pi(s,\cdot)$ から引いて $\delta t$ のあいだ保つとき、$F$ が $C^1$ 級でその導関数が有界、$\pi$ が $C^1$ 級で有界なら、軌道は $\delta t\to0$ で、行動について平均したダイナミクスの決定論的な方程式

$$\frac{ds_t}{dt} = \mathbb{E}_{a\sim\pi(s_t,\cdot)}F(s_t,a) \tag{102}$$

の解に各固定時刻で確率収束する。なお定理の列挙条件とは別に、証明では $F$ 自体の有界性やコンパクトな空間にも言及するため、列挙された条件だけの完全な再証明を本解説が与えるものではない [p.13, Appendix A, (108)–(111)]。ε-greedy の場合、これは (32) と同じ形の (103) になる [p.13, Appendix A, Theorem 10]。

証明は、刻み $\delta t^2$ の軌道を $\lfloor 1/\delta t\rfloor$ ステップ（物理時間でほぼ $\delta t$）ずつまとめて見るものである。その間の変化を平均ダイナミクス $f$ による部分と、期待値 0 で $\mathbb{E}\lVert\xi\rVert^2\le\sigma^2\delta t^3$ (108) を満たす誤差 $\xi$ とに分け、$f$ のオイラー近似から $\varepsilon$ 以上ずれる確率が $O(\sqrt{\delta t})$ であること (124) から確率収束（ずれが一定以上になる確率が 0 に向かうこと）を示す。最後に Lemma 1 と同じ型の議論で連続な方程式への収束を結論する [p.13–14, Appendix A]。

付録 A の途中式のうち、ここで番号だけを挙げたものや触れなかったもの（(41)–(51)、(52)–(70)、(83)–(100)、(104)–(124) の大部分）は、PDF の該当ページを参照されたい。

## B. 実装の詳細（Implementation details）

付録 B は、連続行動版 DAU（Algorithm 2）、ベースラインの DDPG（Algorithm 3）と DQN（Algorithm 4）の擬似コードと、ハイパーパラメータを示す [p.14, Appendix B]。

- Algorithm 2 [p.14, Algorithm 2] は Algorithm 1 に方策ネットワーク $\pi_\phi$ を加えたものである。予測値に含まれる $\max_{a'}\bar A_\psi(s,a')$ を $\bar A_\psi(s,\pi_\phi(s))$ に置き換え、$\pi_\phi$ は $\partial_a\bar A_\psi(s,\pi_\phi(s))\,\partial_\phi\pi_\phi(s)$ のバッチ平均を使って学習率 $\alpha^\pi\delta t$ で、アドバンテージを増やす方向へ更新する。通常の最小化 optimizer に渡すなら、この actor 勾配には負符号が必要である。これは本文の最大化目的と勾配式からの整理で、原典の擬似コードは optimizer の更新方向を明記していない。
- Algorithm 3 と 4 [p.15, Algorithm 3; p.15, Algorithm 4] は target network を使う標準的な DDPG と DQN である。目標値は DDPG では $r + (1-d)\,\gamma\,Q_{\psi'}(s',\pi_{\phi'}(s'))$、DQN では $r + (1-d)\,\gamma\max_{a'}Q_{\psi'}(s',a')$ で、target network のパラメータは学習ステップごとに $\psi'\leftarrow\tau\psi' + (1-\tau)\psi$ と更新する（DDPG の方策の target $\phi'$ も同様）。この式から、$\tau = 0$ なら target network は毎回オンラインのネットワークと同じになる（本解説による補足）。擬似コード自体には $\delta t$ が現れず、割引率・報酬・学習率のスケーリングは Section 5 で述べた 2 つの設定で与える。

2 つの設定の詳細は次のとおりである [p.14, Appendix B; p.16, Appendix B]。設定 2（スケーリングあり）は比較を公平にするためのもので、割引率を $\gamma^{\delta t}$、報酬を $r\,\delta t$ とし、RMSprop を使うので学習率を $\alpha^Q = \tilde\alpha^Q\delta t$、$\alpha^\pi = \tilde\alpha^\pi\delta t$ とする。設定 1（スケーリングなし）は割引率だけを $\gamma^{\delta t}$ とし、報酬と学習率は $\delta t_0 = 10^{-2}$ を使って $r\,\delta t_0$、$\tilde\alpha^Q\delta t_0$ などに固定する。方策側についても原典は $\alpha^\pi=\tilde\alpha^Q\delta t_0$ と印刷しており、設定 2 の $\tilde\alpha^\pi$ と表記が異なる。この不一致を黙って補正せず記しておく。設定 1 は刻みに依存しないが、少なくとも最も大きい刻みではそれなりに機能すると著者は書いている。

### B.1 全実験共通のハイパーパラメータ（Global hyperparameters）

値は [共通の実験設定](#共通の実験設定) に文章でまとめた [p.16, Appendix B.1]。

### B.2 環境ごとのハイパーパラメータ（Environment dependent hyperparameters）

行動空間の種類、状態正規化の有無、$\alpha_2$ と $\tau$ の変更、試した刻みの範囲は [定量的実験（Quantitative experiments）](#定量的実験quantitative-experiments) に文章でまとめた [p.16, Appendix B.2]。

## C. 追加結果（Additional results）

付録 C は、追加結果を示すという短い導入文と図からなり、図別の追加考察はない [p.16, Appendix C]。

- Figure 3 [p.17, Figure 3]: Figure 1 と同じ配置で、設定 1 の DDPG と提案手法の方策を示す。色は行動の値で、明るいほど +1、暗いほど −1 に近い。
- Figure 4 [p.17, Figure 4]: Figure 3 の DDPG を設定 2 に替えた方策の図。
- Figure 5 [p.18, Figure 5]: 設定 2 の DDPG と提案手法の価値関数（Figure 1 と同じ配置）。
- Figure 6 [p.18, Figure 6]: Figure 2 と同じ 5 つの環境で、DAU と設定 2 のベースラインの学習曲線を比べたもの。キャプションと青系の凡例は DDPG と表記するが、Cartpole の手法名については Section 5 と同じ不一致がある。

Figure 3〜5 は Pendulum の方策・価値の可視化であり、本文 Section 5 がこの定性的実験について述べる、DAU は刻みによらず速く収束し DDPG は刻みによって学習の様子が異なるという所見と対応する [p.7, Section 5]。Figure 6 は全 5 環境の定量結果であり、Cartpole はノイズで解釈しにくく Half-Cheetah では DAU も明確に不変でない、という本文の留保を外して全環境の収束を主張するものではない [p.7, Section 5; p.18, Figure 6]。

## 論文が述べている限界

この節は本解説が追加したもので、論文自身が述べている限界を言い換えて集める。

- Half-Cheetah では DAU も刻みに不変とは言えず、Cartpole はノイズが大きく結果を解釈しにくい [p.7, Section 5]。改善の余地が残る環境があることは著者も認めている [p.7–8, Section 6]。
- アルゴリズムを使うには時間刻み $\delta t$ を知っている必要がある [p.8, Section 6]。
- 証明は価値関数の滑らかさを前提にしており、方策が初期状態によって異なる動き方をする環境では、解析はほとんど至るところでしか成り立たない [p.8, Section 6]。
- 理論は逐次的なサンプルを前提としており、実験で使った固定サイズのリプレイバッファは厳密には不変性を崩す [p.8, Section 6]。
- 確率的な環境では RMSProp の正規化のため、適切な学習率のスケーリングが $\sqrt{\delta t}$ に変わりうる [p.8, Section 6]。
- 再帰型モデルなど系の他の構成要素については、連続時間の解析を別に行う必要がある [p.8–9, Section 6]。
- オンポリシー手法と、確率微分方程式に従う状態の形式的な扱いは範囲外である [p.2, Section 2; p.3, footnote 1]。
- 本文の定理は半形式的で、厳密な主張と仮定は付録にある [p.3, Section 3.1]。

## 論文内の表記の食い違い

この節も本解説が追加したもので、読む際に戸惑いやすい原典の表記の不一致を列挙する。数式を突き合わせて得た解説側の整理は、その旨を区別して記す。

- 本文は連続行動版の DAU を補足資料の Alg. 1 と呼ぶが、arXiv 版 PDF の付録では Algorithm 2 である [p.7, Section 5; p.14, Algorithm 2]。
- 本文は付録の図を補足資料の Figs. 1, 2, 3 と呼ぶが、付録の図番号は Figure 3〜6 で、対応は明記されていない [p.7, Section 5; p.17–18]。
- Figure 3〜5 のキャプションは提案手法を AU と書き、図中の見出しは DAU である [p.17, Figure 3; p.17, Figure 4; p.18, Figure 5]。
- Section 5 冒頭は離散行動では DQN と比べるとするが、離散行動の Cartpole を含む定量的実験の段落と Figure 2 の凡例は、比較相手を DDPG と記している [p.6, Section 5; p.7, Section 5; p.8, Figure 2]。
- Algorithm 1 は入力欄で optimizer を $\mathrm{opt}_V$、$\mathrm{opt}_A$ と書き、更新の行では $\mathrm{opt}_1$、$\mathrm{opt}_2$ と書く [p.6, Algorithm 1]。
- 本文 (35) の予測−目標の残差と、それを加算する (36)–(37) は、二乗残差最小化の説明・Algorithm 1・付録 (84)–(85) の更新方向と整合しない。原典表示と、実装に用いる降下方向は Section 4.3 で区別した [p.6; p.12]。
- Theorem 5 は 2 つめの学習率を添字 $\delta t$ のない $\eta^A$ と書く（本解説では $\eta^A_{\delta t}$ とした）[p.6, Theorem 5]。
- Lemma 1 の証明は 1 ステップ内の解を (101) の解と呼ぶが、補題の中でそれを定めているのは (39) である [p.10, Appendix A]。
- Theorem 7 は極限 $A^\pi$ を状態だけの関数（$\mathcal{S}\to\mathbb{R}$）として述べるが、(71) では状態と行動の関数として書かれている [p.11, Appendix A, Theorem 7]。
- Theorem 6 の証明の (63) と (65) では、被積分関数の一部が $r(s_t)$、$\tilde r(\tilde s_t)$ と、前後の式と異なる記号で印刷されている（$\tilde r(s) = r(s,\pi(s))$）[p.11, Appendix A]。
- Theorem 6 の証明は (68) の積分評価で $K+\log\gamma$ の符号による場合分けを省略し、(69) の $\delta t$ の指数には $1/K$ が抜けている。解説側の評価は Theorem 6 の節で区別して示した [p.11, (67)–(70)]。
- Theorem 9 の証明では、(88) の勾配が $\partial_\psi A_{\theta^k_{\delta t}}$ と、(97) が両辺とも $\rho^k_{\delta t}$（$\rho^k_{\delta t} = \log\theta^k_{\delta t}$）と、(99) の初期値が $\rho^0_\delta$ と印刷されている [p.12–13, Appendix A]。また離散軌道の収束を Thm. 1 と同じ筋道で示すとしているが、離散化軌道の収束そのものを扱うのは Lemma 1 である [p.12, Appendix A]。
- Theorem 9 の (98) は主項と剰余で $\delta t$ のべきが異なり、(99) の指数 $(\beta-1)/3$ も、(94) と $K_{\delta t}$ の定義から再計算した $(\beta-1)/2$ と一致しない。両者は付録 Theorem 9 の解説で区別した [p.12–13, (94)–(100)]。
- 付録 B は設定 1 の方策側の学習率を $\alpha^\pi = \tilde\alpha^Q\delta t_0$ と書く [p.16, Appendix B]。
- 付録 B.1 は列挙した値を全実験で一定とするが、B.2 は環境によって $\alpha_2$ と $\tau$ を変えている [p.16, Appendix B.1; p.16, Appendix B.2]。
- (57)、(60)、(96)、(123) は、PDF 上で番号だけがあり式の中身がない行である [p.10–14, Appendix A]。

## キューの論点への答え

reports/INDEX.md のキューに書かれた問いに、論文の記述の範囲で答える。論文が扱っていない点はそう明記する。

1. 時間刻みを細かくすると Q が V に縮退して行動の差が消える、という主張が成り立つ条件
   - 設定: 対象は、連続時間の系 (1) を刻み $\delta t$ で区切り、選んだ行動を $\delta t$ のあいだ保つ準連続 MDP で、行動は離散でも連続でもよい [p.3, Section 3.1]。比べるのは、最初の 1 ステップだけ行動 $a$ をとってその後は $\pi$ に従う $Q^\pi_{\delta t}$ と、$V^\pi_{\delta t}$ である [p.4, Section 3.2]。割引率と報酬を (5) で刻みに合わせた枠組みでの主張であり [p.3, Section 3.1]、著者は割引率を適切に調整しても行動の差は縮むと述べている [p.1, Section 1]。
   - 仮定: Theorem 2 は滑らかさの仮定のもとでの主張である [p.4, Theorem 2]。証明で使うのは、次の状態が $s + F(s,a)\,\delta t + o(\delta t)$ と書けること (14)、$V^\pi_{\delta t}$ が状態について連続微分可能で、勾配が状態と小さい $\delta t$ に関して一様有界、価値自体も小さい $\delta t$ に関して一様有界であること、そして $r\,\delta t = O(\delta t)$ と $\gamma^{\delta t} = 1 + O(\delta t)$ である [p.4, Section 3.2]。ダイナミクスと活用方策は簡単のため決定論的としており、状態が確率微分方程式に従う場合も成り立つと著者は考えているが、形式的には扱っていない [p.3, Section 3.1; p.3, footnote 1]。
   - 何が消えるか: 価値関数 $V^\pi_{\delta t}$ 自体は連続時間の $V^\pi$ に収束し [p.4, Theorem 1]、行動による違いは $O(\delta t)$ にとどまるので、厳密な連続時間では $Q^\pi = V^\pi$ となって行動の順位の情報が無くなる [p.4, Theorem 2; p.4, Section 3.2]。$\delta t$ が 0 でないうちは Q はまだ行動に依存しており、実際に困るのは近似誤差が $O(\delta t)$ より大きく、順位が誤差に支配されるときである [p.4, Section 4.1]。差を $\delta t$ で割ったアドバンテージ (20) は極限をもち、その極限は行動の情報を保つ [p.4, Theorem 3]。
   - 成り立たない所: 著者が考察で挙げる振り子の振り上げの例では、価値関数が不連続になる境界で行動のアドバンテージが無限小にならない [p.8, Section 6]。この例では、往復が要る初期状態と要らない初期状態の境目で最適価値関数が不連続になり、連続時間の解析はほとんど至るところでしか成り立たない [p.8, Section 6]。
   - 適用範囲: 著者は、連続時間を選んだのは理論を明快にするためで、多数の小さな行動の積み重ねで価値が決まる設定にも同じ問題意識が当てはまり、適切なスケーリングとパラメータ化がなければ標準的な Q 学習は失敗しうると示唆している [p.1, Section 1]。環境に時間スケールが 1 つしかなくても起きる問題だとも述べる [p.2, Section 2]。
   - 論文が扱っていない点: 行動間の Q の差を実験で直接測ってはいない。実験で示しているのは、学習した価値関数・方策と性能が刻みによってどう違うかである [p.7, Section 5]。1 つの行動を 1 ステップより長く保つ設定（同じ行動の反復など）も扱っていない。
   - 本解説の推論: 著者は差が縮む理由を、1 つの行動を保つ時間が短いことに求めている [p.1, Section 1]。そこから考えると、行動を保つ物理時間を刻みと無関係に固定する場合には、この定理の「最初の行動を $\delta t$ だけ保つ」という設定が変わる。その場合の差が極限で消えるかは、本論文の結論からは決められない。また、同一方策・遷移・割引で報酬の倍率だけを比べると、報酬を (5) のように $\delta t$ 倍しない場合（実験の設定 1 のベースラインがこれに当たる）は、Q と V がともに (5) に従う場合の $\delta t_0/\delta t$ 倍になるので、差の絶対値が 0 に向かうとは一般には言えない。この倍率関係だけから、値が 0 となる状態も含めて V に対する比を一律に論じることはできない。
2. 対策（advantage updating を深層化した DAU）の具体形
   - 分解: Q の近似器を $V_\theta(s) + \delta t\,A_\psi(s,a)$ に分け (22)、$A_\psi$ はネットワークの出力 $\bar A_\psi$ から活用方策の行動での値を引いて作る (27) ので、整合性条件 $A_\psi(s,\pi(s)) = 0$ (24) が構成上つねに成り立つ [p.5, Section 4.1]。離散行動では活用方策が貪欲方策なので、引く値は $\max_{a'}\bar A_\psi(s,a')$ になる [p.6, Algorithm 1]。$V_\theta$ と $A_\psi$ を $\delta t$ と無関係に初期化すれば、Q に占める状態の寄与と行動の寄与の比が初めから妥当になる [p.5, Section 4.1]。[Baird (1994)] からの変更として論文が最初に挙げるのは、この正規化を学習させずにパラメータ化で満たす点である [p.2, Section 2]。
   - ネットワーク: $V_\theta$ と $\bar A_\psi$ は、パラメータ（$\theta$ と $\psi$）も optimizer も別々に持つ [p.6, Algorithm 1]。実験のネットワークはすべて、256 ユニットの全結合層を 2 層（それぞれに LayerNorm と ReLU）重ねて出力層を付けた形である [p.16, Appendix B.1]。
   - 学習: 予測値 $V_\theta(s) + \delta t\,A_\psi(s,a)$ と目標値 $r\,\delta t + (1-d)\,\gamma^{\delta t}V_\theta(s')$ の差を $\delta t$ で割って $V_\theta$ と $A_\psi$ の勾配に掛け、学習率 $\alpha^V\delta t$ と $\alpha^A\delta t$ の最小化 optimizer でその勾配を減算する。目標値は定数として扱い、アドバンテージ側は最大値基準を引いた項も微分する（Algorithm 1 の式からの整理）[p.6, Algorithm 1]。目標値は学習中の $V_\theta$ でそのまま計算し、target network は使わない [p.6, Algorithm 1]。遷移は探索中の軌道から取ってもリプレイバッファから取ってもよく [p.6, Section 4.3]、実験では容量 $10^6$ の循環バッファを使う [p.16, Appendix B.1]。
   - 刻みに合わせる量: 割引率 $\gamma^{\delta t}$ と報酬 $r\,\delta t$ (5) [p.3, Section 3.1]、そして $\delta t$ に比例する学習率である [p.6, Section 4.4]。Theorem 5 によれば、付録 Theorem 9 の滑らかさ・探索軌道などの仮定の下で、学習率を $\delta t^\beta$ に比例させると、$\beta = 1$ ではパラメータの軌道が連続な軌道に収束し、$\beta > 1$ では初期値から動かず、$\beta < 1$ では任意に短い物理時間で発散しうる [p.6, Theorem 5]。
   - 探索: 離散行動では、行動の数と同じ次元の OU 過程のノイズをアドバンテージに足してから、最大の行動を選ぶ (31) [p.5, Section 4.2]。実験の OU 過程は $\kappa = 7.5$、$\sigma = 1.5$ である [p.16, Appendix B.1]。
   - 連続行動: 行動についての最大化を方策ネットワーク $\pi_\phi$ で代用し、Algorithm 2 の $\partial_a\bar A_\psi(s,\pi_\phi(s))\partial_\phi\pi_\phi(s)$ によって生アドバンテージを増やす方向へ更新する。整合化後の $A_\psi$ に同じ actor を両項とも通して微分する、という意味ではない（Section 4.3 と擬似コードの照合）[p.6, Section 4.3; p.14, Algorithm 2]。
3. 離散行動での結果
   - 環境と所見: 離散行動の環境は Cartpole だけである [p.16, Appendix B.2]。著者が Cartpole について述べるのは、ノイズのため結果を解釈しにくいということだけで、刻みへの頑健さについての結論は出していない [p.7, Section 5]。学習曲線は Figure 2（設定 1）と Figure 6（設定 2）の Cartpole の枠にあり、論文に数値の表は無い [p.8, Figure 2; p.18, Figure 6]。
   - 比較相手: Section 5 の冒頭は離散行動では DQN と比べるとするが、定量的実験の段落と Figure 2 の凡例は比較相手を DDPG と書いている（「論文内の表記の食い違い」に挙げた点）[p.6, Section 5; p.7, Section 5; p.8, Figure 2]。
   - 設定: 刻みは 0.01 から 0.0005 までの 5 段階で、$\tau = 0$、$\alpha_2 = 0.02$ とする [p.16, Appendix B.2]。$\tau$ は DDPG と DQN の擬似コードにだけ現れる target network の更新係数なので、$\tau = 0$ では比較相手の target network が毎回オンラインのネットワークと同じになる（本解説による補足）[p.15, Algorithm 3; p.15, Algorithm 4]。$\alpha_2$ は方策側の学習率の定数だが、Algorithm 1 と Algorithm 4 には方策ネットワークが無く、離散行動でこの値がどこに効くかは書かれていない（本解説による照合）[p.6, Algorithm 1; p.15, Algorithm 4; p.16, Appendix B.1]。
   - 探索: 付録 B.1 は、探索をどの実験でも本文の方式で行うとしている [p.16, Appendix B.1]。ε-greedy を使った実験は無く、小さな刻みでは ε-greedy が探索しなくなるという Theorem 4 は理論上の結果である [p.5, Theorem 4]。
   - 論文が扱っていない点: 価値関数と方策の可視化は、連続行動の Pendulum でしか行っていない [p.7, Section 5]。評価は OpenAI Gym の 5 環境だけで [p.7, Section 5]、Atari のような画像入力の課題は扱っていない。
4. DQN 系に入れるときに要る変更
   - 基準: 論文が DAU と比べた DQN（Algorithm 4）は、リプレイバッファと、学習ステップごとに係数 $\tau$ で追従させる target network を使う形である [p.15, Algorithm 4]。以下は、これと DAU（Algorithm 1）の違いと、刻みに合わせた設定から読み取れる変更である。
   - 出力の分解: Q を直接出す代わりに、$V_\theta(s) + \delta t\,\bigl(\bar A_\psi(s,a) - \max_{a'}\bar A_\psi(s,a')\bigr)$ の形にする [p.5, Section 4.1; p.6, Algorithm 1]。Dueling architecture [Wang et al. (2015)] のように価値とアドバンテージを分けるだけでは、和の Q を通常どおり学習するのでスケーリングが変わらず、刻みに不変にはならない [p.2, Section 2]。
   - 目標値と損失: 目標値は $r\,\delta t + (1-d)\,\gamma^{\delta t}V_\theta(s')$ で、予測−目標の差を $\delta t$ で割って勾配に掛け、最小化 optimizer が減算する。目標値へは逆伝播せず、アドバンテージの最大値基準を引く項は微分に含める（Algorithm 1 の式からの整理）[p.6, Algorithm 1]。この分解では $\max_a Q_\Theta(s',a) = V_\theta(s')$ なので、DQN の max をとる目標値をこの分解で書き直した形に当たる（本解説による整理）。Algorithm 1 は target network を使わず、Algorithm 4 の DQN は使う [p.6, Algorithm 1; p.15, Algorithm 4]。DAU に target network を加えたときの扱いは論文に無い。
   - 刻みに合わせる量: 割引率を $\gamma^{\delta t}$、報酬を $r\,\delta t$ にし [p.3, Section 3.1]、学習率を $\delta t$ に比例させる [p.6, Theorem 5]。実験では RMSprop の設定値 $\alpha$ も $1-\delta t$ にしている（設定 1 のベースラインは $1-\delta t_0$。この $\alpha$ の意味は論文に定義が無い）[p.16, Appendix B.1]。Theorem 5 は軌道に沿って遷移ごとに更新する場合の主張で [p.6, Theorem 5]、実験でも環境とのやりとりあたりの勾配ステップの数は刻みによらず同じである [p.16, Appendix B.1]。このため物理時間あたりの更新回数は $1/\delta t$ に比例して増える。更新に掛かる勾配が $O(1)$ なら、学習率を $\delta t$ に比例させると物理時間あたりの変化量が同じオーダーになる、というのが本文 Section 4.4 のスケーリングの見方である。リプレイや optimizer を含むパラメータ軌道の一致までを、この計算だけで保証するものではない（本解説による整理）。式 (2) のような確率系で、勾配の分散が典型的に $O(1/\delta t)$ となりバッチサイズを固定する場合は、RMSProp のため学習率を $\sqrt{\delta t}$ に比例させれば足りると考察されている [p.8, Section 6]。
   - 探索: 小さな刻みでは ε-greedy の軌道が決定論的な方程式の解に収束して探索しなくなる（Theorem 4）ので、アドバンテージに OU 過程のノイズを足して最大の行動を選ぶ (31) の方式に替える [p.5, Section 4.2; p.5, Theorem 4]。
   - 前提: 時間刻み $\delta t$ を知っている必要がある [p.8, Section 6]。リプレイバッファを全刻みで同じ大きさにすると厳密には不変性が崩れ、物理時間で一定の記憶量になるよう大きさを $1/\delta t$ に比例させる案が挙がっている [p.8, Section 6]。
   - スケーリングだけのベースライン: 割引率・報酬・学習率をすべて刻みに合わせたベースライン（設定 2）も試されたが、本文は割引率だけを合わせた設定 1 のほうがわずかに良かったとして、設定 1 を載せている [p.7, Section 5]。
   - 論文が扱っていない点: Double DQN、優先度付きリプレイ、多段の目標値、分布型の価値（QR や IQN）との組み合わせは扱っていない。再帰型モデルなど系の他の構成要素については、連続時間の解析をやり直す必要があると述べるにとどまる [p.8–9, Section 6]。

## 出典リスト

[Tallec et al., 2019/01] Corentin Tallec, Léonard Blier, Yann Ollivier. "Making Deep Q-learning Methods Robust to Time Discretization." arXiv:1901.09732v2（2019 年 1 月 29 日。ICML 2019, PMLR vol. 97, pp. 6096–6104 に収録）. https://arxiv.org/abs/1901.09732 （PMLR 版: https://proceedings.mlr.press/v97/tallec19a.html ）

[Baird, 1994] L. C. Baird. "Reinforcement learning in continuous time: Advantage updating." Proceedings of 1994 IEEE International Conference on Neural Networks (ICNN'94), vol. 4, pp. 2448–2453. https://doi.org/10.1109/ICNN.1994.374604

[Brockman et al., 2016/06] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, Wojciech Zaremba. "OpenAI Gym." arXiv:1606.01540. https://arxiv.org/abs/1606.01540

[Doya, 2000/01] Kenji Doya. "Reinforcement Learning in Continuous Time and Space." Neural Computation 12(1):219–245. https://doi.org/10.1162/089976600300015961

[Henderson et al., 2017/09] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger. "Deep Reinforcement Learning that Matters." arXiv:1709.06560. https://arxiv.org/abs/1709.06560

[Lillicrap et al., 2015/09] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra. "Continuous control with deep reinforcement learning." arXiv:1509.02971. https://arxiv.org/abs/1509.02971

[Mnih et al., 2015/02] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, et al. "Human-level control through deep reinforcement learning." Nature 518(7540):529–533. https://doi.org/10.1038/nature14236

[Mnih et al., 2016/06] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu. "Asynchronous Methods for Deep Reinforcement Learning." Proceedings of the 33rd International Conference on Machine Learning (ICML 2016), PMLR vol. 48, pp. 1928–1937. https://proceedings.mlr.press/v48/mniha16.html

[Munos & Bourgine, 1998] Rémi Munos, Paul Bourgine. "Reinforcement Learning for Continuous Stochastic Control Problems." Advances in Neural Information Processing Systems 10 (NIPS 1997), pp. 1029–1035（頁は対象論文の参考文献欄による）. https://papers.nips.cc/paper_files/paper/1997/hash/186a157b2992e7daed3677ce8e9fe40f-Abstract.html

[OpenAI, 2018/08] OpenAI（Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, Jonas Schneider, Szymon Sidor, Josh Tobin, Peter Welinder, Lilian Weng, Wojciech Zaremba）. "Learning Dexterous In-Hand Manipulation." arXiv:1808.00177（本文中の [OpenAI (2018a)]）. https://arxiv.org/abs/1808.00177

[OpenAI, 2018/06] OpenAI. "OpenAI Five." OpenAI Blog（本文中の [OpenAI (2018b)]）. https://openai.com/index/openai-five/ （論文に記載の URL は https://blog.openai.com/openai-five/ ）

[Schulman et al., 2015/07] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, Philipp Moritz. "Trust Region Policy Optimization." Proceedings of the 32nd International Conference on Machine Learning (ICML 2015), PMLR vol. 37, pp. 1889–1897. https://proceedings.mlr.press/v37/schulman15.html

[Schulman et al., 2017/07] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. "Proximal Policy Optimization Algorithms." arXiv:1707.06347. https://arxiv.org/abs/1707.06347

[Silver et al., 2017/12] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." arXiv:1712.01815. https://arxiv.org/abs/1712.01815

[Tallec & Ollivier, 2018/03] Corentin Tallec, Yann Ollivier. "Can recurrent neural networks warp time?" arXiv:1804.11188. https://arxiv.org/abs/1804.11188

[Tieleman & Hinton, 2012] Tijmen Tieleman, Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." COURSERA: Neural Networks for Machine Learning, 4(2):26–31（書誌は対象論文の参考文献欄による）. http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf （同講義 Lecture 6 のスライド。rmsprop の回は Lecture 6e と題されている）

[Uhlenbeck & Ornstein, 1930/09] G. E. Uhlenbeck, L. S. Ornstein. "On the Theory of the Brownian Motion." Physical Review 36(5):823–841. https://doi.org/10.1103/PhysRev.36.823

[Wang et al., 2015/11] Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas. "Dueling Network Architectures for Deep Reinforcement Learning." arXiv:1511.06581. https://arxiv.org/abs/1511.06581

[Zhang et al., 2018/11] Amy Zhang, Yuxin Wu, Joelle Pineau. "Natural Environment Benchmarks for Reinforcement Learning." arXiv:1811.06032. https://arxiv.org/abs/1811.06032
