# Offline Actor-Critic Reinforcement Learning Scales to Large Models — 詳細解説

- 著者: Jost Tobias Springenberg, Abbas Abdolmaleki, Jingwei Zhang, Oliver Groth, Michael Bloesch, Thomas Lampe, Philemon Brakel, Sarah Bechtle, Steven Kapturowski, Roland Hafner, Nicolas Heess, Martin Riedmiller [p.1]
- 所属: Google DeepMind（London）[p.1]
- 発表: ICML 2024（Proceedings of the 41st International Conference on Machine Learning, PMLR vol. 235, pp. 46323–46350。PMLR の書誌ページによる）
- 年: 2024
- arXiv: 2402.05546。本解説の底本は v1（2024 年 2 月 8 日、全 25 ページ）で、arXiv の書誌ページに v2 以降の記載は無い。
- URL: https://arxiv.org/abs/2402.05546 （PDF: https://arxiv.org/pdf/2402.05546 、PMLR 版: https://proceedings.mlr.press/v235/springenberg24a.html ）

Date: 2026-11-04

表記の約束:

- 著作権に配慮し、本解説は論文の内容を自分の日本語で要約する。原文より大幅に短くし、節の順番・見出し・位置引用・主要な式・出典は残す。paper-details スキルが求める「原文と訳の引用ブロックが主体の形式」と「図表画像の貼り付け」からは、この理由で外れている。
- 逐語の引用は文書全体で 1 か所（アブストラクトの節、12 語）だけである。
- 図表は転載せず、番号とページで参照して内容を言葉で述べる。議論に必要な数値だけを位置引用つきで書き、表全体は写さない。
- 読んだ版: 上記 arXiv v1 の PDF の 25 ページすべて（本文・付録）。ページ番号は PDF の印字ページと一致する。PMLR 版の本文とは照合していない（書誌情報だけ確認した）。PDF は papers/ に置いていない。原文は上の URL で確認できる。
- 章ごとの監査は未実施（費用のため）。代わりに筆者が数値と式を原文と読み合わせ、表の数値、一部の式、Figure 3 と Figure 9 の図中の印字は PDF のテキスト層でも確かめた。Figure 1 の棒の数値だけはテキスト層に無く、画像からの読み取りで、そう明記した。
- 対象論文への参照は [p.ページ, 節・図・表・式] の形で位置だけを示す。他の文献は [著者 (年)] と書き、末尾の出典リストに対応させる。
- 数式は LaTeX で、本解説の記号に書き直した。(n) は論文の式番号を指す。原典との記号の対応は 3 節の表に置く。
- 本解説が独自に行った計算・照合・推論は「本解説の計算」「本解説の推論」と明記し、論文の主張と区別する。プロジェクトの文脈は「キューの論点への答え」の項目 6 と「このプロジェクトの検討に効く点」にだけ書く。

アブストラクトの要約:

論文の主張は次の一文に集約される。

```
We show that offline actor-critic reinforcement learning can scale to large models

オフラインの actor-critic 強化学習は大きなモデルへスケールできることを示す
```
[p.1, Abstract]

根拠に挙がるのは、Perceiver 型モデルでのスケーリング則の解析と、質の混ざった 132 の連続制御タスク（実ロボットを含む）で教師ありの BC（行動模倣）を上回ったという結果である [p.1, Abstract]。

## Table of Contents

- [1. はじめに（Introduction）](#1-はじめにintroduction)
- [2. 背景と関連研究（Background and Related Work）](#2-背景と関連研究background-and-related-work)
- [3. スケールするオフライン actor-critic 学習（Scalable Offline Actor-Critic Learning）](#3-スケールするオフライン-actor-critic-学習scalable-offline-actor-critic-learning)
  - [3.1 背景と記法（Background and Notation）](#31-背景と記法background-and-notation)
  - [3.2 オフライン KL 正則化 actor-critic（Offline KL-Regularized Actor-Critic）](#32-オフライン-kl-正則化-actor-criticoffline-kl-regularized-actor-critic)
  - [3.3 actor-critic 学習のスケールする構成（Scalable Architecture for Actor-Critic Learning）](#33-actor-critic-学習のスケールする構成scalable-architecture-for-actor-critic-learning)
- [4. 実験（Experiments）](#4-実験experiments)
  - [4.1 オフライン RL の目的関数のスケーリング解析（Scaling Analysis for Offline RL Objectives）](#41-オフライン-rl-の目的関数のスケーリング解析scaling-analysis-for-offline-rl-objectives)
  - [4.2 大規模なオフライン actor-critic 学習（Large-scale Offline Actor-Critic Learning）](#42-大規模なオフライン-actor-critic-学習large-scale-offline-actor-critic-learning)
  - [4.3 RL による微調整と自己改善（RL Fine-tuning and Self-improvement）](#43-rl-による微調整と自己改善rl-fine-tuning-and-self-improvement)
- [5. 考察（Discussion）](#5-考察discussion)
- [Broader Impact](#broader-impact)
- [付録 A. アルゴリズムの詳細（Method Details - Algorithm）](#付録-a-アルゴリズムの詳細method-details---algorithm)
- [付録 B. 構成の詳細（Method Details - Architecture）](#付録-b-構成の詳細method-details---architecture)
- [付録 C. 実験の詳細（Experimental Details）](#付録-c-実験の詳細experimental-details)
- [付録 D. 感度とアブレーション実験（Sensitivity and Ablation Experiments）](#付録-d-感度とアブレーション実験sensitivity-and-ablation-experiments)
- [付録 E. スケーリングの詳細（Scaling Details）](#付録-e-スケーリングの詳細scaling-details)
- [付録 F. 追加実験（Additional Experiments）](#付録-f-追加実験additional-experiments)
- [論文が述べている限界](#論文が述べている限界)
- [論文内の記載の食い違い（本解説による照合）](#論文内の記載の食い違い本解説による照合)
- [論文から確かめられなかった点](#論文から確かめられなかった点)
- [キューの論点への答え](#キューの論点への答え)
- [このプロジェクトの検討に効く点](#このプロジェクトの検討に効く点)
- [出典リスト](#出典リスト)

## 1. はじめに（Introduction）

モデルとデータの規模を上げることが近年の機械学習の進歩を支えており、言語・視覚では大きなモデルを多様なデータで事前学習するのが標準になった [OpenAI (2023)]、[Alayrac et al. (2022)]。画像や音楽の生成も同様である [Rombach et al. (2021)]、[Kang et al. (2023)]。制御でも、似た構成のモデルと大きなデータから教師ありの BC で驚くほど有能なマルチタスク方策が得られている [Reed et al. (2022)]、[Bousmalis et al. (2023)]、[Brohan et al. (2022)]、[Octo Model Team et al. (2023)] [p.1, Section 1]。

ただし BC は質の高いデータを要し、ロボットなどの実世界の制御ではそれが高価か存在しない。そこで準最適なデータや人手を介さず生成したデータを使える RL を、モデルの構成とスケーリングの利点を保ったまま使いたい、というのが動機である [p.1, Section 1]。大きな行動モデルをオフライン RL で学習する研究は少なく、先行例は Transformer でない中規模モデル [Kumar et al. (2022a)] か、比較的小さなモデルで計算の負担が大きいもの [Chebotar et al. (2023)] にとどまる [p.1, Section 1]。脚注は、言語モデルの RLHF で使うオンライン RL は実世界の設定では高すぎるとして対比している [p.1, footnote 1]。

論文の貢献は次のとおりである [p.1–2, Section 1]。

- Perceiver-Actor-Critic（PAC）という手順とモデルを示す（Figure 1）。オフライン actor-critic は計算の追加負担を大きくせずに大きなモデルとデータへスケールし、教師あり学習で観察された法則 [Henighan et al. (2020)]、[Kaplan et al. (2020)] に似た法則に従う。
- アルゴリズムは KL 正則化 RL の目的関数を最適化する単純なもので、MPO/DIME [Abdolmaleki et al. (2018)]、[Abdolmaleki et al. (2022)] の簡略版と見なせる。方策を BC でデータへ正則化するだけで大規模モデルでも安定し、BC と RL の間を連続的に移れる。
- 構成上の工夫は 2 つ。行動を cross-attention で Q 関数へ入れること（複数行動の Q を速く評価できる）と、Perceiver 流の cross-attention で多数の入力を潜在変数へまとめることである。約 10 億パラメータのモデルを手元の機械で 20 Hz で推論できる。
- Control Suite [Tunyasuvunakool et al. (2020)] などで BC を上回り、Gato [Reed et al. (2022)] も超え、質の混ざったデータから実ロボットのベンチマークで専門家の性能を回復する。

Figure 1 [p.2, Figure 1] は PAC の位置づけの概観で、データの質の軸、α による BC と RL の補間、32M と 988M での BC に対する優位、同じデータと自己収集データでの微調整の効果を 1 枚に並べている。

## 2. 背景と関連研究（Background and Related Work）

### Supervised Generalist Agents（教師ありの汎用エージェント）

大きな Transformer [Vaswani et al. (2017)] の汎用エージェントは、制御を系列予測へ変換した研究 [Chen et al. (2021)]、[Janner et al. (2021)] の上に BC で学習されてきた。Gato [Reed et al. (2022)] は Atari からロボット操作までを扱い、続く大規模モデル [Brohan et al. (2022)]、[Bousmalis et al. (2023)]、[Zitkovich et al. (2023)] は多タスク・多物体・多機体のデータで学習され、微調整で新しいタスクへ一般化した [Bousmalis et al. (2023)]、[Open X-Embodiment Collaboration (2023)]。Perceiver 系の cross-attention [Jaegle et al. (2021)] はボクセル観測のロボットで使われ [Shridhar et al. (2023)]、[Ze et al. (2023)]、Octo [Octo Model Team et al. (2023)] も同様の注意機構で出力を予測する [p.2, Section 2]。

### Offline RL（オフライン RL）

オフライン RL [Levine et al. (2020)]、[Lange et al. (2012)] は固定データから学ぶので準最適な軌跡も使えるが、未見の状態行動対を過大評価しやすい。対策は方策をデータ近くに正則化する系 [Peng et al. (2019)]、[Wang et al. (2020)]、[Fujimoto et al. (2019)]、[Wu et al. (2019)] と、価値を罰する系 [Kumar et al. (2020)]、未見の対への価値の伝播を防ぐ系 [Kostrikov et al. (2021)] に分かれる。本論文は BC 正則化をオフポリシー RL に組み合わせる先行研究 [Abdolmaleki et al. (2022)]、[Fujimoto & Gu (2021)] の系統に属する。マルチタスク [Kumar et al. (2022a)] や事前学習 [Kumar et al. (2022b)] の例はあるが、大きなデータで Transformer をオフライン RL で学習した先行例は著者の知る限り Q-Transformer [Chebotar et al. (2023)] だけで、モデルは比較的小さい [p.2, Section 2]。

### Scaling Law Analysis（スケーリング則の解析）

大規模言語モデルではモデル規模と性能の間に滑らかなべき乗則が報告されている [Kaplan et al. (2020)]、[Hoffmann et al. (2022)]、[Henighan et al. (2020)]。オンライン RL の研究 [Neumann & Gros (2022)]、[Hilton et al. (2023)] は 4,000 万パラメータ未満にとどまる。[Lee, K.-H. et al. (2022)] は Decision Transformer 型 [Chen et al. (2021)] を 2 億パラメータまで調べ、CQL [Kumar et al. (2020)] の基準では好ましいスケーリングが見られなかったとした。同時期のモデルベース手法 [Hansen et al. (2023)] は 3 億パラメータまでのマルチタスクで有望である [p.2–3, Section 2]。

## 3. スケールするオフライン actor-critic 学習（Scalable Offline Actor-Critic Learning）

オフライン RL の手法を大きなモデルへ広げるため、3.2 節でアルゴリズムを、3.3 節で Perceiver-IO [Jaegle et al. (2022)] を actor-critic 向けにした構成を示す [p.3, Section 3]。

Section 3 の記号（左が本解説）:

| 本解説 | 原典 | 意味 | 注記 |
|---|---|---|---|
| $s$, $a$, $r$, $s'$ | $s_t$, $a_t$, $r$, $s_{t+1}$ | 状態、行動、報酬、次の状態 | 状態と多モーダル観測を同じ意味で使う |
| $g$ | $\tau$ | タスク記述 | 分位点の意味はない。原典は C.5 で温度にも $\tau$ を使う |
| $\mathcal{D}$, $b$ | $\mathcal{D}$, $b$ | オフラインデータと行動方策 | データは $(s,a,s',g)$ の組、報酬は注釈か関数で得る |
| $\pi_\theta$, $Q_\theta$ | $\pi_\theta$, $Q_\theta$ | 方策と行動価値の推定 | 同じパラメータ $\theta$ に含まれる（付録 D.1.2） |
| $\theta^{-}$ | $\theta'$ | target network のパラメータ | 実験では 100 更新ごと（付録 E.2） |
| $\bar\pi$, $\pi^{+}$ | $\tilde\pi$, $\pi_{\text{imp}}$ | 基準の方策、改善後の方策 | 主版は $\bar\pi = \pi_{\theta^{-}}$ |
| $\eta$ | $\eta$ | KL 正則化の強さ（温度） | PAC での値は論文に見当たらない |
| $\alpha$, $\beta$ | $\alpha$, $\beta$ | BC と RL の配分、TD 損失の倍率 | $\alpha=1$ で純 BC、$\alpha=0$ で純粋な方策最適化 |
| $p_\theta(q\mid s,a,g)$ | 同じ | ビンに離散化した Q の分布 | $Q_\theta=\mathbb{E}_{p_\theta}[q]$ |
| $T_{\theta^{-}}$ | $\Gamma_{\theta'}$ | 分布 TD 演算子で作る目標分布 | (9) |
| $n_a$ | $N$ | 方策から引く行動サンプル数 | 10 |
| $L$ | $N$（B.3） | 入力トークン数 | 既定で 2,634 |

### 3.1 背景と記法（Background and Notation）

マルチタスク MDP を扱う。タスクごとに報酬 $R(s,a,g)$ があり、次の状態は $p(\cdot\mid s,a)$ に従う。状態は完全には観測できないことが多いが、状態と観測を区別せずに書く [p.3, Section 3.1]。方策 $\pi(a\mid s,g)$ はタスクごとの割引累積報酬を最大化し、$Q^\pi$、$V^\pi$、$A^\pi$ は通常の定義である。オフラインデータ $\mathcal{D}$ と報酬の注釈を前提にする。BC は $b$ と $\pi$ の KL の最小化で、定数を除くと $-\mathbb{E}_{\mathcal{D}}[\log\pi(a\mid s,g)]$ になる [p.3, Section 3.1]。

### 3.2 オフライン KL 正則化 actor-critic（Offline KL-Regularized Actor-Critic）

基準の方策 $\bar\pi$ を改善する $\pi^{+}$ を、KL 罰を付けた期待 Q の最大化として求める [p.3, Section 3.2]。

$$J(\pi)=\mathbb{E}_{(s,g)\sim\mathcal{D}}\Big[\mathbb{E}_{a\sim\pi}\big[Q^{\pi}(s,a,g)\big]-\eta\,D_{\mathrm{KL}}\big(\pi(\cdot\mid s,g)\,\big\|\,\bar\pi(\cdot\mid s,g)\big)\Big] \tag{1}$$

解は基準の方策を指数化した Q で重み付けした形で、$Q$ の代わりに $A$ を使っても同じである [p.3, (2)]。

$$\pi^{+}(a\mid s,g)\propto\exp\!\big(Q^{\pi^{+}}(s,a,g)/\eta\big)\,\bar\pi(a\mid s,g) \tag{2}$$

これで RL の問題は重み付きの教師あり学習になる（[Abdolmaleki et al. (2018)] の見方）。パラメトリックな方策を $\pi^{+}$ への KL の最小化で当てはめればよく、損失がすべて重み付き負対数尤度になるので BC 項との配分が容易である [p.3, Section 3.2]。本文は Q 版を扱い、V 版は付録に回す。

Q は分布型 [Bellemare et al. (2017)] で学習する。Q の学習を離散分布の負対数尤度へ変換することで、方策の損失と規模を揃える目的である [p.3, Section 3.2]。同じ target のパラメータから target 方策 $\pi_{\theta^{-}}$ も得て、これを基準の方策に使う。方策の損失、BC 損失、Q の損失を足すと KL が 3 項の総損失になる [p.3, (3)]。

$$L(\theta)=\mathbb{E}_{\mathcal{D}}\Big[(1-\alpha)\,D_{\mathrm{KL}}\big(\pi^{+}\big\|\pi_\theta\big)+\alpha\,D_{\mathrm{KL}}\big(b\big\|\pi_\theta\big)+\beta\,D_{\mathrm{KL}}\big(T_{\theta^{-}}\big\|p_\theta\big)\Big] \tag{3}$$

サンプルで書くと、$C$ を $\theta$ に依らない定数として次の形になる [p.3, (3)]。

$$L(\theta)=-\mathbb{E}_{\mathcal{D}}\Big[(1-\alpha)\,\mathbb{E}_{a'\sim\pi_{\theta^{-}}}\big[w(a')\log\pi_\theta(a'\mid s,g)\big]+\alpha\log\pi_\theta(a\mid s,g)+\beta\,\mathbb{E}_{q\sim T_{\theta^{-}}}\log p_\theta(q\mid s,a,g)\Big]+C$$

$$w(a')=\frac{\exp\big(Q_\theta(s,a',g)/\eta\big)}{\mathbb{E}_{a''\sim\pi_{\theta^{-}}}\big[\exp\big(Q_{\theta^{-}}(s,a'',g)/\eta\big)\big]}$$

データの期待は遷移のサンプルで、行動の期待は $\pi_{\theta^{-}}$ からの $n_a=10$ 個のサンプルで推定し、$T_{\theta^{-}}$ の期待は解析的に計算する [p.3–4, Section 3.2]。原典は $w$ の分子を $Q_\theta$ と印刷するが、付録 (11) は $Q_{\theta^{-}}$ である（「食い違い」の項目 1）。

BC 項の役割は、Q が高くてもデータから遠い（過大評価の危険がある）行動へ方策が収束するのを防ぐことである [Levine et al. (2020)]。先行例 [Abdolmaleki et al. (2022)]、[Fujimoto & Gu (2021)] と同様の単純な正則化で、著者は学習の安定化に要るのはこの項だけだと報告する [p.4, Section 3.2]。$\alpha=1$ から $\alpha=0$ へ連続的に移れるので、高価な学習では $\alpha$ を大きめに置いて安定させ、微調整で下げることができる [p.4, Section 3.2]。

### 3.3 actor-critic 学習のスケールする構成（Scalable Architecture for Actor-Critic Learning）

$\pi_\theta$ と $Q_\theta$ を同じ Perceiver-IO [Jaegle et al. (2022)] 型の構成で実現する。多様な入力を受け、20 Hz の実機制御に足りる計算効率を狙う [p.4, Section 3.3]。全体像は Figure 2 [p.4, Figure 2] にあり、入力ごとの encoder、潜在への cross-attention、潜在上の self-attention の繰り返し、方策と Q の 2 本の cross-attention decoder を左から右へ描く。行動は観測の encoder や潜在を通らず、専用の符号化を経て Q の decoder のクエリとしてだけ入る。

#### Observation Encoding（観測の符号化）

固有受容 $s^{P}$、画像 $s^{V}$、画像と言語のタスク記述 $g^{V}$, $g^{L}$ を、入力ごとの encoder $\phi$ で埋め込みへ写して連結する [p.4, Section 3.3]。

$$e=\phi^{P}(s^{P})\oplus\phi^{V}(s^{V})\oplus\phi^{V}(g^{V})\oplus\phi^{L}(g^{L})\in\mathbb{R}^{L\times D_I}$$

固有受容には任意のスケールの入力を扱う multi-scale normalizer を提案する（付録 B.1、D.2.1）[p.4, Section 3.3]。

#### Transformer on Latent Space（潜在空間上の Transformer）

$e$ を直接 self-attention へ入れると、トークンが 1 ステップあたり数千に及ぶこの設定では二乗の計算とメモリが実時間制御の妨げになる。そこで学習可能な潜在ベクトル $z\in\mathbb{R}^{N_Z\times D_Z}$ をクエリに $e$ を読む cross-attention を先頭に置き、$z_0$ を得る。続く $M$ 個の self-attention を潜在の上だけで行い $z_M\in\mathbb{R}^{N_Z\times D_Z}$ を出す。$N_Z\ll L$ なので計算とメモリは $O(N_Z^2)$ に下がる [p.5, Section 3.3]。

#### Policy and Value Decoding（方策と価値の復号）

方策と Q は同じ $z_M$ を別々のクエリで読む。違いはクエリの作りで、方策は学習ベクトル、Q は行動の符号化である。この配置の狙いとして、行動が無視されにくいことと、$z_M$ の使い回しで多数の行動サンプルを安く評価できることが挙げられる [p.5, Section 3.3]。具体形は付録 B.4 にある。

## 4. 実験（Experiments）

実験は 3 組で、スケーリング則（4.1）、大規模学習での BC との比較（4.2）、実ロボットでの微調整（4.3）である [p.5, Section 4]。

データは共通で 3 つの出所を合わせる。Gato のデータ [Reed et al. (2022)]（Control Suite [Tunyasuvunakool et al. (2020)] の 32 タスクを解く RL エージェントの記録）、RoboCat のデータ [Bousmalis et al. (2023)]（RGB Stacking [Lee, A. X. et al. (2021)] でのピラミッド・タワー課題を 7-DoF Panda でシミュレーションした RL の記録と、歯車を差す Insertion の遠隔操作データ）、CHEF のデータ [Lampe et al. (2023)]（5-DoF Sawyer の積み上げのシミュレーションと実機の RL の記録）である。各フレームに短い言語のタスク記述を付け、RoboCat には目標画像も付ける [p.5, Section 4]。合計はスケーリング実験で 102 のシミュレーション課題と 30 の実機課題、364 万エピソード、約 2.45T トークンである [p.5, Section 4; p.17, Table 6]。

### 4.1 オフライン RL の目的関数のスケーリング解析（Scaling Analysis for Offline RL Objectives）

目的は、教師ありの系列予測で知られるスケーリング則 [Kaplan et al. (2020)] がオフライン RL にも及ぶかを確かめることで、[Hoffmann et al. (2022)] の方法を PAC に適用する [p.5, Section 4.1]。

手順:

1. モデル族: XXS、XS、S、M、L の 5 サイズ（32M〜988M パラメータ）を定める [p.5, Section 4.1; p.15, Table 4]。
2. 目的関数: 各サイズを $\alpha=1.0$（`BC+Q`、BC に Q 損失だけを足した基準）と $\alpha=0.75$（`PAC`）で学習する。Q 損失を足すことは BC の性能を下げなかったという [p.5, Section 4.1; p.5, footnote 2]。
3. 予算: バッチは長さ 5 の軌跡 512 本、データ混合の 1 エポックが約 2.7M ステップなので、各モデルを 3M 更新して単一エポックの範囲に収める [p.5, Section 4.1]。
4. 性能の代理: 本来は損失の包絡線で最適モデルを選ぶが、この設定では損失が性能の指標にならない（付録 E.3）。そこで 6 時点の評価から更新回数→正規化平均リターンのロジスティック曲線を当てはめ（付録 E.1）、モデルごとの return profile を作る [p.6, Section 4.1]。
5. 当てはめ: 5E+18〜5E+20 FLOPs の間に対数等間隔で取った 100 点について、profile の包絡線上で最良のモデルを選び、次のべき乗則を当てはめる [p.6, (4); p.6, Figure 3]。

$$N(C)=N_0\,C^{a},\qquad D(C)=D_0\,C^{b} \tag{4}$$

$C$ は計算量（FLOPs）、$N$ は性能最適なモデルのパラメータ数、$D$ は消費したトークン数である。

結果:

- パラメータの指数は $a(\text{PAC})=0.920$、$a(\text{BC+Q})=0.828$ である。著者は、計算予算が増えると RL の目的関数のほうがパラメータの追加から得るものが大きいように見えるとし、Q 関数に要る容量が大きいことを示唆すると留保つきで述べる [p.6, Section 4.1]。
- データの指数はほぼ同じで $b\approx0.266$ である [p.6, Section 4.1]。
- 単一エポック（2.45T トークン）に絞ると、`BC+Q` は 13.3 億、`PAC` は 9.54 億パラメータが示唆される。988M の L が `PAC` で最良だったことと合い、`BC+Q` はまだ規模を上げる余地があるとされる [p.6, Section 4.1]。Figure 3 の図上の印字では、`PAC` の 1E+21 FLOPs の最適点は 2.37E+09 パラメータ・3.18E+12 トークン、2.45E+12 トークンに対する最適な計算量は 3.72E+20 FLOPs である [p.6, Figure 3]。この 2.45T と 1 エポックの更新回数の対応には不整合の候補がある（「食い違い」の項目 12）。
- 等リターン線（Figure 4）: `BC+Q` の return profile と等リターン線を `PAC` と重ねると、RL の目的関数は等高線を「少ない計算で高い性能」の側へ動かし、この差は計算予算が増えるほど開く。著者はこれをオフライン RL のほうが計算量に対してよくスケールする証拠と読む [p.6, Figure 4; p.6–7, Section 4.1]。

本解説の照合: 表 16 の 1 更新あたりの FLOPs に 3M を掛けると、XXS は約 3.6E+19、L は約 4.8E+20 で、当てはめの範囲（上限 5E+20）と合う [p.22, Table 16]。パラメータは XXS→L で 30.9 倍だが前向き計算の FLOPs は 13.3 倍にとどまる [p.15, Table 4; p.22, Table 16]。また $D$ はデータセットの大きさを変えた結果ではなく、同じデータ混合を読み進めた消費量である（手順 2・3 からの整理。数え方の不整合の候補は「食い違い」の項目 12）。論文は「似た法則」を法則の形の話として述べ、指数の数値は言語モデルと比べていない。参考までに [Hoffmann et al. (2022), Section 3.4, Table 2] の指数は $a=0.46$〜$0.50$、$b=0.50$〜$0.54$ で、PAC の値は大きくパラメータ寄りである（数値の比較は本解説による）。

### 4.2 大規模なオフライン actor-critic 学習（Large-scale Offline Actor-Critic Learning）

強い BC の基準である Gato [Reed et al. (2022)] と RoboCat [Bousmalis et al. (2023)] に対して、RL の目的関数の優位が保たれるかを調べる。大規模モデルの事前学習は発散の恐れから BC で行うのが普通だが、オフライン RL の目的関数を初期段階から安全に使えれば、準最適なデータを最初から学習に使え、後の自己改善に使える Q 関数も持てる [p.7, Section 4.2]。

比べるモデルはいずれも L サイズ（約 10 億パラメータ、1.2B の Gato・RoboCat に合わせた）で、3M 更新する [p.7, Section 4.2]。

- `PAC`: $\alpha=0.75$ の主モデル。
- `BC+Q`: $\alpha=1$、$\beta>0$。Q 関数は学ぶが方策の最適化に使わない。
- `FilteredBC`: $\alpha=1$、$\beta=0$。Q 関数を持たず、成功エピソードだけで学ぶ。
- `α-PAC`: データごとに $\alpha$ を変えた最良の構成（値は付録 C.5）。

データ混合は、本文によればスケーリング実験に RoboCat の Tower と Pyramid の全データを足したもので（表との差は「食い違い」の項目 13）、2.84T トークンである [p.16–17, Section C.4; p.20, Table 10]。指標は、Control Suite では専門家の平均報酬に対する達成率、その他では成功率である（各タスク 100 回の評価）[p.7, Table 1]。

結果 [p.7, Table 1; p.7, Section 4.2]:

- Control Suite（Gato:Control）: Gato の公表値 63.6 に対し、`FilteredBC` 75.8、`BC+Q` 84.6、`PAC` 87.7、`α-PAC` 92.1。著者は自身の BC 基準が Gato を上回る理由に PAC の構成の改善の可能性を挙げる。
- RoboCat: データがほぼ専門家（成功率 75% 超）なので差は小さい。Tower は `PAC` 69.3 に対して `BC+Q` 71.3、Pyramid は `PAC` 63.5 に対して RoboCat の公表値 64.5 で、いずれも区間が重なる。差が大きいのは Insertion で、RoboCat の公表値 71.3 に対し `FilteredBC` と `BC+Q` が約 10 点上回り、`α-PAC` は 89.3 に達する。
- CHEF:sim（データの成功率 28%）: `FilteredBC` 17.0、`BC+Q` 11.0 に対し `PAC` 55.0、`α-PAC` 52.0。BC の基準がデータを集めた方策の平均に届くかどうかなのに対し、PAC は大きく上回る。

### 4.3 RL による微調整と自己改善（RL Fine-tuning and Self-improvement）

組み込みの critic を使い、5-DoF の積み上げ課題 [Lee, A. X. et al. (2021)] を実機の Rethink Sawyer で「習熟」（成功率 90% 超）させる [p.8, Section 4.3]。

- 事前学習: 4.2 のデータで 3M ステップ学習したモデルでは `α-PAC` が 69.8%、`BC+Q` は 7.1% である [p.8, Table 2]。$\alpha=1$ で 3M、続けて $\alpha=0$ で 3M 学習すると 61.9% になり、学習途中でも BC から RL へ安全に移れるとされる [p.8, Section 4.3]。
- 自己改善: [Lampe et al. (2023)] の手順で、評価試行を集めてデータに足し、初期データも残したまま、最初からではなく続けて約 250k ステップ学習することを繰り返す。追加 110k・75k・11k エピソードの 3 回で 84.7、89.8、93.2 と上がる [p.8, Section 4.3; p.8, Table 3]。
- Control Suite: タスクごとに 10,000 エピソードを足して 500k ステップの微調整で 92.1 から 94.3 になる [p.8, Section 4.3]。
- 実行速度: L サイズは手元の RTX 3090 で 20 Hz で動く [p.8, Section 4.3]。

## 5. 考察（Discussion）

著者のまとめは 4 つの観点に整理できる（整理は本解説による）。規模は最大 10 億パラメータ・132 タスクで、BC との差は準最適なデータで大きい。移行のしやすさでは、BC から RL へ徐々に移れ、事前学習後はモデルを変えずに微調整できる。スケーリングでは、計算量に対して BC よりよく伸び、モデルの大きさと学習の長さの指針が得られる。実用性では、多様な入力を同時に扱い、最大のモデルでも 20 Hz で実機を動かせる [p.8, Section 5]。

限界として、報酬の注釈が必要で高価なこと（普遍的な報酬関数 [Du et al. (2023)] や教師なしの報酬付与 [Chebotar et al. (2021)] で緩和しうる）と、多様な領域の間で強い転移が見られなかったこと（領域の重なりがより大きいデータ [Zitkovich et al. (2023)] なら改善を期待）を挙げる [p.8, Section 5]。今後として数十億パラメータへの拡大、事前学習済みの VLM との組み合わせ、言語モデルの事前学習の代替を挙げる [p.8, Section 5]。

## Broader Impact

研究目的で、配備は想定していないとする。安全上の懸念は Gato [Reed et al. (2022)] と RoboCat [Bousmalis et al. (2023)] と多くを共有し、自己改善も人が定めた報酬でオフラインに行い、実世界との相互作用中には学習しないと明記する。報酬で方策を形作ることは、人の好みとの価値整合の研究 [Russell (2019)]、[Christiano et al. (2017)] を適用しやすくするとも述べる [p.9, Broader Impact]。

## 付録 A. アルゴリズムの詳細（Method Details - Algorithm）

### A.1 方策の最適性の必要条件

[Abdolmaleki et al. (2018)] に従い、(1) の解を導く。状態ごとに最適であることを使って状態とタスクを省き、確率分布である条件のラグランジュ乗数 $\lambda$ を加える。$\pi$ で偏微分して 0 と置き、指数をとって $\lambda$ を正規化定数と見なすと、$\pi(a)\sim\exp(Q(a)/\eta)\,\bar\pi(a)$ が得られる [p.12, (5)–(8)]。

### A.2 PAC+Q の詳細

既定の実装（付録では PAC+Q と呼ぶ）は、損失の規模を揃えるために Q を離散分布の負対数尤度へ変換する。ビン幅 $\epsilon$ のビン $q\in\{q_{\min},q_{\min}+\epsilon,\dots,q_{\max}\}$ 上の分布を学び、$Q_\theta=\mathbb{E}_{p_\theta}[q]$ とする [p.12, Section A.2]。目標分布は次のとおりで、指示関数は変換後の Q の確率質量を最も近いビンへ割り当てる [p.12, (9)]。

$$T_{\theta}(q\mid s,a,g)=\mathbb{E}_{s'}\,\mathbb{E}_{a'\sim\pi_\theta(\cdot\mid s',g)}\,\mathbb{E}_{q'\sim p_\theta(\cdot\mid s',a',g)}\Big[\mathbf{1}_{[q-\epsilon/2,\;q+\epsilon/2]}(r+\gamma q')\Big]$$

Q 関数の損失はこの目標との KL (10) で、総損失 (11) は (3) と同じ形である。(11) は重みを指数化した $Q_{\theta^{-}}$ と正規化定数 $K_Q=\log\mathbb{E}_{a'\sim\pi_{\theta^{-}}}[\exp(Q_{\theta^{-}}/\eta)]$ で書く [p.12, (11)]。

### A.3 PAC+V の詳細

V 関数を使う別の実装で、基準の方策を行動方策そのもの（$\bar\pi=b$）にする。アドバンテージを遷移から $A\approx r+\gamma V_\theta(s')-V_\theta(s)$ で推定するので、Q 関数と target network が要らない [p.12–13, Section A.3]。V も行動で条件づけない離散分布で学ぶ。目標は行動方策の遷移でしか得られないので、重要度重み $\pi_\theta(a\mid s,g)/b(a\mid s,g)$ で補正する [p.13, (12)–(13)]。総損失は KL が 2 項で、$\bar\pi=b$ なので BC 項を別に置かない。$\eta\to\infty$ で BC に戻り、これで安定するとされる [p.13, (14)]。

$$L^{V}(\theta)=-\mathbb{E}_{\mathcal{D}}\Big[\exp\big(\tilde A/\eta\big)\log\pi_\theta(a\mid s,g)+\beta\,\frac{\pi_\theta(a\mid s,g)}{b(a\mid s,g)}\,\mathbb{E}_{v}\log p_\theta(v\mid s,g)\Big]+C$$

$\tilde A$ は勾配を止めた $r+\gamma V(s')-V(s)$ である。ここの指数は正規化定数を省いている [p.13, (14)]。$b$ は一定の集合から行動したと仮定して落とすか、最尤で推定する [p.13, Section A.3]。

## 付録 B. 構成の詳細（Method Details - Architecture）

### B.1 Multi-scale Normalizer

スカラー $x$ を $N_G$ 次元の $[-1,1]$ の表現へ写す [p.13, (15)]。

$$\phi^{\mathrm{ms}}(x)=\big[\tanh(\sigma_1 x),\dots,\tanh(\sigma_{N_G}x)\big]$$

$\sigma$ は対数スケールの固定ゲインで、$N_G=8$ なら $10^{-4}$ から $10^{3}$ である。どのゲインを読むかは後段の注意機構に任せる設計で、[Reed et al. (2022)] の $\mu$-law が範囲外の入力で飽和する問題への対策とされる [p.13, Section B.1]。

### B.2 Observation Encoding

入力は 4 種類で、タスク記述はエピソード単位である（後付けの再ラベルも可能 [Andrychowicz et al. (2017)]、[Riedmiller et al. (2018)]）[p.13, Section B.2]。領域で欠ける入力はパディングしてマスクする。入力は 1 タイムステップ分で、部分観測の課題では複数ステップへ容易に広げられると述べる [p.14, Section B.2]。

- 固有受容: $\phi^{\mathrm{ms}}$ のあと線形層で $D_I$ 次元へ写す。1 つの値が $N_G$ トークンになる [p.14, Section B.2]。
- 画像: 最終射影を除いた ResNet [He et al. (2016)] で $N_E$ 個の空間トークンにする。離散化はしない [p.14, Section B.2]。
- 言語: SentencePiece [Kudo & Richardson (2018)] のトークンを学習可能な表で埋め込む [p.14, Section B.2]。

トークン総数は $L=N^{P}N_G+N^{V}N_E+N^{V}_{g}N_E+N^{L}_{g}$ である [p.14, Section B.2]。

### B.3 Transformer on Latent Space

基本演算は $f(X_Q,X_K,X_V)=\mathrm{softmax}\big(X_QX_K^{\top}/\sqrt{D_Q}\big)X_V$ で、計算とメモリの主要部は $O(n_Qn_K)$（$n_Q$、$n_K$ はクエリとキーの本数）である。これにノルム、後段 MLP、スキップ接続を加えた block を ATTN と呼ぶ [p.14, Section B.3]。

- 入力を直接読む self-attention: $O(L^2)$ で、既定の $L=2{,}634$ では実時間の制約と文脈窓の拡張の妨げになる [p.14, Section B.3]。
- 入力 cross-attention: $z_0=\text{X-ATTN}(z,e)$ で $O(N_ZL)$。入力長に線形である [p.15, Section B.3]。
- 潜在 self-attention: $z_{m+1}=\text{S-ATTN}(z_m)$ で $O(N_Z^2)$。$N_Z=32$ なので計算とメモリは約 $(2{,}634/32)^2\approx6{,}775$ 分の 1 になる [p.15, Section B.3]。

Figure 2 の入力 cross-attention は 1 回で、繰り返すのは self-attention だけである [p.4, Figure 2]。

### B.4 Policy and Value Decoding

方策と Q それぞれに専用のクエリを作り、Perceiver-IO の復号に従って $z_M$ を cross-attention で読む [p.15, Section B.4]。

- 方策: 行動の次元数 $N^A$、次元ごとのビン数 $N_B$ として、$N^A$ 本の学習可能なクエリ $q_\pi\in\mathbb{R}^{N^A\times N_O}$ で $z_M$ を読み、線形射影で $N^A\times N_B$ のロジットにする。
- Q: 行動の情報は $z_M$ に無いので、クエリを行動から作る。$\phi^{A}=L^{A2}\circ L^{A1}\circ\phi^{\mathrm{ms}}_{N_G}$ で $\mathbb{R}^{N^A}\to[-1,1]^{N^A\times N_G}\to\mathbb{R}^{(N^A\times N_G)\times D_O}\to\mathbb{R}^{1\times D_O}$ と写す。観測の encoder が要素ごとにトークンを作るのと違い、行動全体を 1 本のクエリ $q_Q$ にまとめる（Q は 1 つの量だけを出せばよいため）。$\text{X-ATTN}(q_Q,z_M)$ の出力 $1\times N_O$ を $1\times N_Q$ のロジットへ写す。
- 利点: 行動が無視されにくく、$z_M$ を 1 回計算すれば複数の行動サンプルを問い合わせられる。

論文は $q_Q$ を「value query」と呼ぶが、これは状態価値を読むクエリではなく、特定の行動の Q を読むクエリである [p.15, Section B.4]。

## 付録 C. 実験の詳細（Experimental Details）

### C.1 スケーリング用の構成パラメータ

動かすのは潜在の大きさ $D_Z$、self-attention の数 $M$、残差 MLP の幅の倍率 $W$ の 3 つである（Table 4）。最小の XXS は 32M（$D_Z=768$、$M=4$、$W=1$）で RT-1 [Brohan et al. (2022)] や Q-Transformer [Chebotar et al. (2023)] と同程度、最大の L は 988M（$D_Z=2048$、$M=18$、$W=4$）で Gato・RoboCat の最大版に近い [p.15, Section C.1; p.15, Table 4]。

### C.2 固定の構成パラメータ

全スケール共通の値が Table 5 にある。本解説で使うものは行動 38 次元、$N_G=8$、$N_Z=32$、$N_B=101$、$D_I=256$ で、入力の数は次の C.3 で述べる [p.16, Table 5]。

### C.3 データの入力とトークン化

- 画像: 80×80 を 3 ブロックの ResNet [He et al. (2016)] で 10×10 の「画像トークン」にし、$D_I$ へ射影する。1 枚 100 トークン、8 枚で 800 トークンである [p.16, Section C.3]。
- 固有受容: 223 値×8 で 1,784 トークン。少ない領域はパディングしてマスクする [p.16, Section C.3]。
- 言語: データセット名から作った命令（例 `humanoid.walk`）を最大 50 トークンで与える [p.16, Section C.3]。

合計 2,634 トークン（画像 800、固有受容 1,784、言語 50）である [p.16, Section C.3]。

### C.4 データ混合

スケーリング実験は 51 データセットを 6 群に分け、群を重み $\lambda$ で引いてから群の中で一様に引く。消費し切らないようデータセットは決定的に周回する [p.16, Section C.4; p.17, Table 6; p.19, Table 9]。領域別の成功率は CHEF:real 12%、CHEF:sim 28%、RC:Insertion 97% など幅があり、平均 42% である [p.17, Table 6]。事前学習実験は、本文では RoboCat の Tower と Pyramid の全データを足すとされるが、表で増えるのは Pyramid だけである（「食い違い」の項目 13）[p.16–17, Section C.4; p.20, Tables 10–11]。

本解説の照合: 1 エポック約 2.7M ステップは、Table 9 の CHEF の実機データ 1 件（28,224,012 軌跡、抽出確率 1/48）を一巡する更新回数 $28{,}224{,}012\times48/512\approx2.65$M に近く、事前学習の 4.7e6 も同じ計算（確率 1/84）で約 4.63M になる。論文は 1 エポックの定義を書いていないが、最も遅く一巡するデータセットで測っていると読める。その場合、軌跡の少ないデータセットは 1 エポックの間に何度も読まれる（例 `swimmer.swimmer6` の 98,000 軌跡は約 220 周）[p.19, Table 9; p.20, Table 11]。ただしこの読みは 2.45T トークンと両立しない（「食い違い」の項目 12）。

### C.5 最適化のハイパーパラメータ

AdamW [Loshchilov & Hutter (2017)] を使い、学習率は線形の warmup のあと 1 エポック分のコサインで減衰する。[Hoffmann et al. (2022)] に従い 1 エポックで 1 桁下げ、減衰ステップはスケーリング実験で 2.7e6、事前学習で 4.7e6 である [p.17, Section C.5]。ピークの学習率はモデルが大きいほど下げ、XXS・XS で 1e-4、S で 5e-5、M・L で 3e-5 である [p.20, Table 12]。

損失の倍率は TD 側の $\beta=38$ を固定して $\alpha$ を動かす（`BC+Q` と `FilteredBC` で 1.0、`PAC` で 0.75）。`α-PAC` は Control Suite と RoboCat で $\beta=19$、$\alpha=0.75$、CHEF で $\beta=1{,}900$、$\alpha=0$ である。`PAC+V` はすべて $\alpha=0$、$\beta=38$、温度 1e-4 である [p.17, Section C.5]。

## 付録 D. 感度とアブレーション実験（Sensitivity and Ablation Experiments）

### D.1 ハイパーパラメータへの敏感さ

D.1.1 BC の倍率 $\alpha$: 専門家データが多いほど BC の比重を上げるのがよく、少ないほど RL から得るものが大きい。XS モデルの Control Suite では $\alpha=0$ で 36.8 と大きく下がり、0.8 付近が最良で 85.1、純 BC の $\alpha=1$ は 82.4 である。データの質が低い CHEF では $\alpha=0$ が最良とされる [p.17, Section D.1.1; p.17, Table 7]。

D.1.2 方策の損失の倍率 $\beta$: 方策と critic はネットワークのパラメータを共有しており、2 つの損失の均衡を取るパラメータが要るという [p.17, Section D.1.2]。XS モデルの CHEF:sim で $\alpha=0$ として $\beta$ を 1.0 から 0.005 まで下げると 22.0 から 83.0 まで上がり、方策の損失の寄与を下げるほどよい。一般に 0.005 がよいとする [p.17–18, Section D.1.2; p.18, Table 8]。この節の $\beta$ は (3) の $\beta$ と意味が違うように読める（「食い違い」の項目 2）。

### D.2 構成のアブレーション

計算の都合から、データの一部と小さいモデルで行う [p.18, Section D.2]。

D.2.1 入力の符号化: multi-scale normalizer を、線形 encoder と Gato 型の $\mu$-law のトークン encoder と比べる。Control Suite（XS）は 79.6、82.7 に対して 86.7 で、線形 encoder は固有受容の値の範囲に対応できないのが原因とみる。RoboCat（S、行動ビンは 1024）では $\mu$-law が最も低く、Pyramid で 36.1 に対し multi-scale は 65.8 である [p.18, Section D.2.1; p.21, Table 13]。

D.2.2 行動の cross-attention: 行動を入力として与え、attention マスクで方策の出力から隠す構成と比べる。行動を出力クエリにする利点として、方策と Q で共通の encoder を使えること、潜在のキャッシュで速いこと、行動から Q までの経路が短く無視されにくいことを挙げる。Control Suite（XS）では出力クエリの 86.7 に対し、入力にすると 76.8 に下がる。RoboCat では実施していない [p.18, Section D.2.2; p.21, Table 13]。

D.2.3 タスクの条件づけ: M サイズの事前学習モデルは言語だけでも目標画像と言語の両方の場合に近い成功率を保つが、目標画像だけでは大きく下がる（Tower で 0.1、Pyramid で 11.4）。入力を欠けさせる微調整で 200k ステップ後に 44.7、47.7 まで回復する [p.18, 21, Section D.2.3; p.21, Table 14]。データの偏りが原因という仮説は、両方の記述が常にある状態で学習し直しても言語に偏ることから否定され、言語のほうが目標画像より識別的なのだろうと著者は推測する。事前学習から欠けさせれば、どの条件でも働く [p.21, Section D.2.3; p.21, Table 15]。

## 付録 E. スケーリングの詳細（Scaling Details）

### E.1 Return Profile の当てはめ

各 Run で 3M 更新を等間隔に分けた 6 時点を評価する。各時点で 102 のシミュレーション課題をタスクごとに 100 回試行し、平均エピソードリターンを求めて、次のロジスティック関数を当てはめる [p.22, (16)]。

$$R(n)=\frac{c_1}{1+\exp\big(-c_2(n-n_0)\big)}+c_3$$

$n$ は更新回数、$c_1,c_2,c_3,n_0$ は当てはめのパラメータである（原典の $a,k,b,n_0$）。

### E.2 FLOPs の数え方

[Hoffmann et al. (2022)] の方式に従い、計算の大半は attention にあるとして数える。cross-attention は入力長×出力長、self-attention は入力長の二乗に比例する。FLOPs を数える部位は Figure 6 にある [p.22, Figure 6; p.22, Section E.2]。乗算と加算の係数は 2 とし、encoder と ResNet も加える。1 回の前向き計算と更新は次の形である [p.23, Section E.2]。

$$F_{\text{fwd}}=E_P+E_V+E_L+E_A+X_{\text{in}}+M\,S+X_\pi+X_Q$$

$$F_{\text{bwd}}=\Big(2+\frac{1}{f}\Big)F_{\text{fwd}},\qquad F_{\text{update}}=B\,\big(F_{\text{fwd}}+F_{\text{bwd}}\big)$$

$E$ は encoder、$X$ は cross-attention、$S$ は self-attention 1 個の FLOPs、$f=100$ は target の更新間隔、$B=512$ はバッチサイズである。原典は右辺も BWD と印刷するが、FWD と読むと Table 16 の値に合う（「食い違い」の項目 4）。1 更新あたり XXS で 1.206E+13、L で 1.602E+14 である [p.22, Table 16]。

### E.3 損失は性能の指標にならない

L サイズの最終の損失は `BC+Q` のほうが `PAC` より約 0.1 低いが、シミュレーションの 73 タスクで比べると性能は逆で、とくに CHEF:sim は 12.0 に対して 40.0 である。[Hilton et al. (2023)] と同様、RL では損失が性能の指標にならないので、損失の包絡線の代わりに return profile を使う [p.23–24, Section E.3; p.23, Figure 7]。

### E.4 PAC+V のスケーリング結果

`PAC+V` の結論は Q 版と同じ傾向で、2.45T トークンに対して示唆される大きさは 8.52 億とやや小さく、指数は $a=0.970$ と大きい。データの指数は $b\approx0.266$ で同じである [p.24, Section E.4; p.24, Figure 9]。

## 付録 F. 追加実験（Additional Experiments）

### F.1 PAC+V の大規模学習

概ね Q 版と同じ結果だが例外が 2 つある。RC:Insertion は `PAC+V` が 89.0 まで伸びる（遠隔操作データのぎこちなさを 1 回の方策改善で避けられたと著者は読む）。CHEF:sim は 42.0 で `PAC` の 55.0 に劣り、データの大半が準最適なら価値関数だけでは遷移の選別に足りないのかもしれないとする [p.24, Section F.1; p.25, Table 17]。

### F.2 RL による微調整と自己改善の補足

RGB Stacking の 5 つの物体セットごとの推移が Table 18 にあり、とくに Set 2（力で物体を倒す）が 53.5 から 91.5 へ伸びる [p.25, Section F.2; p.25, Table 18]。微調整の対象は CHEF:real だけなので他の領域が落ちないかを確かめ、シミュレーションのタスクは影響を受けないとする（Table 19、ただし「食い違い」の項目 7）[p.25, Section F.2]。

`BC+Q` から始める経路も示す。実機では 3.6% と低いが、同じ事前学習データで $\alpha=0$ にして 200k 更新すると 38.2% へ上がる。`α-PAC` の 69.8% には届かないが、最初に非専門家データが無い場合の出発点になるとする [p.25, Section F.2]。

## 論文が述べている限界

- オフライン RL には報酬の注釈が要り、高価である [p.8, Section 5]。
- 多様な領域の間で強い転移は見られなかった [p.8, Section 5]。
- 損失は性能の指標にならず、スケーリングの当てはめは return profile という代理に頼る [p.6, Section 4.1; p.24, Section E.3]。
- アブレーションはデータの一部と小さいモデルでの比較である [p.18, Section D.2]。
- データの大半が準最適な場合、V 版は Q 版に劣る [p.24, Section F.1]。

## 論文内の記載の食い違い（本解説による照合）

1. (3) の重み $w$ は分子が $Q_\theta$、分母が $Q_{\theta'}$ で印刷されているが、付録 (11) では指数の Q も $Q_{\theta'}$ である [p.3, (3); p.12, (11)]。
2. $\beta$ の意味が節で違う。(3)、(11) と C.5 では TD 損失の倍率で値は 19、38、1,900 である [p.3, (3); p.17, Section C.5]。D.1.2 と Table 8 は「方策の損失の倍率」と呼び、1.0〜0.005 の範囲で 0.005 を勧める [p.17–18, Section D.1.2; p.18, Table 8]。
3. `FilteredBC` は Section 4.2 で $\beta=0$ だが、C.5 は $\beta=38$ 固定の文の中で `FilteredBC` の $\alpha$ を述べる [p.7, Section 4.2; p.17, Section C.5]。
4. FLOPs の式は `FLOPS_BWD` の右辺も `FLOPS_BWD`、`FLOPS_UPDATE` も `FLOPS_BWD` の和と印刷されている。右辺を FWD と読むと Table 16 に合う（XXS で BWD/FWD ≈ 2.01、$512\times(7.826\text{E}{+}9+1.573\text{E}{+}10)\approx1.206\text{E}{+}13$）[p.23, Section E.2; p.22, Table 16]。
5. データの指数は本文で $b\approx0.266$ だが、図の印字は `PAC` と `BC+Q` が 0.265、`PAC+V` が 0.267 である [p.6, Section 4.1; p.6, Figure 3; p.24, Figure 9]。差は丸めの範囲である。
6. 実機の `BC+Q` は Table 2 で 7.1%、F.2 の本文で 3.6% である。Figure 1 の棒は 0.07 と読める（画像からの読み取り）[p.8, Table 2; p.25, Section F.2; p.2, Figure 1]。評価の条件の違いかは書かれていない。
7. 微調整後に他の領域は落ちないと本文は述べるが、Table 19 の RC:Insertion は 89.3 [85.0, 92.1] から 80.3 [75.5, 84.4] へ、区間が重ならずに下がっている。80.3 [75.5, 84.4] は Table 1 の `PAC` と同じ値である [p.8, Section 4.3; p.25, Table 19; p.7, Table 1]。
8. 区間の中心が値と合わない箇所がある。Table 19 の Control Suite の 91.3 [89.6, 96.9] は Figure 7 の `PAC` の 93.2 と同じ区間で、Table 1 の `FilteredBC` の 75.8 [62.5, 78.6] は標準誤差に基づくとされるのに非対称である [p.25, Table 19; p.23, Figure 7; p.7, Table 1]。
9. タスク数は Table 6 で CHEF:sim、CHEF:real ともに 30、Table 1 と Table 2 ではそれぞれ 1 と 5 である。評価を積み上げに絞った結果と読めるが、対応は明記されていない [p.17, Table 6; p.7, Table 1; p.8, Table 2]。
10. CHEF:sim について本文は BC の基準がデータ収集方策の平均にかろうじて並ぶと述べるが、Table 1 の値は `FilteredBC` 17.0、`BC+Q` 11.0 で、データの 28 を下回る [p.7, Section 4.2; p.7, Table 1]。
11. 記号と語の小さな乱れ: 原典の $\tau$ はタスク記述と温度の 2 つの意味で使われる [p.3, Section 3.1; p.17, Section C.5]。$N$ はサンプル数とトークン数の 2 つの意味で使われる [p.4, Section 3.2; p.14, Section B.2]。Table 11 の見出しは事前学習の混合なのに「スケーリング実験」と書く [p.20, Table 11]。D.2.3 は言語の記述を $\tau^{V}$ と書く箇所がある [p.18, Section D.2.3]。
12. 不整合の候補（本解説の計算）: 1 エポックの更新回数と 2.45T トークンの対応が合わない。Table 9 の 51 データセットを抽出確率 $P_{\text{eff}}$ で重み付けすると、期待値は 1 更新（512 軌跡）あたり約 224 万トークンで、2.7M 更新では約 6.0T になる。軌跡で見ても $2.7\text{M}\times512\approx13.8$ 億本に対し、Table 9 の合計は約 5.07 億本である。一方、Figure 3 の 2.45E+12 トークンの点は 3.72E+20 FLOPs で、L の 1 更新 1.602E+14 FLOPs で割ると約 232 万更新、約 106 万トークン/更新となり、約 2 倍の差がある。2.45T は Table 9 の #(Tok) の単純な合計に一致する。原因は論文に書かれていない [p.5, Section 4.1; p.6, Figure 3; p.19, Table 9; p.22, Table 16]。
13. 不整合の候補: 事前学習の混合について、本文は RoboCat の Tower と Pyramid の全データを足すと述べるが、Table 6 と Table 10 で増えるのは Pyramid だけ（194k→601k エピソード、195B→604B トークン）で、Tower は減る（121k→100k エピソード、122B→100B トークン）[p.16, Section C.4; p.17, Table 6; p.20, Table 10]。

## 論文から確かめられなかった点

本文と付録の全ページを読んだうえで、書かれていなかったものである。

- $\eta$ の値（PAC）、$N_O$、$D_O$、$N_Q$（Q のビン数）、$q_{\min}$、$q_{\max}$、attention の head 数は記載が無い [p.16, Table 5 ほか]。
- デコーダの cross-attention を方策と Q で別の重みにしているか。Figure 2 は別のブロックとして描くが、文章は共有部分を encoder と共通のパラメータと述べるだけである [p.4, Figure 2; p.17, Section D.1.2; p.18, Section D.2.2]。
- 1 更新で Body を何回通すか、$n_a=10$ のサンプルが FLOPs にどう入るか。式の $X_Q$ は 1 項だけである [p.23, Section E.2]。
- 長さ 5 の軌跡の使い方。損失の式は 1 ステップの遷移で、$F_{\text{update}}$ にも長さ 5 の係数は無い [p.3, (3); p.5, Section 4.1; p.23, Section E.2]。
- seed の数。本文に「seed」の語は無く（テキスト層の検索）、表の区間は試行数に基づくもので Run の間のばらつきではない [p.7, Table 1]。
- オンライン学習の実験。Figure 1 の説明に online の語があるが、本文の実験は集めてからオフラインで学ぶ形だけである [p.2, Figure 1; p.8, Section 4.3; p.9, Broader Impact]。離散行動の実験は無い。
- PMLR 版の本文との差。実装コードは PMLR ページから見つからなかった。

## キューの論点への答え

reports/INDEX.md のキュー（優先 12。コミット 5f1ea9fb の L40）の問いに答える。

1. Perceiver 型の構成と、行動を Q に cross-attention で後から入れる設計の具体形
   - 入力側: トークン $e\in\mathbb{R}^{2634\times256}$ を、学習可能な潜在 $z\in\mathbb{R}^{32\times D_Z}$ をクエリにした cross-attention で 1 回だけ読み、$z_0$ にする。以後の $M$ 個の self-attention は潜在だけで行い、$z_M$ を出す。$D_Z$ と $M$ はモデル規模で 768×4 から 2048×18 まで動かす [p.15, Section B.3; p.15, Table 4; p.16, Table 5]。
   - 方策側: 行動の次元ごとに学習ベクトルを 1 本置く（38 本）。$z_M$ を読み、線形射影で 38×101 のロジットにする [p.15, Section B.4; p.16, Table 5]。
   - Q 側: 行動ベクトルを multi-scale normalizer で 38×8 にし、線形層 2 枚で 1 本のクエリ $q_Q\in\mathbb{R}^{1\times D_O}$ に潰す。$z_M$ を読み、線形射影で Q の分布のロジットにする [p.15, Section B.4]。
   - 検証: 行動を入力にする場合との比較は 1 件だけで、Control Suite（XS）の 86.7 対 76.8 である [p.21, Table 13]。出力クエリと、行動ごとの出力を持つ通常の Head との比較は論文に無い。

2. 状態の表現を使い回して複数の行動候補を評価する仕組みと計算量
   - 仕組み: $z_M$ は行動に依らないので 1 回計算してキャッシュし、$q_Q$ だけを行動サンプルごとに作って問い合わせる [p.5, Section 3.3; p.15, Section B.4]。サンプルは $n_a=10$ で、(3) の重み $w$ の計算に使う [p.4, Section 3.2]。
   - 論文の計算量: 演算は $O(n_Qn_K)$、入力 cross-attention は $O(N_ZL)$、潜在 self-attention は 1 個あたり $O(N_Z^2)$、入力への直接の self-attention に比べて約 6,775 分の 1 である [p.14–15, Section B.3]。
   - 本解説の計算（$D$ の依存を除いたクエリとキーの組の数）: Body は $32\times2{,}634+18\times32^2=102{,}720$（L サイズ）で、Q の decoder は 1 行動あたり $1\times32=32$、10 サンプルで 320 である。行動を入力に入れて Body をサンプルごとに通すと、Body の分が 10 倍になる。

3. actor と critic で共有している部分と、損失（正則化、BC との混ぜ方）
   - 共有: 方策と critic はネットワークのパラメータを共有する [p.17, Section D.1.2]。encoder、潜在、self-attention は共通で $z_M$ までが 1 つの Body であり、出力の cross-attention と射影で分かれる [p.4, Figure 2; p.18, Section D.2.2]。デコーダの重みを共有するかは明記が無い。target のパラメータも 1 組で、100 更新ごとに写し、target 方策と target Q を同時に与える [p.3, Section 3.2; p.23, Section E.2]。
   - 損失: (3) の 3 項の KL で、$\pi^{+}$ への当てはめ（重み $w$ の指数付き）、$b$ への当てはめ（BC）、目標分布への当てはめ（分布 TD）を $(1-\alpha)$、$\alpha$、$\beta$ で足す。RL 側の正則化は $\pi_{\theta^{-}}$ への KL（温度 $\eta$）で、データへの引き寄せは BC 項が受け持つ [p.3–4, Section 3.2]。
   - 配分の決め方: データの質に応じて変える。既定は $\alpha=0.75$ で、CHEF のように質が低ければ $\alpha=0$、$\beta=1{,}900$ である。Control Suite で $\alpha=0$ にすると 36.8 まで下がる [p.17, Section C.5; p.17, Table 7]。
   - V 版: $\bar\pi=b$ にして BC 項を省き、Q と target network を持たない [p.12–13, Section A.3]。

4. スケーリング則の示し方（モデル規模とデータ量）
   - 手順は 4.1 節に書いたとおりである。5 サイズ・2 つの目的関数で 3M 更新、6 時点の評価からロジスティックの return profile を作り、FLOPs に対する包絡線上の 100 点へ (4) を当てはめる [p.5–6, Section 4.1; p.22, Section E.1]。
   - モデル規模の軸は $D_Z$、$M$、$W$ の 3 つを同時に動かす [p.15, Table 4]。データ量の軸は固定のデータ混合の消費トークン数で、データセットの大きさを変えた実験ではない（本解説の整理）。
   - 結果: $a=0.920$（PAC）、0.828（BC+Q）、$b\approx0.266$。2.45T トークンでは 954M と 1.33B が示唆され、L（988M）が最良と合う [p.6, Section 4.1]。2.45T と更新回数の対応には不整合の候補がある（「食い違い」の項目 12）。
   - 注意: 損失ではなく性能を目的変数にした点が言語モデルの方法との違いである [p.6, Section 4.1; p.23–24, Section E.3]。サイズが 5 点なので、パラメータの当てはめは階段状の点に対して行われる（図の読み取り）[p.6, Figure 3]。seed の数は書かれていない。

5. 離散行動やオンライン学習に持ち込むときに要る変更

   論文はどちらも扱っていないので、以下はすべて本解説の推論である。
   - 離散行動: PAC の方策は「各次元の離散化」で、排他的な候補の選択ではない。Q のクエリの multi-scale normalizer は連続値のためのもので、行動 ID なら学習可能な埋め込み（論文の言語 encoder [p.14, Section B.2] の形）で足りる。候補を全部列挙できるので、$n_a$ 個のサンプルはクエリ全部の一括評価に替わり、$O(n_Qn_K)$ の $n_Q$ が行動数になる。
   - actor の扱い: 最大値を列挙で取れるなら方策のネットワークは要らず、(3) の第 1・2 項が消えて TD の項だけが残る。方策を残すなら (2) はサンプルなしで厳密に計算できる。目標分布 (9) の行動の期待は max か Double DQN の形に替える。
   - 分布の型: 離散分布から QR や IQN に替えると、(3) の「対数尤度で規模を揃える」前提と $\beta$ の値は引き継げない。
   - オンライン: BC 項の役割はデータ外の行動の過大評価の抑制 [p.4, Section 3.2] なので、探索で修正が効くなら不要になりうる。ただし大規模モデルの安定性を示す主結果は $\alpha=0.75$ で、$\alpha=0$ の例はオフラインの CHEF などに限られ、オンラインでの保証は論文に無い。探索の仕組み、更新とデータ収集の比、リプレイの入れ替わりも論文の前提（固定データの単一エポック）から変わるので、(4) の指数はそのまま使えない。

6. DropMerge の行動クエリ案（59 行動 + 価値クエリが盤面トークンを読む）との対応

   現行の構成は、盤面 180 トークンを 128 次元の TransformerEncoder 4 層に通したあと `Flatten > Linear_2048 > SiLU` で 1 本にし、IQN の τ の埋め込みと要素積で結合する。そこから Body の枝 `value_stream` と `adv_stream`（各 `HeadFC1024 > SiLU`）を経て、Head の `IQNDuelingHead`（Linear→1 と Linear→action_dim）へ送る（apps/runner/config/DropMerge.txt の `net.branch.@SuikaImpala2_ViT128`、`net.block.[DMImpalaTrans128]`、`net.@iqn`、`net.body.output`。core/anet-core/src/dqn_based_heads.cpp の `IQNDuelingHead`）。行動クエリ案は `[B,180,128]` を 59 本のクエリで読む 1 層の cross-attention である [action_query_papers (2026/08), DropMerge への対応]。「+1」の価値クエリはこのサーベイの案には無く、キューの論点メモ（コミット 5f1ea9fb の reports/INDEX.md L40。今の INDEX からは行を消した）が出所である。

   | 観点 | PAC | DropMerge の案 | 読み方（本解説の推論） |
   |---|---|---|---|
   | 読まれる側 | 潜在 32 本の $z_M$ | self-attention 後の盤面 180 トークン | PAC は圧縮後を読む。位置を保ったまま読む構成は検証していない |
   | クエリの本数 | Q は 1 回に 1 本（サンプルごと）、方策は 38 本 | 59 本を一括（論点メモの形では価値クエリを足して 60 本） | 一括評価は PAC の Q より方策のクエリの形に近い |
   | クエリの作り | Q は行動ベクトルの符号化、方策は学習ベクトル | 行動 ID の学習ベクトルと、落下位置の符号化 | 排他的な候補ごとに固定の意味を持たせる点は AQT と Query2Label に近い |
   | 価値のクエリ | Q 版には無い。論文の「value query」は行動の Q を読むクエリ。PAC+V の V の読み方は書かれていない | 論点メモの形で状態価値を読む 1 本 | 同じ語で別物。Dueling の形は AQT にあるが、AQT の V は Flatten で読む [action_query_papers (2026/08), 1. Action Q-Transformer]。価値クエリの直接の先例は AQT にも PAC にも見当たらず、あるのは Perceiver IO の出力ごとのクエリという一般論からの設計推論だけである |
   | 出力 | Q の分布のロジット（$N_Q$ ビン） | QR の 51 本か、τ と結合した IQN | τ の扱いは PAC に無い |
   | Head の前の Flatten | 無い（cross-attention + 線形） | 無くす | 構成の方向は一致する |

   本解説の計算（$D$ の依存を除いた組の数）: 現行の 180 トークンの self-attention は 1 層 32,400、4 層で 129,600 である。論点メモの形の 60 本のクエリの cross-attention は $60\times180=10{,}800$ で、self-attention 1 層の 3 分の 1 である。合計は 140,400 になる。潜在 32 本を入れる別案では、self-attention の側が $32\times180+4\times32^2=9{,}856$（現行の 129,600 に対応）、デコーダが $60\times32=1{,}920$（上の 10,800 に対応）で、合計 11,776 になる。Flatten 直後の `Linear_2048` は入力が 180×128 なので約 4,700 万パラメータで、案の主な節約はここにある。`vector_feature` は `SpatialEmbedder` で盤面の高さと幅へ広げてチャネル方向に連結され、CNN の入口に入るのでトークン数を変えない（core/anet-core/src/nn_modules.cpp の `Forward`、docs/design/130_neural_networks.jp.md の `bind` の説明）。

## このプロジェクトの検討に効く点

ここはすべて本解説の推論で、論文の主張ではない。

- 根拠の範囲: PAC が示すのは連続制御で「行動を入力に入れるより出力クエリにするほうがよい」こと（XS、Control Suite の 1 比較）である。DropMerge の現行の構成はもともと評価する行動を入力に入れず、全行動を一度に出すので、この結果は「現行の Head より行動クエリがよい」の根拠にならない（直前の命令は `E1.obs_include_prev_action` で観測に入るが、評価する行動とは別である。skip 側の `skip.network_mode=action_in` も別で、`run.@skip` は既定で選ばれていない）。PRD093 の先送りの段階 3 [093 attention エントロピー診断 PRD (2026/10), §7] を試す判断は、複数 seed の比較で決める必要がある。効果の見込みは PAC ではなく、Flatten を無くして列ごとに読むことの是非として評価する。
- 使える設計の原理: 重い共通部を 1 回だけ通し、多数回評価する条件をできるだけ後ろで与える。PAC ではこの条件が行動で、DropMerge では τ（K/N/M=32）に当たる。行動特徴 `[B,59,D]` を 1 回作ってから τ を掛ける U×T の案 [action_query_papers (2026/08), DropMerge への対応] は同じ向きで、現行は τ との結合より後ろ（Body の枝 `value_stream`、`adv_stream` と Head）を K 本分通している。
- 潜在の圧縮は後回しでよい: PAC の動機は 2,634 トークンの二乗回避である。180 トークンでも計算上は 1 層あたり約 32 倍の節約になるが、圧縮後に 58 列の位置を読み分けられるかの証拠は無い。最初の比較は潜在を入れず盤面トークンを直接読む構成にして、変える要素を Head の形だけに絞るのがよい。
- BC 混合は今は不要: DropMerge に方策ネットワークは無く、オンラインなので $\alpha$ の話は直接使えない。データの質で最適な $\alpha$ が大きく変わる事実（0 で 36.8、0.8 で 85.1）は、将来 Run のデータや人のデータから事前学習するときに参考になる。
- スケーリング則の数値は持ち込まない: 固定データの単一エポックの結果で、$a$、$b$ はオンラインに当てはまらない。持ち込めるのは手順で、損失ではなく評価の性能を FLOPs に対して比べること、モデルが大きいほど学習率のピークを下げていること（1e-4 から 3e-5）である [p.20, Table 12]。
- 語の注意: 論文の「value query」は DropMerge の案の「価値クエリ」と別物である。また論文の $\tau$ はタスク記述で、IQN の τ と無関係である。attention の計器を行動クエリへ広げるときは feature 閉包の外の block になり、別の鍵が要るという既存の整理 [093 attention エントロピー診断 PRD (2026/10), §7] もそのまま当てはまる。

## 出典リスト

他文献の書誌は、断りが無ければ対象論文の参考文献欄（p.9–11）による。URL は欄に印字されたものか、印字された arXiv 番号から作ったものだけを書き、それ以外は書かない。著者の省略（et al.）も欄の表記のままである。

[Springenberg et al., 2024/02] Jost Tobias Springenberg, Abbas Abdolmaleki, Jingwei Zhang, Oliver Groth, Michael Bloesch, Thomas Lampe, Philemon Brakel, Sarah Bechtle, Steven Kapturowski, Roland Hafner, Nicolas Heess, Martin Riedmiller. "Offline Actor-Critic Reinforcement Learning Scales to Large Models." arXiv:2402.05546v1（ICML 2024, PMLR vol. 235, pp. 46323–46350 に収録）. https://arxiv.org/abs/2402.05546 （PMLR 版: https://proceedings.mlr.press/v235/springenberg24a.html ）

[Abdolmaleki et al., 2018] Abdolmaleki, A., Springenberg, J. T., Tassa, Y., Munos, R., Heess, N., Riedmiller, M. "Maximum a posteriori policy optimisation." International Conference on Learning Representations, 2018.

[Abdolmaleki et al., 2022] Abdolmaleki, A., Huang, S., Vezzani, G., Shahriari, B., Springenberg, J. T., Mishra, S., et al. "On multi-objective policy optimization as a tool for reinforcement learning: Case studies in offline rl and finetuning." 2022. https://openreview.net/forum?id=bilHNPhT6-

[Alayrac et al., 2022] Alayrac, J.-B., Donahue, J., Luc, P., et al. "Flamingo: a visual language model for few-shot learning." Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

[Andrychowicz et al., 2017] Andrychowicz, M., Wolski, F., Ray, A., et al. "Hindsight experience replay." Advances in Neural Information Processing Systems, 30, 2017.

[Bellemare et al., 2017] Bellemare, M. G., Dabney, W., Munos, R. "A distributional perspective on reinforcement learning." International Conference on Machine Learning, pp. 449–458. PMLR, 2017.

[Bousmalis et al., 2023] Bousmalis, K., Vezzani, G., Rao, D., et al. "Robocat: A self-improving foundation agent for robotic manipulation." arXiv:2306.11706, 2023. https://arxiv.org/abs/2306.11706

[Brohan et al., 2022] Brohan, A., Brown, N., Carbajal, J., et al. "Rt-1: Robotics transformer for real-world control at scale." arXiv:2212.06817, 2022. https://arxiv.org/abs/2212.06817

[Chebotar et al., 2021] Chebotar, Y., Hausman, K., Lu, Y., et al. "Actionable models: Unsupervised offline reinforcement learning of robotic skills." arXiv:2104.07749, 2021. https://arxiv.org/abs/2104.07749

[Chebotar et al., 2023] Chebotar, Y., Vuong, Q., Hausman, K., et al. "Q-transformer: Scalable offline reinforcement learning via autoregressive q-functions." 7th Annual Conference on Robot Learning, 2023.

[Chen et al., 2021] Chen, L., Lu, K., Rajeswaran, A., et al. "Decision transformer: Reinforcement learning via sequence modeling." Advances in Neural Information Processing Systems, 34:15084–15097, 2021.

[Christiano et al., 2017] Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D. "Deep reinforcement learning from human preferences." Advances in Neural Information Processing Systems, 30, 2017. https://proceedings.neurips.cc/paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf

[Du et al., 2023] Du, Y., Konyushkova, K., Denil, M., et al. "Vision-language models as success detectors." arXiv:2303.07280, 2023. https://arxiv.org/abs/2303.07280

[Fujimoto & Gu, 2021] Fujimoto, S., Gu, S. S. "A minimalist approach to offline reinforcement learning." Advances in Neural Information Processing Systems, 34:20132–20145, 2021.

[Fujimoto et al., 2019] Fujimoto, S., Meger, D., Precup, D. "Off-policy deep reinforcement learning without exploration." International Conference on Machine Learning, pp. 2052–2062. PMLR, 2019.

[Hansen et al., 2023] Hansen, N., Su, H., Wang, X. "Td-mpc2: Scalable, robust world models for continuous control." arXiv:2310.16828, 2023. https://arxiv.org/abs/2310.16828

[He et al., 2016] He, K., Zhang, X., Ren, S., Sun, J. "Deep residual learning for image recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[Henighan et al., 2020] Henighan, T., Kaplan, J., Katz, M., et al. "Scaling laws for autoregressive generative modeling." arXiv:2010.14701, 2020. https://arxiv.org/abs/2010.14701

[Hilton et al., 2023] Hilton, J., Tang, J., Schulman, J. "Scaling laws for single-agent reinforcement learning." arXiv:2301.13442, 2023. https://arxiv.org/abs/2301.13442

[Hoffmann et al., 2022] Hoffmann, J., Borgeaud, S., Mensch, A., et al. "Training compute-optimal large language models." arXiv:2203.15556, 2022. https://arxiv.org/abs/2203.15556 （Section 3.4, Table 2 の指数は ar5iv の HTML 版 https://ar5iv.labs.arxiv.org/html/2203.15556 で確認した）

[Jaegle et al., 2021] Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., Carreira, J. "Perceiver: General perception with iterative attention." International Conference on Machine Learning, pp. 4651–4664. PMLR, 2021.

[Jaegle et al., 2022] Jaegle, A., Borgeaud, S., Alayrac, J.-B., et al. "Perceiver IO: A general architecture for structured inputs & outputs." International Conference on Learning Representations, 2022. https://openreview.net/forum?id=fILj7WpI-g

[Janner et al., 2021] Janner, M., Li, Q., Levine, S. "Offline reinforcement learning as one big sequence modeling problem." Advances in Neural Information Processing Systems, 34:1273–1286, 2021.

[Kang et al., 2023] Kang, M., Zhu, J.-Y., Zhang, R., Park, J., Shechtman, E., Paris, S., Park, T. "Scaling up gans for text-to-image synthesis." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[Kaplan et al., 2020] Kaplan, J., McCandlish, S., Henighan, T., et al. "Scaling laws for neural language models." arXiv:2001.08361, 2020. https://arxiv.org/abs/2001.08361

[Kostrikov et al., 2021] Kostrikov, I., Nair, A., Levine, S. "Offline reinforcement learning with implicit q-learning." arXiv:2110.06169, 2021. https://arxiv.org/abs/2110.06169

[Kudo & Richardson, 2018] Kudo, T., Richardson, J. "Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing." arXiv:1808.06226, 2018. https://arxiv.org/abs/1808.06226

[Kumar et al., 2020] Kumar, A., Zhou, A., Tucker, G., Levine, S. "Conservative q-learning for offline reinforcement learning." Advances in Neural Information Processing Systems, 33:1179–1191, 2020.

[Kumar et al., 2022a] Kumar, A., Agarwal, R., Geng, X., Tucker, G., Levine, S. "Offline q-learning on diverse multi-task data both scales and generalizes." arXiv:2211.15144, 2022. https://arxiv.org/abs/2211.15144

[Kumar et al., 2022b] Kumar, A., Singh, A., Ebert, F., et al. "Pre-training for robots: Offline rl enables learning new tasks from a handful of trials." arXiv:2210.05178, 2022. https://arxiv.org/abs/2210.05178

[Lampe et al., 2023] Lampe, T., Abdolmaleki, A., Bechtle, S., et al. "Mastering stacking of diverse shapes with large-scale iterative reinforcement learning on real robots." 2023.

[Lange et al., 2012] Lange, S., Gabel, T., Riedmiller, M. "Batch reinforcement learning." Reinforcement learning: State-of-the-art, pp. 45–73. Springer, 2012.

[Lee, A. X. et al., 2021] Lee, A. X., Devin, C., Zhou, Y., et al. "Beyond pick-and-place: Tackling robotic stacking of diverse shapes." arXiv:2110.06192, 2021. https://arxiv.org/abs/2110.06192

[Lee, K.-H. et al., 2022] Lee, K.-H., Nachum, O., Yang, M. S., et al. "Multi-game decision transformers." Advances in Neural Information Processing Systems, 35:27921–27936, 2022.

[Levine et al., 2020] Levine, S., Kumar, A., Tucker, G., Fu, J. "Offline reinforcement learning: Tutorial, review, and perspectives on open problems." arXiv:2005.01643, 2020. https://arxiv.org/abs/2005.01643

[Loshchilov & Hutter, 2017] Loshchilov, I., Hutter, F. "Decoupled weight decay regularization." arXiv:1711.05101, 2017. https://arxiv.org/abs/1711.05101

[Neumann & Gros, 2022] Neumann, O., Gros, C. "Scaling laws for a multi-agent reinforcement learning model." arXiv:2210.00849, 2022. https://arxiv.org/abs/2210.00849

[Octo Model Team et al., 2023] Octo Model Team, Ghosh, D., Walke, H., et al. "Octo: An open-source generalist robot policy." 2023. https://octo-models.github.io

[Open X-Embodiment Collaboration, 2023] Open X-Embodiment Collaboration. "Open x-embodiment: Robotic learning datasets and rt-x models." arXiv:2310.08864, 2023. https://arxiv.org/abs/2310.08864

[OpenAI, 2023] OpenAI. "Gpt-4 technical report." 2023.

[Peng et al., 2019] Peng, X. B., Kumar, A., Zhang, G., Levine, S. "Advantage-weighted regression: Simple and scalable off-policy reinforcement learning." arXiv:1910.00177, 2019. https://arxiv.org/abs/1910.00177

[Reed et al., 2022] Reed, S., Zolna, K., Parisotto, E., et al. "A generalist agent." Transactions on Machine Learning Research, 2022.

[Riedmiller et al., 2018] Riedmiller, M., Hafner, R., Lampe, T., et al. "Learning by playing solving sparse reward tasks from scratch." International Conference on Machine Learning, pp. 4344–4353. PMLR, 2018.

[Rombach et al., 2021] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. "High-resolution image synthesis with latent diffusion models." 2021.

[Russell, 2019] Russell, S. "Human compatible: Artificial intelligence and the problem of control." Penguin, 2019.

[Shridhar et al., 2023] Shridhar, M., Manuelli, L., Fox, D. "Perceiver-actor: A multi-task transformer for robotic manipulation." Conference on Robot Learning, pp. 785–799. PMLR, 2023.

[Tunyasuvunakool et al., 2020] Tunyasuvunakool, S., Muldal, A., Doron, Y., et al. "dm_control: Software and tasks for continuous control." Software Impacts, 6:100022, 2020. https://doi.org/10.1016/j.simpa.2020.100022

[Vaswani et al., 2017] Vaswani, A., Shazeer, N., Parmar, N., et al. "Attention is all you need." Advances in Neural Information Processing Systems, 30, 2017.

[Wang et al., 2020] Wang, Z., Novikov, A., Zolna, K., et al. "Critic regularized regression." Advances in Neural Information Processing Systems, 33:7768–7778, 2020.

[Wu et al., 2019] Wu, Y., Tucker, G., Nachum, O. "Behavior regularized offline reinforcement learning." arXiv:1911.11361, 2019. https://arxiv.org/abs/1911.11361

[Ze et al., 2023] Ze, Y., Yan, G., Wu, Y.-H., et al. "Gnfactor: Multi-task real robot learning with generalizable neural feature fields." arXiv:2308.16891, 2023. https://arxiv.org/abs/2308.16891

[Zitkovich et al., 2023] Zitkovich, B., Yu, T., Xu, S., et al. "Rt-2: Vision-language-action models transfer web knowledge to robotic control." 7th Annual Conference on Robot Learning, 2023.

[action_query_papers, 2026/08] anet-lab. "Action Query 関連論文サーベイ." reports/action_query_papers_2026-08-13.md

[093 attention エントロピー診断 PRD, 2026/10] anet-lab. "TransformerEncoder の attention エントロピー診断 PRD." docs/memo/done/093_attention_entropy_diagnostics_10prd.md

[DropMerge 設定, 2026/11] anet-lab. "apps/runner/config/DropMerge.txt"（`net.branch.@SuikaImpala2_ViT128`、`net.block.[DMImpalaTrans128]`、`net.@iqn`）
