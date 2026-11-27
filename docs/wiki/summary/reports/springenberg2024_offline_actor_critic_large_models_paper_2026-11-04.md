---
title: Offline Actor-Critic Reinforcement Learning Scales to Large Models — 詳細解説
page_kind: サマリ
material: "[[reports/springenberg2024_offline_actor_critic_large_models_paper_2026-11-04.md|springenberg2024_offline_actor_critic_large_models_paper_2026-11-04]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/action_query_attention.md|Action queryとattentionの証拠]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/subjects/robot_tasks.md|実ロボット課題の選定]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# Offline Actor-Critic Reinforcement Learning Scales to Large Models — 詳細解説

## 元の材料

[Offline Actor-Critic Reinforcement Learning Scales to Large Models — 詳細解説](../../../../reports/springenberg2024_offline_actor_critic_large_models_paper_2026-11-04.md) blob:fa2c3ca64b8ccf8e021e56e317638ad26f857a4f

## 要点

- Perceiver-Actor-Critic（PAC）という手順とモデルを示す（Figure 1）。オフライン actor-critic は計算の追加負担を大きくせずに大きなモデルとデータへスケールし、教師あり学習で観察された法則 [Henighan et al. (2020)]、[Kaplan et al. (2020)] に似た法則に従う。
- 本解説の照合: 表 16 の 1 更新あたりの FLOPs に 3M を掛けると、XXS は約 3.6E+19、L は約 4.8E+20 で、当てはめの範囲（上限 5E+20）と合う [p.22, Table 16]。パラメータは XXS→L で 30.9 倍だが前向き計算の FLOPs は 13.3 倍にとどまる [p.15, Table 4; p.22, Table 16]。また $D$ はデータセットの大きさを変えた結果ではなく、同じデータ混合を読み進めた消費量である（手順 2・3 からの整理。数え方の不整合の候補は「食い違い」の項目 12）。論文は「似た法則」を法則の形の話として述べ、指数の数値は言語モデルと比べていない。参考までに [Hoffmann et al. (2022), Section 3.4, Table 2] の指数は $a=0.46$〜$0.50$、$b=0.50$〜$0.54$ で、PAC の値は大きくパラメータ寄りである（数値の比較は本解説による）。
- オフライン actor-critic を大きくした効果を、教師あり学習との比較と return profile で調べた。
- 損失は性能の指標にならず、スケーリングの当てはめは return profile という代理に頼る [p.6, Section 4.1; p.24, Section E.3]。
- 配分の決め方: データの質に応じて変える。既定は $\alpha=0.75$ で、CHEF のように質が低ければ $\alpha=0$、$\beta=1{,}900$ である。Control Suite で $\alpha=0$ にすると 36.8 まで下がる [p.17, Section C.5; p.17, Table 7]。

## 内容

### 1. はじめに（Introduction）

構成上の工夫は 2 つ。行動を cross-attention で Q 関数へ入れること（複数行動の Q を速く評価できる）と、Perceiver 流の cross-attention で多数の入力を潜在変数へまとめることである。約 10 億パラメータのモデルを手元の機械で 20 Hz で推論できる。

Control Suite [Tunyasuvunakool et al. (2020)] などで BC を上回り、Gato [Reed et al. (2022)] も超え、質の混ざったデータから実ロボットのベンチマークで専門家の性能を回復する。

### 2. 背景と関連研究（Background and Related Work）

大きな Transformer [Vaswani et al. (2017)] の汎用エージェントは、制御を系列予測へ変換した研究 [Chen et al. (2021)]、[Janner et al. (2021)] の上に BC で学習されてきた。Gato [Reed et al. (2022)] は Atari からロボット操作までを扱い、続く大規模モデル [Brohan et al. (2022)]、[Bousmalis et al. (2023)]、[Zitkovich et al. (2023)] は多タスク・多物体・多機体のデータで学習され、微調整で新しいタスクへ一般化した [Bousmalis et al. (2023)]、[Open X-Embodiment Collaboration (2023)]。Perceiver 系の cross-attention [Jaegle et al. (2021)] はボクセル観測のロボットで使われ [Shridhar et al. (2023)]、[Ze et al. (2023)]、Octo [Octo Model Team et al. (2023)] も同様の注意機構で出力を予測する [p.2, Section 2]。

大規模言語モデルではモデル規模と性能の間に滑らかなべき乗則が報告されている [Kaplan et al. (2020)]、[Hoffmann et al. (2022)]、[Henighan et al. (2020)]。オンライン RL の研究 [Neumann & Gros (2022)]、[Hilton et al. (2023)] は 4,000 万パラメータ未満にとどまる。[Lee, K.-H. et al. (2022)] は Decision Transformer 型 [Chen et al. (2021)] を 2 億パラメータまで調べ、CQL [Kumar et al. (2020)] の基準では好ましいスケーリングが見られなかったとした。同時期のモデルベース手法 [Hansen et al. (2023)] は 3 億パラメータまでのマルチタスクで有望である [p.2–3, Section 2]。

### 3. スケールするオフライン actor-critic 学習（Scalable Offline Actor-Critic Learning）

方策と Q は同じ $z_M$ を別々のクエリで読む。違いはクエリの作りで、方策は学習ベクトル、Q は行動の符号化である。この配置の狙いとして、行動が無視されにくいことと、$z_M$ の使い回しで多数の行動サンプルを安く評価できることが挙げられる [p.5, Section 3.3]。具体形は付録 B.4 にある。

### 4. 実験（Experiments）

$C$ は計算量（FLOPs）、$N$ は性能最適なモデルのパラメータ数、$D$ は消費したトークン数である。

パラメータの指数は $a(\text{PAC})=0.920$、$a(\text{BC+Q})=0.828$ である。著者は、計算予算が増えると RL の目的関数のほうがパラメータの追加から得るものが大きいように見えるとし、Q 関数に要る容量が大きいことを示唆すると留保つきで述べる [p.6, Section 4.1]。

### 5. 考察（Discussion）

限界として、報酬の注釈が必要で高価なこと（普遍的な報酬関数 [Du et al. (2023)] や教師なしの報酬付与 [Chebotar et al. (2021)] で緩和しうる）と、多様な領域の間で強い転移が見られなかったこと（領域の重なりがより大きいデータ [Zitkovich et al. (2023)] なら改善を期待）を挙げる [p.8, Section 5]。今後として数十億パラメータへの拡大、事前学習済みの VLM との組み合わせ、言語モデルの事前学習の代替を挙げる [p.8, Section 5]。

### Broader Impact

研究目的で、配備は想定していないとする。安全上の懸念は Gato [Reed et al. (2022)] と RoboCat [Bousmalis et al. (2023)] と多くを共有し、自己改善も人が定めた報酬でオフラインに行い、実世界との相互作用中には学習しないと明記する。報酬で方策を形作ることは、人の好みとの価値整合の研究 [Russell (2019)]、[Christiano et al. (2017)] を適用しやすくするとも述べる [p.9, Broader Impact]。

### 付録 A. アルゴリズムの詳細（Method Details - Algorithm）

Q 関数の損失はこの目標との KL (10) で、総損失 (11) は (3) と同じ形である。(11) は重みを指数化した $Q_{\theta^{-}}$ と正規化定数 $K_Q=\log\mathbb{E}_{a'\sim\pi_{\theta^{-}}}[\exp(Q_{\theta^{-}}/\eta)]$ で書く [p.12, (11)]。

### 付録 B. 構成の詳細（Method Details - Architecture）

$\sigma$ は対数スケールの固定ゲインで、$N_G=8$ なら $10^{-4}$ から $10^{3}$ である。どのゲインを読むかは後段の注意機構に任せる設計で、[Reed et al. (2022)] の $\mu$-law が範囲外の入力で飽和する問題への対策とされる [p.13, Section B.1]。

### 付録 C. 実験の詳細（Experimental Details）

損失の倍率は TD 側の $\beta=38$ を固定して $\alpha$ を動かす（`BC+Q` と `FilteredBC` で 1.0、`PAC` で 0.75）。`α-PAC` は Control Suite と RoboCat で $\beta=19$、$\alpha=0.75$、CHEF で $\beta=1{,}900$、$\alpha=0$ である。`PAC+V` はすべて $\alpha=0$、$\beta=38$、温度 1e-4 である [p.17, Section C.5]。

### 付録 D. 感度とアブレーション実験（Sensitivity and Ablation Experiments）

D.1.1 BC の倍率 $\alpha$: 専門家データが多いほど BC の比重を上げるのがよく、少ないほど RL から得るものが大きい。XS モデルの Control Suite では $\alpha=0$ で 36.8 と大きく下がり、0.8 付近が最良で 85.1、純 BC の $\alpha=1$ は 82.4 である。データの質が低い CHEF では $\alpha=0$ が最良とされる [p.17, Section D.1.1; p.17, Table 7]。

計算の都合から、データの一部と小さいモデルで行う [p.18, Section D.2]。

### 付録 E. スケーリングの詳細（Scaling Details）

`PAC+V` の結論は Q 版と同じ傾向で、2.45T トークンに対して示唆される大きさは 8.52 億とやや小さく、指数は $a=0.970$ と大きい。データの指数は $b\approx0.266$ で同じである [p.24, Section E.4; p.24, Figure 9]。

### 付録 F. 追加実験（Additional Experiments）

RGB Stacking の 5 つの物体セットごとの推移が Table 18 にあり、とくに Set 2（力で物体を倒す）が 53.5 から 91.5 へ伸びる [p.25, Section F.2; p.25, Table 18]。微調整の対象は CHEF:real だけなので他の領域が落ちないかを確かめ、シミュレーションのタスクは影響を受けないとする（Table 19、ただし「食い違い」の項目 7）[p.25, Section F.2]。

### 論文が述べている限界

オフライン RL には報酬の注釈が要り、高価である [p.8, Section 5]。

### 論文内の記載の食い違い（本解説による照合）

データの指数は本文で $b\approx0.266$ だが、図の印字は `PAC` と `BC+Q` が 0.265、`PAC+V` が 0.267 である [p.6, Section 4.1; p.6, Figure 3; p.24, Figure 9]。差は丸めの範囲である。

### 論文から確かめられなかった点

$\eta$ の値（PAC）、$N_O$、$D_O$、$N_Q$（Q のビン数）、$q_{\min}$、$q_{\max}$、attention の head 数は記載が無い [p.16, Table 5 ほか]。

### キューの論点への答え

本解説の計算（$D$ の依存を除いたクエリとキーの組の数）: Body は $32\times2{,}634+18\times32^2=102{,}720$（L サイズ）で、Q の decoder は 1 行動あたり $1\times32=32$、10 サンプルで 320 である。行動を入力に入れて Body をサンプルごとに通すと、Body の分が 10 倍になる。

### このプロジェクトの検討に効く点

潜在の圧縮は後回しでよい: PAC の動機は 2,634 トークンの二乗回避である。180 トークンでも計算上は 1 層あたり約 32 倍の節約になるが、圧縮後に 58 列の位置を読み分けられるかの証拠は無い。最初の比較は潜在を入れず盤面トークンを直接読む構成にして、変える要素を Head の形だけに絞るのがよい。

## 主要な概念

- [Action queryとattentionの証拠](../../concepts/methods/action_query_attention.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [実ロボット課題の選定](../../concepts/subjects/robot_tasks.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
