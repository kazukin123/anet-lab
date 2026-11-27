---
title: "Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning — 詳細解説"
page_kind: サマリ
material: "[[reports/sharma2017_figar_paper_2026-10-03.md|sharma2017_figar_paper_2026-10-03]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning — 詳細解説

## 元の材料

[Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning — 詳細解説](../../../../reports/sharma2017_figar_paper_2026-10-03.md) blob:50e0d3ba8150308074aef9daf0b1b672679e3874

## 要点

- 要旨によれば、強化学習のアルゴリズムは、環境と相互作用して報酬という形のフィードバックを得ながら、逐次的な意思決定のための複雑な行動パターンを学べる。ただし従来のアルゴリズムは、行動の選択を相互作用の時間ステップごとに行ってきた [p.1, Abstract]。論文はこれに対して新しい枠組み FiGAR（Fine Grained Action Repetition）を提案する。FiGAR のエージェントが決めるものを、要旨は次のように書く。
- 枠組みと主張: FiGAR は、方策空間で時間的抽象化を学ばせて方策最適化の DRL アルゴリズムを改善する軽量な枠組みで、A3C、TRPO、DDPG のような連続・離散の行動空間を扱う方策勾配型のアルゴリズムに一般に適用できる。行動の確率分布に、選んだ行動を繰り返す時間スケールの確率分布を加えた構造化された方策を持つ。著者らは、時間的に引き延ばしたマクロ行動の最適な列を見つけることで、FiGAR が複数の領域でよりよい制御方策を学び、方策勾配法と actor-critic 法を大きく改善できることを結果が示したとする。
- 決定論的な環境: Atari、TORCS、MuJoCo はほぼ決定論的で、環境の動力学の確率性が小さい。著者らは、こうした環境では FiGAR のエージェントが環境の動力学の潜在的なモデルを作り、死なずに大きな行動反復を実行できると予想し、Freeway をその例に挙げる。図 1 (a) [p.2, Figure 1] では、ニワトリが近づく車の速さを見積もり、車をぎりぎりでかわしながら道路を渡ってゴールに着く。
- 欠点と今後の課題: 行動反復（マクロ行動）を途中で止める能力は、特に確率的な環境で重要だが、論文の設定は一度実行を決めたマクロ行動を止めることを考えていない。実行中に環境が予想外に変わったときにはこの能力が必要で、マクロ行動を止める行動と開始する行動を動的な時間スケールの基本設定に加えれば、より頑健な方策になりうる。著者らは、この変更が Minecraft のようなより一般的な確率的世界でも機能しうると考え、今後の課題としている。
- 連続制御では平均・標準偏差・行動反復のネットワークを分けた系と、共有を変えた付録の比較がある。

## 内容

### 要旨（Abstract）

FiGAR は、行動空間での時間的抽象化を可能にすることで、方策の推定値を明示的に保持する深層強化学習（DRL）アルゴリズムであればどれでも改善できるとされる。有効性は、異なる領域の 3 つの方策探索アルゴリズムに FiGAR を組み合わせたときの性能向上として、実験で示される。Atari 2600 での Asynchronous Advantage Actor Critic（A3C）、MuJoCo での Trust Region Policy Optimization（TRPO）、自動車レースの TORCS での Deep Deterministic Policy Gradients（DDPG）である [p.1, Abstract]。

### 1. はじめに（Introduction）

FiGAR は、行動を選ぶ方策と行動反復回数を選ぶ方策を独立に分けた、構造化・因子分解された方策の表現によって、行動空間が連続か離散かによらず、どの DRL アルゴリズムにも、時間的に引き延ばしたマクロ行動の形で時間的抽象化を学ばせる枠組みである。論文は、2 つを独立に決めるので、[Vezhnevets et al. (2016)] や [Lakshminarayanan et al. (2016)] と違って行動空間が膨らまないと指摘する [p.2, Section 1]。

時間的抽象化のための因子分解された方策表現により、DRL アルゴリズムの汎用的な拡張を提案した。2 つの Atari 2600 のゲームで学習されたマクロ行動の列を図 1 [p.2, Figure 1] に示す。

### 2. 関連研究（Related Work）

Transition Point Dynamic Programming（TPDP）[Buckland & Lawrence (1994)]: 連続な確率的動的システムの制御に要する学習時間とメモリを減らすよう、表形式の動的計画法を改変したもの。方策は遷移点と呼ぶ状態でだけ変わる。TPDP は、ある状態を遷移点の集合に加えるか除くかを Q 学習の変種で評価して、最適な遷移点の集合を学ぶ。FiGAR は遷移点をその場で学び、状態空間全体へ汎化するのに対し、TPDP は表形式で大規模な問題には使えない。

Dynamic Frameskip Deep Q-network [Lakshminarayanan et al. (2016)]: DQN [Mnih et al. (2015)] に、時間スケールごとに同じ基本行動の出力の流れを別々に加えて、反復の時間スケールを動的に学ぶ。いくつかの Atari 2600 のゲームで大きく性能が上がったが、行動空間の膨張のおそれから多数の時間スケールを扱えず、離散行動空間に限られる。[Durugkar et al. (2016)] も同じ行動を異なる時間スケールで繰り返すマクロ行動を学ぶが、離散行動空間に限られ、性能向上も大きくない。

### 3. 背景（Background）

$\gamma$: 割引率。0 と 1 のあいだの数で [Schulman et al. (2015), Section 2]、3.1 節と 4.2 節の目標値の式では、報酬にかける重みとして $\gamma$ のべき乗が現れる。

$\pi_{\theta_a}(a \mid s)$: パラメータ $\theta_a$ を持つ方策（actor）。状態 $s$ で行動 $a$ を選ぶ確率。

### 4. FiGAR: 細粒度の行動反復（Fine Grained Action Repetition）

$Z$: FiGAR で拡張する元の DRL アルゴリズム。方策を明示的に持つものに限る。論文の例は A3C、TRPO、DDPG。

$A$: 連続行動の次元数。$f_{\theta_a}(s_j)$ の出力の各次元が独立な連続値の行動を表す。4.2 節のアドバンテージ $A(\cdot)$ や行動空間の大きさ $|A|$ と同じ文字だが、別の意味である。

### 5. 実験設定と結果（Experimental Setup and Results）

図 2 [p.7, Figure 2] は、33 ゲームそれぞれの改善率 $i$ を百分率で示す棒グラフである。棒は改善率の大きい順に並び、改善は青、低下は赤で描かれ、各棒に値が書かれている。低下の棒には大きさだけが書かれている（本書による図の読み取り）。棒に書かれた値のうち最大の 2 つは、Enduro の 91822 と Atlantis の 3768 である [p.7, Figure 2]。

問い (1) に答えるため、各行動反復回数が選ばれた割合を記録した。図 3 は任意に選んだゲームでの分布で、値は小数第 2 位に丸めてあるため、ゲームごとの合計が 1 にならないことがある。各ゲームは、図 2 の平均スコアの計算と同じ方策で 10 エピソードプレイした [p.8, Section 5.1]。

### 付録 A. FiGAR-A3C の実験詳細（Appendix A）

アルゴリズム: LSTM を使う A3C の変種 [Mnih et al. (2016)]。

エントロピー正則化: [Mnih et al. (2016)] と同様に探索のために使う。係数 $\beta$ は調整の結果 0.02 で、行動方策 $f_{\theta_a}$ と反復方策 $f_{\theta_x}$ の両方に適用した。

表 4 [p.13, Table 4] は、33 ゲームそれぞれについて、最終的な FiGAR-A3C と A3C の生のスコアを並べ、括弧内に信頼水準 0.95 の信頼区間（下限, 上限）を添えた表である。信頼区間は 100 エピソードのスコアの平均から計算した [p.14, Appendix A]。各行で高い方のスコアが太字になっている（本書による表の読み取り）。A3C の方が高いのは Chopper Command、Crazy Climber、Demon Attack、Name This Game、Phoenix、Q-bert、Road Runner の 7 ゲームで、残る 26 ゲームでは FiGAR-A3C の方が高い（本書による数え上げ）[p.13, Table 4]。差が大きい例として、Enduro は FiGAR-A3C が 707.80、A3C が 0.77、Atlantis は 6330600.00 と 163660.00 である。逆に Name This Game は 1752.60 と 12100.80、Demon Attack は 7499.30 と 26742.75 である [p.13, Table 4]。A3C の Seaquest の値（2799.60）は表 2 のベースラインの値と異なる（食い違いの項目 6）。

### 付録 B. Atari 2600 の追加実験（Appendix B）

この付録は、論文が選んだ評価方式の影響を調べる [p.15, Appendix B]。

表 8 [p.19, Table 8] は、33 ゲームそれぞれについて、確率的な方策（Stochastic の列）と 0.1-greedy の方策（0.1-Greedy の列）での平均行動反復回数を並べた表である。2 つの方策の差が最も大きいのは Bowling で、確率的な方策では 28.64、0.1-greedy の方策では 5.13 である [p.19, Table 8]（差が最大であることは本書による照合）。

### 付録 C. FiGAR-TRPO の実験設定（Appendix C）

ハイパーパラメータ: 最も重要なのは $\beta_{ar}$ と $\beta_{KL}$ である。集合 $\{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28\}$ でのグリッドサーチにより、$\beta_{ar} = 1.28$、$\beta_{KL} = 0.64$ を得た。調整は 5 課題すべてで行った [p.19, Appendix C]。

行動反復の関数 $f_{\theta_x}$: 平均ネットワークに似た隠れ層 2 層のネットワーク（act-rep ネットワーク）で実現する。論文は平均ネットワークより小さい隠れ層としつつ、サイズを (128, 64) と書いている（食い違いの項目 10）。出力の非線形性は、$W$ の大きさに合わせたサイズ 30 の softmax 層である。FiGAR-TRPO のパラメータ数が TRPO より大きく増えないよう、このネットワークは小さくした。

### 付録 D. FiGAR-DDPG の実験詳細（Appendix D）

行動反復集合: 主実験では $W$ のハイパーパラメータを任意に 15 とした。

探索: DDPG は off-policy の actor-critic なので十分な探索が必要である。行動方策の空間では Ornstein-Uhlenbeck 過程 [Lillicrap et al. (2015)] で探索する。反復方策の空間では、探索のための手だてを 2 つとる。1 つ目は学習時に ε-greedy 版の方策を使うことで、$\epsilon$ は 50000 学習ステップで 0.2 から 0 まで減衰する。2 つ目は、確率 $1 - \epsilon$ の場合に貪欲な反復回数を選ぶ代わりに、出力分布 $f_{\theta_x}(s)$ からサンプルすることである。

### 付録 E. FiGAR の変種の詳細（Appendix E）

5.4 節の FiGAR-A3C の変種（行動反復集合 $W$ だけが異なるもの。定義は表 3）について、学習曲線と最終スコアの棒グラフを示す付録である [p.9–10, Section 5.4] [p.10, Table 3]。

図 6 [p.22, Figure 6] は、Space Invaders、Asterix、Seaquest の 3 つのパネルに、6 つの変種と Baseline の学習曲線を重ねて描く（本書による図の読み取り）。キャプションは Sea Quest と Asterix の 2 ゲームの比較と書いている（食い違いの項目 12）。

### 付録 F. 反復方策の重要性（Appendix F）

この付録は、学習後の評価で反復方策 $\pi_{\theta_x}$ を使わず、毎時間ステップ $\pi_{\theta_a}$ だけで行動を選ぶ変種を、通常の FiGAR と比べる。この変種は反復率 1 にあたり、FiGAR-wo-$\pi_{\theta_x}$ と呼ばれる。多くのゲームでこの変種が劣ることから、論文は、$\pi_{\theta_x}$ が学んだ時間的抽象化は自明なものではなく、ゲームプレイの性能に効いていると論じる。評価方式は付録 A と同じである [p.23, Appendix F]。

表 9 [p.23, Table 9] は、ゲームごとに FiGAR（通常の FiGAR-A3C）と FiGAR-wo-$\pi_{\theta_x}$ のスコアを並べた表である。FiGAR の列の値は表 4 の FiGAR-A3C の列と同じである（本書による照合）。各行で高い方のスコアが太字になっているが、太字のない行が 4 つある。Chopper Command、Crazy Climber、Demon Attack、Name this Game で、どれも FiGAR の方が高い。4 つとも、表 4 で A3C が FiGAR-A3C を上回ったゲームである（本書による表の読み取り）[p.13, Table 4]。反復方策を捨てると大きく下がる例として、Pong は 20.32 から -21.00 に、Crazy Climber は 154177.00 から 210.00 になる。逆に反復方策を捨てた方が高い例として、Beam Rider は 2348.78 に対して 3732.00、Enduro は 707.80 に対して 941.10 である [p.23, Table 9]。

### 付録 G. FiGAR-TRPO の共有表現実験（Appendix G）

ハイパーパラメータ: 付録 C と同じ。ただし $\beta_{ar}$ と $\beta_{KL}$ は付録 C と同様のグリッドサーチで 5 課題すべてで調整し直し、$\beta_{ar} = 1.28$、$\beta_{KL} = 0.16$ を得た [p.24, Appendix G]。

学習と評価: 付録 C と同じ手順で、TRPO の 2 段階の学習を行い、P1 の平均収益で選んだ最良方策を 100 エピソードで評価する [p.24, Appendix G] [p.19, Appendix C]。

### 論文が述べている限界

一度実行を決めたマクロ行動を途中で止めることを考えていない。確率的な環境ではこの能力が重要で、止める行動と開始する行動の追加は今後の課題である [p.10, Section 6]。

MuJoCo では TRPO を 5 課題中 3 課題で上回るが、多くの課題で上回る幅はわずかである [p.9, Section 5.2]。

### 論文内の記載の食い違い（本書による照合）

本書が論文の本文・図・表を突き合わせて見つけた食い違いを、論文の位置の順に挙げる。原因の推測はしない。計算値は本書が論文の数値から求めたもので、小数第 3 位を四捨五入した。節の末尾に、論文の参考文献の書誌を論文の外の書誌と照合した結果を別に記す。

### キューの論点への答え

定義から分かること（本書による補足）: 50 未満の素数は 2 から 47 までの 15 個なので、FiGAR-P の $W$ は 1 を含まず、間隔もそろっていない。それでも FiGAR-P は 3 ゲームともベースラインを上回った [p.10, Table 2]。間引いた 2 つの変種について、論文は実際に抽出した集合を示していない [p.10, Table 3]。

## 主要な概念

- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [Atari横断](../../concepts/subjects/atari.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
