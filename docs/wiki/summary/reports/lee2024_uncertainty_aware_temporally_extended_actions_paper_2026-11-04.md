---
title: Learning Uncertainty-Aware Temporally-Extended Actions — 詳細解説
page_kind: サマリ
material: "[[reports/lee2024_uncertainty_aware_temporally_extended_actions_paper_2026-11-04.md|lee2024_uncertainty_aware_temporally_extended_actions_paper_2026-11-04]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
updated: 2026-11-26
---
# Learning Uncertainty-Aware Temporally-Extended Actions — 詳細解説

## 元の材料

[Learning Uncertainty-Aware Temporally-Extended Actions — 詳細解説](../../../../reports/lee2024_uncertainty_aware_temporally_extended_actions_paper_2026-11-04.md) blob:676d1761952076170429f5282e6f9e456ed91b8a

## 要点

- 行動の繰り返しは行動空間の時間抽象の一種で、方策の学習を助ける。しかし従来の研究には、最適でない行動を繰り返すと性能が落ち、利点が打ち消されることがあるという大きな限界がある。論文はこれに対して UTE（Uncertainty-aware Temporal Extension）を提案する。UTE はアンサンブルで、行動を延長している間の不確かさを測る。これにより方策は、探索を重んじる動きと不確かさを避ける動きを、目的に応じて選べる。Gridworld と Atari 2600 の実験で既存の行動繰り返し手法を上回り、方策の学習の効率を高めたとする [p.1, Abstract]。要旨に名前が出るのはこの 2 つだが、本文には Chain MDP と Pendulum-v0 の実験もある [p.5, Chain MDP] [p.7, Control Problem: Pendulum-v0]。
- 従来の行動繰り返し手法（[Lakshminarayanan et al. (2017)]、[Sharma et al. (2017)]、[Dabney et al. (2020)]、[Biedenkapp et al. (2021)]）は不確かさを考えずに延長するので失敗しやすい、と論文はいう。UTE は TempoRL の不確かさを考える版で、$\pi_e$ が $j$ を選ぶときに $\tilde{Q}$ の不確かさを考える [p.4, Ensemble-based Uncertainty Quantification]。
- 状態と行動を与えて、$j$ ごとに平均と分散を集計する [p.4, (3)] [p.4, (4)]。
- 適応的な $\lambda$: 固定より学習がやや遅く、$\lambda$ の最適化に追加のサンプルが要るためだと説明する。それでも他の手法を大きく上回り、$\lambda$ を前もって決める必要をなくせるとする [p.7, Adaptive Uncertainty Parameter λ]。
- 論文で B は10であり、ensemble の大きさを5〜10で変えた感度比較は確認できなかった。

## 内容

### はじめに（Introduction）

ただし繰り返すだけでは性能の向上は保証されない、というのが論文の問題意識である。危険な領域で無謀に繰り返せばゲームが終わるので、不確かさを避ける動きが役に立つ。逆に疎な報酬では、楽観が足りないと近傍に留まるので、探索寄りの動きが役に立つ。どちらの場合も長い時間の幅の価値の不確かさの扱いが要であり、適切な度合いは環境によって違うので適応的であるべきだとする。将来の不確かさを行動の繰り返しに取り入れた研究はこれまで無い、というのが著者らの認識である [p.1–2, Introduction]。

試した環境で、UTE が既存のベースライン（DAR、$\epsilon z$-Greedy、DQN、B-DQN）を最終の評価得点、学習の速さ、状態空間の被覆で上回るとする。

### 関連研究（Related Work）

時間抽象と行動の繰り返し: MDP に option を足すと Semi-Markov Decision Process（SMDP）になる。時間的に延長した行動を定式化したのが option の枠組みで（[Sutton et al. (1999)]、[Precup (2000)]、[Bacon et al. (2017)]）、価値の側では行動価値を一般化した option 価値（[Sutton et al. (1999)]）を、方策の側では option の確率を返す方策を考える。その手前の見方として、時間抽象を AI の行動に合う時間尺度の探索と捉える説明も挙がる [Precup (2000)] [p.2, Related Work]。

単純な option の 1 つが、基本の行動を一定回数繰り返すものである [Schoknecht and Riedmiller (2002)]。繰り返しは探索を深くし（[Dabney et al. (2020)]）、制御の粒度を下げて学習を効率化するとされる（[Lakshminarayanan et al. (2017)]、[Sharma et al. (2017)]、[Metelli et al. (2020)]、[Biedenkapp et al. (2021)]）。延長の長さは分布から引くか（[Dabney et al. (2020)]）、方策が返す（[Lakshminarayanan et al. (2017)]、[Sharma et al. (2017)]）。最も近いのは TempoRL [Biedenkapp et al. (2021)] で、behavior 方策が行動を、skip 方策が繰り返す長さを決める。著者らの直感は、ただ繰り返すだけでは足りず、繰り返しの途中で望ましくない状態に入り、「危険な」領域で破局的な失敗に至りうるというものである [p.2, Related Work]。

### 準備と記法（Preliminaries and Notations）

環境は MDP $\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$ [Puterman (2014)] で、option を組み込むために SMDP $\mathcal{M}_o:=\langle\mathcal{S},\Omega,P_o,R_o\rangle$（[Sutton et al. (1999)]、[Precup (2000)]）を考える。$P_o$ は option を取ったときの遷移、$R_o$ はその報酬である。option 上の方策 $\pi_\omega$ は状態ごとに option の分布を与え、目的は初期状態からの割引収益の期待値の最大化である [p.2–3, Preliminaries and Notations]。

option は一般に、開始からの履歴 $h$ に依存する。semi-Markov の option は開始集合、option 内の方策、終了関数の組である。論文はこのうち、行動 $a$ を $j$ 回取って終わる option を、どの状態でも開始でき、内部の方策が $a$ の one-hot で、履歴の長さが $j$ になると終わる組として定義する [p.3, Preliminaries and Notations]。

### 不確かさを考える時間的延長（Uncertainty-aware Temporal Extension）

Q 学習 [Watkins and Dayan (1992)] に基づき、最適な行動価値と option 価値を次で定義する [p.3, (1)] [p.3, (2)]。$s_{+j}$ は $s$ から $j$ step 後の状態である。

### 実験（Experiments）

主な実験は 3 つである。まず正の $\lambda$ が探索を促し（Chain MDP）、負の $\lambda$ が不確かさを避ける（Gridworlds）という仮説を確かめ、次に Atari で、調整した $\lambda$ の効果と適応的な選択の効果を示す。公平のため、各アルゴリズムの最適な値を広い範囲で探したとする [p.5, Experiments]。

### 結論（Conclusion）

不確かさを明示的に考えて行動を繰り返す手法を提案し、$\lambda$ で度合いを調整することで、Chain MDP、Gridworlds、Atari 2600、制御問題の各環境で他の手法を一貫して大きく上回ったとまとめる。著者らの知る限り、時間的に延長した行動で将来の不確かさを考える最初の深層強化学習アルゴリズムだとする [p.8, Conclusion]。

### 付録: ベースラインの詳細（Details of Baselines）

Fixed Repeat: DDQN で常に同じ回数繰り返す。$j=4$ [p.10, Fixed Repeat]。

### 付録: UTE の実装の詳細（Implementation Details: UTE）

skip 側の損失はメンバーの和で、平均ではない。目標値は全メンバーで同じである。

### 付録: 実験の詳細（Experiments Details）

全実験は GeForce RTX 3090 のクラスタで走らせ、Atari は 1000 万フレームに 13 時間かかった。手法ごとの実時間は書かれていない。Chain MDP と B-DQN の実装は [Touati et al. (2020)]、Gridworlds と Atari の設定と TempoRL の実装は [Biedenkapp et al. (2021)] に基づく [p.11, Experiments Details]。

UTE の skip 側: 同じ構成の 10 個のネットワーク（3 層、26 ユニット）。入力は状態と選んだ行動の連結。

### 付録: 追加の実験結果（Further Experimental Results）

基の DDPG は公開コード（github.com/sfujim/TD3）で、学習 step 数と初期のランダム step 数だけを変えた。FiGAR と t-DDPG の延長方策には一定の ε-greedy を使い、UTE-DDPG は UCB 型の探索を組み込んでいるので使わない。Q 学習を DDPG の更新に置き換え、基のエージェントの Q 関数を skip 側の学習に使える。UTE-DDPG は特に小さい $J$（2、4）で速く、$J$ を増やしても安定し、t-DDPG より繰り返しが少ないとする [p.21, DDPG Implementation Details and Additional Results] [p.23, Figure 14]。

### 論文が述べている限界と、論文に無いこと

繰り返しの最大長は大きければよいわけではなく、学習が遅くなりうる [p.13, Maximum Repeat J]。

### 論文内の記載の食い違い（本解説の照合）

Kangaroo: 本文は Kangaroo で正の $\lambda$ が良いとする [p.18, Per-game Best Parameter]。Table 12 の Kangaroo は $\lambda=-1.5$ の 728.5 が最高で、正の 3 値は 555.2、493.3、543.0 である。正の $\lambda$ が最高なのは Riverraid（$\lambda=+1.0$、938.0）である [p.23, Table 12]。

### キューの論点への答え

依頼時のサーベイキュー（reports/INDEX.md の優先 14）の論点メモへの答えである。その行は完了時に消えて参照先が無いので、論点をここに写す。(1) skip 価値のアンサンブルの作り（メンバーの多様性の出し方、目標値を全メンバーで共有するか）。(2) 平均と標準偏差で j を選ぶ式と、λ の符号の意味と決め方（固定か自動調整か）。(3) TempoRL との差分と、比べた env。

### PRD958 の検討に効く点

PRD958 の §9 の 1 に照らした、本解説の読みである。論文の主張ではない。PRD は変更していない。

### 本解説で省略した内容と、確かめられなかった点

Table 1〜13 の値のうち、議論に使わないもの。とくに Table 7 の残りの値、Table 11 の全値、Table 12 と Table 13 の大部分。

## 主要な概念

- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [Atari横断](../../concepts/subjects/atari.md)
