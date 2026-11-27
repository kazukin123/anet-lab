---
title: "Lazy-MDPs: Towards Interpretable Reinforcement Learning by Learning When to Act — 詳細解説"
page_kind: サマリ
material: "[[reports/jacq2022_lazy_mdps_learning_when_to_act_paper_2026-11-04.md|jacq2022_lazy_mdps_learning_when_to_act_paper_2026-11-04]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# Lazy-MDPs: Towards Interpretable Reinforcement Learning by Learning When to Act — 詳細解説

## 元の材料

[Lazy-MDPs: Towards Interpretable Reinforcement Learning by Learning When to Act — 詳細解説](../../../../reports/jacq2022_lazy_mdps_learning_when_to_act_paper_2026-11-04.md) blob:dc11730d5e79d64b09ba88fc21384af28ae57048

## 要点

- lazy-MDP では既定方策に行動を任せる選択を追加し、自ら制御を取ると費用がかかる。
- 罰を変えると自分が制御する状態が変わり、どの場面の制御が成績に重要かを観察できる。
- 罰の値: 5 章の上下限の式は、常に任せるのでも常に制御を取るのでもない最適方策を保証する。$\eta_{\max}$ は、既定の方策の価値を SARSA [Rummery & Niranjan (1994)] などで推定し、既知の状態やロールアウトで差の最大を取れば近似できる。$\eta_{\min}$ は $Q^{*}$ と $D^{\pi^{*}}$ が普通は手に入らないので、価値反復や Q 学習 [Watkins & Dayan (1992)] で近似する。$D$ の近似は、損失の報酬を全部 1 に置き換えて得る [p.672–673, Section 6]。
- 一様ランダムを既定とした DQN でも、一部の状態だけ制御して標準 MDP に近い性能に達した。
- 最適方策の判定には制御の費用を組み込んだ Q を使う。

## 内容

### アブストラクト（要約）

論文は、強化学習が扱ってきた「どう行動するか」と同じくらい「いつ行動するか」が重要だと主張する。人間が習慣的なふるまいと注意を払ったふるまいを行き来するのになぞらえ、MDP に、決定を既定の方策へ任せる行動を足し、それ以外の行動に罰を与えて、決定的な場面にだけ行動させる。この形式を lazy-MDP と呼び、価値関数・貪欲性・最適解を理論的に調べる。実験では、制御を取る状態が構成上そのまま見えるので一種の解釈可能性が得られると示す。既定の方策が準最適（一様ランダムを含む）でも、Atari で限られた状態でしか制御を取らずに DQN に近づき、ときに上回る [p.669, Abstract]。arXiv 版の要旨は結びの表現が違う（書誌の注記を参照）。

### 1. はじめに（Introduction）

解釈可能性としては、既定の方策と lazy 方策の状態ごとの違いから、既定の方策を上回るのに何が要るか、または課題全体で何が要るかを読み取ることを指す。「なぜ」に答える説明 [Molnar (2020)] は狙わない [p.669–670, Section 1]。

### 2. 関連研究（Related Work）

残差 RL [Johannink et al. (2019)] [Silver et al. (2018)]: 基の方策の行動に足す残差の方策を学ぶ。行動の和が要るので連続制御に限られ、lazy-MDP は離散制御に使える。

### 3. 枠組み（Framework）

lazy-MDP は基の MDP に $\ell,\pi_{\mathrm{d}},\eta$ を足した組で、それ自体も MDP である。基の MDP の報酬を保ち、$\ell$ 以外の全行動で $-\eta$ を追加で受ける [p.671, Section 3]。$\mathcal{S},\gamma,\delta_0$ は変わらない。

### 4. lazy-MDP を解く（Solving Lazy-MDPs）

この規則は $\ell$ の価値を読まないので、論文は $\mathcal{A}$ 上の Q から貪欲な方策を作れるとし、その Q を「基の MDP の Q 関数」と呼ぶ [p.672, Section 4.3]。ただし 4.4 節の不動点は罰込みの Q なので、基の MDP の価値関数 $Q^{*}$ を入れてよいという意味ではない（本解説の補足）。

### 5. 罰の決め方（Setting the Cost）

全状態で $\eta$ が lazy-gap 以上なら、最適方策は全部を任せて費用を払わない。よって $\eta_{\max}$ は、既定の方策のもとでの lazy-gap の最大値である。$Q^{\pi_{\mathrm{d}}}$ は基の MDP での既定の方策の Q である [p.672, Theorem 3]。

### 6. いつどう行動するかの学習（Learning When and How to Act）

既定の方策: 一様ランダムは課題の知識が要らず、行動が本当に効く状態でだけ制御を取るという考えに合う最も単純な選択である。エントロピー正則化に近いが、ランダムさを促すのは一部の状態だけである。複雑な場面では、制御が多すぎると累積費用が高くなり探索を妨げかねない。残差学習のように既知の準最適な方策を置けば、その方策が明らかに悪い状態で制御を取る誘因が生まれる [p.673, Section 6]。

### 7. 実験（Experiments）

Atari [Bellemare et al. (2013)] の、タイミングが中心のゲームで、制御を取る状態を重要な状態とみなし、定性的にフレームで示す。DQN [Mnih et al. (2015)] の Dopamine [Castro et al. (2018)] の実装を使い、既定の方策は一様ランダムである [p.674, Section 7.2]。

### 8. 結論（Conclusion）

論文は、学習の問題に合う MDP の抽象化として lazy-MDP を提案し、価値関数と最適性を調べ、RL で解けることを示した。元の MDP に戻すと解釈しやすく、制御の重要な状態が分かる。一様ランダムを既定にしても、DQN で一部の状態でだけ制御を取りつつ標準の MDP の方策に近い性能に達し、探索の難しい課題ではより高いスコアにも届く、とまとめる [p.676, Section 8]。arXiv 版の結論には、学習済みの方策を既定にすると warm restart より良い傾向があるという 1 文が加わる [arXiv p.14, Section 8]。

### 付録 A〜K（arXiv 版のみ）

位置は A が p.18、B が p.18–19、C が p.19–20、D が p.20、E が p.20–22、F が p.22、G が p.22–23、H が p.23–24、I が p.24、J が p.24–25、K が p.24 と p.26 である。

### 論文内の記載の食い違い（本解説の照合）

系 1 の不等号: 系 1 は $\mu^{*}(\ell\mid s)>0$ と $\Delta^{*}(s)>\eta$ が同値だと印字する [p.672, Corollary 1]。arXiv 版も同じである [arXiv p.6, Corollary 1]。性質 3 と 4.3 節は $\Delta\le\eta$ のときに任せる行動を選ぶので [p.672, Section 4.3] [p.672, Property 3]、向きが逆である。付録 D の証明も $\Delta>\eta$ で制御を取る向きで [arXiv p.20, Appendix D]、5 章の $\eta_{\min}$・$\eta_{\max}$ の定義の説明も性質 3 の向きに合う [p.672, Section 5]。本解説は系 1 を誤植と読み、性質 3 を正とする。

7.2 節の本文は対象の例に Pong、Breakout、Ms Pacman を挙げるが [p.674, Section 7.2]、図で示すのは Breakout、Ms Pacman、Bowling である [p.675, Figure 4–6]。

### 論文に書かれていない点

DQN の実装の細部は、本文にも付録 I にも書かれていない [arXiv p.8, Section 6] [arXiv p.24, Appendix I]。

### キューの論点への答え

依頼時の reports/INDEX.md のキューにあった優先 15 の問いに、論文の記述の範囲で答える。今の INDEX ではこの行が消え、完了表に PRD 958 と書かれているので、問いの文は git の履歴（コミット d37da6f4 で追加）にだけ残る。

切り替える条件は $\max_{a}\tilde{Q}(s,a)-\tilde{Q}(s,b)>\eta$ である。エピソードの最初は $b$ が無いので別に定める必要がある。乱数の MDP で価値反復を回すと、$(s,b)$ に広げて任せる行動を持たせた解とこの式の解の差は最大で約 $2.4\times10^{-13}$ だった（本解説の補足）。条件は 8 状態、4 行動、$\gamma=0.95$、$\eta=0.1$。遷移と報酬の生成法は 5 通りで、遷移は Dirichlet(1)、一様乱数の正規化、行き先 2 状態だけの疎の 3 種、報酬は $[0,1]$ または $[-1,1]$ の一様乱数である（一様乱数の遷移は $[0,1]$ の報酬だけ）。各 20 個で、Python の `random.Random(seed0+k)`（$k=0,\dots,19$、seed0 は 0 と 100）で作った。

### PRD 958 の検討に効く点

§2.2 の説明を直す。壁打ちの「行動を変える価値の差が閾値を超えたときだけ切り替える」は論文の定義ではない。論文の差は最良の行動と既定の方策の期待値の差で、既定が「続ける」のときに限り、続けている行動との差になる。

### 本解説で省略した内容

本文の文章そのもの、図表の転載、Figure 8・9・10・11・12 の点や曲線の細部。

## 主要な概念

- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
