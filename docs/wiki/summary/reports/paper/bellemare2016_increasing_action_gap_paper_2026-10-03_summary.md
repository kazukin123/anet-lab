---
title: "Increasing the Action Gap: New Operators for Reinforcement Learning — 詳細解説"
page_kind: サマリ
material: "[[reports/paper/bellemare2016_increasing_action_gap_paper_2026-10-03.md|bellemare2016_increasing_action_gap_paper_2026-10-03]]"
material_kind: 論文解説
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# Increasing the Action Gap: New Operators for Reinforcement Learning — 詳細解説

## 元の材料

[Increasing the Action Gap: New Operators for Reinforcement Learning — 詳細解説](../../../../../reports/paper/bellemare2016_increasing_action_gap_paper_2026-10-03.md) blob:e4b0fb98daf3d419c135134b0e837735219abcca

## 要点

- 最適行動と次善行動の価値の差を広げ、推定誤差で貪欲方策が変わりにくくなる条件を調べた。
- advantage learning: 同じ差し引きを定数 $\alpha$ で全遷移に当てる [p.6, Section 4.1]。$\alpha<1$ なら価値は $V^*$ に収束する [p.6, Section 4.1, Corollary 2]。
- 定理1は、Q 関数を厳密に表現できる場合に最適な貪欲方策が保たれることを示す。Atari の DQN に対しては実験で成績を確かめた。
- Atari では sticky action とライフ終了の扱いが異なる二通りの設定で DQN・A.L.・P.A.L. を比較した。
- P.A.L. が A.L. を上回ったのは16ゲーム、下回ったのは6ゲームだった。

## 内容

### 概要（Abstract）

最適方策を変えずに、最適でない行動の Q 値だけを下げて行動ギャップ（最適な行動と次善の行動の価値の差、action gap）を広げる作用素を提案する論文である。ギャップが広いと近似誤差や推定誤差で貪欲方策が崩れにくい、というのが論拠になっている。まず表形式向けの一貫ベルマン作用素を示し、離散化した連続系でも良い結果を報告する。次に、作用素が最適性を保つ十分条件を導いて作用素の族を定め、Baird のアドバンテージ学習の最適性をその系として示す。最後に Atari 2600 の 60 ゲームで効果を確かめる [p.1, Abstract]。

### 1. はじめに（原文に見出しなし）

価値ベースの強化学習は、作用素を繰り返し当てて価値関数を改善する。価値反復 [Bellman (1957)] はベルマン作用素 $\mathcal{T}$ を $V_{k+1}=\mathcal{T}V_k$ と当て続け、ダイナミクスが未知なら Q 関数 $Q_k$ とサンプルに基づく作用素で代用する（Q 学習 [Watkins (1989)]）。不動点が最適 Q 関数 $Q^*$ で、そこから最適方策 $\pi^*$ が決まる [p.1, Section 1]。

この修正で最適方策の集合は保たれ、最適でない行動の値は一般に下がるので、行動ギャップが広がる。著者らはこれを誤差への備え [Farahmand (2011)] と位置づけ、時間刻みの細かい系で特に重要とみる。例に挙がるのはビデオゲーム [Togelius et al. (2009)]、[Bellemare et al. (2013)]、リアルタイム市場 [Jiang and Powell (2015)]、ロボット [Riedmiller et al. (2009)]、[Hoburg and Tedrake (2009)]、[Deisenroth and Rasmussen (2011)]、[Sutton et al. (2011)] である。同じ発想は連続時間向けのアドバンテージ学習 [Baird (1999)] や、連続 MDP の離散化解法 [Munos and Moore (1998)]、[Munos and Moore (2002)]（極限はハミルトン・ヤコビ・ベルマン方程式 [Kushner and Dupuis (2001)]）にも見られるという [p.1, Section 1]。

### 2. 背景（Background）

$\mathcal{X}$, $\mathcal{A}$: 状態空間と、有限の行動空間。

貪欲方策の Q 関数が $Q$ 自身に一致するのは $Q=Q^*$ のときだけである。論文は (1) の形で述べるが、結果は SARSA [Rummery and Niranjan (1994)]、方策評価 [Sutton (1988)]、fitted Q-iteration [Ernst et al. (2005)] にも広がり、提案作用素はどれも Q 学習型のサンプル版をもつとする [p.2, Section 2]。

### 3. 一貫ベルマン作用素（The Consistent Bellman Operator）

この例では $\epsilon$ 程度の推定誤差で最悪の方策が貪欲になりうる [p.3, Section 3]。戻るたびに同じ行動をとるとした Q 関数 (4) ならギャップは $\epsilon/(1-\gamma/2)$ に広がるが、そこから有用な作用素は作れそうにない。そこで定常性を、1 ステップ先で同じ状態に戻る場合に限って入れる [p.3, Section 3]。

表現: 状態は 6 次元の特徴で、4 つが角度と角速度、2 つがゴールに対する極座標 $(\psi,d)$ である。$10\times\cdots\times10$ の格子の多重線形補間と $\mathcal{T}_{\mathrm{CQVI}}$ を使う。

### 4. 収束する作用素の族（A Family of Convergent Operators）

定理 1: (1) の $\mathcal{T}$ について、ある $\alpha\in[0,1)$ があり、すべての $Q,x,a$ で次を満たす $\mathcal{T}'$ は、最適性保存的かつギャップ増大的である。右の不等式が論文の条件 1、左が条件 2 にあたる [p.5, Section 4, Theorem 1]。

各反復で下げてよい量の上限は、現在の $Q_k$ の行動ギャップに届かない範囲である。途中で最適行動が貪欲行動より低く評価されることはあっても際限なくは続かず、最適行動 $a^*$ の $Q_k(x,a^*)$ は最後に $V^*(x)$ に届く。最適行動が複数あれば、保証されるのはそのうち 1 つである [p.5, Section 4]、[p.5, footnote 3]。著者らの知る限り、縮小性の議論に頼らない初めての収束の結果で、不動点の存在も仮定しない [p.5, Section 4]。

### 5. Atari 2600 での実験結果（Experimental Results on Atari 2600）

統計的に上回った・下回ったゲーム数: Stochastic Minimal の 60 ゲームでの t 検定の結果。

下回ったゲームとして論文は ATLANTIS と JAMES BOND を挙げている。P.A.L. と A.L. の直接の比較では、P.A.L. が 16 ゲームで上回り 6 ゲームで下回った [p.6, Section 5]。

### 6. 未解決の問題（Open Questions）

本当に要るのは反復の極限 $\tilde Q$ の貪欲行動が最適であることだけで、最適価値の保存すら要らないはずである。定理 1 より弱い条件がありうるが、今の証明技法は届かないという [p.7, Section 6.1]。

### 7. 結び（Concluding Remarks）

総括は、ギャップが近似価値関数の貪欲方策の性能を大きく左右し、ベルマン作用素の小さな修正で性能が大きく伸びる、というものである。あわせて、古典的な Q 関数は実用の方策を得るには心もとないこと、方策と価値の伝統的な関係や価値の概念そのものを見直す余地があることを指摘する [p.7, Section 7]。

### 9. 付録（Appendix）

補題 2: 同じ条件で、$\lvert V_k\rvert$ は $V_0$ と $R$ の大きさと $1/(1-\gamma)$ で一様に抑えられる (12) [p.9, Section 9.1, Lemma 2]。

### 論文が述べている限界

定理 1 が最適な貪欲方策を保証するのは、Q 関数を厳密に表現できる場合に限られる [p.5, Section 4]。

### 論文内の表記の食い違い

系 3 は $\mathcal{T}_{\mathrm{CQVI}}$ に (9) を添えるが、この作用素の式番号は (10) である。$\mathcal{T}_{\mathrm{C}}$ にも系 3 は (5) を、系 1 は (8) を添えている [p.10, Section 9.1, Corollary 3]、[p.4, Section 3.2]、[p.5, Section 4, Corollary 1]。

### キューの論点への答え

効用: 論文が挙げる利点は、推定誤差への頑健さ [p.3, Section 3]、補間で得る方策の改善 [p.4, Section 3.3]、過大評価バイアスの低減（仮説）[p.7, Section 5] の 3 つである。

### SMDP の PRD の検討に効く点

役割が違う。経過時間 $\tau$ は目的（何が最適か）を変えて、待つ値段と計画の見通しを分ける。行動ギャップを広げる作用素は最適方策を変えず（厳密な表現の場合、定理 1）、順位の逆転を減らすだけである。§1 の「本来の差 $(1-\gamma)V$ が揺れで入れ替わる」には直接効くが、§4 のストーリー 1 は満たせない。代わりに Env と replay の契約を変えず、$\tau$ の実装を待たずに比べられ、return の物差し（§5 の 6）も変わらない。

## 主要な概念

- [行動の時間尺度](../../../concepts/methods/temporal_abstraction.md)
- [Atari横断](../../../concepts/subjects/atari.md)
- [324 Agent個別実装 DQN系列](../../../concepts/framework/324_dqn.md)
