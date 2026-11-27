---
title: Unifying Task Specification in Reinforcement Learning — 詳細解説
page_kind: サマリ
material: "[[reports/white2017_unifying_task_specification_paper_2026-11-04.md|white2017_unifying_task_specification_paper_2026-11-04]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-26
---
# Unifying Task Specification in Reinforcement Learning — 詳細解説

## 元の材料

[Unifying Task Specification in Reinforcement Learning — 詳細解説](../../../../reports/white2017_unifying_task_specification_paper_2026-11-04.md) blob:321125f4f6a3f0f3fb10378a766c0e04332ff75e

## 要点

- 環境の状態遷移を共通に保ち、目的ごとに報酬、遷移に応じた割引、どの状態の予測を重視するかを指定する。
- 同じ $s_1$ に入る二つの遷移のうち一方でだけ終了したいとき、次状態だけで決める割引では区別できない。どの状態からどの行動で遷移したかを割引に含めれば表せる（原論文 §2、図1）。
- 硬い終了: $\gamma(s,a,s')=0$。
- 線形 TD では、TD 誤差の次状態の予測値に、その遷移で選んだ割引率を掛ける。適格度トレースにも遷移ごとの割引率を使う（原論文 §4、式3）。
- 状態ベースとの関係: 状態を足さない状態ベースでは表せない [p.3, Figure 1]。仮想状態を足せば等価だが、状態数と手間が増える [p.10, Theorem 2; p.10–11, Appendix B.2]。

## 内容

### 1. はじめに（Introduction）

option、一般価値関数、エピソード型・継続型を、状態や MDP に手を入れず割引関数の変更だけで書けることを示す。

一般化したタスクの Bellman 作用素に新しい縮小の上界を証明し、エピソード型・継続型の既存の上界がその特殊例になることを示す。

### 2. 問題の一般的な定式化（Generalized problem formulation）

RL タスクは、MDP の上に載せる組 $(\mathcal P,r,\gamma,\mathrm i)$ である [p.2, Section 2]。同じ環境の上に、報酬の違う別のタスク（台所へ行く、会議室へ行く、など）を複数置ける。予測でも制御でも、ある方策のリターンを評価するのが中心になる [p.2, Section 2]。リターンは遷移ごとの割引の積で定義される。

### 3. タクシー領域での実演（Demonstration in the taxi domain）

(b) 乗車の遷移で $\gamma=0.1$（柔らかい終了）、他は 0.99: 乗車後の向きまで見越して近づく最適な方策で、総報酬は $-7.7$ である。

### 4. 目的とアルゴリズム（Objectives and algorithms）

線形近似の不動点は $Aw=b$ で書ける [p.6, Section 4]。

### 5. 一般化した理論的性質（Generalized theoretical properties）

既存の上界はほとんど定数割引を仮定しており、ETD の解析 [Hallak et al. (2015)] も定数の $\gamma_c$ だった。本節は $P_{\pi,\gamma}$ の行列ノルムを使って両方の型を一度に扱う [p.6, Section 5]。

### 6. 考察と結論（Discussion and conclusion）

論文の目的は RL タスク形式の使い方を直観と例で示すことで、技術的な貢献は付録に置いた部分もある。著者は改めて 4 点を挙げる。RL タスク形式と遷移ベースの割引の導入、状態ベースと遷移ベースの関係の明示、エピソード型と継続型に共通の近似上界、状態ベース・遷移ベースへ縮小の結果を広げるときの知見と問題点である [p.8, Section 6]。謝辞は遷移ベースの割引と確率的な割引の議論について Hado van Hasselt に向けられている [p.9, Acknowledgements]。

### A. 確率的な割引を含む定式化（More general formulation with probabilistic discounts）

報酬と割引を $\Pr(r,\gamma\mid s,a,s')$ から同時に出す形にしても、扱いはほとんど変わらない [p.10, Appendix A]。次状態 $s'$ を固定すれば $v_\pi(s')$ は定数なので、期待を取ると報酬と割引の項に分かれ、$\gamma(s,a,s')=\mathbb E[\gamma\mid s,a,s']$ と置いた同じ式に戻るからである。

### B. 状態ベースと遷移ベースの割引の関係（Relationship between state-based and transition-based discounting）

Theorem 2: 元の状態へ正規化した定常分布は元の $d_\pi$ に一致し、元の状態の価値も一致する。さらに、どちらで最適化しても同じ方策になる [p.10, Theorem 2, (5); p.11, Theorem 2, (6)]（書き方は項目 6）。関数近似でも解が同じになるという意味で等価だが、大きい MDP で学習する実用面は無視している [p.10, Appendix B]。

### C. 割引と平均報酬（Discounting and average reward for control）

ただしこの議論は遷移ベースには及ばない。割引が遷移によって違うとリターンの重み付けが一様でなくなり、最適方策の選択そのものが変わるからで、タクシー領域がその実演である [p.12, Appendix C]。

### D. アルゴリズム（Algorithms）

Algorithm 1: TD 誤差のブートストラップの係数と、trace の減衰。

### E. 補題（Lemmas）

Lemma 3: A3 のもとで $P_{\pi,\gamma}$ の最大固有値は 1 未満である [p.13, Lemma 3]。既約な $P_\pi$ の成分を減らす形で $P_{\pi,\gamma}$ を作り、固有値が下がることを示す。[Yu (2015)] の Assumption 2.1 を明示的に確かめたものでもある [p.13, Appendix E]。証明の道具は項目 5 を参照。

### F. emphatic アルゴリズムの収束（Convergence of emphatic algorithms for the RL task formalism）

状態ベースの結果は B.1 の等価性で遷移ベースへ自動で移るはずだが、著者は書き方をそろえるため遷移ベースのまま証明する [p.14, Appendix F]。

### G. emphatic 重みなしの遷移ベース trace の問題（Issues with transition-based trace without emphatic weighting）

本解説の照合: この行列を計算すると同じ値になり、$\|P^\lambda_\pi\|_{D_\pi}$ は約 1.27 である。

### 論文が述べている限界

$s_D$ は定数の場合より解釈しにくい [p.8, Section 5.2]。

### 論文内の記載の食い違い

不等式の破れと $s_{D_\pi}=1$ は別である: 確率 0.5 ずつの $s_1\to s_1,s_2$、$s_2\to s_1,s_3$、$s_3\to s_1,s_2$ で $s_1$ へ入る遷移だけ $\gamma=0.5$ の連鎖は、不等式が破れても $\lambda_c=0$ の $s_{D_\pi}$ は約 0.79 である。

### キューの論点への答え

論文が扱っていない点: 結果はすべて固定した方策 $\pi$ の評価についてで、線形の関数近似までである。max を含む制御の作用素の縮小、Q 学習や DQN の収束、深層ネットワークは証明も実験もない。タクシー領域の制御は反復計算での実演である [p.4, Section 3]。リプレイの分布は $D_\pi$ でも $M$ でもないので、これらの重みの結果はそのままは当てはまらない（本解説の読み取り）。

### SMDP の PRD（956）の検討に効く点

§1 の「理論の裏づけ」は範囲を絞る。この論文が支えるのは定式化と固定方策の評価までで、DQN・QR・IQN の max とリプレイの分布は範囲外である。§5.2 の安定性と §5.4 の PER はこの論文では決まらないので、実験で確かめるか別の根拠を探す。

## 主要な概念

- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
