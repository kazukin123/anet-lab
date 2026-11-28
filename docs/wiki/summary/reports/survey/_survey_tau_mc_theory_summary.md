---
title: "Survey: モンテカルロ積分の分散低減サンプリング理論と quantile/τ サンプリングへの適用"
page_kind: サマリ
material: "[[reports/survey/_survey_tau_mc_theory.md|_survey_tau_mc_theory]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# Survey: モンテカルロ積分の分散低減サンプリング理論と quantile/τ サンプリングへの適用

## 元の材料

[Survey: モンテカルロ積分の分散低減サンプリング理論と quantile/τ サンプリングへの適用](../../../../../reports/survey/_survey_tau_mc_theory.md) blob:96aaa9bcc63155c962e005c12a8d8f824af68857

## 要点

- 層化抽出は、積分域を排反な層(strata)に分割し、各層から所定数のサンプルを取る方法である。比例配分(proportional allocation, nj = nωj)の層化は、通常の iid モンテカルロより分散が大きくなることはない。これは層内分散と層間分散への分解(σ² = σ²_W + σ²_B)から従う。
- LHS は d 次元の各座標軸を n 等分し、各軸のすべての層に 1 点ずつ入るようにサンプルを配置する「全軸同時 1 次元層化」である。導入は McKay, Beckman, Conover (1979, Technometrics)。Owen の教科書の章末注は、原論文の内容(重要変数を知らなくても自動的に層化されること、各入力に単調な関数では iid より分散が小さいことの証明)を次のように要約している。
- 対蹠変量の分散は Var(µ̂_anti) = (σ²/n)(1+ρ)(ρ はペア間相関)であり、層化と異なり無条件の保証はない。最良で分散ゼロ、最悪で 2 倍になる。分散低減の十分条件は各入力についての単調性である。
- jittered sampling(規則格子の各セル内で 1 点をランダムに揺らす)は、1 次元の [0,1] を n 等分して各ビンから 1 点取る層化抽出と同型の構成である。CG では Cook (1986, Pixar, ACM Transactions on Graphics) が導入し、規則サンプリングのエイリアシングをノイズに変換する効果を示した。
- 今回の調査で「τ の取り方(等間隔 vs 一様サンプル)と推定分散」を明示的に結び付けている唯一の査読付き論文は、Variance Control for Distributional Reinforcement Learning(Kuang, Zhu, Zhang, Zhou; ICML 2023)であった。被引用数: 約4(Semantic Scholar, 2026-08-12 確認)。同論文は Q 値推定量(分位平均)の分散を制御する QEM 推定量を提案し、その分散低減の十分条件が QR-DQN の等間隔 τ では成り立つが、IQN の一様サンプル τ では成り立たないことを指摘している。

## 内容

### 1. 一般理論: 分散低減サンプリング技法

一般理論の主典拠として、Art Owen(Stanford University)のオンライン教科書 "Monte Carlo theory, methods and examples" の第8章(基本の分散低減)・第10章(発展的分散低減)を用いる。教科書級ソースであり、各命題に原典(Stein 1987, Owen 1997 等)への帰属が明記されている。

原典の書誌情報: Sobol' (1967) "On the distribution of points in a cube and the approximate evaluation of integrals" USSR Computational Mathematics and Mathematical Physics 7(4):86–112、Halton (1960) "On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals" Numerische Mathematik 2:84–90(いずれも書誌のみ確認、原文引用は未取得)。

### 2. 機械学習の期待値推定への適用事例

NeRF(Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng; UC Berkeley / Google Research / UC San Diego, ECCV 2020)は、ボリュームレンダリング積分の求積点を等間隔ビン内一様サンプル(= jittered / 層化)で取る。

NeRF 論文中での層化の動機は分散低減ではなく、最適化の過程で MLP が連続位置で評価されること(連続表現の学習)に置かれている点に注意。

### 3. 分布RLの τ サンプリングへの QMC/stratified/jitter 適用(探索結果)

注意: 同論文は τ のサンプリング方式自体を層化や QMC に置き換える提案はしておらず、推定量(分位値の重み付け)側の改良である。

### 4. 分位点推定における層化・重点サンプリングの統計学的知見

分位点(quantile)推定への分散低減は、金融リスク管理(VaR)と信頼性解析の分野で確立した文献群がある。

### 5. 調査の限界

McKay, Beckman, Conover (1979) と Stein (1987) の原文は paywall のため未入手であり、内容の引用は Owen 教科書による記述(帰属明記あり)を経由している。Sobol' (1967)、Halton (1960)、Hammersley and Morton (1956)、Glynn (1996) も同様に二次ソース経由(書誌のみ確認)。

kuang-qemrl(ICML 2023)の著者所属は未確認。zhang-rqiqn(arXiv 2026/05)は査読状況・所属未確認。

### 6. ソース一覧

[sobol-1967, 1967] I. M. Sobol'. "On the distribution of points in a cube and the approximate evaluation of integrals." USSR Computational Mathematics and Mathematical Physics 7(4):86–112. (書誌のみ、原文未入手)

## 主要な概念

- [τ配置とIQN比較](../../../concepts/methods/tau_sampling.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
- [324 Agent個別実装 DQN系列](../../../concepts/framework/324_dqn.md)
