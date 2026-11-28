---
title: 分布強化学習における τ(quantile fraction)の非一様分布・distortion・適応的選択 — 学術サーベイ
page_kind: サマリ
material: "[[reports/survey/_survey_tau_risk_adaptive.md|_survey_tau_risk_adaptive]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# 分布強化学習における τ(quantile fraction)の非一様分布・distortion・適応的選択 — 学術サーベイ

## 元の材料

[分布強化学習における τ(quantile fraction)の非一様分布・distortion・適応的選択 — 学術サーベイ](../../../../../reports/survey/_survey_tau_risk_adaptive.md) blob:1cb7b156f1ed150f2b2fb3e8cb8432ff753bad93

## 要点

- IQN は分位関数を暗黙的に学習するネットワークで、基本形では τ ~ U([0,1]) をサンプルし、その τ における分位値 Z_τ(x,a) を return 分布からのサンプルとして用いる。risk-sensitive 化は「τ のサンプリング分布を distortion risk measure β で歪める」ことで定式化される。行動価値は β(τ) を通した期待値 Q_β として定義され、方策はその greedy 選択になる。
- CVaR 最適方策をサンプル効率よく学習するために「不確実性に直面したときの楽観主義」を分布 Bellman 作用素に持ち込んだ研究。τ のサンプリング分布を変えるのではなく、推定 CDF 自体を訪問回数ベースのボーナス c/√n(s,a) だけ下方シフトさせる楽観的作用素を定義し、その分布に対する CVaR_α を最大化する行動 a* ← argmax_a CVaR_α(F̃_a) を選ぶ [Keramati+ (2019/11)]。
- 平均でなく「特定の分位」に対して greedy に行動選択する探索法。どの分位を使うかは option 框組の高位方策がオンラインで学習する(N 個の分位を M 個の窓に分け、各 option は窓内 K 個の分位の平均に基づいて行動を提案する)[QUOTA (2018/11)]。
- IQN ベースで、CVaR を「τ のサンプリング区間の制限」として実装することを明示している論文。リスクレベル α 自体は、学習した return 分布の下側テール条件付き分散(右側切断分散 RTV と等価)を intrinsic 不確実性として、exponentially weighted average forecasting(EWAF)でオンライン適応させる(重み更新のフィードバック f = RTV_t − RTV_{t−1})[ART-IQN (2022/03)]。
- τ をランダムサンプルするのではなく、状態(-行動)ごとに τ の離散集合そのものを生成する fraction proposal network を導入し、分位値ネットワークと同時に学習する。fraction proposal network は、近似分布と真の分布の 1-Wasserstein 距離が最小になるように訓練される。「τ を非一様に配置する」ことをリスク目的でなく分布近似精度の目的で行う研究であり、今回の調査範囲では τ 分布を学習で偏らせる研究として最も直接的なもの [FQF (2019/11)]。

## 内容

### 2. CVaR 系 — τ を下位区間に制限する方式

WCPG は actor-critic 構成で CVaR_α を最適化するが、α を固定せず「α を追加入力として条件付けた単一ネットワーク」を学習し、訓練中は α をエピソードごとにランダムサンプルする(論文では α を [0.01, 1.0] からランダムに引く。Algorithm 1 では α ~ U(0,1) と記載)[WCPG (2019/11), Section 4 / Algorithm 1]。これにより推論時に α を変えるだけでリスク態度を連続的に切り替えられる。

### 3. 楽観的 τ 選択・上位分位による探索

QR-DQN の学習済み分位から「左側切断分散」(中央値より上側の分位のみで計算する分散。中央値を使うのは統計的頑健性のため)を探索ボーナスとして使い、パラメトリック不確実性の減衰理論に合わせた減衰スケジュール c_t = c√(log t / t) を掛けて intrinsic な(環境由来の)不確実性の影響を抑える手法 [DLTV (2019/05)]。Atari 49 ゲームで QR-DQN 比 483% の平均累積報酬ゲインを報告している [DLTV (2019/05)]。

### 4. 適応的リスク調整 — 学習中に τ 分布・distortion を自動調整

locomotion 環境で、失敗率を最大 1/7 に低減し、未知環境への汎化性能を最大 14% 改善したと報告している [ARA (2021/06)]。

### 5. Thompson sampling 的な return 分布の利用

τ を確率的に引くこと自体を Thompson sampling として定式化した研究は今回確認できなかった(調査限界を参照)。近接する系統として、return 分布(または価値分布のパラメータの事後分布)からサンプルして行動選択する研究を挙げる。

(引用中の [...] は原文の文献参照(Thompson 1933)を省略した箇所。)

## 主要な概念

- [τ配置とIQN比較](../../../concepts/methods/tau_sampling.md)
- [324 Agent個別実装 DQN系列](../../../concepts/framework/324_dqn.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
