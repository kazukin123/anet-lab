---
title: RNDの害の確認とpredictorの周期リセット
page_kind: サマリ
material: "[[docs/experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md|default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/subjects/lunarlander.md|LunarLander]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# RNDの害の確認とpredictorの周期リセット

## 元の材料

[RNDの害の確認とpredictorの周期リセット](../../../../../experiments/default-dqn/lunarlander/2026-11-08_rnd-harm-check-and-predictor-reset.md) blob:e3bd3a754574c342fcbbbefb5be9e4630c813bb5

## 要点

- RNDを混ぜても、評価側は基準と同じ帯だった（害は見えない）。
- bonusは採点5〜10万件で消えた。
- 周期リセットでbonusは2Mの間続いたが、鋸歯の周期信号になった。
- 周期リセットの成績は終盤で同じ帯、利得は無い。
- seed 2が0.78〜0.9Mで崩れた。
- 周期リセットでは全seedで学習側が荒く、局が長い。

## 内容

### 報酬

8,448 exp step（最初の学習の直後）の評価は学習前のネットワークによるもので、同じseedの全Runで同じ値（UQE full −851／−3,479／−3,739、Greedy −939／−4,290／−6,789）。Run全体の平均はこの1点に引かれる（s3は−3,739/312≈−12）。RNDと基準の軌跡は2点目（14,848）から違う。

### bonusの推移

`13_error_step`は0〜25kの1.71e-2／1.57e-2／1.94e-2から1〜2Mの1.08e-4／9.6e-5／1.05e-4へ下がり、一生平均の`11_error_mean`を常に下回る。

### seed 2の崩れ（周期リセット）

`ll_rnd_rs50k_s2`の20k窓の平均。fullは`02_policy_reward`、shareからerror_meanまでは`56_agent_in`、q_max以降は`37_agent_qtd`と`38_agent_loss`。

### 学習側の指標と局の長さ

1.5〜2Mの平均。局の長さはfull `06_policy_ep_steps`で、Run全体／1.5〜2M。1,000 stepの局はfullの1.5〜2Mの数、−1,000未満の局はfullの0.5〜1.5Mの数。

### 同じseedの軌跡の分岐

周期リセットとリセット無し（同じseed）: s3は最初のリセット（50,432）まで一致し、shareは50,432、fullは53,248から違う。s1・s2はshareが13,056、fullが14,848から違い（s2のtargetは最初の点8,448から）、最初のリセットより前に分岐している。

### 解釈の限界と残る問い

どの群も3 seedである。崩れたのは周期リセットの1本で、bonusの山とQの膨張の並びは1本の観察にとどまる。

### Runと実効config

実効configは各Runの`config/config_data.txt`をそのまま複製し、公開できない値が混ざっていないか走査した。完了済みRunの`metrics_cache.db`の全点から集計した。実時間はlogの最初と最後の行の差で、Montezumaの学習Runと並走している。

## 主要な概念

- [内的報酬による探索の比較](../../../../concepts/methods/intrinsic_novelty.md)
- [LunarLander](../../../../concepts/subjects/lunarlander.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
