---
title: IQN導入とQR比較の探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/lunarlander/2026-08-08_iqn.md|default-dqn/lunarlander/2026-08-08_iqn]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/lunarlander.md|LunarLander]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# IQN導入とQR比較の探索履歴

## 元の材料

[IQN導入とQR比較の探索履歴](../../../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md) blob:b9b271e1c475b30fc1608b768938c9153a69608a

## 要点

- IQNはLunarLanderで正常に学習し、learner update、eval、PER更新、finiteなloss/Q/gradを通過した。
- 当初の「QRの方が立ち上がりが速い」という印象は、warmup不一致とseed差を含んでいた。同じQR設定でもseed 1は良好、seed 2は低調であり、単一RunからQR優位とは言えない。
- LunarLanderの序盤には、報酬0付近でホバリングを維持する局所解と、着陸を学んで報酬200付近へ移る段階がある。比較では最終報酬だけでなく、ホバリング離脱時点と離脱後の成長を分ける。
- learner current tauを64から32へ減らすとraw loss、grad norm、clip ratioが落ち着いた。これはIQN lossがcurrent `N`をsumする契約とも整合し、過学習軽減と断定しない。
- QR 32とIQN `N=32, M=32`のmatched 3-seed A/Bでは、`T50`の範囲と同一seed間の順位が重なり、IQNがホバリング離脱を早める傾向は確認できなかった。
- 離脱後は、持続的`T180`到達がQR 1/3対IQN 3/3、1.5M〜2.0Mのpolicy reward平均が144.3対186.0となり、IQNは発見後の定着と終盤性能で有利な可能性を示した。
- 風の位相とランダム地形が直接観測されないことは発見時点のseed差を増やす有力な仮説だが、探索乱数、NN初期値、ReplayBuffer sampling、backend非決定性とは分離できていない。

## 内容

### この文書の読み方

`T50` / `T180`は、`21_eval/04_policy_reward_ema`がそれぞれ50 / 180以上となる状態を3回連続で満たした最初のexp-stepである。

### 探索ブロック01: IQN tracerとfull distribution表示

IQN head、IQNLearner、tau injectionを通した学習が進行した。

### 探索ブロック02: ReplayBuffer満杯時点の相転移

満杯前後で古い初期経験がoverwriteされ、sample経験が現在方策寄りへ切り替わる説明と整合した。

### 探索ブロック03: learner warmupとホバリング離脱

QRでは8,192のRunが180へ到達せず、76,800のseed 1 Runは早期に到達した。

### 探索ブロック05: IQN learner current/target tau数

64/64から32/64へcurrent `N`だけ減らすと、peak loss EMAとpeak grad norm EMAがほぼ半減した。

## 主要な概念

- [LunarLander](../../../../concepts/subjects/lunarlander.md)
- [τ配置とIQN比較](../../../../concepts/methods/tau_sampling.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
