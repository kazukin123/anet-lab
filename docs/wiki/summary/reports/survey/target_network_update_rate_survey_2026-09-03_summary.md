---
title: "Survey: target network 更新速度（hard C / soft τ）の設定根拠と調整方法"
page_kind: サマリ
material: "[[reports/survey/target_network_update_rate_survey_2026-09-03.md|target_network_update_rate_survey_2026-09-03]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/target_update.md|Target更新の比較]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# Survey: target network 更新速度（hard C / soft τ）の設定根拠と調整方法

## 元の材料

[Survey: target network 更新速度（hard C / soft τ）の設定根拠と調整方法](../../../../../reports/survey/target_network_update_rate_survey_2026-09-03.md) blob:aa021947955f8663f27478089e8d78f96398cb28

## 要点

- C や τ の値を文献間で比較する前に、単位を確認する必要がある。同じ「10,000」が論文と公式実装で 4 倍違う例がある。
- Nature DQN の付録表は、C の単位を parameter update 数と明示している。
- 一方、公式 Lua 実装では同じ 10,000 が agent step 単位のカウンタに対する剰余で判定されており、`update_freq = 4` と組み合わせると 2,500 parameter updates に相当する（本調査で公式実装 `google-deepmind/dqn` の `NeuralQLearner.lua` を確認）。Double DQN 論文が「τ = 10,000 steps」と書いているのは、論文の表ではなく実装側と整合する。
- τ = 0.001 の出所である DDPG は、その値の選定経緯を書いていない。
- 更新周期の両側（速すぎ／遅すぎ）を定量化した解析が存在する。
- soft と hard を等価とする結果は見つからなかった。

## 内容

### 1. 単位の不一致という前提

文献全体では、frames（Rainbow 32K）、agent steps（Dopamine 8000）、learner gradient steps（Ape-X 2,500 training batches、R2D2 2,500 learner steps、BTR 500 gradient steps）が混在している。BTR は付録に frames / steps / transitions の定義節を設けており、ハイパーパラメータ表で両軸を併記している。

### 2. 主要アルゴリズムの設定値と選定根拠

「実験で選ばれた」と明記しているのは少数で、大半は先行研究からの継承と明記されている。

### 3. 更新速度と収束・バイアスの理論

同論文は target 更新周期の掃引としては本調査で確認した中で最も広く、Atari 6 ゲーム × 5 seeds・約 30,000 GPU 時間で T ∈ {500, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000} を試し、TN 側の最良は T = 5,000 だったと報告している（本調査で確認）。既定値として広く使われる 1,000 / 8,000 は、この掃引では最良ではない。

### 4. 自動・適応調整の手法

この系統で唯一二桁の被引用があるのは t-soft update（Neural Networks 2021、被引用 71）である。評価環境は PyBullet 4 タスク・Mountain Car・CartPole どまりで、Atari 規模で優位を示した例は本調査では見つけられなかった。

### 5. 調整に使われる観測指標

target 更新速度そのものを直接評価する標準指標は確立されていない。 実際に測られているのは、過大評価バイアス、TD 誤差、target-online 乖離、action gap、policy churn、可塑性指標である。

### 6. replay ratio との連動

業界の慣行は「gradient step 軸で固定」である。一次情報が複数ある。

### 7. 批判と留保

同じ著者らが Rainbow で hard / Polyak 双方をチューニングし、周期の調整が性能にほとんど影響しないことを追試している。 その際 Polyak 側では τ = 0.005 が最良だったと報告している（Asadi らの追試）。

### 8. 総合評価

第一に、単位が文献間で統一されていない。 frames / agent steps / parameter updates / gradient steps が混在し、原典 DQN では論文の表と公式実装が 4 倍食い違う。値を比較するには毎回単位を確認する必要がある。

## 主要な概念

- [Target更新の比較](../../../concepts/methods/target_update.md)
- [Replay ratioとlane窓](../../../concepts/methods/replay_ratio_lane_window.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
- [324 Agent個別実装 DQN系列](../../../concepts/framework/324_dqn.md)
