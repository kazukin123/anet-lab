---
title: "Survey: 行動を続ける長さ（スキップ）の学習を、補助タスクやパラメータのノイズと組む事例"
page_kind: サマリ
material: "[[reports/skip_auxiliary_task_noise_survey_2026-11-27.md|skip_auxiliary_task_noise_survey_2026-11-27]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
updated: 2026-11-27
---
# Survey: 行動を続ける長さ（スキップ）の学習を、補助タスクやパラメータのノイズと組む事例

## 元の材料

[Survey: 行動を続ける長さ（スキップ）の学習を、補助タスクやパラメータのノイズと組む事例](../../../../reports/skip_auxiliary_task_noise_survey_2026-11-27.md) blob:9b522018e603c23d6ef98fad6463af008b3e0c34

## 要点

- スキップ長（行動を続ける長さ）の価値を、行動に使わず補助タスクとしてだけ学ぶ事例と、スキップ長の選択に NoisyNet などのパラメータのノイズを使う事例は、2026-11-27 の調査では見つからなかった。TempoRL を引用する 45 本の抄録にも無かった [S2 TempoRL-citations (2026/11)]。
- 近いものは、割引率の違う複数の Q を補助として学び、行動は固定の γ で選ぶ多地平の補助タスク（Fedus ら）で、Rainbow をしばしば上回った [Fedus MultiHorizon (2019/02), Abstract, Section 9]。
- 多地平の補助は n-step とはよく組み合わさり、PER とは 10 ゲーム中 4 ゲームで成績を下げた。著者は全 Head の TD 誤差を平均して優先度にしたためと説明する [Fedus MultiHorizon (2019/02), Section 9, Figure 12]。
- 表現の解析では、複数の割引率の価値の予測は平均の割引率の単一の予測と同じ表現に向かい、追加の仕組みが無ければ単一の課題より役立たない [Lyle AuxDynamics (2021/02), Section 4.3]。
- 全部の長さの価値を同時に学ぶ PerDQN は、その価値で行動と長さを選び、探索は ε-greedy である [Sabbioni AllPersistence (2022/11), Section 5]。
- NoisyNet と組んだ近い例は、NoisyNet の Rainbow の上に長さを zeta 分布から引く εz-greedy を重ねた形（長さは学ばない）と、NoisyNet で行動を選び繰り返すかを状態の新しさで決める SNAP（連続制御）だった [Dabney EzGreedy (2020/06), Appendix B.3; Hu SNAP (2024/09), Section 6.3]。
- TAAC は、繰り返すかを Q の Boltzmann 形の分布で選び、温度を entropy の項で自動調整し、ε を使わない（連続制御）[Yu TAAC (2021/04), Section 4.3]。

## 内容

### 1. 長さの価値を補助タスクとしてだけ学ぶ事例

TempoRL の被引用で繰り返しを扱うものは、どれも学んだ価値か別の規則（SNAP は状態の新しさ）で行動か長さを選んでいた [S2 TempoRL-citations (2026/11)]。多地平の補助の Atari の設定は γ が 10 個、最大は 0.99 で、Dopamine の Rainbow（sticky actions 0.25）、各 3 seed である。双曲の割引で行動する Hyper-Rainbow と Multi-Rainbow に有意な差は無く（19 ゲーム）、著者は改善を補助タスクによるものとした。Multi-C51 は 10 ゲーム中 9 ゲームで C51 を上回った。最大の γ で優先度を付ける案は付録 E に予備の結果として置かれ、表現を直接調べた実験は無い [Fedus MultiHorizon (2019/02), Section 6.1, Section 9, Figure 10, Figure 12, Appendix E]。Lyle らの Atari（Double DQN）の実験はランダムな累積量とアンサンブルを補助にしたもので、Montezuma's Revenge での強さを挙げるが本文に数値は無く、多数の割引率は Atari で試していない [Lyle AuxDynamics (2021/02), Section 5.2]。ランダムに選んだ行動の列に条件付けた多段の予測を補助にすると手作りの補助と同じくらいの表現が得られた（抄録で確かめた）[Zheng RandomGVF (2021/02), Abstract]。PerDQN は Atari の 6 ゲーム（各 5 Run）のうち 5 ゲームで DQN と TempoRL より学習曲線が速く、Seaquest では DQN が上回った [Sabbioni AllPersistence (2022/11), Section 7]。Γ-nets は補助タスクとしての利用を次の課題に挙げるだけで、実験は無い [Sherstan GammaNets (2019/11), Section 5]。

### 2. スキップ長の選択をパラメータのノイズで探索する事例

εz-greedy の長さの分布は z(n) ∝ n^(−μ)（既定は μ = 2）で、Atari-57 の人間で正規化した得点の中央値は R2D2（30B フレーム）で 16.44 → 17.77、Rainbow（200M フレーム）で 2.03 → 2.30 だった。NoisyNet とパラメータのノイズは探索の方法として挙げるだけで、直接は比べていない [Dabney EzGreedy (2020/06), Section 3, Section 4.2, Table 1]。SNAP は画像入力の DeepMind Control（DrQ-v2、5 seed）で、NoisyNet の場合の数値は無い [Hu SNAP (2024/09), Section 6.3]。TAAC は 14 の連続制御の課題（3 seed）で持続する探索が生じたとする [Yu TAAC (2021/04), Section 5.6.1]。NoisyNet の原典は、DQN ではノイズを行動の前ごとに引き直し、target の網は別のノイズを使うとし、行動の繰り返しには触れていない [Fortunato NoisyNet (2017/06), Section 3.1]。

### 3. 批判と留保・調査の限界

Fedus らの論文は arXiv の版で、掲載先は確かめられなかった。Lyle らの結論は線形・アンサンブルの解析の前提の下のもので、Fedus らの Atari の結果と同じ条件ではない。SNAP と TAAC は連続制御で、離散の行動とスキップ長の Q を持つ DQN 系の形ではない。被引用の一覧の数本は抄録が無く、題だけで判断した。

## 主要な概念

- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [NoisyNetの観測](../../concepts/methods/noisynet.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [Montezuma's Revenge](../../concepts/subjects/montezuma_revenge.md)
