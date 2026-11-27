---
title: 遷移ごとの経過時間を使う SMDP 割引案
page_kind: サマリ
material: "[[docs/memo/0167_smdp_transition_duration_10rfd.md|0167_smdp_transition_duration_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# 遷移ごとの経過時間を使う SMDP 割引案

## 元の材料

[RFD0167　遷移ごとの経過時間を使う SMDP 割引案](../../../memo/0167_smdp_transition_duration_10rfd.md) blob:2fce7d135f95b4082e995dfb12046e3294735f36

## 要点

- 遷移の時間幅を割引へ反映する SMDP 案であり、時間の単位や既存設定との整合は再グリルの対象である。
- DropMerge では待機の費用と長期計画の見通しを同じ割引率が兼ねるため、行動ごとの所要時間を明示する案を扱う。
- 環境が各遷移の非負の経過時間を返し、既定値を1にする。
- Replay は経過時間を保存し、n-step の報酬と bootstrap に各遷移の割引の積を使う。
- TD・QR・IQN や Munchausen のターゲットなどで同じ時間の意味を使う。
- すべての経過時間が1のときは従来のメトリクスと一致することを確かめる。

## 内容

### 目的と問題

各遷移に「経過時間 τ」を持たせ、学習の目標値の割引を γ^n から γ^τ の積に変える。行動ごとにかかる時間が違う問題の標準的な枠組みである semi-MDPに対応させる。既定は τ = 1 で、今の挙動と変わらない。

### 提案の骨子

γ は Agent が持つ「時間 1 単位あたりの割引」のまま残し、時間の意味は Env が持つ。γ の定義が Env と Agent の 2 か所に分かれないようにするため、Env には割引そのものではなく経過時間を返させる。

### User Stories

研究者として、待つことの値段（明示の罰）と、長い計画の見通し（γ）を別々に決めたい。NEET を抑えながら、ダブルスイカのような長い計画を学ばせるため。
研究者として、既定の設定では今の学習と同じ結果になることを確かめてから、DropMerge の時間の数え方だけを変えて比べたい。

### 決めること

待つ罰の大きさ、γ の値（調整の範囲なので本 RFD では決めない）。
比較の物差し。待つ罰を入れると return が変わるので、085 のスコア軸で比べる。
Rainbow など、DQN 系以外の Agent の扱い（τ を無視してよいか、対応させるか）。

### 対象外

続ける長さを学習すること（TempoRL、0098_temporl_skip_policy）。

## 主要な概念

- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
