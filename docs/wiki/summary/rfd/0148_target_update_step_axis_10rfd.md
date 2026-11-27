---
title: ターゲット Network の更新を数える step 軸
page_kind: サマリ
material: "[[docs/memo/0148_target_update_step_axis_10rfd.md|0148_target_update_step_axis_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/target_update.md|Target更新の比較]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# ターゲット Network の更新を数える step 軸

## 元の材料

[RFD0148　ターゲット Network の更新を数える step 軸](../../../memo/0148_target_update_step_axis_10rfd.md) blob:f0cb9553e539fa16c50ca5af98476c0a3150433b

## 要点

- リプレイ比率に伴うターゲット更新の遅れは確認されたが、step 軸の換算方式と既存設定の移し方は未決である。
- soft update の τ を Learner 更新ごとに使うと、リプレイ比率を下げたときに同じ経験数あたりの追従が遅れる。
- 実験ではリプレイ比率を半分にしたとき、経験数で見たターゲットの遅れが約2倍になった。
- ハード更新の間隔とソフト更新係数を、学習更新と環境経験のどちらで表すかを比較する。
- 軸を変えても同じ有効な追従速度を再現できる条件を検討する。

## 内容

### Context

`replay_ratio` は本記録で最大のレバーだが、これを動かすと target network の遅れが道連れで動く。
exp_step 軸で見た target EMA の時定数は `(1 / τ) × B / RR` になる（B = `replay_batch_size`）。

### 受け入れ観点

同一 `target_lag_exp_steps` で `replay_ratio` を 4 / 2 / 1 と変えたとき、exp 軸の target 遅れが一致すること（実効 τ をログか dump で確認）。

### 非目標

探索ブロック 23・24 の結論の見直し。勾配 step 軸の比較はこの交絡を受けないため、`replay_ratio` が効くという結論は保たれる。

## 主要な概念

- [Target更新の比較](../../concepts/methods/target_update.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
