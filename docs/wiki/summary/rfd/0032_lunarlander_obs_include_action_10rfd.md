---
title: LunarLanderEnv obs_include_action（直前 action の観測化・（暫定）
page_kind: サマリ
material: "[[docs/memo/done/0032_lunarlander_obs_include_action_10rfd.md|0032_lunarlander_obs_include_action_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/lunarlander.md|LunarLander]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
updated: 2026-11-26
---
# LunarLanderEnv obs_include_action（直前 action の観測化・（暫定）

## 元の材料

[LunarLanderEnv obs_include_action（直前 action の観測化・（暫定）](../../../memo/done/0032_lunarlander_obs_include_action_10rfd.md) blob:a935299f7924719d69aa8a6715bebe044a3649d2

## 要点

- LunarLander の観測には風が直接現れず、速度差分にはエンジンの推力も含まれる。
- 直前の行動を観測に加えると、推力と風を区別する手がかりになるかを検証できる。
- obs_include_action で入力を切り替え、行動を含めた観測幅を12、従来の幅を8とする。
- 行動履歴を別経路で再構成せず、フレームごとの観測として扱う。
- 状態表現の変更は暫定の実験条件として扱い、標準の観測と比較する。

## 内容

### 行動と観測の対応

速度変化から風を推定するには、その時どのエンジンを動かしたかが手がかりになる。学習中は探索によって行動が変わるため、複数の観測を積んでも行動が一意に分かるとは限らない。`obs_include_action=true` のとき、従来の8特徴に NOOP・左・主・右の4要素の one-hot を追加する。

リセット直後はまだ行動していないので4要素はすべて0。`Step(a)` が返す観測では実行した `a` だけを1にする。既定の無効時には観測幅・値・乱数消費を変えない。

## 主要な概念

- [LunarLander](../../concepts/subjects/lunarlander.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
