---
title: ReplayBuffer ring折り返し時のframe stack sampleability
page_kind: サマリ
material: "[[docs/memo/done/0050_replay_ring_stack_margin_10rfd.md|0050_replay_ring_stack_margin_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
updated: 2026-11-28
---
# ReplayBuffer ring折り返し時のframe stack sampleability

## 元の材料

[ReplayBuffer ring折り返し時のframe stack sampleability](../../../memo/done/0050_replay_ring_stack_margin_10rfd.md) blob:0fba57f1b9f4b4fa8c028d0391db487766d6cecd

## 要点

- ReplayBuffer のリングが一周すると、遷移が残っていても過去のフレームが上書きされている場合がある。
- 従来の候補列挙では stack_count が使われず、この遷移が抽選され得た。
- n-step など未来側が確定した ready 範囲と、履歴も残った sampleable 範囲を分ける。
- 折り返し後は必要な過去フレーム数を下限に反映し、別世代のフレームの混入を防ぐ。
- サンプリングの利用者が同じ有効範囲を使い、折り返し付近と初期状態を個別に検証する。

## 内容

### 現状の事実（コード確認済み）

`num_envs=1, capacity=8, n_step=2, stack_count=3`、logical time `0..10`をPush。保持されるlogical timeは`3..10`、未来側条件を満たす開始点は`3..8`。

### 契約

ready range（論理区間、env laneごと）: 未上書きであり、N-stepに必要な未来観測が書き込まれ、unroll終端が確定した区間。現行`GetLogicalSampleableRange()`の実態はこれであり、`GetLogicalReadyRange()`へ改名する。
wrap前はmargin 0なので、起動直後のepisode先頭paddingは従来どおり許可される。ring上書きによる履歴喪失はepisode境界ではないため、paddingしてはならない（除外が唯一の正しい扱い）。

### 受け入れ基準

margin近傍にdummy（truncated終端）がある場合の期待集合の固定（dummy除外とmarginの適用順で結果が変わらないこと）。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
