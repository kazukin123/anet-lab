---
title: GridMaze
page_kind: 題材
aliases:
  - GridMaze
  - 6×10
  - 迷路
description: 6×10 の迷路ではスキップ長を選ぶ方策によって判断頻度は変わったが、目標に届く速さと終盤の評価は同程度だった。
updated: 2026-11-26
---
# GridMaze

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [題材](../../index.md#題材)

## 今の結論

6×10 の迷路ではスキップ長を選ぶ方策によって判断頻度は変わったが、目標に届く速さと終盤の評価は同程度だった。

## 分かっていること

### anet の知見

- 到達可能性の保証を入れる前の5×5では、壁密度0.2、穴密度0.1、ゴール (4,4) で、seed 1〜60 のうち解けたのは35個だった。seed 2 ではゴールに到達できず、スキップ方策を比較できなかった（出典: [GridMaze の到達可能性](../../../memo/done/0100_gridmaze_solvable_maze_10rfd.md)、§1・§2）。

- 6×10 の終盤では ε-greedy の判断率が0.88〜0.89で、UQE は0.64〜0.80だった。この差があっても6本の終盤 EMA は同じだった（出典: [TempoRL のスキップ長の方策（UQE 対 ε-greedy）、GridMaze 6×10 探索履歴](../../../experiments/default-dqn/gridmaze/2026-10-28_temporl-skip-policy-6x10.md)、探索ブロック01。Run: `run_20261028-055804_t103_gm_uqe_s1`・`run_20261028-055956_t103_gm_eps_s1`）。
- GridMaze の RND 性能測定では、warm-up 後の同じ経験数に要した時間を比べ、プロファイル用の走行は集計から除いた。 → 詳しくは [計測の負荷と比較条件](../measurement/diagnostic_cost.md)（出典: [GridMaze の RND 性能測定](../../../experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md)、条件と比較方法・通常比較の結果）。
- 6×10・3 seed で、評価 target の EMA が初めて 0.5 以上になった step は、ε-greedy 282k／311k／313k、UQE 269k／323k／359k だった。ε-greedy の方策は、終盤 20% で平均スキップ長 1.16、j=1 が 93〜95% で、ほぼスキップしない形に落ち着いた（出典: [2026-10-28_temporl-skip-policy-6x10.md](../../../experiments/default-dqn/gridmaze/2026-10-28_temporl-skip-policy-6x10.md)、現時点のまとめ 1・2・探索ブロック 01）。

### 外の事例

なし

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

なし

## 撤回・更新

なし

## 関連

- [intrinsic_novelty](../methods/intrinsic_novelty.md#横断的知見)（通らないセルから再開したときの失敗と RND の横断的知見）。
- [evaluation](../measurement/evaluation.md#横断的知見)（固定の開始からの評価と human starts・Q の近似の範囲の補完）。
- [intrinsic_novelty](../methods/intrinsic_novelty.md#見立て)（点数の方策が自分の遷移を持たないと通らない所の Q の粗さが残るという見立て）。
