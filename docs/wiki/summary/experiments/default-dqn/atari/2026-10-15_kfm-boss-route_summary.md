---
title: kung_fu_master のボス行きの定着 探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md|default-dqn/atari/2026-10-15_kfm-boss-route]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/kung_fu_master.md|kung_fu_master（KFM）]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# kung_fu_master のボス行きの定着 探索履歴

## 元の材料

[kung_fu_master のボス行きの定着 探索履歴](../../../../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md) blob:bdce724731884ed0f16634a1d6e35fd9fbac0d20

## 要点

- 設定を変えた 22 本（中断した 1 本を除く）は、どれもボス行きに入らなかった。
- 報酬クリップを外すと（1/200 倍）ボスに挑む回数は増えたが、勝てずに定着しなかった。
- ライフで区切る（`episodic_life`）と、50M までボスに 1 度も当てなかった。
- UQE の挑み方は lane ごとに分かれなかった。
- 挑み始めるきっかけは撃破ではない。 s13 は 4M の窓で命中 31.9% まで挑んだが、10M までに撃破は 0 だった。
- IMPALA+ViT の Body は、挑む割合は既定と同程度だが、勝ち方を覚えるのも全体の学習も遅かった。
- seed 2 を回し直した 2 本（176.8M と、Munchausen α を 0.3 にした 100M）も、撃破は 0 だった。

## 内容

### 探索ブロック 01: 報酬クリップを外す・ライフで区切る（50M）

クリップ外し seed 2 の 0-10M は 5-10M の窓だけに命中がある（5-10M で命中 1.7%・撃破 0.3%）。 ライフで区切る Run でも、ゲーム完了行は実際のゲーム（全機を失うまで）ごとに 1 行出る。 クリップ外しとライフで区切るを両方入れた Run は、6.4M で中断するまで命中 0 だった。

seed 2 を回し直した 2 本も、撃破は 0 だった。 学習中のゲームの命中率（5M 窓）の最高は、`run_20261007-231245_atari_kung_fu_master` が 5-10M の 4.2%、 `run_20261010-215816_atari_kung_fu_master`（Munchausen α 0.3）が 55-60M の 0.3%。どちらも 100M までに撃破も面クリアも無かった。

### 探索ブロック 02: ボス用セット（30M）

UQE の群の 5-10M は撃破 0.2%、撃破 ÷ 命中 0.02。30M までの撃破は 6 ゲーム、面クリアは 1 ゲームだった。

ボス用セット seed 1、n-step 10、n-step 10 + γ 0.999 は、5-10M に 4 万点前後へ上がって横ばいになった。 train の 5-10M 窓は 38,653 / 34,973 / 40,934 で、学習中のゲーム長は 5,631〜6,506 step。 ボスに当てないまま、1 面の稼ぎに早く固まった。ボス用セット seed 1 の eval2 は 10-15M の 37,505 から 25-30M の 27,067 へ下がった。

### 探索ブロック 03: 広い ε の UQE（30M）

広い ε の UQE はボスに挑まなかった。各 1 本なので、低い ε の UQE との違いを seed の差と区別できない。

### 探索ブロック 04: 既定の設定の 10M × 16 seed

0.5M 窓で、学習中のゲームの命中率のピークと、挑んでいる窓（命中 10% 以上）の撃破 ÷ 命中の最高を取った。 「撃破」「面クリア」は 10M までにそれぞれ 1 回以上起きたゲームの数。

### 探索ブロック 05: batch 1024（30M）

実時間は 4.42〜4.43h（約 1,880 exp/s）で、B256 の 30M の群（4.30〜4.58h）と同じだった。

### 探索ブロック 06: IMPALA+ViT の Body（10M × 8 seed）

挑んだ割合（ピーク 30% 以上が 8 本中 3 本）は既定の 16 本中 4 本と同程度だが、撃破 ÷ 命中の最高は既定の s08（0.22）より低い。

### pending

挑んでいる間に撃破 ÷ 命中が上がるかどうかを何が決めるか。seed 3 と s08 は挑み始めの撃破 ÷ 命中がほぼ同じ（0.24 と 0.22）だったが、seed 3 だけが上がった。

## 主要な概念

- [kung_fu_master（KFM）](../../../../concepts/subjects/kung_fu_master.md)
- [報酬尺度と割引率](../../../../concepts/methods/reward_scale_discount.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
