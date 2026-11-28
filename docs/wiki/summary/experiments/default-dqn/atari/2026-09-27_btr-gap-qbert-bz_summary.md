---
title: BTR との差の切り分け（qbert / battle_zone） 探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md|default-dqn/atari/2026-09-27_btr-gap-qbert-bz]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
  - "[[docs/wiki/concepts/subjects/battle_zone.md|Battle Zone]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
updated: 2026-11-28
---
# BTR との差の切り分け（qbert / battle_zone） 探索履歴

## 元の材料

[BTR との差の切り分け（qbert / battle_zone） 探索履歴](../../../../../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md) blob:030470c89a455154914d5a2b8b81799b876cad1c

## 要点

- BTR 型の ε の経過（3 段）は ε 群と同等だった。
- 6〜25M の全 lane ε 0.01 は、ε 群のラダーより第 7〜8 面の経験が少ない。
- qbert の第 8 面突破率は、64 env + replay 1M（BTR と同じ env 数・容量、lane 窓 16,384 据え置き）の Run で高く出やすい。
- 第 8 面の初突破の時期は seed で大きくぶれ、早く出ても広がらない。
- env64 の上に載せた UQE と ViT は、env64 の seed の幅の中だった。
- 得点ループに入った Run では、29,000 点以上で数えた突破率は第 8 面の突破を表さない。

## 内容

### 探索ブロック 01: BTR の ε の経過を 3 段で再現する（qbert）

`round_clear` = 7（第 8 面で終わった）のゲームは、20-21M で 45 本 対 116 本、21-22M で 82 本 対 115 本。 3 段目に入った 25M 以降は逆に 3 段が多く、26-35M の第 8 面到達率は 48〜67%（ε 群 32〜50%）。 lane の ε 帯ごとの到達率も、ε 0.001 未満で 82.8%（ε 群 66.5%）、0.001〜0.003 で 76.0%（61.4%）と高かった。突破率は 2.1% 対 1.3% で、ほぼ同じ。

段の境目に成績の段差は無かった。 ε=0.01 評価は、1 段目の終盤（5.45〜5.95M）4,035〜4,062、2 段目の warmup 中 3,878 / 3,875、学習再開後 4,055（通算 6.7M）。

### 探索ブロック 02: battle_zone の ε ラダー群

2026-09-25 探索ブロック 02 で、battle_zone の train の得点は lane の ε で大きく変わった（25M 以降、ε 0.01〜0.016 の lane 平均 97,023、ε 0.25〜0.4 で 35,340）。 qbert で効いたラダー 1e-4〜0.05 を当てる。

序盤は遅く、30M 前後で追いつき、最良は +2.5%。train の 40-50M 平均は 120,514（旧ラダー 71,888）だが、ε の分布が違うので方策の比較には使わない。

### 探索ブロック 03: env 数・replay 容量・行動ネットの同期（qbert）

64 env / 1M は lane step の年齢が 128 env / 2M と同じで、更新数の年齢が 128 env / 1M と同じ。 PER の選択性（`33_per_selectivity` 約 1.12〜1.20）と、一度も引かれない遷移の割合（約 12.8%）は 64 env / 1M と 128 env / 2M で同じだった。

### 探索ブロック 04: env64 の上の UQE と ViT（qbert）

UQE は 45-50M の突破率・49-50M・ε=0.01 評価が 3 本の中で一番上だが、ε=0 断面は seed 1 と seed 2 の間にあり、env64 の seed の幅を出ない。

### 探索ブロック 05: battle_zone の 64 env / 1M

seed 1 の 25.5M（146,203）は、サンプル効率優先の 50M の Run の中で battle_zone の最高値（2026-09-25 の最良は 115,533）。ただし、その後は下がり続け、45.7M では 97,236 だった。 seed 2 は 128 env / 2M と同じ推移で、40-50M の ε=0.01 評価は 4 本の中で最も低い。

### 探索ブロック 06: seed 4 の得点ループと、第 8 面突破の数え方（qbert）

ε=0 断面は、35.6M 以降ほぼ全部が 2 面目で止まった。 30.5M は 100 本中 81 本が `round_clear` = 7 で平均 28,132 だった。 35.6M・40.6M・45.7M は `round_clear` = 1 がそれぞれ 87 / 99 / 95 本で、その平均は 67,029〜67,050 点。`round_clear` = 1 のゲームはどれも 26,993 step 以上（上限まで）続いた。 断面の平均は 69,468 / 73,400 / 66,820 で、突破（29,000 点以上かつ `round_clear` ≥ 8）は 3 断面とも 0 本。

### pending

64 env / 1M で qbert の第 8 面突破率が上がる仕組みは測れていない。lane step の年齢と更新数の年齢の両方がそろった唯一の組み合わせだが、seed 3 は抜けなかった。

## 主要な概念

- [Qbert](../../../../concepts/subjects/qbert.md)
- [Battle Zone](../../../../concepts/subjects/battle_zone.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [NoisyNetの観測](../../../../concepts/methods/noisynet.md)
