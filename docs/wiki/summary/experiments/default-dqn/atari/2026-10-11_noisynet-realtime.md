---
title: NoisyNet の実時間優先プロファイルへの投入（Atari） 探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md|default-dqn/atari/2026-10-11_noisynet-realtime]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
  - "[[docs/wiki/concepts/subjects/battle_zone.md|Battle Zone]]"
  - "[[docs/wiki/concepts/subjects/double_dunk.md|Double Dunk]]"
  - "[[docs/wiki/concepts/subjects/kung_fu_master.md|kung_fu_master（KFM）]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
updated: 2026-11-26
---
# NoisyNet の実時間優先プロファイルへの投入（Atari） 探索履歴

## 元の材料

[NoisyNet の実時間優先プロファイルへの投入（Atari） 探索履歴](../../../../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md) blob:9262c47cdaf826893bf6408b1cf12bcdb07defe1

## 要点

- Noisy だけ（train の ε 0）の 7 本は、どのゲームでも基準のどの seed も下回った。
- Breakout で下がったのは ε ラダーを外したぶんで、NoisyNet 自体は ±0 だった。
- battle_zone と double_dunk では、Noisy + ラダーで成績は戻ったが、立ち上がりが 10〜20M 遅れた。
- double_dunk の Noisy だけは、50M まで相転移が起きなかった。
- ノイズで行動が変わる割合は、学習の初めに大きく、数 M のうちに数 % まで下がった。
- kung_fu_master の Noisy だけは、序盤は速いがそのまま横ばいになり、面を 1 度も抜けなかった。

## 内容

### 探索ブロック 01: Noisy だけ（ε 0）× 7 ゲーム

kung_fu_master の基準 seed 3 は、40-50M が 76,604、25-50M が 61,722。

kung_fu_master は序盤だけ速い。 eval2 5-10M は 31,447（基準 3 seed は 12,141 / 21,347 / 19,497）、学習中のスコアは 5-10M から 45-50M まで 42,542〜44,248 で横ばい。 ε=0 断面は 5.3M から 45.7M まで 38,666〜44,468。学習中のゲームで、ボスへの命中・撃破・面クリアは 50M まで 0 本だった。

### 探索ブロック 02: ε ラダーの有無と NoisyNet の有無を分ける

ラダーを外すと、NoisyNet の有無によらず eval2 25-50M が約 70〜85 点下がった。NoisyNet の有無の差は、ラダーありで +15.3、ラダーなしで +0.8。

### 探索ブロック 03: NoisyNet の probe

double_dunk の Noisy だけは、5M 以降ノイズが行動を 1 度も変えなかった（5-15M・15-25M・40-50M とも 0.0%）。‖σ‖/‖μ‖ は 0.81 で、7 本の中で最も下がっていない。 Noisy + ラダーの double_dunk は 40-50M も 9.2% が残った。

## 主要な概念

- [NoisyNetの観測](../../../../concepts/methods/noisynet.md)
- [Breakout](../../../../concepts/subjects/breakout.md)
- [Battle Zone](../../../../concepts/subjects/battle_zone.md)
- [Double Dunk](../../../../concepts/subjects/double_dunk.md)
- [kung_fu_master（KFM）](../../../../concepts/subjects/kung_fu_master.md)
- [Atari横断](../../../../concepts/subjects/atari.md)
