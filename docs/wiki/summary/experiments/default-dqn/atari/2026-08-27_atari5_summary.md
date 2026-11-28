---
title: Atari-5 横断スイープ 探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-08-27_atari5.md|default-dqn/atari/2026-08-27_atari5]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/subjects/name_this_game.md|Name This Game]]"
  - "[[docs/wiki/concepts/subjects/battle_zone.md|Battle Zone]]"
  - "[[docs/wiki/concepts/subjects/double_dunk.md|Double Dunk]]"
  - "[[docs/wiki/concepts/subjects/phoenix.md|Phoenix]]"
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
updated: 2026-11-28
---
# Atari-5 横断スイープ 探索履歴

## 元の材料

[Atari-5 横断スイープ 探索履歴](../../../../../experiments/default-dqn/atari/2026-08-27_atari5.md) blob:382d561b630a64b37f2a74c9317d9cf1c5f2efb9

## 要点

- Atari-5 median 推定 = 55.4% HNS（20M step = 80M frames）。5 ゲームの hns57 単純 median 62.3% / 平均 52.3% の間に入り、極端な外れ値に引きずられていない。
- 20M は明確に短い。 battle_zone / name_this_game / qbert の 3 ゲームが 15-20M 窓でも上昇中で、頭打ちの気配がない。特に qbert は 5M ごとにほぼ倍増している（408 → 1,871 → 3,071 → 4,451）。55.4% は到達値ではなく途中経過である。
- name_this_game で初めて人間を超えた（12,316 対 人間基準 8,049 = hns57 174.1%）。酸素補給という資源管理が要る題材で、20M で獲得している。
- double_dunk はランダム以下（−20.3 対 ランダム −18.6、hns57 −77.3%）。20M 通して −22〜−20 の帯から出ない。この 1 ゲームが推定を約 21 ポイント削っている（人間水準なら合計 φ が 1.751 → 1.887、推定 55.4% → 76.1%）。
- phoenix は生存戦略に落ちて劣化し、`bad allocation` で落ちた（12.28M / 全 20M）。エピソード長が 656 → 上限 27,000 に張り付き、スコアは 4,998（5-10M）でピークを打って 4,633 へ低下。throughput も 6,916 → 3,761（−46%）。5 ゲームで phoenix だけがこの挙動を示す。

## 内容

### 探索ブロック 01: 初回スイープ（20M / apex 単群）

27,000 = `max_episode_frames` 108,000 ÷ frameskip 4 で、打ち切り上限そのものである。eval 側も `51_eval1/20_game_len_mean` が 599 → 22,959 → 27,000 と張り付いた。

Breakout は易しい側の題材だった。 旧バンドルの Breakout は 100M で hns57 1685% に達したが、本スイープの 5 ゲームは 20M で −77% 〜 174% に散らばる。バンド幅も 2.2（double_dunk）〜 34,828（battle_zone）と 4 桁違い、Breakout 1 本で見ていた「うまくいっている感」は題材に依存していた。

### pending

予算 20M の不足。3 ゲーム（battle_zone / name_this_game / qbert）が 15-20M 窓でも上昇中で、飽和点が未観測。50M（BTR の 200M frames 相当）で頭打ちするかも未確認。

## 主要な概念

- [Atari横断](../../../../concepts/subjects/atari.md)
- [Name This Game](../../../../concepts/subjects/name_this_game.md)
- [Battle Zone](../../../../concepts/subjects/battle_zone.md)
- [Double Dunk](../../../../concepts/subjects/double_dunk.md)
- [Phoenix](../../../../concepts/subjects/phoenix.md)
- [Qbert](../../../../concepts/subjects/qbert.md)
