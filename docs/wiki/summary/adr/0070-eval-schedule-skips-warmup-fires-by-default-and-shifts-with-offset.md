---
title: eval schedule は既定で学習直後に評価せず、発火の時点を offset でずらす
page_kind: サマリ
material: "[[docs/adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md|0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-26
---
# eval schedule は既定で学習直後に評価せず、発火の時点を offset でずらす

## 元の材料

[ADR 0070](../../../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md) blob:f8e9add0885560e446341d20ad48ca25f85398ff

## 要点

- 評価の候補時点を offset と interval で表す。
- warmup_count は先頭から実行しない回数で、既定は1とする。
- 初期重みの評価を必要とする評価専用 Run や実行体比較では0を指定する。
- 複数評価の重なりは周期をゆがめず offset でずらす。
- この制御は評価スケジュールの Observer が担い、メトリクスの IntervalGate は変更しない。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
