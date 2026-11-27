---
title: PRD の 9xx の番号を、起票の日の順に 900 未満の連番へ振り直し、9xx を再び使える番号に戻す
page_kind: サマリ
material: "[[docs/adr/0077-prd-9xx-renumbered-by-creation-date-and-9xx-reusable.md|0077-prd-9xx-renumbered-by-creation-date-and-9xx-reusable]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# PRD の 9xx の番号を、起票の日の順に 900 未満の連番へ振り直し、9xx を再び使える番号に戻す

## 元の材料

[ADR 0077](../../../adr/0077-prd-9xx-renumbered-by-creation-date-and-9xx-reusable.md) blob:2105e3cc5f5270fee3b9976cbf0fa121ada67fe5

## 要点

- 起票順が番号の並びに反映されるよう旧9xxの文書57本を900未満の連番に振り直す。
- 振り直しより前の利用は例外として、900台を新規の番号として再利用できる。
- 各文書の題の直下に旧番号と振り直しの日を残す。
- 道具はその日以前のコミットだけ旧番号で照合する。
- 別ファイルの対応表や恒久的な予約番号帯は作らない。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
