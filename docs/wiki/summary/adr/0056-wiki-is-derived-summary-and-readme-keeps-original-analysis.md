---
title: wiki と実験記録 README の役割
page_kind: サマリ
material: "[[docs/adr/0056-wiki-is-derived-summary-and-readme-keeps-original-analysis.md|0056-wiki-is-derived-summary-and-readme-keeps-original-analysis]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# wiki と実験記録 README の役割

## 元の材料

[wiki は正本を持たない派生のまとめとし、実験記録の README は要約の部分だけを畳んで独自の分析は正本として残す](../../../adr/0056-wiki-is-derived-summary-and-readme-keeps-original-analysis.md) blob:b0ae21715df676a3db5ac0f4289f8430e3c9cba2

## 要点

- 実験記録の README には、campaign 文書の要約と、そこだけに残る独自の分析が混在する。
- 各記述を campaign 文書と照合し、重複する要約は wiki にまとめる。
- 独自の分析は README に残し、wiki からその記録を出典として引く。
- 全件を移すと分析の正本を失い、全件を残すと要約と撤回の更新先が重複する。
- 後の ADR 0097・0098 で、資料別サマリ、未検証の見立て、横断的知見と疑問の扱い、概念の分類を改めた。確かめた主張と新しく算出した値は、引き続き wiki の外の正本に戻れるようにする。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤](../../concepts/framework/311_dev_tools.md)
