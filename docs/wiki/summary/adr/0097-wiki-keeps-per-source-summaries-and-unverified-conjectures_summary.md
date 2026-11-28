---
title: wiki に資料ごとのサマリのページと、確かめていない見立てを置く
page_kind: サマリ
material: "[[docs/adr/0097-wiki-keeps-per-source-summaries-and-unverified-conjectures.md|0097-wiki-keeps-per-source-summaries-and-unverified-conjectures]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-28
---
# wiki に資料ごとのサマリのページと、確かめていない見立てを置く

## 元の材料

[ADR 0097](../../../adr/0097-wiki-keeps-per-source-summaries-and-unverified-conjectures.md) blob:3190abfdce0fc8f6e60c6c893707e11ad3cf8456

## 要点

- wiki の各材料に対応するサマリを1ページずつ置く。
- 分かっていることは wiki 外の原本を直接出典とする。
- 横断的知見、未検証の見立て、開発者の疑問は専用の節に区別して記録する。
- サマリは調べものの入口で、ほかの主張の出典には使わない。
- 見立てを別文書でも管理すると育てる場所が二重になるため、wiki に直接置く。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
