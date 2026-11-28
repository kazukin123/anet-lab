---
title: wiki の概念のページを、答える問いで分け、種類をフォルダで持つ
page_kind: サマリ
material: "[[docs/adr/0098-concept-pages-are-split-by-the-question-they-answer-and-kinds-live-in-folders.md|0098-concept-pages-are-split-by-the-question-they-answer-and-kinds-live-in-folders]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-28
---
# wiki の概念のページを、答える問いで分け、種類をフォルダで持つ

## 元の材料

[ADR 0098](../../../adr/0098-concept-pages-are-split-by-the-question-they-answer-and-kinds-live-in-folders.md) blob:a87fb4e7fba63545a34e7da83bd1dc44ff0a755c

## 要点

- 以前は題材に当たらない知見を機構にまとめ、現象や測り方が見つけにくかった。
- 知見の種類は、どこで・何を変えたら・何が起き・どう測ったかという問いで決める。
- その問いに対応して題材、手法、現象、測り方に分ける。
- コードの区分を扱うトピック別まとめと合わせ、種類をフォルダで表す。
- 再分類による移動に備え、外から wiki を指すリンクも検査する。
- 先頭の注記（2026-11-28）: ADR 0100（RFD0208）で、分野で分ける種類「ハーネス」を足した。答える問いで題材・手法・測り方・現象に分けることと、種類をフォルダで持つことは残る。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
