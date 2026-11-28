---
title: reports の下は、記録を作った流れでフォルダに分け、直下に報告を置かない
page_kind: サマリ
material: "[[docs/adr/0099-reports-are-split-into-folders-by-the-flow-that-made-them.md|0099-reports-are-split-into-folders-by-the-flow-that-made-them]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-28
---
# reports の下は、記録を作った流れでフォルダに分け、直下に報告を置かない

## 元の材料

[ADR 0099](../../../adr/0099-reports-are-split-into-folders-by-the-flow-that-made-them.md) blob:cddfe724ed5ca94ed6876ad3be3f850effa361bd

## 要点

- `reports/` 直下には、survey スキルの出力、論文解説、コード読解、anet の道具の調査が並び、作業の中で分かったことまで足すと何の記録か分からなくなる。
- RFD0208 で、survey スキルの出力を `reports/survey/`、paper-details スキルの出力を `reports/paper/`、作業の中で分かったこと（作業ノート）を `reports/notes/` に分け、直下には索引だけを置くことにした。
- `docs/` の下は GitHub Pages で配信する公開文書なので、作業ノートの置き場にしなかった。
- 退けた案は、作業ノートを `docs/` の下の新しい置き場にする形、`reports/` 直下に混ぜて名前と索引で分ける形、出どころ（外から得たか、中で分かったか）で分ける形である。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
