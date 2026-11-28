---
title: wiki の知見のページに、分野で分ける種類「ハーネス」を足す
page_kind: サマリ
material: "[[docs/adr/0100-wiki-adds-a-harness-kind-split-by-domain.md|0100-wiki-adds-a-harness-kind-split-by-domain]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-28
---
# wiki の知見のページに、分野で分ける種類「ハーネス」を足す

## 元の材料

[ADR 0100](../../../adr/0100-wiki-adds-a-harness-kind-split-by-domain.md) blob:f37d91a77c623b9c87c67043ddaa95192df03921

## 要点

- ADR 0098 は知見のページを答える問いで題材・手法・測り方・現象に分けたが、ハーネス（AI のエージェントと、その作業の決まりと道具）の知見は、RL の知見と同じ種類に並ぶと読み分けにくい。
- RFD0208 で、主語に AI のエージェント、ハーネスのファイル、ほかのソフトウェア（このリポジトリの `src/` と `apps/` が組み込むか起動するもののほかの、すべてのソフトウェア）のどれかが入る知見は、答える問いを問わず `concepts/harness/` に置く、分野で分ける種類を 1 つ足した。
- RL の実験にも触れる知見も、ハーネスを主にする。答える問いで分けた 4 種と、種類をフォルダで持つことは保つ。
- 退けた案は、今の 4 種で分けて目録に「ハーネス」の小見出しを置く形（ページのフォルダでは混ざる）と、311 のトピック別まとめだけに載せる形（効き目と実測を書く節が無い）である。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
