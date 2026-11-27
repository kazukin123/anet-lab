---
title: DropMerge で合法 DROP を選ばず待ち続ける挙動の診断
page_kind: サマリ
material: "[[docs/memo/0163_dropmerge_neet_diagnostics_10rfd.md|0163_dropmerge_neet_diagnostics_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# DropMerge で合法 DROP を選ばず待ち続ける挙動の診断

## 元の材料

[RFD0163　DropMerge で合法 DROP を選ばず待ち続ける挙動の診断](../../../memo/0163_dropmerge_neet_diagnostics_10rfd.md) blob:e3b4185f053f1b691d6798f2e4eb5469e1d85076

## 要点

- 既存の trace による診断を先に試し、必要になった scalar に限って追加するという暫定案である。
- 合法な DROP があるのに NOOP を選び続ける挙動を、Run ごとに同じ条件で診断する。
- 行動全体に対する NOOP と、実際に DROP を選べた局面での NOOP を区別する。
- 個別 lane やエピソードの終了理由で集計し、長い待機が平均に埋もれないようにする。
- 最初に既存の trace で必要な行動・条件を復元できるかを確かめる。

## 内容

### 目的と問題

NEET（合法 DROP があるのに NOOP を選び続ける病理）を、Run のたびに同じ見方で確かめられるようにする。まず trace チャネルで必要な情報が取れるかを確かめ、足りない集計だけをスカラーの指標にする。

### 起草時点の現状

写した結果、QR51 100M の NEET 期は、差が −0.020〜−0.026 から −0.001〜−0.002 に潰れ、NOOP の選択率は 74〜84% だった。基準 IQN 100M（`run_20261001-191232_dmfb_base`）は −0.021〜−0.033 のままで、選択率は 0〜19% だった。

### 提案の骨子

段階 2: trace では足りない集計（学習の exp 軸に載せた NOOP の差と選択率、lane の段ごとのエピソード長と終了理由、終了理由ごとのスコアと果物数）だけを、スカラーの指標にする。

### User Stories

研究者として、NoDropTimeout が序盤の NEET か終盤の NEET かを、Run のたびに見分けたい。

### 決めること

trace の行の量（全 lane・全 step を載せるか、間引くか）。
RFD0144（エピソードを丸ごと捕まえる仕組み）との分担。本 RFD はスカラーと trace の側に限る案を既定にする。

## 主要な概念

- [DropMerge](../../concepts/subjects/dropmerge.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
