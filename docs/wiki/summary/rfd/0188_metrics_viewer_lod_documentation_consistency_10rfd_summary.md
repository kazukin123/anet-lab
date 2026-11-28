---
title: Metrics Viewer の LOD 集約数の説明
page_kind: サマリ
material: "[[docs/memo/0188_metrics_viewer_lod_documentation_consistency_10rfd.md|0188_metrics_viewer_lod_documentation_consistency_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-28
---
# Metrics Viewer の LOD 集約数の説明

## 元の材料

[RFD0188　Metrics Viewer の LOD 集約数の説明](../../../memo/0188_metrics_viewer_lod_documentation_consistency_10rfd.md) blob:a94eabe4851c1cc969a345d8d0724e36d04f62be

## 要点

- Metrics Viewer の実装は同じレベルの子が4件そろったときに上位バケットを一つ作る。
- 必要な件数がそろう前には上位バケットを書き出さない。
- 設計文書やユーザーガイドに残る異なる集約数の説明を、実装と照合して特定する。
- 各レベルが覆う元のデータ点の数と、親を生成する子の数を区別して記述する。
- 挙動は維持し、文書と例の数字が同じ規則から導けることを確かめる。

## 内容

### 問題

LodIngestWriter.java:109 の `appendChild` は、同じ level の子が `LOD_FACTOR` 件そろうと親を合成して書く。そろうまでは書かない。
ADR 0052 が、保存 LOD を factor16 から factor4 に改めた。ADR 0052 は「ADR 0015のfactor16のみを更新する」と書く。
設計210 jp の 170・184・262・726 行と、wiki の Metrics Viewer のトピック別まとめ「概要・責務」は、すでに 4 で書いている。設計210 jp 163 行の「連続する16点」は序数で数えた幅の例（4² = 16）で、食い違いではない。

### ゴールと非ゴール

ADR 0015 から、factor を改めた ADR 0052 へたどれるようにする。
`LOD_FACTOR` を 16 に戻して文書に合わせる案 — 退ける。ADR 0052 が測定に基づいて 4 を選んだ。実装と ADR が正であり、本当の制約である。
RFD0110 の pin 実装を変える案 — 範囲外。食い違いは pin 対応の前からあり、関係しない。制約ではなく別の関心である。

### 確定事項

正とする集約数は 4 とする。同じ level の子が 4 件そろうと親 1 件を書き、そろわない末尾は行にしない。
wiki の表示粒度のページの「分かっていないこと」から、本件の問いと余分な「なし」を消し、節の本文を、同じページの「撤回・更新」と同じ「なし。」の 1 行にする。4 子という事実は、このページに足さない。wiki の作業の記録に、材料を本 RFD とする `writeback` の行を 1 行足す。設計160 の直しの取り込みは、点検（anet-lint-wiki）の取り込み漏れの検出に任せる。

### 受入

日英の全文の突き合わせの結果（訳元と合っていなかった英語版の段落と直した内容。無ければ無い旨）が、索引 issue の対応内容の「確かめ」に書かれている。上の項目を補う記録である。
設計160 en の、日本語版 270 行に当たる段落が、View data を作る時機（timer が断面を取得した後の最初の Train event と、前回から 200ms 経った Train event だけ）と、Train View を隠している間と FPS が 0 の間も 200ms ごとの作成が続くことを書いている。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
