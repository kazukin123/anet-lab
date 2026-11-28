---
title: metrics query の latest-wins をサーバ側で強制し、その単位を query channel（ブラウザタブ）とする
page_kind: サマリ
material: "[[docs/adr/0023-server-enforced-query-supersede-per-channel.md|0023-server-enforced-query-supersede-per-channel]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
updated: 2026-11-28
---
# metrics query の latest-wins をサーバ側で強制し、その単位を query channel（ブラウザタブ）とする

## 元の材料

[ADR 0023](../../../adr/0023-server-enforced-query-supersede-per-channel.md) blob:eba3321dbe6c9e0d837e4378fd08d088d77f2bd2

## 要点

- 画面で古い通信を中断してもサーバー上の問い合わせが残り、最新の要求が503になることがあった。
- サーバーは同一 channel の新しい要求を受けると古い要求を取り消す。
- 別の channel は互いの要求を取り消さず、同時実行枠だけを共有する。
- 単に枠や待ち時間を増やしても、放棄された処理が枠を使い続ける問題は残る。
- 当初のタブ単位は後の ADR 0090 でタブと問い合わせの種類の組に変更した。

## 内容

### 要求の取り消し

ブラウザーの AbortController は、すでに走っているサーバー処理の終了を保証しない。同じ channel の最新要求をサーバーが特定し、放棄された処理が同時実行枠を占有し続けるのを防ぐ。

### 別 channel と寿命

サーバー全体で1本にすると別タブの表示まで取り消される。workspace が実際に切り替わったときには旧 epoch を取り消し、終了時は Run の後始末より先に実行中の問い合わせを止める。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
