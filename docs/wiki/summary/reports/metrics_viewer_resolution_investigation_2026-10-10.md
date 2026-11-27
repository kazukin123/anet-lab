---
title: "MetricsViewer 複数 Run の表示粒度: 再現調査と対応方針"
page_kind: サマリ
material: "[[reports/metrics_viewer_resolution_investigation_2026-10-10.md|metrics_viewer_resolution_investigation_2026-10-10]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# MetricsViewer 複数 Run の表示粒度: 再現調査と対応方針

## 元の材料

[MetricsViewer 複数 Run の表示粒度: 再現調査と対応方針](../../../../reports/metrics_viewer_resolution_investigation_2026-10-10.md) blob:eed8d15a516a892f0f8810519fed2aa2c8d12dfb

## 要点

- 全タグをまとめて取得すると、request 全体の 500,000 vertex 予算を多数の系列で分ける。学習報酬の LOD が L2（256点単位）から L3（4,096点単位）へ跳び、描画点数が約2,500点から約160点へ減る。
- cache は範囲・zoom・generation・metadata を検査するが、比較する Run 集合や表示粒度を検査しない。操作順によって同一 graph に L2 と L3 が残る。metadata 不変の Reload はこれを修正しない。
- 同じ request と同じ pointBudget でも、level は Run ごとに独立決定する。記録点数が閾値を跨ぐと粒度が16倍異なる。
- 直接の引き金は memory pressure の検出ではなく、点予算と射影・cache の契約である。
- DOM、Plotly 描画、UI 操作は stub 化した。browser 上でクリックを通した E2E、heap 測定、OOM の有無の検証は未実施。
- 今回確認した約0.92 MBは、4 Run × 3タグの転送JSONサイズであり、browser全体のmemory測定ではない。

## 内容

### 結論

症状を、現在起動中の Viewer の API と配信中の実際の client code を使って再現した。

### 調査条件と確認範囲

配信中 metrics-viewer.js と repository のファイルを LF 正規化後に比較し、一致を確認。

### 再現結果

同じ metadata で refresh=true（Reload の再取得判定）を実行。再取得0件。不一致が残る。

### 原因箇所

現行設計正本: docs/design/210_metrics_viewer.jp.md §2.6〜2.8、§9.2。 §2.8の「同じ request なら解像度がそろう」という説明は、現在の独立level選択とcache再利用では保証されない。 実装時にはこの説明を新しい保証へ改める必要がある。

### 推奨方針（未実装・未合意）

粒度は「何点を1つに畳むか」ではなく「横軸のどの step 区間を1つに畳むか」で揃える。 同じ tag、同じ viewport、同じ graph 幅なら、比較 Run 全てに共通の step 境界を使う。

## 主要な概念

- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
