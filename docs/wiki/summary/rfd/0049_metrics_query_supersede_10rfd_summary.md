---
title: metrics query の supersede 機構
page_kind: サマリ
material: "[[docs/memo/done/0049_metrics_query_supersede_10rfd.md|0049_metrics_query_supersede_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-28
---
# metrics query の supersede 機構

## 元の材料

[metrics query の supersede 機構](../../../memo/done/0049_metrics_query_supersede_10rfd.md) blob:18c53fc76c897fc1ba9acf013670a3a7bebeb589

## 要点

- ブラウザが新しい照会を出した後も、古い Metrics Viewer の照会がサーバー側で動き続けていた。
- 同じタブの query channel と連番をヘッダで渡し、新しい要求が古い要求を終了させる。
- 別の channel は相互に取り消さず、同時実行枠だけを共有する。
- ループの中断点と JDBC の Statement.cancel を組み合わせ、キャンセル時の応答を区別する。
- workspace の切替やアプリ終了時も対象とし、古い結果が新しい表示へ戻らないようにする。

## 内容

### Context（背景・目的）

Metrics ViewerでRunやtagを素早く切り替えると、`POST /api/metrics.json`が約5秒後に`503 query_busy`を返し、画面に`Update failed`が出る。

### 受け入れ基準

同一channelで連番1が実行中に2が到着すると、1がcancelされ2が実行される。
真に別channelが枠を使い切った場合だけ`503 query_busy`になる。

### スコープ外

`runs.json`を同時実行枠へ編入すること、およびmetadata用の予約枠。`getRuns()`はRunごとにread connectionを開いて全tagを読む重い処理を無制限に走らせているが、これは枠の外にいるため今回の503には寄与していない。
`RunScanner.listRunId()`のキャッシュ化などquery自体の高速化。permit保持時間を縮める効果はあるが、計測が先である。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
