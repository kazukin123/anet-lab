---
title: C++ から SQLite にメトリクスを書き出す案
page_kind: サマリ
material: "[[docs/memo/frozen/0140_metrics_sqlite_backend_10rfd.md|0140_metrics_sqlite_backend_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# C++ から SQLite にメトリクスを書き出す案

## 元の材料

[RFD0140　C++ から SQLite にメトリクスを書き出す案](../../../memo/frozen/0140_metrics_sqlite_backend_10rfd.md) blob:7282ba591c6120566922cc401283fd73bd484f9b

## 要点

- SQLite 直接出力は凍結中である。RFD0041 のキャッシュ方式の安定運用が再開条件であり、そこから引き継ぐ前提と C++ 側の未決事項を分けている。
- ファイルを読み取って SQLite キャッシュを作る方式に続き、C++ 側で直接書く段階を検討する。
- 既存の backend は1行の JSON オブジェクトを受け取るため、scalar を効率よく保存する入口と対応づける必要がある。
- データの正本と再生成可能なキャッシュの役割、Viewer が参照する契約を区別する。
- 直接書き込みの速度、同時アクセス、障害後の扱いは方式を決めるための論点として残す。

## 内容

### Context(背景・位置づけ)

DB からダンプする任意成果物へ降格する。
Python 標準 `sqlite3` で即クエリできる(Phase 1 ではキャッシュ構築済み Run に限られる)。
フォルダごと退避」という運用が確立したため、当初最優先だった容量動機は副次に降格した。

### 確定済み前提(041 / ADR 0015 から継承。ここは再議論しない)

WAL 並行読み: Run 中(C++ が writer)でも Viewer は read-only 接続で追従できる。
C++ 側の gz 出力はやらない(確定済み。JSONL 圧縮は Phase 1 の手動 gzip + 透過読みの領分)。

### 暫定方針(方向は固めたが、grill 未実施)

実装 `JsonlBackend`)に `SqliteBackend` を追加する。
(tag_id 解決済み・パース不要の型付き書き込み)を持たない。interface 形状の再設計は未決論点へ。

### スコープ外

Phase 1 の設計・実装内容の再説明(041 を正とする)。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
