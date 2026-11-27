---
title: AI向けRun解析CLI `inspect_run.py`
page_kind: サマリ
material: "[[docs/memo/done/0054_inspect_run_10rfd.md|0054_inspect_run_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
updated: 2026-11-26
---
# AI向けRun解析CLI `inspect_run.py`

## 元の材料

[AI向けRun解析CLI `inspect_run.py`](../../../memo/done/0054_inspect_run_10rfd.md) blob:1089863dbdf448dd511396d3695869deeb177955

## 要点

- Run の発見、設定確認、タグ検索、指標集計を機械可読な CLI から実行する。
- runs、tags、config、metrics を分け、各コマンドが必要なデータだけを開く。
- Run 名は現行 workspace から探し、別配置の Run は明示したディレクトリで指定する。
- 複数の Run・タグ・区間をまとめ、元ファイルを読む必要があるときも Run ごとの走査を抑える。
- JSON と Markdown は同じ結果モデルから出力し、集計内容の違いを防ぐ。

## 内容

### Problem Statement

`metrics_cache.db` はMetricsマスタから再生成可能な派生cacheであり、完全に追随している場合だけ解析入力として利用できること。
configには「どのRunnerのcountsか」を表現するtokenが存在しないため、この区別は `@event` と `$runner_scope` の組から間接的に導くほかない。解析側がこの導出をPythonで再実装すると、C++側の解決規則と二重管理になり、仕様変更のたびに乖離する。

### CLI契約

current working directory基準または絶対pathとして既存directoryへ解決できる場合は、そのdirectoryをRun directoryとする。

### step座標系とmetrics定義

削除条件: 現用のRun作業セットが全て `metrics.defs` を持つようになった時点でfallback実装を削除する。
step軸が明示されない場合は、`event == train` なら `train_step`、`learn` と `episode_end` なら `exp_step` とする。

### Metrics sourceとcache選択

条件を満たさないcacheは `absent`、`invalid`、`partial`、`stale`、`error` のいずれかとして理由付きで記録し、Metricsマスタへfallbackする。

### metric抽出契約

同一tag内でstepが逆行した場合、そのtagを `quarantined` とし、逆行前の有効prefixだけを公開する。他tagの走査は継続する。
glob meta characterは `*` と `?` だけとし、`[` と `]` はリテラル文字として照合する。実効configとmetric tagはどちらも `[tag]` 記法を含むため、`fnmatch` のcharacter class解釈を使わない。

### 実効設定

実効設定の正本は `config/config_data.txt` とする。各非空行を最初の `=` で分割し、左辺をtrimしたflat key、右辺をtrimした文字列値として読む。値を数値や真偽値へ暗黙変換しない。
返す各keyについて、いずれかのmodule dumpに同じkeyが存在すれば `effective: true` とする。

### result model

JSON rootは共通のenvelopeを持ち、`subcommand` によって内容が決まる。field名はschema v2のpublic contractとする。

### 実装範囲

既存 `metrics_source.py` をsource選択・openに再利用する。単一CLIのために新規package、component file、共通frameworkを作らない。第三者依存を追加せずPython標準libraryだけで実装する。
C++側とPython側は独立に実装できる。Python側は `metrics.defs` が無くてもconfig fallbackで動作する。

### スコープ外

Metrics Viewer UI、HTTP API、Java実装、SQLite schemaの変更。`metrics.defs` は既存の `type: "json"` を使うためこれらの変更を必要としない。
実Run依存test、網羅的fuzz、性能benchmark。ただしstreaming・1 pass・128点上限の契約はfocused testで確認する。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
