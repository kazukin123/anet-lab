# MetricsViewer 構造リファクタリング(挙動不変) — 取り込み/照会経路の可読性改善

## 元の材料

[MetricsViewer 構造リファクタリング(挙動不変) — 取り込み/照会経路の可読性改善](../../../memo/done/0042_metricsviewer_refactor_10rfd.md) blob:673f2193dc8c182d69279c72bb528ae333d1c721

## 要点

- Metrics Viewer の振る舞いを維持して、Java 側の取り込みと照会を読みやすくする。
- 解像度ごとの定数を集約し、取り込み状態と系列の利用可否を enum で表す。
- 入力ファイルの同一性判定と source_meta の読み書きを共通化する。
- 照会計画の作成を MetricsQueryPlanner に切り出し、取り込み側に SourceReader の境界を設ける。
- 不具合修正や不要コードの一掃は切り分け、既存テストで挙動の一致を確かめる。

## 内容

### 共通原則(全タスク)

配置: 新規トップレベルクラスは最小限にし、既存ファイルへの同居(nested class・同パッケージ)を優先する。

### IngestState / SeriesAvailability の enum 化

`fromDb(String)` はnull・未知値を拒否する。DB妥当性検査では非例外の`isValidDbValue(String)`を使用し、不正値を`PENDING`へ丸めず従来どおり全破棄再構築する。
`isConverting`を廃止し、呼び出し側を`IngestState.isStillIngesting()`へ置き換える。enum化は語彙と判定の一元化に限定し、状態遷移条件自体は変更しない。

### source_meta 読み書きの一機構化

read は `Connection → CacheMetadata` を 1 実装に。parseLong はここへ 1 枚だけ。呼び出し元ごとの現行 fallback 値は維持(既定値を引数で受けるか、呼び出し元別アクセサで吸収)。
Repository の query 経路(RunQueryContext 含む)は生 Map でなく CacheMetadata を保持・参照する。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
