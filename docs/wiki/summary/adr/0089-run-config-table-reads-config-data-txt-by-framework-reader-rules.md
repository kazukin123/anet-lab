# Viewer の設定表は config_data.txt をフレームワークの読み込み側の規則で読む

## 元の材料

[ADR 0089](../../../adr/0089-run-config-table-reads-config-data-txt-by-framework-reader-rules.md) blob:88e3bb3d5008e02fd681364cadc30bbc93a8c5a5

## 要点

- Viewer の設定表は Run の config/config_data.txt を読む。
- C++ の Properties::Load の行解釈を Java 側でも使う。
- UTF-8 として読めない値や $include はエラーにする。
- 同内容の JSON や取り込み済み SQLite より、実行基盤の形式を優先した。
- 設定ファイルの行規則を変更するときは C++ と Java を同じ変更で更新する。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [316 設定管理](../../concepts/framework/316_config.md)
