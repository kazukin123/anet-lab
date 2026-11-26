# リリース zip はリポジトリ配置のミラーサブセットとする

## 元の材料

[ADR 0017](../../../adr/0017-release-zip-mirrors-repo-layout.md) blob:196d4e303055ac39c0bb462d07b4fd2d4d084245

## 要点

- リリース zip にはリポジトリのうち実行に必要な配置をそのまま写す。
- 開発用と配布用で同じ起動 bat を使い、日常の起動で配布手順も検証できる。
- exe や jar を平坦に並べる案はスクリプトやパス記述の二重管理を招く。
- bat は Runner の作業ディレクトリを固定し、起動場所による違いを防ぐ。
- 後の配置変更で data/ を加え、道具を apps/tools、Viewer の jar を apps/metrics-viewer/bin に置いた。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
