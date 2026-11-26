# GitHub Actions リリースパッケージング(実行用 bin パッケージ)

## 元の材料

[GitHub Actions リリースパッケージング(実行用 bin パッケージ)](../../../memo/done/0043_release_packaging_10rfd.md) blob:51ef7c557b014cba75e101111e1b446ea727d3a6

## 要点

- Runner と Metrics Viewer を実行用 ZIP として GitHub Release で配布する。
- 設定・起動用 bat・必要なライブラリに設計資料とライセンスを同梱する。
- 配布物はリポジトリ配置の必要部分を写し、開発環境と同じパスで起動できるようにする。
- 当時の実測では圧縮後約1,798MBで1ファイル2GiBの制限内に収まり、CUDA 版 LibTorch も同梱できた。
- 容量制限を超えた場合はランタイムの ZIP 分割を検討する。

## 内容

### リポジトリ再配置

runner 系は `cd /d "%~dp0runner"` してから実行(runs/logs の出力先が CWD 依存)。

### リリースワークフロー(.github/workflows/release.yml 新設)

サイズガード: 本体 zip > 2,000MB で fail(2GiB 制限の事前検知)。

### 検証

ローカル: 移動後の `apps/10_run.bat` で Runner 起動・run 作成、`build-jar.bat` → `apps/22_metrics_viewer_java.bat` で Viewer 起動を確認。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
