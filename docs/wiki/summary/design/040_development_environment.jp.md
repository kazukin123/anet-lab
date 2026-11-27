---
title: ANET 開発環境構築ガイド
page_kind: サマリ
material: "[[docs/design/040_development_environment.jp.md|040_development_environment.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
updated: 2026-11-26
---
# ANET 開発環境構築ガイド

## 元の材料

[ANET 開発環境構築ガイド](../../../design/040_development_environment.jp.md) blob:b834861588f82cd0f2a4c6d9b9144007fb31270d

## 要点

- Windows の C++ 開発では Visual Studio の CMake preset を使い、Runner と単体テストをビルドする。
- libtorch、wxWidgets、CUDA、Atari 用の ALE は環境変数と CMake の検出結果を対応させる。
- CLI で C++ をビルドするときは MSVC と Windows SDK の環境を初期化する。
- Metrics Viewer は JDK と Maven でテスト・パッケージ化し、Eclipse からも実行できる。
- Python の補助ツールは仮想環境から使う。依存を見つけられないときは実際の環境変数と configure の結果を確認する。

## 内容

### 2. 検証済み環境と推奨 IDE

- 現時点でフレームワーク全体の開発・実行を確認しているのは、Windows 11 x64、MSVC、NVIDIA GPU、CUDA 対応版 libtorch の組み合わせです。
- Java 版 Metrics Viewer のコーディング、Maven build、testには Eclipse を推奨します。JDK 17 以上と Maven integration for Eclipse（m2e）を使用します。
- IDE は C++ に Visual Studio、Java に Eclipse を使い、CLI は再現試験・CI・診断で使用する。

### 3. 初期環境構築

- Box2D、Catch2、nlohmann/json、Tracy、NVTX header は `third_party/` に含まれています。Doxygen、Graphviz、ffmpeg、Tracy server、NVIDIA Nsight は、それぞれ文書生成、図・動画生成、性能分析を行う場合に追加します。最初のビルドには不要です。
- Visual Studio Community 2022 Installerから Visual Studio Community 2022 を導入します。

### 4. Visual Studio による C++ 開発

- Visual Studio でリポジトリのルートを開き、Debug の CMake preset を選ぶ。
- 初回は `x64-Debug` を使用します。自動 configure が始まらない場合は `プロジェクト > Configure anet-lab` を実行します。configure 完了時に CMake 出力に error がないことを確認してください。

### 5. Metrics Viewer のビルドと開発

- Viewer を停止してビルドすると、配布用の JAR が生成される。
- Eclipse の `Window > Preferences > Java > Installed JREs` で、3.7節で導入した JDK を選択します。一覧にない場合は `Add...` から導入先の folderを追加します。

### 6. CLI によるビルド・テスト（AI・CI・診断用）

- CLI から C++ を構築するときも、コンパイラと SDK の環境変数を初期化する。
- Editionやインストール先が異なる場合は `VsDevCmd.bat` のパスを読み替えます。
- 対象を絞る場合は、先にtargetをビルドしてから実行ファイルをリポジトリルートで起動します。
- 起動確認はリポジトリルートから行います。

### 7. Python 補助ツール

- PythonはOptuna harness、Python版Metrics Viewer、TensorBoard bridge、MLflow bridgeなどで使用します。C++本体とJava版Metrics Viewerのビルドだけを行う場合は不要です。
- Python packageをuser siteやglobal環境へ混在させず、リポジトリルートの `.venv` を使用します。

### 8. トラブルシューティング

- Visual Studio Installerで「C++ CMake tools for Windows」が導入されていることを確認します。その後、リポジトリルートをフォルダーとして開き直します。`CMakePresets.json` を編集した直後は、Visual Studioの再起動またはCMake cacheの再構成が必要な場合があります。
- libtorchのCUDA世代、NVIDIA driver、CUDA Toolkitの互換性を確認します。実行ファイル隣のDLLが古い場合は、対象configurationを再ビルドしてpost-build copyをやり直します。

## 主要な概念

- [311 開発ツール](../../concepts/framework/311_dev_tools.md)
- [RunnerApp](../../concepts/apps/110_runner_app.md)
- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
