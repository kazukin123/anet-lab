# Windows テストの行カバレッジ計測案

## 元の材料

[RFD0157　Windows テストの行カバレッジ計測案](../../../memo/0157_test_line_coverage_10rfd.md) blob:e64c4584bfcaf5d169c870e5df2417499b41333f

## 要点

- Windows でどのように行カバレッジを測るかは暫定案で、計測負荷を確かめてから構成や運用を決める。
- テストコード比率やファイル参照からの推定だけでは、実際にどの行を通ったか分からない。
- OpenCppCoverage を開発機に置き、既定は行対応を追いやすい Debug ビルドを測る案である。
- テスト対象と外部依存を区別して収集し、機能カテゴリごとに不足箇所を読めるようにする。
- 既存の履歴ベースの統計と実測の被覆を照合し、推定が外れる条件を調べる。

## 内容

### Context / Problem Statement

誤検出: 公開クラス名がテストに現れるかで判定するため、汎用的な名前で当たる。GUI 共通基盤と MuZero 系列は test 行が 0 なのに被テスト扱いになった。
見逃し: 内部クラスは名前がテストに現れない。設定管理の config_impl は config.hpp 経由で実際にはテストされているのに未テスト扱いになった。

### 効く事実

anet-core は STATIC ライブラリで、anet-core-test.exe に静的リンクされる。モジュールフィルタはテスト実行体 1 本でよく、libtorch などの巨大 DLL は自然に計測対象外になる。

### 提案

OpenCppCoverage を開発機にインストールする（winget または GitHub Releases のインストーラ）。リポジトリには持ち込まない。
手動、または anet-housekeeping からの任意実行。トークンは食わず時間だけ食うので、予算判断は時間側で行う。

### 候補を採る場合の受入条件

L1 で誤検出した GUI 共通基盤と MuZero 系列、見逃した config_impl の実態が数値で確定する。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
