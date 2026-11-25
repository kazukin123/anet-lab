# ソース配置とライブラリの境界

## ライブラリとアプリ

| 置き場 | 責務と依存 |
|---|---|
| `src/core/base/` | Agent、Env、NN、ReplayBuffer、Observer の契約と共通実装。直接の依存は外部ライブラリ |
| `src/algo/{dqn,ppo,muzero,imagecls}/` | Agent の系統別実装。基盤の公開ヘッダを使い、各系統の初期化関数を公開する |
| `src/env/` | 環境ごとの実装と初期化関数。基盤の公開ヘッダを使う |
| `src/apps/runner/` | Runner のビルド入力。Runner は基盤、Agent の各系統、Env の公開初期化関数を呼ぶ |
| `src/apps/metrics-viewer/` | Metrics Viewer のソースと Maven のビルド入力 |

`include/anet/` に置く公開ヘッダの契約は[拡張点の表](010_framework_overview.jp.md#43-拡張点)に載せる。各ライブラリの `src/` には実装と内部ヘッダを置く。

## テスト

単体テストは対象のライブラリに置く。ほかのライブラリを使うときは公開ヘッダとテスト用の口を通る。
テスト用の口のヘッダは `src/core/base/include/anet/test/`、実装は `src/core/base/src_testing/` に置く。
`anet-core-testing` はリンクしたテストに `ANET_TEST_SUPPORT` を定義し、小さな Network の組み立て、参照計算、代役の Agent を提供する。

複数のライブラリを使うテストは、主題となるライブラリに置き、相手は代役で構成する。組み合わせが主題であれば `src/apps/` に専用のテストアプリを置く。出荷設定と検証資材はリポジトリの根から読める。

ターゲットと include の依存は[RFD0182の検証資材](../../src/core/base/testdata/rfd0182/README.md#ライブラリの依存)で確認する。

データの移行と置き場は[RFD0182の移行検証](../../src/core/base/testdata/rfd0182/README.md#実機のデータを移す前の確認)で確認する。

## 出力の置き場

CMake と Maven の中間ファイルは `out/build/` に置く。単体テスト実行体と共通 DLL は `out/test/<構成>/` に出る。
Runner と GUI テストは `apps/runner/bin/<構成>/`、Metrics Viewer の jar は `apps/metrics-viewer/bin/` に出る。
