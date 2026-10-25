# PRD072 検証

設定 resolver の固定17入力比較と、既定葉 `?=` の検査の実行手順。いずれも通常の Debug ビルド済みで、リポジトリルートから実行する。

## 固定17入力比較

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/prepare.py prepare
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/prepare.py compare
```

入力定義・旧golden・新しい記録期待値は`core/anet-core/testdata/prd072/`で管理する。manifestの固定commitを含むGit履歴が必要。生成物は`.scratch/prd072-differential/profiled-<hash>/`（移行入力から計算したhash）。固定設定の原本と`migrate.py`で既定葉の`?=`へ書き換えた設定を別々に保持する。全キー・文字列値を旧goldenと比較し、解決記録・Run差分を`resolution/`の期待値と比較する。空のRun差分も明示検査する。値はキーでソートして厳密比較し、行順の差分は比較しない(dumpに重複キーはない)。`expected.py`は入力宣言・旧goldenの参照値から期待値を検証し、実行結果から自動更新しない。

`prepare.py compare` は準備した `prepare.ROOT` を子プロセスの必須環境変数 `ANET_PRD072_ROOT` に設定してテストを実行する。`prepare.py prepare` は準備だけを行い、親PowerShellの環境変数は設定しない。環境変数名はPythonの `PRD072_ROOT_ENV` とC++の `kPrd072RootEnv` で同じ文字列 `ANET_PRD072_ROOT` を定義する。

テストを直接実行する場合は、上の `prepare` の成功後、同じ入力から求めた `prepare.ROOT` を親PowerShellへ設定して実行する。別hashのディレクトリを選ばないよう、最新ディレクトリの検索は使わない。

```bash
$env:ANET_PRD072_ROOT = (& ./.venv/Scripts/python.exe -c "import sys; sys.path.insert(0, 'core/anet-core/testdata/prd072'); import prepare; print(prepare.ROOT)").Trim(); if ($LASTEXITCODE -eq 0) { & core/anet-core/bin/Debug/anet-core-test.exe '[prd072-baseline]' }
```

## 既定葉の `?=` 検査

共通ベースと環境別のデフォルト設定を`?=`、環境別のその他を`=`として検査する。未選択Run・bat・生成ツールの選択宣言からownerも求め、未分類の個別葉を検出する（ripgrepの`rg`がPATH上に必要）:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

検査器の回帰テスト:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves_test.py
```

一覧は`.scratch/prd072-default-leaf/default-leaf-audit.json`。過去Run、ローカルworkspace、独立resolver fixture、旧goldenは現用移行対象と区別する。

## golden の更新方針

goldenは契約を意図的に変えるときだけ更新する。変更前の最後のcommitで通常テスト実行体をビルドし、manifestのcommitもその値へ更新する。既存goldenは人間が削除した後、`prepare.py capture`で元の固定入力から明示採取する。スクリプト・テストの両方で既存goldenへの上書きを拒否し、自動再生成は行わない。goldenはUTF-8 / LF。manifestや移行処理を変えた場合、古い生成物は退避してからprepareする。

## 関連

- `?=` と `=` の運用規約: [実装 品質確認観点](../../../../docs/quality/impl.md#設定ファイル)「設定ファイル」の規約（代入演算子）、[Run実行ユーザーガイド](../../../../docs/design/020_user_guide_run.jp.md) §3.6〜§3.8
- 設計: [実行基盤と設定](../../../../docs/design/100_runtime_and_configuration.jp.md) §8
