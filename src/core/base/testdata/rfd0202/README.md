# RFD0202 の wiki 検証

リポジトリのルートで `.\.venv\Scripts\python.exe -B src/core/base/testdata/rfd0202/wiki_test.py -v` を実行する。一時 Git リポジトリに材料・作業の記録・サマリ・概念のページを置き、`wiki_support.py scan` の結果と `maps` 実行後のページを調べる。サマリの欠落・鮮度・移動、書式の警告、リンクの移動・修復、`query` の扱いを確認する。

既存の点検と課題一覧への影響は `src/core/base/testdata/rfd0095/wiki_test.py` と `src/core/base/testdata/rfd0114/issues_test.py` でも確認する。実際の移行で使った基準と照合結果は [実装メモ](../../../../../docs/memo/0202_wiki_accumulate_and_lookup_21impl.md) に記録する。
