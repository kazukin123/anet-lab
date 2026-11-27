# RFD0202 の wiki 検証

リポジトリのルートで `.\.venv\Scripts\python.exe -B src/core/base/testdata/rfd0202/wiki_test.py -v` を実行する。一時 Git リポジトリに材料・作業の記録・サマリ・概念のページを置き、`wiki_support.py scan` の結果と `maps` 実行後のページを調べる。サマリの欠落・鮮度・移動、書式の警告、リンクの移動・修復、`query` の扱いを確認する。

会話の記録を引く道具は `.\.venv\Scripts\python.exe -B src/core/base/testdata/rfd0202/chat_utterances_test.py -v` で確かめる。一時フォルダに Claude Code の形と Codex の形の記録の見本を置き、`reports/tools/chat_utterances.py` が開発者の発言だけを、日付・会話の識別子・道具の回数とともに返すことを調べる。期間の端の日（日本時間）、語と期間の組み合わせ、どちらの形にも当たらないファイルと壊れた行の数、引数のエラー（語も期間も無い、日付として読めない、始まりが終わりより後、置き場が無い）を確認する。

既存の点検と課題一覧への影響は `src/core/base/testdata/rfd0095/wiki_test.py` と `src/core/base/testdata/rfd0114/issues_test.py` でも確認する。実際の移行で使った基準と照合結果は [実装メモ](../../../../../docs/memo/0202_wiki_accumulate_and_lookup_21impl.md) に記録する。
