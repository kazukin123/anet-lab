---
name: anet-stats
description: Generate anet-lab's fixed-point statistics from git history and local run folders (activity, code flow by area and Topic, size, test health by Topic category and AI-native era, process metrics), regenerate the reports/stats views, and append a digest entry grounded only in the computed summary and flags. Use when the user invokes $anet-stats or /anet-stats, or from anet-housekeeping.
---

# ANET Stats

「今週も頑張った」を数字で残し、+α として気になる傾向を 1 つ添える。主軸は稼働日、行の流量、Run 起動数、テスト充足。コミット数は参考列に降格してある。

正本は git 履歴とローカル Run フォルダで、`reports/stats/` の生成物は毎回丸ごと再生成される。蓄積するのは Run 台帳 `runs_seen.jsonl` と `digest.md` だけ。

## 手順

1. リポジトリルートで実行する。

   ```
   python reports/tools/stats.py
   ```

   初回は git blame で 1〜2 分かかる。2 回目以降はキャッシュで速い。失敗したら `--no-cloc --no-blame` で再実行し、縮退したことを完了報告に書く。
2. 標準出力の要約を読む。同じ内容が `reports/stats/summary.md` にある。`stats.json` を丸ごと読まない。
   要約の先頭に「digest は追記しない」と出ていれば、前回 digest 以降に新しい期間が無い(同日の再実行)。手順 3 を飛ばし、生成物を更新したことだけ報告する。
3. digest を書く。`reports/stats/digest.md` の末尾に次の節を追記する。既存の節は書き換えない。

   ```
   ## <since> 〜 <until>
   生成 <YYYY-MM-DD> / <claude|codex> / HEAD <sha>

   - 活動: 稼働日、Run 起動、連続稼働
   - 流量: コード・docs・config の追加削除、主種別、主 Topic、docs 同時更新率
   - テスト: L0 と被テスト率、年代の数字、薄いカテゴリ
   - 気になる点: flags の watch か info から 1 つ。無ければ「特になし」
   - 一言: 頑張った点を 1 文
   ```

   期間は summary の「期間」をそのまま使う(前回 digest の翌日から今日まで。週次固定ではない)。数字は summary.md にあるものだけを使い、推測で作らない。
4. summary や trend に `uncategorized` のファイルが出ていれば、完了報告に列挙し `reports/tools/categories.json` への追加案を書く。categories.json 自体は変えない。
5. 完了報告: 期間、digest に書いた要点、`reports/stats/trend.html` のパス、縮退の有無、未分類の追加案。

## 出力

| ファイル | 内容 | 扱い |
|---|---|---|
| `reports/stats/stats.json` | 全期間の集計 | 再生成 |
| `reports/stats/trend.md` | 人が読む表(活動と流量、規模、テスト充足、健全性、プロセス、flags) | 再生成 |
| `reports/stats/trend.html` | 1 ファイル完結、列ソート付きのビュー | 再生成 |
| `reports/stats/summary.md` | digest 用の短い要約と flags | 再生成 |
| `reports/stats/digest.md` | 期間ごとの一言。append-only | 蓄積 |
| `reports/stats/runs_seen.jsonl` | Run 台帳(フォルダが消えても履歴が残る) | 蓄積 |
| `reports/stats/cache/` | blame と cloc のキャッシュ | gitignore、消してよい |

## 設定

`reports/tools/categories.json` が分類の正本。領域、機能カテゴリ(GitHub Topic 31x〜32x への写像)、除外パターン(実験 config ダンプ、archify HTML、testdata)、境界日(AGENTS.md 作成日 2026-05-30)、bulk 閾値。変更は人が行う。

## 禁止

- digest 以外の生成物を手で編集しない(次の再生成で消える)。
- categories.json を変えない(追加案は報告に書く)。
- 統計にない数字を書かない。
- git の書き込み操作をしない。
