# Claude Code の memory の定期整備

## 元の材料

[RFD0112　Claude Code の memory の定期整備](../../../memo/0112_claude_memory_tidy_10rfd.md) blob:3ff2b927fedb0fdd2b9b87f4e5a1e76df952384f

## 要点

- Claude Code の memory の手入れ方を検討した暫定案であり、実行の頻度、候補の選び方、修正の手順は合意前である。
- memory の索引が読み込み上限を超え、新しく追加した項目が読み込まれない問題を扱う。
- 索引と個別ファイルの対応、古くなる作業状態、正本と重複した実験結果や規約を確認する。
- Git やフォルダから分かる状態はそこで確かめ、知見に正本があれば memory はその所在を示す。
- 正本が見つからない知見は個別に残し方を判断できる一覧にする。
- Claude Code のこのリポジトリ用 memory を明示起動または定常整備で点検する設計である。

## 内容

### 目的と問題

知見（Run の数値、調査の結論）が memory に溜まり直している。RFD0095 の段階 3 で一度ポインタ化したが、その後も索引と topic file に結果の数値が書き足されている。memory は Claude Code だけが読めるので、正本と二重に持つと食い違いの元になる。
memory を定期的に見直す仕組みが無い。既存の仕組みは対象が狭いか、anet の規則（状態はフォルダと git、知見は正本から wiki へ）を知らない。

### 起草時点の現状

知見。10/24〜11/02 に更新した project 型の topic file のうち、17 本が数値の知見（%・seed・中央値・倍）を持つ。そのうち 16 本は repo のパスも書いている。正本を指しながら、数値も memory に写している。
Run で確かめると、memory の値に誤りが多数見つかった（打ち切り扱いが完走、複製扱いが別設定、Run フォルダの改名など）。
anet-improve の「memory（Claude Code だけ）」。対象は、その会話で書いた・直した feedback memory のうち規約へ移したものだけである。
Claude Platform の Dreams: 既存の memory store と過去の会話記録を読んで、新しい store を作る。入力の store は変えない。

### 候補を採る場合のゴール・非ゴール・成功の測り方

Codex の memory の整備。退ける案は「段階 1 で Codex の memory も整備する」案である。本当の制約ではなく、書き換えの可否を確かめるまでの先送りである。
memory の置き場を変えること（`autoMemoryDirectory`）。退ける案は「置き場を repo の中へ移す」案である。公開 repo に私的な memory が入るという本当の制約による。
ポインタにした数（知見と feedback）。

### 起草時の暫定案

整備は skill `anet-tidy-memory` に置き、`.agents/skills` と `.claude/skills` の両方に同じ中身を置く。Codex で起動されたときは、何も書かずに「Claude Code の memory は Codex からは整備しない」と報告して終える。
毎回読む範囲は差分にする。索引の全行と、前回から変わった topic file と、状態の語か repo のパスを含む topic file（`scan` の検索で絞る）を読む。引数 `full` を付けたときと、前回の記録が無いときは全部を読む。
機械的な部分は補助スクリプト `reports/tools/memory_support.py` が受け持つ。本文の照合と縮め方の判断は skill を実行する側が行う。
名前は、skill が `anet-tidy-memory`、housekeeping の候補が `memory-tidy`、補助スクリプトが `memory_support.py` とする。

### 候補を採る場合の実装契約

repo の `.venv` で、repo の根から実行する。memory の置き場は引数で受け取り、パスをスクリプトに書かない。
skill の引数は `full` だけである。ほかの語が付いたときは、何も書かずに受け付ける引数（`full`）を示して止まる。
ある topic file が 2 つの直しの組に入るときは、その組どうしを 1 つのまとまりに合わせる。こうして、まとまりどうしは重ならず、各 topic file は 1 回の実行で多くとも 1 回だけ書く。このため、書く直前の判定で比べる先は、つねに実行前の `hashes.json` である。
動線マップ（`docs/agents/`）は、`.agents/skills` の変化を anet-harness-map の前提条件が拾って作り直すので、この変更では直さない。

### 候補を採る場合の検証と受入

`--state` による変わったファイルの検出。`--state` のファイルが無いときは終了コード 0 で前回の記録が無いことを示し、全部の topic file を変わったものとする。壊れた `state.json` では終了コード 1。
`backup` の写しと `hashes.json`。`--out` が既にあるときは何も書かずに終了コード 1。
Codex の `$anet-housekeeping S dry-run` では出ない。
`runs.jsonl` に足された行の `note` が件数だけで、topic file の名前と本文を含まない。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
