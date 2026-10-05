# レビューだけを頼む入口（PRD 087 段階 1.5）実装メモ

## 概要

[PRD 087](087_review_agent_quality_viewpoints_10prd.md) の段階 1.5（§6.5、受入基準 17〜25）を実装する。
レビューだけを頼む入口の skill `anet-review` を新しく置き、PRD の無い変更も `impl` としてレビューに出せるように、観点とレジストリを直す。
段階 2（`anet-create-prd`、`implement-prd-with-docs` への追記、`grill-with-docs` の削除、読み替え表）は、今回は触らない。

`.gitattributes` の改行のルール（§6.5 の 5 行）は、2026-10-05 に入れ済みである。今回は確かめるだけにする。

本メモをファイルの作成前に保存し、実装の正本とする。段階 1 の記録は [20impl](087_review_agent_quality_viewpoints_20impl.md) にある。

## 主な変更

| ファイル | 種類 | 内容 |
|---|---|---|
| `.agents/skills/anet-review/SKILL.md` | 新規 | skill の本体 |
| `.agents/skills/anet-review/agents/openai.yaml` | 新規 | Codex の表示と、暗黙起動の禁止 |
| `.claude/skills/anet-review/` | 新規 | 上の 2 ファイルの実体コピー |
| `docs/quality/impl.md` | 更新 | 位置づけ（対象外の行、入力元が「なし」のときの扱い）と、「PRD への適合」の適用範囲 |
| `docs/process/registry.md` | 更新 | 成果物レジストリの注記に 1 項目 |
| `.agents/skills/README.md`、`.claude/skills/README.md` | 更新 | `anet-review` の行、表の前の文、「anet- スキル共通の約束」の対象 |
| `docs/memo/README.md` | 更新 | §工程と担当 の箇条書き 1 項目 |
| `.scratch/prd087/stage15-acceptance.md` | 新規（git 管理外） | 受入の手順と、結果の保存先 |

変えないもの: `AGENTS.md`、`CONTEXT.md`、共有コアの 2 ファイル、`.codex/agents/an-review.toml`、ほかの skill、`.gitattributes`。

### skill `anet-review`

`SKILL.md` の frontmatter の `description` は、既存の `anet-` skill に合わせて英語で書き、起動条件（`$anet-review` / `/anet-review` だけ）を含める。本文は日本語で、次の節を持つ。

1. **禁止事項**: 成果物を直さない。git の書き込み操作をしない。プロセス契約は 2 節だけを読む。モデルを指定しない。
2. **対象を決める**: PRD §6.5 の「対象の決め方」の表と箇条書き。加えて、下の「前提」の対象の決め方を書く。
3. **依頼を組み立てる**: 読むのは、プロセス契約の 2 節（見出し `## 2. レビュー依頼の形式` から次の `## 3.` の前まで）、`docs/process/registry.md`、`impl-plan` と `impl` のときは `docs/quality/impl.md` の「位置づけ」の節。項目の決め方は PRD §6.5 の表と箇条書きのとおり。依頼文の形は、段階 1 の受入で使った形（冒頭の 1 文と、項目の箇条書き。まとめるときは成果物ごとのブロック）にする。
4. **呼ぶ**: PRD §6.5 の「呼び方」の表と箇条書き。呼ぶ前に、対象・成果物キー・観点ファイル・新しく呼ぶか続けて依頼するかを短く示し、確認は取らない。
5. **応答を示す**: 全文をそのまま示して止まる。保存は頼まれたときだけ。

`agents/openai.yaml`:

```yaml
interface:
  display_name: "ANET Review"
  short_description: "成果物をレビューエージェントに出し、応答をそのまま示す（直さない）"
  default_prompt: "$anet-review を使って、この会話で変更したファイルをレビューに出してください。"

policy:
  allow_implicit_invocation: false
```

### `docs/quality/impl.md`

- 位置づけの表の「対象外」を「設計の妥当性。PRD があれば PRD の側で決め、無ければコミット前の人間のレビューが見る。」にする。「PRD の無い修正も、対象外である」は外す。理由の文も、PRD の無い変更に合うように直す（文書内の整合のため）。
- 位置づけの箇条書きのあとに、小見出し「入力元が「なし」のとき」と、PRD §6.5 の 4 行の表を置く。表の前の文で、PRD に沿わない変更では入力元が「なし」で渡されることと、表に無い項目はそのまま適用することを書く。
- 「PRD への適合」の適用を、「`impl` を渡されたとき。入力元が「なし」のときは、項番 14・15 は対象外、16・17 は位置づけの「入力元が「なし」のとき」による。」にする。
- 項番は変えない。項目の本文も変えない。

### `docs/process/registry.md`

成果物レジストリの注記に、次を 1 項目足す。

- `impl` は、PRD に沿わない変更にも使う。そのときは、入力元を「なし」と書いて渡す。観点の扱いは `docs/quality/impl.md` の位置づけによる。

### skills README（両方のコピー）

- 表の前の文: 「定常整備系は `anet-` 接頭辞で」を、`anet-` の接頭辞が定常整備系に限らないことが分かる文にする。
- 表の最後に `anet-review` の行を足す。用途、起動例（`/anet-review`、`$anet-review 087_review_agent_quality_viewpoints_10prd.md`）、出力先（報告のみ。頼まれたときだけ `.scratch/` に保存）。
- 「anet- スキル共通の約束」の対象を、定常整備系（`anet-housekeeping` と、それが呼ぶ候補）に限る。約束の中身は変えない。

### `docs/memo/README.md` §工程と担当

段階 1 で足した箇条書きを、`anet-review` から頼む形に直す。引数が無ければ会話で変えたファイルを対象にすること、PRD の無い変更も出せること、名指しでも呼べることを書く。観点・プロセス契約・レジストリの所在と、見るのは観点の充足であることは残す。表は変えない。

### 手順

1. 本メモを保存する。
2. `anet-review` の `SKILL.md` と `openai.yaml` を `.agents/skills` に書き、`.claude/skills` へコピーする。
3. `impl.md`、`registry.md`、skills README の両コピー、`docs/memo/README.md` の順に直す。
4. 自己点検として、`impl.md` の全項目を今回の成果物に自分で通す。あわせて「検証」の 1〜7 を実行する。
5. `.scratch/prd087/stage15-acceptance.md` を書き、受入（検証 8・9）をユーザーへ渡す。
6. 変更ファイル・実行した検証・残った未決を報告して止まる。git の操作はしない。受入の結果は、ユーザーが手順書の保存先に置いたあとで、本メモの末尾に転記する。

## テスト

- Public interface / surface: skill `anet-review` の起動（引数なし、PRD のファイル名、パス）と、ユーザーへ示すもの。レビューエージェントへの依頼（プロセス契約の 2 節）。`impl.md` の位置づけと項番。レジストリの注記。
- 優先 behavior:
  1. 引数なしの起動で、その会話で変更したファイルだけが対象になり、PRD に沿わない変更では入力元が「なし」になる。応答が全文で示され、1 行目が `総合判定:` で、適用した項目に項番 14・15 が入らない。対象のファイルは変わらない。
  2. 同じ会話で 2 回目に起動すると、前のレビューエージェントへの続けての依頼になる。
  3. PRD のファイル名を渡すと、`prd` のレビューが返る。
  4. 何も変更していない会話で引数なしで起動すると、レビューエージェントを呼ばずに、`git status --short` を示して対象を尋ねる。
- TDD 順序: 対象外。変更は文書と skill だけで、コードと runner の設定を変えない。behavior は、受入（検証 8・9）で、上の順に確かめる。

## 検証

1. `anet-review` の 2 つのコピーと、skills README の 2 つのコピーに差が無い。
   `git -c core.safecrlf=false diff --no-index --no-renames --name-status .agents/skills/anet-review .claude/skills/anet-review` と、README の 2 ファイルの同じ比較が、何も出力しない。
2. `anet-review` の `SKILL.md` の frontmatter に `name` と `description` があり、`openai.yaml` が `.venv` の Python の `yaml` で読めて、`policy.allow_implicit_invocation` が `false` である。
3. `docs/quality/impl.md` の項番が 1〜53 で、重複が無く、着手時から変わっていない（項番の列を、着手時の `git show HEAD:docs/quality/impl.md` と比べる）。「PRD の無い修正も、対象外である」が無い。入力元が「なし」のときの表が 4 行ある。
4. `docs/process/registry.md` の注記に、PRD に沿わない変更の渡し方がある。見出しは 7 つのままである。
5. 変えた・作った Markdown の相対リンクの参照先が、すべて実在する。
6. `git status --short AGENTS.md CONTEXT.md docs/process/protocol.md .claude/agents/an-review.md .codex/agents/an-review.toml` が空である。`.gitattributes` は、着手の前から未コミットの変更（§6.5 の 5 行）があるので、`git diff .gitattributes` がその 5 行の追加だけであることを確かめる。`git diff docs/memo/README.md` の差分が §工程と担当 の中だけである。
7. `git ls-files --eol -- .agents .claude .codex` の全行が `w/lf` と `attr/text eol=lf` を持つ。`git ls-files --eol --others --exclude-standard -- .agents .claude` で、新しいファイルが `w/lf` と `attr/text eol=lf` を持つ。ほかの変えたファイルも `w/lf` である。
8. Claude Code 側の受入（受入基準 18・19・24）: 手順書に従い、ユーザーが新しい会話で行う。PRD の無い小さな変更を 1 つ行い、`/anet-review` を引数なしで起動する。次に、対象のファイルを 1 つ変えて、もう一度起動する。最後に、1 回目と同じ依頼文で、`an-review` を `sonnet` を指定して直接呼ぶ。各往復の消費・時間・指摘の項番を記録する。対象のファイルの `git hash-object` を、起動の前後で比べる。
9. Codex 側の受入（受入基準 18・19・20）: 手順書に従い、ユーザーが Codex アプリの新しいスレッドで行う。8 の前 2 つと同じことを確かめる。加えて、PRD のファイル名を渡した起動と、何も変更していないスレッドでの引数なしの起動を確かめる。

## 前提

- 実装はこのセッションで行う（2026-10-05 のユーザー回答）。受入は、手順書を `.scratch/prd087/` に置いて、ユーザーが新しい会話で行う（同）。段階 1.5 の成果物そのものは、レビューエージェントにかけない（同）。
- 対象の決め方で、PRD §6.5 に無い点を次のとおり固定する。
  - 「その会話で変更したファイル」は、その会話の中で、ファイルを書く道具かコマンドで作った・変えた・消したファイルとする。リポジトリの外のファイルと、Git が無視するファイル（`.scratch/` など）は除く。作業資材は、レビューの対象ではないためである。
  - 対象に PRD があるとき、同じ会話で変えた `docs/adr/` のファイルと `CONTEXT.md` は、`impl` に入れず、その PRD の依頼の変更箇所に書く。§6.5 の「PRD のレビューでは、あわせて起こした・変えた ADR と、`CONTEXT.md` の変更があれば、変更箇所に書く」を、引数なしの起動に当てはめたものである。
  - PRD と実装計画をファイル名だけで渡されたときは、`docs/memo/` を、`done/` `frozen/` `dropped/` を含めて再帰的に探し、完全一致で解決する。一致が無いときと、複数あるときは、呼ばずに候補を示して尋ねる。`implement-prd-with-docs` の PRD 解決と同じ規則である。
  - `impl` か `impl-plan` の対象に、Git が追跡していて基準コミットから変わっていないファイルがあるときは、呼ばずに、そのファイルを示して基準コミットを尋ねる。そのまま渡すと、`impl.md` の項番 6 の期待値に当たらず、機械検査が NG になるためである。`prd` は §6.5 のとおり「全体」で渡す。
  - 観点ファイルが同じ成果物は、PRD が複数のときも 1 回の依頼にまとめる（プロセス契約の 2 節が許す形）。
- 呼び方で、PRD §6.5 に無い点を次のとおり固定する。
  - 「前の起動と同じ対象」は、観点ファイルが同じで、今回の対象が前の起動の対象をすべて含むこととする。引数なしの起動では、2 回目の対象は 1 回目の対象を含むので、続けての依頼になる。
  - 続けて依頼するときに、前の往復のあとで変えたファイルが無ければ、呼ばずに、その旨を示す。
  - Claude Code の SendMessage は、読み込まれていなければ、先に読み込む。
- 保存を頼まれ、ユーザーが名前を示さないときは、`review-<YYYYMMDD-HHMM>.md` という名前にする。置き場の `NNN` は、対象の PRD か、入力元の PRD の番号にする。
- PRD §7 の「親より軽いモデル」は、`sonnet` にする。親は Opus である。`haiku` は、観点 53 項目の走査に足りるかが測る前から疑わしいので、使わない。
- 受入で使う PRD のファイル名は、コミット済みで変更の無い小さな PRD（`999_nn_forward_amp_consolidation_10prd.md`、4.5KB）にする。変更箇所が「全体」になる経路を確かめられ、Codex の消費も小さい。
- PRD の無い変更の題材は、ユーザーが選ぶ。手順書には、段階 1.5 で触るファイル以外を選ぶよう書く。同じファイルに段階 1.5 の未コミットの変更があると、基準コミットからの差分に両方が出るためである。
- 汎用 skill 46 ファイルが `git status` で変更ありと出るのは、改行を揃えた結果で、中身の差は無い。`git add --renormalize .agents/skills .claude/skills` で消える。実行は人間が行う。
- 新しく書くファイルは LF で書く。

## 検証の結果（2026-10-05）

検証 1〜7 を実行し、すべて満たした。コードは変えていないので、ビルドとテストは実行していない。

- `anet-review` の 2 つのコピーと、skills README の 2 つのコピーは、`git diff --no-index` で差が無い。
- `SKILL.md` の frontmatter は `name` と `description` の 2 キーで、`openai.yaml` は `.venv` の Python の `yaml` で読め、`policy.allow_implicit_invocation` が `false` である。
- `docs/quality/impl.md` の項番の列は、`HEAD` の版と同じ 1〜53 で、重複が無い。「PRD の無い修正も、対象外である」は無く、入力元が「なし」のときの表は 4 行である。
- `docs/process/registry.md` の見出しは 7 つのままで、注記に PRD に沿わない変更の渡し方がある。
- 変えた・作った Markdown の相対リンクは、すべて実在する。`docs/memo/README.md` で検出される 1 件は、コードブロックの中の `sed` の式で、リンクではない（着手前から同じ）。
- `AGENTS.md`、`CONTEXT.md`、共有コアの 2 ファイル、`.codex/agents/an-review.toml` に差分は無い。`.gitattributes` の差分は、§6.5 の 5 行の追加だけで、着手の前（2026-10-05 02:24）からある。`docs/memo/README.md` の差分は、§工程と担当 の箇条書き 1 項目の中だけである。
- `git ls-files --eol -- .agents .claude .codex` の 92 行は、すべて `w/lf` と `attr/text eol=lf` を持つ。新しい 4 ファイル（`anet-review` の 2 コピー）も、`w/lf` と `attr/text eol=lf` である。ほかの変えたファイルも `w/lf` である。

計画から変えた点:

- 検証 6 は、当初 `.gitattributes` の差分が無いことを見るとしていた。着手の前から未コミットの変更があったので、その 5 行だけであることを見る形に直した。
- skills README の「anet- スキル共通の約束」は、名前を残し、括弧で対象（定常整備系。`anet-housekeeping` と、それが呼ぶ候補）を限る書き方にした。PRD 087 と本メモが、この名前で節を指しているためである。

### 自己点検（`docs/quality/impl.md`）

今回の成果物（本メモと、「主な変更」の表の変更ファイル）に、全項目を通した。

- 機械検査: 項番 1〜3 は、`.cpp` と `.hpp` を変えていないので対象外。項番 4 は、共有コアの 2 ファイルで `anet`・`PRD` ともに 0 件。項番 5 は、本メモの名前が形に合い、対の PRD がある。項番 6 は、変更ファイルがすべて差分に現れるか、新規で実在する。項番 7 は、上のとおり `w/lf`。項番 8 は、`anet-review` と README が出力に現れない。出力に出る 3 件（`anet-audit`、`anet-housekeeping`、`grill-with-docs` の `SKILL.md`）は、段階 1 から既知の不一致で、今回は触っていない。
- 実装計画（9〜13）: 充足。コードを変えないので、TDD の対象外とする理由を「テスト」に書いた。
- PRD への適合（14〜17）: 受入基準 17・21・22・23・25 は満たした。18・19・20・24 は受入（検証 8・9）を待つ未解決で、手順書に渡した。PRD §6.5 に無い点は、「前提」に理由とともに書いた。
- 契約の変更（18〜22）: `impl.md` の対象外の行と、README の約束の対象を変えた。現用の範囲（`docs/process/`、`docs/quality/`、`docs/agents/`、`docs/design/`、`.agents/`、`.claude/`、`AGENTS.md`）に、旧い記述を前提にした参照は無い。PRD 087 と 20impl は当時の記録なので、書き換えていない。互換のためのものは持たない。
- 失敗の扱い・C++・コメント・計測・GetScalar・所有権・ログ・テスト（23〜47）: 該当する変更が無いので対象外。
- 検証と報告（48〜52）: 48・49 は本節と「受入」に書いた。50 は、skill とレビューの手順を扱う設計文書が `docs/design/` に無いので対象外。51 は、`.scratch/prd087/stage15-acceptance.md` が別枠へ渡すための作業資材で、常設の検証資材ではないので対象外。52 は、設定ファイルを変えていないので対象外。
- skill（53）: README の両コピーに `anet-review` の行を足した。

### 受入

検証 8・9（受入基準 18・19・20・24）は、手順書 `.scratch/prd087/stage15-acceptance.md` をユーザーへ渡した。結果は `.scratch/prd087/out15/` に置かれたあとで、本メモに転記する。
なお、このセッションでは `anet-review` を作った直後に、Claude Code の skill の一覧に出た。新しいセッションを始めなくても読まれる。
