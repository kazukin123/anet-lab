# 文書の呼び名 RFD・4 桁の番号への移行 実装メモ

対象: [PRD195](0195_document_term_rfd_and_4digit_numbers_10rfd.md)（実装で `195_rename_prd_document_term_10prd.md` から改名した）。課題は [issue-20261121-529662](../management/issues/issue-20261121-529662.md)・[issue-20261116-fe09dc](../management/issues/issue-20261116-fe09dc.md)。

## 概要

PRD195 の D1〜D26 のとおり、文書の呼び名を RFD に、番号を 0 埋めの 4 桁に改め、今使っている場所をすべて移す。コミットは人間が行うので、作業ツリーを D4 の 1 回目のコミットの状態にし、2 回目のコミットの中身は、1 回目のコミットのあとに `.scratch/rfd0195/run_phase2.py` で作る（手順は `.scratch/rfd0195/README.md`）。2 回のコミットの性質（改名の見分け、1 回目のあとも動くこと、2 回目のあとの残りの検索と統計）は、使い捨ての clone の中で 2 回コミットして確かめる。

置き換えは `.scratch/rfd0195/` のスクリプトで機械的に行い、道具とそのテスト、経緯の行（D22）、旧い名前を話題にする記述（D24）、`.scratch/` の今ある材料を指すパス（D5）は手で直す。

## 主な変更

### 1. 改名（D10・D12、1 回目）

- `docs/memo` の下の、3 桁の番号と `_` で始まるファイル（README を除く全部）を、番号を 0 埋めの 4 桁にして改名する。`_10prd.md`・`_12prd.md` は `_10rfd.md`・`_12rfd.md` にする。題の途中の `prd` は `rfd` に、`prdNNN` は `rfd0NNN` にする（114・120・131・179）。195 の題は `document_term_rfd_and_4digit_numbers` にする。実装計画・検証記録・名前の終わりが違う計画と資料は、同じ番号の文書の新しい題にそろえる（文書と題が同じものだけ。今の題が文書と違うものは番号だけを改める）。
- skill の `anet-create-prd`・`anet-impl-prd` のディレクトリを `anet-create-rfd`・`anet-impl-rfd` に、`PRD-IMPLEMENTATION.md` を `RFD-IMPLEMENTATION.md` に改める（`.agents/skills` と `.claude/skills` の両方）。
- `docs/quality/prd.md` を `docs/quality/rfd.md` に改める。
- 検証資材のフォルダ `src/core/base/testdata/prdNNN/`（41 個）を `rfd0NNN/` に、`config_resolver_prd072.json` を `config_resolver_rfd0072.json` に、`src/algo/muzero/src/prd101_muzero_test.cpp` を `rfd0101_muzero_test.cpp` に改める。検証資材の中の Run の設定の写し（`prd092/config/run_*.txt`）と `prd084` の過去の検証の記録 3 本は、名前も本文も残す（D19）。
- 改名は `git mv` を使わず、ファイルの移動で行う（ステージしない）。

### 2. リンクの直し（P5・D4、1 回目）

追跡しているすべてのテキストファイルで、改名したファイルを指すものを新しいパスへ直す。

- Markdown のリンクの丸括弧の中のパス（`#` より前）。ファイルの置き場から解決し、改名したパスに当たるものだけを直す。リンクの文字は、直すファイルでは 3 の規則で直し、残すファイルでは変えない。
- Obsidian のリンクの名前（`[[名前]]`・`[[名前|別名]]`・`[[名前#見出し]]`）。
- 見出しを指す `#` のあと（D26）。改名しないファイルの見出しは 1 回目で変わるので、それを指す `#` も 1 回目で直す。改名したファイルの見出しは 2 回目で変わるので、それを指す `#` は 2 回目で直す。

### 3. 本文の置き換え（D3・D6・D11〜D13・D19）

直すファイル（PRD §9.5 の線引きの表で「直す」の行）の本文に、次の順で当てる。大文字小文字は今の使い分けを写す。

1. 改名したパスと名前（題の途中の `prd` を含む、旧いファイル名・skill 名・品質観点のパス・検証資材のフォルダ名）を新しい名前へ写す。
2. 番号つきの呼び方: `PRD` に空白か無しで続く 3 桁を `RFD` と 4 桁に（`PRD195`・`PRD 195` → `RFD0195`）、`Prd` に続く 3 桁を `Rfd` と 4 桁に、`prd` に続く 3 桁を `rfd` と 4 桁にする。`RFD0nnn/nnn` の形で続く 3 桁の番号も 4 桁にする。`PRDnnn`・`PRDNNN`・`prdNNN`・`{NNN}` のような形の見本は、4 桁の見本（`RFDnnnn`・`RFDNNNN`・`rfdNNNN`・`{NNNN}`）にする。
3. 残りの `PRD`・`Prd`・`prd` を `RFD`・`Rfd`・`rfd` にする。

置き換えない範囲:

- Run 名（`run_YYYYMMDD-HHMMSS_…`）、ADR と実験記録のファイル名、検証資材の中の過去の記録の名前。どれも名前を残すもの（A1 の 51 本）を指す。
- `to-prd`・`grill-with-docs` など、リポジトリの外の skill の名前（§1.3）。
- D5 の今ある材料を指す `.scratch/` のパス（bat の `OLD_EXE_SRC`、検証資材の README の過去の結果の置き場、過去の Run の出力を読む入力のパス）。これから書き出す先（skill の保存先の指示、テストの出力先、検証スクリプトの既定の出力先）は `.scratch/rfdNNNN/…` へ直す。
- D17 で旧い記号 `PRD` と 3 桁、旧い形の振り直しの行を読む、道具の処理とそのテストの行。
- D22 の経緯の行（同じ行に `RFD0195` を書く）と、D24 の旧い名前を話題にする記述。
- 直下の文書（D3）では、題の行と最初の `## ` の見出しより後だけを直し、そのあいだの冒頭の引用行は残す。本文の中でも、`>` で始まる引用の行、発言を引いた「」の中、コードの枠の中の件名の例と、バッククォートで囲んだ件名の引用は残す。本 PRD（195）は題の行の記号だけを `RFD0195` にする（D24）。PRD131 の本文の振り直しの行の形 `（PRDNNN）` の説明は残す（D24）。
- 課題（D11・D6・D18）は前付けだけを直す。`prd` 欄を `rfd` 欄にして値を新しい名前にし、`phase: PRD` を `phase: RFD` にし、`docs` の値を新しいパスにし、題と scope の旧い名前を直す（issue-20261121-529662 の題と scope は D24 で残す）。本文はリンクのパスだけを直す。
- 残すファイル（保管場所の文書、直下の実装計画と検証記録、ADR、実験記録、wiki の log.md、reports のレポートと定常整備の記録、検証資材の中の過去の記録、CHANGELOG のリリース済みの節）は、リンクのパスだけを直す。

### 4. 1 回目と 2 回目への分け方（D4・§9.6）

- 改名しないファイル（道具、src・apps のコード・テスト・タグ・識別子・環境変数・bat、AGENTS.md・CONTEXT.md・README.md、registry・numbering・summary-page、品質観点のほかのファイル、memo と課題の README、wiki の手で書く節、設計文書、CHANGELOG、課題の前付け、ほかの skill）は、3 の全部を 1 回目に入れる。
- 改名したファイルは、1 回目に 2 のリンクと、動作に効く中身だけを入れる。動作に効く中身は、道具のテスト（`wiki_test.py`・`issues_test.py`）の断定、検証資材のスクリプトと README の環境変数の名前とパスの文字列、`rfd0101_muzero_test.cpp` のタグとテスト名、skill の前付けの名前と、skill の本文が名指すほかの skill・品質観点・`RFD-IMPLEMENTATION.md` のパス、`docs/quality/rfd.md` の項番 2・3 の形である。残りの本文の語の置き換えは 2 回目に入れる。
- 2 回目には、改名したファイルの残りの本文の語、改名したファイルの見出しを指す `#` の直し（D26）、`maps` で作り直した生成物、wiki の取り込みの記録（D23）を入れる。

### 5. 道具（D17・D25・§9.1）

- `reports/tools/wiki_support.py`: 文書を `*_10rfd.md` と `\d{4}_` で集める。件名から `RFD` に続く 4 桁と `PRD` に続く 3 桁の両方を、scope と末尾の参照の並びの両方から読み、値で 4 桁のファイル名に当てる。振り直しの行は今の形（3 桁、`（PRDnnn）`）と新しい形（4 桁、`（RFDnnnn）`）を読み、番号を値で比べる。課題の `rfd` 欄を `\d{4}_.+_10rfd` で確かめ、`phase` の値 `RFD` を受け付ける。wiki の節の見出しを `## RFD の一覧` に、行を `- [RFDnnnn]` から始まるリンクの行に、鍵を `unresolved_rfds` に、INDEX の列を `RFD` に、節 `## 課題の無い RFD` に改める。関数名と変数名の `prd` も `rfd` に改める（旧い記号を読む正規表現と、その注記の経緯の行を除く）。
- `reports/tools/stats.py`: 文書を `*_10rfd.md` と `\d{4}_` で集め、課題の `rfd` 欄を読み、関数 `rfd_state`・鍵 `rfd_count`・`rfds`・期間の完了数 `rfds_done` に改め、出力の文言を RFD にする。保管場所の文書の移した日を、改名をたどってパスが初めて `done/`・`frozen/`・`dropped/` の下に入ったコミットの日で取る（D25）。

### 6. ハーネスの文書と用語集（D14・D22）

- AGENTS.md の件名の形を `RFD番号` にし、例の `PRD110` などを 4 桁にする。避ける語の表には足さない（D9）。
- CONTEXT.md に RFD の項目を足し、`_Avoid_` に「PRD（本来は製品の要求書を指す。RFD0195 より前の記録とコミット件名には残る）」と書く。
- CHANGELOG の未リリースの節に、旧い名前と新しい名前の対応を 1 行で書く（経緯の行）。冒頭の説明と「v0.3.0 以降の非互換変更」の節の名前を直す。リリース済みの版の節は残す。
- memo README のファイル名の形と採番を 4 桁にし、採番の確かめで件名と過去のファイル名の両方の形（`PRDnnn` と `RFDnnnn`、3 桁と 4 桁）を見ることと、振り直しの行の 2 つの形を書く（経緯の行）。
- registry の成果物キー `rfd`、出力先 `docs/memo/{NNNN}_{slug}_10rfd.md`、`impl-plan` の出力先 `docs/memo/{NNNN}_{slug}_2{x}impl.md`、担当と読み替え表を直す。

### 7. 生成物と取り込みの記録（D20・D23）

- 課題の一覧と wiki の生成の節は `reports/tools/wiki_support.py maps` で作り直す（1 回目の状態で回し、2 回目のパッチにも 2 回目のあとに回した結果を入れる）。英訳・動線マップ・atlas・統計は作り直さない。
- 実装を始めた時点で wiki に取り込み済みだった材料のうち、名前だけが変わったものに、`anet-ingest-wiki` の手順で取り込みの行を足す（2 回目）。

### 8. 走っている bat と開発者の変更（D7）

- 12 番の bat は、実装を始めた時点で走っていた（キュー `mz_prd191`、Montezuma の Run）。走っている bat を書き換えると段が壊れるので、作業ツリーでは触らない。同じ置き換えを `.scratch/rfd0195/phase1.py --only apps/12_batch_run.bat` で、bat が終わってから 1 回目のコミットの前に当てる。キュー名とラベルは `mz_rfd0191` にし、コメントの番号を 4 桁にし、`OLD_EXE_SRC` は残す。
- 11 番の bat には開発者の未コミットの変更（キュー `mz_lrpl`）がある。直すのは `prd` を含むコメントの行だけで、開発者の変更と同じ hunk には入らない。1 回目のコミットの前に、開発者の変更を先にコミットするか、`anet-prepare-commit` の部分ステージで分ける。

## テスト

- Public interface / surface: `reports/tools/wiki_support.py` の `scan`・`maps` と、その中の件名の番号の読み方・振り直しの行の当て方・課題の欄の検査。`reports/tools/stats.py` の出力（`process` の `rfd_count`・`rfds`・`monthly`、`period` の `rfds_done`）。テストは `src/core/base/testdata/rfd0095/wiki_test.py`・`src/core/base/testdata/rfd0114/issues_test.py`（使い捨ての git リポジトリで道具を回す形）。C++ はテストのタグとテスト名、識別子、環境変数の名前を変えるだけで、振る舞いを変えない。
- 優先 behavior（縦スライスの順）:
  1. 件名の末尾の `RFDnnnn` から番号を読み、4 桁のファイル名 `NNNN_*_10rfd.md` の文書を wiki の「RFD の一覧」に載せる（tracer bullet）。
  2. 旧い記号 `PRDnnn` の件名からも番号を読み、値で同じ 4 桁の文書に当てる（`PRD195` と `RFD0195` は同じ）。scope と末尾の並びの両方から読む。`RFD195`・`PRD0195` は読まない。
  3. 旧い形の振り直しの行（3 桁）と新しい形（4 桁、`（RFDnnnn）`）を読み、行の日付と同じ日のコミットの旧い番号を行を持つ文書に当てる。何度も振り直した文書の旧い番号は行を持つ今の文書に当てる。当たる文書が 2 本以上の旧い番号は解決できない番号にする。
  4. 課題の `rfd` 欄が `\d{4}_.+_10rfd` の直下の文書を指すことを確かめ、`phase` の値 `RFD` を受け付ける。旧い `prd` 欄と `PRD` の値は受け付けない（今の検査のまま、知らない値として止まる・空と扱う）。
  5. 統計の保管場所の文書の移した日が、保管場所の中で改名した文書で、最初に保管場所へ入った日になる（D25）。
- TDD 順序: 上の 1〜5 の順に、テストを 1 本足して落ちることを確かめ、道具を最小に直して通し、次へ進む。改名と本文の置き換えは振る舞いを変えないので TDD の対象外とし、受入の検索（PRD §10.2）で確かめる。

## 検証

```powershell
.\.venv\Scripts\python.exe src\core\base\testdata\rfd0095\wiki_test.py
.\.venv\Scripts\python.exe src\core\base\testdata\rfd0114\issues_test.py
.\.venv\Scripts\python.exe reports\tools\wiki_support.py scan
.\.venv\Scripts\python.exe reports\tools\wiki_support.py maps
.\.venv\Scripts\python.exe reports\tools\stats.py --out .scratch\rfd0195\after\stats --since 2026-10-26 --no-cloc --no-blame --no-runs --quiet
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --preset x64-Debug && cmake --build --preset x64-Debug'
out\test\Debug\anet-core-base-test.exe "[workspace]"
```

- 受入 A1〜A13 は PRD §10.2 の検索を、使い捨ての clone の 1 回目と 2 回目のコミットのあとで回す。A6・A11 の実装の前の値は `.scratch/rfd0195/before/` に保存した（wiki の一覧の組 283 組・解決できない番号 0、全文書 196 本の起票日、統計の出力）。
- C++ のビルドは、テストのファイルの多くを再コンパイルする長いビルドで、実装を始めた時点で正式 Run が走っていた。開発者に確かめてから回す。

## 前提

- 実装を始める前の D7 の確かめ: (1) ほかの枠の作業は、開発者が寝る前にこの実装を起動したことと、作業ツリーの未コミットの変更が 3 件（11 番の bat、`data/workspaces/metrics_viewer_tag_sets.json`、未追跡の `docs/StampFly.md`）だけであることを根拠に、無いものとして進める。コミットの前に開発者が確かめる。(2) 11 番の bat の未コミットの変更は 8 のとおり扱う。ほかの触る場所には未コミットの変更が無い。(3) 最大の番号は 197、次は 198（2026-11-23。ファイル名と `git log --all` で確かめた）。(4) 12 番の bat が走っていたので、8 のとおり扱う。
- 件数の取り直し（2026-11-23）: `prd` を本文かパスに含むファイルは 1,221 本で、PRD の 1,220 本に、起草で足した ADR 0093（行 5）を加えた数と一致する。線引きの表の行ごとの件数も PRD と一致する。検証資材のフォルダは 41 個（PRD の 39 個のあとに 2 個増えた）。
- 2 回目のコミットの中身は、1 回目のコミットのあとに `.scratch/rfd0195/run_phase2.py` で作る。clone の 1 回目と 2 回目のコミットの差分を `.scratch/rfd0195/phase2.patch` に置き、照合に使う。
- 使い捨ての clone は `.scratch/rfd0195/clone/` に置き、作業ツリーのリポジトリの index・ref・作業ツリーには触らない。clone の中のコミットは確かめるためだけのもので、作業ツリーのリポジトリへ持ち込まない。
- Git が追跡しないもの（memory、`~/.claude/CLAUDE.md` の skill 名を名指す節、`cc-review` などユーザー単位の skill、`.claude/settings.local.json`、`docs/memo/prds.base`）は直さず、完了報告で合わせるかを聞く（D21）。

## 経過

- 2026-11-23 計画を保存した。PRD に暫定の印は無く、入口のレビューは不要。未決事項監査のユーザー判断のブロッカーは 0。読んだ観点: 実装 レビュー観点の「実装計画」。
- 2026-11-23 書く前に読んだ観点: 実装 レビュー観点の「名前」「コメントと TODO」「ログ」「設定ファイル」「テスト」「検証と報告」「skill」、AGENTS.md の「クリーンブレーク方針」「ビルド」「検証」「Git 操作・コミットメッセージルール」。
- 2026-11-23 実装の前の値を `.scratch/rfd0195/before/` に保存した（wiki の一覧の組 283 組と解決できない番号 0、全文書 196 本の起票日、統計の出力）。
- 2026-11-23 `.scratch/rfd0195/phase1.py` で 1 回目の状態を作業ツリーに当てた（改名 613 本、本文とリンク 819 本）。手で直したもの: memo README（4 桁、件名と過去のファイル名の両方の形、振り直しの行の 2 つの形）、registry の `{NNNN}`、実装 レビュー観点の項番 5、品質観点 `rfd.md` の項番 2・3、anet-create-issue の `[[NNNN_slug_10rfd]]`、wiki README の見本 `RFDnnnn`、CONTEXT.md の RFD の項目、CHANGELOG（冒頭・非互換の節・未リリースの節と対応の行）。
- 2026-11-23 道具を TDD で直した。1 周ごとにテストを足して落ちることを確かめ、道具を直して通した: (1) 件名の `RFDnnnn` と 4 桁のファイル名、(2) 旧い記号 `PRDnnn` を値で読む・scope と末尾の両方・`RFD195` と `PRD0195` を読まない、(3) 振り直しの行の 2 つの形・同じ日・何度も振り直した文書・桁を混ぜた行で止まる、(4) 課題の `rfd` 欄の 4 桁と `phase: RFD`・旧い `prd` 欄は空と同じ、(5) 統計の移した日を改名をたどって最初に保管場所へ入った日にする（D25）。既存のテストは新しい形へ移した。
- 2026-11-23 置き換えの不具合を 3 つ直した。(1) 名前の写しを順に当てて `0013_` が `00013_` になった（1 回の走査で写す形に直し、作業ツリーの 17 本・82 か所を直した）。(2) 課題の前付けの `docs` の項目のうち `docs/` で始まらない skill のパスを写していなかった（3 本）。(3) `rfd0084・092` のように続けて書いた番号と `PRD0xx` の見本を 4 桁にしていなかった。
- 2026-11-23 改名したファイルを指すのに実装の前から切れていたリンク（昔の skill `implement-prd-with-docs` を指すもの。0114・0120・done/0087）を、同じ文書の今のパス（`anet-impl-rfd`）へ直した。リンクのパスの直しなので 1 回目に入る。
- 2026-11-23 02:33 に、別の枠が `9183b046`（PRD197 のまとめページの見出し、件名は `RFD0197`）をコミットした。作業ツリーの RFD0195 の変更はそのコミットに混ざっていない。前提の D7 (1)（ほかの枠の作業が無い）は崩れていたので、コミットの前に開発者が確かめる。
- 2026-11-23 古い検証資材のフォルダ（`testdata/prd072`・`prd084`・`prd191`）に、Git が無視する `__pycache__` だけが残っていたので消した。
- 2026-11-23 レビューは Codex 側で 8 往復し、8 往復目で OK（下の「レビュー」）。課題の判定を書いた（fe09dc 残件無し、529662 残件有り）。工程末の起票の候補を示して止まった（候補: 実装の前からのリンク切れの整理、既存の文書の直訳の語、テストの重なる文字列の定数化、`check_default_leaves.py` が存在しない `core` を渡して止まる件、受入 A2・A3・A5・A13 の条件の改訂、PRD187 の課題の状態）。
- 2026-11-23 開発者が、候補を推奨どおり記録すると選んだ。新規に 4 件を起票した（リンク切れ [issue-20261123-98baff](../management/issues/issue-20261123-98baff.md)、直訳の語 [issue-20261123-c7c3ae](../management/issues/issue-20261123-c7c3ae.md)、テストの重なる文字列 [issue-20261123-b1205d](../management/issues/issue-20261123-b1205d.md)、受入の条件 [issue-20261123-2338c0](../management/issues/issue-20261123-2338c0.md)）。`check_default_leaves.py` の件は、新規の推奨だったが、同じ原因・同じ直し場所の [issue-20261122-4adabb](../management/issues/issue-20261122-4adabb.md) が既にあったので、その事象へ追記した。PRD187 の課題は [issue-20261103-23d185](../management/issues/issue-20261103-23d185.md) の事象と対応方針へ追記した。リンク切れの件数は、検査の出力の行（270・260）ではなく件数（268・258）に直した。起票した 4 本は 1 回目のコミットに入れる（`.scratch/rfd0195/commit1_paths.txt` に足した）ので、1 回目の新しいファイルは実装計画と合わせて 5 本になる（A11 の c18 の結果は 1 本）。課題の一覧と wiki の課題の区画は、1 回目の予定の件名と対象のファイルを渡して作り直した。
- 2026-11-23 開発者が、割り込みのコミット `9183b046` は把握済みで問題が無いとし（D7 (1) の確かめ）、12 番の bat に置き換えを当てた（変わった行 10。キュー名 `mz_rfd0191`、コメントの番号。`OLD_EXE_SRC` の 1 行は D5 で残る）。C++ の Debug ビルドとテストをこの会話で回してよいとした。D21 は「合わせる」を選んだ（`docs/memo/prds.base` は開発者が直した）。
- 2026-11-23 D21 の名前合わせを Git の外で行った（ユーザー単位の指示と skill の名前・置き場の形、memory の skill 名・文書のパス・検証資材のフォルダ・テストのタグ）。`.claude/settings.local.json` は、旧い名前を含むのが過去に一度だけ許したコマンドの許可 6 件だけなので変えなかった。
- 2026-11-23 レビューのあとに置き換え漏れを 1 か所見つけて直した。anet-review の SKILL（`.agents`・`.claude` の 2 本）の 126 行で、置き場の形 `.scratch/rfdNNNN/` の説明が `NNN` のままだった（3 桁の見本の `NNN` は、`_` が続く形と `{NNN}` だけを 4 桁にしていた）。1 回目のコミットに入る。同じ形の残りは、現用の文書・skill・品質観点・プロセス・道具で 0 件。
- 2026-11-23 Debug の構成とビルド、テストの実行体のタグ、テストの実行体の全部と既定のスイート、Runner の GUI テストを回した（結果は「検証の結果」の A9）。落ちたのは、開発者の作業中の 11 番の bat の変更による `batchrun_fatal_error_handling_test.ps1` と、imagecls（Env）のテストの並びの依存によるダイアログでの停止で、どちらも RFD0195 の変更によるものではないことを確かめた。この 2 つを工程末の起票の候補として示し、開発者の選択で新規に起票した（[issue-20261123-9a0e78](../management/issues/issue-20261123-9a0e78.md)・[issue-20261123-c455b5](../management/issues/issue-20261123-c455b5.md)）。2 本も 1 回目のコミットに入れる（`.scratch/rfd0195/commit1_paths.txt` に足し、課題の一覧を作り直した）ので、1 回目の新しいファイルは実装計画と合わせて 7 本になる。
- 2026-11-23 開発者が 1 回目と 2 回目をコミットした（1 回目の件名は、AGENTS.md の決まりに合わせて 2 文に分けた）。2 回目は `run_phase2.py` で作り、c18 の 2 回目と同じ 160 本で、中身の違いは 1 回目のあとの直しの分だけだった。実リポジトリで受入を確かめ直した（「検証の結果」の「実リポジトリでの確かめ」）。未追跡の `docs/StampFly.md` に、旧いファイル名を指すリンクが 11 本あり切れたことを開発者に伝えた。
- 2026-11-23 開発者の指示で、未追跡の `docs/StampFly.md` を新しい名前に合わせた（切れたリンク 11 本と、本文の「PRD」の呼び名 32 か所）。
- 2026-11-23 開発者が、受入 A2・A3・A5・A13 に残るもの（実装の前からのリンク切れ、名前を残すファイルを指すリンク、過去の Run 名）を受入の対象外とし、RFD0195 を閉じると決めた。issue-20261121-529662 を残件無し・解決にした（fe09dc は 1 回目のコミットの準備で解決済み）。リンク切れは [issue-20261123-98baff](../management/issues/issue-20261123-98baff.md) で扱う。

## 検証の結果

使い捨ての clone（`.scratch/rfd0195/c18`。HEAD `9183b046` から作り、作業ツリーの中身を写した）で、1 回目と 2 回目をコミットして確かめた。2 回目は `run_phase2.py` と同じ順（`phase2.py`、予定の件名を渡した `wiki_support.py maps`、`d23.py`）で作った。改めて確かめる時点は、実リポジトリで 1 回目と 2 回目をコミットしたあとである（コミットは人間が行う）。

| 受入 | 判定 | 結果 |
|---|---|---|
| A1 | 満たす | `git ls-files \| grep -i prd` が ADR 3 本・実験記録 38 本・Run の設定の写し 10 本の 51 本だけ |
| A2 | 満たさない | (1) 「リンクの注意」の検査が空でない。実装の前から 268 件のリンク切れがあり（改名していないファイルの古いパスなど）、本 RFD で 10 件減って 258 件（検査の出力の行の数は、リンク先の中の改行で割れた 2 行を含むので 270 行と 260 行）。空にするには本 RFD と関係の無いリンク切れを直すことになり、範囲（名前と番号の形）を超える（[issue-20261123-98baff](../management/issues/issue-20261123-98baff.md)・[issue-20261123-2338c0](../management/issues/issue-20261123-2338c0.md)）。(2) リンクの丸括弧の中に `prd` が残るのは、A1 の 51 本を指すものと、追跡していないビルド出力（`apps/metrics-viewer/target/prd089-validation/`・`out/test-tmp/prd069/`。保管場所の実装計画の中の 17 本）。改名したファイルではないので移す先が無い。(3) Obsidian のリンクは、動線マップ（`docs/agents/issue-tracker.md`。D20 で次の定常整備に作り直す）の見本、本 RFD の本文（D24）、保管場所の文書の memory の名前の 3 つ。見出しの `#` は 2 回目のあとも当たる（下の A13） |
| A3 | 満たさない | 470 行が残る。(a)〜(f) に当たらないのは、(g) 名前を残すファイル（A1 の 51 本）を指すリンクのパスと過去の Run 名の 3 行（Run のフォルダと ADR は P4・D15 で名前を残す）と、(h) 12 番の bat の 11 行（bat が終わってから当てると 0 になる）。(g) は RFD の種類を足す判断が要る（[issue-20261123-2338c0](../management/issues/issue-20261123-2338c0.md)） |
| A4 | 満たす | 旧い欄と値は 0。題と scope の `prd` は issue-20261121-529662 の 2 行だけ（D24）。前付けに `prd` 欄を持たない課題が 1 件（issue-20261115-d3e094。実装の前から持たない） |
| A5 | 満たさない | 56 行が残る。引用・D5・経緯の行・D24 に当たらないのは、名前を残す ADR を指すリンクと ADR の名前の 9 行と、過去の Run 名の 1 行（A3 の (g) と同じ）。A5 の検索はリンクの丸括弧の中も拾うので、A2 が許す ADR へのリンクも挙がる（[issue-20261123-2338c0](../management/issues/issue-20261123-2338c0.md)） |
| A6 | 満たす | 実装の前の 283 組がすべて残り、増えたのは本 RFD のコミットが触ったページの（ページ、195）の 15 組だけ。解決できない番号は前後とも 0 |
| A7 | 満たす | 1 回目のあとも 2 回目のあとも、道具のテスト（`wiki_test.py` 20 件、`issues_test.py` 34 件）と `scan`・`maps` が通る。2 回目のコミットのあとにもう一度 `maps` を回しても変わらない |
| A8 | 満たす | `anet-create-rfd`・`anet-impl-rfd`（`RFD-IMPLEMENTATION.md`）があり、旧いディレクトリは無い。`diff -rq .agents/skills .claude/skills` は何も出さない |
| A9 | 満たす | 2026-11-23 に作業ツリー（1 回目の状態。2 回目は C++ に触れず、検証資材の `verify.ps1` の関数名だけを変える）で、Debug の構成とビルドが通った（107 工程、エラー 0。既定で作らない `AnetRLRunner-gui-test` も 19 工程で通った）。テストの実行体 10 本の `--list-tags` に `prd` を含むタグは 0。AGENTS.md の検証の既定のスイート（`[workspace]`、PowerShell 2 本、Python 7 本）は、`batchrun_fatal_error_handling_test.ps1` のほかは通った。この 1 本は、作業ツリーの 11 番の bat の開発者の作業中の変更（キュー `mz_lrpl` の、前の Run の重みを読むだけの評価の段が、テストの仮の環境で失敗に数えられる）で落ちた。HEAD の 11 番の bat に RFD0195 のコメントの置き換え（6 行）だけを当てた形と、今の 12 番の bat では通った（[issue-20261123-c455b5](../management/issues/issue-20261123-c455b5.md)）。あわせて全部のテストの実行体を回した（下の「実行できなかった検証」の前の段落） |
| A10 | 満たす | CONTEXT.md の RFD の項目と `_Avoid_` の経緯の行、CHANGELOG の対応の行がある。CHANGELOG の変更は 0.4.0 の節まで（0.3.0 の見出しより前だけ）。ADR 0093 がある |
| A11 | 満たす | 1 回目のステージで改名 612 本がすべて `R`（似ている度合いの最低は 51%）、`D`＋`A` は 0（新しいファイルは実装計画 1 本）。コミットのあとの `git log -1 -M --name-status` も同じで、改名の検出を省いた警告は出ない。全文書 196 本の起票日が実装の前と一致（c11 で確かめた。c18 も改名は同じ）。統計の文書ごとの `created`・`moved` は、P38 の 8 本の `moved` が最初に保管場所へ入った日に変わったほかは一致し、月ごとの数はその分だけ動き（2026-08 の frozen 1→6・dropped 0→1、2026-11 の frozen 5→0・dropped 2→1）、リードタイムと期間の完了数（54）は一致 |
| A12 | 満たす | `docs/memo` の下の追跡しているファイルは README を除いてすべて 4 桁の番号で始まる（未追跡の `prds.base` は D21 で対象外） |
| A13 | 満たさない | 1 回目だけの状態で、既定のスイート（C++ のビルドとテストは A9）と道具のテストと `scan`・`maps`・`stats.py` は通る。skill の前付けの名前は 4 本とも新しい名前で、本文の成果物キー・課題の欄・ファイル名の形も 1 回目で新しい名前になる。品質観点の項番 2 の形が `^\d{4}_.+_10rfd\.md$`、`_10rfd.md` の番号の重なりは 0。見出しを指す `#` のうち `rfd`・`prd` を含む 10 本が、1 回目と 2 回目のどちらでも見出しに当たる（`anchors_check.py`）。「リンクの注意」の検査が空でないのは A2 の (1) と同じ。新しい会話での skill の一覧は開発者が確かめる |

- 作業ツリーでも道具のテストを通した（`wiki_test.py` 20 件、`issues_test.py` 34 件）。`wiki_support.py maps` を作業ツリーで回すと、改名がまだ履歴に無いので、検証資材のフォルダだけを触ったコミットの番号が 1 組（325_muzero の 122）欠ける。1 回目のコミットのあとの maps で戻る（clone で確かめた）。
- 取り込み待ちのまま改名した材料は、点検で「更新あり」ではなく「未取り込み」と出る（旧いパスの記録が当たらないため）。点検の対象の集合は実装の前と同じである。D23 の記録は 146 本に書き、1 本（0114。昔の skill を指す切れたリンクを手で直した）は照合で外れて点検に回る。
- 改行: 変えたファイルのうち `.bat` 以外は作業ツリーをすべて LF にした（実装の前から作業ツリーが CRLF だった 169 本を LF に直した。index は LF のままで、Git の差分には出ない）。
- 出荷設定（`Atari.txt`・`DropMerge.txt`・`LunarLander.txt`）はコメントの行だけを変えた（`#` で始まらない行の差分は 0 行）。項番 52 の `check_default_leaves.py` は、実装の前から `rg` に存在しない `core` を渡して止まる（PRD182 で置き場を移したあと直っていない。[issue-20261122-4adabb](../management/issues/issue-20261122-4adabb.md) へ追記した）ので回せなかった。全キー・文字列値の比較も、コメントだけの変更なので回していない。

### 実リポジトリでの確かめ（2 回のコミットのあと）

2026-11-23 に、開発者が 1 回目と 2 回目をコミットしたあとの実リポジトリで、受入を回し直した（`.scratch/rfd0195/accept_main.sh`、出力は `.scratch/rfd0195/accept_main/`）。A13 は、1 回目のコミットだけの状態を使い捨ての clone（`.scratch/rfd0195/c19`）に取り出して回した。判定は上の表（c18）と同じで、数の違いはどれも理由が分かっている。

開発者は、A2・A3・A5・A13 に残るもの（実装の前からのリンク切れ、名前を残すファイルを指すリンク、過去の Run 名）を受入の対象外とし、RFD0195 を閉じた（下の「経過」）。

- A1: 51 本（ADR 3 本・実験記録 38 本・Run の設定の写し 10 本）。満たす。
- A2: リンクの検索は 76 件・Obsidian 3 件・見出しのリンク 7 件で c18 と一致し、見出しのリンク 10 本はすべて見出しに当たる。リンク切れの検査は 246 件。c18 の 258 件との差は、検査が置き場の外のファイルの有無で判定するためで（実リポジトリには `.scratch` の材料と未追跡の `docs/StampFly.md` があり、c18 には無い。`docs/wiki/README.md` の `../../../README.md` は c18 の置き場の外にたまたまファイルがあった）、RFD0195 による新しいリンク切れは無い。ほかに、未追跡の `docs/StampFly.md` が旧いファイル名を指すリンク 11 本が切れた（追跡していないので受入の検索の外）。満たさない（[issue-20261123-98baff](../management/issues/issue-20261123-98baff.md)・[issue-20261123-2338c0](../management/issues/issue-20261123-2338c0.md)）。
- A3: 460 行。c18 との差は、12 番の bat の 10 行が当てたあとに消えたことと、wiki の課題の行の中身の変化だけ。12 番の bat に残る 1 行は D5 の `OLD_EXE_SRC`（(b) に当たる）。(a)〜(f) に当たらないのは (g) の 3 行で、満たさない（[issue-20261123-2338c0](../management/issues/issue-20261123-2338c0.md)）。
- A4: 旧い欄と値 0、`docs` の旧い名前 0、題と scope は issue-20261121-529662 の 2 行だけ。満たす。
- A5: 56 行で c18 と一致。満たさない（[issue-20261123-2338c0](../management/issues/issue-20261123-2338c0.md)）。
- A6: 実装の前の 283 組がすべて残り、増えたのは（ページ、195）の 15 組だけ。解決できない番号は前後とも 0。満たす。
- A7: 道具のテスト（20 件・34 件）、`scan`・`maps` が通り、`maps` は何も変えない。満たす。
- A8: 新しい skill のファイルがあり、旧いディレクトリは無く、`diff -rq` は何も出さない。満たす。
- A9: 作業ツリーで確かめた結果（上の表）。2 回目のコミットは C++ に触れない。満たす。
- A10: CONTEXT.md の RFD の項目と `_Avoid_` の経緯の行、ADR 0093 がある。CHANGELOG の差分の hunk は 197 行までで、0.3.0 の見出し（199 行）より前だけ。満たす。
- A11: 1 回目のコミットは改名 612（似ている度合いの最低 51%）・追加 7・削除 0・変更 482。全文書 196 本の起票日が実装の前と一致。統計の文書ごとの値（RFD の数・移した日・起票日・リードタイム）は c18 と一致。満たす。
- A12: 4 桁で始まらないファイル 0、`_10rfd.md` の番号の重なり 0、品質観点の項番 2 の形は `^\d{4}_.+_10rfd\.md$`。満たす。
- A13: 1 回目だけの状態で、道具のテスト・`scan`・`maps`・統計が通り、skill の前付けの名前は 4 本とも新しい名前。`maps` は 325_muzero の RFD0122 の行を 1 行戻す（2 回目に入った戻りと同じ）。リンク切れの検査が空でないので、満たさない（A2 と同じ）。

- テストの実行体（2026-11-23、Debug、隠したテストを除く）: dqn 379 件、imagecls（algo）29 件、muzero 7 件、core-base 664 件（ほかに CUDA がある環境では飛ばすテスト 1 件）、atari 56 件、dropmerge 33 件、gridmaze 12 件、lunarlander 8 件、Runner の GUI テスト 42 件がすべて通った。imagecls（Env）は、Catch2 の乱数の並び（seed 1094244747）で 2 件目の「ImageDatasetManager keeps manifest construction failures sticky」が wx のエラーのダイアログを開いたまま止まり、プロセスを止めた。このテストは画像の口の初期化（`EnsureWxImageSupport`）を呼ばず、先に回ると PNG を書けない（単独でも同じ箇所で落ちる）。RFD0195 がこのファイルで変えたのはタグの名前だけで、実装の前からある並びの依存である（[issue-20261123-9a0e78](../management/issues/issue-20261123-9a0e78.md)）。宣言順では 20 件すべて通った。出力は `.scratch/rfd0195/tests-20261123/`。

### 実行できなかった検証

- 新しい会話で `/anet-create-rfd`・`/anet-impl-rfd` が skill の一覧に出ること（A13 の人が確かめる項）。この会話の途中で、一覧が新しい名前で読み直されたことは見えた。
- 項番 52 の `check_default_leaves.py`（上の理由）。

### レビューで見送った指摘と理由

- 項番 76（直訳の語）: 名前を置き換えた既存の行のうち 134 行に、実装の前からある避ける語（「既定葉」「腕」「旗」など）が当たる。本 RFD は名前と番号の形だけを変え、文書の中身は変えない（RFD0195 §1.3 の非ゴール）ので、語は直さない。工程末に別の課題の候補として示し、[issue-20261123-c7c3ae](../management/issues/issue-20261123-c7c3ae.md) に記録した。
- 項番 66・72（リテラルの定数化）: `RFD0094_UI_DARK`（環境変数の名前。テスト 3 か所）と `rfd0099/frames_210x160.u8`（2 か所）は、実装の前から同じ文字列を重ねて書いていたもので、名前だけを置き換えた。定数にするのはコードの形を変える別の関心で、ビルドを回せない今は入れない。工程末に候補として示し、[issue-20261123-b1205d](../management/issues/issue-20261123-b1205d.md) に記録した。

## 決めたこと（実装中）

- コミットは人間が行うので、作業ツリーを 1 回目の状態にし、2 回目の中身は 1 回目のあとに `run_phase2.py` で作る。2 回目の生成の節と D23 の記録は、そのときの履歴と blob で作り直す必要があるので、パッチを当てる形にしなかった（`phase2.patch` は照合用）。1 回目は `commit1_paths.txt` のパスだけをステージし、`apps/11_batch_run.bat` は開発者の変更と hunk で分ける（`.scratch/rfd0195/README.md`）。
- 1 回目に入れる動作に効く中身に、skill と品質観点の本文の小文字の `prd` で書く契約の名前（成果物キー、課題の欄、ファイル名の形）と、ファイル名の見本の 4 桁（`NNNN_slug`）を足した。1 回目のあとも、skill の手順がレジストリと課題の検査と食い違わないためである（レビューの指摘）。語の「PRD」は 2 回目に直す。
- 受入 A3 の種類に、RFD0195 の (a)〜(f) に無い「名前を残すファイル（A1 の 51 本）を指すリンクのパスと過去の Run 名」を足した。A2 の「A1 の 51 本を指すリンク」と同じ扱いである。RFD を改めるかは開発者が決める（未決）。
- D24 の旧い名前そのものを話題にする記述に、昔の skill の名前 `implement-prd-with-docs` と、昔の表記の例を引いた「」（「PRD 930」「PRD」「暫定 PRD」など）を足した。PRD131 の本文は、振り直しの行の形 `（PRDnnn）` の説明に加え、当時の題と件名の引用（「PRD918:」など）を残した（D3 の引用）。
- D5 の `.scratch/` のパスは、用途で決めた。読む入力と過去の証跡（今ある材料）は旧い名前のまま残し、これから書き出す先（`--report`・`--output` などの出力の引数、スクリプトの既定の出力先、手順の中の保存先、同じ手順で書いたものを後で読む参照）は `.scratch/rfdNNNN` へ直した。機械的には、形で決まるもの（過去の出力の `old/`、変更の前に採った基準のパスの中の `before/`、前の版の実行体の `bin/Release`、入れた道具の `tracy-`・`sdl2`、日付つきの採取の `-YYYYMMDD`（パスのどの段でも）、移して追跡した採取の道具 `stage0.py`）を残し、形で決まらないものは 1 件ずつ用途を確かめて、残すパスをファイルごとの一覧（`rfdlib.py` の `FILE_PROTECT`）に書いた。一覧に入れたのは、検証資材の README 11 本（採取済みの入力、保存済みの検証結果・ログ・証跡・設定の写し、旧い実行体とそのタグの組など）、`rfd0084` の集計スクリプトが `--followup` で読む保存済みの比較基準、直下の文書 4 本（RFD0124 の再現ログ、RFD0125 の保存済みの基準、RFD0131 の当時の出力先 `prd918`、RFD0197 の過去の調べとまとめページ）である。1 つのコマンドの中は引数ごとに分けた（`rfd0061` の比較は入力 2 つを旧い名前、`--report` を新しい名前にした）。保存済みの入力を読む先と、新しく書き出す先が同じ置き場の根を兼ねていた検証スクリプト 2 本は、読む側を旧い置き場、書く側を新しい置き場に分けた（置き換えのあとに当てる手直し `MANUAL_EDITS`）。`rfd0084` の `summarize_forward_refactor.py` は変更前の実行体を保存済みの `forward-refactor/before` から読み、採取（`run_forward_refactor.ps1`）は新しい `forward-refactor/after` へ書く。`rfd0103` の `verify_config_migration.py` は、リポジトリのどの段も作らない採取済みの設定の写し `config-{before,after}` を旧い置き場から読み、比較の結果を新しい置き場へ書く（書く前に置き場を作る）。作業ツリーで、新しい置き場が無い状態から回し、保存済みの 8 組を読んで意図しない差 0、終了値 0 で `.scratch/rfd0103/config-comparison.json` を書いた。ほかの検証資材のスクリプトの Run 名の型・指標の tag・プロファイルの名前・環境変数と既定の出力先は 4 桁に直したので、スクリプトを回し直すと新しい名前の置き場へ書き、同じ根の下に保存した過去の採取を読む段は、先に採取の段から回し直す（各機の手元のフォルダは動かさない。D5）。確かめ: `d5_table.py` で、現用の文書（検証資材の README の過去の実行結果の節を除く、直下の文書、課題、wiki など）の `.scratch` の参照 151 件に用途を付けた照合表（`.scratch/rfd0195/d5_table_c18.tsv`）を作った。波括弧の形（`{before,after}`）、番号のあとに名前が続く置き場（`prd197-grill`）、置き場の根だけの参照も拾い、旧い置き場の実在を見る。新しい名前にした 54 件のうち 52 件は書き出す先（出力の引数 9 件のほかは 1 件ずつ確かめた判定を表に書いた）、1 件は実行する道具（本 RFD の作業用の置き場）、1 件は本 RFD の本文である。旧い名前のまま残した 97 件は、読む入力・過去の証跡 86 件（形で決まるものと一覧で残したもの。どれも旧い置き場に実在する）、残す記録 9 件（課題の本文と冒頭の引用行）、本 RFD の本文 2 件である。あわせて `script_refs.py` で、検証資材のスクリプトとテストの中の新しい名前の `.scratch` の文字列のうち旧い置き場が実在するものを並べ、どれも既定の出力先であることを確かめた。
- 検証資材の README の、過去の実行結果の節は、見出しに「結果」「実施記録」「実行記録」「履歴」「以前の記録」を含む節（題の行を除く）とした（一覧は下）。
- 直下の文書（D3）の「引用」は、`>` の行、発言を引いた「」の中（同じ行に「発言」「述べ」などがある）、件名の引用（バッククォートの中とコードの枠の中の件名の形）、昔の表記の例を引いた「」、PRD131 の当時の題の引用とした。それ以外の「」の中の `PRD` は直した。
- D23 の記録は、実装の前に取り込み済みで、実装の前の本文に改名の規則（リンクのパス、1 回目と 2 回目の置き換え）だけを当てた本文と今の本文がちょうど一致する材料にだけ書く（`d23.py`）。別の文書へのリンクの差し替えや、規則のほかの直しがあれば記録せず、点検に任せる。
- 2 回目の生成の節は、予定の件名と変えたファイルを `wiki_support.py maps` に渡して作る（`anet-prepare-commit` の手順 8 と同じ）。

## レビュー

- 1 往復目（2026-11-23）: Claude 側（`cc_review.py`）は「クレジットの残高が足りない」で失敗し、Codex 側だけで受けた。判定は差し戻し（実装計画 2 件、実装 6 件）。直したもの: 受入ごとの判定の記録、1 回目のステージの手順とパスの一覧、1 回目の skill の契約の名前、2 回目の見本の 4 桁、道具のテストの 3 桁の fixture、今ある材料を指すパスの戻し、生成の節に予定の件名を渡すこと、D23 の照合、改行、出荷設定の検査の記録。見送ったもの: 項番 76・66・72（上の「レビューで見送った指摘と理由」）。原文は `.scratch/rfd0195/review-20261123-0340-codex.md`。
- 2 往復目（2026-11-23、Codex 側）: 差し戻し（実装計画 2 件、実装 2 件、範囲外 1 件）。直したもの: 今ある材料のパスを実在で戻す形をやめ、形で決める規則にした（出力先と手順の行がスクリプトとそろう）、D23 の照合を改名の規則を当てた本文との一致にした、見送りの記録の環境変数の名前（`RFD0094_UI_DARK`）、311 のトピック別まとめの成立の経緯の 3 行（置き換えで「呼び名を RFD から RFD に」となっていたのを、旧い名前を残した経緯の行に戻し、同じ行に RFD0195 を書いた）。受入 A3・A5 の一覧と判定は c11 で取り直した。原文は `.scratch/rfd0195/review-20261123-0408-codex.md`。
- 3 往復目（2026-11-23、Codex 側）: 差し戻し（実装計画 1 件、実装 1 件）。形だけで決めると、採取済みの入力や過去の証跡（`rfd0061` の比較の入力、`rfd0122` の `--baseline`、`after-verified`・`baseline-config`・過去のログ、`rfd0099` の旧い実行体とタグ）まで新しい名前になった。用途を 1 件ずつ確かめて残すパスの一覧（`FILE_PROTECT`）を足し、`old` の形をフォルダだけに絞った（PRD131 の手順の `old.md` は書き出す先）。受入と一覧は c13 で取り直した。原文は `.scratch/rfd0195/review-20261123-0426-codex.md`。
- 4 往復目（2026-11-23、Codex 側）: 差し戻し（実装計画 1 件、実装 1 件、範囲外 1 件）。RFD0103 の保存済みの smoke の結果と生成済みの入力、`rfd0084` の集計スクリプトが読む保存済みの比較基準を、旧い名前に戻した（`FILE_PROTECT`）。照合表に用途の列を付けた（`d5_table.py`）。受入と一覧は c14 で取り直した。原文は `.scratch/rfd0195/review-20261123-0451-codex.md`。
- 5 往復目（2026-11-23、Codex 側）: 差し戻し（実装計画 1 件、実装 2 件、範囲外 1 件）。照合表が波括弧の形と名前の続く置き場を取りこぼし、新しい名前の参照を一律に書き出す先としていた。照合の道具を直して全件を挙げ直し、新しい名前の参照はすべて 1 件ずつ判定を表に書いた。rfd0103 の保存済みの検証結果、RFD0124 の再現ログ、RFD0197 の過去の調べとまとめページ、RFD0131 の当時の出力先を旧い名前に戻し、rfd0084 の forward-refactor の置き場を丸ごと旧い名前にした（D5 の例外）。受入と一覧は c16 で取り直した。原文は `.scratch/rfd0195/review-20261123-0504-codex.md`。
- 6 往復目（2026-11-23、Codex 側）: 差し戻し（実装計画 2 件、実装 2 件）。`rfd0084` の例外（置き場を丸ごと旧い名前）は、保存済みの実行体との照合で採取が止まり、A3 の判定ともそろわなかったのでやめ、集計が保存済みの `before` を読むようにスクリプトを直した。`rfd0103` の比較が採取済みの設定の写しを旧い置き場から読むようにした。日付つきの形をパスの途中の段にも当てた。照合表で、実行する道具と、置き場の根だけの参照の実在を区別した。受入と一覧は c17 で取り直した。原文は `.scratch/rfd0195/review-20261123-0526-codex.md`。
- 7 往復目（2026-11-23、Codex 側）: 差し戻し（実装計画 1 件、実装 1 件）。照合表のパスの切り出しが、パスに続く日本語まで含めて実在を誤っていたので、ASCII でない字で止まるようにした。`rfd0103` の比較が新しい置き場を作らずに書こうとしていたので、書く前に作るようにし、作業ツリーで回して確かめた。受入と一覧は c18 で取り直した。原文は `.scratch/rfd0195/review-20261123-0546-codex.md`。
- 8 往復目（2026-11-23、Codex 側）: 実装計画・実装とも OK。原文は `.scratch/rfd0195/review-20261123-0603-codex.md`。
- 課題の判定: issue-20261116-fe09dc（4 桁の形）は残件無し。issue-20261121-529662（呼び名の改名）は残件有り（12 番の bat の直しと 2 回目の中身は 1 回目のコミットの前後に当てる段が残り、C++ のビルドと既定のスイートが未確認で、受入 A2・A3・A5・A13 に満たさない項がある）。

### A3 の残す行の一覧（2 回目のあと。c18 の行番号）

- a 経緯の行: 30 行
  - CONTEXT.md: 778
  - docs/memo/README.md: 16・20
  - docs/wiki/apps/110_runner_app.md: 255
  - docs/wiki/apps/120_metrics_viewer.md: 211
  - docs/wiki/categories/311_dev_tools.md: 46・48・49・229・238
  - docs/wiki/categories/312_util.md: 116
  - docs/wiki/categories/314_tensor.md: 87
  - docs/wiki/categories/315_nn.md: 156
  - docs/wiki/categories/316_config.md: 106
  - docs/wiki/categories/317_metrics.md: 104
  - docs/wiki/categories/318_rl_core.md: 217
  - docs/wiki/categories/319_gui.md: 81
  - docs/wiki/categories/320_env_core.md: 77
  - docs/wiki/categories/321_agent_core.md: 273
  - docs/wiki/categories/322_env_individual.md: 145
  - docs/wiki/categories/324_dqn.md: 299
  - docs/wiki/categories/325_muzero.md: 59
  - reports/tools/wiki_support.py: 184・192・206・210
  - src/core/base/testdata/rfd0095/wiki_test.py: 179・276・277・284
- b 今ある材料を指すパス: 80 行
  - apps/11_batch_run.bat: 50
  - src/core/base/testdata/rfd0061/README.md: 22・31
  - src/core/base/testdata/rfd0084/README.md: 37・44・83・132・181・201・211・217・223
  - src/core/base/testdata/rfd0084/run_forward_refactor.ps1: 15・150・151・155・181
  - src/core/base/testdata/rfd0084/run_review_validation.ps1: 20・23・25・97
  - src/core/base/testdata/rfd0084/summarize_forward_refactor.py: 14・111
  - src/core/base/testdata/rfd0084/summarize_review_validation.py: 46
  - src/core/base/testdata/rfd0092/README.md: 30・162
  - src/core/base/testdata/rfd0094/README.md: 8・14
  - src/core/base/testdata/rfd0094/compare_configs.py: 9
  - src/core/base/testdata/rfd0094/run_validation.py: 15
  - src/core/base/testdata/rfd0097/README.md: 13
  - src/core/base/testdata/rfd0097/run_validation.py: 15
  - src/core/base/testdata/rfd0098/README.md: 5
  - src/core/base/testdata/rfd0098/run_validation.py: 16
  - src/core/base/testdata/rfd0099/README.md: 24・27・31・69・75・81
  - src/core/base/testdata/rfd0099/run_validation.py: 28
  - src/core/base/testdata/rfd0102/README.md: 31
  - src/core/base/testdata/rfd0103/README.md: 24・30・48・80・86・88
  - src/core/base/testdata/rfd0103/verify_config_migration.py: 10
  - src/core/base/testdata/rfd0105/README.md: 17・26・32
  - src/core/base/testdata/rfd0105/run_validation.py: 51
  - src/core/base/testdata/rfd0109/README.md: 9・16・36・46
  - src/core/base/testdata/rfd0110/README.md: 79
  - src/core/base/testdata/rfd0115/verify_memory.py: 106
  - src/core/base/testdata/rfd0116/README.md: 11・31
  - src/core/base/testdata/rfd0116/run_validation.py: 29
  - src/core/base/testdata/rfd0117/README.md: 119・127
  - src/core/base/testdata/rfd0117/run_atari_display_test.py: 16
  - src/core/base/testdata/rfd0121/README.md: 9・13
  - src/core/base/testdata/rfd0122/README.md: 10
  - src/core/base/testdata/rfd0123/README.md: 11
  - src/core/base/testdata/rfd0123/compare_config.py: 5
  - src/core/base/testdata/rfd0123/run_validation.py: 11
  - src/core/base/testdata/rfd0127/README.md: 9
  - src/core/base/testdata/rfd0127/run_validation.py: 14
  - src/core/base/testdata/rfd0127/verify_configuration.py: 9
  - src/core/base/testdata/rfd0130/README.md: 17
  - src/core/base/testdata/rfd0130/run_performance.py: 639
  - src/core/base/testdata/rfd0170/README.md: 33
  - src/core/base/testdata/rfd0191/README.md: 10・11
  - src/core/base/testdata/rfd0191/measure_rollout.py: 127
- c 旧い記号を読む道具とテスト: 35 行
  - reports/tools/wiki_support.py: 186・187・208
  - src/core/base/testdata/rfd0095/wiki_test.py: 181・275・283・285・294・299・303・304・307・308・309・319・321・349・360・371・383・384・395・396・398・412・413
  - src/core/base/testdata/rfd0114/issues_test.py: 260・263・264・265・266・549・552・554・556
- e 生成の節に写る保管場所の文書の題: 132 行
  - docs/wiki/apps/110_runner_app.md（26 行。ファイル単位）
  - docs/wiki/apps/120_metrics_viewer.md（1 行。ファイル単位）
  - docs/wiki/categories/311_dev_tools.md（8 行。ファイル単位）
  - docs/wiki/categories/312_util.md（6 行。ファイル単位）
  - docs/wiki/categories/314_tensor.md（3 行。ファイル単位）
  - docs/wiki/categories/315_nn.md（6 行。ファイル単位）
  - docs/wiki/categories/316_config.md（6 行。ファイル単位）
  - docs/wiki/categories/317_metrics.md（2 行。ファイル単位）
  - docs/wiki/categories/318_rl_core.md（20 行。ファイル単位）
  - docs/wiki/categories/319_gui.md（5 行。ファイル単位）
  - docs/wiki/categories/320_env_core.md（5 行。ファイル単位）
  - docs/wiki/categories/321_agent_core.md（19 行。ファイル単位）
  - docs/wiki/categories/322_env_individual.md（6 行。ファイル単位）
  - docs/wiki/categories/324_dqn.md（16 行。ファイル単位）
  - docs/wiki/categories/325_muzero.md（3 行。ファイル単位）
- f README の過去の実行結果の節: 59 行
  - src/core/base/testdata/rfd0090/README.md: ## 5. 今回の結果（節単位）
  - src/core/base/testdata/rfd0092/README.md: ## 初回受入の実行結果（2026-10-15）（節単位）
  - src/core/base/testdata/rfd0093/README.md: ## GPU検証の最終結果（2026-10-17）、## 編集段階の静的確認（追加許可以前の記録）、## 軽量検証結果（2026-10-17）（節単位）
  - src/core/base/testdata/rfd0097/README.md: ## 履歴（旧M2・不採用、現行手順ではない）、#### 実行記録（節単位）
  - src/core/base/testdata/rfd0105/README.md: ## 結果（節単位）
  - src/core/base/testdata/rfd0109/README.md: ## 2026-10-31 の実機確認・受入結果、## 2026-10-31 の自動検証結果（節単位）
  - src/core/base/testdata/rfd0110/README.md: ## pin改訂の検証結果（2026-11-01、2回目）、## 改訂版の検証結果（2026-11-01）、## 等分・Centre caption改訂の検証結果（2026-11-01、3回目）、## 自動検証結果（2026-10-31）、## 高さ等分・共通Panel改訂の検証結果（2026-11-02、4回目）、### 受入結果（2026-11-02）（節単位）
  - src/core/base/testdata/rfd0117/README.md: ## 通常Runの結果（2026-11-12）（節単位）
  - src/core/base/testdata/rfd0121/README.md: ## 2026-11-15の検証結果（節単位）
  - src/core/base/testdata/rfd0128/README.md: ## 実施記録（節単位）
  - src/core/base/testdata/rfd0130/README.md: ### 2026-11-16 のステージ 1 の結果、### 2026-11-16 のステージ 2 の性能結果、### 2026-11-16 のステージ 2 の機能検証結果（節単位）
- f 検証資材の中の過去の記録: 120 行
  - src/core/base/testdata/rfd0084/forward_refactor_verification.md（44 行。ファイル単位）
  - src/core/base/testdata/rfd0084/review_verification.md（26 行。ファイル単位）
  - src/core/base/testdata/rfd0084/verification.md（12 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-162734_prd092_before_cpu.txt（2 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-171409_prd092_after_off_cpu.txt（4 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-172041_prd092_after_on_cpu.txt（4 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-172808_prd092_perf_1_off.txt（4 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-173051_prd092_perf_1_on.txt（4 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-173317_prd092_perf_2_off.txt（4 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-173545_prd092_perf_2_on.txt（4 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-173813_prd092_perf_3_off.txt（4 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-174034_prd092_perf_3_on.txt（4 行。ファイル単位）
  - src/core/base/testdata/rfd0092/config/run_20261015-174306_prd092_smoke.txt（4 行。ファイル単位）
- g 名前を残すファイルへのリンク・過去の Run 名: 3 行
  - docs/wiki/categories/311_dev_tools.md: 34・41
  - docs/wiki/mechanisms/intrinsic_novelty.md: 23
- h 走行中の bat（1 回目の前に当てる）: 11 行
  - apps/12_batch_run.bat: 7・20・21・50・86・87・365・417・527・569・573

### A5 の残す行の一覧（直下の文書の題と最初の ## より後）

- D24 昔の skill の名前: 15 行
  - docs/memo/0114_findings_intake_and_fix_rfd_10rfd.md: 45・58・94・103・135・138・212・214・291・357・364・415
  - docs/memo/0120_issue_work_type_and_rfd_link_10rfd.md: 64・203
  - docs/memo/0188_metrics_viewer_lod_documentation_consistency_10rfd.md: 71
- D24 本 RFD の題の行: 1 行
  - docs/memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md: 1
- D24・D3 昔の表記・題・件名の引用: 16 行
  - docs/memo/0114_findings_intake_and_fix_rfd_10rfd.md: 256
  - docs/memo/0120_issue_work_type_and_rfd_link_10rfd.md: 112・139・146
  - docs/memo/0131_rfd_9xx_renumber_by_creation_date_10rfd.md: 41・71・72・92・95・97・100・101・138・141・247
  - docs/memo/0197_conversation_options_summary_page_10rfd.md: 313
- D5 の今ある材料を指すパス: 14 行
  - docs/memo/0116_rnd_lifelong_novelty_10rfd.md: 94
  - docs/memo/0124_test_process_mkldnn_off_10rfd.md: 30
  - docs/memo/0125_dqn_learner_update_skeleton_10rfd.md: 340
  - docs/memo/0130_rnd_estimate_statistics_perf_10rfd.md: 174・321・770
  - docs/memo/0131_rfd_9xx_renumber_by_creation_date_10rfd.md: 69・93
  - docs/memo/0179_rfd0097_performance_remeasurement_10rfd.md: 30
  - docs/memo/0191_rnd_raw_error_and_recent_learning_10rfd.md: 937
  - docs/memo/0197_conversation_options_summary_page_10rfd.md: 56・139・140・365
- 名前を残す ADR へのリンク・名前（受入の種類の外）: 9 行
  - docs/memo/0114_findings_intake_and_fix_rfd_10rfd.md: 414
  - docs/memo/0117_warn_policy_and_unread_config_10rfd.md: 130
  - docs/memo/0120_issue_work_type_and_rfd_link_10rfd.md: 91・144・235・306・343
  - docs/memo/0131_rfd_9xx_renumber_by_creation_date_10rfd.md: 191
  - docs/memo/0187_issue_list_views_10rfd.md: 172
- 過去の Run 名（受入の種類の外）: 1 行
  - docs/memo/0098_temporl_skip_policy_10rfd.md: 353
