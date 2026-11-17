# PRD131 9xx の番号の PRD の振り直し 実装メモ

## 概要

[PRD131](131_prd_9xx_renumber_by_creation_date_10prd.md) を実装する。9xx の番号の PRD 57 本と 918 の実装計画 1 本を、§10 の表の並びで 132〜188 へ改名し（先頭の番号は取り直しても 132 のまま）、今使っている文書・課題・コードの表記を新しい名前と番号へ直す。過去の記録はリンクのパスだけを直す。各 PRD に振り直しの行を足し、`wiki_support.py` がその行で件名の旧い番号を引くようにする。規約・観点・skill・wiki の規約を改める。

## 主な変更

1. 道具（TDD）: `reports/tools/wiki_support.py` に、PRD の冒頭の振り直しの行を読む処理と検査（§5.3）を足し、`prd_rows` がコミットの日付と振り直しの行で番号を当てるようにする。`core/anet-core/testdata/prd095/wiki_test.py` に受入 A8 の (a)〜(f) を足し、既存の `git` の差し替えを日付も返す形に直す。
2. 改名: §下の「旧 → 新の一覧」の 58 本を、同じフォルダの中でファイルシステムで改名する（`git mv` は使わない）。`core/anet-core/testdata/prd918/` を `core/anet-core/testdata/prd158/` へ改名する。
3. 振り直しの行: 57 本の題の直下の引用行の先頭に `> 2026-11-17 番号を <旧> から <新> へ振り直した（PRD131）。` を足す。保管場所の 8 本にも足す。
4. 表記の直し（§5.2）: 旧いファイル名・`PRD` 付きの番号・`PRD` を付けない番号・小文字の `prd<旧>` の出現を、スクリプトで候補として出し、行 1〜6 を出現ごとに当てる。行 3・6 で残した出現は下の「残した出現」に一覧で残す。
5. コードの名前（D5）: `config_baseline_test.cpp` のテスト名・タグ・環境変数、`episode_end_test.cpp`・`trainer_test.cpp` のタグ、`testdata/prd158/` の README と `validate.py`（出力先 `.scratch/prd158`、Run 名 `_prd158_ll`）、設計文書 140 の jp・en のリンク。
6. 題の番号（D12）: 918・938・941・910 の題と 918 の実装計画の題、課題 a39330 の `title`・`scope`、`CONTEXT.md` の「PRD900」（921 → 142）。
7. 規約（§5.4）: memo README（採番・リンクの注意）、`docs/quality/prd.md`（項番 3・17・23・28）、`docs/quality/impl.md`（項番 21・82・83）、`anet-create-prd`（128 行、Never rewrite の段落。`.agents` と `.claude` の両方）、`docs/agents/issue-tracker.md`、`docs/wiki/README.md`（67・108 行）、`anet-lint-wiki`（あれば）、`CHANGELOG.md`、PRD114 の冒頭の注記。
8. 課題の `prd` 欄 51 件と本文、`wiki_support.py maps` で一覧とトピック別まとめの区画を作り直す。
9. wiki の取り込みの記録（D14）: この変更で中身かパスが変わり、HEAD の時点では取り込み済みだった wiki の材料を取り込み、`docs/wiki/log.md` に行を足す（実装済みとして取り込み済みだったものは `implemented:` つき）。HEAD の時点ですでに取り込み待ちだった材料は、点検に任せる。
10. Claude Code の memory の旧い番号を直す（リポジトリの外。完了報告に書く）。

## テスト

- Public interface / surface: `reports/tools/wiki_support.py` の `maps`（`regenerate_maps` の結果の `unresolved_prds` と、トピック別まとめの「PRD の一覧」の行）と、振り直しの行の検査で止まる挙動。C++ はテストのタグとテスト名の文字列だけを変え、振る舞いは変えない。
- 優先 behavior:
  1. (a) 振り直しの行の日付以前のコミットの件名にある旧い番号が、振り直した PRD の今の番号の行になる（tracer bullet。件名→当て方→行の表示までを 1 本で通す）。
  2. (d) 今の番号と旧い番号の両方で同じ PRD に当たると、1 行にまとまる。
  3. (b) 同じ番号のファイルが別にあり、コミットが振り直しの行の日付より後なら、そのファイルの PRD の行になる。
  4. (c) 2 本以上の PRD が同じ旧い番号の振り直しの行を持つと、一意に決められない行になる。
  5. (e) 形の崩れた振り直しの行、同じ日付の行が 2 つある PRD、`<新>` がつながらない PRD で、`maps` がファイルと行を示して止まる。
  6. (f) 題のあと最初の見出しより後ろの、振り直しの行と同じ形の行は読まない。
- TDD 順序: 1 → 2 → 3 → 4 → 5 → 6 の順に、1 テストを書いて失敗を確かめ、最小の実装で通してから次へ進む。既存の `git` の差し替えの書き直しは、1 の前に、既存のテストが日付つきの形でも通ることを確かめてから入れる。文書・課題の直しと改名は TDD の対象外（振る舞いを変えない文書の変更）で、受入の検索で確かめる。

## 検証

```powershell
.\.venv\Scripts\python.exe -B core/anet-core/testdata/prd095/wiki_test.py -v
.\.venv\Scripts\python.exe -B core/anet-core/testdata/prd114/issues_test.py -v
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[prd158]"
core\anet-core\bin\Debug\anet-core-test.exe --list-tests "[prd158-config]"
.\.venv\Scripts\python.exe reports/tools/wiki_support.py maps
.\.venv\Scripts\python.exe reports/tools/wiki_support.py scan
```

受入 A1〜A11 の検索とコマンドは PRD の §6 のとおりに回し、結果をこの計画の「検証」に残す。A12 はコミットの準備とコミットのあとで確かめる。

### 結果（2026-11-17）

- `wiki_test.py`: 16 件すべて成功（足した 7 件を含む。ログ `.scratch/prd131/wiki_test.log`）。`issues_test.py`: 32 件すべて成功（`.scratch/prd131/issues_test.log`）。
- ビルド: `anet-core-test` のターゲットだけを作り直し成功（変えたテストの 3 ファイルを再コンパイル。`.scratch/prd131/build-anet-core-test.log`）。
- A5: `anet-core-test.exe "[prd158]"` は 3 件・51 assertions が成功。`--list-tests "[prd158-config]"` は 1 件、`--list-tests "[prd918]"` は 0 件。検索 3 は 0 件。
- A1: `docs/memo` の下の 9xx のファイルは 0 本。58 本は下の一覧のとおり 132〜188 に改名し、番号の重なりは無い。
- A2: 新しい番号の並びは §10 の並びと同じで、差は 0（先頭が 132 のまま）。
- A3: 57 本すべてで、題の直下の最初の引用行が振り直しの行（`（PRD131）。` で終わる）。`maps` は振り直しの行の検査で止まらない。
- A4: 検索 1 は 1 件で、918 の実装計画の「実行した検証コマンドと結果」の節の Run のフォルダ名（`run_…_prd918_ll`。行 3(b)(c)）。検索 3 は 0 件。検索 2 の 60 か所と検索 4 の 304 か所は、すべて下の「残した出現」の行 3・行 6 に当たる（検索 1 を合わせて 365 か所）。改名したファイルはまだ git が追跡していないので、`git grep --untracked` で引いた（`*.txt`・`*.canvas`・`.vscode`・追跡していない `docs/StampFly.md` は除いた）。実装のレビューの 1 往復目の指摘で、検索 1 を `_10prd` の無い `<旧い番号>_<名前>` の形まで、検索 4 を日本語の文字に接した番号まで拾う形に改め（PRD の改訂）、漏れていた 50 か所を直してから引き直した。
- A6: memo README の検査と同じ規則を Python で回し（`.scratch/prd131/linkdiff.py`。前は HEAD の木、後は作業ツリー）、リンク切れは前 230 件・後 191 件で、増えた分は 0 件。
- A7: `docs/adr`・保管場所・実験記録・`reports/` のレポートの差分は、改名しない 32 本がリンクの丸括弧の中の変更だけ、改名した保管場所の 8 本が振り直しの行の追加とリンクの丸括弧の中の変更だけ（元の中身と行ごとに突き合わせた）。
- A8: `maps` の `unresolved_prds` は空。110_runner_app・316_config・318_rl_core の「PRD の一覧」に PRD158 の行がある。
- A9: `maps` は課題の検査で止まらない。課題の `prd` 欄に旧いファイル名は 0 件（検索 1 に含まれる）。
- A10: §5.4 の各行の変更が入っている。`anet-create-prd`・`anet-lint-wiki` の 2 つの写しは同じ（skill の実体コピーの検査で差が出ない）。memo README・`prd.md`・`anet-create-prd` に `900未満`・`900 未満`・`9xx`・`新しい 999` は無い。`docs/agents/issue-tracker.md` に `below 900` は無い。
- A11: 取り込みの行を 79 本ぶん足し（173 は実装のレビューの指摘で直したあとの blob で足し直した）、今回の変更で取り込み待ちになった材料は `scan` の `pending`・`moved` に 0 本。`pending` に残る 63 本は HEAD の時点からの取り込み待ち。実装中に改めた PRD131 は、最後の改訂の blob で書き戻しの行を足した（PRD のレビューで本文がまた変わったら、その blob で足し直す）。
- 未実施: A12（ステージしたあとの改名の見分けと、コミットのあとの日付・`unresolved_prds`）はコミットの準備とコミットのあとで確かめる。

## 前提

- 実装の前提（D14）: 開発者は、ほかの作業の未コミットの変更（課題 aa5758・b754ff・f9790d・6db0f2、課題の一覧、トピック別まとめ 7 枚）が残ったまま始めることを選んだ。混ざった分は、コミットの準備で一部だけをステージして分ける。実装の途中（2026-11-17 00:44 ごろ）から、別の枠が同じ作業ツリーで PRD170 の実装を始めた（経過の 2 往復目の行）。
- 先頭の番号は 132 のまま（2026-11-16、PRD の最大の番号は 131）。
- 振り直しの行の日付は 2026-11-17（実装の日。計画の保存時は 2026-11-16 としていたが、日付が変わったので実装の日にした）。コミットが別の日になったら、コミットの準備で 57 行の日付と、memo README の採番の日付（`2026-11-17`、`--since=2026-11-18`）をその日に直す。
- `wiki_support.py` のコミットの日付は `git log --format=%ad --date=short` の日付とし、これから作るコミットは道具を回した日とする。
- ビルドは `anet-core-test` のターゲットだけにする。

## 経過

- 2026-11-16: PRD131 のグリル（レビュー OK、コミット e2803e44）のあと、同じ枠で実装を始めた。未決事項の監査でユーザー判断が必要なブロッカーは 0（実装の前提は開発者が「このまま始める」を選んだ）。
- 2026-11-16〜17 道具（TDD）: 既存の `git` の差し替えを日付つきの形（`%ad<TAB>%s`）に直し、既存のテストが通ることを確かめてから、優先 behavior の 1（RED を確かめてから最小の実装で GREEN）を入れた。2〜4 と 6 は、1 の最小の実装（PRD の集合でまとめる、日付で比べる、最初の見出しで止める）ですでに満たしていて、テストを足した時点で通った。5（検査で止まる）は RED を確かめてから実装した。テストの架空の番号は、受入 A5 に掛からないよう、振り直さない番号（997・998・911・931）にした。
- 2026-11-17 改名と振り直しの行: 58 本を改名し（`git mv` は使わない）、57 本に振り直しの行を足し、`testdata/prd918/` を `testdata/prd158/` にした。振り直しの行の検査は、実際の 96 行（今回の 57 行と PRD114 の 39 行）で通った。
- 2026-11-17 表記の直し: ファイル名と `PRD` 付きの番号を 209 ファイル・1,007 行で直した（過去の記録はリンクのパスだけ 128 か所）。`PRD` を付けない番号は、候補 369 か所を 1 か所ずつ見て、212 か所を直し、残りを「残した出現」に分けた。`wiki_test.py` はテストの架空の番号なので置き換えから除いた。
- 2026-11-17 個別の直し: 題の番号（134・147、課題 a39330 の `title`・`scope`）、`CONTEXT.md` の「PRD900」→「PRD142」、課題 546bdb の対応方針のコマンドの出力先（`.scratch/prd158/`）、設計文書 140・212 の英語版の翻訳元情報（日本語版と同じリンク 1 か所の変化を確かめて blob と日付を更新）、PRD114 の冒頭の注記、CHANGELOG の見出し、memo README・`docs/agents/issue-tracker.md`・`anet-create-prd`（2 つの写し）・PRD と実装の観点・wiki の規約。`anet-lint-wiki` は実装済みの PRD のコミットの探し方を書いておらず（wiki の規約の 108 行に任せている）、変えていない。
- 2026-11-17 PRD131 の改訂: §5.2 の行 2 で残すとした PRD131 自身のリンク 3 個が、改名で切れて受入 A6 に当たったので、改名したファイルを指すリンクのパスだけ直すことにし、PRD の冒頭に改訂の行を足した（PRD のレビューに別に出す）。
- 2026-11-17 wiki の取り込みの記録: `scan` が拾った 142 本のうち、HEAD の時点で取り込み済みだった 79 本（うち実装済みとして取り込み済みの 2 本は `implemented:` つき）に取り込みの行を足した。残る 63 本は HEAD の時点ですでに取り込み待ちだったもので、点検に任せる。PRD の D14 は 918・934 に `implemented:` を添えるとしていたが、2 本とも HEAD の時点で実装済みとしての取り込みが済んでいなかったので、行を足していない（取り込みの記録が実際より進んで見えないようにするため）。PRD の D14 と受入 A11 も、この扱いに合わせて改め、冒頭に改訂の行を足した。
- 2026-11-17 実装のレビュー 1 往復目: 差し戻し（計画 1 件、実装 2 件）。受入 A4 の検索が `_10prd` の無い `<旧い番号>_<名前>` の形（wiki の出典のリンクの文字など 47 か所）と、日本語の文字に接した番号（3 か所）を拾っていなかったので、50 か所を直し、PRD の §5.2 の定義と A4 の検索 1・4 を改めた（検索 4 の前後の判定を ASCII の英数字に）。本 PRD の実装計画を §5.2 の行 2 と A4 の除外に足した。題材の課題の対応内容の行から旧い番号を外した。PRD131 の書き戻しの行を足した。範囲外の検出（今使っている文書に PRD114 より前の `999_<名前>` の表記が 16 ファイル・63 か所）は、工程末にこの場で直した（下の「工程末のこの場で直す」の行）。
- 2026-11-17 実装のレビュー 2 往復目: 計画は OK、実装は差し戻し 1 件。1 往復目のあと、別の枠が同じ作業ツリーで PRD170（旧 938）の実装を始めていて（00:44 ごろから。新しい名前と番号を使っている）、`trainer_test.cpp`・`episode_end_test.cpp`・課題 6a32b4・課題の一覧とトピック別まとめの「課題」の区画に、この実装の変更と PRD170 の変更が混ざった。D14 の前提（ほかの枠が PRD を実装していない）が実装の途中で崩れた。依頼の変更ファイルの一覧に、ファイルごとにこの実装の分と PRD170 の分を書き分けた。この実装のビルドとテストの記録（00:14）は PRD170 の変更より前の作業ツリーで取ったもので、PRD170 の変更が入ったあとはビルドし直していない。コミットの準備では、混ざったファイルを一部だけステージして分ける。リポジトリの直下にある 0 バイトの `938_nn_trace_capture_unification_10prd.md`（00:44 作成）は、この実装では作っていない（工程末に開発者の選択で消した。下の「工程末のこの場で直す」の行）。
- 2026-11-17 実装のレビュー 3 往復目: 計画は OK、実装は差し戻し 1 件。依頼の一覧を書いた直後（01:18:14）に、PRD170 の枠が `CHANGELOG.md`・設計文書 140 の jp・en（この実装も触ったファイル）と設計文書 130 の jp・en を変えた。一覧に、CHANGELOG と 140 のこの実装の分と PRD170 の分、130 が全体 PRD170 であることを書き足した。140 の en の翻訳元情報の行は、この実装の値（`blob:3a5166f4…`）が PRD170 の枠によって両方の変更を含む値（`blob:c1f82ff6…`）に置き換わっていて、どちらのコミットに入れるかはコミットの準備で決める。
- 2026-11-17 範囲外の気づき: 追跡していない `docs/StampFly.md`（開発者の文書）が、改名した PRD（921・930・910・913・932 ほか）を 30 か所近く指している。git の管理の外なので実装の直しからは外し、工程末にこの場で直した（下の行）。
- 2026-11-17 工程末のこの場で直す（開発者が候補 3 件すべてを選んだ）: (1) `docs/StampFly.md` の旧い番号を 25 行・40 か所、新しい番号とファイル名へ直した（§5.2 の行 5 と同じ扱い）。振り直しの外の古い番号（911 → 060、`999 Munchausen RL` → 067、`999 Notifier Detach` → 146）も今の番号にし、行番号（L59-67・L105-111）は値として残した（行 6）。直す前の写しは `.scratch/prd131/StampFly.before.md`。(2) PRD114 より前の `999_<名前>` の表記を、今使っている文書 15 ファイル（課題 7c9a70・93f8a6、`reports/INDEX.md`、直下の PRD 12 本）の 72 か所で今の番号と名前へ直した（行 5。`999_smdp` は `167_smdp_transition_duration`）。過去の記録は `reports/` の勾配ノイズのサーベイのリンクのパス 2 か所だけを直した（行 4）。PRD084 の検証資材 `forward_refactor_verification.md` の「今回の変更ファイル」の 2 か所は、当時の変更の記録として残した（行 3）。(3) リポジトリ直下の 0 バイトの `938_nn_trace_capture_unification_10prd.md` を消した。wiki: 直した材料のうち実装を始めた時点で取り込み済みだった 10 本と、課題 5adbf9 の判定で実装済みになった PRD131（`implemented:2026-11-17`。311 の成立の経緯を実装後の形にした）の取り込みの行を足し、`scan` の `pending` は開始時の 63 本、`moved` は 0 本。リンク切れは前 230 件・後 190 件で、増えた分は 0 件。
- 2026-11-17 実装のレビュー 5 往復目（工程末のこの場で直す 3 件）: OK。判定には使わない事実として、StampFly.md の行番号（L59-67・L105-111）と課題 93f8a6 の `:66` が振り直しの行の分だけずれていると示された（§5.2 の行 6 で値として残す扱い）。

## 完了報告

- 実施内容: 9xx の PRD 57 本と 918 の実装計画を 132〜188 へ改名し、振り直しの行を足し、今使っている文書・課題・コードの表記を新しい名前と番号へ直した（過去の記録はリンクのパスだけ）。`wiki_support.py` が振り直しの行で件名の旧い番号を引くようにした（TDD）。採番の規約・観点・skill・wiki の規約を改め、wiki の取り込みの記録を足した。工程末に、開発者が選んだ 3 件（StampFly.md の旧い番号、PRD114 より前の `999_<名前>` の表記、直下の 0 バイトのファイル）をこの場で直した。
- 変更ファイル: 上の「主な変更」と「経過」、`.scratch/prd131/review-request-files.md`・`review-request-inplace.md` の一覧のとおり。
- レビュー結果: 実装計画と実装は 5 往復目で OK（1〜3 往復目は差し戻し。4 往復目で OK、5 往復目はこの場で直した分）。実装中に改めた PRD は 5 往復目で OK。根拠が無く見送った指摘は無い。再確認ポイント: 同じ作業ツリーで PRD170 の実装が並んでいて、混ざったファイルはコミットの準備で一部だけをステージして分ける。設計文書 140 の en の翻訳元情報の行は、作業ツリーでは PRD170 の値（`blob:c1f82ff6…`）で、この実装の値（`blob:3a5166f4…`）は残っていない。ビルドとテストの記録（00:14）は PRD170 の変更より前の作業ツリーのもの。StampFly.md と課題 93f8a6 の行番号は、振り直しの行の分だけずれたまま値として残した。
- 決まらなかったこと: この実装で起票した課題は無い（4 桁の形は、グリルで起票した課題 fe09dc）。受入 A12 は、コミットの準備とコミットのあとで確かめる。
- 判断してほしいこと: (1) コミットの準備で、PRD170 の変更と混ざったファイル（`trainer_test.cpp`・`episode_end_test.cpp`・`CHANGELOG.md`・設計文書 140 の jp と en・課題 6a32b4・課題の一覧とトピック別まとめの「課題」の区画）を、この実装の分だけステージするか。(2) 140 の en の翻訳元情報の行を、この実装のコミットでは `blob:3a5166f4…` にして、PRD170 のコミットで `blob:c1f82ff6…` にするか。(3) コミットが 2026-11-17 でなければ、57 本の振り直しの行の日付と memo README の日付・`--since` をその日に直す（受入 A12）。

## 残した出現

受入 A4 の検索 1・2・4 の最終の当たり 365 か所を、§5.2 の判定で分けたもの（検索 3 は 0 件）。

| ファイル | 行（番号） | 判定 |
|---|---|---|
| `core/anet-core/testdata/prd084/forward_refactor_verification.md` | 275（941）、467（918）、467（908）、550（941） | 行 6: 行番号・値 |
| `core/anet-core/testdata/prd084/review_verification.md` | 217（939） | 行 6: 行番号・値 |
| `core/anet-core/testdata/prd092/README.md` | 28（919）、30（920）、38（920）、58（951） | 行 6: 行番号・値 |
| `core/anet-core/testdata/prd117/README.md` | 62（953）、62（952）、157（956）、158（920） | 行 6: 行番号・値 |
| `core/anet-core/testdata/prd121/README.md` | 11（950） | 行 6: 行番号・値 |
| `core/anet-core/testdata/prd128/README.md` | 38（936） | 行 6: 行番号・値 |
| `core/anet-core/testdata/prd130/README.md` | 93（922） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261103-2c9a07.md` | 56（952） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261103-582880.md` | 54（952） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261103-66100f.md` | 58（952） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261103-73a449.md` | 45（951）、46（954） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261105-f4b870.md` | 54（941）、56（902）、57（944） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261107-0c34dd.md` | 64（952） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261107-1226e5.md` | 85（950） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261107-16ffd8.md` | 43（902） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261107-a9a0e3.md` | 91（952） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261107-c17068.md` | 55（952） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261111-24d885.md` | 44（945）、44（958）、76（958） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261111-4a6b6f.md` | 44（934） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261111-585e60.md` | 49（907） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261111-72d640.md` | 67（936） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261111-77ac3c.md` | 61（956） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261112-f9790d.md` | 77（926） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261113-317d32.md` | 61（954） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261115-5adbf9.md` | 38（901）、38（909）、38（914）、38（919）、38（922）、38（929）、38（934）、38（939）、38（942）、38（959）、39（910）、39（941）、49（901）、49（959）、56（918）、56（934）、56（938）、56（925）、63（938） | 行 3(a): 題材の課題の、振り直す前の状態の記述 |
| `docs/memo/088_nn_adaptive_max_pool_10prd.md` | 101（920） | 行 6: 行番号・値 |
| `docs/memo/096_metrics_step_axis_interval_10prd.md` | 43（925） | 行 6: 行番号・値 |
| `docs/memo/098_temporl_skip_policy_10prd.md` | 64（901）、334（955）、378（912） | 行 6: 行番号・値 |
| `docs/memo/098_temporl_skip_policy_20impl.md` | 127（947） | 行 6: 行番号・値 |
| `docs/memo/098_temporl_skip_policy_22impl.md` | 77（906）、186（938） | 行 6: 行番号・値 |
| `docs/memo/107_noisy_gain_lane_ladder_10prd.md` | 25（928）、31（925） | 行 6: 行番号・値 |
| `docs/memo/113_runner_per_monitor_dpi_10prd.md` | 161（922） | 行 6: 行番号・値 |
| `docs/memo/114_findings_intake_and_fix_prd_10prd.md` | 9（959）、17（910）、17（941）、56（910）、56（912）、56（913）、56（920）、56（921）、56（930）、56（932）、56（933）、56（940）、56（941）、129（901）、129（909）、129（914）、129（919）、129（922）、129（929）、129（934）、129（939）、129（942）、129（959）、129（910）、129（941）、254（901）、254（909）、254（914）、254（919）、254（922）、254（929）、254（934）、254（939）、254（942）、254（959）、254（910）、254（941）、333（910）、333（912）、333（913）、333（920）、333（921）、333（930）、333（932）、333（933）、333（940）、333（941）、333（959）、346（910）、346（941）、383（910）、383（941）、256（930） | 行 3(a): 使われた番号・振り直した番号の並びと範囲、「PRD 930」の例 |
| `docs/memo/114_findings_intake_and_fix_prd_20impl.md` | 46（959）、195（910）、195（912）、195（913）、195（920）、195（921）、195（930）、195（932）、195（933）、195（940）、195（941） | 行 3(a): 使われた番号・振り直した番号の並びと範囲、「PRD 930」の例 |
| `docs/memo/116_rnd_lifelong_novelty_20impl.md` | 97（946） | 行 6: 行番号・値 |
| `docs/memo/116_rnd_lifelong_novelty_23impl.md` | 93（946）、120（942） | 行 6: 行番号・値 |
| `docs/memo/117_warn_policy_and_unread_config_20impl.md` | 191（947）、191（941）、197（947）、197（946）、223（953）、223（952）、244（953）、244（952）、253（952） | 行 6: 行番号・値 |
| `docs/memo/118_atari_run_profile_flatten_10prd.md` | 96（906）、96（907）、96（954） | 行 6: 行番号・値 |
| `docs/memo/119_small_doc_comment_config_fixes_20impl.md` | 63（919） | 行 6: 行番号・値 |
| `docs/memo/121_extrinsic_intrinsic_value_decomposition_20impl.md` | 110（952）、261（950）、325（950） | 行 6: 行番号・値 |
| `docs/memo/123_rnd_intrinsic_reward_scale_10prd.md` | 6（956）、91（922） | 行 6: 行番号・値 |
| `docs/memo/123_rnd_intrinsic_reward_scale_20impl.md` | 73（946） | 行 6: 行番号・値 |
| `docs/memo/126_atari_ram_metric_distinct_and_bit_reducers_10prd.md` | 61（942） | 行 6: 行番号・値 |
| `docs/memo/127_eval_trace_intrinsic_reward_columns_20impl.md` | 185（950） | 行 6: 行番号・値 |
| `docs/memo/128_dqn_agent_refactor_bundle_10prd.md` | 74（907）、124（927） | 行 6: 行番号・値 |
| `docs/memo/128_dqn_agent_refactor_bundle_20impl.md` | 191（936）、268（936）、281（936）、308（958）、317（958） | 行 6: 行番号・値 |
| `docs/memo/129_eval_trace_bonus_beta_before_learning_10prd.md` | 69（927） | 行 6: 行番号・値 |
| `docs/memo/129_eval_trace_bonus_beta_before_learning_20impl.md` | 71（954）、84（954）、123（954） | 行 6: 行番号・値 |
| `docs/memo/130_rnd_estimate_statistics_perf_10prd.md` | 275（939）、275（947）、275（950）、275（956）、376（939）、438（925） | 行 6: 行番号・値 |
| `docs/memo/130_rnd_estimate_statistics_perf_20impl.md` | 160（926）、165（922）、168（922）、207（922）、217（922） | 行 6: 行番号・値 |
| `docs/memo/134_network_lock_audit_10prd.md` | 3（910） | 行 3(a): 振り直しの行 |
| `docs/memo/135_nn_block_metrics_10prd.md` | 3（920） | 行 3(a): 振り直しの行 |
| `docs/memo/136_best_checkpoint_10prd.md` | 3（913） | 行 3(a): 振り直しの行 |
| `docs/memo/136_best_checkpoint_10prd.md` | 37（902） | 行 6: 行番号・値 |
| `docs/memo/141_background_eval_snapshot_ordering_10prd.md` | 3（912） | 行 3(a): 振り直しの行 |
| `docs/memo/142_agent_prev_action_obs_10prd.md` | 3（921） | 行 3(a): 振り直しの行 |
| `docs/memo/143_serialize_10prd.md` | 3（930） | 行 3(a): 振り直しの行 |
| `docs/memo/144_episode_forensics_10prd.md` | 3（932） | 行 3(a): 振り直しの行 |
| `docs/memo/145_staged_batch_replay_schedule_10prd.md` | 3（933） | 行 3(a): 振り直しの行 |
| `docs/memo/146_notifier_scoped_detach_10prd.md` | 3（942）、4（942） | 行 3(a): 振り直しの行 |
| `docs/memo/146_notifier_scoped_detach_10prd.md` | 31（936）、34（929）、111（929）、118（929） | 行 6: 行番号・値 |
| `docs/memo/147_iqn_first_priority_quantile_pair_10prd.md` | 3（941） | 行 3(a): 振り直しの行 |
| `docs/memo/148_target_update_step_axis_10prd.md` | 3（940） | 行 3(a): 振り直しの行 |
| `docs/memo/150_residual_alpha_gate_10prd.md` | 3（951）、4（951） | 行 3(a): 振り直しの行 |
| `docs/memo/151_run_queue_snapshot_10prd.md` | 3（952）、4（952） | 行 3(a): 振り直しの行 |
| `docs/memo/152_metric_targeted_hyperparam_control_10prd.md` | 3（927）、4（927） | 行 3(a): 振り直しの行 |
| `docs/memo/153_q_statistics_config_screening_10prd.md` | 3（947）、4（947） | 行 3(a): 振り直しの行 |
| `docs/memo/154_scalar_threshold_rate_10prd.md` | 3（954）、4（954） | 行 3(a): 振り直しの行 |
| `docs/memo/155_config_root_ownership_alignment_10prd.md` | 3（908）、4（908） | 行 3(a): 振り直しの行 |
| `docs/memo/156_cicd_improvement_10prd.md` | 3（906）、4（906） | 行 3(a): 振り直しの行 |
| `docs/memo/157_test_line_coverage_10prd.md` | 3（959）、4（959） | 行 3(a): 振り直しの行 |
| `docs/memo/158_eval_schedule_first_fire_10prd.md` | 3（918）、5（918） | 行 3(a): 振り直しの行 |
| `docs/memo/159_input_mapping_joystick_10prd.md` | 3（922）、4（922） | 行 3(a): 振り直しの行 |
| `docs/memo/160_input_target_train_forced_action_10prd.md` | 3（923）、4（923） | 行 3(a): 振り直しの行 |
| `docs/memo/161_observation_pipeline_10prd.md` | 3（944）、4（944） | 行 3(a): 振り直しの行 |
| `docs/memo/162_decision_aligned_nstep_10prd.md` | 3（914）、4（914） | 行 3(a): 振り直しの行 |
| `docs/memo/163_dropmerge_neet_diagnostics_10prd.md` | 3（915）、4（915） | 行 3(a): 振り直しの行 |
| `docs/memo/164_nn_forward_amp_consolidation_10prd.md` | 3（936）、4（936） | 行 3(a): 振り直しの行 |
| `docs/memo/165_nn_head_projection_separation_10prd.md` | 3（937）、4（937） | 行 3(a): 振り直しの行 |
| `docs/memo/166_noisynet_effective_weight_spectral_norm_10prd.md` | 3（939）、4（939） | 行 3(a): 振り直しの行 |
| `docs/memo/167_smdp_transition_duration_10prd.md` | 3（956）、4（956） | 行 3(a): 振り直しの行 |
| `docs/memo/168_nstep_target_correction_10prd.md` | 3（943）、4（943） | 行 3(a): 振り直しの行 |
| `docs/memo/169_outside_core_extension_layout_10prd.md` | 3（945）、4（945） | 行 3(a): 振り直しの行 |
| `docs/memo/170_nn_trace_capture_unification_10prd.md` | 3（938）、6（938） | 行 3(a): 振り直しの行 |
| `docs/memo/171_replay_eviction_sample_count_10prd.md` | 3（950）、4（950） | 行 3(a): 振り直しの行 |
| `docs/memo/172_bootstrapped_multihead_q_10prd.md` | 3（905）、4（905） | 行 3(a): 振り直しの行 |
| `docs/memo/173_intrinsic_reward_novelty_10prd.md` | 3（924）、5（924） | 行 3(a): 振り直しの行 |
| `docs/memo/173_intrinsic_reward_novelty_10prd.md` | 4（923） | 行 3(a): 番号そのものを述べた改訂の記録 |
| `docs/memo/174_llm_actor_10prd.md` | 3（926）、4（926） | 行 3(a): 振り直しの行 |
| `docs/memo/175_seed_population_early_stop_10prd.md` | 3（955）、4（955） | 行 3(a): 振り直しの行 |
| `docs/memo/176_qvaluepanel_manual_action_state_alignment_10prd.md` | 3（948）、4（948） | 行 3(a): 振り直しの行 |
| `docs/memo/177_episode_end_read_semantics_10prd.md` | 3（916）、4（916） | 行 3(a): 振り直しの行 |
| `docs/memo/179_prd097_performance_remeasurement_10prd.md` | 3（946）、4（946） | 行 3(a): 振り直しの行 |
| `docs/memo/180_core_perf_candidates_202607_10prd.md` | 3（909）、4（909） | 行 3(a): 振り直しの行 |
| `docs/memo/181_config_file_layout_10prd.md` | 3（907）、4（907） | 行 3(a): 振り直しの行 |
| `docs/memo/182_source_layout_10prd.md` | 3（957）、4（957） | 行 3(a): 振り直しの行 |
| `docs/memo/182_source_layout_10prd.md` | 55（948） | 行 6: 行番号・値 |
| `docs/memo/183_temporl_skip_followups_10prd.md` | 3（958）、4（958） | 行 3(a): 振り直しの行 |
| `docs/memo/184_eval_session_step_trace_10prd.md` | 3（919）、4（919） | 行 3(a): 振り直しの行 |
| `docs/memo/185_release_packaging_cmake_10prd.md` | 3（949）、4（949） | 行 3(a): 振り直しの行 |
| `docs/memo/186_runner_theme_switch_10prd.md` | 3（953）、4（953） | 行 3(a): 振り直しの行 |
| `docs/memo/186_runner_theme_switch_10prd.md` | 21（908） | 行 6: 行番号・値 |
| `docs/memo/187_issue_list_views_10prd.md` | 3（925）、6（925） | 行 3(a): 振り直しの行 |
| `docs/memo/188_metrics_viewer_lod_documentation_consistency_10prd.md` | 3（934）、8（934） | 行 3(a): 振り直しの行 |
| `docs/wiki/index.md` | 19（914）、31（914）、37（914） | 行 6: 行番号・値 |
| `docs/wiki/mechanisms/evaluation.md` | 7（914）、15（925）、34（914）、34（949） | 行 6: 行番号・値 |
| `docs/wiki/mechanisms/intrinsic_novelty.md` | 47（956）、48（956） | 行 6: 行番号・値 |
| `docs/wiki/mechanisms/noisynet.md` | 24（955） | 行 6: 行番号・値 |
| `docs/wiki/mechanisms/replay_ratio_lane_window.md` | 14（951） | 行 6: 行番号・値 |
| `docs/wiki/mechanisms/temporal_abstraction.md` | 15（914）、21（952）、24（914）、39（925）、72（904） | 行 6: 行番号・値 |
| `docs/wiki/subjects/battle_zone.md` | 16（955） | 行 6: 行番号・値 |
| `docs/wiki/subjects/beam_rider.md` | 9（914）、13（914） | 行 6: 行番号・値 |
| `docs/wiki/subjects/kung_fu_master.md` | 38（956）、38（933）、40（956）、55（956） | 行 6: 行番号・値 |
| `docs/wiki/subjects/name_this_game.md` | 15（941）、17（913）、17（903） | 行 6: 行番号・値 |
| `docs/wiki/subjects/phoenix.md` | 16（951） | 行 6: 行番号・値 |
| `docs/wiki/subjects/qbert.md` | 17（918）、33（948）、40（955）、43（921） | 行 6: 行番号・値 |
| `docs/management/issues/issue-20261110-546bdb.md` | 43（918） | 行 3(b): 既にある控えの場所 |
| `docs/memo/158_eval_schedule_first_fire_20impl.md` | 36（918）、227（918）、237（918）、238（918）、240（918）、244（918）、245（918） | 行 3(b): 既にある控えの場所 |
| `docs/memo/158_eval_schedule_first_fire_20impl.md` | 41（918）、48（918）、56（918）、64（918）、72（918）、80（918）、88（918）、96（918）、104（918）、112（918）、117（918）、120（918）、128（918）、136（918）、144（918）、152（918）、160（918）、168（918）、176（918）、181（918）、184（918）、187（918）、194（918）、200（918）、206（918） | 行 3(c): 実行した検証コマンドと結果 |

## 旧 → 新の一覧

| 旧 | 新 |
|---|---|
| `docs/memo/dropped/901_actor_priority_per_10prd.md` | `docs/memo/dropped/132_actor_priority_per_10prd.md` |
| `docs/memo/frozen/904_async_loader_ndeep_10prd.md` | `docs/memo/frozen/133_async_loader_ndeep_10prd.md` |
| `docs/memo/910_network_lock_audit_10prd.md` | `docs/memo/134_network_lock_audit_10prd.md` |
| `docs/memo/920_nn_block_metrics_10prd.md` | `docs/memo/135_nn_block_metrics_10prd.md` |
| `docs/memo/913_best_checkpoint_10prd.md` | `docs/memo/136_best_checkpoint_10prd.md` |
| `docs/memo/frozen/935_nn_embedding_tsne_10prd.md` | `docs/memo/frozen/137_nn_embedding_tsne_10prd.md` |
| `docs/memo/frozen/902_adaptive_policy_synchronization_10prd.md` | `docs/memo/frozen/138_adaptive_policy_synchronization_10prd.md` |
| `docs/memo/frozen/903_agent_update_result_view_10prd.md` | `docs/memo/frozen/139_agent_update_result_view_10prd.md` |
| `docs/memo/frozen/929_metrics_sqlite_backend_10prd.md` | `docs/memo/frozen/140_metrics_sqlite_backend_10prd.md` |
| `docs/memo/912_background_eval_snapshot_ordering_10prd.md` | `docs/memo/141_background_eval_snapshot_ordering_10prd.md` |
| `docs/memo/921_agent_prev_action_obs_10prd.md` | `docs/memo/142_agent_prev_action_obs_10prd.md` |
| `docs/memo/930_serialize_10prd.md` | `docs/memo/143_serialize_10prd.md` |
| `docs/memo/932_episode_forensics_10prd.md` | `docs/memo/144_episode_forensics_10prd.md` |
| `docs/memo/933_staged_batch_replay_schedule_10prd.md` | `docs/memo/145_staged_batch_replay_schedule_10prd.md` |
| `docs/memo/942_notifier_scoped_detach_10prd.md` | `docs/memo/146_notifier_scoped_detach_10prd.md` |
| `docs/memo/941_iqn_first_priority_quantile_pair_10prd.md` | `docs/memo/147_iqn_first_priority_quantile_pair_10prd.md` |
| `docs/memo/940_target_update_step_axis_10prd.md` | `docs/memo/148_target_update_step_axis_10prd.md` |
| `docs/memo/dropped/917_episode_start_stack_contract_10prd.md` | `docs/memo/dropped/149_episode_start_stack_contract_10prd.md` |
| `docs/memo/951_residual_alpha_gate_10prd.md` | `docs/memo/150_residual_alpha_gate_10prd.md` |
| `docs/memo/952_run_queue_snapshot_10prd.md` | `docs/memo/151_run_queue_snapshot_10prd.md` |
| `docs/memo/927_metric_targeted_hyperparam_control_10prd.md` | `docs/memo/152_metric_targeted_hyperparam_control_10prd.md` |
| `docs/memo/947_q_statistics_config_screening_10prd.md` | `docs/memo/153_q_statistics_config_screening_10prd.md` |
| `docs/memo/954_scalar_threshold_rate_10prd.md` | `docs/memo/154_scalar_threshold_rate_10prd.md` |
| `docs/memo/908_config_root_ownership_alignment_10prd.md` | `docs/memo/155_config_root_ownership_alignment_10prd.md` |
| `docs/memo/906_cicd_improvement_10prd.md` | `docs/memo/156_cicd_improvement_10prd.md` |
| `docs/memo/959_test_line_coverage_10prd.md` | `docs/memo/157_test_line_coverage_10prd.md` |
| `docs/memo/918_eval_schedule_first_fire_10prd.md` | `docs/memo/158_eval_schedule_first_fire_10prd.md` |
| `docs/memo/918_eval_schedule_first_fire_20impl.md` | `docs/memo/158_eval_schedule_first_fire_20impl.md` |
| `docs/memo/922_input_mapping_joystick_10prd.md` | `docs/memo/159_input_mapping_joystick_10prd.md` |
| `docs/memo/923_input_target_train_forced_action_10prd.md` | `docs/memo/160_input_target_train_forced_action_10prd.md` |
| `docs/memo/944_observation_pipeline_10prd.md` | `docs/memo/161_observation_pipeline_10prd.md` |
| `docs/memo/914_decision_aligned_nstep_10prd.md` | `docs/memo/162_decision_aligned_nstep_10prd.md` |
| `docs/memo/915_dropmerge_neet_diagnostics_10prd.md` | `docs/memo/163_dropmerge_neet_diagnostics_10prd.md` |
| `docs/memo/936_nn_forward_amp_consolidation_10prd.md` | `docs/memo/164_nn_forward_amp_consolidation_10prd.md` |
| `docs/memo/937_nn_head_projection_separation_10prd.md` | `docs/memo/165_nn_head_projection_separation_10prd.md` |
| `docs/memo/939_noisynet_effective_weight_spectral_norm_10prd.md` | `docs/memo/166_noisynet_effective_weight_spectral_norm_10prd.md` |
| `docs/memo/956_smdp_transition_duration_10prd.md` | `docs/memo/167_smdp_transition_duration_10prd.md` |
| `docs/memo/943_nstep_target_correction_10prd.md` | `docs/memo/168_nstep_target_correction_10prd.md` |
| `docs/memo/945_outside_core_extension_layout_10prd.md` | `docs/memo/169_outside_core_extension_layout_10prd.md` |
| `docs/memo/938_nn_trace_capture_unification_10prd.md` | `docs/memo/170_nn_trace_capture_unification_10prd.md` |
| `docs/memo/950_replay_eviction_sample_count_10prd.md` | `docs/memo/171_replay_eviction_sample_count_10prd.md` |
| `docs/memo/905_bootstrapped_multihead_q_10prd.md` | `docs/memo/172_bootstrapped_multihead_q_10prd.md` |
| `docs/memo/924_intrinsic_reward_novelty_10prd.md` | `docs/memo/173_intrinsic_reward_novelty_10prd.md` |
| `docs/memo/926_llm_actor_10prd.md` | `docs/memo/174_llm_actor_10prd.md` |
| `docs/memo/955_seed_population_early_stop_10prd.md` | `docs/memo/175_seed_population_early_stop_10prd.md` |
| `docs/memo/948_qvaluepanel_manual_action_state_alignment_10prd.md` | `docs/memo/176_qvaluepanel_manual_action_state_alignment_10prd.md` |
| `docs/memo/916_episode_end_read_semantics_10prd.md` | `docs/memo/177_episode_end_read_semantics_10prd.md` |
| `docs/memo/done/928_metrics_action_info_dynamic_cast_10prd.md` | `docs/memo/done/178_metrics_action_info_dynamic_cast_10prd.md` |
| `docs/memo/946_prd097_performance_remeasurement_10prd.md` | `docs/memo/179_prd097_performance_remeasurement_10prd.md` |
| `docs/memo/909_core_perf_candidates_202607_10prd.md` | `docs/memo/180_core_perf_candidates_202607_10prd.md` |
| `docs/memo/907_config_file_layout_10prd.md` | `docs/memo/181_config_file_layout_10prd.md` |
| `docs/memo/957_source_layout_10prd.md` | `docs/memo/182_source_layout_10prd.md` |
| `docs/memo/958_temporl_skip_followups_10prd.md` | `docs/memo/183_temporl_skip_followups_10prd.md` |
| `docs/memo/919_eval_session_step_trace_10prd.md` | `docs/memo/184_eval_session_step_trace_10prd.md` |
| `docs/memo/949_release_packaging_cmake_10prd.md` | `docs/memo/185_release_packaging_cmake_10prd.md` |
| `docs/memo/953_runner_theme_switch_10prd.md` | `docs/memo/186_runner_theme_switch_10prd.md` |
| `docs/memo/925_issue_list_views_10prd.md` | `docs/memo/187_issue_list_views_10prd.md` |
| `docs/memo/934_metrics_viewer_lod_documentation_consistency_10prd.md` | `docs/memo/188_metrics_viewer_lod_documentation_consistency_10prd.md` |
| `core/anet-core/testdata/prd918/` | `core/anet-core/testdata/prd158/` |
| `[prd918]`・`[.][prd918-config]`・`ANET_PRD918_OUTPUT` | `[prd158]`・`[.][prd158-config]`・`ANET_PRD158_OUTPUT` |
