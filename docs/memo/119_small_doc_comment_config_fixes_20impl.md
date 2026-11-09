# PRD119 文書・コメント・設定の小さな食い違いの直し 実装メモ

## 概要

[PRD119](119_small_doc_comment_config_fixes_10prd.md) の確定事項 D1〜D15 を実装する。変えるのは文書・コメント・設定ファイルと課題の記録で、C++ はコメントだけを直す。題材 8 件の課題の対応内容に直した内容（ee32bd・fb8a45 では直さない理由）を書き、索引の課題を `実装済み`、題材の課題を `残件無し` にする。

## 主な変更

- D1: `apps/README.md` の 52 行の zip の名前を `anet-lab-doxygen-<version>.zip` にする。
- D2: `docs/design/030_user_guide_analysis.jp.md` の 20・525 行、`docs/design/160_applications_and_tools.jp.md` の 109・444 行のリンク先を `optuna.md` にする。
- D3: `core/anet-core/src/replay_buffer_impl.hpp` の `InitialPriorityCompleter` の `@note` に `GetStats` を足す。
- D4: `apps/runner/config/nn.txt` の 240〜241 行（`[Pattern 4]` と `@ResNet`）と、275〜304 行（「MLP ResBlock」の節）を消す。Pattern の番号は振り直さない。
- D5: nn.txt の type 一覧（16〜41 行）を登録済みの 29 種に合わせ、`Add` を外す。`Reshape`・`StackMerge` は【Base / Routing】、`LayerNorm2d`・`GAP2D`・`MaxPool2d` は【Normalization & Pooling】、`HybridSpatialEmbedder` は【Embedding / Bridge】、`Dropout` は新しい見出し【Regularization】へ入れる。
- D6: nn.txt の init.mode のコメント（5〜10 行）に default・constant・trunc_normal を足す。311 行を 6 種の並びにし、312 行の「1=Xavier」を「xavier」に直す。
- D7: nn.txt の「記法」の節（362〜368 行）から `(=Tag)`・`(@Source)` の行を消し、`>` と `(*N)` を残す。合流は branch の bind で書く旨と 130 §2.4 への参照の 1 行を置く。「表現出来ないパターン」の MEMO の 376 行を、structure の中では書けず、branch を分けて bind の連結でつなげば書けるが branch だらけになる、の趣旨に直す。
- D8: `docs/design/130_neural_networks.jp.md` の 94 行と `.en.md` の 95 行の、structure が出力 tag・入力 tag 参照を表せるという記述を、block 反復（`(*N)`）だけを表せる記述に直す。英語版のヘッダ（`translated-from`）は、日英の対応を確かめてから、直したあとの日本語版の blob と日付に更新する（実装の入口で PRD の D8 を改めた。経過を参照）。
- D9: `apps/runner/config/metrics_scalar.txt` の 11〜14 行を、記法の 1 行と 140 §6 への参照の 2 行に置き換える。`core/anet-core/src/observers.cpp` の 1410〜1414 行を、140 §6 と `ParseMetricTokens` を指す 1 行に置き換える。
- D10: `apps/runner/config/Atari.txt` の 1650 行を `app.online.eval_panel.fps = 14.5` にする。
- D11: Atari.txt の 1652〜1654 行を消し、328 行の `app.online.train_panel.fps = 0` の直前に説明のコメントを置く。
- D12〜D14: 題材 8 件と索引の課題の対応内容に今回の行を足し、`anet-update-issue` の「判定」で、索引の課題を `実装済み`、題材 8 件を `残件無し` にする。ee32bd・fb8a45 は直さない理由を書く。最後に `reports/tools/wiki_support.py maps` を回す。
- D15: PRD949 は変えない。

## テスト

- Public interface / surface: 出荷設定の解決結果（8 env の全キーと文字列値）、既定葉検査、`[shipped_config]` と既定のテストの集合。
- 優先 behavior: (1) 8 env の解決結果は、消すサンプルが使う block の 20 キーが消えるほかは変わらない。Atari の `app.online.train_panel.fps` は 0、`app.online.eval_panel.fps` は 14.5 のまま。(2) 既定葉検査の errors が 0。(3) 既定のテストの集合と `[shipped_config]` の結果が、直す前と同じ。
- TDD 順序: TDD の対象外とする。C++ はコメントだけを変え、公開の振る舞いを変えない。設定の変更のうち振る舞いに当たるのはサンプルを消すことだけで、消したサンプルを選んだときの失敗は、汎用の解決の検査（`core/anet-core/src/config_test.cpp` の 969 行の `material selection target not found`）がすでにテストしている。新しいテストは足さず、上の 3 つを直す前と後で比べる。

## 検証

```powershell
# A9（直す前と後で 1 回ずつ。ビルドと既定のテストの集合・[shipped_config] の JUnit を .scratch\prd119\<label>\ に保存する）
powershell -NoProfile -ExecutionPolicy Bypass -File .scratch\prd119\run_suite.ps1 before
powershell -NoProfile -ExecutionPolicy Bypass -File .scratch\prd119\run_suite.ps1 after
# A3（直す前の設定を .scratch\prd119\config_before\ へ写してから、両方を [prd103-resolve] で解決して比べる）
.\.venv\Scripts\python.exe .scratch\prd119\compare_resolved.py
# A4
.\.venv\Scripts\python.exe -B core\anet-core\testdata\prd072\check_default_leaves.py
.\.venv\Scripts\python.exe -B core\anet-core\testdata\prd072\check_default_leaves_test.py
```

A1・A2・A5〜A8・A10 は PRD §6 に書いた `git grep`・リンク切れ検査・`git diff HEAD` で確かめる。

## 前提

- `run_suite.ps1` の中のビルドは `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'` で行う（AGENTS.md の「AI エージェントでのビルド注意事項」）。
- A3 の入力は、PRD072 の `prepare.py` と同じく `$include <設定のフォルダ/_main.txt>` と `$include <設定のフォルダ/<env>.txt>` の 2 行にする。直す前の側は、編集の前に `apps/runner/config` を写した `.scratch/prd119/config_before/` を使う。比べる道具（`compare_resolved.py`・`run_suite.ps1`）はこの PRD の移行の確かめだけに使う使い捨てで、`.scratch/prd119/` に置き、リポジトリには入れない。
- 直す前と後の解決には、同じ（直したあとの）実行体を使う。C++ の変更はコメントだけなので、解決の結果に効かない。
- A2 の `git diff HEAD` の基準は、実装を始めた時点の HEAD（`0cbfbcf6`）とする。実装の途中で HEAD は、PRD119 のグリルのコミット（`3d99edd4`）と別の作業の課題のコミット（`1dcac500`・`180213e2`）で進んだ。どれも done/069・009・010 に触れていないので、`git diff 0cbfbcf6` と `git diff HEAD` のどちらでも差分は 0 である。レビューの差分の基準は、今の HEAD（`180213e2`）にする。
- コメントの文面: 記法の節の案内は「合流（積・連結）は branch の bind で書く（docs/design/130_neural_networks.jp.md §2.4）」。MEMO の 376 行は「structure の中では書けない。branch を分けて bind の連結（,）でつなげば書けるが、branch だらけになる」。observers.cpp は「記法の正本は docs/design/140_observability.jp.md §6、受け付ける語は metrics.cpp の ParseMetricTokens」。
- 作業ツリーにある別の作業の変更（課題ファイルの `ai_judge` の言い換え、ADR 0015 など）には触れない。課題ファイルへは対応内容の行と判定の 2 欄だけを書く。

## 経過

- 2026-11-09: PRD を解決した（暫定の印なし。再グリルとレビュー OK は同じ日）。未決事項監査のブロッカー 0。直す前のビルドと既定のテストを `.scratch/prd119/before/` へ回し始め、直す前の設定を `.scratch/prd119/config_before/` へ写した。既定葉検査は直す前に `Assignments: 32011, audited: 4015, errors: 2`（Atari.txt の 1650・1654 行）だった。
- 2026-11-09: 書く前に読んだ観点（「設定ファイル」「C++ コード」「コメントと TODO」「検証と報告」）のうち「検証と報告」の規約が、英語版を直すときは対応を確かめた範囲に合う `translated-from` の blob を記録するとしていた。PRD の D8 の「ヘッダは変えない」と食い違うので、PRD の D8・§5.1・§5.2・A5 を改め、改訂行を足した。130 の英語版のヘッダの blob は日本語版の今の blob（`ca1760b3…`）と一致していた。直した PRD は `prd` のレビューに出す。
- 2026-11-09: 直した PRD の `prd` のレビュー（同じレビューエージェントの 4 往復目）は OK。原文は `.scratch/prd119/review-20261109-impl-entry-prd.md`。
- 2026-11-09: D1・D2・D8 を直した。D8 の日英の対応の確認: 直す前の英語版のヘッダの blob（`ca1760b3cd68d89f736ddc7072ac4869e1313a69`）は日本語版の `git hash-object` と一致していた（英語版は直す前の日本語版の全体に対応）。直したのは日本語版の 94 行と英語版の 95 行の同じ 1 文だけで（`git diff --stat` で日英とも 1 行）、英語版の文は日本語版の新しい文（structure は block 名を `>` でつなぐ直列の記法で、block 反復 `(*N)` を表せる。合流は branch の bind で書く）を訳したものである。リンク先は変えていない。そのうえで、ヘッダを `blob:77c388e560b408deeea60861036c233c18d6e50c date:2026-11-09 progress:done` に更新した。
- 2026-11-09: 直す前のビルドとテスト（`.scratch/prd119/before/`）: ビルドは変更なし（`ninja: no work to do`）。既定のテストの集合は終了コード 0（JUnit の `tests="173854" failures="0" errors="0" skipped="1"`）。`[shipped_config]` だけを回すと、`RND shipped Atari networks and Taiga settings match their contracts` で Debug 版 libtorch の oneDNN の assert（`nthr_ == nthr`、`dnnl_thread.hpp`）により終了コード 3 で止まった。2 回回して再現し、そのテストを単独で回しても同じだった。そのテストを除いた `[shipped_config]` は終了コード 0。既定の集合では通るので、テストの順序で MKLDNN の切り替えの有無が変わることによる既知の現象（[PRD028 の実装記録](done/028_cnblock_convnext_20impl.md) の 15〜17 行）と見て、PRD119 の範囲外とした（工程末の候補）。
- 2026-11-09: D3〜D7・D9〜D11 を直した。`metrics_scalar.txt` の作業ツリーは直す前から CRLF（`core.autocrlf=true` による。index は LF）だったので、改行の検査に合わせて作業ツリーを LF に戻した（中身の差分は変わらない）。
- 2026-11-09: A4: `check_default_leaves.py` は `Assignments: 31988, audited: 3992, errors: 0`（直す前より 23 減。消した block の 20 キー、`@ResNet`・`@MLP_RES` の 2 行、train_panel の `?= 10` の 1 行）、`check_default_leaves_test.py` は 16 件 OK。A1・A2・A5〜A8 の検索はすべて期待どおり（A1: 0 件・1 件、A2: BROKEN 0 件・過去の記録の差分 0、A5: 登録 29 種がすべて一覧にあり `\bAdd\b` 0 件、`(=`・`(@`・`タグ` 0 件、`1=Xavier` 0 件、記法と MEMO の節に bind の行、130 の旧い記述 0 件、英語版のヘッダの blob が日本語版と一致、A6: 0 件・記法の 1 行と参照あり・旧例 0 件・参照の行に `ParseMetricTokens`、A7: `@note` に `GetStats`、A8: 0 件）。機械検査: BOM は 10 件で保持、改行は全ファイル `w/lf`、追加行に直訳の語と `TODO` は 0 件。
- 2026-11-09: A9: 直したあとのビルドとテスト（`.scratch/prd119/after/`）。ビルドは終了コード 0 で、`observers.cpp`・`replay_buffer_impl.cpp` を含む 14 段を作り直した。既定のテストの集合は終了コード 0（JUnit の `tests="173854" failures="0" errors="0" skipped="1"`）で、テストケースの一覧（1,117 件）は直す前と一致した。`[shipped_config]` は直す前と同じく RND のテストの oneDNN の assert で終了コード 3。そのテストを除いた `[shipped_config]` は終了コード 0 で、直す前と同じ結果である。
- 2026-11-09: A3: `compare_resolved.py` の結果は `RESULT: OK`。8 env のすべてで、消えたキーは 20（`net.block.[Linear_Proj].`・`[Linear_Hid].`・`[Linear_Out].`・`[ResBlock].` で始まるものだけ）、増えたキーと値の変化は 0。Atari の `app.online.train_panel.fps` は 0→0、`app.online.eval_panel.fps` は 14.5→14.5。キーの数（直す前→後）は CartPole 755→735、GridMaze 803→783、GridMaze_muzero 739→719、ImageCls 1098→1078、Atari 1508→1488、LunarLander 993→973、LunarLander_cpu 919→899、DropMerge 1443→1423。比較の結果は `.scratch/prd119/a3/comparison.json`。
- 2026-11-09: 題材 8 件と索引の課題の対応内容に今回の行を足した（A10 の前半）。AI の判定の更新（索引の課題を `実装済み`、題材 8 件を `残件無し`）は、レビューが OK になったあと、完了の報告の前に行う（A10 の後半）。
- 2026-11-09: 実装計画と実装のレビュー（1 往復目）: 実装は OK、実装計画は差し戻し 1 件。題材の課題に書く AI の判定の値が、改名前の `直った見込み` のままだった（実装の途中のコミット `180213e2` で、`anet-update-issue` の「判定」の値が「残件有り」「残件無し」「判定不能」「実装済み」になった）。計画の 3 か所と、入力元の PRD の §2.2・D14・A10 を `残件無し` に直し、PRD に改訂行を足した。直した PRD は `prd` のレビューに出し直す。原文は `.scratch/prd119/review-20261109-impl-1.md`。
- 2026-11-09: 実装計画と実装のレビュー（2 往復目）: どちらも OK。原文は `.scratch/prd119/review-20261109-impl-2.md`。直した PRD の `prd` のレビュー（5 往復目）も OK。
- 2026-11-09: A10 の後半: `anet-update-issue` の「判定」で、索引の課題 b23c1d を `実装済み`、題材 8 件を `残件無し` にし、`ai_judged` を 2026-11-09 にした。`reports/tools/wiki_support.py maps` で課題の一覧と地図の「課題」の区画を作り直した。
- 2026-11-09: 工程末の候補 2 件を示し、開発者が 1 を記録、2 をこの場で直すと選んだ。1 は [issue-20261109-fa2f69](../management/issues/issue-20261109-fa2f69.md)（`[shipped_config]` を単独で回すと RND のテストが Debug 版 libtorch の oneDNN の assert で止まる）として記録した。2 は、030・160 の英語版の `translated-from` を直したあとの日本語版の blob（030: `c09b31fa…`、160: `41294467…`）と日付に更新した。英語版は直す前の日本語版に同期していて、日本語版の差分はリンク先の 4 か所だけ、英語版の対応する 4 か所はすでに `optuna.md` を指していたことを確かめてから更新した。この直しは、入力元なしの `impl` のレビューに出し、OK だった（原文は `.scratch/prd119/review-20261109-inplace.md`）。依頼に書いた差分の置き場の名前（`diff-20261109-0442`）が実際の名前（`diff-20261109-0419`）と違っていたが、レビューエージェントは `git diff 180213e2` で同じ差分を得て判定した。

## 完了報告

- 実施内容: PRD119 の確定事項 D1〜D15 を実装した。変えたのは文書・コメント・設定ファイルと課題の記録で、C++ はコメントだけ。実装の入口で PRD の D8（英語版のヘッダの扱い）を、レビューの指摘で PRD と計画の AI の判定の値（`残件無し`）を直し、PRD のレビューを受け直した。工程末に、課題 1 件を記録し、030・160 の英語版のヘッダをこの場で直した。
- 変更ファイル: `apps/README.md`、`apps/runner/config/nn.txt`・`metrics_scalar.txt`・`Atari.txt`、`core/anet-core/src/observers.cpp`・`replay_buffer_impl.hpp`、`docs/design/030_user_guide_analysis.jp.md`・`.en.md`、`160_applications_and_tools.jp.md`・`.en.md`、`130_neural_networks.jp.md`・`.en.md`、`docs/memo/119_small_doc_comment_config_fixes_10prd.md`・`_20impl.md`、課題 10 件（題材 8 件、索引、新規の fa2f69）と、`wiki_support.py maps` が作り直した課題の一覧と wiki の地図の「課題」の区画。
- レビュー結果: 実装計画は 2 往復目で OK（1 往復目の指摘は判定の値の改名 1 件）。実装は 1 往復目から OK。PRD は実装の入口の直しで 4 往復目、判定の値の直しで 5 往復目に OK。この場で直した英語版のヘッダも OK。見送った指摘はない。
- 決まらなかったこと: なし（工程末の候補 1 件は [issue-20261109-fa2f69](../management/issues/issue-20261109-fa2f69.md) に記録した）。
- 判断してほしいこと: 索引の課題 b23c1d を含む 9 件の状態を `解決` にするかは開発者が決める（AI の判定は索引 `実装済み`、題材 8 件 `残件無し`）。

## 検証の結果

| 受入 | 結果 |
| --- | --- |
| A1 | 通った。`anet-lab-docs-` 0 件、`anet-lab-doxygen-<version>.zip` 1 件 |
| A2 | 通った。030・160 の日本語版のリンク切れ 0 件、`git diff HEAD` で done/069・009・010 に差分なし、ee32bd・fb8a45 の対応内容に理由がある |
| A3 | 通った。上の経過のとおり |
| A4 | 通った。errors 0、検査のテスト 16 件 OK |
| A5 | 通った。上の経過のとおり |
| A6 | 通った。上の経過のとおり |
| A7 | 通った |
| A8 | 通った。0 件 |
| A9 | 通った。ビルド成功。既定のテストの集合は直す前と同じ（失敗 0）。`[shipped_config]` は直す前と同じ（RND のテストだけ既知の assert で止まり、それを除けば終了コード 0） |
| A10 | 通った。対応内容を書き、索引の課題を `実装済み`、題材 8 件を `残件無し` にした。fb8a45 の親は索引の課題 |

実行できなかった検証: なし。`[shipped_config]` の RND のテストの停止は、直す前から起きる範囲外の現象で、[issue-20261109-fa2f69](../management/issues/issue-20261109-fa2f69.md) に記録した。
