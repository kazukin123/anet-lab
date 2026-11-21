# PRD095 段階2の実装計画

## 概要

PRD095 §4.4 の項目1〜6・8〜10を実装し、受入8〜15・17を確認する。取り込み・点検の仕組みを整備し、RunnerAppと311の地図にリファクタリングの判断材料を加える。

基準コミット: c6425461a3f9635fffc887fce005912afb8fcae6。2026-10-22 のユーザーの PLEASE IMPLEMENT THIS PLAN により承認された計画を本書へ保存した。20implは段階1の記録として保持する。2026年7月の監査結果の移行と受入16はClaude Code側の後続作業とし、段階3も対象外とする。

## 主な変更

### 取り込み・点検と地図

- anet-ingest-wiki と anet-lint-wiki を表示用定義とともに両方のskillディレクトリへ追加する。共通契約はwikiのREADMEに集約する。
- 指定された正本を読む。引数なしでは会話とGit差分を照合し、この会話で作成・変更した材料だけを選ぶ。無関係な既存差分は取り込まない。不正な入力は編集前に止める。
- 本文の取り込みはエージェントが正本を読んで判断する。ファイル収集、blob比較、リンク候補探索、機械的な一覧生成は補助処理へまとめる。
- 点検はPRDの7項目を実施する。取り込み上限は既定3本で更新ありを優先する。fixは取り込み済み判定に使わず、同一blobの移動とdoneへの移動を区別する。
- リンクの修復候補が一意なら修正し、解決できなければ注記して報告する。ファイル・PRDの一覧は全地図で再生成する。
- RunnerAppと311を規定順の8節へ更新する。選ばなかった形と変えるときの条件は、経緯に載せたPRD全文などから対象に関係する内容だけを拾う。採用した仕様や新機能の合否は写さない。RunnerAppにはPRD046のCancel時の不採用案を含める。
- 正本は変更せず、ログは追記だけにする。更新後は出典・リンク・目録・公開できない値を確認する。

### 台帳・既存手順との接続

- registry.py listに--categoryを追加する。分類はJSONの定義順で先勝ちとし、拡張子直前の_testを除く。不明keyは有効なkey一覧を示して失敗させる。
- buildは既存の派生一覧と地図の台帳の指摘を再生成する。対象はopen・stale、掲載内容は番号・優先度・観点・題だけとする。
- listの110は分類上の範囲、RunnerAppの地図はapps/runner全体とする。
- 全地図の目印を検査してから書く。欠落・重複・不正な組があれば台帳の派生一覧を含めて何も書かない。区画外とwikiログには触れない。
- housekeepingにwiki-lintをstatsの次として追加する。前提条件は取り込み漏れか未注記のリンク切れとし、点検自身の出力で再選定されないようにする。
- 監査に地図の参照、key指定時の--category、不採用案の再訪条件、変更時の条件の参照、読んだ地図名の報告を追加する。PRD作成にも地図の2節の利用を追加する。
- skills READMEとAGENTS.mdを追随させ、動線マップはanet-harness-mapで再生成する。skillの2コピーを一致させる。

## テスト

Public interface / surface: 台帳CLI list/buildの標準出力・終了コード・派生ファイル、新しい2つのskillの実行結果、wikiの節・出典・目録・ログ、housekeepingの選定結果、監査の報告。

TDDは各挙動につき1テストずつRED→GREENを完了する。

1. 一時ディレクトリの台帳をCLIから分類抽出できる経路を通す。
2. テストファイルの分類、110の範囲、未知keyの失敗を確認する。
3. buildのopen・stale抽出、アプリケーションの範囲、区画外の保持を確認する。
4. 目印が壊れた場合、全出力ファイルの内容が変わらないことを確認する。

続いて受入8〜15・17を試走する。会話内材料・材料なし・不正入力・未コミット材料・撤回・消えた名前・リンク移動・未追跡ファイル・done移動を確かめる。確定事項と受入だけにある判断材料も取り込み、採用仕様と新機能の合否は取り込まない。housekeepingのdry-runを取り込み前後で比較する。台帳CLIの試験用レコードは写しだけに置く。受入用の一時ページ・PRD・追加行は検証後に除去し、ログは保持する。

## 検証

- anet-audit lens=refactor scope=110_runner_appを1回実施し、読んだ地図名、根拠なく不採用案を再提案した件数、変更時の条件を対処方針に引用した件数を記録する。
- statsのカテゴリ集計不変、skill全体一致、LF、BOM保持、差分検査、出典・公開値検査、動線マップの検証を行う。
- DebugビルドはAGENTS.mdのMSVC初期化経路で実施する。検証はrepoの.venvを使い、コマンド・結果・未実施理由を本書へ記録する。
- 自己点検後、独立したan-reviewへ計画と実装を依頼し、修正可能な指摘をまとめて直して再レビューする。wiki本文は書いた側が出典照合と規定の検査を行う。
- 保存後は再読・hash・更新時刻・Git差分を確認する。

## 前提

- PRD・ADR・CONTEXT.md・categories.json・共有プロセス契約・stats.pyは変更しない。既存の未コミット変更を保持する。
- 補助処理の追加は、会話の判断を持たない機械処理を再利用・検証するため。新しい利用者向け契約は2つのskill、list --category、buildの出力拡張に限定する。
- 検証資材はtestdata/prd095へ置き、関連設計文書から参照する。既存文書への追加は今回の機能と検証導線に限り、別作業の変更を保持する。
- 今回の対象の完了と、受入16を含む段階2全体の完了を区別する。git add・commit・pushは行わない。

## 経過と結果

- 2026-10-22: 未決事項監査のブロッカー0。受入16のClaude Code側への分離をユーザーが選択した。承認済み計画を保存し、実装へ進む。

### 実装と検証（レビュー前）

- 台帳CLIは、未実装の--category、拡張子によらないテスト名、未知key、地図の区画、不正な目印でREDを確認して順にGREENへ進めた。110の既存分類の範囲も回帰テストで固定した。registry_test.pyは6テスト、wiki_test.pyは4テスト成功。
- 機械処理はreports/tools/wiki_support.pyへまとめ、材料・blob・リンク候補の読取りと地図の2節の生成だけを行う。本文の知見・裁定はエージェントが読む。stats.py・分類定義は保持した。
- 受入8（会話内材料）・9〜11・14は、正本を汚さない隔離worktreeでエージェントが手順を適用した。受入12についてtrial2で実測したのはscanのeligible前後だけであり、dry-run選定の実施とは区別する。材料なし・不正入力・dry-run選定は下記のtrial3で追加確認した。結果は.scratch/prd095/stage2/trial-results.json、操作記録はtrial2.py、追記ログはtrial2/docs/wiki/log.md。試験用の材料・PRD・ページ・srcは除去した。skillの意味判断を自動テストで証明したとは扱わない。
- 受入9の正本の写しが初回にBOM/改行の差を持ったため、バイトを保持するコピーで再試走し、正本不変を確認した。消えた名前の照合もRun artifact全体を探索せず現行コード・設定へ絞った。
- 受入13: RunnerAppと311を8節へ更新し、PRD046のCancel案、裁定の語、再開条件、不変条件・等価性の検証、台帳の目印を追加した。受入15の6テストは一時台帳だけを使い、正本のfindings.jsonlへ試験レコードを追加していない。
- 受入17: anet-audit lens=refactor scope=110_runner_appを実施。読んだ地図: RunnerApp。既存open 0、新規0、根拠の無い不採用案の再訪0、変更時の条件を対処方針に引いた件数0（新規候補なし）。pane復元、pause/resume、排水、model同期の境界を点検し、意図した形を重複として提案しなかった。記録はaudit-results.json。台帳build・validate成功、wikiの書く部分は監査で不変。
- Debugビルド: AGENTS.mdのVsDevCmd.bat経由で cmake --build --preset x64-Debug を実行し、終了コード0。ログは.scratch/prd095/stage2/debug-build.log。
- 新skillのquick_validateはWindowsの既定CP932で読み取りに失敗したため、repoの.venvの -X utf8 -B で再実行して両方成功した。実体コピーの全体比較も一致した。
- anet-harness-mapを適用し、現行の文書・skill・参照から在庫と読む道を再生成した。公開版・個人版の4図はshowcaseの9チェック、composition errors/warnings 0、deliver成功。俯瞰図は4画面の表示検査にも成功。工程図は6レーンで縦スクロールを検出し、2回の限定修正でも画面内に収まる条件を満たさなかった。
- 工程図の例外: ユーザーが「縦スクロールを許容して反映する」と明示回答した。この作業では6レーンを保持し、構造検査を通した元の候補を公開した。表示検査の失敗を成功へ読み替えず、そのreceiptと2回の修正結果を保持する。俯瞰図・工程図を目視した。固定Viewer UIは英語。共有Archifyは変更していない。
- 公開5ファイルは一時バックアップ後、JSON/HTMLを置き、READMEを最後に置換した。個人版は公開版の全node idを含む。

- 自己点検で材料の除外がreports/INDEX.md以外のINDEX.mdにも及ぶことを検出した。回帰テストでREDを確認し、除外を指定された1パスだけに修正した。wiki補助のテストは5件になった。
- 開始後、別作業がcore/anet-core/src/episode_end_test.cppとdocs/design/020_user_guide_run.jp.mdを更新した。こちらでは編集せず、観測hashをexternal-changes.jsonへ保存して保持する。境界検査はこの2パス・観測hashだけを外部更新として区別する。
- Archifyの配布資材にCRLFがあり、最初のHTMLは改行が混在した。共有packageは変えず、OS一時領域のコピーのテキストの改行だけをLFへ揃え、同じArchify CLIのdeliverで4図を再生成した。HTMLの後加工は行わず、deliverのhashとLFを確認した。記録はlf-generator.json。

- 最終自己点検: 変更境界・外部更新保持・LF/BOM・8節・出典・リンク・公開値・skillコピー・共有コア・統計不変・CLI/補助テスト・diff --checkを確認した。コマンドは .venv/Scripts/python.exe -X utf8 -B .scratch/prd095/stage2/selfcheck.py、結果はselfcheck.json。registry_test.py 6件、wiki_test.py 5件成功。未解決の設計判断なし。受入16と段階3は合意済みの範囲外。
- docs/wiki/log.mdには今回の規約・設計との照合をfixとして追記した。部分的な照合をingestと偽らず、材料全体の取り込み済み判定には使わせない。試走のlintとingestは隔離環境の追記ログを保持する。
- 境界検査の再実行中、別作業によるCMakeLists.txtの更新も検出した。同様に編集せず、観測hashを追加した。外部更新は計3パスで、今回の実装と分けて保持する。

### 独立レビューと追加確認

- 第1回の独立an-reviewは計画・実装とも差し戻し。計画の必須章名と、試走記録における実測・固定値の混同の2点を修正した。原文は.scratch/prd095/review-2026-10-22-stage2-impl-1.md。
- 計画は「概要」「主な変更」「テスト」「検証」「前提」の構造へ揃え、承認計画の2章は「主な変更」の下位節とした。承認された範囲と内容は変えていない。
- 補助コードの実測: trial2はscanのeligibleが取り込み前true、後falseとなったことを確認した。以前の結果に固定代入していた材料なし応答・dry-run選定値は除き、不正入力の列挙も未実施の入力一覧と明示した。
- エージェントによる追加試走: trial3の隔離シナリオ上で実際にskillの手順を手動適用した。ネイティブのskill起動経路を自動試験したものではない。入力と実際の判断・応答はtrial3-input-results.json、trial3-dryrun-before.json、trial3-dryrun-after.json、trial3-dryrun-self-output.jsonに記録した。操作はtrial3_prepare.py、trial3_inputs.py、trial3_ingest.py、trial3_after.py、trial3_finish.py。
- 材料なし: シナリオ開始前からある別会話の未追跡材料だけの状態で、会話の由来とGit一覧を照合し「この会話の材料なし」で終了した。不正入力: 存在しないmissing.md、集合外のREADME.md、max=0/-1/1.5/abcを入力し、指定値と期待する集合・値域を示して処理前に拒否した。7ケースの前後で隔離環境全ファイルのhashが一致し、wikiの編集・ログ追記なし。
- dry-run: `$anet-housekeeping 10% until=15 only=wiki-lint dry-run` の手順1〜4を適用した。過去実測なしの初期見積10%/15分で、未取り込み1本がある前はwiki-lintを選定した。同材料を取り込み、出典・目録・blob記録を確認した後は前提条件falseで選定なし。各dry-runの前後hashは一致し、runs.jsonlを含めて書き込みなし。
- 自己出力の追加fixtureとして、lintログ、housekeepingログ、台帳の派生一覧、src、注記済み出典リンク切れ1件を置き、再度dry-runした。直近実測からの見積は10%/15分、scanはeligible=false、注記済み1件を報告し選定なし、前後hash一致。追加fixtureは除去し、wikiの追記ログを保持した。集約結果はtrial3-results.json。
- 今回の手順確認に未実施分なし。受入16はClaude Code側の後続作業、段階3は範囲外。Claude Code等の別モデルで同じskillの意味判断を行う検証は今回の試走には含めない。
- 動線マップの参照元探索は一意なファイル名とADRの参照も含めて再生成した。既存のrelease.ymlへのADR参照を保持し、死んだ動線を過剰計上しない。LF版の最終4図を表示検査し、俯瞰図は成功、工程図は許容された縦スクロールだけを検出した。個人版工程図の一時的なブラウザー描画タイムアウトは同じHTMLで再試行し、全capture成功を確認した。READMEを最後に再配置した。

### 完了時の確認

- 第2回の独立an-reviewは計画・実装ともOK、修正要求・範囲外の検出・確認事項なし。原文は.scratch/prd095/review-2026-10-22-stage2-impl-2.md。本文の出典判断は書いた側が担当し、レビューはPRD D4の境界を維持した。
- 最終保存検査中、別作業による.gitignore、apps/runner/CMakeLists.txt、apps/runner/src/RunnerFrame.cpp、core/anet-core/testdata/prd094/README.mdの更新を検出した。こちらでは編集せず、前の3パスと合わせて7パスの観測hashをexternal-changes.jsonへ保持した。
- 公開READMEのHTTPSリンクをWindows絶対パスと誤認する検査式を、ドライブ文字の前境界を持つものへ修正した。公開成果物を変更して隠していない。
- コード・テストは第1回レビュー以降変更なし。CLI6件・wiki補助5件・MSVC初期化経由のDebug成功を再利用し、記録と文書の修正後は保存・差分の検査だけを実施した。保存内容はUTF-8で再読し、hash・更新時刻・Git差分を.scratch/prd095/stage2/saved-files.jsonとfinal-check.jsonに記録する。
- 今回依頼された§4.4の項目1〜6・8〜10と受入8〜15・17を完了した。受入16を含む段階2全体は未完了で、Claude Codeによる監査結果の移行を残す。段階3は着手していない。stage・commit・pushは実施していない。

- 最終保存検査は105/105成功。全37ファイルを再読し、hash・更新時刻とGit差分を保存した。最終検証画像24枚は.scratch/prd095/stage2/visual-finalへ残し、この作業のArchify一時領域・公開前バックアップは片付けた。共有skill・外部更新には触れていない。
- 動線マップの件数: 死んだ動線1、実体コピーのドリフト0、shadow Claude6（内容違い6）/Codex7（内容違い6）、移設候補22節・527行。公開5ファイルはdocs/agents/README.mdとharness_map_architecture/workflowの各.archify.json・.html、個人版は.scratch/harness-map/の同じ構成。既知の発見事項は別のPRDで扱い、この作業では修正していない。
- Debugの終了0は実行時点の作業ツリーについての記録である。その後に観測した別作業の変更の検証まで完了したとは扱わない。

### 2026-10-22 段階2レビュー後の指摘修正

- 入力は.scratch/prd095/codex-request-stage2-fix-20261022.mdの5指摘。改訂済みPRD §4.6を正として照合し、新しい実装計画は作らず本節に記録する。PRD・ADR・CONTEXT.md・分類定義・別作業の変更は保持する。設計160 §7.4の規約へのリンクは変更しない。
- 優先behavior / TDD: PRD候補が0本・複数本でも一覧はリンク無しの注記を作り、全地図を更新し、未解決を報告できる経路を1テストでRED→GREENにする。次に試験用Git indexが本物のindexから分離されることをRED→GREENにする。Gitのread-treeは試験用indexだけを初期化し、試験全体の前後で本物のindexのhash・更新時刻を照合する。
- その他: 計測用の明示監査によるnext_lensをarchitectureへ戻し、last_seen.refactorは保持する。2つのskillのdefault_promptを各用途へ分け、取り込みのmapsにもrepoの.venvを指定する。規約とlintの報告を追随させ、PRD095の現在blobをfixとして追記する。
- 検証予定: registry_test.py・wiki_test.py、skill全体一致、LF/BOM、wikiの出典・リンク・目録・公開値、保存後の再読・hash・更新時刻・差分、品質観点による自己点検と独立an-review。DebugもMSVC初期化経路で確認する。

- TDD結果: `wiki_test.py WikiSupportTest.test_maps_keep_running_and_report_unresolved_prd_numbers -v` はPRD931候補0本、PRD999候補2本の両方でValueErrorのREDを確認した後、リンク無しの注記とページ・番号・候補数のunresolved_prdsを返して全地図を更新する実装でGREENにした。`test_fixture_selects_its_own_git_index` も、本物と同じindexを選ぶREDから専用indexを選ぶGREENへ進めた。ログは.scratch/prd095/stage2-fix/red-prd.log / green-prd.log / red-index.log / green-index.log。
- 全テスト: repoの.venvから `python.exe -X utf8 -B core/anet-core/testdata/prd095/registry_test.py -v` は6件、同wiki_test.pyは7件成功。実行と本物indexのhash・mtime_ns前後比較は.scratch/prd095/stage2-fix/tests.py、結果はtests-index.json（ともに不変）。試験では環境変数GIT_INDEX_FILEを一時ファイルへ向け、git read-tree HEADもその一時indexだけに書く。
- 報告と文書: 未解決PRDはページと番号の組の数・候補数をlintの手順8で報告する。wiki規約へ改訂PRDと同文を足し、PRD095のblob 7f45ddfafdb60c11fe8ef462e855b8691f0f6acfを材料とするfixをREADME宛に追記した。正本PRDは不変、log.mdの元の内容はprefix一致で保持した。
- 巡回: 今回の受入17のような巡回とは別の計測監査ではnext_lensを進めないものとしてarchitectureへ戻した。110_runner_appのlast_seen.refactor=2026-10-22を含むその他の欄は不変。監査skill全般の既存巡回ルールを変更したものではない。
- Debug: 最初の初期化コマンドにパス誤記があり失敗したため、AGENTS.md指定のCommunity/Common7/Tools/VsDevCmd.batへ直した。`cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'` で終了0。初回失敗と成功のログは.scratch/prd095/stage2-fix/debug-build-initial-path-error.log / debug-build.log。
- 自己点検: 実装品質確認観点の全62項目を適用範囲で確認。PRDとの一致、過剰な契約の不在、Pythonのコメントと失敗時の扱い、テストと報告、skill同期を確認した。C++・設定・所有権など今回変更が無い観点は対象外。追加の機械検査は115/115成功（selfcheck.py / selfcheck.json）。全skill実体コピー一致、LF/BOM、出典リンク・目録・公開できない値、改訂句の一致、禁止領域と別作業の変更の不変、log追記のみ、diff --check、再読・hash・更新時刻を確認した。コードに未実施の検証・未決はない。受入16と段階3は従前どおり範囲外。

- 独立an-review結果: 計画追記・実装ともOK。修正要求・範囲外の検出・開発者確認事項なし。原文は.scratch/prd095/review-2026-10-22-stage2-fix-impl.md。13テスト・Debugは実行記録をレビューし、レビュー側の重複実行はしていない。
- 今回の5指摘をすべて修正し、未決・未実施の検証なし。最終保存後にselfcheck.pyを再実行し、15ファイルの再読・hash・mtime_nsと開始snapshotからの差分をsaved-files.json / diffへ保存する。前回からの受入16・段階3の対象外は維持し、stage・commit・pushは実施しない。
- 最終検査中に別作業のdocs/memo/094_evalpanel_step_history_20impl.mdの更新を検出した。こちらでは編集せず、観測hashを.scratch/prd095/stage2-fix/external-changes.jsonへ保存し、今回差分とは分けて保持した。最終機械検査は115/115成功を確認する。
