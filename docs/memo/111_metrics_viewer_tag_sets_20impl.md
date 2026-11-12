# PRD111 タグセット 実装メモ

## 概要

PRD111（`111_metrics_viewer_tag_sets_10prd.md`）のタグ選択全体の保存・呼び出しを実装する。2026-11-02 に提示した計画をユーザーが承認。基準コミットは `addd7d1461473f3c33f79e41fd6fbed21956e711`。

## 主な変更

- `TagSetStore` がパス解決・起動時検証・初期値の複製・一覧・直列化した更新を所有する。初期値は `{}`。作業用の既定は `${metricsviewer.workspaces-dir}/metrics_viewer_tag_sets.json`。
- 保存は名前と tag を UTF-16 コード単位順に並べ、2字下げ・1行1tag・UTF-8 BOMなし・LF・末尾改行。同じディレクトリの一時ファイルから置き換え、成功後だけメモリを更新する。
- `TagSetController` と PRD §9 のモデルを追加。閉じた schema、名前・tag の規則、エラー判定順・code は PRD §6.2 に従う。不正 JSON の変換を他の Controller へ広げない。
- `MetricsViewerClientApp` が選択名と取得済み一覧を所有し、全体の置換を既存 `onTagSelectionChanged()` へ接続する。名前を `anet.metricsviewer.tagSet` に JSON string で保存し、未選択ではキーを削除する。
- `TagSetMenu` が操作行・body直下の固定配置メニュー・編集・確認・キーボードを扱う。印・件数・省略・disabled・失敗時は PRD §6.3–6.4 に従う。通信は既存 metadata/metrics の abort 枠と分ける。
- release のステージから作業用ファイルを除去し不在を検証。210・030 の日英、設定 metadata、永続 state 9件化を追随させる。

## テスト

- Public surface: HTTP API、Store の公開操作と保存ファイル、ブラウザのクリック・キー・画面・localStorage。
- 優先 behavior / TDD順序: (1) 実APIと一時保存先を使う Playwright で Save As→ファイル保存→選択変更→呼び出しを通す。(2) Store の初期作成・既存維持・正確なバイト列・起動時検証・失敗時不変。(3) API の CRUD・空集合・Unicode・閉じたschema・優先順・同名rename・同時更新。(4) 印・Save/Revert・不可視tag・改名/削除/Deselect・復元・通信失敗・複数browser・新規tag。(5) 2操作で切替、1操作でRevert、長い名前・件数・キーボード・タッチ・見切れ。
- 各 behavior を1テストずつ RED→最小実装→GREEN と進める。整理は GREEN 後。既存テストの変更は RunListPlaywrightTest の指定の操作行期待のみ。

## 検証

- Maven の対象テスト→全体 `mvn test`→`mvn package`。リポジトリ規約の Debug ビルドを VsDevCmd.bat 経由で実行。
- jar の初期値、ステージとzipの作業用ファイル不在を確認。workflow_dispatch は変更がリモートrefに必要。未実施なら理由と残る受入を記録する。
- 実装品質確認観点の全項目を適用範囲に従って自己点検。独立 an-review に impl-plan と impl を依頼し、直せる指摘を解消して再レビューする。

## 前提

- 未決事項監査のブロッカー0件。PRD §12 の解決済み事項も現行契約とする。
- 保存・比較は不可視tagを含む選択全体。新規tag自動選択・既存一括操作を維持。
- 作業用ファイルは起動時だけ読む。外部変更検知・保存直前照合・編集ロックは追加しない。
- Run選択・表示設定・グラフ順・パターン指定は範囲外。
- 無関係なdirty変更を保持。git add -f、commit、push は人間が行う。

## 経過

- 2026-11-02: 承認済み計画を保存。実装前の基準状態を確認。検証・レビュー結果は以下へ追記する。
- 最初のPlaywrightはfixtureの初期Run選択を合わせたあと、未実装の開閉ボタンでREDを確認。最小の作成・保存・呼び出しを実装し、1件成功（skipなし）。
- Storeの起動時検証は14入力のパラメータテストでRED→GREEN。正常系・保存失敗時不変・外部編集の非再読込・パス/リソース失敗も確認。
- CRUDとUnicode空白・同名renameの非書き込みをRED→GREEN。閉じたschemaと失敗優先順をRED→GREEN。TagSetStoreTest・TagSetApiIntegrationTest・既存WorkspaceApiIntegrationTestが成功。
- UI編集経路は印が付かないREDから実装。入力行の置き換えを外側クリックと誤認する問題を修正。現在は改名時のToastを検証中。
- UI編集・キーボード・PC/タッチ端末の長い名前と件数がGREEN。改名時のToastはテストの空Workspaceに由来したため、fixtureに既定Workspaceを作った。スクリーンショット `apps/metrics-viewer/target/prd111-desktop.png` と `prd111-mobile.png` を目視確認。
- 通信失敗・競合時の入力維持・別ブラウザ・実マスタ追記がGREEN。実マスタは取り込み完了をAPIで待ってからReloadする（既存の進捗pollはtagを再取得しない契約）。
- Debug: `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'` 終了値0（ninja: no work to do）。
- Unicode White_Space全25文字、64/65コードポイント（補助平面文字）、UTF-16順序、500応答でのGET不変、並列作成と後勝ち更新をAPIで確認済み。
- 自己点検の追試で、有効なobjectのあとに別のJSONが続くbodyが200になるREDを確認。タグセットController専用のObjectMapperコピーで後続tokenを拒否する形にし、他APIの設定は保った。新しい外部契約は追加していない。
- Maven全体の初回実行はログ抑制用 `-Dlogging.level.root=ERROR` が既存のログ検証7件へ影響（274件、errors/skipは0）。この引数を外して通常条件で再実行する。
- 機械確認: 今回の24ファイルはUTF-8で読め、LF、既存11ファイルのBOMを保持。`git diff --check` は対象範囲で問題なし。`_visibleTagSet` の現行JS参照は0。
- 梱包fixture: `.scratch/prd111/package-check.ps1` をPowerShell 7で実行し、実workflowの除外処理、zipの非混入、コピー元の保持を確認。jar検査はpackage後に実施する。

## 自己点検

- `docs/quality/impl.md` 全項目を適用範囲で走査。計画・PRD適合・現用名の移行・失敗・名前・コメント/ログ・public経路のテスト・日英文書・検証資材の配置を確認した。
- C++、GetScalar、Agent所有権、学習ループ計測、skill同期、Runnerのプロファイル/出荷config検査は、その対象を変えていないため対象外。追加設定はSpringのpropertiesとmetadataで、API結合テストの起動により既定の解決先とoverrideを確認する。
- 未決の設計判断なし。初期値は空で作り、作業用ファイルの追跡は人間の操作とする。workflow_dispatchは変更をcommit/pushしない本作業では未実施として残す。
- 後続JSON拒否の修正後にTagSetApiIntegrationTestと既存WorkspaceApiIntegrationTestが通常ログ条件で成功。Maven全体は通常条件で再実行中。

## 検証結果

- `mvn -f apps/metrics-viewer/pom.xml -l C:/dev/anet-lab/.scratch/prd111/full-test.log test`: BUILD SUCCESS、274件、failure/error/skipすべて0、34 suites、4分38秒。タグセットはStore18件・API5件・Playwright10件。既存のテストは指定の操作行期待の追随以外を変更していない。
- `mvn -f apps/metrics-viewer/pom.xml -DskipTests -l C:/dev/anet-lab/.scratch/prd111/package.log package`: BUILD SUCCESS（全体テスト直後なので重複実行を省略）。
- `& ./.scratch/prd111/package-check.ps1 -CheckJar`: PASS。実workflowから取り出した除外処理、ステージとzipの非混入、コピー元の保持、jarリソースの存在を確認。
- jarの `BOOT-INF/classes/initial_tag_sets.json` はバイト列 `{}\n`。初期値0件の契約を確認した。
- Debugビルド・文字コード/改行/BOM・目視は経過のとおり成功。基準コミットからHEADは変えていない。
- 未実施: GitHub Release workflowのworkflow_dispatch。変更をremote refへ公開しない本作業では実行せず、実workflowのローカルfixture検査で補足した。remoteで作成した本体zipの受入確認は残る。commit/pushと作業用ファイルの強制追加は人間が行う。

## レビューと対応

- 1往復目: impl-planはOK。implはブラウザ受入テストの不足で差し戻し（機能コードの修正要求・範囲外の検出・開発者への確認事項はなし）。原文は `.scratch/prd111/review-20261102-1205.md`。
- 対応: A選択済みから開閉→B行のクリック/タップをcaptureで数えて厳密に2入力とし、不可視tagを含む選択全体とSelected Onlyの維持を確認。複数行のUTF-16順序とRun変更前後の見える数を確認。不可視数0のtitleとSave As件数を完全一致で確認。
- 追加テストの初回は既存helperがhidden行も読むため2件失敗し、可視行の検証を `li:visible` に修正した。productionの変更なし。
- `mvn -f apps/metrics-viewer/pom.xml -Dtest=TagSetPlaywrightTest -l C:/dev/anet-lab/.scratch/prd111/review-tests-2.log test`: UI14件、failure/error/skip=0、BUILD SUCCESS。全体274件成功後に受入テスト4件を追加したもので、packageのproduction内容は変えていない。
- 2往復目: 上記の受入検証の追加と結果を、同じan-reviewへ再提出する。
- 2往復目の結果: impl-plan / impl ともOK。修正要求・範囲外の検出・開発者への確認事項はすべてなし。原文は `.scratch/prd111/review-20261102-1209.md`。追加検証で前回の不足が解消されたと確認された。

## 完了報告

- 実施内容: PRD111の保存・呼び出し・Save/Revert・Save As・改名・削除・Deselect・共有・起動復元、初期値/作業用分離、厳密なAPI/ファイル検証を実装した。
- 変更ファイル: 計画を含む24ファイル。serverとrequest/response、browser UI、設定/初期値、テスト、release workflow、210/030日英、PRD111検証README。
- レビュー結果: an-review 2往復で両成果物OK。見送った指摘なし。全体274件と追加後UI14件が成功し、Debug・package・ローカル梱包・jar初期値・改行/BOMも確認した。
- 決まらなかったこと: 設計上の未決・起票候補はなし。remote workflow_dispatchによる本体zipの受入は未実施として残る（検証結果参照）。
- 判断してほしいこと: なし。
- 次にできること: 人間が変更を確認し、公開するrefを用意したあとremote workflow_dispatchで配布zipを確認する。作業用ファイルの追跡・commit/pushも人間が行う。

## cc-review のコメント指摘への対応

- ユーザーの指示で、cc-review初回のClaude側が指摘した段階の説明コメントを補った。対象はJSの4メソッド、release workflowのjar/zip検査、Playwrightの6テスト、Store/APIの各1テスト。仕様や処理は変更していない。
- 既存のライセンス検査コメントは対応するforeachの前へ移し、作業用タグセットの非混入検査には独立した目的の説明を付けた。
- 変更前を `.scratch/prd111/comments-before-20261102-1340/` に退避し、対象5ファイルについて空行と独立した行コメントを除いた各行が変更前と完全一致することをPythonで確認した。UTF-8・既存BOMの有無・LFも確認し、`git diff --check` は問題なし。
- 実行文・テスト条件の変更が無いため、新しいテストを作らず、ビルド・Mavenテストの再実行も省略した。前述の全体274件・追加後UI14件、Debug、package、ローカル梱包の検証結果を使う。
- 再レビューは同じCodexのan-reviewとClaudeセッション `7ed559e1-4546-4eb3-a180-46c0a1809069` に、同一の依頼を渡して行う。ユーザーガイド本文の再確認候補とremote workflow_dispatch未実施は、今回のコメント修正では変更しない。
