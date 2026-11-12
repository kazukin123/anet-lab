# MetricsViewer のタグセット（名前付きのタグ選択の保存と呼び出し）PRD

> 2026-11-02 起票（Claude Code、`anet-create-prd`。グリルあり。2026-11-01〜02）。グリルの前に画面モック（`.scratch/prd111/tag-list-mock.html`、git 対象外）で 3 案を比べ、案 B（開閉式の一覧）を採った。
> 2026-11-02 別モデル（Fable）の検討のあと（同じ枠）: 「見えている tag」を、コードのとおり選択中の Run の tag の和集合（Tags 一覧に出ている tag）と定めて文言を直した（§1.1 の 4、§3）。作業用ファイルの既定を `metricsviewer.workspaces-dir` の直下に改め、親が無ければ作ることにした（D16。既存テストと 210 §7.3 の起動を壊さないため。ユーザーの裁定）。外で変わった作業用ファイルを書く直前に照合して拒む案は退け、D12 に git の切り替えでも上書きされることを書いた（ユーザーの裁定）。Save As の入力欄の脇に保存される数を出すことにした（D17。ユーザーの裁定）。開閉ボタンの幅と省略、押せないボタン、localStorage の「なし」の表し方、tag の制御文字、release.yml の確かめ方を直した。
> 2026-11-10 PRD を起こさずに変えた（ユーザーの指示）: 選択中でないタグセットの未保存の変更をセットごとに browser へ残し、そのセットを呼ぶと保存済みの中身の代わりに戻すようにした。選択中のセットを選び直しても選択は変わらず、E10 の「選び直すと `*` が消える」は Revert だけになった（§6.3 の「呼び出し」も同じ）。正本は [210](../design/210_metrics_viewer.jp.md) §7.4・§9.6。
> 起点: 観点ごとに見たいタグの組を、130〜190 本のタグからクリックで毎回組み直している（§1.2）。
> 関連: [210 Metrics Viewer](../design/210_metrics_viewer.jp.md) §3.3・§7.1・§7.3・§7.4・§9、[030 Run分析ユーザーガイド](../design/030_user_guide_analysis.jp.md) §3、[ADR 0015](../adr/0015-metrics-cache-disposable-derivative.md)、[ADR 0021](../adr/0021-run-classification-by-workspace-folder.md)、[ADR 0067](../adr/0067-tag-sets-split-tracked-working-file-and-bundled-initial-copied-only-when-absent.md)。

## 1. 問題

### 1.1 今の事実（2026-11-01〜02 にコードと文書で確かめた）

| # | 事実 | 位置 |
|---|---|---|
| 1 | 選択中の tag は `activeTags` として localStorage に保存する。Workspace 名をキーに含めないので Workspace をまたいで 1 つで、Tags 一覧に出ていない tag（他の Workspace や選んでいない Run の tag）も選択に残る。`Clear All` は見えている tag しか外さないので、残った tag は消えない | [metrics-viewer.js:2864-2866](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)、:2919-2922、:1835-1841、[210 §7.4](../design/210_metrics_viewer.jp.md) |
| 2 | 一度も見たことのない tag（`knownTags` に無いもの）が見えると、自動で選択に入れる | [metrics-viewer.js:2548-2558](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) |
| 3 | Tags 一覧は tag 名の UTF-16 コード単位順に並び、クリックで選択を切り替える。`Select All` / `Clear All` は今見えている tag だけに効き、`Selected Only` を解除する | [metrics-viewer.js:1775-1847](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) |
| 4 | 「見えている tag」は、選択中の Run の tag の和集合である。Tags 一覧はこれを描き、グラフは見えている選択中の tag を名前順に描く。Workspace 全体の tag の和集合はクライアントに無い | [metrics-viewer.js:2535-2546](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)、:2192 |
| 5 | Workspace selector は、開く（フォーカス）たびに一覧をサーバーから取り直す | [metrics-viewer.js:1852](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)、:2111 |
| 6 | localStorage へ保存する state は 8 件と定めている。選択中の workspace が無いときは、そのキーを消して表す | [210 §7.4](../design/210_metrics_viewer.jp.md)、[metrics-viewer.js:2018-2019](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) |
| 7 | HTTP API は workspace・runs・metrics・prioritize だけで、利用者が作るデータを保存する API と保存先は無い。workspace 切替の不正 JSON を 400 `invalid_request` に変える処理は `WorkspaceController` の中に限り、metrics・prioritize の失敗の形には影響させない | [210 §9.1](../design/210_metrics_viewer.jp.md)、[WorkspaceController.java:50-57](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/WorkspaceController.java) |
| 8 | エラー応答の body は `{"code","message"}`（`ApiError`） | [ApiError.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model/ApiError.java) |
| 9 | Viewer は `apps/runner` を作業ディレクトリにして起動し、`metricsviewer.workspaces-dir` の既定は `workspaces`。別の Workspace 群を見るときは起動引数でこれを差し替える。設定の契約違反は起動を止める | [22_metrics_viewer_java.bat](../../apps/22_metrics_viewer_java.bat)、[210 §7.1・§7.3](../design/210_metrics_viewer.jp.md)、[MetricsViewerSettings.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/config/MetricsViewerSettings.java) |
| 10 | `metricsviewer.workspaces-dir` が存在しなくても Viewer は止まらず、Workspace の一覧が空のまま起動する | [WorkspaceManager.java:114](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceManager.java) |
| 11 | Workspace の一覧は、Runner・Viewer のどちらも Workspace 群の親の直下の**ディレクトリ**だけを数える。直下のファイルはどちらの一覧にも入らない | [app_util.cpp:423-438](../../core/anet-core/src/app_util.cpp)、[WorkspaceManager.java:114-118](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceManager.java)、:311-313 |
| 12 | `apps/runner/workspaces` は `.gitignore` の対象だが、各 Workspace の `config/_main.txt` は強制追加で追跡している。リリースの梱包は `apps/runner/workspaces` を丸ごとコピーし、`apps/metrics-viewer` からは jar だけをコピーする | [.gitignore:66](../../.gitignore)、[release.yml:211](../../.github/workflows/release.yml)、:219-221 |
| 13 | Spring を起動する既存のテストは、`metricsviewer.workspaces-dir` を一時ディレクトリか `target/playwright-test-empty-workspaces`（存在しないことがある）へ向けて走る。`mvn test` の作業ディレクトリは `apps/metrics-viewer` で、そこに `workspaces/` は無い | 例 [TagListPlaywrightTest.java:20](../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/TagListPlaywrightTest.java)、[WorkspaceApiIntegrationTest.java:46](../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceApiIntegrationTest.java) |
| 14 | browser の control は即時実行・トグル・巡回の 3 種類だけで、どれも button で作る。サイドパネルの button は同じ寸法にし、巡回のラベルは最長に合わせた固定幅にする。一括操作は無効化しない。サイドパネルで無効（disabled）になる control は、切替中の workspace selector だけ | [210 §3.3](../design/210_metrics_viewer.jp.md)、[metrics-viewer.js:2046](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)、:2073 |
| 15 | 左パネルのタグ一覧そのものを「Tag list」と呼んでいる（`#tag-list`、`renderTagList`、`TagListPlaywrightTest`、210 §3.2 の `UIController`） | [index.html:55-64](../../apps/metrics-viewer/src/main/resources/static/index.html)、[210 §3.2](../design/210_metrics_viewer.jp.md) |
| 16 | `html, body` と `#tag-section` は `overflow: hidden` である | [metrics-viewer.css:23-32](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.css)、:314-318 |

### 1.2 困っていること

観点ごとに見たいタグの組（成績、Q/TD、PER、可塑性など）があるが、保存できるのは今の選択 1 組だけで、しかもブラウザごとに別である。観点を変えるたびに `Clear All` と 1 本ずつのクリックで組み直している。

## 2. ゴールと非ゴール

### 2.1 ゴール

- **G1**: 今のタグの選択を名前付きで保存し、1 操作で呼び出せる。
- **G2**: 保存したものの名前変更・削除・中身の編集ができる。
- **G3**: サーバー側に保存し、Workspace とブラウザをまたいで共有する。
- **G4**: 操作を Tags の見出しの下の 1 行に収める。
- **G5**: 最後に選んだものを localStorage に随時保存し、クライアントの起動時に戻す。
- **G6**: リリース物に入れるきれいな定義と、普段の作業で散らかる定義を、リポジトリ上で別のファイルにする（グリルの途中でユーザーの指示により追加）。

### 2.2 非ゴール

| # | 非ゴール | 退ける案と、それが本当の制約か |
|---|---|---|
| NG1 | Run の選択・Run 色の保存 | 「Run の組も保存する」案を退ける。制約ではなく範囲の選択（別の関心）。Run 色は比較ごとの役割で保存しない（[ADR 0041](../adr/0041-run-color-is-comparison-role-not-run-identity.md)） |
| NG2 | 表示設定（Log、percentile、LOD、viewport）をタグセットに含めること | 「見え方ごと保存する」案を退ける。範囲の選択。Log と percentile は今も tag ごとに localStorage で共有している |
| NG3 | グラフの並び順をタグセットで決めること | 「タグセットの順にグラフを並べる」案を退ける。範囲の選択。グラフは今どおり tag 名順（§1.1 の 4） |
| NG4 | パターン（接頭辞・ワイルドカード）で tag を指すこと | 「群ごと指す」案を退ける。範囲の選択 |
| NG5 | 複数人で使うときの権限・ロック | 同時に書いたら後勝ち。範囲の選択 |
| NG6 | Workspace ごとのタグセット | G3 と反対の形。範囲の選択 |
| NG7 | ファイルへの書き出し・読み込みの UI | ファイルは手で写せる（D8）。範囲の選択 |

### 2.3 足す機構とゴールへの寄与

| # | 機構・概念 | 寄与 |
|---|---|---|
| S1 | タグセット（§3） | G1・G2・G3 |
| S2 | サーバー側の作業用ファイルと HTTP API（§6.1・§6.2） | G3 |
| S3 | 選択中のタグセットと localStorage のキー（§6.3） | G5、G2 の編集 |
| S4 | 変更の印 `*`、Save・Revert（§6.4） | G2 の編集 |
| S5 | Tags の下の 1 行と開閉する一覧（§6.4） | G4 |
| S6 | jar 同梱の初期値ファイル（§6.1） | G6 |

先送りしたものは無い。

## 3. 用語

- **タグセット（Tag Set）**: Metrics Viewer で名前を付けて保存した、表示する tag の順序の無い集合。Workspace をまたいで共有し、呼び出すと tag の選択がその中身に置き換わる。[CONTEXT.md](../../CONTEXT.md) に追加した。「タグリスト」は既存の Tags 一覧（§1.1 の 15）と紛れ、「プリセット」は CONTEXT.md が別の概念で避けている語、「タググループ」は tag 番号の群と紛れるので使わない。
- **見えている tag**: Tags 一覧に出ている tag。選択中の Run の tag の和集合である（§1.1 の 4）。「Tags 一覧に出ていない tag」は、選択中のどの Run にも無い tag をいう（他の Workspace の tag も、今の Workspace の選んでいない Run だけにある tag も含む）。
- **作業用ファイル**: Viewer が読み書きするタグセットの保存ファイル（§6.1）。
- **初期値ファイル**: jar に入れて配る、きれいに整えたタグセットの定義。作業用ファイルが無いときだけ写す（§6.1）。
- **選択中のタグセット**: 最後に呼び出した（または保存した）タグセット。無いこともある。
- **変更の印（`*`）**: 選択中のタグセットの中身と、今のタグの選択が違うことを示す印。

## 4. 確定事項

| # | 決定 | 理由 | 依存 |
|---|---|---|---|
| D1 | 呼び出すと、タグの選択全体をタグセットの中身で置き換える。Tags 一覧に出ていない tag も選択に入り、それまでの他の選択は外れる | どの Workspace・どの Run を選んでもタグセットどおりに見えるのは、全体の置き換えだけである。「見えている部分だけ置き換える」は呼んだときの選択で結果が変わり、「足し込む」は観点を切り替えるたびに `Clear All` が要る | G3、§1.1 の 1・4 |
| D2 | 保存（Save As・Save）は選択全体を書く。変更の印も選択全体で比べる | ユーザーの選択。最も単純に「呼び出し後の選択 = タグセット」が保たれる。Tags 一覧に出ていない tag の差でも `*` が付き、他の Workspace で選んだまま見えない tag も Save As に入ることを承知で選んだ（推奨は「見えている部分だけ」だった）。紛れ込みには D17 で保存の前に気づけるようにする | D1、§1.1 の 1 |
| D3 | タグセットを選んでいる間も、新しい tag は今どおり自動で選択に入れる。そのため `*` が付く | 新しいメトリクスを見落とさないための既存の働きを、タグセットの有無で変えない。崩れた見え方は Revert（D11）で戻せる | D2、§1.1 の 2 |
| D4 | 起動時は、選択中のタグセットの名前だけを localStorage から戻す。タグの選択は従来どおり `activeTags` から戻すので、未保存の変更と `*` も残る | 要望の「最後に選択したタグリストの選択状況を復帰」の 2 つの読み（どれを選んでいたか／その下での選択）を両方満たし、起動で変更を失わない。起動時に取った一覧と比べるので、他のブラウザで中身が変わっていれば起動後に `*` で分かる（起動後の変更は、一覧を開き直すまで比べない。E11） | D2 |
| D5 | 入口はモックの案 B。Tags の下の 1 行を `[選択中の名前 ▾][Save][Revert]` とし、▾ で開く一覧の中で呼び出し・名前変更・削除・新規保存を行う | ユーザーの選択。名前を行の残りの幅いっぱいに出せ、各タグセットの「Tags 一覧に出ている tag 数 / tag 数」を出せる（D2 のもとで見えない差の目安になる）。210 §3.3 に新しい部品（開閉）を足す（推奨は案 A のセレクト＋動詞ボタンだった） | §1.1 の 14 |
| D6 | 呼び名は「タグセット（Tag Set）」とする | 中身が順序の無い集合であることが名前で分かり、既存の「Tag list」とぶつからない。既存コードで「見えている tag の集合」の意味に一般語として使っている名前は、紛れないよう改名する（§8） | §1.1 の 15 |
| D7 | 作業用ファイルの置き場は設定キー `metricsviewer.tag-sets-file` で差し替えられる | ユーザーの選択。起動引数で別の場所を指して共有もできる | — |
| D8 | 作業用ファイルと初期値ファイルは、整形した JSON にする | ユーザーの選択。手で直せ、行単位の diff が読め、既存の Jackson で検証でき、名前にどんな文字が入っても曖昧にならない | — |
| D9 | HTTP API は 1 件ずつの操作を POST で受け、名前は body に入れる。書いたら全件を返す | 既存 API と同じ形である。名前に `/` や日本語が入っても通る。別々のタグセットへの同時操作が互いを消さない（全体を PUT する形では、古い一覧を持つ側の保存が他の追加・削除を消す） | §1.1 の 7 |
| D10 | 選択中のタグセットを外す操作（タグの選択は変えない）は、一覧の下の操作の区画に、名前を含めない固定文の動詞の行 `Deselect tag set` として置く | 呼び出し（選択が変わる）とは別の操作なので、呼び出しの行と同じ並びに置かず、区画と動詞で区別する。めったに使わないので常に見えるボタンにはしない。ラベルに名前を入れると名前が必要以上に目立つ（ユーザーの指示） | D1、D5 |
| D11 | Save の横に `Revert` を置き、選択中のタグセットを呼び出し直す | 「呼び出してから少し足し引きし、元に戻す」が頻繁に起きる（ユーザー）。選び直しより 1 操作で済む | D1 |
| D12 | 作業用ファイルは起動時だけ読む。稼働中に外で変わった内容（手で直したもの、git のブランチの切り替え・pull・stash で変わったもの）は再起動まで反映されず、その前に UI から保存すると上書きされて消える | ユーザーの選択（推奨は「変更を検知して読み直す」だった）。読み込みの経路を起動時の 1 つにする。書く直前にファイルの内容を照合し、変わっていれば 409 で拒む案も、別モデルの検討（2026-11-02）で git の切り替えでも消えると分かったうえで、ユーザーが退けた | D8、D14 |
| D13 | 初期値ファイルは、作業用ファイルが無いときだけ写して作業用ファイルを作る。以後は作業用ファイルだけを使う | G6 を、UI を 1 種類のまま満たす。新しい版を上書き展開しても利用者の作業用ファイルを壊さない。次のリリースで初期値が増えても、既に作業用ファイルを持つ利用者には届かない（作業用ファイルを消せば写し直される） | G6 |
| D14 | 作業用ファイルは Workspace 群の親の直下に置き（D16）、git で追跡する（`git add -f`）。リリースの梱包からは除く | ユーザーの選択。散らかっていても有用で、記録になり、失われると困るので追跡する。`apps/runner/workspaces` は `.gitignore` の対象なので強制追加が要り、梱包は Workspace を丸ごとコピーするので除外が要る（§1.1 の 12。推奨は Viewer のフォルダだった）。直下のファイルは Workspace の一覧に入らない（§1.1 の 11） | D7、D13、D16 |
| D15 | 初期値ファイルは jar の中のリソース `apps/metrics-viewer/src/main/resources/initial_tag_sets.json` に置く | 設定キーも梱包の変更も要らず、作業用ファイルと取り違えようがない | D13、§1.1 の 12 |
| D16 | `metricsviewer.tag-sets-file` の既定は、設定の式 `${metricsviewer.workspaces-dir}/metrics_viewer_tag_sets.json` とする（既定の起動では `apps/runner/workspaces/metrics_viewer_tag_sets.json`）。親ディレクトリが無ければ作り、作れなければ起動を止める | ユーザーの選択（別モデルの検討で、作業ディレクトリ基準の固定値では既存テストが起動で落ち、210 §7.3 の別の Workspace 群での起動で元の群に書くか起動が止まると分かった。§1.1 の 9・10・13）。既定を式で書くので特別な値の意味は持たない。別の Workspace 群で起動すれば、その群の中にタグセットが閉じる。workspaces-dir を打ち間違えるとその名前のフォルダができるが、今も Workspace が空で起動するので気づける。「親が無ければ起動を止める」案に対しては、既存テストの手当ては本当の制約ではない（クリーンブレーク方針のとおり同じ変更の中で移せる）。退けたのは、その手当ての手間と、workspaces-dir が無いだけで起動が止まる今より厳しい振る舞いを、ユーザーが比べて選ばなかったからである | D7、D14 |
| D18 | 初期値ファイルは、実装の時点では `{}`（0 件）で置く。整えたタグセットは、ユーザーが作業用ファイルから項目を写して別に commit する | ユーザーの選択。実装を初期値の中身の検討で待たせない。形式が同じ（D8）なので写すだけで済む | D8、D15 |
| D17 | Save As の入力欄の脇に、保存される tag の数（選択全体の数）と、そのうち Tags 一覧に出ていない数を出す（例 `55 tags (50 not in the list)`） | ユーザーの選択。D2 のもとで、見えない tag が新しいタグセットに紛れ込むことに保存の前に気づける（別モデルの検討の指摘） | D2、§1.1 の 1 |

## 5. 決定と規約から導いた事項

| # | 事項 | 根拠 |
|---|---|---|
| E1 | 保存・名前変更・削除・Revert・Deselect は即時実行の操作で、ラベルは動詞で始める。checkbox は使わない | [210 §3.3](../design/210_metrics_viewer.jp.md) |
| E2 | エラーは既存の Toast で知らせる | [210 §3.2](../design/210_metrics_viewer.jp.md) |
| E3 | API は閉じた schema で、未知の field・型違反は 400 `invalid_request`、名前と tag は JSON string だけを受け付ける | [210 §7.2](../design/210_metrics_viewer.jp.md) |
| E4 | タグセットを Metricsキャッシュにも Run フォルダにも置かない。作業用ファイルを Workspace 群の親の直下に置くのは Run の分類ではないので、ADR 0021 の「workspace フォルダ外に分類のメタデータを持たない」とは矛盾しない | [ADR 0015](../adr/0015-metrics-cache-disposable-derivative.md)、[ADR 0021](../adr/0021-run-classification-by-workspace-folder.md) |
| E5 | localStorage の値が読めない・形が不正なら、警告して「選択中のタグセットなし」で起動する | 既存の読み込み（[metrics-viewer.js:2878-2891](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)） |
| E6 | Workspace を切り替えても、選択中のタグセットは変えない | G3 |
| E7 | 同時に書いたら後勝ち | NG5 |
| E8 | 変更の印は、選択全体とタグセットの中身の差で決まる | D2 |
| E9 | Save As の直後は `*` が付かない | D2 |
| E10 | 同じタグセットを選び直すか Revert すると `*` が消える。Tags 一覧に出ていない tag の差も戻る | D1、D2、D11 |
| E11 | 一覧を開くたびにサーバーから取り直す。開き直すか Revert するまで、他のブラウザの変更は手元の一覧に入らない | D5、§1.1 の 5 |
| E12 | 210 §3.3 に「開閉」の部品を足す。あわせて、前提の無い即時実行を disabled にすることを足す（サイドパネルで初めての使い方） | D5、§6.4、§1.1 の 14 |
| E13 | 作業用ファイル・初期値ファイルが壊れていれば、パスかリソース名を含むエラーで起動を止める | [AGENTS.md](../../AGENTS.md) §Fail-Fast 原則（壊れた入力データ、起動時に検証できるものを潜伏させない）、[210 §7.1](../design/210_metrics_viewer.jp.md) |
| E14 | 初期値ファイルは、作業用ファイルがあっても起動のたびに同じ規則で検証する | E13 と同じ（開発中に壊したまま出荷しない） |
| E15 | 稼働中に外で変わった作業用ファイルの内容（手で直したもの、git の切り替えなど）は再起動まで反映されず、その前の UI 保存で消える。このことをユーザーガイドに書く | D12 |
| E16 | `Select All`・`Clear All`・`Selected Only` の動きは変えない。選択を変えるので `*` の判定に入る | D2、§1.1 の 3 |
| E17 | 既存のテストは、サイドパネルの操作行の一覧を確かめるテスト（§8）の期待に新しい行を足すほかは、変えずに通る。workspaces-dir を向けた先（一時ディレクトリか `target/` の下）に、作業用ファイルが作られる | D16、§6.4、§1.1 の 13 |
| E18 | 必須の field が無い body は 400 `invalid_request` | E3、[210 §9.1](../design/210_metrics_viewer.jp.md)（workspace API は必須 field の違反を 400 にしている） |
| E19 | 起動時に作業用ファイルを書けない（初期値を写せない）ときは、パスを含むエラーで起動を止める | [AGENTS.md](../../AGENTS.md) §Fail-Fast 原則（起動時に検証できるものを潜伏させない）、E13 |
| E20 | 一覧をいつ取り直しても、選択中のタグセットがその一覧に無ければ「なし」にして Toast を出す（起動時・一覧を開いたとき・自分が書いた応答・Revert の前・404 と 409 のあとで同じ）。ただし自分の名前変更・削除の応答では、先にその操作の扱い（選択中の名前を新しい名前へ移す、選択中を「なし」にする）を当ててから判定するので、自分が改名・削除した分には Toast を出さない | §6.3 の起動時・Revert・404/409 の扱いを、取り直すすべての時機に揃える。名前だけ残すと、変更の印に比べる相手が無くなる。自分の操作の結果を「他で消えた」と知らせない |

## 6. 契約

### 6.1 ファイル

**作業用ファイル**

- 設定キー `metricsviewer.tag-sets-file`。値は文字列のパスで、相対パスは作業ディレクトリ基準（`metricsviewer.workspaces-dir` と同じ基準）。既定は設定の式 `${metricsviewer.workspaces-dir}/metrics_viewer_tag_sets.json`（D16）。
- 起動時の検証（違反は `Invalid configuration: key=metricsviewer.tag-sets-file value=<値> expected=<期待>` で起動を止める）:
  - パスが既存のディレクトリである。
  - 親ディレクトリが無く、作ることもできない（親ディレクトリが無ければ作る）。
- 読み込みは起動時だけ（D12）。以後はメモリ上の一覧を正とし、変更のたびにファイルへ書く。

**初期値ファイル**

- jar のリソース `initial_tag_sets.json`（ソースは `apps/metrics-viewer/src/main/resources/initial_tag_sets.json`）。設定キーは持たない。
- リソースが無いときは、リソース名を含むエラーで起動を止める（壊れた配布物）。

**起動の手順**

1. 作業用ファイルのパスを解決し、上の検証を行い、親ディレクトリが無ければ作る。
2. 初期値ファイルを読み、下の「形と検証」を通す（E14）。
3. 作業用ファイルがあれば読み、同じ検証を通す。無ければ、初期値ファイルの中身を下の「書き方」で作業用ファイルへ書き、INFO で `Created tag sets file <パス> from initial_tag_sets.json` を出す（D13）。書けなければ、パスを含むエラーで起動を止める（E19）。

**形と検証**

- 最上位は JSON オブジェクトで、キーがタグセットの名前、値が tag の文字列の配列である。

```json
{
  "PER": [
    "39_agent_per/05_sample_actor_init_ratio",
    "39_agent_per/64_priority_stale_ratio"
  ],
  "eval 成績": [
    "20_eps/11_train_reward_ema",
    "21_eval/09_greedy_reward"
  ]
}
```

- 次のどれかに当たれば、パス（初期値ならリソース名）と、何が違うか（名前、位置）を含むエラーにする: JSON として読めない、最上位がオブジェクトでない、値が文字列の配列でない、名前が §6.2 の名前の規則に反する（前後の空白を含む名前も反則として扱う）、同じ名前のキーが 2 回ある、同じタグセットに同じ tag が 2 回ある、tag が §6.2 の tag の規則に反する。
- 空のタグセット（`[]`）と、タグセット 0 件（`{}`）は正しい。
- tag がどの Run にも無いことは検証しない（Run を消してもタグセットの中身を失わない）。

**書き方**

- 名前の UTF-16 コード単位順、各タグセットの中は tag の同じ順で、2 字下げ・1 行 1 tag・UTF-8（BOM なし）・LF・末尾に改行 1 つ。サーバーは保存のたびにファイル全体をこの形で書き直す（手で入れた並びや空白は保存で整う）。
- 同じディレクトリの一時ファイルに書いてから置き換える。書けなかったときは、その要求を 500 `tag_sets_write_failed`（パスを含む）にし、メモリ上の一覧も変えない。

**リポジトリとリリース**

- 作業用ファイル（既定の起動の場所 `apps/runner/workspaces/metrics_viewer_tag_sets.json`）は `git add -f` で追跡する（D14。追跡と commit は人間が行う）。
- `.github/workflows/release.yml` は Workspace をコピーしたあと、ステージの作業用ファイルを消し、無いことを確かめる（残っていれば梱包を失敗させる。§8）。初期値ファイルは jar に入るので、梱包の変更はこの除外だけである。

### 6.2 HTTP API

すべて `/api` 配下。body は閉じた schema で、未知の field・必須の field の欠け・型違反・JSON として読めない body は 400 `invalid_request`（E3・E18）。この変換は workspace 切替とタグセットの Controller だけに当て、metrics・prioritize の失敗の形には影響させない（§1.1 の 7）。エラーの body は `{"code","message"}`（§1.1 の 8）。

| 呼び出し | body | 成功 | 失敗 |
|---|---|---|---|
| `GET /api/tag-sets.json` | なし | 200 全件 | — |
| `POST /api/tag-sets/create` | `{"name": string, "tags": [string]}` | 200 全件 | 400 `invalid_request`（名前の規則・tag の規則違反を含む）、409 `tag_set_exists`、500 `tag_sets_write_failed` |
| `POST /api/tag-sets/update` | `{"name": string, "tags": [string]}` | 200 全件 | 400、404 `unknown_tag_set`、500 |
| `POST /api/tag-sets/rename` | `{"name": string, "newName": string}` | 200 全件。前後の空白を除いたあとで `newName` が `name` と同じなら何もしない（下の判定の順） | 400、404（`name` が無い）、409（`newName` が別のタグセットにある）、500 |
| `POST /api/tag-sets/delete` | `{"name": string}` | 200 全件 | 400、404、500 |

- 全件の応答: `{"tagSets": [{"name": string, "tags": [string]}, …]}`。名前の UTF-16 コード単位順、tag も同じ順。
- **前後の空白**: Unicode の White_Space 属性を持つ文字（全角空白 U+3000、タブ、改行、改行しない空白 U+00A0 を含む）。クライアントとサーバーで同じ定義を使う。
- **名前の規則**: 前後の空白を除いてから判定・照合・保存する。除いたあとが空なら不可、64 文字（Unicode のコードポイント数）まで、制御文字は不可（除いたあとの内側のタブなど）。大文字小文字は区別する（`PER` と `per` は別のタグセット）。この規則は、作る名前（`create` の `name`、`rename` の `newName`）にも、照合に使う名前（`update`・`rename`・`delete` の `name`）にも当てる。例: `"　PER"`・`"\tPER"` は `PER` として扱う。`"   "` は 400。`"P\tER"` は 400。
- **tag の規則**: 空でない文字列で、制御文字を含まない。同じ要求の中に同じ tag が 2 回あれば 400。tag がどの Run にも無いことは検証しない。
- **判定の順**: 失敗の条件が重なったときは、次の順で最初に当たったものを返す。(1) body の形（E3・E18）→ 400。(2) 名前の規則（上の両方の名前）→ 400。(3) tag の規則 → 400。(4) 照合に使う `name` が無い（`update`・`rename`・`delete`）→ 404。(5) 作る名前が既にある（`create` の `name`、`rename` で空白を除いて `name` と違う `newName`）→ 409。(6) 書き込み → 500。`rename` で空白を除いた `newName` が `name` と同じときは、(4) を通ったあとで何もせず 200 を返す。例: `delete {"name":"   "}` は 400、名前が無い `update` に重複した tag があれば 400、既にある名前の `create` に制御文字の tag があれば 400、`name` が無く `newName` が別のタグセットにある `rename` は 404、`name` が無く `newName` が同じ `rename` も 404。
- 書き込みはサーバー内で 1 件ずつ順に行う。同じタグセットを 2 つのブラウザが書けば後勝ち（E7）。

### 6.3 クライアントの状態

- 選択中のタグセットの名前を持ち、変えるたびに localStorage のキー `anet.metricsviewer.tagSet` へ書く。値は名前の JSON string で、選択中が無いときはキーを消す（§1.1 の 6 の workspace のキーと同じ表し方）。210 §7.4 の localStorage の state は 9 件になる。
- 手元の一覧を持つ。取り直すのは、起動時、一覧を開いたとき、自分が書いた応答を受けたとき、Revert の前である。
- 一覧を取り直すたびに、選択中のタグセットがその一覧に無ければ「なし」にして Toast（`Tag set "<名前>" no longer exists.`）を出す（E20）。
- 起動時は、名前を localStorage から戻し、一覧を取ったあとで上の扱いを当てる。起動時に一覧の取得に失敗したときは Toast を出し、名前は変えず、変更の印は判定しない（Save・Revert は押せない）。
- 起動時以外で一覧の取得に失敗したとき: 一覧を開くときなら Toast を出し、手元の古い一覧で開く（E11 のとおり手元の一覧はもともと古いことがある）。起動時の取得にも失敗していて手元に一覧が無いときは、開かずに Toast だけを出す（空の一覧を「まだ無い」と見せない）。Revert の前なら Toast を出し、Revert を行わない（Revert は最新の中身へ戻す操作なので、古い一覧では戻さない）。
- 書き込み（create・update・rename・delete）の要求が応答を受け取れずに失敗したときは、Toast を出し、選択中のタグセットとタグの選択は変えない。
- 変更の印は、選択中のタグセットがあり、選択全体（localStorage の `anet.metricsviewer.activeTags` に保つタグの選択）と手元の一覧のそのタグセットの中身が集合として違うときに付く。
- **呼び出し**: 選択全体を中身で置き換え、選択中にし、一覧を閉じる。`Selected Only` の状態は変えない。グラフの描き直しとデータの取得は、tag の選択を変えたときと同じ流れで行う。
- **Save**: `update` に選択全体を送る。
- **Save As**: 入力した名前で `create` に選択全体を送り、成功したらそのタグセット（前後の空白を除いた名前）を選択中にして一覧を閉じる。
- **Revert**: 一覧を取り直してから、選択中のタグセットを呼び出す。取り直した一覧に無ければ「なし」にして Toast（E20）。
- **Deselect**: 選択中を「なし」にする。タグの選択は変えない。一覧を閉じる。
- **名前変更**: `rename` を送る。成功の応答を受けたら、選択中のタグセットなら選択中の名前と localStorage を新しい名前（前後の空白を除いた名前）へ移し、そのあとで応答の一覧に E20 を当てる（Toast は出ない）。
- **削除**: 確認のあと `delete` を送る。成功の応答を受けたら、選択中のタグセットなら「なし」にし、そのあとで応答の一覧に E20 を当てる（自分で消したので Toast は出さない）。タグの選択は変えない。
- 404・409 を受けたら、サーバーの message を Toast に出し、一覧を取り直す（選択中のタグセットが無くなっていれば E20）。
- 新しい tag の自動選択（§1.1 の 2）は変えない（D3）。
- 「Tags 一覧に出ている tag」の数は、見えている tag（§3）で数える。選択する Run を変えると、一覧の `n/m` と差の数も変わる。

### 6.4 UI

**行（Tags の見出しの下の 1 行）**: `[開閉ボタン][Save][Revert]`。Runs の見出しの下の操作行（`Recolor`・`Auto Recolor`）と同じ作りの操作行にし、サイドパネルの操作行が溢れず押し潰れないことを確かめる既存のテスト（§8）の対象に入れる。

- 開閉ボタン: 行のうち `Save`・`Revert` を除いた残りの幅を占める。ラベルは「選択中の名前」＋変更の印のときだけ ` *`＋`▾`。名前が入りきらなければ名前の文字だけを省略記号で切り、` *` と `▾` は常に見えるようにする。ボタン自体の中身は溢れさせない（上のテストの「押し潰れ」に当たらない）。選択中が無いときは薄い色の `Tag Set`。一覧を開いている間は `.active` と `aria-expanded="true"` を付ける。変更の印があるとき、title に差を出す（例: `Modified: +2 −1 (1 not in the list)`。足した数・外した数・そのうち Tags 一覧に出ていない tag の数。出ていない数が 0 なら括弧ごと省いて `Modified: +2 −1`）。
- `Save`・`Revert`: サイドパネルの他の button と同じ寸法で、位置は名前の長さで動かない。変更の印があるときだけ押せ、無いときは disabled にする（押しても何もしない状態を作らない）。disabled の見た目は、文字と枠を `--control-border` より弱い段にする。
- 210 §3.3 に足すこと: 4 つめの種別「開閉」（ラベルは今の値と `▾`、行の残りの幅を占めて省略記号で切る、開いている間は `.active` と `aria-expanded` を対で更新する。トグルの `aria-pressed` は使わない）と、「前提の無い即時実行は disabled にする（一括操作は従来どおり無効化しない）」。

**開閉する一覧**

- 上の区画: タグセットの行を名前の UTF-16 コード単位順に並べる。各行は、名前（選択中なら強調し、変更の印があれば ` *`）、`<Tags 一覧に出ている tag 数>/<tag 数>`（title: `Tags shown in the list / tags in the set`）、`✎`（名前変更）、`×`（削除）。`✎`・`×` は常に薄く表示し、行に乗ると明るくする（タッチ端末には「乗る」操作が無いため）。行を押すと呼び出す。タグセットが 0 件なら `No tag sets yet.` を出す。
- 下の区画（区切り線の下）: `+ Save current selection as…`（押すとその場に名前の入力欄と `Save`・`Cancel`、入力欄の脇に `<保存される tag の数（選択全体の数）> tags (<そのうち Tags 一覧に出ていない数> not in the list)` を出す。例: 見えている 3 本と見えない 2 本を選んでいれば `5 tags (2 not in the list)`。出ていない数が 0 なら括弧ごと省いて `3 tags`。D17）。選択中のタグセットがあるときだけ `Deselect tag set`（title: `Keep the current tag selection and stop tracking the tag set`）。
- 名前変更: 行の中が入力欄と `Rename`・`Cancel` に替わる。Enter で確定、Esc で取り消し。
- 削除: 行の中が `Delete "<名前>"?` と `Delete`・`Cancel` に替わる。
- 同名の Save As・Rename（409）は Toast を出し、入力欄は値を残したまま開いておく。
- 閉じる: 外側のクリック、Esc（入力欄の中の Esc は入力の取り消しだけ）、呼び出し、Save As の成功、Deselect。名前変更・削除のあとは開いたまま。
- キーボード: 開いている間、↑↓ で行（タグセットの行と下の区画の行）を移り、Enter で呼び出しまたは操作、Esc で閉じる。
- 一覧はページの最前面に重ね、`html, body`・`#tag-section` の `overflow: hidden`（§1.1 の 16）で切れないようにする。

## 7. 退けた形

グリルと別モデルの検討で比べて退けた形と、その理由。wiki の「選ばなかった形」の出典になる。

| 論点 | 退けた形 | 理由 |
|---|---|---|
| 呼び出し（D1） | 見えている部分だけ置き換える／足し込む | 呼んだときの選択で結果が変わる／観点の切替に `Clear All` が要る |
| 保存の範囲（D2） | 見えている部分だけ書き、他の部分は残す | ユーザーが単純さを選んだ |
| 入口の形（D5） | セレクト＋動詞ボタン（案 A）／チップの横並び（案 C） | 名前の欄が約 93px に細る／数が増えると隠れ、名前変更・削除が隠れ操作になる |
| API（D9） | 名前をパスに入れる REST 風／全体を GET・PUT | `%2F` を Tomcat が既定で拒む／古い一覧を持つ側の保存が他の変更を消す |
| 外す操作（D10） | 一覧の先頭の `(none)` 行／行の専用ボタン／開閉ボタンの中の × | 呼び出しと見分けがつかない／めったに使わないのに幅を取る／§3.3 に無い形でタッチで押しにくい |
| 稼働中の外の変更（D12） | 変更を検知して読み直す／書く直前に照合して 409 で拒む | ユーザーが読み込みの経路を 1 つにし、照合も足さない単純さを選んだ |
| リリース用の定義（D13） | 組み込みの区画として作業用と並べる（書き換え不可）／梱包で作業用の場所に置くだけ | UI・API が一回り大きくなり同名の扱いが要る／上書き展開で利用者の作業用ファイルを消す |
| 作業用の置き場（D14） | Viewer のフォルダ（`apps/metrics-viewer/tag_sets.json`） | ユーザーが Workspace 群と一緒に置くことを選んだ |
| 初期値の置き場（D15） | 追跡するファイルを新しい設定キーで指す | 設定キーが増え、Runner の config フォルダに Viewer のファイルが混ざる |
| 既定のパス（D16） | 作業ディレクトリ基準の固定値／workspaces-dir 直下で親が無ければ起動を止める | 既存テストが起動で落ち、別の Workspace 群での起動で元の群に書くか止まる／存在しないフォルダで起動する既存テストに手当てが要り、workspaces-dir が無いだけで起動が止まる（テストの手当て自体は本当の制約ではなく、ユーザーが手間と厳しさを比べて選ばなかった） |
| Save As の数（D17） | 出さない | 見えない tag の紛れ込みに保存のあとでしか気づけない |

## 8. 移行対象

既存の契約を置き換えるものは無い（足すだけ）。今の振る舞いのうち変わるのは、localStorage の state の件数、サイドパネルの Tags の区画の高さ（1 行増える）、Viewer の起動で workspaces-dir（既定の起動では `apps/runner/workspaces`）が無ければ作られることである。

| 対象 | 直すこと |
|---|---|
| [210 Metrics Viewer](../design/210_metrics_viewer.jp.md) と [英語版](../design/210_metrics_viewer.en.md) | §3.1 に `TagSetStore`・`TagSetController`、§3.2 に `TagSetMenu`、§3.3 に「開閉」の種別と disabled の規則（§6.4）、§4 のコードマップ、§7.1 に `metricsviewer.tag-sets-file`、§7.3 の別の Workspace 群の起動例にタグセットもその群の中に作られること、§7.4 の「8件」を9件にし `anet.metricsviewer.tagSet` を足す、§9 にタグセットの API の節、§9.1 の「不正 JSON の変換は `WorkspaceController` 内に限定」をタグセットの Controller を含む形に |
| [030 Run分析ユーザーガイド](../design/030_user_guide_analysis.jp.md) と [英語版](../design/030_user_guide_analysis.en.md) | §3 の操作表にタグセットの行（呼び出し・Save・Revert・Save As・名前変更・削除・Deselect）。§3 の localStorage の段落に選択中のタグセット。作業用ファイルの場所、初期値から写されること、稼働中に外で変わった内容は再起動まで反映されず UI 保存で消えること（E15） |
| `apps/metrics-viewer/src/main/resources/application.properties` | `metricsviewer.tag-sets-file=${metricsviewer.workspaces-dir}/metrics_viewer_tag_sets.json` |
| `apps/metrics-viewer/src/main/resources/META-INF/additional-spring-configuration-metadata.json` | 同じキーの説明 |
| `apps/metrics-viewer/src/main/resources/initial_tag_sets.json` | 新規。中身は `{}`（D18） |
| `apps/metrics-viewer/src/main/resources/static/index.html`・`metrics-viewer.css`・`metrics-viewer.js` | 行と一覧。`_visibleTagSet()` を紛れない名前（例 `_visibleTagKeys()`）へ改名し、呼び出し元を追随（D6） |
| [.github/workflows/release.yml](../../.github/workflows/release.yml) | Workspace のコピーのあとで、ステージの `apps/runner/workspaces/metrics_viewer_tag_sets.json` を消し、無いことを確かめる（残っていれば失敗） |
| [RunListPlaywrightTest.java](../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/RunListPlaywrightTest.java) の `sidePanelControlRowsFitOnOneLineWithoutShrinkingButtons`（:719-739） | サイドパネルの操作行の一覧の期待に、Tags の新しい行 `tag-section/controls:false:false` を `tag-section/header` のあとへ足す（新しい行もこの溢れ・押し潰れの検査に入れる。§6.4、E17） |
| [CONTEXT.md](../../CONTEXT.md) | 「タグセット」を追加済み（起草の前に足した） |
| [wiki の Metrics Viewer のページ](../wiki/apps/120_metrics_viewer.md) | 起草で書き戻し済み |

## 9. クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場 |
|---|---|---|---|
| `TagSetStore` | 新規 | 作業用ファイルのパスの解決と検証・親ディレクトリの作成、初期値ファイルの検証と写し、メモリ上の一覧、1 件ずつの変更とファイルへの書き込み | server の service |
| `TagSetController` | 新規 | `/api/tag-sets*` を公開し、閉じた schema の body の読み取り失敗と `TagSetStore` の失敗を `ApiError` の応答へ変える | server の view |
| `GetTagSetsResponse` | 新規 | 全件の応答（既存の `GetRunsResponse`・`GetWorkspacesResponse` と同じ名付け） | server の view/model |
| `TagSetInfo` | 新規 | 応答の 1 件（名前と tag。既存の `RunInfo`・`TagInfo` と同じ名付け） | server の view/model |
| `SaveTagSetRequest` | 新規 | create・update の body | server の view/model |
| `RenameTagSetRequest` | 新規 | rename の body | server の view/model |
| `DeleteTagSetRequest` | 新規 | delete の body | server の view/model |
| `TagSetMenu` | 新規 | Tags の下の行と開閉する一覧の描画、入力欄・確認・キーボード・外側クリックの扱い | browser（`metrics-viewer.js`） |
| `MetricsViewerClientApp` | 変更 | 選択中のタグセットと手元の一覧、変更の印の判定、呼び出し・Save・Save As・Revert・Deselect・名前変更・削除、localStorage のキー | browser |
| `DataFetcher` | 変更 | タグセットの API の呼び出し | browser |
| `UIController` | 変更 | 行と一覧の bind、Tags 一覧の描き直しとの連動 | browser |

## 10. 受入基準

1. **API の結合テスト**（`mvn test`、新規 `TagSetApiIntegrationTest`）: §6.2 の各呼び出しで、成功時に全件が名前順で返る。409・404・400・500 がそれぞれの code で返る。400 は、未知の field、必須の field の欠け（`tags` の無い `create`、`newName` の無い `rename`）、型違反、名前の前後の空白を除いて空（`"   "`）、65 文字、名前と tag の制御文字（`"P\tER"`）、同じ tag の重複で確かめる。前後の全角空白とタブは除かれる（`"　PER"`・`"\tPER"` で `create` すると `PER` ができ、既に `PER` があれば 409）。rename の同名（空白を除いたあとで同じ）は何も変えずに 200。§6.2 の判定の順の例（`delete {"name":"   "}` は 400、名前が無い `update` に重複した tag は 400、既にある名前の `create` に制御文字の tag は 400、`name` が無く `newName` が別のタグセットにある `rename` は 404、`name` が無く `newName` が同じ `rename` は 404）がそのとおりに返る。書き込み先を書けなくした状態で 500 `tag_sets_write_failed` が返り、続く `GET` の一覧が変わっていない。
2. **Store の単体テスト**（新規 `TagSetStoreTest`）: 作業用ファイルが無いとき、初期値ファイルの中身が §6.1 の書き方どおりのバイト列で作業用ファイルに書かれる（名前順、tag 順、2 字下げ、LF、BOM なし）。作業用ファイルがあるときは初期値を写さない。親ディレクトリが無いときは作ってから書く。作業用ファイルが壊れているとき（§6.1 の各違反。同じ名前のキーが 2 回ある場合を含む）、初期値ファイルが壊れているとき、パスがディレクトリのとき、作業用ファイルが無く初期値を書けないとき（親ディレクトリに書けない）に、それぞれ起動に当たる生成が失敗し、message にパスかリソース名かキー名が入る。設定を与えないとき、作業用ファイルが workspaces-dir の直下に作られる。
3. **Playwright**（新規 `TagSetPlaywrightTest`）: 利用者の入力（クリック・キー）で次を確かめる。
   - タグセット A を選んだ状態から B への切り替えが、開閉ボタンを押す・B の行を押すの 2 回の入力で済み、そのあとの選択が B の中身と一致する（§11 の測り方）。
   - 呼び出すと、Tags 一覧に出ていない tag を含めて選択がタグセットの中身と一致する（localStorage の `anet.metricsviewer.activeTags` で確かめる）。
   - tag を 1 本切り替えると開閉ボタンに ` *` が付き、title に差の数が出て（Tags 一覧に出ていない差が 0 なら括弧が無い）、`Save`・`Revert` が押せるようになる。`Save` で印が消え、`GET` の中身が選択と一致する。印が無いとき `Save`・`Revert` は disabled である。
   - 切り替えたあと `Revert` で選択が戻り、印が消える。
   - `Deselect tag set` で選択中が無くなり、`anet.metricsviewer.activeTags` は変わらず、localStorage の `anet.metricsviewer.tagSet` が消える。
   - 見えている 3 本と見えない 2 本を選んだ状態で `+ Save current selection as…` を押すと、入力欄の脇に `5 tags (2 not in the list)` が出る。見えない tag が無ければ `3 tags` のように括弧が無い。作ったタグセットが選択中になる。同名なら Toast が出て入力欄が残る。
   - 選択中のタグセットを行の中で名前変更すると、選択中の名前と localStorage の `anet.metricsviewer.tagSet` が新しい名前（前後の空白を除いた名前）に追随し、Toast は出ない。選択中のタグセットを削除の確認で消すと「なし」になり、Toast は出ない。
   - 起動時と一覧を開くときの取得がどちらも失敗する状態（サーバーの API を失敗させる）で開閉ボタンを押すと、一覧は開かず Toast が出る。
   - 行の並びが UTF-16 コード単位順（例: `Atari スコア`、`PER`、`eval 成績`、`可塑性` の順）で、`<見える数>/<数>` が出る。選択する Run を変えると見える数が変わる。
   - 長い名前は開閉ボタンで省略記号で切れ、` *` と `▾` が見え、`Save`・`Revert` の位置が動かない。
   - ページを読み直すと、選択中の名前と ` *` が戻る。
   - 2 つめのブラウザ文脈で一覧を開くと、1 つめで作ったタグセットが見える。1 つめで消したタグセットを 2 つめで選んでいるとき、2 つめで Revert しても、一覧を開いても、Toast が出て「なし」になる（E20）。
   - タグセットを選んだ状態で、Run の metrics に新しい tag を追記してから `Reload` を押すと、新しい tag が自動で選択に入り ` *` が付く（D3）。
   - 一覧を開いて ↑↓・Enter で呼び出せ、Esc で閉じる。
4. **既存のテスト**: `mvn test` の既存のテスト（`TagListPlaywrightTest`、`WorkspaceSelectorPlaywrightTest` などを含む）がすべて通る。テストのコードと設定の変更は、§8 の `RunListPlaywrightTest` の操作行の期待に新しい行を足すことだけで、そのテストで新しい行も溢れず押し潰れていない（E17）。
5. **梱包**: release workflow のドライラン（workflow_dispatch）で作った zip に `apps/runner/workspaces/metrics_viewer_tag_sets.json` が無く、`apps/metrics-viewer/target/metrics-viewer.jar` の中に `initial_tag_sets.json` がある。
6. **文書**: §8 の 210・030（日英）が直っている。210 §7.4 の表に `anet.metricsviewer.tagSet` があり、件数が 9 件になっている。210 §3.3 に「開閉」の種別と disabled の規則がある。

## 11. 成功の測り方

- 観点の切り替え（タグセット A から B へ）が、開閉ボタンを開く・行を押すの 2 操作で済む。受入 3 の Playwright で、利用者の操作の数として数える（今は `Clear All` と tag の本数分のクリック）。
- 別のブラウザ文脈で、同じタグセットが見える（受入 3）。
- 少し足し引きしたあと元へ戻すのが `Revert` の 1 操作で済む（受入 3）。

## 12. 未決表

なし。

- 初期値ファイルの中身は、自己点検のあとの確認で D18 に決めた。
- レビュー（1 往復目）の指摘で決める必要が出た 3 件は、推奨で本文に書き、レビューが OK になったあとでユーザーが推奨どおりに決めた（2026-11-02）: 失敗の条件が重なったときの判定の順と、照合に使う名前にも名前の規則を当てること（§6.2「判定の順」）、「前後の空白」の範囲（§6.2）、起動時以外で一覧の取得や書き込みが失敗したときのクライアントの扱い（§6.3）。
