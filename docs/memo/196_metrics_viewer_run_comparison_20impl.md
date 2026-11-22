# PRD196 Metrics Viewer の Run 比較 実装メモ

## 概要

[PRD196](196_metrics_viewer_run_comparison_10prd.md) に従い、選んだ Run の設定表、Run・tag の文字による絞り込み、tag の情報パネル、Plotly のオフライン配信を実装する。対象課題は [issue-20261121-201669](../management/issues/issue-20261121-201669.md)。

## 主な変更

- `POST /api/config.json` の要求検証、Run ごとの読み込み結果、TXT の行規則とエラー診断を実装する。選択に追随する設定表、固定タブとグラフの更新停止・再開を加える。長いキーと値は折り返し、横スクロールが必要なときも検索と `Diff Only` を表示する。
- Run・tag・設定キーで Unicode 空白区切りの部分一致を共有する。Run・tag の一括操作は文字の一致を使い、Selected Only を解除する。
- `POST /api/range-stats.json` で各系列の全点を集計する。tag 名から開く情報パネルで All と View を並べ、表示範囲の更新を 200 ms 待って問い合わせる。コピー操作は Run の情報パネルと共有する。
- Plotly を jar 内から配信し、210・030・160 の日本語版を更新する。

## テスト

観測面は Playwright で操作する Viewer と、HTTP API 2 本。各行の最初の失敗を確認し、対応する挙動を実装してから次へ進む。

1. 設定表を経由してファイルの設定を読めることを 1 本の Playwright テストで通す（A5）。値と有無の差、検索、タブ再表示・再読み込みを確認する。
2. 設定 API が 1 要求の列ごとに状態を返し、不正な本文を 400 にすること、TXT の R1〜R13 が反映されることを確認する（A2・A4）。
3. 絞り込みに一致する行を選ぶ・外す操作を確認する（A6）。
4. 範囲統計の端点、空の範囲と未登録 Run および異なる channel の並行問い合わせを確認する（A7・A9）。
5. 表示範囲をドラッグで変え、情報パネルの値と範囲が更新されることを確認する。行ごとの状態、コピー、channel の再利用を確認する（A8・A9）。
6. ブラウザで外部通信を遮断し、同梱の Plotly で描画できることを確認する（A1）。
7. 狭い画面でもタブを切り替える。設定の長い行は折り返し、幅が足りず表を横へ送ったあとも検索と `Diff Only` を操作できることを確認する。

## 検証

- Viewer ディレクトリで対象テストを個別に実行し、最後に `mvn test` と `mvn package` を実行する。長時間の実行ログは `.scratch/prd196/` に保存し、要約と失敗箇所を確認する。
- A3 は実リポジトリの `data/workspaces` にある Run の TXT を実装した reader で読み、同じ Run の JSON と突き合わせる。比較件数、片方だけある件数、不一致件数、reader の失敗件数を記録する。
- A1 は配布形の jar 内 WebJar とライセンス一覧を確認する。A10 は日本語設計文書の旧 CDN 表記と API・用語を検索する。
- A12 は実 workspace で指定された描画・操作を目視比較し、開発者の判定を記録する。

## 前提

- PRD は起草時のレビュー済みで、D1〜D32 が確定している。用語集の「範囲統計」と「query channel」および ADR 0089・0090 を確認した。ユーザー判断が必要な実装前の未決は 0 件。
- `Graphs` を既定にし、タブの選択を保存する。絞り込みと `Diff Only` はページ内で保持する。設定の列は選んだ Run、差分の判定は読み込み成功の列を対象とする。
- 設定の読み込みは `Properties::Load` と PRD §6.3 に従う。workspace の固定には既存の lease、Run の解決には RunScanner を使う。
- 範囲統計は commit 済みの L0 を既存の step 検索と同じ閉区間で集計する。workspace の切り替えと同時実行枠を既存の query coordinator で扱う。グラフと範囲統計はタブ内で別系列とする。
- Plotly は Maven から WebJar 2.35.3 を同梱する。日本語の設計・利用ガイドを更新し、英訳の作り直しは別工程とする。

## 経過

- 2026-11-22 PRD、用語集、関連 ADR、現在の API とテストを照合し、実装計画を保存。
- 設定表、文字の絞り込み、範囲統計、tag の情報パネル、Plotly 同梱の順で Playwright と HTTP のテストを進めた。ブラウザ側の部品は既存の `metrics-viewer.js` に置いた。
- A3 は `data/workspaces` 配下の 741 Run を調べた。両方ある 738 Run の TXT を実装した `ConfigTextReader` で読み、JSON の `data` のキーと文字列値を照合した。不一致 0、片方だけ 0、両方なし 3、reader の失敗 0、JSON の失敗 0。実行ログ `.scratch/prd196/workspace-config-parity.log`。

## 検証結果

| 受入 | 確認結果 |
|---|---|
| A1 | `RunComparisonPlaywrightTest.graphRendersWithoutAnyRequestOutsideTheViewer` は CDN を読む変更前の実装で失敗し、同梱後は外部要求 0 件で描画。実行 jar の WebJar の中に `META-INF/resources/webjars/plotly.js-dist-min/2.35.3/plotly.min.js` があり、jar の `META-INF/THIRD-PARTY.txt` およびリリースの入力となるライセンス一覧に MIT の行がある |
| A2・A4 | `ConfigTextReaderTest` 11 件と `MetricsApiIntegrationTest.configRequestReportsEachRunIndependentlyAndRejectsInvalidBodies` が通過 |
| A3 | TXT と JSON の 738 件が一致。LunarLander-01 の実 Run 2 件では `inspect_run.py config --diff` の 35 キーと設定 API から求めた差分の 35 キーが一致 |
| A5・A6 | `RunComparisonPlaywrightTest` が実ファイルから設定表までの経路と、空白区切り・部分一致・絞り込み中の一括選択を確認。設定タブ、Runの文字絞り込みは、それぞれ追加前にテストが失敗することを確認 |
| A7・A9 | `MetricsApiIntegrationTest` で重複stepを含む閉区間の値、空・未登録・取り込み待ち、本文の違反、別channel間と同一channel内の打ち切りを確認。範囲統計のHTTPテストは追加前に404で失敗 |
| A8・A9 | `RunComparisonPlaywrightTest` でhover、全期間と範囲の列、欠けているRun、実際のドラッグによるViewの更新、コピー、別channelの再利用を確認。旧見出しの統計の期待値は既存の2テストを移し替えて確認 |
| A10 | 日本語の設計文書に旧CDNの3表記が0行。210の操作規約・API節と用語集の「範囲統計」を確認 |
| A11 | 最終版の `mvn test` 320件通過（失敗・エラー・スキップ各0件）。実行ログ `.scratch/prd196/full-test-verified.log` |
| A12 | LunarLander-01 の実 Run 2 件を作業用に複製し、旧版と新版で同じRunが ready になることを確認した。旧版は 127.0.0.1:18082、新版は 127.0.0.1:18083。開発者が指定の範囲で描画・操作に差がないことを確認し、合格 |

Maven はリポジトリルートから `mvn -B -f src/apps/metrics-viewer/pom.xml -Danet.viewer.build.directory=C:/dev/anet-lab/.scratch/prd196/mobile-build test` を実行した。テスト後に同じビルド出力から `mvn package` を `-DskipTests` と作業用の runtime 出力先を指定して実行した。最終版は `.scratch/prd196/runtime-final/metrics-viewer.jar`。梱包ログは `.scratch/prd196/package-verified.log`、WebJar内の資材とライセンスの確認は `.scratch/prd196/package-contents.log`。

Java のコンパイルとテストは Maven がデバッグ情報付きで行った。設定のTXTとJSONの比較は実装したJava readerを直接使用した。Runnerの起動を伴う試験はPRDの対象外で、英語版の追随はD28の別工程で行う。

- 追加依頼により、Configの長いキーと値は折り返す。横スクロールは表だけに限定し、キー検索とDiff Onlyは表の外に置く。
- 横幅390 pxではメインエリア左のタブが画面外になることを実ブラウザで確認した。狭い表示では右上の操作の下に一段を確保する。縦方向の余白が増える代わりに、タブ2個と既存の右上の操作を押せる。

## 自己点検

- 対象に当たる実装品質確認観点を通し、PRD の受入とテストの期待値を照合した。グラフの旧統計は情報パネルへ移し、既存の設定・Metrics API とキャッシュ様式を確認した。
- 閉じた本文、系列ごとの状態、workspace固定、グラフと範囲統計のキャンセル境界を確認した。追加のJavaクラスは設定読み込み、範囲集計とHTTP応答の責務を持つ。
- A12 の目視受入は開発者の確認で合格。

## レビューと追加の確認

- 初回は実装計画の節順と、幅390 pxのタブ、Config中のスクリーンショット切り替え、点なしと取り込み待ちの区別、整数stepの範囲、受入テスト、160のAPI一覧に修正が必要と判定された。原文は `.scratch/prd196/review-20261122-0320.md`。
- 計画の節順を直し、幅390 pxでのタブ操作を確認した。Config中はスクリーンショットを切り替えてもPlotlyの更新を保留する。範囲統計は表示範囲とsafe integer範囲の共通部分を使い、範囲が決まらない場合は既知の点なしと未確定の状態を区別する。
- TXTで個別指定の次に既定キーがあっても個別指定が残ること、未ズーム時の全行でAllとViewの各列が一致することを追加で確かめた。160のControllerのAPI一覧を更新した。
- ユーザー追加依頼のConfigを、短い行と長い行の両方で確認した。幅950 pxでは2 Runの長いキーと値を折り返して表を収め、幅660 pxでは表を横へ送った後も検索とDiff Onlyを操作できた。狭いタブ操作と点なしのテストでは変更前に失敗することを確認。対象のMavenテスト18件が通った（`.scratch/prd196/review-fixes-targeted.log`）。

- 2回目のレビューは計画がOK、実装は390 pxで検索に使える幅が足りず差し戻し。原文は `.scratch/prd196/review-20261122-0325.md`。狭いConfigではサイドパネルを上、操作と表を下へ並べ、キー列の幅は表を表示できる領域に合わせた。
- 幅390 pxで横スクロールした後に検索で入力し、Diff Onlyを押して目的のキーだけ残る回帰テストを追加した。修正前は検索欄の幅で失敗し、修正後のRun比較Playwright 7件が通った（`.scratch/prd196/narrow-controls-red.log`、`.scratch/prd196/narrow-controls-green.log`）。

- 最終の確認用jarでLunarLander-01の実Run 2件を表示。幅1280 pxで表の表示幅と内容幅はともに898 px。幅660 pxでは表を138 px、幅390 pxでは50 px横に送った。390 pxでキー検索は183 pxあり、スクロール後に入力してDiff Onlyを使えた。画面に出た差分キー35件はCLIと一致（`.scratch/prd196/visual/config-table-final.log`）。
- 3回目の独立レビューで、実装計画と実装がともにOK。指摘の残りはなし。原文は `.scratch/prd196/review-20261122-0329.md`。

## 完了報告

- 設定表・文字の絞り込み・tagの範囲統計・オフライン描画を実装した。Configは長いキーと値を折り返し、表を横に送ったあともキー検索とDiff Onlyを操作できる。
- 最終版でMaven全320テストが通過し、配布形のjarとPlotlyのMITライセンスを確認した。実Runの設定は738件一致。新旧描画の目視受入は開発者が合格と確認した。
- 独立レビューは3回目で実装計画・実装ともにOK。残件と新規課題候補はなし。

## 2026-11-22 追加調整

ユーザーの指示で、情報パネルは横軸が全点を含むときに `All` だけを表示する形に改めた。PRDのD21で扱った全期間と同じ集計は再取得しない。横軸がデータの一部を表示している間は `Viewing (開始-終了)` を表示する。手動で指定した範囲が全点を含む場合も `All` だけに戻す。

`node --check` と Maven の関連 Playwright テスト37件を確認した。MetricsPlotとOutlierRangeは各15件、テストでパネル再表示を待つようにしたあとのRunComparisonは7件が通過した。実 Run の LunarLander-01 で全体表示、`Viewing (15,559-46,674)` の追加、リセット後に `All` だけへ戻ることを確認した。ログは `.scratch/prd196/viewing-green.log`、`.scratch/prd196/viewing-comparison.log`、`.scratch/prd196/visual/viewing-smoke.log`。確認用jarを梱包し、ポート18083の確認用Viewerを更新した。

### タッチで情報パネルを開く

グラフのtag名をタップすると情報パネルを開く。変更前は短いタップで開かず、長押しではグラフのtag名をブラウザが選択できた。見出しの文字選択を抑え、パネル内のtag名はコピー用に選択できる。マウスではhover、クリック、右クリックで開く。

スマホ相当の幅390 pxで変更前のタップが失敗することを確認した。変更後はタップで開き、外側のタップで閉じる。スワイプで開かず、長押しでtag名を選択しない。関連Playwright 24件が通り、実RunのLunarLander-01でもスマホ相当のタップを確認した。ログは `.scratch/prd196/tag-touch-red.log`、`.scratch/prd196/tag-touch-green.log`、`.scratch/prd196/visual/tag-touch-smoke.log`。






