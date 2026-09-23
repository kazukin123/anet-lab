# 指摘台帳(派生ビュー)

生成 2026-09-23。正本は findings.jsonl。この文書は毎回再生成される。

## 概況

| 状態 | 件数 |
|---|---|
| open | 5 |

open の優先度別: P0 0, P1 2, P2 3, P3 0

open の観点別: architecture 0, backlog 0, defect 4, drift 0, process 0, refactor 1, test-gap 0

## 最近の変化(14 日)

| id | 更新 | 状態 | 優先度 | 観点 | 場所 | タイトル |
|---|---|---|---|---|---|---|
| F-0001 | 2026-09-23 | open | P1 | refactor | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.outlierPercentile | 排他的な外れ値表示モードを二つのSetと分散した排他処理で保持している |
| F-0002 | 2026-09-23 | open | P2 | defect | `apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java`::LodPageCache.find | 未完成のLOD pageをbucketごとに全行読み直すため、LOD射影で読む行数が表示bucket数の2乗になる |
| F-0003 | 2026-09-23 | open | P1 | defect | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::PlotlyController.renderBySelection | window内に点が無いtagはgraph blockごと消え、全tag一括のReset View以外では戻せない |
| F-0004 | 2026-09-23 | open | P2 | defect | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.onReload / onWorkspaceChanged | DataFetcherの単一のabort枠を独立した呼び出し元が奪い合い、abortが失敗Toastや初期Run選択の喪失になる |
| F-0005 | 2026-09-23 | open | P2 | defect | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.init / onToggleScreenshot | 初期化中にScreenshotへ切り替えるとinit完了時にmodeが上書きされ、init失敗時はScreenshotから抜けられない |

## open(優先度順)

| id | 優先度 | 確度 | 観点 | 分類 | 場所 | タイトル | 初出 | 最終確認 |
|---|---|---|---|---|---|---|---|---|
| F-0001 | P1 | high | refactor | 1関心1機構違反 | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.outlierPercentile | 排他的な外れ値表示モードを二つのSetと分散した排他処理で保持している | 2026-09-19 | 2026-09-23 |
| F-0003 | P1 | high | defect | 境界 | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::PlotlyController.renderBySelection | window内に点が無いtagはgraph blockごと消え、全tag一括のReset View以外では戻せない | 2026-09-23 | 2026-09-23 |
| F-0002 | P2 | high | defect | 境界 | `apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java`::LodPageCache.find | 未完成のLOD pageをbucketごとに全行読み直すため、LOD射影で読む行数が表示bucket数の2乗になる | 2026-09-23 | 2026-09-23 |
| F-0004 | P2 | high | defect | 並行 | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.onReload / onWorkspaceChanged | DataFetcherの単一のabort枠を独立した呼び出し元が奪い合い、abortが失敗Toastや初期Run選択の喪失になる | 2026-09-23 | 2026-09-23 |
| F-0005 | P2 | high | defect | 並行 | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.init / onToggleScreenshot | 初期化中にScreenshotへ切り替えるとinit完了時にmodeが上書きされ、init失敗時はScreenshotから抜けられない | 2026-09-23 | 2026-09-23 |

### 詳細

#### F-0001 P1 排他的な外れ値表示モードを二つのSetと分散した排他処理で保持している

- 観点/分類: refactor / 1関心1機構違反
- 場所: `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.outlierPercentile
- 要約: 同じtagの外れ値表示はoff・p5–p95・p1–p99の択一だが、ignoreOutlierTagsとp1P99Tagsの二つの集合で表現している。排他条件をtoggle、復元、描画値の選択へ分散させている。現状の補正は存在し、表示バグを指摘するものではない。 元の優先度P2を、対象範囲内で最多のchurn90=10のファイルとして1段上げた。
- シナリオ: 新しいpercentile表示を追加すると、集合と保存キーに加えて既存toggle双方、_loadStateの競合解消、outlierPercentileの優先順を同時に変更する必要がある。現状でも二つの保存集合へ同じtagが入った状態を_loadStateが補正している。
- 対処: 外れ値表示の状態遷移を一箇所へ集約し、内部ではtagごとの単一モードとして扱う。既存保存形式と競合時p1–p99優先の挙動を保つ入出力境界を明確にする。OutlierRangePlaywrightTestの排他・永続復元・手動Y範囲を特性テストとして確認し、描画経路の性能が同等であることを確認する。新しい保存契約へ変更する場合は別途設計判断を行う。
- 根拠: apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2001 二つのtoggleが相手のSetを消す; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2023 二つのSetから優先順でpercentileを復元; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2423 永続状態の復元にも排他違反の補正が必要

#### F-0003 P1 window内に点が無いtagはgraph blockごと消え、全tag一括のReset View以外では戻せない

- 観点/分類: defect / 境界
- 場所: `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::PlotlyController.renderBySelection
- 要約: serverは要求範囲内に点が無い系列をempty/pendingで返す。clientはok以外の系列を描かず、全Runがok以外のtagではgraph block自体を出さない。explicit viewportは残り、空windowが範囲を覆うので再取得も起きないため、そのgraphは操作手段ごと消えたままになる。元の優先度P2を、対象範囲内で最多のchurn90=10のファイルとして1段上げた。
- シナリオ: LOD表示中のtagで、点の無い区間（長いepisodeによる間隔など）の中へ、その幅の1/3未満でbox zoomする。または末尾付近を拡大して右へ2〜3画面panし、window[from-w,to+w]が最終点を越える → 全Runがempty → そのtagのgraphが消える。double clickする対象も無く、tagの再選択やReloadでも同じviewportで再びemptyになる。floatingのReset Viewで全tagのzoomを捨てるか、ページを再読込するしか戻す手段がない。
- 対処: explicit viewportを持つtagは、traceが0本でも空の軸だけのgraph blockを描く。またはserverが範囲外の直近点を両端に1点ずつ含める
- 根拠: apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepository.java:241 ordinal範囲は閉区間[fromStep,toStep]内の点だけで、範囲外の隣接点は含めない; apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryPlanner.java:86 範囲内が0点ならempty（取り込み中はpending）; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2082 empty/pendingの結果もwindowとして保存する; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:951 availabilityがok以外の系列はtraceを作らない; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:954 traceが0本のtagは、headerとLog/percentileボタンを含むgraph block自体を作らない; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:412 空windowが要求範囲を覆うので、以後needsFetchはfalseのまま; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1976 tag単位のautorange復帰はgraph上のrelayout経由だけで、graphが無いと発火しない; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2284 graph無しでviewportをautorangeへ戻す経路は、全tag一括のonResetViewだけ

#### F-0002 P2 未完成のLOD pageをbucketごとに全行読み直すため、LOD射影で読む行数が表示bucket数の2乗になる

- 観点/分類: defect / 境界
- 場所: `apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java`::LodPageCache.find
- 要約: LodPageCache.findは未完成page（各tagの末尾page）を保持せず、呼ばれるたびにpage全体を読み直す。一方でMetricsRangeProjector.loadBucketsは表示範囲のbucketごとにfindを呼ぶ。このため末尾pageに入るB個のbucketについて、B回×最大B行を読み、12本の配列もB回確保することになり、query時間がO(B^2)になる。結果は正しいが、cacheを無効にした場合（capacity=0、bucket単位SELECT）より遅い。
- シナリオ: 10万点のtagを全域表示する（maxPoints=8000）→ selectLevelはlevel 2の391 bucketを選ぶ → 永続化済みの390 bucketはすべて未完成のpage 0にあり、391回のfindがそれぞれ390行を読む。合計約15万行で、生の点数10万より多い。10 Run×20 tagを同時表示すると1 requestで約3,000万行になり、permit（既定2）を長く占有して他のタブが503 query_busyになる。Auto Reloadのforce再取得のたびに同じことが起きる。
- 対処: 未完成pageをquery単位でmemoするか、loadBucketsで範囲を一括SELECTしてからbucketを引く
- 根拠: apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java:50 cache missのたびに、1 bucketのためにpage全体（最大1024行）をSELECTする; apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java:51 未完成pageは保持せず捨てるので、同じpageの次のbucketで再びloadPageが走る; apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java:252 1024 bucket未満のpageは常に未完成扱い。つまり各tag・各levelの末尾pageは一度もcacheされない; apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRangeProjector.java:170 loadBucketsは選択したlevelの全bucketについて1件ずつpageCache.findを呼ぶ

#### F-0004 P2 DataFetcherの単一のabort枠を独立した呼び出し元が奪い合い、abortが失敗Toastや初期Run選択の喪失になる

- 観点/分類: defect / 並行
- 場所: `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.onReload / onWorkspaceChanged
- 要約: runs.jsonとmetrics.jsonのAbortControllerはDataFetcherにそれぞれ1枠しかなく、Reload・Auto Reload・workspace切替・viewport変更・Run/tagのclickが互いの要求をabortする。abortされたonReload/onWorkspaceChangedは、abortやsupersededを失敗として通知してしまう。さらにworkspace切替のinitial=true要求がAuto Reloadのinitial=false要求にabortされると、初期Run選択が行われない。元の優先度P3を、対象範囲内で最多のchurn90=10のファイルとして1段上げた。
- シナリオ: (a) Reloadを押した後、metrics.jsonの応答前にRunやtagをclickする → _bumpQueryRevisionがReloadの要求をabort → 『Reload failed.』が出る。workspace切替後の描画待ちにRunをclickした場合も『Workspace switched, but data refresh failed.』が出る。(b) Auto Reload ONでworkspaceを切り替え、切替側のruns.jsonの応答待ちにtickが来る → 切替側のrefreshMetadata（initial=true）がabortされてToastが出る。tick側（initial=false）が新workspaceを反映するが、選択Runは空のまま『No selection.』になる。
- 対処: AbortError/supersededを失敗Toastの対象から外す。workspace切替の後処理中はAuto Reloadのmetadata取得を止めるか、initial要求を後発の要求へ引き継ぐ
- 根拠: apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:243 fetchRunsは呼び出し元を問わず直前のruns.jsonをabortする; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:280 fetchMetricsも直前のmetrics.jsonをabortする。_bumpQueryRevisionも同じ枠をabortする; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2087 requestVisibleDataはabortも含めて例外を再送出する; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2441 onReloadはAbortErrorや409 supersededでも無条件に『Reload failed.』を出す; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1812 onWorkspaceChangedも無条件に『Workspace switched, but data refresh failed.』を出す; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2519 _handleQueryErrorとinit（1702行）はAbortError/supersededを除外しており、扱いが一致しない; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1861 初期Run選択はinitial=trueのrefreshMetadataだけが行う; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2414 Auto Reloadのtickはinitial=falseでrefreshMetadataを呼ぶ

#### F-0005 P2 初期化中にScreenshotへ切り替えるとinit完了時にmodeが上書きされ、init失敗時はScreenshotから抜けられない

- 観点/分類: defect / 並行
- 場所: `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.init / onToggleScreenshot
- 要約: META_LOADING中にScreenshotへ切り替えるとbodyはscreenshot-modeになるが、init完了時にmodeが無条件にNORMALかERRORへ上書きされ、画面とmodeが食い違う。scroll lockと取り込み進捗pollの停止はmode==SCREENSHOTで判定するため、画面はscreenshotのまま通常時の挙動になる。元の優先度P3を、対象範囲内で最多のchurn90=10のファイルとして1段上げた。
- シナリオ: (a) runs.jsonが遅い起動中に×/⬅を押す → init完了でmode=NORMALだが画面はscreenshotのまま → scroll lockが外れてgraphのdragがzoom/panになり、止まるはずの進捗pollも続く。(b) 同じ操作の後にinitが失敗する → mode=ERROR、side panelは非表示、graphも無い → Reloadボタンもgraphのdouble click Reloadも使えず、onToggleScreenshotはERRORで何もせず戻る。F5以外で復帰できない。
- 対処: META_LOADING中はscreenshot切替を無効にする。またはinit完了時にscreenshot状態を保ったままmodeを決め、ERROR中も解除だけは許可する
- 根拠: apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1695 initはMETA_LOADINGにしてからworkspaceとmetadataをawaitする; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2446 onToggleScreenshotはERROR以外（META_LOADINGを含む）で切替を許し、ERROR中は解除も拒否する; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1699 init成功時は現在のmodeを見ずにNORMALへ上書きする; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1704 init失敗時はERRORへ上書きし、screenshot-mode classは残る; apps/metrics-viewer/src/main/resources/static/metrics-viewer.css:645 screenshot中はReloadを含むside panelが非表示


## クローズ済み・保留

無し。
