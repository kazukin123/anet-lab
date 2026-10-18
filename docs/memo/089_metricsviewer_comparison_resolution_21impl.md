# MetricsViewerスクロール時の概観表示

[PRD089](089_metricsviewer_comparison_resolution_10prd.md)の追加要求を実装する。保存factor4/schema2と正確なStep区間集約は維持する。

## 方針

- 全選択タグについて、全記録範囲を約64共通Step区間へ集約した概観を取得する。画面幅・ズームに応じた詳細データは選択対象を64MiB以内のLRUで保持し、描画のみ近傍へ制限する。画面外も近い順に残容量内で詳細を先読みする。
- 概観は軽量SVGで表示し、詳細待ちと画面外でも線を残す。概観と詳細を同じグラフへ混在させず、全Runでそろった結果だけ表示する。近傍の詳細が完成したら既存Plotlyへ更新する。
- 概観は複数グラフを小さいbatchで取り、詳細を待つ前に表示する。概観を取得済みの通常Reloadやzoomで不要な照会を増やさない。世代変更、Run/タグ選択変更で不要な概観を捨てる。
- 凡例の非表示、色、Log、Mean/Band、軸範囲を描画に反映する。Screenshotでは詳細を取得する従来動作を維持する。

## 検証

実APIの4Run×64タグテストに、遠いタグも概観があること、詳細応答を保留したスクロール中も4Runの線が見えること、応答後の詳細化、詳細cacheの容量制限、余裕内で全64タグを先読みしスクロール往復の再取得が0回であることを追加する。RED→GREEN後に既存ブラウザ回帰を実行し、稼働中Viewerを変更せず検証JARを更新する。C++/Python/cache形式を変更しない。

## 検証記録

2026-10-11、追加実装と検証を完了。コミット・稼働中Viewerの差し替えは行っていない。

- 現行実装では遠いタグの概観が存在せず、新規テストの4本の線のassertが0本で失敗した（RED）。実装後に実APIの比較・スクロールテスト2件が成功（GREEN）。
- JDK25 / Maven3.9.15、POMの一時コピーを使い出力先を`target/prd089-validation`に固定。`-Dtest=*PlaywrightTest test`で全94件がfailure/error/skipped=0。zoom/pan、Run選択、凡例、Reload、Mean/Band、signed Log、percentile、Scroll Lock、Screenshot、workspace切替を含む。Java service/infraとPythonは変更しておらず、前回の検証結果を引き継ぐ。
- 実JSONL/APIの4Run×64タグで全256系列の概観と詳細を先読みできた。画面外の詳細も保持され、スクロール往復のmetrics再取得は0回。画面幅変更で詳細の再取得が必要な場合も、応答を保留している間に4Runの概観が表示され、応答後に4Runの詳細へ置き換わった。同一graph内で粒度を混在させない。
- このfixtureはtrainがRunごと8,192点、他63タグが各16点。decoded詳細と系列管理費用の合計は571,564 bytes（約0.55MiB）。CDPの`Runtime.getHeapUsage`はusedSize=23,456,428、totalSize=46,235,648、embedderHeapUsedSize=13,395,632、backingStorageSize=4,630,171 bytes。この小さいfixtureの観測であり、実DropMergeのメモリ余裕・応答時間を測った結果ではない。
- 通常の詳細cacheは64MiBのLRU。背景先読みは推定追加量が残容量へ収まるときだけ実行する。画面近傍の表示要求は優先し、容量超過時に遠いwindowから解放する。概観配列・SVG・近傍Plotly・server page cacheは別であり、64MiBをブラウザ全体のhard limitとは扱わない。Screenshotは全選択graphを描く明示操作なので近傍/LRU制限の対象外。
- `-DskipTests package`が成功。検証JARのJSはソースとbyte一致し、LICENSE/NOTICE/THIRD-PARTYを同梱する。一時POMは削除済み。既存の`target/metrics-viewer.jar`、実Runキャッシュ、8082は変更していない。

成果物: [更新した検証JAR](../../apps/metrics-viewer/target/prd089-validation/metrics-viewer-prd089.jar)、[ブラウザ回帰ログ](../../apps/metrics-viewer/target/prd089-validation/preview-regression.log)、[packageログ](../../apps/metrics-viewer/target/prd089-validation/preview-package.log)。JARは35,508,583 bytes、SHA-256 `087630c427dcba12ef5ab0f152f991213902a181d1962a20dc5a16807d3e9751`。


## 後続改善

2026-10-14、手書きSVGを共通Plotly描画からのexportへ置き換えた。現行の描画設計・検証結果と隔離JARのhashは[追加実装メモ22](089_metricsviewer_comparison_resolution_22impl.md)を参照する。上記2026-10-11の検証記録は当時の結果として保持する。
