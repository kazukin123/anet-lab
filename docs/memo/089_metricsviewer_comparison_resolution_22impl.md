# 概観と詳細の描画共通化 実装メモ

## 概要

[PRD089](089_metricsviewer_comparison_resolution_10prd.md)の追加改善として、ユーザーが承認した「概観と詳細の描画を共通化する」計画を実装する。手書きSVGを廃止し、既存Plotly 2.27による描画から概観SVGと詳細を作る。ユーザー判断のブロッカーは0。保存factor4/schema2、公開API、PRD090の未コミット高速化とステージ状態を保持する。

## 主な変更

- グラフごとのデータ選択と表示モデル、共通の外枠・ヘッダー、概観/詳細の描画を分離する。モデルには選択Run、window世代、traces、幅、軸範囲、表示設定をまとめる。既存Plotlyレイアウトを両表示で共有し、SVG独自の軸・座標・Band描画を削除する。
- 概観SVGは1つの作業用Plotlyで直列生成する。タグごと最新1件を保持し、描画キーの変更・選択解除・workspace切替で更新/破棄する。生成結果は現在モデルへ一致する場合だけ反映する。
- 生成は画面の近い順に処理し、グラフごとにイベントループへ制御を返す。通信や詳細描画でSVG生成をawaitしない。全選択グラフのPlotlyを常駐させない。
- 共通外枠と操作ヘッダーを保持し、内側の描画完成時だけ内容を置き換える。更新中は直前の表示を維持する。概観は静止画、詳細のhover/zoom/pan/凡例操作は従来どおりとし、共通ヘッダーのLog/percentileと凡例表示状態を両表示へ反映する。
- 64MiBの詳細LRU、近い順の先読み、同一graph内の全Runで共通Step粒度・一括反映を維持する。

## テスト

Public surfaceは実JSONL/API、ブラウザのRun選択・scroll・resize・Log/percentile・凡例と描画。最初に詳細応答を保留した概観の軸/凡例/ヘッダーをRED→GREENで確認する。次に初期ロード・Run追加・スクロールの切替で、非ゼロEMA、スパイク、期間の異なる4Run、MinMax/Mean/BandとLogを確認する。外枠/スクロール位置/操作状態の保持、遅着描画の排除、workspace切替、先読み後のmetrics再取得0回を順に追加・検証する。

## 検証

変更前のsame-dataハーネスはPlotly 2.27、4Run×64graph（192 vertices/Run）で生成合計約0.85秒、手書きSVG比の保持メモリ増分約16MiB。受け入れ目安は同条件で生成合計1.5秒以内、最初の概観表示の追加遅延p95が100ms以内、GC後の保持メモリ増分32MiB以内とする。単純なfixtureでの観測であり実Run規模の応答保証ではない。

JDK25/Maven3.9.15、POMの一時コピーでbuild.directoryを`target/prd089-validation`へ変更して、対象Playwright→全Java test→隔離JARを検証する。画像と性能ログも同ディレクトリへ保存する。通常JAR、8082、実Run cache、PRD090性能試験資材、git indexは変更しない。新しいライブラリ・汎用ジョブ基盤は追加しない。

## 検証記録

2026-10-14、実装・検証を完了。コミット・稼働中Viewerの差し替えは行っていない。

- 概観の軸・凡例と外枠保持を先にテストし、手書きSVG／block再作成で失敗するREDを確認した。共通モデル・Plotly export・完成時の内容差し替えでGREENとなった。
- JDK25 / Maven3.9.15で全Javaテスト239件（ブラウザ96件）がfailure/error/skipped=0。初期ロード、1→4Run追加、初回スクロールで詳細応答を保留し、非ゼロEMA・スパイク・期間差・MinMax/Mean/Band・Log・percentileを確認した。外枠/スクロール位置/操作状態の保持、先読み後のmetrics再取得0回、容量制限、workspace切替、最新Run選択への遅着画像の混入防止を含む。[全テストログ](../../apps/metrics-viewer/target/prd089-validation/render-full.log)、[ビルドログ](../../apps/metrics-viewer/target/prd089-validation/render-package.log)。
- 実JSONL/APIの比較fixtureは4Run×64タグで、train各8,192点、EMA各16点、残り62タグ各16点。画像を保存し、概観と詳細の軸・凡例・余白・共通ヘッダーの配置を確認した：[初期概観](../../apps/metrics-viewer/target/prd089-validation/initial-overview.png)、[4Run概観](../../apps/metrics-viewer/target/prd089-validation/multi-run-overview.png)、[Log概観](../../apps/metrics-viewer/target/prd089-validation/multi-run-overview-log.png)、[Log詳細](../../apps/metrics-viewer/target/prd089-validation/multi-run-detail.png)。
- 独立した同一入力の描画ベンチマークで、4Run×64graph・192 vertices/Run・Plotly 2.27.0を変更前/後それぞれ5回測定した。SVG生成合計p95は935.5ms（中央値905.3ms）、最初の表示の追加遅延p95は36.0ms、GC後の保持メモリ増分p95は18,945,016 bytes（18.07MiB）。制御を返す待ち時間と画像decodeを含む全64件完了は最大1,332ms。作業完了後に保持するPlotly instanceは0。全受け入れ目安を満たした。[測定値とsource SHA256](../../apps/metrics-viewer/target/prd089-validation/render-performance.json)、[再実行手順](../../core/anet-core/testdata/prd089/README.md)。メモリはCDPのusedSize＋embedderHeapUsedSizeの変更前後差分であり、ブラウザ全processのメモリではない。単純な描画fixtureでの値であり、実Run全体の初期ロード時間を保証するものではない。
- [隔離JAR](../../apps/metrics-viewer/target/prd089-validation/metrics-viewer-prd089.jar)を作成した。35,539,539 bytes、SHA256 `8e71388c239bcefb4767facda5f4bd38b3aaeab6964564a6247156b1def99198`。同梱JSは検証対象の作業treeと一致し、LICENSE/NOTICE/THIRD-PARTYを確認した。
- 実装前のhash記録と比較して、Java production/service test、PRD090文書の変更は0件。git indexも実装前と同一。保存factor/schema/API/マスタ、通常JAR、8082は変更していない。既存の64MiB詳細LRUと近傍先読みを維持した。

