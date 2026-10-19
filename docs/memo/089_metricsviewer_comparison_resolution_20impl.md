# MetricsViewer比較粒度の実装メモ

## 概要

[PRD 089](089_metricsviewer_comparison_resolution_10prd.md)を正本として実装する。保存factor=4、schema=2。表示Step区間は保存LOD階層と分離し、区間起点は0。ユーザー判断のブロッカーは0で、ユーザーは必要判断後の実装・テストまで実行を依頼済み。

## 主な変更

1. `MetricsSeriesRequest`に必須のsafe integer `stepBucketWidth` (>0) を加える。`maxPoints`は応答頂点の上限であり粒度を変更する配分値にはしない。幅1でraw全点が指定上限に収まるときはraw件数、それ以外は区間数×3またはraw件数の小さい方を必要量として上限を検証し、request全体の制限も超過時422で拒否する。グラフの各Runは同じrange/widthを要求する。
2. `MetricsRangeProjector`は対象序数範囲をfactor4の完成bucketで分解する。bucket全体が1つのStep区間へ入れば加重合成し、跨ぐ部分だけ下位LOD、最後にL0へ降りる。L0 SQLはprepared statementを再利用しキャンセル登録を維持する。Step幅1かつraw全点が予算内ならraw、それ以外は共通Step区間のLODを返す。summary.stepsは区間の左端、min/maxStepsは実Step。空区間は生成しない。
3. responseの`level`/序数`bucketWidth`を削除し`stepBucketWidth`を公開する。旧ordinal射影選択処理を残さない。
4. 表示区間数の目標はグラフ幅pxを256..2000へ収めた値。Step幅はmax(1,ceil(viewport.width/目標))。Run数・タグ数は算出に使わない。左右1画面分をprefetchし、window境界は共通Step区間へ整列する。各系列予算はwindowの区間数×3。
5. 1グラフの全Runを1requestで取得する。全タグのrequestをまとめず順次処理し、revision/generation/全系列の応答一致を確認してから一括反映する。再利用条件にStep幅を含め、異なる幅のwindowを同時描画しない。
6. 全選択タグに高さ300pxのblockを置く。通常モードはscroll viewport前後1画面の近傍だけwindow/Plotlyを保持する。スクロール・resizeで再計画し、画面外のPlotlyをpurgeする。Screenshotモードは明示された全選択グラフを順次取得する。既存軸範囲・凡例・Scroll Lockを保持する。
7. `LodBucket` factor、Java/Python cache schemaとfixturesを更新する。既存のschema不一致による自動全破棄を利用する。CONTEXTには表示Step区間だけ定義し、ADR 0052に判断理由、210設計書の日本語/英語に現行契約を記録する。

## テスト

Public surface: `POST /api/metrics.json`、`GET /api/runs.json`、ブラウザのRun選択・Plotly relayout・scroll、マスタ追記とcache再構築。

優先behaviorとTDD順序:

1. tracer bullet: 明示Step幅を要求したAPIが、極値の実Stepと共通区間左端を返す。1テストRED→最小射影実装GREEN。
2. 異なる頻度/長さ、重複Step、端の部分区間を持つ4Runの区間集約が直接計算と一致し、4Runで粒度不変。RED→GREEN。
3. 上限超過を粒度低下で解決せず422、不正幅は400。RED→GREEN。
4. 実ブラウザで多数タグ・選択順序・zoom/pan/reset/resizeの再現。共通Step幅、一括応答反映、supersede、近傍取得を確認。RED→GREEN。
5. factor4の追記/再起動、旧schemaから再構築、gzip、既存Viewer操作の回帰を更新して実行。

## 検証

MetricsViewerはJava/JSだけを変更するためMavenのcompile/test/packageとPlaywrightを使う。C++のDebugビルドは対象外。JDK25のJAVA_HOMEと既存Maven3.9.15を使用し、全テストと必要なPython inspect_runテストを実行する。検証用Runはtarget内に作り、稼働中8082や実Runキャッシュを変更しない。PRD固有READMEは`core/anet-core/testdata/prd089/README.md`に置き設計書から参照する。

## 前提と範囲外

既存UIはMinMax/Mean/Bandを維持する。cache短命接続、snapshot、TagStats、generation、キャンセル登録を維持する。全Run常時raw、新ストア、migration、commit/pushは範囲外。Screenshotで全選択を描く場合は通常モードの近傍保持制限から外れる。区間幅は記録頻度によらず同じだが、観測点が存在しない区間を補間しない。

## 検証記録

2026-10-10、実装と検証を完了。コミット前なのでPRDは`docs/memo/`に置いたままとする。

- JDK `25.0.3.9` / Maven `3.9.15`、Java release=17。通常POMの一時コピーで出力先だけを`target/prd089-validation`、finalNameを`metrics-viewer-prd089`へ変更し、`mvn -B -f pom.prd089-validation.xml test package`で全209件を実行した。初回に残った新規テストの頂点数仮定と旧UIテストのHTTP待機・回数仮定を修正し、`-Dtest=ComparisonResolutionPlaywrightTest,MetricsPlotPlaywrightTest test`、最後に`-Dtest=MetricsPlotPlaywrightTest test`で該当クラスを再実行した。最終surefire集計は30クラス・209件、failure/error/skipped=0。そのうちPlaywrightは93件。
- `ComparisonResolutionPlaywrightTest`は実JSONL、実API、4Run×64タグを使う。固定Step範囲0..131072、同じ画面幅の1→2→3→4RunでStep幅146が維持された。密なRunは898区間・1,861頂点。Run追加前後で既存Runの頂点数も不変。異なる記録頻度・重複Step、zoom/pan、solo raw拡大後の4Run再選択、Reset View、resize、画面外の解放・再取得が成功した。
- `MetricsLodIntegrationTest`の5件で区間左端、極値の実Step、部分端、保存上位より広い区間、異なる頻度/長さの1〜4Runのmin/max/加重平均とマスタ直接計算一致、幅1の重複Step全点raw保持を確認した。
- `MetricsIngestorIntegrationTest`23件、`MetricsCacheDatabaseIntegrationTest`9件が成功。factor4階層、追記・再起動・gzip、schema1→2の全破棄とgeneration更新、マスタ無変更を確認した。422の固定粒度上限、不正幅400、queryキャンセル・snapshot・workspace切替も回帰に含む。
- `C:\dev\anet-lab\.venv\Scripts\python.exe -B viewers\metrics-tools\inspect_run_test.py`は72件成功。Pythonのcache eligibilityはschema2へ更新した。
- テスト通過後に同じ一時POMで`mvn -B -f pom.prd089-validation.xml -DskipTests package`を実行し、Spring Boot JARの作成が成功した。一時POMは削除済み。JARのJSと主要classは検証済みbuild出力とbyte一致し、LICENSE/NOTICE/THIRD-PARTYを同梱する。

成果物: [検証用JAR](../../apps/metrics-viewer/target/prd089-validation/metrics-viewer-prd089.jar)、[全件実行ログ](../../apps/metrics-viewer/target/prd089-validation/full-validation.log)、[比較・描画再検証ログ](../../apps/metrics-viewer/target/prd089-validation/retest.log)、[最終描画再検証ログ](../../apps/metrics-viewer/target/prd089-validation/final-retest.log)、[packageログ](../../apps/metrics-viewer/target/prd089-validation/package.log)。JARは35,506,179 bytes、SHA-256 `5761cccdb2ba26093bbe597fed34c9aebbfeeb5c2a026f508460856ef0f572f6`。

稼働中8082、通常`target/metrics-viewer.jar`、実Runキャッシュは差し替えていない。新binary利用時は初回Loadでキャッシュ再構築を待つ。実DropMergeの約200k点/系列でのJava応答時間と再構築時間は未計測で、今回の検証は比較粒度・集約の正確性・既存操作の回帰を保証する。Screenshotは全選択を描くため通常の近傍保持制限から外れる。

2026-10-11、スクロール時の概観・容量内詳細保持・先読みを追加し、検証JARを更新した。現行成果物と最新検証は[追加実装メモ](089_metricsviewer_comparison_resolution_21impl.md)を参照する。上記SHA-256は2026-10-10の初回ビルドの記録である。
