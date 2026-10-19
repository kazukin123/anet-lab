# PRD089 MetricsViewer比較粒度の検証

[PRD](../../../../docs/memo/089_metricsviewer_comparison_resolution_10prd.md)と[実装計画](../../../../docs/memo/089_metricsviewer_comparison_resolution_20impl.md)を参照する。

検証コードは `apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/` に置く。fixtureは `apps/metrics-viewer/target/` 以下だけに生成し、実Run・8082の稼働中Viewerを変更しない。

- `service/MetricsLodIntegrationTest`: マスタを直接集約した期待値と1〜4RunのStep区間、重複Step、部分端、極値・平均を比較。
- `view/ComparisonResolutionPlaywrightTest`: 実APIの4Run・64タグ、粒度維持、zoom/pan/reset、scrollでの詳細保持と先読み、詳細応答保留中の概観表示、容量内で先読み済みのスクロール往復の再取得ゼロ。
- `infra/MetricsCacheDatabaseIntegrationTest`: schema1から2への全破棄、generation変更、マスタ無変更。
- `service/MetricsIngestorIntegrationTest`: factor4の全階層、再起動後の未完成末尾、gzip/JSONL同値。
- 既存Playwrightとservice tests: MinMax/Mean/Band、hover、Log、percentile、凡例、Run色、Reload、Scroll Lock、workspace、supersede。

PowerShellからテストを実行（既存JDK/Mavenを使用。稼働中JARを差し替えない）:

```bash
$env:JAVA_HOME='C:\Program Files\Eclipse Adoptium\jdk-25.0.3.9-hotspot'; Set-Location C:\dev\anet-lab\apps\metrics-viewer; & 'C:\dev\apache-maven-3.9.15\bin\mvn.cmd' -B test
```

Python側の現用cache eligibility:

```bash
Set-Location C:\dev\anet-lab; & .\.venv\Scripts\python.exe -B viewers\metrics-tools\inspect_run_test.py
```

表示粒度の比較では同じStep viewport・同じ画面幅を固定する。autorangeのまま長いRunを加えると表示範囲自体が広がるため、Step幅が変わるのは正常である。Screenshotモードは全選択グラフの描画を明示する操作であり、通常モードの詳細cache容量制限の対象外。

今回の検証ビルドはPOMの一時コピーでbuild.directoryを`target/prd089-validation`、finalNameを`metrics-viewer-prd089`へ変え、同じソース・依存・resourcesで実施した。通常の`target/metrics-viewer.jar`は変更せず、一時POMは検証後に削除する。結果と実行条件は実装メモの検証記録を参照する。

## 概観/詳細の描画検証

[22impl](../../../../docs/memo/089_metricsviewer_comparison_resolution_22impl.md)を参照。`ComparisonResolutionPlaywrightTest`は非ゼロEMA・スパイク・異なる4Run期間、初期ロード/Run追加/scroll中の詳細応答保留、Log/Mean/Band、外枠/ヘッダー保持、遅着SVG、再取得0回を確認する。PNGは`apps/metrics-viewer/target/prd089-validation/`へ出す。

`render_overview_benchmark.cjs`は4Run×64graph、192vertices/Run、各方式5回でPlotly 2.27の実装を比較する。baselineは変更前のJSを`target/prd089-validation/baseline-metrics-viewer.js`へ保存したものを使う（今回の変更では更新前の隔離JARから復元）。生成時間、初回表示の追加遅延、GC後のJS/Blink保持メモリをJSONへ記録する。実Run規模の応答・プロセス全体のメモリ保証ではない。

```bash
Set-Location C:\dev\anet-lab; & 'C:\Users\kazuk\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' core/anet-core/testdata/prd089/render_overview_benchmark.cjs
```
