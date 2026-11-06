# PRD090 キャッシュ保守の検証

## 1. 対象と隔離

未選択workspaceの旧cache再構築、process全体の3:1配分、gzip resource保持、buffer/batch境界、transaction rollback、Run単位の失敗停止、ログ、query cancel、shutdownを検証する。既存のJava全テスト・Playwrightも回帰対象とする。C++は変更対象外。

`verify_cache_rebuild.py` は通常JARと実Run cacheを変更せず、`apps/metrics-viewer/target/prd090-validation` と `target/prd090-performance` に生成物を置く。試行DBは一つずつ作成し、入力handleを閉じてから検証領域内だけで削除する。性能測定を同時に走らせない。

## 2. 自動回帰

repo rootから実行する。Java/Mavenのパスはスクリプト既定値を使うか `--java` / `--maven` で実環境の絶対パスを指定する。

```bash
./.venv/Scripts/python.exe core/anet-core/testdata/prd090/verify_cache_rebuild.py validate
```

一時POMで `package` を一回実行し、関連テスト、全Maven test、Playwright、隔離JAR生成を検証する。結果は `target/prd090-performance/all-tests-package-final.log` に残す。

## 3. 性能比較

変更前の作業ツリーは、PRD089を含む未コミット変更を保持した状態で `target/prd090-baseline` にコピーしてビルドした。Git HEADとの差分比較ではなく、PRD090実装直前の実効コードをbaselineとする。再現時は同じように変更前snapshotを用意し、同じbenchmark mainをコンパイルする。

停止済みDropMergeマスタを `target/prd090-performance/input/metrics.jsonl.gz` へ複製し、同内容rawを生成する。gzipは534,402,924bytes、rawは4,699,829,874bytes。factor4、JDK25.0.3、`-Xms512m -Xmx2g`、各raw/gzip 3回、試行ごとに新しいDBを使う。入力は複製へhardlinkして再利用し、実Run cacheへは接続しない。変更後コードは `new-project` へ凍結コピーしてビルドする。

```bash
./.venv/Scripts/python.exe core/anet-core/testdata/prd090/verify_cache_rebuild.py new
```

計測時間はDB初期化から最終commit・input解放まで。検証用の全rowハッシュ計算時間は含めない。`tags`、`scalars`、`scalars_lod`、`tag_stats`、`json_lines` をPK順に全row比較し、世代/sourceパス/圧縮offset等の取り込みmetadataを除いた内容の一致を確認する。47946167 scalar点、207 tag、15981218 LOD行、19非scalar行。

heap peakは各heap poolのpeak used bytesの合計で、同時点peakの上限である。DB容量は最終DBファイルサイズ。書込用SQLiteの16MiB page cache目標はnative memoryであり、heap peakへは含めない。min/max、母標準偏差、各試行値も保存する。cold OS cacheは強制せず、同条件の逐次試行として比較する。

## 4. 選択Runの反映待ち

```bash
./.venv/Scripts/python.exe core/anet-core/testdata/prd090/verify_cache_rebuild.py priority
```

大きい背景gzip 3Runと選択raw 1Runを別workspaceに置く。背景stream確保後、選択rawへ20回逐次追記し、公開閲覧APIのTagStatsへ新しい確定点が現れるまで100ms周期で測る。追記ごとのwakeや優先再要求は行わない。入力resource確保済みの選択Runが受入母集団であり、rawはstream枠待ち0ms。gzip枠待ちは別のresource上限テストとINFO waiting/resumedログで確認する。Plotly描画までのend-to-end遅延ではない。

```bash
./.venv/Scripts/python.exe core/anet-core/testdata/prd090/verify_cache_rebuild.py report
```

12試行の全tableハッシュ一致、raw/gzip中央値短縮、反映待ちp95<=10秒を検証し、`target/prd090-performance/results.json` に集計する。結果の正本は本ディレクトリの `performance-results.json` に保存する。

## 5. 今回の結果

2026-10-12 21:57:05に最終ソースで全237 testが成功（failure/error/skipいずれも0）。Playwrightを含む隔離Maven packageは3分28秒で完了した。性能受入条件もすべて満たした。

`CacheMaintenanceTest` は128Run検査、同名Run分離、stream上限、無変化のterminal、workspace集計、実SQLite triggerによる失敗停止・再処理、cycle途中の選択Run追記検出、変換中gzipからrawへの変更時の旧stream解放を検証する。`WorkspaceManagerTest` はbrowserなしの旧版再構築、workspace往復、古いepoch、一覧失敗隔離、残りqueueの継続、query leaseとshutdownを検証する。`MetricsIngestorIntegrationTest` はUTF-8/CRLF/buffer/batch境界、長い非scalar、rollback整合、fake clockの進捗公開を検証する。

behaviorごとのRED/GREENログは `target/prd090-validation/*-red.log` / `*-green.log` に保存した。

### 5.1 再構築時間と資源

| 入力 | 変更前中央値（秒） | 変更後中央値（秒） | 短縮率 | 変更前標準偏差（秒） | 変更後標準偏差（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 1033.863 | 688.271 | 33.4% | 60.252 | 3.710 |
| gzip | 1704.032 | 760.164 | 55.4% | 25.277 | 14.418 |

| 試行群 | 3回の時間（秒） | heap peak範囲（MiB） |
| --- | --- | ---: |
| baseline-raw | 1145.587, 1006.212, 1033.863 | 320.83–321.16 |
| baseline-gzip | 1704.032, 1668.549, 1730.233 | 321.28–321.64 |
| new-raw | 688.271, 694.645, 685.849 | 322.82–324.33 |
| new-gzip | 786.705, 760.164, 753.257 | 322.13–324.18 |

全12試行で最終DB容量は2,337,267,712bytes（約2.18GiB）、5tableの全row SHA-256が一致した。SQLite各列をbenchmarkが取得した値のJava表現で比較し、浮動小数点値にも許容誤差を設けていない。全tableの件数とhash、各試行の工程別時間・速度・block数は [performance-results.json](performance-results.json) に保存する。各試行のstream枠待ちは0msで、単独再構築のためgzip上限に達しない。

buffer/batch化だけの最初の候補はraw 1087.835秒、commit合計687.159秒だった。page spill抑制を狙った書込接続の16MiB cache調整後に、最終比較をraw/gzip各3回やり直した。初期候補は比較へ含めない。最終比較でもcommit合計はraw約465–475秒、gzip約522–562秒で主要費用として残る。1秒はread/parse/writeのsoft予算であり、commitを含むblock全体のhard期限ではない。

### 5.2 選択Runの反映待ちと測定コード

20sampleのp50は1,217.0325ms、p95は1,343.0929ms、最大1,353.1645ms。背景gzipの最大同時streamは3本、選択rawのstream枠待ちは0ms、heap peakは341,325,680bytes。背景3workspaceのgzipを維持しながら選択rawの確定点を公開した。resource確保済みrawへのAPI反映の結果であり、gzip枠待ちや画面描画までの遅延を含まない。枠飽和・解除は自動回帰で別途確認している。

最初の反映待ち測定はp95=11,302.7262msで不合格だった。優先Runの属性確認を各block境界へ移し、cycle途中でも追記を再発見する回帰をRED→GREENで追加した。gzipからrawへの変更時には旧streamを明示解放する回帰も追加した。最終237testと上記20sampleはこれらの修正を含むソースで実行した。

最後の修正対象はIngestSchedulerと回帰testだけで、単独MetricsIngestorを呼ぶ再構築ベンチの実行コードは変更していない。測定時source/class manifestを保存し、scheduler以外のmain 103 classがbyte単位で一致することを集計時に検証した。入力SHA-256、測定時・最終source manifestのSHA-256、JVM versionもJSONへ保存する。生log・凍結snapshot・manifestは `target/prd090-performance` と `target/prd090-baseline` に保持した。
