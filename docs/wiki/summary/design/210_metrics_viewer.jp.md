# Metrics Viewer

## 元の材料

[Metrics Viewer](../../../design/210_metrics_viewer.jp.md) blob:f468bce7519636a35fb4b858171f50e94916e1d9

## 要点

- Run の Metrics マスタを読み、再構築できる SQLite キャッシュに取り込んでブラウザへ表示範囲の系列を返す。
- step と系列内の序数を区別する。step の閉区間を序数の半開区間に変換して取得する。
- 複数の解像度と点数上限を使い、広い範囲の俯瞰と狭い範囲の詳細を同じ画面で切り替える。
- サーバの取り込み状態とキャッシュ世代、ブラウザが保持する範囲を照合し、古い応答で新しい表示を上書きしない。
- API は系列の取得、設定、タグセット、範囲統計などを分担し、workspace と Run の境界を保つ。

## 内容

### 2. Metrics Viewerの概要と基本概念

- Runnerは学習中、`train/loss`のようなtagごとの数値を、1行1点のJSON行としてfileへ追記し続ける。
- フォルダを入れる・出す・リネームするというファイル操作だけが、可視化対象（Run作業セット）の登録・解除・改名の手段であり、表示はこの作業セットに限定し、キャッシュ保守は設定済みdirectory内の全workspaceへ及ぶ。設定範囲外のRunは追跡しない。
- `metrics_cache.db`はマスタから従属構築される破棄可能なMetricsキャッシュであり、第2のマスタにはしない。
- そこで、点を捨てるのではなく区間ごとに畳み込む。連続する4点を1つの単位にまとめ、その中の最小値・最大値・最終値・件数・平均を保存しておく。
- バケットの幅は「連続する16点」のように序数で数え、「step 1000ごと」のようにstep幅では数えない。
- 1バケットから取り出す描画候補は`min`、`max`、`last`の3点である。表示区間ごとの描画頂点は最大3点であり、同じ序数の候補は1点だけ出す。
- タグ全体の統計 `TagStats` は、表示の解像度や空区間に左右されないよう、取り込み済みの全有効点から計算する。
- 件数上限に加え1秒の時間予算を持ち、優先変更・workspace切替・shutdown要求でも次の完全行境界で早期commitする。列挙、初期化、commit、単独の長い行には厳密な時間上限を保証せず、工程別時間を記録する。

### 3. コンポーネント定義

- `SourceReader` — 改行終端済みの完全行だけをblock単位で読み出す抽象。`RawFileReader`と`GzipSessionReader`を持つ。
- 連続値は`input type="range"`で、ほかは`button`で作り、見分けはラベルの言葉づかいで付ける。値の一覧から1つを選ぶ`Workspace`とLODは`select`のままで、この分類に含めない。
- 前提の無い即時実行は disabled にする。タグセットの `Save`・`Revert` は選択中のセットと現在の選択に差があるときだけ押せ、文字と枠を `--control-border` より弱く表示する。
- スクリーンショットモードは表示部品を隠す。初期化中も切り替えられ、データ更新は続く。初期化失敗時は復帰に使う Reload を隠さないよう、モードへの移行を止める。
- `scrollElement()`が現在のスクロール主体を返し、再描画とモード切替は前後で「先頭に見えているgraph blockのtagKeyと、そのblockが表示領域の上端からはみ出している量」を覚えて復元する。

### 5. 静的構造

- `MetricsCacheDatabase`はwriteとreadで別connectionを返すが、Run directory単位のlifecycle read-write lockを共有する。
- 全破棄・再構築だけがwrite lockを取り、通常のtransaction同士はWALへ委ねる。

### 6. 主要フロー

- JSONL は最後に改行が確定した行まで取り込み、書きかけの末尾を次回に残す。gzip は展開後のストリームが未終端行のまま終わると `error` になる。
- scalar値の検証は行い、`null`・非数値・非finite・float32へ収まらない値はその行だけskipして`(run, tag, 理由)`ごとに1回WARNする。
- 同一tag内でstepが逆行した場合は、マスタ全体ではなくそのtagだけを`status='error'`へ隔離する。隔離前のL0、LOD、`TagStats`は公開し続け、隔離解除はsource変更による全再構築時だけ行う。
- Runごとのread connectionは、計画から射影完了まで1本を保持する。lifecycle read lockが全再構築を排他し、SQLite transactionにより取り込み側の通常commitをまたいでも同じsnapshotを参照する。
- 各rangeは前回応答へ依存しない完結した結果であり、clientは差分mergeを行わずwindowごと置換する。
- 明示viewportを持つtagで、取得した全Runのwindowに点が無いときは、軸だけのgraphを描く。graphが無いと、ダブルクリックやpanで戻す手段が無くなるためである。
- 描画はgraph単位で突き合わせる。`PlotlyController`はgraph blockをtagKeyで引き、外枠と共通headerを使い回す。データwindow、概観/詳細の種別、選択Run、幅、高さ、LOD表示モード、Log、percentile、軸範囲、Scroll Lock、凡例の表示状態・色を表示モデルの描画keyへ含める。同じkeyでは再描画せず、metadata由来のheaderの警告を更新する。

### 7. 設定一覧

- すべて`application.properties`または起動引数（`--key=value`）で与える。数値設定は`MetricsViewerSettings`、workspace path/nameは`WorkspaceManager`、タグセットのパス・ファイルは`TagSetStore`のconstructorで検証し、契約違反はapplication起動を中止する。
- `cache-memory-mb`の上限判定は`Runtime.maxMemory()`に依存する。起動scriptの`-Xmx`を下げるとこの設定だけで起動に失敗しうる。
- リクエスト本文に未知のフィールドを含むと `invalid_request` を返す。
- 22_metrics_viewer_java.batは作業ディレクトリをJARの置き場に揃え、portを固定する。Optuna用の別Viewer launcherは持たず、同じworkspace selectorを使う。
- Run色は`MetricsViewerClientApp`が所有し、`refreshLists()`の先頭で解決する。まず色を持たないRunへrunId昇順で`RUN_COLORS`から先着順に配る（基本色）。
- `Auto Recolor`がONなら、前回の解決時に選択されていなかったRun（初回表示で自動選択する最新Runを含む）を新しく選んだRunとし、選択順に、ほかの選択中Runの色から最も離れた色を配る。選択中だったRunの色は変えず、新しく選んだRunの基本色は見ない。OFFからONへ切り替えた時点では色を変えない。

### 8. Metricsキャッシュのデータベース定義

- 完成済みの bucket に限り、アプリケーション側の `LodPageCache` が LRU で保持する。
- 子4件がそろった時点でのみ親を書くため、末尾の未完成bucketは行として存在しない。同値が並ぶ場合は序数が小さい点を代表にする。
- `TagStats` はコミット済みの全 L0 データから範囲非依存の統計を計算し、同じトランザクションで更新する。

### 9. HTTP API

- `GET /api/workspaces.json`はcurrent workspaceと選択肢を返す。選択肢は`metricsviewer.workspaces-dir`直下で`runs/`または`config/`を持つdirectoryの名前昇順である。
- `percentage`は`state`から導く。`pending`は0、`ready`は100、それ以外は`committed_offset / source_size`を0〜99へclampする（`source_size`が0のときは`error`が0、それ以外は100）。
- `POST /api/runs/prioritize` は受け取った Run ID の集合で取り込みの優先対象を置換し、成功時に 204 を返す。
- `minMax`は各bucketの`min` / `max` / `last`を序数順に並べた折れ線用の点列で、同一序数が重なる場合は1点へ畳む。
- タグセット名は前後の空白を除いた1〜64コードポイントで、制御文字は使えない。タグは同じセット内で重複できない。

### 10. lifetime・並行制御・エラー境界

- gzip変換中のRunは展開済みstreamを`GzipInputSessions`がblock間で保持する。未選択workspaceの背景変換中にも、そのRun folderの移動をサポートしない。`ready`到達、失敗、登録対象からの消失、shutdownで解放する。
- 1点だけの異常（`null`、非数値、非finite、float32外）はその点をskipし、`(run, tag, 理由)`ごとに1回WARNする。
- 完全検証済みの`ready` / `error` Runはprocess memoryにsource/cache属性を保持し、属性不変のpollではfingerprint、Metricsマスタ本文、SQLite connectionへ入らない。観測はRun消失とprocess restartで失われ、表示snapshotの切替だけでは失われない。
- gzip streamは全体16本まで。新しい非優先Runは、非優先の開始済みstreamが4本未満の場合だけ開始する。選択Runは全体上限まで開始できる。選択を外れた開始済みstreamも継続する。sourceがgzipからrawへ変更された場合は、旧gzip streamを取り込み前に解放する。枠待ちはevictionせず、理由の状態遷移でINFOを出し、再開時に`waitMs`を出す。

### 11. ビルドと依存ライブラリ

- `org.webjars.npm:plotly.js-dist-min` — versionは2.35.3。scopeはcompile。用途はjar内から配信するPlotly。
- `org.springframework.boot:spring-boot-devtools` — versionは3.5.7。scopeはcompile（optional）。用途は開発時のみ。`spring-devtools.properties`でrestartは無効化済み。
- `jackson-databind`とloggingはstarterからも入るが、版を固定するため明示宣言している。
- `spring-boot-maven-plugin` — versionは3.5.7。設定は`repackage`実行可能JARを生成する。
- Java側のコード規約はcheckstyle.xmlで定義する。行頭indentはtab（tab幅4、継続行8）、行長上限120、`FinalLocalVariable`と標準の命名規則を要求する。Eclipse用の設定は`.checkstyle`とformatter定義に置く。
- frontendはnpm等のbuild工程を持たず、`src/main/resources/static`をそのまま配信する。
- Plotlyはjarに含むWebJarから配信するため、ブラウザが外部へ接続できない環境でもグラフを描ける。

### 12. テストと拡張時の確認事項

- タグセットは `TagSetStoreTest`・`TagSetApiIntegrationTest`・`TagSetPlaywrightTest` で検証する。梱包を含む手順は RFD0111検証README。
- Playwrightテストは既定でMicrosoft Edgeを起動する。Edgeが無い環境では`Assumptions`によりskipされ、失敗にはならない。
- テストごとにcontextを開き直し、route、`localStorage`、Plotly stateを共有しない。
- cache schemaを変えるときは`SCHEMA_VERSION`を上げる。migrationは書かず、旧cacheが警告なしで破棄・再構築されることをtestする。
- LODを触るときは、完成bucketだけ永続化する契約と、端の部分bucketを下位levelから再集約する経路の両方を検証する。`LodPageCache`は完成bucket数をtag点数から導くので、親を書く契機を変えるときはこの導出も合わせて変える。
- TagStatsはcommit済みL0全点から、範囲統計は指定step範囲のcommit済みL0全点から作る。設定TXTのフレームワークの行規則を変えたら、Viewerの `ConfigTextReader` と検証も同じ変更で更新する。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
