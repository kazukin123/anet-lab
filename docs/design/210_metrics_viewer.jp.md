# Metrics Viewer

> 主たる観点: 具象application単位（取り込み、cache DB、range query、browser描画を工程順に併記）

## 1. はじめに

### 1.1 目的

本書は、Java/Spring Boot製のMetrics Viewerについて、Metricsマスタからの取り込み、Run-local SQLite cacheの構造、range queryの解決、browser側の描画までを説明する。
Runner processとの境界、cacheをどこまで信頼してよいか、どの層に責務を追加すべきかを明確にする。

### 1.2 対象読者

- Metrics Viewerのserver・frontendを変更する開発者
- 取り込みscheduling、LOD、点予算、query同時実行を調整する開発者
- cache schema、HTTP API、process間contractの互換性をレビューする担当者

### 1.3 記載範囲

現行の`apps/metrics-viewer`を扱う。Runner側のMetricsマスタ生成、Event、metric登録の内部は[可観測性](140_observability.jp.md)を参照する。
Runner GUI、Optuna harnessとのprocess境界は[アプリケーションとツール](160_applications_and_tools.jp.md)を正本とする。
利用手順は[Run分析ガイド](030_user_guide_analysis.jp.md)、build手順は[開発環境](040_development_environment.jp.md)を参照する。

2章は前提知識なしで読めるように、解きたい問題からデータの流れ、用語の定義までを順に説明する。
3章以降は2章の語彙を使って実装を説明するため、先に2章を通して読むことを勧める。

## 2. Metrics Viewerの概要と基本概念

### 2.1 解きたい問題

Runnerは学習中、`train/loss`のような**tag**ごとの数値を、1行1点のJSON行としてfileへ追記し続ける。
横軸に使う**step**（学習の進行度）と値の組が延々と並ぶ形である。長いRunでは1 tagが数百万点、Run全体で数千万点に達する。

一方、browserのgraphは横幅がせいぜい1,000〜2,000ピクセルしかない。
数百万点をそのまま送っても、1ピクセルへ数千点が重なって潰れるだけで、転送量とmemoryだけを消費する。
かといって「100点に1点だけ残す」ような素朴な間引きをすると、学習の異常検知でいちばん見たい単発のスパイクが消えてしまう。

Metrics Viewerは次の3つを同時に満たすことを目標とする。

1. 数千万点のRunでも、開いた瞬間に全体像が出る。
2. zoomすれば、その範囲だけ細かい形が出る。十分に狭めれば最後は生の点そのものへ到達する。
3. 学習が走っている最中のRunでも、追記済みの範囲を止めずに見られる。

このために、「マスタfileを毎回読み直さずに済む形へ一度変換しておき、表示に必要な粒度だけを切り出して送る」構造を採る。
次節でその流れ全体を示す。

### 2.2 データフロー全体像

データは次の6段を一方向に流れる。上流ほど正確で重く、下流ほど粗くて軽い。

```mermaid
flowchart TD
  A["① Metricsマスタ<br/>metrics.jsonl / metrics.jsonl.gz<br/>1行1点のJSON、追記のみ"]
  B["② Metricsキャッシュ<br/>metrics_cache.db（Runフォルダ内のSQLite）<br/>全点 + 集約バケット + tag統計"]
  C["③ LOD page cache<br/>server processのheap<br/>集約バケットの塊"]
  D["④ range応答<br/>HTTP JSON、数値はBase64 binary<br/>1系列あたり数千点"]
  E["⑤ browser DataCache<br/>tabのmemory<br/>系列ごとに詳細windowと概観"]
  F["⑥ Plotly描画"]

  A -->|LoadingThreadが1 blockずつ取り込む| B
  B -->|request時に読み出す| C
  C -->|集約済みバケット| D
  B -->|生点と端の再集約はDBを直接読む| D
  D -->|fetch| E
  E -->|選択中のtagだけ| F
```

データを保持する①〜⑤の性質は次のとおりである（⑥は描画そのもので、独自のデータ保持を持たない）。

| 段 | 置き場所 | データの単位 | 作る主体 | 消えるとき |
|---|---|---|---|---|
| ① Metricsマスタ | Runフォルダのfile | 1行1点 | Runner process | 手動で消すまで残る。**唯一の正** |
| ② Metricsキャッシュ | Runフォルダのfile | tag内の序数 | Viewerの取り込みthread | 不整合検出時に全破棄。①から再構築できる |
| ③ LOD page cache | server processのheap | 1024バケットのpage | range query | 容量超過、Run消失、世代変更 |
| ④ range応答 | 通信路 | viewport 3画面分 | range query | 応答ごとに使い切り |
| ⑤ browser DataCache | tabのmemory | 系列ごとの詳細windowと軽量概観 | client app | 容量超過、選択解除、世代変更、tab再読込 |

②〜⑤はすべて①から再生成できる。どこを消しても失われるのは速度だけで、情報は失われない。
この一方向性が全体の設計原則であり、下流から上流へ書き戻す経路は存在しない。

各段を詳しく扱う章は次のとおりである。

| 段 | 詳細を扱う章 |
|---|---|
| ①→② 取り込み | 6.1（フロー）、8.2〜8.3（格納先と再構築条件） |
| ② データ構造 | 8.2（table定義） |
| ②→④ range query | 6.2（フロー）、9.2（点予算とavailability） |
| ③ LOD page cache | 10.1（lifetime）、10.4（性能） |
| ④ 通信形式 | 9.4（binary符号化） |
| ⑤⑥ client | 6.3（フロー）、7.4（定数と永続state） |

なお、Metrics ViewerはRunner processへ直接接続しない。`runs`ディレクトリのRunフォルダだけが入力である。
フォルダを入れる・出す・リネームするというファイル操作だけが、可視化対象（**Run作業セット**）の登録・解除・改名の手段であり、表示はこの作業セットに限定し、キャッシュ保守は設定済みdirectory内の全workspaceへ及ぶ。設定範囲外のRunは追跡しない。

process内のthreadは次の2系統だけで、両者はSQLite fileを介してのみ結合する。

| 系統 | thread | 役割 |
|---|---|---|
| 取り込み | 単一の`Metrics-LoadingThread` | ①を1 blockずつ読み、②へ書く唯一のwriter |
| 応答 | Tomcat request thread | ②へ短命なread connectionを開き、③〜④を組み立てる |

HTTP requestは①を一切読まない。取り込みthreadがcommit済みのsnapshotだけを見る。
この分離により、巨大なマスタを読んでいる最中でもHTTP応答が止まらない。

### 2.3 「キャッシュ」と呼ぶものが3つある

上表のとおり、性質の違う3つを「キャッシュ」と呼ぶ場面がある。寿命と破棄条件が異なるので、本書では常に区別して呼ぶ。

| 呼称 | 実体 | 何を省くためか | 寿命 | 容量制御 |
|---|---|---|---|---|
| Metricsキャッシュ | Run-localのSQLite file | マスタの再parseと、範囲検索のための全走査 | Runフォルダと同じ。processをまたいで残る | なし（マスタに比例） |
| LOD page cache | server heapのLRU | SQLiteからのバケット読み出し | process内 | `cache-memory-mb` |
| browser DataCache | client JavaScriptのMap | HTTPの往復 | 選択中、容量内 | 詳細は64MiBのLRU、概観は約64区間/系列 |

4つ目として、SQLite自身が持つconnection内のpage cacheがあるが、**これには依存しない**。
Metrics Viewerはconnectionを読み＝request単位、書き＝取り込みblock単位の短命に保つため、connection内のcacheは毎回捨てられる。
常時接続にしないのは、Windowsではopen中のfileを移動・削除できず、「Runフォルダの出し入れ＝可視化対象の登録・解除」という運用契約が壊れるためである。
その代わりをapplication層のLOD page cacheが務める。

### 2.4 マスタとキャッシュの従属関係

`metrics.jsonl`（workspace metrics圧縮ツールで移行した後は`metrics.jsonl.gz`）が**Metricsマスタ**である。
`metrics_cache.db`はマスタから従属構築される破棄可能な**Metricsキャッシュ**であり、第2のマスタにはしない。

この区別が効いてくるのは、schemaを変えたときとfileが壊れたときである。
キャッシュは「いつ削除してもマスタから同一内容を再構築できる」ことを前提にするので、schema様式やsourceの同一性に不整合があれば、migrationを書かずに全破棄・再構築で解決してよい。
逆にキャッシュ側にしか無い情報を持たせると、この前提が崩れて破棄できなくなる。詳細は[ADR 0015](../adr/0015-metrics-cache-disposable-derivative.md)を正本とする。

同一Runフォルダに`metrics.jsonl`と`metrics.jsonl.gz`が両方ある場合は`metrics.jsonl`を選び、Runごと1回だけWARNする。

### 2.5 tag、step、序数

3つの語を区別する。

- **tag**: 系列の名前。`train/loss`のような文字列で、1 tag = graph 1本分の時系列である。
- **step**: 横軸の座標値。学習の進行度を表す整数で、Runnerが各点へ付ける。
- **序数**（ordinal）: 1つのtag内での記録の出現順。0始まりの通し番号で、Viewerが取り込み時に振る。

点のidentityと順序は**序数**が持ち、stepは座標値として列に置くだけである。stepを主キーにしない理由は、stepが一意にならないためである。
同一tagのstepは非減少だが、同一stepへ複数episodeの値が正当に載る。実測では1つのepisode系tagに同一stepの点が240,109件あった。
ここで`(tag, step)`をUNIQUEにすると、正当なデータをREPLACEで失う。

この選択の帰結として、step範囲での検索は「二分探索でstep境界を序数へ写像し、以降は序数区間として扱う」形になる。
step用の補助indexは持たない。`[fromStep, toStep]`は両端を含む閉区間で、対応する序数区間は`[ordinalFrom, ordinalTo)`の半開区間になる。

### 2.6 LOD ― 粗い絵をあらかじめ作っておく

**LOD**（Level of Detail、詳細度）は、3Dグラフィクス由来の考え方で、「遠くのものは粗いモデルで描く」ことを指す。
ここでは「広い範囲を見るときは、あらかじめ集約しておいた粗い系列を描く」という意味で使う。

#### なぜ間引きではだめか

素朴な案は「16点に1点だけ残す」である。しかしこれは、残らなかった15点にスパイクがあると消してしまう。
学習metricsでは、lossが一瞬跳ねた・報酬が一度だけ落ちたという単発の外れ値こそ見たい情報なので、この方式は採れない。

そこで、点を捨てるのではなく**区間ごとに畳み込む**。連続する4点を1つの単位にまとめ、その中の最小値・最大値・最終値・件数・平均を保存しておく。
最小値と最大値が残るので、区間内にスパイクがあれば必ず絵に出る。最終値が残るので、隣の単位との繋がりも保たれる。

この集約単位を**LODバケット**と呼ぶ。

#### バケットの幅は序数で数える

バケットの幅は「連続する16点」のように**序数**で数え、「step 1000ごと」のようにstep幅では数えない。
tagごとに記録頻度が違うため、step幅で切ると、密なtagでは1バケットに数万点が入り、疎なtagでは空バケットが大量に並ぶ。
序数で切れば、どのtagでも1バケットの中身が常に一定件数になる。

#### 階層にする

zoomの倍率に応じて必要な粗さが変わるので、バケットを階層にする。
4点ずつまとめたものを**level 1**、level 1のバケットを4個まとめたものを**level 2**、というように、幅は`4^level`になる。
集約前の生の点そのものは**level 0**、略して**L0**と呼ぶ。以降、生の点の層とそれを格納するtableを「L0」と書く。

| level | 1バケットが覆う点数 |
|---:|---:|
| 0（L0） | 1（生の点） |
| 1 | 4 |
| 2 | 16 |
| 3 | 64 |
| 4 | 256 |
| 5 | 1,024 |

保存階層は表示区間へ集約するための再利用部品である。描画粒度はStep区間幅で決まり、保存levelを表示へ直接割り当てない。

階層は追記だけで作れる。子が4個そろった瞬間に親を1個書き、その親をさらに上のlevelの子として渡す。
したがって未完成のバケットはDBに存在せず、点が増えるたびに作り直す必要もない。

#### 1バケットから描画に使う点は3つ

1バケットから取り出す描画候補は`min`、`max`、`last`の3点である。表示区間ごとの描画頂点は最大3点であり、同じ序数の候補は1点だけ出す。
Step区間幅が1で、range内の生点数が予算以下なら、生の点をそのまま返す。これが2.1の目標2にある「最後は生の点そのものへ到達する」の実装である。

#### 統計はLODから作らない

tagごとの平均・分散・最小最大などの正確な統計は、**TagStats**として別に保持する。
これはcommit済みの有効な全点に対する範囲非依存の統計であり、LODからは導出しない。
LODは表示解像度のための近似なので、そこから統計を復元すると、表示Step幅や空区間の有無で結果が変わってしまうためである。

### 2.7 共通Step区間と点数上限

保存LODは序数区間、表示は**共通Step区間**である。同じgraphの全Runへ、Step=0を起点とする同じ半開区間 `[k*w,(k+1)*w)` を適用する。幅`w`は`max(1,ceil(viewport幅 / 目標区間数))`で、目標区間数はgraph幅pxを256〜2000に収めた値。Run数と選択tag数は算出に使わない。

各表示区間へ完全に収まる保存bucketはそのまま加重合成し、跨ぐbucketだけ下位LOD、最後にL0へ降りる。平均は点数で加重し、min/maxの実Step、重複Step、末尾点を保持する。点が無い区間を生成しない。幅1で全点が上限内ならraw、その他は表示区間のmin/max/lastとsummaryを返す。

`maxPoints`と`max-points-per-request`は上限であり、系列間の予算配分で粒度を変更しない。raw幅1で全点が指定上限内なら必要量はraw件数で、重複Stepも全点を保持する。それ以外の必要頂点数上界は`min(range内の生点数, Step区間数*3)`。上限を超えたら422を返す。clientは1graphの全Runを1requestにまとめ、graphごとに順次取得する。

### 2.8 viewportとwindow

**viewport**は、いま画面に映っているstep範囲である。**window**は、そのviewportに対してclientがserverへ要求する取得範囲である。

clientはviewportそのものではなく、左右へ1画面ずつ広げた**3画面分**をwindowとして要求する。
こうすると、少しpanしただけでは再取得が起きず、graphが空白になる時間を減らせる。

windowは系列ごとに1件だけ保持し、新しい応答が来たら差分mergeせず丸ごと置き換える。
各range応答はそれ単体で完結しており、前回の応答へ依存しないため、この単純な置換で足りる。

1つのgraphに載る系列（同じtagの選択中Run）は同じrequestで取る。表示粒度はviewportと画面幅から先に決め、prefetchで広げても変えない。window境界は共通Step区間へ整列し、パン時に区間起点を動かさない。通常モードのPlotly描画はscroll viewport前後1画面の近傍に限る。詳細windowは選択対象を64MiBのLRU内で保持し、容量内で画面に近い順に先読みする。スクロールだけでwindowを捨てない。全選択tagに全記録範囲の約64共通Step区間の概観を8graph単位で先に取得し、画面外と詳細待ちの間は、詳細と共通のPlotly figureから生成した軽量SVGで線を残す。概観と詳細を同じgraphで混在させない。blockの高さ300px、headerとscroll位置を維持し、描画が完成するまで直前の内容を残す。Screenshotモードは全選択graphを順次取得する。
どれか1系列でも取り直しが要れば、そのgraphの系列をまとめて取り直す。概観は同じ共通Step幅の全Runを一括反映し、詳細がそろうまでは概観を表示する。64MiBはdecoded詳細配列と系列ごとの管理費用の上限であり、SVG、Plotly、server heapを含むプロセス全体のメモリ上限ではない。

概観SVGは手書きで座標・軸・Bandを計算せず、詳細と共通のtraces/layoutをPlotly 2.27で描いてexportする。共通headerのLog/percentileと凡例表示状態を反映するが、概観自身は静止画であり、hover/zoom/pan/凡例クリックは詳細表示で扱う。生成は画面近くから1件ずつ進め、イベントループへ制御を返す。通信や詳細描画はSVG生成をawaitしない。タグごと最新1画像を保持し、データ・選択・幅・表示設定が変われば生成し直す。workspace切替では世代を進めて古い生成結果を破棄する。

概観も詳細も完成後に内側だけを公開する。更新中は直前の描画を残し、選択Run、cache世代、表示設定、幅が要求時と一致しない遅着結果は公開しない。scrollによる描画更新はrequestAnimationFrameで行い、通信のdebounceとは分離する。

### 2.9 Metricsキャッシュ世代

同じ名前のRunフォルダでも、中身のマスタが差し替われば別物である。これを識別するのが**Metricsキャッシュ世代**である。

全再構築のたびに新しいUUIDを発行し、通常の追記では維持する。
HTTP応答とbrowser DataCacheはこの世代を突き合わせ、古い世代の応答を新しい世代の絵へ混ぜない。

`PRAGMA user_version`と混同しないこと。`user_version`はschemaの様式版であり、世代はキャッシュ内容の同一性である。

### 2.10 取り込みの進行state

`source_meta.state`はキャッシュの取り込みがどこまで進んだかを表す。文字列値は永続値かつHTTP公開値であり、互換性なく変更しない。

| state | 意味 | 次のcycleの扱い |
|---|---|---|
| `pending` | キャッシュを作った直後で1 blockも確定していない | 続けて読む |
| `converting` | 1 block以上をcommit済みで、まだ末尾に達していない | 続けて読む |
| `ready` | 取得時点のsource末尾（gzipはstream EOF）まで確定した | rawは追記があれば再開、gzipは読まない |
| `error` | source側の致命的な不正を検出した | sourceのsize/mtimeが変わるまで読まない |

ここでいう**block**は、1回のtransactionで取り込む行のまとまりである。
巨大なマスタを1回のtransactionで読み切ると、その間ずっとRunが表示できないため、最大1,000,000行ずつに区切って途中経過をcommitする。
件数上限に加え1秒の時間予算を持ち、優先変更・workspace切替・shutdown要求でも次の完全行境界で早期commitする。列挙、初期化、commit、単独の長い行には厳密な時間上限を保証せず、工程別時間を記録する。
`converting`はこの「途中まで見えている」状態を表し、browserは進捗率つきで表示する。

## 3. コンポーネント定義

### 3.1 server

| コンポーネント | 定義 |
|---|---|
| `MetricsViewerApplication` | Spring Boot application entry |
| `WorkspaceManager` | current workspaceをepoch付きsnapshotとしてatomicに保持し、query lease、切替gate、表示resourceのclose-on-zeroとprocess schedulerのshutdownを管理する |
| `RunScanner` | snapshotの`<workspace>/runs`直下からMetricsマスタを持つRunフォルダを列挙し、Run idをRun directoryへ解決する |
| `MetricsSource` | 選択したマスタfileのkind、size、mtime、先頭・commit直前のSHA-256 fingerprintを表すvalue |
| `MetricsCacheDatabase` | cache fileの検証、破棄・再構築、`source_meta`、read/write connectionのlifecycleを管理する |
| `SourceReader` | 改行終端済みの完全行だけをblock単位で読み出す抽象。`RawFileReader`と`GzipSessionReader`を持つ |
| `GzipInputSessions` | process単位でgzip streamと64KiBの未消費展開bufferを保持し、完了・失敗・消失・shutdownで解放する |
| `MetricsIngestor` | 1 blockのJSONL parse、L0書込み、LOD追記、`TagStats`、source位置を同一transactionで確定する |
| `LodIngestWriter` / `LodBucket` | 子4件がそろうたびに親bucketを合成して`scalars_lod`へ書く追記専用writerとbucket値 |
| `IngestScheduler` | 全workspaceを軽量検査し、処理可能なRunへpriority 3 : background 1で配分するprocess scheduler。背景はworkspace間・各Run間で巡回し、失敗Runの再試行を属性変化まで停止する |
| `LoadingThread` | `WorkspaceManager.runIngestCycle()`を回す単一writer thread。即時処理可能なbacklogもworkspace切替も残らないときだけidle sleepする |
| `MetricsQueryCoordinator` | process-globalなfair permit、query channelごとの最新sequence、live ticket、workspace epoch、terminal shutdownを一体管理し、supersede時にcheckpointと実行中SQLを停止する |
| `MetricsRepository` | Runごとに1本のread snapshotを開き、Run metadataとseries queryを解決する |
| `MetricsQueryPlanner` | 系列ごとのavailability判定と、固定Step幅の必要頂点数上界を検証する |
| `MetricsRangeProjector` | raw射影とLOD射影を組み立て、部分bucketだけ下位levelから再集約する |
| `LodPageCache` | 完成済みbucketだけを1024件単位のpageとしてheapへ持つLRU cache。末尾pageは、snapshotのtag点数から求めた完成bucket数に足りなくなったときだけ読み直す |
| `MetricsService` | LoadingThreadのlifecycle、metrics body/header検証、coordinator実行、snapshot lease取得、HTTP error変換を担う |
| `MetricsViewerController` | Run、metrics、priorityのREST APIを公開する |
| `WorkspaceController` | workspace一覧・切替APIを公開し、同Controllerの不正JSONだけを400 `invalid_request`へ変換する |
| `MetricTraceEncoder` | double/float配列をlittle-endian Base64 chunk列へ符号化する |
| `RunWarningRegistry` | 正規化したRun絶対パスごとに、登録workspace内に存在する間、世代をまたいで同じWARNを抑止する |
| `HttpAccessLogFilter` | 全requestの開始・終了・所要時間をINFOで記録する |

### 3.2 browser

| コンポーネント | 定義 |
|---|---|
| `MetricsViewerClientApp` | Run/tag選択、Run色、viewport、描画世代（revision）、poll timerを所有するclient app |
| `DataFetcher` | REST呼出し、ページ単位のquery channelとsequence、AbortControllerによる旧request打ち切りを担当する |
| `DataCache` | Run metadataと、`(runId, tagKey)`ごとの詳細windowと概観を別々に保持する。windowには取得時のtag点数・最終step・statusを添え、Reload時の鮮度判定に使う |
| `OverviewSvgRenderer` | 共通figureを1つの作業用Plotlyで直列exportし、タグごとの最新SVGだけを保持する。現在モデル・世代に一致する結果だけを公開する |
| `PlotlyController` | raw/MinMax/Mean/Band描画、signed-log軸、zoom/pan、scroll lock、凡例状態を扱う。graphごとのデータ選択・共通表示モデル・外枠を管理し、描画keyが変わったgraphの内側だけを完成時に差し替える |
| `UIController` | Run list、Tag list、進捗表示、静的controlのbindを担当する |
| `Toast` | CSSの`.toast`表示規則を使って一時的なerror通知を表示する |

### 3.3 UIコントロールの規約

browserのcontrolは、押した瞬間に効果が出るもの、on/offが持続するもの、決まった段を順に巡るものの3種類しかない。
どれも`button`で作り、見分けはラベルの言葉づかいで付ける。

| 種別 | ラベル | 状態 |
|---|---|---|
| 即時実行 | 動詞で始める。`Reload`、`Select All`、`Select Latest`、`Recolor`、`Clear All`、`Reset View` | 持たない |
| トグル | 動詞で始めない名詞句。`Auto Reload`、`Auto Recolor`、`Selected Only`、`Scroll Lock`、`Log` | `.active`と`aria-pressed`を対で更新する |
| 巡回 | 現在の段をそのまま書く。graphの下限`p0–` / `p1–` / `p5–`と上限`–p100` / `–p99` / `–p95` | 押すたびに固定順で次の段へ進み、最後の段の次は先頭へ戻る。先頭の段以外で`.active`と`aria-pressed`をONにする |

sectionの見出し行には、ラベルに続けてそのリストへの操作を左詰めで並べる。
入りきらない分だけを次の行へ送る（Runsの`Recolor` / `Auto Recolor`）。
サイドパネルのbuttonは、見出し行・ボタン行・globalのどこにあっても同じ寸法にする。
絞り込み中に一括操作（`Select All`など）を押したときは、ボタンを無効化せず、絞り込みを解除してから実行する。

on/offはラベル文字へ書かず、押下色だけで示す。状態の反映は`setToggleState()`へ集約し、
見た目の`.active`と意味の`aria-pressed`が食い違わないようにする。checkboxは使わない。
巡回は段が3つ以上あって押下色だけでは区別できないので、ラベルに現在の段を書き、押下色は先頭の段から外れていることだけを示す。
段によってラベルの文字数が変わっても寸法と後続要素の位置が動かないよう、最長のラベルに合わせた固定幅にする。

controlの枠線は`--control-border`（hoverは`--control-border-hover`）の1段で、
`.section`などのcontainer枠`--container-border`より弱くする。
ON時の配色は`--active-background` / `--active-accent` / `--active-text`を使う。

### 3.4 スクリーンショットモード

スクリーンショットモードは貼り付け用の絵を作るために表示だけを畳むモードであり、データの更新は止めない。
side panelとfloating controlのうちgraph操作用のものを隠し、`#main-area`のoverflowをvisibleにしてスクロールをdocument側へ移し、Run名を載せたheaderを出す。
Auto Reloadも手動Reloadも通常と同じに効き、side panelが隠れてReloadボタンを押せない間はgraphのdouble clickがReloadの入口になる。
進捗を出すRun listが隠れている間は、取り込み進捗のpollだけを止める。

スクロール主体が`#main-area`とdocumentで入れ替わるため、位置はpixelのままでは引き継げない。
`scrollElement()`が現在のスクロール主体を返し、再描画とモード切替は前後で
「先頭に見えているgraph blockのtagKeyと、そのblockが表示領域の上端からはみ出している量」を覚えて復元する。
覚えたtagのgraphが消えていたときだけ、同じスクローラの続きとしてpixelで戻す。

## 4. コードマップ

| 領域 | 主なファイル |
|---|---|
| entry / 設定 | [MetricsViewerApplication.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/MetricsViewerApplication.java)、[MetricsViewerSettings.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/config/MetricsViewerSettings.java)、[application.properties](../../apps/metrics-viewer/src/main/resources/application.properties) |
| scan / source同一性 | [RunScanner.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/RunScanner.java)、[MetricsSource.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsSource.java) |
| cache DB | [MetricsCacheDatabase.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsCacheDatabase.java) |
| source読み出し | [SourceReader.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/SourceReader.java)、[RawFileReader.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/RawFileReader.java)、[GzipSessionReader.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/GzipSessionReader.java)、[GzipInputSessions.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/GzipInputSessions.java) |
| 取り込み | [MetricsIngestor.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java)、[LodIngestWriter.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodIngestWriter.java)、[LodBucket.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodBucket.java) |
| workspace / scheduling | [WorkspaceManager.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/WorkspaceManager.java)、[IngestScheduler.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/IngestScheduler.java)、[LoadingThread.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LoadingThread.java) |
| query | [MetricsQueryCoordinator.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryCoordinator.java)、[QueryCancelledException.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/QueryCancelledException.java)、[MetricsRepository.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepository.java)、[MetricsQueryPlanner.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryPlanner.java)、[MetricsRangeProjector.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRangeProjector.java)、[LodPageCache.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java) |
| API | [MetricsService.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsService.java)、[MetricsViewerController.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/MetricsViewerController.java)、[WorkspaceController.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/WorkspaceController.java)、[view/model](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/view/model)、[MetricTraceEncoder.java](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/util/MetricTraceEncoder.java) |
| browser UI | [index.html](../../apps/metrics-viewer/src/main/resources/static/index.html)、[metrics-viewer.js](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)、[metrics-viewer.css](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.css) |
| test | [src/test/java](../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer) |
| build | [pom.xml](../../apps/metrics-viewer/pom.xml)、[checkstyle.xml](../../apps/metrics-viewer/checkstyle.xml) |

## 5. 静的構造

```mermaid
classDiagram
direction LR

class LoadingThread
class WorkspaceManager
class IngestScheduler
class MetricsIngestor
class SourceReader
class RawFileReader
class GzipSessionReader
class GzipInputSessions
class LodIngestWriter
class MetricsCacheDatabase
class RunScanner
class MetricsService
class MetricsQueryCoordinator
class MetricsRepository
class MetricsQueryPlanner
class MetricsRangeProjector
class LodPageCache
class MetricsViewerController
class WorkspaceController
class MetricsViewerSettings

LoadingThread --> WorkspaceManager
WorkspaceManager *-- IngestScheduler
WorkspaceManager *-- RunScanner
WorkspaceManager *-- MetricsRepository
WorkspaceManager *-- LodPageCache
IngestScheduler *-- GzipInputSessions
IngestScheduler --> MetricsIngestor
IngestScheduler --> MetricsSource
IngestScheduler --> GzipInputSessions
MetricsIngestor --> SourceReader
MetricsIngestor --> LodIngestWriter
MetricsIngestor --> MetricsCacheDatabase
SourceReader <|.. RawFileReader
SourceReader <|.. GzipSessionReader
GzipSessionReader --> GzipInputSessions

MetricsViewerController --> MetricsService
WorkspaceController --> MetricsService
MetricsService --> WorkspaceManager
MetricsService --> LoadingThread
MetricsService --> MetricsQueryCoordinator
WorkspaceManager --> MetricsQueryCoordinator
MetricsRepository --> MetricsCacheDatabase
MetricsRepository --> RunScanner
MetricsRepository *-- MetricsQueryPlanner
MetricsRepository --> MetricsRangeProjector
MetricsRepository ..> MetricsQueryCoordinator : execution token
MetricsRangeProjector --> LodPageCache
MetricsService --> MetricsViewerSettings
MetricsQueryPlanner --> MetricsViewerSettings
LodPageCache --> MetricsViewerSettings
```

`MetricsCacheDatabase`はwriteとreadで別connectionを返すが、Run directory単位のlifecycle read-write lockを共有する。
全破棄・再構築だけがwrite lockを取り、通常のtransaction同士はWALへ委ねる。

## 6. 主要フロー

### 6.1 block取り込み

```mermaid
sequenceDiagram
    participant L as LoadingThread
    participant W as WorkspaceManager
    participant S as IngestScheduler
    participant I as MetricsIngestor
    participant R as SourceReader
    participant D as metrics_cache.db

    L->>W: runIngestCycle()
    W->>W: snapshot lease取得
    loop 4 slot（priority 3 : background 1）
        W->>W: 切替gate取得、epoch/shutdown確認
        W->>S: runNextBlock()
        S->>S: 全workspaceの分割検査、処理可能Runを配分
        S->>I: ingestBlock(runId, runDir, source属性)
        I->>I: source/cache属性を検証済み観測と照合
        alt 属性一致かつ前回stateがready/error
            I-->>S: didWork=false, immediateRetry=false
        else
            I->>R: prepare(database)
            R->>D: source fingerprint照合
            alt 不一致
                D->>D: cache全破棄と新generationで再作成
            end
            alt readRequired = false
                I-->>S: didWork=false, immediateRetry=false
            else
                I->>D: BEGIN
                loop 最大1,000,000行 / 1秒 / 早期終了要求
                    I->>R: 64KiB bufferから完全行を抽出
                    I->>I: JSON parse
                    alt type = scalar
                        I->>D: scalars batch（最大4096）
                        I->>D: 子4件のscalars_lod batch（最大4096）
                    else 非scalar
                        I->>D: json_lines batch（4096件 / 1MiB）
                    end
                end
                I->>D: 残りbatch flush、tag_stats UPSERT
                I->>D: source_meta（offset・fingerprint・state）更新
                I->>D: COMMIT
                I-->>S: didWork、次state、immediateRetry
            end
        end
        S->>S: terminal/no-op Runをcycle内でexhausted化
        S-->>W: immediateRetry
        W->>W: 切替gate解放
    end
    W->>W: snapshot lease解放
    alt immediateRetry = false
        L->>L: 最大10秒wait（優先変更・切替・停止でwake）
    end
```

改行で終端していない末尾行はblock外へ公開せず、commit可能offsetも最後の完全行より先へ進めない。
これによりRunnerが書き込み途中の行を取り込まない。gzipではstream EOFに未終端行が残った場合だけcorruptとして`error`にする。

scalar値の検証は次の順序で行い、`null`・非数値・非finite・float32へ収まらない値はその行だけskipして`(run, tag, 理由)`ごとに1回WARNする。
同一tag内でstepが逆行した場合は、マスタ全体ではなくそのtagだけを`status='error'`へ隔離する。隔離前のL0、LOD、`TagStats`は公開し続け、隔離解除はsource変更による全再構築時だけ行う。

### 6.2 range query

```mermaid
sequenceDiagram
    participant B as Browser
    participant C as MetricsViewerController
    participant S as MetricsService
    participant Q as MetricsQueryCoordinator
    participant W as WorkspaceManager
    participant R as MetricsRepository
    participant P as MetricsQueryPlanner
    participant J as MetricsRangeProjector
    participant D as metrics_cache.db

    B->>C: POST /api/metrics.json（series配列 + channel/sequence header）
    C->>S: getMetrics(request, channel, sequence)
    S->>S: body/header検証
    S->>Q: run(channel, sequence, work)
    Q->>Q: 同一channelの旧ticketをcancelし、fair permitを最大5秒待つ
    Q->>S: work(execution token)を呼び出す
    S->>W: workspace lease取得
    W-->>S: lease(epoch)
    S->>Q: ticketをworkspace epochへ束縛
    Q-->>S: epoch束縛完了
    S->>R: query(series, execution token)
    loop Runごと
        R->>D: openRead + setAutoCommit(false)
        R->>D: source_meta読み取り
        R->>D: tagとtag_stats読み取り
        R->>D: stepの二分探索でordinal範囲へ写像（tagのstep範囲外の端は点数で決まる）
    end
    R->>P: plan(inputs)
    P-->>R: availability、固定Step幅の必要頂点数上界
    loop ok系列ごと
        R->>J: project(ordinal範囲, Step区間幅, 点数上限)
        alt Step幅=1、raw全点が指定上限内
            J->>D: scalarsをordinal順に読む
        else
            J->>J: Step=0起点の共通Step区間へ射影
            J->>D: 区間内の完成bucketを再利用、跨ぐ部分だけ下位LOD/L0を読む
        end
        J-->>R: raw または lod 射影
    end
    R-->>S: 系列結果
    S-->>Q: lease解放後に系列結果
    Q-->>S: permit返却後に系列結果
    S-->>C: GetMetricsResponse
    C-->>B: Base64 chunk列を含むJSON
```

Runごとのread connectionは、計画から射影完了まで1本を保持する。lifecycle read lockが全再構築を排他し、SQLite transactionにより取り込み側の通常commitをまたいでも同じsnapshotを参照する。

### 6.3 client viewportとwindow置換

```mermaid
sequenceDiagram
    participant U as User
    participant A as MetricsViewerClientApp
    participant F as DataFetcher
    participant K as DataCache
    participant P as PlotlyController

    A->>F: GET /api/runs.json
    F-->>A: Run/tag metadata
    A->>A: 新規tagを自動active化、最新Runを初期選択
    A->>F: POST /api/runs/prioritize（選択Run）
    A->>A: tagごとのviewportを決定（明示範囲またはstep全域）
    A->>K: needsFetch判定
    A->>F: POST /api/metrics.json（viewport左右1画面を含む3画面window）
    F-->>A: 系列結果
    A->>K: 全系列のgeneration・revision・Step幅を確認してgraphごと一括置換
    A->>P: raw/MinMax/Mean/Bandで再描画

    U->>P: zoom / pan
    P->>A: onViewportChanged(tagKey, range)
    A->>A: revisionを進めて進行中requestをabort、150ms debounce
    A->>F: 新しい3画面windowを要求

    U->>A: Reload
    A->>F: GET /api/runs.json
    F-->>A: Run/tag metadata
    A->>K: isOutdated判定（取得時のtag点数・最終step・statusと比べる）
    A->>F: 古くなったgraphの系列だけPOST /api/metrics.json（無ければ送らない）
    A->>P: 描画keyが変わったgraphの内容を完成時に差し替える
```

各rangeは前回応答へ依存しない完結した結果であり、clientは差分mergeを行わずwindowごと置換する。

手動ReloadとAuto Reloadは、metadataを取り直したあと、取得後に中身が古くなったwindowを含むgraphだけを取り直す。何も変わっていなければ`metrics.json`は送らない。
windowには取得を決めた時点のtag点数・最終step・statusを添えておき、次のどれかに当たれば古いとみなす。

| 条件 | 理由 |
|---|---|
| windowが無い、世代が違う、表示viewportを覆っていない、要求Step幅が違う | zoom/panや選択変更と同じ判定 |
| availabilityが`ok` / `empty`以外 | 取り込み中や照会失敗で値が確定していない |
| tagのstatusが変わった | 隔離などでissueが変わった |
| tagの点数が変わり、windowの右端が取得時の最終step以上 | 追記点がwindowへ入りうる |

tag内のstepは非減少なので、追記点のstepは取得時の最終step以上になる。過去の区間を拡大していてwindowの右端がそこへ届かないgraphは、点数が増えても取り直さない。

描画はgraph単位で突き合わせる。`PlotlyController`はgraph blockをtagKeyで引き、外枠と共通headerを使い回す。データwindow、概観/詳細の種別、選択Run、幅、LOD表示モード、Log、percentile、軸範囲、Scroll Lock、凡例の表示状態・色を表示モデルの描画keyへ含める。同じkeyでは再描画せず、metadata由来のheader（統計と警告）を更新する。
凡例の表示、軸範囲、drag modeは描画前に`capturePlotState`で読み戻す。描画keyが変わったときは新しいSVGのdecodeまたは詳細Plotlyの描画が完了してから内側だけを差し替え、直前の表示を保持する。遅着した描画は現在の選択・世代・表示設定と一致した場合だけ反映する。
Plotlyの再描画はgraph 1枚あたり数十msかかり（従来の実測で137 graph×5 Runの全再描画が約5秒）、対象graphを絞ることがReloadの応答時間を決める。

取り込み中Runがある間はRun metadataを4秒間隔でpollし、進捗表示だけを更新する。この進捗pollだけはスクリーンショットモード中は止まり、Auto Reloadと手動Reloadはモードに関わらず動く。Auto Reloadは30秒間隔でworkspace一覧とmetadataを取り直し、最新stepへ追従中のgraphのうち古くなったものだけrangeを更新する。workspace一覧専用のtimerは持たず、初期表示、workspace selectorへのfocus、切替結果、手動Reload、Auto Reloadを再取得境界とする。

## 7. 設定一覧

### 7.1 Metrics Viewer固有設定

すべて`application.properties`または起動引数（`--key=value`）で与える。数値設定は
`MetricsViewerSettings`、workspace path/nameは`WorkspaceManager`のconstructorで検証し、
契約違反はapplication起動を中止する。

| key | 既定 | 有効範囲 | 意味 |
|---|---:|---|---|
| `metricsviewer.workspaces-dir` | `workspaces` | local path | workspace群の親directory。UNC rootは起動時に拒否する |
| `metricsviewer.initial-workspace` | `_default` | 直下directory名 | 起動時のcurrent workspace。妥当だが不在ならWARNと空のRun一覧で起動する |
| `metricsviewer.target-points-per-series` | `8000` | 3 〜 `max-points-per-request` | requestが`maxPoints`を省略したときの1系列あたり既定vertex予算 |
| `metricsviewer.max-points-per-request` | `500000` | 3 〜 1,000,000 | 1 requestで返せるvertex総数 |
| `metricsviewer.cache-memory-mb` | `256` | 0以上、かつ最大heapの50%以下 | LOD page（完成bucketの塊）のheap上限。`0`でpage cacheを使わずbucket単位で読む |
| `metricsviewer.max-concurrent-queries` | `2` | 1 〜 4 | `/api/metrics.json`のprocess-globalな同時実行数。coordinatorのfair permitを最大5秒待つ |

`cache-memory-mb`の上限判定は`Runtime.maxMemory()`に依存する。起動scriptの`-Xmx`を下げるとこの設定だけで起動に失敗しうる。

### 7.2 Spring / Tomcat / Jackson設定

| key | 設定値 | 意図 |
|---|---|---|
| `server.port` | `8082` | Metrics Viewerの既定port |
| `server.compression.enabled` | `false` | 応答本体は既にBase64 binaryのため、圧縮を既定で無効にする |
| `server.compression.mime-types` / `min-response-size` | JSON系 / `512` | 圧縮を有効化した場合の対象 |
| `server.tomcat.connection-timeout` | `600000` | 大きなrange応答の生成待ちで切断させない |
| `server.tomcat.keep-alive-timeout` | `600000` | 同上 |
| `server.tomcat.max-keep-alive-requests` | `1000` | 連続pollでconnectionを再確立させない |
| `spring.mvc.async.request-timeout` | `600000` | 同上 |
| `spring.jackson.mapper.allow-coercion-of-scalars` | `false` | `"123"`のような型の緩い入力をfail-fastさせる |
| `spring.jackson.deserialization.accept-float-as-int` | `false` | step等へ小数を渡した場合にfail-fastさせる |
| `logging.file.path` | `logs` | 起動ディレクトリ配下へlogを出す |

request bodyは上記に加えて、`@JsonAnySetter`で未知fieldを捕捉し、1つでもあれば`invalid_request`にする。
`runId`と`tagKey`は`StrictStringDeserializer`によりJSON stringだけを受け付ける。

### 7.3 起動引数

| 用途 | 例 |
|---|---|
| 既定workspace群の可視化 | `java -Xmx1g -jar target\metrics-viewer.jar --server.port=8082` |
| 別のworkspace群の可視化 | `java -Xmx1g -jar target\metrics-viewer.jar --metricsviewer.workspaces-dir=<path> --metricsviewer.initial-workspace=<name>` |

既定pathとportは[22_metrics_viewer_java.bat](../../apps/22_metrics_viewer_java.bat)が固定する。Optuna用の別Viewer launcherは持たず、同じworkspace selectorを使う。

### 7.4 browser側の定数と永続state

serverから配布しないclient定数は[metrics-viewer.js](../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)の先頭で定義する。

| 定数 | 値 | 意味 |
|---|---:|---|
| `AUTO_RELOAD_INTERVAL_MS` | 30,000 | Auto Reload ONのときのworkspace一覧・metadata再取得間隔 |
| `INGEST_POLL_INTERVAL_MS` | 4,000 | 取り込み中Runがある間の進捗poll間隔 |
| `VIEWPORT_DEBOUNCE_MS` | 150 | zoom/pan後にrange requestを出すまでのdebounce |
| `RUN_SOLO_INTERVAL_MS` | 350 | 同じRun行の連続clickをsolo選択とみなす閾値 |
| `HOVER_SCROLL_DELAY_MS` | 300 | Tag list hoverから該当graphへscrollするまでの待ち |
| `GRAPH_SCROLL_LOCK_DRAG_THRESHOLD_PX` | 1 | scroll lock中にdrag scrollへ切り替える移動量 |

Run色は`MetricsViewerClientApp`が所有し、`refreshLists()`の先頭で解決する。まず色を持たないRunへrunId昇順で`RUN_COLORS`から先着順に配る（基本色）。
`Auto Recolor`がONなら、前回の解決時に選択されていなかったRun（初回表示で自動選択する最新Runを含む）を新しく選んだRunとし、選択順に、ほかの選択中Runの色から最も離れた色を配る。選択中だったRunの色は変えず、新しく選んだRunの基本色は見ない。OFFからONへ切り替えた時点では色を変えない。
`Recolor`は選択中Runをすべて新しく選んだRunとして扱い、現在の色を無視して選択順にpalette先頭`#2F7DE1`を起点としたfarthest-pointで配る。選択1本なら`#2F7DE1`になる。
solo選択（同じRun行の連続click）は2回のclickを1つの操作として扱う。元から選択中だったRunは色を保ち、選択していなかったRunは新しく選んだ1本として`#2F7DE1`を取る。
どちらも未選択Runの色を変えず、配る色がpaletteの20色を使い切るとラウンドを改める。

`localStorage`へ保存するstateは次の8件だけである。viewport、凡例の表示状態、Run選択、Run色は保存しない。Logとpercentileの下限・上限はworkspace名をkeyへ含めず、同名tagで共有する。

| key | 内容 |
|---|---|
| `anet.metricsviewer.workspace` | 最後に選択したworkspace。列挙に無ければserver currentで上書きする |
| `anet.metricsviewer.activeTags` | 現在選択中のtag集合 |
| `anet.metricsviewer.knownTags` | 一度でも観測したtag集合。未知tagだけを自動でactiveにするために使う |
| `anet.metricsviewer.graphScrollLockEnabled` | Scroll Lockのon/off |
| `anet.metricsviewer.autoRecolorEnabled` | `Auto Recolor`のon/off。既定はONで、`"false"`のときだけOFFとして読む |
| `anet.metricsviewer.lodDisplayMode` | `MinMax` / `Mean` / `Band` |
| `anet.metricsviewer.logScaleTags` | signed-logを有効にしたtag集合。文字列JSON配列を辞書順で保存する |
| `anet.metricsviewer.percentileBounds` | percentileの下限か上限を制限したtagごとの段。`{"<tagKey>": [下限, 上限]}`のJSON objectをtagKeyの辞書順で保存する。下限は0 / 1 / 5、上限は100 / 99 / 95で、両方とも制限なし（0と100）のtagは含めない |

Logの集合とpercentileの下限・上限は独立して復元する。Logの値が文字列JSON配列でないとき、percentileの値がJSON objectでないか段に無い値を1つでも含むときは、警告してそれぞれ空へフォールバックする。
下限・上限を2つのボタンへ分ける前の`anet.metricsviewer.ignoreOutlierTags`（p5–p95）と`anet.metricsviewer.p1P99Tags`（p1–p99）は読まずに削除する。

## 8. Metricsキャッシュのデータベース定義

### 8.1 file identityとconnection設定

| 項目 | 値 |
|---|---|
| file名 | Run directory直下の`metrics_cache.db`（WAL使用時は`-wal`、`-shm`を伴う） |
| `PRAGMA application_id` | `0x414E4554`（`ANET`） |
| `PRAGMA user_version` | `2`（`SCHEMA_VERSION`） |
| write connection | `busy_timeout=5000`、`journal_mode=WAL`、`synchronous=NORMAL` |
| read connection | `busy_timeout=5000`、`query_only=ON` |
| connectionの寿命 | 読み＝request単位、書き＝取り込みblock単位の短命 |

常時接続にしないのは、Windowsでopen中のfileを移動・削除できず、「Runフォルダの出し入れ＝可視化対象の登録・解除」という運用契約を壊すためである。
失われるpage cacheは、完成済みbucketに限定したapplication層のLRU（`LodPageCache`）で代替する。

### 8.2 table定義

#### `tags`

| column | 型 | 制約 | 意味 |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY | tagの内部id |
| `key` | TEXT | UNIQUE NOT NULL | tag文字列 |
| `type` | TEXT | NOT NULL CHECK(`'scalar'`) | 現行はscalarのみ |
| `status` | TEXT | NOT NULL CHECK(`'ok'` / `'error'`) | tag隔離の有無 |
| `error_code` | TEXT | | 隔離理由（現行は`tag_step_regression`） |
| `error_message` | TEXT | | 人間向けの補足 |
| `error_source_offset` | INTEGER | | 逆行を検出したsource offset |
| `error_previous_step` | INTEGER | | 直前のstep |
| `error_step` | INTEGER | | 逆行したstep |

#### `scalars`（L0全点）

| column | 型 | 制約 | 意味 |
|---|---|---|---|
| `tag_id` | INTEGER | NOT NULL | `tags.id` |
| `ordinal` | INTEGER | NOT NULL | tag内の出現順（0始まり） |
| `step` | INTEGER | NOT NULL | 座標値としてのstep |
| `value` | REAL | NOT NULL | finiteかつfloat32へ収まる値だけを格納する |

PRIMARY KEY `(tag_id, ordinal)`、`WITHOUT ROWID`。

#### `scalars_lod`

| column | 型 | 意味 |
|---|---|---|
| `tag_id` | INTEGER | `tags.id` |
| `level` | INTEGER | 1以上。bucket幅は`4^level` |
| `bucket` | INTEGER | `ordinalFrom / 4^level` |
| `cnt` | INTEGER | bucket内の点数（完成bucketは常に`4^level`） |
| `step_first` / `step_last` | INTEGER | bucket先頭・末尾のstep |
| `min_ordinal` / `min_step` / `vmin` | INTEGER / INTEGER / REAL | 最小値をとった点 |
| `max_ordinal` / `max_step` / `vmax` | INTEGER / INTEGER / REAL | 最大値をとった点 |
| `vmean` | REAL | bucket内平均（子の件数重み付き） |
| `vlast` | REAL | bucket末尾の値 |

PRIMARY KEY `(tag_id, level, bucket)`、`WITHOUT ROWID`。
子4件がそろった時点でのみ親を書くため、末尾の未完成bucketは行として存在しない。同値が並ぶ場合は序数が小さい点を代表にする。

#### `tag_stats`

| column | 型 | 意味 |
|---|---|---|
| `tag_id` | INTEGER | PRIMARY KEY |
| `count` | INTEGER | commit済みの有効点数 |
| `mean` | REAL | Welford法の平均 |
| `m2` | REAL | Welford法の二次モーメント。APIでは`variance = m2 / count`と`stdDev`へ変換して公開する |
| `min_value` / `max_value` | REAL | 値域 |
| `min_step` / `max_step` | INTEGER | step範囲 |
| `last_value` | REAL | 最新値 |

`WITHOUT ROWID`。`TagStats`はLODから導出せず、commit済みL0全点に対する範囲非依存の統計として同じtransactionで更新する。
LODは表示解像度のための近似であり、そこから統計を復元すると表示Step幅や空区間に結果が依存するためである。

#### `json_lines`

| column | 型 | 意味 |
|---|---|---|
| `ordinal` | INTEGER | PRIMARY KEY。非scalar行の挿入順であり、source全体の行番号ではない |
| `type` | TEXT | `meta` / `json` / `video`など、scalar以外のrecord種別 |
| `tag` / `step` / `timestamp` | TEXT / INTEGER / TEXT | 存在する場合だけ取り出した索引用の値 |
| `json` | TEXT | 元のJSON行そのまま |

現行のHTTP APIはこのtableを読まない。config dumpやmeta行を保全し、外部からSQLで解析できるようにするための領域である。

#### `source_meta`

`k TEXT PRIMARY KEY` / `v TEXT NOT NULL`の`WITHOUT ROWID`なkey-value table。

| key | 意味 |
|---|---|
| `generation` | Metricsキャッシュ世代のUUID。全再構築ごとに再発行する |
| `source_kind` | `jsonl` または `jsonl.gz` |
| `source_size` | 直近に観測したマスタのbyte数 |
| `source_mtime` | 直近に観測したマスタの更新時刻（ミリ秒） |
| `source_head_sha256` | 先頭64 KiBのSHA-256 |
| `source_commit_tail_sha256` | commit済みoffset直前64 KiBのSHA-256 |
| `committed_offset` | 確定済みのsource offset。rawは非圧縮byte、gzipは圧縮stream消費byte |
| `state` | `pending` / `converting` / `ready` / `error` |
| `error_code` / `error_message` | `state=error`のときだけ存在する |

### 8.3 破棄・再構築の判定

cache fileを開く前後で次を判定し、いずれかに該当したらWAL/SHMごと削除して新しい世代で作り直す。判定理由はINFOに残す。資源・DB接続の失敗を破損扱いで削除せず、対象Runを停止する。

| 分類 | 理由code |
|---|---|
| 様式 | `application_id_mismatch`、`schema_version_mismatch`、`required_table_missing`、`required_column_missing`、`database_corrupt` |
| メタ値 | `generation_invalid`、`state_invalid`、`source_metadata_invalid` |
| source同一性 | `source_kind_changed`、`source_head_changed`、`committed_source_tail_changed`、`source_truncated_below_committed_offset`、`source_truncated_below_previous_size` |
| gzip固有 | `gzip_conversion_session_missing`、`gzip_source_size_changed`、`gzip_source_mtime_changed` |
| error再検査 | `errored_source_size_changed`、`errored_source_mtime_changed` |

通常起動ではDBサイズに比例する`PRAGMA quick_check` / `integrity_check`を実行しない。
`application_id`、schema version、必須table/column、`source_meta`、source fingerprintの軽量検査だけを行い、
検査で接続不能または不整合を検出したcacheを再構築する。
軽量検査が参照しないdata pageの破損は起動時には検出せず、実際のingest/queryでSQLiteエラーとして扱う。
fingerprint計算のI/O失敗はcache不一致へ丸めず、`IOException`として呼び出し元へ伝播する。

`error`状態のRunは、sourceのsizeとmtimeが変わるまで再読込しない。sourceを直したら、そのRun folderのマスタを更新すれば次cycleで自動的に再構築される。

## 9. HTTP API

すべて`/api`配下で、`static/index.html`をrootとして配信する。

### 9.1 workspace API

`GET /api/workspaces.json`はcurrent workspaceと選択肢を返す。選択肢は
`metricsviewer.workspaces-dir`直下で`runs/`または`config/`を持つdirectoryの名前昇順である。
currentが不在でもcurrent名は返し、選択肢には含めない。

```json
{"current":"dm_long","workspaces":["_default","dm_long","dm_opt"]}
```

`POST /api/workspace`は`{"name":"dm_long"}`だけを受け付ける閉じたschemaで、成功時は
204 No Contentを返す。同じcurrent名は存在確認より先に204 no-opとし、epochを増やさない。
未知workspaceは404 `unknown_workspace`、bodyの型・必須field・未知field違反は400
`invalid_request`とする。不正JSONの応答変換は`WorkspaceController`内に限定し、
metrics・prioritize APIのparse失敗形式へ影響させない。

clientがselector表示後に外部でrename・削除されたworkspaceへ切り替え、`unknown_workspace`を受けた場合は、
その選択肢を除去して切替前のworkspaceへ戻し、workspace一覧を再取得してToastで通知する。
404以外の切替失敗では選択肢を除去せず、切替前のworkspaceへ戻す。
POST成功後のworkspace一覧・metadata・data再取得に失敗した場合は切替済みworkspaceを維持し、
`Workspace switched, but data refresh failed.`をToastで通知する。
一方、server current自体が一覧から消えている場合は自動的に別workspaceへ切り替えず、
`(missing) <name>`のdisabled optionとして現在値を表示し、同じmissing状態につき1回だけToastで通知する。

clientは`POST /api/workspace`の応答を待つ間だけworkspace selectorをdisabledにする。応答後に
workspace状態とselectorを同期した時点で再び操作可能にし、workspace一覧・metadata・dataのrefresh中でも
次の切替を許可する。切替ごとにworkspace switch revisionを進め、新しい切替は古いmetadata/metrics requestを
abortして各描画revisionを無効化する。古い切替世代は遅着した一覧・refresh結果、失敗表示、Toastを反映せず、
最新の切替世代だけが後処理を所有する。

切替はingest cycleと共通のgate内で、新snapshotを作ってcurrentへatomic swapしてから旧snapshotを
retireする。API queryは開始時に取得したsnapshot leaseを処理終了まで使うため、切替中も異なる
workspaceの同名Runやcacheが混ざらない。旧表示snapshotは最後のquery leaseで閉じる。gzip streamと取り込み状態はprocess schedulerが保持し、旧workspaceの変換を背景で継続する。優先更新はepochと正規化したRun絶対パス集合で渡し、旧epochの遅着要求は無視する。

### 9.2 `GET /api/runs.json`

Run作業セット全体のmetadataを返す。呼び出しのたびにruns directoryを走査するため、フォルダ操作が即座に反映される。

```
{ "runs": [ {
    "id": "run_20260612-015116",
    "generation": "0f7c...",              // UUID。cacheが無い間はnull
    "stats": { "maxStep": 2000000 },
    "ingest": { "state": "converting", "percentage": 37,
                "error": { "code": "...", "message": "..." } },   // errorはnullなら省略
    "tags": [ { "key": "train/loss", "type": "scalar", "status": "ok",
                "stats": { "minStep": 0, "maxStep": 2000000, "count": 125000,
                           "lastValue": 0.01, "minValue": 0.0, "maxValue": 1.2,
                           "mean": 0.08, "variance": 0.004, "stdDev": 0.063 } } ]
} ] }
```

- `percentage`は`state`から導く。`pending`は0、`ready`は100、それ以外は`committed_offset / source_size`を0〜99へclampする（`source_size`が0のときは`error`が0、それ以外は100）。
- cacheをまだ開けないRunは`state=pending`、`generation=null`、`tags=[]`として返し、request全体は失敗させない。
- `ingest.error`、`tags[].stats`、`tags[].error`はnullなら省略する。`generation`はnullのまま返す。
- この呼び出しはLOD page cacheの世代整理も行う。作業セットから消えたRunのpageと、世代が変わったRunの旧pageを破棄する。

### 9.3 `POST /api/metrics.json`

request:

```
{ "series": [ { "runId": "...", "tagKey": "...",
                "fromStep": 0, "toStep": 2000000, "stepBucketWidth": 1000, "maxPoints": 8000 } ] }
```

必須header:

| header | 契約 |
|---|---|
| `X-Query-Channel` | 1つのブラウザタブに対応する1〜128文字の非blank文字列。値はtrimしない |
| `X-Query-Sequence` | channel内でPOSTごとに増える0以上のJavaScript safe integer |

browserはページ生成時に`crypto.randomUUID()`が利用可能ならその値をchannelに使う。非secureなリモートHTTPなどで同APIが未定義の場合は、時刻と複数の乱数片から128文字以内のtab固有channelを生成するため、`http://<host>:8082`での閲覧も維持する。

`fromStep` / `toStep`は必須の閉区間で、JavaScriptのsafe integer範囲（±9,007,199,254,740,991）に収める。`stepBucketWidth`は必須で正のsafe integer。`maxPoints`は省略可で、既定は`target-points-per-series`である。

response（系列ごと）:

| field | 意味 |
|---|---|
| `runId` / `tagKey` / `fromStep` / `toStep` | requestの反映値 |
| `generation` | 応答を作ったcache世代。clientはmetadataの世代と一致しない結果を捨てる |
| `availability` | `ok` / `pending` / `not_found` / `empty` |
| `pointBudget` | 固定Step幅に対する応答頂点数の上界。`ok`以外は0 |
| `stepBucketWidth` | 表示の共通Step区間幅。`ok`以外はnull。保存levelは公開しない |
| `issues` | `{scope, code, message}`の配列。空なら省略 |
| `projection` | `kind = "raw"` または `"lod"` |

`availability`の決まり方は次のとおりで、取り込み継続中かどうかで`pending`と確定値を分ける。

| 状況 | 取り込み中 | 取り込み完了 |
|---|---|---|
| Runが作業セットに無い | `not_found` | `not_found` |
| cacheを開けない / query失敗 | `pending` | `pending` |
| tagがまだ無い | `pending` | `not_found` |
| tagはあるが点が0 | `pending` | `empty` |
| range内に点が無い | `pending` | `empty` |
| range内に点がある | `ok` | `ok` |

`MetricsQueryPlanner`は各`ok`系列の必要頂点数上界を計算する。各系列の`maxPoints`超過、または合計の`max-points-per-request`超過は422で拒否し、Step幅を変更しない。

error応答:

| status | body | 契機 |
|---|---|---|
| 400 | `{"code":"invalid_request","message":...}` | body違反、query headerの欠落・空・長さ超過・形式/範囲違反 |
| 409 | `{"code":"superseded","message":...}` | 同一channelの新query、workspace切替、shutdown、遅着sequenceにより停止された |
| 422 | `{"seriesCount":N,"requiredPoints":M,"maxPointsPerRequest":K}` または `{"seriesIndex":I,"requiredPoints":M,"maxPoints":K}` | 固定Step粒度の必要量がrequest全体または系列上限を超過 |
| 503 | `{"code":"query_busy","message":...}` + `Retry-After: 2` | 別channelが枠を占有して5秒以内にpermitを取得できない、またはshutdown中にleaseを取得できない |

### 9.4 `POST /api/runs/prioritize`

`{"runIds": ["...", "..."]}`を受け取り、取り込み優先Run集合を丸ごと置換する。成功時は204 No Contentである。
存在しないRun id、空文字、未知fieldは400にする。clientはRun選択が変わるたびに送り、前回の送信完了を待って直列化する。

### 9.5 binary projectionの符号化

数値列はJSON配列ではなく、little-endianのbinaryをBase64化した文字列の配列として返す。1 chunkは250,000要素で、float換算1 MBである。

| 系列 | 要素型 |
|---|---|
| `steps`、`minSteps`、`maxSteps` | float64 |
| `values`、`mins`、`maxs`、`means` | float32 |

```
raw : { "kind":"raw", "steps":[...], "values":[...] }
lod : { "kind":"lod",
        "minMax":  { "steps":[...], "values":[...] },
        "summary": { "steps":[...], "mins":[...], "maxs":[...],
                     "means":[...], "minSteps":[...], "maxSteps":[...] } }
```

`minMax`は各bucketの`min` / `max` / `last`を序数順に並べた折れ線用の点列で、同一序数が重なる場合は1点へ畳む。
`summary.steps`は共通Step区間の左端であり、band描画とbucket単位のhover表示に使う。

## 10. lifetime・並行制御・エラー境界

### 10.1 lifetime

- `MetricsService`の`@PostConstruct`で`LoadingThread`を開始する。`@PreDestroy`は`MetricsQueryCoordinator.cancelAll()`、`LoadingThread`の最大30秒停止待ち、`WorkspaceManager.shutdown()`の順で実行する。
- metrics queryはcoordinatorのpermit取得後にworkspace leaseを取得する。終了時はSQL/connection、lease、permitの順に解放し、切替でretireされた旧resourceをpermit返却前に閉じる。
- `LoadingThread`はdaemonの単一writer。即時処理可能なbacklogがなければ最大10秒waitし、優先変更・切替・shutdownで起こす。予期しないRuntimeExceptionはERRORに残しwriterを停止する。
- `WorkspaceManager.shutdown()`はterminalである。開始後の新規leaseとworkspace切替は`IllegalStateException`で即時終了し、進行中cycleは現在blockの安全な終了後に停止する。取得済みquery leaseは利用を継続でき、最後のreleaseがretire済み表示resourceを1回だけ閉じる。取り込みresourceはwriter終了時またはshutdownでprocess単位に閉じる。
- gzip変換中のRunは展開済みstreamを`GzipInputSessions`がblock間で保持する。未選択workspaceの背景変換中にも、そのRun folderの移動をサポートしない。`ready`到達、失敗、登録対象からの消失、shutdownで解放する。
- `LodPageCache`は完成bucketから成るpageを末尾pageも含めて保持し、Run消失と世代変更で破棄する。容量超過時はアクセス順のLRUで追い出す。

### 10.2 並行制御

| 境界 | 機構 |
|---|---|
| cacheの全破棄・再構築 | Run directory単位のlifecycle write lock |
| 通常のread/write transaction | 同lockのread lock + WAL |
| query 1本のsnapshot固定 | Runごとに1本のread connectionを`autoCommit=false`で保持 |
| query同時実行・supersede | `MetricsQueryCoordinator`のfair permit（取得待ち最大5秒）、channel別最新sequence、identity付きlive ticket。枠待ちは待機threadだけを起こし、実行中SQLは`Statement.cancel()`とcheckpointで止める |
| workspace切替とquery | `SWITCHED`の旧epochだけを切替gate解放後にcancelする。`NO_OP`と`UNKNOWN`はcancelしない |
| 取り込みwriter | `LoadingThread` 1本のみ。writerの多重化を前提にしない |
| workspace切替と取り込み | fairな切替gateを取り込みblockごとに解放する。待機中の切替要求は現在blockを完全行境界で早期commitさせ、POST成功後は新epochで配分し、旧workspaceの作業は背景として継続する |
| priority集合の更新 | epoch検証を切替gate内で行い、旧epochの要求は無視する。待機要求はwriterを早期commit・wakeさせる。scan開始後に追加されたpriorityを古いscan結果で削除しない |

### 10.3 エラーとWARNの方針

- source側の致命的な不正（不正JSON、`type`欠落、`tag`欠落、`value`欠落、不正step、gzip corrupt）はtransactionをrollbackし、Runを`error`にする。行を読み飛ばして先へ進めない。
- 1点だけの異常（`null`、非数値、非finite、float32外）はその点をskipし、`(run, tag, 理由)`ごとに1回WARNする。
- step逆行は該当tagだけを隔離し、他tagのcommit済みデータを隠さない。
- `runs.json`のRun単位読み取り失敗は例外を投げず`pending`へ丸め、1 Runの破損で一覧全体を失わせない。
- WARN抑止は`RunWarningRegistry`が持ち、正規化したRun絶対パスをキーとし、全workspaceの登録対象から消えたときに解除する。異なるworkspaceの同名Runは独立し、再投入すると再度WARNする。

- DB書き込み・読み取り失敗はrollbackとstream解放後のsource/cache属性と理由をprocess内に保持し、そのRunだけを停止する。属性変化、除去後の再投入、process再起動まで自動retryしない。他Runとqueryは継続する。正常な版更新はINFO、読取・書込失敗はERROR。

### 10.4 性能特性

- 取り込みは1 block最大1,000,000行のstreaming parseで、中間Listを作らない。64KiB bufferと4096件のSQL batchを使い、1秒または優先変更・切替・停止要求で完全行境界から短いblockを確定する。非scalarの保留文字列は1MiB（UTF-16文字数×2）でもflushし、単独の長い行は単独で処理する。例外時のcloseではbatchを実行しない。L0、LOD、`TagStats`、source位置はどちらも同一commit境界で確定する。
- 完全検証済みの`ready` / `error` Runはprocess memoryにsource/cache属性を保持し、属性不変のpollではfingerprint、Metricsマスタ本文、SQLite connectionへ入らない。観測はRun消失とprocess restartで失われ、表示snapshotの切替だけでは失われない。
- range queryのコストはstep二分探索、bucket読み出し、部分bucketの再集約に分かれる。下位への読み出しが必要なのは、保存bucketが表示Step区間を跨ぐ部分と、range端・未完成末尾である。
- step境界がtagのstep範囲（`min_step`〜`max_step`）の外側にある端は、序数が0または点数に決まるので二分探索を省く。autorangeのwindowは左右へ1画面広げるため、全域表示では両端ともこの経路になる。
- LOD pageは1024 bucket単位で読み、末尾pageも含めてheapに残す。levelごとの完成bucket数は、同じsnapshotのtag点数から`点数 / 4^level`（切り捨て）で決まる。子4件がそろった瞬間に親を書くためである。
  これより後ろのbucketはDBを引かずに下位levelから再集約し、保持中の末尾pageが完成bucket数に足りないときだけ読み直す。
  完成bucketは書き換わらないので、新しいsnapshotで読んだpageを古いsnapshotのqueryが使っても結果は変わらない。
  読んだpageが完成bucket数に足りなければcacheの不変条件違反として、その系列を`query_error`にする。
- 末尾pageも保持するのは、取り込みを終えたRunの末尾pageは満杯にならず、保持しないとbucketを1個引くたびにpage全体を読み直すためである。100M stepのRunを5本並べた616系列のrequestでは、これがquery時間の8割を占めていた。
- 1 pageは最大`1024 × 96` byte（long 8列 + double 4列）である。
- 応答はBase64 binaryのため、HTTP圧縮は既定で無効にしてCPUを使わない。

### 10.5 全workspace保守とresource予算

起動直後と前回全走査完了から10秒周期で全workspaceを検査する。1区間は最大128 Runまたは50msとし、残りのiteratorを次区間へ引き継ぐ。directory列挙自体は原子的で、別途時間を記録する。属性が不変ならSQLite・fingerprint・本文へ入らない。初回またはsource/cache/WALの属性変更時だけ短いread connectionでheaderを検査し、旧schemaでは旧tableを読まず待ち行列へ登録する。削除と完全検証は実際のblock開始時に行う。優先Runはblock境界で属性だけを追加確認する。選択Runがある場合のidle待ちは最大1秒とし、全workspaceの10秒検査周期を短縮せずに反映待ちを抑える。

gzip streamは全体16本まで。新しい非優先Runは、非優先の開始済みstreamが4本未満の場合だけ開始する。選択Runは全体上限まで開始できる。選択を外れた開始済みstreamも継続する。sourceがgzipからrawへ変更された場合は、旧gzip streamを取り込み前に解放する。枠待ちはevictionせず、理由の状態遷移でINFOを出し、再開時に`waitMs`を出す。

検査結果はworkspaceごとに初回・結果変更時のみINFOで集計する。再構築の開始・進捗・完了はworkspace、Run、generationを含み、行数・有効点数は`scope=since_start`のcommit済み値である。進捗は30秒以上空けて次のcommit後に出す。gzipの百分率は概算であり、EOF検証・最終commit前は100%や完了を出さない。`elapsedMs`は待機込み、`processingMs`は実処理時間。prepare、reader、parse、L0、LOD、非scalar、flush/statistics、metadata、commit、resource解放を計測し、実処理時間から処理速度を出す。正常な通常追記を再構築として記録しない。

書込用SQLite接続には `PRAGMA cache_size=-16384` で16MiBのpage cache目標を与える。短いblockでのcommit負担を実測したため、tag別L0/LODページのspill抑制を狙う。これは接続単位の内部値であり、保存schemaや読み取り用接続を変更しない。SQLiteのnative memoryで、JVM heap peakには含まれない。[SQLite cache_size仕様](https://www.sqlite.org/pragma.html#pragma_cache_size)。

## 11. ビルドと依存ライブラリ

### 11.1 プロジェクト座標

| 項目 | 値 |
|---|---|
| groupId / artifactId | `io.github.kazukin123.anetlab` / `metrics-viewer` |
| version | `0.1.0-SNAPSHOT` |
| packaging | `jar`（`finalName = metrics-viewer`） |
| Java | 17（`maven.compiler.release`） |
| encoding | UTF-8 |

### 11.2 依存ライブラリ

| groupId:artifactId | version | scope | 用途 |
|---|---|---|---|
| `org.springframework.boot:spring-boot-starter-web` | 3.5.7 | compile | 組み込みTomcat、Spring MVC、Jackson auto-config、静的resource配信 |
| `org.springframework.boot:spring-boot-starter-logging` | 3.5.7 | compile | SLF4J + Logback |
| `com.fasterxml.jackson.core:jackson-databind` | 2.20.1 | compile | JSONL 1行のparseとAPI request/responseのbind |
| `org.xerial:sqlite-jdbc` | 3.53.1.0 | compile | Metricsキャッシュ（`jdbc:sqlite:`） |
| `org.projectlombok:lombok` | 1.18.42 | compile（optional） | view modelの`@Data` / `@Builder` |
| `org.springframework.boot:spring-boot-devtools` | 3.5.7 | compile（optional） | 開発時のみ。`spring-devtools.properties`でrestartは無効化済み |
| `org.springframework.boot:spring-boot-starter-test` | 3.5.7 | test | JUnit 5、`@SpringBootTest`、MockMvc |
| `com.microsoft.playwright:playwright` | 1.60.0 | test | browser表示テスト。`com.google.gson`も本依存から推移的に入る |

`jackson-databind`とloggingはstarterからも入るが、版を固定するため明示宣言している。

### 11.3 build plugin

| plugin | version | 設定 |
|---|---|---|
| `spring-boot-maven-plugin` | 3.5.7 | `repackage`実行可能JARを生成する |
| `maven-compiler-plugin` | 3.15.0 | `release=17`、`parameters=true`、`proc=full`、annotation processorにlombokを指定 |

Java側のコード規約は[checkstyle.xml](../../apps/metrics-viewer/checkstyle.xml)で定義する。行頭indentはtab（tab幅4、継続行8）、行長上限120、`FinalLocalVariable`と標準の命名規則を要求する。Eclipse用の設定は`.checkstyle`とformatter定義に置く。

### 11.4 build・実行

```powershell
cd apps\metrics-viewer
mvn -B test
mvn -B package
java -Xmx1g -jar target\metrics-viewer.jar --server.port=8082
```

### 11.5 frontend asset

frontendはnpm等のbuild工程を持たず、`src/main/resources/static`をそのまま配信する。
外部依存はCDNから読むPlotly（`https://cdn.plot.ly/plotly-2.27.0.min.js`）だけで、frameworkは使わない。
完全offline環境で使う場合はasset取得方法を別途用意する必要がある。

## 12. テストと拡張時の確認事項

### 12.1 テストの構成

| 領域 | 主なテスト |
|---|---|
| cache DBの様式・破棄再構築 | `MetricsCacheDatabaseIntegrationTest`、`MetricsCacheIntegrationTest` |
| 取り込み（block、gzip、error、隔離） | `MetricsIngestorIntegrationTest` |
| LODの構築と射影 | `MetricsLodIntegrationTest`、`LodPageCacheTest` |
| scheduling / workspace lifetime | `IngestSchedulerTest`、`LoadingThreadTest`、`WorkspaceManagerTest`、`WorkspaceSnapshotIntegrationTest` |
| query計画とsnapshot | `MetricsQueryPlannerTest`、`MetricsRepositorySnapshotIntegrationTest`、`MetricsRepositoryStepRangeIntegrationTest`、`MetricsQueryConcurrencyTest` |
| HTTP API | `MetricsApiIntegrationTest`、`WorkspaceApiIntegrationTest`、`SeriesAvailabilityTest`、`HttpAccessLogFilterTest` |
| 走査・設定 | `RunScannerTest`、`MetricsViewerSettingsTest` |
| browser UI | `RunListPlaywrightTest`、`TagListPlaywrightTest`、`MetricsPlotPlaywrightTest`、`GraphInteractionPlaywrightTest`、`ReloadPlaywrightTest`、`SignedLogPlaywrightTest`、`OutlierRangePlaywrightTest`、`WorkspaceSelectorPlaywrightTest` |

Playwrightテストは既定でMicrosoft Edgeを起動する。Edgeが無い環境では`Assumptions`によりskipされ、失敗にはならない。
テストごとにcontextを開き直し、route、`localStorage`、Plotly stateを共有しない。

### 12.2 変更時の確認事項

検証手順・性能比較・データ一致の記録は[PRD090検証README](../../core/anet-core/testdata/prd090/README.md)を参照する。

1. cache schemaを変えるときは`SCHEMA_VERSION`を上げる。migrationは書かず、旧cacheが警告なしで破棄・再構築されることをtestする。
2. `IngestState`と`SeriesAvailability`のexternalNameは永続値かつHTTP公開値である。値の追加は可、既存値の改名・削除は非互換とみなす。
3. 取り込みの新しい処理は同一transactionへ入れる。L0、LOD、`TagStats`、`source_meta`が別commitへ分かれるとcrash時に整合が壊れる。
4. LODを触るときは、完成bucketだけ永続化する契約と、端の部分bucketを下位levelから再集約する経路の両方を検証する。`LodPageCache`は完成bucket数をtag点数から導くので、親を書く契機を変えるときはこの導出も合わせて変える。
5. 統計をLODから導出しない。範囲非依存の統計はcommit済みL0全点から作る。
6. Step粒度または点数上限を変えたら、1〜4Runの幅一致、選択順序・zoom/pan/reset/resize、422と503をtestする。PRD089の手順は[検証README](../../core/anet-core/testdata/prd089/README.md)を参照する。
7. clientへ新しい状態を足すときは、Reload時にPlotly DOMを再構築しても保たれるようclient app側に持たせる。
   graphの見た目を決める入力なら、共通表示モデルの描画key（`PlotlyController._graphModel` / `_graphDisplayKey`）へ加える。Plotly上で直接変わる状態も`capturePlotState`で読み戻してモデルへ反映する。モデルへ反映されない入力は、変わっても描画更新や遅着結果の排除に使われない。
   windowの中身を左右する条件なら、Reloadの鮮度判定（`DataCache.isOutdated`）にも反映する。
8. 長時間実行のRunに対しては、追記中（`converting`）と完了後（`ready`）の両方で同じrangeが同じ結果を返すことを確認する。
9. Runフォルダの変更が次の検査で反映され、raw input handleはblockごとに閉じることを確認する。gzip変換は未選択workspaceの背景処理もhandleを保持するため、フォルダ移動前にViewerを停止する。

## 13. 関連文書

- [Run分析ユーザーガイド](030_user_guide_analysis.jp.md)
- [開発環境](040_development_environment.jp.md)
- [可観測性](140_observability.jp.md)
- [アプリケーションとツール](160_applications_and_tools.jp.md)
- [Metricsキャッシュを破棄可能な従属導出物とするADR](../adr/0015-metrics-cache-disposable-derivative.md)
- [ドメイン用語集](../../CONTEXT.md)
