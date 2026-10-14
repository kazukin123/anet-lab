# PRD 090: MetricsViewerのキャッシュ再構築を高速化し全ワークスペースで事前実行する

## 1. 問題

MetricsViewerのキャッシュ版が上がると、利用者はキャッシュの作り直しに長時間待たされる。未選択のワークスペースはキャッシュ版の確認と取り込みの対象にならず、そのワークスペースを開くまで再構築の必要性を検出できない。必要になってから待つ運用を、未選択でも検出し、背景で準備する運用へ変える。

2026-10-12の着手前コード確認では、次の構造を確認した。

- schema 2と保存LOD factor4への変更により、旧キャッシュはMetricsマスタから全再構築される。
- 取り込みはcurrent workspaceのsnapshotに属する。workspace一覧は名前を列挙するだけで、未選択workspaceのキャッシュを検査しない。
- 完全行の読み出しは1byte単位である。L0とLODのINSERTはPreparedStatementを再利用するが、1件ごとにSQLを実行する。
- transactionは既にblock単位であり、1点ごとのcommitではない。
- workspace切替で旧snapshotのgzip streamを閉じる。変換途中のgzipでstreamが無くなると、次回取り込み時にキャッシュを先頭から再構築する。

読み出しとSQL実行は高速化候補であり、実Runでどの工程が支配的か、何倍短縮できるかは未計測である。gzipの切替時やり直しはコード上の条件として確認しており、利用者の今回の待ち時間すべてがこの条件によるとは断定しない。

## 2. 目標と解決策

1. バッファ単位の入力と件数上限付きSQL batchで、Metricsマスタからの再構築時間を短縮する。
2. 起動後と定期走査で全ワークスペースのキャッシュ版・存在・未完了状態を軽量に確認し、必要なRunを再構築キューへ登録する。
3. 単一writerを維持し、選択Runを優先しながら、現在の未選択Runと他workspaceにも処理枠を配分する。
4. 取り込み作業とgzip streamの寿命を表示workspaceの寿命から分離し、workspace切替だけを理由に進捗を捨てない。
5. 検査結果、再構築開始、確定済み進捗、完了、失敗をログへ出力する。

再構築キューは、未処理のRunと再構築理由をprocess内で管理し、次に処理するRunを選ぶためのものである。汎用ジョブ基盤や新しい永続ストアを作らない。process起動時にはキャッシュ自身を確認して必要な作業を再発見する。

画面の可視化対象は引き続き選択中workspaceとする。背景のキャッシュ保守対象は設定済みworkspaces directory直下の全workspaceへ広げる。背景作業のために表示workspaceを切り替えたり、別workspaceのRunを画面一覧へ混ぜたりしない。

## 3. 利用者の要求

1. 利用者として、キャッシュ版更新後の再構築を短縮し、グラフを待つ時間を減らしたい。
2. 利用者として、まだ開いていないworkspaceの旧キャッシュも自動検出し、開く前に準備したい。
3. 利用者として、キャッシュのないRunも同じ背景処理で準備したい。
4. 利用者として、選択したRunを優先し、他workspaceの大きな再構築で長時間待たされたくない。
5. 利用者として、選択Runへの継続追記中でも背景Runの再構築を進めたい。
6. 利用者として、workspaceを切り替えても、途中まで作ったgzipキャッシュを捨てたくない。
7. 利用者として、同名Runを複数workspaceへ置いても、キャッシュ・進捗・優先度が混ざらないことを求める。
8. 利用者として、Runやworkspaceをフォルダ操作で追加・除去でき、消えた対象への作業が終了することを求める。
9. 利用者として、再構築理由と旧版・新版をログで確認したい。
10. 利用者として、変換が進んでいるかを確定済みの行数・点数・進捗率・経過時間から判断したい。
11. 利用者として、完了時に所要時間と処理速度を確認したい。
12. 利用者として、失敗したworkspace・Run・工程・理由をログから特定したい。
13. 利用者として、正常なキャッシュを定期的に作り直すことや、進捗ログの大量出力を避けたい。
14. 利用者として、Metricsマスタ、重複Step、全点統計、表示粒度を保持したまま高速化したい。
15. 利用者として、更新状況のための新しい画面や操作を増やしたくない。

## 4. 確定事項と実装上の契約

### 4.1 入力と書き込みの高速化

- SourceReaderはバッファ単位で読み、バッファ内の改行を探索して完全行だけを取り込みへ渡す。巨大なファイル全体や全行のListをheapへ持たない。
- UTF-8文字・CRLF・JSON行がバッファ境界を跨いでも同じレコードを生成する。rawは取得時点の末尾より先を次blockへ送り、未終端の末尾行を公開しない。gzipはEOF・CRC等の終端検証を維持する。
- L0、LOD、非scalar行のINSERTを件数上限付きbatchへまとめる。batchのflushとtransactionのcommitを別の境界として扱い、同じtransaction内で複数batchをflushできる。
- L0、LOD、TagStats、tag隔離状態、source位置は同じcommitで確定する。SQL batch失敗時はblockをrollbackし、読取位置や統計だけを先行公開しない。
- batch化でtagの序数、同一Stepの複数点、極値の同値時の選択、点数加重平均、非scalar行の順序を変えない。
- Reader、JSON parse、L0書き込み、LOD生成・書き込み、commitの時間を分けて測定する。JSONのstreaming parseへの変更は、parseが主要費用と確認できた場合の追加候補とし、初期の必須変更に含めない。
- read buffer、SQL batch、blockの件数・時間予算は上限を持つ。具体値と調整根拠は実装計画・測定記録に固定し、新しい利用者向け設定を既定では追加しない。

### 4.2 全ワークスペースの軽量検査

- 起動後、browser接続やworkspace選択を待たずに、設定済みworkspaces directory直下の既存判定で認識できる全workspaceを検査する。定期走査は追加・除去・キャッシュ属性変更を再検出する。
- 対象は各workspaceのRun directory直下でMetricsマスタを持つRunとする。設定範囲外の退避Runや別のdirectoryを再帰探索しない。
- 軽量検査は列挙、file属性、初回または変更時のキャッシュheader検査を中心にする。版不一致の発見だけで旧テーブルを解釈せず、Metricsマスタ本文の全走査・全fingerprint・書き込みを行わない。
- キャッシュ未作成、現行版不一致、未完了の再構築、変更により通常の検証が必要な対象をキューへ登録する。検査だけではキャッシュを削除しない。
- 作業開始時にRunの存在、Metricsマスタ、キャッシュを再確認する。検出後にsourceが変わった場合は古い観測のまま破棄・追記せず、既存の完全検証へ戻る。
- 正常に検証済みでsource/cache属性が不変なready/error Runは、通常pollでSQLite connection・fingerprint・Metricsマスタ本文へ入らない。PRD 064の定常I/O抑制を全workspaceでも保持する。
- 未選択workspaceで検査に失敗しても、他workspaceの検査や選択中workspaceのAPIを失わせない。対象と原因をログへ残し、成功扱い・未作成扱いに丸めない。
- 走査周期と1回の走査・処理予算は実装計画で定める。周期内に走査を開始し、有限個の対象を公平に検査する。大量Runがある場合も起動・HTTP要求を全走査の完了待ちにしない。

### 4.3 全体の取り込み配分

- LoadingThreadの単一writerを維持する。process全体で同じRunへ複数writerが到達しない。
- 優先対象は現在の表示workspaceで選択されたRun。背景対象は現在の未選択Runと他workspaceの作業対象である。
- 既存の優先3:背景1を、処理可能なblockへの配分として全体へ適用する。片側に作業が無ければ他側へ枠を渡す。terminal/no-op対象の確認だけで処理枠を消費しない。
- 背景対象はworkspaceを跨いで公平に巡回し、継続追記中の優先Runがあっても有限長の背景再構築が進み、最終的に完了できる。
- 処理単位は件数上限に加えて時間予算を持ち、完全行の境界で早期commitできる。優先度変更・workspace切替は現在の安全な処理境界で反映する。必要な入力resourceを確保できる優先対象の処理を、背景Run全体の完了まで待たせない。
- 検査、初期化、commitなど分割しにくい工程の時間も測定する。時間予算を超えたblockが残る場合は、工程と理由を記録して受入判断へ含める。
- 作業identityは正規化したRunの絶対パスを使う。Run名だけをキーにせず、同名Runのworkspace間衝突と重複登録を防ぐ。
- フォルダ除去時はキュー・検証済み観測・gzip streamを解放する。追加・再投入は新しい観測として検査する。取り込み停止後にdirectoryを再作成して作業を続けない。

### 4.4 変換作業と表示workspaceの寿命

- WorkspaceManagerは表示workspace、query epoch、query leaseの切替を管理する。取り込み作業とGzipInputSessionsは表示snapshotより長いprocess内の寿命で管理する。
- 表示切替で旧queryをcancelし、旧表示側のresourceを解放する既存契約は維持する。切替成功後も、旧workspaceの未完了取り込みは背景対象として継続できる。
- 同じMetricsマスタのgzip変換をworkspace切替だけで先頭へ戻さない。切替前後でキャッシュ世代を維持し、commit済み点数とsource位置を後退させない。
- gzip stream、buffer、Runごとの状態には上限を設ける。未開始のgzip作業はstream枠が空くまでキューで待ち、開始済み作業をworkspace切替や新しい選択だけで捨てない。上限飽和時の新しい選択Runは入力resourceの確保待ちとなり、blockの配分待ちと区別してログへ理由を出す。優先対象用の枠の確保を含む具体的な上限・割当と待機特性は実装計画で定め、検証する。
- streamは完了、失敗、対象消失、process shutdownで解放する。SQLite connectionは引き続き取り込みblock単位・read request単位で閉じる。
- gzip変換中にそのRunフォルダの移動をサポートしない既存制約は維持する。未選択workspaceでも変換中のRunに同じ制約が及ぶことを現行文書へ明記する。
- process再起動後のgzip途中位置からの再開は本PRDの保証対象外とし、現行のstream喪失時の再構築契約を維持する。source不一致やキャッシュ破損を検出したときの全再構築も維持する。

### 4.5 変換状況のログ

ログは英語の固定messageと識別可能なfieldで出す。正常な検出・開始・進捗・完了はINFOとし、通常の版更新をWARN扱いにしない。新しいUI、更新状況API、永続的な進捗tableは追加しない。

| 契機 | 出力内容 | 出力頻度 |
| --- | --- | --- |
| workspace検査 | workspace、検査Run数、再構築必要数、理由別件数 | 初回検査と結果変更時。変化のない定期走査ごとには繰り返さない |
| 再構築開始 | workspace、Run、Run path、理由、旧版、新版、source種別・bytes、キャッシュ世代 | 実際に作業を開始した再構築ごとに1回 |
| 再構築進捗 | workspace、Run、キャッシュ世代、commit済み行数・有効scalar点数、確定source offset、進捗率、経過時間 | Runごとに既定30秒以上の間隔で、次のcommit後に出す |
| 再構築完了 | workspace、Run、キャッシュ世代、総確定行数・有効scalar点数、経過時間、処理時間、処理速度 | EOF検証と最終commit後に1回 |
| 失敗・取消 | workspace、Run、工程、理由・error code、確定済み位置、経過時間 | 状態遷移時。対象消失・shutdownは正常な取消としてINFO |

- キュー登録を開始や完了と表現しない。行数は消費した完全なJSON行数、有効scalar点数はL0へ確定した点数とし、skip・非scalar行との区別を保つ。既存cacheを途中から継続する場合に通算値を取得できなければ、その開始以降の値と明記する。
- rawの進捗率は検証済みsource snapshotのbytesとcommit済みoffsetから算出する。gzipの圧縮byte比率は先読みを含む概算と明記し、未完了なのに完了を示す100%を出さない。完了判定は率に依存しない。
- 経過時間は待機を含む。処理速度は、この作業で確定した行数・点数を作業に費やした処理時間で割り、待機込みの時間と混同しない。0除算や未成立値を偽の速度で埋めない。
- 健全な通常追記を毎block「再構築」として記録しない。変更なしのready Runから進捗・完了ログを繰り返さない。
- 進捗集計のためのMetricsマスタ再走査や毎blockの全表COUNTを行わない。作業中の集計と既存の確定情報を使い、追加の永続counterを既定にしない。
- 取得できない旧版や旧generationを取得済みの値として捏造せず、unknown等の明示表記を用いる。版不一致cacheの旧tableをログ生成のために読むことを要求しない。
- 読取・変換・DB書込の失敗は既存のRun単位のerror扱いと整合させる。同じ失敗がsource/cacheの変化なしに繰り返されてもログをspamせず、黙って再構築を無限retryしない。

### 4.6 維持する永続・表示契約

- Metricsマスタを唯一の正とし、旧キャッシュはmigrationせず全破棄・再構築する。過去のMetricsマスタ・Run artifactを加工しない。
- 本PRDの高速化・配分・寿命・ログ変更だけを理由にschema 2を上げない。保存LOD factor4、序数PK、TagStats、tag隔離、source fingerprint、キャッシュ世代の意味を維持する。
- queryのsnapshot、WAL、再構築時のRun単位排他、query supersede、共通Step区間、概観・詳細描画の整合性を維持する。
- 新設定、旧API alias、旧版専用の読替層を追加しない。明示された現行設定の不正値は既存どおり起動時に失敗させる。

### 4.7 主な責務と文書追随

| 責務 | 主な変更対象 | 変更の目的 |
| --- | --- | --- |
| 完全行の入力 | SourceReader、RawFileReader、GzipSessionReader | buffer境界・完全行・source位置をまとめて扱う |
| 原子的な取り込み | MetricsIngestor、LodIngestWriter | 上限付きbatch、集計、commitとログの整合を保つ |
| キャッシュ判定 | MetricsCacheDatabase | 軽量な観測と実際の準備・破棄を分ける |
| 全体配分 | LoadingThread、IngestScheduler | 全workspaceの発見・重複排除・公平な作業配分をまとめる |
| 表示と作業の寿命 | WorkspaceManager、GzipInputSessions | query切替と変換継続の所有権を分ける |

新しいmoduleが必要な場合もMetricsViewer内のこの責務に限定し、他機能向けの抽象ジョブ基盤へ広げない。具体的なinterface・class名は実装計画で定める。

実装時は現用コードとテスト、MetricsViewer設計文書の日英版、分析ユーザーガイドの日英版を同じ変更で追随させる。CONTEXTの「Run作業セット」は現在の可視化対象を表す語として維持し、キャッシュ構築を選択中workspaceだけに限定する記述を新しい背景保守契約へ更新する。新しいdomain用語はこの起票では追加しない。

ADR 0015のマスタ・短命connection・migrationなし、ADR 0052の保存factor4と表示契約は維持する。選択中workspaceだけに取り込みを限定する現行lifecycleは本PRDで変更する。ADR 0015のフォルダ操作を妨げない意図に対し、gzip変換中のhandle保持という現行例外が未選択workspaceにも広がる点を明記する。コミット済みADRを黙って書き換えない。

### 4.6 実装計画で固定した予算と失敗境界

2026-10-12に承認された[実装計画](090_metricsviewer_cache_rebuild_20impl.md)を実装の正本とする。入力64KiB、SQL batch4096件、非scalar保留文字列1MiB（UTF-16文字数×2）、block最大100万行／1秒を内部既定値として固定する。単独の長い行・列挙・初期化・commitにはsoft budgetを超える可能性があり、工程別時間を記録する。

全workspace検査は起動直後と前回全走査完了から10秒周期、1区間最大128 Run／50ms。gzip streamは全体16本、新規背景gzipは非優先の開始済みstreamが4本未満の場合だけ開始する。選択Runは全体上限まで開始可能とし、選択から外れたstreamも継続する。進捗ログはRunごと30秒以上空け、commit後に公開する。選択Runがある場合のidle待ちは最大1秒とし、全workspaceの検査周期は10秒を保つ。

DB書き込み失敗はrollbackとstream解放を行い、終了後のsource/cache属性と失敗理由をprocess内へ保存して、そのRunだけを停止する。同じ属性では接続・再試行を行わず、属性変化・対象再投入・Viewer再起動で再検査する。他Runと閲覧は継続する。予期しない内部例外は無限retryせずwriterを停止する。

性能受入は、同じ停止済み約510MiB gzipマスタとそのraw版、同じJVM条件、保存factor4、各3回の比較でraw/gzip双方の再構築時間中央値がbaselineより短いこと。全永続データの一致と、入力resource確保済み選択Runの反映待ちp95が10秒以内であることを要求する。ばらつき・heap peak・DB容量・stream枠待ちを併記し、実Run cacheと通常JARは変更しない。

書込SQLiteのpage cache目標は接続単位で16MiBとする。短いblockのcommit負担を実測し、page spill抑制を狙った内部調整であり、schema・read接続・設定キーは維持する。native memoryのためJVM heap peakと分けて記録する。

## 5. テストと受入条件

テストは出力データ、世代、確定位置、API結果、ログ、resource解放を観測する。batchの内部実装だけを写すテストや、壁時計のsleepに依存するログ試験は作らない。小さな処理予算・fake clockを注入して境界を再現し、実Run性能の測定とは分ける。

1. 正式PRD番号が090で一意であり、既存PRD・実装計画を上書きしていない。
2. raw/gzipの同一マスタを現行方式と新方式で取り込み、L0、全levelのLOD、TagStats、非scalar行、tag隔離状態が一致する。浮動小数点の比較条件を明記する。
3. バッファ境界を跨ぐUTF-8・CRLF・長いJSON行・raw未終端末尾・gzip corruptで既存の完全行と失敗契約を保つ。
4. batch flush、block早期終了、SQL失敗の各境界で、source位置・L0・LOD・統計が同じtransactionに属する。失敗したblockの点やoffsetを公開しない。
5. 未選択workspaceに旧版・未作成・変換途中のRunを置き、画面切替やbrowser接続なしで検出ログが出て、有限長マスタの再構築が完了する。
6. 優先Runへ継続追記しながら複数workspaceの背景Runを処理し、処理可能な3:1配分と背景対象の継続進捗を確認する。
7. 背景の巨大Run処理中に選択Run・workspaceを変更し、入力resourceを確保できる優先対象へ安全な処理境界で戻る。gzip stream枠が飽和する場合は待機理由がログへ出て、枠の割当方針に従って処理を開始する。stream枠待ちを含む待機時間も記録する。
8. gzip変換中にworkspaceを往復し、世代が変わらず、commit済み点数・位置が後退しない。選択していない間も背景枠で進む。
9. 同名Runを複数workspaceに置き、SQL内容・世代・進捗・優先度・ログが混ざらない。
10. 検出後のsource差替え、Run除去、workspace除去、再投入で誤った書込や消失directoryの再作成が起きず、不要resourceが解放される。
11. 開始済みgzip stream数・buffer・SQL batchの上限を確認する。完了・失敗・shutdown後にhandleが残らず、開始済み変換を切替だけで捨てない。
12. 検証済みで属性不変のready/error Runでは、定期走査によるSQLite connection・fingerprint・本文読取・DB更新が発生しない。
13. ログは検査・開始・進捗・完了・失敗の対象を識別できる。30秒未満の進捗連発、rollback分の進捗公開、EOF前の完了、変化なしの完了再送がない。
14. MetricsIngestor、MetricsCache、IngestScheduler、WorkspaceManager/WorkspaceSnapshot、MetricsQueryConcurrencyの既存回帰を基礎に、選択中API・query cancel・表示内容を維持する。UIに更新状況の追加がない。
15. 同じ有限長マスタの複製・同じJVM条件・同じ保存factor4で、変更前後の全再構築を比較する。rawとgzipを分け、工程別時間、総時間、行/秒・点/秒、heap peak、DB容量、優先Runの反映待ち時間を記録する。
16. 本PRDの完了には、代表的な大規模マスタで再構築時間が短縮したことと、背景処理併用で優先Runの待機を長いblockへ閉じ込めないことの実測が必要である。複数回のばらつき・改善率を記録し、小規模fixtureの成功を実Run性能の証明に代えない。

実装計画ではbaseline取得条件、代表マスタ、処理予算、resource上限と性能判定の数値を変更前の測定後に明記する。現時点で根拠のない速度倍率を受入値として確約しない。性能測定中・batch_run中という利用者の実行禁止指示がある場合は、負荷を伴う検証を開始しない。

## 6. 範囲と簡素化の記録

| 要素 | 裁定 | 根拠 |
| --- | --- | --- |
| バッファ入力と上限付きSQL batch | keep | 再構築が長いという実際の待ち時間と、byte/SQL単位の処理構造 |
| 全workspaceの検査と背景配分 | keep | 開くまで版更新に気付かない現行挙動 |
| 作業と表示snapshotの寿命分離 | keep | gzipが切替でstreamを失い、再構築する現行条件 |
| INFOの変換状況ログ | keep | 利用者がUI追加の代わりに明示要求 |
| 更新状況UI・一時停止/再開UI・新API | cut | 利用者が案⑤を不要と判断 |
| 旧DBのmigration、既存L0からの版変換 | cut | マスタから再構築する既存方針を維持 |
| 複数writer・新永続ストア・汎用ジョブ基盤 | cut | 単一writerとprocess内配分で今回の要求を満たせる |
| JSON streaming parse | defer | parseが支配的と実測された場合にだけ検討 |
| process再起動後のgzip途中再開 | defer | 今回保証するのは表示workspace切替時の継続であり、再起動時の保存形式変更は別課題 |
| ユーザー向け調整設定の増設 | defer | 内部予算では要求を満たせない運用上の根拠が得られた場合に検討 |

最小解は既存のReader・writerを高速化し、キャッシュ保守を全workspaceへ広げ、作業の寿命を表示切替から分離して、確定後にログを出すことである。ログ専用の永続状態や新しい表示機構は要らない。

入力・batch高速化は単独でも価値を持つ。全workspaceの配分とgzip寿命変更は一体で検証し、途中のworkspace切替で進捗を失う状態を完成としない。全体の完了は第5節の正確性・公平性・実測性能で判断する。

## 7. 範囲外と関連資料

範囲外はC++のMetrics出力変更、Metricsマスタ形式変更、過去artifactの加工、保存factorの再選定、表示粒度の変更、旧キャッシュのmigration、複数process間のwriter調停、新しいUI/API、コミット・push、稼働中Viewerの停止・差し替えとする。起票時は設計のみを対象とし、後続の承認済み実装計画によりコード・文書・隔離検証の実施が許可された。実Runのキャッシュ再構築は行わない。

- [MetricsViewer設計](../design/210_metrics_viewer.jp.md)
- [分析ユーザーガイド](../design/030_user_guide_analysis.jp.md)
- [PRD 089: 比較粒度と保存factor4](089_metricsviewer_comparison_resolution_10prd.md)
- [PRD 064: 定常取り込みI/O抑制](done/064_metricsviewer_idle_ingest_io_10prd.md)
- [ADR 0015: 破棄可能な従属導出物](../adr/0015-metrics-cache-disposable-derivative.md)
- [ADR 0052: 共通Step区間と保存factor4](../adr/0052-metrics-display-uses-common-step-bins-and-factor4-storage.md)

2026-10-12の会話で、利用者は取り込み高速化、全workspaceの軽量検査と背景再構築、gzip変換作業の寿命分離を含む方針でのPRD作成を依頼した。更新状況UIは不要とし、ログ出力を要求した。正式採番は利用者の連番指定に従い、既存正式番号089の次の090とする。
