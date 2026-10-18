# PRD 090 実装計画

## 1. 概要と正本

2026-10-12に承認された計画。キャッシュ取り込みを高速化し、未選択workspaceも自動検査・背景再構築する。gzip変換はworkspace切替で継続し、状況を英語ログへ出す。DB書き込み失敗は該当Runだけを停止する。本ファイルを実装の正本とする。

未決事項監査のブロッカーは0。既存のSQL transaction、schema 2、保存factor4、query epochはrepo evidenceで確認済み。Run単位停止はユーザー決定。以下の予算・上限は承認済み内部既定値。migration、JSON parser変更、複数writer、再起動後gzip途中再開は範囲外。

## 2. 主な変更

- 入力を64KiB単位で読み完全行を抽出する。rawは確定offsetから再開し、gzipは展開bufferと未消費部分をstreamと保持する。UTF-8・CRLF・未終端行・gzip終端検証を維持する。
- L0・LOD・非scalar行を各4,096件までSQL batch化する。非scalar行は保留文字列1MiBでもflushし、単独の長い行はそのまま処理する。正常終了時はbatch、TagStats、source位置を同一transactionで確定する。例外時のcloseではflushせずrollbackする。
- 既存IngestSchedulerをprocess単位へ移し、取り込み・gzip stream・検証済み観測・ログ集計を所有させる。Run identityとWARN抑止キーは正規化した絶対パス。
- WorkspaceManager snapshotはquery・表示cacheを管理する。切替で旧queryをcancelし、取り込みresourceは閉じない。優先更新はworkspace epochとパス集合で渡し、旧epochの遅着要求は反映しない。
- 起動直後と10秒周期で全workspaceを検査する。区間は最大128Run／50ms、未完了走査は次区間へ継続。cache接続は初回・属性変更時だけ。旧版では旧tableを読まず再構築待ちへ登録。削除・完全検証は作業開始時。
- 処理可能なblockを優先3：背景1で配分し、背景はworkspace間で巡回。blockは最大100万行／1秒、完全行境界で早期commit。優先変更・切替・shutdownでwriterを起こし早期終了要求。
- DB失敗はrollback・stream解放後のsource/cache観測と理由をprocess内に保持。属性変化・対象再投入・Viewer再起動まで同じ失敗を再試行しない。他Runと閲覧は継続。予期しない内部例外はwriter停止。
- 列挙・初期化・commit・単独の長い行はsoft時間予算を超え得る。工程別に計測して記録。

## 3. Resourceとログ

| 項目 | 値・挙動 |
| --- | --- |
| gzip stream全体 | 16本まで |
| 新規背景gzip | 非優先の開始済みstreamが4本未満のとき開始 |
| 選択gzip | 全体上限まで開始可能、切替で非優先になっても継続 |
| 上限到達 | 新規作業を待機、理由を状態遷移時にINFO |
| 進捗 | Runごと30秒以上空け、次のcommit後に出力 |
| resource解放 | 完了・失敗・対象消失・shutdown |

開始・進捗・完了にworkspace、Run、世代。行数・有効点数はscope=since_startで確定値を出す。経過時間と実処理時間を区別。gzip率は概算、EOF検証・最終commit前に100%や完了を出さない。正常版更新INFO、読取・書込失敗ERROR。通常追記を再構築と記録せず、無変化検査や同じ失敗をspamしない。

## 4. テスト

Public surfaceは既存workspace/metrics API、IngestScheduler、SourceReader、MetricsIngestorの取り込み結果、永続cache、ログ、shutdown。private構造に依存せず外から観測する。各behaviorを1テストのRED→最小GREENで進める。

1. browserなしで未選択workspaceの旧cacheを検出・再構築し、既存閲覧APIから正しい結果。
2. 同名Run分離、3：1配分、背景継続。
3. gzip変換中workspace往復で世代・確定点数維持、上限待機・解放。
4. buffer/batch境界、tag隔離、破損、rollbackのL0/LOD/TagStats整合。
5. DB失敗Run停止、他Run継続、属性変更後再処理。
6. fake clockでログ間隔・commit後公開・完了・重複抑止。
7. 属性不変ready/error RunでDB接続・fingerprint・本文読取・更新なし。
8. workspace切替、query cancel、shutdown、既存ブラウザ回帰。

## 5. 検証

変更前作業ツリーを隔離ビルドしてbaselineを保存。停止済みマスタ `apps/runner/workspaces/DropMerge-02/runs/run_20261006-033243_dm_fb_munch_tau3e-2_ndtpen20_nstep3_100m/metrics.jsonl.gz` （534,402,924bytes）を検証領域へ複製し同内容rawを生成。同じJVM条件・factor4・各3回で比較。raw/gzip中央値がbaselineより短いこと、出力一致、入力resource確保済み選択Runの反映待ちp95<=10秒を受入条件とする。ばらつき・heap peak・DB容量・stream待ちを併記。試行DBは一つずつ処理し検証領域内だけで解放。

関連Javaテスト→全Maven test/package→Playwright。通常JAR、実Run cache、稼働中Viewerは変更しない。C++変更なしなのでDebugビルド対象外。手順・結果は `core/anet-core/testdata/prd090/README.md` に記録する。

## 6. 前提・文書追随

UI・HTTP形式・設定キー・schema 2・保存factor4を維持。MetricsViewer設計と分析ガイドの日英版、CONTEXTのRun作業セット説明を更新。背景gzip変換中のフォルダ移動制約を明記。PRD090へRun単位停止と内部既定値を反映する。PRD089を含む既存未コミット変更を保持。commit・push・稼働Viewer差し替えを行わない。

## 7. 実装時の補足

選択Runの属性確認はblock境界で行い、優先集合が空でないidle待ちは最大1秒とした。背景作業が枯渇しても、選択Runへの通常追記の反映待ちを10秒の全体検査周期に依存させないため。全workspace検査周期は前回全走査完了から10秒を維持する。同じ優先集合の再要求ではwake・block中断・検査の前倒しを行わない。

現在のblockの完了と待ち行列全体の枯渇を分け、別の処理可能Runが残る場合はidleへ戻らない。gzip枠待ちだけが残る場合はbusy loopを避ける。個別workspaceの列挙失敗は直前の一覧を保持し、他workspaceの検査を続ける。

軽量検査は旧版tableを解釈しない。現行schemaでも必要table/columnの欠損は再構築へ登録し、SQLite資源・書込失敗はRun単位停止へ渡す。同サイズ・mtimeのファイル置換はfileKey、creationTime、完全精度mtimeで区別し、source種別/pathの変更も属性比較に含める。

大容量候補の最初のraw試行は1087.835秒、うちcommitは687.159秒だった。単純なbuffer/batch化だけでは、1秒blockのcommit負担が高速化を相殺しているため、page spill抑制を狙って、書込接続のpage cache目標を16MiB（`PRAGMA cache_size=-16384`）へ固定した。接続単位のnative memoryであり、heap peakには含まれない。schema・設定キー・read接続は維持する。この候補は最終比較へ含めず、調整後のソースを凍結し直して全テストと各3試行を取り直す。

## 8. 検証記録

2026-10-12 21:57:05、最終ソースでMaven packageが成功。全237 test、failure/error/skipはいずれも0、Playwrightを含み3分28秒。通常JARは隔離POMで保持し、完了後に一時POMを除去した。

変更前のraw/gzip各3試行は全table内容が一致し、中央値はraw1033.862971秒、gzip1704.031628秒。最終ソースの各3試行と選択Run反映待ちの集計はPRD固有READMEとperformance-results.jsonへ記録する。

選択Run反映待ちの最初の20 sampleはp95=11,302.7262msで受入条件を超えた。cycle先頭だけの属性probeでは、追記後に背景blockが3つ続くため。各block境界で優先Runの属性を調べ、変化したRunだけcycle内のexhaustedを解除する。3:1のslot配分、128Run/50ms検査上限、全workspace周期は維持する。cycle途中の追記を独立回帰でRED→GREENとして確認した。

この修正はschedulerだけで、単独MetricsIngestorを呼ぶ再構築ベンチの実行コードは不変。測定時のsource/class manifestを保持し、変更差分とclass一致を検証して再構築各3試行を保持する。修正後snapshotでp95を再測定し、全体回帰も成功した。

入力が変換中gzipから優先rawへ切り替わるときは、旧decoderの枠とhandleを取り込み前に解放する。実streamを用いた回帰で解放漏れをRED→GREENとして修正した。workspace切替だけでは解放しない契約は維持する。

最終比較はraw中央値688.271364秒（baseline1033.862971秒から33.4%短縮）、gzip中央値760.164143秒（baseline1704.031628秒から55.4%短縮）。raw/gzip双方で各3試行の中央値短縮と、全12試行の5table全row一致を確認した。DB容量は全試行2,337,267,712bytes。ばらつき・heap peak・工程別時間・入力hashはPRD固有READMEとperformance-results.jsonへ保存した。

選択raw 1Runと背景gzip 3Runの20sampleで、反映待ちp95は1,343.0929ms、最大1,353.1645ms、選択Runのstream枠待ちは0ms。入力resource確保済みの受入母集団で10秒以内を満たす。画面描画・gzip枠待ちまでの測定ではない。再構築を測った実行class 103個は最後のscheduler修正後も一致し、source/class manifestで証拠を保持した。
