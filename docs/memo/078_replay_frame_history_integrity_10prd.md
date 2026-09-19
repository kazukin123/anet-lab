# ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD

起点: 2026-09-20。ReplayBuffer整合性アッセイと公開APIの最小再現で検出した2件を、共通の境界管理の修復としてまとめる。観測事実、コードから説明できる想定原因、未測定の学習影響、想定対応案を区別する。

本書は要求と対応案のドラフトである。境界情報の具体的な保持形式・内部APIは実装計画で確定する。採番は[PRD運用規約](README.md)に従い、実装着手時に正式番号へ変更する。

## 1. 問題と目的（Problem Statement / Solution）

ReplayBufferから返るTensorのshape、replay item key、n-step長が正しくても、観測stackの一部が別時刻の観測や過剰なpaddingに置き換わる。学習側はこれを正常な経験として受け取り、誤った状態を学習に使う可能性がある。

| ID | 観測された問題 | 確認した主条件 | 修復の目的 |
|---|---|---|---|
| A | 初回満杯時、`obs`の起動時paddingへ末尾の新しい観測が混入する | stack=4、n_step=1 | 起動時の論理的な履歴不足を正しくpaddingし、物理ring末尾を過去として読まない |
| B | `next_obs`の履歴を実エピソード境界と誤認し、最新frameで過剰にpaddingする | stack=4、n_step=3/5、wrap前およびwrap後 | 観測の実境界をn-step結果の確定状態・前世代のmetadataから独立させる |

公開サンプルが、入力された同一lane・同一エピソードの履歴に一致することを保証する。ring上書きで失われた履歴をpaddingで補うことはせず、既存のsampleable range契約で除外する。サンプル集合を余分に狭めて不具合を隠す対応も採らない。

## 2. 調査結果と確度

### 2.1 整合性アッセイ

Debugで、stack `{1,4}` × n_step `{1,3,5}` × lane数 `{1,4,16,128}` × 実lane容量 `{17,31}` × Uniform/PER × direct/CPU Prefetch の192条件を実行した。入力生成seedは`20260919`。

| stack | n_step | 成功条件 | 失敗条件 |
|---|---|---:|---:|
| 1 | 1 / 3 / 5 | 96 | 0 |
| 4 | 1 | 4 | 28 |
| 4 | 3 | 8 | 24 |
| 4 | 5 | 12 | 20 |
| 合計 | | 120 | 72 |

初回実行は100条件成功後の101番目で失敗した。残り91条件を独立実行した結果は20成功・71失敗。追加91条件のプロセス実行時間合計は586.032秒で、成功20条件だけの集計は180検査地点、延べ7,308 key、49,185サンプル照合だった。単一プロセスずつ実行し、メモリ観測例はworking set 519 MiB / private memory 931 MiB。全プロセスの生涯最大メモリは採取していない。

**失敗条件は最初の不一致で打ち切られている。** 全192条件を起動・終了したことと、全検査地点の照合完了は異なる。完走は120条件であり、失敗72条件の後続検査地点には別の不整合が隠れている可能性がある。上表は不具合の発生率や学習影響の大きさを表さない。

### 2.2 確定事項と未確定事項

- 確定: Aの`obs`汚染、Bの`next_obs`過剰padding、Bのwrap後再発を公開APIで再現した。
- 原因の確度が高い: 現行の書込み・n-step確定・境界判定のコードは、各再現結果を説明する。ただし、原因を除去した修正による緑化はまだ確認していない。
- 未測定: 実験Runのスコア、TD誤差分布、学習安定性への寄与量。既存の成績差をこの不具合だけの結果とは断定しない。
- 未検証: CUDA転送経路、matrix外の全構成。stack=1の96条件成功も、全機能・全入力の無欠陥証明ではない。

## 3. 不具合A: 初回満杯時の起動時padding汚染

### 3.1 観測された現象・再現条件

最小再現は単一lane、容量8、stack=4、n_step=1、Uniform、direct。観測を時刻そのものとし、done/truncatedを入れずに0〜7をpushする。最初の入力だけepisode_startとする。`SampleUniqueUniform`で全7件を取得し、key=8（slot 0、第1世代）を調べる。

| 項目 | 期待値 | 実値 |
|---|---|---|
| 時刻0の`obs` stack | `[0,0,0,0]` | `[5,6,7,0]` |

matrixではstack=4 / n_step=1の28条件で同系統の不一致が出た。代表例はcase 101/102、実lane容量31、lane数1、Uniform、direct/CPU Prefetch、pushed=27、key=31で、期待時刻0に対し25が返る。入力件数27でも、truncationの終端観測dummyを含む書込み位置は31に達する。入力時刻を物理slotと同一視してはならない。

### 3.2 想定原因

初回満杯ちょうどでは保持最古の論理位置が0なので、history marginは0のままであり、時刻0付近も正当なsample候補に残る。一方、extractorは負の履歴位置を物理ring末尾へ折り返して探索する。

初回満杯より前は、末尾に残る未書込みslotの初期terminal値が偶然に起動境界として働く。満杯時には末尾も書込み済みになるため、この暗黙の境界が消え、新しい観測を時刻0より前の履歴として拾う。

したがって、問題は起動境界を物理slotの未書込み状態へ依存させている点にある。時刻0をsample不可にすることや、初回満杯でhistory marginを追加することは、正当にpaddingできる遷移を排除する回避策になる。

### 3.3 想定影響範囲

- 確認済みの汚染は初回満杯境界の`obs`。開始時刻付近の遷移をその状態で抽出すると、現在状態の表現に未来のframeが混入する。
- Uniform/PER、direct/CPU Prefetch、複数laneで発生した。Prefetch固有の競合を必要としない。
- 各laneの書込み位置、実境界の位置、dummy挿入により、同じpush回数でも発生可否が異なる。常にすべてのlane・サンプルが壊れるわけではない。
- この再現だけから毎周回同じ`obs`汚染が起きるとは言えない。wrap後は既存のhistory marginが最古側を保護する。
- 誤った現在状態が価値・方策表現、TD誤差、学習更新へ影響する可能性があるが、実験上の影響量は未測定。

### 3.4 想定対応案

laneの起動境界を論理的に判定し、論理位置0より前の履歴要求は最初のframeでpaddingする。物理位置への剰余変換に先立って起動境界を扱うか、同等の意味を保持する境界metadataを用いる。

推奨はBと共通の境界復元処理にまとめ、起動境界・実エピソード境界・保持範囲を別々に判定すること。wrap後の上書き欠損は既存のhistory marginで除外し、起動paddingと混同しない。

## 4. 不具合B: 未確定metadataによるbootstrap stackの過剰padding

### 4.1 観測された現象・再現条件

単一lane、stack=4、Uniform、directで以下を再現した。各サンプルの`obs`、実n-step数、terminal=falseの検査は成功し、`next_obs`の検査が失敗する。

| 再現 | 入力 | 対象 | `next_obs`期待値 | 実値 |
|---|---|---|---|---|
| B1 | 容量32、n_step=3、終端なし、0〜3をpush | 時刻0 | `[0,1,2,3]` | `[3,3,3,3]` |
| B2 | 容量32、n_step=5、終端なし、0〜5をpush | 時刻0 | `[2,3,4,5]` | `[5,5,5,5]` |
| B3 | 容量8、n_step=3、0〜11をpush。時刻2だけdone、時刻3でepisode_start | 時刻8、key=16（slot 0、第2世代） | `[8,9,10,11]` | `[11,11,11,11]` |

B3の`obs`は期待どおり`[5,6,7,8]`。初期充填だけでなく、wrap後にも再現する。B1/B2はdone/truncatedなしで成立し、小容量や複数laneを必要としない。

matrixではn_step=3の24条件、n_step=5の20条件で同系統のフレーム不一致を検出した。case 129はlane容量17、lane数1、pushed=14、key=29、期待時刻10に対し13。case 149はlane容量31、lane数16で、pushed=4の最初の検査地点から発生した。

### 4.2 想定原因

1. storageは観測・actionを即時に書くが、そのslotの`terminal`と`actual_n_steps`はn-step結果が確定してから更新する。
2. 未確定期間には、初回なら初期値`terminal=true / actual_n_steps=0`、wrap後なら前世代の値が残る。
3. extractorは`terminal && actual_n_steps <= 1`をframe stackの境界として使う。この判定では、現在の観測が書込み済みでも、その観測から始まるn-step結果が未確定なslotを、実終端・dummy・未書込み領域と誤認し得る。
4. sample対象の開始遷移は確定済みでも、bootstrap stack内の後続遷移のn-step結果まで確定済みとは限らない。そのslotを境界と判定すると、正当な過去frameを切り捨て、後ろのframeで埋め直す。

B3では、物理slot 2にあった時刻2の終端metadataが、時刻10の通常観測への上書き後にも残る。時刻10自身のn-step結果が未確定な間、その旧終端を時刻11の直前の境界として読んでしまう。

### 4.3 想定影響範囲

- 非terminalサンプルのbootstrap入力が誤るため、bootstrap価値、Bellmanターゲット、TD誤差、Learner優先度更新へ波及する可能性がある。
- n_step=3/5、Uniform/PER、direct/CPU Prefetchで観測した。n_step=2等は今回のmatrix外だが、原因からは検証対象に含める必要がある。
- wrap前の初期値だけでなく、wrap後の旧世代の実終端等が原因となり得るため、運用中にも繰り返し起こる可能性がある。頻度は未測定。
- `next_obs`の末尾frameやkeyが正しくても、中間frameが正しいとは限らない。報酬和やn-step長だけの検査では検出できない。
- stack=1は今回の96条件で成功した。Actor側のオンラインstackまで壊れること、laneをまたぐ混入、メモリ破壊は確認していない。
- CUDAへ転送される前の共通抽出処理には関係するが、CUDA転送経路自体の検証済みという意味ではない。

### 4.4 想定対応案

観測の実エピソード境界を、n-step結果とは別の情報として書込み時点で確定させる。物理slot再利用時には観測とその境界情報を同じ書込みで置き換え、旧世代の値を残さない。n-step確定処理はtarget return、terminal、実n-step数だけを完成させ、観測の境界の意味を変えない。

境界の表現は、観測の直前が開始境界かを示すslot単位の情報、または同等のエピソード識別・開始位置情報を候補とする。いずれも実境界と起動境界を復元でき、現在世代の観測と対応することを要件とする。具体的な型・メンバ名はここでは固定しない。

`terminal`をPush時にfalseへ戻すだけの対応は、n-step結果と観測境界の兼用を残し、実終端・dummy・起動時の扱いを再び暗黙化するため採らない。後続遷移のn-step結果がすべて確定するまでsampleを遅らせる案も、既存のready/sampleable契約を余分に狭めるため採らない。

## 5. 利用者の要求（User Stories）

1. 学習実験の実行者として、起動直後から正しい観測stackを受け取り、誤った現在状態で学習しないようにしたい。
2. 学習実験の実行者として、n-stepのbootstrap先に正しい時系列を受け取り、正しいBellmanターゲットを計算したい。
3. 学習実験の実行者として、長時間のring周回後も同じ整合性を保ち、旧世代の終端情報を使わないようにしたい。
4. 複数環境を使う実行者として、laneごとに異なる終端と書込み位置を独立に扱ってほしい。
5. PER利用者として、Uniformと同じ正しい経験を抽出し、その経験に対応する優先度を更新したい。
6. Prefetch利用者として、先読み時点の履歴として整合した経験を受け取りたい。
7. 環境実装者として、doneとtruncationそれぞれの終端観測・bootstrap契約が維持されてほしい。
8. ReplayBuffer保守者として、起動padding、実境界padding、上書きによる除外を別の条件として説明・検証できるようにしたい。
9. テスト保守者として、本体の復元処理を使わない正解と全frameを比較し、同じバグをテスト側にも複製しないようにしたい。
10. 障害調査者として、1条件の失敗でmatrix全体を打ち切らず、未被覆を成功扱いしない結果を得たい。

## 6. 共通の設計要求と対応方針（Implementation Decisions）

### 6.1 境界・世代の不変条件

| 状況 | 必要な扱い |
|---|---|
| laneの最初の観測 | 起動境界として確定する。必要な先頭paddingはこの観測で埋める |
| 同一episodeの通常観測 | n-step未確定でも実境界にしない |
| doneで終わる遷移の観測 | その観測は終了episodeに属する。次episodeの観測の直前を境界とする |
| truncationの終端観測dummy | 終了episodeの最後の観測としてbootstrap stackへ含める。dummy直前でそのepisodeの履歴を切らない |
| done/truncation後の次episodeの観測 | 新しい開始境界とし、前episodeをstackに混ぜない |
| slotの上書き | 観測と境界情報を同じ世代へ置き換える。前世代のn-step結果を境界判定へ使わない |
| ring上書きで必要な過去frameが消失 | sampleable rangeから除外する。paddingで救済しない |

`obs`とbootstrap対象の`next_obs`は同じ境界規則で組み立てる。truncationは公開サンプルに専用フラグがないため、terminal=false、実n-step短縮、正しい終端観測を含むstackで外部契約を確認する。true terminalの`next_obs`に新たな値保証は設けない。

### 6.2 モジュールの責務

| モジュール | 想定する変更・責務 |
|---|---|
| DefaultReplayBuffer / 書込み経路 | 入力の実終端とdummy挿入から観測の境界を即時に確定し、既存のstorage排他境界内で渡す |
| ReplayExperienceStorage | 観測と現在世代の境界情報を所有し、slot再利用時に一緒に置き換える |
| ExperienceSampleExtractor | 起動位置・実境界・保持範囲からstackを復元する。n-step結果を観測境界として解釈しない |
| ValidIndexManager | ready range / sampleable range / history margin / dummy除外の既存契約を維持する |
| n-step queue / builder | 報酬和・terminal・実n-step数を確定する。観測境界の遅延確定源にはしない |
| PrefetchingReplayBuffer | 既存の1-deep先読みとwrite-behind順序を維持し、生成時点の整合したsnapshotを返す |

境界計算は内部の小さな責務として集約する。新しい公開サブシステムや汎用フレームワークは要求しない。公開ReplayBuffer API、設定キー、replay item keyの符号化、sampling分布、`Size()`の意味、PER初期優先度完成・eviction統計のready基準は変更しない。

追加metadataはslot数に比例する小さな固定幅情報とlane単位の状態に抑える。全観測のstack複製、無制限の履歴保持、追加の画像Tensorコピーは導入しない。実装案で要素幅と実容量から追加メモリを算出し、既存ProfileRangeを維持してPush/Sampleの時間も確認する。

### 6.3 既存設計との関係

[ADR 0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)の「上書き履歴は除外し、episode由来の不足だけpaddingする」判断は維持する。一方、同ADRの「安全な開始indexならextractorは正しく動くためextractorは変更しない」という当時の判断は、AとBの再現により本件では成立しない。**本件ではextractor側の境界復元も修復対象へ含める。** これはhistory marginの撤回ではなく、当時の検証範囲を超えた不具合への対応である。実装時にこの判断の更新理由をADRへ追記または後続ADRに記録する。

[ADR 0011](../adr/0011-generation-aware-replay-item-key.md)のkey世代・slot identity、[ADR 0005](../adr/0005-sample-prefetch-stale-per.md)のstale samplingとwrite-behind順序は維持する。先読み済みの古いbatchが生成時点の履歴に正しく対応することは正常であり、今回のフレーム汚染と混同しない。

## 7. テスト方針・受け入れ基準（Testing Decisions）

### 7.1 最小再現の緑化

- A: 容量8、stack=4、n_step=1、0〜7 push後のkey=8について、`obs=[0,0,0,0]`になる。既存のsampleable件数7を保つ。
- B1/B2: 容量32の終端なし入力で、`next_obs`がそれぞれ`[0,1,2,3]` / `[2,3,4,5]`になる。`obs`、terminal、n-step長も従来の期待値に一致する。
- B3: 容量8のwrap後、key=16で`obs=[5,6,7,8]`、`next_obs=[8,9,10,11]`、terminal=false、n_steps=3となる。
- 再現テストを削除・skip・期待失敗化せず、期待値も実装出力へ合わせて変更しない。

### 7.2 全matrixの再検査

192条件すべてが全検査地点まで完走すること。失敗した72条件だけの再実行では完了としない。

1. 元の入力履歴をring形式ではない正解として用い、本体のstack/n-step復元処理を流用しない。
2. laneごとに長さ`{1,2,3,4,5,7}`のepisodeをずらして巡回し、done/truncationを交互に配置する。各lane容量の5倍まで入力し、初期充填、初回wrap前後、各周回後を検査する。
3. 各地点でpushを止め、全sampleable要素をunique probeで検査する。通常`Sample()`も全対象を引くまで照合し、抽出上限到達は未被覆として失敗させる。
4. 全frame、起動・実境界padding、bootstrap対象`next_obs`、割引報酬和、実n-step数、terminal、keyのslot/世代を確認する。dummyを含む実書込み位置と実容量でkeyを計算する。
5. Prefetchは最初の抽出後もpushを続け、write-behindを通す。同期後も残る先読み済みbatchは生成時点の履歴で検査する。
6. 失敗時は設定、seed、lane、入力時刻、key、対象が`obs`か`next_obs`か、stack内のframe位置、期待値・実値を出す。条件ごとの実行結果と未完了地点を区別する。

### 7.3 原因に対応する境界追加検査

- 初回容量到達の直前・ちょうど・直後をlane単位で検査し、dummyにより入力回数と書込み位置がずれるケースを含める。
- 観測書込み済み・n-step未確定の期間を直接検査する。n_step=2、stack=2/3等の小さい代表値も補い、matrixの値だけに依存する修正を避ける。
- wrap先の旧slotが実終端、dummy、通常遷移だった各ケースを作り、旧世代の種類で新世代のstackが変わらないことを確認する。
- episode長がstack/n-stepより短い場合、truncation直前の履歴と終端観測を保ち、その後のreset観測を混ぜないことを確認する。
- 同一論理履歴のサンプルを後続push前後に再検査し、必要な観測を上書きしていない間、n-step metadataの確定だけでstackが変化しないことを確認する。
- 既存のstack_keys、stack=1、ring history margin、unroll、PER、probe、key世代のテストに退行がないことを確認する。unrollの契約自体は拡張しない。

テストの正解は公開サンプルの挙動に置き、内部metadata配列の形や具体的なbit配置へ固定しない。

### 7.4 実行と完了条件

VsDevCmd経由のDebugビルド後、最小再現、`[replay_buffer][integrity_assay]`、既存`[replay_buffer]`の順に確認する。既存で仕様未裁定の「doneを伴わないepisode_start」2件の期待失敗を除き、新規・既存の予期しない失敗は0件とする。修正により後続地点で別の不整合が出た場合は、原因と対応範囲を追記し、matrix未完走のまま完了扱いにしない。

実測時間、検査地点数、延べ被覆key数、照合サンプル数、メモリと追加metadata量を報告する。実行時間を数秒へ収めることは合格条件にしない。条件を順次実行し、大量の履歴や複数プロセスを同時保持しない。性能測定中のRunがある場合は、その終了確認後にビルド・実行する。

修正に合わせ、現行ReplayBuffer設計書の境界復元説明と関連ADRを更新する。公開用語の意味を変更しない限り、用語集へ実装詳細を持ち込まない。

## 8. スコープ外（Out of Scope）

- done/truncationを伴わないepisode_startの仕様裁定。
- CUDA転送・CUDAイベント管理の修正と新規検証。共通抽出の修復を転送経路の全面保証とは扱わない。
- Learnerの目的関数、PERアルゴリズム、Actor初期優先度推定式の変更。
- 公開API・設定・保存データ契約の追加、stale sampling廃止、N-deep prefetch。
- 全観測stackの保存方式への変更、広範なReplayBuffer再設計。
- 実験Runのスコア差の因果確定や過去Run artifactの書換え。
- §9の修正済みshutdown問題の再実装、および原因未確定のheap assertionをA/Bと同一原因と決めつけること。

## 9. 調査中に見つかった関連事項（Further Notes）

### 9.1 PinnedThreadPool停止時の通知取りこぼし

整合性アッセイのCPU Prefetch破棄時に停止した。想定原因は、停止predicateの確認からwaitに入る間へ通知が割り込み、workerが停止を再確認せず待機する競合。影響は破棄・停止処理のhangであり、観測stackの誤値とは別問題。

停止通知を各workerの待機mutexと同期させる修正と回帰テストは、既にこの調査の作業ツリーへ追加済み。修正前は30秒で打ち切り、修正後は単独6回と既存ReplayBuffer・threadの合同回帰で完了を確認した。本PRDではA/Bの修復前提・非退行対象として扱う。

### 9.2 Debug CRT heap assertionの単発観測

shutdown修正直後の1回で`_CrtIsValidHeapPointer` / `is_block_type_valid`を観測した。同じseedの後続6回と合同回帰、追加91条件では同じ症状を確認していない。想定原因は特定できておらず、影響範囲も未確定。shutdown修正で解消したとも、ReplayBufferのA/Bが原因とも扱わない。再発時は独立した再現条件・stack・実行ログを採取して対応範囲を裁定する。

## 10. 根拠・参照

本書の最小再現・結果表は、ローカルログが失われても現象を追えるよう本文に記録した。`.scratch`内のログは補助証跡であり、PRDの成立をその永続保持へ依存させない。

- [ReplayBuffer設計書](../design/150_replay_buffer.jp.md): 現行の公開サンプル、sampleable range、frame stacking、Prefetchの契約。
- [PRD050](done/050_replay_ring_stack_margin_10prd.md): 上書き履歴の除外と起動・実境界paddingの区別。
- [ドメイン用語](../../CONTEXT.md): ready range、sampleable range、history margin、slot index、replay item key、target return。
- [ReplayBuffer実装](../../core/anet-core/src/replay_buffer_impl.cpp): `ReplayExperienceStorage::Push` / `Update` / `PushTerminalDummy`、`ExperienceSampleExtractor`の実装。2026-09-20時点の初期値487〜488行付近、書込み・確定495〜545行付近、境界判定1168行付近、stack復元1195〜1222行付近。
- [ReplayBuffer内部宣言](../../core/anet-core/src/replay_buffer_impl.hpp): `ForEachSampleableIndex`のhistory margin適用。
- [公開API回帰テスト](../../core/anet-core/src/replay_buffer_test.cpp): `[integrity_assay]`、`[initial_fill]`、`[pending_metadata]`、`[wrapped_metadata]`。本書作成時点ではA/Bの再現は失敗する。
- [継続結果レポート](../../.scratch/rb-integrity-assay/remaining-20260920-041125/report.md)、[条件別終了コード・時間](../../.scratch/rb-integrity-assay/remaining-20260920-041125/results.csv)、[Bの最小再現ログ](../../.scratch/rb-integrity-assay/remaining-20260920-041125/pending-metadata-repro-final.log)。
- [調査全体の検証記録](../../.scratch/rb-integrity-assay/verification.md): 初回実行、Aの最小再現、shutdown修正、既存テスト結果、heap assertionの記録。
