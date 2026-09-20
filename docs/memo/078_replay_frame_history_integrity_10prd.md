# ReplayBuffer の観測履歴・bootstrap stack 整合性修復 PRD

起点: 2026-09-20。ReplayBuffer整合性アッセイと公開APIの最小再現で検出した2件を、共通の境界管理の修復としてまとめる。観測事実、コードから説明できる原因、未測定の学習影響、決定した対応を区別する。

契約・責務・受入条件は本書で確定している。内部の型名・メンバ名だけは実装計画で決める。

## 0. 決定一覧（グリル確定値）

| ID | 決定 |
|---|---|
| D1 | 境界の源は入力`BatchState::episode_start`とする。done/truncationを伴わない`episode_start`は正規入力ではなく契約違反として扱う。Env継ぎ目の契約（Resetは全laneの`state.episode_start == true`、Stepは`continue_state.episode_start == (done \|\| truncated)`。[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、[Env設計書 §2](../design/120_environments.jp.md)、`ValidateEpisodeStructure`）をReplayBufferも前提にする。[PRD050](done/050_replay_ring_stack_margin_10prd.md) D15以来未裁定だった「doneを伴わないepisode_start」の期待失敗2件は、契約外入力として決着する |
| D2 | frame stackの境界の基準はSTART（履歴開始）とする。過去方向の走査には過去側の信号を使う。slot単位に持つ境界情報は履歴開始1件だけで、END（done/truncated）はn-step queueの一時値のままslot単位には保持しない。`terminals_` / `actual_n_steps_`はn-step出力としてだけ残し、境界判定に使わない。lane状態からSTARTを導出する方式は採らない |
| D3 | `DefaultReplayBuffer::Push`は整合検証を行う。規則は「laneの初回の実pushは`episode_start == true`、以降は次の実pushの`episode_start`が直前の実pushの`done \|\| truncated`と一致、dummyは対象外」。全laneをstorageへ書く前にpreflightで検証し（`UpdatePriorities`と同じ流儀）、違反は`ANET_SYSTEM_ERROR`（lane、logical index、期待値、実値を含む）。同一slotのSTARTとENDは独立な2事実で、長さ1のepisodeは正常入力。Prefetch経由のwrite-behind Pushで起きた違反は既存契約どおり次の同期境界で再送出される |
| D4 | 履歴開始フラグは`ReplayExperienceStorage`が所有し、実観測のPushで引数の値を、`PushTerminalDummy`でfalseを、観測と同じ書込みで書いて旧世代を置換する。検証用のlane状態（直前の実pushがENDだったか、初期値true）は`DefaultReplayBuffer::Push`が照合・更新する。形式はCPU常駐のslot当たり1 byte相当の配列とlane当たり1 boolで、Tensorである必要はない。extractorは履歴開始だけを新→旧に走査し、最初のtrueで止める。`next_obs`は`L = t + actual_n`から同じ規則で復元する。`stack_count == 1`は走査しない |
| D5 | 全条件matrixは`[.][integrity_assay]`のhiddenテストにし、`[replay_buffer]`タグは付けない。Catch2は正のフィルタに一致すればhiddenも実行するため、通常タグを付けると`"[replay_buffer]"`指定で走る。最小再現3本と§7.3の追加テストは`[replay_buffer][frame_stack][history_start]`を共通タグにして可視のまま置く。CI/CDの既定スイートにmatrixは含めない |
| D6 | matrixはstack `{1,2,4}` × n_step `{1,2,3,5}` × lane数 `{1,4,16,128}` × 実lane容量 `{17,31}` × Uniform/PER × direct/CPU Prefetchの384条件へ拡張する。全条件の全検査地点完走が受入条件。狙い撃ちの単体テストも追加する（§7.3） |
| D7 | アッセイランナー`core/anet-core/testdata/prd078/run_integrity_assay.py`（`.venv`のPython、標準ライブラリのみ）を追加する。caseごとに`anet-core-test.exe "[integrity_assay]" -c "case N" --rng-seed <seed>`を別プロセスで順次実行し、失敗しても続行する。seed（既定`20260919`）はmatrixのTEST_CASEが`Catch::getSeed()`で読んでReplayBufferの抽選とunique probeに使い、最小再現と単体テストは固定seedのまま。case当たりの時間上限（既定300秒）の超過は失敗として記録する。`.scratch/prd078/<timestamp>/`にcase別ログ、`results.csv`、`report.md`を書く。実行手順は`core/anet-core/testdata/prd078/README.md`に置き、設計書150 §8から参照する |
| D8 | `terminals_` / `actual_n_steps_`の初期値（true / 0）は現状維持。既存テスト「ReplayExperienceStorage initializes unwritten slots as episode boundaries」は名前と目的を「Storage metadataの初期値」に限定し、「未書込みslotの履歴開始は立っていない（境界は書込みでだけ付く）」を足す。`DefaultExperienceBuilder`の`sequence.back().is_dummy`分岐は到達不能で注記のみ（スコープ外）。`DumpToLog`に履歴開始を出力する |
| D9 | [ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)を新設し、[ADR 0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)の「安全な開始indexならextractorは変更しない」判断を名指しで更新する（history margin自体は維持）。`CONTEXT.md`に用語「履歴開始」を追加する。設計書150と実装コメントの更新は実装と同じ変更で行う |

## 1. 問題と目的（Problem Statement / Solution）

ReplayBufferから返るTensorのshape、replay item key、n-step長が正しくても、観測stackの一部が別時刻の観測や過剰なpaddingに置き換わる。学習側はこれを正常な経験として受け取り、誤った状態を学習に使う可能性がある。

| ID | 観測された問題 | 確認した主条件 | 修復の目的 |
|---|---|---|---|
| A | 初回満杯時、`obs`の起動時paddingへ末尾の新しい観測が混入する | stack=4、n_step=1 | laneの最初の観測を履歴開始として保存し、物理ring末尾を過去として読まない |
| B | `next_obs`の履歴を実エピソード境界と誤認し、最新frameで過剰にpaddingする | stack=4、n_step=3/5、wrap前およびwrap後 | 観測の履歴開始をn-step結果の確定状態・前世代のmetadataから独立させる |

公開サンプルが、入力された同一lane・同一エピソードの履歴に一致することを保証する。履歴不足のpaddingが許されるのはepisode由来の不足だけであり、ring上書きで失われた履歴は既存のsampleable range契約で除外する。サンプル集合を余分に狭めて不具合を隠す対応も採らない。

両方の欠陥の共通根源は、履歴開始を表す一次情報がslotに無く、extractorが`terminal && actual_n_steps <= 1`という代理指標から境界を推定していることである。Aは代理指標の消失、Bは代理指標の遅延と残骸であり、引き金は別だが穴は同じである。修復は「Push時に確定する履歴開始をslotへ書き、extractorはそれだけを読む」の1機構で両方を閉じる。

## 2. 調査結果と確度

### 2.1 整合性アッセイ

Debugで、stack `{1,4}` × n_step `{1,3,5}` × lane数 `{1,4,16,128}` × 実lane容量 `{17,31}` × Uniform/PER × direct/CPU Prefetch の192条件を実行した。入力履歴は乱数を使わない固定の生成規則による。ReplayBufferの抽選とunique probeのseedは`20260919`。

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
- 確定: 現行の書込み・n-step確定・境界判定のコードは各再現結果を説明する（§3.2、§4.2）。原因を除去した修正による緑化はまだ確認していない。
- 確定: production経路では、Runnerが`continue_state`を次stepの`state`として`Push`へ渡し、`ValidateEpisodeStructure`が毎stepで`continue_state.episode_start == (done || truncated)`とResetの`episode_start == true`を検証している。したがって「done/truncationを伴わないepisode_start」はproductionからReplayBufferへ到達せず、直接APIを叩くテストだけが作れる。
- 未測定: 実験Runのスコア、TD誤差分布、学習安定性への寄与量。既存の成績差をこの不具合だけの結果とは断定しない。
- 未検証: CUDA転送経路、matrix外の全構成。stack=1の96条件成功も、全機能・全入力の無欠陥証明ではない。

## 3. 不具合A: 初回満杯時の起動時padding汚染

### 3.1 観測された現象・再現条件

最小再現は単一lane、容量8、stack=4、n_step=1、Uniform、direct。観測を時刻そのものとし、done/truncatedを入れずに0〜7をpushする。最初の入力だけepisode_startとする。`SampleUniqueUniform`で全7件を取得し、key=8（slot 0、第1世代）を調べる。

| 項目 | 期待値 | 実値 |
|---|---|---|
| 時刻0の`obs` stack | `[0,0,0,0]` | `[5,6,7,0]` |

matrixではstack=4 / n_step=1の28条件で同系統の不一致が出た。代表例はcase 101/102、実lane容量31、lane数1、Uniform、direct/CPU Prefetch、pushed=27、key=31で、期待時刻0に対し25が返る。入力件数27でも、truncationの終端観測dummyを含む書込み位置は31に達する。入力時刻を物理slotと同一視してはならない。

### 3.2 原因

初回満杯ちょうどでは保持最古の論理位置が0なので、history marginは0のままであり、時刻0付近も正当なsample候補に残る。一方、extractorは負の履歴位置を物理ring末尾へ折り返して探索する。

初回満杯より前は、末尾に残る未書込みslotの初期値`terminal=true / actual_n_steps=0`が偶然に起動境界として働く。満杯時には末尾も書込み済みになるため、この暗黙の境界が消え、新しい観測を時刻0より前の履歴として拾う。

発現条件を正確に書くと「write cursorが実lane容量とちょうど等しく、かつ末尾の`stack_count - 1` slotがすべてn-step確定済みの通常遷移」である。n_step≧2では末尾slotのn-step結果が未確定で初期値が残るため偶然の境界が生き残り、matrixではn_step=1でしか顕在化しなかった。これは偶然であり、Bだけを直すと（たとえば未確定slotを境界と見なさなくすると）Aは全n_stepへ広がる。AとBは同時に修復しなければならない。

Aは`next_obs`にも及ぶ。時刻0、n=1、stack=4なら`next_obs`のframeは論理位置-2〜1で、同じ折り返しが起きる。アッセイは最初の不一致で止まるため、記録には`obs`しか残っていない。

問題は起動境界を物理slotの未書込み状態へ依存させている点にある。時刻0をsample不可にすることや、初回満杯でhistory marginを追加することは、正当にpaddingできる遷移を排除する回避策になる。

### 3.3 影響範囲

- 汚染は初回満杯境界の`obs`と`next_obs`。開始時刻付近の遷移をその状態で抽出すると、現在状態または bootstrap 状態の表現に未来のframeが混入する。
- Uniform/PER、direct/CPU Prefetch、複数laneで発生した。Prefetch固有の競合を必要としない。
- 各laneの書込み位置、実境界の位置、dummy挿入により、同じpush回数でも発生可否が異なる。常にすべてのlane・サンプルが壊れるわけではない。
- 毎周回同じ汚染が起きるわけではない。wrap後は既存のhistory marginが最古側を保護する。
- 誤った現在状態が価値・方策表現、TD誤差、学習更新へ影響する可能性があるが、実験上の影響量は未測定。

### 3.4 対応

D2/D4により、laneの最初の観測は履歴開始としてslotに保存される。extractorは時刻tから新→旧に走査し、履歴開始で止まるため、論理位置0より前に到達することがなく、物理ring末尾への折り返しは起きない。物理位置からlogical indexを復元する必要もない。

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

### 4.2 原因

1. storageは観測・actionを即時に書くが、そのslotの`terminal`と`actual_n_steps`はn-step結果が確定してから更新する。
2. 未確定期間には、初回なら初期値`terminal=true / actual_n_steps=0`、wrap後なら前世代の値が残る。
3. extractorは`terminal && actual_n_steps <= 1`をframe stackの境界として使う。この判定では、現在の観測が書込み済みでも、その観測から始まるn-step結果が未確定なslotを、実終端・dummy・未書込み領域と誤認する。
4. sample対象の開始遷移は確定済みでも、bootstrap stack内の後続遷移のn-step結果まで確定済みとは限らない。そのslotを境界と判定すると、正当な過去frameを切り捨て、後ろのframeで埋め直す。

B3では、物理slot 2にあった時刻2の終端metadataが、時刻10の通常観測への上書き後にも残る。時刻10自身のn-step結果が未確定な間、その旧終端を時刻11の直前の境界として読んでしまう。

必要条件を書くと次のとおり。slot jのn-step結果が確定するのは`j <= cursor - n_step`のとき、tがreadyなのは`t <= cursor - n_step - 1`のときなので、bootstrap末尾`L = t + actual_n`は`cursor - 1`まで届く。走査対象は`k <= L - 1`で、kが未確定になるのは`k > cursor - n_step`のときだから、**Bはn_step≧3かつstack≧2で起きる。** n_step=2では未確定slotは走査対象外のL自身だけで、Bは原理的に起きない。`obs`側にも起きない。`k < t`は queue のFIFO確定により常に確定済みで、history marginにより現世代だからである。

### 4.3 影響範囲

- 非terminalサンプルのbootstrap入力が誤るため、bootstrap価値、Bellmanターゲット、TD誤差、Learner優先度更新へ波及する可能性がある。
- n_step=3/5、Uniform/PER、direct/CPU Prefetchで観測した。n_step=2は原理的に非該当だが、否定制御として検証対象に含める。
- wrap前の初期値だけでなく、wrap後の旧世代の実終端等が原因となり得るため、運用中にも繰り返し起こる可能性がある。頻度は未測定。
- `next_obs`の末尾frameやkeyが正しくても、中間frameが正しいとは限らない。報酬和やn-step長だけの検査では検出できない。
- stack=1は今回の96条件で成功した。Actor側のオンラインstackまで壊れること、laneをまたぐ混入、メモリ破壊は確認していない。
- CUDAへ転送される前の共通抽出処理には関係するが、CUDA転送経路自体の検証済みという意味ではない。

### 4.4 対応

D2/D3/D4により、履歴開始はPush時に入力の`episode_start`から確定し、n-step確定処理はtarget return、terminal、実n-step数だけを完成させる。extractorは`terminal`と`actual_n_steps`を境界に使わないので、未確定期間や旧世代の値は境界へ影響しない。

採らない案と理由:

| 案 | 却下理由 |
|---|---|
| Push時に`terminal=false`（または`terminal=done, actual_n_steps=1`の暫定値）を書く | Bは消えるが、境界がn-step出力の派生述語のまま残り、Builderやunrollの変更で再び壊れ得る。Aは別途必要になる |
| 後続遷移のn-step結果がすべて確定するまでsampleを遅らせる | 既存のready/sampleable契約を余分に狭め、episode開始直後の経験が遅れて供給される |
| lane状態からSTARTを導出する（Dopamine型） | 1 byte/slotは同じ費用で、1つずらしとlane先頭の初期状態という導出がextractor側に残る。Actor stackerと別信号になり、入力`episode_start`との不整合を検出できない |
| `episode_start`とdone/dummyの和集合を境界にする | 源が2つになり、done無しの`episode_start`でstackだけ切れてn-stepは切れない非対称が残る。冗長性を検証に使えない |
| done無しの`episode_start`を正規入力として受理し、n-stepも切る | productionではEnv継ぎ目のfail-fastで到達不能な入力のための新機構になる。直前遷移をterminal扱いするか破棄するかの裁定も要る |
| sample結果へgenerationを運びextractorでassertする | ADR 0024で保留した防御。今回の欠陥は同世代の未確定metadataでも起きるため世代検証では防げない |

## 5. 利用者の要求（User Stories）

1. 学習実験の実行者として、起動直後から正しい観測stackを受け取り、誤った現在状態で学習しないようにしたい。
2. 学習実験の実行者として、n-stepのbootstrap先に正しい時系列を受け取り、正しいBellmanターゲットを計算したい。
3. 学習実験の実行者として、長時間のring周回後も同じ整合性を保ち、旧世代の終端情報を使わないようにしたい。
4. 複数環境を使う実行者として、laneごとに異なる終端と書込み位置を独立に扱ってほしい。
5. PER利用者として、Uniformと同じ正しい経験を抽出し、その経験に対応する優先度を更新したい。
6. Prefetch利用者として、先読み時点の履歴として整合した経験を受け取りたい。
7. 環境実装者として、doneとtruncationそれぞれの終端観測・bootstrap契約が維持されてほしい。
8. ReplayBuffer保守者として、履歴開始（lane先頭とepisode先頭）によるpaddingと、ring上書きによる除外（history margin）を別の条件として説明・検証できるようにしたい。
9. テスト保守者として、本体の復元処理を使わない正解と全frameを比較し、同じバグをテスト側にも複製しないようにしたい。
10. 障害調査者として、1条件の失敗でmatrix全体を打ち切らず、未被覆を成功扱いしない結果を得たい。
11. Agent実装者として、学習時のstackがActorの`StackerActionContext`と同じ信号（`episode_start`）から作られ、行動時と学習時で境界の定義が食い違わないようにしたい。
12. Env実装者として、`episode_start`とdone/truncatedの不整合をReplayBufferが無音で飲み込まず、Push時点で即時に検出してほしい。

## 6. 共通の設計要求と対応方針（Implementation Decisions）

### 6.1 境界・世代の不変条件

| 状況 | 必要な扱い |
|---|---|
| laneの最初の観測 | 履歴開始。入力`episode_start == true`が契約。先頭paddingはこの観測で埋める |
| 同一episodeの通常観測 | 履歴開始ではない。n-step未確定でも境界にしない |
| doneで終わる遷移の観測 | 履歴開始ではない。その観測は終了episodeに属し、次の実pushの観測が履歴開始になる |
| truncationの終端観測dummy | 履歴開始ではない。終了episodeの最後の観測としてbootstrap stackへ含める |
| done/truncation後の次episodeの観測 | 履歴開始。入力`episode_start == true`が契約。前episodeをstackに混ぜない |
| 同一slotのSTARTとEND | 独立な2事実。長さ1のepisodeはSTARTかつENDであり正常 |
| slotの上書き | 観測と履歴開始を同じPushで置き換える。前世代の値を残さない。n-step結果を境界判定へ使わない |
| ring上書きで必要な過去frameが消失 | sampleable rangeから除外する（history margin）。paddingで救済しない |
| done/truncationを伴わない`episode_start`、ENDの次が`episode_start`でない | 契約違反。Pushがfail-fastする |

`obs`とbootstrap対象の`next_obs`は同じ境界規則で組み立てる。truncationは公開サンプルに専用フラグがないため、terminal=false、実n-step短縮、正しい終端観測を含むstackで外部契約を確認する。true terminalの`next_obs`は次episodeの先頭frameのpaddingになり決定的だが、値保証は設けない（学習に使わない）。

### 6.2 モジュールの責務

| モジュール | 変更・責務 |
|---|---|
| `DefaultReplayBuffer::Push`（書込み経路） | 入力の`state.episode_start`、`next_state.done`、`next_state.truncated`をlaneごとに読む。storageへ書く前に全laneをpreflightし、lane状態（初期値true）と`episode_start`の不一致を`ANET_SYSTEM_ERROR`で止める。実観測のPushに履歴開始を渡し、dummy挿入後も含めてlane状態を`done \|\| truncated`へ更新する |
| `ReplayExperienceStorage` | slot単位の履歴開始を所有する。実観測のPushで引数の値、`PushTerminalDummy`でfalseを、観測と同じ書込みで書く。`DumpToLog`に履歴開始を出す |
| `ExperienceSampleExtractor` | 履歴開始だけから stack を復元する。`terminal` / `actual_n_steps`を境界として解釈しない |
| `ValidIndexManager` | ready range / sampleable range / history margin / dummy除外の既存契約を維持する |
| n-step queue / builder | 報酬和・terminal・実n-step数を確定する。境界情報の源にはしない |
| `PrefetchingReplayBuffer` | 変更しない。write-behind Pushで起きたfail-fastは既存契約どおり次の同期境界で再送出する |

extractorの走査規則: 開始slot tに対し`k = t, t-1, …, t - stack_count + 1`の順に履歴開始を調べ、最初にtrueとなったkをstackの先頭にする。見つからなければ`t - stack_count + 1`が先頭。先頭より前の不足分は先頭frameの複製で埋める（既存のpadding規則）。`next_obs`は`L = t + actual_n`を開始slotとして同じ規則を適用する。走査範囲はhistory marginにより常に保持中の現世代slotだけであり、lane先頭の履歴開始は範囲内に必ず現れるため、負の論理位置や物理折り返しを扱う必要がない。

Push検証の規則: laneごとに`expected = lane_expects_start`（初期値true）、`actual = state.episode_start[lane]`。`expected != actual`なら、lane、write cursor（logical index）、期待値、実値を含めて`ANET_SYSTEM_ERROR`。全laneの検証をどのlaneの書込みよりも前に終える。検証後、実観測を履歴開始`actual`で書き、truncatedならdummy（履歴開始false）を書き、`lane_expects_start = done \|\| truncated`とする。dummyは検証の対象にならない。

追加メモリはslot当たり1 byte相当とlane当たり1 boolに収める。

| 項目 | slot当たり |
|---|---|
| 既存metadata（target_return f32、terminal、actual_n_steps i64、generation i64、sampled_once） | 22 byte超 |
| PER追加（SumTree double 2本分、source） | 約17 byte |
| 履歴開始 | 1 byte |
| 参考: Atari 84×84 uint8観測 | 7,056 byte |

Atari 1M slotで約1 MiB。全観測のstack複製、無制限の履歴保持、追加の画像Tensorコピーは導入しない。Pushにはslot当たり1 byteの履歴開始の書込みに加え、全laneの事前検証とlane状態の更新が増える。Sampleの走査は2つのaccessor読みから1 byte読みへ置き換わる。既存のProfileRangeを維持し、Push/Sampleの時間を同条件で変更前後に測定・比較して報告する。固定の性能劣化率を合否基準にはせず、明確な悪化が出た場合は原因を調査する。

境界計算は内部の小さな責務として集約する。新しい公開サブシステムや汎用フレームワークは要求しない。公開ReplayBuffer APIのシグネチャ、設定キー、replay item keyの符号化、sampling分布、`Size()`の意味、PER初期優先度完成・eviction統計のready基準は変更しない。Pushが`state.episode_start`を必須入力として読む点だけが入力契約の追加である。

### 6.3 既存設計との関係

[ADR 0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)の「上書き履歴は除外し、episode由来の不足だけpaddingする」判断は維持する。同ADRの「安全な開始indexならextractorは正しく動くためextractorは変更しない」という判断は、AとBの再現により成立しない。**本件ではextractor側の境界復元も修復対象へ含める。** これはhistory marginの撤回ではなく、当時の検証範囲を超えた不具合への対応であり、[ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)に記録した。

[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)が定めるEnv継ぎ目の構造契約（Resetの`episode_start`、`continue_state.episode_start == (done || truncated)`）をReplayBufferの入力前提とし、D3の検証はその契約をPush側で確認するものである。契約自体は変更しない。

[ADR 0011](../adr/0011-generation-aware-replay-item-key.md)のkey世代・slot identity、[ADR 0005](../adr/0005-sample-prefetch-stale-per.md)のstale samplingとwrite-behind順序は維持する。先読み済みの古いbatchが生成時点の履歴に正しく対応することは正常であり、今回のフレーム汚染と混同しない。

一般実装との比較（2026-09-20時点のソースで確認）:

| 実装 | 構造 | 境界の作り方 |
|---|---|---|
| DeepMind DQN `dqn/TransitionTable.lua` | 単一frame保存、sample時復元 | `add`でterminalを即時書込み、`concatFrames`が過去へ走査してterminalを越えたframeをzero-out。1-stepなのでENDはPush時に確定している |
| Dopamine v2 `dopamine/replay_memory/circular_replay_buffer.py` | 単一frame保存、sample時復元、n-stepあり | `add()`で直前がterminalなら`stack_size - 1`個のzero transitionを先に挿入。n-step returnはsample時に即時terminalから計算する |
| Dopamine JAX `dopamine/jax/replay_memory/accumulator.py` | storage前に解決 | per-episodeのdequeで stack と n-step を作り、`is_terminal` / `episode_end`でclear |
| Gymnasium `gymnasium/wrappers/stateful_observation.py` `FrameStackObservation` | Actor側wrapper | `reset()`だけで再初期化。stepではterminated/truncatedを見ない |
| SB3 `stable_baselines3/common/vec_env/stacked_observations.py` | vec env autoreset | 同stepの`dones`でstackをクリアしてreset観測を書く |

共通則は「境界は書込みまたはresetの時点で確定した情報から作る」であり、後から確定する派生値を境界に使う実装はない。anetはn-step導入で`terminal`が派生値になった時点でこの前提が崩れていた。明示的なSTART信号を持つvec APIは少なく、持たない実装はENDから導出するかzero slotで実体化している。1 byte/slotの履歴開始はこれらの中で最も安い実体化である。

## 7. テスト方針・受け入れ基準（Testing Decisions）

### 7.1 最小再現の緑化

- A: 容量8、stack=4、n_step=1、0〜7 push後のkey=8について、`obs=[0,0,0,0]`になる。既存のsampleable件数7を保つ。
- B1/B2: 容量32の終端なし入力で、`next_obs`がそれぞれ`[0,1,2,3]` / `[2,3,4,5]`になる。`obs`、terminal、n-step長も従来の期待値に一致する。
- B3: 容量8のwrap後、key=16で`obs=[5,6,7,8]`、`next_obs=[8,9,10,11]`、terminal=false、n_steps=3となる。
- 3本のタグは`[replay_buffer][frame_stack][history_start]`に個別タグ（`[initial_fill]`、`[pending_metadata]`、`[wrapped_metadata]`）を加えたものにし、`[integrity_assay]`は外す。再現テストを削除・skip・期待失敗化せず、期待値も実装出力へ合わせて変更しない。

### 7.2 全matrixの再検査

384条件（D6）すべてが全検査地点まで完走すること。受入実行では抽選・probeのseedに`20260919`を使用する。失敗した条件だけの再実行では完了としない。

1. 元の入力履歴をring形式ではない正解として用い、本体のstack/n-step復元処理を流用しない。入力生成器は`episode_start`をepisode先頭の入力にだけ立て、done/truncatedと整合する契約どおりの参照実装として維持する。
2. laneごとに長さ`{1,2,3,4,5,7}`のepisodeをずらして巡回し、done/truncationを交互に配置する。各lane容量の5倍まで入力し、初期充填、初回wrap前後、各周回後を検査する。
3. 各地点でpushを止め、全sampleable要素をunique probeで検査する。通常`Sample()`も全対象を引くまで照合し、抽出上限到達は未被覆として失敗させる。
4. 全frame、履歴開始のpadding、bootstrap対象`next_obs`、割引報酬和、実n-step数、terminal、keyのslot/世代を確認する。dummyを含む実書込み位置と実容量でkeyを計算する。
5. Prefetchは最初の抽出後もpushを続け、write-behindを通す。同期後も残る先読み済みbatchは生成時点の履歴で検査する。
6. 失敗時は設定、seed、lane、入力時刻、key、対象が`obs`か`next_obs`か、stack内のframe位置、期待値・実値を出す。条件ごとの実行結果と未完了地点を区別する。

matrixのTEST_CASEは`[.][integrity_assay]`だけをタグに持つ（D5）。実行はD7のランナーで行う。

ランナーの仕様:

- 位置と実行系: `core/anet-core/testdata/prd078/run_integrity_assay.py`。`.venv`のPythonで実行し、標準ライブラリだけを使う。
- 引数: テスト実行体のパス（既定`core/anet-core/bin/Debug/anet-core-test.exe`）、対象case（既定は全384。範囲または列挙で再開できる）、出力先（既定`.scratch/prd078/<timestamp>/`）、seed（既定`20260919`。Catch2の`--rng-seed`として渡し、matrixのTEST_CASEが`Catch::getSeed()`で読み取ってReplayBufferの抽選とunique probeへ適用する）、case当たりの時間上限（既定300秒、引数で変更可能。超過は失敗扱い）。
- 動作: caseごとに`<exe> "[integrity_assay]" -c "case N" --rng-seed <seed>`を1プロセスで実行し、stdout/stderrを`case-NNN.log`へ、終了コードと秒を`results.csv`へ書く。失敗しても次のcaseへ続行する。
- seedの適用と記録: `--rng-seed`をCatch2へ渡すだけで済ませず、matrixのTEST_CASEはCatch2の実行seedを`Catch::getSeed()`（uint32。`20260919`は収まる）で読み、ReplayBufferの抽選seedとunique probeのRNGに使う。切り替える対象はhiddenのmatrixだけで、最小再現3本と§7.3の単体テストは固定seedのままにする。Catch2は`--rng-seed`未指定だと乱数でseedを決めるため、ランナーを介さずmatrixを直接実行した場合の再現には出力先頭の`Randomness seeded to:`の値を使う。入力履歴の生成規則は固定のままとし、seedを変えても観測・報酬・episode境界の配置は変えない。実際に抽選・probeへ適用したseedをcase別ログと`report.md`に記録する。
- 集計: Catch2出力の標記行（`Replay integrity passed: case=… snapshots=… covered_keys=… checked_samples=… seconds=…`、`Replay integrity snapshot complete: pushed=…`）から、成功/失敗数、失敗caseの最後に完了した検査地点と失敗した検査地点、合計時間、検査地点数、延べkey数、照合サンプル数を`report.md`に書く。失敗が1件でもあればランナーの終了コードを非0にする。
- 未完走を成功扱いしない。時間上限超過、クラッシュ、標記行の欠落はすべて失敗として数える。時間上限を超えたcaseは打ち切り、失敗を記録して次のcaseへ進む。300秒はhangによる全体停止を防ぐための上限であり、性能の合格基準ではない。

### 7.3 原因に対応する境界追加検査

すべて`[replay_buffer][frame_stack][history_start]`を共通タグとする単体テストで、公開サンプルの挙動を正解に置く。

- 初回容量到達の直前・ちょうど・直後をlane単位で検査し、dummyにより入力回数と書込み位置がずれるケースを含める。`obs`と`next_obs`の両方を見る。
- 観測書込み済み・n-step未確定の期間を直接検査する。n_step=2を否定制御（Bが起きないことの確認）とし、stack=2/3の小さい代表値も補う。
- wrap先の旧slotが実終端、dummy、通常遷移だった各ケースを作り、旧世代の種類で新世代のstackが変わらないことを確認する。
- episode長がstack/n-stepより短い場合、truncation直前の履歴と終端観測を保ち、その後のreset観測を混ぜないことを確認する。
- 長さ1のepisode（STARTとENDが同一slot）をdoneとtruncationの両方で受理し、`obs`、`next_obs`、n-step長、terminalが期待どおりであることを確認する。
- 同一論理履歴のサンプルを後続push前後に再検査し、必要な観測を上書きしていない間、n-step metadataの確定だけでstackが変化しないことを確認する。
- 契約違反のfail-fast 3方向: 初回pushの`episode_start=false`、done/truncatedの次のpushの`episode_start=false`、done/truncatedを伴わない`episode_start=true`。いずれも`Push`が例外で止まり、preflightにより当該Pushの書込みが起きていないことを確認する。Prefetch経由の再送出は既存契約のままなので専用テストは要求しない。
- 既存の`[!shouldfail]` 2件（`n-step returns stop at episode_start without done`、`frame stacking starts a new stack at episode_start without done`）は削除し、上記のfail-fastテストに置き換える。
- 初回pushの`episode_start`を省略していた既存テストを移行する。replay_buffer_test.cppの6件（`sampled indices are valid sampleable storage indices`、`reads a caller-random unique uniform probe batch without replacement`、`sampling history returns requested disjoint groups from one snapshot`、`history probes preserve ring generations and sample reconstruction`、`caller-owned probe RNG is deterministic and isolated`、`samples while push and priority update run concurrently`）とdqn_munchausen_test.cppの3件（`Replay fit TD and quantile losses match independent fixed-distribution oracles`、`Replay fit preserves stochastic training and uses fixed eval distributions`、`DQN target evaluation preserves the PRD073 pre-change baseline`）。いずれも初回pushの`episode_start`をtrueにするだけで、期待値は変えない。
- D8のとおり、`ReplayExperienceStorage initializes unwritten slots as episode boundaries`は名前と目的を「metadataの初期値」に限定し、未書込みslotの履歴開始が立っていないことを足す。
- 既存のstack_keys、stack=1、ring history margin、unroll、PER、probe、key世代のテストに退行がないことを確認する。unrollの契約自体は拡張しない。

テストの正解は公開サンプルの挙動に置き、内部metadata配列の形や具体的なbit配置へ固定しない。

### 7.4 実行と完了条件

VsDevCmd経由のDebugビルド後、次の順に確認する。

1. `anet-core-test.exe "[history_start]"`で最小再現3本と§7.3の追加テストが緑。
2. D7のランナーで384条件を実行し、全条件が全検査地点まで完走。
3. 引数なしの既定スイート全体（hidden除外、CI相当）で失敗0件、期待失敗0件。テスト一覧とタグを確認し、matrixが引数なしの既定スイートと`"[replay_buffer]"`指定の対象に含まれないことを確認する。所要時間からの推測では判定しない。

新規・既存の予期しない失敗は0件とする。修正により後続地点で別の不整合が出た場合は、原因と対応範囲を追記し、matrix未完走のまま完了扱いにしない。

履歴開始の保存、全laneの事前検証、stack復元、既存テストの移行は一体で完成させる。ランナー整備は独立して進められるが、片方だけを受入としない。サンプル候補を余分に減らしていないことは、matrixの各検査地点で`Size()`が契約から導いた期待集合の件数と一致すること、および§7.1の件数維持で確認する。学習スコアの改善は要求しない。

実測時間、検査地点数、延べ被覆key数、照合サンプル数、メモリ（実行者の観測値）と追加metadata量を報告する。加えて、§6.2のPush/Sampleの変更前後比較を報告する。実行時間を数秒へ収めることは合格条件にしない。条件を順次実行し、大量の履歴や複数プロセスを同時保持しない。性能測定中のRunがある場合は、その終了確認後にビルド・実行する。

実装と同じ変更で更新する文書:

- [ReplayBuffer設計書](../design/150_replay_buffer.jp.md): §2.2の`Push`行に`state.episode_start`の読取りと整合検証を追記。§2.3の「起動直後の未書込領域または保存済みterminalによる実episode境界をpadding」を「Push時に保存した履歴開始（`BatchState::episode_start`）より前を先頭frameでpadding」へ書き換え、「`DefaultReplayBuffer::Push`は`episode_start`を保存・参照しない」段落を削除して整合検証の契約に置き換える。§3のStorage行に履歴開始の所有を追記。§7.3のエラー一覧にPushの契約違反を追加。§8のテスト一覧に整合性アッセイ（hidden）とランナーを追加。
- `core/anet-core/src/replay_buffer_impl.cpp`先頭の「[設計仕様]」コメント（エピソード開始時のpadding）とextractorの境界コメントを履歴開始へ書き換える。
- `core/anet-core/testdata/prd078/README.md`にランナーの実行手順を置き、150 §8から参照する。
- `CONTEXT.md`の用語「履歴開始」と[ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)は本書作成時に追加済み。

## 8. スコープ外（Out of Scope）

- Env継ぎ目の構造契約（ADR 0034）自体の変更、Actor側`StackerActionContext`の変更。
- CUDA転送・CUDAイベント管理の修正と新規検証。共通抽出の修復を転送経路の全面保証とは扱わない。
- Learnerの目的関数、PERアルゴリズム、Actor初期優先度推定式の変更。
- 公開API・設定・保存データ契約の追加、stale sampling廃止、N-deep prefetch。
- 全観測stackの保存方式への変更、広範なReplayBuffer再設計。
- metadata幅の最適化（int64の`actual_n_steps_`等）。
- sample結果へのgeneration運搬とextractor側assert（ADR 0024で保留のまま）。
- 実験Runのスコア差の因果確定や過去Run artifactの書換え。
- §9の修正済みshutdown問題の再実装、および原因未確定のheap assertionをA/Bと同一原因と決めつけること。

## 9. 調査中に見つかった関連事項（Further Notes）

### 9.1 PinnedThreadPool停止時の通知取りこぼし

整合性アッセイのCPU Prefetch破棄時に停止した。想定原因は、停止predicateの確認からwaitに入る間へ通知が割り込み、workerが停止を再確認せず待機する競合。影響は破棄・停止処理のhangであり、観測stackの誤値とは別問題。

停止通知を各workerの待機mutexと同期させる修正と回帰テストは、既にこの調査の作業ツリーへ追加済み。修正前は30秒で打ち切り、修正後は単独6回と既存ReplayBuffer・threadの合同回帰で完了を確認した。本PRDではA/Bの修復前提・非退行対象として扱う。

### 9.2 Debug CRT heap assertionの単発観測

shutdown修正直後の1回で`_CrtIsValidHeapPointer` / `is_block_type_valid`を観測した。同じseedの後続6回と合同回帰、追加91条件では同じ症状を確認していない。想定原因は特定できておらず、影響範囲も未確定。shutdown修正で解消したとも、ReplayBufferのA/Bが原因とも扱わない。再発時は独立した再現条件・stack・実行ログを採取して対応範囲を裁定する。

### 9.3 到達不能なBuilder分岐

`DefaultExperienceBuilder::Build`の`sequence.back().is_dummy`分岐は到達しない。dummyは必ずtruncatedレコードの直後にあり、`NStepQueueController`はそのtruncatedを先に終端として系列を切るためである。本PRDでは変更せず注記に留める。

### 9.4 再グリルでの簡素化判断（2026-09-20）

目的の軸は、A/Bを一体で修復し、正当なサンプル候補を余分に減らさず、入力と一致する観測履歴を返すことである。D1〜D9の中核判断は維持し、次の項目を判定した。再グリルで前提を補正した箇所は、seedの用途（入力生成ではなく抽選・probe）とPushの追加コスト（書込みだけでなく検証・lane状態更新も含む）の2点で、境界管理の設計は変えていない。

| 判定 | 対象 | 理由・再検討条件 |
|---|---|---|
| 維持 | slotごとの履歴開始 | 削ると、起動境界の消失とn-step未確定metadataへの依存が残り、A/Bを修復できない |
| 維持 | laneごとの入力整合検証 | stackを切るepisode_startとn-stepを切るdone/truncatedの一致をPush境界で保証する |
| 維持 | PRD078専用ランナー | 最初の失敗後に残り91条件を別途実行した作業を再現可能にし、全条件の完走を確認する |
| 限定 | 可変seedとtimeout | seedは抽選・probeだけへ、hiddenのmatrixだけに適用する。入力履歴は固定し、timeoutは既定300秒の実行オプションに収める |
| 保留 | ランダム入力履歴の生成 | 固定境界の384条件で捉えられない実際の不具合・検証不足が生じた時点で再検討する |
| 保留 | 汎用テスト実行基盤 | 他の検査でも同じ実行・集計処理の重複が実際に負担になった時点で再検討する |
| 保留 | 性能専用基盤 | 既存のProfileRangeと同条件比較では悪化の判定・原因調査ができない時点で再検討する |
| 対象外を維持 | 学習スコア改善の立証、CUDA経路の全面検証、ReplayBuffer全体の再設計 | 今回の観測履歴修復の受入に必要な範囲を超える |

## 10. 根拠・参照

本書の最小再現・結果表は、ローカルログが失われても現象を追えるよう本文に記録した。`.scratch`内のログは補助証跡であり、PRDの成立をその永続保持へ依存させない。

- [ReplayBuffer設計書](../design/150_replay_buffer.jp.md): 現行の公開サンプル、sampleable range、frame stacking、Prefetchの契約。
- [Env設計書](../design/120_environments.jp.md) §2: Reset/Stepの`episode_start`構造契約。
- [PRD050](done/050_replay_ring_stack_margin_10prd.md): 上書き履歴の除外と起動・実境界paddingの区別。
- [ADR 0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、[ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)。
- [ドメイン用語](../../CONTEXT.md): ready range、sampleable range、history margin、履歴開始、slot index、replay item key、target return。
- [ReplayBuffer実装](../../core/anet-core/src/replay_buffer_impl.cpp): `ReplayExperienceStorage::Push` / `Update` / `PushTerminalDummy`、`ExperienceSampleExtractor`の実装。2026-09-20時点の初期値487〜488行付近、書込み・確定495〜545行付近、境界判定1168行付近、stack復元1195〜1222行付近、`DefaultReplayBuffer::Push` 1283〜1376行付近。
- [ReplayBuffer内部宣言](../../core/anet-core/src/replay_buffer_impl.hpp): `ForEachSampleableIndex`のhistory margin適用。
- [Env構造契約の検証](../../core/anet-core/src/env.cpp): `ValidateEpisodeStructure`（112〜175行付近）。Runner側の呼び出しは`trainer.cpp`の各step・reset経路。
- [Catch2のhidden判定](../../third_party/catch2/src/catch.cpp): `TestSpec::Filter::matches`（1901行付近）。正のフィルタに一致するhiddenテストは実行される。
- [公開API回帰テスト](../../core/anet-core/src/replay_buffer_test.cpp): `[integrity_assay]`、`[initial_fill]`、`[pending_metadata]`、`[wrapped_metadata]`。本書作成時点ではA/Bの再現は失敗する。
- [継続結果レポート](../../.scratch/rb-integrity-assay/remaining-20260920-041125/report.md)、[条件別終了コード・時間](../../.scratch/rb-integrity-assay/remaining-20260920-041125/results.csv)、[Bの最小再現ログ](../../.scratch/rb-integrity-assay/remaining-20260920-041125/pending-metadata-repro-final.log)。
- [調査全体の検証記録](../../.scratch/rb-integrity-assay/verification.md): 初回実行、Aの最小再現、shutdown修正、既存テスト結果、heap assertionの記録。
