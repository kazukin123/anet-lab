# Episode Forensics（特殊挙動エピソード自動捕捉・事後調査）暫定 PRD

> 状態: 暫定メモ。長期 Run 中に発生する希少挙動を、発生後でも元エピソードまで遡って調査できる仕組みの要求と設計候補を記録する。
> 起点: `run_20260803-080518_apx-ll_cy05_b512-rr200-a5e5` で観測した一過性の `q_max_max` スパイク、および DropMerge の NEET 方策、長時間 NOOP、NoLegal 終端などの事後分析。
> 実装順・保存形式・既定閾値は未確定。本 PRD は実装着手を意味しない。

## Problem Statement

長期 Run では、スカラー指標のスパイク、NEET 方策、異常に長い NOOP 列、珍しい終端、急激な Q 値・TD 誤差・PER 優先度の変化など、短時間だけ現れる特徴的な挙動がある。

現状のメトリクスからは「いつ、何が起きたか」の一部は分かるが、その値を生んだ元エピソード、遷移、Observation、Action、報酬、終端理由、Actor/Learner の判断値まで遡れない場合がある。特に Learner が ReplayBuffer から過去の遷移を Sample した時点で異常を検出した場合、現在表示されている Env やエピソードとは時間的にも主体的にも一致しない。

実例として、上記 Run では `q_max_max = 105.069` が約 52.15M step で発生し、同時に real-space Q、TD 誤差、loss も一時的に上昇した。全体の恒常的な発散ではなく、希少な Observation または遷移が Replay で繰り返し Sample された可能性が高いが、Sample 元を識別する情報が保存されていないため、仮説以上には進められなかった。

長期 Run は実時間と計算資源の費用が大きい。特徴的な挙動を人間がリアルタイムで目撃して手動録画する運用に依存せず、該当エピソードを自動で捕捉し、Run 終了後でも詳細調査できる証拠を残したい。

## Solution

Run 内に **Episode Forensics** 機構を追加する。

Episode Forensics は、全エピソードを無制限に動画保存する仕組みではない。次の二段階で、通常時の負荷と調査可能性を両立する。

1. 全 Env lane の直近エピソードについて、Action、報酬、終端、時刻軸、識別子、Observation の要約または設定された診断 payload を、容量制限付きの **Episode Journal** に保持する。
2. Env、Actor、Learner、Replay、または利用者操作から特徴的な挙動を検出したら、そのエピソードと関連遷移を **Forensics Bundle** へ昇格し、Run artifact として永続化する。

検出時点が元エピソードより後でも追跡できるよう、Run 内で一意な `EpisodeId` と `TransitionId` を Experience、Replay、Sample まで伝播させる。検出理由、検出値、閾値、step 軸、Env instance/lane、Actor と Learner の model version、関連 Replay item を同じ Bundle に記録する。

最初の tracer bullet は、Learner の Q 値外れ値を検出し、最大値を出した batch 要素から元の DropMerge エピソードへ遡れる経路とする。その後、長時間 NOOP や NoLegal などのエピソード系列検出へ一般化する。

## User Stories

1. 長期 Run の実施者として、特徴的なエピソードを自動保存したい。リアルタイムで画面を監視し続けなくても希少事象を失わないためである。
2. Run 分析者として、メトリクス上の Q 値スパイクから最大値を生んだ Sample 要素を特定したい。batch 集約値だけでなく原因候補の遷移を確認するためである。
3. Run 分析者として、Learner で検出した Sample を元のエピソードへ紐付けたい。現在動いている Env の映像と過去の Replay 経験を混同しないためである。
4. DropMerge 開発者として、長時間 NOOP、NOOP Q margin の偏り、NoDrop timeout、NoLegal 終端などを検出条件にしたい。NEET 方策と正当な待機を具体的な系列で比較するためである。
5. DropMerge 開発者として、Double Suika、最大 rank 更新、極端に長い成功エピソードなどの良い希少事象も捕捉したい。異常だけでなく獲得技能も調査するためである。
6. Agent 開発者として、Q 値、TD 誤差、loss、PER priority、IS weight、gradient などを組み合わせた trigger を定義したい。単一指標では分からない機構上の連鎖を調べるためである。
7. Env 開発者として、終端理由や Env 固有の診断値を Bundle に追加したい。共通層へ Env 固有知識を埋め込まずに詳細を残すためである。
8. Run 分析者として、trigger より前の数十から数百 step も確認したい。異常が見えた瞬間ではなく、そこへ至る系列を調査するためである。
9. Run 分析者として、trigger 後からエピソード終端までの系列も確認したい。異常から回復したのか、方策が固定化したのかを判定するためである。
10. Run 分析者として、同じエピソードに複数 trigger が発生した場合は一つの Bundle で時系列に見たい。重複保存を避け、因果候補を一続きで読むためである。
11. Run 分析者として、証拠が完全か、一部が保持期限切れかを明示的に知りたい。不完全な記録を完全な再現と誤認しないためである。
12. Run 分析者として、train と eval、Env instance、lane、step 軸を区別したい。異なる母集団や時間軸を混ぜないためである。
13. Run 分析者として、行動時の Actor model version と、後から評価した Learner model version を区別したい。異なるネットワークの Q 値を同一判断として扱わないためである。
14. 利用者として、興味深い場面を手動で bookmark したい。自動 detector が未対応の挙動も同じ調査形式で保存するためである。
15. 利用者として、通常 Run では保存容量と追加負荷に上限を設定したい。Forensics 自体が学習 throughput やマシン安定性を損なわないためである。
16. 利用者として、Forensics を無効にした場合は学習結果と hot path が従来通りであってほしい。観測機構の有無を実験条件へ不用意に混入させないためである。
17. 実装者として、新しい detector を学習本体から独立して単体検証したい。検出条件を安全に追加・調整するためである。
18. 実装者として、保存失敗、queue 飽和、保持期限切れをメトリクスと WARN で確認したい。証拠が残っていない状態を黙って見逃さないためである。
19. 将来の Viewer 利用者として、Bundle 一覧から trigger 理由を絞り込み、エピソード系列を再生または表形式で確認したい。生データを手作業で結合せず調査するためである。
20. 実験比較者として、Run 間で同じ detector 定義と閾値を確認したい。自動検出件数を比較可能な補助指標として扱うためである。

## Implementation Decisions

以下は暫定判断であり、実装計画のレビュー時に確定する。

### 1. 永続的なエピソード・遷移識別子を導入する

- Env instance と lane を含む Run-scoped な `EpisodeId` を発番する。
- エピソード内の各遷移へ Run-scoped な `TransitionId` を付与する。
- 識別子は Experience から ReplayBuffer、`ExperienceSamples` まで値を変えずに伝播させる。
- PRD035 の generation 付き replay item key は、物理 slot 上書き後の誤更新を防ぐ一時的な Replay identity である。Episode Forensics の永続識別子とは目的を分け、相互に記録できるようにする。
- ID は Run artifact 内で一意とし、Run を跨いだ global uniqueness は要求しない。Run ID と組み合わせて識別する。

### 2. 容量制限付き Episode Journal を常時保持する

- 各 lane の active episode と直近完了 episode を、容量、件数、または時間 horizon で制限されたリング形式の Journal に保持する。
- 最小 payload は、識別子、step 軸、Action、報酬、terminal/truncated、終端理由、Observation fingerprint、Env/Actor の診断 scalar とする。
- Observation 本体、Env state、画像、動画は payload tier と保存予算に応じて選択可能にする。既定で全エピソードの無圧縮画像を保存しない。
- 各 Run は、どの fidelity tier で記録したかを明示する。要約のみの Bundle を完全な再生可能記録として扱わない。
- Learner trigger が発生する前に元履歴が期限切れになる可能性があるため、保持 horizon は Replay の滞留時間との関係を設定・メトリクスで確認可能にする。

### 3. Detector と保存処理を分離する

- Detector は入力されたイベントまたは診断値から、trigger reason、severity、対象 ID、値、閾値を含む小さな `ForensicsTrigger` を生成する深いモジュールとする。
- Detector はファイル I/O、UI、ReplayBuffer の内部構造を直接扱わない。
- trigger source は少なくとも Env/episode、Actor/action、Learner/update、Replay/sample、manual bookmark を区別する。
- 単純閾値、連続回数、移動統計からの外れ値、複数条件の AND/OR、短時間の cluster を表現できる余地を残す。ただし最初から汎用ルール言語は作らない。
- 「特殊」は必ずしも「エラー」を意味しない。Double Suika のような良い希少事象も同じ仕組みで扱い、検出だけで Run を fail-fast させない。

### 4. Run-level の Forensics Coordinator が証拠を昇格する

- Runner/Run 側の Coordinator が Episode Journal、trigger の集約、Bundle 昇格、保存 queue を管理する。
- Env、Actor、Learner は Coordinator の内部状態へ依存せず、既存イベントまたは狭い trigger sink を介して事実だけを通知する。
- 同一 `EpisodeId` に複数 trigger が来た場合、同じ Bundle へ追記し、trigger ごとの時刻と根拠を残す。
- trigger 時点で対象履歴が Journal から消えている場合は、利用可能な Sample/Experience と trigger 情報だけを保存し、`source_history_expired` 相当の不完全状態を明記する。黙って保存成功扱いにしない。
- active episode が trigger された場合、trigger 前の Journal を固定し、終端または設定された post-trigger step まで追記してから Bundle を確定する。

### 5. Forensics Bundle を Run artifact の独立した証拠とする

- `metrics.jsonl` は集約メトリクスの正本として維持し、サイズの大きいエピソード証拠を混在させない。
- Bundle は Run artifact 内の専用領域へ保存し、manifest、trigger 一覧、episode/transition series、任意の Observation/画像/動画、整合性情報を持つ。
- manifest には Run ID、実効 config fingerprint、train/eval、Env instance/lane、全 step 軸、Episode/Transition ID、trigger 定義、記録 fidelity、完全性、保存 schema version を含める。
- 一部ファイルだけが書かれた Bundle を完成済みとして公開しない。確定境界を設け、異常終了時は incomplete と判別できるようにする。
- metrics cache と同様の派生 cache にせず、調査の一次証拠として Run の保存・移動対象へ含める。

### 6. モデルと判断値の provenance を記録する

- Actor action には、使用した Actor snapshot の role/version と action values、選択理由を可能な範囲で紐付ける。
- Learner trigger には、検出時の online/target network role、learner update step、Sample 内 index、`q_sa`、`q_max`、target、TD error、loss 寄与、PER priority、IS weight など該当 detector が必要とする値を記録する。
- full network checkpoint を trigger ごとに必須保存しない。model version と既存 snapshot/checkpoint との対応を基本とし、checkpoint 昇格は容量と同期コストを評価する別オプションとする。
- 非決定論、BF16、物理シミュレーションのカオス性があるため、seed と action 列だけによる完全再実行を前提にしない。保存済み事実と再現実験を区別する。

### 7. Hot path と資源消費を有界にする

- 通常 step の Journal 追加は固定上限を持つ処理とし、既定では追加の NN forward や GPU synchronize を発生させない。
- 重い serialize、圧縮、画像化は background writer へ渡す。ただし queue の所有権と終了時 flush 境界を明示する。
- 容量超過時は、manifest と trigger 根拠を優先し、設定された方針に従って optional heavy payload を落とす。drop 件数と理由を metric/WARN に残す。
- 明示された必須 fidelity を満たせない場合は、Fail-Fast 原則に従い黙って低品質へ fallback しない。`auto` tier のみ、警告付き fallback を許可する。
- Forensics 自身の CPU/GPU/メモリ/I/O 時間と queue 深度を profile/metric で観測可能にする。

### 8. 検出ルールは設定と実効 artifact に残す

- 検出の有効/無効、閾値、連続回数、cooldown、保持予算、fidelity tier を Run config として扱う。
- 同一イベントの連打を防ぐ hysteresis、cooldown、episode 単位 dedup を設ける。
- detector の設定は `config_data.txt` に解決済み実効値として残し、Run 間比較時は同じ定義であったか確認できるようにする。
- detector 発火数、Bundle 完成数、不完全数、dedup 数、budget drop 数、Journal horizon を集約メトリクスとして記録する。

### 9. 最初の tracer bullet は Q 外れ値から元エピソードへ遡る

初回実装は次の縦断経路へ限定する。

1. Experience に `EpisodeId` / `TransitionId` を付与し、Replay Sample まで保持する。
2. DropMerge episode の低コストな Journal を保持する。
3. Learner で `q_max` の閾値または外れ値 detector を評価し、最大値を出した batch 要素を特定する。
4. 該当 Sample の Replay identity、episode/transition identity、Q/TD/PER 診断値を trigger として通知する。
5. 元 episode の利用可能な pre/post context を Bundle に昇格する。
6. GUI を必須とせず、保存物を一覧・要約表示できる offline inspector で内容を確認する。

この経路で、今回の `q_max_max = 105.069` に相当する事象が再発した際、「一件の異常値があった」から「どの Observation と系列が、どのモデル・Sample・PER 状態で異常値を生んだか」まで進めることを最初の成功条件とする。

### 10. 後続段階でエピソード系列 detector と Viewer を追加する

- Phase 2: 長時間 NOOP、NOOP margin、NoDrop timeout、NoLegal、終端理由、rank 更新など、Env/Actor 系 detector を追加する。
- Phase 3: Bundle 一覧、条件検索、step 表、Observation 表示、動画または既存 View による再生を Viewer へ統合する。
- 個別 detector の学習制御への利用は、本機構の観測結果を得てから別 PRD で判断する。

## Testing Decisions

- テストは private container の形ではなく、入力イベント、発生 trigger、保存された Bundle、明示された不完全状態という外部契約を検証する。
- ID 伝播テストでは、複数 Env/lane、episode reset、terminal/truncated、n-step、Replay 上書き、prefetch を通っても対象 episode/transition が一致することを確認する。
- PRD035 の replay item key テストを先例とし、物理 slot の再利用と永続 identity を混同しないことを検証する。
- Detector 単体テストでは、閾値境界、連続条件、hysteresis、cooldown、cluster、非 finite 入力、良い希少事象を含む reason/severity を確認する。
- Journal lifecycle テストでは、trigger 前履歴、active episode の固定、post-trigger 追記、終端確定、容量 eviction を確認する。
- Learner tracer bullet の統合テストでは、既知の一要素だけが極端な Q/TD 値を持つ batch を与え、該当 Sample index と元 episode が保存されることを確認する。
- 同一 episode への複数 trigger が一 Bundle に統合され、異なる episode が誤って統合されないことを確認する。
- 元履歴が保持期限切れの場合、Bundle が incomplete として確定し、利用可能な trigger/Sample 証拠は失われないことを確認する。
- writer queue 飽和、容量上限、I/O 失敗、Run close、異常終了途中を模擬し、silent loss や完成済み偽装がないことを確認する。
- 無効設定では従来と同じ学習経路となり、追加 forward、意図しない同期、学習入力の変更がないことを確認する。
- performance test/profile では、通常時、trigger burst 時、Run close 時を分け、step throughput、CPU/GPU memory、I/O queue の上限を確認する。
- 保存 schema は version を持ち、未知 version、欠損必須 field、破損 manifest を inspector が明示的に拒否することを確認する。

## Out of Scope

- 特殊挙動の根本原因を自動分類または自動修正すること。
- detector の発火を reward shaping、NOOP penalty、学習停止、Replay priority 操作へ直接接続すること。
- 全 Run の全 Observation、全フレーム、全ネットワーク checkpoint を無制限に保存すること。
- 既に詳細データを記録せず完了した過去 Run から、元エピソードを遡及復元すること。
- seed と action 列だけで、非決定論またはカオス的な Env を完全再現できると保証すること。
- 最初の実装で、全 Env/Agent に共通する汎用 detector DSL を完成させること。
- 最初の実装で、GUI、動画生成、対話的 timeline を必須化すること。
- Episode Forensics のために Env observation や Agent API の責務を歪めること。
- NEET 方策、Q 値過大評価、PER 偏り自体のアルゴリズム上の解決。

## Further Notes

### 想定する初期 detector 候補

| 分類 | detector 候補 | 主な調査目的 |
|---|---|---|
| Learner/Q | `q_max` 絶対閾値、rolling 分布からの外れ値、real-space Q 外れ値 | Q 過大評価、TBO inverse 増幅、希少入力の特定 |
| Learner/TD | TD error、sample loss、gradient、priority の同時上昇 | Replay で強調された遷移と更新影響の特定 |
| Replay | 同一遷移または同一 episode の短時間反復 Sample | PER feedback loop 仮説の確認 |
| Actor | NOOP Q margin、Action gap、低 entropy 張り付き | NEET 方策の意思決定根拠の確認 |
| Episode | 連続 NOOP、NoDrop timeout、NoLegal、極端な episode 長 | 失敗系列と終端契約の確認 |
| Success | rank 更新、Double Suika、極端な高報酬 | 良い希少技能と長期成長の確認 |
| Manual | UI/shortcut による bookmark | 未定義の興味深い挙動の保存 |

### 未決事項

- DropMerge で「詳細調査可能」と呼べる最小 Observation/Env state payload は何か。画像、盤面オブジェクト列、物理状態のどれを一次証拠とするか。
- Replay capacity と実時間 throughput に対して、Journal の保持 horizon を何 episode/step/byte にするか。
- episode 全体を compact journal に保持するか、trigger 対象遷移の周辺 window を中心にするか。
- model version を既存の Actor sync、target sync、snapshot とどう対応付けるか。
- trigger 時の network checkpoint 昇格を実用的なコストで行えるか。
- train と eval で detector、保持予算、fidelity の既定値を分けるか。
- Observation 本体の圧縮形式と schema migration をどうするか。
- `EvalEpisodeEndEvent` など既存 Observer/Event 系へ統合するか、Run 共通の独立サービス境界を設けるか。
- Bundle の offline inspector を CLI、metrics-viewer、Runner のどこから開始するか。
- Replay 内の同一 episode Sample 反復を数えるため、どの粒度で identity 集計を保持するか。

### 設計上の注意

- transient な `AuxData` や画面表示だけでは、Run 終了後の調査証拠にならない。
- `replay_item_keys` は同じ ReplayBuffer の現世代 slot を検証するための opaque key であり、Run artifact の永続 identity として直列化しない現行契約を維持する。
- Actor が行動した時点の Q と、Learner が後から同じ遷移を評価した Q は別の事実である。Bundle では model role/version と評価時点を必ず分離する。
- 一件の trigger は因果の証明ではない。Bundle は仮説を再現可能な証拠へ近づけるための仕組みであり、seed A/B や再発頻度の検証は別途必要である。
