# DefaultDQN Train Actor定期network snapshot PRD

## 問題

`DefaultDQNAgent`のTrain Actorは、現状ではLearnerが更新するonline networkを直接共有している。そのため、同じepisodeの途中でもLearner更新に応じて行動forwardが参照するparameterとbufferが変化する。

DropMergeのTBO/UQE spatial explorationでは、ENV laneごとの`epsilon`と`tau`を維持したまま、near-greedy laneで連続行動中のnetwork変化を切り分けたい。Eval専用Actor、別ENV batch、追加network forwardを導入すると、評価条件への特化とGPUコストが増える。既存Train Actorのnetworkだけを一定期間固定し、周期的にLearner online networkから更新できるopt-in機能が必要である。

この機能はNoDropTimeoutなど特定の現象を必ず改善する修正ではない。一般的なActor parameter lagを制御し、DropMerge固有の仮説をA/B検証するための実験機能である。

また、共通`Agent::CreateActor()`が必須の`bool clone_model`を受け取り、Train Runnerが`false`を決め打ちしている。network cloneを持たないAgentも存在し得るため、Agent固有のActor resource方針をRunnerが常に決めるinterfaceは責務が逆転している。

## 目的

1. `DefaultDQNAgent`のTrain Actorで、Learner online networkのparameterとbufferを複製した **Train Actor network snapshot** を任意に使用できるようにする。
2. snapshotの同期周期を`ProfiledValue<step_t>`で表し、既定は固定値としながら、必要なら`exp_step`に応じてannealingできるようにする。
3. Serial/PipelineのRunner実装差に依存せず、Actor自身がaction forward直前に必要な同期を判断する。
4. 既存の明示的な`Actor::Sync()`を、呼ばれた時点で強制同期するAPIとして維持する。
5. shared networkを既定として既存Runのaction選択forward回数、保存形式、Train動作を維持する。
6. snapshotの現在周期とageを、既存の`$action_info`メトリクス経路から観測できるようにする。
7. `CreateActor()`のclone指定をoptional overrideへ変更し、Agentが自身の既定方針を所有できるようにする。

## 対象範囲

周期snapshot機能の対象は`DefaultDQNAgent`のTrain Actorだけとする。

- Rainbowには設定、周期判定、メトリクスを公開しない。Rainbowは本プロジェクトでは限定機能のDQN実装であり、Ape-X由来の要素を追加しない。
- ImageClsのTrain Actorはshared networkによるリアルタイム反映を維持する。episodeの概念がなく、同期を遅らせる理由がないため、設定と同期実装を変更しない。
- MuZeroには周期snapshot機能を追加しない。`clone_model_override`を受け取っても、既存どおり値を無視してshared networkを使用する。
- Eval Actorは従来どおり外部からの`Sync()`で同期し、`MakeAction()`内の周期同期を行わない。
- `Agent::CreateActor()`のoptional override化は共通interfaceの責務修正として全Agentのsignatureへ反映するが、周期snapshot機能を他Agentへ拡張するものではない。

PRD 035のReplay初期優先度計算、hint carrier、PER計算、Learner優先度の最終権威は変更しない。`actor_approx`時はsnapshot Actorが既存forwardから生成するQヒントをそのまま利用し、snapshot ageによる近似品質の差はPRD 035の既存メトリクスで評価する。

## 用語

**Train Actor network snapshot** は、`DefaultDQNAgent`のTrain Actorがaction forwardに使用する、Learner online networkから複製されたparameterとbufferの時点コピーである。

固定対象はnetworkだけである。次はsnapshotへ含めない。

- `ActionPolicy`とその`epsilon`、`tau`などの状態
- `ObservationNormalizer`
- RNGと乱数消費状態
- ReplayBuffer、Learner、optimizer

したがって本機能を`policy snapshot`、`frozen policy`、方策固定とは呼ばない。同じObservationに対するactionの完全一致も保証しない。

## 解決方針

### 1. Actor生成override

共通interfaceを次の意味へ変更する。

```cpp
virtual std::shared_ptr<Actor> CreateActor(
    const BatchEnvSpec& batch_env_spec,
    RunMode run_mode,
    std::optional<bool> clone_model_override = std::nullopt,
    std::optional<torch::Device> device = std::nullopt) const = 0;
```

`clone_model_override`の契約は次のとおりとする。

| 値 | 意味 |
| --- | --- |
| `std::nullopt` | AgentがRunModeと自身の設定から決定する |
| `true` | 呼び出し側が`clone_model = true`を明示する。具体的な処理は各Agentの既存契約に従う |
| `false` | 呼び出し側が`clone_model = false`を明示する。具体的な処理は各Agentの既存契約に従う |

Train Runnerは`std::nullopt`を渡す。Eval Runnerは既存の`train.eval.[tag].clone_model`を明示overrideとして渡し、現在のper-Eval設定を維持する。

`DefaultDQNAgent`は次の優先順位で解決する。

| RunMode | override | 動作 |
| --- | --- | --- |
| Train | `nullopt` | `DefaultDQNAgent.train_actor.clone_model`に従う |
| Train | `true` | 設定に関係なくcloneし、周期同期を有効にする |
| Train | `false` | 設定に関係なくshared networkを使用する |
| Eval | `true` / `false` | 従来の明示指定に従う。周期同期は行わない |
| Eval | `nullopt` | `DefaultDQNAgent`のEval既定としてshared networkを使用する |

ImageClsとRainbowは、Trainの`nullopt`を現在と同じshared networkとして解決し、明示`true` / `false`の既存挙動を維持する。MuZeroは`nullopt` / `true` / `false`のいずれでも既存どおりshared networkを使用する。共通interfaceのoptional化を理由に、clone非対応Agentへ新しいfail-fastを追加しない。

Runnerは`nullopt`をshared networkと仮定してdevice整合性を判定してはならない。effective clone方針を解決したAgentが、shared networkとActor deviceの互換性を検証する。既存の明示overrideに対するRunner側の早期検証は維持してよい。

### 2. snapshotの生成と所有権

`DefaultDQNAgentConfig`は`train_actor`設定と`ProfiledValueConfig<step_t>`を所有する。AgentはLearnerが更新するsource online networkを所有する。

Train Actorは、clone有効時に次を所有する。

- action forwardに使用するprivate snapshot network
- runtimeの`ProfiledValue<step_t> sync_interval_`
- 最終同期の`train_step`と、強制同期後の基準更新に必要な状態

Actor自身が`MakeAction()`と`Sync()`でこれらを更新するため、同期周期とageはActor Stateである。private snapshot networkは他moduleと共有しないActor-private resourceであり、Agent所有のshared/source networkとは区別する。PolicyからLearnerへの依存は追加せず、ActorはAgentからsource online networkの参照だけを受け取る。

### 3. 初期snapshot

`Network::Clone()`が構築時に実行するparameter・buffer copyを初回同期とみなす。Train RunnerはRun開始時にstep loopより前でTrain Actorを生成し、最初の`MakeAction()`へ`train_step = 0`を渡す。このlifecycleを前提として、初期`last_sync_train_step`は`0`とする。

- 最初の`MakeAction()`で重複copyしない。
- Actor生成は現行の`auto_load_file`処理後なので、load済みonline networkから初期snapshotを作る。
- 最初のactionで観測するsnapshot ageは`0`とする。

Run途中で初めてTrain Actorを生成するlifecycleはv1の対象外とする。将来その経路を追加する場合は、生成時点の`train_step`を初期age基準として渡す契約を別途設計する。

### 4. action境界の周期同期

cloneを有効にしたTrain Actorの`MakeAction()`は、action forwardより前に次の順序で処理する。

1. `sync_interval_.Update(step.exp_step)`で現在周期を評価する。
2. 強制`Sync()`後の基準更新がpendingなら、copyを繰り返さず、現在の`step.train_step`を新しい同期基準にする。
3. `step.train_step - last_sync_train_step >= sync_interval_.Value()`を判定する。
4. 条件成立時はsource online networkのparameterとbufferをsnapshotへcopyし、`last_sync_train_step = step.train_step`とする。
5. 同期後のsnapshotを使ってaction forwardを1回実行する。

周期値を評価するschedule軸はglobalな`exp_step`、周期とsnapshot ageの単位はglobalな`train_step`で固定する。軸を選択する追加設定は設けない。

既定周期400の場合、初期snapshotは`train_step = 0`から`399`のactionで使用し、`train_step = 400`のforward直前に同期する。同期が起きたactionのageは`0`である。

schedule変化時は将来の同期点を予約しない。各action境界で現在値と現在ageを比較する。

- 周期が短縮され、短縮後の周期が現在age以下になった場合は、そのactionで直ちに同期する。
- 周期が延長された場合は、現在snapshotの寿命を新しい周期まで延長する。

### 5. 強制`Sync()`

`Actor::Sync()`は呼び出された時点でsourceからsnapshotへ強制copyする。周期未到達でも必ず同期する。shared network Actorでは従来どおり実コピーを行わない。

`Sync()`は`StepCounts`を受け取らない。周期Train Actorで強制同期した場合、次の`MakeAction()`が受け取る現在の`train_step`を新しいage基準として採用し、その`MakeAction()`では同じ内容を再copyしない。したがって次のactionで観測するageは`0`となる。

Pipeline Train Runnerがaction直前に毎step呼んでいる`actor_->Sync()`は削除する。Serial/Pipelineとも、Train Actorの`MakeAction()`内同期契約へ統一する。両Runnerで一致させるのは、同じ`train_step`で同期判定が成立するtrigger境界であり、Learnerとの実行タイミング差まで吸収して同一のsource parameter versionをcopyすることは保証しない。Evalの明示`Sync()`経路は変更しない。

### 6. thread safety

同一Actor instanceに対する`MakeAction()`と`Sync()`はthread-safeではない。各メソッド同士、および同じメソッドの複数呼び出しを並行実行してはならない。

公開基底interfaceである`Actor::MakeAction()`と`Actor::Sync()`の両方へ、この非スレッドセーフ契約をDoxygenコメントで記載する。Actor-local mutexは追加せず、Train/Eval Runnerが同一Actorへの呼び出しを直列化する。

source online networkのcopyは、Learner更新と同じAgent mutexの既存境界で保護する。

## 設定

設定はRunner共通設定ではなく、`DefaultDQNAgent`配下へ置く。

| 設定キー | 型 | 既定値 | 意味 |
| --- | --- | --- | --- |
| `DefaultDQNAgent.train_actor.clone_model` | bool | `false` | Train Runnerがoverrideを指定しない場合にprivate network cloneを使うか |
| `DefaultDQNAgent.train_actor.sync_interval.type` | string | `constant` | 既存`ProfiledValue`のprofile種別 |
| `DefaultDQNAgent.train_actor.sync_interval.value` | `step_t` | `400` | `constant`時の同期周期。単位は`train_step` |

`sync_interval`は既存`ProfiledValueConfig<step_t>`の全schemaをそのまま使う。

- root field: `type`、`value`、`start`、`end`、`steps`、`cycle_mult`、`phases`
- phase field: `phase.[name].type`、`value`、`start`、`end`、`steps`、`cycle_mult`
- type: `constant`、`linear`、`cosine`、`cosine_restart`、`phased`

`ProfiledValueConfig`には設定schemaへ公開しないcode-ownedな`min_value` / `max_value`制約を持たせる。Train Actorの`sync_interval`は`min_value = 1`を指定し、この制約は設定成果物へ出力しない。

既定値400に2の累乗としての意味はない。既存事例のある一般既定として採用するだけで、Ape-Xのframe軸と本機能の`train_step`軸が等価であるとはみなさない。DropMergeなど個別experimentでは、目的に応じたprofileを設定layerで明示的に上書きし、PRDは個別Runの最適周期を規定しない。

### バリデーション

- 汎用`ConfigData::Read()`はキー欠落時だけ呼出側の値を使い、存在する値の型変換失敗を`ANET_SYSTEM_ERROR`でfail-fastする。`ConfigData::Get()`も同じ契約に従う。
- default prefixとoverride prefixの各layerは独立して書式検証する。後続layerに正常なoverrideがあっても、先行layerの書式不正を隠さず、その実keyとraw値でfail-fastする。
- typed readerは前後空白を除去し、値全体の消費、overflow、負のunsigned値、nonfinite値、不正bool、vector内の不正tokenを検出する。数値中のカンマは既存互換として位置を問わず除去する。明示的な空値はstring / vectorでは有効、数値 / boolでは不正とする。
- `ConfigReader<ProfiledValueConfig<T>>`と`ProfiledValue<T>` constructorは同じ共通validatorを使い、未知type、必要な`steps == 0`、非正またはnonfiniteな`cycle_mult`、空の`phases`、未定義phaseを拒否する。
- `min_value` / `max_value`はinclusiveな構築時制約とし、`constant`の`value`、補間profileの`start` / `end`、列挙された各phaseのactive fieldだけを検証する。dormant fieldと未列挙phaseは検証しない。
- `ProfiledValueConfig`の構造・boundsエラーはlayer provenanceを推測せず、Config所有者から見た`train_actor.sync_interval.value`などの論理keyを示す。
- Train Actorの`min_value = 1`により、activeなprofileが生成し得るすべての周期を1以上にする。
- `clone_model = false`でも`sync_interval`を検証する。不正値をinactiveとして無視しない。
- `clone_model = false`かつ有効な`sync_interval`は、将来のA/B overrideに備えたdormant設定としてWARNなしで許可する。
- エラーには設定キー、raw指定値、期待される型または範囲を含め、暗黙のclampや既定値へのfallbackを行わない。

`ProfiledValue<step_t>`の補間は既存実装どおり、非負の小数部分を整数変換で切り捨てる。例えば計算結果`399.8`は`399 train_step`として扱う。DQN側だけで丸め規則を変更しない。

## メトリクス

次のscalarを`DQNActionInfo::GetScalar()`から公開し、`@train $action_info`で観測できるようにする。

| キー | 意味 |
| --- | --- |
| `train_actor_snapshot_interval` | そのactionで`exp_step`から評価した現在周期。単位は`train_step` |
| `train_actor_snapshot_age` | そのactionのforwardに使ったsnapshotの同期後age。単位は`train_step` |

観測値は、`MakeAction()`内のprofile更新と必要な同期が完了した後の値とする。同期が発生したactionでは`train_actor_snapshot_age = 0`である。

DefaultDQNで周期Train snapshotが無効な場合は、キーを省略せず両方とも`NaN`を返す。`std::nullopt`は返さない。これにはshared Train Actorと、周期同期を行わないDefaultDQN Eval Actorを含む。`DQNActionInfo::To()`と`WithAction()`を通しても値を維持する。Rainbowは両キーを公開せず、照会時に`std::nullopt`を返す。

両キーは利用可能なメトリクスのカタログである`metrics.scalar.full`にだけ登録し、`metrics.scalar.baseline`には追加しない。v1では試験的な診断メトリクスとして扱い、NEET現象などの分析で有用性が確認できなければ将来削除してよい。

次のメトリクスは追加しない。

- `train_actor_snapshot_learn_age`
- `train_actor_snapshot_sync_count`
- copy完了時間のscalar metric

## GPU copyとprofiling契約

同一GPU上のsnapshot同期は既存`Network::CopyTo()`を使い、parameterとbufferの`copy_`をenqueueする。v1ではLearner更新、D2D copy、Actor forwardが同じdefault CUDA stream上で順序付けられる既存契約へ依存する。

- CUDA Event、別stream、明示的な`cudaDeviceSynchronize`や`torch::cuda::synchronize`を追加しない。
- `Network::CopyTo()`の既存ProfileRangeは、host側のparameter列挙、ATen dispatch、CUDA enqueueに要した範囲として解釈する。GPU上のD2D完了時間とはみなさない。
- 実際のD2D時間はNsight SystemsのGPU CUDA memcpy activityで確認する。
- 計測だけを目的にproduction同期を追加しない。

将来Actor copyを別CUDA streamへ移す場合は、Learner更新完了eventを記録し、Actor streamがそのeventを待つ明示契約を別途設計する。本PRDでは扱わない。

## 保存・読込

現在のarchive契約であるModelとAdamの保存・読込を維持する。

- private snapshot network、`sync_interval_`のruntime値、最終同期step、強制同期pending状態は保存しない。
- `auto_load_file`は新しいRunを開始し、`StepCounts`とprofileの`exp_step`は0から始まる。
- load済みonline networkからActorを生成し、clone時の初回copyを初期snapshotとする。
- 将来の真のRun再開機能でStepCountsやschedule stateを復元する際に、snapshot/profile復元を改めて設計する。

## 性能と互換性

- `clone_model = false`を既定とし、shared networkの現行動作を維持する。
- snapshot有効化によるRun中の追加network inferenceを禁止し、各actionのaction選択network forwardを従来どおり1回とする。`Network::Clone()`が構築時に行うdummy forwardは、この「追加forward」には数えない。
- clone有無によって実行経路が変わるため、RNG利用回数または順序の一致は互換性要件としない。
- snapshot有効時の追加コストは、network 1個分のdevice memoryと周期的なparameter・buffer copyである。
- すべてのENV laneは同じsnapshot networkを1回のbatched forwardで共有する。ENVごとのsnapshotや追加forward群を作らない。
- PRD 035のReplay payload、PER数式、Learner更新回数を変更しない。
- Rainbow、ImageCls、MuZero、Evalの学習・同期挙動を周期snapshotの有効化対象にしない。

## 受入条件

1. `DefaultDQNAgent.train_actor.clone_model = false`が既定で、Train Actorは現在と同じshared online networkを使う。
2. Train Runnerが`clone_model_override = nullopt`を渡し、DefaultDQN Train ActorがAgent設定を解決する。
3. DefaultDQN Trainの明示`true` / `false`がAgent設定をoverrideし、Evalの明示指定が従来どおり機能する。ImageCls、Rainbow、MuZeroにもoptional signatureを反映するが、既存のclone解釈を変更せず、新しいfail-fastを追加しない。
4. Train Runnerがstep loop前にActorを生成し、最初の`MakeAction()`へ`train_step = 0`を渡す。Actor生成時のclone copyを初回同期とし、最初の`MakeAction()`で重複copyしない。
5. source online networkを更新しても、周期未満ではTrain Actorのsnapshot parameterとbufferが変化しない。
6. 既定周期400では`train_step = 0..399`が初期snapshotを使い、`train_step = 400`のforward前に1回だけ同期する。
7. 各actionで`sync_interval_.Update(step.exp_step)`を先に行い、短縮時は必要なら即時同期、延長時はsnapshot寿命を延ばす。
8. `Sync()`が周期に関係なく強制copyし、次の`MakeAction()`で現在`train_step`をage 0の基準にして重複copyしない。
9. Pipelineの毎step`Sync()`を削除し、Serial/Pipelineで同じ`train_step`のtrigger境界を使う。両Runnerが同じsource parameter versionをcopyすることは要求しない。
10. DefaultDQNの`train_actor_snapshot_interval`と`train_actor_snapshot_age`が同期後のaction情報を返し、周期snapshot無効時は`NaN`を返す。Rainbowでは両キーを公開しない。
11. 両snapshot metricを`metrics.scalar.full`だけに登録し、`metrics.scalar.baseline`へ追加しない。
12. `train_actor_snapshot_learn_age`、`train_actor_snapshot_sync_count`、copy時間scalarを追加しない。
13. 共通`ConfigData`が不正bool、負のunsigned値、末尾文字を含む数値、overflow、nonfinite値などを、既定値へfallbackせずfail-fastする。
14. `ProfiledValue<step_t>`が全profile種別、整数切り捨て、`min_value = 1`による正値検証を共通契約どおり扱う。
15. snapshot有効時もRun中のaction選択network forwardは各action 1回で、probeや優先度計算のための追加forwardを行わない。RNG利用回数または順序の一致は要求しない。
16. `auto_load_file`後はload済みonline networkから初期snapshotを作り、profileを`exp_step = 0`から開始する。
17. 同一Actorへの`MakeAction()` / `Sync()`非並行契約を公開Doxygenへ記載する。
18. CUDA完了待ちを追加せず、同一default streamと既存mutexでLearner更新、copy、forwardの順序を保つ。
19. Rainbow、ImageCls、MuZero、Evalに周期snapshotの設定または`MakeAction()`内同期を追加しない。

## テスト方針

決定論的なCPUテストを必須とする。

- shared modeでsource更新が従来どおりActor forwardへ反映されること。
- snapshot modeでsource更新が周期前に反映されず、境界で1回だけ反映されること。
- 400周期の`0..399` / `400`境界と、同期直後age 0を検証すること。
- `exp_step`で変化するprofileについて、周期短縮による即時同期と周期延長による寿命延長を検証すること。
- `constant`、`linear`、`cosine`、`cosine_restart`、`phased`を`step_t`で評価し、小数切り捨てと1以上の検証を行うこと。
- 強制`Sync()`、次actionでのage基準更新、重複copyなしをcopy回数で検証すること。
- Actor生成時の初期clone copy後、最初のactionでcopy回数が増えないこと。
- `clone_model_override`のDefaultDQN Train/Eval/nullopt/true/false組み合わせを検証すること。ImageCls、Rainbow、MuZeroはoptional signature移行後も既存挙動を維持し、MuZeroの明示`true`がshared networkのままであることを回帰テストすること。
- Serial/Pipelineで同じ`train_step`のtrigger境界となり、Pipelineが毎stepcopyしないことを検証すること。同じsource parameter versionのcopyはテスト要件にしない。
- 両action-info scalarの値、DefaultDQN無効時`NaN`、Rainbow照会時`std::nullopt`、`To()` / `WithAction()`での保持を検証すること。
- `metrics.scalar.full`に両キーがあり、`metrics.scalar.baseline`にないことを設定テストで検証すること。
- 共通`ConfigData`の不正bool、負unsigned値、末尾文字、overflow、nonfinite値がfail-fastし、`clone_model = false`でも`sync_interval`の共通boundsを検証すること。
- counting networkまたは同等の公開境界で、Run中のaction選択network forwardが各action 1回であることを検証すること。RNG消費回数は比較しない。
- `auto_load_file`相当のload後Actor生成で、復元済みnetworkから初期snapshotが作られること。
- Evalの明示`Sync()`、Rainbow、ImageCls、MuZeroの回帰テストを実行すること。

CUDA D2Dの非同期完了時刻を必須CIテストにはしない。CUDA挙動は既存`Network::CopyTo()`の契約とNsight Systemsで確認し、テストのためのproduction同期を追加しない。

実装後は少なくともfocused Debug test、`anet-core-test`全体、`git diff --check`を実行する。C++ビルドは`VsDevCmd.bat`経由で行う。

## 実装時のドキュメント更新

実装と同じ変更で、少なくとも次を現行仕様へ更新する。

- `docs/design/100_runtime_and_configuration.jp.md`: Train RunnerからActorへのclone overrideと、Pipeline/Serial共通の同期境界
- `docs/design/110_agents_and_learning.jp.md`: `CreateActor()`のoptional override、Train Actor network snapshot、`Sync()`契約
- `docs/design/140_observability.jp.md`: `$action_info`から読むsnapshot interval/ageと`NaN`条件
- Doxygen: `Actor::MakeAction()`と`Actor::Sync()`の非スレッドセーフ契約

本PRDの確定時点で、共通interfaceとresource所有権の判断は`docs/adr/0013-actor-network-resource-policy.md`へ記録し、両snapshot metricは`apps/runner/config/metrics_scalar.txt`の`metrics.scalar.full`だけへ先行登録する。現行動作を説明する`docs/design/`本文は実装前に先行変更しない。

## 対象外

- Rainbow、ImageCls、MuZeroへの周期Train snapshot機能
- Eval episode中の同期方式変更
- ENV laneごとのnetwork snapshot
- episode終端基準の同期
- Learner更新の停止
- Actor lagを補正する新しいoff-policy数式
- 追加のgreedy probe Actor、別ENV batch、追加network forward
- target networkをTrain behavior networkとして使うこと
- snapshot、StepCounts、profile runtime stateを含む真のRun再開
- CUDA別stream化とEvent同期
- Actor/Learner全体の包括的lock監査
- Run途中で初めてTrain Actorを生成するlifecycle
- `clone_model_override`を`Shared` / `Independent`などの意味ベースresource policyへ置き換える共通interface再設計
- DropMerge固有の最適周期の決定

## 補足

- 既定400は一般既定にすぎず、DropMerge向け最適値ではない。experiment側で明示的に上書きする。
- snapshot ageはLearner update回数ではなく`train_step`で表す。replay ratioにより同じageでもLearner更新回数は変わり得る。
- 本機能の採否は、NoDropTimeoutだけでなく、PRD 035のActor/Learner一致度、`exp_step_per_sec`、主目的scoreを別々に比較して判断する。
- 周期400、snapshot metric、強制`Sync()`pending処理などの機能固有詳細はPRDで管理し、ADR 0013には共通interfaceとresource所有権の判断だけを記録する。
