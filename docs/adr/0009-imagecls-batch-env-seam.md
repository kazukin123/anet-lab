# ImageCls batch-native env の seam：EnvRepository を single/batch variant 化＋Factory/Builder 再レイヤ

ImageCls を batch-native 化するにあたり、既存の `EnvRepository`（class_id → `SingleDiscreteEnvFactory` の1本 map）は「single env を N 個 wrap して batch を作る」前提で、env 固有の batch 実装を差し込めない。案A（single/batch を別 registry に分離）／案B（1 factory に single と batch の両 capability を持たせ ImageCls だけ batch を override）／案C-new（registry は1本のまま、class_id ごとに single XOR batch を variant として排他登録）を比較し、**案C-new** を採用する。根拠: standalone single env の本番消費者はゼロ（train/eval/wrap いずれも batch 経由で、tests は concrete class を直接 new し registry 非依存）なので single/batch は排他で十分。案B は ImageCls に使われない single capability を死蔵させ、eval が惰性で B=1 に留まりやすい。案A は registry・登録・lookup が二重化する。

あわせて命名/レイヤも整理する。現行 `BatchEnvFactory`(IF) は実装が `DefaultBatchEnvFactory` の1つだけで、trainer も concrete を直接保持しており、**単一実装の死んだ抽象**（IF が polymorphism に使われていない）。命名規則を「AbstractFactory 相当（IF＋種別ごと具象、registry 登録）= Factory、config と Factory で必要なインスタンスを組む単一の上位層 = Builder（1種1インスタンス＝IF 不要）」と定め、①現行 `BatchEnvFactory` IF を削除、②上位層 `DefaultBatchEnvFactory` を単一 concrete `BatchEnvBuilder`（+`BatchEnvBuilderConfig`）へ改名、③空いた `BatchEnvFactory` 名を **per-class abstract factory IF**（`SingleDiscreteEnvFactory` と対、`BatchEnv` を作る）として再定義する。さらに `BatchEnvBuilder::CreateBatchEnv` に `config_prefix` を追加し、eval env の構築も builder 経由に統一する（従来 eval は config_prefix を渡せず factory をバイパスして `VectorizedDiscreteBatchEnv` を直接構築していた＝`docs/memo/999` の FLAG）。詳細設計・実装フェーズは `docs/memo/034_imagecls_batch_input_10prd.md`。

## Consequences

- `EnvRepository` の値型は `std::variant<std::shared_ptr<SingleDiscreteEnvFactory>, std::shared_ptr<BatchEnvFactory>>`。class_id ごと single XOR batch を排他登録し、同一 class_id への二重登録は `LOG::warn` 後 throw（fail-fast）。`RegistEnvFactory` は両型 overload。**登録は `Init*()` manual（RunnerApp.cpp）に一本化し、`ANET_REGISTER_ENV_FACTORY` static 登録マクロ（使用4箇所＋定義）は撤去する**（現状 GridMaze/LunarLander/CartPole/DropMerge で manual と static の二重登録が起きており、fail-fast 有効化の前提として解消する）。
- 現行 `BatchEnvFactory`(IF) を削除し、上位層は単一 concrete `BatchEnvBuilder`（旧 `DefaultBatchEnvFactory`）に。新 `BatchEnvFactory` は per-class abstract factory IF（`CreateBatchEnv(config_data, device, seed, num_envs, config_prefix)` + `GetTargetEnvClassId()`）。
- `BatchEnvBuilder::CreateBatchEnv(seed, num_envs, config_prefix="")`：entry が batch factory なら `factory->CreateBatchEnv(config_data_, device_, seed, num_envs, config_prefix)` を直接呼び、single factory なら従来の `VectorizedDiscreteBatchEnv`/`ThreadPoolDiscreteEnv` で N 個 wrap（worker_type 分岐はこの経路のみ）。
- eval env は `env_factory_->CreateBatchEnv(eval_seed, eval_batch_size, config_prefix)` 経由に統一（`trainer.cpp` の直接 `VectorizedDiscreteBatchEnv` 構築を廃止）。これで ImageCls eval が自動で batch adapter に載り、docs/memo/999 の config_prefix FLAG も解消する。
- ImageCls は `ImageClsEnv` を `SingleDiscreteEnv`→`BatchEnv`、`ImageClsEnvFactory` を `SingleDiscreteEnvFactory`→`BatchEnvFactory` に**名前踏襲で作り替え**（single の上位互換）。旧 single result は削除。他 env は従来どおり single factory + wrap で不変。ImageCls 固有は batch のみのため "Batch" 接尾辞を付けない。
- batch factory は自前で並列 decode を持つため、worker 数ヒューリスティック（`GetLogicalCores`/`ResolveWorkerThreads`）を無状態 mixin `WorkerThreadResolver` に抽出し `BatchEnvBuilder` と共有する。
- ImageClsEnv の `GetScalar` は global 2キーのみ（`accuracy`＝直近に確定した採点サイクルの正解率〔train=epoch wrap／eval=eval 1回〕、`epoch_count`）。per-lane stream キーは持たず、不明キーは `ANET_SYSTEM_ERROR`。あわせて `DiscreteBatchEnvBase::GetScalar` の無 prefix WARN+mean fallback を廃止し SYSTEM_ERROR とする（mean./max./min. 必須。全 config は prefix 付きのみで影響ゼロ、詳細は PRD 034 C4）。
- ImageClsEnv は central batch RNG を使うため、同 seed でも旧 per-env run と bit 一致しない。「同 seed・同 config で新 contract として再現保証」とする（network 決定性は ADR 0006 と直交）。
- 実装は挙動不変の framework refactor（旧 IF 削除＋`BatchEnvBuilder` 改名＋mixin 抽出）を Phase 0 として先行し、seam→source→env の順に積む（PRD 034 の実装フェーズ）。

## Follow-up: PRD 037先行によるEnv name伝播

このfollow-upは、上記のregistry variant化、Factory/Builder再レイヤ、configured EvalのBuilder統一という元の決定と理由を変更しない。[PRD 037](../memo/037_env_instance_name_10prd.md)を[PRD 034](../memo/034_imagecls_batch_input_10prd.md)より先に実装することに伴い、Env生成seamへ人間向けの必須`name`を追加する。

- 実装順序をPRD 037→PRD 034とし、PRD 034 Phase 0はPRD 037完了後をbaselineとする。
- PRD 034 Phase 0では、旧top-level `BatchEnvFactory::CreateBatchEnv(name, seed, num_envs)`を維持したまま`DefaultBatchEnvFactory`を`BatchEnvBuilder`へ改名する。
- Phase 1の恒久seamでは、`BatchEnvBuilder::CreateBatchEnv(name, seed, num_envs, RunMode, config_prefix)`と、新per-class `BatchEnvFactory::CreateBatchEnv(config_data, device, name, seed, num_envs, RunMode, config_prefix)`の両方で`name`を必須とする。RunModeの生成時受け渡しはPRD 034 D3で採用と決定した（`name`とともに必須）。
- Builderは`name`を加工・解析せず、single経路ではwrapperへ、batch-native経路ではper-class factoryへ転送する。single wrapperはlane nameを`<name>[lane_index]`として完成させ、batch-native Envはnameを共通`BatchEnv`基底へ渡す。
- configured EvalをBuilderへ統一する際も、`name=tag`、configured Eval tag、`RunMode`、`config_prefix=train.eval.[tag].env`を別契約として維持する。
- EvalPanelは`name=EvalPanel`を維持し、PRD 034 D7で選ぶconfigured Eval tag、その`RunMode`、config prefixとは分離する。
- `BatchEnv`は状態を持たないinterfaceとして`GetName()`と`GetEnvName(lane_index)`をpure virtualで公開し、`BatchEnvBase`がnameとlane nameを保持して両accessorを`final override`する。native `ImageClsEnv`は`BatchEnvBase`を継承して受け取ったnameを渡すだけとし、lane name accessorを独自実装しない。
- `name`は人間向けの不透明な文字列であり、DatasetKey、Source、cache、seed、RNG domain、sample列、Env挙動の決定に使用しない。
- 同一Run内のBatchEnv nameはcase-sensitiveな完全一致で一意とする。Runをまたぐ一意性は保証しない。
- `RunManager`は固定名`train`、全configured Eval tag、固定名`EvalPanel`を最初のBatchEnv構築前に一括検証する。`train`と`EvalPanel`はconfigured Eval tagの予約名とし、衝突は`ANET_SYSTEM_ERROR`とする。
- 生成成功済みnameの`name -> owner説明` registryは`RunManager`だけがrun-localに所有する。動的な`CreateEvalRunner(name, ...)`も第二のEnv構築前に重複を検出し、既存runnerを上書きしない。
- 生成失敗したnameはregistryへ残さず、生成成功済みnameはRunManager破棄まで再利用しない。Builder、single/per-class Factory、wrapper、native Envへregistryまたはowner情報を渡さない。
- 重複nameはWARN、自動suffix、暗黙のrenameへfallbackせず、重複name、既存owner、要求ownerを含む`ANET_SYSTEM_ERROR`でfail-fastする。

元のConsequencesに記載したname追加前のシグネチャは歴史的な判断時点の表記として残す。実装時の規範シグネチャはこのfollow-upとPRD 034のPhase定義を優先する。

PRD 034 D3の決定（生成時RunMode固定）に伴い、次を追記する。`BatchEnv` / `SingleDiscreteEnv`の`Reset` / `Step`から実行時`RunMode`引数を撤去し、`SingleDiscreteEnvBase` / `BatchEnvBase`が`name`と同じパターンで生成時RunModeを保持して`GetRunMode()`を公開する。`SingleDiscreteEnvFactory::CreateSingleEnv`にもRunModeを追加する。既存Envの実行時mode分岐（CartPoleのeval初期状態固定）は保持値参照へ置換し挙動を変えない。役割特化インスタンスを構築時に確定する形は、Gym系env API（SB3 / Tianshou / DI-engine / RLlib）およびPyTorch DataLoaderのtrain/val分離と同型であり、誤modeでのReset/Stepは引数の不存在により構造的に不可能になる。

## Follow-up: PRD 034実装時のDataset ownershipとdormant検証

PRD 034の確定判断に従い、ImageClsのimmutable Dataset資源はprocess singleton `ImageDatasetManager`が`DatasetKey`単位で所有する。同一key・同一resolved configはmanifest/cacheを共有し、同一key・異configは登録済みDatasetの利用有無にかかわらずfield差付きでfail-fastする。catalog登録はI/Oなしの全件preflight後にcommitし、manifest構築またはcache entry decodeの失敗はprocess lifetime中stickyとする。一方、Sampler、cursor、RNG、augment、decode pool、episode stateはEnv-local `ImageDataSource`が所有し、singletonへ移さない。

`interval=0`のconfigured Evalは全Env共通のdormant宣言として、name予約とschema検証だけを行う。ImageCls batch factoryの`ValidateConfig`はcatalog/SourceをI/Oなしで検証するため、dormant tagはmanifest、Env、Actor、Observer、poolを生成しない。enabled configured EvalとEvalPanelは生成したEnvの`EnvSpec.info["image_dataset_key"]`、state/action specをmain Envのcanonical specと接続前に比較する。

## Follow-up: ImageCls固有判断をAgent・Module seamへ移動

後続レビューにより、直前follow-up末尾のcanonical DatasetKey/EnvSpec比較を撤回する。ImageCls設定は`ImageClsEnv.train.*`と`ImageClsEnv.eval.*`を標準の組として必須化し、tagなしEvalは標準Eval設定、configured Evalはtag固有overlayを使用する。両manifestはImageCls factoryが起動時に個別検証するが、Train/Eval間でspec一致を強制しない。

Runnerは生成対象の`BatchEnvSpec`と`EnvSpec`を`Agent::CreateActor()`へ渡し、Actor生成可否はAgentが判断する。通常の同一state/action契約には`EnvSpec::CheckSameStateActionSpec()`を使用できるが、RunManagerはImageCls class ID、DatasetKey、`EnvSpec.info`を解釈せず、canonical specを保持しない。DatasetKeyも`EnvSpec.info`へ特別格納しない。

適用済み設定は`Module::GetConfigData()`の共通自己記述情報として扱う。Envは子を含む不変の実効設定snapshotを返し、RunManagerは`config/env.<Env name>.txt`へ共通dumpする。これに伴い、具象Envからの個別設定ログとRun直下`config.txt`集約を廃止する。実動情報はこのConfigへ混ぜず、将来のProperty seamへ分離する。
