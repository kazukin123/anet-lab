# ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 出自: `grill-with-docs` によるカテゴリ別（C0–C9）詳細化。seam 決定は `docs/adr/0009-imagecls-batch-env-seam.md` に併記。
> レビュー: Codex による PRD レビュー指摘 7 件（double-buffer 契約 / cache fill race / eval metrics 重複 / cache fail-fast / episode_start 契約 / eval config path / manifest validation）を反映済み。
> レビュー2巡目（Claude、前提事実の全数照合）: 環境登録の一本化（既存二重登録の解消）／GetScalar キー体系の全面改定（global 2キー・stream キー全廃）／eval_samples ローテーション化／snapshot 契約 等を反映。
> C カテゴリ: C0=seam・factory／C1=データ供給 source／C2=並列 decode／C3=cache／C4=env（RL scaffolding）／C5=eval／C6=config／C7=RNG・再現性／C8=旧 single 実装・テストの B=1 移行。
> 既存 runner/observer/GUI 互換のため、当面は `BatchEnv` interface で動く adapter として実装する。

## Context（背景・目的）

ImageCls は RL フレームワーク内に `SingleDiscreteEnv`（[`ImageClsEnv`](../../core/envs/imagecls1/src/ImageClsEnv.hpp:76)）として作られ、
本流 train/eval は **N 個の single env を `VectorizedDiscreteBatchEnv`/`ThreadPoolDiscreteEnv` でラップ**して batch を作る。
分類は本来 batch dataset だが RL の env/episode に押し込まれており、以下の無駄がある:

- 各 env が train+eval の `ImageDataSource` を2本保持（[`ImageClsEnv.cpp:46-52`](../../core/envs/imagecls1/src/ImageClsEnv.cpp:46)）→ N env で 2N 本。
- 1枚ずつ decode/resize/tensorize（[`ImageData.hpp:92`](../../core/envs/imagecls1/src/ImageData.hpp:92)）、**cache 無し**（epoch ごと全再 decode）。
- `CopyBatchItem` で1行ずつ collate（[`tensor_util.hpp:380`](../../core/anet-core/include/anet/tensor_util.hpp:380)）、`getStepResult` は毎回新規確保（[`env.cpp:196`](../../core/anet-core/src/env.cpp:196)）。
- eval が **B=1 固定**（[`trainer.cpp:817`](../../core/anet-core/src/trainer.cpp:817)）で、max_steps 枚のランダム復元抽出という意味論の悪い評価。

目的: **ImageCls を batch-native 入力に置き換え**、大 dataset（Food101 / 将来 ImageNet）でメモリ/性能・eval 品質を改善する。
既存の contract（obs=`grid[3,H,W] uint8`＋`vector[1] int64`、network 境界で `float32/255`）は維持する。

## 確定した設計判断

1. **seam=案C-new（C0）**: `EnvRepository` は1本、値を `std::variant<SingleDiscreteEnvFactory, BatchEnvFactory>` にし、
   class_id ごと **single XOR batch を排他登録**（二重登録は `LOG::warn` 後 throw）。ImageCls は batch のみ登録。→ ADR 0009。
   **前提整備（Phase 0）**: 現状 GridMaze/LunarLander/CartPole/DropMerge は `ANET_REGISTER_ENV_FACTORY`（static 自動登録）と `Init*()` manual 登録の**二重登録が現実に起きている**（現行 Regist は上書きで無害）。
   fail-fast 有効化に先立ち、**マクロ使用4箇所＋使用ゼロになるマクロ定義（env.hpp:176）を削除し `Init*()`（[`RunnerApp.cpp:205`](../../apps/runner/src/RunnerApp.cpp:205)）に一本化**する。テストは registry 非依存（確認済み）。
2. **Factory/Builder 再レイヤ（C0）**: 現行 `BatchEnvFactory`(IF、実装は `DefaultBatchEnvFactory` のみ＝**単一実装の死んだ抽象**) を**削除**し、
   上位オーケストレータを **単一 concrete `BatchEnvBuilder`**（IF 無し、旧 `DefaultBatchEnvFactory`＋`BatchEnvBuilderConfig`）に改名。
   空いた `BatchEnvFactory` 名を **per-class abstract factory IF**（`SingleDiscreteEnvFactory` と対、`BatchEnv` を作る）として再定義。
   `BatchEnvBuilder::CreateBatchEnv(seed, num_envs, config_prefix="")` に prefix を追加し、eval の直接 `VectorizedDiscreteBatchEnv` 構築を
   `env_factory_->CreateBatchEnv(eval_seed, eval_batch_size, config_prefix)` に置換（docs/memo/999 の FLAG 解消）。
3. **`ImageClsEnv`/`ImageClsEnvFactory` を batch 版に作り替え（C0/C4、旧名フル踏襲）**: `ImageClsEnv` は `SingleDiscreteEnv`→`BatchEnv`
   （旧 single の上位互換、B=1 で同挙動）、`ImageClsEnvFactory` は `SingleDiscreteEnvFactory`→`BatchEnvFactory`。
   旧 `ImageDataSource`(単品 Dataset) は batch source へ作り替え（C1、名前踏襲）、旧 single result（`ImageClsResetResult`/`ImageClsStepResult`）は削除。
4. **standalone データ供給 source（C1）**: manifest / sampler / decode / augment / cache を1コンポーネントに集約し
   `NextBatch(B, mode) → {grid:[B,3,H,W] uint8, targets:[B]}` を返す。env は RL scaffolding で薄く包む。名前は `ImageDataSource` 踏襲。
5. **train サンプリング=共有 shuffled epoch カーソル（C4）**: 全 index を shuffle、Step ごと B 個 consume、
   端数 <B は **wrap**（reshuffle した次 epoch 先頭で補填、B 固定・データ非破棄・`data_size<B` でも頑健）。
6. **episode ⊥ epoch（C4, X）**: **episode=max_steps の metrics/可視化窓**（既存機構のまま、データ非依存、同期境界。役割は Conv2d 可視化頻度・train 側 EpisodeEndEvent・`$episode_step` 軸の cadence 供給のみで、**accuracy 系キーとは無関係**）、
   **epoch=cursor wrap カウンタ**（scalar `epoch_count`＋reshuffle トリガのみ、boundary 機構にしない）。
7. **reward/next_state は RL scaffolding（C4）**: learner は `experiences.state.obs` のみ使用
   （[`image_cls_agent.cpp:329,348`](../../core/anet-core/src/image_cls_agent.cpp:329)）。reward は metrics 専用、
   `next_state`/`continue_state` は次バッチ（データは同一、フラグのみ差し替え）。
8. **prefetch は作らない・double-buffer 契約（C2+C3）**: 既存 `PipelineTrainRunner` が env.Step を async learn と 1-deep overlap 済み。
   NextBatch は env.Step 内で **PinnedThreadPool により B 枚並列 decode**。出力は **double-buffer（ping-pong 2枚）**に書き、
   `continue_state`(=次 `state_`) を次 Step の `state_.Clone()`（[`trainer.cpp:642`](../../core/anet-core/src/trainer.cpp:642)）まで生存させる
   （単一バッファ再利用は `env.Step(t)` が `state_(t-1)` を上書きし clone を汚すため不可）。**batch 内の重複 index は dedupe**してから decode。
9. **cache=`SampleCachePolicy` strategy・fail-fast（C2+C3）**: `cache.mode = auto | none | full_ram`。
   **`auto`（既定）**のみ size 判定で `full_ram`/`none` を選ぶ（fallback は auto のみ）、**明示 `full_ram` の cap 超過は `ANET_SYSTEM_ERROR`**（明示設定を勝手に落とさない）。
   `FullRamCachePolicy` の fill は **dedupe 済み unique idx** を書く（同一 idx 同時 fill の race を排除）。bounded LRU は epoch 非復元と非整合（hit≈0）で **不採用**。
   `MmapCachePolicy`/`PreprocessedFileCachePolicy` は future（seam のみ）。
10. **eval=deterministic pass・代表 lane done（C5）**: eval モードは eval source。`eval_samples=all`（既定）は **sequential full pass**、
    `eval_samples=N<all` は **ローテーション**（`EpochShuffleSampler` を eval 呼び出し間で永続させ全 test 画像を均等使用。固定部分集合は「同じ画像だけが使われ続ける」ため不採用）。
    pass 完了 Step で **代表 lane(0) のみ `done=true`**（`EpisodeEndEvent` 1個＝metrics B 重複を回避）。既存 `RunEvaluationEpisode` の do-while を**無改修**で駆動。
    accuracy は env `GetScalar("accuracy")` が **eval 1回分の snapshot** で返す。full pass の末尾は **pad-to-B＋valid-prefix count**（exact、pad step の `n_transitions`=valid count）。
11. **RNG（C7）**: sampler RNG と augment RNG を **別 stream**（source seed から `SeedMaker` 導出）。
    augment は **per-sample 決定的 seed**（`(augment_seed, epoch, dataset_index)`）で並列 decode 下でも thread 順非依存。
    epoch permutation は `(sampler_seed, epoch)` 由来で再現可能。**新 reproducibility 契約**（旧 per-env run と bit 非一致）。
12. **worker 数ヒューリスティックの共有（C0）**: 旧 `DefaultBatchEnvFactory`（→ `BatchEnvBuilder`）の `GetLogicalCores`/`ResolveWorkerThreads` を
    無状態 mixin base `WorkerThreadResolver`（`RandomHolder` と同型）へ抽出し、`BatchEnvBuilder` と source（decode pool）で共有。
13. **manifest は fail-fast（C1）**: 現行 `ImageDataSource` の malformed line / unknown class の silent skip を廃止し、
    **list 行番号・class 名・path 付き `ANET_SYSTEM_ERROR`**（repo の fail-fast 方針）。
14. **GetScalar は global 2キーのみ（C4/C5）**: 無印 **`accuracy`**＝「直近に確定した採点サイクルの正解率」（train=epoch wrap／eval=eval 1回）＋**`epoch_count`**。
    旧 per-episode stream キー（`episode_len`/`reward_sum`/窓 `accuracy`）は**全廃**（train の fresh 精度は agent 側 `accuracy @learn`＝38_agent/03 が既存で二重化不要、lane は共有カーソルのスロットに過ぎず per-lane 値に診断価値なし）。
    不明キー・global キーへの prefix 付与は `ANET_SYSTEM_ERROR`。framework 側 `DiscreteBatchEnvBase::GetScalar` の「無 prefix→WARN+mean fallback」も**廃止して `ANET_SYSTEM_ERROR`**
    （mean./max./min. 必須。全 config audit＝prefix 無し `$env` キー使用ゼロ・テスト依存ゼロを確認済み）。

## クラス命名（確定）

> 方針: **AbstractFactory 相当（IF＋種別ごと具象、registry 登録）= Factory**、**config と Factory で必要なインスタンスを組む単一の上位層 = Builder（1種1インスタンス＝IF 無し）**。
> framework は Batch/Single を保持。ImageCls 固有は batch 版のみのため "Batch" を付けず旧名踏襲（`ImageClsEnv`/`ImageClsEnvFactory` は base を Single→Batch へ作り替え）。

| Before | After | 種別 | 概要 |
|---|---|---|---|
| `BatchEnvFactory`(IF、単一実装) | （削除） | 削除 | 死んだ抽象（Builder に IF 不要） |
| `DefaultBatchEnvFactory`(+Config) | `BatchEnvBuilder`(+`BatchEnvBuilderConfig`) | 改名（concrete 単一） | config＋registry の Factory＋wrap 戦略で `BatchEnv` を組む上位層 |
| `SingleDiscreteEnvFactory` | 同左 | 温存 | per-class abstract factory（single） |
| （旧 `BatchEnvFactory` 名を再利用） | `BatchEnvFactory` | 新規 IF | per-class abstract factory（`BatchEnv` を作る、`SingleDiscreteEnvFactory` と対） |
| — | `WorkerThreadResolver` | 新規 | worker 数ヒューリスティック mixin（framework） |
| — | `PlainBatchResetResult`/`PlainBatchStepResult` | 新規 | 空 aux の最小 concrete batch result（framework） |
| `ImageClsEnv`（SingleDiscreteEnv） | `ImageClsEnv`（BatchEnv） | 作り替え | batch-native env、旧 single の上位互換 |
| `ImageClsEnvFactory`（SingleDiscreteEnvFactory） | `ImageClsEnvFactory`（BatchEnvFactory） | 作り替え | ImageCls の per-class batch factory |
| `ImageClsResetResult`/`ImageClsStepResult` | （削除） | 削除 | 旧 single result（aux 未消費） |
| `ImageDataSource`（単品 Dataset） | `ImageDataSource`（batch source） | 作り替え | `NextBatch(B,mode)`。manifest+sampler+decode+augment+cache 集約 |
| （path/label/class 部） | `ImageManifest` | 新規 | `paths`/`targets`/`class_names`（validation は fail-fast） |
| （env の sampling） | `IndexSampler`+`EpochShuffleSampler`+`SequentialPassSampler` | 新規 | train shuffle+wrap／eval sequential+pad |
| `ImageDataSource::get`（decode 部） | `DecodeResizedImage`（free fn） | 抽出 | pure I/O |
| `ApplyTrainAugment`/`ApplyRandomResizedCrop`（env method） | `ApplyTrainAugment`（free fn, seed 引数） | 移設 | per-sample 決定的 seed |
| — | `SampleCachePolicy`(IF)+`NoCachePolicy`+`FullRamCachePolicy` | 新規 | cache 戦略（`Mmap`/`PreprocessedFile` は future） |
| `ImageClsView` 一式 / `ImageClsEnvConfig` | 同左 | 温存 | GUI（batch[0]）/ config（`cache.*` 追加） |
| `RegistEnvFactory`（single 用） | `RegistEnvFactory`（single/batch overload） | 改修 | 両 factory 型を受ける。登録は `Init*()` manual に一本化 |
| `ANET_REGISTER_ENV_FACTORY`（マクロ） | （削除） | 削除 | static 自動登録を撤去（使用4箇所＋定義）。既存二重登録の解消 |

## 前提事実（実コード確認済み）

> 基準: working tree（`main`）。行番号は現 working tree 基準。**以下は現行コードの記述なので旧名（`DefaultBatchEnvFactory` 等）のまま**。

**seam / factory**
- `EnvRepository`（[`env.hpp:92`](../../core/anet-core/include/anet/env.hpp:92) / [`env.cpp:640`](../../core/anet-core/src/env.cpp:640)）は
  `unordered_map<string, shared_ptr<SingleDiscreteEnvFactory>>` の1本。`@todo SingleDiscreteEnvFactory → SingleDiscreteEnvCreator`（[`env.hpp:100`](../../core/anet-core/include/anet/env.hpp:100)）は**採用しない**（Factory 命名を保持）。
- `DefaultBatchEnvFactory::CreateBatchEnv(seed, num_envs)`（[`env.cpp:599`](../../core/anet-core/src/env.cpp:599)）は class_id で single factory を引き、
  num_envs==1/SINGLE_THREAD なら Vectorized、他は ThreadPool で **N 個 wrap**。specialized batch 分岐なし。**config_prefix を通さない**。
- `getStepResult()` は毎 Step `createEmptyStepResult()` で **StepResult を新規確保**（[`env.cpp:196`](../../core/anet-core/src/env.cpp:196)）＝現状は continue_state のバッファ aliasing が起きない。
- eval は `single_env_factory = env_factory_->GetSingleFactory()`（[`trainer.cpp:754`](../../core/anet-core/src/trainer.cpp:754)）を
  取り、`VectorizedDiscreteBatchEnv(config_data, single_env_factory, 1, env_device, eval_obs_seed, config_prefix)`
  （[`trainer.cpp:817-818`](../../core/anet-core/src/trainer.cpp:817)）で**直接構築**（config_prefix=`train.eval.[tag].env`、[`trainer.cpp:793`](../../core/anet-core/src/trainer.cpp:793)）。
- 現行 `BatchEnvFactory` IF（[`rl.hpp:644`](../../core/anet-core/include/anet/rl.hpp:644)）= CreateBatchEnv のみ、**実装は `DefaultBatchEnvFactory` だけ**。trainer は `unique_ptr<DefaultBatchEnvFactory>`（[`trainer.hpp:230`](../../core/anet-core/include/anet/trainer.hpp:230)）で **concrete 保持＝IF 経由なし**。
- worker 数ヒューリスティック `ResolveWorkerThreads`（[`env.cpp:565`](../../core/anet-core/src/env.cpp:565)）の状態依存は `config_.worker_threads` のみ、`GetLogicalCores`（[`env.cpp:558`](../../core/anet-core/src/env.cpp:558)）は無状態。
- env 登録の実経路は `Init*()`（[`RunnerApp.cpp:205-209`](../../apps/runner/src/RunnerApp.cpp:205)）。GridMaze/LunarLander/CartPole/DropMerge は加えて `ANET_REGISTER_ENV_FACTORY`
  （[`GridMazeEnv.cpp:294`](../../core/envs/gridmaze1/src/GridMazeEnv.cpp:294) 等4箇所）の static 登録も持ち**同一 class_id へ二重 Regist が起きている**（現行は上書き）。*_test.cpp は EnvRepository/RegistEnvFactory を使用しない。
- 全 config の `$env` scalar キーは **prefix 付きのみ**（DropMerge/GridMaze/ImageCls 全数 grep 済み）＝`DiscreteBatchEnvBase::GetScalar` の無 prefix WARN+mean fallback（[`env.cpp:227`](../../core/anet-core/src/env.cpp:227)）の使用実績ゼロ。
- agent 側の train 精度 `accuracy @learn`（[`ImageCls.txt:504`](../../apps/runner/config/ImageCls.txt:504)、learner が update ごとに算出・EMA 行も設定済み）が既存＝env 側の fresh 精度キーは冗長。
- eval 起動頻度は `train.eval.[eval1].interval = 50`（learn step 基準、[`ImageCls.txt:623`](../../apps/runner/config/ImageCls.txt:623)）。accuracy 記録は 1 eval につき 1 点で、pass 長は点の間隔に影響しない（background 実行）。

**learner / パイプライン**
- `ImageClsLearner::UpdateFromBatch`（[`image_cls_agent.cpp:316`](../../core/anet-core/src/image_cls_agent.cpp:316)）は
  `experiences.state.obs`（[grid](../../core/anet-core/src/image_cls_agent.cpp:348) / [vector→targets](../../core/anet-core/src/image_cls_agent.cpp:329)）**のみ**使用。RB 無し（prev_exp 直投入）。
- `PipelineTrainRunner`: DoStep 冒頭で前回 learn を待ち（[`trainer.cpp:587`](../../core/anet-core/src/trainer.cpp:587)）→ learn を `learn_pool_`（PinnedThreadPool 1本、[`trainer.cpp:546`](../../core/anet-core/src/trainer.cpp:546)）へ enqueue（[`trainer.cpp:625`](../../core/anet-core/src/trainer.cpp:625)）→ **env.Step は learn の裏で実行**（[`trainer.cpp:633`](../../core/anet-core/src/trainer.cpp:633)）。critical path ≈ max(decode, learn)。
- `prev_exp_` は `env.Step` **後**に `state_.Clone()`＋`result->next_state.Clone()`（[`trainer.cpp:642`](../../core/anet-core/src/trainer.cpp:642)）→ その後 `state_ = result->continue_state`（[`trainer.cpp:654`](../../core/anet-core/src/trainer.cpp:654)）。**`state_(=前 continue_state)` は clone まで生存が必要**＝出力バッファ再利用は double-buffer 必須。
- `AccumulateAndNotifyEpisodeEnd`（[`trainer.cpp:111`](../../core/anet-core/src/trainer.cpp:111)）は `result->next_state.done|truncated`[B]（[`trainer.cpp:127`](../../core/anet-core/src/trainer.cpp:127)）で per-env 終端検出し **終端 lane ごとに** `EpisodeEndEvent` 発火（[`trainer.cpp:139`](../../core/anet-core/src/trainer.cpp:139) 付近）。shape assert `[num_envs]`（[`trainer.cpp:129`](../../core/anet-core/src/trainer.cpp:129)）。
- `PinnedThreadPool`（[`thread.hpp:64`](../../core/anet-core/include/anet/thread.hpp:64)）= `Enqueue(worker_id, fn)`/`WaitAll()`。ThreadPoolDiscreteEnv の per-env 並列（[`env.cpp:475`](../../core/anet-core/src/env.cpp:475)）と同パターンで decode 並列化可（wxImage 並列 decode の実績あり）。

**eval driving / metrics**
- `EpisodeEvalObserver.OnLearn`（[`observers.cpp:539`](../../core/anet-core/src/observers.cpp:539)）が eval_interval ごとに `RunEvaluationEpisode` を起動。
- `RunEvaluationEpisode`（[`observers.cpp:514`](../../core/anet-core/src/observers.cpp:514)）= `eval_runner->Sync(); do { DoStep } while(!LastStepHadEpisodeEnd());`（**1 個でも lane が終端したら止まる**）。
- `MetricsLogEpisodeEndObserver`（[`observers.hpp:380`](../../core/anet-core/include/anet/observers.hpp:380)）は `EpisodeEndEvent` ごとに **env 全体（env_index 不使用）の scalar を記録＋EMA 前進** → 1 pass で B event を出すと **B 重複**（C5 で lane0 のみ done にする理由）。eval config は `$runner`/`$env` の scalar 名で metric 源泉を指定（[`ImageCls.txt:495`](../../apps/runner/config/ImageCls.txt:495)）。
- `EvalRunner` は `BatchEnv` を駆動（[`trainer.hpp:88`](../../core/anet-core/include/anet/trainer.hpp:88)）、`DoStep` は env.Step→AccumulateAndNotifyEpisodeEnd→state_=continue_state（[`trainer.cpp:285-315`](../../core/anet-core/src/trainer.cpp:285)）。

**データ供給 / view / config**
- `ImageDataSource`（[`ImageData.hpp:19`](../../core/envs/imagecls1/src/ImageData.hpp:19)）は `torch::data::datasets::Dataset` 継承だが **DataLoader 使用箇所はゼロ**（`get()` は手動 [`ImageClsEnv.cpp:100`](../../core/envs/imagecls1/src/ImageClsEnv.cpp:100)）→ Dataset 基底は撤去可。
  parse は malformed line / unknown class を **silent skip**（[`ImageData.hpp:72`](../../core/envs/imagecls1/src/ImageData.hpp:72)）。`labels_`=per-image class ID（長さ N_img [`ImageData.hpp:87`](../../core/envs/imagecls1/src/ImageData.hpp:87)）、`classes_`=per-class 名（長さ N_class [`ImageData.hpp:56`](../../core/envs/imagecls1/src/ImageData.hpp:56)）。
- 現行 sampling は `rnd_->RandUint64() % size` の**復元抽出・epoch 無し**（[`ImageClsEnv.cpp:99`](../../core/envs/imagecls1/src/ImageClsEnv.cpp:99)）。augment は env 内 train-only（[`ImageClsEnv.cpp:105-107`](../../core/envs/imagecls1/src/ImageClsEnv.cpp:105)、022 の判断）。
- `ImageClsView` は experience の batch[0] を表示（[`ImageClsView.cpp:253-260`](../../core/envs/imagecls1/src/ImageClsView.cpp:253)）、`value_labels` は GetSpec 由来（[`ImageClsView.cpp:245`](../../core/envs/imagecls1/src/ImageClsView.cpp:245)）。ImageCls は **AuxData を消費しない**（GetAuxDataList を使うのは LunarLander/DropMerge のみ）。
- 既存テスト（[`ImageClsEnv_test.cpp:182`](../../core/envs/imagecls1/src/ImageClsEnv_test.cpp:182)）は **terminal `next_state.episode_start=false`／auto-reset 後 `continue_state.episode_start=true`** を期待。
- `ImageClsEnvConfig`（[`ImageClsEnv.hpp:15`](../../core/envs/imagecls1/src/ImageClsEnv.hpp:15)）= root/list/classes/image_wh/max_steps/augment.*。Food101 は 224×224、train 75,750 / test 25,250。
- eval 設定は `train.eval.[tag].*`（interval/use_background/clone_model 等を trainer が読む [`trainer.cpp:793-813`](../../core/anet-core/src/trainer.cpp:793)、env override は `train.eval.[tag].env` prefix）。
- tests は concrete 直接 new（`ImageClsEnv_test.cpp` / `image_cls_agent_test.cpp` の `MakeImageClsEnvSpec()`）で registry 非依存。

## 設計方針

### C0. Factory seam / registry / builder

- `rl.hpp`: **旧 `BatchEnvFactory` IF（top-level dispatch、単一実装）を削除**し、**新 `BatchEnvFactory`（per-class abstract factory IF）** を追加:
  ```cpp
  class BatchEnvFactory {
  public:
      virtual std::shared_ptr<BatchEnv> CreateBatchEnv(
          const ConfigData& config_data, const torch::Device& device,
          std::optional<seed_t> seed, int num_envs, const std::string& config_prefix) = 0;
      virtual std::string GetTargetEnvClassId() const = 0;
      virtual ~BatchEnvFactory() = default;
  };
  ```
  `PlainBatchResetResult`/`PlainBatchStepResult`（`GetAuxDataList` が空 vector を返す最小 concrete。直接 `BatchEnv` 実装が使う）。
- `env.hpp`/`env.cpp`:
  - `DefaultBatchEnvFactory`（+Config）→ **`BatchEnvBuilder`（+`BatchEnvBuilderConfig`）** に改名（単一 concrete、IF 無し）。
  - `EnvRepository` 値を `std::variant<shared_ptr<SingleDiscreteEnvFactory>, shared_ptr<BatchEnvFactory>>`、`RegistEnvFactory` を両型 overload。同一 class_id 二重登録は `LOG::warn` 後 throw。
    **事前（Phase 0）に `ANET_REGISTER_ENV_FACTORY` マクロ（定義＋使用4箇所）を削除し `Init*()` 登録に一本化**（throw が既存 env 起動で発火しないための前提）。
  - `BatchEnvBuilder::CreateBatchEnv(seed, num_envs, config_prefix="")`: entry を引き、**batch factory → `factory->CreateBatchEnv(config_data_, device_, seed, num_envs, config_prefix)`**、
    **single factory → 従来 Vectorized/ThreadPool wrap**（worker_type 分岐はこの経路のみ、prefix も渡す）。
  - 無状態 mixin `WorkerThreadResolver`（`protected int GetLogicalCores() const; protected int ResolveWorkerThreads(int num_envs, int worker_threads_mode) const;`）を抽出。`BatchEnvBuilder` と source が継承。`CreatePool` は `BatchEnvBuilder` 側に残す。
- `trainer.cpp`: `env_factory_` 型を `unique_ptr<BatchEnvBuilder>` に。eval env 構築（817-818）を `env_factory_->CreateBatchEnv(eval_env_seed, eval_batch_size, config_prefix)` へ置換。
  実装時に builder の `device_` と `env_device` の一致を確認（ずれるなら device override 引数を検討）。
- `ImageCls.cpp`: `RegistEnvFactory(make_shared<ImageClsEnvFactory>())`（batch factory 版に差替。View 登録は不変）。

### C1. データ供給 source（standalone コンポーネント）

`ImageDataSource` を分解し、**1 データセット単位の供給コンポーネント**（`ImageDataSource` 名踏襲）に再編。`NextBatch(B, mode) → {grid:[B,3,H,W] uint8, targets:[B] int64}`。内部構成:

- **`ImageManifest`**（immutable）: classes.txt + list.txt を parse。`targets`（per-image class ID、旧 `labels_`）＋
  `class_names`（per-class 名、旧 `classes_`）。`class_names` を GetSpec の `value_labels` 供給元に。RNG 無し。global cache 無し。
  **malformed line / unknown class は fail-fast**（list 行番号・class 名・path を含む `ANET_SYSTEM_ERROR`。現行 silent skip [`ImageData.hpp:72`] を廃止）。
- **sampler**（RNG）: `IndexSampler`(IF)＋`EpochShuffleSampler`(train=共有 shuffle epoch cursor＋wrap)＋`SequentialPassSampler`(eval=全件＋pad)。source seed から導出。
- **`DecodeResizedImage`**（pure I/O、free 関数）: `LoadFile→Scale→[3,H,W] uint8`。`torch::data::datasets::Dataset` 基底は撤去。RNG 無し。
- **`ApplyTrainAugment`**（train transform、free 関数、seed 引数）: `decode(cache) → augment(per-sample, RNG) → collate`。022 の「場所=env」を「場所=source の transform 層」へ更新（原則: manifest/decode は純粋、RNG は transform/sampler 層は維持）。
- **cache**（C2）: `SampleCachePolicy`（下記）。

env は train source（`EpochShuffleSampler`）＋eval source（`SequentialPassSampler`）を保持し `RunMode` で選択。cache は **lazy alloc**（未使用 mode の無駄なし）。

### C2+C3. データパイプライン（並列 decode ＋ cache）

- **prefetch は作らない**。NextBatch は env.Step 内で走り既存 overlap に乗る。B 個の index を得たら **batch 内で dedupe**
  （wrap や `data_size<B` で同一 dataset index が同一 batch 内に再登場し得るため）、unique idx を `PinnedThreadPool` で並列 decode/cache し、各 batch slot は該当 idx から copy。
  → unique fill なので cache への**同時書き込み race が無い**。pool sizing は `WorkerThreadResolver`。
  注: 同一 batch 内の重複 slot は同一 `(epoch, dataset_index)` seed のため **augment も同一結果**になる（wrap 端数由来で頻度極小・許容仕様）。
- **出力は double-buffer（ping-pong 2枚）**。`continue_state`(=次 `state_`) のバッファは、次 Step の `state_.Clone()`（[`trainer.cpp:642`](../../core/anet-core/src/trainer.cpp:642)）が済むまで生存させる契約。
  単一バッファ再利用は `env.Step(t)` が `state_(t-1)` を上書きし clone を汚すため不可。`next_state` は同 Step 内で即 Clone されるので単一で可。collation は decode task が batch slot `[i]` へ直書き（slot は distinct）。
- **`SampleCachePolicy` strategy**（`cache.mode = auto | none | full_ram`）:
  - `NoCachePolicy`: 毎回 decode（ImageNet train 標準、GPU 計算が decode を隠す）。
  - `FullRamCachePolicy`: source ごと `[N,3,H,W] uint8` を1本、epoch1 で lazy fill（**dedupe 済み unique idx** を書くので distinct、thread-safe）、epoch2+ は copy。key=dataset index、**pre-augment** 保持（augment は cache 後）。
  - **`auto`（既定）**のみ size 判定：`cache.max_bytes` に収まれば `FullRamCachePolicy`、超で `NoCachePolicy`（`LOG::warn`）。**明示 `full_ram` の cap 超過は `ANET_SYSTEM_ERROR`**（fail-fast、明示設定を勝手に落とさない）。明示 `none` は none。
  - **future strategy（seam のみ・未実装）**: `MmapCachePolicy`（pre-decoded uint8 の mmap ファイル、FFCV 類）/ `PreprocessedFileCachePolicy`。
- スケール: Food101 train 10.6GiB/eval 3.5GiB（224²uint8）、ImageNet-1K train ~180GiB（RAM 不可→none）/val ~7GiB（full 可）。
- **wxImage 前提**: 並列 decode 前に image handler の初期化を一度だけ済ませる（handler 登録 race 回避）。並列 decode＋cache fill は stress test で確認（Phase 2）。

### C4. ImageClsEnv（RL scaffolding 薄層）

- `BatchEnv` 直接実装（`DiscreteBatchEnvBase` は継承しない＝`envs_` 由来 fan-out 不要）。GetSpec/GetBatchSpec/GetDevice は single 踏襲（obs grid/vector、num_envs=B）。result は `PlainBatch{Reset,Step}Result`。
- Reset/Step: source から NextBatch を引き `state.obs` を構成。`current_labels_[B]` を保持し `reward[i]=(action[i]==label_i)`。
- **train mode**: global step counter で max_steps ごとに境界。**max_steps は「可視化/ログ cadence 専用の metrics 窓」**（データ非依存。役割＝Conv2d 可視化頻度・train 側 EpisodeEndEvent（`$runner train_episode_reward` が依存）・`$episode_step` 軸の供給。accuracy 系キーとは無関係）。
  境界 Step は **`next_state.done=true, episode_start=false`（terminal）**、
  **`continue_state.done=false, episode_start=true`（auto-reset 後の新 episode）**（既存テスト [`ImageClsEnv_test.cpp:182`] の契約。Conv2d metrics image が episode_start をキーにするため維持）。
  auto-reset は window リセットのみ（cursor 連続、次バッチを引く）。`epoch_count` は cursor wrap で++。
- **GetScalar は global 2キーのみ**（stream キー廃止、index 引数は不使用）:
  - `accuracy` ＝ **直近に確定した採点サイクルの正解率**。train のサイクル＝epoch（wrap で確定。初回 wrap 前は NaN。epoch 中もモデルが更新されるため「epoch 平均性能」であり、eval の固定モデル精度とは意味が微妙に異なる点を注記）。eval のサイクル＝eval 1回分（C5）。
  - `epoch_count` ＝ train: wrap 回数／eval: pass（all）・cycle（subset）回数。
  - 上記以外のキー、および global キーへの prefix 付与（`mean.accuracy` 等）は **`ANET_SYSTEM_ERROR`**（config typo の即死検出）。
  - **snapshot 契約**: accumulate→サイクル境界で snapshot＋reset。GetScalar は常に snapshot を読む（EpisodeEndEvent 後の読み出しとクリアが衝突しない）。
    train の境界確定は「per-sample epoch tag 別 accumulator の total==data_size」（wrap 混在 batch でも順序仮定不要。epoch tag は sampler が付与）、eval は「pass_target 分の採点完了」（C5）。
  - 命名の経緯: 無印 `accuracy` は当初 stream キーの prefix 必須規則と衝突するため不採用予定だったが、**stream キー全廃で衝突相手が消えたため global キーとして採用**。`batch_accuracy` は batch=B（1 Step の枚数）と衝突するため不採用。
- framework 側: `DiscreteBatchEnvBase::GetScalar`（wrap 環境用）の「無 prefix→WARN+mean fallback」を**廃止し `ANET_SYSTEM_ERROR`**（mean./max./min. 必須。subkey 不明時の nullopt 伝播は維持）。
  config 全数 audit（prefix 無し使用ゼロ）・テスト依存ゼロのため実質挙動不変（Phase 0、in-place 変更。`AggregateBatchScalar` free 関数の抽出は ImageCls が集計を使わなくなったため**不要**）。
- decode pool/Shutdown を持つ（source 内）。出力バッファは double-buffer（上記）。

### C5. batched eval

- **sampler は eval_samples で切替**:
  - `eval_samples=all`（既定）: `SequentialPassSampler`（毎 pass index 0 から deterministic の full pass）。**末尾 `eval_size%B`** は pad-to-B ＋ valid-prefix count（valid counter を eval_size で頭打ち、explicit mask 不要）。pad step の **`n_transitions`=valid count（B でなく実サンプル数）**。exact 全件・無バイアス。
  - `eval_samples=N<all`: **ローテーション**。`EpochShuffleSampler`（train と同一クラス、eval 専用 seed）を **eval 呼び出しをまたいで永続するカーソル**として使い、1 回の eval で **ceil(N/B)×B 枚（B 単位切り上げ）**を消費・採点。
    wrap-fill のため **pad 不要・全 batch valid**。ceil(eval_size/N) 回の eval で全件を一様被覆し、cycle ごとに reshuffle。固定部分集合は「同じ画像だけが使われ続け残りが未評価」になるため不採用。
- **pass 終端は両モード共通機構**: pass_target（all→eval_size、subset→切り上げ後 N）分を採点した Step で pass end。**代表 lane(0) のみ `done=true`**（他 lane false）。
  `AccumulateAndNotifyEpisodeEnd` は 1 個の `EpisodeEndEvent` を発火（B 個バースト＝`MetricsLogEpisodeEndObserver` の B 重複を回避）、`RunEvaluationEpisode` の do-while は `LastStepHadEpisodeEnd()` で抜ける（**observer/runner 無改修**）。
- **accuracy は eval 1回分の snapshot**: env が pass の correct/total（all の pad 除外）を集計し、pass end で snapshot→`GetScalar("accuracy")` が返す（C4 のキー体系・snapshot 契約）。
  eval metrics config は **`$env accuracy`** を読む（`$runner eps_total_reward` は代表 lane 方式では lane0 のみの部分値になるため eval では使わない→config から削除）。
- sampler の pass-end reset／カーソル前進は done mask と独立に env 内部で実施（次 eval も deterministic）。**pass-end Step の `continue_state` は次 pass の先頭バッチ**
  （train の auto-reset と同型。sampler は 1 バッチ先読みになるが pass 会計はズレない）。
- **時系列解像度**: accuracy の記録頻度は `train.eval.[tag].interval`（learn step 基準）が決め、pass 長は点の間隔に影響しない（background 実行）。
  1 点の質は旧（ランダム 100 枚）→ 新 all（全件 exact）で大幅向上。eval が重い場合は「**頻度=interval、1 点の質=eval_samples**」で独立調整。
- eval は C0 再レイヤで `env_factory_->CreateBatchEnv(eval_seed, eval_batch_size, config_prefix)` 経由。

### C6. config surface

- 名前空間は `ImageClsEnv.*` 踏襲（既存 config 互換）。追加:
  - `ImageClsEnv.cache.mode` = `auto` | `none` | `full_ram`（future: mmap/preprocessed_file）。既定 **`auto`**。
  - `ImageClsEnv.cache.max_bytes` = `auto`/`full_ram` の RAM 上限（既定 ~4GiB）。**`auto` は超過で none に落ち、明示 `full_ram` は超過で `ANET_SYSTEM_ERROR`**。
  - decode 並列度は既存 `env.worker_threads`（`WorkerThreadResolver`）を流用（新キー追加しない）。
- **eval config path 固定**:
  - `eval_batch_size` = **`train.eval.[tag].eval_batch_size`（trainer が top-level eval key として読む → CreateBatchEnv num_envs）**。
  - `eval_samples` = **env config**（`train.eval.[tag].env` prefix 経由で env が読む）。既定 all=full pass。`N<all` は**ローテーション**（C5。B 単位に切り上げ）。**`eval_samples>eval_size`→eval_size に clamp（WARN）、`eval_samples<=0`→all**。
  - **max_steps と分離**（eval は max_steps を使わない）。
- `cache.max_bytes` は **source ごと**の上限（train/eval source で独立に判定）。
- **metrics config 差し替え（Phase 3）**: 出力タグは維持し、ソースキーを新キー体系へ。**各行の行末コメントで定義を明記し、実 config にもそのまま入れる**:
  ```
  metrics.scalar.[42_env/04_accuracy_mean]     = $env accuracy @train $exp_step                               # 直近確定epochのtrain正解率(wrap毎更新/初回wrap前NaN/epoch中もモデル更新=epoch平均性能)
  metrics.scalar.[42_env/05_accuracy_mean_ema] = $env accuracy @train $exp_step $ema ema_alpha:0.001          # 上のEMA
  metrics.scalar.[42_env/07_epoch_count]       = $env epoch_count @train $exp_step interval:100               # データセット周回数(shuffled cursorのwrap回数)
  metrics.scalar.[51_eval1/03_accuracy]        = $eval.[eval1] $env accuracy @episode_end                     # 直近eval1回分の正解率(all=テスト全件exact/subset=ローテーションのチャンク分)
  metrics.scalar.[51_eval1/04_accuracy_ema]    = $eval.[eval1] $env accuracy @episode_end $ema ema_alpha:0.01 # 上のEMA(subset時のチャンク差を平滑化)
  ```
  - `21_eval/01,02`（`$runner eps_total_reward`）は**削除**（代表 lane 方式では lane0 のみの部分値になり誤解を招く。51_eval1 に一本化）。
  - `20_eps/10,11`（`$runner train_episode_reward`）は**無改修**（runner 側キー。episode 機構は温存され従来どおり動く）。
  - コメントアウト中の `42_env/02,03`（`mean.reward_sum`）は**削除**（stream キー廃止）。
- 既存 ImageClsEnvConfig（root/list/classes/image_wh/max_steps/augment.*）は不変。

### C7. RNG / 再現性

- source seed から `SeedMaker` で **sampler RNG と augment RNG を別 stream**に導出（augment ON/OFF が sample 順を乱さない）。
- epoch permutation は `(sampler_seed, epoch)` 由来で決定的・再現可能。eval は all=sequential（RNG 不使用）、subset=eval 専用 seed の `EpochShuffleSampler`（決定的・eval 呼び出し間で永続）。
- augment は **per-sample 決定的 seed**（`(augment_seed, epoch, dataset_index)` 由来の thread-local RNG）。並列 decode 下でも thread 順非依存。
- decode は pure（index 割当）で並列でも order 非依存＝決定的。dedupe された unique idx の割当も index 決定的。
- **新 reproducibility 契約**: central batch RNG により同 seed でも旧 per-env run と bit 一致しない。「同 seed・同 config で新 contract として再現保証」。network 決定性は既存 ADR 0006（determinism flag）と直交。

## 実装フェーズ

> 各フェーズは**独立してビルド緑＋テスト可能**な単位。名称変更が framework に及ぶため、まず挙動不変の refactor（Phase 0）で足場を固めてから seam / ImageCls を積む。

### Phase 0 — framework refactor（実質挙動不変の準備）
- 旧 `BatchEnvFactory` IF 削除、`DefaultBatchEnvFactory`(+Config) → `BatchEnvBuilder`(+`BatchEnvBuilderConfig`)（concrete 単一）。`trainer` の型追従（`unique_ptr<BatchEnvBuilder>`）。
- `WorkerThreadResolver` mixin 抽出（`GetLogicalCores`/`ResolveWorkerThreads` 移設）。
- **`ANET_REGISTER_ENV_FACTORY` 撤去**: 使用4箇所（GridMaze/LunarLander/CartPole/DropMerge）＋マクロ定義（env.hpp）を削除し `Init*()` 登録に一本化（Phase 1 の二重登録 fail-fast の前提）。
- **`DiscreteBatchEnvBase::GetScalar` の無 prefix fallback 廃止**: WARN+mean → `ANET_SYSTEM_ERROR`（mean./max./min. 必須）。config 全数 audit（prefix 無し使用ゼロ）・テスト依存ゼロ確認済みのため実質挙動不変。
- 対象 C: C0（命名・抽出部）＋C4（framework 側 GetScalar）。
- 検証: 全 env（CartPole/LunarLander/DropMerge/GridMaze/ImageCls 現状 single）が従来どおり train/eval で回る。既存テスト緑。

### Phase 1 — seam 追加（single 経路で検証）
- 新 `BatchEnvFactory`(per-class IF)＋`PlainBatch{Reset,Step}Result`。`EnvRepository` variant 化＋`RegistEnvFactory` overload＋二重登録 fail-fast。
- `BatchEnvBuilder::CreateBatchEnv` に `config_prefix` 追加＋dispatch（batch factory 直接／single wrap）。eval を builder 経由へ（`trainer.cpp:817` 置換、FLAG 解消）。
- 対象 C: C0（seam）＋C5（eval routing）。ImageCls は**まだ single factory 登録のまま**。
- 検証: 既存 env が variant/dispatch/eval-routing 経由で不変動作（eval も builder 経由で回る）。ImageCls も現状 single のまま緑。

### Phase 2 — ImageCls データ供給 source（env 非依存で単体テスト）
- `ImageData.hpp` 作り替え: `ImageManifest`（fail-fast validation）、`DecodeResizedImage`(free)、`ImageDataSource`(batch source, `NextBatch`, double-buffer, dedupe)。
  `IndexSampler`+`EpochShuffleSampler`+`SequentialPassSampler`、`SampleCachePolicy`+`NoCachePolicy`+`FullRamCachePolicy`（`auto`/fail-fast）、`ApplyTrainAugment`(free, per-sample seed)。
- 対象 C: C1＋C2＋C3＋C7。
- 検証: **source 単体テスト**（manifest 件数＋malformed/unknown fail-fast、sampler epoch/wrap/deterministic、eval all=sequential+pad／subset=ローテーション（呼び出し間カーソル永続・cycle 全件被覆・B 切り上げ）、decode 形状、cache fill/hit/`auto` fallback/明示 full_ram 超で error、
  **並列 decode＋重複 idx dedupe＋cache 同時 fill の race stress**、double-buffer 契約、augment 決定性、wxImage handler 事前初期化）。env 不要。

### Phase 3 — ImageClsEnv 作り替え＋eval＋旧削除
- `ImageClsEnv` を `SingleDiscreteEnv`→`BatchEnv`（source を薄く包む、reward/done/episode_start(terminal=false/reset=true)/GetScalar(`accuracy`=サイクル snapshot＋`epoch_count`、他は SYSTEM_ERROR)、`PlainBatch*Result`）。
  `ImageClsEnvFactory` を `BatchEnvFactory` 版に。旧 single result 削除、`ImageCls.cpp` を batch 登録に。config（`cache.*`/`eval_batch_size`/`eval_samples`＋metrics 行差し替え・行末コメント）。
- 旧 single `ImageClsEnv` 実装・test を Batch(B=1) へ移行（C8）。eval pass＋代表 lane done（C5）。
- 対象 C: C4＋C5（env eval）＋C6＋C8。
- 検証: B=1 superset（旧 single 等価。GetScalar は新キー体系のため旧キー互換は対象外）、train が回る、eval が pass 駆動（all=eval set 全件・EpisodeEndEvent 1個・accuracy 毎 eval 1回／subset=ローテーション被覆）、Food101 学習曲線・eval accuracy が旧相当（±ブレ幅）、メモリ 2N→2 ソース。

## 影響ファイル

| ファイル | 変更 | Phase |
|---|---|---|
| `core/anet-core/include/anet/rl.hpp` | 旧 `BatchEnvFactory` IF 削除／新 `BatchEnvFactory`(per-class IF) 追加／`PlainBatchResetResult`/`PlainBatchStepResult` | 0(削除),1(新規) |
| `core/anet-core/include/anet/env.hpp` | `DefaultBatchEnvFactory`→`BatchEnvBuilder`(+Config)、`WorkerThreadResolver` 宣言、`ANET_REGISTER_ENV_FACTORY` マクロ削除／`EnvRepository` variant＋`RegistEnvFactory` overload | 0／1 |
| `core/anet-core/src/env.cpp` | `BatchEnvBuilder` 改名／`ResolveWorkerThreads`/`GetLogicalCores` を base へ／GetScalar 無 prefix fallback を SYSTEM_ERROR 化／variant 登録＋dispatch＋config_prefix | 0／1 |
| `core/envs/{gridmaze1,lunarlander1,cartpole2,dropmerge1}/src/*Env.cpp` | `ANET_REGISTER_ENV_FACTORY` 使用行の削除（`Init*()` 一本化） | 0 |
| `core/anet-core/src/trainer.cpp` | `env_factory_` 型追従（0）／eval を `env_factory_->CreateBatchEnv(..., eval_batch_size, config_prefix)` へ（1） | 0／1 |
| `core/envs/imagecls1/src/ImageData.hpp`（作り替え） | `ImageDataSource`→batch source(`NextBatch`, double-buffer, dedupe)／`ImageManifest`(fail-fast, `labels_→targets`/`classes_→class_names`)／`DecodeResizedImage`／`Dataset` 基底撤去 | 2 |
| `core/envs/imagecls1/src/`（新規ファイル） | `IndexSampler`+`EpochShuffleSampler`+`SequentialPassSampler`、`SampleCachePolicy`+`NoCachePolicy`+`FullRamCachePolicy`、`ApplyTrainAugment`(free) | 2 |
| `core/envs/imagecls1/CMakeLists.txt` | 新規ファイルの追加 | 2 |
| `core/envs/imagecls1/src/ImageClsEnv.{hpp,cpp}`（作り替え） | `ImageClsEnv` を `SingleDiscreteEnv`→`BatchEnv`（episode_start/eval done/`accuracy`=サイクル snapshot/`epoch_count`）／`ImageClsEnvFactory` を `BatchEnvFactory` 版／旧 single result 削除 | 3 |
| `core/envs/imagecls1/src/ImageCls.cpp` | `RegistEnvFactory(make_shared<ImageClsEnvFactory>())`（batch 版、View 登録不変） | 3 |
| `core/envs/imagecls1/src/ImageClsEnv_test.cpp` | source 単体テスト（2）＋ ImageClsEnv(B=1)/eval metrics テスト（3） | 2／3 |
| `apps/runner/config/ImageCls.txt` | `cache.mode`(auto)/`cache.max_bytes`、eval の `eval_batch_size`(top-level)/`eval_samples`(env)、metrics を `$env accuracy`/`epoch_count` へ差し替え（行末コメント付き）＋`21_eval/01,02`・`42_env/02,03` 削除 | 3 |
| `docs/adr/0009-imagecls-batch-env-seam.md` | seam 決定 ADR | — |

## 受け入れ基準

1. 各 Phase 末でビルド緑（x64-Debug）＋既存テスト緑。Phase 0/1 で他 env（CartPole/LunarLander/DropMerge/GridMaze）が不変動作。
2. seam: `class_id="ImageClsEnv"` が batch factory を引き `CreateBatchEnv` が `ImageClsEnv`(BatchEnv) を返す。他 env は single→Vectorized/ThreadPool。二重登録は fail-fast（**マクロ撤去済みのため既存 env 起動では発火しない**）。旧 `BatchEnvFactory` IF が消え `BatchEnvBuilder` が concrete 単一。
3. env: `Reset`/`Step` が `grid[B,3,H,W] uint8`＋`vector[B,1] int64` を返す。`reward[i]=(action==target)`。train は max_steps ごと **terminal `next_state.episode_start=false` / reset `continue_state.episode_start=true`**。`GetScalar` は **`accuracy`（直近確定サイクル snapshot。train は wrap 境界で確定・初回 wrap 前 NaN）＋`epoch_count` のみ**、不明キー・prefix 付き global キーは `ANET_SYSTEM_ERROR`（旧 stream キーは廃止）。
4. **B=1 superset**: `ImageClsEnv`(B=1) が旧 single と同 shape/label/reward/episode_start（GetScalar は新キー体系のため旧キー互換は対象外）。
5. **double-buffer**: 出力バッファが double-buffer で、`state_.Clone()`（prev_exp）が **action を作った元 state** を保持（次バッチで上書きされない）。
6. sampler: train 共有 epoch cursor が非復元・全件被覆・wrap 端数・`epoch_count`。**同 seed で sample 列一致**。eval all=sequential が全件 deterministic＋pad valid-prefix で **accuracy が全件 exact**（pad step の `n_transitions`=valid count）。eval subset=ローテーションが **eval 呼び出し間でカーソル継続・cycle で全件を一様被覆・B 単位切り上げ**（同 seed で schedule 一致）。
7. **cache/race**: `FullRamCachePolicy` が epoch1 fill→epoch2 hit（同値）。**batch 内重複 idx を dedupe し同時 fill が race しない**（TSan/stress 緑）。`cache.mode=auto` は cap 超で none、**明示 `full_ram` は cap 超で `ANET_SYSTEM_ERROR`**。
8. **eval metrics**: 1 pass で `EpisodeEndEvent` **1個**、eval accuracy が **pass 単位 snapshot で毎 eval 1回記録**（B 重複せず EMA も1回前進。all=全件 exact）。config は `$env accuracy`（行末コメントで定義明記）。
9. determinism: 同 seed+config で sample 列＋augment が run 間一致（並列 thread 順非依存）。
10. manifest: malformed line / unknown class で **行番号・class・path 付き fail-fast**。
11. train 学習: 既存 Food101 config で学習曲線・eval accuracy が旧 single 相当（±終盤ブレ幅）。メモリが 2N ソース→2 ソースへ削減。

## 正直なリスク

- **同期 episode バースト（train）**: 全 B が max_steps ごとに同時終端し `EpisodeEndEvent` が B 個バースト発火。既存機構は平均で吸収し observer は interval 判定だが、per-episode 動作の observer（Conv2d/video）で cadence 変化がないか要確認。（eval は代表 lane で回避済み。**現行の N env wrap でも全 env 同時 Reset→同時 max_steps 到達で B 個バーストしており、新規の悪化ではない**。）
- **double-buffer の実装ミス**: ping-pong を誤ると `prev_exp_.state` が次バッチを掴む。受け入れ基準5で明示検証。
- **cache fill race**: dedupe 漏れ／per-index 未保護だと epoch1 fill が data race。dedupe→unique fill を stress test（受け入れ基準7）。
- **wxImage 並列 decode**: `LoadFile/Scale` 並列は `ThreadPoolDiscreteEnv` で実績ありだが、handler 登録 race 回避のため並列前に handler 初期化を一度だけ。新 source で stress。
- **full_ram の既定 cap**: `auto` 既定 ~4GiB は Food101 eval(3.5)は載り train(10.6)は none。マシン RAM 依存で cap 値は要調整（大 RAM は引上げ、明示 full_ram なら fail-fast で気付ける）。
- **eval pad の valid counter**: 末尾バッチの valid 数を eval_size で頭打ちする実装ミスは accuracy を汚す。counter は source 側で一元管理、`n_transitions`=valid。accuracy のクリア忘れ／早すぎるクリアは **snapshot 契約**（境界で snapshot→reset、GetScalar は snapshot を読む）で構造的に排除。
- **新 RNG 契約**: 旧 run と bit 非一致。等価性は「同等設定の複数 seed の終盤平均ブレ幅」で判断（構成比較はブレ幅基準）。
- **EvalRunner の TrainEvent は浅参照**: eval の `BatchExperience` は Clone 無し（[`trainer.cpp:306`](../../core/anet-core/src/trainer.cpp:306)。Pipeline/Serial は Clone 済み）。Notify は同期なので即時読みは安全だが、**eval scope の TrainEvent 購読者がテンソル参照を保持すると double-buffer 再利用で中身が差し替わる**。Phase 3 で購読者の非保持を確認。
- **ローテーション eval のデータ抽選ノイズ**: subset では eval ごとに chunk が異なる（既設 EMA が平滑化）。同 seed で schedule は決定的だが、**eval interval を変えると同 learn_step で読む chunk が変わる**（run 比較はブレ幅基準で吸収）。
- **キー体系の fail-fast**: 旧 config（`mean.accuracy` 等）をそのまま流用すると最初の metrics 読みで `ANET_SYSTEM_ERROR`（意図した typo 検出。config 差し替えは Phase 3 に含む）。
- **decode 律速**: 軽量モデル/高速 GPU では並列 decode が learn を超え律速化しうる。その場合の解は future `MmapCachePolicy`/`PreprocessedFileCachePolicy`（本 PRD 非対象）。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[image_cls]"
core\anet-core\bin\Debug\anet-core-test.exe "[env]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

- 機能確認: runner を online で起動し、MODEL SHAPE DUMP と batch obs `[B,3,224,224]`、eval が pass 駆動（all=eval set 全件、accuracy 毎 eval 1回記録）で `$env accuracy` を出し、train 側 `accuracy` が初回 wrap 後に階段更新されることを確認。
- perf/精度（exp_step_per_sec / eval accuracy / host RAM）はユーザーが seed 違い複数 run の終盤平均で評価。

## 非対象（Out of Scope）

- supervised runner の全面導入（batch input が安定した後の follow-up）。
- 全 env 向け汎用 DataLoader 抽象。
- `SingleDiscreteEnvFactory` の改名（Factory 命名を保持。env.hpp:100 の Creator 化 @todo は不採用）。
- `MmapCachePolicy`/`PreprocessedFileCachePolicy` の実装（seam のみ。ImageNet train が decode 律速化した時の別 PRD）。
- eval 末尾の可変バッチ（PyTorch 式）: `batch_env_spec.num_envs` 可変化＝共有 runner/env 機構改修になるため別 PRD。
- MixUp/CutMix 可視化、mean/std normalize、新 dataset format 汎用化。
- 旧 checkpoint / 旧 run との bit 一致（新 RNG 契約）。

## 後続

1. 実装は Codex。上記 Phase 0→1→2→3 の順で 1 段ずつビルド緑を確認。
2. `MmapCachePolicy`/`PreprocessedFileCachePolicy`（ImageNet train 向け）。
3. supervised runner 化（`images+labels` を直接 learner へ）。
4. CONTEXT.md に用語（targets/class_names、epoch/episode、`accuracy`＝直近確定採点サイクル）を追記済み。
