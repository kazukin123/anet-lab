# 999: train/eval mode 整理後の排他精査 PRD（草案）

## Status

- 番号: 999
- 種別: PRD draft
- 対象リポジトリ: `C:\dev\anet-lab`
- 対象領域: DQN / Rainbow / ImageCls / MuZero の Agent・Actor・Learner・NetworkModel 周辺
- 起点メモ: `C:\tmp\anet-lab-lock-audit-handoff.md`

## Context（背景）

Dropout 導入前の前提整理として、NN module の `train()` / `eval()` mode 契約を見直した。

現在の方針は、共有 network を通常 `eval` とし、Learner の学習 forward/backward/update 中だけ scoped に `train` にすること。Actor の行動選択 path では `eval()` / `train()` を呼ばず、推論は `NoGradGuard` と既存の `shared_lock` のもとで実行する。

この整理により、DQN 系では `NetworkModel::ForwardOnlineWithTrain()` が online network を一時的に `train` にする API になった。一方で、`ForwardOnline()` / `ForwardTarget()` は mode を変更しない API として扱う。

ただし、mode 切替は LibTorch module の mutable state を変更するため、排他制御と不可分である。現状は Agent / Learner / Actor のクラス構成上、lock の取得階層が一箇所に集約されておらず、次のような懸念が残る。

- `TrainingModeGuard` の呼び出しが常に `unique_lock` 内にあることを体系的に確認できていない。
- clone なし共有 network の Actor forward は `shared_lock` で守られているが、Actor 作成・clone・sync・load/save・可視化 callback などの補助経路まで同じ粒度で確認できていない。
- `DefaultDQNAgent::LoadNetwork()` は constructor 中の auto-load では安全に見えるが、runtime API として呼ばれる場合の排他契約が明確でない。
- Actor clone network は共有 source とは別インスタンスだが、`Sync()` と `MakeAction()` が同一 Actor 上で並行し得る場合、clone 側にも actor-local な排他が必要になる可能性がある。

`NetworkModel::SoftUpdate()` / `HardUpdate()` の `target_net_->eval()` は、mode 契約を曖昧にする「念のため」の修復として削除済み。本 PRD では、その後に残る排他制御の安全性を精査する。

## Problem（解くべき問題）

共有 network と actor-private clone network に対して、以下の操作がどの lock によって保護されるべきかを明確にする。

- forward read
- parameter / buffer copy
- optimizer update
- `train()` / `eval()` mode 切替
- `torch::save` / `torch::load`
- visualization / TensorFunction callback 経由の read

現状のコードで安全な経路、ライフサイクル仮定に依存している経路、実際に race し得る経路を分類し、最小修正案を決められる状態にする。

## Goals（目的）

1. DQN / Rainbow / ImageCls / MuZero について、Actor / Learner / Agent / NetworkModel の lock 契約を棚卸しする。
2. `TrainingModeGuard` を使う経路が、必ず `unique_lock` 内で実行されることを確認する。
3. clone なし Actor forward が、共有 network read として必ず `shared_lock` 内で実行されることを確認する。
4. clone あり Actor の `CreateActor()` / `Sync()` / `MakeAction()` について、source network read と clone network write/read の排他契約を確認する。
5. `LoadNetwork()` / `Save()` / visualization callback のような補助経路について、runtime 中の並行実行可否と必要 lock を明確にする。
6. 実装変更が必要な場合、lock 階層の大整理と局所修正を分けて提案する。

## Non-Goals（非対象）

- Dropout config の追加や `MultiheadAttentionOptions::dropout` 配線。
- `CustomTransformerEncoderLayer` や ViT/HybridVIT の構造変更。
- Agent / Learner / Actor クラス構成の全面整理。
- ReplayBuffer、Prefetch、CUDA transfer lifetime の排他再設計。
- MuZero の API 分離、clone 対応、MCTS バッチ化。
- lock-free 化、fine-grained lock 化、性能最適化を主目的とした設計変更。

## Current Contract（現時点の設計契約）

- 共有 network は通常 `eval` mode とする。
- Learner の学習区間だけ `TrainingModeGuard` で scoped `train` にする。
- Actor の行動選択中に `eval()` / `train()` を呼ばない。
- `ForwardOnline()` / `ForwardTarget()` は mode を変更しない。
- `ForwardOnlineWithTrain()` は mode を変更するため、呼び出し側が `unique_lock` を保持している前提とする。
- 初期化中の `eval()` は、network がまだ Actor/Learner に公開されていないため排他不要とする。
- actor-private clone に対する `eval()` は共有 network の mode 変更ではない。ただし clone 自体の read/write 並行性は別途確認する。

## Audit Targets（精査対象）

### DQN / Rainbow

対象ファイル:

- `core/anet-core/src/dqn_based_agent.cpp`
- `core/anet-core/src/dqn_based_agent.hpp`
- `core/anet-core/src/default_dqn_agent.cpp`
- `core/anet-core/src/rainbow_agent.cpp`

確認観点:

- `DefaultDQNAgent::UpdateFromBatch()` / `RainbowAgent::UpdateFromBatch()` の `unique_lock` が Learner 内の `ForwardOnlineWithTrain()`、optimizer step、target update 全体を覆っているか。
- `TDLearner` / `QRLearner` へ外部から直接到達できる bypass がないか。
- `Actor::MakeAction()` の clone なし経路が `shared_lock` 内で forward しているか。
- `CreateActor()` の `src_network->Clone(device)` が runtime 中に呼ばれ得るか。呼ばれ得る場合、source network read に `shared_lock` が必要か。
- `Actor::Sync()` が source network read を `shared_lock` で守れているか。
- `Actor::Sync()` と同一 Actor の `MakeAction()` が並行し得るか。並行し得る場合、actor-private `network_` の `CopyTo()` と forward が race しないか。
- `NetworkModel::Load()` の `eval()` は呼び出し元 lock によって守られているか。
- `DefaultDQNAgent::LoadNetwork()` を runtime API として扱うなら、model / learner load 全体に `unique_lock` が必要か。
- `Save()` / `torch::save` 経路が Actor/Learner と並行し得るか。PRD 055でRunnerからのruntime Saveを正式な経路とし、`DefaultDQNAgent::Save`のserialization全体をAgentの`shared_lock`で保護することを確定した。
- TensorFunction / visualization callback が lock scope 外へ shared network access を持ち出していないか。

### ImageCls

対象ファイル:

- `core/anet-core/src/image_cls_agent.cpp`

確認観点:

- `ImageClsLearner::UpdateFromBatch()` の `unique_lock` と `TrainingModeGuard` の scope が forward/loss/backward/clip/optimizer step を覆っているか。
- `ImageClsActor::MakeAction()` が `shared_lock` 内で forward し、mode 切替を行っていないか。
- `run_mode_` が NN mode 切替に使われていないこと。
- ImageCls に load/save 経路や clone 経路がない前提でよいか。

### MuZero

対象ファイル:

- `core/anet-core/src/muzero_proto_agent.cpp`
- `core/anet-core/src/muzero_based_agent.cpp`

確認観点:

- `MuZeroLearner::UpdateFromBatch()` の `unique_lock` と `TrainingModeGuard` の scope が initial/recurrent inference、loss、backward、optimizer step を覆っているか。
- `MuZeroActor::MakeAction()` / `MCTSEngine::Search()` が mode 切替を行っていないか。
- Actor/MCTS の `shared_lock` scope が長すぎる場合でも、correctness 上は安全か。
- MuZero は clone 非対応という現状前提でよいか。

## Required Output（成果物）

精査結果として、次の分類を含む短い日本語メモを作成する。

1. confirmed-safe paths
   - 現状の lock と lifecycle で安全と判断できる経路。
2. lifecycle-assumption paths
   - constructor 中のみ、Actor 作成時のみ、runner sequencing 前提など、コード上の lock ではなく呼び出し順に依存している経路。
3. actual race risks
   - runtime 並行実行で実際に race し得る経路。
4. minimal fixes
   - 局所修正で閉じられるもの。
5. deferred architecture cleanup
   - Agent 側への lock 集約、Actor clone lifecycle 整理など、別 PRD に分けるべきもの。

## Acceptance Criteria（受け入れ条件）

- `train()` / `eval()` / `TrainingModeGuard` の全呼び出し箇所について、排他状態が説明されている。
- shared network を読む forward / clone / copy / save / visualization 経路について、`shared_lock` または `unique_lock` の有無が説明されている。
- shared network を変更する train/eval / optimizer step / load / target update 経路について、`unique_lock` の有無が説明されている。
- actor-private clone network について、source 側排他と clone 側排他を分けて整理している。
- `DefaultDQNAgent::LoadNetwork()` の runtime 安全性について、結論または追加調査項目が明記されている。
- `Actor::Sync()` と `MakeAction()` の同一 Actor 並行可能性について、結論または runner 側調査項目が明記されている。
- 実装変更案を出す場合、最小修正とクラス構成整理を分離している。
- dropout 導入可否の議論に脱線していない。

## Suggested Investigation Commands

```powershell
rg -n "eval\(\)|train\(|TrainingModeGuard|unique_lock|shared_lock|LoadNetwork|CreateActor|Sync\(|Save\(|Load\(" core/anet-core/src core/anet-core/include

git diff -- core/anet-core/src/dqn_based_agent.cpp core/anet-core/src/default_dqn_agent.cpp core/anet-core/src/rainbow_agent.cpp core/anet-core/src/image_cls_agent.cpp core/anet-core/src/muzero_based_agent.cpp core/anet-core/src/muzero_proto_agent.cpp core/anet-core/include/anet/nn_util.hpp
```

## Validation Plan（実装修正が発生した場合）

実装変更が発生した場合のみ、以下を実施する。

```powershell
git diff --check -- core/anet-core/src/dqn_based_agent.cpp core/anet-core/src/default_dqn_agent.cpp core/anet-core/src/rainbow_agent.cpp core/anet-core/src/image_cls_agent.cpp core/anet-core/src/muzero_based_agent.cpp core/anet-core/src/muzero_proto_agent.cpp core/anet-core/include/anet/nn_util.hpp

cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'

core\anet-core\bin\Debug\anet-core-test.exe
```

## Open Questions（未確定事項）

- `CreateActor()` は学習開始後にも呼ばれ得るか、それとも runner lifecycle 上は初期化時だけか。
- `Actor::Sync()` は同一 Actor の `MakeAction()` と並行し得るか。
- `LoadNetwork()` は constructor auto-load 専用の実質 private API と見なせるか、runtime UI/API から呼ばれ得るか。
- `Save()` は runner shutdown など Actor/Learner 停止後だけか、runtime 中にも呼ばれ得るか。**PRD 055で解決済み**: toolbarからruntime中にも呼ばれ、`DefaultDQNAgent`ではAgentの`shared_lock`を取得してLearnerの`unique_lock`と排他する。
- TensorFunction / visualization callback が lock scope 外で network を再 forward する経路が残っているか。

## Draft Recommendation（草案時点の仮説）

- `TrainingModeGuard` 本体に lock を持たせるのではなく、呼び出し側の Agent/Learner lock 契約を明示する方針を維持する。
- `DefaultDQNAgent::LoadNetwork()` は runtime 呼び出し可能なら `unique_lock` で囲む候補とする。
- `CreateActor()` の clone は、runtime 呼び出し可能なら source read を `shared_lock` で囲む候補とする。
- `Actor::Sync()` と `MakeAction()` が同一 Actor で並行し得るなら、actor-private clone 用の local mutex を検討する。
- lock 所有者を Agent に寄せる大整理は、今回の精査結果を踏まえて別 PRD に分ける。
