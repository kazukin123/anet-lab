# Background eval の network snapshot 順序保証

> 状態: backlog。既存の潜在的な再現性問題を記録し、解決方式の選択と実装は後続セッションで行う。
> 関連: `docs/memo/003_eval_10prd.md` / `docs/memo/003_eval_20impl.md`、`docs/memo/910_network_lock_audit_10prd.md`、`docs/adr/0006-deterministic-algorithms.md`。
> 非関連: `docs/memo/034_imagecls_batch_input_10prd.md` の batch input / Dataset / cache 設計。本書は 034 から独立した既存 eval 基盤の課題を扱う。

## Context（背景）

`backend.deterministic_algorithms=true` は ATen 演算に決定論的アルゴリズムを選択させ、決定版が無い既知の演算を
`warn_only=false` で明示的に失敗させる。しかし、この backend 設定は OS の thread scheduling、worker の起床順、
`std::shared_mutex` の取得順、または「どの learner update 後の network を eval が参照したか」を固定しない。

現行 background eval では、`LearnEvent[n]` を契機に eval job を投入した後、TrainThread が次の learner update を投入する。
eval worker 上の `EvalRunner::Sync()` / actor forward と LearnThread 上の learner update の順序は明示的に確定されていない。
そのため、個々の forward/update が data race 無く決定論的に実行されても、eval が採点する network version は run 間で変わり得る。

本 PRD の目的は、演算レベルの決定論とは別に、background eval の **時間的順序**と **network snapshot の一貫性**を
明文化し、`LearnEvent[n]` に対応する eval が明示的に確定した単一 network version だけを評価する契約を定めることである。

本競合の適用範囲は次のとおりである。

- **background eval（`use_background=true`）固有の問題である。** foreground eval は `OnLearn()` 内で
  `RunEvaluationEpisode()` を同期実行するため、eval は次 learner update の開始前に必ず完了し、順序は既に確定している
  （[`observers.cpp:568-571`](../../core/anet-core/src/observers.cpp:568)）。foreground への切り替えは暫定回避策として機能する。
- **Runner 実装には依存しない。** eval worker が別 thread である限り、`SerialTrainRunner`（learner update を
  TrainThread 上で同期実行）でも `PipelineTrainRunner`（LearnThread_0 上で非同期実行）でも成立する。

## 現行コードで確定している事実

### 1. 関係する thread

以下は `PipelineTrainRunner` 構成を代表として記述する。

- **TrainThread**: `PipelineTrainRunner::DoStep()` を駆動する Runner 本体。前回 learn の `future.get()`、
  `LearnEvent` の同期通知、train actor forward、次 learn の投入を順番に行う
  （[`trainer.cpp:587-627`](../../core/anet-core/src/trainer.cpp:587)）。
- **LearnThread_0**: `PipelineTrainRunner` が所有する 1 worker の `PinnedThreadPool`。
  `ImageClsLearner::UpdateFromBatch()` を実行する（[`trainer.cpp:541-546`](../../core/anet-core/src/trainer.cpp:541)）。
- **EpisodeEvalObserver_0**: `EpisodeEvalObserver` が background eval 用に所有する 1 worker の `PinnedThreadPool`。
  `RunEvaluationEpisode()` を実行する（[`observers.cpp:485-519`](../../core/anet-core/src/observers.cpp:485)）。

`SerialTrainRunner` は LearnThread_0 を持たず、`ImageClsLearner::UpdateFromBatch()` を TrainThread 上で同期実行する
（[`trainer.cpp:421`](../../core/anet-core/src/trainer.cpp:421), [`trainer.cpp:502`](../../core/anet-core/src/trainer.cpp:502)）。
この構成では eval worker の競合相手は LearnThread_0 ではなく TrainThread 自身になる。

### 2. 現行の時系列

```text
LearnThread_0: Update[n] 完了
        |
        v  future.get()
TrainThread: LearnEvent[n] を同期 Notify
        |
        +-- EpisodeEvalObserver::OnLearn()
        |     `-- background eval job を EpisodeEvalObserver_0 へ enqueue
        |
        +-- train actor forward
        |
        `-- Update[n+1] を LearnThread_0 へ enqueue

EpisodeEvalObserver_0                      LearnThread_0
  EvalRunner::Sync()                 vs     Update[n+1]
  eval actor forward                 vs     Update[n+1], Update[n+2], ...
```

`EpisodeEvalObserver::OnLearn()` は eval job を enqueue した時点で戻り、worker が `Sync()` を完了するまで TrainThread を待たせない
（[`observers.cpp:539-567`](../../core/anet-core/src/observers.cpp:539)）。したがって、eval worker と次 LearnThread のどちらが先に
network lock を取得するかは未規定である。

`SerialTrainRunner` では `LearnEvent[n]` は Update[n] の同期完了直後に Notify され、Update[n+1] は次の `DoStep()` で
TrainThread 上で実行される。eval worker の競合相手が LearnThread_0 から TrainThread に変わるだけで、
`LearnEvent[n]` → eval job enqueue → Update[n+1] という順序未規定の構造は同一である。
一方、foreground eval（`use_background=false`）は enqueue ではなく同期実行のため、この競合自体が発生しない。

### 3. 現在の lock が保証する範囲

ImageCls の共有 network には `std::shared_mutex` があり、次を保証する。

- clone なし actor は `MakeAction()` の各 forward を `shared_lock` で保護する
  （[`image_cls_agent.cpp:48-73`](../../core/anet-core/src/image_cls_agent.cpp:48)）。
- learner は forward / backward / optimizer step を含む update 全体を `unique_lock` で保護する
  （[`image_cls_agent.cpp:338-390`](../../core/anet-core/src/image_cls_agent.cpp:338)）。
- clone あり actor の `Sync()` は source network を `shared_lock` で読み、clone network へコピーする
  （[`image_cls_agent.cpp:93-100`](../../core/anet-core/src/image_cls_agent.cpp:93)）。

この排他は、update 途中の parameter を actor が読むことや、同じ shared network に forward と optimizer step が同時アクセスすることを防ぐ。
一方、各 forward / update 単位の排他であり、eval pass 全体の network version や、`LearnEvent[n]` と snapshot 取得の順序までは保証しない。

## 問題

### A. `clone_model=false`: pass 内で複数 network version を参照できる

clone なし actor は learner と同じ network を共有し、各 eval batch の forward ごとに `shared_lock` を取得・解放する。
そのため、memory safety は保たれるが、batch 間に learner update が入ることを許す。

```text
eval batch 0 -> weight[n]
learner update
eval batch 1 -> weight[n+1]
learner update
eval batch 2 -> weight[n+2]
```

各 forward が決定論的でも、どの update 後の weight を参照するかが scheduling に依存するため、pass 集計値の完全再現は保証できない。

### B. `clone_model=true`: snapshot version の取得時点が未確定

clone あり actor は一度 `Sync()` すれば、その後の eval pass を単一 clone network で実行できる。
ただし現行 `Sync()` は background worker 上の `RunEvaluationEpisode()` 冒頭で呼ばれる
（[`observers.cpp:514-519`](../../core/anet-core/src/observers.cpp:514)）。

eval job 投入後に TrainThread が次 learner update を投入するため、次の両方が成立し得る。

```text
run A: Sync(weight[n])   -> Update[n+1]
run B: Update[n+1]      -> Sync(weight[n+1])
```

clone 後の pass 内一貫性は保たれるが、`LearnEvent[n]` がどの version の snapshot に対応するかは固定されない。

## 目標契約

1. `LearnEvent[n]` によって起動された snapshot eval は、明示的に確定した **単一 network version**だけを全 batch で使用する。
2. eval に使用する network version の確定は、次 learner update との順序が定義された同期境界で行う。
3. 同じ seed、config、入力、learner update 列では、同じ eval trigger が同じ network version を評価する。
4. `backend.deterministic_algorithms` は演算レベル、本 PRD の snapshot 契約は application thread の時間的順序を担当する。
5. 現行の per-forward `shared_lock` / per-update `unique_lock` による memory safety は維持する。
6. snapshot を使わず学習中 network を追従する eval を残す場合は、snapshot eval と混同しない明示的な mode / 契約として分離する。

## 解決候補（最終選択は後続セッション）

### 候補 A: TrainThread 上で snapshot を確定してから background job を投入する

- `LearnEvent[n]` の通知処理中、次 learn を投入する前に eval actor の snapshot copy を完了する。
- background worker は準備済み snapshot を受け取り、worker 内では source network から再同期しない。
- trigger と snapshot version の対応を最も直接的に固定でき、learner との overlap も snapshot 完了後は維持できる。
- clone network の所有権、複数 eval runner の同期順、snapshot copy の待ち時間を設計する必要がある。
- 現行でも `WaitBackgroundEval()` による前回 eval の完了待ちは存在するため
  （[`observers.cpp:552-553`](../../core/anet-core/src/observers.cpp:552)）、本候補が TrainThread に追加するブロックは
  snapshot copy 時間のみである。
- ただし copy 先の clone network は前回 eval が使用中であり得るため、snapshot copy は `WaitBackgroundEval()` の
  完了後に行うという順序制約が付く。

### 候補 B: clone なし経路に pass-level read lease を導入する

- eval worker が pass 開始時に shared network の read lease を取得し、pass 終了まで保持する。
- learner の `unique_lock` は eval 完了まで待つため、追加 network copy 無しで pass 内一貫性を得られる。
- eval job 投入後、次 learner update を許可する前に lease 取得完了を確認する barrier が必要。
- background thread 自体は維持できるが、eval 中は learner が停止するため学習との計算 overlap は失われる。

### 候補 C: live-weight eval を明示的な非 snapshot mode として分離する

- 現行 clone なし挙動を、学習中 network の変化を追従する別 mode として明文化する。
- snapshot accuracy と同じ metric / mode 名では扱わず、完全再現の対象外であることを明示する。
- 後方互換性は高いが、deterministic run の評価値として何を保証するかを config と metrics 上で区別する必要がある。

## Grill 候補（未決事項）

以下は本 PRD 作成時点で未決の論点であり、後続の grill セッションで確定する。

- **G1. 解決方式の選択**: 候補 A / B / C の性能・互換性・API 影響を比較し、採用方式を確定する。
- **G2. 機構を置く層**: ImageCls agent 層（actor `Sync()` 相当の snapshot copy）か、
  `EpisodeEvalObserver` / `EvalRunner` の共通層か。採用層によって適用範囲（ImageCls 限定か全 agent か）が変わる。
  `910_network_lock_audit_10prd.md` との棲み分けもここで確定する。
- **G3. 候補 A の詳細設計**: clone network の所有権、複数 eval runner 時の同期順、
  TrainThread ブロック増分（= snapshot copy 時間）の許容量。
- **G4. live-weight eval の扱い**: 候補 C を残すか。残す場合の config / metrics / mode 名での区別方法。
- **G5. 検証対象経路**: barrier テストの対象を Serial / Pipeline の両方とするか、Pipeline 代表で足りるか。

## 既存ドキュメントとの責務分担

- `003_eval_10prd.md` / `003_eval_20impl.md`: `EpisodeEvalObserver` と `EvalRunner`、`EpisodeEndEvent`、metrics の構造を扱う。
  worker 内 `Sync()` を採用した既存構造は本 PRD の出発点とする（候補 A は `Sync()` の TrainThread 側移設を含むため、
  この構造の維持までは意味しない）。
- `910_network_lock_audit_10prd.md`: network read/write、Actor clone/sync、load/save 等の排他と memory safety を広く監査する。
  本 PRD は lock の有無ではなく、eval trigger と snapshot の時間的順序へ限定する。
- ADR 0006: ATen/cuDNN の演算決定論を扱う。本 PRD は backend flag が対象としない application thread scheduling を扱う。
- `034_imagecls_batch_input_10prd.md`: ImageCls の batch-native input、Dataset/Sampler/cacheを扱う。本問題の競合構造は034以前から存在し、034の実装範囲へ含めない。

## 検証方針

### 1. 現行競合の決定的な再現

- 単調増加する `network_version` を出力できる test network / test actor / test learner と、
  少ステップで決定的に episode 終端する test env を用意する（`EvalRunner::DoStep()` は env を回して episode 終端まで走るため）。
- barrier または latch で eval worker の `Sync()` / forward と learner update の順序を制御する。
- 現行構造で次の2経路を意図的に再現する。
  - `Sync(version=n)` の後に `Update(n+1)`
  - `Update(n+1)` の後に `Sync(version=n+1)`
- sleep や確率的な race 再現には依存しない。

### 2. snapshot version の固定

- 同じ `LearnEvent[n]` から起動した eval の全 batch が同一 version を観測することを検証する。
- 次 learner update を意図的に競合させても、trigger に対応する snapshot version が変わらないことを検証する。
- 同一の seed / event 列で複数回実行し、eval trigger と version の対応列が一致することを検証する。

### 3. 互換性

- snapshot準備後も background eval の例外伝播・前回eval待機契約を維持する。
- training の learner update 順、weight列、TrainEvent / LearnEvent の順序を変更しない。
- foreground eval と RunnerFrame の EvalPanel は、採用案が直接対象にしない限り既存動作を維持する。

## 受け入れ基準

1. Serial / Pipeline 両経路のthread時系列、2種類の非決定境界、既存mutexが保証する範囲がテストとドキュメントで一致している。
2. snapshot eval は1 pass中に単一network versionだけを参照する。
3. `LearnEvent[n]` とsnapshot versionの対応が、barrierで競合順を揺らしても変化しない。
4. `backend.deterministic_algorithms`が演算決定論、snapshot機構がapplication-level順序を担当する責務分担が明示されている。
5. live-weight evalを残す場合、そのmodeはsnapshot evalと設定・ログ・metric上で区別される。
6. training weight列、foreground eval（本問題の対象外）、EvalPanelの互換条件または非対象範囲が明示されている。

## 非対象（Out of Scope）

- ImageCls batch input、Dataset/Sampler、decode、cacheの設計・実装。
- ATen/cuDNNの決定論アルゴリズム選択やADR 0006の再設計。
- eval accuracyのsample数、batch size、episode/pass終端、metrics集計方法。
- network lock全経路の包括監査、load/save、可視化callbackの排他。
- 本PRD作成時点での解決候補の最終選択およびコード実装。

## 後続

1. 本PRDを起点に、Grill 候補 G1〜G5 をgrillし、採用方式と適用範囲を確定する。
2. 採用方式確定後に実装メモを作成し、barrier制御の回帰テストから実装する。
3. 本PRDの作成後は `034_imagecls_batch_input_10prd.md` のレビューへ戻る。
