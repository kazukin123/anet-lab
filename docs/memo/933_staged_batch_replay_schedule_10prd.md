# DefaultDQN Staged Batch / Replay Schedule 暫定 PRD

> 番号 999。backlog / 検討草案。
> 関連: [長期 Run における batch / replay 探索履歴](../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md)、[Run 全体 serialize 暫定 PRD](930_serialize_10prd.md)。
> 状態: 長期 Run で得た仮説と実装上の契約を忘れないための暫定記録。実装承認、正式設定名、既定 schedule、切替 step は未確定。

## Problem Statement

DropMerge の長期 Run では、結果的に batch size と replay ratio を段階的に変更した lineage が成立した。

- B128/RR1.25 は scratch 初期の Q / loss が安定し、初期学習の参照条件になった。
- B256/RR1 は 100M 級の scratch QR Run で品質を維持しながら、実時間効率の良い探索基準になった。
- B512/RR2 は成熟 checkpoint からの継続で、B256/RR1 より高い Eval と PER coverage を示し、最終成績優先ラインになった。
- B512/RR2 を scratch から開始した Run では Q バブル、条件付き NEET、Eval reward 低下が同期した。

この観測は「batch size は学習とともに必ず大きくすべき」という一般則を証明しない。
一方で、初期の不安定な価値推定には小さい batch / 高い update 密度、成熟後には大きい batch / 広い replay sampling が適する可能性を、将来検証する価値はある。

現在の Learner は `replay_batch_size` と `replay_ratio` を Run 中一定の値として扱う。
replay ratio mode の update credit は初期化時に一度だけ計算され、sampling、shape 検証、TD / QR / IQN loss も静的な batch size を参照する。
したがって `replay_batch_size` だけを単純に `ProfiledValue` 化すると、次の問題が起きる。

1. batch size と replay ratio の切替がずれ、optimizer update 頻度と replay sample budget が意図せず変わる。
2. 旧 phase で蓄積した update credit の単位が、新 batch size では同じ意味を持たない。
3. 1-deep prefetch 済みの旧 batch と新 phase の期待 shape が衝突する。
4. batch size 増加時に、ReplayBuffer が新 batch size を sample 可能か再確認されない。
5. loss 実装が静的 batch size を shape の正本にすると、動的 batch に追従できない。
6. Agent checkpoint を別 Run で load すると Run-local `exp_step` が再開せず、schedule が初期 phase へ戻る。

段階的 schedule を導入する場合は、値を時間変化させるだけでなく、Learner の replay budget、prefetch、shape、save / load を一つの契約として扱う必要がある。

## Solution

DefaultDQN Learner に、batch size と replay ratio を一組として解決する複合 **ReplayUpdateSchedule** を導入する案を採用候補とする。

ReplayUpdateSchedule は `constant` と `phased` を持つ。
各 phase は期間、batch size、replay ratio を保持し、Learner は `UpdateFromBatch` の開始時に lineage exp-step から現在 phase を一度だけ解決する。
batch size と replay ratio は同じ phase transition で原子的に切り替え、独立した二つの `ProfiledValue` として管理しない。

段階的増加は DropMerge の実験候補であり、フレームワーク契約ではない。
phase ごとの batch size の増減を許容し、利用目的に合った schedule の選択は利用側の責任とする。

update budget は optimizer update 単位ではなく、replay sample 単位の **sample credit** として保持する。

```text
sample_credit += num_envs * current_replay_ratio

while sample_credit >= current_batch_size:
    sample current_batch_size experiences
    update learner once
    sample_credit -= current_batch_size
```

この単位なら、phase 境界で batch size が変わっても未消化の replay sample budget を同じ意味のまま持ち越せる。
定数設定では、現行の `num_envs * replay_ratio / replay_batch_size` による平均 optimizer update 頻度と数学的に一致させる。

phase transition は次の同期境界で行う。

1. `UpdateFromBatch` の開始時に、lineage exp-step から新しい phase を解決する。
2. phase が変わった場合は、ReplayBuffer prefetch と遅延 Push を quiesce する。
3. 旧 batch size で予約済みの prefetched batch は学習へ使用せず破棄し、破棄件数を診断可能にする。
4. 新 batch size と replay ratio を同時に有効化する。
5. ReplayBuffer の sample 可能件数を新 batch size に対して再検証する。
6. 現在の Experience を Push し、sample credit を加算して、新 phase の batch size で update を再開する。
7. 次の prefetch は新 batch size で cold start する。

Learner が ReplayBuffer へ要求した batch size と、返却された ExperienceSamples の実 batch size は一致しなければならない。
不一致は fail-fast とし、TD / QR / IQN の shape 計算は検証済み ExperienceSamples の実 batch sizeを正本とする。

cross-Run で schedule を連続させるには、Run-local `exp_step` ではなく lineage exp-step が必要である。
lineage exp-step、sample credit、現在 phase を復元できる Run 全体 save / load 契約が成立するまで、段階 schedule は scratch から単一 Run 内で完結する実験に限定する。
Agent checkpoint だけを別 Run へ load し、schedule を自動で初期 phase から再評価する運用は認めない。

## User Stories

1. As a DropMerge experimenter, I want to schedule batch size and replay ratio as one phase, so that a batch change does not silently alter the intended replay budget.
2. As a long-Run operator, I want to use a stable scratch phase before a performance-oriented mature phase, so that an aggressive mature setting is not forced onto unstable initial value estimates.
3. As a hyperparameter researcher, I want constant and phased modes to share the same accounting semantics, so that fixed controls and staged Runs remain comparable.
4. As a researcher, I want fractional replay budget to survive phase transitions, so that the boundary neither creates nor loses a full optimizer update arbitrarily.
5. As a Learner maintainer, I want the actual sampled batch to be the shape authority, so that TD, QR, and IQN implementations do not depend on stale static configuration.
6. As a ReplayBuffer maintainer, I want a defined prefetch quiesce boundary, so that a prefetched B128 sample cannot be consumed after switching to B256 or B512.
7. As a Run operator, I want batch increases to recheck sample availability, so that an early phase transition waits safely instead of sampling an undersized ReplayBuffer.
8. As an experiment analyst, I want the active phase, batch size, replay ratio, sample credit, and effective update rate exposed as metrics, so that generated artifacts prove what actually ran.
9. As an experiment analyst, I want transition stalls and discarded prefetched batches observable, so that schedule overhead is not mistaken for GPU or Env slowdown.
10. As a reproducibility-conscious user, I want schedule progress and sample credit restored with the Run, so that a continued lineage does not silently restart its learning regime.
11. As a configuration author, I want invalid phase definitions to fail at startup, so that malformed long Runs do not consume hours before the mistake becomes visible.
12. As a framework maintainer, I want batch monotonicity left to the experiment configuration, so that the generic mechanism does not encode a DropMerge-specific assumption.
13. As a researcher, I want B512/RR1 measured separately before attributing gains to staged batch growth, so that batch-size effects and replay-sample effects are not conflated.
14. As a project maintainer, I want this proposal kept as a provisional backlog PRD, so that the long-Run finding is preserved without implying implementation approval.

## Implementation Decisions

### 1. Deep module and ownership

- ReplayUpdateSchedule is a small value-resolution module that owns phase definitions and resolves an immutable pair of current batch size and replay ratio from lineage exp-step.
- Learner owns mutable schedule State: sample credit, last resolved phase, transition count, and last transition step.
- Agent owns schedule configuration and save / load Resource integration according to the project ownership guideline.
- ReplayBuffer は phase の切替時期を判断せず、Learner が要求する quiesce / cold-rearm 境界だけを提供する。
- ActionPolicy、Actor、Env は schedule を参照しない。

### 2. Configuration semantics

- 将来の設定モデルは `constant | phased` とする。正式なキー名は本暫定 PRD では決めない。
- constant は単一の batch size と replay ratio を持ち、現行の replay ratio mode と同じ結果を生成する。
- phased は順序付き phase を持ち、各 phase が正の期間、正の batch size、正の replay ratio を持つ。
- phased mode は replay ratio mode 専用とし、`update_interval` mode と混在させない。
- phase の batch size が単調増加することは検証しない。
- phase 境界は lineage exp-step の半開区間として一意に解決し、同じ step で複数回評価しても同じ phase を返す。

### 3. Sample credit

- sample credit の単位は「ReplayBuffer から学習へ投入すべき experience 数」とする。
- Train batch ごとに `num_envs * replay_ratio` を加算する。
- current batch size 以上の credit がある間だけ optimizer update を行い、update ごとに実 batch size を減算する。
- phase transition では sample credit を reset、round、batch size 比で変換しない。
- floating-point の長期誤差が update 数を変えないよう、実装時に十分な精度と境界テストを用意する。
- target network update と `learn_step` は従来どおり optimizer update ごとに一度進める。

### 4. Atomic phase transition

- phase 解決と transition は `UpdateFromBatch` 冒頭の一箇所で行う。
- prefetch quiesce、設定 pair の反映、sample availability の無効化、新 Experience の Push、update 再開の順序を固定する。
- transition 中に旧 batch と新 batch を同じ update loop 内で混在させない。
- 旧 phase で完了済みの optimizer update、priority update、target update は巻き戻さない。
- quiesce 中の worker 例外は握り潰さず、呼び出し元へ再送出して Run を停止する。

### 5. Prefetch contract

- 1-deep prefetch が未起動なら transition は待機せず新 phase を反映する。
- prefetch が in-flight なら完了を待ち、queued Push を FIFO 順に完了させる。
- 完了した旧 prefetched batch は学習に使用せず破棄する。次の sample は新 batch size で同期取得し、その後に新しい 1-deep prefetch を arm する。
- prefetched batch、worker future、CUDA event は save 対象にしない。save / load の quiesce 後は cold state から再開する。
- transition の待機時間と破棄 batch 数を観測可能にする。

### 6. Sample availability and shape

- batch size が変わるたびに「十分な Replay sample がある」という one-way cache を無効化する。
- 現在の ReplayBuffer sampleable count が新 batch size 未満なら、Experience の Push は継続し、update だけを停止する。
- ReplayBuffer は要求 batch size と同数の sample を返す。返却数不一致は契約違反として fail-fast する。
- Learner 共通処理と TD / QR / IQN loss は、検証済み sample の先頭次元から batch size を取得する。

### 7. Save / load dependency

- lineage exp-step と sample credit は Run 全体の継続 State として保存する。
- current phase は lineage exp-step から再解決し、保存値との不一致がある場合は fail-fast または明示診断する。具体契約は正式実装 PRD で確定する。
- 現在の Agent-only checkpoint を別 Run へ load する方式は、段階 schedule の継続手段として扱わない。
- 本機能の cross-Run 対応は Run 全体 serialize 機構の成立を前提とする。

### 8. Diagnostics

少なくとも次を scalar または transition event として観測可能にする。正式な metric tag は実装 PRD で既存 taxonomy に合わせて決める。

- current batch size
- current replay ratio
- current optimizer updates / exp-step
- current replay samples / new experience
- current sample credit
- current phase index または安定した phase ID
- phase transition count
- last phase transition exp-step
- actual sampled batch size
- prefetch quiesce wait time
- discarded prefetched batch count
- sample availability による update skip count

### 9. Validation

- batch size が 1 未満、replay ratio が非正または non-finite、phase 期間が 1 未満なら fail-fast する。
- phased mode で phase が空、一覧にある phase の定義がない、phase 名が重複する場合は fail-fast する。
- phased mode と `update_interval` mode の混在は fail-fast する。
- schedule が現在の Run save / load 能力では継続できない構成の場合、黙って初期 phase へ戻さない。
- 利用目的に対して phase 値が良いか、batch size が増加すべきかはフレームワークでは検証しない。

## Testing Decisions

テストは内部変数の代入方法ではなく、外部から観測できる update 回数、sample サイズ、transition、例外、metric、save / load の結果を検証する。

1. constant B128/RR1.25 で、固定step列に対する optimizer update 回数が現行計算と一致する。
2. phase 境界直前は旧 pair、境界stepから新 pairが使われ、同一 `UpdateFromBatch` 内で混在しない。
3. B128/RR1.25からB256/RR1へ切り替えても、sample credit換算でexperience budgetが失われない。
4. B256/RR1からB512/RR2への切替で、平均updates / exp-stepが同じでもsamples / new experienceが変わることを確認する。
5. batch増加時にReplayBuffer不足ならupdateを待機し、十分になった最初の境界で再開する。
6. batch減少も有効であり、単調増加を要求しない。
7. prefetch済み旧batchを破棄し、新batch sizeでcold sample / rearmする。
8. prefetch quiesce中のworker例外がLearner呼び出し元へ再送出される。
9. TD、QR、IQNがB128、B256、B512の実sample shapeでupdateできる。
10. ReplayBufferが要求数と異なるsampleを返した場合にfail-fastする。
11. phase、sample credit、lineage exp-stepをsave / loadし、再開後の次updateが中断なしRunと一致する。
12. load後にprefetchがcold stateから再開し、保存前のfutureを要求しない。
13. 各phaseでcurrent batch、RR、effective update rate、sample credit、transition metricが期待値を返す。
14. 不正mode、空phase、非正batch / RR / duration、`update_interval`混在を設定読み込み時にfail-fastする。

既存のLearner credit、ReplayBuffer prefetch、DefaultDQN config、TD / QR / IQN updateのテストをprior artとし、production APIをtest専用に歪めない。

実験採用前に、同一checkpointからB512/RR1とB512/RR2を比較し、batch sizeとreplay sample budgetの寄与を分離する。
その後、少なくとも固定B256/RR1、固定B512/RR2、段階scheduleを、同一NN / Agent / Env条件で比較する。
評価はmatched exp-stepとwall-clockの双方を使い、Eval reward、Double Suika / max rank、Q / TD / loss、PER coverage、NEET /終端、throughput、transition overheadを分けて判断する。

## Out of Scope

- 具体的なphase切替step、既定schedule、正式設定キーの決定。
- B128→B256→B512をDropMergeまたはDQN全般の普遍的最適scheduleとみなすこと。
- reward、loss、Q、TD、NEET発生率を入力にしたadaptive / closed-loop制御。
- ReplayBuffer capacity、PER alpha / beta、learning rate、gamma、n-step、target update方式の同時変更。
- `noop_penalty`、`time_penalty`、終端契約によるNEET対策。
- Run全体serialize機構そのものの実装。
- B512/RR1分離診断の実行と結果確定。
- RainbowAgentやDefaultDQN以外のAgentへ設定surfaceを公開すること。共有Learner変更による回帰は防止する。

## Further Notes

長期Runのlineageは手動の設定変更を含み、single-seed・非決定論である。
段階移行が結果的に成立したことは設計仮説の根拠にはなるが、各phaseが性能向上の原因だったとは断定しない。

B512/RR2の改善は、batch size 512、replay ratio 2.0、PER coverage、optimizer update回数、target update cadenceが組み合わさった結果である。
最初の追加実験はB512/RR1による機構分離を優先し、その結果を踏まえて正式番号の実装PRDへ昇格するか判断する。

本PRDは`ready-for-agent`ではない。
Run全体serializeの設計確定、B512/RR1分離診断、正式なconfig / metric契約のレビューを経た後に、実装用PRDを別途作成する。
