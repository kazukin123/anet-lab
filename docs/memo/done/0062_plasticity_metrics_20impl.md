# PRD062 可塑性メトリクス実装計画

## 概要

- 実装開始時に本計画を `docs/memo/062_plasticity_metrics_20impl.md` へ保存し、PRD062 の拘束力ある実装計画とする。
- DQN へ「実際の学習 forward の特徴」と「ReplayBuffer 一様 probe」の2チャネルを追加する。ImageCls は実学習 forward のみ対象とする。
- target-network capture は実装・テストするが、メトリクス行は既定でコメントアウトする。
- PRD の metrics 例を修正し、すべての新規 `@learn` 行へ `$learn_step` を明示する。`ForwardUpTo` の依存解決には、現行 builder と同じ「input key を branch 名より優先する」規則を明記する。
- t-SNE、Rainbow、MuZero、ReDo等の保護機構、RR科学検証は対象外とする。

## 実装変更

### NN・統計基盤

- `NetworkBranchCapture { branch_key, output }` を追加し、`Network::Forward` と `NetworkBody::Forward` に既定 `nullptr` の capture 引数を追加する。branch loop 後・`output_keys` 変換前に対象 tensor を detach して返し、追加 forward、TraceCallback変更、actorへの波及を発生させない。
- `Network::ForwardUpTo(input, branch_key)` と `Network::GetBranchNames()` を公開する。`ForwardUpTo` は対象 branch の ancestor closure だけをトポロジカル順に実行し、Format済み入力と実行済みbranchを含む `TensorDict` を返す。
- closure 構築では bind factor が input spec に存在すれば終端入力として扱い、同名branchを依存先にしない。未知branchは指定値と利用可能一覧を含めて fail-fast する。
- `anet::nn::PlasticityMetrics`、`ComputePlasticityMetrics()`、`kPlasticitySrankDelta=0.01`、`kPlasticityDormantTau=0.025` を追加する。rank-2 `(N,D)` を必須とし、detach → FP32 → CPUで一度だけ計算・キャッシュする。
- 指標は srank、`srank/min(N,D)`、τ=0.025 dormant率、τ=0 dead率、平均L2 feature normとする。全ゼロ特徴は順に `0, 0, 1, 1, 0` とする。

### 購読・Observer契約

- typedな `ScalarMetricSubscription` と、既定no-opの `Agent::ConfigureScalarMetricSubscriptions()` を追加する。source key、event、optional target、interval、runner scope、eval名を保持する。
- `RunManager` が実際にattachした定義だけを購読情報へ変換し、学習開始前にAgentへ1回渡す。DQN/ImageClsだけが train-scope `LEARN` の関心キーを消費する。
- online/target actual と probe の有効化、各cadenceは該当行の最小 `interval` から決める。購読ゼロならcapture request、probe sample、統計計算を一切行わない。
- 既知だが未計測のplasticity keyは `NaN`、未知keyだけを `nullopt` とする。Observerは非finite値をEMA更新前と非LEARN平均前に除外し、後続の有限値で正常に再開できるようにする。

### ReplayBuffer probe

- `ReplayBuffer` にpure virtual `SampleUniqueUniform(ExperienceSamples&, int64_t) const` を追加し、`DefaultReplayBuffer`、`PrefetchingReplayBuffer`、既存の全test doubleを同一変更で移行する。
- sampleable数不足時は、出力と専用RNGを変更せず `false`。十分なら一様・非復元で一意indexを返し、CPU samples、IS weight=1を設定する。`MarkSampledOnce`、priority、eviction統計、通常sampler RNGには触れない。
- probe samplerのseedはReplayBuffer seedから `SeedMaker(seed).MakeNamedSeed("plasticity_probe")` で派生する。件数一致時は全件を返してRNGを消費しない。
- Prefetch decoratorでは、呼出時までに受理したPushとin-flight prefetchを既存FIFO順でsettleしてからinnerへ委譲する。通常prefetched batchを消費・並べ替えず、通常sample列を不変に保つ。

### Agent・config・docs

- DQNではincrement前の `vars_.learn_step` でgateし、actualは既存train-mode/autocast forwardから捕捉する。targetはTD計算で使用したforward、target更新前の特徴を捕捉する。mutableな `MakeBatchUpdateResult` 境界で各結果へ焼き込み、その後requestをclearする。
- probeは同じstep gateで unique sample → device転送 →既存obs正規化 → learnerと同じautocast + NoGrad + eval-mode `ForwardOnlineUpTo` の順に実行する。最新capture/cacheはLearner-owned Stateとし、既存Agent lock境界で公開する。
- ImageClsは `StepCounts.learn_step` でgateし、ApplyMix後・train-mode・既存autocast内の実forwardを捕捉して `ImageClsUpdateResult` に格納する。ReplayBuffer probeは追加しない。
- DQN baselineへ `learner.plasticity.feature_key=main_feature` と `probe.batch_size=512`、ImageClsへ `feature_key=main_feature` を追加する。feature keyは購読時のみ必須・存在検証し、購読ゼロならNoCare。batch size `<1` は常時fail-fastする。
- group `34_agent_plasticity` にDQN actual 5本・probe 5本を既定ON、target 5本をコメントアウトで追加する。ImageClsはactual 5本だけを追加する。全行を `@learn $learn_step interval:500` とする。番号はdecade=チャネル、下1桁=統計種。
- PRD、ADR0031の追補、Agent/NN/observability/ReplayBuffer/DQN設計ページを現在契約へ同期する。既存用語で足りるため `CONTEXT.md` には新語を追加しない。

## テスト計画

- NN: capture有無で通常出力・traceが不変、対象branch一致、未知branch、closure外branch不実行、IQN pre-fusion、input/branch同名時のinput優先を検証する。
- 統計・Observer: 解析解のあるfull-rank/rank-1/all-zero行列、rank違反、lazy cache、疎なNaN、EMA/平均の非finite除外と回復を検証する。
- ReplayBuffer: index一意性、seed再現、件数不足と全件時のRNG不変、通常sample/PER統計不変、prefetch settle後snapshotと通常prefetch列不変を検証する。
- Agent: cadence最小値、購読ゼロ、feature key検証、actualと同一forward、target更新前capture、probe鮮度、ImageClsのApplyMix後capture、actor/nn_trace不変を検証する。
- MSVC Debugでcore testsとRunnerをビルドし、関連test tagsおよび全core testを実行する。LunarLander DefaultDQNとImageClsのsmokeで `inspect_run.py tags` が新タグを `status=ok`・`count>0` と報告することを確認する。
- deterministicなLunarLander短Runを同seedでON/OFF実行し、既存learning scalar系列と `agent_close.anet` を完全一致で比較する。通常ReplayBuffer sample列の不変はunit testでも直接確認する。
- ReleaseでAtariとLunarLanderをON/OFF各3回、順序を交互にして同一予算で測定し、warmup後throughput中央値を比較する。差が±3%を超えた場合はinterval/batch size案を報告するが、既定値は自動変更しない。

## 前提

- target対応は維持するが既定OFFとする。
- Rainbow/MuZeroへの配線、t-SNE利用側、plasticity保護機構、25M RR実験は実装・受入対象外。
- 互換aliasや旧契約分岐は追加せず、公開ReplayBuffer契約の全現用実装をクリーンブレークで移行する。
- 既存の未コミット変更には触れず、Git staging・commit・pushは行わない。
