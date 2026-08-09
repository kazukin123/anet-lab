# 追加 forward なしの近似 Actor PER 初期優先度 PRD

## 問題

DropMerge 系の実行では、PER の新規遷移に固定 `per_initial_priority` を与えている。固定値が高いと、Learner 未更新の新規遷移がリプレイ優先度の質量を支配する。

固定値が低いと、一度もサンプリングされないまま追い出されやすい。実測でも、`replay_ratio = 1.0` では `per_sample_initial_ratio` と `last_evicted_never_sampled_ratio` の間に強い収支がある。

固定値の調整だけでは、新規遷移の網羅性とLearner更新済み遷移の再利用を両立しにくい。

一般的なPER実装の既定である最大優先度初期化（現行実装では負の`per_initial_priority`で選択できる）も試した。しかし新規遷移が常にその時点の最大優先度で入るため、初期優先度の質量支配は固定高値と同等かそれ以上に強く、実測でも問題は解消しなかった。

固定初期値は、遷移ごとのTD誤差、学習段階、報酬スケール、TBO、ネットワークの変化を反映できない。一方、単一マシンで優先度算出専用のネットワークforwardを追加するのは計算効率が悪い。

同じGPU計算資源をLearner更新に使う方が有効である可能性が高い。分散Ape-Xをそのまま導入せず、既存の行動推論が生成済みのQ値を再利用し、追加forwardなしで固定値より情報量のある初期優先度を作りたい。

行動選択時点では、報酬、n-step収益、ブートストラップ状態が確定していない。そのため、Actorが最終優先度まで計算する責務を持つのは不自然である。

また、`PipelineRunner`が偶然保持する前後stepをTrainerで結合すると、DQN/PER固有の意味がRunnerの実行順序へ漏れる。通常の同期Trainerへも展開しにくくなる。

## 解決方針

Actorは最終優先度を計算しない。既存の行動推論から得られる最小限のアルゴリズム固有payloadを、`BatchActionInfo`へ追加するoptionalな`ReplayInitialPriorityHint`に載せる。

TrainerとLearnerは、このメタデータを通常の`BatchExperience`と同じ経路で`ReplayBuffer`へ運ぶ。アルゴリズム固有の統合処理は行わない。

`ReplayBuffer`は、遷移がサンプリング可能になる時点で、開始slotと必要なブートストラップ状態slotのopaqueなhint行、確定済みn-step収益、終端情報、実n-step数を初期優先度推定器へ渡す。DQN推定器がhint行から`Q(s,a)`と状態価値を解釈する。

DQN Agentが注入する推定器は、Learnerと同じ割引率、TBO変換、PER epsilonを使って近似TD誤差を計算する。報酬スケールはAgentがPush前に適用済みで、保存済みtarget returnへ反映されているため、推定器では扱わない。truncated遷移は開始hintを検証したうえで、終端観測由来のbootstrap hintが存在しないため最大優先度初期化へフォールバックする。契約違反はバグとしてエラーで停止する。

この方式は、`PipelineRunner`と同期Trainerの両方で動作する。必要な前提は、経験が`ReplayBuffer`へ時系列順に到着するという既存契約だけである。

優先度専用forward、Actor用target network、前後stepを結合するTrainer固有処理は追加しない。

一方、現行の1-step先読みでは、Sample後のwrite-behind Pushが同じ物理slotを別の遷移へ上書きしてから、古いSampleの`UpdatePriorities`が到着し得る。この問題は本機能で新規に発生するものではないが、source追跡とActor/Learner比較を正しく成立させるため、本PRDでgeneration付きの`replay item key`を導入し、上書き前の更新だけを要素単位で棄却する。判断の経緯はADR 0011へ記録する。

## ユーザーストーリー

1. 強化学習の実験者として、既存の行動推論情報で新規遷移を順位付けしたい。固定初期優先度ですべての遷移を同じ価値として扱わないためである。
2. 強化学習の実験者として、近似Actor初期優先度のためにネットワークforwardを増やしたくない。GPU時間をLearner更新に使うためである。
3. 強化学習の実験者として、固定・最大・近似Actorの初期化方式を明示的に切り替えたい。同条件の実行を比較するためである。
4. 強化学習の実験者として、仕様上bootstrap hintを利用できないtruncated遷移には最大優先度初期化を使いたい。有効化によって遷移が暗黙にリプレイ対象外にならないためである。
5. 強化学習の実験者として、サンプリング後はLearnerのTD誤差を最終権威にしたい。Actorの近似情報を初回サンプリング前だけに限定するためである。
6. 強化学習の実験者として、Actor初期化とフォールバック初期化をメトリクスで区別したい。近似値が実際に使われているか診断するためである。
7. 強化学習の実験者として、近似Actor初期優先度とLearner優先度を比較したい。最終報酬とは独立に近似品質を評価するためである。
8. 性能調査者として、forward呼び出し回数とプロファイル範囲を確認したい。優先度専用forwardが増えていないことを証明するためである。
9. 性能調査者として、固定値基準のスループットを測りたい。小さなメタデータ転送による実行速度低下を見落とさないためである。
10. 保守担当者として、TrainerにはDQN優先度の意味を理解させず、標準的な経験だけを運ばせたい。Runnerをアルゴリズム非依存に保つためである。
11. 保守担当者として、`ActionPolicy`から小さなReplay初期優先度ヒントだけを出したい。`ReplayBuffer`に全行動価値や全quantileを保持させないためである。
12. 保守担当者として、DQN固有のtarget計算とTBO計算を優先度推定器へ閉じ込めたい。汎用`ReplayBuffer`にDQN分岐を持ち込まないためである。
13. 保守担当者として、既存の`ReplayBuffer::Push`インターフェースを維持したい。先読みと転送のデコレータ契約を保つためである。
14. 保守担当者として、リングバッファ上書き時に優先度sourceを正しく初期化したい。上書き済みslotの初期状態が診断とメトリクスに残らないためである。
15. テスト作成者として、ヒント使用、フォールバック、終端処理、Learner上書きを決定的に検証したい。非公開のSumTree配置に依存しないためである。
16. 同期学習の利用者として、`PipelineRunner`を使わず同じ機能を利用したい。初期化方針を実行スケジュールへ結合しないためである。
17. QR-DQNの利用者として、既存の平均Q出力を再利用したい。全quantileの保存を不要にするためである。
18. TBOの利用者として、LearnerのTD誤差と同じ空間で近似targetを変換したい。近似Actor初期優先度とLearner優先度を比較可能にするためである。
19. 先読みReplayBufferの利用者として、Sample後に物理slotが上書きされても、古いLearner優先度で新しい遷移を更新したくない。staleな更新だけを無視し、同じbatchの有効な更新を維持するためである。

## 実装上の決定

### 1. 設定はLearnerとリプレイ方針に属する

- DQNの`LearnerConfig`に`per_initial_priority_mode`を追加し、`fixed`、`max`、`actor_approx`の3値を受け付ける。
- 既定値は`fixed`とし、既存設定とチェックポイントの動作を維持する。
- `max`は、現行の負値`per_initial_priority`が行う暗黙の最大優先度初期化（Learner更新済み優先度の最大値で初期化。初期値1.0、`per_alpha`適用済み空間で追跡）を明示モードへ昇格したものである。
- `per_initial_priority`は`fixed`モードでのみ使用する。`max`と`actor_approx`では非負の有限値を読み込むが使用しない。同一configで`per_initial_priority_mode`だけを切り替えるA/B比較を想定し、不使用モードでの非負指定はエラーやWARNにしない。
- 負値`per_initial_priority`による暗黙の最大優先度初期化は廃止し、モードに関係なく負値は`per_initial_priority_mode = max`への移行方法を示してエラーにする。NaN/Infも全モードでエラーにする。既存configの負値使用箇所は本PRDで移行する（「設定」の章を参照）。
- `fixed`モードの`per_initial_priority = 0`は意図的な指定として許可する。leafが0でもsourceは`fixed_initial`であり、値0を無効化protocolとして解釈しない。
- `ActionPolicyConfig`へ重複するユーザー設定を追加しない。Agent構築時にLearnerのモードからReplay初期優先度ヒント出力の要否を決め、学習用Actorへ明示的に配線する。
- `max`と`actor_approx`はPERを前提とする。`use_per = false`との組み合わせは、両設定と修正方法を示してエラーにする。
- `actor_approx`でActor推定値を利用できない遷移のフォールバック初期化は、`max`モードと同一の最大優先度初期化とする。

### 2. Actorは優先度ではなくReplay初期優先度ヒントを出力する

- 行動選択で生成済みのonline network出力を再利用する。online、target、優先度専用のforwardは追加しない。
- `BatchActionInfo`にoptionalな`ReplayInitialPriorityHint`を追加する。共通carrierは、アルゴリズム固有payloadを単一の連続した`float32[B,K]`（`K > 0`）として保持し、列の意味を解釈しない。carrierが存在する場合はbatchの全行が有効であり、別のvalidity maskは持たない。`fixed`と`max`モードではcarrierを設定しない（未設定＝確保・転送ゼロ）。
- DQN系ではpayloadを`K = 2`のActor Qヒントとし、次の2列を保持する。列数、列index、pack/decodeはDQN module内の共通helperで一元管理し、汎用RL層とReplayBufferへQ値の意味を持ち込まない。
  - 環境へ実際に渡した行動の平均価値`actor_q_sa`
  - 全行動のonline平均Qの最大値として定義する状態価値近似`actor_state_value`
- carrier tensorのdevice方針はaction本体と同一とする。計算グラフから切り離した連続tensorを出生デバイスで保持し、初回のCPU要求時にpack済みtensorを1本だけ同期変換してキャッシュする（non_blockingは使わない）。CPU消費者はPER（CPU側）の`Push`だけであり、物理D2HはPush実行スレッドで1stepあたり1本だけ発生してactor/envのcritical pathに乗らない。GPU方向の変換APIは実装しない。
- Replay初期優先度ヒントを生成するのは`actor_approx`が有効な学習Actorだけとする。評価ActorとLearner内部のtarget action選択、ならびに`fixed`/`max`モードではヒントを生成しない。
- QR-DQNではquantileから算出済みの平均`q`出力を使う。`q_dist`を本機能のために追加保持せず、初期優先度用にquantile lossを再計算しない。
- `actor_q_sa`は、epsilonやUQEを含む最終実行行動で収集する。探索の有無で定義を変えない。
- `actor_state_value`は、Learnerのtarget選択を意図的に`max_a Q_online(s,a)`で近似する。
- Learner側のtarget-network評価は再現しない。Double DQN有効時の`Q_target(s', argmax_a Q_online)`だけでなく、無効時もLearnerはtarget networkを行動選択・評価に使う。いずれもActor側での再現にはtarget networkの保持と追加forwardが必要で、forward追加禁止の目的に反する。また行動時点とLearnerサンプル時点ではtarget networkのスナップショットが別物のため、コストを払っても厳密再現は原理的に不可能である。
- UQE/楽観的target選択も再現しない。こちらはquantileが行動時に手元にありforwardは増えないが、UQE版状態価値という追加ヒントtensorが必要になる。UQEのtauは減衰スケジュールで時間変化し、ThompsonSamplingのtauは呼び出しごとに再抽選されるため、いずれも行動時とLearnerサンプル時で一致しない。順位付け目的には平均Q近似で足りると割り切る。`greedy_only`指定でもUQEの楽観的選択基準は維持される実装のため、UQE構成では近似Actor初期優先度とLearner優先度に系統差が出ることを許容する。判断の記録と再訪条件はADR 0010を参照。
- `DQNActionInfo::WithAction`で実行行動を差し替える場合、DQN payload helperでschemaを検証し、既存の一時的な`aux["q_values"]`から差し替え後の`actor_q_sa`をgatherし直して`actor_state_value`を維持する。追加forwardやフォールバックは行わない。ヒントが存在するのに`q_values`が欠ける場合は契約違反としてエラーにする。現状の強制行動経路は評価用でヒント自体を持たないが、将来の学習利用でも正しい値を保つ契約とする。
- Replay初期優先度ヒントは小さなtensorとし、本機能のために全network出力を追加保持しない。既存auxの全行動Q/quantileは補助診断用の一時データでReplayBufferへ保存せず、全行動Qは`WithAction`時の再計算に限って再利用する。

### 3. 既存の経験転送経路だけを使用する

- Replay初期優先度ヒントは、`BatchActionInfo`のoptional carrier、`BatchExperience`、Trainer、`Learner::UpdateFromBatch`、`ReplayBuffer::Push`の順に流す。
- 永続info（`info_storage_`へ保存されサンプル抽出でLearnerへ渡る経路）はヒントの置き場にしない。サンプルバッチへの漏出と抽出コストを避けるためである。auxの補助診断データもReplayBufferへ保存されないため、ヒントの運搬には使用しない。`WithAction`内で既存Qを一時参照する場合を除き、infoでもauxでもない第三の器としてoptionalな専用carrierを採用する。
- TrainerはReplay初期優先度ヒントを読み取り、結合、改名、解釈しない。特に`PipelineRunner`で前回経験と現在の`ActionInfo`を結合しない。
- `ReplayBuffer::Push`は変更しない。`PrefetchingReplayBuffer`は同じ不変`BatchExperience`スナップショットを運び、既存のFIFO順序を維持する。
- `actor_approx`が有効な学習Actorは、dummyではない実stepごとにcarrierを必ず設定する。開始stepの設定漏れは、後にtruncatedになった場合もフォールバック理由とはせず、契約違反としてサンプリング可能化境界のエラー検出（決定4）で停止する。

### 4. サンプリング可能化の境界で優先度を完成させる

- ReplayBufferはReplay初期優先度ヒントをslot単位で長期保存しない。`Push`はpack済みtensorを1回だけCPU化し、envごとのopaqueな行をReplayBuffer内部の`c10::SmallVector<float, 4>`へコピーする。開始stepの行はn-step確定キューのレコード（`QueueRecord`相当）に載せて確定まで運び、確定後は完成待ちエントリへ移す。ブートストラップstepの行は、そのslotを書き込む`Push`の入力から同一call内で直接消費し、保存しない。推定器へは所有権を渡さない`std::span<const float>`として同期的に渡す。
- 保存不要が成り立つ根拠: 非終端遷移は構造上`actual_n_steps = n_step`（短縮は終端到達時のみ）であり、開始slot tのサンプリング可能化はslot t+n を書くPushで起きるため、ブートストラップヒントの到着と消費は常に同一Pushになる。
- 非終端遷移は、n-step収益と`start + actual_n_steps`にあるブートストラップ状態slotの両方が利用可能になった後に限り、近似Actor初期優先度を受け取る。このタイミングを既存の未来観測有効性ルール（`write_cursor >= start + n_step + 1`）と揃える。未初期化または古いSumTree leafを持つ遷移をサンプリング可能にしない。
- 既存の`-1.0f`特殊フラグによる初期化は、シーケンス確定時すなわちブートストラップslot書き込み前に走るため、そのままの位置では近似初期化に流用できない。流用すると非終端遷移が全件ブートストラップ欠損になる。Push毎に、新たにサンプリング可能化境界へ達したエントリの優先度を完成させるsweepを設ける。確定順と完成順は一致するため、完成待ちはenv毎のFIFOで足りる。
- 完成待ちFIFO、mode分岐、推定器、フォールバック/エラー判定、初期系sourceの決定、completion理由カウンタは`InitialPriorityCompleter`が所有する。`DefaultReplayBuffer`は`ValidIndexManager`、`ReplayPriorityStore`、`metadata_mutex_`の所有者としてcompleterの生成と呼び出しを行い、同じ同期区間内で狭いinterface経由のsampleable照会と初期優先度適用を許可する。completer自身はsource配列を含む共有resourceとmutexを所有しない。
- `InitialPriorityCompleter`はPERでのみ構築する。PERの`fixed`、`max`、`actor_approx`は同じcompletion経路とサンプリング可能化境界を使う。一様ReplayBufferではcompleterも完成待ちFIFOも生成せず、不要なpendingを蓄積しない。
- sampleable範囲式は`ValidIndexManager`内の小関数へ括り出し、列挙、単点判定、上書き判定から共用して式の二重保守を避ける。
- Replay初期優先度ヒントのtransport表現とcompletionの所有権判断はADR 0012へ記録する。
- 真の終端遷移はブートストラップを0とし、自身の開始hintと確定収益だけで推定する。ブートストラップslotの検査は行わない（次エピソードのslotを誤検査しないため）。
- 正当なフォールバックはtruncated遷移のみとする。現在のRunner契約では終端観測を推論せず、dummy slotにヒントが存在しないため、v1では最大優先度初期化へフォールバックする。
- 契約違反はフォールバックせずエラーで停止する。`actor_approx`有効時の実stepのcarrier未設定、共通carrierのrank/dtype/batch不整合、DQN payloadの`K != 2`、非終端遷移でのブートストラップ不整合、負の算出優先度は、いずれもバグでしか発生しないため`ANET_SYSTEM_ERROR`とする。算出優先度0は有効であり、sourceを`actor_initial`に保つ。
- payloadのschemaとfinite判定はアルゴリズム固有推定器が所有する。`InitialPriorityEstimator::ValidateHint`はschema違反をエラーにし、payloadにNaN/Infがあればfalseを返す。truncatedでも開始hintへ先に`ValidateHint`を適用し、Actor由来の異常をtruncation理由へ隠さない。通常の`Estimate`は開始hintと必要なbootstrap hintを同じ規則で検証する。
- 推定器が検出したhintまたは算出値のNaN/Infは、Debugビルドでは`ANET_SYSTEM_ERROR`で停止し、Release/RelWithDebInfoでは最大優先度初期化へフォールバックして非finite理由カウンタに計上する。Learner算出優先度のNaN/Infは最終権威の破綻なので、全buildでReplayBuffer更新前に`ANET_SYSTEM_ERROR`とする。
- 未対応アルゴリズムの実行時フォールバックは存在しない。`per_initial_priority_mode`はDQN系Learner設定にのみ存在し、推定器を持たない構成は`actor_approx`になり得ないためである。
- フォールバック初期化もActor初期化と同じサンプリング可能化境界で適用し、全モード（fixed/max/actor_approx）で有効性と順序の契約を共通化する。
- 構築時に、丸め後の`capacity_per_env`が`max(1, n_step) + 1`以上であることを全モード共通で検証する。不足時は開始slotが未来観測到着前に上書きされるため、指定`replay_capacity`、`num_envs`、丸め後の`capacity_per_env`、必要最小値を含む`ANET_SYSTEM_ERROR`とする。既知のwrap/sampleabilityバグ自体は本PRDで修正しない。

### 5. DQN固有計算を深い戦略モジュールへ閉じ込める

- ReplayBufferは構築時に任意の初期優先度推定器を受け取る。completerが呼び出し時期、フォールバック、初期系sourceの決定を所有し、DQN計算とDQN payload検証は推定器が所有する。
- 推定器の入力は、opaqueな開始/ブートストラップhint行を表す`std::span<const float>`、確定済みtarget return、終端情報、実n-step数とする。真の終端ではbootstrap spanを空にし、非終端では論理時刻が一致するhint行を必須とする。ReplayBuffer内部やSumTreeへのアクセスは渡さない。
- `InitialPriorityEstimator::ValidateHint`はpayload schemaを検証し、finiteならtrue、NaN/Infならfalseを返す。DQN推定器は`K = 2`と列意味を所有し、Actor、`WithAction`、推定器で同じpack/decode helperを使用する。
- 報酬スケールはAgentがPush前に適用済みで、保存済みtarget returnへ反映されている。推定器はRewardScalerを保持せず、スケールを再適用しない。
- 非TBOのDQN/QR-DQNでは、確定済みn-step収益と割引済みonline状態価値ブートストラップから近似targetを作る。
- TBOでは、h空間のブートストラップを逆変換し、実空間で確定収益と結合する。再びh空間へ変換して`actor_q_sa`と比較し、LearnerのTD誤差空間へ揃える。
- 生のActor初期優先度は近似TD誤差の絶対値に`per_eps`を加えた値とする。Samplerが`per_alpha`を適用する前に、Learner優先度と共通のclipを適用する。`per_eps = 0`やclip上限0による優先度0は有効とする。
- LearnerとのTD誤差計算はできる限り数式と命名を共有する。`TransformH`/`TransformHInv`は名前付きDQN namespaceへtensor版とscalar版のoverloadを隣接定義し、Learnerはtensor版、推定器はPush hot pathでtensor生成を避けるscalar版を呼ぶ。`|近似TD誤差| + per_eps`とclipの優先度確定ポリシーも同じnamespaceのtensor/scalar helperへ揃え、両版が同一数式であることを数値一致テストで拘束する。
- `fixed`初期値は既存互換のため新しいActor/Learner共通clipの対象にせず、非負の指定値へ`per_alpha`を1回だけ適用する。`max_initial`は既に`per_alpha`適用済みのLearner最大値を使い、再度clipや`per_alpha`を適用しない。raw priorityとSumTree leaf priorityの内部設定経路を分け、二重変換を防ぐ。
- n-stepターゲット合成（`G + γ^n (1 - T) V`のTBO挟み）は、Learner側がquantile次元付きテンソル、推定器側がスカラーで形が異なるため無理に共通化せず、同一入力でLearner計算と数値一致するテストで拘束する。なおLearnerの優先度用TD誤差は既に平均空間（`mean(Z(s,a)) - mean(target)`）であり、推定器の式と構造的に同形である。
- QR×TBOでは、Learnerが分位点毎に変換して平均を取り、推定器が平均を変換するため、hの非線形性により完全一致は原理的に不可能である。一致テストは非TBOまたはスカラーDQN構成で行う。
- `Estimate`の`std::nullopt`は入力または計算結果のnonfiniteに限定する。schema違反はエラーとし、値がある場合はcompleterがfiniteかつ非負であることを防御的に検証する。推定器自身はフォールバック値を選ばない。
- 一様ReplayBuffer、MuZero、推定器を持たないAgentは既存の構築と動作を維持する。

### 6. 初期状態と優先度sourceを別の概念として扱う

- slotごとのsourceを`none`、`fixed_initial`、`max_initial`、`actor_initial`、`learner_updated`として追跡する。`fixed_initial`、`max_initial`、`actor_initial`はいずれもLearner未更新の初期状態である。
- `actor_approx`のフォールバック適用はsourceを`max_initial`とする。sourceは適用した値の種類を表し、フォールバックへ至った理由は診断カウンタで別管理する。
- generation一致のLearner更新は、適用優先度が0の場合もsourceを`learner_updated`へ変更する。stale更新ではsourceを変更しない。
- `ExperienceSamples::per_is_initial_priority`はCPU `int8` tensorの`per_priority_sources[B]`へ置き換える。LearnerはSample時点のsourceから、既存の`per_sample_initial_ratio`とsource別サンプル比率を導出する。source enumは診断上の意味情報であり、物理slotやgenerationは公開しない。
- 既存のReplayBufferとSampler間の`-1.0f`=初期化、`0.0f`=無効化、正値=Learner更新というmagic protocolは廃止する。内部では、物理slot無効化、raw初期優先度設定、`per_alpha`適用済みmax初期優先度設定、Learner更新を別の明示APIにする。優先度0と無効化をsourceで区別する。
- per-slot source配列とsource別優先度質量は、`DefaultReplayBuffer`が所有する`ReplayPriorityStore`に保持する。completerは`fixed_initial`、`max_initial`、`actor_initial`の初期遷移を決定し、リング上書き/dummy書き込み経路は`none`、generation一致のLearner更新経路は`learner_updated`を、同じstoreの明示APIから適用する。
- 初期系（fixed/max/actor）の優先度設定は最大優先度追跡を更新しない。最大優先度追跡を押し上げるのは、generation検証を通過して実際に適用された有限なLearner優先度だけとする。staleとして棄却したLearner値は最大値にも反映しない。
- リング上書きとdummy書き込みでは、generationを進めると同時にsourceを`none`へ戻し、leafを無効化する。優先度0の正当な初期sourceは上書きまで維持する。
- 既存の`per_sample_initial_ratio`と`initial_mass_ratio`は、Learner未更新を集約する履歴互換メトリクスとして維持し、本PRDでは改名しない。

### 7. generation付きreplay item keyで上書き後の更新を防ぐ

- 外向けの`ExperienceSamples::indices`を`replay_item_keys`へ、`ReplayBuffer::UpdatePriorities`の引数を`item_keys`へ改名する。keyはopaqueなCPU `int64`値とし、`ReplayItemKey`型やaliasは作らない。
- 内部の物理位置は`slot_index`、全envをflattenした位置は`flat_slot_index`と呼ぶ。外向けkeyを`logical_index`とは呼ばない。
- keyの基数は丸め後の`actual_capacity = capacity_per_env * num_envs`とし、SumTree容量も`config.capacity`ではなく`actual_capacity`へ統一する。keyは`generation * actual_capacity + flat_slot_index`でencodeする。
- per-slot generationは未書き込み時0、realまたはdummyを書き込むたびに1増加し、n-step確定や優先度変更では増加させない。最初にSample可能なitemのgenerationは1以上とする。encode時の乗算・加算overflowは`ANET_SYSTEM_ERROR`とする。
- `UpdatePriorities`はkeyをdecodeし、key generationが現在より小さければ上書き済みstaleとしてその要素だけを無視する。同じなら適用し、現在より大きいgeneration、generation 0、負値、形状不一致、非finite/負のLearner優先度はプログラムエラーとして、どのleafも変更する前に`ANET_SYSTEM_ERROR`とする。keyは生成元ReplayBufferだけで有効であり、直列化しない。
- stale要素はleaf、source、最大優先度、Actor/Learner比較を一切変更せず、stale-dropカウンタだけを増やす。同じbatchのgeneration一致要素は通常どおり適用する。
- 同じkeyがbatch内に重複した場合は入力順に逐次適用し、leafは既存互換のlast-winsとする。Actor/Learner比較は最初に`actor_initial`から遷移させた要素だけを1ペアとして記録する。
- `UpdatePriorities(item_keys, priorities)`はvoidではなく呼び出し単位の`ReplayPriorityUpdateResult`を返す。結果には適用数、stale数、Actor/Learner比較ペア数、正値ペア率、`A/L`中央値、平均`log(A/L)`、Spearman順位相関を含める。raw slot、generation配列、比較ペア配列は返さない。`PrefetchingReplayBuffer`は既存wait境界の後でinner結果をそのまま返す。
- この変更は、ADR 0005で許容した「1-step staleな経験で学習すること」と、上書き後の別itemへ古い優先度を書き込むことを区別する。後者だけを防ぐ判断はADR 0011へ記録する。raw物理indexを受ける互換経路は設けない。

### 8. 採用判断に必要な診断情報を追加する

- `fixed_initial`、`max_initial`、`actor_initial`のsource別サンプル比率は各Learner minibatchで、source別優先度質量比率は取得時点のSumTree全体で報告する。
- Actor推定の試行数、利用成功数、truncationフォールバック数、Release系の非finiteフォールバック数と各比率は、ReplayBuffer生成後の累積値として報告する。契約違反はエラーで停止するため理由カウンタには現れない。
- 既存の未サンプル追い出しメトリクスはsource非依存で維持する。stepごとのtensor走査なしで実現できる場合はsource別も追加する。
- 適用した初期優先度をslotごとに追加保存せず、sample時leaf priorityも`ExperienceSamples`へ追加しない。generation一致かつ更新直前のsourceが`actor_initial`なら、現在のSumTree leafがActor初期優先度である。このitemを最初に`learner_updated`へ変える1回だけ、新しいLearner優先度と突き合わせる。既に`learner_updated`のitemやstale itemは比較ペアへ追加しない。
- Actor/Learner比較は各`UpdatePriorities`呼び出し、すなわちLearner minibatch単位で集計する。全比較ペア数と、両優先度が正のペア率を報告する。正値ペアについて`A/L`の中央値と`log(A/L)`の算術平均を計算し、正値ペア0件なら両方`NaN`とする。
- Spearman順位相関はzeroを含む全finiteペアを平均rankで計算し、ペア数2未満または片側が全tieなら`NaN`とする。`per_alpha = 0`で全leafがtieになる場合も`NaN`とし、source/useメトリクスは通常どおり報告する。比較はすべて`per_alpha`適用済み空間で行う。
- `ReplayPriorityUpdateResult`のstale数からminibatch単位のstale率を報告し、ReplayBuffer側にも累積stale-drop数を持つ。メトリクスのためにサンプリングを変えたり、環境stepごとの追加GPU同期を行ったりしない。
- 既存の`per_prio_clip_ratio`は、clip適用前の`abs(td_error) + per_eps`が`per_prio_clip_value`を超え、実際に値が変更された要素の割合とする。上限と元から等しい要素はclip件数へ含めない。
- ヒント抽出、初期優先度推定、source管理、key検証に安定したプロファイル範囲を追加する。

#### 8.1 追加メトリクス辞書

`metrics_scalar.txt`へ追加する20項目を以下に定義する。比率の分母が0の場合は`NaN`とし、値が未定義であることと正常な0を区別する。累積countはReplayBuffer生成後からの絶対件数であり、異常調査にのみ使う。累積countにはEMAを付けず、trial間の直接比較や最適化objectiveにはしない。

source構成:

| メトリクス | 意味 | 分母・単位 | 期待する見方 | 主な利用場面 | `NaN`条件 |
| --- | --- | --- | --- | --- | --- |
| `per_sample_fixed_initial_ratio` | minibatch中で`fixed_initial`のままsampleされた割合 | minibatch size、比率 | `fixed` modeの曝露量。`actor_approx`では原則0 | mode配線とsource遷移の確認 | minibatch sizeが0 |
| `per_sample_max_initial_ratio` | minibatch中で`max_initial`のままsampleされた割合 | minibatch size、比率 | `max` modeの曝露量。`actor_approx`ではtruncation/nonfinite fallbackの影響も含む | fallbackが学習batchへ届く強さの確認 | minibatch sizeが0 |
| `per_sample_actor_initial_ratio` | minibatch中で`actor_initial`のままsampleされた割合 | minibatch size、比率 | 高低そのものを優劣とせず、Actor初期値が実際に学習へ曝露した量として読む | ハイパラ探索の常時比較 | minibatch sizeが0 |
| `replaybuffer.per.fixed_initial_mass_ratio` | SumTree全質量に占める`fixed_initial`の質量割合 | SumTree total、比率 | `fixed` modeでの初期値の影響量。`actor_approx`では原則0 | source構成とsample比率の差の診断 | SumTree totalが0 |
| `replaybuffer.per.max_initial_mass_ratio` | SumTree全質量に占める`max_initial`の質量割合 | SumTree total、比率 | `max` modeまたはfallback初期値のsampling影響量 | fallbackが過大な質量を持つかの診断 | SumTree totalが0 |
| `replaybuffer.per.actor_initial_mass_ratio` | SumTree全質量に占める`actor_initial`の質量割合 | SumTree total、比率 | 高低そのものを優劣とせず、Actor初期値がReplay samplingへ与える影響量として読む | ハイパラ探索の常時比較 | SumTree totalが0 |

stale・上書き:

| メトリクス | 意味 | 分母・単位 | 期待する見方 | 主な利用場面 | `NaN`条件 |
| --- | --- | --- | --- | --- | --- |
| `per_priority_update_stale_ratio` | 1回のLearner更新でgeneration不一致により棄却したpriority更新の割合 | `applied + stale`、比率 | 低いほどよく、通常は0付近。pipeline深度やReplay回転速度との組合せで読む | ハイパラ探索結果の原因分析 | `applied + stale`が0 |
| `replaybuffer.per.priority_update_stale_drop_count` | ReplayBuffer生成後に棄却したstale更新の累積件数 | 件数 | 増加有無と増加速度だけを見る | pipeline・prefetch・小容量Replayの障害調査 | なし。PER無効時は項目自体を公開しない |

Actor/Learner一致度:

| メトリクス | 意味 | 分母・単位 | 期待する見方 | 主な利用場面 | `NaN`条件 |
| --- | --- | --- | --- | --- | --- |
| `per_actor_learner_pair_count` | その更新で初めて`actor_initial -> learner_updated`になり比較できたitem数 | 件数 | 大きさ自体は目的にせず、相関・倍率統計を採用できる標本数か確認する | 全比較統計の信頼性gate | なし。比較なしは0 |
| `per_actor_learner_positive_pair_ratio` | Actor優先度とLearner優先度がともに正だった比較ペアの割合 | 全比較ペア数、比率 | 1に近いほど倍率比較に使えるペアが多い。0 priorityを許す設定では低下を許容する | 倍率統計が欠ける理由の診断 | 比較ペア数が0 |
| `per_actor_learner_ratio_median` | 正値ペアにおける`Actor priority / Learner priority`の中央値 | 正値ペア、倍率 | 1付近なら倍率校正が良い。1超はActor過大、1未満はActor過小の傾向 | 相関が良いtrialの倍率校正診断 | 正値ペア数が0 |
| `per_actor_learner_log_ratio_mean` | 正値ペアにおける`log(Actor priority / Learner priority)`の平均 | 正値ペア、対数倍率 | 0付近が目安。正はActor過大、負はActor過小 | ハイパラ探索の常時比較 | 正値ペア数が0 |
| `per_actor_learner_spearman` | Actor初期優先度と最初のLearner優先度の順位相関 | zeroを含む全finite比較ペア、相関係数 | 高いほどActorの順位付けがLearnerと整合する | ハイパラ探索の常時比較 | ペア数2未満、または片側が全tie |

Actor完成品質:

| メトリクス | 意味 | 分母・単位 | 期待する見方 | 主な利用場面 | `NaN`条件 |
| --- | --- | --- | --- | --- | --- |
| `replaybuffer.per.actor_completion_attempt_count` | sampleable化境界でActor近似初期化を試みた累積件数 | 件数 | 単調増加を確認するだけでtrial間比較には使わない | mode配線・処理停止の障害調査 | なし |
| `replaybuffer.per.actor_completion_success_count` | `actor_initial`を正常適用できた累積件数 | 件数 | attemptとともに増えることを確認する | completion経路の障害調査 | なし |
| `replaybuffer.per.actor_completion_success_ratio` | Actor近似初期化を正常適用できた割合 | completion attempt数、比率 | non-truncated主体の環境では1付近。truncationが多い環境ではfallback率と合わせて読む | ハイパラ探索結果の原因分析 | completion attempt数が0 |
| `replaybuffer.per.actor_truncation_fallback_count` | `truncated && !done`により`max_initial`へfallbackした累積件数 | 件数 | 環境契約由来のため単独で良否判定しない | truncation頻度とfallback影響の障害調査 | なし |
| `replaybuffer.per.actor_truncation_fallback_ratio` | truncation fallbackの割合 | completion attempt数、比率 | baselineとの差ではなく環境のtruncation率と整合するかを見る | ハイパラ探索結果の原因分析 | completion attempt数が0 |
| `replaybuffer.per.actor_nonfinite_fallback_count` | Release系で推定器がActor payloadまたは算出priorityのnonfiniteを検出し、`max_initial`へfallbackした累積件数 | 件数 | 原則0。1以上なら数値異常として扱う。Debugではfallbackせず停止する | 障害調査 | なし |
| `replaybuffer.per.actor_nonfinite_fallback_ratio` | nonfinite fallbackの割合 | completion attempt数、比率 | 原則0 | ハイパラ探索結果の健全性確認 | completion attempt数が0 |

#### 8.2 ハイパラ探索で見る最小セット

Viewer tagは`39_agent_per/`配下だけを互換性なしで再編する。`metrics.scalar.baseline`と`metrics.scalar.full`は独立profileなので、A/Bは両方へ明示的に定義し、Cは`full`だけへ定義する。`39_agent_per/`でPERは明らかなためtag suffix先頭の`per_`は省略するが、suffix単独でも意味が分かる名前を使う。

| 優先度 | Profile・番号帯 | Viewer tag suffix | 判断目的 |
| --- | --- | --- | --- |
| A: trial比較で常に見る | `baseline` / `full`の01–08 | `actor_learner_rank_corr`、`actor_learner_log_ratio`、`sample_actor_init_ratio`、`evicted_unsampled_ratio`をraw＋`_ema`のペアで配置 | 順位付け品質、倍率校正、Actor値のsample曝露、経験の追い出しを独立に判定する |
| B: Aが悪いtrialで見る | `baseline` / `full`の50–65 | `actor_init_mass_ratio`、`actor_learner_pair_count`、`actor_learner_pos_ratio`、`actor_learner_ratio_median`、`actor_completion_ok_ratio`、`actor_trunc_fallback_ratio`、`actor_nonfinite_fb_ratio`、`priority_stale_ratio`をraw＋`_ema`のペアで配置 | Aの悪化がReplay質量への影響、標本不足、倍率ずれ、fallback、staleのどれに由来するかを特定する |
| C: 通常は非表示 | `full`だけの66–98 | TD/priority/IS分布、集約初期source、fixed/max sourceをraw＋`_ema`で配置し、94–98だけは累積countをrawで配置 | mode配線、source遷移、数値異常、pipeline障害を調べる |

`baseline`の`39_agent_per/`はA＋Bの24 tag、`full`はA＋B＋Cの57 tagとする。累積count 5項目以外は、事後的にEMAを再構成できない現行契約に合わせてraw直後へEMA版を必ず定義する。EMA係数は明示せず既定の`0.01`を使う。`EmaFilter`は`NaN`/Inf入力を更新対象から除外するため、pair不足や全tieの区間はEMAを汚染しない。単調増加する次の累積countには意味の薄いEMAを作らない。

`sample_actor_init_ratio`をActor初期値がLearner minibatchへ届いたことを示す優先度Aの主要曝露指標とする。`actor_init_mass_ratio`はそれだけで採用判断せず、sample比率に異常やtrial差が出たときにReplay全質量への影響を調べる優先度Bの補助指標とする。

- `actor_completion_attempt_count`
- `actor_completion_success_count`
- `actor_trunc_fallback_count`
- `actor_nonfinite_fallback_count`
- `priority_stale_drop_count`

tag名の短縮語は`init`、`corr`、`pos`、`fb`、`ok`に限定する。`per_is_rss_ratio`という旧tag typoは`is_ess_ratio`へ修正し、旧tagとの互換性は持たない。終盤報酬または主目的scoreと`exp_step_per_sec`は`39_agent_per/`外の既存tagを利用し、この再編では移動しない。

探索時は次の規則を適用する。

- baselineと`actor_approx`は同一seed条件かつ同一`exp_step`区間で比較する。時刻やwall-clock長だけを揃えた比較は行わない。
- Spearman、中央値、平均log比は`per_actor_learner_pair_count`が十分な区間だけ採用する。pair不足や全tieによる`NaN`は低性能値へ置換せず、比較対象外とする。
- `per_actor_learner_log_ratio_mean`は0付近、`per_actor_learner_ratio_median`は1付近を校正目安とする。Spearmanが高くても倍率が極端なtrialは採用候補ではなく診断対象とする。
- Actorのsample比率とmass比率は「高いほど良い」objectiveではない。Actor初期値がReplayとLearnerへ影響したことを確認する曝露指標として使う。
- 主目的score、throughput、未サンプル追い出し率、Spearmanを単一の合成scoreへ混ぜない。終盤報酬低下、baseline比3%超のthroughput低下、追い出し悪化、弱い順位相関を個別の棄却信号とする。
- nonfinite fallbackは原則0とする。stale率はReplay容量、replay ratio、pipeline/prefetch構成と合わせ、truncation fallback率は環境のtruncation契約と合わせて判断する。

### 9. 実行性能と互換性の方針

- 本機能のためにActor、Trainer、Learner、ReplayBufferへ新しいネットワークforwardを追加しない。
- 想定する実行時コストは、連続`float32[B,K]`carrier（DQNでは`K = 2`）の抽出、1stepあたり物理1本の小さなD2H（Push実行スレッド）、inline保持するopaque行のコピー、スカラー優先度計算、per-slot generation、診断処理だけとする。
- `fixed`と`max`モード、および評価/target action選択ではReplay初期優先度ヒント用メタデータを確保または転送しない。一様ReplayBufferではcompletion用FIFOとcounterも確保しない。
- `ReplayBuffer::Push`、Agent、network、optimizerのarchive契約は維持する。既存チェックポイントはリプレイ内容を直列化しないためarchive移行は不要とする。一方、Sampleのkey/source metadataと`UpdatePriorities`の引数名・戻り値は意図的に変更し、raw物理index互換は持たない。
- `fixed`を既定値とし、非負の`per_initial_priority`を使う既存の実行設定はそのまま有効とする。負値を使う既存設定は`per_initial_priority_mode = max`へ移行する（「設定」の章を参照）。

## 設定

追加・変更する設定項目を示す。キーはDQN系AgentのLearner設定配下（例: `DefaultDQNAgent.learner.per_initial_priority_mode`）である。

| 設定キー | 区分 | 値 | 既定値 | 意味 |
| --- | --- | --- | --- | --- |
| `learner.per_initial_priority_mode` | 新規 | `fixed` / `max` / `actor_approx` | `fixed` | PER新規遷移の初期優先度方式。`fixed`=固定値、`max`=Learner更新済み優先度の最大値、`actor_approx`=近似Actor初期化（フォールバックは`max`と同一動作） |
| `learner.per_initial_priority` | 意味変更 | 非負finite float | 1.0 | `fixed`モードの初期優先度。0も有効。`fixed`以外では非負finite値を使用しない。負値は全モードでエラー（`per_initial_priority_mode = max`へ移行） |

バリデーション:

- `use_per = false`かつ`per_initial_priority_mode`が`fixed`以外の組み合わせは、両設定と修正方法を示してエラーにする。
- `per_initial_priority`の負値は全モードで、`per_initial_priority_mode = max`への移行方法を示してエラーにする。NaN/Infも全モードでエラーにする。`fixed`以外の非負finite指定だけを、A/B比較のためエラーやWARNなしで無視する。
- `per_eps = 0`は全モードで許可し、PER有効時は`"zero-TD-error transitions may receive zero sampling priority"`を起動時に1回WARNする。負値・NaN・Infは全モードでエラーにする。
- `actor_approx`かつ`per_alpha = 0`は許可し、`"Actor priority does not affect sampling when per_alpha is 0"`を起動時に1回WARNする。source/useメトリクスは有効だが、adjusted-spaceの順位相関は全tieなら`NaN`になる。PERのalpha/beta方針自体は本PRDで変更しない。
- `use_per_prio_clip = true`のとき、`per_prio_clip_value = 0`を許可して近似Actor初期優先度とLearner優先度がすべて0へclipされる旨を1回WARNする。負値・NaN・Infはエラーにする。`0 < per_prio_clip_value <= per_eps`も許可するが、優先度差が潰れ得る旨を1回WARNする。`use_per_prio_clip = false`では値を使用せず、検証やWARNを行わない。
- SumTree全質量0は、`per_alpha > 0`で優先度0を許可する設定により生じる有効な状態とする。Samplerはrejection samplingを行わず直接uniform samplingへフォールバックし、全質量0を初めて検出したときだけWARNする。`fixed`初期値0は正のLearner優先度が適用されれば質量が回復する一方、`per_prio_clip_value = 0`ではLearner更新後も全raw priorityが0なので実行中はuniform samplingを継続する。後者でもsample済みitemのsourceは`learner_updated`へ遷移し、source遷移と質量回復を同一視しない。`per_alpha = 0`ではadjusted leafが全tieとなる別契約であり、全質量0フォールバックには入らない。
- 全モードで、丸め後の`capacity_per_env >= max(1, n_step) + 1`を構築時に要求する。不足時は`replay_capacity`、`num_envs`、丸め後容量、必要最小値をエラーへ含める。

既存configの移行（本PRDの作業に含める）:

- `apps/runner/config/LunarLander.txt`の`R.learner.per_initial_priority = -1`を`R.learner.per_initial_priority_mode = max`へ書き換える。

## テスト方針

- 外部から観測できる動作と安定したインターフェースを中心にテストし、production APIへtest-only経路を追加しない。任意leaf設定が必要な低水準テストは内部の物理slot APIを直接テストし、外部`UpdatePriorities`へraw indexを再導入しない。
- `ReplayInitialPriorityHint`について、rank、`float32`、batch size、`K > 0`、detach/contiguous、CPU cacheを検証する。validity tensorが存在せず、GPU出生時もCPU化が物理D2H 1本で再利用されることを検証する。
- 初期優先度推定器について、DQN payloadの`K = 2`、列decode、QR平均Q、1-step、n-step、真の終端、TBO、有限値、優先度0、`ValidateHint`、`nullopt`をテストする。schema不一致はエラー、nonfiniteはfalseまたは`nullopt`になることを区別する。`TransformH`/`TransformHInv`と優先度確定policyのtensor/scalar版が代表値・境界値で数値一致することを検証し、非TBOまたはスカラーDQN構成では推定器とLearnerのtarget/priorityも同一入力で一致させる（QR×TBOは変換と平均の順序差により原理的に不一致）。
- 探索行動を含む最終実行行動からpack済み`actor_q_sa`/`actor_state_value`が生成されることを検証する。`WithAction`は同じDQN schema helperを通じて既存Qから差し替え後の`actor_q_sa`を再計算し、追加forwardやフォールバックを行わないことを検証する。
- `fixed`/`max`モード、評価Actor、Learner内部のtarget action選択ではReplay初期優先度ヒントが生成されないことを検証する。forward回数を数えるfake networkまたは同等の公開境界を使い、`actor_approx`有効化でもforward回数が増えないことを証明する。一様ReplayBufferではcompleterと完成待ちFIFOを生成せず、Pushを継続しても未消費pendingを蓄積しないことを内部component境界で検証する。
- 非終端遷移がサンプリング可能になった時点でのみ、開始stepとブートストラップstepのヒントから初期化されることを検証する。true terminalはbootstrapを検査せず、truncatedは`max_initial`へフォールバックして累積理由カウンタへ入ることを検証する。
- carrier未設定、共通形式不整合、DQN schema不一致、非終端遷移のブートストラップ欠落・論理時刻不整合、負の算出優先度をエラーとして検証し、優先度0は`actor_initial`/`fixed_initial`を維持することを検証する。
- Actor payloadのNaN/Infは、truncatedの開始hintを含めてDebugで停止し、Release/RelWithDebInfoで`max_initial`へフォールバックすることを構成別に検証する。true terminalは空のbootstrap spanで成功し、非終端だけbootstrap hintを要求することを検証する。Learner優先度のNaN/Infは全buildでSumTree更新前に停止することを検証する。
- 負値・NaN/Infの`per_initial_priority`、負値・NaN/Infの`per_eps`、clip設定、`per_alpha = 0`のWARN、容量不足について設定契約を検証する。`per_initial_priority_mode = max`が旧負値指定と同じ最大優先度初期化になることも検証する。`per_clipped_count`はclip前priorityが上限未満・等値・超過の3境界で、超過した要素だけを数えることを検証する。
- `per_alpha > 0`のSumTree全質量0ではrejection samplingを経ずuniform samplingでき、WARNが初回1回だけであることを検証する。`fixed = 0`は正のLearner優先度でPER samplingへ復帰し、`per_prio_clip_value = 0`はsourceを`learner_updated`へ遷移させながら全質量0とuniform samplingを維持することを検証する。`per_alpha = 0`ではraw priority 0でも全tieの正質量となり、このfallbackへ入らないことを区別する。
- real/dummy書き込みでgenerationが進み、Sampleが`replay_item_keys`と`per_priority_sources`をCPU metadataとして返すことを検証する。丸め後`actual_capacity`をkey基数とSumTree容量の両方に使うことも、非整除capacityで検証する。
- generation一致、過去generation、未来generation、generation 0、負値、overflow、key/priorities長不一致を検証する。プログラムエラー時はbatchを部分適用せず、stale時はその要素だけを無視して他要素を更新することを検証する。
- `F1 Sample -> write-behind Push overwrite -> F2 Sample -> F1 UpdatePriorities`を再現し、F1が新しいslotのleaf/source/最大優先度を変更せずstaleへ計上され、F2は新generationで更新できることを検証する。
- 同一batchのduplicate keyは入力順のlast-winsになり、Actor/Learner比較は最初の`actor_initial -> learner_updated`だけを1ペアとして記録することを検証する。
- source別サンプル比率、取得時点のsource別質量比率、累積フォールバック率、既存の集約メトリクスが整合することを検証する。Actor/Learner比較の正値ペア選別、中央値、平均log比、平均rankのSpearman、ペア不足・全tie時の`NaN`も決定的な値で検証する。
- 同期学習とpipeline学習でTrainer側のDQN分岐なしに同じヒントがReplayBufferへ届き、`PrefetchingReplayBuffer`が`ReplayPriorityUpdateResult`を対応するLearner minibatchへ返すことを検証する。
- 既存のReplayBuffer、DQN/QR Learner、TBO、転送、先読み、Trainerの回帰テストを実行する。
- 現行ReplayBuffer系の既知失敗は、次のテスト名だけをallowlistとする。完了条件は、allowlist外の失敗0、本PRD追加テスト全passである。allowlist内の失敗が減ることは許容するが、件数だけでbaselineを判定しない。
  - `ReplayBuffer n-step returns stop at episode_start without done`
  - `ReplayBuffer excludes wrapped samples whose frame stack would read overwritten frames`
  - `ReplayBuffer PER samples only safe wrapped frame-stack indices`
  - `ReplayBuffer wrapped sampleability honors both frame stack and unroll horizons`
  - `ReplayBuffer frame stacking starts a new stack at episode_start without done`
- 実験採用時は、基準モード（`fixed`または`max`）とActor近似モードを同じ`exp_step`区間で比較する。3%を超えるスループット低下、未サンプル追い出し悪化、弱い優先度相関、終盤報酬低下を別々の棄却信号とする。

## 対象外

- 分散Ape-XのActor/Learner processまたはリプレイサービス
- Actor側target network、Learner側target-network評価の正確な再現、正確なUQE/楽観的target選択（判断の記録と再訪条件はADR 0010）
- 優先度専用のネットワークforward
- `PipelineRunner`固有の前回/現在経験の結合
- ReplayBuffer全体の非同期優先度更新またはReanalyse型のtarget再生成
- すべての遷移を最低1回サンプリングする保証
- `replay_ratio`または近似Actor初期優先度floorの自動schedule
- PERのalpha/beta方針変更またはimportance sampling補正の削除
- MuZero、ImageCls、非DQNのリプレイ経路
- 既存PER履歴メトリクスの改名
- `per_initial_priority`の削除、または最大優先度初期化の計算方式（最大値追跡）自体の再設計
- Action Masking連動のstate value補正。`max_a Q_online`が無効行動のQを拾い得る問題は、Action Masking基盤の導入時に再訪する
- 既知の未対策バグ（ReplayBufferのepisode_start境界、wrap時のsampleability等）の修正

## 補足

- 目的はLearner優先度の正確な再現ではない。追加のネットワークコストなしで、初回サンプリング前の有用な順位付け信号を提供することである。
- Actor近似ではtarget networkとtarget policyの差を意図的に省くため、絶対値の一致より近似Actor初期優先度とLearner優先度の順位一致を重視する。
- `per_initial_priority`は`fixed`モード専用の固定基準として残す。Actor推定値を利用できない場合のフォールバックは最大優先度初期化とし、`per_initial_priority`を有効なActor推定値のfloorには暗黙利用しない。floorはActor由来の順位を平坦化し得るため、将来の別方針とする。
- 本機能は、保証された成績改善ではなく、まずリプレイ分布の変更として評価する。採用判断にはPERメトリクス、Actor/Learner一致度、スループット、同じ終盤区間の報酬が必要である。
- 本正式PRDは、以前の999近似Actor初期優先度草案を置き換える。
