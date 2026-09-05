# 評価セッションの N 本集約は BatchEnv デコレータと開始時動的グラントで行う

`eval_batch_size > 1` は eval 用 BatchEnv の lane 数として配線されているが、評価ループが最初の lane 終端で止まるため、最短 episode 1 本へ偏る。停止条件だけを延ばしても、同じ step の `@episode_end` は metrics gate で最初の 1 点しか記録されず、AtariEnv の `game_score` 等も完了 Step の直後に保存しなければ次の Step で失われる。ImageCls は batch-level の eval window を lane 0 だけの終端として報告する妥協も持つ。したがって、lane 数 L と評価本数 N を分離し、同じ network snapshot に対する論理 episode N 本を一つの評価セッションとして収集する必要がある。

**評価セッションの N 本集約は、eval 用 BatchEnv を包む `EvalSessionEnv` が担当する**ことを決定する。`EvalSessionEnv` は通常の spec / state / action / `Step()` を inner へ委譲する decorator であり、EvalRunner の actor 経路からは通常の BatchEnv と等価に見える。セッション操作には既存の`Reset()`と追加する`GetSessionResult()`だけを使い、domain 固有の終端判定、metrics記録、scheduleは所有しない。

`BatchEnvSpec`には閉じた`EpisodeScope { PER_LANE, SHARED }`を追加する。`PER_LANE`は1 laneを1 episode groupとしてG=L、`SHARED`は全laneを1 groupとしてG=1とする。既定は`PER_LANE`、JSON表現は`"per_lane"` / `"shared"`。DiscreteBatchEnv系とImageCls trainは`PER_LANE`、ImageCls evalは`SHARED`を宣言する。

採用episodeはR4の開始境界動的グラントで決める。セッション開始時はgroup index順に先頭`min(N, G)` groupへ採用権を発行する。採用episodeが完了した時点で発行済み採用権がN未満なら、そのgroupのauto-reset後の次episodeへ採用権を発行する。同一Stepで複数groupが完了した場合はgroup index順に残りの採用権を割り当てる。採用権はepisode開始後に変更せず、発行済み採用権がNに達した後のepisodeは集約しない。採用権のないgroupも通常どおりStep / auto-resetし、partial Resetやmasked StepはP2の契約へ追加しない。

P2対象の`PER_LANE`では全groupが同じ固定policy / Env分布から独立にepisodeを生成する。採用権は次episodeの結果を見る前に過去の履歴だけで発行されるため、各採用episodeの条件付き期待値は同じμである。採用数Nは固定なので`E[Σ return] = Nμ`となり、動的グラントでも標本平均は不偏である。異なるepisode分布を持つgroupの混在はP2対象外とする。

`EvalSessionEnv::Reset()`をセッション開始境界とする。保存済みの最終`continue_state`で全groupがfreshなら再利用し、一つでもnon-freshならinnerを全lane Resetする。初回もinnerをResetする。fresh再利用時のReset resultはcached `continue_state`とlane数分の空AuxDataで組み立て、直前StepのAuxDataを引き継がない。ImageClsではStepのAuxDataが採点済み旧batch、`continue_state`が先読み済み次batchを表すため、両者を混在させない。これにより前のnetwork snapshotで途中まで進んだepisodeを混ぜず、auto-reset直後の二重ResetによるN=1のRNG系列変化とImageClsの先読みbatch破棄を避ける。`BeginSession()`は追加しない。

`EvalSessionEnv::Step()`は採用groupの完了直後に購読済みenv scalarをcaptureし、採用権を発行したN episodeがすべて完了した時点で結果を確定する。未完了の採用episodeを捨てて最初のN完了だけを選ばない。`PER_LANE`は`inner->GetScalar(base_key, lane)`、`SHARED`は`inner->GetScalar(base_key, -1)`を1回読む。`EvalSessionResult::episode_returns`は採用episodeの完了順、同時完了はgroup index順とする。`GetSessionResult()`は進行中`nullopt`、完成後は次のResetまで同じ`EvalSessionResult { episode_returns }`を返す。`SessionComplete()`や`GetCompletedSession()`は追加しない。

RunManagerは現在のconfigured eval tagについて、`scope=EVAL`、`event=EPISODE_END`、`target=ENV`の`ScalarMetricSubscription`だけを抽出し、正確なsource key一覧をdecoratorへ渡す。N>1では`mean.` / `max.` / `min.` / `std.` prefixを起動時に必須とし、N=1の無prefix keyも完了時にidentity値としてcaptureする。購読対象外keyとindexed lookupはwrapper transparencyとしてinnerへ委譲できるが、セッション集約を保証するのは購読済みkeyだけである。

EvalRunnerは`RunSession(event_counts)`で`Sync()`、`EvalSessionEnv::Reset()`、結果完成までの`DoStep()`、最終`EpisodeEndEvent` 1回を所有する。group、採用権、scalar captureの詳細は知らない。configured eval中のper-group eventは抑制し、最終eventは`env=EvalSessionEnv`、`env_index=-1`とする。EpisodeEvalObserverはschedule、foreground / background、`WaitBackgroundEval`、例外伝播だけを担当する。（採用 episode とセッション完了のイベント分離、および対応する scalar 購読抽出の変更は [ADR 0037](0037-metrics-trace-channel-and-session-end-event.md) を参照。）

## Considered Options

- **終了順採用**: 走行中の全episodeから終了後に最初のN完了を選ぶ方式は、未完了の長いepisodeを打ち切って短いepisodeを重くするため却下した。開始境界で採用権を発行し、発行したN本をすべて待つ動的グラントとは区別する。
- **固定group quota**: 動的グラントと同じく不偏で、laneごとの採用本数がconfigだけで確定する。Stable-Baselines3のvector laneごとの固定episode targetはこの形に近い（[evaluation.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py)）。一方、episode長が不均一なほど最も遅いgroupのquota合計へ律速されるため、R3の効率動機を優先して採用しなかった。
- **採用権を持たないgroupの停止**: RLlibの余剰actorをidleにする方式は通常1 env / EnvRunnerのactor単位であり、vector env内のmasked Stepではない（[RLlib advanced evaluation](https://docs.ray.io/en/latest/rllib/rllib-advanced-api.html)、[SingleAgentEnvRunner](https://github.com/ray-project/ray/blob/master/rllib/env/single_agent_env_runner.py)）。TorchRLのpartial step/resetは専用maskを公開契約に含めている（[TorchRL vectorized environments](https://docs.pytorch.org/rl/main/reference/envs_vectorized.html)）。同等の契約を持たないP2では通常stepを続け、parkingは実測後の別設計へdeferする。
- **全セッション無条件Reset**: auto-reset直後にも再度ResetしてN=1のRNG系列を変え、ImageClsの先読みbatchを捨てるため、fresh-state再利用へ置き換えた。
- **Runner集中集約**: `$env` metricsを`$runner`へ全面移行させ、env固有の`game_score`や`hns57`を失うため却下した。
- **metrics gate変更**: bucket-crossingという共通metrics契約へ波及し、早く終わったlaneの確定値も復元できないため却下した。
- **ImageCls indexed GetScalar / Reset preload特例**: `SHARED`のglobal captureとfresh-state再利用で不要になったためcutした。

## Consequences

- 構造契約はframeworkが所有する。Reset結果は各groupの`episode_start`がすべてtrue、Step結果はgroup内の`done`、`truncated`、`continue_state.episode_start`がそれぞれ同値、laneごとに`continue_state.episode_start == (done || truncated)`、`n_episode_end`が完了group数と一致しなければならない。`done`と`truncated`は独立であり同時成立を許容する。共通pure helperをEvalSessionEnvのcapture前とRunnerBaseのreward / event解釈前に呼び、違反時はenv名、group / lane、mask、期待値、実値を含めてfail-fastする。
- episode returnは共通`EpisodeReturnAccumulator`で計算する。`PER_LANE`はlane別、`SHARED`は全lane・全stepのreward総和を1 returnとする。RunnerBaseとEvalSessionEnvは別instanceを持ち、configured evalでは通常のper-group event集計を迂回する。
- scalar集約はDiscreteBatchEnvBase、EvalSessionEnv、RunnerBaseで共通化する。`nullopt`はpoison、NaNは除外し、有効値0件はNaN、stdは有効値2件未満でNaN、2件以上は安定計算した母集団標準偏差とする。
- `$runner eps_total_reward`は`$runner mean.episode_return`、`$runner train_episode_reward`は`$runner max.episode_return`へクリーンブレークし、`min.episode_return` / `std.episode_return`も提供する。metrics tagとscalar JSONL形式は維持し、旧aliasやWARNは残さない。`EpisodeEndEvent::eps_total_reward`も削除する。
- `N < G`では先頭N groupだけへ採用権を発行する。残りgroupの処理が評価結果へ寄与しないためtagごとに1回WARNする。ImageCls evalはL=128でもSHAREDのG=1なのでN=1でWARNしない。
- 動的グラントではlaneごとの採用本数が実行結果に依存するが、同じseed・固定policy・決定論backendでは完了順、グラント列、結果順を再現できる。
- ImageCls evalは1 windowを1 episodeとして、全lane同時終端、`n_episode_end=1`、global scalar captureにする。reward metrics tagは削除し、accuracyを残す。
- DropMergeのP2出荷既定はL=N=1を維持する。一時test configでL=N=16を有効化できることだけを確認し、N / L / intervalは`WaitBackgroundEval`の停止時間を含むP3実測で一体として決める。
- EvalPanelはdecorator対象外。background evalのschedule、counts、step軸、例外伝播、PRD 912のsnapshot順序は変更しない。
- 詳細契約と受入条件は`docs/memo/done/060_eval_batch_episodes_10prd.md`に置く。
