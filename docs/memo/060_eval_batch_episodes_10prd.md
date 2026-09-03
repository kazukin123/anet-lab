# 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する

- 起票日: 2026-08-16
- グリル: 2026-08-30〜09-03（`/grill-with-docs`。裁定は §10・§11、決定の記録は ADR 0034）
- 状態: implementation ready（P1 は 2026-09-03 に採用決定。P2 は self-contained）
- 対象: `core/anet-core`（`env.hpp` / `env.cpp` の BatchEnv 群、`trainer.cpp` の EvalRunner / RunManager、`observers.cpp` の EpisodeEvalObserver）、`core/envs/imagecls1`（ImageClsEnv）、`apps/runner/config`（common.txt / Atari.txt / DropMerge.txt / ImageCls metrics）
- 関連:
  - ADR 0034（本 PRD の決定と却下案）
  - PRD 034 / ADR 0009（ImageCls batch-native。034 D5 の「eval は lane 0 代表」を本 PRD が supersede する。§12）
  - PRD 052 / ADR 0027（eval 定義とスケジュールの分離。052 が据え置いた EpisodeEvalObserver の実行機構側を本 PRD が変える）
  - PRD 061（eval スロットごとの ε / network 指定。独立。着手は本 PRD が先）
  - PRD 912（background eval の snapshot 順序。軸は直交だが、セッションが長くなると 912 の窓が広がる）
  - PRD 913（best checkpoint。本 PRD の N 本平均に依存）
  - 探索記録 `docs/experiments/default-dqn/atari/2026-08-17_baseline.md`（:13 比較上の注意、:21 eval の決定論性、:955 探索ブロック 14、:2132-2134 未解決項目。:2134 の「PRD 911 待ち」は本 PRD（060）を指す誤記と思われる）
- 発見経緯: Atari Breakout の A/B（2026-08-16）で eval 値の分散が大きく、文献比較に耐えないことが判明

## 1. 背景

`eval_batch_size` は設定として存在し、`CreateBatchEnv(tag, seed, eval_batch_size, ...)`（trainer.cpp:920）で eval 用 BatchEnv の lane 数として渡されている。しかし評価ループの停止条件が「どれか 1 lane が終端したら止める」であるため、**2 以上を指定しても 1 本ぶんの結果しか得られず、しかもその 1 本は最短エピソードに偏る**。

現状の実害は eval 値の分散である。Breakout で 32M step まで学習した Run の `51_eval1/10_game_score_mean`（旧名 `10_episode_score_mean`）直近 7 点は `[44, 57, 83, 56, 86, 97, 38]` で、38〜97 と 2.5 倍の幅がある。探索記録では後期 Breakout で 44〜227 の幅も観測されている。文献値（Nature DQN Breakout 401.2）は 30 エピソードの平均で報告されており、1 本評価のままでは学習曲線の判読にも文献比較にも精度が足りない。探索ブロック 14 では 40〜50M step の eval1 で 5% 差（n=25〜29）が判定不能だった（探索記録 :955）。

現在は 2.5M step の window 平均（3〜4 点）で代用しているが、これは「異なる時点の方策の平均」であり、単一時点の N 本平均とは意味が違う。

本 PRD は「1 回の評価セッションで採用エピソード N 本を完走させ、その集約を 1 点として記録する」契約を確定する。lane 数（並列度）と本数（統計サンプル数）は独立のキーにし、既定値では現行と挙動等価にする。ImageCls（native batch）は lane 0 代表の妥協を撤廃して同じ規則に載せる。

## 2. 現行コードで確定している事実（2026-09-02 時点）

### F1. metrics 経路は既に「1 セッション 1 点」であり、停止条件だけ変えても N 本平均は録れない

- `EpisodeEvalObserver::OnLearn` はイベントの `counts` を値コピーして評価セッション全体に渡す（observers.cpp:569）。`@episode_end` metrics の出力 step は `counts.GetByAxis(step_axis_)`（observers.cpp:807）なので、**同一セッション内の全 @episode_end は同じ step になる**。
- `IntervalGate::ShouldFire`（util.hpp:252-269）は bucket が厳密に増加したときだけ true。metrics の既定 interval は 1（observers.cpp:1219）で Atari.txt の M1 定義は interval 未指定。
- よって**1 セッションで記録されるのは最初の @episode_end 1 点だけ**で、2 個目以降は `gate_.ShouldFire`（observers.cpp:1009）で捨てられる。EMA だけは gate 前に毎イベント更新される（observers.cpp:992-1001「出力しない場合も更新」）。
- 帰結: 旧 PRD の「候補 A では 2 本目以降も記録されてしまう」は現行経路では起きない。逆に、停止条件をどう変えても**記録経路に手を入れない限り N 本平均は録れない**。「集約側の改修は不要」（旧 §3）は、同一 step で複数 lane が終端したときにしか成立せず、実測では 128 env でも 97.1% が単独完了だった（探索記録 :24）。

### F2. metrics observer は env_index を捨てる

`MetricsLogEpisodeEndObserver::OnEpisodeEnd` は `OnGenericUpdate(event.counts, event.agent, event.runner, event.env, nullptr, nullptr)` を呼び、`event.env_index` を渡さない（observers.hpp:390-393）。`target->GetScalar(key_)` は index 既定 -1 の**バッチ全体集約**であり、`$env mean.game_score` は「イベントを出した lane の値」ではなく「その瞬間に確定値を持つ全 lane の平均」である。

### F3. キー名と interval は旧 PRD の表記から変わっている

- `episode_score` / `episode_len` / `episode_frames` は `game_score` / `game_len` / `game_frames` へ改名済み（コミット 6db2dc5、2026-08-19）。改名理由は `episodic_life=true` で RL のエピソード境界（life 単位）と確定境界（ゲーム 1 回）が一致しないため（docs/design/220_atari_env.jp.md:194）。同コミットで `hns57` / `hns49` が追加された。
- metrics 定義は `M1.[51_eval1/10_game_score_mean] = $eval.[eval1] @episode_end $env mean.game_score`（Atari.txt:750）。11_game_score_mean_ema / 12_hns57_mean / 14_hns49_mean / 16_game_score_max / 20_game_len_mean / 22_game_frames_mean / 24_game_len_max / 30_lives_mean が同型。
- `train.eval_schedule.[eval1].interval` は現行 1000（Atari.txt:708、コメント「旧 2500」）。

### F4. env の確定値は「終端した Step が返った直後」にしか読めない

- AtariEnv の `completion_available_` は `Step()` 末尾で `real_done || truncated` のとき立ち（AtariEnv.cpp:575-580）、**`Step()` 冒頭（:531）で無条件に false に戻る**。`Reset()`（:492-526）はこの flag に触らない。
- BatchEnv の auto-reset は done になった同一 `Step()` 呼び出しの内側で走る（VectorizedDiscreteBatchEnv env.cpp:427-442、ThreadPoolDiscreteEnv env.cpp:569-584）が、`AtariEnv::Reset()` は確定値を消さないので、**BatchEnv の Step が返った直後に `GetScalar` を読めば確定値が読める**（AtariEnv_test.cpp:798-806 で固定）。消えるのは次の `Step()` だけである。
- 帰結: 旧候補 B の「完走した lane に no-op を送り続ける」は不成立（no-op でも Step 冒頭で消える）。N 本の確定値は、完走した Step の直後に誰かが控えなければ集約できない。
- `DiscreteBatchEnvBase::GetScalar`（env.cpp:269-342）の集約 prefix は `max.` / `mean.` / `min.` の 3 種で prefix 無しはエラー（:294-296）。`index >= 0` なら集約せず lane 直読み（:274-277）。1 lane でも `nullopt` なら全体 `nullopt`（:310-312）、NaN は分母から除外（:317-318）、全 lane NaN なら NaN（:331-334）。

### F5. ImageCls は eval で lane 0 だけを代表 episode にしている（PRD 034 が「既知の妥協」と明記）

- `ImageClsEnv` は `BatchEnvBase` を直接継承する native batch で（ImageClsEnv.hpp:43）、`BatchEnvBuilder::CreateBatchEnv` の factory 経路（env.cpp:686-690）で生成される。`DiscreteBatchEnvBase` の集約・auto-reset は通らない。
- eval の window 終端では **lane 0 だけ** `done` を立て（ImageClsEnv.cpp:164）、`continue_state.episode_start` も lane 0 だけ（:171）、`n_episode_end` は 1（:183）。train は全 lane 同時 done。
- 一方 `Reset()` は eval でも全 lane に `episode_start=1` を返し（:87）、Step 側の lane 0 規約と食い違っている。`GetScalar` は index を一律拒否する（:200-203）。
- PRD 034:1064 は lane 0 代表を「per-lane flags と batch-level 境界の不整合（既知の妥協として明記）」と記録し、根拠は 034:613「`MetricsLogEpisodeEndObserver` は EpisodeEndEvent ごとに env 全体の scalar を記録するため、1 window で B event を出すと同じ accuracy と EMA が B 回前進する」の 1 点である。034:1063 は「将来の driver 側 protocol knob『1 評価あたり episode 数』（n_eval_episodes 相当、現在は 1 固定）は本決定と直交して追加可能」とも書いている。
- `Reset()` は `source_->NextBatch()` を呼ぶ（:81）。`Step()` は次バッチを先読みして `current_batch_ = next_batch` にしている（:155-157, :185）ため、**Step 後に Reset を呼ぶと先読み済みの未採点バッチが 1 つ捨てられる**。

### F6. `episodic_life` は eval env 側の overlay で false にできる

`train.eval.[tag].env.*` は eval env の SingleEnv 設定に効く（config_prefix、trainer.cpp:920）。`train.eval.[tag].env.episodic_life = false` は現行機構でそのまま書ける。現用の `AtariEnv.@v5_noop30`（Atari.txt:26 で選択）は `episodic_life = false` であり、旧 §D の症状は `@classic` / `@100k` 固有である。`fire_reset=true` でも life loss 後の FIRE は soft-reset 経路（AtariEnv.cpp:497-513）でしか打たれないため、eval で `episodic_life=false` にすると life loss 後は方策の確率性（v5: sticky 0.25、Classic: eval ε=0.05 は PRD 061）で FIRE を出すことになる。これは Nature 2015 の評価条件と同じである。

### F7. eval の worker スレッドはタグ単位で絞れない

`BatchEnvBuilderConfig` はグローバル prefix `env` 固定（env.hpp:245-262）で、`ResolveWorkerThreads` は `env.worker_threads` をそのまま使う（env.cpp:671-674）。現行は `-1`（AUTO = `min(num_envs, 論理コア-2)`、common.txt:56）。`eval_batch_size = 1` なら `VectorizedDiscreteBatchEnv`（追加スレッド 0）、L > 1 なら eval タグごとに `min(L, 論理コア-2)` 本の PinnedThreadPool が新規生成される（env.cpp:703-713）。投入は `i % worker_count` の固定 round-robin で Step ごとに `WaitAll()`（env.cpp:546-590）。

### F8. その他

- lane seed は `SeedMaker::MakeIndexedSeed(i) = splitmix64(base + i)`（env.cpp:143-152、random.cpp:195-198）で決定論的に分配され、AtariEnv では ALE 内部 RNG（sticky）と env RNG（noop 回数、AtariEnv.cpp:437）の 2 系統に分かれる。`@v5_noop30` / `@classic` / `@100k` のいずれも lane は独立である（旧設定 `@v5_noop0` + Greedy ε=0 だけが同一軌道の複製になっていた。探索記録 :955）。
- `train.eval.[tag].clone_model` は既定 true（trainer.cpp:887-888）。`EvalRunner::Sync()` はセッション冒頭に 1 回だけ走る（observers.cpp:527）ため、セッション中の重みは固定である。
- `EvalRunner::DoStep` は `env_initialized_` が false のとき（プロセス中 1 回だけ）`env_->Reset()` を呼ぶ（trainer.cpp:278-284）。以降は auto-reset 任せなので、**完走しなかった lane の途中エピソードは次セッションへ持ち越される**（複数の重み世代にまたがる軌道になる）。
- `RunnerBase::AccumulateAndNotifyEpisodeEnd`（trainer.cpp:129-185）は第 1 パスで done|truncated の lane があれば `last_step_had_episode_end_ = true` を立て、第 2 パスで終端 lane ごとに `EpisodeEndEvent{runner, counts, agent, env, env_index, eps_total_reward}` を通知する（rl.hpp:917-924）。
- eval runner の `step_counts_.episode_count` は metrics に消費されていない（`$runner episode_count` は train scope の `10_train/11_eps_count` のみ）。
- 購読ヒント `ScalarMetricSubscription{source_key, event, target, interval, scope, eval_name}`（rl.hpp:891-898）は `RunManager` 初期化の末尾で attach 済み定義から生成され `agent_->ConfigureScalarMetricSubscriptions` へ渡される（trainer.cpp:990-994）。`scope = EVAL` / `eval_name = tag` / `target = ENV` / `event = EPISODE_END` でフィルタすれば「その eval タグの @episode_end で読まれる `$env` キー一覧」になる。
- EvalPanel の eval 経路は `eval_batch_size` を無視して num_envs=1 をハードコードしている（trainer.cpp:1068-1069）。`EvalRunner::DoStep` の `WithAction` 強制は 1 要素テンソルで（trainer.cpp:296-300）、到達元は GUI のみ。

## 3. 問題

### A. 最短エピソードバイアス

N lane のうち最初に終わった 1 本だけが採用される。Breakout / Pong では「早く終わる = 下手」なので、評価値が系統的に下振れする。lane 数を増やすほどバイアスが強くなる（最小値統計に近づく）。「終わった順に N 本」へ変えても同型のバイアスが残る（§4.2）。

### B. 計算の破棄

残り N-1 lane 分の env ステップと NN 推論が捨てられる。

### C. GPU 効率（隠れた利点）

現在の eval は batch=1 の CNN forward を 1 step ずつ回しており GPU がほぼ遊んでいる（`train.eval_device_type = cuda`）。lane 数を増やしても GPU 推論は 1 回のバッチ処理で済むため、N 本取るコストは N 倍にならない。

### D. episodic_life との相互作用（Classic 固有、設定で解く）

`episodic_life=true` では life-loss でも done が立つため、eval が 1 ライフで停止し、実ゲーム完走まで 5 セッションを要する（実測: eval1 の記録間隔が 5 倍、1 ゲームが 5 時点の方策にまたがる）。これは eval env の overlay `env.episodic_life = false` で発生源から消す（§4.8、F6。P1 として採用決定）。旧 PRD の「完走判定を `completion_available_` に置く」機構は建てない（§10）。


### 3.1 ゴールと非ゴール

- ゴール: `eval_batch_size`（並列度）と `eval_episodes`（統計サンプル数）を分離し、1 回の network snapshot に対する N episode の集約を `@episode_end` 1 点として記録する。Atari / DropMerge / ImageCls の契約を同じモデルで説明できる状態にする。
- 非ゴール: P3 の runtime 値を本 PRD の実装時に固定すること、PRD 061 の eval policy override、PRD 912 の snapshot 順序、過去 Run artifact の変換。

## 4. 確定契約

### 4.1 設定キー（`train.eval.[tag]` 定義側）

本PRDでは`L = eval_batch_size`（物理lane数・並列度）、`N = eval_episodes`（1評価点へ集約する論理episode数）、`G = episode group count`（episode scopeから導出）と表記する。

| キー | 意味 | 既定 | 検証 |
|---|---|---|---|
| `eval_batch_size` | eval 用 BatchEnv の lane 数＝並列度 | 1 | 正整数以外は fail-fast（現行どおり） |
| `eval_episodes`（新設） | 1 評価セッションで完走・集計する論理 episode group の本数 | 1 | 正整数以外は fail-fast |

- `eval_episodes` は lane 数ではなく、§4.2 の `episode_scope` が定義する episode group の本数を数える。
- `eval_episodes < episode_group_count` は、採用権を一度も受け取らない group が生じ、その処理が評価結果へ寄与しないためタグごと 1 回 WARN する。`episode_group_count` は `PER_LANE` なら lane 数、`SHARED` なら 1。したがって ImageCls eval の L=128, N=1 は WARN しない。
- `apps/runner/config/common.txt` に eval1 / eval2 の `eval_episodes = 1` をデフォルト直書きする。
- `eval_episodes > 1` のタグでは、`$eval.[tag] @episode_end $env <key>` の `<key>` に集約 prefix（`max.` / `mean.` / `min.` / `std.`）を必須とし、無 prefix は起動時に fail-fast する。N=1 の無 prefixは採用 episode の完了時に identity 値としてキャプチャし、後から inner env へ委譲しない。
- `eval_episodes` の消費者は EvalSessionEnv、`eval_batch_size` の消費者は BatchEnvBuilder のままとする。

### 4.2 BatchEnvSpec の episode scope

`BatchEnvSpec` に `EpisodeScope episode_scope` を追加する。既定は `PER_LANE` とし、`ToJson()` は `"per_lane"` / `"shared"` の文字列で出力する。

| scope | 論理 episode group | 完了と reward の意味 |
|---|---|---|
| `PER_LANE` | 1 lane = 1 group（group 数 L） | lane ごとに完了。reward は当該 lane の episode 内総和 |
| `SHARED` | 全 lane = 1 group（group 数 1） | 全 lane が同時に完了。reward は全 step・全 lane の総和 |

- DiscreteBatchEnvBase 系と ImageCls train は `PER_LANE`、ImageCls eval は `SHARED`。
- `episode_start` / `done` / `truncated` は各 group 内でそれぞれ全要素が一致しなければならない。`done` と `truncated` は独立した flag とし、同時成立を許容する。
- `Step()` では lane ごとに `continue_state.episode_start[i] == (done[i] || truncated[i])` を必須とする。`next_state.episode_start` はこの対応検証に使わない。
- `n_episode_end` はその Step で完了した group 数と一致しなければならない。`Reset()` 結果では各 group の全 lane が `episode_start = true` でなければならない。
- 上記はフレームワークが所有する構造契約として fail-fast する。共通の pure helper を `EvalSessionEnv` では capture 前、`RunnerBase` では reward / event 解釈前に呼ぶ。エラーには env name、group index、lane index、違反した mask 名、期待値、実値を含める。

### 4.3 評価セッションと採用規則（R4: 開始時の動的グラント）

- **評価セッション** = eval schedule の 1 回の発火で行う評価の単位。`Sync()` → session state の開始 → 採用 episode group N 本の完走 → `@episode_end` 1 点。
- セッション開始時、group index 順に先頭 `min(N, G)` group の現在 episode へ採用権を発行する。採用 episode が完了した時点で発行済み採用権が N 未満なら、その group の auto-reset 後の次 episode へ採用権を発行する。同一 Step で複数 group が完了した場合は group index 順に残りの採用権を割り当てる。
- 採用権は episode の開始境界でだけ発行し、開始後に採否を変えたり、途中 episode を採用したりしない。発行済み採用権が N に達した後の episode も通常どおり Step / auto-reset するが、return と scalar は集約しない。
- セッション終了条件は「採用権を発行した N episode がすべて完了したこと」。未完了の長い採用 episode を捨てて最初の N 完了だけを選ぶ終了順採用は禁止する。`EvalSessionResult::episode_returns` は採用 episode の完了順とし、同時完了は group index 順に格納する。
- `N < G` では先頭 N group だけが採用権を持ち、残りの group は評価結果へ寄与しない。全 group は通常どおり Step / auto-reset する。
- P2 対象の `PER_LANE` は、全 group が同じ固定 policy / Env 分布から独立に episode を生成することを前提とする。各採用権は次 episode の結果を見る前に、それまでの履歴だけで発行されるため、各採用 episode の条件付き期待値は常に同じ μ である。採用数 N は固定なので `E[Σ return] = Nμ`、標本平均は不偏になる。異なる episode 分布を持つ group を一つの `PER_LANE` 評価へ混在させることは P2 の対象外とする。
- セッション開始時、キャッシュした直前の `continue_state` で**全 group が fresh**（group 内の全 lane が `episode_start = true`）ならその state を再利用する。初回、または 1 group でも non-fresh なら inner を全 lane `Reset()` する。これにより PER_LANE L=1,N=1 の軌道・RNGを維持し、ImageCls eval の先読み済み batch も捨てない。
- fresh 再利用時の `Reset()` 戻り値は、`state` に cached `continue_state`、AuxData に lane 数 L 個の空 `AuxData` を持つ `BatchResetResult` とする。直前 Step の AuxData は終端側の情報であり、再利用する開始 state の情報ではないため引き継がない。
- partial Reset / masked Step は P2 に導入しない。採用権を持たない group を Step 対象から外す最適化は、P3 の実測で必要性が出た場合に別契約として設計する。

### 4.4 EvalSessionEnv（BatchEnv デコレータ）

metrics は `EpisodeEndEvent.env->GetScalar` を読む一方、inner env の確定値は完了 Step 直後にしか読めない。eval 用 BatchEnv を包む `EvalSessionEnv` が、採用権の発行・キャプチャ・集約・結果完成判定を同じ境界で所有する。domain 固有の episode 終端判定、metrics の記録、eval schedule は所有しない。

- `BatchEnvBase` を実装して通常の spec / state / action / `Step()` 結果を inner へ委譲し、EvalRunner の actor 経路からは通常の BatchEnv と等価に見せる。RunManager が configured eval tag の BatchEnv を常に包み、EvalPanel は対象外。
- 構築時に `eval_episodes` と、その eval tag の `scope=EVAL && target=ENV && event=EPISODE_END` の購読 key を受け取る。N>1 の prefix 必須検証を起動時に行う。
- `Reset()` をセッション開始境界とし、§4.3 の fresh 判定に従って cached `continue_state` の再利用または inner の全 lane Reset を選び、発行済み採用権数、完了済み採用 episode 数、group ごとの現在の採用状態、集約 state を初期化する。fresh 再利用時は cached state と lane 数分の空 AuxData から Reset result を組み立て、直前 Step の AuxData を載せない。ImageCls では Step の AuxData が採点済み旧 batch、`continue_state` が先読み済み次 batch を表すため、両者を混在させない。`BeginSession()` は追加しない。
- `Step()` は inner の結果を §4.2 の共通 helper で検証し、reward を `EpisodeReturnAccumulator` へ渡す。完了した return のうち採用権を持つ group だけを集約し、その Step 直後に購読 base key をキャプチャして、group index 順に完了処理と次 episode への再グラントを行う。`PER_LANE` は `inner->GetScalar(key, lane)`、`SHARED` は group 全体を `inner->GetScalar(key, -1)` で 1 回読む。採用権のない group の完了は return / scalar / 結果件数へ反映しない。
- `GetScalar(key, -1)` は購読済み key について、採用 episode のキャプチャ値に対する `mean` / `max` / `min` / `std` を返す。N=1 の無 prefix key はキャプチャした identity 値を返す。購読対象外 key と `index >= 0` は通常の decorator transparency として inner へ委譲するが、セッション集約を保証するのは購読済み key だけとする。
- セッション結果は次の API で公開する。進行中は `std::nullopt`、完成後は次の `Reset()` まで同じ値を返す。`SessionComplete()` / `GetCompletedSession()` は追加しない。

```cpp
struct EvalSessionResult {
  std::vector<float> episode_returns;
};

std::optional<EvalSessionResult> GetSessionResult() const;
```

- `ANET_PROFILE_SCOPE` は `EvalSessionEnv::Step` の capture を含む意味のある処理境界へ置く。

### 4.5 RunnerBase / EvalRunner / EpisodeEvalObserver

- RunnerBase と EvalSessionEnv に重複している reward 集計を、`BatchStepResult + EpisodeScope` を入力し、完了した `{group_index, episode_return}` を返す共通 `EpisodeReturnAccumulator` へ置き換える。両者は別 instance を持つ。
- `PER_LANE` は lane ごとに reward を加算し、`SHARED` は各 Step の全 lane reward を一つの group return に加算する。train の `train_episode_reward` 相当は、直近 Step で完了した return 群の max という現行意味を維持する。
- `EvalRunner::RunSession(event_counts)` が `Sync()` → `EvalSessionEnv::Reset()` → `GetSessionResult()` が値を返すまで `DoStep()` → 最終 `EpisodeEndEvent` 1 回、を所有する。EvalRunner は group、採用権、capture の詳細を知らない。
- configured eval では通常の per-group event / return 集計を迂回し、中間 event を抑制する。最終 event は `env` にデコレータ、`env_index=-1`を設定し、return payload は持たせない。`EpisodeEndEvent::eps_total_reward` は production consumer がないことを確認して削除する。
- `EpisodeEvalObserver` は schedule、foreground / background、`WaitBackgroundEval`、例外伝播だけを所有し、`RunSession()` を呼ぶ。`counts` と step 座標系は変更しない。

### 4.6 集約 prefix

- `DiscreteBatchEnvBase`、EvalSessionEnv、RunnerBase は同じ key parser / accumulator を使い、`max.` / `mean.` / `min.` / `std.` を一般 API として提供する。`std.` は session 専用機能にしない。
- 共通要素は `ScalarAggregation { MEAN, MAX, MIN, STD }`、`ScalarAggregationKey { aggregation, base_key }`、`ScalarSampleAccumulator::Add(std::optional<float>) / Get(ScalarAggregation)` 相当とする。
- `nullopt` は key 未認識として poison、NaN は値未成立として除外する。有効値 0 件は全集約で NaN、`std.` は有効値 2 件未満で NaN、2 件以上では Welford 法相当の安定計算による母集団標準偏差を返す。
- `se.` prefix は追加しない。標準誤差が必要なら P3 で `std.<key> / sqrt(N)` を概算として読む。

### 4.7 ImageClsEnv（shared episode）

- train は `episode_scope=PER_LANE`、eval は `episode_scope=SHARED` を返す。
- eval window 終端では `done` と `continue_state.episode_start` を全 lane に立て、`n_episode_end=1` とする。N は画像 lane 数ではなく評価 window 数を数える。
- SHARED のキャプチャは `GetScalar(key, -1)` を使うため、ImageCls に indexed `GetScalar` は追加しない。
- §4.3 の fresh state 再利用により Step 後の先読み済み batch を次 session の先頭に使えるため、ImageCls `Reset()` に preloaded-batch 特例を追加しない。
- ImageCls config の `21_eval/01_test_reward` / `02_test_reward_ema` は削除し、正準 metrics の accuracy を残す。SHARED episode return は全 lane・全 step の reward 総和として共通 accumulator から得る。

### 4.8 Atari / DropMerge の設定

- **P1（採用決定 2026-09-03）**: configured eval の eval1 / eval2 と GUI 観戦用 eval_panel で `env.episodic_life=false` を上書きし、Classic / 100k / v5 の評価単位を実ゲームに揃える。AtariEnv のコードは変更しない。これは framework 機構ではなく、Classic で N が「N lives」になることを防ぐ Atari 固有の前提補正である。P1 の設定変更は次の3行に固定する。

```text
train.eval.[eval1].env.episodic_life = false
train.eval.[eval2].env.episodic_life = false
train.eval.[eval_panel].env.episodic_life = false
```

eval_panel は引き続き `EvalSessionEnv` の対象外とする。変更するのは AtariEnv の episode 境界だけで、GUI の schedule と既定の time-based model sync は変えない。episode-based sync を選んだ場合だけ、その同期境界も life loss から実ゲーム終了へ変わる。

#### P1 の採用根拠

**(a) ALE のゲーム進行と実ゲーム単位の scalar 集約は変わらない。** `AtariEnv::Reset()` は life-loss では ALE を維持し、`ale_->reset_game()` は実ゲーム境界でだけ呼ぶ（`AtariEnv.cpp:497-520`）。`completion_available_` が立つのも `real_done || truncated` の枝だけなので、`game_score` / `game_len` / `game_frames` は `episodic_life` に関係なく実ゲーム単位で確定する。一方、`done` に従属する方策入力の履歴は (c) のとおり変わりうる。

**(b) `episodic_life=true` の eval は測定として成立していない。** soft reset のためゲームは次の eval セッションへ持ち越される。結果として 5 セッション中 4 つは `game_score` を 1 本も記録せず、記録される 1 本は **5 つの異なる network snapshot でプレイされたゲーム**のスコアになる。どの方策のスコアかが定義できない。P1 は「eval の都合で条件を変える」のではなく、この不成立を直すものである。

**(c) eval の `done` はセッション管理だけでなく frame stack にも影響する。** 行動選択は `done` を直接参照せず、TD target も計算されないが、`StackerActionContext::PushObservation()` は `BatchState::episode_start` を `FrameStacker` へ渡し、frame stacker は true の lane の stack を現在フレームで初期化する（`stacker.cpp:21-42, 86-93, 133-140`）。life loss の `done` は auto-reset 後の `episode_start` に反映されるため、P1 前は life ごとに stack が切れ、P1 後は実ゲーム終了まで継続する。train の `episodic_life=true` とは観測履歴が変わるが、評価・testing では life loss を終端にしない標準的な構成であり、意図したプロトコル差とする（[Stable-Baselines3 `EpisodicLifeEnv`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py#L89-L100)）。

**(d) train=true / eval=false は標準的な構成である。** `episodic_life` は学習器のブートストラップ整形であってタスクの定義ではない。Machado et al. 2018 のプロトコルは life 情報を使わず、BTR も `--life_info` 既定 0 で eval している（`reports/btr_hyperparams_survey_2026-08-26.md` §7）。Classic / 100k の train 側は `episodic_life=true` のままでよい。

**制約**。プロファイルごとに `fire_reset` が異なる（`Atari.txt:1208,1214,1221,1227`）。

| プロファイル | `episodic_life` | `fire_reset` | P1 の影響 |
|---|---|---|---|
| `@v5_noop0` / `@v5_noop30` | false | false | 対象外（もともと false） |
| `@classic` | true | **true** | **frame stack が life loss をまたいで継続し、強制 FIRE も無くなる**（下記） |
| `@100k` | true | false | **frame stack が life loss をまたいで継続する** |

**`@classic` では P1 により軌跡が変わる。** `life_loss_pending_` の枝が `ApplyFireReset()` を呼ぶため、`episodic_life=true` では life loss ごとに env が FIRE を押していた。P1 後はその再投入が無くなり、**ボールの再投入は方策の確率的行動に委ねられる**。これは Nature 2015 の評価条件（ε=0.05）と同じ状況であり、採用判断を変えるものではないが、Classic の eval では **ε が life loss 後の FIRE を担う**ことになる。

したがって **eval ε > 0 の維持が必須である**（v5 系 0.01 = `A2.eval_policy.eps_start/end`、Classic 系 0.05 = `run.@classic_iqn_impala_x2` / `run.@nature_dqn` の上書き）。**ε=0 との併用は不可。** 加えて P1 後は eval エピソードが約 5 倍長くなるため、FIRE デッドロックへの露出時間も同じだけ増える。
- DropMerge は P2 で eval1 / eval2 の既定を変更せず、`eval_batch_size=1`、`eval_episodes=1` を維持する。一時的な test config で `L=N=16` を有効化でき、旧 workaround が不要になったことだけを受入確認する。
- Atari と DropMerge の N / L / interval は P3 で測って一体で決める。Atari の `game_score_std` は既定 metrics に追加せず、必要な P3 tag だけ既存 `std.game_score` API を使う。

### 4.9 metrics 定義への影響

- 既存の metrics tag と step 座標系、scalar JSONL の `(tag, step, value)` 形式は変えない。`mean.game_score` は session N 本の平均、`max.game_score` は最大になる。
- `eval_episodes > 1` にした Run 以降は 1 点の母集団が変わるため、旧 Run と比較するときは N を併記する。
- 完了時にキャプチャする値はその Step の確定値であり、次 Step へ持ち越さない。
- Runner source key は `$runner eps_total_reward` を `$runner mean.episode_return`、`$runner train_episode_reward` を `$runner max.episode_return` へクリーンブレークし、`min.episode_return` / `std.episode_return` も同じ共通集約で提供する。旧 alias、旧 key 専用 WARN、旧定数は残さず、metrics.defs の source key を同じ変更で移行する。

## 5. 実装ノート（Codex 向け）

| 対象 | 変更 |
|---|---|
| `core/anet-core/include/anet/env.hpp` / `src/env.cpp` | `EpisodeScope` / `BatchEnvSpec`、group 構造検証 helper、EvalSessionEnv、EpisodeReturnAccumulator、共通 scalar key parser / accumulator（`std.` 含む） |
| `core/anet-core/include/anet/trainer.hpp` / `src/trainer.cpp` | `eval_episodes`、RunnerBase の共通 return 集約、EvalRunner::RunSession、Runner source key のクリーンブレーク、EpisodeEndEvent の return payload 削除 |
| `core/anet-core/src/observers.cpp` | EpisodeEvalObserver を session の schedule / background 制御へ限定し、`RunSession()` 呼び出しへ置換 |
| `core/envs/imagecls1/src/ImageClsEnv.cpp` | eval を SHARED episode として宣言し、全 lane 同時終端・`n_episode_end=1` |
| `apps/runner/config/common.txt` | eval1 / eval2 の `eval_episodes=1` |
| `apps/runner/config/Atari.txt` | P1 の eval1 / eval2 / eval_panel overlay（P2 と独立に反映可）。P2 / P3 の恒久 N / L / interval や既定 std metric は入れない |
| `apps/runner/config/DropMerge.txt` | P2 では L=N=1 を維持。L=N=16 は一時 test config だけで確認し、恒久値は P3 で決める |
| metrics config | Runner source key を `mean.episode_return` / `max.episode_return` へ移行。ImageCls の `01_test_reward` / `02_test_reward_ema` を削除し accuracy を維持 |
| docs | 本文書 / ADR / CONTEXT は先行して確定契約へ更新し、現行実装を説明する `docs/design` は P2 の code / config と同時に同期 |

実装は P2 を atomic に進める。順序は構造契約と prefix → EvalSessionEnv → Runner / Observer → ImageCls → config → docs → 受入テスト。公開 API を test-only seam で歪めない。

## 6. 受入条件

1. `BatchEnvSpec`: `PER_LANE` / `SHARED` と JSON 文字列、既定 `PER_LANE` を固定する。
2. 構造契約: group 内の各 mask 同値、`n_episode_end`、Reset start、`continue_state.episode_start == done|truncated` の正常系と各違反の fail-fast を固定する。`done && truncated` は許容し、エラーには env name / group / lane / mask / expected / actual を含める。
3. 動的グラント: PER_LANE L=2,N=3 は両 group の初回 episode を採用し、先に完了した group の次 episode へ3本目をグラントする。3本目より遅い他 group の初回採用 episode も最後まで待ち、最初の3完了だけを選ばない。L=4,N=2 は先頭2 groupだけへ初回グラントし、他groupの結果を無視する。同一Stepで複数groupが完了し残りグラントが不足する場合はgroup index順に割り当てる。SHARED G=1,N>1は同じgroupへ直列にグラントする。発行、capture、結果は常に正確にN件で、同じseed・決定論backendではグラント列と結果順が再現する。
4. fresh 開始: 全 group fresh なら cached `continue_state` を再利用し、初回または一部 non-fresh なら全 lane Reset する。N=1 の二重 Reset、RNG系列の余分な前進、ImageCls の先読み batch 破棄がない。fresh 再利用の単体テストでは inner `Reset()` の呼び出し回数が増えず、戻り値の state が cached `continue_state` と一致し、`GetAuxDataList()` が lane 数 L 個の空 `AuxData` を返して直前 Step の marker を含まないことを固定する。
5. SHARED: 全 lane の同時完了を1 episodeと数え、scalarはindex=-1で1回キャプチャし、episode returnは全 lane・全 stepの総和になる。
6. session lifecycle: `GetSessionResult()` は進行中 `nullopt`、完成後は次の Reset まで同じ raw `episode_returns` を返す。中間 event は抑制され、完了時にデコレータをenv、`env_index=-1`とするpayloadなしeventが1回だけ出る。
7. scalar / return: 共通utilityのmean/max/min/std、`nullopt` poison、NaN除外、0件NaN、1件std NaN、2件以上のpopulation stdをgolden traceで固定する。N>1無prefixはfail-fast、N=1無prefixは完了時captureのidentity値になる。
8. N=1非退行: PER_LANE L=1,N=1はstate / action / reward / eventのgolden traceが完全一致する。さらにAtari `@v5_noop30`と現行DropMerge設定を、同一seed・deterministic backend・sync eval・1回以上の完了evalでbase/branch比較し、`90_perf`と意図的変更tagを除いた順序付き `(tag, step, value)` を完全一致させる。この2設定では実質的に全non-perf scalarの一致を要求する。timestamp、JSON metadata / metrics.defs、raw JSONL byte、checkpoint hashは比較しない。
9. config: `eval_episodes<=0` は起動時error、N<Gは採用権を一度も受け取らないgroupが生じるためタグごと1回WARN。ImageCls eval L=128,N=1はG=1なのでWARNしない。DropMergeは既定L=N=1のまま、一時test configでL=N=16を有効化できる。
10. ImageCls: Nはwindow数を数え、global accuracy / epoch_countを完了時に取得する。indexed GetScalarやReset preload特例を追加せず、eval reward tag削除後もaccuracyを維持する。
11. RunnerBase train smoke: `out/test-tmp/prd060-smoke/<env>/` にGit管理外の一時main configとRun出力を作り、`apps/runner/bin/Debug/AnetRLRunner.exe --config <temporary-main>`でLunarLander / GridMaze / CartPole / ImageClsを各1回起動する。各main configは`apps/runner/config/_main.txt`の後に対象Env configをincludeし、次を上書きする。
    - 共通: `app.$=app.batchrun`、`train.num_envs=2`、`app.train_exit_step=12`、全configured eval scheduleの`interval=0`。一時metric `metrics.scalar.[00_smoke/01_episode_count] = $runner episode_count @train $runner interval:1`を追加する。
    - episode上限: `LunarLanderEnv.limit_step=4`、`GridMazeEnv.max_steps=4`、`CartPoleEnv.limit_step=4`、`ImageClsEnv.max_steps=4`。
    - 各Runでprocess exit code 0、構造契約errorなし、最終`episode_count >= 6`を要求する。少なくとも3世代のepisode完了をRunnerBaseの毎Step検証へ通す追加スモークであり、Atari / DropMergeのchecksum比較は置き換えない。

## 7. 成功指標（P3 の一回限りの運用検証）

同一 Run に同期 eval tag A/B/C を置き、A は N=1、B/C は N=30、`run_mode` / `interval` / `use_background=false` を揃える。tag ごとの seed domain は独立させる。これは手順を PRD へ残すだけで、恒久 profile や P2 受入条件にはしない。

- 30 回以上の paired trigger を取る。
- `σ30 ≈ std(B-C) / sqrt(2)`、`σ1 ≈ std(A-(B+C)/2)` とし、`σ1 / σ30 >= 4` を確認する（理想は `sqrt(30) ≈ 5.5`）。
- 同じ snapshot 内の概算指標 `std.game_score / sqrt(N)` を見て、Breakout で概ね 10 以下になる N を runtime 候補にする。厳密な不偏標準誤差ではなく、N=30 では補正差が小さい運用概算とする。
- DropMerge は代表的な eval1 / eval2 実行で各 session が次の同一 tag trigger より前に完了し、`WaitBackgroundEval` が train を待たせないことを昇格条件にする。満たさない場合は既定 L=N=1 を維持するか、N / L / interval の組を調整する。
- P3 で決めるのは Atari / DropMerge の runtime N / L / interval。framework code や既定 `game_score_std` metric を追加しない。

## 8. フェーズ

| Phase | 定義 | 完了条件・位置付け |
|---|---|---|
| P1 | **Atari 評価 episode 境界の正規化（config のみ、採用決定）**。eval1 / eval2 / eval_panel で `env.episodic_life=false` | configured eval の N を life 数ではなく game 数にし、GUI 観戦も同じ episode 境界へ揃える Atari 固有の前提補正。framework 実装の受入条件ではないので P2 と独立に反映できる |
| P2 | **N-episode session 機構の atomic 実装**。`eval_episodes`、episode scope、EvalSessionEnv、Runner/Observer、ImageCls、共通return/scalar集約、source key移行、tests、docs | Atari / DropMerge / ImageCls について §6 を満たす。DropMergeの出荷既定は変更しない。途中分割では契約が成立しないため1 phase |
| P3 | **任意の一回限りの分散・runtime測定**。§7のpaired検証、DropMerge停止時間確認、N/L/interval選定 | P2のcode acceptanceではない。恒久profile化と出荷既定変更は測定結果を見て別途判断 |

P1 は PRD060 の中核機構ではないが、Classic で `eval_episodes` の 1 本を「実ゲーム 1 回」と解釈するために関係する。P2 をブロックしないので独立に反映してよい。採用根拠は §4.8 を参照。

## 9. コスト見積もり（P3 で実測すること）

PER_LANEでgroup数G=Lなら、採用episodeを完了したgroupが残りの採用権を受け取るため、N>Gでは固定quotaよりlane間の仕事量を平準化できる。session長は概ね採用episodeの総step数をGで分担する時間に、最後まで残る採用episodeのtailを加えたものになる。N=Gでは各groupに1本だけをグラントするためL本の平均長ではなく最長episodeまで待つ。SHAREDはG=1なので、N windowを直列に評価する。

- GPU 推論は L lane を batch 化できる。CPU env は worker 並列度の影響を受ける。
- 全N件の採用権を発行した後も、採用権を持たないgroupを含む全groupを通常どおりStepするため、最後の採用episodeを待つtailではCPU浪費が残る。実測で支配的になった場合だけfreeze / Step除外を別設計する。
- background eval が interval をまたぐと次のgateでtrainが待つ。DropMergeはeval1/eval2が同位相で発火し、L>1ではタグごとにthread poolも増えるため、N/L/intervalは必ずセットで決める。
- PRD 912 の snapshot 時点は変更しない。`Sync()` 後の session 中は同じ network snapshot を使う。

## 10. 複雑さ監査（グリル 2026-09-03 最終）

| 機構 | 裁定 | 理由 |
|---|---|---|
| `eval_episodes` と `eval_batch_size` の分離 | keep | 統計本数と並列度は別の調整軸 |
| `EpisodeScope {PER_LANE, SHARED}` | keep | lane episode と ImageCls の batch window を同じ group モデルで表す最小宣言 |
| R4: 開始時の動的グラント | keep | 採用を開始境界で確定して不偏性を保ち、完了groupへ次の採用権を渡して固定quotaのtailを減らす |
| fresh なら continue state 再利用、他は全 Reset | keep | N=1/RNG と ImageCls preload を保ちつつ、重み世代をまたぐ途中 episode を除く |
| EvalSessionEnv と session event 1 回 | keep | 完了 Step 直後の scalar capture と metrics gate/EMA の多重前進を同時に解く |
| 共通 EpisodeReturnAccumulator | keep | shared / per-lane reward解釈をRunnerとdecoratorで重複させない |
| 汎用 `std.` | keep | P3 で設定だけで分散を観測でき、lane/session で同じ語彙になる |
| Atari eval1 / eval2 / eval_panel の `episodic_life=false` | 採用（2026-09-03） | 評価単位と frame stack を実ゲーム境界へ揃える。中核 mechanism とは独立 |
| 採用権を持たない group の freeze | defer | 性能問題が未実測 |
| P3 runtime 値・恒久 profile | defer | 測定前に固定しない |
| Atari 既定 `game_score_std` metric | cut | ID 衝突と恒久 metric 追加を避け、P3 tag だけで使う |
| ImageCls indexed `GetScalar` / Reset preload 特例 | cut | SHARED capture と fresh state 再利用で不要 |
| 任意 group map | cut | 現要件は PER_LANE/SHARED の 2 形で足りる |
| `se.` prefix | cut | `std./sqrt(N)` で運用可能 |
| metrics gate 改変 | cut | 共通 gate 契約へ波及させない |

## 11. 却下した旧解決候補（経緯）

旧候補 A/B の停止条件だけの変更は F1 により N 本平均にならない。走行中の全episodeから終了後に最初のN完了を選ぶ終了順採用は、長いepisodeを打ち切って短いepisodeへ偏るため却下する。一方、開始時の動的グラントは次episodeの結果を見る前に採用を決め、グラントした固定N本をすべて待つため、このbiasを持たない。

groupごとの固定quotaも不偏で、laneごとの採用本数がconfigだけで確定する固定デザインになる。Stable-Baselines3のvector laneごとの固定episode targetはこの形に近い（[evaluation.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py)、[DummyVecEnv](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/dummy_vec_env.py)）。ただしepisode長が不均一なほど最も遅いgroupのquota合計へ律速されるため、P2はR3の効率動機に合うR4の開始時動的グラントを採用する。採用権を持たないgroupもauto-reset後に通常stepを続けて結果だけを無視する。RLlibの余剰actorをidleにする方式は通常1 env / EnvRunnerのactor単位であり、vector env内のmasked Step契約ではない（[RLlib advanced evaluation](https://docs.ray.io/en/latest/rllib/rllib-advanced-api.html)、[SingleAgentEnvRunner](https://github.com/ray-project/ray/blob/master/rllib/env/single_agent_env_runner.py)）。TorchRLのpartial step/resetは専用maskを公開契約に含めているため（[TorchRL vectorized environments](https://docs.pytorch.org/rl/main/reference/envs_vectorized.html)）、同等契約を持たないP2では導入しない。

Runner集中集約は`$env` metricsの全面移行を招くため、captureと集約はBatchEnv decoratorへ置く。全session無条件Reset案はauto-reset済みstateを二重Resetし、N=1のRNG系列とImageCls preloadを壊すためfresh判定へ置き換えた。ImageClsのlane別採用案はbatch-level accuracyと一致しないためSHARED groupへ置き換えた。

## 12. 上書き・文書同期

- PRD 034 D5 の「eval は lane 0 だけを代表 episode とする」を、ImageCls eval の `episode_scope=SHARED`、全 lane 同時終端、`n_episode_end=1` で supersede する。PRD 034 自体は履歴として保持する。
- 本PRD、ADR 0034、CONTEXT.mdは受入済みの目標契約へ先行して同期する。現行コードを説明する`docs/design`はP2のcode / config変更と同じ変更で同期する。
- 旧 PRD の未調査事項 4 件は F4/F7/F8 と §4.7 で解決済み。

## 13. Atari プロトコルとの関係

- Classic / 100k は train で `episodic_life=true` を使い続けてよい。eval だけ false にする。v5 はもともと false。
- lane 独立性は Classic の noop、v5 の sticky action で確保される。旧 `@v5_noop0` かつ Greedy ε=0 の Run は同一軌道複製なので N 本平均の比較対象にしない。
- Nature 系 30 本、BTR 系 100 本は P3 の候補であって出荷既定値ではない。本数は PRD060、eval policy/ε は PRD061 の範囲。
