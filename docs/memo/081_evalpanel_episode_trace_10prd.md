# EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る PRD

> 起点: 2026-09-23、kung_fu_master 100M Run の EvalPanel を録画と突き合わせようとしたが、EvalPanel のゲームは metrics / trace に 1 本も残っておらず、スコア分布の断絶（74,900 → 84,200）を越えたゲームがあったかを後から確かめられなかったこと。
> 改訂: 2026-09-24 グリル（`/grill-with-docs`、9 問）で §1.4 の前提修正（EvalPanel の trace `step` は学習 `exp_step` にならない）と裁定。2026-09-26 再グリル: 起動時宣言（旧 D2。`CreateEvalRunner` 撤去）と定義レコードの `env_name` 欄（旧 D5）を取り下げ、`CreateEvalRunner` を残す原案 A1 の形へ戻した。突き合わせは序数の手順から trace 行の `timestamp` へ変更（ADR 0037 の行契約を改訂）。ログ prefix は `env.[<Env name>].[<lane>]:` 記法に追従。決定の記録は ADR 0047。
> 関連: [ADR 0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md)（本 PRD で新設）、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)（trace チャネル。`episode_id` / `model_version` 欄は需要が出た時点で同じゲートで足す）、[069](done/069_metrics_trace_channel_10prd.md)、[ADR 0029](../adr/0029-analysis-metadata-emitted-by-runner.md)（解析メタデータは Runner が出す）、[932](932_episode_forensics_10prd.md)（Run-scoped `EpisodeId`）、[912](912_background_eval_snapshot_ordering_10prd.md)、[220 §4.7](../design/220_atari_env.jp.md)（AtariEnv のゲーム完了ログ）、[140](../design/140_observability.jp.md)、[160](../design/160_applications_and_tools.jp.md)。
> 担当: 実装（コード・テスト・設計文書・Atari.txt の同期）は実装枠（Codex）。コミットは人間。

## 1. Context / Problem Statement

### 1.1 EvalPanel は記録を持たない

EvalPanel はデモ用のパネルで、`RunnerFrame::Initialize` が `RunManager::CreateEvalRunner("EvalPanel", "eval_panel")` で専用の EvalRunner（1 lane）を RunManager 構築後に作って回す（[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp) の `Initialize`、[trainer.cpp](../../core/anet-core/src/trainer.cpp) の `CreateEvalRunner`）。`eval_panel` は `run.eval_schedule` を持たない definition-only の eval 定義で、[Atari.txt](../../apps/runner/config/Atari.txt) は「eval_panel は metrics を 1 つも出さない（51/52/53 はいずれも別の eval 定義に紐づく）ので、記録側の値には混ざらない」と明記している。

デモ用という位置付けは変えない。一方で、録画した EvalPanel のあるゲームで何が起きたかを後から調べたい。いま残るのは AtariEnv のゲーム完了ログ（220 §4.7）だけである。

```
<時刻>: [V] env.[EvalPanel].[0]: Game over. game_score=<点> game_len=<step> game_frames=<frame>
```

時刻は分かるが、ログに出した値以外の詳細（`hns57` など trace が取れる任意の scalar key、どの方策スナップショットが遊んだか）は残らない。

### 1.2 設定だけでは trace が付かない

`metrics.trace.@atari.[<tag>] = $eval.[eval_panel] @episode_end $env ...` と書いても何も出ない。理由は 3 つある。

1. **結び付けは起動時だけ。** `RunManager` は構築時に `run.eval_schedule.[tag]` ごとに EvalRunner を作り、そのあとで `ObserverFactory` が組んだ scalar / trace observer を「`$eval.[tag]` → 作成済み Runner」の対応で attach する（`trainer.cpp` の `resolve_runner`）。
2. **definition-only の tag は捨てられる。** 対応する Runner が無く tag が definition-only（dormant）なら、`Skipping metrics for unscheduled eval tag` の WARN を tag ごとに 1 回出して observer を捨てる。`eval_panel` はこれに当たる。
3. **後から作る Runner には何も付かない。** `CreateEvalRunner` は env と EvalRunner を作って登録するだけで、observer を attach しない。

加えて、trace の定義（`metrics.trace.defs`）は起動時に 1 レコードだけ出し、dormant tag の定義は除く（`trainer.cpp` の `complete_metric_definition`）。[inspect_run.py](../../viewers/metrics-tools/inspect_run.py) は `metrics.trace.defs` を 1 レコードとして読む（master は最初の 1 件で読み終え、cache は序数最大の 1 件）。後から結び付く trace の定義をどう載せるかを決める必要がある（§3.3）。

### 1.3 trace 行は時刻を持たない

trace 行は `tag` / `step` / `lane` と `data` だけで、時刻を持たない（ADR 0037 / PRD 069 D6「scalar と同じ最小形」）。ログ行は時刻を持つが通し番号を持たない。序数（CONTEXT.md「序数」）で k 番目同士を対応させる手順は組めるが、録画の時刻から該当ゲームを引くたびにログと trace の両方を数えることになる。

一方、`type:"json"` のレコードは既に `timestamp`（`MetricsLogger::GetCurrentTimeStr()`、`%Y-%m-%dT%H:%M:%S`、ローカル時刻、秒精度）を持ち（[metrics_logger.cpp:645](../../core/anet-core/src/metrics_logger.cpp:645)）、Metrics Viewer の cache は `json_lines.timestamp` 列を持ち、ingest は `timestamp` を汎用に拾う（[MetricsIngestor.java:329](../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java:329)、同 351）。trace 行に同じ欄を足せば、trace 1 行で「いつ・何が」が閉じる。

### 1.4 EvalPanel の trace `step` は定義レコードと食い違う（原案の前提の修正）

原案は「EvalPanel はモデル同期のたびに学習側のカウンタをコピーするので、trace 行の `step` は方策スナップショットの学習 `exp_step` になる」と書いた。現行コードではそうならない。

- EvalPanel の timer 経路は `DoUpdateFrame` → `DoStep()` → `DoStep(-1, step_counts_)` で、`EpisodeEndEvent.counts` は **EvalPanel runner 自身の `step_counts_`**（[trainer.cpp:146](../../core/anet-core/src/trainer.cpp:146)、[同 300-313](../../core/anet-core/src/trainer.cpp:300)）。手動行動の `DoStep(action)` も同じ。
- `EvalRunner::Sync(source_counts)` は `source_counts_` を更新するだけで、それは `actor_->MakeAction(source_counts_, state_)` にしか渡らない（[同 219-224](../../core/anet-core/src/trainer.cpp:219)、CONTEXT.md「学習側 counts」）。
- 一方、定義レコードの `runner`（座標系所有者）は `OwningRunner(EVAL, EPISODE_END)` = `"train"` 固定である（[observers.cpp:1288](../../core/anet-core/src/observers.cpp:1288)）。

つまり結び付けだけを実装すると、trace の `step` は EvalPanel 自身の遷移数（EvalPanel 座標系）になり、定義レコードは train 座標系だと主張する。ADR 0029（解析側は定義を正本として読む）と CONTEXT.md「step座標系」（同じ eval tag でも `@episode_end` は train 側のカウンタに載る）の両方に反する。

configured eval は `RunSession(event_counts)` の冒頭で `Sync(event_counts)` してから同じ `event_counts` を `EpisodeEndEvent` / `SessionEndEvent` に載せているので、configured eval では `source_counts_` と event の counts は常に同値である。

### 1.5 その他の確定事実（2026-09-26）

- `CreateEvalRunner` の非テスト呼び出しは [RunnerFrame.cpp:471](../../apps/runner/src/RunnerFrame.cpp:471) の 1 箇所。online / batchrun とも常に生成する（batchrun は `auto_start=false` で止めているだけ）。
- `eval_runners` は name キー（configured は tag、`CreateEvalRunner` は name）。`resolve_runner` は eval_name（tag）で検索する。`RunnerScopedEpisodeEndObserver` は `event.runner == target_runner_` のポインタ一致で振り分ける（[rl.cpp:890](../../core/anet-core/src/rl.cpp:890)）。
- trace observer は `ParsedEpisodeEndObserver{scope, eval_name, obs}` として `episode_end_observers_` に入る。eval scope の `@episode_end` scalar は ADR 0037 で fail-fast なので、**eval scope の episode_end observer は trace observer に限られる**。
- trace 定義レコードは 9 欄（`step_axis / runner / scope / eval_name / eval_episodes / num_envs / event / target / keys`。[observers.cpp:1321](../../core/anet-core/src/observers.cpp:1321)）。`eval_episodes` / `num_envs` は `std::optional` で `null` を書ける。
- `inspect_run.py trace-csv` の固定列は `row_no, run, tag, step, lane`（inspect_run.py:53）。`_trace_fields` は `tag / step / lane / data` を検証して返す（同 1005-1020）。
- 手動行動の経路は RunnerFrame がハンドラを繋ぐ `EvalPanel::DoStep(action)` の 1 本（RunnerFrame.cpp:477-480 → EvalPanel.cpp:160-167）で、今は何もログしない（pause / resume は `LOG::info` で時刻付きに残る）。
- ゲーム完了ログ（`RecordGameCompletion`、`Game over.` / `Game truncated by max_episode_frames.`、prefix は `env.[<Env name>].[<lane>]:`）と PRD 082 の `ram_metric.[n]` は未コミット差分にある。Atari.txt の trace 4 行へのキー追加はまだである。

## 2. ゴール / 非ゴール

**ゴール:**

- **G1**: `CreateEvalRunner` で後から作る鏡写しインスタンス（§3.1）に、その config tag（`$eval.[tag]`）に結び付いた trace を付ける。EvalPanel はその最初の利用者であり、EvalPanel 専用の経路は作らない。
- **G2**: EvalPanel のゲーム 1 回が trace 1 行になり、その行に時刻（`timestamp`）と詳細が同居する。録画との突き合わせは時刻で行い、ログを数えない。
- **G3**: 解析側が `metrics.trace.defs` から鏡写しの trace 定義を読め、定義が言う座標系（`runner`）と行の `step` が一致する（ADR 0029）。
- **G4**: 学習中 eval の系列（`51_eval1` / `52_eval2` / `53_evalg`）と混ざらない。

**非ゴール:**

- **NG1**: EvalPanel の scalar metrics。人が開始・停止し、QValuePanel から手で行動も打てるので、系列として読むと eval の測定値と誤読される。dormant tag に結び付いた scalar は現行どおり捨てる。
- **NG2**: ゲーム固有の進行情報の新設。PRD 082 の `ram_metric.[n]` が Atari.txt の trace 行に入ったら、本 PRD の行もキー列を揃えるだけ（§3.7）。
- **NG3**: 932 の `EpisodeId` 全体（ReplayBuffer までの伝播）と trace 行の `episode_id` 欄（ADR 0037 のゲート）。
- **NG4**: Metrics Viewer 側の trace 表示。
- **NG5**: 突き合わせのための明示キー（`game_index` 等）と序数結合の手順の文書化。時刻で足りる。
- **NG6**: ゲーム開始時の方策スナップショット（`model_version`。912 と ADR 0037 のゲート）。
- **NG7**: 手動行動の trace 欄。ログ 1 行で足りる（§3.6）。
- **NG8**: RunManager の生成 API の変更（鏡写しの起動時宣言、`CreateEvalRunner` の撤去）。§8 参照。

## 3. 確定契約

### 3.1 用語（CONTEXT.md。本 PRD と同じ変更で改訂）

- **鏡写しインスタンス（mirror eval instance）**（新語）: configured eval tag の内容（run_mode / env overlay / actor）を参照して、アプリケーションが名前を付けて `CreateEvalRunner(name, tag)` で作る 1 lane の EvalRunner インスタンス。タグ自身のインスタンス（configured eval）とは別で、eval schedule に駆動されず評価セッションを持たない。人が開始・停止・手動操作するので scalar の購読先にはならず、タグが dormant のときだけ最初に作られた鏡写しが trace の購読先になる。EvalPanel が唯一の利用者。_Avoid_: 動的 Eval、EvalPanel runner（アプリ側の名前）、ad-hoc eval / on-demand eval。
- **dormant** に追記: dormant タグを参照する scalar は WARN で skip。trace は鏡写しインスタンスがあればそこへ結び付き、無ければ同じく WARN で skip。
- **学習側 counts** に追記: EvalRunner の `@episode_end` / `@session_end` イベントは常にこの値に載る（configured eval はセッション開始時、鏡写しは直近の Sync 時点）。

### 3.2 D2 結び付け: dormant タグの trace observer を保持し、`CreateEvalRunner` で付ける

- 構築時、`resolve_runner` が dormant tag に当たった observer のうち、**trace observer は捨てずに RunManager が tag 別に保持する**（`pending_trace_observers_: tag → observer 列`）。scalar（train / learn / session_end、および train scope の episode_end）は現行どおり。eval scope の episode_end observer は trace に限られる（§1.5）ので、「eval scope かつ dormant の episode_end observer を保持する」で足り、ObserverFactory は変えない。
- WARN `Skipping metrics for unscheduled eval tag`（tag ごと 1 回）は **scalar が dormant tag を参照したときだけ**出す。trace だけが参照する tag（`eval_panel`）では出ない。
- `CreateEvalRunner(name, config_tag)` は現行どおり Runner を作った後、保持分に `config_tag` があればその Runner へ `notifier_->AttachScoped(obs, runner)` で付け、**保持から外す**。これで同じ tag の 2 つ目の Runner には付かず、scheduled tag の鏡写し（`app.eval_panel.eval_config_tag = eval`）には最初から何も付かない（trace は configured インスタンスに付いている）。1 tag の trace 購読先は 1 インスタンスに閉じる。
- 保持分が無く、かつその tag に trace 定義があるとき（scheduled tag の鏡写し、2 つ目以降の鏡写し）は INFO を 1 行: `Trace for eval tag is already bound. tag='<tag>' runner='<name>'.` fail-fast にはしない。scheduled tag の鏡写しは現行と同じ結果（パネルに metrics は付かない）で、正当な設定のまま。
- attach は現行の `RunnerScopedEpisodeEndObserver`（runner ポインタ一致）でよい。trace observer のインスタンスは 1 tag に 1 つで、2 つの Runner へ scoped attach しない。
- `CreateEvalRunner` のシグネチャ、名前 registry、seed 領域（env `eval_panel/<tag>`、actor `actor/<name>`）、`LogEnvConfig` は変えない。

### 3.3 D3 定義レコード: 保持した trace の定義も起動時に載せる

- `metrics.trace.defs` は従来どおり構築時に 1 レコード。`complete_metric_definition` は **dormant tag を参照する trace 定義を除外せず**、`eval_episodes=null`（評価セッションではない）、`num_envs=null`（構築時点で Runner が無い）で載せる。`runner` は `OwningRunner` のとおり `train`（D1 で正しい）。scalar 定義の dormant 除外は現行どおり。
- 誰も `CreateEvalRunner` を呼ばなかった tag の定義は、行が 1 本も出ないまま残る（例: `@btreval` で `eval_target` が dormant のときの `51_eval1/episode`）。定義は「構築した observer」を表し、行の有無は保証しない。inspect_run は行の無い定義を空として扱えるので読み手は変えない。
- 欄は 9 欄のまま。`metrics.scalar.defs` は変えない。

### 3.4 D1 step 座標: EvalRunner のイベント counts は学習側 counts

- `EvalRunner::DoStep(int64_t action)` は `DoStep(action, source_counts_)` へ（現行は `step_counts_`）。`DoStep()` → `DoStep(-1)` は不変。`DoStep(const StepCounts&)` と `RunSession(event_counts)` は不変（`RunSession` は `Sync(event_counts)` 直後なので同値）。
- これで EvalRunner の `@episode_end` / `@session_end` は常に学習側 counts（train 座標系）に載り、定義レコードの `runner=train` がそのまま正しい。CONTEXT.md「step座標系」の「同じ eval tag でも `@episode_end` は train 側のカウンタに載る」が鏡写しにも成立する。
- View 用の `TrainEvent.counts` と `GetCounts()` は EvalRunner 自身の `step_counts_` のまま（EvalPanel の表示・`SyncAfterStep` は変えない）。
- 帰結（原案 Q4）: 鏡写しの trace `step` は**終局直前の Sync 時点の学習側 `exp_step`**。1 ゲームの途中で Sync が起きても終局時の値になる。学習を一時停止しているあいだは全ゲームが同じ step になる（identity は序数、時刻は `timestamp`）。最初の Sync 前に終わったゲームは `StepCounts` の既定値 0。開始時のスナップショットが要るなら `model_version` 欄の話（NG6）。

### 3.5 D5 trace 行に `timestamp` を持たせる（ADR 0037 の行契約改訂）

```
{"type":"trace","tag":"54_evalpanel/episode","step":12345678,"lane":0,"timestamp":"2026-09-23T21:04:17","data":{"game_score":84200,...}}
```

- `MetricsLogger::LogTrace` が全 trace 行（`42_env` / `51`〜`53` / `54` すべて）に `timestamp` を足す。値は `GetCurrentTimeStr()`（`type:"json"` レコードと同じ書式・同じローカル時刻・秒精度）。書式を新設しない。
- 固定属性は `tag / step / lane / timestamp`、個別値は `data` 下。ADR 0037 の読み手 3 制約（`type` は文字列、`step` は整数、top-level に数値 `value` を置かない）は変えない。`timestamp` は identity ではない（identity は序数、CONTEXT.md）。同じ秒に複数行が並ぶのは正常（configured eval の複数 lane）。
- 読み手: Metrics Viewer は無変更（ingest が `timestamp` を `json_lines.timestamp` へ汎用に格納する）。`inspect_run.py trace-csv` は固定列を `row_no, run, tag, step, lane, timestamp` にし、`_trace_fields` は `timestamp` を文字列または欠落（旧 Run は空欄）として返す。
- 同 seed 比較（ADR 0037 の受入方式）では trace 行の `timestamp` を除いて比べる。`type:"json"` レコードの `timestamp` と同じ扱い。
- 突き合わせの手順: 録画の時刻 → `trace-csv --tag "54_evalpanel/*"` で `timestamp` が一致する行。ログ側と対応させたいときも時刻で引く。秒精度で足りる（EvalPanel は 1 lane・15 fps で、1 秒に 2 ゲームは終わらない）。
- 精度を上げる（ミリ秒）案は採らない（§8）。

### 3.6 D6 手動操作のログ

`EvalPanel::DoStep(int64_t action)` で `LOG::info() << "EvalPanel: manual action=" << action;` を 1 行出す。ゲーム完了行の間に手動行動の時刻が残り、trace 行の `timestamp` と突き合わせられる。trace 欄は足さない（NG7）。QValuePanel 側の `runner_->DoStep(action)` フォールバックはハンドラ未設定時だけの経路で RunnerFrame は常に設定するので触らない。

### 3.7 D7 設定（Atari.txt）

```
metrics.trace.@atari.[54_evalpanel/episode] = $eval.[eval_panel] @episode_end $env game_score game_len game_frames hns57
```

- 番号は `42_env`（train）/ `51`〜`53`（eval）の次で `54`。既定 ON（Atari.txt は online 観戦構成が既定。batchrun で EvalPanel を動かさなければ行は出ない）。
- キー列は `52_eval2/episode` 行と同じにする。PRD 082 で `ram_metric.[1] ram_metric.[2] ram_metric.[3]` が 51〜53 行に入ったら本行も揃える（同じ Env なので同じキーが取れる）。
- Atari.txt の「デモ専用パネルなので … eval_panel は metrics を 1 つも出さない（51/52/53 はいずれも別の eval 定義に紐づく）ので、記録側の値には混ざらない」を「scalar は出さず trace（54）だけ出す。51/52/53 は別の eval 定義に紐づくので混ざらない」へ直す。
- 他 env（DropMerge / GridMaze / LunarLander / ImageCls）は trace 宣言自体が無いので足さない（069 と同じ方針）。common.txt は変更なし。

### 3.8 帰結として決めたこと

- batchrun でも鏡写しは生成される（現行どおり）。`auto_start=false` なら Step が進まないので行は出ないが、定義レコードには載る（§3.3）。
- 鏡写しの trace は `episodic_life=false`（既定の `run.eval.[eval_panel].env.episodic_life = false`）でゲーム完了 = episode 終端になる。`episodic_life=true` の tag を鏡写しにすると、ライフ喪失の終端でも行が出て `game_score` は `null`。これは train の `42_env` と同じ挙動で、本 PRD では変えない。

## 4. 実装ノート

- [trainer.hpp](../../core/anet-core/include/anet/trainer.hpp) / [trainer.cpp](../../core/anet-core/src/trainer.cpp): `pending_trace_observers_`、`resolve_runner` の dormant 分岐（trace は保持、scalar は WARN）、`complete_metric_definition` の trace 側 dormant 許容（`null` 2 欄）、`CreateEvalRunner` 末尾の attach と INFO、`EvalRunner::DoStep(int64_t)` の counts。
- [metrics_logger.hpp](../../core/anet-core/include/anet/metrics_logger.hpp): `LogTrace` に `{"timestamp", GetCurrentTimeStr()}`。
- [inspect_run.py](../../viewers/metrics-tools/inspect_run.py): `TRACE_FIXED_COLUMNS` に `timestamp`、`_trace_fields` / `iter_trace_rows` / CSV 行の組み立てに 1 欄。
- [EvalPanel.cpp](../../apps/runner/src/EvalPanel.cpp): ログ 1 行。[Atari.txt](../../apps/runner/config/Atari.txt): 54 行とコメント。
- [trainer_test.cpp](../../core/anet-core/src/trainer_test.cpp)、`metrics_logger_test.cpp`、`inspect_run_test.py`: §5。

## 5. テストの方針と受入条件

**既存テストの更新**: `RunManager writes attached scalar and trace definitions separately` は、dormant tag `sleep` の trace 定義が `eval_episodes=null` / `num_envs=null` で載ることに変える（scalar 側の `sleep` 除外は現行どおり）。`CreateEvalRunner` を使う既存テスト（name registry 系）は変えない。

**新規（先行例は `trainer_test.cpp` の `$eval.[active]` / `$eval.[sleep]` と `episode_end_test.cpp`）**:

1. dormant tag の trace を参照し、その tag で `CreateEvalRunner` すると、その Runner のゲーム完了ごとに trace 行が 1 行出る。`step` は直前に `Sync` した学習側 counts の `exp_step`（EvalRunner 自身の counts ではない）。
2. 同じ tag の scalar は行が出ず、WARN は scalar 参照があるときだけ tag ごと 1 回。trace だけの参照では WARN が出ない。
3. 定義レコードに当該 tag が `runner=train / scope=eval / eval_name=<tag> / eval_episodes=null / num_envs=null` で 1 回だけ載る。
4. scheduled tag で `CreateEvalRunner` しても trace 行は増えず（configured 側の行は変わらない）、INFO が 1 行出る。
5. 同じ dormant tag で `CreateEvalRunner` を 2 回すると、最初の Runner だけが行を出し、2 つ目は INFO。
6. `EvalRunner` 単体: `Sync(source)` 後の `DoStep()` / `DoStep(action)` で届く `EpisodeEndEvent.counts` が `source` と一致し、`GetCounts()` は自身の counts のまま（既存 `EvalRunner uses synchronized learning counts while keeping evaluation counts` に追記）。
7. `LogTrace` の行に `timestamp` が `%Y-%m-%dT%H:%M:%S` 形式で入る（`metrics_logger_test.cpp`）。
8. `trace-csv` が `timestamp` 列を出し、欠落した旧 Run の行では空欄になる（`inspect_run_test.py`）。

**受入**:

- 上記テストが緑。既存 trace（`42_env` / `51`〜`53`）の行は `timestamp` を除いて変わらない。ADR 0037 の受入方式（編集前 baseline との同 seed 比較）に従い、比較は `timestamp` を除く。
- Atari online 構成で EvalPanel を数ゲーム回し、`trace-csv --tag "54_evalpanel/*"` の行数 = ログの `env.[EvalPanel].[0]: Game over.` 行数、各行の `timestamp` が対応するログ行の時刻と同じ秒（境界で ±1 秒）、`game_score` が一致、`step` が直近 Sync 時点の学習 `exp_step` であること。
- `app.eval_panel.eval_config_tag = eval` の構成で、52_eval2 の行が編集前 baseline と（`timestamp` を除いて）一致し、INFO が 1 行出ること。

## 6. 文書同期

### 6.1 今回の文書改訂（本 PRD と同じ変更）

| 文書 | 変更 |
|---|---|
| 本 PRD | 再グリル反映（§1.3、§2 NG5/NG8、§3.2〜3.5、§5、§8） |
| `CONTEXT.md` | 「鏡写しインスタンス」の定義を `CreateEvalRunner` で作る形へ。dormant / 学習側 counts の追記は維持 |
| `docs/adr/0047-…md` | 決定の記録（改題・書き直し） |
| `docs/adr/0037-…md` | Consequences に `timestamp` 欄と dormant trace の保持を追記（ADR 0047 参照） |

### 6.2 コード実装時に同期する対象（実装枠が同じ変更で更新）

| 対象 | 変更 |
|---|---|
| [140 §6.x](../design/140_observability.jp.md) | 行契約に `timestamp`（例の行、「`timestamp` は持たず」の削除）、dormant tag の trace 保持と `CreateEvalRunner` での結び付け、定義レコードの dormant trace（`null` 2 欄）、EvalRunner の event counts = 学習側 counts |
| [160](../design/160_applications_and_tools.jp.md) | 213 行の EvalPanel 説明に trace（`54_evalpanel/episode`）と手動行動のログ |
| [220 §4.7](../design/220_atari_env.jp.md) | 「metrics / trace を持たない EvalPanel のゲームも、ここからミリ秒の時刻付きで追える」→ trace は `54_evalpanel/episode`（時刻は行の `timestamp`）、ログは ms 精度の補助 |
| [030 §6.9](../design/030_user_guide_analysis.jp.md) | 出力列に `timestamp`、録画の時刻から行を引く例 |
| [Atari.txt](../../apps/runner/config/Atari.txt) | §3.7 |

100（`CreateEvalRunner` の記述）と AGENTS.md に変更は無い。

## 7. スコープ外（再掲）

- NG1〜NG8。特に `game_index` 等の明示キー、`episode_id` / `model_version` 欄、手動行動の trace 欄、Viewer 表示、生成 API の変更。
- DropMerge 等、他 env への trace 宣言の追加。
- trace 行 `timestamp` のミリ秒化・UTC 化（`type:"json"` と揃えたまま）。

## 8. 却下した案（経緯）

| 案 | 却下理由 |
|---|---|
| 起動時宣言（旧 D2）: RunManager ctor に `{name, config_tag}` を渡して鏡写しを構築時に生成し、`CreateEvalRunner` を撤去 | 「定義 = 実購読先・1 回」を厳密に守るための案だったが、ctor API 変更・RunnerApp/RunnerFrame 変更・既存テスト 6 件の移行を伴い、10 行程度の結び付けに対して大きすぎる。定義に dormant trace を載せる（§3.3）だけで目的は満たせる（2026-09-26 取り下げ） |
| 定義レコードの `env_name` 欄（旧 D5） | 序数結合でログの lane 名と対応させるために要った。時刻を trace 行に持てばログとの対応自体が不要（2026-09-26 取り下げ） |
| 序数で結合（旧 D3） | 手順としては成立するが、録画の時刻から引くたびにログと trace を数える。時刻を行に持てば直接引ける |
| eval 定義に `on_demand` 印を置いて載せる定義を選ぶ | 用途ラベル。アプリが作るかどうかは設定の関心ではない |
| `CreateEvalRunner` 時に定義レコードを追記し、読み手を複数レコードのマージへ変える | 読み手契約の変更。起動時に載せれば不要 |
| 鏡写しの `step` を自身の counts に載せ、定義の `runner` をインスタンス名にする | 定義の座標規則がインスタンス依存になり、CONTEXT.md「`@episode_end` は train 側」と矛盾。学習側 counts に載せれば既存規則のまま正しくなる |
| EvalPanel が counts を明示で渡す新オーバーロード | 結果は D1 と同じで API が増えるだけ |
| 突き合わせに env の `game_index` | 時刻で足りる。後付け可能（env キー 1 つ） |
| 突き合わせに `episode_id` 欄 | ADR 0037 のゲート。ログ行は env が出すので Runner の番号を知らない |
| `timestamp` をミリ秒・UTC にする | `type:"json"` レコードと書式が割れる。EvalPanel は 1 秒に 2 ゲーム終わらず、identity は序数が持つ |
| 全鏡写しにも trace を付け、行に Runner 名欄を足して区別 | scheduled tag の鏡写しでは eval の分布にパネルのゲームが混ざり G4 違反 |
| scalar も鏡写しに付ける | 人が開始・停止・手動操作するので統計の母集団ではない（NG1） |
| 同じ tag の 2 つ目の `CreateEvalRunner` を fail-fast | 保持分を手放す実装で自然に 1 つに閉じる。INFO で足りる |
| 手動行動を trace 欄（別 tag の `$runner forced_actions`）で残す | Runner にエピソード単位のカウンタと key が増え、行も 2 系列になる。ログ 1 行で足りる |
| ADR を 2 本（結び付け / EvalRunner の counts）に分ける | D1 は configured eval の挙動を変えず、鏡写しの step を定義と一致させるための帰結。一つの契約として 0047 に置く |
