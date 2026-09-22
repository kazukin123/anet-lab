# PRD 076: 終了時の評価セッション排水

> 起点: 2026-09-18。`run_20260917-225635_rr4_btrsn12_envs128_cap2m_50m` で、予算末尾に発火した
> ε=0 断面セッション（400 エピソード）が 70 エピソードまで進んだところでプロセス終了に巻き込まれ、
> 集計もログも出ずに消えた。判定に使う断面そのものを失い、復旧に評価専用 Run を 1 本要した。
> 関連: [PRD 026](frozen/026_metricslog_lifecycle_10prd.md)（`MetricsLogger` ライフサイクル。本 PRD は同 R2 を切り出して拡張する）、
> PRD 052 / [ADR 0027](../adr/0027-eval-definition-schedule-separation.md)（eval の定義とスケジュールの分離）、PRD 060（評価セッション）、
> [ADR 0045](../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)（本 PRD の判断理由）、用語「排水」は [CONTEXT.md](../../CONTEXT.md)、
> `core/anet-core/include/anet/rl.hpp`、`core/anet-core/include/anet/thread.hpp`、`core/anet-core/src/observers.cpp`、
> `core/anet-core/src/trainer.cpp`、`apps/runner/src/RunnerFrame.cpp`、`apps/runner/src/RunnerApp.cpp`
>
> 改訂: 2026-09-21 再グリル（第 1 回、Claude、`/grill-with-docs`）。起点 Run の死因を裏取りに合わせて書き換え、
> wrapper 転送・例外の帰結（D7）・実装順序を追加。`common.txt` の既定行は cut、受入に exit code 0 を追加。
>
> 改訂: 2026-09-21 再グリル（第 2 回、Claude、Atari 枠レビューの指摘 1 を受けて）。「待たずに捨てる」は
> `PinnedThreadPool` の join で成立しないことが判明し、`RunSession` の協調キャンセル（D9）、排水全体の上限時間（G4 / D10）、
> 人が閉じたときの 3 択ダイアログ（D8）へ改めた。非ゴールを採番し直し、旧 NG2（タイムアウト）は撤回した。
>
> 改訂: 2026-09-21 再グリル（第 3 回、Claude、Atari 枠レビューの残り 5 件）。キャンセル完了の有界仮定と NG6、
> キャンセルの json レコード、deadline 共有の条件、`stop_source_` の作り直し、999 との移行依存を反映。

## Context（背景・目的）

### 実害

背景実行の評価セッションが in-flight のまま学習予算に到達すると、**セッションは完了を待たれずに破棄される**。

実測（`run_20260917-225635`、`replay_capacity` 2,097,152 / `num_envs` 128 / RR4）:

| 事象 | 時刻 | 値 |
|---|---|---|
| `[greedy_dist]` セッション開始 | 06:18:25 | `learn_step=773,430` / `exp_step=49,699,712` |
| 学習予算到達・プロセス終了 | 06:21:42 | `exp_step=49,998,592` |
| 経過 | — | 3 分 17 秒（セッション所要は約 22 分） |

trace チャネルには**途中まで残る**。`53_evalg/episode` の step 別行数は完了セッションが 400 行ずつなのに対し、
`step=49,699,712` だけ **70 行**。一方で scalar（`53_evalg/10_game_score_mean` 等）はセッション終端で
初めて確定するため**一切出ない**。`session end` の実行ログも出ない。`stderr` に例外も FATAL も無い。

**プロセスは完走を待たずに落ちている。** 次 Run `run_20260918-062145_rr1_va_hard125_100m` の起動時刻は
`RunnerFrame::OnClose()`（06:21:42.442）の 3 秒後で、`~EpisodeEvalObserver()` の `eval_future_.wait()`
（残り約 19 分）には到達していない。`MetricsLogger::Reset()` の後に eval スレッドが次の trace 行を書こうとして
null の `Instance()` を参照した（PRD 026 が診断した経路）とみるのが最も整合的で、その場合 exit code は非 0 になり
bat の集計では失敗 Run に数えられる。実行ログに痕跡が残らないのは、file logger が `ShutdownRunLogging()` で
先に閉じているためである。

同時に in-flight だったのは 1 本ではない。`eval`（10 エピソード）も 06:21:03 に `session start` を出したまま
終わっていない。同じ条件は次 Run（RR1 / 100M）でも成立しており、末尾発火は構造的に取りこぼす。

失われるのは「余分な 1 点」ではない。**予算末尾の断面は腕の判定に使う点**であり、
本件では復旧のために評価専用 Run（`learner.enabled=false`）を 1 本 25 分かけて回し直した。

### 経路

```
train が exp_exit_step 到達 → wxEVT_TRAINER_EXIT を post
  RunnerFrame::OnClose()                          [apps/runner/src/RunnerFrame.cpp]
    ① wxGetApp().StopTraining()                   → RunnerThread::Stop()（train スレッドのみ）
    ② DetachTrainStatusObserver()
    ③ TrySaveAgent(agent_close.anet)
    ④ ShutdownRunLogging()                        ← eval 走行中に file logger を閉じる
    ⑤ eval_panel_->DoClose() / aui_mgr_.UnInit()
  RunnerApp::OnExit()                             [apps/runner/src/RunnerApp.cpp]
    trainer_thread_->Stop() / ShutdownRunLogging()
    MetricsLogger::Reset()                        ← eval 走行中に metrics を落とす
  プロセス終了（実際には Reset 後の eval スレッドが落として終わる。「実害」参照）
```

`EpisodeEvalObserver` は自前の `PinnedThreadPool(1)` を持つ（`observers.cpp`）。
`RunnerThread::Stop()` は train スレッドしか止めないため、**このプールは停止経路のどこからも触られない**。

`~EpisodeEvalObserver()` には `eval_future_.wait()` があるが、そこへ到達するのは
`MetricsLogger::Reset()` より後であり、待てたとしても結果を記録できない（起点 Run では到達前に落ちている）。

停止経路で確定している事実:

- `RunnerThread::Stop()`（実体は `ThreadBase::Stop()`、`thread.cpp`）は train スレッドを **join** する。
  `StopTraining()` の後に `OnLearn()` が走ることは無く、新しいセッションは投入されない。
  前景実行（`use_background=false`）のセッションは `OnLearn()` の中で同期実行されるため、この join の中で完走する。
- `MetricsLogger::Reset()` は **flush しない**（`instance_.reset()` だけ）。metrics の flush は
  `ShutdownRunLogging()` が呼ぶ `FlushRunOutputs()` だけが行う。排水を `ShutdownRunLogging()` より前に置かないと、
  完走しても scalar は書き出されない。
- `Notifier::AttachScoped()` は observer を `RunnerScoped*Observer` の wrapper で包んで attach する（`rl.hpp`）。
  wrapper は event だけを転送するので、基底に足した仮想関数は wrapper が転送しない限り実体へ届かない。
- `PinnedThreadPool::WorkerLoop`（`thread.cpp`）は `stop_flag_` を wait の述語とキューが空のときにしか見ず、
  `task()` はロック外で中断点なしに走る。`Stop()` は flag を立てて notify した後、全 worker を **join** する。
  `~PinnedThreadPool()` も `Stop()` を呼び、`workers_` は `std::thread[]` なので join せずに破棄はできない。
  `EnqueueFuture` は `std::packaged_task` の future を返すので future のデストラクタはブロックしない。
  **待たせているのは future ではなく pool の join**であり、`get()` を飛ばしても待ち時間は変わらない。
- `TrainRunner::Shutdown()`（`env_->Shutdown()` + `notifier_->Clear()`）は application のどこからも呼ばれていない。
  設計文書 100 §7.2 の記述と食い違うが、本 PRD の範囲外として記録に留める。

### 中断点は無いが、足せる

`EvalRunner::RunSession`（`trainer.cpp`）は

```cpp
while (!session_env_->GetSessionResult().has_value()) { DoStepInternal(-1, event_counts, false); ... }
```

で、**停止フラグを見る点が無い**。`GetSessionResult()` は全採用エピソード完了でのみ値を返すため、
「途中まででセッションを確定する」口も無い。

一方でこのループは**ステップ単位**で、1 反復は全 lane の 1 Step（Atari 16 lane でミリ秒級）である。
反復の頭でキャンセル要求を見るだけで、eval スレッドは 1 Step 以内に抜け、pool の join はその直後に戻る。
C++ にスレッドを外から安全に止める手段は無く、止められるのは「走っている本人が要求を見て自分で返る」協調キャンセルだけなので、
中断点は `RunSession` 側に足す。`PinnedThreadPool` の「投入済みタスクを捌いてから止まる」意味論はキャンセルとは独立で、
pool を直しても解決しない（NG4）。

「途中まででセッションを確定する」ことは引き続きしない（NG1）。キャンセルしたセッションは何も集計しない。

### 排水が生む新しい実害

排水を入れると、今度は **exe が数分から数十分落ちない**事態が起こり得る。起点 Run の断面は 22 分級、
[999_eval_schedule_first_fire](999_eval_schedule_first_fire_10prd.md) が実測した t=0 の貪欲方策の張り付きは 1 時間級である。
設定を間違えた Run を即座に落としたい場面や、online 構成で Eval 中と意識せずに × を押す場面で、待つしかない設計は
人を強制終了へ追い込む。強制終了は最後の flush 以降の metrics と exit code 0 を失うので、回避策として劣る。
したがって「抜ける手段」を本 PRD の要件に含める（G4 / D8）。

### PRD 026 との関係

PRD 026 R2 が、本 PRD が必要とする排水機構をすでに設計している（observer 基底への `Shutdown()` 追加、
`Notifier::Shutdown()`、`MetricsLogger::Reset()` の前で呼ぶ）。本 PRD はそれを土台に次を変える。

- **026 は `MetricsLogger::Reset()` の直前に置く**。それでクラッシュは防げるが、
  `RunnerFrame::OnClose` の `ShutdownRunLogging()` が先に走るため**結果は残らない**
  （file logger は閉じており、`Reset()` は flush しないので scalar も書き出されない）。位置をさらに前へ出す。
- **026 は常に待ち、抜ける手段が無い**。スロット単位の選択（D2）、排水全体の上限時間（D10）、
  人が閉じたときの選択（D8）を足す。

026 本体は R1（`MetricsLogger` の null-safe static API への全面移行）と抱き合わせで範囲が広い。
R1 は systemic なガードで別の関心なので、本 PRD は R2 系の**ライフサイクル整合だけ**を扱う。

## ゴール / 非ゴール

**ゴール**

- **G1**: 学習側の終了時に、background 実行中の評価セッションが完了まで待たれ、**結果が通常どおり記録される**。
- **G2**: 「完走を待つ / 終了時に即キャンセル」を、評価スロット単位で設定から選べる。
- **G3**: 排水は冪等で、通常終了・GUI からの明示クローズ・`OnExit` のいずれの経路でも同じ状態に落ちる。
- **G4**: 排水全体の所要時間に上限を置け、超過したら残りのセッションをキャンセルして flush 付きで正常終了（exit code 0）する。

**非ゴール**

- **NG1**: 部分集計。途中まで進んだセッションを確定して scalar にすることはしない。
  `eval_episodes` が Run 内で不揃いになって比較不能な点を新しく作るためである。キャンセルは何も集計しない。
- **NG2**: 発火の開始下限・終了マージン。初期重みの貪欲方策が上限まで張り付く問題
  （RR1 実測 2,940 秒 / スコア 0）は別の関心なので、`run.eval_schedule.[tag]` の別キーとして切り分ける
  （[999_eval_schedule_first_fire](999_eval_schedule_first_fire_10prd.md)）。ただし移行上の依存はある。
  999 が入るまでは、短い Run で t=0 に発火した `greedy_dist` セッションが終了時に in-flight になり、本 PRD の既定では
  上限時間まで待つ。配線 arm 側の手当ては「影響・移行」に書く。
- **NG3**: PRD 026 R1（null-safe static ログ API）。
- **NG4**: `PinnedThreadPool` の改造。キャンセルは `RunSession` の協調停止で作る（「中断点は無いが、足せる」参照）。
- **NG5**: 待機中にも中断できる進捗ダイアログ。人が閉じたときの選択は排水に入る前に 1 回聞く（D8）だけにし、
  待機中の UI は従来どおりブロックする。待機中に強制終了して記録を失った事例が出たら足す。
- **NG6**: Env の `Step()` が返らない故障。協調停止は本人が中断点へ戻る前提なので、WAIT でも CANCEL でも待ち続ける。
  未観測の故障であり、実例が出たら「第 2 期限後の flush 付き強制終了」を別 PRD で扱う。

## 確定事項

- **D1: 排水は無条件で実装する。** 終了時のセッションの結果は「完走して記録」「設定によるキャンセル」「上限時間によるキャンセル」の 3 つで、
  人が閉じたときは D8 の選択でキャンセルにもできる。いずれの場合も記録先を閉じる前に eval スレッドは止まっている。
- **D2: 設定キーは `run.eval_schedule.[tag].wait_on_exit`（bool、既定 `true`）。**
  `true` は完走を待つ（上限時間内）、`false` は終了時に即キャンセルする。
  粒度を評価スロット単位にするのは、1 本の Run の中で要求が分かれるため。
  ε=0.01 の時系列（10 エピソード = 数十秒）は待っても無コスト、ε=0 の断面（400 エピソード = 約 20 分）は
  判定点そのものなので待つ価値が高い。throughput 測定 Run は捨ててよい。
  既定を `true` にするのは、**判定点が黙って消える損失が、終了が遅い損失より大きい**ため。
  なお `false` を書く現用 Run は今のところ無い（throughput 測定は `run.@evaloff` で eval 自体を切る）。
  D9 のキャンセル経路をそのまま使うので追加コストはキー読み取りだけであり、スロット間のコスト差を理由に先行して持つ。
- **D3: 呼び出し位置は `RunnerFrame::OnClose()` の `TrySaveAgent()` の直後、`ShutdownRunLogging()` の前。**
  checkpoint を先に確定させてから排水に入る（排水中に異常が起きても checkpoint を失わない）。
  train スレッドは `StopTraining()` で join 済みなので排水中に network は変化せず、save と排水の順序で
  内容が変わらないのは clone 設定によらない。
  `RunnerApp::OnExit()` にも冪等に置き、GUI を経ない経路（main loop の fatal 後は `OnClose()` を通らない）を塞ぐ。
- **D4: `Shutdown()` は冪等。** デストラクタからも呼ぶ（二重停止で安全）。デストラクタ経由は安全網であり、
  未処理の in-flight があれば待たずにキャンセルする。明示 `Shutdown()` は worker 例外を呼び出し元へ再送出するが、
  デストラクタ安全網だけは例外を捕捉して FATAL を記録し、`noexcept` デストラクタからの `std::terminate` を避ける。
- **D5: 待機に入ることを実行ログへ出す。** 排水は最大で上限時間まで伸び、`OnClose` は UI スレッドなので
  ウィンドウが無応答になる。ログが無いとハングと区別できない。
  `RunnerApp` は排水開始時に `Draining background observers: mode=wait|cancel timeout=3600s` を 1 行出す（終了ごとに 1 回）。
  observer は既存の `eval.[{}]: waited for previous session elapsed={:.2f}s ...`（`observers.cpp`）と書式を揃え、
  同じ規則で、実際に待った場合だけ `draining in-flight session on exit` を出す（完了済みなら出さない）。
- **D6: `use_background = false` のスロットでは no-op。** 前景実行なので in-flight が存在しない
  （進行中の前景セッションは `StopTraining()` の join の中で完走している）。
- **D7: 排水中の例外は `Shutdown()` が再送出し、呼び出し側では捕捉しない。**
  `OnClose()` から抜けた例外は `RunnerApp::OnExceptionInMainLoop()` に届き、fatal として報告されて exit code は 1 になる
  （PRD 068 の契約）。このとき `OnClose()` の残り（`ShutdownRunLogging()`、EvalPanel の `DoClose()`、`aui_mgr_.UnInit()`）は
  実行されず、`Notifier::Shutdown()` のループも投げた observer で止まる。冪等なので `OnExit()` の 2 回目の
  `Shutdown()` が残りのスロットを排水し、投げた observer は future 消費済みのため pool の停止だけを行う。
  キャンセル後の `get()` も同じ契約で、キャンセル前に失敗していた例外はそこで再送出される。
- **D8: 人が閉じたとき（× / メニューの Exit）は、待ちが発生するなら 3 択を聞く。**
  判定は `wxCloseEvent::CanVeto()` で行う。× はデフォルトで veto 可能、予算到達の `wxEVT_TRAINER_EXIT` は `Close(true)` なので
  veto 不可となり区別できる。メニューの Exit は現在 `Close(true)` なので `Close()` に変えて × と同じ側に揃える。
  手動 close でも最初に train thread を Stop/join し、新しい評価が投入されない状態で `WillBlockOnDrain()` を照会する。
  pause は要求を出すだけで境界到達を待たないため、照会してから Stop すると、その間に評価が投入されてダイアログ無しで待つ競合が残るためである。
  `CanVeto()` かつ `WillBlockOnDrain()`（`wait_on_exit=true` のセッションが進行中）のとき、
  「**Wait and close**（完走次第自動で閉じる。上限時間内）/ **Cancel and close**（セッションを中断して即閉じる）/
  **Keep running**（閉じるのをやめる）」を出す。Keep running は `event.Veto()` し、同じ `RunnerThread` を close 前の pause 状態のまま再開する。
  予算到達では聞かず、設定と上限時間に従う。online / batchrun の構成は問わない（× を押したのは人である）。
  待機中の「応答なし」表示は仕様（NG5）。
- **D9: キャンセルは `RunSession` の協調停止で作る。**
  要求を出すのは observer なので `std::stop_source` は `EpisodeEvalObserver` が所有し（更新する側が所有する所有権ルール）、
  `RunSession(event_counts, stop_token)` にトークンを渡す。`RunSession` はループ条件で `stop_requested()` を見て、
  要求があれば `SessionEndEvent` を出さずに返る。scalar は出ず、完了済み採用エピソードの trace 行は既に書かれているので残る
  （個体記録であって集計ではない）。
  解析側が「trace 行あり・scalar 無し」だけではクラッシュと区別できないので、ログと json レコードで区別する。理由を知る observer が
  `eval.[tag]: cancelling in-flight session on exit reason=close|config|timeout` を、進捗を知る `RunSession` が
  `eval.[tag]: session cancelled learn_step= exp_step= elapsed= completed=n` を出す。
  加えて `RunSession` はキャンセルの事実を Metrics マスタにも残す。既存の json チャネルへ
  `MetricsLogger::Instance()->Log("eval.[<tag>].session_cancelled", {learn_step, exp_step, elapsed_sec, completed})` を 1 行出す
  （`{type:"json", tag, data}` の行。scalar / trace の契約と定義レコードは触らない。`inspect_run.py` はこの行を `json_lines` に保持し、表示は別件）。
  「trace 行数 < `eval_episodes`」はキャンセルでも起点のクラッシュでも同じ形になるので、解析ではこのレコードか `session cancelled` 行と対で判定する。
  キャンセルの完了は 1 Step の所要で有界と仮定する。`Step()` が返らない故障には安全な強制手段が無く（detach は解放後アクセス、
  プロセス強制終了は flush の意味論が別）、WAIT の timeout 後も同じ `get()` で待つので、両経路に共通の前提である（NG6）。
  `stop_source_` はセッション投入のたびに作り直し、`Shutdown()` は終了時専用とする（Run 途中で呼ぶ用途は想定しない）。
- **D10: 排水全体の上限は `app.drain_timeout_sec`（正の整数、既定 3600）。**
  無人の batchrun で唯一効く自動弁であり、設定し忘れても exe が有界に終わるよう既定で効かせる。
  3600 は ε=0 断面（22 分級）を失わず、t=0 flare（1 時間超）や異常に長いセッションだけを切る。
  無制限モードは設けない（AGENTS.md は `0=無制限` のような番兵値を禁じる。長く待ちたければ大きな値を書く）。
  0 以下・非整数は `ANET_SYSTEM_ERROR`。`common.txt` に既定行は置かない（`?=` は既存の実効値を保つ葉だけに置く運用）。
  `RunnerApp` が排水開始時に deadline（`steady_clock`）を 1 つ計算し、全スロットが同じ deadline を共有するので合計が有界になる。
  `Notifier` の走査順は `unordered_map` 由来で不定だが、各セッションは自分の pool で並行に進むので、どの順で待っても
  全員が deadline まで進める。順序で損をするスロットは無い。
  スロット単位の上限は置かない（合計が有界にならず、スケジュール定義に終了ポリシーが混ざる）。core は設定を読まず、deadline を API 引数で受ける。

## 実装契約

### `core/anet-core/include/anet/rl.hpp`

observer 基底 4 種（`TrainObserver` / `LearnObserver` / `EpisodeEndObserver` / `SessionEndObserver`）へ既定 no-op の
teardown フック 2 本を追加する。背景スレッドを持つ observer 一般の口として置く（現在それを持つのは `EpisodeEvalObserver` だけ）。

```cpp
enum class ShutdownMode { WAIT, CANCEL };

virtual void Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode) { }
virtual bool WillBlockOnShutdown() const { return false; }   // Shutdown(WAIT) が待ちを伴うなら true
```

`RunnerScopedTrainObserver` / `RunnerScopedLearnObserver` / `RunnerScopedEpisodeEndObserver` /
`RunnerScopedSessionEndObserver` は両方を override して `real_observer_` へ転送する。
`AttachScoped()` で attach した observer には wrapper 経由でしか到達しないため、この転送が無いと
`Notifier::Shutdown()` は `EpisodeEvalObserver` に届かず no-op になる。

`Notifier` へ一括呼び出しを追加する。冪等であることを契約に含める。

```cpp
void Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode);   // 4 列すべてを巡回。複数回呼んでよい
bool WillBlockOnShutdown() const;                                                    // いずれかが true なら true
```

途中の observer が例外を投げた場合はそこで抜ける（D7）。残りは次の呼び出しで排水されるので、
ループ内で捕捉して続行する実装にはしない。

`Runner::Shutdown()` / `BatchEnv::Shutdown()` とは別クラスの別物である（同名だが無関係）。

### `core/anet-core/src/observers.cpp`

`EpisodeEvalObserver` に `wait_on_exit_` と `std::stop_source stop_source_` を持たせ、2 本を override する。
現在デストラクタにある処理を抽出し、冪等化する。

```cpp
bool EpisodeEvalObserver::WillBlockOnShutdown() const
{
    return wait_on_exit_ && eval_future_.valid()
        && eval_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
}

void EpisodeEvalObserver::Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode)
{
    if (eval_future_.valid()) {
        const bool in_flight = eval_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
        if (in_flight) {
            if (mode == ShutdownMode::CANCEL || !wait_on_exit_) {
                LOG::info() << std::format("eval.[{}]: cancelling in-flight session on exit reason={}",
                    eval_runner_->GetName(), mode == ShutdownMode::CANCEL ? "close" : "config");
                stop_source_.request_stop();
            } else {
                LOG::info() << std::format("eval.[{}]: draining in-flight session on exit", eval_runner_->GetName());
                if (eval_future_.wait_until(deadline) == std::future_status::timeout) {
                    LOG::info() << std::format("eval.[{}]: cancelling in-flight session on exit reason=timeout",
                        eval_runner_->GetName());
                    stop_source_.request_stop();
                }
            }
        }
        eval_future_.get();     // 例外はここで再送出される（D7）。キャンセル後は 1 Step 以内に戻る
    }
    if (eval_pool_) { eval_pool_->Stop(); eval_pool_.reset(); }
}

EpisodeEvalObserver::~EpisodeEvalObserver()
{
    // 安全網。通常は Shutdown 済みで future は無効。未処理なら待たずにキャンセルする（D4）
    try {
        Shutdown(std::chrono::steady_clock::now(), ShutdownMode::CANCEL);
    } catch (const std::exception& e) {
        LOG::fatal() << ToString() << " shutdown failed in destructor: " << e.what();
    } catch (...) {
        LOG::fatal() << ToString() << " shutdown failed in destructor: unknown exception";
    }
}
```

`OnLearn()` の背景投入では `stop_source_ = std::stop_source{};` で作り直してから
`RunEvaluationSession(event_counts, stop_source_.get_token())` とし（一度 `request_stop()` した source は戻らないため）、
`RunEvaluationSession` は `eval_runner_->RunSession(event_counts, token)` へ渡す。
コンストラクタへ `bool wait_on_exit` を追加する（現在は `eval_runner` / `eval_interval` / `use_background` の 3 引数）。

### `core/anet-core/src/trainer.cpp`

`EvalRunner::RunSession` にキャンセル要求を受ける引数を足す。既定値付きなので既存の呼び出しは変えない。

```cpp
void RunSession(const StepCounts& event_counts, std::stop_token stop = {});
```

ループ条件を `while (!stop.stop_requested() && !session_env_->GetSessionResult().has_value())` にし、
採用エピソードの完了数を数えておく。ループを抜けた時点で結果が無ければキャンセルであり、
確定・`SessionEndEvent` 通知・`session end` ログのいずれも行わず、次を 1 行出して返る。

```cpp
LOG::info() << std::format(
    "eval.[{}]: session cancelled learn_step={} exp_step={} elapsed={:.2f}s completed={}",
    name_, event_counts.learn_step, event_counts.exp_step, elapsed, completed);
anet::MetricsLogger::Instance()->Log("eval.[" + name_ + "].session_cancelled", json{
    {"learn_step", event_counts.learn_step}, {"exp_step", event_counts.exp_step},
    {"elapsed_sec", elapsed}, {"completed", completed}});
```

`EvalSessionEnv` は途中状態のまま残るが、次の `Reset()` が非 fresh な lane をすべて Reset する契約（ADR 0034）なので整合は保てる。

`EvalScheduleConfig` へ `wait_on_exit` を追加し、`run.eval_schedule.[tag]` から読む。
現在このスケジュールが持つのは `interval` と `use_background` の 2 つだけである。

```cpp
bool wait_on_exit = true;
schedule_config.Read("wait_on_exit", wait_on_exit, wait_on_exit);
```

`notifier_->AttachScoped<EpisodeEvalObserver>(...)` の引数へ渡す。
起動時の `scheduled` 行に `wait_on_exit` を含め、設定が効いていることを実行ログから確認できるようにする。

### `apps/runner/src/RunnerApp.cpp`

`Config` に `int drain_timeout_sec = 3600;` を追加し、`app.drain_timeout_sec` から読む。0 以下は `ANET_SYSTEM_ERROR`
（キー・指定値・期待範囲を含める）。

```cpp
bool RunnerApp::WillBlockOnDrain() const
{
    return run_manager_ != nullptr && run_manager_->GetNotifier()->WillBlockOnShutdown();
}

void RunnerApp::DrainBackgroundObservers(anet::rl::ShutdownMode mode)
{
    if (drain_completed_ || run_manager_ == nullptr) return;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(config_->drain_timeout_sec);
    LOG::info() << std::format("Draining background observers: mode={} timeout={}s",
        mode == anet::rl::ShutdownMode::WAIT ? "wait" : "cancel", config_->drain_timeout_sec);
    run_manager_->GetNotifier()->Shutdown(deadline, mode);   // 例外は捕捉せず外へ通す（D7）
    drain_completed_ = true;
}
```

`OnExit()` でも `MetricsLogger::Reset()` の前に呼ぶ。`Shutdown()` が冪等なので二重呼び出しでよい。

```cpp
trainer_thread_->Stop();
DrainBackgroundObservers(anet::rl::ShutdownMode::WAIT);   // ← 追加
ShutdownRunLogging();
anet::MetricsLogger::Reset();
```

明示 `Reset()` を行う他の経路は無い。in-process の Optuna run 切替は存在せず、`Reset()` の呼び出しは `OnExit()` とテストだけである。

### `apps/runner/src/RunnerFrame.cpp`

`OnClose()` の先頭で D8 の判定と 3 択を行い、続けて順序を変える。

```cpp
const bool was_paused = wxGetApp().IsTrainingPaused();
wxGetApp().StopTraining();
auto mode = anet::rl::ShutdownMode::WAIT;
if (event.CanVeto() && wxGetApp().WillBlockOnDrain()) {
    wxMessageDialog dialog(this,
        "A background evaluation session is still running.\n"
        "Wait for it to finish (up to the drain timeout), cancel it and close now, or keep running?",
        "Close Run", wxYES_NO | wxCANCEL | wxICON_QUESTION);
    dialog.SetYesNoCancelLabels("Wait and close", "Cancel and close", "Keep running");
    switch (dialog.ShowModal()) {
    case wxID_YES: mode = anet::rl::ShutdownMode::WAIT;   break;
    case wxID_NO:  mode = anet::rl::ShutdownMode::CANCEL; break;
    default:
        wxGetApp().RestartTrainingAfterCloseVeto(was_paused);
        event.Veto();
        return;
    }
}
DetachTrainStatusObserver();
if (wxGetApp().ShouldSaveAgentOnClose()) {
    TrySaveAgent(wxGetApp().GetRunDir() / "agent_close.anet");
}
wxGetApp().DrainBackgroundObservers(mode);   // ← 追加。ShutdownRunLogging より前
wxGetApp().ShutdownRunLogging();
```

メニューの Exit（`RunnerFrame::OnExit`）は `Close(true)` を `Close()` に変える。`wxEVT_TRAINER_EXIT` の `Close(true)` は変えない。
ダイアログの文言は既存 UI に合わせて英語にする。

### `apps/runner/config`

`common.txt` に `wait_on_exit` と `drain_timeout_sec` の既定行は置かない。C++ 側の既定（`true` / 3600）で足り、
効いているかは起動時の `scheduled` 行と排水開始行で確認できる。`?=` の既定葉は既存の実効値を保つために必要な葉だけに置く運用
（AGENTS.md）に当たらないためである。捨ててよいスロットにだけ `wait_on_exit = false` を明示する（現時点で該当する現用 Run は無い）。

### 実装順序

1 コミットでよいが、次の順に進めると途中で止まっても現状より悪くならない。

1. core 側: 基底 4 種のフック 2 本、wrapper 4 種の転送、`Notifier::Shutdown()` / `WillBlockOnShutdown()`、
   `RunSession(counts, stop_token)`、`EpisodeEvalObserver::Shutdown()` の冪等化。T1 / T2 / T3 / T4 / T6 / T9 / T11。
   ここまでで「常に待つ」と「キャンセル」が core で成立する。
2. `RunnerApp`: 排水の順序、`app.drain_timeout_sec`、排水開始ログ、予算到達経路。T8 / T10 と T7 の実 Run。ここまでで G1 / G3 / G4 を満たす。
3. `RunnerFrame`: `CanVeto()` 判定、Pause、3 択ダイアログ、veto と復帰、メニュー Exit の `Close()`。手動確認。ここで D8 を満たす。
4. `wait_on_exit` キーの読み取りと `scheduled` 行。T5。ここで G2 を満たす。

## テスト

- **T1**: background 有効・in-flight ありで `Notifier::Shutdown(deadline, WAIT)` を呼ぶと、セッションが完了し
  `SessionEndEvent` が通知され、scalar が記録される。observer は本番と同じく `AttachScoped()` で attach し、
  wrapper の転送を被覆する。
- **T2**: `wait_on_exit = false`、または `mode = CANCEL` では 1 Step 以内に返り、`SessionEndEvent` は通知されず、
  `cancelling` と `session cancelled` の行が出る。
- **T3**: `Shutdown()` の 2 回呼び出し、および `Shutdown()` 後のデストラクタで落ちない（冪等）。
- **T4**: `use_background = false` のスロットで `Shutdown()` が no-op、`WillBlockOnShutdown()` が false。
- **T5**: `wait_on_exit` の既定が `true` で、`run.eval_schedule.[tag].wait_on_exit = false` が効く。
- **T6**: セッションが例外を投げた場合、明示 `Shutdown()` 経由では `eval_future_.get()` により再送出され、
  デストラクタ安全網では例外を捕捉して FATAL を記録し `std::terminate` しない
  （現行の `RethrowCompletedBackgroundEval` と同じ契約を保つ）。
- **T7**: 実 Run。予算末尾に断面を発火させ、`session end` が出て scalar が記録され、
  trace の行数が `eval_episodes` と一致する。
- **T8**: 過去の deadline を渡した `Shutdown(deadline, WAIT)` で、`cancelling ... reason=timeout` が出てキャンセルされ、
  1 Step 以内に返る。
- **T9**: `RunSession` に stop 要求済みのトークンを渡すと、`SessionEndEvent` を出さず scalar も出さずに返り、
  `session cancelled` の行に完了数が載り、`eval.[<tag>].session_cancelled` の json レコードが Metrics マスタに出る。
  完了済み採用エピソードの trace 行は残る。
- **T10**: `app.drain_timeout_sec` の 0 以下・非整数が `ANET_SYSTEM_ERROR` になる。
- **T11**: `WillBlockOnShutdown()` の 4 ケース（進行中かつ `true` / 完了済み / `false` / 前景）。

## Complexity audit

| # | 機構 | 裁定 | 理由 / 切ったら戻る痛み |
|---|---|---|---|
| 1 | observer 基底の `Shutdown(deadline, mode)` / `WillBlockOnShutdown()` フック + `RunnerScoped*` wrapper の転送 | keep | 背景スレッドを持つ observer 一般の teardown 口。現状 override は 1 つだが、口が無いと停止経路から触れない。wrapper 転送が無いと口が届かない |
| 2 | `Notifier::Shutdown()` / `WillBlockOnShutdown()` | keep | 呼び出し側が observer の実体を知らずに一括排水・問い合わせできる |
| 3 | `wait_on_exit`（bool） | keep | 待ちが 20 分級になるスロットと数十秒のスロットが同居する。`false` の利用先は現用 Run に無いが、12 のキャンセル経路を共用するので追加コストはキー読み取りだけ |
| 4 | 排水の上限時間（`app.drain_timeout_sec`） | keep | 無人 batchrun で唯一効く自動弁。切ると異常に長いセッションで exe が落ちない。中断点（12）ができたので「時間切れ＝捨てると同義」という旧 NG2 の反論は消えた |
| 5 | 部分セッションの確定 | cut（NG1） | `eval_episodes` が Run 内で不揃いになり比較不能を新設する |
| 6 | 待機開始ログ・排水開始ログ | keep | UI スレッドが最大で上限時間まで無応答になるため、ハングと区別できない |
| 7 | `OnClose` と `OnExit` の二重呼び出し | keep | main loop の fatal 後は `OnClose()` を通らず `OnExit()` だけが走る。D7 で例外が抜けた後の残りスロットも 2 回目が排水する |
| 8 | `common.txt` の既定行（`wait_on_exit` / `drain_timeout_sec`） | cut | C++ 既定で実効値は変わらず、ログで確認できる。`?=` は既存の実効値を保つ葉だけに置く運用に当たらない |
| 9 | `RunManager` が `EpisodeEvalObserver` を直接保持して排水する案 | 不採用 | 基底フック・`Notifier`・wrapper 転送を不要にする最小解だが、背景スレッドを持つ observer 一般の口を優先した（再グリルの裁定） |
| 10 | キー無しで常に待つ案 | 不採用 | 最小解との差は 3 だけ。スロット間のコスト差を理由に 3 を保持した（再グリルの裁定） |
| 11 | `RunnerApp::DrainBackgroundObservers()` での例外捕捉 | 不採用 | 既存の再送出契約と fatal 経路（`OnExceptionInMainLoop`）で exit code 1 に到達できる。捕捉層を足さない（D7） |
| 12 | `RunSession` の協調キャンセル（`std::stop_token`） | keep | 無いと「捨てる」も上限時間も成立しない（pool の join で必ず完走を待つ）。`PinnedThreadPool` は触らない |
| 13 | 人が閉じたときの 3 択ダイアログ | keep | 設定ミスの Run を即落としたい場面と、Eval 中と意識せず × を押す場面で、待つしかない設計は強制終了へ追い込む。強制終了は flush と exit code 0 を失う |
| 14 | 待機中にも中断できる進捗ダイアログ | defer（NG5） | 排水に入る前の 1 回の選択で足りる。待機中に強制終了して記録を失った事例が出たら足す |
| 15 | pool を detach / leak して「捨てる」 | cut | スレッドが `Reset()` と `RunManager` 破棄を生き延び、026 の経路に加えて解放後アクセスになる。`std::thread` は join せずに破棄できない |
| 16 | `PinnedThreadPool` にタスク単位の stop token を配る改造 | cut（NG4） | 利用者が 1 つしか無い口を汎用層に置く。pool の「投入済みを捌いて止まる」意味論はキャンセルと独立 |
| 17 | スロット単位の上限時間 | cut | 合計が有界にならず、スケジュール定義に終了ポリシーが混ざる |
| 18 | 上限時間の無制限モード / キー無しで無制限 | cut | 番兵値は禁止（AGENTS.md）。設定し忘れで exe が有界にならない。長く待つなら大きな値を書く |
| 19 | Metrics マスタへのキャンセル記録 | keep（json 1 行） | 「trace 行あり・scalar 無し」が起点のクラッシュと同型になり、metrics だけを読む `inspect_run.py` で区別できなくなる。既存 json チャネルなら scalar / trace の契約と読み手に波及しない |
| 20 | T7 の手順の具体化 | 実装側裁量 | 受入 1 の確認項目で十分 |
| 21 | CANCEL 側の第 2 期限と強制終了 | defer（NG6） | 協調停止は本人が中断点へ戻る前提で、`Step()` が返らない故障に安全な強制手段は無い。WAIT の timeout 後も同じ前提。未観測なので実例が出たら別 PRD |
| 22 | deadline 共有の順序緩和 | cut | 各セッションは並行に進むので、走査順によらず全員が deadline まで進める。受入 8 に「上限時間内」の条件を書くだけで足りる |

## 受入基準

1. 予算末尾に発火した断面セッションが完走し、`session end` ログ・scalar・trace が揃う。
   trace の行数が `eval_episodes` と一致する。
2. `wait_on_exit = false` のスロットは終了時に 1 Step 以内にキャンセルされる。
3. `Notifier::Shutdown()` が `MetricsLogger::Reset()` および `ShutdownRunLogging()` より前に実行される。
4. `anet-core-test` 全緑（既知の既存失敗を除く）+ T1〜T6、T8〜T11。
5. 排水が不要な Run（in-flight 無し）で終了時間が増えない。
6. 起動時の `scheduled` 行に `wait_on_exit` が出る。
7. batchrun 構成で予算末尾に発火した Run の exit code が 0 になる（起点 Run は非 0 だったとみられる）。
8. 複数スロットが同時に in-flight でも、上限時間内に終わるものはすべて完走して記録される（起点 Run は `greedy_dist` と `eval` の 2 本）。
9. 待機開始ログは実際に待った場合だけ出る（完了済みなら出ない）。
10. 上限時間を超えた Run は残りをキャンセルして exit code 0 で終わり、`OnClose()` から終了までが上限 + 1 Step + 数秒以内。
11. キャンセルしたセッションは scalar を出さず、`cancelling` と `session cancelled` の 2 行と、Metrics マスタの
    `eval.[<tag>].session_cancelled` レコードが出る。
12. Eval 中に × を押すと Stop/join 後の照会でダイアログが出て、Keep running で同じ thread が元の pause 状態のまま再開し、Cancel and close で数秒以内に閉じ、Wait and close で完走後に閉じる。
13. 予算到達ではダイアログが出ない。

## 影響・移行

- 既定が `true` なので、**現用設定はすべて挙動が変わる**。予算末尾に断面が乗る Run では
  終了が最大でセッション 1 本ぶん（Atari の ε=0 / 400 エピソードで約 20 分）伸び、上限は既定 3600 秒である。
- 本 PRD が入ると、**末尾発火を避けるための interval 回避策が不要になる**。
  むしろ末尾ぎりぎりに発火させて排水で完走させるほうが、予算終端の断面を Run 内で得られる。
  評価専用 Run（`run.@evalonly` 系）を断面のためだけに立てる運用を畳める。
  `Atari.txt` の `run.@eval2ch` にある回避注記の更新は Atari 側の作業として別途行う。
- **999 が入るまでの移行依存（Atari 側の作業）**: 配線 arm（bat の `run.$=%TP%>run.@to_400k` は `run.@eval2ch_r1` を含む）と
  `run.@pl_check`（`Atari.txt:377-380`。`[eval_target]` / `[eval]` は切っているが `[greedy_dist]` は切っていない）では、
  t=0 に発火した `greedy_dist` セッション（遅いゲームで 1 時間級）が 2〜4 分の Run の終了時に in-flight になり、
  既定では上限時間まで待つ。これらの arm に `run.eval_schedule.[greedy_dist].interval = 0` か
  `run.eval_schedule.[greedy_dist].wait_on_exit = false` を入れる。999 が入れば t=0 発火自体が消える。
- 人が閉じたときの挙動が変わる。Eval 中なら 3 択ダイアログが出る（D8）。メニューの Exit は veto 可能な close になる。
- PRD 026 は R1 を含むため凍結のままでよい。本 PRD が R2 相当を先に満たすことで、
  026 が対象としていたクラッシュ経路のうち正常シャットダウン分は塞がる。
  in-process の run 切替経路は無いので、026 R1 に残る動機は「想定外の背景呼び出し」だけになる。
- 同一変更で移行する現行文書（クリーンブレーク方針）:
  1. `docs/design/140_observability.jp.md` §5.2 の終了シーケンスに 3 択ダイアログと排水（`DrainBackgroundObservers(mode)` →
     `Notifier::Shutdown(deadline, mode)`）を `SaveAgent` と `ShutdownRunLogging()` の間へ挿入。§7.2 に排水開始行、
     `draining` / `cancelling` / `session cancelled` 行の規則と、`scheduled` 行が `wait_on_exit` を含むことを追記。
     出力ファイルの表に json レコード `eval.[<tag>].session_cancelled` を追記。
  2. `docs/design/100_runtime_and_configuration.jp.md` のキー表 `run.eval_schedule.[tag].*` に `wait_on_exit` を追加し、
     §7.2 lifetime に排水の位置（checkpoint 保存の後、`ShutdownRunLogging()` の前）と上限時間を追記。
  3. `docs/design/020_user_guide_run.jp.md` の `app.*` 表に `app.drain_timeout_sec` を追加し、× を押したときの 3 択と
     eval_schedule の `wait_on_exit` を利用者向けに記述。`docs/design/010_framework_overview.jp.md` の eval_schedule の記述に
     終了時に background セッションを完走またはキャンセルすることを 1 文で追記。
  4. [frozen/026](frozen/026_metricslog_lifecycle_10prd.md) の先頭に「R2 相当は 076 が先行実装。残りは R1」を 1 行。
  5. 用語「排水」は [CONTEXT.md](../../CONTEXT.md)、判断理由は [ADR 0045](../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)。
  `.en.md` は translate-docs スキルの管轄なので含めない。

### 成果の確認

1. 予算末尾に発火する Run で、実行ログに `draining in-flight session on exit` → `session end` の順に行が並び、
   trace の行数が `eval_episodes` と一致し、scalar が記録され、exit code が 0 であること。
2. 上限時間を短くした Run（例 `app.drain_timeout_sec = 60` で長いセッション）で、`cancelling ... reason=timeout` →
   `session cancelled` の順に行が並び、scalar が無く、Metrics マスタに `session_cancelled` レコードがあり、exit code が 0 で、
   `OnClose()` から終了までが上限 + 数秒以内であること。
3. Eval 中に × を押すとダイアログが出て、Keep running で学習が続き、Cancel and close で数秒以内に閉じること。
