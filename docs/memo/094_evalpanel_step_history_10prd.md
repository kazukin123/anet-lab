# Evaluation Trace — EvalPanel の遷移を step trace として表で見て、その時点を再生する PRD

> 2026-10-18 起票（グリル前の暫定版）。ユーザーの希望で、999 ではなく正式番号で起票した。
> 2026-10-18 改訂: 追加するパネルを近似優先度の専用から、EvalPanel の step 単位の行を直近の履歴として表で見る汎用のパネルに改めた（ファイル名も `094_evalpanel_approx_priority_view` から変更）。
> 2026-10-19 グリル（`/anet-create-prd 094`、Claude Code）で全面改訂した。位置付けを「step 単位の詳細を記録して見るシリーズの第一弾」とし、近似優先度はその最初の利用例にした。主な決定は、値の読み口の統一とメトリクス配線の分離を段階 1 に入れること、step trace を trace チャネルへ出す段（旧段階 2）を PRD 081 へ移すこと、近似優先度を Agent の step 部品が計算すること（Actor の設定キーは足さない）、列をメトリクスの定義と分けて番号付きで宣言すること、`$env_step`・`$agent_step` を足すこと、行を選ぶか再生すると EvalPanel・QValuePanel・Conv2d が Trace モードでその時点を出すこと、自動リセットの時機は別の基盤 PRD で先に直すこと。§3 の現状をコードと照合して直した（毎秒の step 数、式の所在、ヒントの所在、env の値の出口など）。起草後に別モデル（Fable）の検討を受け、eval_panel の Actor の Q が既定で target network であること（§3.2、D6）、Trace モードに入ったら EvalPanel を一時停止すること（D11）などを反映した。ファイル名は、ほかの文書と wiki の作業記録が指しているので変えない（題は Evaluation Trace）。決定の記録は [ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md) と [ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)。
> 2026-10-19 基盤 PRD [097](097_env_auto_reset_after_notification_10prd.md) のグリルの決定に合わせて、NG6・§3.3・D4・D11（Trace モード）・§7 の 1 を書き直した。097 の決定は、BatchEnv の `Step` はリセットせず、Runner が通知のあとに `ResetEndedEpisodes()` を呼ぶこと（097 の D1）と、リセット後の情報は Runner だけが受け取ること（097 の D4）である。本 PRD の決定は変えていない。
> 2026-10-20 追記: 097 を再グリルで改めた（[ADR 0060](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)）。遷移の結果は Step の結果で運び、env への問い合わせ（`$env`）は今の env の状態を返す。BatchEnv は Reset と Step のままで、`ResetEndedEpisodes()` は作らない。そのため、§3.3 の 097 の行と §7 の 1 を事実に合わせて直した。D4 の「`$env` は遷移のあと・リセット前の env を読む」は成り立たなくなったので、扱いを §7 の 1 の未決に置いた。D11 の「終端の行の画面はリセット前」と、097 の D4（リセット後の情報は Runner だけが受け取る）は変わらない。
> 2026-10-20 改訂（Claude Code、097 の申し送りの反映）: step trace の列から `$env` を外した（ユーザーの裁定）。`$env_step` を 097 のメトリクスと同じ target にし、metrics の DSL へ `$env_step` を足すのは 097、`$agent_step` は 081 とした。D4・D5・D8・§7 の 1・受入 4 を直した。
> 2026-10-21 訂正（Claude Code、実装のコードレビューで判明）: §3.1 の「1 frame に複数の step を進める構成では、View のスナップショットは step ごとには作られない」は誤りだった。EvalPanel は View の更新を強制の指定で呼んでおり、変更前から step ごとに作っていた。§3.1 と D11 の該当文を事実に合わせて直した。決定は変えていない。
> 2026-10-21 改訂（Claude Code、実装のコードレビューで判明した、共通の仕組みへの DQN の鍵の漏れの直し）:
> - 旧 D5 は、`$agent_step` の鍵を共通の側で `q_sa`・`state_value`・`approx_priority` に決め、構築時に検査していた。そのため、共通の型と列の解析が DQN の鍵の名前を持っていた。
> - 鍵は Agent の step 部品が決め、共通の仕組みは鍵の名前を持たない形に改めた。部品が答えない鍵は最初の行で止める。
> - 部品を出さない Agent では、`$agent_step` の列は鍵によらず空欄にする。そのため、その場合は鍵の書き間違いに気づけない。
> - 実装（20impl）が先に書き換えた D5・D6・D9 を、契約の文に整えた。D4・D8・D10・§4.4・受入 5 も合わせた。
> 2026-10-21 改訂（実装の名前・継承・置き場の見直し。ユーザーと Claude Code が決め、Codex が実装した）:
> - Agent の step 部品の名前を、責務が分かる `AgentStepDiagnosticsProvider` に改めた。関連する型も `AgentStepDiagnostics` で始まる名前にそろえた。
> - `BatchActionInfo` を Module にした。実行した行動（`action`）は、その型の `GetScalar` が答える（D5）。
> - 追従 pane の口の名前を `CreateTraceSnapshot` にし、`MetricsWiring` を `RunManager` の内側のクラスにした。
> - §4.4 のクラス一覧を合わせ、`BatchActionInfo` の行を足した。
> 2026-10-22 改訂（表示と操作の調整。ユーザーが承認した）:
> - D10〜D13 を改めた。列の幅は、長い値が出た列だけを広げる。表の配色はシステム色を基本にし、報酬と終端の行には薄い色を重ねる。プレイヤーは、再生と一時停止を 1 つにまとめた 6 個のアイコンボタンにする。Trace モードの間は、操作バー全体を赤枠で囲む。既定の列ラベルを `Q_sa`・`V_s` にする。
> - コマ送りのアイコンは縦棒の右に右向き三角（❙▶）、コマ戻しは左向き三角の右に縦棒（◀❙）にする。メインツールバーの Step のアイコン（今は三角の右に縦棒）も、コマ送りと同じ形に直す。
> - Live でコマ戻しを押すと、いちばん新しい行の 1 つ前の行を選んで `Trace: step back` を出す。行が 1 件なら、その行を選んで `Trace: start` を出す（2026-10-19 の裁定では、どちらもいちばん新しい行を選んで `Trace: step back` を出していた）。Live でコマ送りを押すと、いちばん新しい行を選び、いちばん新しい行でのコマ送りと同じく `Trace: end` を出す（2026-10-19 の裁定では `Trace: step`）。
> - 再生の向き、速度の段階、停止で Live に戻ること、行の持ち方、記録する値は変えない。受入基準（7〜12）、クラス一覧、文書同期の対象を合わせた。
> 2026-10-22 追加改訂（ユーザー指定）:
> - 既定のtruncated列は取得キーを変えず見出しをtruncとする。低速倍率は小数表記にし、操作バー全体とアクション列までを守る最小幅を設ける。行選択でも列見出しの色を変えず、Traceの目印を赤枠とcaptionにそろえる。
> 2026-10-22 列順改訂（ユーザー指定）: 行動と結果を隣り合わせで読むため、既定のreward列をaction列の直後へ移す（D13）。
> 2026-10-23 改訂: KFM の Run 名 2 つを今のフォルダ名に直した（`…_atari_kung_fu_master_…` などから `…_kfm_…` への改名。値は変えていない）。
> 起点: Atari kung_fu_master（以下 KFM）でボス討伐が定着しない件で、「ボス撃破のような貴重な経験は、TD 誤差による PER の優先度で本当に目立つのか」という疑問が出たこと（§2）。学習側の計器（[999_replay_eviction_sample_count](999_replay_eviction_sample_count_10prd.md)）は Run 全体の集計を出すが、1 つ 1 つの場面で行動・報酬・優先度がどう並ぶかを見る手段が無い。
> 関連: [done/035_approx_actor_priority_per_10prd.md](done/035_approx_actor_priority_per_10prd.md)（近似 Actor 初期優先度）、[ADR 0010](../adr/0010-actor-priority-mean-q-approx.md)、[ADR 0012](../adr/0012-replay-initial-priority-hint-completion.md)、[ADR 0036](../adr/0036-actor-q-hint-three-columns-munchausen.md)、[081_evalpanel_episode_trace_10prd.md](081_evalpanel_episode_trace_10prd.md)（鏡写しインスタンスの trace。step trace の記録はここへ移す）、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)、[ADR 0047](../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md)、[140_observability.jp.md](../design/140_observability.jp.md)、[160_applications_and_tools.jp.md](../design/160_applications_and_tools.jp.md)、[CONTEXT.md](../../CONTEXT.md) の「Actor Qヒント」「近似Actor初期優先度」「raw priority」「鏡写しインスタンス」「補助 pane」「実行時 UI 操作」「trace」「step trace」「Live モード／Trace モード」。

## 1. 目的と問題

EvalPanel（鏡写しインスタンス）で遊ばせているゲームを、1 遷移ずつ追う手段が無い。行動・報酬・Q・優先度・env の途中の値を並べて見られず、ある場面で Agent が何を見て何を選び、その結果どうなったかを、あとから確かめられない。

本 PRD は、EvalPanel の 1 遷移を 1 件とする記録（**step trace**）を作り、新しいパネル **Evaluation Trace** で直近の step trace を表で見られるようにする。行を選ぶか再生すると、EvalPanel・QValuePanel・Conv2d がその時点の表示に切り替わる（**Trace モード**）。step 単位の詳細を記録して見るシリーズの第一弾であり、近似 Actor 初期優先度（以下、近似優先度）の列はその最初の利用例である。KFM では、ボス撃破や出口の換算の場面で近似優先度がふつうの場面より高く出るかを、その場で確かめる用途がまずある。

### ゴール

- **G1**: メトリクスの値の読み口（scalar と trace が別々に持っている target の解決）を 1 つにし、メトリクスの配線を RunManager から 1 つのクラスへ分ける。振る舞いは変えない（段階 1）。
- **G2**: EvalPanel の 1 遷移ごとに step trace を作る。列は番号付きで宣言し、値の持ち主ごとの target から、その遷移の時点に合う値だけを読む（段階 2）。
- **G3**: 近似優先度の列を、学習側の初期優先度と同じ式で出す（段階 2）。
- **G4**: Evaluation Trace パネルで直近の step trace を表で見て、行を選ぶか再生すると、EvalPanel・QValuePanel・Conv2d がその時点を出す（段階 2）。列幅・テーマに合う配色・動画プレイヤーに近い操作・Trace中の識別により、狭いパネルでも履歴を読み、操作できるようにする（2026-10-22のUI調整、D10〜D13）。

### 非ゴール

- **NG1**: 学習側の優先度そのものの表示。replay の中の遷移の優先度や抽選回数は [999_replay_eviction_sample_count](999_replay_eviction_sample_count_10prd.md) が扱う。
- **NG2**: 学習・抽選・優先度の値を変えること。本 PRD は観測だけで、学習系列を変えない。
- **NG3**: 定期評価（configured eval）と学習側（train runner）の step trace。量が大きすぎる（§3.1）。
- **NG4**: step trace を trace チャネル（Run artifact）へ出すこと。PRD 081 を改訂して取り込む（§7 の 2）。本 PRD は、081 がそのまま使える生成器を作るところまで。
- **NG5**: Event / Observer の型の複製（`TrainEvent`・`LearnEvent`・`EpisodeEndEvent`・`SessionEndEvent` ごとの interface・RunnerScoped・Notifier のメソッド）の統一。step trace は既存の `TrainEvent` で受けられるので、今回の要求からは導けない別の関心である。
- **NG6**: 通知の中で読む遷移の結果が終端の lane でリセット後になる問題への対処（遷移の結果を Step の結果で運ぶ直し）。基盤 PRD [097](097_env_auto_reset_after_notification_10prd.md) で決め、本 PRD より先に実装する（§7 の 1）。

## 2. 背景（KFM）

- KFM の実時間優先 3 seed（100M）のうち、ボス行き（面クリア）を学んだのは seed 3（`run_20260925-130017_atari_kfm_rr1_100m_seed3`）だけで、報酬クリップを外す、ライフで区切る、ε を下げる、PER の `per_alpha` を 0.6 にする、UQE にする、などの 2026-10-14〜17 の 30〜50M の Run ではボス行きは定着しなかった（出典: [999_intrinsic_reward_novelty §2.2](999_intrinsic_reward_novelty_10prd.md#22-学習の結果)）。
- クリップ後の 1 step あたりの報酬は、ボス行きの seed 3 が 0.042、1 面の稼ぎが 0.048 でほぼ同じ。当て 1 発は 0 点、撃破はクリップ後 +1 で雑魚と同じで、成功を区別できるのは出口の換算だけ（出典: [999_intrinsic_reward_novelty §2.3](999_intrinsic_reward_novelty_10prd.md#23-分析の結論2026-1016〜17)）。
- 実時間優先の構成では、1 遷移が寿命中に引かれる回数の期待値は約 1.0 で、4 件に 1 件は一度も引かれずに消える。`per_alpha` 0.2 では、raw priority が 20 倍でも引かれやすさは約 1.8 倍にとどまる（出典: [999_replay_eviction_sample_count §2](999_replay_eviction_sample_count_10prd.md#2-現状2026-10-16-に確認したこと)、`run_20261016-042902_kfm_rr1_50m_noclip200_seed2`）。
- TD 誤差による優先度が珍しい経験を目立たせないと考えられる理由は、報酬クリップの下では撃破の報酬が雑魚と同じ +1 であること、撃破の先の価値（出口の換算）がまだ target に入っていないこと、優先度は引かれたときにしか更新されないこと、IQN の TD は τ の揺らぎを含むこと。どれも推測で、場面ごとに確かめる手段が無い。

## 3. 現状（2026-10-19 にコードで確かめたこと。§3.6 は 2026-10-22）

### 3.1 EvalPanel と 1 step の流れ

- EvalPanel は、`RunnerFrame` が `CreateEvalRunner("EvalPanel", eval_config_tag)` で作る 1 lane の EvalRunner（鏡写しインスタンス、Actor キーは `eval_panel`）を、UI スレッドの wxTimer で回す（[RunnerFrame.cpp:477-480](../../apps/runner/src/RunnerFrame.cpp)、[EvalPanel.cpp:243-267](../../apps/runner/src/EvalPanel.cpp)）。`Notifier::Notify` は呼んだスレッドで同期に購読者を呼ぶので、EvalRunner の購読者は UI スレッドで呼ばれる（[rl.cpp:1110-1117](../../core/anet-core/src/rl.cpp)）。
- 1 回の `EvalRunner::DoStepInternal` は、`MakeAction(s_t)` → 行動の強制があれば `WithAction` → `env->Step(a_t)` → `TrainEvent` の通知 → `state_ = continue_state` の順に進む（[trainer.cpp:231-298](../../core/anet-core/src/trainer.cpp)）。外から見える通知は 1 step に 1 回で、行動の決定と env の step は同じ呼び出しの中で続けて起きる。
  ```
  （待ち）runner は s_t を持つ。行動はまだ決まっていない
  DoStep: MakeAction(s_t) → a_t、Q(s_t,·)
          env.Step(a_t)    → r_t、done、truncated、next_state=s_{t+1}、info（aux）。終端なら env がこの中で自動リセットする
          Notify(TrainEvent)   ← 完了した遷移 t と、次の状態 s_{t+1} が一緒に届く
          state_ = continue_state（終端ならリセット後の s_0）
  ```
- 毎秒の step 数は、構造体の既定（30 fps × 3 step/frame、[EvalPanel.hpp:31-32](../../apps/runner/src/EvalPanel.hpp)）ではなく設定で決まる。online の既定は 30 fps × 1 step/frame（[common.txt:133-134](../../apps/runner/config/common.txt)）、Atari の online は 14.5 fps（[Atari.txt:1413](../../apps/runner/config/Atari.txt)）。1 時間で約 5〜11 万遷移。定期評価に step 単位の記録を付けると、30M の Run で数百万行になる。
- 設定キーは `app.eval_panel.*`（`app.online.*`・`app.batchrun.*` は `app.$` で流し込む層）で、パネルごとに `app.<パネル>_panel.*` の並びになっている（[RunnerApp.cpp:75](../../apps/runner/src/RunnerApp.cpp)、同 101-111）。
- EvalPanel の View は、通知を受けて表示用のスナップショットを作り（`CreateViewData` → `UIDataStore`、[gui.hpp:261-279](../../core/anet-core/include/anet/gui.hpp)）、timer の 1 frame ごとに 1 回取り出して描く。`UIDataStore` は、取り出しの要求があったか一定時間たったときだけ中身を入れ替える（同 186-227）。ただし EvalPanel は View の更新を強制の指定（`UpdateViewData(event, true)`）で呼ぶので、この間引きは効かず、スナップショットは step ごとに作られる（[EvalPanel.cpp](../../apps/runner/src/EvalPanel.cpp) の `Initialize` の購読。コミット 4d191af3 の時点で 126 行）。間引きが効くのは、強制しない TrainPanel の View である。1 frame に複数の step を進める構成（`step_per_frame` > 1。batchrun の既定は 10）でも作るのは step ごとで、描くのはその frame の最後の step のものだけである（[EvalPanel.cpp:245-267](../../apps/runner/src/EvalPanel.cpp)）。Atari の View は、env の `rgb_frame`（通知の時点の画面。終端ではリセット後）と、step_result の `next_state` の観測・aux の得点・残機・RAM（リセット前）、行動名と報酬を出す（[AtariView.cpp:218-257](../../core/envs/atari1/src/AtariView.cpp)）。LunarLander の View は `next_state`（リセット前）と行動・報酬を出す（[LunarLanderView.cpp:28-44](../../core/envs/lunarlander1/src/LunarLanderView.cpp)）。どちらも「この状態に来た行動」を新しい状態に重ねて描いている。
- QValuePanel は同じ通知を `FunctionTrainObserver` で受け、`action_info` の aux の `q_values`（分位点系は `q_quantiles` など）と選んだ行動を表示する（[QValuePanel.cpp:281-308](../../apps/runner/src/QValuePanel.cpp)、同 315-397）。表示用のデータは通知ごと（毎 step）に `CreateData` で作り、画面の更新だけを frame に 1 回へまとめている（同 293-302）。表示は Q(s_t, ·) で、EvalPanel の画面（s_{t+1}）より 1 step 遅れている。
- Conv2dPanel は鏡写しインスタンス（EvalPanel）の通知から、`action_info` の aux の `nn_trace/*`（その step の判断で出した NN の活性）を画像にする（[RunnerFrame.cpp:1060](../../apps/runner/src/RunnerFrame.cpp)、[Conv2dPanel.cpp:76-112](../../apps/runner/src/Conv2dPanel.cpp)）。SweepHeatMapPanel は学習側の runner の `LearnEvent` で、学習側のネットワークを状態の 2 次元の格子の上で掃いた地図を作る（[RunnerFrame.cpp:1045](../../apps/runner/src/RunnerFrame.cpp)、[observers.cpp:390](../../core/anet-core/src/observers.cpp)）。EvalPanel の step とは関係しない。
- パネルを閉じても購読は外れない。`AttachScoped` は内側の observer を返し、`Detach` はそれと一致しないので何も外れない（[rl.hpp:1104-1124](../../core/anet-core/include/anet/rl.hpp)、[999_notifier_scoped_detach](999_notifier_scoped_detach_10prd.md)）。Notifier はロックを持たない。
- 表を出す部品は QValuePanel の非仮想の `wxGrid` だけ（[QValuePanel.cpp:228-279](../../apps/runner/src/QValuePanel.cpp)）。補助 pane（HeatMap・Conv2d）は `AddAuxPane` で右端の列へ足す（[RunnerFrame.cpp:959-974](../../apps/runner/src/RunnerFrame.cpp)）。折れ線の部品 `PlotPanel` はどこからも使われていない。
- ステータスバーの左端の欄（field 0）は操作結果専用で、EvalPanel の pause/resume もここへ出す（[RunnerFrame.cpp:313-330](../../apps/runner/src/RunnerFrame.cpp)、[EvalPanel.cpp:170-179](../../apps/runner/src/EvalPanel.cpp)）。

### 3.2 近似 Actor 初期優先度の式とヒント

- 式は 1 か所ではない。`DqnInitialPriorityEstimator::Estimate`（[dqn_based_agent.cpp:379-423](../../core/anet-core/src/dqn_based_agent.cpp)、無名空間）は、target = 割引報酬和 ＋ 開始時の Munchausen 項 ＋ discount × 状態価値（終端なら 0、TBO なら h 空間）から |target − Q(s,a)| を作り、per_eps と clip を適用する。生成関数 `dqn::CreateInitialPriorityEstimator(const LearnerConfig&)` は内部ヘッダにある（[dqn_based_agent.hpp:168](../../core/anet-core/src/dqn_based_agent.hpp)）。
- n step の割引報酬和、実 n step 数、`discount = γ^実n` は replay 側が作る（系列の確定 [replay_buffer_impl.cpp:150-178](../../core/anet-core/src/replay_buffer_impl.cpp)、割引報酬和 同 189-215、推定器の呼び出し 同 421-428）。n step の途中で終端になれば短い長さで確定する。打ち切り（truncated）の窓にかかる遷移は bootstrap せず最大優先度へ逃がす（同 393-398）。実効の n は `use_n_step ? n_step : 1`（[dqn_based_agent.cpp:2680](../../core/anet-core/src/dqn_based_agent.cpp) 付近）。
- 材料の Actor Q ヒント（実行した行動の Q、状態価値、Munchausen 項の 3 列、[ADR 0036](../adr/0036-actor-q-hint-three-columns-munchausen.md)）は、aux ではなく別の欄 `GetReplayInitialPriorityHint()` にあり（[rl.hpp:544-547](../../core/anet-core/include/anet/rl.hpp)）、`Actor::MakeAction` が aux の `q_values` から作る（[dqn_based_agent.cpp:1917-1952](../../core/anet-core/src/dqn_based_agent.cpp)）。作るのは「PER あり かつ `per_initial_priority_mode = actor_approx`」のときだけで、Actor キーによらない（[default_dqn_agent.cpp:532-533](../../core/anet-core/src/default_dqn_agent.cpp)）。KFM の実時間優先の構成（`max`）では、eval_panel を含むどの Actor もヒントを作らない。`actor_approx` の構成では、eval_panel の Actor も作る。
- eval_panel の Actor は、評価用の Actor を継いで方策だけを Greedy にしたもので、ネットワークは既定で **target** である（`actor.[eval_panel]` ← `actor.[eval_target]` ← `network = target`。[agent.txt:344-345](../../apps/runner/config/agent.txt)、同 394、[160](../design/160_applications_and_tools.jp.md) の EvalPanel の説明）。学習側の近似優先度は、学習 Actor の online の Q で計算される。
- 報酬は、Agent が Push の前に reward scaler を通す（[default_dqn_agent.cpp:557-574](../../core/anet-core/src/default_dqn_agent.cpp)、[PRD 035](done/035_approx_actor_priority_per_10prd.md) の 29 行目）。倍率は統計を更新しない `GetScalar("reward_scaler.scale")` で読める（[scaler_impl.cpp:626-635](../../core/anet-core/src/scaler_impl.cpp)）。EvalRunner の報酬は env の報酬（Atari は env で符号クリップ）で、reward scaler は通っていない。
- 学習側の Actor が UQE のときは学習側 Actor の aux `q_values` が上側の分位点の平均になる（[dqn_based_agent.cpp:1657-1674](../../core/anet-core/src/dqn_based_agent.cpp)）。同じ状態でも、eval_panel の Q は学習側 Actor の Q と違いうる（[ADR 0010](../adr/0010-actor-priority-mean-q-approx.md) が UQE での系統差を許容している）。
- 近似の精度は、`actor_approx` の Run の `39_agent_per/01_actor_learner_rank_corr`（近似と学習側の優先度の順位相関）で測れる（[metrics_scalar.txt:113](../../apps/runner/config/metrics_scalar.txt)）。実時間優先の KFM の Run は `max` なので、この値が無い。

### 3.3 env の値の出口と自動リセット

- env の値の出口は 2 つある。`GetScalar(key, lane)` は完了したゲームの統計用で、Atari の `game_score`・`game_len`・`ram_metric.[n]`・`hns57` などは完了時だけ値を返し、それ以外は NaN。`lives` は今の値（[AtariEnv.cpp:839-891](../../core/envs/atari1/src/AtariEnv.cpp)）。step_result の aux（`名前 → Tensor`、[rl.hpp:349](../../core/anet-core/include/anet/rl.hpp)）は 1 step ごとの報告で、Atari は途中の `game_score`・`game_len`・`game_frames`・`lives`・`ram_metric.<label>` を毎 step 入れる（[AtariEnv.cpp:682-694](../../core/envs/atari1/src/AtariEnv.cpp)）。GridMaze・LunarLander・DropMerge も aux を持つ。
- 単一の env をまとめる BatchEnv は、Step の中で lane ごとに env の Step を呼び、終端なら続けて Reset する（[env.cpp:712-742](../../core/anet-core/src/env.cpp)）。aux と `next_state` は Reset の前に結果へ入るのでリセット前の値、`continue_state` はリセット後の状態になる。通知の時点で env へ問い合わせる `GetScalar` と `GetTensor("rgb_frame")` は、終端の lane ではもうリセット後の env を読む。完了値が終端の通知で読めるのは、env がそれをリセットをまたいで持っているからである。
- 定期評価の `$env mean.hns57 @session_end` は、RunManager が宣言から鍵を集めて EvalSessionEnv へ購読として渡し、EvalSessionEnv が中の BatchEnv の Step のあとで完了した lane の `GetScalar` を溜める経路で読んでいる（[trainer.cpp:954-972](../../core/anet-core/src/trainer.cpp)、[env.cpp:336-369](../../core/anet-core/src/env.cpp)）。読むのはリセット後だが、読む鍵が完了値なので正しく読めている。
- 基盤 PRD [097](097_env_auto_reset_after_notification_10prd.md)（2026-10-20 改訂）の実装のあとも、`Step` の中の自動リセットは残る。遷移の結果（得点・残機・画面・完了の値）は Step の結果が答え、通知の読み手はそこから読む。env への問い合わせ（`GetScalar`・`GetTensor`）は今の env の状態を返し、終端の lane ではリセット後になる（097 の D1・D2）。
- `continue_state` は、Runner の `state_ = result->continue_state`（[trainer.cpp:295](../../core/anet-core/src/trainer.cpp)、同 566、同 700）のほか、エピソード構造の検査（[env.cpp:142-176](../../core/anet-core/src/env.cpp)）と、EvalSessionEnv がセッションの継ぎ目で Reset を省く処理（[env.cpp:317-327](../../core/anet-core/src/env.cpp)）で使われる。

### 3.4 メトリクスの配線と値の読み口

- メトリクスの配線は RunManager の構築処理の中にある（約 106 行、RunManager の約 24%）。scope を Runner へ解決する `resolve_runner`（dormant なら WARN して捨てる）、event 4 種ごとの attach、定義レコードの補完と `metrics.scalar.defs`・`metrics.trace.defs` の書き出し、Agent への購読ヒントである（[trainer.cpp:1010-1093](../../core/anet-core/src/trainer.cpp)）。`CreateEvalRunner` は何も配線しない（同 1147-1168）。PRD 081 が入ると、構築後に配線する最初の箇所になる。
- 値の読み口が 2 つある。scalar は `MetricsLogObserverBase::GetMetricsData`（lane を持たず、target を省くと探索順で探す。[observers.cpp:873-934](../../core/anet-core/src/observers.cpp)）、trace は `MetricsLogTraceObserver`（lane 付き・宣言順・値が無ければ fail-fast。同 831-870）で、target の switch を別々に持つ。step_result の aux と任意の aux キーを読む target は無い。`$action_info` は `anet::Module` の `GetScalar` を読むので、行動そのもの（`GetAction()`）は読めない。`$exp`（`BatchExperience`）の `GetScalar` は常に値を返さない（[rl.hpp:615](../../core/anet-core/include/anet/rl.hpp)）。
- trace の event は `@episode_end` に固定されている（[observers.cpp:1617](../../core/anet-core/src/observers.cpp)、同 1660、同 1666）。
- DSL の `$<名前>_step` は step 軸のトークンで、`$episode_step` はエピソード数の軸（`episode_count`）を指す（[observers.cpp:1359-1361](../../core/anet-core/src/observers.cpp)、[rl.hpp:108-118](../../core/anet-core/include/anet/rl.hpp)）。

### 3.5 Runner の時点

- 学習側の Runner は、`UpdateMetrics`（`reward`・`reward_ema`）とエピソードの集計・`EpisodeEndEvent` の通知を `TrainEvent` の前に行い、カウンタは通知のあとで数える（[trainer.cpp:536-567](../../core/anet-core/src/trainer.cpp)）。
- EvalRunner は、カウンタを通知の前に数え（[trainer.cpp:279-282](../../core/anet-core/src/trainer.cpp)）、`UpdateMetrics` はコメントアウトされていて呼ばれず（同 277）、エピソードの集計と `EpisodeEndEvent` の通知は `TrainEvent` のあとに、引数 `notify_episode_end` が真のときだけ行う（同 289-294）。定期評価のセッションはこれを偽で呼ぶ（同 334）。このため EvalRunner の `TrainEvent` の時点では、`$runner reward` は更新されず、`$runner` のエピソード集計は 1 step 前のままである。定期評価のセッションはこの集計を通らず、セッションの最後に `SetCompletedEpisodes` でまとめて確定する（同 334、同 366）。
- 今の設定で、eval の `$runner reward` を読む metrics は無い。eval scope の `@train` で読めるのは `$action_info` だけである（[140 §6](../design/140_observability.jp.md)、observers.cpp 1509-1513）。

### 3.6 表示・操作の現状（2026-10-22）

- 列は一律95pxで、内容に応じた幅になっていない。上部には文字の7ボタンがあり、再生と一時停止が別になっている（[EvalTracePanel.cpp](../../apps/runner/src/EvalTracePanel.cpp)、構築とInitialize）。
- 報酬行の背景は明るい緑、終端行は明るい橙、文字は黒の固定色で、ダークテーマのシステム色に合わせていない。同じファイルのTableが配色している。
- Trace中は追従paneの見出しに`Trace: step ...`を出す。Trace専用のヘッダ色変更はなく、上部の操作バーを囲む表示もない（同ファイル、RefreshCaptions）。メインツールバーには、システム文字色へ追従するアイコンがある（[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp)、MakeSystemTextSvg）。
- メインツールバーの Step（EvalPanel を 1 step 進める）のアイコンは、右向き三角の右に縦棒を置いた形である（同ファイル、`kStepSvg`）。動画のプレイヤーで一般的なコマ送りの形（縦棒の右に右向き三角）とは、順が逆である。

## 4. 確定事項

導入する機構とゴールの対応:

| 機構 | 寄与するゴール |
|---|---|
| 値の読み口の統一（D1） | G1。step trace（G2）も同じ読み口で列を読む |
| メトリクス配線の分離（D2） | G1。PRD 081 とシリーズの後続（step trace の記録）の置き場 |
| step trace と生成器、新しい target、列の宣言（D4・D5・D8・D9） | G2 |
| Agent の step 部品（D6）と EvalRunner の時点合わせ（D7） | G3、G2 |
| Evaluation Trace、追従 pane と Live / Trace モード、プレイヤー（D10〜D12） | G4 |

### 4.1 段階 1: 値の読み口の統一とメトリクス配線の分離

**D1 値の読み口を 1 つにする。** target（`$runner`・`$actor`・`$agent`・`$env`・`$exp`・`$update_result`・`$action_info`）と鍵と lane から値を読む部品を 1 つ作り、scalar の `GetMetricsData` と trace の `MetricsLogTraceObserver` はそれを使う。段階 2 の step trace も同じ部品を使い、新しい target（§4.2 D5）はこの部品に足す。
理由: step trace の列を読む口を別に作ると、target の解決が 3 つ目になる。

**D2 メトリクスの配線を RunManager から分ける。** scope の解決、dormant の WARN、event ごとの attach、定義レコードの補完と書き出し、購読ヒントを 1 つのクラスへ移し、RunManager はそれを持って構築時に呼ぶ。PRD 081 の追加分（dormant の trace の保持、`CreateEvalRunner` での後付けの結び付け、WARN の出し分け、定義の分岐）は、このクラスへ入る。
理由: 081 と、シリーズの後続（step trace の記録など）が同じ配線に乗るので、RunManager を太らせずに済む。本 PRD の段階 2 はこのクラスを通らない（列は metrics の定義と分けて宣言し、生成器は `TrainEvent` を直接受ける。D8・D9）。それでも本 PRD に置くのは、081 が構築後の結び付けを足す前に置き場を作り、段階 1 の整理として等価性で独立に受け入れるためである（2026-10-19 のユーザーの裁定）。Event / Observer の型の統一は別の関心として扱わない（NG5）。

**D3 段階 1 は振る舞いを変えない。** metrics の出力、定義レコード、WARN、fail-fast の文言と時機を変えない。受け入れは等価性で確かめる（§8 の受入 1）。
理由: 段階 1 は整理であり、段階 2 の前提。等価性で独立に受け入れられる形にする。

### 4.2 段階 2: step trace

**D4 step trace の単位と時点。** step trace は EvalRunner の完了した遷移 t の 1 件で、通知 1 回から 1 件を作る。列の値は、次のどの時点に属するかが決まっている。

| 時点 | target と鍵 | 中身 |
|---|---|---|
| 判断の時点（s_t。行動を選ぶ前） | `$action_info`（`action` を含む）、`$agent_step` のうちその step で決まる鍵（DQN 系では `q_sa`・`state_value`）、`$actor` | s_t で何を見て何を選んだか。強制した行動も反映される |
| 遷移のあと（a_t の結果、リセット前） | `$env_step`（`reward`・`done`・`truncated` と、Step の結果が答える鍵） | r_t、s_{t+1} が終端か打ち切りか、step 後の得点・残機・RAM（Atari の hns などの派生の鍵も） |
| 遷移の全体 | `$agent_step` のうちあとで決まる鍵（DQN 系では `approx_priority`） | 遷移 t の近似優先度。n step 後に決まる |
| 通知の時点のカウンタ・集計 | `$runner` | この step を数えたあとのカウンタ、この step の報酬、この step で完了したエピソードの集計（R1・R2 のあと） |

- 1 行の値はすべて同じ遷移に属する。「前の判断」は行 t−1 に、「次の判断」は行 t+1 にあるので、前後を指す target（`_prev`・`_next`）は作らない。
- `$agent`（学習側の統計。学習スレッドで step と関係なく変わり、EvalPanel が同期した複製のモデルとも一致しない）、`$exp`（scalar の鍵を持たない、§3.4）、`$update_result`（EvalRunner では空）、`$env`（今の env の状態を返し、終端の行ではリセット後の値になる。097 の D1）は step trace で読めない（構築時に fail-fast）。
- env の値は `$env_step` で読む。遷移の結果は Step の結果で運ぶので（097 の D1）、終端の行でもリセット前の値になる。`$env` を外しても、今 `$env` で読める遷移の結果（Atari の完了値や hns などの派生の鍵）は、097 で Step の結果が答えるので `$env_step` で読める（097 の D5）。LunarLander の風（`wind_x`・`wind_torque`）のように env の今の状態だけにある鍵は、その env の Step の結果が答えるようにするまで列にできない（2026-10-20 のユーザーの裁定。097 の §8 の申し送り）。
理由: 行の値がすべて同じ遷移に属すると保証できれば、表でも trace でも読み違えない。

**D5 新しい target と鍵。**
- `$env_step`: その step の Step の結果（lane ごと）。097 のメトリクスの `$env_step` と同じ target である（097 の §4.2 の 8）。鍵は `reward`・`done`・`truncated`（Step の結果の欄。done・truncated は 0/1）と、Step の結果が答える鍵（既定は env がその step の info（aux）に入れたスカラーの鍵。Atari は hns・`game_score.ge.[N]`・`ram_metric.[n]` などの派生の鍵も。097 の D5）。値はその遷移の時点の値（速報値）で、エピソードが終わった step ではエピソード単位の鍵が確定値になる（097 の D3・D6）。鍵が無い、またはスカラーでなければ、その鍵を最初に読んだ行で fail-fast（Step の結果が答える鍵は実行するまで分からない）。`reward`・`done`・`truncated` は、どの env でも Step の結果そのものが lane ごとの欄の値で答える鍵で、env の aux に同じ名前の鍵があっても読まない（env の Step の結果の型が lane ごとの問い合わせを欄へ回す。097 の途中の実装もこの順）。これは不正な設定を別の値へ置き換える挙動ではなく、target の契約が持つ予約の鍵である。env は aux にこの 3 つの名前を使わない（使うと読まれない）。
- `$agent_step`: Agent の step 部品（D6）が答える値。
  - どの鍵があり何を意味するかは部品が決める。共通の仕組み（列の解析、生成器、パネル）は鍵の名前を持たない。共通の側で鍵を決めると、共通の型と列の解析が DQN の鍵の名前を持つことになるためである（2026-10-21 のコードレビューで判明）。
  - DQN 系の部品が答える鍵は、`q_sa`（Q(s_t, a_t)）、`state_value`（max_a Q(s_t, a)。Munchausen では soft value）、`approx_priority`（あとで決まる）の 3 つ。
  - 部品が答えない鍵は、ほかの target と同じく、最初に読んだ行で fail-fast にする（D8 の実行時の fail-fast）。部品が知っている鍵でも、値がまだ無ければ NaN（空欄）になる。
  - 部品を出さない Agent では、`$agent_step` の列は、鍵が何であってもすべて NaN（空欄）にし、止めない。既定の列（D13、common.txt）をどの env・Agent でも使えるようにするためである。そのかわり、部品を出さない Agent では、鍵の書き間違いに気づけない。
- `$runner` の鍵 `step_in_episode`（新規、lane ごと）: この遷移を数えたあとの、今のエピソードの中の step 数（1 始まり。終端の行ではエピソードの長さ）。Runner がエピソードの集計に使っている lane ごとの step 数（`EpisodeStatsAccumulator`、[env.cpp:230-241](../../core/anet-core/src/env.cpp)）から読み、別のカウンタを持たない。`episode_step` という名前は、DSL でエピソード数の軸を指すので使わない（§3.4）。
- `$runner` のエピソード集計の鍵（`episode_return`・`episode_steps`）は、Runner の鍵の形のとおり集約の接頭辞つき（`mean.episode_return` など）で書く。step trace では、その step で完了したエピソードの値（完了が無ければ NaN）になる。集約の接頭辞を書けるのはこの鍵だけで、ほかの鍵と target では fail-fast（trace と同じく、step trace は lane ごとの値を扱うため）。
- `$action_info` の鍵 `action`: 実行した行動（強制した行動を含む）。離散なら `action`、連続なら次元ごとの `action.[<次元>]`。連続の行動に `action` を書いたら fail-fast で `action.[<次元>]` を案内する。どの Agent でも読める（ActionInfo の基底の型が、lane を指定した問い合わせに `GetAction()` の値で答えるので、Agent ごとの実装を要しない）。
- 新しい target の名前 `$env_step`・`$agent_step` は、DSL の step 軸のトークン（`$exp_step` など）と同じ `_step` で終わる。字句としては衝突せず（step 軸のトークンは決まった 6 つ、§3.4）、名前はユーザーの指定なので、このままにする（2026-10-19 の裁定）。
- 新しい target は列の宣言（D8）の解析と読み口（D1）に足す。metrics の DSL（`ParseMetricTokens` と trace の target の制限）へは、`$env_step` を 097 が足し（097 の §4.2 の 9）、`$agent_step` を step trace を trace チャネルへ出す PRD 081 の改訂で足す。
理由: 値の持ち主ごとに target を分けると、設定の階層がフレームワークの概念（Runner・Actor・Env・Agent）と一致し、PRD 081 の step trace でも同じ書き方で使える。遷移の時点の env の値は Step の結果にしか無い（§3.3、097 の D1）。

**D6 Agent の step 部品（近似優先度）。** Agent が step trace 用の部品を作って渡す。DQN 系（DefaultDQNAgent の DQN・QR・IQN、Munchausen を含む。同じ DQN の部品を使う RainbowAgent）だけが作り、ほかの Agent は作らない（`$agent_step` は空欄）。
- 部品と生成器（D9）の取り決め（どの Agent の部品にも共通）:
  - 部品は、step ごとに lane ごとの `action_info`・報酬・done・truncated を受け取る。
  - その step の値は、lane と鍵で 1 つずつ引ける形で返す。ほかの target と同じく、Module の `GetScalar(鍵, lane)` で読む。鍵の名前と値の型は部品の中に置き、共通の型には置かない。
  - その step の行で、あとで決まる鍵を lane ごとに示す。「値がまだ決まっていない」と「決まったが値が無い（NaN）」は別のものとして扱う。
  - あとで決まると示した鍵も、その step の値の中では、知っている鍵として NaN で答える。
  - 値が決まったら、「どの lane の何行前の行か」「決まった鍵」「その値」を添えて返す。打ち切りなどで決まらないと分かったときも、NaN で決まったものとして返す。あとで決まると示した鍵は、必ずどちらかの形で返す。
- DQN 系の部品は、`q_sa`・`state_value` をその step のうちに返す。`approx_priority` は、あとで決まる鍵として示し、n step 後に返す。DQN の鍵の名前と値の型は、DQN の実装の中だけに置く。
- Actor Q ヒントは、Actor と同じ共有のヘルパーで aux の `q_values` から作る（`Actor::MakeAction` の該当部分をヘルパーへ切り出し、Actor もそれを呼ぶ）。`actor_approx` の構成で `action_info` に既にヒントがあっても使わず、常にヘルパーで作り直す（gather と max が 1 回ずつで軽く、構成によって値の出どころを変えないため）。n step の組み立ては replay と同じ処理を使い、式は `DqnInitialPriorityEstimator` を使う。GUI に式を書かない。Actor の設定キーは足さない。
- 使う設定は Learner の γ・実効 n・per_eps・clip・TBO・Munchausen。PER の有無と `per_initial_priority_mode` によらず計算する（見たいのは TD の大きさで、PER が無くても意味がある）。
- 報酬には、学習側と同じ reward scaler の変換（倍率とクリップ）を、統計を更新せずに掛ける。RunningStd の scaler では、replay に入った報酬は Push の時点の倍率なので、今の倍率とは違いうる。
- n step の途中で終端になったら短い長さで確定する。打ち切りが窓にかかる遷移は空欄（replay は最大優先度へ逃がすが、その値はパネルに無い）。エピソードの始まりで窓を空にする。
- Q は eval_panel の Actor の出力（Greedy、EvalPanel が同期した複製のモデル）で計算する。eval_panel の Actor のネットワークは既定で target なので（§3.2）、学習側の近似優先度（online の Q）とは値の出どころが違う。既定は変えず（EvalPanel のデモの方策を変えないため。2026-10-19 のユーザーの裁定）、学習側と同じ online の Q で見たいときは次の 1 行で切り替える。追加の forward をしない方針（ADR 0010）なので、step 部品がネットワークを選ぶことはしない。
  ```
  DefaultDQNAgent.@baseline.actor.[eval_panel].network = online
  ```
- 学習側の Actor が UQE なら、そのほかにも系統差が出る（ADR 0010）。n step のあいだにモデルの同期があれば、開始と bootstrap で別のモデルの値になる。
理由: 設定キーが増えず、式・ヒント・n step の組み立てが DQN の module に 1 つずつにまとまり、step trace の生成器はアルゴリズムを知らずに済む（[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)）。学習側の値は変えない（§8 の受入 3）。

**D7 EvalRunner の時点合わせ。**
- R1: `EvalRunner::DoStepInternal` で、通知の前に `UpdateMetrics` を呼ぶ（`$runner reward`・`reward_ema` がその step の値になる）。
- R2: エピソードの集計と `EpisodeEndEvent` の通知を `TrainEvent` の前に行う（学習側の Runner と同じ順。`notify_episode_end` が真の経路だけ）。
- カウンタを数える時機（EvalRunner は通知の前、学習側は通知のあと）は変えない。そろえると、eval scope の `@train $action_info` の metrics の step が 1 つずれる（DropMerge の設定）。step trace の `$runner exp_step` は「この step を数えたあとの数」と定義する。
理由: R1・R2 で `$runner` のすべての鍵が行の時点に合う。R1 は定期評価のセッションでも走るが（セッションも `DoStepInternal` を通る、[trainer.cpp:334](../../core/anet-core/src/trainer.cpp)）、eval scope で `$runner` を読む metrics は無く、セッションの集計は `SetCompletedEpisodes` で確定するので（同 366）、既存の metrics の出力は変わらない。R2 はセッションでは通らない（`notify_episode_end` が偽、§3.5）。

**D8 列の宣言。** 列は metrics の定義とは分けて、Evaluation Trace の設定に番号付きで 1 行 1 列で書く。

```
app.eval_trace_panel.columns.[<番号>] = $<target> <鍵> [label:<列ラベル>] [display:<表示>]
```

- 番号は 1 以上の整数で、飛び番を許し、小さい順に左から並べる。既定の列も含めて、すべての列をこの形で選ぶ。
- 値のトークンは target 1 つ、鍵 1 つ、任意の `label:`・`display:` だけ。event・scope・step 軸・EMA・`interval:` は書けない。集約の接頭辞（`mean.` など）は `$runner` のエピソード集計の鍵だけに書ける（D5）。
- 解析は列の宣言の専用のもので、字句の分類だけを metrics の DSL と共有する。
- `label:` は列の見出し（空白を含まない、1 文字以上）。省くと鍵が見出しになる。見出しの重複は許す。
- 1 行に同じ属性（`label:`・`display:`）は 1 回だけ書ける（trace の宣言で指定の重複を拒むのと同じ。[140 の trace チャネルの節](../design/140_observability.jp.md)）。
- `display:` は表示の仕方。行動の列（`$action_info action`）は `value`（番号）・`name`（EnvSpec の `action_spec.value_labels` の名前）・`value_name`（`3:UPFIRE`。既定）。数値の列は `value`（既定）・`pct`（直近の値の中の百分位）。連続行動の `action.[<次元>]` は `value` だけ。行の値は数値のままで、名前は表示だけで置き換える。離散の行動では、行動の数が `value_labels` の数そのものなので（[rl.hpp:269-273](../../core/anet-core/include/anet/rl.hpp)）、名前の無い行動は無く、代わりの表示も置かない。連続行動の env では既定の列の `$action_info action` が構築時に止まるので、その env の設定で列を上書きする（PPO に対応するときに見直す）。
- 構築時の fail-fast: 番号が 1 以上の整数でない、target が無い・2 つ以上、鍵が 1 つでない、未知の target や step trace で読めない target（`$agent`・`$exp`・`$update_result`・`$env`）、`$runner` のエピソード集計の鍵以外への集約の接頭辞、未知のトークンや属性、同じ属性の重複、空の `label:`、`display:` の値と列の組み合わせが合わない（連続行動に `name`・`value_name` を含む）。
- 実行時の fail-fast: 読み手が値を返さない鍵（未知の鍵）は、最初に読んだ行で止める（trace と同じ規則）。値が NaN・±Inf なら空欄。部品を出さない Agent の `$agent_step` は、読み手が無いので止めずに空欄にする（D5）。
理由: 記録の宣言と切り離して見られ、本 PRD だけで完結し、GUI がメトリクスの配線に依存しない（[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)）。表示上の鍵は列番号なので、番号を設定の鍵にする。連続行動（PPO）に備えて、`display:` の値は番号を前提にしない名前にした。

**D9 step trace の生成器。** core に、EvalRunner の `TrainEvent` から step trace を作る生成器を置く。宣言（D8）を解析し、読み口（D1）で列を読み、Agent の step 部品（D6）を lane ごとに呼ぶ。
- 生成器は、列ごとに値が決まったかを持つ。待つのは、部品があとで決まると示した鍵（D6）の列だけである。NaN で決まった列も、そこで待つのをやめる。鍵の名前で待つ列を決める処理は置かない。
- 生成器は 2 つのことを知らせる。「step trace を足した」と「列が決まった」である。
  - 「列が決まった」は、あとで決まる列が決まるたびに、その行と、新たに決まった列の位置（列の並びの中の位置）を添えて知らせる。1 回の知らせで全部の列が決まるとは限らない。
  - 行のすべての列が決まると、その行は「決まった」状態になる。あとで決まる列が無い行も、足した直後に、決まった列なしで 1 回知らせる。
  - パネルは、前者で行を足し、後者で決まった列を埋める。PRD 081 の trace の出口は、後者のうち、行が「決まった」状態になった知らせでだけ 1 行を書く。途中の知らせでは書かない。
- 色付けのため、報酬・done・truncated は宣言の有無にかかわらず持つ。
- UI スレッドで同期に動き、ロックを持たない（§3.1）。
理由: 表と trace が同じ生成器を使えば、シリーズの後続は列を足すだけで済む。

**D10 Evaluation Trace パネル。**
- 1 つだけ作る pane で、View メニューとツールバーから表示・非表示を切り替える。置き場は補助 pane（Conv2d・HeatMap）と同じ右端の列（`AddAuxPane` の列）だが、補助 pane のように何個も作らない（Trace モードを動かす表を 1 つに決めるため）。既定は非表示。クラスは `EvalTracePanel`、表示名は「Evaluation Trace」。GUI の表示は英語。
- 表は新しい行が上。持つ行数は `app.eval_trace_panel.max_rows`（既定 1000、1 以上の整数。整数に直せない値と範囲外は構築時に fail-fast）。Atari の 14.5 step/s で約 70 秒分で、ボス戦の 1 回が入る。
- メモリの見積もり（Atari、1000 行）: 1 行あたり、EvalPanel の View のスナップショット（210×160 の RGB 画面と 84×84 の観測で約 120〜220 KB）、QValuePanel の値（数 KB）、Conv2d を開いていれば活性（Nature CNN で約 85 KB）。合わせて約 130〜300 MB。表示している間は Live でも持つ。
- 数値: 値が整数（小数部が 0）なら整数で、そうでなければ有効数字 4 桁で出す。NaN は空欄、done・truncated は 0/1（論理値の表示をそろえ、空欄を「値が無い」だけに使うため。得点のような大きな整数を丸めないため）。
- 自動調整する列の幅は見出しと実際に表示する文字列が、標準のセル余白を含めて収まる必要な幅とし、短い列を一律に広くしない。数値の書式は上記のまま、行動名はD8のdisplay指定に従う。初期は見出しに合う幅とし、新規行・遅れて確定した値で長い文字列が現れた列だけ自動拡張する。同じ表示中は行の入替えで自動縮小しない。
- ユーザーが幅を調整した列は、その表示中は自動拡張もしない。パネルを閉じて開き直すと、手動指定を含めて初期幅へ戻す。幅・配色・アイコンはフォント・DPI・テーマに追従する。全列の合計が表示領域を超えるときは横スクロールを使い、収めるために文字をつぶさない。
- パネルの最小幅は、6ボタン・スライダー・速度表示を含む操作バー全体の必要幅と、左端から最初の行動列までの現在の列幅の合計（標準余白・縦スクロールバーを含む）の大きい方とする。行動列が無ければ操作バーを基準とする。連続行動は最初の次元別行動列までとする。列幅・フォント・DPI・テーマの変更に合わせて更新し、ドック中・フロート中ともそれ未満へ縮めない。残りの列は横スクロールで読む。
- 性能の見積もり: 列の幅を決める処理は、新しい行の表示値 C 件と、あとで決まった表示値 K 件だけを対象にする（表示する 1 step あたり C+K 件の幅の判定）。持っている全行を step ごとに測り直さず、列を広げるのは、それまでより長い値が出たときだけにする。学習の step の順序と worker の構成は変えない。
  - 頻度×単価: [common.txt](../../apps/runner/config/common.txt) の既定 10 列と [Atari.txt](../../apps/runner/config/Atari.txt) の 2 列から C=12、通常の 1 行の追加と 1 行の遅れた確定を代表として K=2（prio・prio_pct）と置く。1 件の処理（文字の幅の判定と、要るときの列幅の更新）の単価を 0.01〜0.10 ms と仮定すると、増える時間は 14 件 × 0.01〜0.10 ms = 0.14〜1.40 ms/step である。この単価は実測ではなく、方式を評価するための仮定である。
  - Atari の 14.5 step/s では 1 step が約 68.97 ms なので、上限の 1.40 ms は時間で約 2.03%、速度で約 98.0% に当たる。受入 8 の 95% 以上（許す低下 5%）に対し、その半分の 2.5% を下回る。列が多いとき、終端でまとめて確定するとき、より速い再生元のときは、同じ比率にならない。実装では、列の数・確定の件数・表示の速さでこの見積もりを確かめ直し、低下が 2.5% を超える見込みなら、方式を決める前に UI だけの試作で測る。実装のあとは、受入 8 の表示の速さと UI の受入で確かめる。
- 通常セル・見出し・文字・罫線・選択はシステム色を基本にする。報酬が0でない行は緑系、doneかtruncatedの行は橙系の薄い強調を標準背景へ重ね、ライト／ダークの双方で文字を読めるようにする。両方に当たるときは終端の橙系を優先する。選択中は行の強調よりシステム選択色・選択文字色を優先する。色付けの有無の設定は追加しない（今回合意したのは色の調整であり、表示切替の操作要件は追加しない）。
- `display:pct` の百分位は、その列の直近 `app.eval_trace_panel.pct_window` 個（既定 10000、1 以上の整数。違反は構築時に fail-fast）の値の中での位置（0〜100、NaN を除く）。表の行数より長い窓で、ゲーム数本分の分布の中の位置を出す。百分位は、列の値が決まったときに 1 回だけ計算する（近似優先度のようにあとで決まる値は、決まったとき）。表の上限で消えた行の値も、決まったときに窓へ入れる。
- パネルを閉じている間も購読は付けたままにし、非表示なら何もしない（step trace もスナップショットも作らず、Agent の step 部品も呼ばない）。開き直したら、表、`pct_window` の値、Agent の step 部品の n step の窓をすべて空から始める。
理由: `AttachScoped` の Detach は効かず、走行中の attach と detach は学習スレッドの通知と競合しうる（§3.1）。列幅は必要なときだけ広げ、読んでいる途中の列位置の揺れと保持済み全行の再測定を避ける（上の性能の見積もり）。配色は固定の明色背景・黒文字をやめ、テーマに合う文字の読みやすさと報酬／終端の手がかりを両立する。

**D11 Live モードと Trace モード（追従 pane）。**
- 鏡写しインスタンス（EvalPanel）の step から描く pane（EvalPanel の View、QValuePanel、Conv2dPanel）を**追従 pane** とし、共通の口を持たせる。口は「step の通知から表示用のスナップショットを作る」「渡されたスナップショットを出す」「Live に戻る」の 3 つ。
- Evaluation Trace は、開いている追従 pane の分だけ、step trace ごとにスナップショットを持つ。QValuePanel と EvalPanel の View は、今も step ごとにスナップショットを作っているので（§3.1）、それを共有して持ち、計算は増えない。Conv2d は活性（`nn_trace/*`）の CPU コピーを持ち、選ばれたときに描く（画像のままでは 1 行 0.7 MB を超えるので持たない）。追従 pane を開く前の行には、その pane の「no data for this step」を出す。
- 行を選ぶ（クリック・上下キー・スライダー・再生）と、追従 pane は **Trace モード**に入り、その step trace のスナップショットを出し続ける。見出しに `Trace: step <番号>` を出す。行 t の EvalPanel は遷移のあとの画面 s_{t+1} を、QValuePanel は判断の時点の Q(s_t, ·) を出す（今のライブの表示と同じ組み合わせ）。判断の入力 s_t の画面は行 t−1 で見られる。ただし、エピソードの最初の行の判断の入力 s_0（リセット直後の画面）は、どの行にも入らない。基盤 PRD 097 で、リセット後の情報は Runner だけが受け取り、Observer には知らせないと決めたためである（097 の D4）。終端の行の画面は、097 の実装のあとはリセット前の終端の画面になる（097 の D1）。
- Traceモード中は、EvalTracePanel上部のボタン・スライダー・速度表示を含む操作バー全体を2 DIPの赤枠で囲む。再生中・一時停止中・手動の行選択で共通に表示し、Liveへ戻ると消す。枠の有無で部品の位置を動かさない。見出しの`Trace: step <番号>`と通常のシステムによるactive／inactive表示は保ち、列見出しはLive／Traceと行選択の有無で同じシステム配色を保ち、行選択に伴う見出しの強調もしない。選択行のシステム選択色は保つ。操作する場所でTrace中と分かるようにするためである。
- Trace モードに入ったら、EvalPanel を一時停止する（調べている間に、選んでいる行が上限で消えないようにするため）。Live に戻っても再開はしない。再開は EvalPanel の Resume で行い、それが Live に戻るきっかけにもなる（2026-10-19 のユーザーの裁定）。
- **Live モード**に戻るきっかけは、停止ボタン（D12）、選択の解除（表にフォーカスがあるときの Esc キー。行の無いところのクリックは解除にしない。押し間違いで Live に戻らないため。2026-10-19 のユーザーの裁定）、手動の行動（QValuePanel のクリック、キー、ツールバーの Step）、EvalPanel の Resume、Evaluation Trace を閉じること（閉じた表で Trace のまま残らないように）。手動の行動は Live に戻してから行う（行動は今の状態に当たるため）。Live に戻ると再生も止まる。
- Trace の間に runner が進むことは、上の一時停止でほぼ無くなる。それでも step trace が増えたときは、選んだ行と表の表示位置を動かさず、表の上限で選んでいる行が消えたら、残っている中でいちばん古い行へ移る。
- 上下キーは、フォーカスのある窓が受ける。表にフォーカスがあれば行を動かし、EvalPanel にフォーカスがあれば今までどおり行動になる（EvalPanel から転送されるキー、[RunnerFrame.cpp:993-994](../../apps/runner/src/RunnerFrame.cpp)）。
- SweepHeatMapPanel は追従 pane ではない（学習側のモデルの地図で、EvalPanel の step と関係しない）。Trace の間も今のまま更新する。
理由: 既存のスナップショットをそのまま使え、どの env の View でも同じ口で動き、シリーズの後続で足す pane も同じ口で追従できる。先頭の行は今のライブの表示と同じなので、まだ行動の決まっていない「現在の状態」の仮の行は要らない。

**D12 プレイヤー。** Evaluation Traceの上部に、左から「再生／一時停止・停止・コマ戻し・コマ送り・速度減少・速度増加」の6個のアイコンボタン、スライダー、速度の表示（`x1`など）を置く。
- 再生／一時停止は1個のボタンとし、再生中は一時停止のアイコンとツールチップ`Pause`、LiveまたはTraceの一時停止中は再生のアイコンと`Play`を出す。押すと現在の状態に応じた操作を行い、最新行への到達などで再生が止まった場合も表示を同期する。
- ボタンには文字を並べず、次の一般的な動画プレイヤーの形を使う。メインツールバーと基準サイズ・描画の調子をそろえ、システム文字色と無効時の表現を使う。テーマ変更時も追従させる。ツールチップは英語とする。

| 操作 | アイコンの形 | ツールチップ |
|---|---|---|
| 再生／一時停止 | 右向き三角／二本の縦線 | `Play`／`Pause` |
| 停止 | 四角 | `Stop and return to Live` |
| コマ戻し | 左向き三角の右に縦棒（◀❙） | `Step back` |
| コマ送り | 縦棒の右に右向き三角（❙▶） | `Step forward` |
| 速度減少 | 左向きの二重三角 | `Slower` |
| 速度増加 | 右向きの二重三角 | `Faster` |

コマは表の 1 行を指す。メインツールバーの Step（EvalPanel を 1 step 進める）のアイコンも、コマ送りと同じ「縦棒の右に右向き三角」に直す（今は順が逆、§3.6）。
- 再生: Live のときはいちばん古い行から（Trace モードに入るので EvalPanel も一時停止する、D11）、Trace で一時停止しているときは選んでいる行から、新しい側へ 1 行ずつ自動で送る。いちばん新しい行で一時停止しているときは、いちばん古い行から再生する。
- いちばん新しい行に追いついたら一時停止する（Trace のまま）。Trace の間は EvalPanel も一時停止しているので（D11）、追いつく先は持っている最後の行である。
- 一時停止: 今の行で止める（Trace のまま）。停止: 再生を止めて Live に戻る。
- コマ送り: 1 行新しい側へ進めて一時停止する。コマ戻し: 1 行古い側へ戻して一時停止する。表にフォーカスがあれば、キーの上下でも動ける（D11）。プレイヤーのコマ送りは表の行を送る操作で、EvalPanel の Step（runner を進める）とは別である。
- 速度: 1x は EvalPanel が実際に進む速さ（fps × step_per_frame、Atari は 14.5 step/s。View の FPS の変更に追随する）。段階は 1/8・1/4・1/2・1・2・4・8 倍で、ボタン 1 回で 2 倍か半分。端でさらに押しても変わらない。表示は `x0.125`・`x0.25`・`x0.5`・`x1`・`x2`・`x4`・`x8` とし、分数表記は使わない。既定は 1x。設定には残さない（実行時 UI 操作）。
- スライダー: 範囲は持っている行（いちばん古い行からいちばん新しい行まで）、位置は選んでいる行。動かすと Trace に入ってその行を出す。再生中に動かしたら、その位置から再生を続ける。
- ボタンを押すたびに、ステータスバーの通知欄（field 0）に操作を出す。文言は `Trace: play`・`Trace: pause`・`Trace: stop (live)`・`Trace: step`・`Trace: step back`・`Trace: speed x2`（端では `Trace: speed x8 (max)`・`Trace: speed x0.125 (min)`）・`Trace: end`（再生が追いついて止まったとき、いちばん新しい行でのコマ送り）・`Trace: start`（いちばん古い行でのコマ戻し）。ログには出さない（run に影響しない表示の操作のため）。
- Live の画面はいちばん新しい行と同じなので（D11）、Live でのコマ送り・コマ戻しは、いちばん新しい行を今の位置として扱う。どちらも Trace モードに入り、EvalPanel を一時停止する。
  - コマ送り: いちばん新しい行を選ぶ。それより新しい行は無いので、いちばん新しい行でのコマ送りと同じく、通知欄に `Trace: end` を出す。
  - コマ戻し: いちばん新しい行の 1 つ前の行を選び、通知欄に `Trace: step back` を出す。行が 1 件だけのときは、その行を選び、いちばん古い行でのコマ戻しと同じく `Trace: start` を出す。
  - Live 中の再生／一時停止ボタンは再生操作なので、上記の再生の規則に従う。独立した一時停止ボタンは置かない。
- いちばん新しい行でのコマ送りと、いちばん古い行でのコマ戻しは、回らずにその行に留まり、通知欄にそれぞれ `Trace: end`・`Trace: start` を出す（動画のコマ送りと同じく端で止まる。回るのは、いちばん新しい行で一時停止しているときの再生だけ）。
- 表に行が 1 件も無い間（開いた直後、開き直した直後）は、プレイヤーのボタンとスライダーを無効にする（押せない）。
（2026-10-19 のユーザーの裁定。Live の箇条〔再生／一時停止ボタンの扱い、コマ送りの通知、コマ戻しで選ぶ行と通知（行が 1 件のときの `Trace: start` を含む）〕とアイコンは、2026-10-22 に改めた）
- 再生の制御（今の行、速度、端の扱い、行の消失）は GUI に依らない部品にし、単体テストで確かめる。
理由: 履歴を 1 行ずつ手で選ぶのはつらく、時系列で自動に送って見たい。停止を Live に寄せるとボタンが減り、Live と Trace の行き来が「選ぶ・再生する」と「停止」の対になる。最後まで再生したら止まり、その位置で見直せる。

**D13 既定の設定。**
- 行動と結果を隣り合わせで読むため、既定列はstep、ep_step、action、reward、Q_sa、V_s、done、trunc、prio、prio_pctの順とする。
- 既定の`truncated`列は`label:trunc`で短く表示する。取得キー・値・設定文法は変えない。

```
# common.txt（行動と報酬を隣接させ、Q・終端・優先度を続ける。env 固有の列は 20 番から）
app.eval_trace_panel.max_rows ?= 1000
app.eval_trace_panel.pct_window ?= 10000
app.eval_trace_panel.columns.[1] ?= $runner exp_step label:step
app.eval_trace_panel.columns.[2] ?= $runner step_in_episode label:ep_step
app.eval_trace_panel.columns.[3] ?= $action_info action
app.eval_trace_panel.columns.[4] ?= $env_step reward
app.eval_trace_panel.columns.[5] ?= $agent_step q_sa label:Q_sa
app.eval_trace_panel.columns.[6] ?= $agent_step state_value label:V_s
app.eval_trace_panel.columns.[7] ?= $env_step done
app.eval_trace_panel.columns.[8] ?= $env_step truncated label:trunc
app.eval_trace_panel.columns.[9] ?= $agent_step approx_priority label:prio
app.eval_trace_panel.columns.[10] ?= $agent_step approx_priority display:pct label:prio_pct

# Atari.txt
app.eval_trace_panel.columns.[20] = $env_step game_score
app.eval_trace_panel.columns.[21] = $env_step lives
```

理由: KFM のボス撃破・出口の換算は、学習報酬（クリップ後 +1）では雑魚と区別できず、途中の得点の増え方で見分ける（§2）。`Q_sa`／`V_s`は短く対応のそろった表示名であり、既存の`label:`だけで指定する。取得キー`q_sa`・`state_value`、値とその意味、設定文法は変えず、GUIにDQN固有の鍵判定を足さない。ユーザーが別のlabelを指定した列はその指定に従う。

### 4.3 規約・ADR・コードから導いたこと

1. 未知の鍵は fail-fast、NaN・±Inf は空欄（trace の規則。根拠: [140 の trace チャネルの節](../design/140_observability.jp.md)、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)）。
2. n step の途中の終端は短い長さで確定し、エピソードの始まりで窓を空にする（根拠: [replay_buffer_impl.cpp:150-178](../../core/anet-core/src/replay_buffer_impl.cpp)）。
3. 打ち切りが窓にかかる行の近似優先度は空欄（根拠: [replay_buffer_impl.cpp:393-398](../../core/anet-core/src/replay_buffer_impl.cpp)、[ADR 0012](../adr/0012-replay-initial-priority-hint-completion.md)）。
4. 実効の n は `use_n_step ? n_step : 1`、割引は γ の実 n 乗（根拠: [dqn_based_agent.cpp:2680](../../core/anet-core/src/dqn_based_agent.cpp) 付近、[replay_buffer_impl.cpp:421-428](../../core/anet-core/src/replay_buffer_impl.cpp)）。
5. 報酬には学習側と同じ reward scaler の変換を、統計を更新せずに掛ける（根拠: [PRD 035](done/035_approx_actor_priority_per_10prd.md) の 29 行目、[scaler_impl.cpp:626-635](../../core/anet-core/src/scaler_impl.cpp)）。
6. Q は eval_panel の Actor の出力で計算し、UQE の系統差を許す（根拠: [ADR 0010](../adr/0010-actor-priority-mean-q-approx.md)）。
7. step 単位の trace の event を開くのは PRD 081 で、そのとき新しい ADR を起こし ADR 0037 の先頭に注記する（根拠: [ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md) の「トリガは `@episode_end` のみ。要るときに開ける」）。
8. 設定キーは `app.eval_trace_panel.*`（根拠: [RunnerApp.cpp:75](../../apps/runner/src/RunnerApp.cpp) の `app.<パネル>_panel` の並び）。
9. step trace は UI スレッドで同期に作り、ロックを持たない（根拠: [EvalPanel.cpp:243-267](../../apps/runner/src/EvalPanel.cpp)、[rl.cpp:1110-1117](../../core/anet-core/src/rl.cpp)）。

### 4.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `MetricValueReader` | 新規 | target・鍵・lane と通知から値を読む共通の読み口。scalar・trace・step trace が使う | `metrics` |
| `RunManager::MetricsWiring` | 新規（RunManager の内側のクラス） | RunManager から分けたメトリクス配線（scope の解決、dormant の WARN、attach、定義レコード、購読ヒント） | `trainer` |
| `RunManager` | 変更 | メトリクスの配線を内側の `MetricsWiring` へ任せる | `trainer` |
| `MetricsLogObserverBase`・`MetricsLogTraceObserver` | 変更 | target の解決を `MetricValueReader` に任せる | `observers` |
| `BatchActionInfo` | 変更 | Module になる。lane を指定した `action`・`action.[<次元>]` の問い合わせに、実行した行動で答える | `rl` |
| `StepTraceColumnSpec` | 新規 | 列の宣言（番号・target・鍵・label・display）の解析と構築時の検査 | `metrics` |
| `StepTrace` | 新規 | step trace 1 件（列の値と列ごとに決まったか、報酬・done・truncated） | `metrics` |
| `StepTraceBuilder` | 新規 | EvalRunner の `TrainEvent` から step trace を作り、あとで決まる列を埋め、「足した」「列が決まった」を知らせる。鍵の名前で処理を分けない | `metrics` |
| `AgentStepDiagnosticsProvider` | 新規 | Agent の step 部品の interface（lane ごとの step の入力、その step の値、あとで決まる鍵、決まった値）。鍵の名前と値の型は持たない | `rl` |
| `Agent` | 変更 | step 部品を作る口 `CreateStepDiagnosticsProvider` を足す（既定は作らない） | `rl` |
| `DqnAgentStepDiagnosticsProvider` | 新規 | DQN 系の step 部品。ヒントの共有ヘルパー、replay と同じ n step の処理、`DqnInitialPriorityEstimator` で値を出す。DQN の鍵（`q_sa`・`state_value`・`approx_priority`）と値の型はこの中に置く | `dqn_based_agent` |
| Actor Q ヒントの共有ヘルパー | 変更 | `Actor::MakeAction` から切り出し、Actor と `DqnAgentStepDiagnosticsProvider` が使う | `dqn_based_agent` |
| n step の組み立て（`NStepQueueController`・`DefaultExperienceBuilder`） | 変更 | replay と step 部品が共有できる置き場へ移す | `replay_buffer_impl` |
| `RunnerBase` | 変更 | `step_in_episode` を返す（`EpisodeStatsAccumulator` の lane ごとの step 数と、終端の行では完了したエピソードの長さから） | `trainer` |
| `EpisodeStatsAccumulator` | 変更 | lane ごとの今の step 数を読めるようにする | `env` |
| `EvalRunner` | 変更 | R1・R2（通知の前の `UpdateMetrics` と、エピソード集計の順） | `trainer` |
| `EvalTracePanel` | 新規（初回）／変更（2026-10-22 のUI調整） | Evaluation Trace の pane。列幅・最小幅・テーマ配色・アイコンと再生状態・操作バーの赤枠、百分位、スナップショットの保持、Live / Traceの切り替え | `EvalTracePanel` |
| `TracePlayer` | 新規（GUI の機能グループ） | 再生の制御（今の行、速度、端の扱い、行の消失）。wx の画面の部品に依らない。2026-10-22 のUI調整ではこの再生動作を維持する（Live でのコマ送り・コマ戻しの扱いは D12 のとおり改めた） | `gui` |
| `TraceFollower` | 新規 | 追従 pane の共通の口（スナップショットを作る `CreateTraceSnapshot`、渡されたものを出す、Live に戻る） | `gui` |
| `ViewBase`（gui.hpp） | 変更 | View のスナップショットを追従 pane の口で渡し、出せるようにする | `gui` |
| `EvalPanel`・`QValuePanel`・`Conv2dPanel` | 変更 | 追従 pane として実装する | `EvalPanel・QValuePanel・Conv2dPanel` |
| `RunnerFrame` | 変更 | Evaluation Trace の出し入れ、Live に戻すきっかけ（Resume・手動の行動・Evaluation Trace を閉じる）の配線、メインツールバーの Step のアイコンの形（D12） | `RunnerFrame` |

## 5. User Stories

1. 研究者として、EvalPanel で遊ばせているゲームの直近の遷移を、行動・報酬・Q・近似優先度・env の途中の値を並べた表で見たい。
2. 研究者として、表の行を選んで、その時点の EvalPanel・QValuePanel・Conv2d の表示（画面、Q、活性）を見たい。
3. 研究者として、選んだ時点から時系列で自動に送って、場面の流れを見たい。
4. 研究者として、KFM のボス撃破や出口の換算の step で、近似優先度がふつうの step より高いかを確かめたい。
5. 研究者として、env ごとに見たい値（Atari の得点・残機・RAM）を、設定で列に足したい。

## 6. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| 学習側の優先度（online と target の 2 つのネットワーク、分位点の平均の差）の列 | 不採用 | 学習側の優先度は、replay から引いた遷移について online と target の両方の forward で決まる。EvalPanel は 1 つのネットワークの forward しか持たず、replay も無い。近似の精度は `actor_approx` の Run の順位相関で確かめる |
| 定期評価・学習側の step trace | 不採用 | 量が大きすぎる（§3.1）。Run 全体の集計は 999_replay_eviction_sample_count の側 |
| step trace を trace チャネルへ出す | PRD 081 へ移す | 鏡写しインスタンスを trace の購読先にする 081 と同じ配線の関心。081 を改訂して取り込む（§7 の 2） |
| Agent の入力（raw_obs）の表示 | 先送り | 84×84 の Tensor を半端に描いても読みにくく、画面は EvalPanel の Trace モードで見られる。要ると分かったら、選んだ行の Agent の入力を文字列でログへ出す形から起こす |
| 列の値の折れ線（未使用の `PlotPanel` の流用） | 先送り | まず表と再生で足りるかを見る。近似優先度の推移を一目で見たくなったときに起こす |
| 条件に合う行で自動で一時停止する | 先送り | 再生で場面を探す手間が大きいと分かったときに起こす |
| Event / Observer の型の統一 | 別 PRD | 別の関心（NG5）。新しい Event の型を足す変更が出たときに起票する |

## 7. 未決と依存

| # | 事項 | 扱い |
|---|---|---|
| 1 | BatchEnv の自動リセットと通知の時機 | 基盤 PRD [097_env_auto_reset_after_notification](097_env_auto_reset_after_notification_10prd.md) は、2026-10-19 に M2 に決めたあと、2026-10-20 の再グリルで改めた（[ADR 0060](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)）。遷移の結果は Step の結果で運び、`$env` は今の env の状態を返す。BatchEnv は Reset と Step のまま（097 の D1・D2）。リセット後の情報は Runner だけが受け取る（097 の D4、変わらない）。`$env_step` は、097 の Step の結果が答える鍵（既定は aux のスカラー、Atari は派生の鍵も）を読む、メトリクスと同じ target になる（097 の §4.2 の 8）。097 を本 PRD より先に実装する。step trace の列の `$env` は、097 の §8 の申し送り（ADR 0057 の「時点が決まっている target だけを読む」）どおり列の target から外し、遷移の結果は `$env_step` で読む（2026-10-20 のユーザーの裁定、D4） |
| 2 | step trace を trace チャネルへ出す段 | PRD 081 を改訂して取り込む（081 の冒頭に 2026-10-19 に追記済み。改訂は 081 のグリルで行う）。081 の実装ノートが trainer.cpp を名指ししている箇所は、`MetricsWiring` を指すよう直す。改訂で決めること: trace 行の step 座標（step trace は `TrainEvent` から作るので、`@train` 系の規則なら EvalPanel 自身の counts になり、`54_evalpanel/episode` の学習側 counts と座標が違う。ADR 0047、CONTEXT.md「学習側 counts」「step座標系」）、metrics の DSL への `$agent_step` の追加（`$env_step` は 097 が足す）、あとで決まる列を待って 1 行を書くこと（生成器の知らせ方は D9） |

## 8. 成功の測り方と受入基準

成功の測り方:

- KFM の EvalPanel（ボス行きに入った seed 3 のチェックポイント、または現行の Run。学習側の近似と同じ online の Q で見るときは D6 の 1 行を足す）で、ボス撃破の step、出口の換算の step、ふつうの雑魚を倒した step を、`$env_step game_score` の増え方と RAM の列で表から見つけ、それぞれの近似優先度と百分位を読み取れる。Trace モードで、その時点の画面と Q を確かめられる。
- 近似の確からしさ: `actor_approx` で回した Run で `39_agent_per/01_actor_learner_rank_corr` を確かめ、近似と学習側の優先度がどれだけ揃うかを把握する。

受入基準:

1. 段階 1 の等価性: 編集前と後で、同 seed・決定的モードの短い Run（Atari、DropMerge、LunarLander の `run.@cpu`、GridMaze）の metrics checksum、`metrics.scalar.defs`・`metrics.trace.defs` の中身、WARN の行が一致する。
2. 単体テスト（Agent の step 部品）: 合成したヒント・報酬の系列について、`approx_priority` が replay の初期優先度の推定（`InitialPriorityCompleter` 経由の値）と一致する。n step の途中の終端、打ち切り（空欄）、TBO、Munchausen の有無、reward scaler の倍率を含む。
3. 学習の値が変わらない: ヒントのヘルパーを切り出す前と後で、`actor_approx` の同 seed・決定的モードの短い Run の metrics checksum が一致する。
4. 単体テスト（列の宣言）: 構築時の fail-fast（番号が整数でない・0 以下、鍵が 2 つ、target が無い、未知の target、`$agent`・`$exp`・`$update_result`・`$env`、`$runner` のエピソード集計以外への集約の接頭辞、未知の属性、同じ属性の重複、空の `label:`、`display:name` を数値の列に、連続行動の `action` と `display:name`）と、正しい宣言の解析結果（番号順、label の既定）。
5. 単体テスト（時点）: 終端の行の `$env_step` の値がリセット前（097 の実装のあと）、`$runner step_in_episode` が終端の行でエピソードの長さになり次の行で 1 に戻る、強制した行動が `$action_info action` に出る、aux に `reward` という鍵を持つ env の Step の結果でも `$env_step reward` が lane ごとの欄の値を返す、`approx_priority` が n step 後に正しい行へ入る。DQN 以外の部品（テスト用）が答える任意の鍵の列を読める、あとで決まる鍵は部品が示した行の列だけが埋まる（NaN で決まった列も待つのをやめる）、部品が答えない鍵は最初の行で止まる、部品を出さない Agent では `$agent_step` の列が鍵によらず空欄になる。
6. 単体テスト（R1・R2）: EvalRunner の `TrainEvent` の時点で `$runner reward` がその step の報酬、`$runner mean.episode_steps` がその step で完了したエピソードの値（完了が無ければ NaN）。定期評価のセッションの scalar と trace は編集前と一致する。
7. 単体テスト（再生の制御）: 速度の段階と端、Live からの再生がいちばん古い行から始まる、いちばん新しい行で止まる、一時停止中の再生が選んでいる行から始まる、選んでいる行が消えたらいちばん古い行へ移る。Live のときのコマ送りがいちばん新しい行で Trace に入って `Trace: end` を出し、コマ戻しがその 1 つ前の行で Trace に入って `Trace: step back` を出す（行が 1 件ならコマ戻しはその行で `Trace: start`）、いちばん新しい行でのコマ送りといちばん古い行でのコマ戻しがその行に留まる（`Trace: end`・`Trace: start`）、行が 0 件の間はボタンとスライダーが無効。
8. 手動の確認（Runner、Atari の online）:
   - Evaluation Trace を開くと行が流れ、報酬と終端の行に色が付き、`prio_pct` が出る。閉じると行を作らない（開き直すと空から）。
   - 行 t を選ぶと、EvalPanel・QValuePanel・Conv2d の見出しに `Trace: step <その行の step>` が出る。QValuePanel が示す選んだ行動とその Q が、その行の `action`・`q_sa` と一致する。EvalPanel の画面に重ねて出る行動名と報酬が、その行の `action`・`reward` と一致する。Conv2d は、同じ行を選び直すと同じ活性が出て、開く前の行では no data を出す。HeatMap は Trace モードに入らず（`Trace: step` を出さず）、今のまま更新を続ける。
   - 再生／一時停止・停止・コマ送り・コマ戻し・速度・スライダーが D12 のとおりに動き、ステータスバーの通知欄に D12 の文言が出る。停止・選択の解除（表で Esc キー。行の無いところのクリックでは戻らない）・手動の行動・EvalPanel の Resume・Evaluation Trace を閉じることで Live に戻る。
   - 行を選ぶと EvalPanel が一時停止し、Resume で再開すると Live に戻る。表にフォーカスがあるときの上下キーは行を動かし、EvalPanel にフォーカスがあるときは行動になる。
   - Evaluation Trace を開いても、EvalPanel の実際の速さが落ちない: Live で 1 分間に増えた `step` 列の値が、fps × step_per_frame × 60 の 95% 以上。学習側の速さ（ステータスバーの field 1）が、開く前の 1 分ずつ 3 回の幅に収まる。
   - Evaluation Trace を表示して 1000 行たまったときの Runner のメモリの増え方が、D10 の見積もりの範囲に収まる。

9. UI調整（列幅・ラベル）: 既定列の見出しが`Q_sa`／`V_s`／`trunc`となり、取得キーと値が変わらない。短い数値・長い整数・小数・負値・D8の行動名表示を使い、見出しと値が余白付きで収まる。新規行の長い値、遅れて確定した近似優先度で必要な列だけ広がり、短い値への入替えや古い行の削除では縮まない。手動で狭めた列を含め手動幅を維持し、開き直すと手動指定を含む幅を初期化する。最小幅でも操作バー全体と左端から最初の行動列までが収まり、ドック／フロート・列幅・フォント・DPI変更後も守られる。合計幅が領域を超えても横スクロールで読める。
10. UI調整（テーマ）: ライト／ダーク双方で通常・報酬・終端・選択行の文字が読め、報酬と終端が重なる行は終端色になる。選択中はシステムの選択背景と文字色になる。開いたままテーマを変えても配色・アイコンが追従し、DPIやフォントが変わっても見出し・アイコン・枠が欠けない。列見出しはLive／Traceと行選択で色が変わらない。色付け有無の設定は増えていない。
11. UI調整（プレイヤーとTrace表示）: D12の6ボタンが指定順のアイコンと英語ツールチップで出る。速度の全段階がD12の分数を使わない倍率表記で表示される。コマ送りのアイコンは縦棒の右に右向き三角、コマ戻しは左向き三角の右に縦棒で、メインツールバーの Step のアイコンもコマ送りと同じ形である。Live／一時停止中はPlay、再生中はPauseとなり、最新行に到達するとPlayへ戻る。履歴0件ではボタンとスライダーが無効。行選択・再生・一時停止中は操作バー全体に2 DIPの赤枠が出て、Liveへ戻るすべてのきっかけ（D11）で消える。枠の切替で配置が動かず、見出しの`Trace: step ...`も残る。
12. UI調整の回帰:
    - 受入 5（時点）と受入 7（再生の制御）の単体テストが通る。2026-10-22 の調整の前から期待値を変えるのは、D12 の Live の箇条と速度の表示で改めたものだけである。すなわち、低速倍率の表示（`x0.125`・`x0.25`・`x0.5`）、Live での一時停止（独立したボタンが無くなる）、Live でのコマ送りの通知（`Trace: end`）、Live でのコマ戻しで選ぶ行と通知（1 つ前の行で `Trace: step back`、行が 1 件ならその行で `Trace: start`）である。
    - 受入 8 の手動確認のうち、再生・停止・スライダー・Live に戻るきっかけ・開き直しで表が空から始まることが、D11・D12 のとおりである。
    - [common.txt](../../apps/runner/config/common.txt)・[Atari.txt](../../apps/runner/config/Atari.txt) の `app.eval_trace_panel.columns.*` の、2026-10-22 の調整の前との差分が、`Q_sa`・`V_s`・`trunc`の`label:`指定と、rewardをaction直後へ移す列4〜6の並べ替えだけである。各列のtarget・取得キー・値は変わらない。

## 9. 依存と順序

基盤 PRD（[097](097_env_auto_reset_after_notification_10prd.md)、§7 の 1）→ 本 PRD の段階 1 → 段階 2 → PRD 081 の改訂と実装。段階 1 は基盤 PRD に依らないが、ユーザーの指示で基盤 PRD を先に行う。

## 10. 文書同期（実装と同じ変更で更新する）

| 対象 | 変更 |
|---|---|
| [100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) | RunManager の責務とクラス図から、メトリクスの配線を `MetricsWiring` へ分けたこと |
| [140_observability.jp.md](../design/140_observability.jp.md) | 値の読み口と配線のクラス、step trace の列と target（`$env_step`・`$agent_step`・`step_in_episode`・`action`）と時点の表 |
| [160_applications_and_tools.jp.md](../design/160_applications_and_tools.jp.md) | Evaluation Trace、列幅・最小幅・テーマ配色、追従paneとLive／Trace・操作バーの赤枠、6個のアイコンボタンと再生／一時停止トグル、Live でのコマ戻し、メインツールバーの Step のアイコン |
| [020_user_guide_run.jp.md](../design/020_user_guide_run.jp.md) | Evaluation Traceのアイコン操作（Live でのコマ戻しを含む）・赤枠・手動列幅と開き直しの使い方、列の設定とQ_sa／V_s／truncの表示名 |
| [common.txt](../../apps/runner/config/common.txt)・[Atari.txt](../../apps/runner/config/Atari.txt) | D13 |
