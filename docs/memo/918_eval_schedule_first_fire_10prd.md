# PRD918: eval スケジュールの発火の列 — 学習直後の評価を既定で止め、発火の時点をずらせるようにする

> 2026-11-10 再グリル。暫定の印を外し、旧 D1〜D6（キーの形は未決、既定は今の挙動のまま）を §4 の確定事項 D1〜D10 に置き換えた。ゴールを、既定で学習直後の評価を走らせないこと（全 env）と、interval を変えずに発火の時点をずらす `offset` に広げた（ユーザーの指示）。足すキーは `warmup_count`（既定 1）と `offset`（既定 0）。別モデル（Fable）の検討を受けて、`IntervalGate` は変えずに `EpisodeEvalObserver` が受け持つ形にし（D5）、offset を当てるのは位相のために間隔を曲げた箇所だけに狭め（D8。同じ間隔の組は §8 へ）、`run.@evalonly` に Greedy を組み（D7）、検証資材と CHANGELOG を移行対象に足した（§5.4）。今のコードと設定に合わせて行番号とタグ名（`greedy_dist` → `[greedy]`、400 本 → 100 本）を直し、BTR・SB3・DI-Engine が学習開始時に評価するかを §2.4 に足した。節を §1〜§11 に組み直し、旧「Context」は §1・§2 へ、旧「テスト」と「受入基準」は §7 へ移した。旧受入 6（name_this_game か double_dunk を 10M 流して `waited` が出ないこと）は、100 本の今の既定では抑止の効果を見分けられないので外した。観点のレビューの指摘で、§5.4 の移行対象（observer を直接作るテスト 13 か所、prd084 の検証資材）と触らないものの区分、§5.3 の末尾の検索と件数、D7 の根拠の行、§3.3・§6 の回数の書き分け、D8 の例外（`[greedy_single]` を `[greedy]` にそろえる）、D7 の検証資材の範囲（§5.4 の行を参照する書き方）を直した。工程末に、offset の仕事を新しい課題にせず、開発者の指示で課題 a86dcc の題と範囲を改めて覆わせ（§5.6）、§8 にあった Atari.txt の古い注記はこの場で直した。
> 2026-11-03 番号を 999 から 918 へ振り直した（PRD114）。
> 起点: 2026-09-21 の Atari-5 地ならし。`run.@eval2ch` の ε=0 の評価（当時のタグ名 `greedy_dist`、400 エピソード）が learn_step=0 で発火し、5 ゲーム中 2 つで背景のセッションが 1 時間級になり、次の発火で学習を 692.6s・885.5s 止めた。この点は未学習の方策の測定で、実験記録が使っていない捨て点である。
> 関連: [076_eval_session_drain_on_exit](done/076_eval_session_drain_on_exit_10prd.md)（Run 終了側の対）、[052 eval 定義/スケジュール分離](done/052_eval_schedule_separation_10prd.md)、[053 interval bucket-crossing](done/053_interval_gate_perf_ema_10prd.md)、[118 Atari の Run プロファイルの整理](118_atari_run_profile_flatten_10prd.md)。

## 1. 問題

### 1.1 学習直後に必ず 1 回評価する

`run.eval_schedule.[tag].interval` で組んだ評価は、step の値によらず最初に必ず 1 回発火する。最初の学習イベントは warmup が明けた最初の学習 step で、そのときの learn step は 0 である（§2.1）。そのため、どの評価の系列も、先頭に初期の重みを測った点が入る。

Atari の ε=0（Greedy）の評価では、この 1 回が高くつく。未学習の貪欲方策は行動を退化させ、全エピソードが打ち切り上限の 27,000 step（`max_episode_frames` 108,000 / `frame_skip` 4）に張り付くことがある。背景で走るこの評価が次の発火までに終わらないと、次の発火のときに学習が待つ。

この発火は、bucket の境目の副作用ではなく、明示した規則である（[ADR 0028](../adr/0028-interval-fires-on-bucket-crossing.md) の Consequences、[PRD053](done/053_interval_gate_perf_ema_10prd.md) の D4）。発火の格子の位相をずらすだけでは止まらない。

### 1.2 実測（2026-09-21）

Atari-5 の 5 ゲームを同じ設定（10M / RR4 / `run.@eval2ch`）で回したときの、learn_step=0 の ε=0 の評価セッション。当時のタグ名は `greedy_dist` で、400 エピソード・`eval_batch_size` 16。

| game | セッション所要 | 次の発火の `waited` | `mean.episode_steps` | `mean.episode_return` |
|---|---|---|---|---|
| qbert | 53.6s | なし | 275 | 0 |
| phoenix | 63.4s | なし | 386 | 0.90 |
| battle_zone | 292.8s | なし | 1,578 | 3.50 |
| name_this_game | 3,806.4s | **692.6s** | **26,963** | 0 |
| double_dunk | 4,110.2s | **885.5s** | **26,992** | −1.46 |

- セッションの所要時間は、学習の停止時間ではない。`use_background = true` では評価は別のスレッドで走り、学習が止まるのは、次の発火のときに前回が終わっていない場合だけである（上の表の `waited` 列）。
- 学習の停止は 2/5 ゲームで 692.6s と 885.5s、合計約 26 分だった。5 ゲーム × 50M のスイープ（約 47.5h）に対して 0.9% である。
- どのゲームで起きるかは事前に分からなかった。同じ未学習の方策・同じ設定で 5 ゲーム中 2 つだけが該当し、スコアの高低やエピソード長とは対応していなかった。
- Breakout でも起きていた（[Atari.txt](../../apps/runner/config/Atari.txt) の 509〜510 行の注記「2026-09-18 の RR1 で 49 分」）。
- この点は使われていない。実験記録 [2026-09-15 探索ブロック 07](../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md) の ε=0 の 9 点の対の表は 5.2M から始まり、learn_step=0 の点を使っていない。

### 1.3 2 つの評価を同じ時点に発火させないために、間隔をずらしている

背景の評価が 2 本同じ学習 step で発火すると、GPU と CPU を学習と取り合い、前回の待ちも重なりやすい（DropMerge で 2026-07 に診断された学習の定期停止。[PRD052](done/052_eval_schedule_separation_10prd.md) の「6. スコープ外」、[ADR 0027](../adr/0027-eval-definition-schedule-separation.md)）。今の設定はこれを避けるのに、間隔を互いに素にするしかない。Atari 本体の `[eval]` は `[eval_target]` の 1000 に対して 1009 にしている（[Atari.txt](../../apps/runner/config/Atari.txt) の 1415〜1416 行）。間隔は予算や比較の都合で決めたい値なので、位相のために間隔を曲げるのは本末転倒である。ただし、同じ時点に発火させないことで分けられるのは評価の始まる時点だけで、1 回の評価が interval より長ければ実行は重なったままになる（§2.1）。

## 2. 現状（2026-11-10 にコードで確かめたこと）

### 2.1 発火の仕組み

- 発火の判定は `IntervalGate::ShouldFire`（[util.hpp](../../core/anet-core/include/anet/util.hpp) の 287〜304 行）。`step / interval` の商（バケット）が前回の発火より増えた呼び出しで 1 回だけ発火し、最初の呼び出しは step によらず発火する。複数のバケットを一度に跨いでも発火は 1 回で、catch-up しない（ADR 0028）。
- `IntervalGate` は eval のほかに、メトリクスの observer（[observers.hpp](../../core/anet-core/include/anet/observers.hpp) の 136・137・213・364 行）、DQN の診断（[dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp) の 6 か所）、ImageCls（[image_cls_agent.hpp](../../core/anet-core/include/anet/image_cls_agent.hpp) の 300 行）が使う。
- `EpisodeEvalObserver` は interval が 1 以上のときだけゲートを持ち（[observers.cpp](../../core/anet-core/src/observers.cpp) の 511〜512 行）、学習イベントごとに learn step の軸でゲートに問い合わせる（同 621〜624 行）。発火したら評価セッションを背景に投げ、前回が終わっていなければ終わるまで待つ（同 586〜611 行、`waiting for previous session` と `waited for previous session` のログ）。背景のスレッドはタグごとに 1 本で（同 515〜517 行）、待つのは自分のタグの前回だけである。
- 学習イベントは、Learner が更新したときだけ通知され、learn step は通知のあとに足される（[trainer.cpp](../../core/anet-core/src/trainer.cpp) の 613〜639 行）。最初の学習イベントは warmup 明けで、learn step は 0 である。
- Run の step の数は checkpoint から戻さない（[200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) の 616 行）。checkpoint を読む継続 Run と評価専用 Run（`learner.enabled = false`）でも、learn step 0 で発火する。そのときの評価は読み込んだ重みを測る。
- `RunManager` は `run.eval_schedule.[tag]` の `interval`（必須、負はエラー）・`use_background`・`wait_on_exit` を読む（[trainer.cpp](../../core/anet-core/src/trainer.cpp) の 960〜984 行）。`run.eval_schedule.[tag]` に何かのキーがあれば、そのタグは schedule の一覧に入り、`interval` が無いと止まる（同 965〜968 行）。`use_background`・`wait_on_exit` は `interval = 0` でも読んで検証する。休止（dormant）の判定はそのあと（同 1093〜1111 行）。
- 起動時に `eval.[<tag>]: scheduled (interval=…, background=…, wait_on_exit=…, episodes=…, batch_size=…)` を INFO で出す（同 1113〜1116 行）。セッションの始まりと終わりは verbose の `eval.[<tag>]: session start learn_step=… exp_step=…`（[observers.cpp](../../core/anet-core/src/observers.cpp) の 626〜628 行）と `eval.[<tag>]: session end learn_step=… exp_step=… elapsed=…`（[trainer.cpp](../../core/anet-core/src/trainer.cpp) の 423〜428 行）で出る。
- 1 回の学習イベントで進む learn step（刻み）は `num_envs × replay_ratio / replay_batch_size` で、LunarLander は 256 × 8 / 1024 = 2（[LunarLander.txt](../../apps/runner/config/LunarLander.txt) の 39・176 行、[agent.txt](../../apps/runner/config/agent.txt) の 221 行）、GridMaze は 256 × 1 / 128 = 2（[GridMaze.txt](../../apps/runner/config/GridMaze.txt) の 11・175・177 行）、DropMerge は 1〜2（[DropMerge.txt](../../apps/runner/config/DropMerge.txt) の 13・174・177・679・682 行）。

### 2.2 今の設定

- Atari の ε=0 の評価は今 `[greedy]` で、既定は 100 本（[Atari.txt](../../apps/runner/config/Atari.txt) の 1434〜1440 行。400 本は `run.@gdist400`、530 行）。`[greedy_single]` は `[greedy]` の schedule を `$` で継承し、同じ時点に並ぶ（同 517・525 行、1442〜1449 行）。既定のチェーンは `run.@eval2ch_r1` を含む（同 156 行）。
- 間隔を互いに素にして同じ時点を避けているのは、Atari 本体の `[eval]` 1009（1415 行）と、Atari の `run.@evalonly` の 10・11（551〜552 行）、12_batch_run.bat の評価専用の 10・11・12（103〜106 行）である。
- 素数の間隔で、理由の注記が無いもの: `run.@skip` の `[greedy]`（LunarLander 1009・GridMaze 53・DropMerge 503。[LunarLander.txt](../../apps/runner/config/LunarLander.txt) の 90 行、[GridMaze.txt](../../apps/runner/config/GridMaze.txt) の 70 行、[DropMerge.txt](../../apps/runner/config/DropMerge.txt) の 509 行）と、12_batch_run.bat の `TWSES` の `[greedy_single]` 1009（287 行）。
- Atari の `run.@rainbow_like` は `[eval]` 500,000・`[greedy]` 1,250,000 で（109〜110 行）、2,500,000 learn step ごとに重なる。間隔は予算から決めた値で、位相のために曲げたものではない。
- 同じ間隔で同じ時点に発火する組: DropMerge 100・100（[DropMerge.txt](../../apps/runner/config/DropMerge.txt) の 1695〜1696 行）、LunarLander 200・200（[LunarLander.txt](../../apps/runner/config/LunarLander.txt) の 461〜462 行）、GridMaze 10・10（[GridMaze.txt](../../apps/runner/config/GridMaze.txt) の 312〜313 行）、common.txt の既定 100・100（[common.txt](../../apps/runner/config/common.txt) の 6・13 行。CartPole が使う）、12_batch_run.bat の `GEV`（`[eval_target_greedy]` 200 が DropMerge の `[eval]` 100 と重なる。74 行）・`B1024RR1`（25・25・50。89 行）・`TWSES`（`[eval_single]` 200 が LunarLander の `[eval]` 200 と重なる。287 行）。
- 評価専用 Run は、Atari の `run.@evalonly`（[Atari.txt](../../apps/runner/config/Atari.txt) の 543〜552 行。`max_exp_step / 10` 回のセッション）と、bat の変数 `EV2X2`（11_batch_run.bat の 53 行、`exp_exit_step = 40`、間隔 1000 で各タグ 1 回）・`EVONLY`・`GEVONLY`・`GSESBIG`（12_batch_run.bat の 103〜106 行、`exp_exit_step = 40`、注記「4 sessions x 1250 episodes on each of the five slots」）・`EVTAU`（13_batch_run.bat の 76 行、`exp_exit_step = 5`、間隔 10 で各タグ 1 回）である。どれも learn step 0 の発火を数に入れている。`run.@evalonly` は `[eval_target]`・`[eval]` だけを組み、`[greedy]`・`[greedy_single]` は `run.@eval2ch*` を重ねたときだけ learn step 0 に 1 回走る（重ねないと走らない）。Atari.txt の 1444 行は、`[greedy]`（53_evalg）と `[greedy_single]`（54_evalgs）を「全 Run で重ねて比べるために常に置く」と書いている。
- 検証資材のスクリプトも、学習直後の発火を前提に評価を組む: [prd104/run_experiments.py](../../core/anet-core/testdata/prd104/run_experiments.py) の 89〜103 行（`learner.enabled = false`、`[greedy]` の interval 1,000,000,000 で learn step 0 の 1 回だけを測る）、[prd094/run_validation.py](../../core/anet-core/testdata/prd094/run_validation.py) の 73 行・[prd097/run_validation.py](../../core/anet-core/testdata/prd097/run_validation.py) の 77 行（interval 1000、予算 65,536・512,000 exp step）、[prd103/run_current.py](../../core/anet-core/testdata/prd103/run_current.py) の 71〜72 行（smoke の `[greedy]` 256）、[prd084/run_forward_refactor.ps1](../../core/anet-core/testdata/prd084/run_forward_refactor.ps1)・[prd084/run_review_validation.ps1](../../core/anet-core/testdata/prd084/run_review_validation.ps1)（`[greedy]` だけを休止にし、ほかの評価は組んだまま）。どれも新旧の実行ファイルの出力を比べる資材で、wiki の地図の「変えるときの条件」が PRD103 の資材での比較を挙げている。
- 設定の検査 [config_baseline_test.cpp](../../core/anet-core/src/config_baseline_test.cpp) の 522 行が、`run.@skip` の `[greedy]` の間隔 53・1009・503 を断定している。

### 2.3 規約と過去の判断

- ADR 0028 の Consequences は「初回（step=0）は発火する。…学習開始直後のベースライン点が残る」と書く。PRD053 の D4 も「warmup 明け直後のベースライン点を残す」とした。
- [ADR 0045](../adr/0045-eval-session-drain-on-exit-wait-or-discard.md) の Consequences は、初回発火の抑止が入るまで、短い配線 Run で t=0 のセッションが終了時に走り続ける移行の依存を書いている。
- クリーンブレーク方針（[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針）により、契約を変えるときは現用の設定・テスト・文書を同じ変更で移し、過去の Run・ADR・実験記録は書き換えない。
- 不正な設定値は fail-fast、意図された休止（`interval = 0`）はエラーにしない（[AGENTS.md](../../AGENTS.md) §Fail-Fast 原則、§設定値の扱い）。

### 2.4 ほかの実装（2026-11-10 にソースで確かめた）

| | 学習開始時の評価 | 間隔の設定と既定 | 開始時の評価を選ぶ設定 | 発火の時点をずらす設定 |
|---|---|---|---|---|
| BTR（公開実装） | しない。最初は 250,000 step（`next_eval = eval_every`、`steps >= next_eval` で評価） | `eval_every = 250000` はコードに固定。10 env・100 本 | なし（評価全体の on/off だけ） | なし |
| Stable-Baselines3（`EvalCallback`） | しない。`n_calls` を足してから `n_calls % eval_freq == 0` で判定する | `eval_freq=10000`（callback の呼び出し回数）、`n_eval_episodes=5`、`deterministic=True` | なし | なし |
| DI-Engine | する。前回の評価の反復を −1 で始め、`train_iter` 0 の初回の判定が通る | `eval_freq=1000`（学習の反復） | なし | なし |

出典は §11 の [BTR]・[SB3]・[DI-Engine]。3 つのうち 2 つが学習開始時に評価しない。どれも、開始時の評価やずらしを選ぶ設定を持たない。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

1. eval schedule の既定で、学習直後（発火の列の先頭）の評価を走らせない。全 env・全タグに効く。要る Run（評価専用 Run など）は、タグごとに走らせるよう書ける。
2. interval を変えずに、評価ごとに発火の時点をずらせる。今ある、位相を避けるために間隔を曲げた設定を置き換える。
3. `IntervalGate` と、そのほかの利用者（メトリクスの observer、DQN の診断、ImageCls）の挙動は変えない。

### 3.2 非ゴール

| 退けた案・範囲 | 理由（本当の制約か） |
|---|---|
| Run 終了時に背景の評価が捨てられる件 | [PRD076](done/076_eval_session_drain_on_exit_10prd.md) で扱った、発火の列の末尾の話である。別の関心 |
| 未学習の方策が行動を退化させること自体 | 方策として正当な挙動で、本 PRD は「それを測らない」だけを扱う |
| 評価のエピソード上限（`max_episode_frames`）を下げる案 | 上限を下げると学習後のエピソードも打ち切られ、測定値に偏りが入る。本当の制約（測定の比較条件） |
| メトリクス側の interval にも同じキーを足す案 | メトリクスの学習直後の点は安く、解析の邪魔になった例が無い。解く問題が無く、クリーンブレークの制約とは関係しない |
| 背景の評価が終わっていないとき、次の発火を待たずに見送る案 | 発火の列の時点ではなく、待ち方の関心である。別の関心 |
| 評価のセッションを足し引きすると、同じ seed でも学習の軌跡が変わる件（[wiki「評価と比較の条件」](../wiki/mechanisms/evaluation.md)） | 乱数の系列の共有の関心である。本 PRD は、この性質があることを §9 の影響に書くだけにする |
| 評価が 1 回も走らないまま Run が終わったときに WARN か INFO で知らせる案 | 最初の発火より短い配線・確認の Run で毎回出て、毎回出る WARN が本物を埋もれさせる課題（[issue-20261107-a9a0e3](../management/issues/issue-20261107-a9a0e3.md)）と同じ形になる。評価専用 Run の書き忘れは結果が出ないことで気づける。本当の制約ではなく、再グリルでのユーザーの選択 |
| 「既定では今の挙動を変えない」（起票時の旧ゴール） | クリーンブレークで移せるので本当の制約ではない。再グリルでユーザーが撤回した（D2） |

### 3.3 成功の測り方

- 既定の Run（どの env でも）で、各タグの最初の評価セッションの learn step が `offset + interval` 以上である（Run ログの最初の `session end` 行）。
- offset でずらした組（Atari 本体の `[eval_target]` と `[eval]`、`run.@skip` の `[greedy]` と `[eval_target]`・`[eval]`、12 の bat の `TWSES` の `[greedy_single]` と LunarLander の `[eval_target]`・`[eval]`）で、`session start` の learn step が重ならない。
- 評価専用 Run のうち、今も組んでいるタグ（`run.@evalonly` の `[eval_target]`・`[eval]`、bat の評価専用の変数の各タグ）が、今と同じ回数のセッションを出す。`run.@evalonly` の `[greedy]`・`[greedy_single]` は、D7 で各 `max_exp_step / 10` 回に増える（今は `run.@eval2ch*` を重ねたときの 1 回か、重ねないときの 0 回）。
- 設定に、位相のために間隔を曲げた値（互いに素の 1009・11・12、素数の 53・503・1009）と、その注記が残っていない。

## 4. 確定事項

2026-11-10 の再グリルで決めた。D1・D2・D3・D8 と、D7 のうち「評価が 1 回も走らないまま Run が終わっても知らせない」はユーザーの選択で、ほかは規約・コード・ほかの決定から導いた。D8 は、別モデル（Fable）の検討の反対を受けて、ユーザーが範囲を「間隔を曲げた箇所だけ」に選び直した。D8 の値と、D7 の評価専用 Run の移し方は、ユーザーが「推奨で埋める」を選んだ細部（§6）による。

| # | 決定 | 理由と依存 |
|---|---|---|
| D1 | `run.eval_schedule.[tag].offset` は、発火の列 {offset + k·interval} 全体をずらす。最初の発火は、learn step が offset に届いた最初の学習イベント。単位は interval と同じ learn step。0 以上だけを検証し、上限は置かない。offset が interval 以上なら、その step から始める（開始の遅れ）として、書いたとおりに動く。開始の遅れとして使い、offset の点から評価したいときは、`warmup_count = 0` も書く（D2 の既定 1 では offset の点も止まる） | ずらす目的は 2 本の評価を同じ時点に発火させないことで、学習直後の 1 回も含めて重ならないのは列ごとずらす形だけである（格子だけずらす形は学習直後の 1 回が重なる）。上限は、書いたとおりに動く以上は要らない（ユーザーの指摘）。単位は、ゲートが learn step の軸で判定することによる（§2.1） |
| D2 | `run.eval_schedule.[tag].warmup_count` は、発火の列の先頭から評価を走らせない発火の回数。発火の回数で数え、評価を走らせなかった発火も「前回の発火」として扱う（後ろの発火は遅れない）。eval schedule の既定は 1（全 env・全タグ）。評価専用 Run など、学習直後の評価が要る Run は 0 を書く | 初期の重みの評価は普通は要らない（ユーザー）。BTR と SB3 も学習開始時に評価しない（§2.4）。止めた発火を前回に数えるのは、発火の格子を step から決める ADR 0028 の形を保つため |
| D3 | キーの名前は `warmup_count` と `offset` | `warmup_count` は、既存の `robust_warmup_count`（[agent.txt](../../apps/runner/config/agent.txt) の 277 行）・`warmup_steps` に合わせた（ユーザー）。`skip_first` は `skip` が TempoRL のスキップ長の語（[CONTEXT.md](../../CONTEXT.md) の 81 行）、`first` が「最初の 1 回」の bool に読めるので避けた。`offset` は今の設定に同じ名のキーが無い |
| D4 | 2 つのキーは `interval` と同じ読み口で読む。負の値は Env を作る前に `ANET_SYSTEM_ERROR` で止め、メッセージにキー・指定値・期待（0 以上）を入れる。`interval = 0` のタグ（休止）でも、書いてあれば読んで検証し、エラーにも WARN にもしない。`run.eval_schedule.[tag]` に 2 つのキーだけを書いて `interval` を書かないと、今と同じく `interval` が無いエラーで止まる | 不正値は fail-fast（AGENTS.md §Fail-Fast 原則）。休止は意図された状態で、同じ階層の `use_background`・`wait_on_exit` も休止で読んで検証している（§2.1）。`interval` が必須なのは今の契約のまま（§2.1） |
| D5 | `IntervalGate` は変えない。offset と warmup_count は `EpisodeEvalObserver` が受け持つ。learn step が offset に届くまではゲートに問い合わせず、届いたら learn step から offset を引いた値でゲートに問い合わせ、発火を数えて先頭の warmup_count 回では評価を走らせない。bucket-crossing と、catch-up しない規則はゲートのまま | 今のゲートは「最初の呼び出しで発火、以後は商が増えたら発火」なので、offset を引いた値を渡せば §5.1 の列になる（別モデルの検討の指摘。[util.hpp](../../core/anet-core/include/anet/util.hpp) の 287〜304 行）。共通部品を変えなければ、ゴール 3 は確かめるまでもなく満たされる。offset と warmup_count を使うのは eval schedule だけである（非ゴールの「メトリクス側の interval」） |
| D6 | 起動時の行を `eval.[<tag>]: scheduled (interval=…, offset=…, warmup_count=…, background=…, wait_on_exit=…, episodes=…, batch_size=…)` にする。既定のときも 2 つの値を出す。評価を走らせなかった発火ごとの行は出さない | scheduled の行に発火の条件をそろえて出す今の決まり（[140_observability.jp.md](../design/140_observability.jp.md) の 330 行）を保ち、Run ログだけから発火の列を組み立て直せるようにする。発火の列を決める 3 つを並べる |
| D7 | 評価専用 Run は、組むタグに `warmup_count = 0` を書く。Atari の `run.@evalonly` と、bat の `EV2X2`・`EVONLY`・`GEVONLY`・`GSESBIG`・`EVTAU`、新旧の実行ファイルを比べるか、学習直後の発火に頼って測る検証資材（§5.4 の移行対象の検証資材の行）が当たる（値は §5.3・§5.4）。`run.@evalonly` には `[greedy]`・`[greedy_single]` も、ほかのタグと同じ間隔 10 で組む。書いていないタグは既定に従う。checkpoint を読んで学習を続ける継続 Run には書かない。評価が 1 回も走らないまま Run が終わっても、知らせない | 評価専用 Run は learn step 0 の発火を数に入れていて、bat の評価専用 Run は 1 のままだと終わるまでに評価が 1 回も走らないことがある（§2.2）。継続 Run の learn step 0 の点は元 Run の最後の重みで、元 Run の最後の評価から 1 間隔以内にあるので、点の価値が小さい（D2 の「普通は要らない」と同じ扱い）。`run.@evalonly` の Greedy は、Atari.txt の 1444 行の「全 Run で重ねて比べるために常に置く」と、469〜471 行（`run.@to_50` の注記）の「interval 10 と合わせて max_exp_step / 10 セッション」による。検証資材は新旧の実行ファイルの比較を保つため（§2.2）。知らせない理由は §3.2（ユーザーの選択） |
| D8 | offset を、位相を避けるために間隔を曲げている箇所にだけ当てる。Atari 本体の `[eval]` 1009 は `[eval_target]` に offset を持たせて 1000 に戻し、`run.@skip` の `[greedy]` の素数（1009・53・503）と 12 の bat の `TWSES` の `[greedy_single]` 1009 は、きりのよい間隔と offset に直す。評価専用 Run の互いに素の間隔（`run.@evalonly` の 11、12 の bat の 11・12）は 10 にそろえ、ずらすための offset は付けない（本体か `run.@skip` の offset が残るタグには、打ち消す `offset = 0` だけを書く。D9）。同じ間隔で同じ時点に発火する組（§2.2）は変えず、§8 の確かめのあとに回す。値は §5.3 の表を正本とする | ゴール 2（ユーザーの選択）。同じ時点の発火をずらして分けられるのは始まる時点だけで、待ちが減るかは測っていない（§1.3・§2.1）。評価専用 Run は 1 lane で learn step が一瞬で進むので、offset を付けても評価はほぼ同時に始まる。Atari で `[eval_target]` に持たせるのは、Atari の評価のプロファイル（`run.@eval2ch` 系・`run.@btreval_r*`）が `[eval_target]` を 0 にするので、`[eval]` の間隔を変えるプロファイルに offset が残らないため。offset は、刻み（§2.1）より離して、同じ env のほかの評価と同じ学習イベントで発火しないように選ぶ。ただし `[greedy_single]` は `[greedy]` と同じ時点にそろえる（12 の bat の `TWSES` の `[greedy_single]` は LunarLander の `[greedy]` と同じ 1000・offset 50）。2 本は同じ点を Viewer で重ねて比べるために置いている（Atari.txt の 1444 行） |
| D9 | 本体の設定で offset を持つタグの interval をプロファイルや bat で変える箇所では、その offset が新しい interval でも意図どおりかを確かめ、違えば offset も書く | offset は step で書くので、interval だけを変えると本体の offset が残る（D1）。評価専用 Run では、残った offset のせいで終わるまでに 1 回も走らないことがある |
| D10 | ADR を 1 本起こし（[ADR 0070](../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md)）、ADR 0028 の先頭に、eval schedule についてだけ初回の発火の判断を改めた旨の注記を足す。Atari の Greedy の間隔は変えない（`run.@eval2ch` の 50M も `run.@eval2ch_r1` の 100M も 9 点になる） | D2 は ADR 0028 の「初回は発火する」を eval schedule について改める。コミット済みの ADR は書き換えず、注記と新しい ADR で残す（anet-create-prd の「Never rewrite an existing ADR」）。Greedy の間隔は予算から決めた値で、点を 10 に戻すには終わり際の発火の扱いの見直しが要り、本 PRD の範囲外である |

## 5. 実装契約

### 5.1 発火の列

eval schedule が有効なタグ（`interval` が 1 以上）について、学習イベントごとに、そのときの learn step s を次の表に上から順に当て、最初に当たった行で扱う。

| 順 | 条件 | 扱い |
|---|---|---|
| 1 | s < offset | 何もしない（発火しない。前回の発火も変えない） |
| 2 | まだ一度も発火していない | 発火する |
| 3 | ⌊(s − offset) / interval⌋ が、前回の発火のときより大きい | 発火する |
| 4 | 上のどれでもない | 何もしない |

発火のうち、最初から数えて `warmup_count` 回目までは評価を走らせない。それより後の発火では、今と同じく評価セッションを 1 回始める（背景で走らせ、前回が終わっていなければ待つ）。評価を走らせなかった発火も、3 の「前回の発火」になる。learn step は減らない。

例（interval 1000、刻み 1、learn step は 0 から）:

| offset | warmup_count | 評価が始まる learn step |
|---|---|---|
| 0 | 0 | 0, 1000, 2000, … |
| 0 | 1（既定） | 1000, 2000, … |
| 500 | 0 | 500, 1500, 2500, … |
| 500 | 1 | 1500, 2500, … |
| 2500 | 0 | 2500, 3500, … |
| 2500 | 1（既定） | 3500, 4500, … |

刻みが interval を超えるときも、1 回の学習イベントで発火は 1 回で、catch-up しない（ADR 0028）。最初の学習イベントの時点で s がすでに 2 つ以上先の格子にいても、その学習イベントが 1 回目の発火になる。

この列は、`EpisodeEvalObserver` が、learn step が offset に届くまでゲートに問い合わせず、届いたら learn step から offset を引いた値で今の `IntervalGate` に問い合わせ、発火を数えて先頭の warmup_count 回を止めることで作る。`IntervalGate` は変えない（D5）。

### 5.2 設定キー

| キー | 値 | 既定 | 値の範囲と、外れたとき |
|---|---|---|---|
| `run.eval_schedule.[tag].offset` | 整数（learn step） | 0 | 0 以上。上限なし。負なら Env を作る前に `ANET_SYSTEM_ERROR`（キー・指定値・期待「0 以上」を含む） |
| `run.eval_schedule.[tag].warmup_count` | 整数（発火の回数） | 1 | 0 以上。上限なし（発火の総数以上なら、評価は 1 回も走らない）。負なら同上 |

型の読み取りは `interval` と同じ読み口で行い、型の不正の扱いも `interval` と同じにする。`interval = 0` のタグでは、書いてあれば読んで検証し、使わない。`interval` が必須なのは今のまま（D4）。

`warmup_count` の warmup は、Learner の `update_warmup_steps`（学習を始める前に経験をためる期間）とは別のもので、評価の発火の列の先頭から数える回数である。設計文書とユーザーガイドのキーの説明にも、この 1 文を書く。

設定例:

```text
run.@evalonly : run.eval_schedule.[eval].warmup_count = 0
run.eval_schedule.[eval_target].offset = 500
```

### 5.3 設定の書き換え（D7・D8 の値の正本）

| 場所 | 今 | 変えたあと | 決定 |
|---|---|---|---|
| [Atari.txt](../../apps/runner/config/Atari.txt) 本体の 1412・1415〜1416 行 | `[eval_target]` 1000、`[eval]` 1009（注記: 互いに素にして同じ時点を避ける） | `[eval_target]` 1000・offset 500、`[eval]` 1000。注記は「`[eval_target]` を offset 500 でずらして同じ時点を避ける」の趣旨に | D8 |
| Atari.txt の `run.@evalonly`（551〜552 行） | `[eval_target]` 10、`[eval]` 11。`[greedy]`・`[greedy_single]` は組まない | `[eval_target]` 10・offset 0（本体の 500 を打ち消す）、`[eval]` 10、`[greedy]` 10、`[greedy_single]` 10。4 タグに `warmup_count = 0`。`[greedy_single]` は継承に頼らず葉を書く | D7・D8・D9 |
| Atari.txt の 509〜510 行の注記 | 「先頭は learn_step 0 でも発火する。…この 1 回は捨て点になる」 | 消す | D2 |
| Atari.txt の 515・523 行の行末の注記 | 「50M 予算で 0〜45.45M の 10 回」「100M 予算で 0〜90.9M の 10 回」 | 「50M 予算で 5.05〜45.45M の 9 回」「100M 予算で 10.1〜90.9M の 9 回」 | D10 |
| [LunarLander.txt](../../apps/runner/config/LunarLander.txt) の `run.@skip`（90 行） | `[greedy]` 1009 | `[greedy]` 1000・offset 50 | D8 |
| [GridMaze.txt](../../apps/runner/config/GridMaze.txt) の `run.@skip`（70 行） | `[greedy]` 53 | `[greedy]` 50・offset 3 | D8 |
| [DropMerge.txt](../../apps/runner/config/DropMerge.txt) の `run.@skip`（509 行） | `[greedy]` 503 | `[greedy]` 500・offset 25 | D8 |
| [11_batch_run.bat](../../apps/11_batch_run.bat) の `EV2X2`（53 行） | 4 タグ（`[greedy]`・`[greedy_single]`・`[eval]`・`[eval_target]`）interval 1000 | 4 タグに `warmup_count=0`、`[eval_target]` に `offset=0`（Atari 本体の 500 を打ち消す）。間隔はそのまま | D7・D9 |
| [12_batch_run.bat](../../apps/12_batch_run.bat) の `EVONLY`・`GEVONLY`・`GSESBIG`（103〜106 行） | `[eval_target]` 10、`[eval]` 11、`[eval_target_greedy]` 12、`[greedy]`・`[greedy_single]` 12 | 間隔をすべて 10。5 タグに `warmup_count=0`。`[greedy]`・`[greedy_single]` に `offset=0`（DropMerge の `run.@skip` の 25 を打ち消す）。ほかの offset は書かない | D7・D8・D9 |
| 12_batch_run.bat の `TWSES`（287 行） | `[greedy_single]` 1009 | `[greedy_single]` 1000・offset 50（LunarLander の `[greedy]` と同じ時点）。`[eval_single]` 200 はそのまま | D8 |
| [13_batch_run.bat](../../apps/13_batch_run.bat) の `EVTAU`（76 行） | 3 タグ（`[eval]`・`[greedy]`・`[greedy_single]`）interval 10 | 3 タグに `warmup_count=0`。間隔はそのまま | D7 |

bat は Shift-JIS（CP932）・CRLF のまま書き換える（AGENTS.md §改行と文字コード）。

D9 の確かめは、apps/ では `git grep -n -E "run\.eval_schedule\.\[[a-z_0-9]+\]\.interval" -- apps`（設定と bat の全体。コメントの行も含めて当たりを見る）で、検証資材では、キーを組み立てる資材も当たるように `git grep -n -F "eval_schedule" -- core/anet-core/testdata` で行う。2026-11-10 の時点の当たりは、apps/ が 82 行（設定 65・bat 17）、検証資材が 53 ファイル・261 行である。apps/ のうち上の表に載らない行の扱いは次のとおり。検証資材の当たりは、ファイルごとに §5.4 の移行対象か「触らないもの」のどちらかに入れた。

- 間隔を 0 にする行（休止）と、コメントの行: 変えない。
- `[eval]` や `[greedy]` の間隔を予算から決めている行（Atari.txt の `run.@rainbow_like`・`run.@no_munch`・`run.@btreval_r*`・`run.@eval2ch` 系、11・13 の bat の学習 Run の変数）: 変えない。`[eval]` と Atari の `[greedy]` は offset を持たない。`run.@rainbow_like`・`run.@no_munch` では本体の `[eval_target]`（1000・offset 500）が生きたままだが、`[eval]` の 500,000 と 7,816 の倍数は 1000 で割った余りが 500 にならないので、重ならない。
- `run.@skip` の `[greedy]` の間隔を予算から決め直す bat の行（12 の `GSES`・`GSES1K`、13 の `FW10`）: 間隔は変えない。DropMerge では `run.@skip` の offset 25 が残るが、その Run の予算の中で `[eval_target]`・`[eval]` の 100 の倍数と重ならないので書き足さない。
- 同じ間隔で同じ時点に発火する組（12 の bat の `GEV`・`B1024RR1`、`TWSES` の `[eval_single]`、各 env の本体の `[eval_target]`・`[eval]`、common.txt）: 変えない（D8、§8）。
- ImageCls・GridMaze_muzero・LunarLander の `[test1]`・`[test2]`: 評価のタグが 1 本だけか休止なので、変えない。

### 5.4 移行対象と、触らないもの

移行対象（同じ変更で新しい契約へ移す）:

| 種類 | 対象 | 直すこと | 決定 |
|---|---|---|---|
| コード | [observers.hpp](../../core/anet-core/include/anet/observers.hpp)・[observers.cpp](../../core/anet-core/src/observers.cpp) の `EpisodeEvalObserver`、[trainer.cpp](../../core/anet-core/src/trainer.cpp) の `RunManager`（eval schedule の読み込みと scheduled の行） | §5.1〜§5.2 | D1〜D6 |
| 設定・bat | §5.3 の表 | §5.3 | D7〜D10 |
| 検証資材 | [prd084/run_forward_refactor.ps1](../../core/anet-core/testdata/prd084/run_forward_refactor.ps1) の 56・135・164 行と [prd084/run_review_validation.ps1](../../core/anet-core/testdata/prd084/run_review_validation.ps1) の 67 行（`[greedy]` だけを休止にし、ほかの評価を含む metrics を旧い実行ファイルの出力と比べる。手順を書いた [prd084/README.md](../../core/anet-core/testdata/prd084/README.md) の 63 行も合わせる）、[prd104/run_experiments.py](../../core/anet-core/testdata/prd104/run_experiments.py) の 93〜95 行、[prd094/run_validation.py](../../core/anet-core/testdata/prd094/run_validation.py) の 73 行、[prd097/run_validation.py](../../core/anet-core/testdata/prd097/run_validation.py) の 77 行（あわせて [prd097/audit_validation.py](../../core/anet-core/testdata/prd097/audit_validation.py) の 63〜64 行の解決結果の検査）、[prd103/run_current.py](../../core/anet-core/testdata/prd103/run_current.py) の 71〜72 行 | 休止にしない評価のタグに `warmup_count=0` を足し、学習直後の発火を前提にした測定と、新旧の実行ファイルの比較を保つ。古い実行ファイルはこのキーを読まないので、同じスクリプトで新旧を回せる | D7 |
| 変更履歴 | [CHANGELOG.md](../../CHANGELOG.md) の「v0.3.0 以降の非互換変更」の「黙って挙動・値が変わる変更」、「0.4.0: (未リリース)」、「将来バージョンの対応候補」の 469 行 | 前の 2 つに、既定で学習直後の評価を走らせなくなったことと `offset`・`warmup_count` を 1 行ずつ足す。469 行の候補を消す | D1・D2 |
| テスト | [trainer_test.cpp](../../core/anet-core/src/trainer_test.cpp) の 1278・1306 行（scheduled の行の期待の文字列）と `[eval_schedule]` のテスト | 期待の文字列に `offset=`・`warmup_count=` を足す。learn step 0 の発火を前提にした断定があれば直す | D2・D6 |
| テスト | `EpisodeEvalObserver` を直接作るテストの 13 か所: [episode_end_test.cpp](../../core/anet-core/src/episode_end_test.cpp) の 597・772・824・887・961・975・996・1015・1058・1105 行、[observers_test.cpp](../../core/anet-core/src/observers_test.cpp) の 1570・1603・1630 行（検索: `git grep -n -E "EpisodeEvalObserver(>| [a-z_]+\(|\()" -- "core/anet-core/src/*_test.cpp" "apps/runner/src/*_test.cpp"` の、TEST_CASE の行を除いた当たり） | 13 か所すべてを offset 0・`warmup_count` 0 で作る。多くは最初の学習イベントで評価が始まることを確かめており、残り（961 行の foreground、975 行など）も同じ作り方にそろえる | D2・D5 |
| テスト | [config_baseline_test.cpp](../../core/anet-core/src/config_baseline_test.cpp) に、Atari の `run.@evalonly` を重ねた解決結果の断定を足す | §7 の A4 | D7・D8 |
| テスト | [config_baseline_test.cpp](../../core/anet-core/src/config_baseline_test.cpp) の 522 行 | `[greedy]` の間隔 53・1009・503 の断定を、50・1000・500 と offset 3・50・25 の断定に直す | D8 |
| テスト | [RunnerApp_test.cpp](../../apps/runner/src/RunnerApp_test.cpp) の 517〜518 行（hidden の `[runner_close_drain]`、interval 1） | 最初の評価が learn step 1 に移る。通らなければ `warmup_count` 0 を書く | D2 |
| 設計文書 | [140_observability.jp.md](../design/140_observability.jp.md) の 228 行の近く（`IntervalGate` は今のまま初回に発火し、eval schedule は offset と warmup_count で列を決める）と 330 行（scheduled の行の中身）、[100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) の 329 行（`run.eval_schedule.[tag].*` のキーの表）、[010_framework_overview.jp.md](../design/010_framework_overview.jp.md) の 296 行（「学習更新数が interval に達したとき」起動する）、[020_user_guide_run.jp.md](../design/020_user_guide_run.jp.md) の 83〜85 行の例と 509 行の近く（キーの説明）、[200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) の 448 行（評価専用 Run は `warmup_count = 0` を書く） | 新しいキーと既定、評価専用 Run の書き方、`warmup_count` が Learner の warmup と別であること（§5.2）。英語版（`.en.md`）は、日本語版との対応を確かめた範囲に合わせて直す（[実装 品質確認観点](../quality/impl.md) の「検証と報告」の規約） | D1〜D7 |
| 用語集 | [CONTEXT.md](../../CONTEXT.md) の 381 行（eval schedule の定義の「interval / use_background」） | 足さない。用語の定義で、キーの一覧ではないため | — |
| ADR | ADR 0028 の先頭の注記、ADR 0070 の新設 | 本 PRD の工程で書く | D10 |
| wiki | [評価と比較の条件](../wiki/mechanisms/evaluation.md) の「初回に必ず 1 回発火する」、[計測の負荷と比較条件](../wiki/mechanisms/diagnostic_cost.md) の 13〜14 行 | 本 PRD の工程では §2.4 の事実を書き戻し、出典の節の名前を直す（決定はまだ挙動ではないので書かない）。実装のあとに「今の結論」と「分かっていること」の「初回に必ず 1 回発火する」を新しい挙動に直す | — |

旧い挙動を探す検索語は、`learn_step 0`・`learn step 0`・`learn_step=0`・`初回`・`互いに素`・`同位相`・`ShouldFire`・`scheduled (`・`1009`・`53`・`503`・`11`・`12` である。`1009` などの数は、`run.eval_schedule` を含む行に限って引く。

触らないもの:

- 過去の Run の設定の写し（`apps/runner/workspaces/**/runs/`、`docs/experiments/**`）と実験記録、`docs/memo/done/` の PRD・実装記録、CHANGELOG の過去の版の記述。当時の記録である（AGENTS.md §クリーンブレーク方針）。
- [ADR 0045](../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)・[ADR 0027](../adr/0027-eval-definition-schedule-separation.md)。本 PRD はどちらの判断も変えない（0045 の移行の依存が要らなくなるだけ）。
- [util.hpp](../../core/anet-core/include/anet/util.hpp) の `IntervalGate` と、メトリクス側の interval の記法、`IntervalGate` のほかの利用者のコード（D5）。
- [prd072](../../core/anet-core/testdata/prd072/manifest.json)・[prd102](../../core/anet-core/testdata/prd102/manifest.json) の検証資材（`run.@evalonly` の連結を含む）。どちらも、その PRD の作業の前に固めた設定か、保存した基準と比べる資材で、本 PRD の設定の変更で解決結果が変わっても直さない。
- 検証資材のうち、学習直後の発火に頼っていないもの。§5.3 の末尾の検索の当たりのうち、移行対象の行に挙げたもの以外は、次のどれかに当たる。
  - 評価を休止（interval 0）にするもの: [prd084/run_performance.ps1](../../core/anet-core/testdata/prd084/run_performance.ps1)、[prd092/run_validation.py](../../core/anet-core/testdata/prd092/run_validation.py)、[prd093/run_validation.py](../../core/anet-core/testdata/prd093/run_validation.py)、[prd099/run_validation.py](../../core/anet-core/testdata/prd099/run_validation.py)、[prd104/preflight_checkpoints.py](../../core/anet-core/testdata/prd104/preflight_checkpoints.py)・[prd104/run_validation.py](../../core/anet-core/testdata/prd104/run_validation.py)、[prd106/run_validation.py](../../core/anet-core/testdata/prd106/run_validation.py)、[prd116/run_performance.py](../../core/anet-core/testdata/prd116/run_performance.py)。[prd098/run_validation.py](../../core/anet-core/testdata/prd098/run_validation.py) も、等価性を比べる短い条件では全タグを休止にする（67 行の `[eval_target_greedy]` 10000 は長い条件の学習の測定で、新しい既定に従ってよい）。[prd100/run_validation.py](../../core/anet-core/testdata/prd100/run_validation.py) は、ログの Run では update_warmup_steps 100,000 が予算 256 exp step より長く学習イベントが起きず、ほかの Run では評価を休止にする。
  - 休止を検査するか、キーを選り分けるだけのもの: [prd084/summarize_forward_refactor.py](../../core/anet-core/testdata/prd084/summarize_forward_refactor.py)・[prd084/summarize_performance.py](../../core/anet-core/testdata/prd084/summarize_performance.py)、[prd103/verify_config_migration.py](../../core/anet-core/testdata/prd103/verify_config_migration.py)。
  - 過去の Run の設定の写しと、作業の前に固めた解決結果: [prd092/config/](../../core/anet-core/testdata/prd092/config/) の 11 本、[prd072](../../core/anet-core/testdata/prd072/manifest.json) の baseline と resolution。
  - 未定義のタグのエラーを確かめるもの: [prd109/run_validation.py](../../core/anet-core/testdata/prd109/run_validation.py) の 142 行（`[nosuch]`）と [prd109/README.md](../../core/anet-core/testdata/prd109/README.md)。
  - PRD の番号の対応表: [prd114/migration.json](../../core/anet-core/testdata/prd114/migration.json)。
- 課題の単票の題と事象に書かれた当時の名前（`greedy_dist` など）。

### 5.5 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場 |
|---|---|---|---|
| `EpisodeEvalObserver` | 変更 | eval schedule の offset と warmup_count を持つ。learn step が offset に届くまではゲートに問い合わせず、届いたら offset を引いた値で問い合わせ、先頭の warmup_count 回の発火では評価を走らせない | 318 強化学習 共通基盤 |
| `RunManager` | 変更 | `run.eval_schedule.[tag]` の offset・warmup_count を読み、検証し、scheduled の行に出して observer へ渡す | 318 強化学習 共通基盤 |

### 5.6 issue の対応表

| issueのID | 題 | area | 現在のコードで確認した根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261103-a86dcc](../management/issues/issue-20261103-a86dcc.md) | eval スケジュールの初回発火の抑止と、発火の時点を offset でずらすこと | 318_rl_core, 312_util | core/anet-core/include/anet/util.hpp:287、core/anet-core/src/observers.cpp:624、core/anet-core/src/trainer.cpp:960、apps/runner/config/Atari.txt:1415 | D1〜D10 | A1〜A8 |

offset（ゴール 2、D1・D8・D9）の仕事も、この課題が覆う（2026-11-10 に開発者の指示で題と範囲を改めた）。

## 6. 細部

細部はユーザーが「推奨で埋める」を選んだ。値の正本は §5.3 で、ここには決め方だけを書く。

1. Atari 本体では offset を `[eval_target]` に持たせ、`[eval]` は 0（D8 の理由）。
2. `run.@skip` の `[greedy]` の offset は、刻み（§2.1）より十分離し、`[eval_target]`・`[eval]` の格子（LunarLander 200・GridMaze 10・DropMerge 100 の倍数）と同じ学習イベントで発火しないように選ぶ（50・3・25）。間隔が小さい GridMaze では端数になる。`[greedy_single]` は `[greedy]` と同じ値にする（D8）。
3. 素数の間隔は、近いきりのよい数（1000・50・500）に直す。
4. 評価専用 Run は、今組んでいるタグのセッションの回数を保つ（`run.@evalonly` の `[eval_target]`・`[eval]` は各 `max_exp_step / 10` 回、12 の bat は各タグ 4 回、11・13 の bat は各タグ 1 回）。`run.@evalonly` に足す `[greedy]`・`[greedy_single]` も各 `max_exp_step / 10` 回になる（D7）。間隔は 10 にそろえ、offset は付けない。本体か `run.@skip` の offset が残るタグには `offset = 0` を書く。
5. 継続 Run には `warmup_count = 0` を書かない（D7）。
6. Atari の Greedy の間隔は変えず、注記の点の数を直す（D10）。
7. 評価を走らせなかった発火ごとのログは出さない（D6）。
8. scheduled の行の書式は D6 のとおり。
9. テストと短い Run の受入は §7 のとおり。

## 7. 受入

### A1. `IntervalGate` を変えない

`git diff <実装前のコミット> -- core/anet-core/include/anet/util.hpp` が空で、[util_test.cpp](../../core/anet-core/src/util_test.cpp) の `[interval_gate]` の既存の 4 件（203・222・239・296 行）が、変えずに通る。

### A2. `EpisodeEvalObserver` の列

fake の EvalRunner で学習イベントを送り、評価セッションの始まった learn step を記録して、§5.1 の表どおりかを比べる。入力は次のとおり。

- interval 10、learn step 0 から刻み 1: 既定（offset 0・warmup_count 1）では最初が 10、warmup_count 0 では 0、offset 5・warmup_count 1 では 15。
- interval 1000: offset 0・500・2500 と warmup_count 0・1・3 の組み合わせ。刻み 1・刻み 7・刻み 1500（interval を超える）。
- 最初の学習イベントが offset を過ぎている（例: offset 500 で最初が learn step 2700）。
- warmup_count が発火の総数より多い（評価が 1 回も走らないまま学習イベントが終わる）。

### A3. 設定の読み込み

trainer_test の `[eval_schedule]` で、次を確かめる。

- 負の offset と負の warmup_count が、Env を作る前に `ANET_SYSTEM_ERROR` で止まる。メッセージにキー名・指定値・`0 以上` の期待が入る。
- `interval = 0` のタグに 2 つのキーがあっても、エラーにならない。負なら止まる。
- scheduled の行が D6 の書式で、書いていないタグでは `offset=0, warmup_count=1` になる。

### A4. 出荷する設定

config_baseline_test で、`_main.txt` と各 env の設定を解決した結果について、§5.3 の表の設定ファイルの行の interval・offset・warmup_count を断定する。Atari は本体の既定の連結と、それに `run.@evalonly` を重ねた連結（`[greedy_single]` の葉を含む）、LunarLander・GridMaze・DropMerge は `run.@skip` を含む連結である。

### A5. bat の評価専用 Run

`iconv -f CP932 -t UTF-8` で読んだ 11・12・13 の bat で、変数 `EV2X2`・`EVONLY`・`GEVONLY`・`GSESBIG`・`EVTAU`・`TWSES` の定義の行（11 の 53 行、12 の 103〜106 行・287 行、13 の 76 行）を見て、§5.3 の表の値が書かれている。評価専用の 5 つの変数では、`run.eval_schedule.[<tag>].interval=` を書いているタグごとに、同じ行に `run.eval_schedule.[<tag>].warmup_count=0` がある。

### A6. 短い Run

LunarLander の既定（`run.@skip` を含む）を `app.batchrun.exp_exit_step=300000` で 1 本流す。Run ログで次を確かめる。

- scheduled の行に、`[eval_target]`・`[eval]` は `offset=0`、`[greedy]` は `offset=50` と、3 タグとも `warmup_count=1` が出る。
- 各タグの最初の `session end` の learn_step が、そのタグの offset + interval 以上である（`[eval_target]`・`[eval]` は 200、`[greedy]` は 1,050）。
- `[greedy]` の `session start` の learn_step が、`[eval_target]`・`[eval]` の `session start` のどれとも一致しない。

300,000 exp step は、warmup の 8,192 exp step（[LunarLander.txt](../../apps/runner/config/LunarLander.txt) の 130 行）を除いて、1 学習イベントあたり 256 exp step・刻み 2 で learn step 約 2,280 にあたり、`[greedy]` の 2 回目（learn step 2,050）が入る長さである。Run が予算まで走らずに止まったら、受入は不合格として原因を調べる。

### A7. ほかの利用者

既定のテストスイートが通る。既知の失敗（PRD093 の attention のテスト 1 件）は、本 PRD の前後で変わらないことを確かめる。

### A8. 旧い記述の検索

`git grep -n -E "互いに素|同位相" -- apps/runner/config` と、`git grep -n -F "learn_step 0" -- apps` が 0 件になる。`git grep -n -E "run\.eval_schedule\.\[[a-z_]+\]\.interval *=? *(1009|53|503|11|12)\b" -- apps` が 0 件になる（コメントの行も含む。11・12 は評価専用 Run の互いに素の間隔）。

## 8. 範囲外・先送り

| 事項 | 区分 | 理由と、何が揃ったら扱うか |
|---|---|---|
| 同じ間隔で同じ時点に発火する組に offset を当てること（各 env の本体の `[eval_target]`・`[eval]`、common.txt、12 の bat の `GEV`・`B1024RR1`・`TWSES` の `[eval_single]`、Atari の `run.@rainbow_like` の `[eval]` と `[greedy]`） | 実験の結果が要る | ずらして分けられるのは始まる時点だけで、学習の待ちが減るかを測っていない（§1.3、D8）。本 PRD の実装のあと、DropMerge の Run で `[eval_target]` に offset 50 を当てた群と当てない群の `waited for previous session` の回数と学習の実時間を比べ、減ると分かったら当てる。そのとき `GEV` の `[eval_target_greedy]` の offset は、DropMerge の `run.@skip` の `[greedy]`（500・offset 25）と 1025 などで重ならない値（例: 75）にする |
| Atari.txt の 505〜508 行の注記「間隔は予算 / 10 より大きく取る（最後の発火が捨てられる）」 | この場で直した（2026-11-10） | PRD076 の排水で最後のセッションは捨てられなくなった（ADR 0045）ので、同じ 4 行を「終了時に完走を待ってから閉じる（`wait_on_exit` 既定 true、`app.drain_timeout_sec` 既定 3600 秒）。間隔を予算 / 10 より大きく取る必要は無い」に書き換えた（[Atari.txt](../../apps/runner/config/Atari.txt) の 505〜508 行）。行数は変えていない。Greedy の間隔と点の数は変えない（D10） |
| 背景の評価が前回を待つ代わりに、次の発火を見送る案 | 別の関心 | 待ち方の関心で、発火の列とは別。DropMerge の確かめ（1 行目）で停止が残ったら起こす |

## 9. 影響

- 書かない Run は、どの env でも、各タグの評価が 1 回（学習直後の 1 回）減る。評価のセッションの足し引きで同じ seed の学習の軌跡が変わるので（wiki「評価と比較の条件」）、本 PRD のあとの Run は、前の同じ seed の Run と、学習の初めから軌跡が変わりうる。新旧の実行ファイルを決定的モードで突き合わせる比較は、新しい設定で取り直す。
- Atari 本体の `[eval_target]`（1000 の倍数から 1500, 2500, …へ）と `[eval]`（1009 の倍数から 1000 の倍数へ）、`run.@skip` の `[greedy]`（例: LunarLander は 1009 の倍数から 1050, 2050, …へ）の点の位置が動く。点の位置での対比較は、step で並べ直す。
- Atari の ε=0 の評価の系列は、5.05M（`run.@eval2ch`）か 10.1M（`run.@eval2ch_r1`）から始まる。旧い Run と k 番目の点で対にするときは、新しい Run の k 番目を旧い Run の k+1 番目と対応させる。実験記録の 9 点の対の表は learn_step=0 の点を使っていないので、作り方は変わらない。
- ADR 0045 の Consequences にある「初回発火の抑止が入るまでの移行の依存」は、要らなくなる。
- PRD118（Atari の Run プロファイルの整理、暫定）は `run.@eval2ch` 系と本体の評価の設定を置き換える。どちらが先に実装されても、あとの側が §5.3 の Atari の行を新しい置き場へ移す。

## 10. 未決

なし。

## 11. 参照

- [BTR] VIPTankz. BTR, main.py. [commit 093ca77 の 299〜300 行](https://github.com/VIPTankz/BTR/blob/093ca77f08540d646f12fd942ee5fda763e0fd8a/main.py#L299-L300)（`eval_every = 250000`、`next_eval = eval_every`）、[377 行](https://github.com/VIPTankz/BTR/blob/093ca77f08540d646f12fd942ee5fda763e0fd8a/main.py#L377)（`steps >= next_eval`）。同じ commit の手元の写しで確かめた。
- [SB3] DLR-RM. Stable-Baselines3, [callbacks.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py)（`BaseCallback.on_step`、`EvalCallback.__init__`・`_on_step`）。2026-11-10 に master を参照。
- [DI-Engine] OpenDILab. DI-engine, [serial_entry.py](https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry.py)（`serial_pipeline` のループの先頭の `should_eval(learner.train_iter)`）、[interaction_serial_evaluator.py](https://github.com/opendilab/DI-engine/blob/main/ding/worker/collector/interaction_serial_evaluator.py)（`eval_freq=1000`、`_last_eval_iter = -1`、`should_eval`）、[base_learner.py](https://github.com/opendilab/DI-engine/blob/main/ding/worker/learner/base_learner.py)（`CountVar(init_val=0)`）、[evaluator.py（middleware）](https://github.com/opendilab/DI-engine/blob/main/ding/framework/middleware/functional/evaluator.py)、[context.py](https://github.com/opendilab/DI-engine/blob/main/ding/framework/context.py)（`last_eval_iter: int = -1`）。2026-11-10 に main を参照。
- [ADR 0028](../adr/0028-interval-fires-on-bucket-crossing.md)、[ADR 0070](../adr/0070-eval-schedule-skips-warmup-fires-by-default-and-shifts-with-offset.md)、[ADR 0045](../adr/0045-eval-session-drain-on-exit-wait-or-discard.md)、[ADR 0027](../adr/0027-eval-definition-schedule-separation.md)。
