# 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD

> 2026-10-19 起票（暫定。グリル前）。ユーザーの希望で、999 ではなく正式番号で起票した。
> 2026-10-19 グリル（`/anet-create-prd 097`、Claude Code）で M2 に確定した。M2 は、BatchEnv の `Step` はリセットせず、Runner が通知のあとに `ResetEndedEpisodes()` を毎回呼ぶ形である（ADR 0059）。
>
> 2026-10-20 再グリル（`/anet-create-prd 097`、Claude Code）で全面改訂した。開き直した理由は 2 つある。
> - M2 の実装で、Atari の非同期の学習 Runner の速さが改修前より 5〜8% 落ち、受入 4 に届かなかった（§3.8）。Atari 以外の env も落ちていた。
> - ユーザーの指摘: 遷移の結果を Step の結果ではなく env 本体から取ろうとすること自体が、ずれの根である。
>
> 主な決定は次のとおり。
> - 遷移の結果は Step の結果で運び、env への問い合わせは今の env の状態を返すものとする（D1）。
> - 操作を Reset と Step の 2 つに戻し、自動リセットは Step の中で行う（D2）。
> - 速報値か確定値かは、読むイベントで分け、鍵は分けない（D3）。Atari は、残機を失ったがゲームが続く step だけ、ゲーム単位の鍵を NaN にする。
> - 派生・引数つきの鍵は、Step の結果が問い合わせに答える（D5）。
>
> 起草後に別モデル（Fable）の検討を受け、次を反映した。
> - `$env_step` の `@episode_end` の scalar を 1 step に 1 回だけ読むこと（EMA を同じ step で何回も更新しない）。
> - step 軸を明示して移すこと。
> - 非同期の学習 Runner の最初と最後の step を比較から外すこと。
> - Atari の速報値の時点（NOOP の前）。
> - GridMaze の `reward_sum`。
> - 094 への申し送り。
>
> あわせて、W・S1 と Atari 以外の env の追加の計測の結果を §3.8 に足し、受入 4 に判定できない構成の扱いを足した。
>
> 決定の記録は [ADR 0060](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md) で、[ADR 0059](../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)・[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)・[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)・[ADR 0046](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md) の先頭に注記を足した。ファイル名は、ほかの文書と wiki の作業記録が指しているので変えない。
> 起点: [PRD 094](094_evalpanel_step_history_10prd.md)（Evaluation Trace）のグリルで、step trace の列 `$env` を「遷移のあと・リセット前」の env と定義したところ、今の BatchEnv では終端の lane だけがリセット後になると分かったこと（094 の §3.3、D4、§7 の 1）。ユーザーの裁定で、この直しは別の PRD とし、094 より先に実施する。
> 関連: [094_evalpanel_step_history_10prd.md](094_evalpanel_step_history_10prd.md)、[081_evalpanel_episode_trace_10prd.md](081_evalpanel_episode_trace_10prd.md)、[999_qvaluepanel_manual_action_state_alignment_10prd.md](999_qvaluepanel_manual_action_state_alignment_10prd.md)、[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)、[ADR 0046](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、[ADR 0059](../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)、[120_environments.jp.md](../design/120_environments.jp.md)、[140_observability.jp.md](../design/140_observability.jp.md)、[220_atari_env.jp.md](../design/220_atari_env.jp.md)、[CONTEXT.md](../../CONTEXT.md) の「episode scope」「episode group」「評価セッション」「trace」「速報値／確定値」。

## 1. 目的と問題

BatchEnv の `Step` は、lane ごとに env を 1 step 進め、終端（done か truncated）になった lane をその場で `Reset` する。通知（`TrainEvent`・`EpisodeEndEvent`）の読み手は、遷移の結果（得点・残機・画面・完了値）を env 本体へ問い合わせて読んでいる（`GetScalar`・`GetTensor`）。そのため、終端の lane だけがリセット後の値を読む。env は完了値をリセットをまたいで持ち越して、これを埋め合わせている（§3.4）が、残機や画面のような今の値は直らない。

2026-10-19 の決定（M2）は、通知が終わるまで env をリセットさせない形で直した。しかし、リセットが学習と重ならなくなり、遅くなった（§3.8）。ずれの根は、イベントの時点の事実（遷移の結果）を、そのあとも動き続ける env 本体に問い合わせていることにある。遷移の結果はイベントのデータとして Step の結果で運ぶのが正しい。

### ゴール

- **G1**: 通知の中で読む遷移の結果は、どの lane でも、その遷移の時点（リセット前）の値になる。遷移の結果は Step の結果で運ぶ。
- **G2**: 速さは、改修前の流れ（リセットを Step の中で、学習と重ねて行う）を保つ。
- **G3**: 学習の系列を変えない。学習の系列とは、Agent と replay に入るもの、loss・Q、episode の return と長さである。env のメトリクスは、設定を移したあとも同じ意味の系列になる。変わってよい系列は、受入で先に挙げる（§7）。

### 非ゴール

- **NG1**: replay と Agent が扱う遷移（`next_state` はリセット前の終端の観測）の意味を変えること。
- **NG2**: env の `GetScalar`・`GetTensor` を、遷移の結果に合わせて直すこと。env への問い合わせは、今の env の状態を返すもののままにする。遷移の結果は Step の結果から読む。
- **NG3**: 定期評価の採用 episode の規則（ADR 0034・0037）を変えること。
- **NG4**: エピソードの開始（リセット直後の s_0）を Observer に知らせる通知を足すこと（§5）。

## 2. 背景

- ユーザーの説明（2026-10-19）: 自動リセットが要るようになったとき、`Step` の中でリセットまで行えば、既存の流れとも単一の env とも同じ形を保て、`Reset` と `Step` の 2 つだけという単純さも保てた。
- ユーザーの方向（同日）: 欲しいのは step の結果なので、リセットの前に Observer を呼ぶのが正しい。この方向を受けて M2 を選んだ。
- ユーザーの指摘（2026-10-20）: 速度の観点では修正前の流れがよい。また、env の Step の結果を StepResult ではなく env 本体から取ろうとすること自体が矛盾している。そのうえで、score を例に次の方向を示した: 鍵ではなく取得のタイミング（イベントの種類）で切り分けるべきで、ステップ時点の速報値なのか、エピソード終了時点の確定値なのか。イベントの間で鍵を共有するか個別にするか、NaN を返すかは、実装ごとに決めてよい。

## 3. 現状（2026-10-19〜20 にコードと計測で確かめたこと）

コードの行番号は、コミット済みの HEAD のものである。作業ツリーには、M2 の実装がまだコミットされずに入っている（§3.8）。

### 3.1 BatchEnv と Step の結果

- 単一の env をまとめる BatchEnv は 2 つある。逐次の `VectorizedDiscreteBatchEnv`（[env.cpp:697-742](../../core/anet-core/src/env.cpp)）と、スレッドの `ThreadPoolDiscreteEnv`（同 830-890）。
  - どちらも lane ごとに `envs_[i]->Step(a)` を呼び、done か truncated なら続けて `envs_[i]->Reset()` を呼ぶ。
  - `next_state` にはリセット前、`continue_state` にはリセット後の状態を詰める。
  - lane ごとの Step の結果（aux を含む）は `single_results` として持つ（[env.cpp:424-447](../../core/anet-core/src/env.cpp)）。リセットの aux は捨てている。
- 単一の env の結果の型（`SingleEnvResult`）は `GetAuxData()` だけを持つ（[rl.hpp:355-381](../../core/anet-core/include/anet/rl.hpp)）。env の型ごとの派生がある（`AtariStepResult` など。[AtariEnv.cpp:40-46](../../core/envs/atari1/src/AtariEnv.cpp)）。
- 自前でバッチを持つ `ImageClsEnv` は、`Step` の中で次のバッチを先読みし、エピソードの終わりで `continue_state` の `episode_start` を立てる（[ImageClsEnv.cpp:139-190](../../core/envs/imagecls1/src/ImageClsEnv.cpp)）。

### 3.2 Runner の通知の時機

- EvalRunner: `MakeAction` → `Step` → `TrainEvent` → `EpisodeEndEvent` → `state_ = continue_state`（[trainer.cpp:231-298](../../core/anet-core/src/trainer.cpp)）。
- 評価セッション（`RunSession`）: 採用 episode の `EpisodeEndEvent` は、Step の処理を抜けたあとに `LastAdoptedGroups()` から出す（同 315-342）。
- 逐次の学習 Runner: `MakeAction` → `Step` → `EpisodeEndEvent` → 学習 → `LearnEvent` → `TrainEvent` → `state_ = continue_state`（同 509-566）。
- 非同期の学習 Runner（既定。[common.txt:4](../../apps/runner/config/common.txt)）: step t の `TrainEvent` は、次の DoStep の頭で学習の待ち合わせのあとに出す。`EpisodeEndEvent(t)` は Step の直後に出す（[trainer.cpp:634-700](../../core/anet-core/src/trainer.cpp)）。
- `EpisodeEndEvent` は runner・counts・agent・env・env_index だけを持ち、Step の結果は持たない（[rl.hpp:952-958](../../core/anet-core/include/anet/rl.hpp)。通知は [trainer.cpp:96-103](../../core/anet-core/src/trainer.cpp) で、終わった group ごとに出す）。`TrainEvent` は Step の結果（`step_result`）を持つ。

### 3.3 評価セッション

- `EvalSessionEnv` は中の BatchEnv を包み、採用 episode が完了した Step のあとで、購読した鍵を中の env の `GetScalar` で読んで溜める（[env.cpp:336-390](../../core/anet-core/src/env.cpp)）。`@session_end` の scalar は、その溜めた集約を `EvalSessionEnv::GetScalar(key, -1)` で読む（同 392-402）。
- セッションの継ぎ目では、前の Step の `continue_state` が全 group とも新しいエピソードなら、`Reset` を呼ばずに使い回す（同 317-334、ADR 0034）。

### 3.4 env の完了値と持ち越し

- **Atari**
  - aux には途中の値（`game_score`・`game_len`・`game_frames`・`lives`・`ram_metric.<ラベル>`）を毎 step 入れる（[AtariEnv.cpp:682-694](../../core/envs/atari1/src/AtariEnv.cpp)）。
  - `GetScalar` の完了の鍵（`game_score`・`game_len`・`game_frames`・`hns57`・`hns49`・`game_score.ge.[N]`・`ram_metric.[n]`）は、完了時に写した `completed_*` から、完了した Step のあとだけ返す（同 696-717、839-891）。
  - `Reset` の中でもゲームの完了を記録することがある。残機を失った直後の NOOP でゲームオーバーになった場合である（同 725-733）。
  - `lives` は今の値。`ram_metric.[n]` の番号からラベルへの対応と、hns の基準表は env が持つ。
- **DropMerge**
  - 完了の鍵（`ep_*`・`term_reason_*`・`blocked_drop_on_candidate` など）は、`GetScalar` が `last_*` の写しやエピソードの統計から、完了した Step のあとだけ返す（[DropMergeEnv.cpp:1556-1670](../../core/envs/dropmerge1/src/DropMergeEnv.cpp)）。
  - aux は途中の値と、View の「前回の結果」の表示用の `last_score`・`last_step`・`last_term_reason` を持つ（同 1534-1547、[DropMergeView.cpp:447-500](../../core/envs/dropmerge1/src/DropMergeView.cpp)）。
- **GridMaze**: 完了の鍵（`reward_sum`・`wall_collisions`・`is_success`・`is_hole`）は `last_*` の写しから返す（[GridMazeEnv.cpp:233-273](../../core/envs/gridmaze1/src/GridMazeEnv.cpp)）。aux は位置などの途中の値（同 94-107）。
- **ImageCls**: `accuracy`（直近に確定した採点の snapshot）と `epoch_count`（累積）は env 全体の値で、index −1 だけを受け付ける（[ImageClsEnv.cpp:196-206](../../core/envs/imagecls1/src/ImageClsEnv.cpp)）。採点の区切りで `Step` の中で更新され、リセットでは変わらない。

### 3.5 env を読む読み手

- `$env` の scalar（`@train`）は、`TrainEvent` の中で BatchEnv の `GetScalar` を lane 方向に集約して読む（[observers.cpp:1078-1088](../../core/anet-core/src/observers.cpp)、[env.cpp:617-640](../../core/anet-core/src/env.cpp)）。間隔の既定は 1（[observers.cpp:1437](../../core/anet-core/src/observers.cpp)）。そのため、完了の鍵の `@train` の宣言は、その step で完了した lane があるときだけ値が出る。
- `@episode_end` の trace は、`EpisodeEndEvent` の中で env の `GetScalar(key, lane)` を読む（[observers.cpp:837-872](../../core/anet-core/src/observers.cpp)、ADR 0037）。
- AtariView は、終端の通知で env の `rgb_frame`（リセット後）を出す。一緒に出す得点・残機・RAM は、Step の結果の aux（リセット前）から取る（[AtariView.cpp:218-257](../../core/envs/atari1/src/AtariView.cpp)）。
- View は `TrainEvent` だけを受ける（[gui.hpp:259](../../core/anet-core/include/anet/gui.hpp)）。Run の最初のエピソードの s_0 は、今も画面に出ない。

### 3.6 乱数

env の乱数は env ごとに持ち、torch の共有の乱数は使わない（`core/envs` に `torch::rand` などは無い）。リセットを Step の中で行っても、学習の乱数の系列は変わらない。

### 3.7 `$env` を読む既存の宣言

宣言は 4 つのファイルにある（Atari: `@train` 27・`@session_end` 70・`@episode_end` 4、DropMerge: 30・44・0、GridMaze: 8・0・0、ImageCls: 3・3・0。[Atari.txt](../../apps/runner/config/Atari.txt) 1259-1374 行、[DropMerge.txt](../../apps/runner/config/DropMerge.txt) 1271-1390 行、[GridMaze.txt](../../apps/runner/config/GridMaze.txt) 161-169 行、[ImageCls.txt](../../apps/runner/config/ImageCls.txt) 551-557 行）。性格は次のとおり。

| 区分 | 該当 | 性格 |
|---|---|---|
| 学習側の完了値 | Atari の game_score・game_len・game_frames（8 行）、DropMerge（30 行）、GridMaze（8 行） | その step で完了した lane だけ値がある、遷移の結果 |
| 学習側の遷移のあとの状態 | Atari の lives（2 行） | 遷移のあとの今の値 |
| 派生・引数つきの鍵 | Atari の hns57・hns49（4 行）、game_score.ge.[432]・[600]（4 行）、ram_metric.[1〜3]（9 行） | 完了値に env の規則（基準表、閾値の引数、番号からラベルへの対応）を当てて計算する値 |
| `@episode_end` の trace | Atari の 4 行 | 1 本のエピソードの完了値 |
| 評価セッションの集約 | Atari 70 行、DropMerge 44 行、ImageCls 3 行（`@session_end`） | 採用 episode の値を、セッション全体でまとめた二次の値 |
| env 全体の状態 | ImageCls の accuracy・epoch_count（`@train`、3 行） | lane に属さず、遷移ごとの結果でもない。リセットで変わらない |

Atari.txt の `mean.lives` の 7 本は、改修前はリセット後の残機を読んでいる（`42_env/30・31` は終端の lane があった step、`51_eval1/30・31`・`52_eval2/30・31`・`53_evalg/30` は全部の値）。

### 3.8 M2 の実装と計測（2026-10-20）

M2 の実装は Codex が行い、まだコミットしていない。作業記録は 097_env_auto_reset_after_notification_20impl.md。

- **等価性**: 7 構成で全点が一致した（残機の 7 本は点の数と step）。
- **実時間（Atari）**: 非同期の学習 Runner（Breakout、128 lane、`@classic`）で、改修前と交互に測った。改修前比（中央値の比）は世代ごとに 0.917・0.919・0.958・0.947 で、受入 4 の 2% に届かなかった。
- **原因**: 学習は 2 step に 1 回だけ起きる（`IQNLearner::UpdateFromSamples` は 500 step で 249 回、1 回 33.6 ms）。
  - 改修前は、リセットが、学習を待つ時間と Step の中の並列に隠れていた。
  - M2 では、段（1 step あたり約 4 ms）が学習の待ちのあとに足され、ゲームの作り直し 1 回（約 6.4 ms）の遅れがそのまま表に出る。
- **試作との同じ組の比較**: before・M2 の実装・S1・W の 4 つを 1 組にし、交互に 3 組測った。
  - S1: 予備の ALE で次のゲームを先に用意する試作。
  - W: リセットを Step の中に戻し、外から読む値だけを段まで写しで出す試作。
  - 結果:

    | 側 | 改修前比 | 起動 | メモリ（Private） |
    |---|---|---|---|
    | M2 の実装 | 0.947 | 10.1 s | 改修前と同じ |
    | S1 | 0.971 | 15.6 s（予備の ALE の生成 5.7 s） | +88 MiB |
    | W | 0.985 | 10.1 s | 改修前と同じ |

  - W の段に残る仕事は状態の複製（0.15 ms）だけで、リセットは Step の中で走る。
  - 後半の実時間の約 53% は同期の評価で、exp/s の差はそのままの割合では実時間に出ない。
- **実時間（Atari 以外。改修前と M2 の実装を交互に）**:
  - GridMaze: 0.955。3 組ともそろって落ちた。段が 1 step 0.44 ms かかり、thread pool の往復の待ちの回数が 2 倍になっていた。
  - DropMerge（逐次）: 0.977。
  - DropMerge（非同期）: 0.972（1 組に外れ値）。
  - LunarLander: 0.839。ただし判定できない。改修前の 5 本が約 4250 と約 4900 の 2 つの山に分かれた（ばらつき 16%）。
  - M2 の段は、Atari 以外でも固定費の分だけ遅くしていた。

## 4. 確定事項

導入する機構とゴールの対応:

| 機構 | 寄与するゴール |
|---|---|
| Step の結果に遷移の結果を載せ、Step の結果が鍵に答える（D1・D5） | G1 |
| Reset と Step の 2 つに戻し、自動リセットを Step の中で行う（D2） | G2 |
| 速報値と確定値を読むイベントで分ける（D3）、`$env_step` の読み方（§4.3.2） | G1・G3 |
| `EpisodeEndEvent` に Step の結果を持たせる・評価セッションが Step の結果から溜める（§4.3.3・§4.3.4） | G1・G3 |
| env の `GetScalar` から、完了したときだけ値を返す鍵を外す（§4.3.5） | NG2 の線引き（env への問い合わせは今の状態） |

### 4.1 決定

**D1 遷移の結果は Step の結果で運ぶ。env への問い合わせは、今の env の状態を返すものとする。**
- 通知の読み手（メトリクス、trace、評価セッションの溜め込み、View）は、遷移の結果を Step の結果から読む。
- env の `GetScalar`・`GetTensor` は、今の env の状態（リセット後もあり得る）を返す。遷移の時点との一致は保証しない。

理由: 遷移の結果はイベントの時点の事実なので、イベントのデータとして運べば、リセットがいつ起きても値は変わらない。2026-10-19 の M2（通知の間 env を止める）、検討した「外から読む値の写し」（W）、予備の ALE（S1）は、どれも、動き続ける env にイベントの時点の値を答えさせるための埋め合わせだった（ユーザーの指摘、2026-10-20）。
依存: §3.5（読み手の一覧）、§3.8（M2 の計測）。
既存 ADR との関係:
- ADR 0059（M2）を取り消す。
- ADR 0034 の継ぎ目は、`continue_state` の使い回しに戻る。
- ADR 0037 の「trace の読み手は完了 Step の直後に `GetScalar(key, env_index)` を読む」は、`EpisodeEndEvent` が持つ Step の結果から読む形に変わる。イベントの型の分け方と trace チャネルは変えない。
- ADR 0046 の「AtariEnv が `ram_metric.[n]` を返す」は、Step の結果が答える形に変わる。番号付きの汎用の鍵、ゲーム別の RAM 定義、集計の単位（実ゲーム 1 回）は変えない。0046 の「確定のタイミングと NaN の慣行は `game_score` と同じ」は保ち、その `game_score` の慣行が D3・§4.3.6 で変わる。0046 が「NaN の規則を変えるときは新しい ADR で行う」としている新しい ADR は ADR 0060 である。
- ADR 0057 は「時点が決まっている target だけを読む」で、step trace の `$env` の扱いは 094 で直す（§8）。

**D2 BatchEnv の操作を Reset と Step の 2 つに戻す。**
- 自動リセットは改修前と同じく `Step` の中で行い、`continue_state` を Step の結果に戻す。
- 2026-10-19 の決定のうち、`ResetEndedEpisodes()`・呼ぶ順番の fail-fast・検査の分割・`EvalSessionEnv` の段の結果の使い回し・ImageCls の段は、M2 の実装から元の形に戻す。

理由:
- 段は「通知の間 env を止める」ための仕組みで、D1 のもとでは守るものが無い。
- 改修前の流れに戻れば、リセットが学習の待ちと Step の中の並列に重なり、速さが戻る（G2）。段の固定費（毎 step の thread pool の往復、状態の組み立て）も無くなる。
- ユーザーが元々守りたかった単純さに戻る（§2）。

退けた案: 段を残し、中身は Step の中で済ませたリセットの結果を返すだけにする。097 で直した箇所を多く残せるが、段が守るものが無く、使われない境目と呼ぶ順番の規則が残る。
依存: D1、G2、§3.8。

**D3 速報値か確定値かは、読むイベントで分ける。**
- 同じ鍵でも、`@train` で読めばその step の途中の値（速報値）、`@episode_end` で読めばエピソードの確定値になる。
- env は、Step の結果に、その遷移の時点の値を載せる。
- 鍵ごとの次の 2 点は、env の実装が決める。
  - 鍵をイベントの間で共有するか、イベントごとに別の鍵にするか。
  - どんなときに NaN を返すか。
- そのうえで、次の 3 つを守る。
  1. 決めたことを、env の設計文書の鍵の表に書く。鍵ごとに、`@train` と `@episode_end` で読んだときの意味と、NaN の条件を書く。専用の設計文書が無い env（DropMerge・GridMaze）は、120_environments.jp.md に節を足して書く。
  2. 知らない鍵は nullopt、知っている鍵で値が成り立たないときは NaN とする（AGENTS.md §GetScalar 実装ルール）。
  3. 設定の行を、env の選択に合わせて同じ変更で移し、系列が一致することを受入で確かめる。

理由: 値の意味（途中か確定か）は、読むイベントで決まる性格のものである（ユーザーの指摘、2026-10-20）。鍵の名前で区別すると、Atari の aux の `game_score`（途中の値）と `GetScalar("game_score")`（完了したときだけ値を返す値）のように、同じ名前で意味が分かれる（§3.4）。
用語: 「完了値」の語は、鍵の名前で区別していた旧来の仕組みを指すので、契約では使わない。CONTEXT.md に「速報値／確定値」を足した。
env の確定の単位がフレームワークのエピソードより長いとき:
- イベントだけでは確定を表せない。Atari の学習側で `episodic_life = true` のときは、エピソードは残機 1 つ分で、ゲーム 1 回より短い。
- この場合も鍵は分けない。ゲーム単位の鍵は、「エピソードは終わったがゲームは続いている step」だけ NaN にする（§4.3.6）。
- そうすれば、`@episode_end` はゲームが終わった step だけを数え、`@train` は途中の値を読める（その step だけ欠ける）。
- 鍵を分ける案（確定値に `game_end.` を付ける）は、Step の結果に速報値と確定値の 2 組を持つ作りになり、設定の鍵の名前も書き換えることになる。欠ける速報値を読む宣言は今は無いので、退けた（ユーザーの判断、2026-10-20）。

依存: D1、G3。

**D4 リセット後の情報は、Runner だけが `continue_state` として受け取る。Observer への通知は足さない。**
- リセット後の情報とは、次のエピソードの最初の観測・画面 s_0 である。
- View は、終端の step では Step の結果の画面 s_T を出す。手動で進めるときは、エピソードの境目で s_0 を見ずに最初の行動を選ぶことになる。Run の最初のエピソードの今の扱い（§3.5）と同じである。

理由: 「エピソードの開始」の通知は、ゴールに効かない機能の追加である。また、境目で何を画面に出すかは、[999_qvaluepanel_manual_action_state_alignment](999_qvaluepanel_manual_action_state_alignment_10prd.md) と同じ関心である（§5）。
依存: NG4。

**D5 派生・引数つきの鍵は、Step の結果が問い合わせに答える。**
- 単一の env の Step の結果の型（`AtariStepResult` など）が `GetScalar(鍵)` に答え、派生の鍵（hns57・hns49・`game_score.ge.[N]`・`ram_metric.[n]`）もそこで計算する。
- 上書きしない env の Step の結果は、aux のスカラーの鍵をそのまま返す。
- BatchEnv の Step の結果は `GetScalar(鍵, lane)` で、lane の結果に問い合わせる。
- 設定の鍵の書き方は今と同じ（例: `M1.[42_env/40_game_score_ge432] = $env_step mean.game_score.ge.[432] @episode_end $exp_step`）。

理由: 計算の規則を env の中に 1 つだけ置け、aux は増えない。`.ge.[N]` の N も今どおり任意に書ける。
退けた案: hns と番号つきの ram_metric を固定の鍵で aux に載せ、`.ge.[N]` は DSL の汎用の演算にする。aux が毎 step・毎 lane 増え、DSL に演算が増える。汎用の閾値の判定は [999_scalar_threshold_rate](999_scalar_threshold_rate_10prd.md) の関心として別に扱う。
依存: D1、D3、§3.7。

**D6 Atari の残機喪失直後の NOOP は `Step` の中で打つ。速報値は NOOP の前の値、ゲームが終わった step の確定値は NOOP のあとの値にする。**
- NOOP を打つ条件は今と同じである。`episodic_life = true` で、その step で実ゲームは終わっておらず、残機が減って 0 より大きく、そのため done になった step（今の `life_loss_pending_` の条件。[AtariEnv.cpp:787-798](../../core/envs/atari1/src/AtariEnv.cpp)）だけで打つ。`episodic_life = false` の step と、実ゲームが終わった step では打たない。
- その step では、観測と画面を作ったあとで NOOP を 1 フレーム打つ。それでゲームオーバーになれば、その `Step` の中でゲームの終わりを記録する。
- Step の結果の速報値（`lives`、途中の得点・フレーム数・RAM。aux と同じ値）は、NOOP の前の値、つまり遷移の時点の値にする。
- ゲームが終わった step（NOOP でゲームオーバーになった場合を含む）では、ゲーム単位の鍵は、ゲームの確定値（NOOP の報酬とフレームを含む）を返す。これは改修前の完了値と同じ値である。
- リセットは、ゲームの作り直しか FIRE だけを行う。ALE に渡るフレームの列と、乱数を消費する順番は今と同じ。

理由:
- リセットは Step の中に戻るが、`Reset` の中で記録したゲームの終わりは、その step の Step の結果に載らない。NOOP を Step の中で打てば、確定値を Step の結果で運べる（G1・G3）。
- 速報値を NOOP の前にするのは、遷移の時点の値という G1 の意味を守るためである（ユーザーの判断、2026-10-20）。改修前も、aux は `Step` の中で作られ、NOOP は `Reset` の中で打つので、aux の値は NOOP の前である（[AtariEnv.cpp:682-694](../../core/envs/atari1/src/AtariEnv.cpp)・725-733）。aux の値は今と変わらない。
- NOOP でゲームが終わる経路は `episodic_life = true` の学習側にしか無く、その step のゲーム単位の鍵は確定値を返すので、速報値との衝突は起きない（D3）。

依存: D1、D3、G3、[AtariEnv.cpp:725-733](../../core/envs/atari1/src/AtariEnv.cpp)。

**D7 学習の系列が変わらないことは、決定的モードでの完全一致で確かめる。**
決定的モード（[common.txt:98-100](../../apps/runner/config/common.txt) の `backend.@deterministic`）を使い、次の順で確かめる。

1. 改修前のビルドで同じコマンドを 2 回走らせ、一致することを先に確かめる。
2. 改修前と改修後で、次が完全に一致することを確かめる。
   - scalar の全点（tag・step・値のビット列）
   - trace の全行（tag・step・lane・data）
3. 比べる対象から外すのは、実時間の tag（`90_perf/`・`91_resource/`・`92_resource/`）と trace の `timestamp` だけ。
4. Atari の `mean.lives` の 7 本（§3.7）は、点の数と step が同じことだけを確かめる。
5. `@episode_end` へ移した tag は、非同期の学習 Runner の最初と最後の step の点を比較から外す。非同期の学習 Runner はこの 2 つの step の `TrainEvent` を出さない（§3.2）。そのため、その step に終わった lane があると、改修後にだけ点が出ることがある。

設定の行を `$env_step` へ移しても、tag の名前は変えないので、tag ごとに比べられる。

理由: G3 から、改修前後の値はビットまで一致するはずである。改修前どうしが一致しない構成では、ずれの原因を切り分けられない。
前例: [PRD 084 の比較スクリプト](../../core/anet-core/testdata/prd084/compare_metrics.py)。trace の比較は足す。
依存: G3。

### 4.2 規約・ADR・コードから導いたこと

1. **クリーンブレークで移す。** 今使っている設定・テスト・文書を同じ変更で移し、別名や互換の分岐は残さない。過去の Run の記録は書き換えない。コミット済みの ADR は本文を変えず、先頭に注記を足す（根拠: [AGENTS.md](../../AGENTS.md) §クリーンブレーク方針）。
2. **scalar の `$env_step` の点の打ち方。** `@train` では全 lane を、`@episode_end` ではその step で終わった lane を、lane 方向に集約し、step ごとに 1 点にする。lane ごとの個々の値は trace が持つ（根拠: [ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md) の「scalar は集約、個体は trace」。今の `@train` の宣言も、間隔の既定 1 でこの点の出方になっている、§3.5）。
3. **`EpisodeEndEvent` は、そのエピソードが終わった Step の結果を持つ。** `@episode_end` の trace と scalar はそこから読む。評価セッションの `RunSession` が出す採用 episode の通知も同じにする（根拠: D1、§3.2）。
4. **評価セッションは、採用 episode が終わった Step の結果から値を溜める。** `@session_end` の宣言（`$env`）と、その読み口（`EvalSessionEnv::GetScalar(key, -1)`）は変えない（根拠: D1、ADR 0034）。
5. **AtariView は `rgb_frame` を Step の結果から読む**（根拠: D1、§3.5）。
6. **env の `GetScalar` から、Step の結果へ移した鍵（完了したときだけ値を返していたもの）を外す。** 自動リセットのもとでは意味を保てず、読む所も無くなるためである（根拠: クリーンブレーク方針、D2）。今の状態を返す鍵（Atari の `lives`、LunarLander の `wind_*`、ImageCls の `accuracy`・`epoch_count`）と、`EvalSessionEnv` の集約は残す。
7. **Atari のゲーム単位の鍵は、ゲーム 1 回の確定値の系列を保つ。** 鍵の名前は今のまま、NaN の条件で扱う（D3・§4.3.6）。条件は 220_atari_env.jp.md の鍵の表に書く（根拠: D3、G3）。
8. **メトリクスの `$env_step` は、094 の step trace の `$env_step` と同じ target とする。** Step の結果の欄（`reward`・`done`・`truncated`）と、Step の結果が答える鍵を読む。鍵が無い、またはスカラーでなければ、最初に読んだときに止める（根拠: [PRD094 の D5](094_evalpanel_step_history_10prd.md)）。
9. **メトリクスへの `$env_step` の追加は、本 PRD で行う。** 081 で予定していた追加のうち、残るのは `$agent_step` だけになる（根拠: G1、PRD094 §7 の 2）。
10. **Step の結果の値にも、NaN・nullopt の決まりを当てる。** 知らない鍵は nullopt、知っている鍵で値が成り立たないときは NaN。集約は NaN を分母から外す（根拠: [AGENTS.md](../../AGENTS.md) §GetScalar 実装ルール）。
11. **リセットの aux は捨てる（今のまま）。** `continue_state` と一緒に Runner が受け取る情報は、状態だけで足りる（根拠: D4、§3.1）。
12. **`$env_step` の `@episode_end` の scalar は、1 step に 1 回だけ読む。** 対象は `$env_step` の scalar に限る。ほかの target（`$runner`・`$agent`・`$actor`・`$env`）の `@episode_end` の scalar は今のまま、通知のたびに読む（リポジトリの設定に宣言が無く G3 に響かない。範囲を広げると、本 PRD の関心でない target の意味を変えることになる）。`EpisodeEndEvent` は終わった lane ごとに出る（[trainer.cpp:96-103](../../core/anet-core/src/trainer.cpp)）。今の読み口は、値を取って EMA を更新してから間隔の判定を見る（[observers.cpp:1110-1127](../../core/anet-core/src/observers.cpp)）。そのため、同じ step に 2 lane 以上が終わると、EMA が何回も更新されてしまう。同じ step の 2 つ目以降の通知では値を読まず、EMA も更新しない。終わった lane は `env_index` ではなく、Step の結果の done・truncated から取る（根拠: G3 の系列の一致、§4.2 の 2）。
13. **step 軸を書いていない行は、軸を明示して移す。** 軸の既定は `@train` なら train_step、`@episode_end` なら exp_step である（[observers.cpp:1524-1536](../../core/anet-core/src/observers.cpp)）。そのまま移すと座標が変わるので、軸を省略していた行（DropMerge の 28 行、GridMaze の 8 行）には `$train_step` を書いて移す（根拠: G3）。
14. **lane 方向の集約は、lane の昇順で、今と同じ集約の部品に入れる。** ビットまで一致させるためである。Welford の計算は、加える順番で値が変わる（[util.cpp:39-60](../../core/anet-core/src/util.cpp)）。今の `DiscreteBatchEnvBase::GetScalar` は lane 0 から順に加えている（[env.cpp:633-639](../../core/anet-core/src/env.cpp)）（根拠: D7）。
15. **`$env_step` を `@learn` か `@session_end` で使ったら、構築のときに止める。** `LearnEvent` は Step の結果を持たず、セッションは 1 つの step に属さないためである（根拠: AGENTS.md §Fail-Fast 原則）。
16. **外した鍵を `$env` で読む古い宣言は、今の汎用の仕組みのまま扱う**（未知の鍵は WARN を出して進む。[observers.cpp:1120-1123](../../core/anet-core/src/observers.cpp)）。特別な検出は足さない。リポジトリの設定は同じ変更で移す（根拠: [AGENTS.md](../../AGENTS.md) §クリーンブレーク方針の「現行契約に含まれない識別子を特別扱いするためだけのコードは追加しない」）。
17. **定義レコードの `target` に `env_step` が増える。** 読む側（`inspect_run.py` の `TARGET_TOKENS`、Viewer の target の検証）も同じ変更で移す（根拠: クリーンブレーク方針、[observers.cpp:1268-1280](../../core/anet-core/src/observers.cpp)）。

### 4.3 実装契約

#### 4.3.1 BatchEnv と Step の結果

- `BatchEnv` の操作は `Reset` と `Step`。`Step` の中で終端の lane を自動リセットし、`BatchStepResult` は `reward`・`next_state`・`continue_state`・`n_transitions`・`n_episode_end` を持つ（改修前の契約）。エピソード構造の検査は、改修前と同じく `continue_state.episode_start == (done || truncated)` を見る。
- 単一の env の結果（`SingleEnvResult`）に、鍵の問い合わせを足す。

```cpp
class SingleEnvResult {
public:
    virtual AuxData GetAuxData() const { return AuxData(); }
    // Step（または Reset）の時点の値を答える。既定は aux のスカラーの鍵。知らない鍵は nullopt
    virtual std::optional<float> GetScalar(const std::string& key) const;
    ...
};
```

- `BatchStepResult` は `GetScalar(鍵, lane)` を持つ。wrapper（`DiscreteBatchEnvBase` の結果）は、lane の単一の env の結果に問い合わせる。自前でバッチを持つ env（ImageCls）は、自分で答える。`SHARED` の env は lane −1 でバッチの値に答える。
- Step の結果は、作られたあとで変わらない。読み手は、通知のあとにも同じ値を読める。
- 鍵の読み出しで aux を複製しない（G2 のため）。今の `GetAuxData()` は値返しなので、結果の型はメンバか参照から答える。終わった lane の mask は、1 step に 1 回だけ CPU へ移す。

#### 4.3.2 `$env_step` の読み方（メトリクスと trace）

| 宣言 | 読む値 | 点 |
|---|---|---|
| scalar `$env_step <集約>.<鍵> @train` | その step の全 lane の `step_result->GetScalar(鍵, lane)` を集約する | step ごとに 1 点 |
| scalar `$env_step <集約>.<鍵> @episode_end` | その step で終わった lane だけを集約する | step ごとに 1 点（終わった lane が無い step には出さない） |
| trace `$env_step <鍵> ... @episode_end` | 終わった lane の `step_result->GetScalar(鍵, lane)` | lane ごとに 1 行（今の trace と同じ） |
| scalar `$eval.[x] @session_end $env <集約>.<鍵>` | 評価セッションの集約（変えない） | セッションごとに 1 点 |

- 集約の接頭辞（`mean.`・`max.`・`min.`・`std.`）は今の `$env` と同じに扱う。NaN は分母から外す。lane は昇順で、今と同じ集約の部品に入れる（§4.2 の 14）。
- `$env_step` の `@episode_end` の scalar は 1 step に 1 回だけ読み、EMA も 1 回だけ更新する（§4.2 の 12）。
- `@episode_end` の scalar の step 座標は、`EpisodeEndEvent` の counts（今の `@train` の宣言と同じ値）とする。軸の既定は exp_step なので、train_step で記録する行は `$train_step` を明示する（§4.2 の 13）。
- `$env_step` を `@learn` か `@session_end` で使うと、構築のときに fail-fast にする（§4.2 の 15）。
- `$env` は、今の env の状態を読む target として残す（NG2）。`@episode_end` で `$env` を読むと、終わった lane はリセット後の今の状態になる。遷移の結果は `$env_step` で読む。

#### 4.3.3 `EpisodeEndEvent`

- そのエピソードが終わった Step の結果（`step_result`）を持たせる。lane の番号（`env_index`）は今のまま。
- 逐次・非同期の学習 Runner と EvalRunner は、Step の直後に出す通知に、その Step の結果を入れる。
- `RunSession` は、採用 episode の通知に、その採用 episode が終わった Step の結果を入れる。

#### 4.3.4 評価セッション（`EvalSessionEnv`）

- 採用 episode が完了した Step のあとで、購読した鍵を、その Step の結果の `GetScalar(鍵, group)` から溜める（`SHARED` では −1）。中の env の `GetScalar` は読まない。
- `@session_end` の集約の読み口と、採用の規則は変えない（NG3）。
- セッションの継ぎ目は、改修前と同じく前の Step の `continue_state` を使い回す（ADR 0034 の元の判断）。

#### 4.3.5 env の `GetScalar` から外す鍵

| env | 外す鍵（Step の結果へ移す） | 残す鍵（今の状態） |
|---|---|---|
| Atari | `game_score`・`game_len`・`game_frames`・`hns57`・`hns49`・`game_score.ge.[N]`・`ram_metric.[n]` | `lives`、`GetTensor("rgb_frame")` |
| DropMerge | `ep_*`・`term_reason_*`・`blocked_drop_on_candidate`・`no_drop_timeout_on_candidate` | なし |
| GridMaze | `reward_sum`・`wall_collisions`・`is_success`・`is_hole` | なし |
| LunarLander | なし | `wind_x`・`wind_torque`、`GetTensor`・`GetTensorVector` |
| ImageCls | なし | `accuracy`・`epoch_count` |

外した鍵のための持ち越し（Atari の `completed_*`、DropMerge・GridMaze の `GetScalar` 用の `last_*`）も外す。DropMerge の View が使う aux の `last_score`・`last_step`・`last_term_reason` は、表示の機能として残す。

#### 4.3.6 Atari

- `Step` の中身（D6）:
  1. フレームを進める。
  2. 観測・`rgb_frame`・速報値（aux と同じ値）を作る。これは NOOP の前の時点である。
  3. D6 の条件の step（`episodic_life = true`・実ゲームは続いている・残機が減って 0 より大きい）なら NOOP を打つ。それでゲームオーバーなら、ゲームの終わりを記録する。
  4. ゲームが終わった step なら、ゲームの確定値（NOOP の報酬とフレームを含む）を Step の結果に入れる。
- `AtariStepResult` が持つもの:
  - その Step の速報値（得点・長さ・フレーム数・残機・RAM metric の値）。
  - そのエピソードが終わったか、ゲームが終わったか。
  - ゲームが終わった step では、ゲームの確定値。
  - `retain_rgb_frame` のときは `rgb_frame` の参照。
- `GetScalar(鍵)` は、派生の鍵（hns の基準表、`.ge.[N]`、番号からラベルへの対応）を計算する（D5）。
- ゲーム単位の鍵（`game_score`・`game_len`・`game_frames`・`hns57`・`hns49`・`game_score.ge.[N]`・`ram_metric.[n]`）は、鍵の名前を今のまま共有し、次のように答える（D3）。

  | その step | ゲーム単位の鍵が返す値 |
  |---|---|
  | エピソードが終わっていない | 速報値（NOOP の前） |
  | エピソードは終わったが、ゲームは続いている（`episodic_life = true` で残機を失った） | NaN |
  | ゲームが終わった（実ゲームの終わり・打ち切り・NOOP でのゲームオーバー） | ゲームの確定値 |

  - `lives` はゲーム単位の鍵ではないので、どの step でも NOOP の前の値を返す。
  - `episodic_life = false`（評価・EvalPanel・v5 系）では、エピソードの終わりは必ずゲームの終わりなので、NaN は出ない。
  - 上の表を、220_atari_env.jp.md の鍵の表に書く。
- `Reset`（Step の中の自動リセット）は、ゲームの作り直しか FIRE だけを行い、最後に `current_lives_` を ALE から読み直す（今と同じ）。
- ゲームの終わりのログの行は、`Step` の時点の確定値で出す（今と同じ値）。

#### 4.3.7 DropMerge・GridMaze・ImageCls

- **DropMerge・GridMaze**: Step の結果の型が、§4.3.5 で外した鍵に答える。値は、エピソードが終わった step の統計と、その step の途中の値である。鍵の共有と NaN の条件は各 env の実装が決め、120_environments.jp.md に足す節の鍵の表に書く（D3）。
  - 平均のように組み立てる値は、今と同じ式・同じ順番の float の演算で作る（D7 のビットの一致のため）。
  - DropMerge の View 用の aux は、今のまま残す。
- **ImageCls**: Step の結果が、採点の区切りの step で確定した `accuracy` と、その時点の `epoch_count` に答える。評価セッションは、ここから溜める（§4.3.4）。学習側の `$env accuracy @train`・`$env epoch_count @train` は、env 全体の状態として `$env` のまま残す（§3.7）。

#### 4.3.8 設定の移行

| 宣言 | 移し方 |
|---|---|
| Atari・DropMerge・GridMaze の `$env ... @train` のうち、完了したときだけ値を返す鍵の行（§3.7 の「学習側の完了値」「派生・引数つきの鍵」） | `$env_step ... @episode_end` へ。鍵の名前は今のまま（例: `$env mean.game_score @train $exp_step` → `$env_step mean.game_score @episode_end $exp_step`） |
| Atari の lives の 2 行（`42_env/30・31`） | `$env_step mean.lives @train` へ |
| `@episode_end` の trace（Atari の 4 行） | `$env` を `$env_step` に替える |
| 全 env の `@session_end` の行、ImageCls の `@train` の行 | 変えない |

- tag の名前、EMA などの指定は変えない。
- step 軸を書いていない行（DropMerge の 28 行、GridMaze の 8 行）は、`@episode_end` へ移すときに `$train_step` を書き足す（§4.2 の 13）。

#### 4.3.9 M2 の実装から戻すもの（移行の対象）

作業ツリーの M2 の実装（未コミット）は、次のとおり扱う。

- **改修前の形に戻すもの**:
  - `BatchEnv`・`BatchStepResult` の段と `continue_state` の扱い
  - `DiscreteBatchEnvBase`・`VectorizedDiscreteBatchEnv`・`ThreadPoolDiscreteEnv` の段
  - `BatchEnvBase` の呼ぶ順番の検査
  - `ValidateEpisodeStructure` の分割
  - `EvalSessionEnv` の段の結果の使い回し
  - Runner 3 つの段の呼び出し
  - ImageCls の段
  - テスト用の BatchEnv の段
  - 設計文書（120・010・100・150・220）の段の記述
- **今回の決定に合わせて直すもの**:
  - Atari の NOOP の位置（D6）、持ち越しの撤去（§4.3.5）
  - §4.3.5 で外した鍵を env の `GetScalar` で読む既存のテスト（`AtariEnv_test.cpp`・`DropMergeEnv_test.cpp`・`GridMazeEnv_test.cpp` など）を、Step の結果から読む形へ移す
  - `$env` の宣言を使うテスト（`observers_test.cpp`・`episode_end_test.cpp`・`trainer_test.cpp`）のうち、外した鍵を読むものを `$env_step` へ移す
- **新しく足すもの**:
  - 単一の env の結果の `GetScalar`、`BatchStepResult::GetScalar`
  - `EpisodeEndEvent` の Step の結果
  - `$env_step` の scalar・trace の読み方
  - 評価セッションの溜め込みの元
  - AtariView の読み口
  - 各 env の Step の結果の型
  - 設定の移行（§4.3.8）
  - 定義レコードの target `env_step` を読む側（`inspect_run.py`・Viewer。§4.2 の 17）

### 4.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 |
|---|---|---|
| `SingleEnvResult`（rl.hpp） | 変更 | Step・Reset の時点の値を `GetScalar(鍵)` で答える（既定は aux のスカラー） |
| `BatchStepResult`（rl.hpp）・wrapper の結果（env.cpp） | 変更 | `GetScalar(鍵, lane)` で lane の結果に問い合わせる |
| `BatchEnv`・`BatchEnvBase`・`DiscreteBatchEnvBase`・`VectorizedDiscreteBatchEnv`・`ThreadPoolDiscreteEnv` | 変更 | M2 の段を外し、改修前の自動リセットと `continue_state` に戻す |
| `ValidateEpisodeStructure`（env.hpp の関数） | 変更 | 改修前の 1 つの検査に戻す |
| `EvalSessionEnv` | 変更 | 採用 episode の値を Step の結果から溜める。継ぎ目は `continue_state` の使い回しに戻す |
| `EpisodeEndEvent` | 変更 | 終わった Step の結果を持つ |
| `RunnerBase`・`EvalRunner`・`SerialTrainRunner`・`PipelineTrainRunner` | 変更 | 段の呼び出しを外し、`EpisodeEndEvent` に Step の結果を入れる |
| scalar・trace の読み口（`MetricsLogObserverBase`・`MetricsLogEpisodeEndObserver`・`MetricsLogTraceObserver`・解析の `ObserverFactory`） | 変更 | `$env_step` を読む（§4.3.2） |
| `AtariStepResult`・`AtariEnv` | 変更 | Step の時点の値と派生の鍵に答える（§4.3.6 の NaN の条件）。NOOP の位置（D6）。env の `GetScalar` からゲーム単位の鍵と `completed_*` を外す |
| `AtariView` | 変更 | `rgb_frame` を Step の結果から読む |
| DropMerge・GridMaze の Step の結果の型と env | 新規・変更 | §4.3.5 の鍵に答える結果の型を足し、env の `GetScalar` からその鍵と `last_*` を外す |
| `ImageClsEnv` の Step の結果 | 変更 | 採点の区切りの step の `accuracy`・`epoch_count` に答える。M2 の段を外す |
| テスト用の BatchEnv | 変更 | M2 の段を外す |

## 5. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| エピソードの開始（s_0）を Observer に知らせる通知と、View の開始の表示 | 先送り | ゴールに効かない（D4）。境目の画面は [999_qvaluepanel_manual_action_state_alignment](999_qvaluepanel_manual_action_state_alignment_10prd.md) と同じ関心で、その PRD で手動の時点のずれを決めるときに作る |
| `.ge.[N]` を DSL の汎用の演算にすること | 別 PRD | D5 で Step の結果が答える。env を問わない閾値の判定は [999_scalar_threshold_rate](999_scalar_threshold_rate_10prd.md) で扱う |
| `$agent_step` のメトリクスへの追加 | 081 のまま | G1 に要らない |
| 段（`ResetEndedEpisodes()`）、外から読む値の写し（W）、予備の ALE（S1） | 不採用 | D1・D2 |

## 6. 未決

なし。

## 7. 成功の測り方と受入基準

成功の測り方:

- 通知の中で読む遷移の結果（Step の結果）が、終端の lane でもリセット前の値になる。
- 速さが、改修前の流れと同じになる（受入 4）。
- 学習の系列が変わらない（D7）。
- PRD 094 の step trace の `$env_step` が、終端の行でも遷移の時点の値になる。

受入基準:

1. **等価性**（D7）: 次の 7 本を、改修前は 2 回、改修後は 1 回、同じコマンドで走らせる。step 数と seed は実装計画で固定する。
   - Run の組:
     - Atari 1: 非同期の学習 Runner、定期評価のセッションを含む、`AtariEnv.episodic_life = true`・`fire_reset = true`（`@classic` の組み合わせ）
     - Atari 2: `episodic_life = true`・`fire_reset = false`（`@100k` の組み合わせ）
     - DropMerge
     - LunarLander の `run.@cpu`
     - GridMaze
     - ImageCls
     - `run.train.runner_type = serial` を CLI で渡した DropMerge
   - 各 Run の `config/config_data.txt` で、`AtariEnv.episodic_life` と `fire_reset` の値を確かめる。
   - 合格の条件は D7 のとおり。
2. **単体テスト**
   - (a) scalar の `$env_step` の集約が、`@train` では全 lane、`@episode_end` ではその step で終わった lane だけになり、step ごとに 1 点になる。同じ step に 2 lane が終わっても、`$env_step` の `@episode_end` の scalar の EMA は 1 回だけ更新される。
   - (b) `EpisodeEndEvent` が、終わった Step の結果を持つ。逐次・非同期・評価の Runner と、`RunSession` の採用 episode の通知で確かめる。
   - (c) 評価セッションが Step の結果から値を溜め、`@session_end` の集約が改修前と一致する。
   - (d) Atari の派生・引数つきの鍵を Step の結果から計算でき、改修前の env の値と一致する。`episodic_life = true` では、ゲーム単位の鍵が §4.3.6 の表のとおりに答える（残機を失ったがゲームが続く step は NaN、ゲームが終わった step は確定値、それ以外は速報値）。`lives` はどの step でも NOOP の前の値になる。
   - (e) Atari で、残機を失った直後の NOOP でゲームが終わる経路を通る試験がある。その Step の結果に、ゲームの確定値（NOOP の報酬とフレームを含む）が載ることを確かめる。経路の作り方（ROM の操作の列か、ALE の RAM の書き換えなど）は実装計画で決める。どの作り方でも通せないと分かったら、受入を満たしたとせず、PRD の改訂に戻る。
   - (f) env の `GetScalar` から外した鍵が、未知の鍵（nullopt）になる。
   - (g) Step の結果は、通知のあとに読んでも同じ値を返す。
   - (h) `$env_step` を `@learn` か `@session_end` で宣言すると、構築のときに止まる。
3. **既存のテスト**: 移したテストを含め、`[.]`（隠し）を除く全テストが通る。
4. **実時間**: 次の構成で、改修前と改修後を交互に各 3 回走らせる。
   - 構成: Atari 1 と同じ構成、GridMaze、LunarLander（`run.@cpu`）、DropMerge（非同期・逐次）。
   - 1 本の値: その Run の `90_perf/12_exp_step_per_sec` の点のうち、`exp_step` が最後の値の半分を超えるものの中央値。
   - 合格の条件: 構成ごとに、改修後 3 本の中央値の、改修前 3 本の中央値からの低下が 2% 以内。改修前の 3 本のばらつき（最大と最小の差）が 2% を超えた構成は、5 組に増やす。
   - 5 組にしても改修前のばらつきが 2% を超える構成は、合格とも不合格ともせず「判定できない」とする（2026-10-20 の計測で、LunarLander の改修前が 2 つの山に分かれたため。§3.8）。
     - 判定できない構成は受入 4 の合否から外す。ほかの構成がすべて合格なら、受入 4 を満たしたとし、実装を完了できる。
     - 判定できない構成については、その結果と、ばらつきの原因を調べた結果を実装の記録に書く。
     - 原因が分かって安定させられたら、測り直して判定する。原因が分からない、または安定させられないときは、調べた結果を記録したうえで、測り直しを起票の候補として示して終える。
5. **手動の確認**: EvalPanel の Atari（`episodic_life = false`、[Atari.txt](../../apps/runner/config/Atari.txt) 1406 行）で、最後の残機を失ってゲームが終わった step の画面が、ゲームオーバーの瞬間の画面になる（改修前は、新しいゲームの最初の画面になる）。

## 8. 依存と順序

- 順序: 本 PRD → PRD 094（段階 1・2）→ PRD 081 の改訂。
- PRD 094 は、本 PRD に合わせて直す。
  - step trace の `$env` は、遷移の時点の値を返さない target になる（D1）。ADR 0057 の「時点が決まっている target だけを読む」に従い、列の target から `$env` を外し、遷移の結果は `$env_step` で読む。
  - `$env_step` を、本 PRD のメトリクスの target と同じものにする（§4.2 の 8）。
- PRD 081 は、メトリクスの DSL への `$env_step` の追加を本 PRD に譲り、`$agent_step` だけを扱う。
- [999_qvaluepanel_manual_action_state_alignment](999_qvaluepanel_manual_action_state_alignment_10prd.md) の案 A（次の状態の `MakeAction` を通知のあとに先に行う）を採る場合、次の状態は `continue_state` から取る。

## 9. 文書同期（実装と同じ変更で更新する）

| 対象 | 変更 |
|---|---|
| [120_environments.jp.md](../design/120_environments.jp.md) | §2.2 の Step の結果（遷移の結果は Step の結果で運ぶ、`GetScalar`）。M2 の段の記述を、改修前の自動リセットに戻す |
| [120_environments.jp.md](../design/120_environments.jp.md) の DropMerge・GridMaze の節（新設） | Step の結果の鍵と、`@train`・`@episode_end` で読んだときの意味・NaN の条件の表（D3）。env の `GetScalar` に残す鍵 |
| [140_observability.jp.md](../design/140_observability.jp.md) | `$env_step` の target（`@learn`・`@session_end` では止まる）、速報値と確定値（読むイベント）、`$env_step` の `@episode_end` の scalar を 1 step に 1 回読むこと、`@episode_end` の `$env` は今の状態を読むこと、`EpisodeEndEvent` の Step の結果、評価セッションの溜め込みの元、定義レコードの target `env_step`。`@episode_end` の trace の例（244 行）の `$env` を `$env_step` に替える |
| [220_atari_env.jp.md](../design/220_atari_env.jp.md) | 鍵の表（230-237 行）を、Step の結果の鍵と、`@train`・`@episode_end` で読んだときの意味・NaN の条件の表（§4.3.6）に改める。env の `GetScalar` に残す鍵の表。198 行の soft-reset の NOOP の位置（D6） |
| [010_framework_overview.jp.md](../design/010_framework_overview.jp.md)・[100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md)・[150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) | M2 の実装で変えた段の記述を、改修前の形に戻す |
| `docs/archify/anet_lab_20_training_step_sequence` | M2 の実装で変えていれば戻す。アトラスの再生成で直す |
| [CONTEXT.md](../../CONTEXT.md) | 「終わったエピソードのリセット」を外し、「速報値／確定値」を足す（本 PRD の起草で反映済み） |
| [ADR 0059](../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md)・[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)・[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)・[ADR 0046](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md) | 先頭の注記（本 PRD の起草で反映済み） |
| 設計文書の英訳（`*.en.md`） | 手では直さない。翻訳の手順（`anet-translate-docs`）が追随する |
