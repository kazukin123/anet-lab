# 終わったエピソードのリセットを通知のあとへ — 通知の間は env を「遷移のあと・リセット前」に保つ PRD

> 2026-10-19 起票（暫定。グリル前）。ユーザーの希望で、999 ではなく正式番号で起票した。
> 2026-10-19 グリル（`/anet-create-prd 097`、Claude Code）で全面改訂した。主な決定は次のとおり。
> - M2: BatchEnv の `Step` はリセットせず、Runner が通知のあとに `ResetEndedEpisodes()` を毎回呼ぶ。
> - ゴール 3 の言い直し:「metrics checksum を変えない」から、「学習の系列は変えず、終端の lane があった step で env の今の値を読むメトリクスだけは変わってよい」へ。
> - 完了値をリセットのあとまで持ち越す仕組みを外す。ただし、DropMerge の View が出す前回値の写しは残す。
> - リセット後の情報は Runner だけが受け取る。
> - Atari の残機喪失直後の NOOP を `Step` へ移す。
> - 受入は、決定的モードでの改修前後の完全一致で確かめる。
>
> 起草後に別モデル（Fable）の検討を受け、次を反映した。
> - 評価セッションでは、採用 episode の `EpisodeEndEvent` を出してから段を呼ぶ。
> - `Step` の直後の全体の `Reset()` を fail-fast にする（D8）。
> - 受入の Atari を残機制の設定で走らせる。
> - §3.7 の残機が変わる lane を正す。
> - 移行の対象に Atari の残機の試験と `rl.hpp` のコメントを足す。
>
> レビュー（an-review）を受け、次を直した。
> - 移行の対象のテストを全数で挙げた。
> - まだ何も呼んでいない状態の `Step()` を fail-fast にした。
> - 受入 2 の (e2)、受入 4・5 を、合否が一つに決まる書き方にした。
>
> 工程の終わりに、ユーザーが選んだ候補として、受入 2(b) に「まだ何も呼んでいない状態の段」を足した。あわせて、PRD 094 の 097 を待つ記述と、999_qvaluepanel_manual_action_state_alignment に、この PRD の決定を書き足した。
>
> §3 の現状をコードと照合して書き足した。決定の記録は [ADR 0059](../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md) で、[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md) の先頭に注記を足した。ファイル名は、ほかの文書と wiki の作業記録が指しているので変えない。
> 起点: [PRD 094](094_evalpanel_step_history_10prd.md)（Evaluation Trace）のグリルで、step trace の列 `$env` を「遷移のあと・リセット前」の env と定義した。そのとき、今の BatchEnv では終端の lane だけがリセット後になると分かった（094 の §3.3、D4、§7 の 1）。ユーザーの裁定で、この直しは別の PRD とし、094 より先に実施する。
> 関連: [094_evalpanel_step_history_10prd.md](094_evalpanel_step_history_10prd.md)、[081_evalpanel_episode_trace_10prd.md](081_evalpanel_episode_trace_10prd.md)、[999_qvaluepanel_manual_action_state_alignment_10prd.md](999_qvaluepanel_manual_action_state_alignment_10prd.md)、[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)（trace は `@episode_end` の通知の中で env の `GetScalar` を読む）、[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)、[120_environments.jp.md](../design/120_environments.jp.md)、[CONTEXT.md](../../CONTEXT.md) の「episode scope」「episode group」「評価セッション」「終わったエピソードのリセット」。

## 1. 目的と問題

BatchEnv の `Step` は、lane ごとに env を 1 step 進め、終端（done か truncated）になった lane をその場で `Reset` する。そのため、`Step` が戻って通知（`TrainEvent`・`EpisodeEndEvent`）が届いた時点では、終端の lane の env はもう次のエピソードになっている。通知の中で env へ問い合わせる読み手（`GetScalar`・`GetTensor`）は、終端の lane だけリセット後の値を読む。

これを埋め合わせるために、env は完了したエピソードの値をリセットをまたいで持ち続け（§3.4）、エピソード完了の読み手はそれに頼っている。一方、残機のような「今の値」は、終端の通知では次のゲームの値になる。EvalPanel の Atari の画面も、終端の step だけリセット後の画面になる。

### ゴール

- **G1**: 通知（`TrainEvent`・`EpisodeEndEvent`）の間、どの lane の env も「遷移のあと・リセット前」の状態にある。通知の中の `GetScalar`・`GetTensor` は、終端の lane でもその遷移の結果を返す。
- **G2**: 次の判断に使う状態は、通知のあとで得る。
- **G3**: 学習の系列を変えない。学習の系列とは、Agent と replay に入るもの、loss・Q など学習のメトリクス、エピソードの統計と完了値である。変わってよいのは、終端の lane があった step で env の「今の値」を読むメトリクスだけで、変わるのはこの PRD が直す食い違いそのものである（今の設定では Atari の `mean.lives` を読む 7 本。§3.7）。

### 非ゴール

- **NG1**: replay と Agent が扱う遷移（`next_state` はリセット前の終端の観測）の意味を変えること。
- **NG2**: env の「完了したときだけ値を返す」という `GetScalar` の仕様を変えること。エピソード完了の統計はこれに依っている。
- **NG3**: 定期評価の採用 episode の規則（ADR 0034・0037）を変えること。
- **NG4**: エピソードの開始（リセット直後の s_0）を Observer に知らせる通知を足すこと（§5）。

## 2. 背景: なぜ今の形か

ユーザーの説明（2026-10-19）: 自動リセットが要るようになったとき、`Step` の中でリセットまで行えば、既存の流れとも単一の env とも同じ形を保てた。`Reset` と `Step` の 2 つだけという単純さも残るので、この形にした。`ResetFinished` のような段を足すのは、その単純さを崩すので慎重に決めたい。

ユーザーの方向（同日）: 欲しいのは step の結果なので、リセットの前に Observer を呼ぶのが正しい。気がかりは、リセット後の情報をいつ・なぜ取るか（その必要性を含む）。この問いへの答えは D4 である。

## 3. 現状（2026-10-19 にコードで確かめたこと）

### 3.1 BatchEnv と Step の結果

- 単一の env をまとめる BatchEnv は 2 つある。逐次の `VectorizedDiscreteBatchEnv`（[env.cpp:697-742](../../core/anet-core/src/env.cpp)）と、スレッドの `ThreadPoolDiscreteEnv`（同 830-890）。
  - どちらも lane ごとに `envs_[i]->Step(a)` を呼び、done か truncated なら続けて `envs_[i]->Reset()` を呼ぶ。
  - `next_state` にはリセット前の状態、`continue_state` にはリセット後の状態を詰める。続く lane の `continue_state` は `next_state` の写しである。
  - aux（step の info）は `Step` の結果なのでリセット前。リセットの aux は捨てている。
- `BatchStepResult` は `reward`・`next_state`・`continue_state`・`n_transitions`・`n_episode_end` を持つ（[rl.hpp:577-590](../../core/anet-core/include/anet/rl.hpp)）。
  - `BatchEnv` の操作は `Reset` と `Step` の 2 つ（同 670-686）。
  - wrapper の `Step` の結果は呼ぶたびに新しく作る（[env.cpp:609-613](../../core/anet-core/src/env.cpp)）。`Reset` の結果は使い回す（同 600-606）。
- 自前でバッチを持つ `ImageClsEnv` は、`Step` の中で次のバッチを先読みし、それを `next_state` と `continue_state` の両方の観測にする。エピソードの終わりでは `continue_state` の `episode_start` を立て、step 数を 0 に戻す（[ImageClsEnv.cpp:139-190](../../core/envs/imagecls1/src/ImageClsEnv.cpp)）。
- `continue_state` を使うのは次の 2 つで、Agent・replay・View は使わない。
  - Runner の `state_ = result->continue_state`（評価 [trainer.cpp:295](../../core/anet-core/src/trainer.cpp)、逐次の学習 同 566、非同期の学習 同 700）
  - エピソード構造の検査（`continue_state.episode_start` が done か truncated と一致すること。[env.cpp:142-176](../../core/anet-core/src/env.cpp)）

### 3.2 Runner の通知の時機

- EvalRunner: `MakeAction` → `Step` → `TrainEvent` → `EpisodeEndEvent` → `state_ = continue_state`（[trainer.cpp:231-298](../../core/anet-core/src/trainer.cpp)）。
- 逐次の学習 Runner: `MakeAction` → `Step` → `EpisodeEndEvent` → 学習（`UpdateFromBatch`）→ `LearnEvent` → `TrainEvent` → `state_ = continue_state`（同 509-566）。
- 非同期の学習 Runner（既定。[common.txt:4](../../apps/runner/config/common.txt) の `run.train.runner_type ?= pipeline`）は、次の順に進む（同 634-700）。
  - step t の `TrainEvent` は、次の DoStep の頭で、学習の待ち合わせのあとに通知する。
  - 続けて `MakeAction(t+1)` → 学習の投入 → `Step(t+1)` → `EpisodeEndEvent(t+1)` → `state_ = continue_state` と進む。
  - 最初の 2 回の DoStep では、前の step の `TrainEvent` を出さない（学習の結果がまだ無い。step 0 の `TrainEvent` は出ない）。
  - 今は、終端の lane のリセットが `Step` の中にあるので、GPU の学習と重なる時間に行われている。

### 3.3 評価セッション

- `EvalSessionEnv` は中の BatchEnv を包み、中の `Step` が戻ったあとで、完了した採用 episode の購読した鍵の `GetScalar` を溜める（[env.cpp:336-390](../../core/anet-core/src/env.cpp)）。読むのはリセット後だが、完了値の鍵は env が持ち越しているので正しく読めている。残機の `lives` のような今の値は、リセット後の値を溜めている。
- セッションの継ぎ目では、前の `Step` の `continue_state` が全 group とも新しいエピソードなら、`Reset` を呼ばずにそれを使い回す（同 317-334、[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)）。これは、N=1 の乱数の系列と ImageCls の先読みのバッチを二重のリセットで壊さないためである。

### 3.4 env の完了値と持ち越し

- **Atari**
  - ゲームの完了時に `RecordGameCompletion` が `completed_*` へ写しを取る（[AtariEnv.cpp:696-717](../../core/envs/atari1/src/AtariEnv.cpp)）。`GetScalar` の完了の鍵（`game_score`・`game_len`・`game_frames`・`hns57`・`hns49`・`game_score.ge.[N]`・`ram_metric.[n]`）は、完了した Step のあとだけ写しを返し、次の `Step` の頭で消える（同 756、839-891）。
  - **`Reset` の中でもゲームの完了を記録する**: `episodic_life` で残機を失った直後の `Reset` は NOOP を 1 フレーム打ち、それでゲームオーバーになれば、そこで完了を記録する（同 725-733）。今は `Step` の中でリセットするので、この完了は、残機を失った step の通知で読める。
  - `lives` は今の値を返す（同 865）。
- **DropMerge**
  - エピソードの終わりに `last_*` へ写しを取る（[DropMergeEnv.cpp:1222-1236](../../core/envs/dropmerge1/src/DropMergeEnv.cpp)）。
  - エピソードの統計は `Reset` では消さず、次のエピソードの最初の `Step` で消す（同 963-980）。完了の鍵は、完了した Step のあとだけ値を返す（同 1556-1670）。
  - 写しのうち `last_score`・`last_step`・`last_term_reason` は aux にも入れ（同 1542-1547）、View が「前回のスコア・ステップ数・終了要因」を出すのに使う（[DropMergeView.cpp:447-500](../../core/envs/dropmerge1/src/DropMergeView.cpp)）。
- **GridMaze**: エピソードの和は `Reset` で消すので、終わりに `last_*` へ写しを取り、完了の鍵はそれを返す（[GridMazeEnv.cpp:166-273](../../core/envs/gridmaze1/src/GridMazeEnv.cpp)）。
- **LunarLander**（`wind_x`・`wind_torque`）と **ImageCls**（`accuracy`・`epoch_count`）は、今の値か、採点の区切りで確定した snapshot を返す。持ち越しの仕組みは無い。

### 3.5 env を読む読み手

- `$env` の scalar（`@train`）は、`TrainEvent` の中で BatchEnv の `GetScalar` を lane 方向に集約して読む（[observers.cpp:1078-1088](../../core/anet-core/src/observers.cpp)、[env.cpp:617-640](../../core/anet-core/src/env.cpp)）。
- trace（`@episode_end`）は、`EpisodeEndEvent` の中で env の `GetScalar(key, lane)` を読む（[observers.cpp:837-872](../../core/anet-core/src/observers.cpp)、ADR 0037）。
- **AtariView**
  - 終端の通知で、env の `rgb_frame`（リセット後）を出す（[AtariView.cpp:218-257](../../core/envs/atari1/src/AtariView.cpp)）。
  - 一緒に出す step の結果の観測・aux（リセット前）とは時点がずれている。
- View は `TrainEvent` だけを受ける（[gui.hpp:259](../../core/anet-core/include/anet/gui.hpp)）。
  - そのため、Run の最初のエピソードの s_0 は今も画面に出ない。
  - 終端の step では、偶然リセット後の s_0 が出ている。手動で QValuePanel から進めるとき（[EvalPanel.cpp:161-168](../../apps/runner/src/EvalPanel.cpp)）は、その s_0 が次のクリックの当たる状態になっている。

### 3.6 乱数

- env の乱数は env ごとに持つ。torch の共有の乱数は使わない（`core/envs` に `torch::rand` などの呼び出しは無い）。
- そのため、逐次の学習 Runner でリセットが学習のあとへ動いても、学習の乱数の系列は変わらない。
- Actor の `MakeAction` の回数と順番も変わらない。

### 3.7 env の今の値を読むメトリクス

[Atari.txt](../../apps/runner/config/Atari.txt) の `mean.lives` を読む 7 本は、次の場所で値が変わる。

- `42_env/30_lives_mean`・`31_lives_mean_ema`（`@train`、1274-1275 行）: 値が変わるのは、実ゲームの終わりか打ち切りでゲームを作り直した lane があった step だけである。今はリセット後の満タンの残機を読み、改修後は終わったときの残機（ゲームオーバーなら 0）を読む。残機を失っただけの lane では値は変わらない。`Step` が `current_lives_` を失ったあとの値にし（[AtariEnv.cpp:797](../../core/envs/atari1/src/AtariEnv.cpp)）、soft reset も ALE から同じ値を読み直す（同 747）からである。
- `51_eval1/30`・`31`、`52_eval2/30`・`31`、`53_evalg/30`（`@session_end`。評価セッションが採用 episode の完了ごとに溜める。1301-1302・1326-1327・1348 行）: すべての値が、リセット後の満タンの残機から、採用 episode が終わったときの残機に変わる。評価は `episodic_life=false` なので、ほとんど 0 になる（[220_atari_env.jp.md](../design/220_atari_env.jp.md) 198 行）。

ほかの `$env` の鍵（`@train` の Atari の完了の鍵と RAM、DropMerge の `ep_*`、trace の完了の鍵。同 1258-1289 行、[DropMerge.txt](../../apps/runner/config/DropMerge.txt) 1271-1280 行）は完了の鍵なので、値は変わらない。

## 4. 確定事項

導入する機構とゴールの対応:

| 機構 | 寄与するゴール |
|---|---|
| `Step` はリセットせず、終わったエピソードのリセットを別の段にする（D1〜D3、D8） | G1、G2 |
| Runner 3 つと評価セッションの呼び出しの順（§4.3.2・§4.3.3） | G1、G2 |
| Atari の残機喪失直後の NOOP を `Step` へ移す（D6） | G3（完了値を失わない） |
| エピソード構造の検査を 2 つに分ける（§4.3.4） | G1・G2 の形になっても、今の構造の契約の fail-fast を保つ |
| 完了値の持ち越しを外す（D5） | G1 の結果、使われなくなる仕組みを消す |

### 4.1 決定

**D1 M2: `Step` はリセットせず、Runner が通知のあとに終わったエピソードをリセットする段を呼ぶ。**
BatchEnv の `Step` は、終端になった lane もリセットしない。Runner は、その `Step` の通知をすべて済ませたあとで、終わったエピソードのリセット（D3 の `ResetEndedEpisodes()`）を呼び、次の判断の状態を受け取る。評価セッションで採用 episode ごとに出す `EpisodeEndEvent` も、済ませる通知に含む。通知の間、env は全 lane とも遷移のあとにある。
理由: 読む側が何もしなくても、通知の中の値が正しくなる。
退けた M1 は、`Reset`・`Step` を保ったまま、宣言で購読した鍵の値を BatchEnv がリセットの前に読んで結果へ写す案である。M1 では次の問題が残る。
- 購読の口と、2 つの読み口（終端の lane は写し、ほかは今の env）が要る。
- 購読していない鍵を通知の中で env から直接読むと、終端の lane だけ黙ってリセット後になる。
- D5 と両立しない。

性能の差が出るのは、非同期の学習 Runner だけである。そこではリセットと続く lane の状態の写しが、GPU の学習と重なる時間から外れる（§4.3.2）。lane が多く `episodic_life` のときは、ほぼ毎 step どこかの lane が終わるが、重いゲームの作り直しはゲーム 1 回に 1 度で、残機を失ったときのリセットは数フレームなので、量は小さい見込みである。量は §7 の受入 4 で測る。
既存 ADR との関係: [ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md) の「採用権のない group も通常どおり Step / auto-reset する」「保存済みの最終 `continue_state` で全 group が fresh なら再利用する」と矛盾する。[ADR 0059](../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md) で置き換え、0034 の先頭に、置き換えた点と残る点を注記した。
依存: ユーザーの方向（§2）と、§3.5 の読み手の一覧。

**D2 終わったエピソードのリセットは、`Step` のたびに毎回 1 回呼ぶ。**
終わった lane が無い step でも呼び、そのときは何もリセットせずに状態だけを返す。次の 3 つは、どれも fail-fast にする。
- 段を呼ばずに次の `Step` を呼ぶ。
- `Step` を挟まずに段を 2 回呼ぶ。
- `Reset` の直後に段を呼ぶ。

理由: Runner 3 つと評価セッションの順番が、「`Step` → 通知 → 段 → `MakeAction`」の 1 つにそろい、分岐が無くなる。状態を組み立てる手間は、今 `continue_state` を毎 step 組み立てているのと同じなので、性能は変わらない。
退けた案: 終わった lane があるときだけ呼ぶ。終わった lane の無い step で写しを 1 回省けるが、Runner 3 つに同じ分岐が入る。
依存: D1。

**D3 段の名前は `BatchEnv::ResetEndedEpisodes()`。呼び名は「終わったエピソードのリセット」とする。**
理由: 既存の `EpisodeEndEvent`・`n_episode_end` の「episode end」の語にそろい、何をする操作かが名前で読める。
退けた候補:
- `ResetFinished()`: 仮の名。「finished」はコードの語彙に無い。
- `ResetCompleted()`: 「リセットが完了した」と読み違えやすい。
- `ContinueEpisodes()`: リセットが起きることが名前から読めない。

用語は [CONTEXT.md](../../CONTEXT.md) の「Env・実行」に足した。既存の語では言い換えられない。「Reset」は全 group を新しいエピソードにする操作で、終わった group だけを対象にしない。「自動リセット」（auto-reset）は `Step` の中でリセットする旧来の形を指すので、`_Avoid_` に挙げた。

**D4 リセット後の情報は、Runner だけが段の戻り値として受け取る。Observer への通知は足さない。**
リセット後の情報とは、次のエピソードの最初の観測・画面 s_0 と、リセットの aux である。Runner はこれを次の `MakeAction` に使う。View は終端の画面 s_T を出したままにする。
結果として、手動で進めるときは、エピソードの境目で s_0 を見ずに最初の行動を選ぶことになる。これは Run の最初のエピソードの今の扱い（§3.5）と同じである。
理由: 「エピソードの開始」の通知は G1〜G3 に効かない機能の追加である。また、境目で何を画面に出すかは、「画面と、行動が当たる状態のずれ」という [999_qvaluepanel_manual_action_state_alignment](999_qvaluepanel_manual_action_state_alignment_10prd.md) と同じ関心である（§5）。
依存: D1、§3.5（今 s_0 が出るのは終端の step の偶然だけ）。

**D5 完了値をリセットのあとまで持ち越す仕組みを外す。DropMerge の View が出す前回値の写しは残す。**
- 外すもの:
  - Atari の `completed_*`
  - GridMaze の `last_*`
  - DropMerge の `GetScalar` が使う `last_*` と、「統計を次のエピソードの最初の `Step` で消す」作り
- 外したあとの形: 完了の鍵は、完了した Step の通知の間は今の値をそのまま返し、統計はリセットで消す。リセットのあとに完了の鍵を読むと NaN になる（§4.2 の 3）。
- 残すもの: DropMerge の aux の `last_score`・`last_step`・`last_term_reason`。View の「前回の結果」の表示の機能として残す（§3.4）。

理由: D1 では、読み手はすべて通知の中、つまりリセットの前に読む。そのため今の値が完了値と同じになり、持ち越しは使われない仕組みになる（ユーザーの選択。持ち越しはゴールに直接は効かないが、残すと「通知の中で今の値を読む」という規則が 2 つに分かれる）。
依存: D1。M1 を選んでいたら成り立たなかった。
経緯: 初めの問いで DropMerge の View の用途を説明し漏れたので、訂正して聞き直した（2026-10-19）。

**D6 Atari の残機喪失直後の NOOP を `Step` の終わりへ移す。**
残機を失った step では、観測・画面（`rgb_frame`）・aux を作ったあとで NOOP を 1 フレーム打つ。それでゲームオーバーになれば、その `Step` の中で完了を記録する。リセットの段は、ゲームの作り直しか FIRE だけを行う。

- 変わらないもの: ALE に渡るフレームの列と、sticky actions の乱数を消費する順番は今と同じ。得点・長さ・フレーム数・学習の系列も変わらない。
- 通知の間の状態: ALE は観測より 1 フレーム先に進んでいる。画面・aux・`lives` は観測と同じ時点の値である。

理由: D1 のままでは、`Reset` の中で記録した完了（§3.4）が通知のあとになり、次の `Step` の頭で誰にも読まれずに消える。まれだが、起きるとゲーム 1 回分の完了値が抜け、G3 に反する。
退けた案:
- ALE の状態を複製して先読みする: 複製と復元が増え、乱数の系列が変わらないことを確かめる手間がかかる。
- 失ってよいとする: G3 に反する。

依存: D1、G3、[220_atari_env.jp.md](../design/220_atari_env.jp.md) の soft-reset の契約（198 行）。

**D7 学習の系列が変わらないことは、決定的モードでの完全一致で確かめる。**
決定的モード（[common.txt:98-100](../../apps/runner/config/common.txt) の `backend.@deterministic`）を使い、次の順で確かめる。

1. 改修前のビルドで同じコマンドを 2 回走らせ、一致することを先に確かめる。一致しない構成があれば、改修前からある問題として原因を調べ、比べられるようにしてから進む。
2. 改修前と改修後で、次が完全に一致することを確かめる。
   - scalar の全点（tag・step・値のビット列）
   - trace の全行（tag・step・lane・data）
3. 比べる対象から外すのは、次のものだけである。
   - 実時間の tag（`90_perf/`・`91_resource/`・`92_resource/`）
   - trace の `timestamp`
4. §3.7 の 7 本は完全一致の対象から外し、点の数と step が同じことだけを確かめる。

理由: G3 は「学習の系列を変えない」なので、改修前後の値はビットまで一致するはずである。改修前どうしが一致しない構成では、ずれの原因が改修か元の非決定性かを切り分けられない。実時間の tag と `timestamp` は、走らせるたびに違う。
前例: [PRD 084 の比較スクリプト](../../core/anet-core/testdata/prd084/compare_metrics.py)。これは scalar の全点を比べ、実時間と資源の tag を外している。trace の比較は足す。
経緯: 初めは「全 tag の checksum」と書いていたが、実時間の tag と `timestamp` は必ず違うので、ユーザーの問い（決定的モードで完全一致か）を受けて直した（2026-10-19）。

**D8 `Step` の直後に、段を呼ばずに全体の `Reset()` を呼ぶことも fail-fast にする。**
理由: D6 のあと、残機を失った直後の Atari の lane は「段で FIRE を打つ待ち」のまま残る。ここで全体の `Reset()` を呼ぶと、単一の env の `Reset` は待ちを見て FIRE だけでゲームを続けてしまう（今は `Step` の中で FIRE まで済んでいるので、同じ呼び方でゲームを作り直す）。今の Runner にこの呼び方は無い。`Reset()` を呼ぶのは初期化と評価セッションの開始だけで、後者は必ず段のあとである（§3.2・§3.3）。そのため、止めても困る呼び手は無い。D2 の「`Step` のあとは必ず段」とそろい、Atari の全体の `Reset` の振る舞いを新しく決めずに済む（ユーザーの選択）。
退けた案:
- 許し、待ちの lane は段と同じに扱う: 全体の `Reset` がゲームを作り直すとは限らなくなる。
- 許し、常にゲームを作り直す: 単一の env の `Reset` が、全体のリセットか段かを見分ける口が要る。

依存: D2、D6。経緯: 別モデル（Fable）の検討で挙がった（2026-10-19）。

### 4.2 規約・ADR・コードから導いたこと

1. **移行はクリーンブレークで行う。** 現用のコード、テスト用の BatchEnv（§4.3.8）、設定、現行の文書を、同じ変更で新しい契約へ移す。旧契約の別名・互換の分岐・WARN は残さない（根拠: [AGENTS.md](../../AGENTS.md) §クリーンブレーク方針）。
2. **過去の記録は書き換えない。** 過去の Run の記録と実験記録は書き換えない。コミット済みの ADR は本文を変えず、ADR 0034 の先頭に注記を足す（根拠: 同上、anet-create-prd の「既存の ADR を書き換えない」）。
3. **リセットのあとに完了の鍵を読んだときは NaN を返し、前回の値を返さない**（根拠: [AGENTS.md](../../AGENTS.md) §GetScalar 実装ルール）。
4. **`BatchStepResult` から `continue_state` を外す。** D1 では、`Step` の時点でリセット後の状態がまだ無いためである（根拠: D1、§クリーンブレーク方針）。
5. **段の戻り値は、既存の `BatchResetResult`（`state` と `GetAuxDataList`）にする。**
   - 終わった lane: リセットの結果の状態（`episode_start` = true）と、リセットの aux。
   - 続く lane: 直前の `Step` の `next_state`（`episode_start` = false）と、空の aux。空にするのは、ADR 0034 が使い回した状態の aux を空にしているのと揃えるためである。
   - aux の一覧は、空の lane を詰めずに lane の数だけ返す（`GetAuxDataList(lane)` の番号がずれないようにするため）。今の wrapper の結果は、中身の無い lane を飛ばして詰める（[env.cpp:428-447](../../core/anet-core/src/env.cpp)）ので、段の結果ではそうしない。
   - 戻り値は、`Step` の結果と同じく呼ぶたびに新しく作る。今 `Step` の結果に入っている `continue_state` の分が段へ移るだけなので、確保の量は変わらない。使い回すと、Runner が次の `Step` のあとまで使う `state_`（経験の `state_.Clone()`）と、`EvalSessionEnv` が覚えた結果が同じ領域を指し、次の段で書き換わる。
   - 根拠: [env.cpp:609-613・722-741](../../core/anet-core/src/env.cpp)、[ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)
6. **非同期の学習 Runner の順番**は §4.3.2 のとおりとする。根拠は、次の 3 つを同時に満たす順番がこれ 1 通りしかないことである。
   - `TrainEvent` は学習の結果を載せるので、待ち合わせのあとになる。
   - `MakeAction` はリセット後の状態を使い、今と同じく学習の投入より前に置く。GPU が空いている間に推論するという今の意図である（同 658-661 のコメント）。段を学習と重ねるために、学習の投入を段より前へ動かすと、この意図が崩れる。
   - D1 により、`TrainEvent` の間は env を遷移のあとに保つ。
   - 根拠: [trainer.cpp:634-700](../../core/anet-core/src/trainer.cpp)
7. **逐次の学習 Runner と評価の Runner**は、今の通知の順番のあとに段を置く（根拠: D1・D2、同 231-298・509-566）。
8. **評価セッションの継ぎ目は、ADR 0034 の判断を段の結果の上へそのまま移す**（§4.3.3）。採用の規則は変えない（根拠: [ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、NG3）。
9. **エピソード構造の検査は、`Step` の結果と段の結果の 2 つに分ける**（§4.3.4）。検査は今と同じく枠組みの側が持つ（根拠: ADR 0034 の「構造の契約は枠組みが持つ」、[env.cpp:120-176](../../core/anet-core/src/env.cpp)）。
10. **ImageCls も同じ契約に合わせる**（§4.3.5）。BatchEnv の契約は 1 つだからである（根拠: §クリーンブレーク方針、[ImageClsEnv.cpp:139-190](../../core/envs/imagecls1/src/ImageClsEnv.cpp)）。
11. **等価性と性能の比べ方は前例に従う。** checkpoint のファイルそのもののハッシュは合否に使わず（根拠: [PRD 065](done/065_nn_spectral_norm_10prd.md) の受入 1、[PRD 083](083_action_context_retire_10prd.md) の確定 11）、実時間は改修前後を交互に走らせて比べる（根拠: PRD 065 の受入 5）。

### 4.3 実装契約

#### 4.3.1 BatchEnv

```cpp
class BatchEnv : public Module {
public:
    // すべての episode group を新しいエピソードの開始状態にする。Step の直後（段の前）には呼べない
    virtual std::shared_ptr<const BatchResetResult> Reset() = 0;
    // 全 lane を 1 step 進める。終端の lane もリセットしない
    virtual std::shared_ptr<const BatchStepResult> Step(std::shared_ptr<BatchActionInfo> action_info) = 0;
    // 直前の Step で終わった episode group だけを新しいエピソードの開始状態にし、全 lane の次の判断の状態を返す
    virtual std::shared_ptr<const BatchResetResult> ResetEndedEpisodes() = 0;
    ...
};
```

| 呼び出し | 前の呼び出し | 振る舞い |
|---|---|---|
| `Reset()` | まだ何も呼んでいない・`Reset()`・`ResetEndedEpisodes()` | 全 group をリセットする |
| `Reset()` | `Step()` | fail-fast（`ResetEndedEpisodes()` を呼んでいない。D8） |
| `Step()` | `Reset()` か `ResetEndedEpisodes()` | 全 lane を進める。終わった lane の env は遷移のあとのまま残る |
| `Step()` | `Step()` | fail-fast（`ResetEndedEpisodes()` を呼んでいない） |
| `Step()` | まだ何も呼んでいない | fail-fast（`Reset()` を呼んでいない） |
| `ResetEndedEpisodes()` | `Step()` | 終わった lane だけ `Reset` し、全 lane の状態を返す（§4.2 の 5）。終わった lane が無ければ何もリセットしない |
| `ResetEndedEpisodes()` | `Reset()`・`ResetEndedEpisodes()`・まだ何も呼んでいない | fail-fast（段の対象の `Step` が無い） |

- まだ何も呼んでいない `Step()` を止めるのは、初期の状態の無い呼び出しが契約に反する内部状態だからである（根拠: [AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の「契約違反の内部状態」）。前例として、`EvalSessionEnv` はすでに `Reset` の前の `Step` を止めている（[env.cpp:352-353](../../core/anet-core/src/env.cpp)）。今の Runner とテストは、どれも最初に `Reset()` を呼ぶ。
- fail-fast のメッセージには、env 名・期待した前の呼び出し・実際の前の呼び出しを含め、今の `ValidateEpisodeStructure` と同じ書き方にする。
- 呼び出しの順番の検査は、`BatchEnvBase` に共通の補助を置き、各実装が使う。
- `ThreadPoolDiscreteEnv` の段は、終わった lane があるときだけ thread pool に仕事を配る。lane と worker の対応は `Step` と同じ（`i % worker_count`）にする。
- `next_state`・`reward`・aux の意味は変えない（NG1）。

#### 4.3.2 Runner の順番

```
評価:        MakeAction(s_t) → Step(a_t) → TrainEvent → EpisodeEndEvent → s_{t+1} = ResetEndedEpisodes().state
評価セッション（RunSession）:
             MakeAction(s_t) → Step(a_t) → TrainEvent
             → 採用 episode の EpisodeEndEvent（LastAdoptedGroups から RunSession が出す）
             → s_{t+1} = ResetEndedEpisodes().state
逐次の学習:  MakeAction(s_t) → Step(a_t) → EpisodeEndEvent → UpdateFromBatch → LearnEvent → TrainEvent
             → s_{t+1} = ResetEndedEpisodes().state
非同期の学習（DoStep の頭）:
             学習の待ち合わせ → 前の step の LearnEvent・TrainEvent（出す回だけ）
             → 前の Step があれば s_t = ResetEndedEpisodes().state
             → MakeAction(s_t) → 学習の投入 → Step(a_t) → EpisodeEndEvent
```

- 評価セッションでは、採用 episode の `EpisodeEndEvent` を、Step の処理を抜けたあとで `RunSession` が別に出している（[trainer.cpp:334-342](../../core/anet-core/src/trainer.cpp)）。段はこの通知のあとに呼ぶ。Step の処理の末尾に段を置くと、この通知の中で env の完了の鍵が NaN になり（D5）、評価の trace（[Atari.txt](../../apps/runner/config/Atari.txt) 1259-1261 行など）が全行空になる。段を呼ぶのが Step の処理の中か、呼び手（`DoStep`・`RunSession`）かは実装で決めてよい。
- 非同期の学習 Runner では、リセットと続く lane の状態の写しが、GPU の学習と重なる時間から外れる。量は小さい見込みだが、毎 step 起き得るので、影響は §7 の受入 4 で測る（D1）。
- 段の結果の構造の検査（§4.3.4）は、段を呼んだ直後に Runner が行う。
- `DoStep` が返ってから次の `DoStep` までの間（非同期の学習 Runner では、次の DoStep の頭で段を呼ぶまで）、学習用の env は遷移のあとのまま残る。この間に学習用の env を読む処理は無い（§3.5）。

#### 4.3.3 評価セッション（`EvalSessionEnv`）

- `Step` は中の `Step` を呼び、Step の結果の検査のあとで、完了した採用 episode の購読した鍵を溜める（今と同じ。読むのはリセット前）。
- `ResetEndedEpisodes()` は中の段を呼び、段の結果の検査のあとで、その結果を覚える。セッションの結果が確定したあと（`Step` は fail-fast になる状態。[env.cpp:352-355](../../core/anet-core/src/env.cpp)）でも呼べる。最後の step のあとにも段を呼ぶため（D2）。
- `Reset()`（セッションの開始）は、覚えた段の結果の全 group が新しいエピソードならそれを使い回し、そうでなければ中を全 lane `Reset` する。使い回すときの aux は空にする（ADR 0034 の判断をそのまま移したもの）。
- EvalRunner の `RunSession` は、各 step で採用 episode の `EpisodeEndEvent` を出してから段を呼ぶ（§4.3.2）。セッションの最後の step でも段を呼ぶ（D2）。`SessionEndEvent` は今と同じく、セッションのループのあとに出す。

#### 4.3.4 エピソード構造の検査

| 検査 | 対象 | 条件 | 呼ぶ所 |
|---|---|---|---|
| Reset の結果 | `Reset()` の結果 | 全 group の `episode_start` が true（今と同じ） | Runner の初期化、`EvalSessionEnv::Reset` |
| Step の結果 | `Step()` の結果 | `SHARED` では group 内の done・truncated が一致する。`n_episode_end` は終わった group の数と一致する。終わった group の一覧を返す | `RunnerBase` のエピソード集計の前、`EvalSessionEnv` の溜め込みの前 |
| 段の結果 | `ResetEndedEpisodes()` の結果と、直前の `Step()` の結果 | lane ごとに `episode_start == (done \|\| truncated)`、`SHARED` では group 内で一致する | Runner が段を呼んだ直後、`EvalSessionEnv::ResetEndedEpisodes` |

違反したときのメッセージは今と同じ書き方にする（env 名、group・lane、mask、期待した値、実際の値）。

#### 4.3.5 ImageCls

- `Step` は今と同じく、採点して次のバッチを先読みし、`next_state` を返す。エピソードの切り替え（step 数を 0 に戻すこと）は行わない。
- `ResetEndedEpisodes()` は、先読み済みのバッチに、直前の `Step` で終わったかどうかで `episode_start` を立てた状態を返し、終わっていれば step 数を 0 に戻す。新しいバッチは読まない。
- `Reset()` は今と同じ。

#### 4.3.6 Atari

- `Step` の中身: 残機を失った step（`life_done`）では、観測・`rgb_frame`・aux を作ったあとで NOOP を 1 フレーム打ち、その報酬を `game_score_` へ加える。ゲームオーバーになれば、その `Step` で完了を記録する（D6）。
- リセットの段（単一の env の `Reset`）の中身:
  - NOOP でゲームオーバーになっていた場合: ゲームを作り直し、リセットの操作を行う。
  - ゲームが続いている場合: FIRE だけを打つ。
  - 実ゲームの終わりと打ち切りの場合: 今と同じく作り直す。
  - どの場合も、最後に `current_lives_` を ALE から読み直す（今と同じ。[AtariEnv.cpp:747](../../core/envs/atari1/src/AtariEnv.cpp)）。
- 完了の鍵の扱い（D5）:
  - `completed_*` の写しを外す。
  - 完了の鍵は、完了した `Step` のあとは今の `game_score_`・`game_len_`・ALE のフレーム数・RAM の値を返す。
  - `Reset` で完了の印を下ろすので、そのあとは NaN になる。
  - 完了のログの行は、`Step` の完了の時点の値で出す（今と同じ値）。

#### 4.3.7 DropMerge・GridMaze

- **DropMerge**
  - エピソードの統計を消す処理を、次のエピソードの最初の `Step` から `Reset` へ移す。
  - 完了の鍵は、完了した `Step` のあとは今の統計から返す。`GetScalar` のための `last_*` の写しは外す。
  - 平均のように組み立てる値は、読むときに統計から計算する。
  - aux の `last_score`・`last_step`・`last_term_reason` は、View の表示の機能として今のまま残す（D5）。
- **GridMaze**
  - `last_*` の写しを外し、完了の鍵は、完了した `Step` のあとは今のエピソードの値を返す。
  - 成功・穴の判定に使う最後の報酬は、そのエピソードの中だけで持つ。
- **共通**: 両方とも、完了の印（`episode_just_ended_`）は `Reset` と次の `Step` の頭で下ろす。

#### 4.3.8 移行の対象

- コード
  - `rl.hpp` の `BatchEnv`・`BatchStepResult`、`rl.cpp` の `BatchStepResult`
  - `env.hpp`・`env.cpp` の `BatchEnvBase`・`DiscreteBatchEnvBase`・`VectorizedDiscreteBatchEnv`・`ThreadPoolDiscreteEnv`・`EvalSessionEnv`・`ValidateEpisodeStructure`
  - `trainer.cpp` の Runner 3 つ
  - `ImageClsEnv`・`AtariEnv`・`DropMergeEnv`・`GridMazeEnv`
  - `observers.cpp` の、「次の Step が確定値を消す前に読む」という趣旨のコメント
  - `rl.hpp` の `BatchEnv` の操作のコメント（[rl.hpp:679-680](../../core/anet-core/include/anet/rl.hpp)）。今も「BatchStepResult は使い回される」と書いてあるが、`Step` の結果は毎回新しい（§3.1）。3 つの操作のコメントを、§4.2 の 5 の契約に合わせる
- テスト。洗い出しは、テストの中の BatchEnv の `Step`・`Reset` の直接の呼び出し、`BatchEnvBase` を継ぐ試験用 env、`BatchStepResult` を組み立てる箇所、`continue_state` の参照を検索して行った（2026-10-19）。単一の env の `Step` を呼ぶテストは、単一の env の操作（`Reset`・`Step`）を変えないので対象外である。ただし、Atari の単一の env の振る舞いを見る下の試験は除く。
  - `BatchEnvBase` を継ぐ試験用 env（`ResetEndedEpisodes()` を実装し、`Step` でリセットしない）
    - `env_test.cpp` の `NameTestBatchEnv`・`ScriptedSessionEnv`
    - `trainer_test.cpp` の `TestBatchEnv`
    - `episode_end_test.cpp` の `TestBatchEnv`・`SessionRunnerEnv`・`SlowSessionRunnerEnv`
    - `observers_test.cpp` の `TestBatchEnv`・`ProbeEnv`
    - `dqn_based_agent_test.cpp` の `JitterBatchEnv`
  - `continue_state` を持つ `BatchStepResult` を組み立てる箇所
    - `env_test.cpp` の `MakeEpisodeStep`（141 行・260 行）
    - `trainer_test.cpp` 136 行、`episode_end_test.cpp` 81 行、`observers_test.cpp` 96 行、`dqn_based_agent_test.cpp` 1180-1250 行
  - 段を挟まずに BatchEnv の `Step` を続けて呼ぶ試験（段を挟むよう移す）
    - `env_test.cpp` の「EvalSessionEnv dynamically grants exactly N per-lane episodes」（404・407 行）、「EvalSessionEnv supports shared episodes and resets a partially fresh batch」（438・440・458 行）、「EvalSessionEnv waits for every adopted per-lane episode」（498-516 行）
    - `ImageClsEnv_test.cpp` の「ImageCls shared eval completes an N-window EvalSession」（220・222 行）、「ImageCls rotating eval window uses exact valid count and padding」（507・511 行）
  - `continue_state` を見る箇所
    - `env_test.cpp` の「Episode structure validation distinguishes per-lane and shared groups」（329・339 行）、「EvalSessionEnv dynamically grants exactly N per-lane episodes」の使い回しの確認（417 行付近）、「EvalSessionEnv keeps the N=1 single-lane trace and scalar identity」（540 行）
    - `AtariEnv_test.cpp` の「AtariEnv keeps vectorized and thread-pool batch semantics」（1184-1185 行）
    - `ImageClsEnv_test.cpp` の「ImageClsEnv returns fresh observations and shared eval termination」（486-487 行）
  - Atari の単一の env の残機の試験 2 本: `AtariEnv_test.cpp` の 1188 行「episodic life の判定」と 1311 行「soft reset の FIRE」
    - 1188 行の試験は、soft reset の NOOP でゲームが終わったときの完了を `Reset` のあとに読んでいる（1296-1302 行）。D6 のあとは、残機を失った `Step` の直後に読むように移す。
  - 対象外と確かめたもの: `DropMergeEnv_test.cpp` 326 行・`LunarLanderEnv_test.cpp` 216 行・`AtariEnv_test.cpp` 1149 行（1184-1185 行を除く）の試験。BatchEnv を `Reset` のあと `Step` 1 回だけ呼ぶので、新しい契約でもそのまま通る。
- 現行の文書: §9。

#### 4.3.9 実装の注意（契約ではない）

- `EvalSessionEnv` の `Reset()` が段の結果を使い回して中の `Reset` を省くときも、`EvalSessionEnv` 自身の呼び出しの順番は「Reset のあと」へ進める。中の BatchEnv は段のあとなので、そのまま `Step` を呼べる。
- DropMerge の平均のように組み立てる値（[DropMergeEnv.cpp:1226-1236](../../core/envs/dropmerge1/src/DropMergeEnv.cpp)）を読むときに計算するなら、今と同じ式・同じ順番の float の演算にする。順番が変わると D7 のビットの一致が崩れる。

### 4.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 |
|---|---|---|
| `BatchEnv`（rl.hpp） | 変更 | 操作に `ResetEndedEpisodes()` を足す。`Step` はリセットしない契約にする |
| `BatchStepResult`（rl.hpp） | 変更 | `continue_state` を外す |
| `BatchEnvBase`（env.hpp） | 変更 | `Reset`・`Step`・段の呼び出しの順番を検査する共通の補助を持つ |
| `DiscreteBatchEnvBase`・`VectorizedDiscreteBatchEnv`・`ThreadPoolDiscreteEnv` | 変更 | `Step` で lane をリセットせず、段で終わった lane だけを `Reset` して状態と aux を返す |
| `EvalSessionEnv` | 変更 | 段を中へ任せて結果を覚え、セッションの開始で使い回す |
| `ValidateEpisodeStructure`（env.hpp の関数） | 変更 | Step の結果の検査と、段の結果の検査に分ける |
| `RunnerBase`・`EvalRunner`・`SerialTrainRunner`・`PipelineTrainRunner` | 変更 | 通知のあとに段を呼び、その結果を次の判断の状態にする（§4.3.2） |
| `ImageClsEnv` | 変更 | エピソードの切り替えを段へ移す |
| `AtariEnv` | 変更 | 残機喪失直後の NOOP を `Step` へ移し、`completed_*` の写しを外す |
| `DropMergeEnv` | 変更 | 統計を消す時機を `Reset` へ移し、`GetScalar` 用の `last_*` の写しを外す（View 用の aux は残す） |
| `GridMazeEnv` | 変更 | `last_*` の写しを外す |
| テスト用の BatchEnv（§4.3.8） | 変更 | `ResetEndedEpisodes()` を実装し、`Step` でリセットしない |

## 5. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| エピソードの開始（s_0）を Observer に知らせる通知と、View の開始の表示 | 先送り | ゴールに効かない機能の追加（D4）。境目で何を画面に出すかは、[999_qvaluepanel_manual_action_state_alignment](999_qvaluepanel_manual_action_state_alignment_10prd.md) と同じ関心である。その PRD で手動の時点のずれを決めるときに、境目の画面と合わせて作る |
| 購読した鍵をリセットの前に結果へ写す口（M1） | 不採用 | D1 |
| ALE の状態を複製しての先読み | 不採用 | D6 |
| 完了値の持ち越しを外す範囲に、DropMerge の View の前回値の表示を含めること | 不採用 | View の表示の機能なので残す（D5） |

## 6. 未決

なし。

## 7. 成功の測り方と受入基準

成功の測り方:

- 終端の lane の通知の中で、env の `GetScalar`・`GetTensor` が、その遷移の結果（リセット前）を返す。
- 学習の系列が変わらない（D7）。
- PRD 094 の step trace の `$env` が、終端の行でも「遷移のあと・リセット前」になる。

受入基準:

1. **等価性**（D7）: 次の 7 本を、改修前は 2 回、改修後は 1 回、同じコマンドで走らせる。step 数と seed は実装計画で固定する。
   - Run の組:
     - Atari 1: 非同期の学習 Runner で、定期評価のセッションを含む。`AtariEnv.episodic_life = true`・`fire_reset = true`（`AtariEnv.@classic` の組み合わせ）で走らせる
     - Atari 2: `AtariEnv.episodic_life = true`・`fire_reset = false`（`AtariEnv.@100k` の組み合わせ）で走らせる
     - DropMerge
     - LunarLander の `run.@cpu`
     - GridMaze
     - ImageCls
     - `run.train.runner_type = serial` を CLI で渡した DropMerge
   - Atari を 2 本にするのは、Atari.txt の既定の Run（`run.@v5_iqn_impala_x2`）が `AtariEnv.@v5_noop30`（`episodic_life = false`）で、D6 で移す残機喪失の経路を通らないためである（[Atari.txt](../../apps/runner/config/Atari.txt) 256 行・1151-1170 行）。渡し方は実装計画で決め、各 Run の `config/config_data.txt` で `AtariEnv.episodic_life` と `AtariEnv.fire_reset` の値を確かめる。
   - 合格の条件:
     - 改修前の 2 回が一致する。
     - 改修前と改修後で、scalar の全点（tag・step・値のビット列）と trace の全行（tag・step・lane・data）が完全に一致する。外すのは、実時間の tag（`90_perf/`・`91_resource/`・`92_resource/`）と trace の `timestamp` だけ。
     - §3.7 の 7 本は、点の数と step が同じである。
     - ほかに差が出たら不合格。
2. **単体テスト**
   - (a) 試験用の BatchEnv で、終端の lane の `TrainEvent`・`EpisodeEndEvent` の中で env の今の値を読むと、リセット前の値になる。次の 4 つの経路で確かめる。
     - 逐次の学習 Runner
     - 非同期の学習 Runner
     - 評価の Runner（`DoStep`）
     - 評価セッション（`RunSession`）の採用 episode の `EpisodeEndEvent`
   - (b) 次の 6 つが fail-fast で止まる: 段を呼ばずに `Step` を呼ぶ／`Step` を挟まずに段を 2 回呼ぶ／`Reset` の直後に段を呼ぶ／`Step` の直後に段を呼ばずに `Reset` を呼ぶ（D8）／まだ何も呼んでいない状態で `Step` を呼ぶ／まだ何も呼んでいない状態で段を呼ぶ。
   - (c) 段の結果の `episode_start` が直前の `Step` の done・truncated と食い違っていたら、検査が止める。
   - (d) `EvalSessionEnv` が、全 group が新しいエピソードなら、段の結果を空の aux で使い回して中の `Reset` を省き、そうでなければ全 lane を `Reset` する。
   - (e) Atari: 残機を失う既存の ROM の試験（[AtariEnv_test.cpp:737](../../core/envs/atari1/src/AtariEnv_test.cpp)）と、同じ seed の再現の試験（同 901）が、改修前と同じ値で通る。§4.3.8 のとおり移した残機の試験 2 本（同 1188・1311）が通り、1311 行の試験の `game_frames` の差（soft reset で FIRE ありなら +3、FIRE なしなら +1）は改修前と同じである。
   - (e2) Atari: 残機を失った直後の NOOP でゲームが終わる経路を通る試験がある。その `Step` の直後（段の前）に完了の鍵（`game_score`・`game_len`・`game_frames`）が NaN でない値を返し、段のあとで NaN になることを確かめる。
     - 経路の作り方（ROM で再現できる操作の列か、ALE の RAM を書き換えて残機喪失の直後にゲームが終わる状態にする、など）は実装計画で決める。
     - どの作り方でもこの経路を通せないと分かったら、受入を満たしたとしない。実装を止めて PRD の改訂に戻り、D6 の扱いを決め直す。
   - (f) Atari・DropMerge・GridMaze で、段のあとに完了の鍵が NaN になる。DropMerge の aux の `last_score`・`last_step`・`last_term_reason` は、段のあとも前回の値のまま残る。
3. **既存のテスト**: 移行したテストを含め、`[.]`（隠し）を除く全テストが通る。
4. **実時間**: 受入 1 の Atari 1 と同じ構成（非同期の学習 Runner、`episodic_life = true`）で、改修前と改修後を交互に各 3 回走らせる。
   - 1 本の値: その Run の `90_perf/12_exp_step_per_sec` の点のうち、`exp_step` が Run の最後の値の半分を超えるものの中央値。replay を溜める立ち上がりの区間を外すためである。Run の長さは、半分の時点で学習が始まっているように実装計画で決める。
   - 合格の条件: 改修後 3 本の値の中央値の、改修前 3 本の値の中央値からの低下が 2% 以内。
5. **手動の確認**: EvalPanel の Atari（`episodic_life = false`、[Atari.txt](../../apps/runner/config/Atari.txt) 1406 行）で、最後の残機を失ってゲームが終わった step の画面が、ゲームオーバーの瞬間の画面になる（改修前は、新しいゲームの最初の画面になる）。残機を失っただけの step は、改修前からリセットしないので確かめる対象にしない。

## 8. 依存と順序

- 順序: 本 PRD → PRD 094（段階 1・2）→ PRD 081 の改訂。
- [999_qvaluepanel_manual_action_state_alignment](999_qvaluepanel_manual_action_state_alignment_10prd.md) の案 A（次の状態の `MakeAction` を通知のあとに先に行う）を採る場合、その先読みは本 PRD の段のあとに置く（次の状態は段の戻り値である）。

## 9. 文書同期（実装と同じ変更で更新する）

| 対象 | 変更 |
|---|---|
| [120_environments.jp.md](../design/120_environments.jp.md) | §2.2 の Reset と Step の契約（`ResetEndedEpisodes()`、呼び出しの順番、検査の 2 つ）、§2.3 の ImageCls の終端（56 行）、189 行の EvalSessionEnv の継ぎ目と「Reset 時期は wrapper と Runner の契約で決まる」の記述 |
| [010_framework_overview.jp.md](../design/010_framework_overview.jp.md) | 220 行の `continue_state` の説明と 270 行の図 |
| [100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) | 212 行・274 行の図（`continue_state` を段の戻り値へ） |
| [150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) | 35 行の `continue_state` の説明 |
| [220_atari_env.jp.md](../design/220_atari_env.jp.md) | 28 行の「auto-reset は既存 wrapper」、35 行の「autoreset はフレームワーク既存機構」、198 行の soft-reset の NOOP の位置と「auto-reset wrapper（done\|\|truncated → Reset）の契約」（D6） |
| `docs/archify/anet_lab_20_training_step_sequence`（.archify.json・.html） | 学習 step の図の `continue_state`。アトラスの再生成で直す |
| [CONTEXT.md](../../CONTEXT.md) | 「終わったエピソードのリセット」（本 PRD の起草で追加済み） |
| [ADR 0034](../adr/0034-eval-session-aggregation-in-batchenv-decorator.md) | 先頭の注記（本 PRD の起草で追加済み） |
| 設計文書の英訳（`*.en.md`） | 手では直さない。日本語版の更新を受けて、翻訳の手順（`anet-translate-docs`）が追随する（英訳は元の blob を記録して古さを判定する） |
