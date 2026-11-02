# QValuePanel の Q の表示と、手動で選んだ行動が当たる状態のずれ 暫定 PRD

> 2026-10-19 起票（暫定。グリル前）。質問はせず、暫定の既定を置いて §5 に決めることを残した。
> 起点: [PRD 094](094_evalpanel_step_history_10prd.md)（Evaluation Trace）のグリルで、EvalPanel・QValuePanel が待っている間に出している時点を整理したとき、QValuePanel の Q は 1 つ前の状態のもので、クリックした行動は次の状態に当たると分かったこと（094 の §3.1）。
> 2026-10-19 PRD [097](097_env_auto_reset_after_notification_10prd.md) のグリルで、エピソードの境目の画面をこの PRD で決めることになったので、§2 の末尾と §5 の 5・6 を書き足した。
> 2026-10-20 追記: 097 を再グリルで改めた（[ADR 0060](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)）。BatchEnv は Reset と Step のままで、`ResetEndedEpisodes()` は作らない。そのため、§2 の末尾と §5 の 5・6 の 097 の段を指す記述を、Step の結果の `continue_state` に合わせて直した。
> 関連: [097_env_auto_reset_after_notification_10prd.md](097_env_auto_reset_after_notification_10prd.md)（遷移の結果を Step の結果で運ぶ）、[999_input_target_train_forced_action_10prd.md](999_input_target_train_forced_action_10prd.md)（入力先 train の強制行動）、[CONTEXT.md](../../CONTEXT.md) の「鏡写しインスタンス」「実行時 UI 操作」。

## 1. 目的と問題

QValuePanel で行動をクリックして EvalPanel を手動で進めるとき、画面に出ている Q と、クリックした行動が当たる状態が 1 step ずれている。研究者は「今の画面の状態で、どの行動の Q が高いか」を見て選んでいるつもりだが、実際には 1 つ前の状態の Q を見て、次の状態の行動を選んでいる。

### ゴール

- 手動で行動を選ぶとき、その行動が当たる状態の Q を見て選べる。

### 非ゴール

- Evaluation Trace（PRD 094）の Trace モードの表示。行 t の Q(s_t, ·) と行動 a_t は同じ時点にそろっていて、ずれは無い。
- 学習側（train runner）への手動の入力。999_input_target_train_forced_action の範囲である。

## 2. 現状（2026-10-19 にコードで確かめたこと）

- EvalRunner の 1 回の step は、`MakeAction(s_t)` → 行動の強制があれば `WithAction` → `env->Step(a_t)` → `TrainEvent` の通知 → `state_ = continue_state` の順に進む（[trainer.cpp:231-298](../../core/anet-core/src/trainer.cpp)）。
- QValuePanel は、この通知の `action_info` の aux から Q(s_t, ·) と a_t を出す（[QValuePanel.cpp:281-308](../../apps/runner/src/QValuePanel.cpp)、同 315-397）。EvalPanel の画面は、同じ通知の時点の env の画面 s_{t+1} を出す（[AtariView.cpp:218-257](../../core/envs/atari1/src/AtariView.cpp)）。待っている間、画面は s_{t+1}、Q は s_t のものである。
- QValuePanel で行動をクリックすると `EvalPanel::DoStep(action)` が呼ばれ（[RunnerFrame.cpp:484-487](../../apps/runner/src/RunnerFrame.cpp)、[EvalPanel.cpp:161-168](../../apps/runner/src/EvalPanel.cpp)）、`MakeAction(s_{t+1})` のあとに、その行動が s_{t+1} に強制される。クリックしたときに見ていた Q は、s_{t+1} の Q ではない。
- 行動の決定と env の step は同じ呼び出しの中で続けて起きるので、「s_{t+1} の Q を出して、行動を待つ」状態は今は無い。
- PRD 097 の実装のあとも、自動リセットは `Step` の中で行い、Runner は次の状態を Step の結果の `continue_state` で受け取る（097 の D2）。画面は Step の結果から読むので、終端の step の通知では、画面は終端の画面 s_T になる（097 の D1）。次のクリックが当たるリセット後の s_0 は、画面に出ない（097 の D4。今は終端の step でだけ、リセット後の s_0 が偶然出ている）。手動で進めると、エピソードの境目では s_0 を見ずに最初の行動を選ぶことになる。

## 3. 提案の骨子（暫定）

暫定の既定は案 B（表示で時点を明示する）とし、案 A・C は §5 で比べる。

| 案 | 中身 | 性能 | 保守性 |
|---|---|---|---|
| A. 先読み | 通知のあと、次の状態 s_{t+1} について `MakeAction` だけを先に行い、その Q を QValuePanel に出す。次の step はその結果を使う | 1 step あたりの forward は増えない（先に行うだけ）。ただし ε・NoisyNet の乱数の消費の時機が変わる | EvalRunner の step を「決定」と「実行」に分ける変更になる。強制の経路（`WithAction`）と同期（Sync）の時機も合わせる |
| B. 表示で明示（暫定の既定） | QValuePanel の見出しに「Q は step N（1 つ前の状態）の判断」と出し、クリックの対象は次の状態だと分かるようにする | 変わらない | 小さい。ずれそのものは残る |
| C. 2 段階のクリック | 1 回目のクリックで s_{t+1} の Q を計算して出し、2 回目で確定する | 手動のときだけ forward が 1 回増える | クリックの意味が変わる。キーでの手動の行動とそろえる必要がある |

## 4. 作らないもの・先送り

なし（暫定の起票のため、グリルで決める）。

## 5. 決めること（グリルで詰める）

| # | 決めること | 暫定の既定 |
|---|---|---|
| 1 | 案 A・B・C のどれにするか | B。ずれは残るが、変更が小さく、乱数の消費の時機を変えない |
| 2 | 案 A のとき、先読みした行動を、モデルの同期（Sync）をまたいで使ってよいか | 同期をまたいだら先読みをやり直す |
| 3 | キー（EvalPanel から転送される矢印キーなど）での手動の行動も同じ扱いにするか | 同じ扱いにする |
| 4 | PRD 094 の Trace モードとの関係 | Trace モードの間は手動の行動の前に Live へ戻る（094 の D11）ので、Live の表示だけを対象にする |
| 5 | エピソードの境目で、次のクリックが当たる s_0 の画面を出すか（PRD 097 の D4 で先送りした件。097 は Observer にエピソードの開始の通知を足していない） | 出す。出し方（エピソードの開始の通知を足すか、手動の step のあとに `continue_state` から画面を作るか）はグリルで決める |
| 6 | 案 A のとき、先読みを置く位置 | Step の結果の通知を済ませたあと。次の状態は Step の結果の `continue_state` だから（097 の §8） |

## 6. 成功の測り方の目安

- 手動で行動を選ぶとき、QValuePanel に出ている Q が、その行動が当たる状態のものだと画面から分かる（案 B）、またはその状態の Q そのものになっている（案 A・C）。
