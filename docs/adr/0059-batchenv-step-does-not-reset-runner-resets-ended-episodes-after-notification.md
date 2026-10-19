# BatchEnv の Step は終端の lane をリセットせず、Runner が通知のあとに終わったエピソードをリセットする

`Step` の中で終端の lane を自動リセットすると、通知（`TrainEvent`・`EpisodeEndEvent`）の中で env に問い合わせる読み手が終端の lane だけリセット後の値を読み、env の完了値の持ち越しで埋め合わせても、残機や画面のような今の値は直らない。そこで **BatchEnv の `Step` はリセットせず、Runner がその `Step` の通知をすべて済ませてから `ResetEndedEpisodes()` を毎回 1 回呼び、その戻り値を次の判断の状態にする**ことを決定する。通知の間はどの lane の env も「遷移のあと・リセット前」になり読む側は何もしなくても正しい値を読めるので、`Reset` と `Step` の 2 つだけという単純さを手放し、増えた呼び出しの順番は fail-fast で守る。

## Considered Options

- **Reset/Step を保ち、購読した鍵の値をリセットの前に Step の結果へ写す（M1）**: 操作は 2 つのまま。ただし購読の口と、終端の lane だけ写しを読む 2 つ目の読み口が要り、購読していない鍵を env から直接読むと黙ってリセット後の値になる。却下。
- **今の形（Step の中でリセットし、env が完了値をリセットのあとまで持ち越す）**: 完了値は読めるが、今の値の食い違いは残り、env ごとに持ち越しの仕組みが要る。却下。
- **終わった lane があるときだけ段を呼ぶ**: 終わった lane が無い step で状態の写しを 1 回省けるが、Runner ごとに同じ分岐が入る。却下。

## Consequences

- `BatchStepResult` から `continue_state` を外す。エピソード構造の検査は、Step の結果の検査と段の結果の検査に分ける。
- 評価セッションの継ぎ目（[ADR 0034](0034-eval-session-aggregation-in-batchenv-decorator.md)）は、直前の段の結果を使い回す形へ移る。
- 完了値の持ち越し（Atari の `completed_*` など）は外す。Atari が残機喪失直後のリセットで打つ NOOP は、完了を通知の中で読めるよう `Step` へ移す。
- 詳細は [PRD 097](../memo/097_env_auto_reset_after_notification_10prd.md)。
